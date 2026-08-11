"""T2 relative navigation — policy-on-estimated-state harness.

Converts the orbital rendezvous agent from guidance (full-state feedback) to
GN&C by inserting a measurement model + EKF between the environment and the
policy. Nothing in the C env is modified: the harness reads the truth
observation, recovers both absolute states from it, synthesizes a noisy
range/bearing measurement of the target, runs the EKF, and hands the policy a
38-dim observation whose target-derived slots have been recomputed from the
*estimate*. Chaser-only slots stay truth (GPS/INS assumption).

Stages
  sanity   quick truth-obs control run, anchors the harness against 97.5%
  qsweep   process-noise / sigma_v0 sweep, selected on NEES in-bounds fraction
  validate open-loop filter Monte Carlo: NEES / NIS / RMSE + plots
  eval     closed loop: truth, recon (rebuild path with zero estimation error),
           and EKF at 1x / 3x / 10x sensor noise, plus a 30x / 100x / 300x
           degradation tail
  all      qsweep -> validate -> eval

Run from /Users/pete/space_training/pufferlib.
"""

import argparse
import csv
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/Users/pete/space_training/pufferlib")

from pufferlib.ocean.orbital.orbital import Orbital          # noqa: E402
from pufferlib.models import Default, LSTMWrapper            # noqa: E402

import orbital_math as om                                    # noqa: E402
from ekf import (TargetEKF, measure, wrap_pi, NEES_LO, NEES_HI,  # noqa: E402
                 NIS_LO, NIS_HI, SIGMA_RHO_M, SIGMA_BETA_RAD,
                 Q_ACCEL_PSD, SIGMA_V0)

ROOT = "/Users/pete/space_training"
CKPT = f"{ROOT}/pufferlib/experiments/puffer_orbital_177765503091/model_puffer_orbital_000325.pt"
PLOT_DIR = f"{ROOT}/plots/relnav"
RESULTS_CSV = f"{ROOT}/web_data/results/relnav_results.csv"

# Headline eval conditions (canonical ckpt's training distribution).
ENV_KWARGS = dict(
    num_envs=1,
    num_debris_min=0,
    num_debris_max=0,
    e_max_target=0.05,
    e_max_sat=0.05,
    init_phase_gap_max=3.14159,
    valid_init_only=1,
    gave_up_action="terminate",
    max_valid_init_attempts=4096,
    obs_alt_scale_m=1.6e6,
    lvlh_scale_m=6.371e6,
    legacy_action_space=10,
)

# ── T3 corrected-dynamics config (--t3) ──────────────────────────────────────
# The T3 recovery checkpoint trains on phase_obs_mode=1 (mean-longitude phase
# channels + episode clock + apsidal alignment in obs[13-16]) with a Discrete(16)
# head, so legacy_action_space is deliberately absent. Everything the filter
# touches — the physical relative state — is unchanged; only the *encoding* of
# the target into the observation moves.
T3_CKPT = (f"{ROOT}/pufferlib/experiments/puffer_orbital_178642097817/"
           f"model_puffer_orbital_000382.pt")
T3_PLOT_DIR = f"{ROOT}/plots/relnav_t3"
T3_RESULTS_CSV = f"{ROOT}/web_data/results/t3_relnav_corrected.csv"
T3_ENV_KWARGS = dict(
    num_envs=1,
    num_debris_min=0,
    num_debris_max=0,
    e_max_target=0.05,
    e_max_sat=0.05,
    init_phase_gap_max=3.14159,
    valid_init_only=1,
    gave_up_action="terminate",
    max_valid_init_attempts=4096,
    obs_alt_scale_m=1.6e6,
    lvlh_scale_m=6.371e6,
    shaping_mode=1,
    shape_gamma=1.0,
    phase_gap_mode=1,
    phase_obs_mode=1,
    episode_cap_steps=3000,
    cap_terminal_reward=0.0,
)

# ── T4 wide-envelope config (--wide) ─────────────────────────────────────────
# The WL4 rung of scripts/orbital/t3/t3_wide_envelope.sh. Same corrected-dynamics
# observation *layout* as --t3 (phase_obs_mode=1, Discrete(16) head, so no
# legacy_action_space), but a 5x wider altitude domain (300-8000 km alt), target
# eccentricity to 0.30, and — the reason this needs its own flag — different
# observation *normalizers*: obs_alt_scale_m=8e6 (vs 1.6e6) and lvlh_scale_m=1.5e7
# (vs 6.371e6). The encode/decode layer must use the same constants the policy was
# trained under, or every target-derived slot is mis-scaled by 5x (altitudes) /
# 2.35x (LVLH) and the closed loop measures the wrong thing entirely.
#
# e_max_sat is deliberately absent: at WL3+ the chaser's eccentricity is set by
# de_max (e-vector offset from the target), which overrides e_max_sat sampling.
# This kwarg set is exactly what eval_checkpoint.py was invoked with for the
# 200/200 held-out (seed 123) WL4 number.
WIDE_CKPT = f"{ROOT}/models/t3/seed42_WL4_wide.pt"
WIDE_PLOT_DIR = f"{ROOT}/plots/relnav_wl4"
WIDE_RESULTS_CSV = f"{ROOT}/web_data/results/t4_relnav_wl4.csv"
WIDE_ENV_KWARGS = dict(
    num_envs=1,
    num_debris_min=0,
    num_debris_max=0,
    e_max_target=0.30,
    de_max=0.08,
    da_max_m=600e3,
    a_min_override=6.671e6,
    a_max_override=14.371e6,
    init_phase_gap_max=3.14159,
    valid_init_only=1,
    gave_up_action="terminate",
    max_valid_init_attempts=4096,
    obs_alt_scale_m=8e6,
    lvlh_scale_m=1.5e7,
    shaping_mode=1,
    shape_gamma=1.0,
    shape_dv_ref_ms=700.0,
    phase_gap_mode=1,
    phase_obs_mode=1,
    episode_cap_steps=6000,
    cap_terminal_reward=0.0,
)

# 0 = legacy true-anomaly obs[13-16]; 1 = T3 mean-longitude layout. Set by main().
PHASE_OBS_MODE = 0

# Sensor sampling period (s). 0 = legacy: exactly one measurement per env.step(),
# i.e. the navigation rate is welded to the guidance rate. That is harmless while
# decisions are <= 5 min apart (the T2 policy warped 5 min, 97% of the time) but
# becomes a sensor blackout under T3, whose policy spends 51% of its decisions on
# the 1-hour warp. Set to e.g. 60 to run the filter at a fixed sensor cadence
# through coasts and warps while the policy still acts only at decision epochs.
SENSOR_DT = 0.0

# Appended to condition names in the results CSV (keeps rows from two cadences
# distinguishable without touching the CSV schema).
COND_SUFFIX = ''

SETTLE = 10   # filter-acquisition transient, excluded from 'settled' RMSE

CAUSE_NAMES = ['none', 'success', 'collision', 'escape', 'safety_cap',
               'stranded', 'hyperbolic', 'gave_up']


# ── observation decode / re-encode, phase_obs_mode aware ─────────────────────
def _mean_anomaly(theta, e):
    """theta -> M using the *corrected* inverse, i.e. orbital.h at HEAD.

    orbital_math.true_to_mean deliberately mirrors the pre-fix (inverted)
    C routine and is inert there because M is never read. Under
    phase_obs_mode=1, M *is* read — lambda = M + omega is the phase channel —
    so the corrected map (orbital_math.mean_from_true) is the one that matches
    the environment. e is clamped below 1 so a hyperbolic estimate degrades
    instead of producing NaN.
    """
    m = om.mean_from_true(theta, min(max(e, 0.0), 1.0 - 1e-12))
    return m % (2.0 * math.pi)


def recover_states_t3(o, obs_alt_scale_m=om.OBS_ALT_SCALE_M):
    """Invert fill_observations() under phase_obs_mode=1.

    Chaser is unchanged (a,e from [0,1], theta from [2,3], omega from [9,10]).
    The target's anomaly is no longer in the observation directly: obs[13,14]
    carry sin/cos of the mean-longitude gap dlambda = lambda_s - lambda_t, so
        lambda_t = lambda_s - dlambda,  M_t = lambda_t - omega_t,
    and theta_t follows from Kepler. obs[15] is the episode clock and obs[16]
    is cos(omega_s - omega_t) — neither adds target information beyond
    obs[11,12], so the recovery is exact up to float32 obs quantization.
    """
    sat = {
        'a': float(o[0]) * obs_alt_scale_m + om.R_EARTH,
        'e': float(o[1]),
        'theta': math.atan2(float(o[2]), float(o[3])),
        'omega': math.atan2(float(o[9]), float(o[10])),
    }
    sat['M'] = _mean_anomaly(sat['theta'], sat['e'])
    tgt = {
        'a': float(o[7]) * obs_alt_scale_m + om.R_EARTH,
        'e': float(o[8]),
        'omega': math.atan2(float(o[11]), float(o[12])),
    }
    dlam = math.atan2(float(o[13]), float(o[14]))       # lambda_s - lambda_t
    lam_s = sat['M'] + sat['omega']
    tgt['M'] = ((lam_s - dlam) - tgt['omega']) % (2.0 * math.pi)
    tgt['theta'] = om.eccentric_to_true(om.solve_kepler(tgt['M'], tgt['e']),
                                        tgt['e'])
    return sat, tgt


def build_obs_t3(o, sat_el, tgt_el, obs_alt_scale_m, lvlh_scale_m,
                 tgt_cart=None):
    """Re-emit the observation with every target-derived slot from tgt_el.

    om.build_obs already handles the slots whose meaning is mode-independent
    ([7] a_t, [8] e_t, [11,12] sin/cos omega_t, [33-37] LVLH relative state);
    only obs[13-16] are re-encoded here:
        [13,14] sin/cos(lambda_s - lambda_t), lambda = M + omega
        [15]    episode clock — chaser-side, NOT target-derived: restored to
                truth (overwriting it would inject a fake mission deadline)
        [16]    cos(omega_s - omega_t)
    """
    out = om.build_obs(o, sat_el, tgt_el, obs_alt_scale_m, lvlh_scale_m,
                       tgt_cart=tgt_cart)
    lam_s = _mean_anomaly(sat_el['theta'], sat_el['e']) + sat_el['omega']
    lam_t = _mean_anomaly(tgt_el['theta'], tgt_el['e']) + tgt_el['omega']
    dlam = lam_s - lam_t
    out[13] = math.sin(dlam)
    out[14] = math.cos(dlam)
    out[15] = o[15]
    out[16] = math.cos(sat_el['omega'] - tgt_el['omega'])
    return np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=-1.0)


def _obs_scales():
    """(obs_alt_scale_m, lvlh_scale_m) actually in force for the selected config.

    Policy-facing normalizers, not physics. A LEO checkpoint reads obs[0]/obs[7]
    as (a - R_EARTH)/1.6e6 and the LVLH block in units of 6.371e6 m; the WL4
    wide-envelope checkpoint reads 8e6 and 1.5e7. Decoding or re-encoding with
    the wrong pair silently mis-scales every target slot, so both ends of the
    layer read the constants from the live env config instead of from
    orbital_math's LEO module defaults. The .get() fallbacks are inert for the
    three shipped configs (all three name both keys explicitly).
    """
    return (ENV_KWARGS.get('obs_alt_scale_m', om.OBS_ALT_SCALE_M),
            ENV_KWARGS.get('lvlh_scale_m', om.LVLH_SCALE_M))


def recover_states(o):
    alt_scale, _ = _obs_scales()
    if PHASE_OBS_MODE == 1:
        return recover_states_t3(o, alt_scale)
    return om.recover_states(o, alt_scale)


# Target-derived observation slots, grouped so a closed-loop loss can be
# attributed to the entry that carries it. Mode 1 moves cos(omega_s - omega_t)
# into [16] and hands [15] to the episode clock, which is chaser-side and never
# injected.
INJECT_GROUPS = {
    0: {'a_e': (7, 8), 'omega': (11, 12), 'lam': (13, 14), 'anom': (15, 16),
        'lvlh': (33, 34, 35, 36, 37)},
    1: {'a_e': (7, 8), 'omega': (11, 12, 16), 'lam': (13, 14),
        'lvlh': (33, 34, 35, 36, 37)},
}
# None = inject every target-derived slot (the only behaviour before this flag).
INJECT = None


def build_obs(o, sat_el, tgt_el, tgt_cart=None):
    fn = build_obs_t3 if PHASE_OBS_MODE == 1 else om.build_obs
    alt_scale, lvlh_scale = _obs_scales()
    out = fn(o, sat_el, tgt_el, alt_scale, lvlh_scale, tgt_cart=tgt_cart)
    if INJECT is not None:
        for g, slots in INJECT_GROUPS[PHASE_OBS_MODE].items():
            if g not in INJECT:
                for i in slots:
                    out[i] = o[i]        # this group stays truth
    return out


def _ulp_diff(a32, b32):
    """Exact float32 ULP distance, elementwise.

    A relative-error figure is meaningless for observation slots that legitimately
    sit near zero (sin/cos channels at a zero crossing), so the reconstruction
    residual is reported as a count of representable float32 values between the
    two — 0 means bit-identical, 1 means adjacent, and anything larger is a real
    disagreement rather than rounding.
    """
    ai = np.asarray(a32, dtype=np.float32).view(np.int32).astype(np.int64)
    bi = np.asarray(b32, dtype=np.float32).view(np.int32).astype(np.int64)
    ai = np.where(ai < 0, np.int64(0x80000000) - ai, ai)
    bi = np.where(bi < 0, np.int64(0x80000000) - bi, bi)
    return np.abs(ai - bi)


def make_env():
    return Orbital(**ENV_KWARGS)


def load_policy(env):
    policy = LSTMWrapper(env, Default(env))
    policy.load_state_dict(torch.load(CKPT, map_location="cpu", weights_only=True))
    policy.eval()
    return policy


def zero_state(policy):
    return {'lstm_h': torch.zeros(1, policy.hidden_size),
            'lstm_c': torch.zeros(1, policy.hidden_size)}


def run(num_episodes, mode="truth", noise_scale=1.0, q_a=Q_ACCEL_PSD,
        sigma_v0=SIGMA_V0, seed=42, meas_seed=1234, collect_traces=0,
        max_episodes_traced=0, label="", verbose=True, recon_check=0):
    """Roll out the policy for num_episodes.

    mode="truth"  policy sees the truth observation; the EKF still runs, so the
                  same pass yields the open-loop filter statistics.
    mode="ekf"    policy sees the reconstructed observation (target slots from
                  the estimate). Filter errors close the loop.

    Returns a dict of per-episode arrays plus optional per-step traces.
    """
    env = make_env()
    policy = load_policy(env)
    obs, _ = env.reset(seed=seed)
    rng = np.random.default_rng(meas_seed)
    state = zero_state(policy)

    s_rho = SIGMA_RHO_M * noise_scale
    s_beta = SIGMA_BETA_RAD * noise_scale
    ekf = TargetEKF(sigma_rho=s_rho, sigma_beta=s_beta, q_a=q_a,
                    sigma_v0=sigma_v0)

    ep_success, ep_cause, ep_len = [], [], []
    ep_pos_rmse, ep_vel_rmse = [], []
    ep_term_pos, ep_term_vel = [], []
    ep_pos_rmse_s, ep_vel_rmse_s = [], []   # steps >= SETTLE only
    all_nees, all_nis = [], []
    step_nees, step_nis = {}, {}     # step index -> list
    step_perr, step_verr, step_rho = {}, {}, {}
    v0_guess_err = []
    traces = []
    n_hyperbolic = 0     # steps whose *estimate* went non-elliptical
    ep_diverged = []     # per-episode: estimate left the elliptical regime
    hyp_this_ep = 0

    # Flat (range, error) telemetry: the step-indexed dicts above answer "how does
    # the filter behave over time", which is the wrong axis once the envelope is
    # wide. A 1 mrad bearing sigma is a *transverse position* sigma of rho * 1e-3,
    # so accuracy is a function of separation, not of step number — at WL4 drift
    # legs the two differ by three orders of magnitude.
    flat_rho, flat_pe, flat_ve, flat_sigp = [], [], [], []
    # recon_check: dim-wise |recon - truth| and whether the policy's argmax moves.
    recon_maxabs = np.zeros(38)
    recon_exact = 0
    recon_steps = 0
    recon_action_mismatch = 0
    recon_ulp_max = 0.0

    fresh = True
    k = 0
    ep_pe, ep_ve = [], []
    trace = None
    episodes = 0
    t0 = time.time()

    while episodes < num_episodes:
        o = np.array(obs[0], dtype=np.float32, copy=True)
        sat_el, tgt_el = recover_states(o)
        sat_cart = om.orbit_to_cartesian(sat_el)
        tgt_cart = om.orbit_to_cartesian(tgt_el)

        rho, beta = measure(sat_cart, tgt_cart, rng, s_rho, s_beta)

        if fresh:
            ekf.initialize(sat_cart, rho, beta)
            v0_guess_err.append(ekf.x[2] - tgt_cart[2])
            v0_guess_err.append(ekf.x[3] - tgt_cart[3])
            nis = None
            fresh = False
            k = 0
            ep_pe, ep_ve = [], []
            trace = {'t': [], 'err': [], 'sig': []} if (
                collect_traces and len(traces) < max_episodes_traced) else None
        else:
            nu, S = ekf.update(sat_cart, rho, beta)
            nis = float(nu @ np.linalg.solve(S, nu)) / 2.0

        err = ekf.x - np.asarray(tgt_cart)
        pe = math.hypot(err[0], err[1])
        ve = math.hypot(err[2], err[3])
        nees = ekf.nees(tgt_cart)

        rho_true = math.hypot(tgt_cart[0] - sat_cart[0], tgt_cart[1] - sat_cart[1])
        ep_pe.append(pe)
        ep_ve.append(ve)
        all_nees.append(nees)
        step_nees.setdefault(k, []).append(nees)
        step_perr.setdefault(k, []).append(pe)
        step_verr.setdefault(k, []).append(ve)
        step_rho.setdefault(k, []).append(rho_true)
        flat_rho.append(rho_true)
        flat_pe.append(pe)
        flat_ve.append(ve)
        sg = ekf.sigmas()
        flat_sigp.append(math.hypot(sg[0], sg[1]))
        if nis is not None:
            all_nis.append(nis)
            step_nis.setdefault(k, []).append(nis)
        if trace is not None:
            trace['t'].append(k)
            trace['err'].append(err.copy())
            trace['sig'].append(ekf.sigmas())

        # ── policy observation ──────────────────────────────────────────────
        if mode == "truth":
            pol_obs = o
        elif mode == "recon":
            # Control: same reconstruction path, zero estimation error. Isolates
            # the obs-rebuild from the filter, and must reproduce truth exactly.
            pol_obs = build_obs(o, sat_el, tgt_el)
        else:
            est_el = om.cartesian_to_elements(*ekf.x)
            if est_el['a'] <= 0.0:
                n_hyperbolic += 1
                hyp_this_ep += 1
            pol_obs = build_obs(o, sat_el, est_el, tgt_cart=tuple(ekf.x))

        # forward_eval mutates `state` in place, so the recon probe forward must
        # run against a snapshot taken *before* the real one.
        state_snapshot = ({'lstm_h': state['lstm_h'].clone(),
                           'lstm_c': state['lstm_c'].clone()}
                          if recon_check else None)

        with torch.no_grad():
            logits, _ = policy.forward_eval(
                torch.from_numpy(pol_obs).float().unsqueeze(0).unsqueeze(0), state)
            action = int(torch.argmax(logits, dim=-1).item())

        if recon_check:
            # Zero-estimation-error control on the encode/decode layer alone:
            # decode the truth obs into elements, re-encode, compare. Any scale
            # constant that does not match the env shows up here as a gross,
            # not a rounding, error — and the argmax comparison converts the
            # residual into the only currency the closed loop cares about.
            rec = build_obs(o, sat_el, tgt_el)
            d = np.abs(rec.astype(np.float64) - o.astype(np.float64))
            recon_maxabs = np.maximum(recon_maxabs, d)
            recon_steps += 1
            if not np.any(d):
                recon_exact += 1
            else:
                recon_ulp_max = max(recon_ulp_max, float(_ulp_diff(rec, o).max()))
            with torch.no_grad():
                lg2, _ = policy.forward_eval(
                    torch.from_numpy(rec).float().unsqueeze(0).unsqueeze(0),
                    state_snapshot)
                if int(torch.argmax(lg2, dim=-1).item()) != action:
                    recon_action_mismatch += 1

        obs, rewards, terms, truncs, _ = env.step(np.array([action], dtype=np.int32))
        k += 1

        if terms[0]:
            episodes += 1
            sim_steps, cause = env.last_episode_result(0)
            ep_cause.append(int(cause))
            ep_success.append(1 if cause == 1 else 0)
            ep_len.append(int(sim_steps))
            ep_pos_rmse.append(float(np.sqrt(np.mean(np.square(ep_pe)))))
            ep_vel_rmse.append(float(np.sqrt(np.mean(np.square(ep_ve)))))
            ep_term_pos.append(ep_pe[-1])
            ep_term_vel.append(ep_ve[-1])
            ep_diverged.append(1 if hyp_this_ep else 0)
            hyp_this_ep = 0
            tail_p, tail_v = ep_pe[SETTLE:], ep_ve[SETTLE:]
            if tail_p:
                ep_pos_rmse_s.append(float(np.sqrt(np.mean(np.square(tail_p)))))
                ep_vel_rmse_s.append(float(np.sqrt(np.mean(np.square(tail_v)))))
            if trace is not None:
                trace['err'] = np.array(trace['err'])
                trace['sig'] = np.array(trace['sig'])
                trace['t'] = np.array(trace['t'])
                traces.append(trace)
            fresh = True
            state = zero_state(policy)
            if verbose and episodes % 25 == 0:
                sr = float(np.mean(ep_success))
                print(f"    [{label}] ep {episodes}/{num_episodes} "
                      f"success={sr:.1%} "
                      f"pos_rmse={np.mean(ep_pos_rmse):.1f} m "
                      f"({time.time()-t0:.0f}s)", flush=True)
        else:
            dt_total = om.ACTION_TAU[action] * om.DT
            if SENSOR_DT > 0.0 and dt_total > SENSOR_DT + 1e-9:
                # Guidance decided once; navigation keeps running. A warp applies
                # no impulse, so both truth states simply coast, and
                # propagate_cartesian is the same exact Kepler map the env
                # sub-steps with — no env access and no dynamics mismatch.
                n_sub = max(1, int(round(dt_total / SENSOR_DT)))
                h = dt_total / n_sub
                s_c, t_c = sat_cart, tgt_cart
                for i in range(n_sub):
                    s_c = om.propagate_cartesian(s_c, h)
                    t_c = om.propagate_cartesian(t_c, h)
                    ekf.predict(h)
                    if i < n_sub - 1:
                        # The last sub-interval's update is the next loop
                        # iteration's, taken against the env's own truth obs, so
                        # the NEES/NIS tallies stay decision-epoch statistics.
                        r_i, b_i = measure(s_c, t_c, rng, s_rho, s_beta)
                        ekf.update(s_c, r_i, b_i)
            else:
                ekf.predict(dt_total)

    env.close()
    return dict(
        label=label, mode=mode, noise_scale=noise_scale, q_a=q_a,
        success=np.array(ep_success), cause=np.array(ep_cause),
        length=np.array(ep_len),
        pos_rmse=np.array(ep_pos_rmse), vel_rmse=np.array(ep_vel_rmse),
        pos_rmse_s=np.array(ep_pos_rmse_s), vel_rmse_s=np.array(ep_vel_rmse_s),
        term_pos=np.array(ep_term_pos), term_vel=np.array(ep_term_vel),
        nees=np.array(all_nees), nis=np.array(all_nis),
        step_nees=step_nees, step_nis=step_nis,
        step_perr=step_perr, step_verr=step_verr, step_rho=step_rho,
        v0_guess_err=np.array(v0_guess_err), traces=traces,
        n_hyperbolic=n_hyperbolic, diverged=np.array(ep_diverged),
        flat_rho=np.array(flat_rho), flat_pe=np.array(flat_pe),
        flat_ve=np.array(flat_ve), flat_sigp=np.array(flat_sigp),
        recon_maxabs=recon_maxabs, recon_exact=recon_exact,
        recon_steps=recon_steps, recon_action_mismatch=recon_action_mismatch,
        recon_ulp_max=recon_ulp_max,
        wall_s=time.time() - t0,
    )


def in_bounds(v, lo, hi):
    v = np.asarray(v)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float('nan')
    return float(np.mean((v >= lo) & (v <= hi)))


def bounds_split(v, lo, hi):
    """(in, below, above). 'below' = conservative filter, 'above' = overconfident.

    Only the 'above' tail is a safety problem: it means the reported covariance
    understates the true error."""
    v = np.asarray(v)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float('nan'), float('nan'), float('nan')
    return (float(np.mean((v >= lo) & (v <= hi))),
            float(np.mean(v < lo)), float(np.mean(v > hi)))


RANGE_EDGES = (0.0, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9)


def range_report(r, title="filter accuracy vs separation"):
    """Bin the per-step filter error by true chaser-target range.

    The point of the table is the `err/(sigma_b*rho)` column. The bearing channel
    contributes a *transverse* position uncertainty of rho * sigma_beta, so a
    filter that is doing nothing but inverting the current measurement sits at
    ~1.0 there; a filter that is genuinely integrating a dynamics model over many
    epochs sits well below 1. That ratio, not the raw metre count, is what says
    whether the sensor suite is adequate at a given separation.
    """
    rho, pe, ve, sp = r['flat_rho'], r['flat_pe'], r['flat_ve'], r['flat_sigp']
    if rho.size == 0:
        return []
    s_beta = SIGMA_BETA_RAD * r['noise_scale']
    s_rho = SIGMA_RHO_M * r['noise_scale']
    print(f"  {title} (sigma_rho={s_rho:.0f} m, sigma_beta={s_beta:.1e} rad):")
    print("    range bin        n      med rho     med |dp|   sigma_b*rho  "
          "err/(sb*rho)  med |dv|   med sigma_p")
    rows = []
    for lo, hi in zip(RANGE_EDGES[:-1], RANGE_EDGES[1:]):
        m = (rho >= lo) & (rho < hi)
        n = int(m.sum())
        if n < 20:
            continue
        mr, mp = float(np.median(rho[m])), float(np.median(pe[m]))
        trans = mr * s_beta
        row = dict(lo=lo, hi=hi, n=n, med_rho=mr, med_pos=mp, trans=trans,
                   ratio=mp / max(trans, 1e-12),
                   med_vel=float(np.median(ve[m])),
                   med_sig=float(np.median(sp[m])))
        rows.append(row)
        print(f"    {lo:8.0e}-{hi:<8.0e} {n:7d} {mr/1e3:10.1f} km "
              f"{mp:11.1f} m {trans:11.1f} m {row['ratio']:11.3f} "
              f"{row['med_vel']:10.4f} m/s {row['med_sig']:11.1f} m")
    return rows


def stage_reconcheck(args):
    """Verify the encode/decode layer with zero estimation error.

    Rebuild every target-derived slot from the states decoded out of the truth
    observation and compare against that same truth observation, dim by dim, plus
    the policy's argmax on both. This is the test that catches a wrong
    obs_alt_scale_m / lvlh_scale_m: a 5x scale error is a ~0.5 absolute residual
    on obs[7], four million times the float32 round-trip floor.
    """
    print("── Stage: reconcheck (encode/decode layer, zero estimation error) ──")
    alt, lv = _obs_scales()
    print(f"  scales in force: obs_alt_scale_m={alt:g}  lvlh_scale_m={lv:g}")
    r = run(args.recon_eps, mode="truth", label="reconcheck", seed=args.seed,
            q_a=args.q_a if args.q_a else Q_ACCEL_PSD,
            sigma_v0=args.sigma_v0, recon_check=1, verbose=False)
    ma = r['recon_maxabs']
    worst = int(np.argmax(ma))
    tgt_idx = INJECT_GROUPS[PHASE_OBS_MODE]
    tgt_slots = sorted(i for v in tgt_idx.values() for i in v)
    print(f"  episodes {len(r['success'])}, steps compared {r['recon_steps']}")
    print(f"  bit-identical steps        {r['recon_exact']}/{r['recon_steps']} "
          f"= {r['recon_exact']/max(r['recon_steps'],1):.3%}")
    print(f"  max |recon - truth| any dim  {ma.max():.3e}  (dim {worst})")
    print(f"  max |recon - truth| target   "
          f"{max(ma[i] for i in tgt_slots):.3e}")
    print(f"  max residual / eps_f32(1.0)  {ma.max()/float(np.spacing(np.float32(1.0))):.2f}")
    print(f"  max residual, float32 ULP    {r['recon_ulp_max']:.0f}  "
          f"(large only where the slot itself is ~0: a sin/cos channel at a "
          f"zero crossing has ULPs of ~1e-45)")
    print(f"  mean decisions/episode       "
          f"{r['recon_steps']/max(len(r['success']),1):.1f}  "
          f"(mean sim length {r['length'].mean():.0f} sub-steps)")
    print(f"  policy argmax mismatches     {r['recon_action_mismatch']}/"
          f"{r['recon_steps']}")
    nz = [(i, ma[i]) for i in range(38) if ma[i] > 0]
    if nz:
        print("  non-zero dims: " +
              ", ".join(f"[{i}]={v:.2e}" for i, v in nz))
    return r


# ── stages ───────────────────────────────────────────────────────────────────
def stage_sanity(args):
    print("── Stage: sanity (truth-obs control, harness anchor) ──────────")
    r = run(args.sanity_eps, mode="truth", label="sanity", seed=args.seed)
    n_gave = int(np.sum(r['cause'] == 7))
    n_valid = len(r['success']) - n_gave
    sr = r['success'].sum() / max(n_valid, 1)
    print(f"  success {r['success'].sum()}/{n_valid} = {sr:.1%}   "
          f"(gave_up {n_gave})")
    print(f"  mean ep length {r['length'].mean():.0f} steps, "
          f"wall {r['wall_s']:.0f}s")
    print(f"  passive-filter pos RMSE {np.mean(r['pos_rmse']):.1f} m, "
          f"vel RMSE {np.mean(r['vel_rmse']):.3f} m/s")
    print(f"  circular-guess v0 error RMS {np.sqrt(np.mean(r['v0_guess_err']**2)):.1f} m/s")
    print(f"  NEES in-bounds {in_bounds(r['nees'], NEES_LO, NEES_HI):.3f}  "
          f"NIS in-bounds {in_bounds(r['nis'], NIS_LO, NIS_HI):.3f}")
    return r


def stage_qsweep(args):
    """Select (q_a, sigma_v0) on NEES in-bounds fraction.

    Dynamics are exact, so q_a has no model error to absorb — it exists purely
    as a covariance floor. sigma_v0 must match the actual spread of the
    circular-velocity init guess, which the sweep also measures directly.
    """
    print("── Stage: qsweep (Q / sigma_v0 selection via NEES) ────────────")
    rows = []
    for sv in args.sigma_v0_grid:
        for q in args.q_grid:
            r = run(args.qsweep_eps, mode="truth", q_a=q, sigma_v0=sv,
                    label=f"q={q:g}", seed=args.seed, verbose=False)
            fn = in_bounds(r['nees'], NEES_LO, NEES_HI)
            fi = in_bounds(r['nis'], NIS_LO, NIS_HI)
            v0rms = float(np.sqrt(np.mean(r['v0_guess_err'] ** 2)))
            rows.append(dict(q_a=q, sigma_v0=sv, nees_ib=fn, nis_ib=fi,
                             nees_med=float(np.nanmedian(r['nees'])),
                             pos=float(np.mean(r['pos_rmse'])),
                             vel=float(np.mean(r['vel_rmse'])), v0rms=v0rms))
            _, blo, bhi = bounds_split(r['nees'], NEES_LO, NEES_HI)
            rows[-1]['nees_below'], rows[-1]['nees_above'] = blo, bhi
            print(f"  sigma_v0={sv:6.0f}  q_a={q:8.1e}  NEES in-bounds {fn:.3f}"
                  f"  (below {blo:.3f} / above {bhi:.3f})"
                  f"  median {rows[-1]['nees_med']:6.3f}   NIS in-bounds {fi:.3f}"
                  f"   pos RMSE {rows[-1]['pos']:7.1f} m", flush=True)
    print(f"  measured circular-guess v0 error RMS: {rows[-1]['v0rms']:.1f} m/s")
    best = max(rows, key=lambda d: d['nees_ib'])
    print(f"  -> selected q_a={best['q_a']:g}, sigma_v0={best['sigma_v0']:g} "
          f"(NEES in-bounds {best['nees_ib']:.3f}, NIS {best['nis_ib']:.3f})")
    return best['q_a'], best['sigma_v0'], rows


def stage_validate(args, q_a):
    print("── Stage: validate (open-loop filter Monte Carlo) ─────────────")
    r = run(args.validate_eps, mode="truth", q_a=q_a, sigma_v0=args.sigma_v0,
            label="validate", seed=args.seed, collect_traces=1,
            max_episodes_traced=3)
    fn, nlo, nhi = bounds_split(r['nees'], NEES_LO, NEES_HI)
    fi, ilo, ihi = bounds_split(r['nis'], NIS_LO, NIS_HI)
    settled = np.concatenate([np.asarray(r['step_nees'][k])
                              for k in sorted(r['step_nees']) if k >= 10])
    print(f"  episodes {len(r['success'])}, filter updates {len(r['nis'])}, "
          f"mean sim length {r['length'].mean():.0f} steps")
    print(f"  NEES  in-bounds {fn:.3f} (below {nlo:.3f} / above {nhi:.3f})  "
          f"median {np.nanmedian(r['nees']):.3f}   bounds [{NEES_LO:.3f}, {NEES_HI:.3f}]")
    print(f"  NEES  in-bounds after step 10: "
          f"{in_bounds(settled, NEES_LO, NEES_HI):.3f}")
    print(f"  NIS   in-bounds {fi:.3f} (below {ilo:.3f} / above {ihi:.3f})  "
          f"median {np.nanmedian(r['nis']):.3f}   bounds [{NIS_LO:.3f}, {NIS_HI:.3f}]")
    print(f"  pos RMSE mean {np.mean(r['pos_rmse']):.1f} m  median "
          f"{np.median(r['pos_rmse']):.1f} m")
    print(f"  vel RMSE mean {np.mean(r['vel_rmse']):.3f} m/s  median "
          f"{np.median(r['vel_rmse']):.3f} m/s")
    print(f"  settled (step >= {SETTLE}) pos RMSE {np.mean(r['pos_rmse_s']):.1f} m, "
          f"vel RMSE {np.mean(r['vel_rmse_s']):.4f} m/s")
    print(f"  terminal pos err mean {np.mean(r['term_pos']):.1f} m, "
          f"vel err mean {np.mean(r['term_vel']):.3f} m/s")
    range_report(r)
    make_plots(r)
    return r


def stage_eval(args, q_a):
    print("── Stage: eval (closed loop) ──────────────────────────────────")
    conds = [("truth", "truth", 0.0),
             ("recon", "recon", 0.0),
             ("ekf_1x", "ekf", 1.0),
             ("ekf_3x", "ekf", 3.0),
             ("ekf_10x", "ekf", 10.0)]
    conds += [(f"ekf_{int(n)}x", "ekf", float(n)) for n in args.extra_noise]
    out = []
    for name, mode, ns in conds:
        r = run(args.eval_eps, mode=mode, noise_scale=max(ns, 1.0), q_a=q_a,
                sigma_v0=args.sigma_v0, seed=args.seed, label=name)
        r['cond'] = name + COND_SUFFIX
        r['applied_noise'] = ns
        out.append(r)
        n_gave = int(np.sum(r['cause'] == 7))
        n_valid = len(r['success']) - n_gave
        dv = float(np.mean(r['diverged'])) if r['diverged'].size else 0.0
        causes = ", ".join(f"{CAUSE_NAMES[c]}={int(np.sum(r['cause'] == c))}"
                           for c in range(8) if np.sum(r['cause'] == c))
        print(f"  {name:8s} success {r['success'].sum()}/{n_valid} = "
              f"{r['success'].sum()/max(n_valid,1):.1%}   "
              f"pos RMSE med {np.median(r['pos_rmse']):8.1f} m "
              f"(settled med {np.median(r['pos_rmse_s']):8.1f} m)   "
              f"vel RMSE med {np.median(r['vel_rmse']):7.3f} m/s   "
              f"diverged {dv:.1%}   ({r['wall_s']:.0f}s)")
        print(f"           causes: {causes}")
        if mode == "ekf" and args.range_report:
            range_report(r, title=f"{name}: filter accuracy vs separation")
    write_csv(out)
    plot_success(out)
    return out


# ── outputs ──────────────────────────────────────────────────────────────────
def write_csv(runs):
    os.makedirs(os.path.dirname(RESULTS_CSV), exist_ok=True)
    with open(RESULTS_CSV, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['condition', 'obs_source', 'sigma_rho_m', 'sigma_beta_rad',
                    'episodes', 'gave_up', 'successes', 'success_rate',
                    'mean_ep_len', 'pos_rmse_mean_m', 'pos_rmse_median_m',
                    'pos_rmse_p95_m', 'vel_rmse_mean_ms', 'vel_rmse_median_ms',
                    'pos_rmse_settled_m', 'vel_rmse_settled_ms',
                    'pos_rmse_settled_median_m', 'vel_rmse_settled_median_ms',
                    'term_pos_err_mean_m', 'term_vel_err_mean_ms',
                    'nees_in_bounds', 'nis_in_bounds', 'hyperbolic_est_steps',
                    'frac_episodes_diverged',
                    'cause_success', 'cause_collision', 'cause_escape',
                    'cause_safety_cap', 'cause_stranded', 'cause_hyperbolic'])
        for r in runs:
            n_gave = int(np.sum(r['cause'] == 7))
            n_valid = len(r['success']) - n_gave
            ns = r['applied_noise']
            src = r['mode']
            w.writerow([
                r['cond'], src,
                f"{SIGMA_RHO_M * r['noise_scale']:.1f}",
                f"{SIGMA_BETA_RAD * r['noise_scale']:.1e}",
                len(r['success']), n_gave, int(r['success'].sum()),
                f"{r['success'].sum()/max(n_valid,1):.4f}",
                f"{r['length'].mean():.1f}",
                f"{np.mean(r['pos_rmse']):.2f}", f"{np.median(r['pos_rmse']):.2f}",
                f"{np.percentile(r['pos_rmse'], 95):.2f}",
                f"{np.mean(r['vel_rmse']):.4f}", f"{np.median(r['vel_rmse']):.4f}",
                f"{np.mean(r['pos_rmse_s']):.2f}", f"{np.mean(r['vel_rmse_s']):.4f}",
                f"{np.median(r['pos_rmse_s']):.2f}", f"{np.median(r['vel_rmse_s']):.4f}",
                f"{np.mean(r['term_pos']):.2f}", f"{np.mean(r['term_vel']):.4f}",
                f"{in_bounds(r['nees'], NEES_LO, NEES_HI):.4f}",
                f"{in_bounds(r['nis'], NIS_LO, NIS_HI):.4f}", r['n_hyperbolic'],
                f"{float(np.mean(r['diverged'])) if r['diverged'].size else 0.0:.4f}",
                *[int(np.sum(r['cause'] == c)) for c in (1, 2, 3, 4, 5, 6)],
            ])
    print(f"  wrote {RESULTS_CSV}")


def _band(step_dict, max_k):
    ks = sorted(k for k in step_dict if k <= max_k and len(step_dict[k]) >= 5)
    med = np.array([np.nanmedian(step_dict[k]) for k in ks])
    lo = np.array([np.nanpercentile(step_dict[k], 5) for k in ks])
    hi = np.array([np.nanpercentile(step_dict[k], 95) for k in ks])
    return np.array(ks), med, lo, hi


def make_plots(r):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(PLOT_DIR, exist_ok=True)
    max_k = int(np.percentile(r['length'], 75))

    # NEES vs step
    ks, med, lo, hi = _band(r['step_nees'], max_k)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.fill_between(ks, lo, hi, alpha=0.25, color='C0', label='5-95 pct')
    ax.plot(ks, med, color='C0', lw=1.4, label='median NEES')
    ax.axhline(NEES_LO, ls='--', c='r', lw=1)
    ax.axhline(NEES_HI, ls='--', c='r', lw=1,
               label=f'95% chi2(4) bounds [{NEES_LO:.2f}, {NEES_HI:.2f}]')
    ax.axhline(1.0, ls=':', c='k', lw=1, label='consistent = 1')
    ax.set_yscale('log'); ax.set_xlabel('step'); ax.set_ylabel('NEES / 4')
    ax.set_title(f"EKF NEES consistency ({len(r['success'])} episodes, "
                 f"in-bounds {in_bounds(r['nees'], NEES_LO, NEES_HI):.1%})")
    ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(f"{PLOT_DIR}/nees.png", dpi=140); plt.close(fig)

    # NIS vs step
    ks, med, lo, hi = _band(r['step_nis'], max_k)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.fill_between(ks, lo, hi, alpha=0.25, color='C2', label='5-95 pct')
    ax.plot(ks, med, color='C2', lw=1.4, label='median NIS')
    ax.axhline(NIS_LO, ls='--', c='r', lw=1)
    ax.axhline(NIS_HI, ls='--', c='r', lw=1,
               label=f'95% chi2(2) bounds [{NIS_LO:.2f}, {NIS_HI:.2f}]')
    ax.axhline(1.0, ls=':', c='k', lw=1, label='consistent = 1')
    ax.set_yscale('log'); ax.set_xlabel('step'); ax.set_ylabel('NIS / 2')
    ax.set_title(f"EKF NIS consistency (in-bounds "
                 f"{in_bounds(r['nis'], NIS_LO, NIS_HI):.1%})")
    ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(f"{PLOT_DIR}/nis.png", dpi=140); plt.close(fig)

    # RMSE vs time
    ksp, medp, lop, hip = _band(r['step_perr'], max_k)
    ksv, medv, lov, hiv = _band(r['step_verr'], max_k)
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    axes[0].fill_between(ksp, lop, hip, alpha=0.25, color='C0')
    axes[0].plot(ksp, medp, color='C0', lw=1.4)
    axes[0].set_yscale('log'); axes[0].set_ylabel('|pos error| (m)')
    axes[0].set_title('Target-state estimation error vs time (median, 5-95 pct)')
    axes[1].fill_between(ksv, lov, hiv, alpha=0.25, color='C3')
    axes[1].plot(ksv, medv, color='C3', lw=1.4)
    axes[1].set_yscale('log'); axes[1].set_ylabel('|vel error| (m/s)')
    axes[1].set_xlabel('step (60 s each)')
    ksr, medr, _, _ = _band(r['step_rho'], max_k)
    for ax in axes:
        tw = ax.twinx()
        tw.plot(ksr, medr / 1e3, color='0.5', lw=1.0, ls=':')
        tw.set_yscale('log'); tw.set_ylabel('median true range (km)', color='0.4')
        tw.tick_params(axis='y', colors='0.4')
    fig.tight_layout(); fig.savefig(f"{PLOT_DIR}/rmse_vs_time.png", dpi=140)
    plt.close(fig)

    # error vs +-3 sigma, representative episodes
    names = ['x (m)', 'y (m)', 'vx (m/s)', 'vy (m/s)']
    for i, tr in enumerate(r['traces']):
        fig, axes = plt.subplots(4, 1, figsize=(8, 9), sharex=True)
        for j, ax in enumerate(axes):
            ax.plot(tr['t'], tr['err'][:, j], lw=1.0, color='C0', label='error')
            ax.plot(tr['t'], 3 * tr['sig'][:, j], lw=1.0, ls='--', color='r',
                    label=r'$\pm 3\sigma$')
            ax.plot(tr['t'], -3 * tr['sig'][:, j], lw=1.0, ls='--', color='r')
            ax.set_yscale('symlog', linthresh=(1.0 if j < 2 else 1e-3))
            ax.set_ylabel(names[j]); ax.grid(alpha=0.25)
            if j == 0:
                ax.legend(fontsize=8)
        axes[0].set_title(f'Estimation error vs $\\pm3\\sigma$ — episode {i+1}')
        axes[-1].set_xlabel('step (60 s each)')
        fig.tight_layout()
        fig.savefig(f"{PLOT_DIR}/err_3sigma_ep{i+1}.png", dpi=140); plt.close(fig)

    print(f"  wrote plots to {PLOT_DIR}/")


def plot_success(runs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(PLOT_DIR, exist_ok=True)
    labels, rates, errs = [], [], []
    for r in runs:
        n_gave = int(np.sum(r['cause'] == 7))
        n = len(r['success']) - n_gave
        p = r['success'].sum() / max(n, 1)
        labels.append(r['cond'])
        rates.append(100 * p)
        errs.append(100 * math.sqrt(max(p * (1 - p), 1e-9) / max(n, 1)))
    fig, ax = plt.subplots(figsize=(6.5, 4))
    cols = ['0.4', '0.6'] + [f'C{i}' for i in range(len(labels) - 2)]
    ax.bar(labels, rates, yerr=errs, capsize=4, color=cols[:len(labels)])
    ax.tick_params(axis='x', labelrotation=20)
    for i, v in enumerate(rates):
        ax.text(i, v + 1.5, f"{v:.1f}%", ha='center', fontsize=10)
    ax.set_ylabel('rendezvous success rate (%)')
    ax.set_ylim(0, 105)
    ax.set_title('Policy on estimated state — sensor-noise sweep')
    fig.tight_layout(); fig.savefig(f"{PLOT_DIR}/success_vs_noise.png", dpi=140)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--stage', default='all',
                   choices=['sanity', 'reconcheck', 'qsweep', 'validate',
                            'eval', 'all'])
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--sanity-eps', type=int, default=50)
    p.add_argument('--recon-eps', type=int, default=10)
    p.add_argument('--range-report', action='store_true',
                   help='per-condition filter-accuracy-vs-separation table in '
                        'the eval stage')
    p.add_argument('--qsweep-eps', type=int, default=30)
    p.add_argument('--validate-eps', type=int, default=100)
    p.add_argument('--eval-eps', type=int, default=200)
    p.add_argument('--q-a', type=float, default=None,
                   help='skip qsweep and use this process-noise PSD')
    p.add_argument('--sigma-v0', type=float, default=SIGMA_V0)
    p.add_argument('--q-grid', type=float, nargs='*',
                   default=[1e-13, 1e-12, 1e-11, 1e-10, 1e-9, 1e-8])
    p.add_argument('--sigma-v0-grid', type=float, nargs='*',
                   default=[SIGMA_V0])
    p.add_argument('--extra-noise', type=float, nargs='*', default=[30, 100, 300],
                   help='degradation-curve points beyond the headline 1/3/10x')
    p.add_argument('--ckpt', default=None,
                   help='policy checkpoint (default: the T2 legacy 10-action ckpt, '
                        'or the T3 headline ckpt under --t3)')
    p.add_argument('--t3', action='store_true',
                   help='corrected-dynamics T3 config: shaping_mode=1, '
                        'shape_gamma=1, phase_gap_mode=1, phase_obs_mode=1, '
                        'cap 3000 @ reward 0, Discrete(16) head, and the '
                        'mean-longitude obs[13-16] injection layout')
    p.add_argument('--wide', action='store_true',
                   help='T4 wide-envelope WL4 config: same obs layout as --t3 '
                        'but 300-8000 km altitudes, e_target<=0.30, de_max=0.08, '
                        'da_max=600 km, cap 6000, and the WIDE observation '
                        'normalizers obs_alt_scale_m=8e6 / lvlh_scale_m=1.5e7')
    p.add_argument('--results-csv', default=None)
    p.add_argument('--plot-dir', default=None)
    p.add_argument('--sensor-dt', type=float, default=0.0,
                   help='sensor sampling period in s; 0 (default) welds the '
                        'measurement cadence to the decision cadence')
    p.add_argument('--cond-suffix', default='',
                   help='appended to condition names in the results CSV')
    p.add_argument('--inject', default=None,
                   help="comma-separated subset of target-obs groups to feed "
                        "from the estimate (a_e, omega, lam, anom, lvlh); the "
                        "rest stay truth. Default: all of them.")
    args = p.parse_args()
    if args.t3 and args.wide:
        p.error("--t3 and --wide are mutually exclusive (different obs scales)")

    global CKPT, ENV_KWARGS, PHASE_OBS_MODE, RESULTS_CSV, PLOT_DIR, SENSOR_DT
    global COND_SUFFIX, INJECT
    SENSOR_DT = args.sensor_dt
    COND_SUFFIX = args.cond_suffix
    if args.inject:
        INJECT = set(s.strip() for s in args.inject.split(',') if s.strip())
    if args.t3:
        ENV_KWARGS = T3_ENV_KWARGS
        PHASE_OBS_MODE = 1
        CKPT = T3_CKPT
        RESULTS_CSV = T3_RESULTS_CSV
        PLOT_DIR = T3_PLOT_DIR
    if args.wide:
        ENV_KWARGS = WIDE_ENV_KWARGS
        PHASE_OBS_MODE = 1        # same corrected-dynamics obs layout as T3
        CKPT = WIDE_CKPT
        RESULTS_CSV = WIDE_RESULTS_CSV
        PLOT_DIR = WIDE_PLOT_DIR
    if args.ckpt:
        CKPT = args.ckpt
    if args.results_csv:
        RESULTS_CSV = args.results_csv
    if args.plot_dir:
        PLOT_DIR = args.plot_dir
    _cfg = ('T4 wide-envelope WL4' if args.wide else
            'T3 corrected-dynamics' if args.t3 else 'legacy T2')
    _alt, _lv = _obs_scales()
    print(f"  ckpt      {CKPT}")
    print(f"  config    {_cfg} (phase_obs_mode={PHASE_OBS_MODE})")
    print(f"  scales    obs_alt_scale_m={_alt:g}  lvlh_scale_m={_lv:g}")
    print(f"  sensor    {'1 measurement per decision (legacy)' if SENSOR_DT <= 0 else f'{SENSOR_DT:g} s cadence'}")
    if INJECT is not None:
        print(f"  inject    {sorted(INJECT)} (all other target slots stay truth)")

    torch.manual_seed(args.seed)
    torch.set_num_threads(1)

    if args.stage == 'sanity':
        stage_sanity(args); return
    if args.stage == 'reconcheck':
        stage_reconcheck(args); return

    q_a, sv0 = args.q_a, args.sigma_v0
    if args.stage in ('qsweep', 'all') and q_a is None:
        q_a, sv0, _ = stage_qsweep(args)
    if q_a is None:
        q_a = Q_ACCEL_PSD
    args.sigma_v0 = sv0
    if args.stage == 'qsweep':
        return

    if args.stage in ('validate', 'all'):
        stage_validate(args, q_a)
    if args.stage in ('eval', 'all'):
        stage_eval(args, q_a)


if __name__ == '__main__':
    main()
