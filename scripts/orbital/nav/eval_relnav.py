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

_HERE = os.path.dirname(os.path.abspath(__file__))
# Repository root of THIS checkout (works in the ext-nav worktree and in the
# main tree). The pufferlib import, results and plots must resolve here so a
# worktree run never writes into the main checkout; only `experiments/`
# checkpoints, which are untracked, fall back to the main tree.
ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
MAIN = "/Users/pete/space_training"
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(ROOT, "pufferlib"))
sys.path.insert(0, os.path.join(ROOT, "scripts", "orbital", "ext_recon"))

from pufferlib.ocean.orbital.orbital import Orbital          # noqa: E402
from pufferlib.models import Default, LSTMWrapper            # noqa: E402

import orbital_math as om                                    # noqa: E402
from ekf import (TargetEKF, measure, wrap_pi, NEES_LO, NEES_HI,  # noqa: E402
                 NIS_LO, NIS_HI, SIGMA_RHO_M, SIGMA_BETA_RAD,
                 Q_ACCEL_PSD, SIGMA_V0)
CKPT = f"{MAIN}/pufferlib/experiments/puffer_orbital_177765503091/model_puffer_orbital_000325.pt"
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
T3_CKPT = (f"{MAIN}/pufferlib/experiments/puffer_orbital_178642097817/"
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


# ═════════════════════════════════════════════════════════════════════════════
# Bearings-only (angles-only) closed loop — NAV-G's shipped filter, for real
# ═════════════════════════════════════════════════════════════════════════════
# The range row is deleted from the measurement. What is left is a scalar
# inertial line-of-sight bearing, and the target's range must be recovered from
# the nonlinearity of gravity over the separation plus whatever parallax the
# chaser's own maneuvers generate.
#
# TRAINING uses a calibrated surrogate for the acquisition step, because the
# batch solver is 17-90x the entire training step (red-team BLOCKER-1). EVAL —
# this code — runs the REAL `bls_acquire_adaptive`: dense log-spaced range grid
# over the analytic feasible set, binned multi-start Levenberg-damped
# Gauss-Newton, arc grown x1.6 until a chi-square gate, an ambiguity-margin
# gate and a covariance gate all pass. 200 episodes x ~0.55 s is ~2 minutes.
#
# Carry the red-team MAJOR-2 caveat with any number produced here: those three
# gates test the self-consistency of the chosen range basin, NOT the
# correctness of the basin choice. At reduced grid density the `minimal` config
# passed 18/18 gates while returning a 2,951 km epoch solution at G2. The gates
# are not an acceptance test; the grid knobs below are the shipped ones and
# must not be tuned without re-measuring epoch error against truth.
# Tsiolkovsky inversion of obs[6]. obs[6] is fuel_mass/(dry+fuel); the tank
# starts at FUEL_FRAC = 0.15, so the cumulative delta-v ACTUALLY APPLIED is
#     dv = VE * ln( (1 - obs[6]) / (1 - 0.15) ),   VE = ISP*G0 = 2942 m/s
# This reads the mass the C env really burned rather than the delta-v the
# policy commanded, so a fuel-limited burn is counted correctly, and it
# saturates at 478 m/s — the documented budget.
VE_MS = 300.0 * 9.80665
FUEL_FRAC0 = 0.15


def dv_spent(fuel_frac):
    return VE_MS * math.log(max(1.0 - float(fuel_frac), 1e-9) / (1.0 - FUEL_FRAC0))


def fuel_budget_left(fuel_frac):
    """Remaining fraction of the delta-v BUDGET (1.0 = full tank)."""
    f = max(min(float(fuel_frac), FUEL_FRAC0), 0.0)
    return (f / max(1.0 - f, 1e-9)) / (FUEL_FRAC0 / (1.0 - FUEL_FRAC0))


def _is_burn(a):
    """Any tau==1 non-coast action — exactly c_step's own burn test."""
    return a != 0 and om.ACTION_TAU[a] == 1


BO_Q_A = 1.0e-13          # NAV-G's bearings-only covariance floor
BO_W0 = 45                # initial batch window (observations)
BO_ACQ_EVERY = 15         # retry cadence, in new observations
BO_ARC_MAX = 400          # cap the batch arc (cost is superlinear in window)


def _bo_env_annulus():
    """Altitude annulus [r_min, r_max] the target is known to live in."""
    a_min = ENV_KWARGS.get('a_min_override', -1.0)
    a_max = ENV_KWARGS.get('a_max_override', -1.0)
    a_min = a_min if a_min >= om.R_EARTH else om.R_EARTH + 300e3
    a_max = a_max if a_max > a_min else om.R_EARTH + 800e3
    e_max = max(float(ENV_KWARGS.get('e_max_target', 0.0)), 0.0)
    return a_min * (1.0 - e_max), a_max * (1.0 + e_max)


def _bo_sigma_v_ecc():
    """Circular-velocity guess error. T4 §8.2: it scales with ECCENTRICITY
    (~v_c*e), not with range."""
    e_max = max(float(ENV_KWARGS.get('e_max_target', 0.0)), 0.0)
    return max(7.7e3 * max(e_max, 0.02), 100.0)


def _bo_blind_state(sat_cart, beta, r_min, r_max):
    """The unacquired estimate: geometric mean of the analytic feasible range
    set along the measured bearing, plus a prograde circular velocity.

    This is exactly the object red-team `ext_rtnav_blind_window.py` injected to
    measure what blindness costs (median injected range error 9,276 km; median
    max target-slot obs perturbation 1.74 obs-units).
    """
    import ext_bo_filter as ebf
    iv = ebf.range_prior_intervals(sat_cart, beta, r_min, r_max)
    lo, hi = ebf.prior_span(iv)
    rho0 = math.sqrt(max(lo, 1.0) * hi)
    px = sat_cart[0] + rho0 * math.cos(beta)
    py = sat_cart[1] + rho0 * math.sin(beta)
    vx, vy = ebf.circular_state(px, py)
    return np.array([px, py, vx, vy], dtype=float), iv, (lo, hi)


def _bo_seed_mpc(mpc, x0, sat_cart, lo, hi, sigma_v_ecc, sigma_beta):
    """Seed the modified-polar filter from the blind prior.

    Seeded in MSC, not Cartesian: the 4-decade range ignorance then lives
    entirely in ln rho where its variance is (ln(hi/lo))^2/12 <= ~10, instead
    of being pushed through the encoding Jacobian as a (sig_rho/rho0)^2 ~ 1e3
    term that the finite-difference transition Jacobian cannot survive.
    """
    import ext_bo_filter as ebf
    y0 = ebf.msc_encode(sat_cart, x0)
    rho0 = math.exp(min(y0[3], 25.0))
    sig_lnr = min(max(math.log(hi / max(lo, 1.0)) / math.sqrt(12.0), 0.5), 4.0)
    sig_rate = min(max(sigma_v_ecc / max(rho0, 1.0), 1e-5), 1e-1)
    mpc.y = y0
    mpc.Py = np.diag([sigma_beta ** 2, sig_rate ** 2, sig_rate ** 2,
                      sig_lnr ** 2])
    mpc.sat = tuple(sat_cart)
    mpc.alive = True
    return mpc


def run_bo(num_episodes, noise_scale=1.0, q_a=BO_Q_A, seed=42, meas_seed=1234,
           label="", blindclose_m=0.0, blind_all=0, verbose=True):
    """Closed loop on an angles-only estimate.

    blindclose_m > 0 reproduces red-team MAJOR-4: whenever the TRUE separation
    is below the threshold, the policy is handed the *unacquired* prior
    estimate instead of the filter's. That cell is the single most informative
    one in the matrix — blindness at episode start is free (100/100 at every
    depth to 4 decisions), blindness inside 200 km scores 45/100 and inside
    1000 km scores 18/100, all by fuel exhaustion.

    blind_all=1 is the permanently-blind control. It also retires the
    "reward leaks truth" worry at inference time: if truth leaked usefully
    through the reward, a permanently blind policy would not collapse.
    """
    import ext_bo_filter as ebf

    env = make_env()
    policy = load_policy(env)
    obs, _ = env.reset(seed=seed)
    rng = np.random.default_rng(meas_seed)
    state = zero_state(policy)
    s_beta = SIGMA_BETA_RAD * noise_scale
    r_min, r_max = _bo_env_annulus()
    sigma_v_ecc = _bo_sigma_v_ecc()

    mpc = ebf.BearingMPC(sigma_beta=s_beta, q_a=q_a)
    ep_success, ep_cause, ep_len = [], [], []
    flat_rho, flat_pe, flat_ve, flat_slr, flat_acq = [], [], [], [], []
    acq_latency, acq_epoch_err, n_acq_fail, n_diverge = [], [], 0, 0
    ep_blind_dec, ep_dec, ep_minrho = [], [], []
    # ── blind-window behaviour (the mechanism training is supposed to fix) ──
    # Canonical baseline: fed a ~9,000 km-wrong prior, the truth-trained policy
    # dumps its whole 478 m/s budget in ~16 min of sim time and strands itself
    # BEFORE any angles-only solver could have converged (NAV-G's w0 is 45
    # observations = 45 min). These counters measure exactly that.
    acq_dv, acq_fuel_left = [], []          # at the acquisition instant
    ep_blind_burns, ep_acq_burns = [], []   # burns, split by nav state
    ep_blind_dv, ep_acq_dv = [], []         # delta-v applied, split by nav state
    ep_acq_dec = []
    ep_total_dv, ep_ever_acq, ep_blind_at_end = [], [], []
    blind_burns = acq_burns = acq_dec = 0
    blind_dv = acq_dv_ep = 0.0
    fuel_prev = None
    logged_acq = False

    fresh, episodes, t0 = True, 0, time.time()
    times, sats, betas = [], [], []
    intervals, prior = None, (1.0, 1.0)
    acquired, t_clock, blind_dec, dec, minrho = False, 0.0, 0, 0, float('inf')
    next_try = BO_W0
    prev_sat, prev_tgt, prev_tau = None, None, 1

    while episodes < num_episodes:
        o = np.array(obs[0], dtype=np.float32, copy=True)
        sat_el, tgt_el = recover_states(o)
        sat_cart = om.orbit_to_cartesian(sat_el)
        tgt_cart = om.orbit_to_cartesian(tgt_el)
        beta_t = math.atan2(tgt_cart[1] - sat_cart[1], tgt_cart[0] - sat_cart[0])
        beta = wrap_pi(beta_t + rng.normal(0.0, s_beta))

        if fresh:
            t_clock = 0.0
            times, sats, betas = [0.0], [sat_cart], [beta]
            x0, intervals, prior = _bo_blind_state(sat_cart, beta, r_min, r_max)
            _bo_seed_mpc(mpc, x0, sat_cart, prior[0], prior[1], sigma_v_ecc,
                         s_beta)
            acquired, next_try, fresh = False, BO_W0, False
            blind_dec, dec, minrho = 0, 0, float('inf')
            blind_burns = acq_burns = acq_dec = 0
            blind_dv = acq_dv_ep = 0.0
            fuel_prev = float(o[6])
            logged_acq = False
        else:
            # Advance navigation from the PREVIOUS decision epoch to this one,
            # at the fixed sensor cadence. Done here, not at the end of the
            # previous iteration, because the last sub-interval's `sat_to` is
            # the chaser state AFTER any impulse — and the modified-polar state
            # is RELATIVE, so re-encoding against a pre-burn chaser silently
            # attributes the chaser's own delta-v to the target. Measured: that
            # single ordering error put 25 m/s into the target velocity
            # estimate and 169 km of position error into the closed loop.
            n_sub = max(1, int(prev_tau))
            s_c, t_c = prev_sat, prev_tgt
            for i in range(n_sub):
                s_prev = s_c
                s_c = om.propagate_cartesian(s_c, om.DT)
                t_c = om.propagate_cartesian(t_c, om.DT)
                t_clock += om.DT
                if i == n_sub - 1:
                    # the env's own epoch: pin truth to the observation
                    s_c, t_c = sat_cart, tgt_cart
                    b_i = beta
                else:
                    b_i = wrap_pi(math.atan2(t_c[1] - s_c[1], t_c[0] - s_c[0])
                                  + rng.normal(0.0, s_beta))
                mpc.predict(om.DT, s_prev, s_c)
                mpc.sat = tuple(s_c)
                mpc.update(s_c, b_i)
                times.append(t_clock)
                sats.append(s_c)
                betas.append(b_i)

        # ── real angles-only acquisition ────────────────────────────────────
        if not acquired and len(times) >= next_try:
            w = min(len(times), BO_ARC_MAX)
            acq = ebf.bls_acquire_adaptive(
                times[-w:], sats[-w:], betas[-w:],
                ebf.range_prior_intervals(sats[-w], betas[-w], r_min, r_max),
                s_beta, sigma_v_ecc, w0=min(BO_W0, w))
            next_try = len(times) + BO_ACQ_EVERY
            if acq is not None and acq[4]:
                x_e, P_e = acq[0], acq[1]
                dt_fwd = times[-1] - times[-w]
                acq_latency.append(dt_fwd / 60.0 + BO_W0)
                # epoch error against truth, the number MAJOR-2 says the gates
                # do NOT certify
                tgt_epoch = om.propagate_cartesian(tgt_cart, -dt_fwd)
                acq_epoch_err.append(math.hypot(x_e[0] - tgt_epoch[0],
                                                x_e[1] - tgt_epoch[1]))
                F = om.stm_numerical(x_e, dt_fwd) if dt_fwd > 0 else np.eye(4)
                x_n = (om.propagate_cartesian(x_e, dt_fwd) if dt_fwd > 0
                       else x_e)
                mpc.set_cart(np.asarray(x_n, dtype=float), F @ P_e @ F.T,
                             sat_cart)
                acquired = True
                if not logged_acq:
                    acq_dv.append(dv_spent(o[6]))
                    acq_fuel_left.append(fuel_budget_left(o[6]))
                    logged_acq = True
            elif len(times) > BO_ARC_MAX:
                n_acq_fail += 1

        # ── divergence guard ────────────────────────────────────────────────
        x_est = np.asarray(ebf.msc_decode(mpc.y, mpc.sat), dtype=float)
        r_e = math.hypot(x_est[0], x_est[1])
        v2 = x_est[2] ** 2 + x_est[3] ** 2
        bad = (not np.all(np.isfinite(x_est))) or (not mpc.alive) \
            or r_e < 1.0 or (2.0 / max(r_e, 1.0) - v2 / om.MU) <= 0.0
        if bad:
            n_diverge += 1
            x0, intervals, prior = _bo_blind_state(sat_cart, beta, r_min, r_max)
            _bo_seed_mpc(mpc, x0, sat_cart, prior[0], prior[1], sigma_v_ecc,
                         s_beta)
            acquired = False
            times, sats, betas = [t_clock], [sat_cart], [beta]
            next_try = BO_W0
            x_est = x0

        # ── telemetry ───────────────────────────────────────────────────────
        rho_true = math.hypot(tgt_cart[0] - sat_cart[0],
                              tgt_cart[1] - sat_cart[1])
        minrho = min(minrho, rho_true)
        _, P_c = mpc.mean_cov()
        u = np.array([tgt_cart[0] - sat_cart[0], tgt_cart[1] - sat_cart[1]])
        u = u / max(np.linalg.norm(u), 1e-12)
        slr = math.sqrt(max(float(u @ P_c[:2, :2] @ u), 0.0)) / max(rho_true, 1.0)
        flat_rho.append(rho_true)
        flat_pe.append(math.hypot(x_est[0] - tgt_cart[0], x_est[1] - tgt_cart[1]))
        flat_ve.append(math.hypot(x_est[2] - tgt_cart[2], x_est[3] - tgt_cart[3]))
        flat_slr.append(slr)
        flat_acq.append(1.0 if acquired else 0.0)

        # ── policy observation ──────────────────────────────────────────────
        show_blind = bool(blind_all) or (blindclose_m > 0.0
                                         and rho_true < blindclose_m)
        if show_blind:
            xb, _, _ = _bo_blind_state(sat_cart, beta, r_min, r_max)
            x_pol = xb
        else:
            x_pol = x_est
        dec += 1
        if show_blind or not acquired:
            blind_dec += 1
        est_el = om.cartesian_to_elements(*x_pol)
        pol_obs = build_obs(o, sat_el, est_el, tgt_cart=tuple(x_pol))

        with torch.no_grad():
            logits, _ = policy.forward_eval(
                torch.from_numpy(pol_obs).float().unsqueeze(0).unsqueeze(0), state)
            action = int(torch.argmax(logits, dim=-1).item())

        obs, rewards, terms, truncs, _ = env.step(np.array([action], dtype=np.int32))
        prev_sat, prev_tgt, prev_tau = sat_cart, tgt_cart, om.ACTION_TAU[action]

        # Attribute this decision's realised delta-v to the nav state it was
        # taken under. `blind` here means "the policy was flying an unacquired
        # estimate", which is the condition the training arm is meant to change.
        blind_now = show_blind or not acquired
        f_now = float(obs[0][6]) if not terms[0] else 0.0
        d_dv = max(dv_spent(f_now) - dv_spent(fuel_prev), 0.0) if not terms[0] else 0.0
        if blind_now:
            blind_burns += 1 if _is_burn(action) else 0
            blind_dv += d_dv
        else:
            acq_dec += 1
            acq_burns += 1 if _is_burn(action) else 0
            acq_dv_ep += d_dv
        if not terms[0]:
            fuel_prev = f_now

        if terms[0]:
            episodes += 1
            sim_steps, cause = env.last_episode_result(0)
            ep_cause.append(int(cause))
            ep_success.append(1 if cause == 1 else 0)
            ep_len.append(int(sim_steps))
            ep_blind_dec.append(blind_dec)
            ep_dec.append(dec)
            ep_minrho.append(minrho)
            ep_blind_burns.append(blind_burns)
            ep_acq_burns.append(acq_burns)
            ep_acq_dec.append(acq_dec)
            ep_blind_dv.append(blind_dv)
            ep_acq_dv.append(acq_dv_ep)
            ep_total_dv.append(dv_spent(fuel_prev))
            ep_ever_acq.append(1 if logged_acq else 0)
            ep_blind_at_end.append(1 if blind_now else 0)
            fresh = True
            state = zero_state(policy)
            if verbose and episodes % 25 == 0:
                print(f"    [{label}] ep {episodes}/{num_episodes} "
                      f"success={np.mean(ep_success):.1%} "
                      f"({time.time()-t0:.0f}s)", flush=True)

    env.close()
    return dict(
        label=label, mode='bo', noise_scale=noise_scale,
        success=np.array(ep_success), cause=np.array(ep_cause),
        length=np.array(ep_len),
        flat_rho=np.array(flat_rho), flat_pe=np.array(flat_pe),
        flat_ve=np.array(flat_ve), flat_slr=np.array(flat_slr),
        flat_acq=np.array(flat_acq),
        acq_latency=np.array(acq_latency), acq_epoch_err=np.array(acq_epoch_err),
        n_acq_fail=n_acq_fail, n_diverge=n_diverge,
        blind_dec=np.array(ep_blind_dec), dec=np.array(ep_dec),
        minrho=np.array(ep_minrho),
        acq_dv=np.array(acq_dv), acq_fuel_left=np.array(acq_fuel_left),
        blind_burns=np.array(ep_blind_burns), acq_burns=np.array(ep_acq_burns),
        acq_dec=np.array(ep_acq_dec),
        blind_dv=np.array(ep_blind_dv), acq_dv_ep=np.array(ep_acq_dv),
        total_dv=np.array(ep_total_dv), ever_acq=np.array(ep_ever_acq),
        blind_at_end=np.array(ep_blind_at_end),
        wall=time.time() - t0)


def behaviour_report(r, indent='  ', quiet=False):
    """Blind-window behaviour block — the mechanism, not the score."""
    if quiet:
        import io
        import contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            return behaviour_report(r, indent)
    n = len(r['success'])
    bd, ad = r['blind_dec'].sum(), r['acq_dec'].sum()
    strand = r['cause'] == 5
    strand_blind = strand & (r['blind_at_end'] == 1)
    out = {}
    print(f"{indent}blind-window behaviour")
    if r['acq_dv'].size:
        out['acq_dv_med'] = float(np.median(r['acq_dv']))
        out['acq_fuel_med'] = float(np.median(r['acq_fuel_left']))
        print(f"{indent}  dv spent BEFORE acquisition   median "
              f"{np.median(r['acq_dv']):6.1f} m/s   p90 "
              f"{np.percentile(r['acq_dv'], 90):6.1f}   max "
              f"{r['acq_dv'].max():6.1f}   (budget 478)")
        print(f"{indent}  fuel budget left AT acquisition median "
              f"{np.median(r['acq_fuel_left']):.3f}   p10 "
              f"{np.percentile(r['acq_fuel_left'], 10):.3f}")
    else:
        out['acq_dv_med'] = out['acq_fuel_med'] = float('nan')
        print(f"{indent}  (no episode ever acquired)")
    br_b = r['blind_burns'].sum() / max(bd, 1)
    br_a = r['acq_burns'].sum() / max(ad, 1)
    dv_b = r['blind_dv'].sum() / max(bd, 1)
    dv_a = r['acq_dv_ep'].sum() / max(ad, 1)
    out.update(burn_rate_blind=float(br_b), burn_rate_acq=float(br_a),
               dv_per_dec_blind=float(dv_b), dv_per_dec_acq=float(dv_a),
               strand_rate=float(strand.mean()),
               strand_blind_rate=float(strand_blind.mean()),
               total_dv_med=float(np.median(r['total_dv'])),
               ever_acq=float(r['ever_acq'].mean()))
    print(f"{indent}  burn rate   blind {br_b:.3f} burns/decision   "
          f"acquired {br_a:.3f}   ratio {br_b/max(br_a,1e-9):5.2f}x")
    print(f"{indent}  dv per decision  blind {dv_b:6.2f} m/s   "
          f"acquired {dv_a:6.2f} m/s   ratio {dv_b/max(dv_a,1e-9):5.2f}x")
    print(f"{indent}  stranded {int(strand.sum())}/{n} = {strand.mean():.1%}   "
          f"of which BLIND at terminal {int(strand_blind.sum())} "
          f"({strand_blind.mean():.1%} of all episodes)")
    print(f"{indent}  total dv/episode median {np.median(r['total_dv']):.1f} m/s   "
          f"episodes that ever acquired {r['ever_acq'].mean():.1%}")
    return out


def _bo_report(r):
    n_gave = int(np.sum(r['cause'] == 7))
    n_valid = len(r['success']) - n_gave
    causes = ", ".join(f"{CAUSE_NAMES[c]}={int(np.sum(r['cause'] == c))}"
                       for c in range(8) if np.sum(r['cause'] == c))
    print(f"  {r['label']:22s} success {int(r['success'].sum())}/{n_valid} = "
          f"{r['success'].sum()/max(n_valid,1):6.1%}   "
          f"acquired {r['flat_acq'].mean():.3f}   "
          f"diverge {r['n_diverge']}   acq_fail {r['n_acq_fail']}   "
          f"({r['wall']:.0f}s)")
    print(f"  {'':22s} causes: {causes}")
    if r['acq_latency'].size:
        print(f"  {'':22s} acquisition: latency median "
              f"{np.median(r['acq_latency']):.0f} min "
              f"(p90 {np.percentile(r['acq_latency'],90):.0f}), "
              f"EPOCH error vs truth median "
              f"{np.median(r['acq_epoch_err'])/1e3:.1f} km "
              f"(p90 {np.percentile(r['acq_epoch_err'],90)/1e3:.1f} km, "
              f"max {r['acq_epoch_err'].max()/1e3:.1f} km)")
    blind_frac = r['blind_dec'].sum() / max(r['dec'].sum(), 1)
    print(f"  {'':22s} blind decisions {blind_frac:.1%}; episodes never "
          f"acquiring {int((r['blind_dec'] == r['dec']).sum())}/{len(r['dec'])}")
    # close-range covariance gate, the MAJOR-4 first-class output
    rho, slr, pe = r['flat_rho'], r['flat_slr'], r['flat_pe']
    print(f"  {'':22s} {'sep bin':>16s} {'n':>7s} {'med sigLOS/rho':>16s} "
          f"{'med |dp|':>12s} {'acq frac':>9s}")
    for lo, hi in ((0, 5e4), (5e4, 2e5), (2e5, 1e6), (1e6, 1e7), (1e7, 1e9)):
        m = (rho >= lo) & (rho < hi)
        if m.sum() < 10:
            continue
        print(f"  {'':22s} {lo:7.0e}-{hi:<7.0e} {int(m.sum()):7d} "
              f"{np.median(slr[m]):16.4f} {np.median(pe[m]):10.1f} m "
              f"{r['flat_acq'][m].mean():9.3f}")
    close = r['minrho'] < 2e5
    if close.any():
        print(f"  {'':22s} conditional success | min separation < 200 km: "
              f"{int(r['success'][close].sum())}/{int(close.sum())}")
    behaviour_report(r, indent='  ' + ' ' * 22)


def stage_bearings(args):
    print("── Stage: bearings-only (angles-only, real BLS acquisition) ────")
    print(f"  annulus r in [{_bo_env_annulus()[0]/1e3:.0f}, "
          f"{_bo_env_annulus()[1]/1e3:.0f}] km   sigma_v_ecc "
          f"{_bo_sigma_v_ecc():.0f} m/s   sigma_beta "
          f"{SIGMA_BETA_RAD*args.bo_noise:.1e} rad")
    out = []
    r = run_bo(args.eval_eps, noise_scale=args.bo_noise, seed=args.seed,
               label=f"bo_{args.bo_noise:g}x")
    _bo_report(r)
    out.append(r)
    for km in args.blindclose:
        rb = run_bo(args.blindclose_eps, noise_scale=args.bo_noise,
                    seed=args.seed, blindclose_m=km * 1e3,
                    label=f"blindclose<{int(km)}km", verbose=False)
        _bo_report(rb)
        out.append(rb)
    if args.blindall:
        ra = run_bo(args.blindclose_eps, noise_scale=args.bo_noise,
                    seed=args.seed, blind_all=1, label="blindall",
                    verbose=False)
        _bo_report(ra)
        out.append(ra)
    return out


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
                            'eval', 'bearings', 'all'])
    p.add_argument('--bo-noise', type=float, default=1.0,
                   help='bearings-only sigma_beta multiplier')
    p.add_argument('--blindclose', type=float, nargs='*', default=[],
                   help='blindclose ablation thresholds in km (red-team '
                        'MAJOR-4); e.g. --blindclose 50 200 1000')
    p.add_argument('--blindclose-eps', type=int, default=100)
    p.add_argument('--blindall', action='store_true',
                   help='permanently-blind control cell')
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
    if args.stage == 'bearings':
        stage_bearings(args); return

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
