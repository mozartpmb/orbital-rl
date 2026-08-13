#!/usr/bin/env python3
"""3D-nav eval harness: policy-on-estimated-state under `dim3_mode=1`.

Why this is a new file rather than a 3D branch inside `eval_relnav.py`. The 2D
harness is a SCALAR implementation built on `orbital_math` (4-state) and
`ext_bo_filter` (4-state) — its decode, re-encode, filter and acquisition are
all 2D by construction, and its own anchors run through them. Forking a `dim3`
path through it would put the 2D lineage's anchors at risk for no benefit,
because Phase A already produced batched, gated 3D versions of every piece:

    recover_states_3d   -> `Orbital.get_state()`, the read-only C getter.
                           MAJOR-5: obs[21-28] is SO(3)-invariant by
                           construction, so the chaser's plane is ABSENT from
                           the observation and NO decoder can recover it. The
                           getter is not an optimisation, it is the only
                           correct route.
    build_obs_t3 (19)   -> `nav_encode3d.Encoder3D`, gated bitwise against the
                           C's own observation (verify_n3dnav stage c1).
    the filter          -> `nav_math3d.BatchedBearingMSC6` (BLOCKER-1) and
                           `BatchedRangeBearingEKF3D` (the N1 control).
    z-term in the       -> `bls3d.range_prior_intervals3`, the 3-vector
    range prior            quadratic; and the wrapper's own blind seed.

So this harness drives `OrbitalNav` directly and adds the one thing training
does not have: the REAL 6-unknown / 2-angle batch IOD (`bls3d`) in place of the
calibrated acquisition surrogate.

Stages
  anchor  THE GATE. Truth-mode closed loop on a known 3D checkpoint at its own
          rung; must reproduce that checkpoint's published held-out score. Run
          this before any nav-mode eval — a nav result on an unvalidated
          harness is not a result.
  eval    closed loop in a chosen nav_mode, JSON out.

    cd /Users/pete/space_training/pufferlib
    python3 ../scripts/orbital/nav/eval_relnav3d.py --stage anchor
    python3 ../scripts/orbital/nav/eval_relnav3d.py --stage eval \
        --ckpt <path> --nav-mode bearings_only --episodes 200 --out <json>
"""

import argparse
import hashlib
import json
import math
import os
import sys
import time

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(_HERE, '..', '..', '..'))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(ROOT, 'pufferlib'))

from pufferlib.ocean.orbital.orbital import Orbital                # noqa: E402
from pufferlib.ocean.orbital_nav.orbital_nav import OrbitalNav     # noqa: E402
from pufferlib.ocean.orbital_nav import nav_math as nm             # noqa: E402
from pufferlib.ocean.orbital_nav import nav_math3d as n3           # noqa: E402
from pufferlib.models import Default, LSTMWrapper                  # noqa: E402
import bls3d                                                       # noqa: E402

CAUSES = ['none', 'success', 'collision', 'escape', 'safety_cap',
          'stranded', 'hyperbolic', 'gave_up']

# ── RUNG 1 = X3, exactly the rung `seed42_X3_3d_di1deg.pt` was trained and
# scored at (scripts/orbital/ext3d/t5_x3_seeds.sh). n3d_C section 5 puts every
# first-generation nav rung — N0-truth, N0-recon, N1-rb3d, N2-bo3d — at X3, and
# for a reason worth keeping in view: its box is the DEFAULT 30 km / 50 m/s,
# which is the same box as the 2D nav lineage, so the NB1 98.0-99.5% numbers
# are a like-for-like comparator rather than a different experiment.
X3_KW = dict(
    num_debris_min=0, num_debris_max=0,
    e_max_target=0.05, e_max_sat=0.05, same_orbit_init=0,
    init_phase_gap_max=3.14159, valid_init_only=1,
    gave_up_action='terminate', max_valid_init_attempts=4096,
    obs_alt_scale_m=1.6e6, lvlh_scale_m=6.371e6,
    rendezvous_radius_m=30000.0, rel_vel_tol_ms=50.0,
    shaping_mode=2, shape_w_lambda=1.0, shape_w_match=0.8166667,
    shape_dv_ref_ms=700.0, shape_gamma=1.0,
    phase_gap_mode=1, phase_obs_mode=1,
    episode_cap_steps=3000, cap_terminal_reward=0.0,
    dim3_mode=1, di_max_rad=0.017453,          # 1.0 deg relative inclination
    legacy_action_space=30,
)
RUNGS = {'X3': X3_KW}

# ── success boxes ───────────────────────────────────────────────────────────
# The 3D task is held fixed and ONLY the box moves, exactly as the TB3D ladder
# (scripts/orbital/t3/t6_tb3d.sh) moved it: B1 10 km/10 m/s -> B2 5 km/2 m/s ->
# B3 5 km/1 m/s. Named here so an eval cannot silently be run at a box the
# checkpoint was not trained for, which is the single easiest way to
# manufacture a wrong number in this project.
BOXES = {
    'X3':      (30000.0, 50.0),     # the loose 2D-lineage box, rung 1
    'TB3-3D':  (10000.0, 10.0),     # TB3D ladder B1, published 199/200
    'TB4-3D':  (5000.0,  2.0),      # TB3D ladder B2, published 199/200
    'TB5-3D':  (5000.0,  1.0),      # TB3D ladder B3, published 194/200
}

ANCHOR_CKPT = os.path.join(ROOT, 'models', 't3', 'seed42_X3_3d_di1deg.pt')
ANCHOR_EXPECT = (200, 200)          # published held-out score at seed 123


# ═══════════════════════════════════════════════════════════════════════════
# Real batch IOD at eval, in place of the training surrogate
# ═══════════════════════════════════════════════════════════════════════════
class RealAcq3D:
    """Drop-in replacement for `AcquisitionSurrogate` that runs the real solver.

    Same call surface, so `OrbitalNav` does not know the difference. It records
    the realised measurement arc — the very az/el the filter was fed, not a
    re-draw — and hands it to `bls3d.bls_acquire_adaptive3`.

    The acceptance window `w0` is DERIVED from the 45-minute sim-time floor and
    the cadence (BLOCKER-2), never written as an observation count: at
    dt_tick = 300 s a hard-coded 45 would be a 3.75-hour floor, and the
    resulting failure would present as "bearings-only does not work".
    """

    def __init__(self, n, sigma_beta, dt_tick, gate=0.20,
                 min_sec=2700.0, cov_inflate=4.0, r_min=None, r_max=None,
                 arc_max=400, retry_every=15, retry_growth=1.6,
                 retry_max=240):
        self.n = n
        self.sigma_beta = float(sigma_beta)
        self.dt = float(dt_tick)
        self.gate = float(gate)
        self.min_sec = float(min_sec)
        self.cov_inflate = float(cov_inflate)
        self.w0 = max(2, int(round(self.min_sec / max(self.dt, 1e-9))))
        self.arc_max = int(arc_max)
        # Retry cadence GROWS after each failure, and it has to. The 2D
        # harness retries every 15 observations, which is affordable there
        # (4 states, 1 angle, a cheap scalar solver). Here each attempt is a
        # 4320-node grid plus binned multi-start Gauss-Newton over an arc that
        # `bls_acquire_adaptive3` itself already grows x1.6 internally, so a
        # fixed 15-observation retry re-runs almost the same computation on
        # almost the same arc. Measured on the floor arm: a failing episode
        # that runs to the 3000-substep cap would attempt acquisition ~200
        # times at seconds apiece — tens of minutes for ONE episode, and the
        # bearings-only evals are 200 episodes x 3 arms.
        #
        # Growing the interval by the same 1.6 the solver uses means a failure
        # is retried only once the arc is MATERIALLY longer, which is the only
        # condition under which the answer can change. ~8-10 attempts per
        # capped episode instead of ~200, with no change to WHEN a solvable
        # geometry first solves (the first attempt is still at the 45-minute
        # floor).
        self.retry_every = int(retry_every)
        self.retry_growth = float(retry_growth)
        self.retry_max = int(retry_max)
        self.n_attempt = 0
        self.r_min, self.r_max = r_min, r_max
        self.acquired = np.zeros(n, dtype=bool)
        self.elapsed = np.zeros(n)
        self.ticks = np.zeros(n, dtype=np.int64)
        self.dv = np.zeros(n)
        self.n_acq = 0
        self.n_reset = 0
        self.n_fail = 0
        self.n_chol_fallback = 0
        self.acq_latency_s = []
        self.acq_epoch_err_m = []
        self._arc = [dict(t=[], sat=[], az=[], el=[], next_try=self.w0,
                          tries=0) for _ in range(n)]
        self._Rp = np.tile(np.eye(3), (n, 1, 1))
        self._pending = {}

    # -- surrogate API --------------------------------------------------------
    def reset_rows(self, idx, sat_cart, tgt_cart, period_s, a_ref=None):
        for i in np.atleast_1d(idx):
            self._arc[int(i)] = dict(t=[], sat=[], az=[], el=[],
                                     next_try=self.w0, tries=0)
        self.acquired[idx] = False
        self.elapsed[idx] = 0.0
        self.ticks[idx] = 0
        self.dv[idx] = 0.0
        self.n_reset += int(np.size(idx))

    def add_dv(self, idx, dv):
        m = ~self.acquired[idx]
        if m.any():
            self.dv[idx[m]] += dv[m]

    def accumulate(self, idx, sat_now, tgt_now, tgt_prev, dt, first=False):
        sel = ~self.acquired[idx]
        if not sel.any():
            return
        pend = idx[sel]
        self.ticks[pend] += 1
        self.elapsed[pend] += float(dt)

    def record_meas(self, idx, Rp, sat_cart, az, el):
        """The hook `OrbitalNav._tick3` calls with the REALISED measurement."""
        for k, i in enumerate(np.atleast_1d(idx)):
            i = int(i)
            if self.acquired[i]:
                continue
            a = self._arc[i]
            a['t'].append(float(self.elapsed[i]))
            a['sat'].append(np.array(sat_cart[k], dtype=np.float64))
            a['az'].append(float(az[k]))
            a['el'].append(float(el[k]))
            self._Rp[i] = Rp[k]

    def ready(self, idx, dt=60.0):
        rows, sig = [], []
        for i in np.atleast_1d(idx):
            i = int(i)
            if self.acquired[i]:
                continue
            a = self._arc[i]
            if self.elapsed[i] < self.min_sec or len(a['t']) < a['next_try']:
                continue
            w = min(len(a['t']), self.arc_max)
            Rp = self._Rp[i]
            ca, se = math.cos(a['el'][-w]), math.sin(a['el'][-w])
            u0 = Rp.T @ np.array([ca * math.cos(a['az'][-w]),
                                  ca * math.sin(a['az'][-w]), se])
            iv = bls3d.range_prior_intervals3(a['sat'][-w][:3], u0,
                                              self.r_min, self.r_max)
            step = min(self.retry_max,
                       int(round(self.retry_every
                                 * self.retry_growth ** a['tries'])))
            a['tries'] += 1
            self.n_attempt += 1
            a['next_try'] = len(a['t']) + step
            if not iv:
                continue
            res = bls3d.bls_acquire_adaptive3(
                a['t'][-w:], a['sat'][-w:], a['az'][-w:], a['el'][-w:], iv,
                self.sigma_beta, Rp, w0=min(self.w0, w), gate=self.gate,
                cov_inflate=self.cov_inflate)
            if res is None:
                self.n_fail += 1
                continue
            x_e, P_e, cost, ww, ok = res
            if not ok:
                continue
            # Forward the epoch solution to NOW with the same analytic STM the
            # filter uses, exactly as the 2D harness does.
            dt_fwd = a['t'][-1] - a['t'][-w]
            if dt_fwd > 0:
                F, _, Y = n3.stm_analytic_nd(x_e[None], dt_fwd)
                x_n, P_n = Y[0], F[0] @ P_e @ F[0].T
            else:
                x_n, P_n = x_e, P_e
            self._pending[i] = (x_n, P_n, x_e, dt_fwd, w)
            rows.append(i)
            rho_v = x_e[:3] - a['sat'][-w][:3]
            rho = max(float(np.linalg.norm(rho_v)), 1.0)
            u = rho_v / rho
            sig.append(math.sqrt(max(float(u @ P_e[:3, :3] @ u), 0.0)))
        return np.array(rows, dtype=np.int64), np.array(sig)

    def draw(self, idx, sigma_crlb, x_true, sat_cart, rng):
        """No draw: the real solver already produced a state and a covariance."""
        m = np.size(idx)
        X = np.zeros((m, 6))
        P = np.zeros((m, 6, 6))
        for k, i in enumerate(np.atleast_1d(idx)):
            i = int(i)
            x_n, P_n, x_e, dt_fwd, w = self._pending.pop(i)
            X[k] = x_n
            P[k] = P_n
            self.acquired[i] = True
            self.acq_latency_s.append(float(self.elapsed[i]))
            # The number MAJOR-2 says the gates do NOT certify: epoch error
            # against truth, back-propagated to the batch epoch.
            if dt_fwd > 0:
                tgt_epoch = n3.propagate_cartesian_nd(
                    np.asarray(x_true[k], dtype=np.float64)[None], -dt_fwd)[0][0]
            else:
                tgt_epoch = np.asarray(x_true[k], dtype=np.float64)
            self.acq_epoch_err_m.append(
                float(np.linalg.norm(x_e[:3] - tgt_epoch[:3])))
        self.n_acq += m
        return X, P, np.zeros(m)


# ═══════════════════════════════════════════════════════════════════════════
def box_kw(box=None, radius_m=None, vel_ms=None):
    """Box override as env kwargs. Named preset, or an explicit pair."""
    if radius_m is not None and vel_ms is not None:
        return dict(rendezvous_radius_m=float(radius_m),
                    rel_vel_tol_ms=float(vel_ms))
    if box:
        r, v = BOXES[box]
        return dict(rendezvous_radius_m=r, rel_vel_tol_ms=v)
    return {}


def make_env(rung, nav_mode, acq='surrogate', noise_mult=1.0, nav_seed=20260812,
             **over):
    kw = dict(RUNGS[rung])
    kw.update(over)
    env = OrbitalNav(num_envs=1, nav_mode=nav_mode, nav_noise_mult=noise_mult,
                     nav_seed=nav_seed, log_interval=10 ** 9, **kw)
    env._acq_real = (acq == 'real' and nav_mode == 'bearings_only')
    # The C env keeps the box; nothing on the Python object exposes it, and
    # the in-box terminal statistics are meaningless if they are computed at
    # the wrong radius (a 30 km window at a 5 km box measures the approach,
    # not the capture). Stash it explicitly.
    env._box_radius_m = float(kw['rendezvous_radius_m'])
    env._box_vel_ms = float(kw['rel_vel_tol_ms'])
    return env


def _install_real_acq(env):
    env._nav_alloc()
    env._acq = RealAcq3D(env.num_agents, env._s_beta, env._nav_dt,
                         gate=env._acq_gate, min_sec=env._acq_min_sec,
                         cov_inflate=env._cov_inflate,
                         r_min=env._r_min, r_max=env._r_max)


def load_policy(env, ckpt):
    p = LSTMWrapper(env, Default(env))
    p.load_state_dict(torch.load(ckpt, map_location='cpu', weights_only=True))
    p.eval()
    return p


def rollout(env, ckpt, episodes, seed, label='', verbose=True):
    """eval_checkpoint.py's protocol, verbatim: num_envs=1, greedy argmax,
    LSTM zeroed per episode, success = terminal cause == 1, gave-up inits
    excluded from the denominator."""
    policy = load_policy(env, ckpt)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    obs, _ = env.reset(seed=seed)
    st = {'lstm_h': torch.zeros(1, policy.hidden_size),
          'lstm_c': torch.zeros(1, policy.hidden_size)}
    actions, causes, lengths = [], [], []
    dv_used, minrho, ep_minrho = [], float('inf'), []
    fuel_last = float(np.asarray(obs)[0, 6])
    # ── terminal-geometry error statistics (n3d_B section 2.3) ─────────────
    # N3D-B predicts the 3D angles-only dividend lands on VELOCITY (2-4x),
    # not on range, and NAV-F section 2.6 established that a rendezvous box
    # binds on sigma_vel rather than sigma_range — which is exactly what makes
    # TB5's 1 m/s tolerance the binding constraint. Sampling the estimator's
    # velocity error while the chaser is INSIDE the position box turns that
    # prediction into a measurement instead of an inherited claim.
    box_r = float(getattr(env, '_box_radius_m', 0.0) or 0.0)
    box_v = float(getattr(env, '_box_vel_ms', 0.0) or 0.0)
    if not box_r:
        raise RuntimeError('rollout: box radius not set on the env; build it '
                           'with make_env() or set _box_radius_m explicitly')
    v_err_box, p_err_box, relv_box = [], [], []
    relv_last, ep_relv_last = [], float('nan')
    n_done, k, t0 = 0, 0, time.time()
    while n_done < episodes:
        with torch.no_grad():
            logits, _ = policy.forward_eval(
                torch.from_numpy(np.asarray(obs)).float().unsqueeze(0), st)
            a = int(torch.argmax(logits, dim=-1).item())
        actions.append(a)
        if getattr(env, '_prev_tgt', None) is not None and env._dim3 \
                and env._prev_tgt.shape[1] == 6:
            # `_prev_sat`/`_prev_tgt` hold TRUTH at the CURRENT decision epoch
            # (they are written at the end of the previous _nav_step), so this
            # pairs with the estimate the policy is about to act on.
            d = env._prev_tgt[0, :3] - env._prev_sat[0, :3]
            rho_t = float(np.linalg.norm(d))
            minrho = min(minrho, rho_t)
            dv_rel = float(np.linalg.norm(env._prev_tgt[0, 3:6]
                                          - env._prev_sat[0, 3:6]))
            ep_relv_last = dv_rel
            if rho_t < box_r:
                relv_box.append(dv_rel)
                if getattr(env, '_filt', None) is not None:
                    est = env._mean()
                    v_err_box.append(float(np.linalg.norm(
                        est[0, 3:6] - env._prev_tgt[0, 3:6])))
                    p_err_box.append(float(np.linalg.norm(
                        est[0, :3] - env._prev_tgt[0, :3])))
        fuel_last = float(np.asarray(obs)[0, 6])
        obs, rew, term, trunc, _ = env.step(np.array([a], dtype=np.int32))
        k += 1
        if term[0]:
            n_done += 1
            _, cause = env.last_episode_result(0)
            causes.append(int(cause))
            lengths.append(k)
            ep_minrho.append(minrho if np.isfinite(minrho) else 0.0)
            if np.isfinite(ep_relv_last):
                relv_last.append(ep_relv_last)
            ep_relv_last = float('nan')
            # obs after a terminal step is the NEXT episode's reset, so the
            # fuel state has to come from before the step.
            dv_used.append(_dv_spent(fuel_last))
            k, minrho = 0, float('inf')
            st = {'lstm_h': torch.zeros(1, policy.hidden_size),
                  'lstm_c': torch.zeros(1, policy.hidden_size)}
            if verbose and n_done % 25 == 0:
                sr = sum(1 for c in causes if c == 1) / n_done
                print(f'    [{label}] {n_done}/{episodes} success {sr:.1%} '
                      f'({time.time() - t0:.0f}s)', flush=True)
    causes = np.array(causes)
    n_gave = int((causes == 7).sum())
    n_valid = len(causes) - n_gave
    succ = int((causes == 1).sum())
    out = dict(
        label=label, success=succ, n_valid=n_valid, gave_up=n_gave,
        rate=succ / max(n_valid, 1),
        causes={CAUSES[c]: int((causes == c).sum())
                for c in range(8) if (causes == c).any()},
        md5=hashlib.md5(np.array(actions, dtype=np.int32).tobytes()).hexdigest(),
        n_actions=len(actions), mean_len=float(np.mean(lengths)),
        dv_med_ms=float(np.median(dv_used)),
        minrho_med_m=float(np.median(ep_minrho)),
        box_radius_m=box_r, box_vel_ms=box_v,
        wall_s=float(time.time() - t0))
    # Inside-the-box terminal geometry. `rel_vel` is the physical quantity the
    # success test uses (|v_sat - v_tgt|, inertial 3-vector, orbital.h:1470);
    # `vel_err` is the ESTIMATOR's error on it, which is what n3d_B predicts
    # the 3D dividend lands on.
    if relv_box:
        out['relvel_inbox_med_ms'] = float(np.median(relv_box))
        out['relvel_inbox_p90_ms'] = float(np.percentile(relv_box, 90))
        out['n_inbox_dec'] = len(relv_box)
        # What fraction of in-box decisions already satisfy the VELOCITY half
        # of the success test — i.e. how much of the endgame is spent waiting
        # on velocity rather than on position. At TB5 this is the binding half.
        if box_v:
            out['relvel_inbox_under_tol_frac'] = float(
                np.mean(np.asarray(relv_box) < box_v))
    if relv_last:
        out['relvel_lastdec_med_ms'] = float(np.median(relv_last))
    if v_err_box:
        out['navvel_err_inbox_med_ms'] = float(np.median(v_err_box))
        out['navvel_err_inbox_p90_ms'] = float(np.percentile(v_err_box, 90))
        out['navpos_err_inbox_med_m'] = float(np.median(p_err_box))
    acq = getattr(env, '_acq', None)
    if acq is not None:
        lat = list(getattr(acq, 'acq_latency_s', []))
        out['acq_n'] = int(getattr(acq, 'n_acq', 0))
        out['acq_per_ep'] = float(acq.n_acq / max(acq.n_reset, 1))
        # BLOCKER-2: SIM SECONDS. Never decisions — the 3D policy packs 2.45x
        # more sim-time into a decision than the 2D one, so a decision-counted
        # latency shows 3D winning by 2.45x from the tau mix alone.
        out['acq_latency_med_s'] = float(np.median(lat)) if lat else None
        out['acq_latency_p90_s'] = (float(np.percentile(lat, 90))
                                    if lat else None)
        out['acq_fail'] = int(getattr(acq, 'n_fail', 0))
        out['acq_attempts'] = int(getattr(acq, 'n_attempt', 0))
        out['chol_fallback'] = int(getattr(acq, 'n_chol_fallback', 0))
        ee = list(getattr(acq, 'acq_epoch_err_m', []))
        if ee:
            # MAJOR-2 carries over verbatim: the acceptance gates test the
            # self-consistency of the chosen range basin, NOT the correctness
            # of the basin choice. Report the epoch error against truth so a
            # certified-but-wrong basin is visible.
            out['acq_epoch_err_med_m'] = float(np.median(ee))
            out['acq_epoch_err_max_m'] = float(np.max(ee))
    filt = getattr(env, '_filt', None)
    if filt is not None and hasattr(filt, 'n_repole_total'):
        out['repole_total'] = int(filt.n_repole_total)
    env.close()
    return out


_VE = 300.0 * 9.80665


def _dv_spent(fuel_frac):
    return _VE * math.log(max(1.0 - float(fuel_frac), 1e-9) / (1.0 - 0.15))


def show(r):
    print(f"  {r['label']:28s} success {r['success']}/{r['n_valid']} = "
          f"{r['rate']:6.1%}   gave_up {r['gave_up']}   "
          f"decisions {r['n_actions']} (mean {r['mean_len']:.1f}/ep)   "
          f"md5 {r['md5'][:12]}   {r['wall_s']:.0f}s")
    print(f"  {'':28s} causes: {r['causes']}   dv_med {r['dv_med_ms']:.0f} m/s")
    if 'relvel_inbox_med_ms' in r:
        line = (f"  {'':28s} inside the {r['box_radius_m']/1e3:.0f} km box "
                f"({r['n_inbox_dec']} decisions): |v_rel| median "
                f"{r['relvel_inbox_med_ms']:.3f} m/s (p90 "
                f"{r['relvel_inbox_p90_ms']:.3f}, "
                f"{r.get('relvel_inbox_under_tol_frac', float('nan')):.2f} "
                f"under the {r['box_vel_ms']:.0f} m/s tol)")
        if 'navvel_err_inbox_med_ms' in r:
            line += (f"; NAV vel error median "
                     f"{r['navvel_err_inbox_med_ms']:.3f} m/s (p90 "
                     f"{r['navvel_err_inbox_p90_ms']:.3f}), pos error median "
                     f"{r['navpos_err_inbox_med_m']/1e3:.2f} km")
        print(line)
    if r.get('acq_latency_med_s') is not None:
        print(f"  {'':28s} acquisition: {r['acq_n']} acquired "
              f"({r['acq_per_ep']:.2f}/ep), latency median "
              f"{r['acq_latency_med_s']:.0f} s "
              f"({r['acq_latency_med_s'] / 60:.1f} min), p90 "
              f"{r['acq_latency_p90_s']:.0f} s, fails {r['acq_fail']}")
        if 'acq_epoch_err_med_m' in r:
            print(f"  {'':28s} EPOCH error vs truth (NOT certified by the "
                  f"gates): median {r['acq_epoch_err_med_m'] / 1e3:.1f} km, "
                  f"max {r['acq_epoch_err_max_m'] / 1e3:.1f} km")


# ═══════════════════════════════════════════════════════════════════════════
def stage_anchor(args):
    """THE harness gate. If this fails, nothing downstream means anything."""
    ck = args.ckpt or ANCHOR_CKPT
    over = _cli_over(args)
    exp_s, exp_n = ANCHOR_EXPECT
    if args.expect:
        exp_s, exp_n = (int(x) for x in args.expect.split('/'))
    print('== HARNESS ANCHOR ================================================')
    print(f'  rung        {args.rung}')
    print(f'  box         {args.box or "rung default"}  '
          f'{over.get("rendezvous_radius_m", RUNGS[args.rung]["rendezvous_radius_m"])/1e3:.0f} km / '
          f'{over.get("rel_vel_tol_ms", RUNGS[args.rung]["rel_vel_tol_ms"]):.0f} m/s')
    print(f'  checkpoint  {ck}')
    print(f'  expect      {exp_s}/{exp_n} at seed {args.seed} (PUBLISHED '
          f'held-out score)')
    if not os.path.exists(ck):
        print(f'  FAIL: checkpoint missing')
        return 1

    base = dict(RUNGS[args.rung]); base.update(over)
    _plain = Orbital(num_envs=1, **base)
    _plain._box_radius_m = float(base['rendezvous_radius_m'])
    _plain._box_vel_ms = float(base['rel_vel_tol_ms'])
    a = rollout(_plain, ck,
                args.episodes, args.seed, 'plain Orbital (dim3)', verbose=False)
    show(a)
    b = rollout(make_env(args.rung, 'truth', **over), ck, args.episodes,
                args.seed, 'OrbitalNav truth (dim3)', verbose=False)
    show(b)

    ok_plain = (a['success'], a['n_valid']) == (
        exp_s * args.episodes // 200, exp_n * args.episodes // 200)
    ok_same = (a['md5'] == b['md5'] and a['causes'] == b['causes'])
    print(f"\n  [{'PASS' if ok_plain else 'FAIL'}] plain Orbital reproduces the "
          f"published score at this rung")
    print(f"  [{'PASS' if ok_same else 'FAIL'}] OrbitalNav(nav_mode='truth') is "
          f"a byte-identical passthrough (md5 {a['md5'][:16]} vs "
          f"{b['md5'][:16]})")

    c = rollout(make_env(args.rung, 'recon', **over), ck, args.episodes,
                args.seed, 'OrbitalNav recon (dim3)', verbose=False)
    show(c)
    ok_recon = (a['md5'] == c['md5'] and a['causes'] == c['causes'])
    print(f"  [{'PASS' if ok_recon else 'FAIL'}] the 19-slot encode/decode layer "
          f"is transparent at zero estimation error")

    ok = ok_plain and ok_same and ok_recon
    print(f"\n  HARNESS ANCHOR: {'PASS' if ok else 'FAIL'}")
    if args.out:
        json.dump(dict(anchor_pass=bool(ok), plain=a, truth=b, recon=c),
                  open(args.out, 'w'), indent=1)
        print(f'  wrote {args.out}')
    return 0 if ok else 1


def stage_eval(args):
    print(f'== EVAL  {args.nav_mode}  ({args.acq} acquisition) ===============')
    over = _cli_over(args)
    env = make_env(args.rung, args.nav_mode, acq=args.acq,
                   noise_mult=args.noise_mult, nav_seed=args.nav_seed, **over)
    if args.acq == 'real' and args.nav_mode == 'bearings_only':
        _install_real_acq(env)
    r = rollout(env, args.ckpt, args.episodes, args.seed,
                args.label or args.nav_mode)
    show(r)
    r['rung'] = args.rung
    r['box'] = args.box or 'rung default'
    r['nav_mode'] = args.nav_mode
    r['acq'] = args.acq
    r['ckpt'] = args.ckpt
    r['episodes'] = args.episodes
    r['eval_seed'] = args.seed
    r['noise_mult'] = args.noise_mult
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        json.dump(r, open(args.out, 'w'), indent=1)
        print(f'  wrote {args.out}')
    return 0


def _cli_over(args):
    """Env-kwarg overrides carried by the CLI. Kept in ONE place so the anchor
    and the eval stages cannot drift apart on the box."""
    over = box_kw(args.box, args.box_radius_m, args.box_vel_ms)
    if args.shape_w_match is not None:
        over['shape_w_match'] = float(args.shape_w_match)
    return over


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--stage', default='anchor', choices=['anchor', 'eval'])
    p.add_argument('--rung', default='X3', choices=sorted(RUNGS))
    p.add_argument('--ckpt', default=ANCHOR_CKPT)
    p.add_argument('--nav-mode', default='truth',
                   choices=['truth', 'recon', 'rb_ekf', 'bearings_only'])
    p.add_argument('--acq', default='surrogate', choices=['surrogate', 'real'])
    p.add_argument('--episodes', type=int, default=200)
    p.add_argument('--seed', type=int, default=123)
    p.add_argument('--nav-seed', type=int, default=20260812)
    p.add_argument('--noise-mult', type=float, default=1.0)
    p.add_argument('--box', default='', choices=[''] + sorted(BOXES),
                   help='named success box; overrides the rung default')
    p.add_argument('--box-radius-m', type=float, default=None)
    p.add_argument('--box-vel-ms', type=float, default=None)
    # The TB3D ladder trained with orbital.py's DEFAULT shape_w_match (0.35) at
    # dv_ref 700, not the 0.8166667 the rung-1 nav campaign used. A nav arm
    # warm-started from a TB3D checkpoint must match ITS regime, or the
    # observation pipeline stops being the only variable.
    p.add_argument('--shape-w-match', type=float, default=None)
    p.add_argument('--expect', default='',
                   help='anchor stage: published score as N/M (default 200/200)')
    p.add_argument('--label', default='')
    p.add_argument('--out', default='')
    args = p.parse_args()
    return stage_anchor(args) if args.stage == 'anchor' else stage_eval(args)


if __name__ == '__main__':
    sys.exit(main())
