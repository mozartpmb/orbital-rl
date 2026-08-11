#!/usr/bin/env python3
"""T3 — adversarial physics fuzz of the CORRECTED orbital env.

Goal: establish (or refute) that no further dynamics bugs remain after the
true_to_mean fix (commit f55d9cb, 2026-08-10).

Method
------
Drive the real C env with adversarial random / state-conditioned action
sequences, pull the ground-truth per-sub-step trajectory out of the binding,
and re-simulate the SAME impulse sequence in an independent oracle
(scripts/orbital/nav/orbital_math.py, exact Lagrange f&g propagation written
for a separate EKF project — it shares no code path with the env's
element-wise Kepler propagator).

Checks
------
  L  true_to_mean is the correct inverse of eccentric_to_true (static/unit)
  A  step-local oracle re-sim of chaser state (burn + 60 s coast)
  G  step-local oracle re-sim of target state (pure coast)
  B  energy / angular-momentum invariance across coast sub-steps
  C  energy / angular-momentum jump across a burn matches the applied dv
  D  Tsiolkovsky fuel bookkeeping
  E  logged dv magnitude matches the action table (modulo fuel clamp)
  F  warp semantics: tau sub-steps per decision, zero dv
  J  logged elements <-> logged Cartesian self-consistency
  H  observation cross-check (obs[0..16], obs[17..32], obs[33..37])
  I  phase-gap sign convention vs LVLH along-track offset
  K  terminal-cause consistency with the terminal state
  M  reward = gated potential shaping delta (+ NHR clamp at terminal)

Everything is compared against a float32-quantization noise model, since the
trajectory log is float32 while the env runs in double. Per-check thresholds
are >= 8x the modelled noise floor.

Run:
    cd /Users/pete/space_training/pufferlib
    python3 ../scripts/orbital/t3/fuzz_dynamics.py --episodes-per-cell 12
"""

import argparse
import math
import os
import sys
import time
from collections import defaultdict

import numpy as np

REPO = "/Users/pete/space_training"
sys.path.insert(0, os.path.join(REPO, "pufferlib"))
sys.path.insert(0, os.path.join(REPO, "scripts", "orbital", "nav"))

import orbital_math as om  # noqa: E402
from pufferlib.ocean.orbital.orbital import Orbital, TRAJ_COLS  # noqa: E402
from pufferlib.ocean.orbital import binding  # noqa: E402

MU = 3.986004418e14
R_EARTH = 6.371e6
DT = 60.0
MAX_STEPS = 2000
ISP, G0 = 300.0, 9.80665
VE = ISP * G0
FUEL_FRAC = 0.15
DRY_MASS = 850.0
EARTH_KEEPOUT = R_EARTH + 200e3
SUCCESS_TOL_A = 10000.0
REL_VEL_TOL = 50.0
BETA_SHAPE = 1.0
W_ORBIT = W_PHASE = W_VEL = 0.01
EPS_ORBIT, EPS_PHASE = 2.0, 0.3
TAU_ORBIT, TAU_PHASE = 0.2, 0.03
GAMMA = 0.995

# (dv_prograde, dv_radial) — orbital.h ACTION_DV
ACTION_DV = [
    (0.0, 0.0), (5.0, 0.0), (10.0, 0.0), (25.0, 0.0),
    (-5.0, 0.0), (-10.0, 0.0), (-25.0, 0.0),
    (0.0, 10.0), (0.0, -10.0),
    (0.0, 0.0), (0.0, 0.0), (0.0, 0.0),
    (1.0, 0.0), (-1.0, 0.0), (2.0, 0.0), (-2.0, 0.0),
]
ACTION_TAU = (1, 1, 1, 1, 1, 1, 1, 1, 1, 5, 30, 60, 1, 1, 1, 1)
WARP_ACTIONS = {9, 10, 11}
BURN_ACTIONS = [a for a in range(16) if ACTION_TAU[a] == 1 and a != 0]

TERM_NAME = {0: "none", 1: "success", 2: "collision", 3: "escape",
             4: "safety_cap", 5: "stranded", 6: "hyperbolic", 7: "gave_up"}

COL = {c: i for i, c in enumerate(TRAJ_COLS)}
F32_HALF_ULP = 5.9604645e-8  # 0.5 * 2^-23


# ───────────────────────────────────────────────────────────── helpers ──
def energy(x, y, vx, vy):
    return 0.5 * (vx * vx + vy * vy) - MU / math.hypot(x, y)


def angmom(x, y, vx, vy):
    return x * vy - y * vx


def wrap_pi(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def compute_phi(sa, se, sw, sth, ta, te, tw, tth,
                phi_tol_eff=SUCCESS_TOL_A):
    """Python transcription of orbital.h compute_phi()."""
    da = abs(sa - ta)
    de = math.hypot(se * math.cos(sw) - te * math.cos(tw),
                    se * math.sin(sw) - te * math.sin(tw))
    phi_orbit = da / phi_tol_eff + de
    phi_phase = 1.0 - math.cos(sth - tth)

    sx, sy, svx, svy = om.orbit_to_cartesian(
        {'a': sa, 'e': se, 'theta': sth, 'omega': sw})
    tx, ty, tvx, tvy = om.orbit_to_cartesian(
        {'a': ta, 'e': te, 'theta': tth, 'omega': tw})
    theta_t = tth + tw
    ct, st = math.cos(theta_t), math.sin(theta_t)
    dvxi, dvyi = svx - tvx, svy - tvy
    dxi, dyi = sx - tx, sy - ty
    dvx_l = ct * dvxi + st * dvyi
    dvy_l = -st * dvxi + ct * dvyi
    dx_l = ct * dxi + st * dyi
    dy_l = -st * dxi + ct * dyi
    n_t = math.sqrt(MU / (ta ** 3))
    dvx_l += n_t * dy_l
    dvy_l -= n_t * dx_l
    phi_vel = math.hypot(dvx_l, dvy_l) / REL_VEL_TOL

    s2 = 1.0 / (1.0 + math.exp(-(EPS_ORBIT - phi_orbit) / TAU_ORBIT))
    s3 = s2 / (1.0 + math.exp(-(EPS_PHASE - phi_phase) / TAU_PHASE))
    return -(W_ORBIT * phi_orbit + W_PHASE * phi_phase * s2
             + W_VEL * phi_vel * s3)


class Acc:
    """Error accumulator: max, p99, count, worst-case context."""

    def __init__(self, name, thresh):
        self.name, self.thresh = name, thresh
        self.vals = []
        self.ctx = []
        self.worst = (-1.0, None)

    def add(self, v, ctx=None):
        if not math.isfinite(v):
            v = 1e30
        self.vals.append(v)
        if v > self.worst[0]:
            self.worst = (v, ctx)
        if v > self.thresh and len(self.ctx) < 40:
            self.ctx.append((v, ctx))

    def summary(self):
        if not self.vals:
            return dict(name=self.name, n=0, verdict="SKIP", max=0.0, p99=0.0,
                        p50=0.0, mean=0.0, thresh=self.thresh, n_over=0,
                        worst=None)
        a = np.asarray(self.vals, dtype=np.float64)
        n_over = int((a > self.thresh).sum())
        return dict(name=self.name, n=len(a),
                    verdict="PASS" if n_over == 0 else "FAIL",
                    max=float(a.max()), p99=float(np.percentile(a, 99)),
                    p50=float(np.percentile(a, 50)), mean=float(a.mean()),
                    thresh=self.thresh, n_over=n_over, worst=self.worst)


# ───────────────────────────────────────────────────────── check L (unit) ──
def c_true_to_mean_fixed(theta, e):
    """Transcription of the CURRENT orbital.h true_to_mean (post-fix)."""
    x = math.sqrt(1.0 + e) * math.cos(theta / 2.0)
    y = math.sqrt(1.0 - e) * math.sin(theta / 2.0)
    E = 2.0 * math.atan2(y, x)
    return E - e * math.sin(E)


def check_L(e_hi=0.15):
    """theta -> M -> (Kepler) -> E -> theta round trip must be identity.

    Restricted to the operational eccentricity domain; check_L2 sweeps e.
    """
    rng = np.random.default_rng(0)
    worst_M, worst_rt = 0.0, 0.0
    for _ in range(20000):
        e = float(rng.uniform(0.0, e_hi))
        th = float(rng.uniform(-math.pi, math.pi))
        M_ref = om.mean_from_true(th, e)
        M_c = c_true_to_mean_fixed(th, e)
        worst_M = max(worst_M, abs(wrap_pi(M_c - M_ref)))
        M = M_c % (2 * math.pi)
        E = om.solve_kepler(M, e)
        th_rt = om.eccentric_to_true(E, e)
        worst_rt = max(worst_rt, abs(wrap_pi(th_rt - th)))
    return worst_M, worst_rt


def check_L2():
    """5-iteration Newton solve_kepler convergence vs e (env uses 5 iters,
    initial guess E=M for e<0.8 else E=pi, early-out at |dE|<1e-12)."""
    rng = np.random.default_rng(3)
    out = []
    for e in (0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95):
        worst = 0.0
        for _ in range(4000):
            M = float(rng.uniform(0.0, 2 * math.pi))
            E = om.solve_kepler(M, e)                      # 5-iter, as in C
            res = abs(wrap_pi(E - e * math.sin(E) - M))    # Kepler residual
            worst = max(worst, res)
        # convert residual in M to along-track position error at LEO
        out.append((e, worst, worst * 7.0e6))
    return out


def check_R(n_env=512):
    """Does init_phase_gap_max / phase_gap_fixed control the PHYSICAL angular
    separation at reset?

    c_reset sets target.M = sat.M + phase_gap. That is a MEAN-ANOMALY offset in
    each body's own perifocal frame. The physically meaningful separation is
    the inertial-angle gap (theta+omega)_sat - (theta+omega)_tgt. When the two
    orbits have independent omega (any e_max_sat > 0 with same_orbit_init=0),
    the omega difference is uniform on [0, 2pi) and swamps the requested gap.
    """
    import numpy as _np
    cases = [
        ("e=0 both (circular)",
         dict(e_max_target=0.0, e_max_sat=0.0, same_orbit_init=0)),
        ("e=0.05 fixed, indep omega",
         dict(e_target_fixed=0.05, e_sat_fixed=0.05, same_orbit_init=0)),
        ("e=0.05 fixed, same_orbit_init=1",
         dict(e_target_fixed=0.05, e_sat_fixed=0.05, same_orbit_init=1)),
    ]
    rows = []
    for label, extra in cases:
        kw = dict(BASE_KW)
        kw.update(extra)
        kw['init_phase_gap_max'] = 0.0
        kw['phase_gap_fixed'] = math.radians(30.0)
        env = Orbital(num_envs=n_env, seed=7, **kw)
        obs, _ = env.reset(seed=7)
        o = _np.asarray(obs, dtype=_np.float64)
        th_s = _np.arctan2(o[:, 2], o[:, 3])
        w_s = _np.arctan2(o[:, 9], o[:, 10])
        th_t = _np.arctan2(o[:, 15], o[:, 16])
        w_t = _np.arctan2(o[:, 11], o[:, 12])
        phys = _np.degrees(_np.angle(_np.exp(1j * ((th_s + w_s) - (th_t + w_t)))))
        true_gap = _np.degrees(_np.angle(_np.exp(1j * (th_s - th_t))))
        env.close()
        rows.append((label, float(_np.mean(phys)), float(_np.std(phys)),
                     float(_np.percentile(_np.abs(phys), 5)),
                     float(_np.percentile(_np.abs(phys), 95)),
                     float(_np.mean(true_gap)), float(_np.std(true_gap))))
    return rows


def check_S():
    """Drift-orbit phasing budget vs the hard 2000-sub-step (33.3 h) clock.

    MAX_STEPS counts SUB-steps, not agent decisions, so time-warp buys decision
    economy but NOT wall-clock: every episode ends at 2000 * 60 s = 33.33 h.
    Under corrected dynamics the only way to change phase is a drift orbit:
    d(phase)/dt = -1.5 * (da/a) * n. Round-trip dv ~ v * |da| / a (two Hohmann
    pairs: out and back).
    """
    a = R_EARTH + 550e3
    n = math.sqrt(MU / a ** 3)
    v = math.sqrt(MU / a)
    horizon_h = MAX_STEPS * DT / 3600.0
    rows = []
    for da_km in (50, 100, 150, 200, 300, 400, 500):
        da = da_km * 1e3
        rate_deg_h = 1.5 * (da / a) * n * 3600.0 * 180.0 / math.pi
        t180 = 180.0 / rate_deg_h
        dv_round = 2.0 * (v * da / (2.0 * a))
        rows.append((da_km, rate_deg_h, t180, dv_round,
                     t180 <= horizon_h, dv_round <= 478.0))
    return horizon_h, rows


# ───────────────────────────────────────────────────────── scenarios ──
BASE_KW = dict(num_debris_min=0, num_debris_max=0, valid_init_only=1,
               init_phase_gap_max=math.pi, obs_alt_scale_m=1.6e6,
               lvlh_scale_m=6.371e6, phi_orbit_scale_k=0.001,
               rendezvous_radius_m=30000.0, rel_vel_tol_ms=50.0)

SCENARIOS = [
    ("S0_circular",      dict(e_max_target=0.0,  e_max_sat=0.0,  same_orbit_init=0)),
    ("S1_headline",      dict(e_max_target=0.05, e_max_sat=0.05, same_orbit_init=0)),
    ("S2_sameorbit",     dict(e_max_target=0.05, e_max_sat=0.05, same_orbit_init=1)),
    ("S3_hi_e",          dict(e_max_target=0.10, e_max_sat=0.10, same_orbit_init=0)),
    ("S4_hi_e_sameorb",  dict(e_max_target=0.10, e_max_sat=0.10, same_orbit_init=1)),
    ("S5_fixed_e10",     dict(e_target_fixed=0.10, e_sat_fixed=0.10, same_orbit_init=0)),
    ("S6_debris_path",   dict(e_max_target=0.05, e_max_sat=0.05, same_orbit_init=0,
                              num_debris_min=4, num_debris_max=8)),
]


# ───────────────────────────────────────────────────────── policies ──
def pol_uniform16(rng, obs):
    return int(rng.integers(0, 16))


def pol_burn_heavy(rng, obs):
    if rng.random() < 0.08:
        return int(rng.choice([0, 9, 10, 11]))
    return int(rng.choice(BURN_ACTIONS))


def pol_warp_burn(rng, obs):
    """Burn, then a long warp — stresses burn-then-warp-1hr sequences."""
    if rng.random() < 0.5:
        return int(rng.choice(BURN_ACTIONS))
    return int(rng.choice([9, 10, 11, 11]))


def _theta_gated(rng, obs, want, tol=0.30):
    """Burn only when sin(theta_sat) (obs[2]) / cos(theta_sat) (obs[3]) is
    near the requested quadrant; otherwise warp forward."""
    s, c = float(obs[2]), float(obs[3])
    if want == "quad90":
        hit = abs(s) > 1.0 - tol
    elif want == "peri":
        hit = c > 1.0 - tol
    else:  # apo
        hit = c < -1.0 + tol
    if hit:
        return int(rng.choice(BURN_ACTIONS))
    return int(rng.choice([0, 9]))


def pol_theta90(rng, obs):
    return _theta_gated(rng, obs, "quad90")


def pol_periapsis(rng, obs):
    return _theta_gated(rng, obs, "peri")


def pol_burst(rng, obs):
    """Bursts of consecutive identical burns (quick succession)."""
    if not hasattr(pol_burst, "st") or pol_burst.st[1] <= 0:
        pol_burst.st = (int(rng.choice(BURN_ACTIONS)), int(rng.integers(3, 12)))
    a, n = pol_burst.st
    pol_burst.st = (a, n - 1)
    return a


def pol_coast(rng, obs):
    r = rng.random()
    if r < 0.55:
        return 11
    if r < 0.85:
        return 10
    if r < 0.95:
        return 0
    return int(rng.choice(BURN_ACTIONS))


POLICIES = [("uniform16", pol_uniform16), ("burn_heavy", pol_burn_heavy),
            ("warp_burn", pol_warp_burn), ("theta90", pol_theta90),
            ("periapsis", pol_periapsis), ("burst", pol_burst),
            ("coast_warp", pol_coast)]


# ───────────────────────────────────────────────────────── episode run ──
def run_episode(env, buf, rng, policy, max_decisions=4000):
    """Step until terminal; return (traj array, list of (rec_idx, obs, action,
    reward, tau)) for the NON-terminal decisions."""
    obs = env.observations
    recs = []
    cum = 0
    term = False
    for _ in range(max_decisions):
        a = policy(rng, obs[0])
        prev_obs_cum = cum
        o, r, t, tr, _ = env.step(np.array([a], dtype=np.int32))
        done = bool(t[0])
        cum_after = cum + ACTION_TAU[a]
        if not done:
            recs.append((prev_obs_cum, cum_after, np.array(o[0], copy=True),
                         a, float(r[0])))
            cum = cum_after
        else:
            term = True
            break
    n = binding.vec_get_trajectory(env.c_envs, 0, buf)
    steps, cause = binding.vec_get_episode_result(env.c_envs, 0)
    return buf[:n].copy(), recs, cause, term


# ───────────────────────────────────────────────────────── analysis ──
def analyse(traj, recs, cause, kw, A):  # noqa: C901
    """Fill accumulators A (dict of Acc) from one episode."""
    n = traj.shape[0]
    if n < 2:
        return
    g = lambda c: traj[:, COL[c]].astype(np.float64)  # noqa: E731
    sx, sy, svx, svy = g('sat_x'), g('sat_y'), g('sat_vx'), g('sat_vy')
    sa, se, sth, sw = g('sat_a'), g('sat_e'), g('sat_theta'), g('sat_omega')
    tx, ty, tvx, tvy = g('target_x'), g('target_y'), g('target_vx'), g('target_vy')
    ta, te, tw = g('target_a'), g('target_e'), g('target_omega')
    fuel, act, rew, dv = g('fuel'), g('action').astype(int), g('reward'), g('delta_v')
    nb = g('num_bodies').astype(int)

    obs_alt = kw.get('obs_alt_scale_m', 1.6e6)
    lvlh_s = kw.get('lvlh_scale_m', 6.371e6)
    scale_dist = R_EARTH + obs_alt

    # target true anomaly: inertial angle = omega + theta (exact in 2D)
    tth = np.array([wrap_pi(math.atan2(ty[i], tx[i]) - tw[i]) for i in range(n)])

    # ── J: elements <-> cartesian self-consistency ────────────────────
    for i in range(n):
        if sa[i] <= 0:
            continue
        el = om.cartesian_to_elements(sx[i], sy[i], svx[i], svy[i])
        A['J_a'].add(abs(el['a'] - sa[i]), (i, sa[i], se[i]))
        A['J_e'].add(abs(el['e'] - se[i]), (i, sa[i], se[i]))
        # theta/omega individually ill-conditioned at e->0; their SUM is not
        A['J_thw'].add(abs(wrap_pi(el['theta'] + el['omega'] - sth[i] - sw[i])),
                       (i, sa[i], se[i]))

    # ── A / G: step-local oracle re-sim ───────────────────────────────
    for i in range(1, n):
        if sa[i - 1] <= 0 or sa[i] <= 0:
            continue
        st = [sx[i - 1], sy[i - 1], svx[i - 1], svy[i - 1]]
        d = dv[i]
        a_i = int(act[i])
        if d > 0.0:
            pro, rad = ACTION_DV[a_i]
            vm = math.hypot(st[2], st[3])
            rm = math.hypot(st[0], st[1])
            ux = pro * st[2] / vm + rad * st[0] / rm
            uy = pro * st[3] / vm + rad * st[1] / rm
            um = math.hypot(ux, uy)
            if um < 1e-12:
                continue
            st[2] += d * ux / um
            st[3] += d * uy / um
        px, py, pvx, pvy = om.propagate_cartesian(tuple(st), DT)
        ep = math.hypot(px - sx[i], py - sy[i])
        ev = math.hypot(pvx - svx[i], pvy - svy[i])
        ctx = dict(i=i, act=a_i, dv=d, e_prev=se[i - 1], th_prev=sth[i - 1],
                   a_prev=sa[i - 1])
        A['A_pos'].add(ep, ctx)
        A['A_vel'].add(ev, ctx)
        if d > 0.0:
            A['A_pos_burn'].add(ep, ctx)
            # stratify: burns near the 90/270 quadrant with e in [0.02, 0.10]
            if abs(math.sin(sth[i - 1])) > 0.7 and 0.02 <= se[i - 1] <= 0.12:
                A['A_pos_quad_e'].add(ep, ctx)
        else:
            A['A_pos_coast'].add(ep, ctx)

        tp = om.propagate_cartesian((tx[i - 1], ty[i - 1], tvx[i - 1], tvy[i - 1]), DT)
        A['G_pos'].add(math.hypot(tp[0] - tx[i], tp[1] - ty[i]),
                       dict(i=i, e=te[i], a=ta[i]))
        A['G_vel'].add(math.hypot(tp[2] - tvx[i], tp[3] - tvy[i]),
                       dict(i=i, e=te[i], a=ta[i]))

    # ── B / C: invariants ─────────────────────────────────────────────
    for i in range(1, n):
        if sa[i - 1] <= 0 or sa[i] <= 0:
            continue
        E0 = energy(sx[i - 1], sy[i - 1], svx[i - 1], svy[i - 1])
        E1 = energy(sx[i], sy[i], svx[i], svy[i])
        h0 = angmom(sx[i - 1], sy[i - 1], svx[i - 1], svy[i - 1])
        h1 = angmom(sx[i], sy[i], svx[i], svy[i])
        v0 = math.hypot(svx[i - 1], svy[i - 1])
        r0 = math.hypot(sx[i - 1], sy[i - 1])
        # float32 quantization noise on E and h
        sE = v0 * (v0 * F32_HALF_ULP) + (MU / (r0 * r0)) * (r0 * F32_HALF_ULP)
        sh = 2.0 * (r0 * F32_HALF_ULP * v0 + v0 * F32_HALF_ULP * r0)
        sE, sh = max(sE, 1e-6), max(sh, 1e-6)
        if dv[i] == 0.0:
            A['B_E'].add(abs(E1 - E0) / sE, dict(i=i, a=sa[i], e=se[i]))
            A['B_h'].add(abs(h1 - h0) / sh, dict(i=i, a=sa[i], e=se[i]))
        else:
            pro, rad = ACTION_DV[int(act[i])]
            vm, rm = v0, r0
            ux = pro * svx[i - 1] / vm + rad * sx[i - 1] / rm
            uy = pro * svy[i - 1] / vm + rad * sy[i - 1] / rm
            um = math.hypot(ux, uy)
            if um < 1e-12:
                continue
            dvx, dvy = dv[i] * ux / um, dv[i] * uy / um
            dE_pred = svx[i - 1] * dvx + svy[i - 1] * dvy + 0.5 * dv[i] ** 2
            dh_pred = sx[i - 1] * dvy - sy[i - 1] * dvx
            A['C_E'].add(abs((E1 - E0) - dE_pred) / sE,
                         dict(i=i, act=int(act[i]), dv=dv[i], e=se[i - 1],
                              th=sth[i - 1]))
            A['C_h'].add(abs((h1 - h0) - dh_pred) / sh,
                         dict(i=i, act=int(act[i]), dv=dv[i], e=se[i - 1]))

    # ── D / E: fuel + dv table ────────────────────────────────────────
    for i in range(1, n):
        if dv[i] <= 0.0:
            continue
        f0, f1 = fuel[i - 1], fuel[i]
        m0 = DRY_MASS / (1.0 - f0)
        fm0, fm1 = DRY_MASS * f0 / (1.0 - f0), DRY_MASS * f1 / (1.0 - f1)
        dm_pred = m0 * (1.0 - math.exp(-dv[i] / VE))
        A['D_fuel'].add(abs((fm0 - fm1) - dm_pred),
                        dict(i=i, dv=dv[i], f0=f0))
        want = abs(ACTION_DV[int(act[i])][0]) + abs(ACTION_DV[int(act[i])][1])
        clamped = fm1 <= 1e-9
        if not clamped:
            A['E_dv'].add(abs(dv[i] - want), dict(i=i, act=int(act[i]), dv=dv[i]))
        else:
            A['E_dv_clamped'].add(max(0.0, dv[i] - want - 1e-6),
                                  dict(i=i, act=int(act[i]), dv=dv[i]))
        if int(act[i]) in WARP_ACTIONS or int(act[i]) == 0:
            A['F_warp_dv'].add(dv[i], dict(i=i, act=int(act[i])))

    # ── F: warp tau bookkeeping (non-terminal decisions only) ─────────
    for (i0, i1, o, a, r) in recs:
        A['F_tau'].add(abs((i1 - i0) - ACTION_TAU[a]), dict(i=i1, act=a))
        if a in WARP_ACTIONS:
            seg = dv[i0 + 1:i1 + 1]
            A['F_warp_dv'].add(float(np.abs(seg).max()) if seg.size else 0.0,
                               dict(i=i1, act=a))

    # ── H / I: observation cross-check ────────────────────────────────
    for (i0, i, o, a, r) in recs:
        if i >= n or sa[i] <= 0:
            continue
        rr = math.hypot(sx[i], sy[i])
        vr = (sx[i] * svx[i] + sy[i] * svy[i]) / rr
        vt = (sx[i] * svy[i] - sy[i] * svx[i]) / rr
        vc = math.sqrt(MU / rr)
        exp = {
            0: (sa[i] - R_EARTH) / obs_alt,
            1: se[i],
            2: math.sin(sth[i]), 3: math.cos(sth[i]),
            4: vr / vc, 5: vt / vc, 6: fuel[i],
            7: (ta[i] - R_EARTH) / obs_alt, 8: te[i],
            9: math.sin(sw[i]), 10: math.cos(sw[i]),
            11: math.sin(tw[i]), 12: math.cos(tw[i]),
            13: math.sin(sth[i] - tth[i]), 14: math.cos(sth[i] - tth[i]),
            15: math.sin(tth[i]), 16: math.cos(tth[i]),
        }
        for k, v in exp.items():
            A['H_obs%02d' % k].add(abs(float(o[k]) - v), dict(i=i, k=k))

        # Earth body slot (obs[17..20]) — Earth is always bodies[0] at origin
        # and, with no debris, always the closest body.
        if nb[i] == 1:
            sat_ang = math.atan2(sy[i], sx[i])
            exp_b = {17: rr / scale_dist, 18: wrap_pi(sat_ang) / math.pi,
                     19: vr / vc, 20: EARTH_KEEPOUT / scale_dist}
            for k, v in exp_b.items():
                A['H_obs%02d' % k].add(abs(float(o[k]) - v), dict(i=i, k=k))
            for k in range(21, 33):
                A['H_obs_pad'].add(abs(float(o[k])), dict(i=i, k=k))

        # LVLH block
        theta_t = tth[i] + tw[i]
        ct, st_ = math.cos(theta_t), math.sin(theta_t)
        dxi, dyi = sx[i] - tx[i], sy[i] - ty[i]
        dvxi, dvyi = svx[i] - tvx[i], svy[i] - tvy[i]
        dx_l = ct * dxi + st_ * dyi
        dy_l = -st_ * dxi + ct * dyi
        dvx_l = ct * dvxi + st_ * dvyi
        dvy_l = -st_ * dvxi + ct * dvyi
        n_t = math.sqrt(MU / (ta[i] ** 3))
        dvx_l += n_t * dy_l
        dvy_l -= n_t * dx_l
        vct = math.sqrt(MU / ta[i])
        expl = {33: dx_l / lvlh_s, 34: dy_l / lvlh_s,
                35: dvx_l / vct, 36: dvy_l / vct, 37: n_t / 1e-3}
        for k, v in expl.items():
            A['H_obs%02d' % k].add(abs(float(o[k]) - v), dict(i=i, k=k))

        # I: phase sign — obs[13] (sin of true-anomaly gap) vs LVLH
        # along-track offset obs[34]. Consistent iff same sign whenever the
        # physical along-track gap is unambiguous.
        phys = wrap_pi((sth[i] + sw[i]) - (tth[i] + tw[i]))
        if abs(phys) > 0.05 and abs(phys) < math.pi - 0.05:
            A['I_sign_lvlh'].add(0.0 if (np.sign(phys) == np.sign(dy_l)) else 1.0,
                                 dict(i=i, phys=phys, dy_l=dy_l))
            same_w = abs(wrap_pi(sw[i] - tw[i])) < 1e-6
            bad = 0.0 if (np.sign(math.sin(wrap_pi(sth[i] - tth[i])))
                          == np.sign(phys)) else 1.0
            gap_err = abs(math.degrees(wrap_pi(wrap_pi(sth[i] - tth[i]) - phys)))
            c2 = dict(i=i, phys=phys, dth_true=wrap_pi(sth[i] - tth[i]),
                      dw=wrap_pi(sw[i] - tw[i]))
            if same_w:
                A['I_sign_obs13_sameW'].add(bad, c2)
                A['I_gap_err_deg_sameW'].add(gap_err, c2)
            else:
                A['I_sign_obs13_diffW'].add(bad, c2)
                A['I_gap_err_deg_diffW'].add(gap_err, c2)

    # ── N: (omega, theta) split conditioning across an impulse ────────
    # An impulse cannot move the satellite, so the inertial angle (theta+omega)
    # is invariant across a burn. theta ALONE is not: at small e the
    # eccentricity vector is tiny and a burn rotates omega by O(1) rad, with
    # theta absorbing the compensating jump. obs[2],obs[3],obs[13],obs[14] and
    # Phi_phase are all functions of theta alone.
    for i in range(1, n):
        if dv[i] <= 0.0 or sa[i - 1] <= 0:
            continue
        pro, rad = ACTION_DV[int(act[i])]
        vm = math.hypot(svx[i - 1], svy[i - 1])
        rm = math.hypot(sx[i - 1], sy[i - 1])
        ux = pro * svx[i - 1] / vm + rad * sx[i - 1] / rm
        uy = pro * svy[i - 1] / vm + rad * sy[i - 1] / rm
        um = math.hypot(ux, uy)
        if um < 1e-12:
            continue
        el = om.cartesian_to_elements(sx[i - 1], sy[i - 1],
                                      svx[i - 1] + dv[i] * ux / um,
                                      svy[i - 1] + dv[i] * uy / um)
        dth = abs(math.degrees(wrap_pi(el['theta'] - sth[i - 1])))
        dw = abs(math.degrees(wrap_pi(el['omega'] - sw[i - 1])))
        dinert = abs(math.degrees(wrap_pi(el['theta'] + el['omega']
                                          - sth[i - 1] - sw[i - 1])))
        ctx = dict(i=i, act=int(act[i]), dv=dv[i], e_pre=se[i - 1],
                   e_post=el['e'], dth_deg=dth, dw_deg=dw)
        A['N_inertial_inv'].add(dinert, ctx)
        A['N_theta_jump'].add(dth, ctx)
        if se[i - 1] < 0.005:
            A['N_theta_jump_loe'].add(dth, ctx)

    # ── M: reward = shaping delta ─────────────────────────────────────
    prev_phi = None
    for (i0, i, o, a, r) in recs:
        if i >= n or sa[i] <= 0 or sa[i0] <= 0:
            continue
        phi_prev = compute_phi(sa[i0], se[i0], sw[i0], sth[i0],
                               ta[i0], te[i0], tw[i0], tth[i0])
        phi_cur = compute_phi(sa[i], se[i], sw[i], sth[i],
                              ta[i], te[i], tw[i], tth[i])
        pred = BETA_SHAPE * (GAMMA ** ACTION_TAU[a] * phi_cur - phi_prev)
        A['M_reward'].add(abs(pred - r), dict(i=i, act=a, pred=pred, got=r))

    # ── P: shaping audit — how much of Phi_phase is a frame artifact ──
    # Phi_phase = 1 - cos(theta_sat - theta_tgt) uses TRUE ANOMALY, which is
    # measured from each body's own periapsis. The physically meaningful gap is
    # the inertial-angle gap (theta+omega). Compare the per-decision shaping
    # delta the env actually pays against the same delta computed with the
    # inertial-angle phase, and record the gate state.
    for (i0, i, o, a, r) in recs:
        if i >= n or sa[i] <= 0 or sa[i0] <= 0:
            continue
        def phi_terms(k):
            da = abs(sa[k] - ta[k])
            de = math.hypot(se[k] * math.cos(sw[k]) - te[k] * math.cos(tw[k]),
                            se[k] * math.sin(sw[k]) - te[k] * math.sin(tw[k]))
            po = da / SUCCESS_TOL_A + de
            s2 = 1.0 / (1.0 + math.exp(-(EPS_ORBIT - po) / TAU_ORBIT))
            pp_true = 1.0 - math.cos(sth[k] - tth[k])
            pp_inert = 1.0 - math.cos((sth[k] + sw[k]) - (tth[k] + tw[k]))
            return po, s2, pp_true, pp_inert
        po1, s2_1, ppt1, ppi1 = phi_terms(i)
        po0, s2_0, ppt0, ppi0 = phi_terms(i0)
        gt = GAMMA ** ACTION_TAU[a]
        d_true = W_PHASE * (gt * ppt1 * s2_1 - ppt0 * s2_0)
        d_inert = W_PHASE * (gt * ppi1 * s2_1 - ppi0 * s2_0)
        A['P_gate_open'].add(1.0 if s2_1 > 0.5 else 0.0, dict(i=i, phi_orbit=po1))
        A['P_phase_artifact'].add(abs(d_true - d_inert),
                                  dict(i=i, act=a, s2=s2_1,
                                       dth_true=wrap_pi(sth[i] - tth[i]),
                                       dth_inert=wrap_pi(sth[i] + sw[i]
                                                         - tth[i] - tw[i])))
        if s2_1 > 0.5:
            A['P_phase_artifact_open'].add(abs(d_true - d_inert),
                                           dict(i=i, act=a, s2=s2_1))
            A['P_phase_true_open'].add(abs(d_true), dict(i=i, act=a))
            if abs(d_true) > 1e-9:
                A['P_artifact_ratio_open'].add(
                    abs(d_true - d_inert) / abs(d_true),
                    dict(i=i, act=a, d_true=d_true, d_inert=d_inert))

    # ── Q: discrete-sampling collision tunnelling ─────────────────────
    # Termination samples r only at 60 s boundaries. A LEO chaser covers
    # ~460 km per sub-step, so an arc whose perigee dips below the surface
    # between two sampled points is never detected.
    for i in range(1, n):
        if sa[i] <= 0 or se[i] >= 1.0:
            continue
        th0 = sth[i - 1]
        if dv[i] > 0.0:
            pro, rad = ACTION_DV[int(act[i])]
            vm = math.hypot(svx[i - 1], svy[i - 1])
            rm = math.hypot(sx[i - 1], sy[i - 1])
            ux = pro * svx[i - 1] / vm + rad * sx[i - 1] / rm
            uy = pro * svy[i - 1] / vm + rad * sy[i - 1] / rm
            um = math.hypot(ux, uy)
            if um < 1e-12:
                continue
            th0 = om.cartesian_to_elements(sx[i - 1], sy[i - 1],
                                           svx[i - 1] + dv[i] * ux / um,
                                           svy[i - 1] + dv[i] * uy / um)['theta']
        adv = (sth[i] - th0) % (2.0 * math.pi)
        crosses_peri = ((-th0) % (2.0 * math.pi)) < adv
        r0 = math.hypot(sx[i - 1], sy[i - 1])
        r1 = math.hypot(sx[i], sy[i])
        r_min = sa[i] * (1.0 - se[i]) if crosses_peri else min(r0, r1)
        ctx = dict(i=i, act=int(act[i]), r_min=r_min, r0=r0, r1=r1,
                   a=sa[i], e=se[i], adv_deg=math.degrees(adv))
        A['Q_tunnel_surface'].add(
            1.0 if (r_min < R_EARTH and min(r0, r1) >= R_EARTH) else 0.0, ctx)
        A['Q_tunnel_keepout'].add(
            1.0 if (r_min < EARTH_KEEPOUT and min(r0, r1) >= EARTH_KEEPOUT)
            else 0.0, ctx)
        A['Q_arc_deg'].add(math.degrees(adv), ctx)
        A['Q_tunnel_depth_m'].add(max(0.0, min(r0, r1) - r_min), ctx)

    # ── Q2: debris conjunction sampling ───────────────────────────────
    if nb[0] > 1:
        for i in range(n):
            dmin = 1e30
            for b in range(1, int(nb[i])):
                bx = traj[i, COL['body_x_%d' % b]]
                by = traj[i, COL['body_y_%d' % b]]
                dmin = min(dmin, math.hypot(sx[i] - bx, sy[i] - by))
            if dmin < 1e29:
                A['Q_debris_min_km'].add(-dmin / 1000.0, dict(i=i, d_km=dmin / 1e3))

    # ── O: cumulative re-sim vs float32-ULP divergence envelope ───────
    # Re-simulate the whole impulse sequence from record 0 forward without
    # re-seeding from the log, and compare the final error against the spread
    # produced by perturbing the seed state by +/-1 float32 ULP. A systematic
    # dynamics error shows up as drift outside the envelope.
    if analyse.do_cumulative and n >= 120:
        rngp = np.random.default_rng(1234 + n)

        def resim(seed_state):
            st = list(seed_state)
            for i in range(1, n):
                if sa[i - 1] <= 0:
                    return None
                d = dv[i]
                if d > 0.0:
                    pro, rad = ACTION_DV[int(act[i])]
                    vm = math.hypot(st[2], st[3])
                    rm = math.hypot(st[0], st[1])
                    ux = pro * st[2] / vm + rad * st[0] / rm
                    uy = pro * st[3] / vm + rad * st[1] / rm
                    um = math.hypot(ux, uy)
                    if um < 1e-12:
                        return None
                    st[2] += d * ux / um
                    st[3] += d * uy / um
                st = list(om.propagate_cartesian(tuple(st), DT))
            return st

        seed0 = (sx[0], sy[0], svx[0], svy[0])
        base = resim(seed0)
        if base is not None:
            err = math.hypot(base[0] - sx[n - 1], base[1] - sy[n - 1])
            env_max = 0.0
            for _ in range(4):
                p = [seed0[j] * (1.0 + F32_HALF_ULP * 2.0
                                 * float(rngp.integers(-1, 2))) for j in range(4)]
                alt = resim(p)
                if alt is None:
                    continue
                env_max = max(env_max, math.hypot(alt[0] - base[0],
                                                  alt[1] - base[1]))
            env_max = max(env_max, 1.0)
            A['O_cum_pos'].add(err / env_max,
                               dict(n=n, err_m=err, envelope_m=env_max))

    # ── K: terminal-cause consistency ─────────────────────────────────
    j = n - 1
    if sa[j] > 0:
        d = math.hypot(sx[j] - tx[j], sy[j] - ty[j])
        rv = math.hypot(svx[j] - tvx[j], svy[j] - tvy[j])
        rr = math.hypot(sx[j], sy[j])
        Ej = energy(sx[j], sy[j], svx[j], svy[j])
        rad_m = kw.get('rendezvous_radius_m', 30000.0)
        rvt = kw.get('rel_vel_tol_ms', 50.0)
        bad = 0.0
        if cause == 1 and not (d < rad_m * 1.02 and rv < rvt * 1.02):
            bad = 1.0
        if cause == 2 and rr > R_EARTH * 1.001:
            bad = 1.0
        if cause == 3 and Ej < -1.0:
            bad = 1.0
        if cause == 5 and fuel[j] > 1e-4:
            bad = 1.0
        if cause == 4 and j < MAX_STEPS - 1:
            bad = 1.0
        A['K_term'].add(bad, dict(cause=cause, d=d, rv=rv, r=rr, E=Ej,
                                  fuel=fuel[j], j=j))


analyse.do_cumulative = False


# ───────────────────────────────────────────────────────── main ──
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--episodes-per-cell', type=int, default=6)
    ap.add_argument('--seed', type=int, default=20260810)
    ap.add_argument('--max-decisions', type=int, default=4000)
    ap.add_argument('--cumulative-per-cell', type=int, default=1)
    ap.add_argument('--out', default=os.path.join(REPO, 'web_data', 'results'))
    args = ap.parse_args()

    # thresholds: >= 8x modelled float32 noise floor
    A = {
        'A_pos': Acc('A chaser step-local |dpos| (m)', 10.0),
        'A_vel': Acc('A chaser step-local |dvel| (m/s)', 0.02),
        'A_pos_burn': Acc('A  ...on burn sub-steps (m)', 10.0),
        'A_pos_coast': Acc('A  ...on coast sub-steps (m)', 10.0),
        'A_pos_quad_e': Acc('A  ...burn @|sin th|>0.7, e in [.02,.12] (m)', 10.0),
        'G_pos': Acc('G target step-local |dpos| (m)', 10.0),
        'G_vel': Acc('G target step-local |dvel| (m/s)', 0.02),
        'B_E': Acc('B coast dE / sigma_E (z)', 20.0),
        'B_h': Acc('B coast dh / sigma_h (z)', 20.0),
        'C_E': Acc('C burn dE residual / sigma_E (z)', 20.0),
        'C_h': Acc('C burn dh residual / sigma_h (z)', 20.0),
        'D_fuel': Acc('D Tsiolkovsky dm residual (kg)', 1e-3),
        'E_dv': Acc('E |dv - table| unclamped (m/s)', 1e-4),
        'E_dv_clamped': Acc('E clamped dv overshoot (m/s)', 1e-4),
        'F_tau': Acc('F sub-steps per decision - ACTION_TAU', 0.5),
        'F_warp_dv': Acc('F dv on warp/coast actions (m/s)', 1e-9),
        'J_a': Acc('J |a_cart - a_log| (m)', 5.0),
        'J_e': Acc('J |e_cart - e_log|', 1e-5),
        'J_thw': Acc('J |(th+w)_cart - (th+w)_log| (rad)', 1e-5),
        'H_obs_pad': Acc('H empty body slots == 0', 1e-9),
        'I_sign_lvlh': Acc('I sign(phys gap) == sign(LVLH dy) [1=bad]', 0.0),
        'I_sign_obs13_sameW': Acc('I sign(obs13) wrong, w_sat==w_tgt [1=bad]', 0.0),
        'I_sign_obs13_diffW': Acc('I sign(obs13) wrong, w_sat!=w_tgt [1=bad]', 0.0),
        'I_gap_err_deg_sameW': Acc('I |true-anom - physical| gap, same w (deg)', 1.0),
        'I_gap_err_deg_diffW': Acc('I |true-anom - physical| gap, diff w (deg)', 1.0),
        'N_inertial_inv': Acc('N |d(theta+omega)| across impulse (deg)', 1e-3),
        'N_theta_jump': Acc('N |d theta| across impulse (deg)', 1e-3),
        'N_theta_jump_loe': Acc('N  ...for e_pre < 0.005 (deg)', 1e-3),
        'O_cum_pos': Acc('O cumulative re-sim |dpos| / ULP envelope', 8.0),
        'P_gate_open': Acc('P sigma2 gate open rate [audit, 1=open]', 1.1),
        'P_phase_artifact': Acc('P |shaping_true - shaping_inertial| (reward)', 1e9),
        'P_phase_artifact_open': Acc('P  ...gate-open decisions only', 1e9),
        'P_phase_true_open': Acc('P |phase shaping paid|, gate open', 1e9),
        'Q_tunnel_surface': Acc('Q undetected sub-surface arc [1=bad]', 0.0),
        'Q_tunnel_keepout': Acc('Q undetected sub-keepout arc [1=bad]', 0.0),
        'Q_arc_deg': Acc('Q true-anomaly arc per 60 s sub-step (deg)', 1e9),
        'Q_debris_min_km': Acc('Q debris closest approach, -km (>-5 = keepout)', -5.0),
        'Q_tunnel_depth_m': Acc('Q unsampled dip below sampled r_min (m)', 1e9),
        'P_artifact_ratio_open': Acc('P artifact / |phase shaping paid|, gate open', 1e9),
        'M_reward': Acc('M |reward - shaping delta|', 2e-5),
        'K_term': Acc('K terminal cause inconsistent [1=bad]', 0.0),
    }
    for k in list(range(0, 21)) + list(range(33, 38)):
        A['H_obs%02d' % k] = Acc('H obs[%d]' % k, 3e-6 if k not in (0, 7, 37) else 3e-5)

    print('== Check L: true_to_mean inverse (unit test, 20k draws, e<=0.15) ==')
    wM, wRT = check_L()
    print(f'   max |M_c - M_ref|          = {wM:.3e} rad')
    print(f'   max |theta -> M -> theta|  = {wRT:.3e} rad')
    L_pass = wM < 1e-12 and wRT < 1e-9
    print(f'   verdict: {"PASS" if L_pass else "FAIL"}')

    print('\n== Check L2: 5-iteration solve_kepler residual vs e ==')
    print(f'   {"e":>6s} {"max |M_resid| (rad)":>21s} {"~along-track err (m)":>21s}')
    l2 = check_L2()
    for e, res, m in l2:
        print(f'   {e:6.2f} {res:21.3e} {m:21.3e}')
    L2_pass = all(res < 1e-9 for e, res, _ in l2 if e <= 0.15)
    print(f'   verdict (operational e<=0.15): {"PASS" if L2_pass else "FAIL"}')

    print('\n== Check R: does phase_gap_fixed=30 deg set the PHYSICAL gap? ==')
    print(f'   {"case":36s} {"phys mean":>10s} {"phys sd":>9s} '
          f'{"|phys| p5":>10s} {"|phys| p95":>11s} {"trueanom mean":>14s} '
          f'{"sd":>7s}')
    for lab, m, sd, p5, p95, tm, tsd in check_R():
        print(f'   {lab:36s} {m:10.2f} {sd:9.2f} {p5:10.2f} {p95:11.2f} '
              f'{tm:14.2f} {tsd:7.2f}')

    hz, srows = check_S()
    print(f'\n== Check S: drift-orbit phasing vs the {hz:.1f} h episode clock '
          f'(a = 550 km alt) ==')
    print(f'   {"|da| km":>8s} {"deg/h":>8s} {"h for 180deg":>13s} '
          f'{"round-trip dv":>14s} {"fits clock":>11s} {"fits fuel":>10s}')
    for da, rate, t180, dv_r, okc, okf in srows:
        print(f'   {da:8d} {rate:8.2f} {t180:13.1f} {dv_r:14.1f} '
              f'{str(okc):>11s} {str(okf):>10s}')

    buf = np.zeros((MAX_STEPS, len(TRAJ_COLS)), dtype=np.float32)
    rows = []
    ep_total = 0
    t0 = time.time()
    for si, (sname, skw) in enumerate(SCENARIOS):
        kw = dict(BASE_KW)
        kw.update(skw)
        for pi, (pname, pol) in enumerate(POLICIES):
            seed = args.seed + 1009 * si + 31 * pi
            env = Orbital(num_envs=1, seed=seed, **kw)
            env.reset(seed=seed)
            rng = np.random.default_rng(seed)
            causes = defaultdict(int)
            for _ek in range(args.episodes_per_cell):
                analyse.do_cumulative = (_ek < args.cumulative_per_cell)
                traj, recs, cause, term = run_episode(
                    env, buf, rng, pol, args.max_decisions)
                causes[cause] += 1
                ep_total += 1
                analyse(traj, recs, cause, kw, A)
            env.close()
            rows.append((sname, pname, dict(causes)))
            print(f'  {sname:18s} {pname:11s} n={args.episodes_per_cell:3d} '
                  f'causes={ {TERM_NAME[k]: v for k, v in causes.items()} }')

    dt = time.time() - t0
    print(f'\n{ep_total} episodes in {dt:.1f}s\n')

    # ── report ────────────────────────────────────────────────────────
    order = ([k for k in A if not k.startswith('H_obs')]
             + sorted(k for k in A if k.startswith('H_obs')))
    print(f'{"check":52s} {"n":>8s} {"mean":>11s} {"p50":>11s} {"p99":>11s} '
          f'{"max":>11s} {"thresh":>10s} {"over":>6s}  verdict')
    print('-' * 138)
    out = []
    for k in order:
        s = A[k].summary()
        print(f'{s["name"]:52s} {s["n"]:8d} {s["mean"]:11.3e} {s["p50"]:11.3e} '
              f'{s["p99"]:11.3e} {s["max"]:11.3e} {s["thresh"]:10.3e} '
              f'{s["n_over"]:6d}  {s["verdict"]}')
        out.append(dict(key=k, **{kk: s[kk] for kk in
                                  ('name', 'n', 'mean', 'p50', 'p99', 'max',
                                   'thresh', 'n_over', 'verdict')}))

    print('\n== anomalies (first few contexts over threshold) ==')
    any_bad = False
    for k in order:
        s = A[k].summary()
        if s['verdict'] == 'FAIL':
            any_bad = True
            print(f'  {s["name"]}: {s["n_over"]}/{s["n"]} over {s["thresh"]:.2e}')
            for v, c in A[k].ctx[:5]:
                print(f'      val={v:.4e}  ctx={c}')
    if not any_bad:
        print('  (none)')

    print('\n== worst-case context per check ==')
    for k in order:
        w = A[k].worst
        if w[1] is not None:
            print(f'  {A[k].name:52s} max={w[0]:.3e}  ctx={w[1]}')

    os.makedirs(args.out, exist_ok=True)
    import csv
    p = os.path.join(args.out, 't3_fuzz_dynamics.csv')
    with open(p, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['key', 'name', 'n', 'mean', 'p50',
                                          'p99', 'max', 'thresh', 'n_over',
                                          'verdict'])
        w.writeheader()
        for r in out:
            w.writerow(r)
    print(f'\nwrote {p}')

    p2 = os.path.join(args.out, 't3_fuzz_coverage.csv')
    with open(p2, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['scenario', 'policy', 'episodes', 'terminal_causes'])
        for sname, pname, causes in rows:
            w.writerow([sname, pname, args.episodes_per_cell,
                        ';'.join(f'{TERM_NAME[k]}={v}' for k, v in sorted(causes.items()))])
    print(f'wrote {p2}')


if __name__ == '__main__':
    main()
