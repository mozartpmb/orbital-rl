#!/usr/bin/env python3
"""ext-3d anchor A1 — adversarial 3D dynamics fuzz of the real C env.

Structure mirrors scripts/orbital/t3/fuzz_dynamics.py; the checks come from the
sibling recon's MUTATION-TESTED battery, imported unmodified:

    scripts/orbital/ext_recon/ext_invariants3d.py :: INVARIANTS
    scripts/orbital/ext_recon/orbital_math3d.py   :: the oracle

The oracle is deliberately a different algebra from the env — universal
variables + Stumpff c2/c3 with bracketed Newton, signed atan2 element recovery,
no acos-plus-if-sign anywhere — so agreement is evidence rather than a shared
assumption. The battery was mutation-tested at 15/15 seeded 3D bug classes
caught with zero dead checks, which is why it is used verbatim rather than
re-derived here.

What this harness supplies that the 2D one could not: the env now logs the TRUE
post-impulse, pre-propagation state (`burn_post_*`), so I7 / I8 / I9 / I11 / I15
test the environment instead of testing a reconstruction of the burn against
itself.

Conventions and their justification:
  · rv_pre  = logged state at sub-step k−1, rv_end = logged state at sub-step k,
    dt = 60 s. Every row is one sub-step, warps included.
  · rv_post = burn_post_* when the row carries a burn, else rv_pre.
  · Angles are wrapped into [0, 2π) before I14 sees them. The env's EQUATORIAL
    branch deliberately returns ω, θ on (−π, π] — those are the verbatim legacy
    statements that make the 2D anchor bit-exact — so the branch is a documented
    reporting convention, not a domain violation. I14 still tests the inclined
    branch, which is where new code lives.
  · Thresholds are ext_invariants3d.FLOAT32_THRESHOLDS: the trajectory log is
    float32 while the env runs in double.

Run (from the pufferlib dir):
    python3 ../scripts/orbital/ext3d/fuzz_dynamics3d.py --episodes-per-cell 10
"""
import argparse
import math
import os
import sys
import time
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'pufferlib'))
sys.path.insert(0, os.path.join(ROOT, 'scripts', 'orbital', 'ext_recon'))

import orbital_math3d as o3                                        # noqa: E402
from ext_invariants3d import INVARIANTS, FLOAT32_THRESHOLDS        # noqa: E402
from pufferlib.ocean.orbital.orbital import Orbital, TRAJ_COLS     # noqa: E402
from pufferlib.ocean.orbital import binding                        # noqa: E402

COL = {c: i for i, c in enumerate(TRAJ_COLS)}
DT = 60.0
TWO_PI = 2.0 * math.pi

# orbital.h ACTION_DV / ACTION_TAU, Discrete-30
ACTION_DV = [
    (0, 0, 0), (5, 0, 0), (10, 0, 0), (25, 0, 0), (-5, 0, 0), (-10, 0, 0),
    (-25, 0, 0), (0, 10, 0), (0, -10, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0),
    (1, 0, 0), (-1, 0, 0), (2, 0, 0), (-2, 0, 0), (0, 0, 0), (0, 0, 0),
    (0, 1, 0), (0, -1, 0),
    (0, 0, 1), (0, 0, -1), (0, 0, 10), (0, 0, -10), (0, 0, 25), (0, 0, -25),
    (25, 0, 25), (25, 0, -25), (-25, 0, 25), (-25, 0, -25),
]
ACTION_TAU = [1] * 30
for a, t in ((9, 5), (10, 30), (11, 60), (16, 180), (17, 360)):
    ACTION_TAU[a] = t
WARP_ACTIONS = {9, 10, 11, 16, 17}
INPLANE_BURNS = [1, 2, 3, 4, 5, 6, 7, 8, 12, 13, 14, 15, 18, 19]
NORMAL_BURNS = [20, 21, 22, 23, 24, 25]
COMBINED_BURNS = [26, 27, 28, 29]
ALL_BURNS = INPLANE_BURNS + NORMAL_BURNS + COMBINED_BURNS

TERM_NAME = {0: "none", 1: "success", 2: "collision", 3: "escape",
             4: "safety_cap", 5: "stranded", 6: "hyperbolic", 7: "gave_up"}


# ─────────────────────────────────────────────────────────────── policies ──
# obs[21] = δı⃗·R̂_s / scale, obs[22] = δı⃗·T̂_s / scale: the relative node sits
# where the T̂ component vanishes. Using the real observation for the node/
# antinode policies exercises the channel as well as the dynamics.
def _node_phase(o):
    r, t = float(o[21]), float(o[22])
    m = math.hypot(r, t)
    return (abs(t) / m if m > 1e-9 else 0.0,
            abs(r) / m if m > 1e-9 else 0.0)


# Restricted per-cell burn pool. The C0 planar-anchor cell must never take an
# out-of-plane action or I13 is measuring the harness, not the env.
BURN_POOL = list(ALL_BURNS)


def pol_uniform(rng, o):
    return int(rng.integers(0, 30))


def pol_inplane(rng, o):
    r = rng.random()
    if r < 0.15:
        return 0
    if r < 0.35:
        return int(rng.choice([9, 10, 11]))
    return int(rng.choice(INPLANE_BURNS))


def pol_normal(rng, o):
    r = rng.random()
    if r < 0.25:
        return int(rng.choice([0, 9, 10]))
    return int(rng.choice(NORMAL_BURNS))


def pol_node(rng, o):
    frac_t, _ = _node_phase(o)
    if frac_t < 0.20:
        return int(rng.choice(NORMAL_BURNS + COMBINED_BURNS))
    return 9


def pol_antinode(rng, o):
    _, frac_r = _node_phase(o)
    if frac_r < 0.20:
        return int(rng.choice(NORMAL_BURNS + COMBINED_BURNS))
    return 9


def pol_burst(rng, o):
    st = getattr(pol_burst, 'st', None)
    if st is None or st[1] <= 0:
        pol_burst.st = (int(rng.choice(BURN_POOL)), int(rng.integers(3, 12)))
    a, n = pol_burst.st
    pol_burst.st = (a, n - 1)
    return a


def pol_coast(rng, o):
    r = rng.random()
    if r < 0.5:
        return 11
    if r < 0.8:
        return 10
    if r < 0.92:
        return 0
    return int(rng.choice(BURN_POOL))


def pol_combined(rng, o):
    r = rng.random()
    if r < 0.2:
        return int(rng.choice([0, 9]))
    return int(rng.choice(COMBINED_BURNS))


POLICIES = [('uniform', pol_uniform), ('inplane_only', pol_inplane),
            ('normal_only', pol_normal), ('normal_at_node', pol_node),
            ('normal_at_antinode', pol_antinode), ('burn_burst', pol_burst),
            ('coast_heavy', pol_coast), ('combined_only', pol_combined)]


# ──────────────────────────────────────────────────────── traj -> battery ──
def build_traj(arr):
    """Convert the env's float32 trajectory rows into the battery's format."""
    g = lambda c: arr[:, COL[c]].astype(np.float64)               # noqa: E731
    x, y, z = g('sat_x'), g('sat_y'), g('sat_z')
    vx, vy, vz = g('sat_vx'), g('sat_vy'), g('sat_vz')
    a, e = g('sat_a'), g('sat_e')
    inc, raan, argp, nu = g('sat_inc'), g('sat_raan'), g('sat_omega'), g('sat_theta')
    act = g('action').astype(int)
    dvm = g('delta_v')
    fuel = g('fuel')
    bpx, bpy, bpz = g('burn_post_x'), g('burn_post_y'), g('burn_post_z')
    bvx, bvy, bvz = g('burn_post_vx'), g('burn_post_vy'), g('burn_post_vz')

    out = []
    for k in range(1, arr.shape[0]):
        if a[k] <= 0 or a[k - 1] <= 0:
            continue                                   # hyperbolic terminal row
        rv_pre = np.array([x[k - 1], y[k - 1], z[k - 1],
                           vx[k - 1], vy[k - 1], vz[k - 1]])
        rv_end = np.array([x[k], y[k], z[k], vx[k], vy[k], vz[k]])
        d = float(dvm[k])
        if d > 0.0:
            rv_post = np.array([bpx[k], bpy[k], bpz[k], bvx[k], bvy[k], bvz[k]])
            pro, rad, nor = ACTION_DV[int(act[k])]
        else:
            rv_post = rv_pre.copy()
            pro = rad = nor = 0.0
        out.append(dict(
            rv_pre=rv_pre, rv_post=rv_post, rv_end=rv_end, dt=DT,
            dv_pro=float(pro), dv_rad=float(rad), dv_nor=float(nor),
            dv_mag=d, fuel=float(fuel[k]),
            el=dict(a=float(a[k]), e=float(e[k]), i=float(inc[k]),
                    raan=float(raan[k]) % TWO_PI,
                    argp=float(argp[k]) % TWO_PI,
                    nu=float(nu[k]) % TWO_PI)))
    return out


def I15abs_transverse(traj):
    """Harness-added, scale-free companion to I15.

    I15 reports the ANGLE between the realised and commanded Δv, with a fixed
    0.1 m/s exclusion calibrated for a DOUBLE-precision log. Against a float32
    log that exclusion is wrong by construction (trap T20 — the noise model must
    scale per column): the realised Δv is a difference of two float32 velocity
    vectors whose components are ~7.6e3 m/s, so it carries an ABSOLUTE error of
    about sqrt(3)·ulp ≈ 1.7e-3 m/s, and the resulting angle floor is that over
    |Δv| — measured 3.9e-4 rad at 1 m/s decaying as 1/|Δv| to 1.2e-5 at 35 m/s,
    exactly the 1/|Δv| signature. Fuel-CLAMPED burns land arbitrarily close to
    the 0.1 m/s cut and blow the angle up further.

    So the angle form is gated at |Δv| >= --dv-floor, and this check carries the
    small-burn stratum in the units where the floor is constant:

        max over burns of  |Δv| · angle(realised, commanded)   [m/s]

    i.e. the transverse component of the Δv error. Bounded by the float32 floor
    (~1.7e-3 m/s) for a correct env, and O(|Δv|) — i.e. 1-25 m/s, three to four
    decades larger — for any of the burn-frame bug classes the battery indicts
    (B1 normal-along-ẑ, B12 normal-forced-north, B13 radial-inertial)."""
    w = 0.0
    for r in traj:
        if r['dv_mag'] <= 0.0:
            continue
        rv = r['rv_pre']
        vv, rr = rv[3:], rv[:3]
        nvec = np.cross(rr, vv)
        pro = vv / np.linalg.norm(vv)
        rad = rr / np.linalg.norm(rr)
        nor = nvec / np.linalg.norm(nvec)
        want = r['dv_pro'] * pro + r['dv_rad'] * rad + r['dv_nor'] * nor
        got = r['rv_post'][3:] - rv[3:]
        nw, ng = float(np.linalg.norm(want)), float(np.linalg.norm(got))
        if nw < 1e-12 or ng < 1e-12:
            continue
        a, b = want / nw, got / ng
        ang = math.atan2(float(np.linalg.norm(np.cross(a, b))), float(np.dot(a, b)))
        w = max(w, r['dv_mag'] * ang)
    return w


class Acc:
    def __init__(self, name, thresh):
        self.name, self.thresh, self.vals, self.n_over = name, thresh, [], 0
        self.worst = (-1.0, None)

    def add(self, v, ctx=None):
        if not np.isfinite(v):
            v = 1e30
        self.vals.append(v)
        if v > self.worst[0]:
            self.worst = (v, ctx)
        if v > self.thresh:
            self.n_over += 1

    def row(self):
        if not self.vals:
            return (self.name, 0, 'SKIP', 0.0, 0.0, self.thresh, 0, None)
        arr = np.asarray(self.vals)
        return (self.name, len(arr), 'PASS' if self.n_over == 0 else 'FAIL',
                float(arr.max()), float(np.percentile(arr, 99)), self.thresh,
                self.n_over, self.worst[1])


BASE = dict(
    num_debris_min=0, num_debris_max=0, valid_init_only=1,
    gave_up_action="terminate",
    init_phase_gap_max=math.pi, obs_alt_scale_m=1.6e6, lvlh_scale_m=6.371e6,
    phi_orbit_scale_k=0.001, rendezvous_radius_m=30000.0, rel_vel_tol_ms=50.0,
    shaping_mode=2, shape_w_lambda=1.0, shape_w_match=0.8166667,
    shape_dv_ref_ms=700.0, shape_gamma=1.0,
    phase_gap_mode=1, phase_obs_mode=1, episode_cap_steps=3000,
    cap_terminal_reward=0.0, legacy_action_space=30,
    dim3_mode=1, traj_log_every=10 ** 9,
)

D = math.pi / 180.0
SCENARIOS = [
    # (label, env kwarg delta). i_t is pure gauge, so it is swept purely as an
    # adversary against the conversion branch table and the target-plane gauge.
    ('C0_eq_planar',   dict(i_target_rad=0.0,      di_max_rad=-1.0,
                            e_max_target=0.0, e_max_sat=0.0)),
    ('C1_eq_di0.4',    dict(i_target_rad=0.0,      di_max_rad=0.40 * D,
                            e_max_target=0.05, e_max_sat=0.05)),
    ('C2_eq_di2.0',    dict(i_target_rad=0.0,      di_max_rad=2.00 * D,
                            e_max_target=0.05, e_max_sat=0.05)),
    ('C3_iss_di1.0',   dict(i_target_rad=51.6 * D, di_max_rad=1.00 * D,
                            e_max_target=0.05, e_max_sat=0.05)),
    ('C4_iss_di2.0_e', dict(i_target_rad=51.6 * D, di_max_rad=2.00 * D,
                            e_max_target=0.30, e_max_sat=0.30,
                            de_max=0.08, da_max_m=600e3,
                            a_min_override=6.671e6, a_max_override=1.4371e7,
                            episode_cap_steps=6000)),
    ('C5_polar_di2.0', dict(i_target_rad=90.0 * D, di_max_rad=2.00 * D,
                            e_max_target=0.05, e_max_sat=0.05)),
]
# The planar anchor cell (I13) needs a trajectory that never leaves z = 0.
PLANAR_CELL = 'C0_eq_planar'
PLANAR_POLICIES = ('inplane_only', 'coast_heavy')


def run_episode(env, buf, rng, policy, max_decisions=4000):
    obs = env.observations
    for _ in range(max_decisions):
        a = policy(rng, obs[0])
        _, _, t, _, _ = env.step(np.array([a], dtype=np.int32))
        if bool(t[0]):
            break
    n = binding.vec_get_trajectory(env.c_envs, 0, buf)
    _, cause = binding.vec_get_episode_result(env.c_envs, 0)
    return buf[:n].copy(), cause


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--episodes-per-cell', type=int, default=5)
    ap.add_argument('--seed', type=int, default=20260811)
    ap.add_argument('--max-decisions', type=int, default=1200)
    ap.add_argument('--dv-floor', type=float, default=2.0,
                    help='|dv| below which the ANGLE form of I15 is not '
                         'resolvable against a float32 log (m/s); that stratum '
                         'is carried by I15abs instead')
    args = ap.parse_args()

    accs = {}
    for label, fn, thr_d, applies in INVARIANTS:
        key = label.split()[0]
        accs[key] = (Acc(label, FLOAT32_THRESHOLDS[key]), fn, applies)
    _ALL = lambda s, p: True                                       # noqa: E731
    accs['I15abs'] = (Acc('I15abs |dv|·angle, transverse err (m/s)', 1e-2),
                      I15abs_transverse, _ALL)
    # I15's own fixed 0.1 m/s exclusion is a double-precision calibration; see
    # I15abs_transverse.__doc__. Re-gate it at the float32-log floor.
    _i15_acc, _i15_fn, _i15_ap = accs['I15']
    _floor = args.dv_floor
    accs['I15'] = (_i15_acc,
                   lambda tr: _i15_fn([r for r in tr if r['dv_mag'] >= _floor]),
                   _i15_ap)

    buf = np.zeros((12000, len(TRAJ_COLS)), dtype=np.float32)
    causes = defaultdict(int)
    n_ep = n_rows = n_burn = n_normal = 0
    t0 = time.time()
    tmp = '/tmp/ext3d_fuzz_traj'
    os.makedirs(tmp, exist_ok=True)

    for si, (sname, sdelta) in enumerate(SCENARIOS):
        planar = (sname == PLANAR_CELL)
        globals()['BURN_POOL'] = list(INPLANE_BURNS) if planar else list(ALL_BURNS)
        for pi, (pname, pol) in enumerate(POLICIES):
            if planar and pname not in PLANAR_POLICIES:
                # C0 is the 2D-compat anchor: any out-of-plane action leaves
                # z = 0 and I13 would be measuring the harness, not the env.
                continue
            kw = dict(BASE)
            kw.update(sdelta)
            kw['traj_log_dir'] = tmp
            seed = args.seed + 1009 * si + 31 * pi
            env = Orbital(num_envs=1, seed=seed, **kw)
            env.reset(seed=seed)
            rng = np.random.default_rng(seed)
            if hasattr(pol_burst, 'st'):
                del pol_burst.st
            for _ in range(args.episodes_per_cell):
                arr, cause = run_episode(env, buf, rng, pol, args.max_decisions)
                causes[TERM_NAME[cause]] += 1
                traj = build_traj(arr)
                if not traj:
                    continue
                n_ep += 1
                n_rows += len(traj)
                n_burn += sum(1 for r in traj if r['dv_mag'] > 0)
                n_normal += sum(1 for r in traj if r['dv_nor'] != 0.0)
                # I13 applies only where the battery says it does.
                scen_key = ('C0_circ_eq' if sname == PLANAR_CELL else sname)
                pol_key = pname
                for key, (acc, fn, applies) in accs.items():
                    if not applies(scen_key, pol_key):
                        continue
                    acc.add(float(fn(traj)), dict(scen=sname, pol=pname))
            env.close()

    print(f"\n=== ext-3d A1 dynamics fuzz — {n_ep} episodes, {n_rows} sub-steps, "
          f"{n_burn} burns ({n_normal} with a normal component), "
          f"{time.time() - t0:.0f} s ===")
    print("terminal causes:", dict(causes))
    print(f"\n{'invariant':46s} {'n':>5s} {'max':>11s} {'p99':>11s} "
          f"{'thresh':>9s} {'over':>5s}  verdict")
    n_fail = 0
    for key, (acc, _, _) in accs.items():
        name, n, verdict, mx, p99, thr, over, worst = acc.row()
        if verdict == 'FAIL':
            n_fail += 1
        print(f"{name:46s} {n:5d} {mx:11.3e} {p99:11.3e} {thr:9.1e} "
              f"{over:5d}  {verdict}"
              + (f"   worst={worst}" if verdict == 'FAIL' else ""))
    print(f"\nOVERALL: {'PASS' if n_fail == 0 else f'FAIL ({n_fail} checks)'}")
    return 0 if n_fail == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
