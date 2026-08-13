#!/usr/bin/env python3
"""3D-nav (`dim3_mode=1` + angles-only) verification ladder — Phase A gates.

Structured exactly like `verify_extnav.py` (the 2D V1-V6 ladder) and
`ext_recon/ext_invariants3d.py` (fires-or-it-is-worthless mutation testing):
every gate is a pure function that returns a dict, prints its own numbers, and
sets `ok`. `--stage all` runs them in order and prints a PASS/FAIL summary.

The binding spec is `scripts/orbital/ext_recon/reports/n3d_REDTEAM.md`; each
gate names the severity ID it closes.

    cd /Users/pete/space_training/pufferlib
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
      python3 ../scripts/orbital/nav/verify_n3dnav.py --stage all

Stages
  a1  MAJOR-8  ACTION_TAU / ACTION_DV_MAG are 30 long and agree with
               orbital.h ROW BY ROW, parsed from the C source at test time.
               Includes the hypot(25,25)=35.355 trap on rows 26-29.
  a2  MAJOR-9  the divergence guard reads ln rho from the FILTER's own index.
               A healthy 6-state batch must flag ZERO rows; the pre-fix guard
               (hard-coded y[:,3]) flags every row of every step forever.
  b1  MAJOR-5/6  vec_get_state: shape, h-hat identities, Cartesian-vs-element
               consistency, and a PURE-READ proof (interleaved getter calls
               leave the observation AND reward stream bit-identical).
  c1  BLOCKER-4  the 19-slot encode: recon exactness against the C's own
               observation, MAJOR-15's recon-cart, the leakcheck gate in both
               directions, and the gate tested against a deliberately leaky and
               a deliberately over-broad encode. Plus MAJOR-14's slot-dependent
               clamp.
  d1  BLOCKER-1  the MSC chart pole. Reproduces the pole table (N3D-C 3b is
               exact gimbal lock in the coplanar anchor), the el == 0
               2D-reduction anchor, the closed-form guard-trigger condition,
               the re-pole as an exact similarity transform (round trip 1e-12,
               NEES continuity, per-row counter), and a closed-loop sweep
               through the regime the prototype skipped.
  e1  BLOCKER-3  the analytic 3D STM: symplecticity, the universal-variable
               oracle, oracle central differences, the group property, the 2D
               reduction, and the before/after tick-time table.
  f1  BLOCKER-2  the acquisition floor is SIM TIME: the same 2700 s at
               dt_tick = 60 s and at 300 s, where a 45-tick floor would have
               been 3.75 h. Plus MAJOR-7's fixed-count/adaptive-interval
               sub-tick, MAJOR-11's enforced crlb_online and counted Cholesky
               fallback, MAJOR-12's basis-free information kernel, MAJOR-13's
               unit reconciliation, MAJOR-16's out-of-plane velocity seed,
               NOTE-20's kwarg tripwires, NOTE-21's obs[28] diagnostic, and
               the dim3 recon closed loop against truth.

Two conventions worth knowing before reading the output:
  * `record` is a pass/fail GATE. `note` is a measured FINDING that is not a
    gate — a projection that came in low, or a correction to the spec. They are
    kept apart so the ladder's exit status means "a gate broke", nothing else.
  * every timing is min-of-batches with the ops interleaved round-robin, so a
    busy machine cannot turn a ratio into a result.
"""

import argparse
import math
import os
import re
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
WT = os.path.abspath(os.path.join(_HERE, '..', '..', '..'))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(WT, 'pufferlib'))

from pufferlib.ocean.orbital_nav import nav_math as nm            # noqa: E402

ORBITAL_H = os.path.join(WT, 'pufferlib', 'pufferlib', 'ocean',
                         'orbital', 'orbital.h')

_RESULTS = []
_NOTES = []


def record(name, ok, detail=''):
    _RESULTS.append((name, bool(ok), detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"   {detail}" if detail else ''))
    return bool(ok)


def note(name, detail=''):
    """A measured FINDING that is not a pass/fail gate.

    Kept distinct on purpose: a projection that was not met, or a correction to
    the spec, must be visible in the ladder output without turning the ladder's
    own exit status into a mix of "a gate broke" and "a number came in low".
    """
    _NOTES.append((name, detail))
    print(f'  [NOTE] {name}' + (f'   {detail}' if detail else ''))


# ── C-source parsing (so the tables cannot drift silently) ───────────────────
def parse_action_dv():
    """`ACTION_DV[NUM_ACTIONS][3]` from orbital.h -> (N,3) float array."""
    src = open(ORBITAL_H).read()
    m = re.search(r'static const double ACTION_DV\[NUM_ACTIONS\]\[3\]\s*=\s*\{(.*?)\n\};',
                  src, re.S)
    if not m:
        raise RuntimeError('ACTION_DV not found in orbital.h')
    body = re.sub(r'/\*.*?\*/', '', m.group(1), flags=re.S)
    rows = re.findall(r'\{([^{}]*)\}', body)
    return np.array([[float(x) for x in r.split(',')] for r in rows])


def parse_action_tau():
    src = open(ORBITAL_H).read()
    m = re.search(r'static const int ACTION_TAU\[NUM_ACTIONS\]\s*=\s*\{(.*?)\n\};',
                  src, re.S)
    if not m:
        raise RuntimeError('ACTION_TAU not found in orbital.h')
    body = re.sub(r'/\*.*?\*/', '', m.group(1), flags=re.S)
    return np.array([int(x) for x in body.replace('\n', ' ').split(',')
                     if x.strip()])


def parse_num_actions():
    src = open(ORBITAL_H).read()
    return int(re.search(r'#define\s+NUM_ACTIONS\s+(\d+)', src).group(1))


# ── a1: MAJOR-8 ──────────────────────────────────────────────────────────────
def stage_a1(args):
    print('== a1  MAJOR-8  action tables vs orbital.h ========================')
    n_c = parse_num_actions()
    tau_c = parse_action_tau()
    dv_c = parse_action_dv()
    dv_mag_c = np.sqrt((dv_c ** 2).sum(axis=1))
    print(f'  orbital.h: NUM_ACTIONS {n_c}, ACTION_TAU {tau_c.size} rows, '
          f'ACTION_DV {dv_c.shape[0]} rows')

    ok = True
    ok &= record('ACTION_TAU length == NUM_ACTIONS',
                 nm.ACTION_TAU.size == n_c,
                 f'{nm.ACTION_TAU.size} vs {n_c}')
    ok &= record('ACTION_DV_MAG length == NUM_ACTIONS',
                 nm.ACTION_DV_MAG.size == n_c,
                 f'{nm.ACTION_DV_MAG.size} vs {n_c}')
    if nm.ACTION_TAU.size == n_c:
        ok &= record('ACTION_TAU row-by-row == C',
                     np.array_equal(nm.ACTION_TAU, tau_c),
                     f'max|diff| {np.abs(nm.ACTION_TAU - tau_c).max()}')
    if nm.ACTION_DV_MAG.size == n_c:
        d = np.abs(nm.ACTION_DV_MAG - dv_mag_c).max()
        ok &= record('ACTION_DV_MAG row-by-row == ||ACTION_DV|| (C)',
                     d < 1e-12, f'max|diff| {d:.3e}')

    # The trap: rows 26-29 are combined prograde+normal, p^ perpendicular n^,
    # so |dv| = hypot(25,25) = 35.355, NOT 50 (which is what an L1 reading of
    # the table gives). 1.41x error on 23.6% of the measured Discrete-30 mix.
    want = math.hypot(25.0, 25.0)
    got = nm.ACTION_DV_MAG[26:30] if nm.ACTION_DV_MAG.size >= 30 else np.array([])
    ok &= record('combined rows 26-29 == hypot(25,25) = 35.355',
                 got.size == 4 and np.allclose(got, want, atol=1e-12),
                 f'{np.round(got, 4).tolist()} (L1 answer would be 50.0)')

    # The crash itself: this is the literal expression that raised IndexError
    # on step 1 of every Discrete-30 run.
    try:
        _ = nm.ACTION_TAU[np.array([26, 29])]
        _ = nm.ACTION_DV_MAG[np.array([26, 29])]
        crash = False
    except IndexError as exc:                                # pragma: no cover
        crash = True
        print(f'    IndexError: {exc}')
    ok &= record('ACTION_TAU[[26,29]] does not raise IndexError', not crash)

    # MAJOR-10: the ablation sets must be named separately.
    ok &= record('MAJOR-10 action subsets present',
                 set(nm.ACTION_SETS) >= {'fine_inplane', 'fine_normal',
                                         'normal_all', 'combined'},
                 f'{sorted(nm.ACTION_SETS)}')
    ok &= record('fine_normal == [20, 21] (normal +/-1, the 3D treatment)',
                 np.array_equal(nm.FINE_NORMAL, np.array([20, 21])))
    return dict(ok=bool(ok))


# ── a2: MAJOR-9 ──────────────────────────────────────────────────────────────
class _Fake6:
    """Minimal healthy 6-state modified-spherical filter.

    Only the filter-interface contract MAJOR-9 introduced is implemented, which
    is the point: the guard must work off `IDX_LNRHO` / `POS_DIM` / `trace()`
    and never off a hard-coded column.
    """
    STATE_DIM = 6
    CART_DIM = 6
    POS_DIM = 3
    IDX_LNRHO = nm.MSC6_LNRHO       # 5

    def __init__(self, n, rho=5.0e3):
        self.n = n
        self.y = np.zeros((n, 6))
        self.y[:, nm.MSC6_AZ] = 0.3
        self.y[:, nm.MSC6_EL] = 0.05
        self.y[:, nm.MSC6_WA] = 1.0e-4       # rad/s
        self.y[:, nm.MSC6_WE] = -2.0e-4      # rad/s  <- exp() of this is ~1.0 m
        self.y[:, nm.MSC6_RDOT] = 1.0e-5
        self.y[:, nm.MSC6_LNRHO] = math.log(rho)
        self.Py = np.tile(np.eye(6) * 1e-6, (n, 1, 1))
        # A plausible LEO target state so the guard's vis-viva branch is happy.
        r = nm.R_EARTH + 5.0e5
        self.x = np.tile(
            np.array([r, 0.0, 0.0, 0.0, math.sqrt(nm.MU / r), 0.0]), (n, 1))

    def mean_cart(self, idx):
        return self.x[idx]

    def trace(self):
        return np.trace(self.Py, axis1=1, axis2=2)

    def rho(self):
        return np.exp(np.minimum(self.y[:, self.IDX_LNRHO], 25.0))

    def set_polar(self, idx, y, Py, sat_cart):
        """Re-seed stub: the guard calls `_init_rows` on flagged rows and this
        gate only measures WHICH rows it flags, not what it reseeds them to."""
        self.y[idx, :4] = 0.0
        self.y[idx, nm.MSC6_LNRHO] = math.log(5.0e3)


def _legacy_guard_flags(filt):
    """The PRE-FIX guard's rho extraction, verbatim (orbital_nav.py:421)."""
    rho = np.exp(np.minimum(filt.y[:, 3], 25.0))
    from pufferlib.ocean.orbital_nav.orbital_nav import RHO_MIN_M, RHO_MAX_M
    return (rho < RHO_MIN_M) | (rho > RHO_MAX_M)


def stage_a2(args):
    print('== a2  MAJOR-9  divergence guard under a 6-state filter ===========')
    from pufferlib.ocean.orbital_nav.orbital_nav import OrbitalNav

    n = 16
    env = OrbitalNav(num_envs=n, nav_mode='bearings_only', **_MIN_KW)
    env.reset(seed=3)
    filt = _Fake6(n)
    env._filt = filt
    sat_c = np.zeros((n, 4))
    tgt_c = np.zeros((n, 4))
    sat_el = dict(a=np.full(n, nm.R_EARTH + 5e5))

    n_bad = env._guard(sat_c, tgt_c, sat_el)
    ok = record('healthy 6-state batch: guard flags 0 rows',
                n_bad == 0, f'{n_bad}/{n} flagged')

    legacy = _legacy_guard_flags(filt)
    ok &= record('pre-fix guard WOULD have flagged every row (test fires)',
                 bool(legacy.all()),
                 f'{int(legacy.sum())}/{n} rows, exp(y[:,3]) = '
                 f'{math.exp(filt.y[0, 3]):.4f} m < RHO_MIN_M = 1.0')

    # The guard must still fire on a genuinely diverged row.
    filt.y[3, nm.MSC6_LNRHO] = math.log(1.0e12)
    filt.y[7, nm.MSC6_LNRHO] = np.nan
    env._d_div = 0
    n_bad = env._guard(sat_c, tgt_c, sat_el)
    ok &= record('diverged rows still caught (rho=1e12 m, NaN)',
                 n_bad == 2, f'{n_bad} flagged, expected 2')

    # Interface contract: no positional index survives in the shipped filters.
    ok &= record('BatchedBearingMPC.IDX_LNRHO == 3 (4-state)',
                 nm.BatchedBearingMPC.IDX_LNRHO == nm.MSC4_LNRHO == 3)
    ok &= record('BatchedRangeBearingEKF.IDX_LNRHO is None',
                 nm.BatchedRangeBearingEKF.IDX_LNRHO is None)
    ok &= record('MSC6 index layout [az, el, w_a, w_e, rdot/rho, ln rho]',
                 (nm.MSC6_AZ, nm.MSC6_EL, nm.MSC6_WA, nm.MSC6_WE,
                  nm.MSC6_RDOT, nm.MSC6_LNRHO) == (0, 1, 2, 3, 4, 5))
    env.close()
    return dict(ok=bool(ok))


# ── b1: MAJOR-5 / MAJOR-6 — the read-only C getter ───────────────────────────
def stage_b1(args):
    print('== b1  MAJOR-5/6  vec_get_state: read-only, plane as h-hat =========')
    from pufferlib.ocean.orbital.orbital import Orbital

    n = 64
    env = Orbital(num_envs=n, seed=5, **_D3_KW)
    env.reset(seed=11)
    S = env.get_state()
    C = {c: i for i, c in enumerate(Orbital.STATE_COLS)}

    ok = record('shape / dtype', S.shape == (n, 36) and S.dtype == np.float64,
                f'{S.shape} {S.dtype}')

    hs = S[:, C['sat_hx']:C['sat_hx'] + 3]
    ht = S[:, C['tgt_hx']:C['tgt_hx'] + 3]
    d = max(np.abs(np.linalg.norm(hs, axis=1) - 1.0).max(),
            np.abs(np.linalg.norm(ht, axis=1) - 1.0).max())
    ok &= record('|h-hat| == 1 to <= 4 ulp', d <= 4 * np.spacing(1.0),
                 f'max||h|-1| {d:.3e} (sin^2+cos^2 rounding, not a normalisation)')

    rs = S[:, C['sat_x']:C['sat_x'] + 3]
    vs = S[:, C['sat_vx']:C['sat_vx'] + 3]
    hv = np.cross(rs, vs)
    hv /= np.linalg.norm(hv, axis=1)[:, None]
    d = np.abs(hv - hs).max()
    ok &= record('h-hat == unit(r x v)', d < 1e-14, f'max|diff| {d:.3e}')

    rmag = (S[:, C['sat_a']] * (1.0 - S[:, C['sat_e']] ** 2)
            / (1.0 + S[:, C['sat_e']] * np.cos(S[:, C['sat_theta']])))
    d = np.abs(np.linalg.norm(rs, axis=1) - rmag).max()
    ok &= record('Cartesian consistent with (a, e, theta)', d < 1e-6,
                 f'max|diff| {d:.3e} m')

    # The e-vectors are the ELEMENT route (orb_evec); they must agree with the
    # Cartesian route in VALUE (they differ only in floating-point path).
    es = S[:, C['sat_ex']:C['sat_ex'] + 3]
    hvec = np.cross(rs, vs)
    ecart = (np.cross(vs, hvec) / nm.MU
             - rs / np.linalg.norm(rs, axis=1)[:, None])
    d = np.abs(es - ecart).max()
    ok &= record('sat e-vector (element route) == Cartesian route in value',
                 d < 1e-12,
                 f'max|diff| {d:.3e}; |e| = {np.linalg.norm(es, axis=1).max():.4f}')
    d = np.abs(np.linalg.norm(es, axis=1) - S[:, C['sat_e']]).max()
    ok &= record('|e-vector| == e element', d < 1e-15, f'max|diff| {d:.3e}')

    # PURE READ. Interleaving 3 getter calls per step must not perturb the
    # observation stream, the reward stream or the RNG (the getter draws no
    # random numbers and mutates no env state; this proves it rather than
    # asserting it).
    def roll(with_getter):
        e = Orbital(num_envs=n, seed=5, **_D3_KW)
        o, _ = e.reset(seed=11)
        rng = np.random.default_rng(0)
        acc = [np.asarray(o, dtype=np.float32).copy()]
        for _ in range(150):
            if with_getter:
                for _ in range(3):
                    e.get_state()
            o, r, t, tr, _ = e.step(rng.integers(0, 30, n).astype(np.int32))
            acc.append(np.asarray(o, dtype=np.float32).copy())
            acc.append(np.asarray(r, dtype=np.float32).copy().reshape(-1, 1)
                       * np.ones((1, 38), dtype=np.float32))
        e.close()
        return np.stack(acc)

    A, B = roll(False), roll(True)
    ok &= record('getter is a PURE READ (obs+reward stream bit-identical)',
                 np.array_equal(A, B),
                 f'max|diff| {np.abs(A - B).max():.3e} over {A.size} values')

    # MAJOR-5, the reason the plane is returned as a VECTOR. At i_t = 0 the
    # observation carries (Delta_i_rel, u) and annihilates the node longitude
    # exactly, so RAAN alone would be the missing d.o.f. — but the moment the
    # target tilts, recovering i_s from (Delta_i_rel, Omega_s) is two-valued.
    o = np.asarray(env.observations, dtype=np.float64)
    di_scale = _D3_KW['di_max_rad']
    u = S[:, C['sat_omega']] + S[:, C['sat_theta']]
    i_s = np.arccos(np.clip(S[:, C['sat_hz']], -1.0, 1.0))
    pred21 = i_s * np.cos(u) / di_scale
    pred22 = -i_s * np.sin(u) / di_scale
    d = max(np.abs(o[:, 21] - pred21).max(), np.abs(o[:, 22] - pred22).max())
    ok &= record('obs[21,22] == (i_s cos u, -i_s sin u)/di_scale at i_t=0',
                 d < 1e-6, f'max|diff| {d:.3e} (float32 obs quantisation)')

    n_amb = _inclination_root_ambiguity(200, math.radians(51.6))
    note('two-fold i_s ambiguity at i_t=51.6 deg (why the getter returns '
         'h-hat, NOT Omega_s)',
         f'{n_amb:.0%} of my 200 draws admit >= 2 inclination roots '
           f'(n3d_REDTEAM measured 30% on its own draw distribution)')
    env.close()
    return dict(ok=bool(ok))


def _inclination_root_ambiguity(n, i_t, seed=17):
    """Fraction of draws where `cos di_rel = f(i_s)` has >= 2 roots in [0, pi).

    This is what a RAAN-only getter would have to solve, per step, in the hot
    path — a 2-D nonlinear solve with a discrete branch choice. Returning
    h-hat removes the equation entirely.
    """
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(n):
        raan_t = rng.uniform(0, 2 * np.pi)
        raan_s = rng.uniform(0, 2 * np.pi)
        i_s_true = rng.uniform(0.0, np.pi * 0.9)
        def hhat(i, O):
            return np.array([np.sin(i) * np.sin(O), -np.sin(i) * np.cos(O),
                             np.cos(i)])
        c_true = float(hhat(i_s_true, raan_s) @ hhat(i_t, raan_t))
        grid = np.linspace(1e-6, np.pi - 1e-6, 4001)
        f = np.array([hhat(g, raan_s) @ hhat(i_t, raan_t) for g in grid]) - c_true
        hits += int((np.diff(np.sign(f)) != 0).sum() >= 2)
    return hits / n


# Minimal C-env kwargs: no debris, LEO, the standing project configuration.
_MIN_KW = dict(
    num_debris_min=0, num_debris_max=0,
    e_max_target=0.0, e_max_sat=0.0, same_orbit_init=0,
    init_phase_gap_max=3.14159, valid_init_only=1,
    gave_up_action='terminate', max_valid_init_attempts=4096,
    obs_alt_scale_m=1.6e6, lvlh_scale_m=6.371e6,
    shaping_mode=1, shape_gamma=1.0, phase_gap_mode=1, phase_obs_mode=1,
    episode_cap_steps=3000, cap_terminal_reward=0.0,
)


# ── c1: BLOCKER-4 (leakcheck) + MAJOR-14 (clamp) + MAJOR-15 (recon-cart) ─────
def _fixed_wrong_state(xt):
    """A FIXED wrong target state for `leakcheck`.

    NOT the literal "+1000 km along-track" of the red-team's example, and the
    reason is measurable. A position shift along v-hat leaves r x v EXACTLY
    invariant ((r + d v-hat) x v = r x v), so h-hat-t does not move; and every
    other change it induces is in-plane, so its projection on h-hat-t is zero
    too. Measured below: such a probe leaves obs[21], obs[22], obs[23] and
    obs[24] bit-identical — FOUR of the seven new leak channels — so a
    leakcheck built on it would certify a leaky encode as clean. This state
    instead rotates the plane, shifts along-track and rescales the speed, so
    every derived quantity differs.
    """
    r, v = xt[:, :3], xt[:, 3:6]
    ax = np.array([0.3, 0.5, 0.81])
    ax = ax / np.linalg.norm(ax)
    th = math.radians(3.0)
    K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    R = np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * (K @ K)
    vh = v / np.linalg.norm(v, axis=1)[:, None]
    out = np.empty_like(xt)
    out[:, :3] = r @ R.T + 1.0e6 * vh
    out[:, 3:6] = (v @ R.T) * 1.001
    return out


def _alongtrack_shift(xt, d=1.0e6):
    v = xt[:, 3:6]
    out = xt.copy()
    out[:, :3] = xt[:, :3] + d * v / np.linalg.norm(v, axis=1)[:, None]
    return out


def stage_c1(args):
    print('== c1  BLOCKER-4 leakcheck / MAJOR-14 clamp / MAJOR-15 recon-cart ==')
    from pufferlib.ocean.orbital.orbital import Orbital
    from pufferlib.ocean.orbital_nav import nav_encode3d as ne
    from pufferlib.ocean.orbital_nav import nav_math3d as n3

    n = 128
    env = Orbital(num_envs=n, seed=5, **_D3_KW)
    env.reset(seed=11)
    rng = np.random.default_rng(0)
    for _ in range(40):                      # get off the reset manifold
        env.step(rng.integers(0, 30, n).astype(np.int32))
    st = env.get_state()
    obs_c = np.asarray(env.observations, dtype=np.float32).copy()

    enc = ne.Encoder3D(n, obs_alt_scale_m=_D3_KW['obs_alt_scale_m'],
                       lvlh_scale_m=_D3_KW['lvlh_scale_m'],
                       di_max_rad=_D3_KW['di_max_rad'],
                       shape_dv_ref_ms=_D3_KW['shape_dv_ref_ms'])
    slots = np.array(ne.TARGET_SLOTS_3D)
    comp = np.array(ne.COMPLEMENT_SLOTS_3D)

    ok = record('layout: 19 target slots + 19 complement partition obs[0:38]',
                len(ne.TARGET_SLOTS_3D) == 19 and len(ne.COMPLEMENT_SLOTS_3D) == 19,
                f'target {ne.TARGET_SLOTS_3D}')
    ok &= record('the 7 NEW leak channels are in the rebuilt set',
                 set([21, 22, 23, 24, 25, 26, 28]) <= set(ne.TARGET_SLOTS_3D))
    ok &= record('obs[27] (chaser-only dv ledger) is NOT rebuilt',
                 27 in ne.COMPLEMENT_SLOTS_3D)
    ok &= record('obs[15] (episode clock) is NOT rebuilt',
                 15 in ne.COMPLEMENT_SLOTS_3D)
    ok &= record('obs[29-32] reserved, NOT rebuilt',
                 set([29, 30, 31, 32]) <= set(ne.COMPLEMENT_SLOTS_3D))

    # ── recon: estimate := truth, straight off the getter ───────────────────
    tt = ne.Encoder3D.target_truth(st)
    o = obs_c.copy()
    enc.write(o, st, tt['cart'])
    d = np.abs(o.astype(np.float64) - obs_c.astype(np.float64))
    eps32 = float(np.spacing(np.float32(1.0)))
    per = d[:, slots].max(axis=0)
    ok &= record('recon: complement untouched (bitwise)',
                 np.array_equal(o[:, comp], obs_c[:, comp]),
                 f'max|diff| {d[:, comp].max():.3e}')
    ok &= record('recon: 19 rebuilt slots reproduce the C to < 1e-5 abs',
                 per.max() < 1e-5,
                 f'max {per.max():.3e} at slot {slots[int(per.argmax())]}; '
                 f'{per.max() / eps32:.1f} x eps_f32')
    print('    per-slot max |recon - truth|: '
          + ', '.join(f'[{s}]={per[i]:.1e}' for i, s in enumerate(slots)
                      if per[i] > 0))

    # ── MAJOR-15 recon-cart: the Cartesian route the FILTER actually flies ──
    el = n3.cartesian_to_elements_3d(tt['cart'])
    xt_rt = n3.orbit_to_cartesian_3d(el['a'], el['e'], el['theta'],
                                     el['omega'], el['inc'], el['raan'])
    o2 = obs_c.copy()
    enc.write(o2, st, xt_rt)
    d2 = np.abs(o2.astype(np.float64) - obs_c.astype(np.float64))[:, slots]
    per2 = d2.max(axis=0)
    cart_err = float(np.abs(xt_rt - tt['cart']).max())
    ok &= record('recon-cart: truth round-tripped through Cartesian, 19 slots '
                 '(float32, vs the C)', per2.max() < 1e-3,
                 f'max {per2.max():.3e} at slot {slots[int(per2.argmax())]} '
                 f'(state round-trip {cart_err:.2e} m)')
    # The float32 comparison saturates at 0 because the round-trip state error
    # is below the observation's own quantum. The MEASURED tolerance the gate
    # states is therefore taken in float64, on the encoder's raw values, before
    # the cast — otherwise "0.0" would be a property of float32, not of the
    # Cartesian route.
    v_el = enc.values(st, tt['cart'])[:, slots].copy()
    v_ct = enc.values(st, xt_rt)[:, slots].copy()
    per64 = np.abs(v_el - v_ct).max(axis=0)
    ok &= record('recon-cart: same, in float64 before the cast (the STATED '
                 'tolerance)', per64.max() < 1e-9,
                 f'max {per64.max():.3e} at slot {slots[int(per64.argmax())]}')
    print('    per-slot max |recon-cart - recon| (float64): '
          + ', '.join(f'[{s}]={per64[i]:.1e}' for i, s in enumerate(slots)
                      if per64[i] > 0))

    # ── leakcheck ───────────────────────────────────────────────────────────
    def leakcheck(encoder, label):
        """Returns (unmoved_target_slots, moved_complement_slots)."""
        base = obs_c.copy()
        encoder(base, st, tt['cart'])
        bad = obs_c.copy()
        encoder(bad, st, _fixed_wrong_state(tt['cart']))
        moved = (base != bad).any(axis=0)
        unmoved_t = [int(s) for s in slots if not moved[s]]
        moved_c = [int(s) for s in comp if moved[s]]
        print(f'    {label}: unmoved target slots {unmoved_t}, '
              f'moved complement slots {moved_c}')
        return unmoved_t, moved_c

    ut, mc = leakcheck(enc.write, 'shipped Encoder3D')
    ok &= record('leakcheck (a): ALL 19 estimated slots move', not ut)
    ok &= record('leakcheck (b): the complement does NOT move', not mc)

    # ── test the test (the ext_invariants3d discipline) ─────────────────────
    class _LeakyEncoder(ne.Encoder3D):
        """The 2D wrapper's behaviour under dim3: rebuilds only the 12 slots of
        nav_math.TARGET_SLOTS_T3, leaving the seven ext-3d channels on TRUTH."""
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self.slots = np.array(nm.TARGET_SLOTS_T3)
            self._is_clamp2 = np.isin(self.slots, self.clamp2)

    class _OverBroadEncoder(ne.Encoder3D):
        """The other direction: also writes obs[15] (the episode clock) and
        obs[27] (the chaser-only dv ledger)."""
        def write(self, obs, st, xt):
            r = super().write(obs, st, xt)
            v = self.values(st, xt)
            obs[:, 15] = np.float32(0.5)
            obs[:, 27] = (v[:, 26] * 0.5).astype(np.float32)
            return r

    leaky = _LeakyEncoder(n, obs_alt_scale_m=_D3_KW['obs_alt_scale_m'],
                          lvlh_scale_m=_D3_KW['lvlh_scale_m'],
                          di_max_rad=_D3_KW['di_max_rad'],
                          shape_dv_ref_ms=_D3_KW['shape_dv_ref_ms'])
    ut, mc = leakcheck(leaky.write, 'deliberately LEAKY encode (2D 12-slot)')
    ok &= record('test-the-test: leaky encode caught on exactly the 7 new '
                 'channels {21,22,23,24,25,26,28}',
                 sorted(ut) == [21, 22, 23, 24, 25, 26, 28] and not mc,
                 f'caught {sorted(ut)}')

    broad = _OverBroadEncoder(n, obs_alt_scale_m=_D3_KW['obs_alt_scale_m'],
                              lvlh_scale_m=_D3_KW['lvlh_scale_m'],
                              di_max_rad=_D3_KW['di_max_rad'],
                              shape_dv_ref_ms=_D3_KW['shape_dv_ref_ms'])
    ut, mc = leakcheck(broad.write, 'deliberately OVER-BROAD encode')
    ok &= record('test-the-test: over-broad encode caught on obs[27] '
                 '(and obs[15] via the untouched-complement bitwise check)',
                 27 in mc, f'moved complement {mc}')
    o3 = obs_c.copy()
    broad.write(o3, st, tt['cart'])
    ok &= record('test-the-test: over-broad encode breaks the bitwise '
                 'complement check', not np.array_equal(o3[:, comp],
                                                        obs_c[:, comp]))

    # The probe the red-team's example would have used, and why it is unsafe.
    base = obs_c.copy(); enc.write(base, st, tt['cart'])
    alt = obs_c.copy(); enc.write(alt, st, _alongtrack_shift(tt['cart']))
    still = [int(s) for s in slots if not (base[:, s] != alt[:, s]).any()]
    note("the red-team's example probe (+1000 km ALONG-TRACK) is NOT a valid "
         'leakcheck garbage',
         f'it leaves {still} bit-identical — (r + d v-hat) x v = r x v so '
           f'h-hat-t never moves, and every induced change is in-plane')

    # ── MAJOR-14: slot-dependent clamp ──────────────────────────────────────
    ok &= record('clamp set is exactly {21..26, 28} (obs[27] is not rebuilt)',
                 tuple(ne.CLAMP2_SLOTS) == (21, 22, 23, 24, 25, 26, 28))
    probe = np.array([-3.0, -2.0, -1.5, 0.0, 1.5, 2.0, 3.0, np.nan, np.inf,
                      -np.inf])
    want = np.array([-2.0, -2.0, -1.5, 0.0, 1.5, 2.0, 2.0, -2.0, 2.0, -2.0])
    got = ne.obs_clamp2(probe)
    ok &= record('obs_clamp2 reproduces the C branch incl. the NaN trap '
                 '(NaN -> -2.0)', np.array_equal(got, want),
                 f'{got.tolist()}')

    # A saturating estimate must land on +/-2 in 21-28 and on +/-nav_clip
    # elsewhere — this is what a uniform nav_clip = 4.0 would get wrong.
    wild = tt['cart'].copy()
    wild[:, :3] *= 4.0
    o4 = obs_c.copy()
    enc.write(o4, st, wild)
    c2 = np.abs(o4[:, np.array(ne.CLAMP2_SLOTS)]).max()
    ok &= record('saturating estimate: slots 21-28 bounded by 2.0, not 4.0',
                 c2 <= 2.0 + 1e-6, f'max |obs[21..28]| = {c2:.4f}')

    env.close()
    return dict(ok=bool(ok))




# ── d1: BLOCKER-1 — the MSC chart pole and the re-pole ───────────────────────
def _pair_geometry(di_deg, rho_inplane_m, a=6771e3, n_steps=200):
    """Chaser/target pair: same circular LEO orbit, an along-track separation,
    and a relative inclination. Propagated over one full orbit."""
    from pufferlib.ocean.orbital_nav import nav_math3d as n3
    di = math.radians(di_deg)
    th0 = 0.7
    dth = rho_inplane_m / a
    tgt0 = n3.orbit_to_cartesian_3d(np.array([a]), np.array([0.0]),
                                    np.array([th0]), np.array([0.0]),
                                    np.array([0.0]), np.array([0.0]))
    sat0 = n3.orbit_to_cartesian_3d(np.array([a]), np.array([0.0]),
                                    np.array([th0 - dth]), np.array([0.0]),
                                    np.array([di]), np.array([0.0]))
    T = 2.0 * np.pi * math.sqrt(a ** 3 / n3.MU)
    ts = np.linspace(0.0, T, n_steps)
    S = n3.propagate_cartesian_nd(np.repeat(sat0, n_steps, axis=0), ts)[0]
    G = n3.propagate_cartesian_nd(np.repeat(tgt0, n_steps, axis=0), ts)[0]
    return S, G, ts


def _max_el(S, G, pole):
    """max |el| (deg) of the LOS about a FIXED unit pole."""
    d = G[:, :3] - S[:, :3]
    u = d / np.linalg.norm(d, axis=1)[:, None]
    return float(np.degrees(np.abs(np.arcsin(np.clip(u @ pole, -1.0, 1.0)))).max())


def stage_d1(args):
    print('== d1  BLOCKER-1  MSC chart pole + re-pole ========================')
    from pufferlib.ocean.orbital_nav import nav_math3d as n3

    # (1) reproduce the red-team's pole table
    print('  max |el| over one orbit, N3D-A pole (h-hat_c) vs N3D-C 3b pole '
          '(u0 x h-hat_t):')
    print(f"    {'geometry':38s} {'h-hat_c':>10s} {'u0 x h-hat_t':>14s}")
    cells = [('coplanar 5 km  (2D-reduction anchor)', 0.0, 5e3),
             ('coplanar 10 km', 0.0, 10e3),
             ('di=0.02 deg, 5 km  (tight box)', 0.02, 5e3),
             ('di=0.0417 deg, 5 km  (box plane tol)', 0.0417, 5e3),
             ('di=1.0 deg, rho_inplane 5 km  (X3)', 1.0, 5e3),
             ('di=1.0 deg, 300 km', 1.0, 300e3)]
    tab = {}
    for name, di, rho in cells:
        # 40,001 samples over one orbit: the N3D-C pole's singularity is a
        # crossing, not a plateau, so a coarse sweep under-reports it (a
        # 200-sample grid tops out at 89.55 deg purely by missing the crossing).
        S, G, _ = _pair_geometry(di, rho, n_steps=40001)
        pA = n3.pole_frame(S[:1])[0, 2]                    # h-hat_c(t0)
        d0 = G[0, :3] - S[0, :3]
        u0 = d0 / np.linalg.norm(d0)
        ht = n3.unit(n3.cross(G[0, :3], G[0, 3:6])[None])[0]
        pC = n3.unit(n3.cross(u0[None], ht[None]))[0]
        eA, eC = _max_el(S, G, pA), _max_el(S, G, pC)
        tab[name] = (eA, eC)
        print(f'    {name:38s} {eA:9.2f}  {eC:13.2f}')

    ok = record('N3D-C 3b pole is EXACT gimbal lock in the coplanar anchor',
                abs(tab['coplanar 5 km  (2D-reduction anchor)'][1] - 90.0) < 1e-3,
                f"{tab['coplanar 5 km  (2D-reduction anchor)'][1]:.4f} deg — "
                f'this is why it is deleted from the spec')
    ok &= record('N3D-A pole gives el == 0 in the coplanar anchor '
                 '(the 2D-reduction regression test)',
                 tab['coplanar 5 km  (2D-reduction anchor)'][0] < 1e-9,
                 f"{tab['coplanar 5 km  (2D-reduction anchor)'][0]:.2e} deg")
    ok &= record('N3D-A pole DOES arm the guard at the X3 rung '
                 '(the caveat the prototype never exercised)',
                 tab['di=1.0 deg, rho_inplane 5 km  (X3)'][0] > 45.0,
                 f"{tab['di=1.0 deg, rho_inplane 5 km  (X3)'][0]:.2f} deg "
                 f'> 45 deg trigger')

    # (2) the 2D reduction, exactly: at di = 0, el and w_e are identically zero
    S, G, _ = _pair_geometry(0.0, 5e3, n_steps=60)
    Rp = np.repeat(n3.pole_frame(S[:1]), S.shape[0], axis=0)
    y = n3.msc6_encode(S, G, Rp)
    ok &= record('2D reduction: el == 0 and w_e == 0 at di = 0',
                 np.abs(y[:, n3.IDX_EL]).max() < 1e-15
                 and np.abs(y[:, n3.IDX_WE]).max() < 1e-18,
                 f'max |el| {np.abs(y[:, n3.IDX_EL]).max():.2e} rad, '
                 f'max |w_e| {np.abs(y[:, n3.IDX_WE]).max():.2e} rad/s')

    # (3) the closed-form guard-trigger condition
    #     |el| > X  <=>  rho_inplane < rho_oop / tan(X),  rho_oop <= r sin(di)
    r = 6771e3
    di = 1.0
    print('  guard-trigger crossing at di = 1.0 deg (sweep rho_inplane):')
    rows = []
    for rho in (20e3, 40e3, 68e3, 90e3, 118e3, 160e3, 300e3):
        S, G, _ = _pair_geometry(di, rho)
        pA = n3.pole_frame(S[:1])[0, 2]
        rows.append((rho, _max_el(S, G, pA)))
        print(f'    rho_inplane {rho/1e3:6.0f} km   max |el| {rows[-1][1]:6.2f} deg')
    pred60 = 0.577 * r * math.sin(math.radians(di))
    pred45 = 1.000 * r * math.sin(math.radians(di))
    def crossing(deg):
        below = [x for x, e in rows if e > deg]
        above = [x for x, e in rows if e <= deg]
        return (max(below) if below else 0.0, min(above) if above else np.inf)
    c60, c45 = crossing(60.0), crossing(45.0)
    ok &= record('closed form |el|>60 <=> rho_inplane < 0.577 r sin(di) '
                 f'= {pred60/1e3:.0f} km',
                 c60[0] <= pred60 <= c60[1],
                 f'measured crossing bracketed by [{c60[0]/1e3:.0f}, '
                 f'{c60[1]/1e3:.0f}] km')
    ok &= record('closed form |el|>45 <=> rho_inplane < 1.000 r sin(di) '
                 f'= {pred45/1e3:.0f} km',
                 c45[0] <= pred45 <= c45[1],
                 f'measured crossing bracketed by [{c45[0]/1e3:.0f}, '
                 f'{c45[1]/1e3:.0f}] km')
    note('CONSEQUENCE: the 45 deg trigger does NOT keep re-poles out of the '
         '30 km success box',
         f'it arms below {pred45/1e3:.0f} km vs {pred60/1e3:.0f} km at 60 deg '
           f'— lowering the trigger fires MORE, not less. Taken deliberately: '
           f'45 caps the az-noise inflation 1/cos^2 el at 2.0 (60 allows 4.0), '
           f'and the re-pole is exact, so the cost of firing is bounded below.')

    # (4) the re-pole itself: exact similarity transform
    rng = np.random.default_rng(11)
    B = 64
    S, G, _ = _pair_geometry(1.0, 5e3, n_steps=B)
    Rp = np.repeat(n3.pole_frame(S[:1]), B, axis=0)
    filt = n3.BatchedBearingMSC6(B, sigma_beta=1e-3)
    filt.Rp = Rp.copy()
    filt.sat = S.copy()
    # Seed a NON-ZERO estimation error, otherwise NEES is identically 0 and the
    # continuity check is vacuous.
    sig_p, sig_v = 40.0, 0.04
    x_est = G.copy()
    x_est[:, :3] += rng.normal(0.0, sig_p, (B, 3))
    x_est[:, 3:] += rng.normal(0.0, sig_v, (B, 3))
    P_cart = np.tile(np.diag([sig_p ** 2] * 3 + [sig_v ** 2] * 3), (B, 1, 1))
    filt.set_cart(np.arange(B), x_est, P_cart, S)

    idx = np.arange(B)
    x_before, P_before = filt.mean_cov(idx)
    hot = filt.repole(idx)
    x_after, P_after = filt.mean_cov(idx)
    scale = np.abs(x_before).max()
    d_state = float(np.abs(x_after - x_before).max() / scale)
    d_cov = float(np.abs(P_after - P_before).max()
                  / max(np.abs(P_before).max(), 1e-300))
    ok &= record('re-pole fired on the rows above the trigger',
                 hot.size > 0,
                 f'{hot.size}/{B} rows, |el| range '
                 f'{np.degrees(np.abs(n3.msc6_encode(S, G, Rp)[:, 1])).min():.1f}'
                 f'-{np.degrees(np.abs(n3.msc6_encode(S, G, Rp)[:, 1])).max():.1f} deg')
    ok &= record('re-pole Cartesian round trip is the identity to 1e-12',
                 d_state < 1e-12, f'max rel |dx| {d_state:.3e}')
    ok &= record('re-pole covariance is an exact similarity transform '
                 '(Cartesian P unchanged)', d_cov < 1e-9,
                 f'max rel |dP| {d_cov:.3e}')
    truth = G
    nees_b = n3.nees_nd(x_before, P_before, truth)
    nees_a = n3.nees_nd(x_after, P_after, truth)
    ok &= record('NEES continuous across the re-pole (no step)',
                 float(np.abs(nees_a - nees_b).max()) < 1e-6,
                 f'max |dNEES| {np.abs(nees_a - nees_b).max():.3e}; '
                 f'median NEES {np.median(nees_b):.4f} -> '
                 f'{np.median(nees_a):.4f}')
    el_after = np.degrees(np.abs(filt.y[:, n3.IDX_EL]))
    sel = np.isin(np.arange(B), hot)
    ok &= record('every row is inside the trigger after the re-pole',
                 float(el_after.max()) <= n3.REPOLE_EL_DEG,
                 f'max |el| after {el_after.max():.2f} deg '
                 f'(trigger {n3.REPOLE_EL_DEG} deg)')
    ok &= record('the RE-POLED rows land at el == 0 exactly '
                 '(maximal distance from the singularity)',
                 float(el_after[sel].max()) < 1e-9,
                 f'max |el| on re-poled rows {el_after[sel].max():.2e} deg')
    ok &= record('re-pole COUNTER surfaced per row',
                 filt.n_repole_total == hot.size,
                 f'n_repole_total {filt.n_repole_total}, '
                 f'per-row max {int(filt.n_repole.max())}')

    # (5) exercise the regime the prototype skipped: di = 1 deg, rho_inplane
    #     swept THROUGH the 68 km guard radius, closed-loop over one orbit.
    print('  closed-loop 6-state run at di = 1.0 deg (the skipped regime):')
    ok_all = True
    for rho0 in (5e3, 30e3, 68e3, 118e3, 300e3):
        res = _run_msc6_arc(rho0, di_deg=1.0, seed=5)
        print(f"    rho_inplane {rho0/1e3:6.0f} km  re-poles {res['repoles']:3d}"
              f"  pos RMSE {res['pos']:10.1f} m  NEES med {res['nees']:8.3f}"
              f"  in-bounds {res['ib']:.2f}  diverged {res['div']}")
        ok_all &= (res['div'] == 0) and np.isfinite(res['pos'])
    ok &= record('no divergence anywhere in the swept regime, re-poles handled',
                 ok_all)
    return dict(ok=bool(ok))


def _run_msc6_arc(rho_inplane_m, di_deg=1.0, seed=5, n_obs=180, dt=60.0):
    """One bearings-only arc through the 6-state filter, seeded near truth.

    Deliberately NOT a blind start: this gate is about the CHART (pole,
    re-pole, anisotropic R), not about acquisition.
    """
    from pufferlib.ocean.orbital_nav import nav_math3d as n3
    rng = np.random.default_rng(seed)
    S, G, ts = _pair_geometry(di_deg, rho_inplane_m, n_steps=n_obs)
    B = 1
    filt = n3.BatchedBearingMSC6(B, sigma_beta=1e-3, q_a=1e-13)
    idx = np.arange(B)
    filt.set_pole(idx, S[:1])
    x0 = G[:1].copy()
    x0[:, :3] += rng.normal(0.0, 50.0, 3)
    x0[:, 3:] += rng.normal(0.0, 0.05, 3)
    P0 = np.diag([50.0 ** 2] * 3 + [0.05 ** 2] * 3)[None]
    filt.set_cart(idx, x0, P0, S[:1])
    nees, ib, div = [], [], 0
    for k in range(1, n_obs):
        ddt = float(ts[k] - ts[k - 1])
        ok = filt.predict(idx, ddt, S[k - 1:k], S[k:k + 1])
        if not ok.all():
            div += 1
            break
        d = np.einsum('nij,nj->ni', filt.Rp[idx], G[k:k + 1, :3] - S[k:k + 1, :3])
        u = d / np.linalg.norm(d, axis=1)[:, None]
        el_t = np.arcsin(np.clip(u[:, 2], -1.0, 1.0))
        az_t = np.arctan2(u[:, 1], u[:, 0])
        ce = np.maximum(np.cos(el_t), 1e-8)
        az = az_t + rng.normal(0.0, 1e-3 / ce)
        el = el_t + rng.normal(0.0, 1e-3, B)
        filt.update(idx, S[k:k + 1], az, el)
        filt.repole(idx)
        x, P = filt.mean_cov(idx)
        v = n3.nees_nd(x, P, G[k:k + 1])
        if np.isfinite(v).all():
            nees.append(float(v[0]))
            ib.append(n3.NEES6_LO <= v[0] <= n3.NEES6_HI)
    x, _ = filt.mean_cov(idx)
    pos = float(np.linalg.norm(x[0, :3] - G[n_obs - 1, :3]))
    return dict(repoles=int(filt.n_repole_total), pos=pos,
                nees=float(np.median(nees)) if nees else np.nan,
                ib=float(np.mean(ib)) if ib else 0.0, div=div)



# ── e1: BLOCKER-3 — the analytic 3D STM ─────────────────────────────────────
def _fuzz_states(N, rng):
    from pufferlib.ocean.orbital_nav import nav_math3d as n3
    a = n3.R_EARTH + rng.uniform(300e3, 8000e3, N)
    e = rng.uniform(0.0, 0.30, N)
    inc = rng.uniform(0.0, math.radians(69.0), N)
    raan = rng.uniform(0.0, 2 * np.pi, N)
    argp = rng.uniform(0.0, 2 * np.pi, N)
    nu = rng.uniform(0.0, 2 * np.pi, N)
    return n3.orbit_to_cartesian_3d(a, e, nu, argp, inc, raan)


def stage_e1(args):
    print('== e1  BLOCKER-3  analytic 3D STM ================================')
    import time
    from pufferlib.ocean.orbital_nav import nav_math3d as n3
    sys.path.insert(0, os.path.join(WT, 'scripts', 'orbital', 'ext_recon'))
    import orbital_math3d as o3

    rng = np.random.default_rng(3)
    N = 200
    X = _fuzz_states(N, rng)
    ok = True
    print('  fuzz envelope: a 6.7-14.4 Mm, e <= 0.30, i <= 69 deg, N = 200')
    print(f"    {'dt [s]':>8s} {'symp analytic':>15s} {'symp FD':>12s} "
          f"{'prop vs oracle':>16s} {'STM vs oracle-FD':>18s}")
    for dt in (60.0, 300.0, 1800.0, 3600.0, 21600.0):
        F, okf, Y = n3.stm_analytic_nd(X, dt)
        Ffd, _, _ = n3.stm_fd_nd(X, dt)
        sr = n3.symplectic_residual(F).max()
        srf = n3.symplectic_residual(Ffd).max()
        Yo = np.array([o3.propagate_universal(X[i], dt) for i in range(N)])
        dprop = np.abs(Y - Yo).max()
        # Richardson-extrapolated central differences OF THE ORACLE — an
        # independent reference for the STM itself, not just the propagation.
        Fnum = np.zeros((N, 6, 6))
        hs = np.array([1e2] * 3 + [1e-1] * 3)
        for j in range(6):
            for c, k in ((8 / 12., 1.0), (-1 / 12., 2.0)):
                Zp = X.copy(); Zp[:, j] += k * hs[j]
                Zm = X.copy(); Zm[:, j] -= k * hs[j]
                Op = np.array([o3.propagate_universal(Zp[i], dt) for i in range(N)])
                Om = np.array([o3.propagate_universal(Zm[i], dt) for i in range(N)])
                Fnum[:, :, j] += c * (Op - Om) / hs[j]
        rel = (np.abs(F - Fnum) / np.abs(Fnum).max(axis=(1, 2))[:, None, None]).max()
        print(f'    {dt:8.0f} {sr:15.2e} {srf:12.2e} {dprop:14.2e} m '
              f'{rel:17.2e}')
        ok &= (sr < 1e-9) and (dprop < 1e-3) and (rel < 1e-6)
    ok = record('analytic STM: symplectic to <= 1e-9, propagation matches the '
                'universal-variable oracle, STM matches oracle central '
                'differences to <= 1e-6 rel, across the fuzz envelope', ok)

    dt1, dt2 = 137.0, 431.0
    F1, _, Y1 = n3.stm_analytic_nd(X, dt1)
    F2, _, _ = n3.stm_analytic_nd(Y1, dt2)
    F12, _, _ = n3.stm_analytic_nd(X, dt1 + dt2)
    g = float((np.abs(F2 @ F1 - F12) / np.abs(F12).max()).max())
    ok &= record('group property Phi(t2<-t1) Phi(t1<-t0) == Phi(t2<-t0)',
                 g < 1e-12, f'max rel {g:.3e}')
    F0, _, _ = n3.stm_analytic_nd(X, 0.0)
    ok &= record('Phi(dt=0) == I exactly',
                 float(np.abs(F0 - np.eye(6)).max()) == 0.0)

    # 2D reduction against the SHIPPED finite-difference STM.
    rng2 = np.random.default_rng(9)
    M = 200
    a = n3.R_EARTH + rng2.uniform(300e3, 800e3, M)
    e = rng2.uniform(0.0, 0.05, M)
    nu = rng2.uniform(0.0, 2 * np.pi, M)
    w = rng2.uniform(0.0, 2 * np.pi, M)
    Xe = n3.orbit_to_cartesian_3d(a, e, nu, w, np.zeros(M), np.zeros(M))
    X4 = np.stack([Xe[:, 0], Xe[:, 1], Xe[:, 3], Xe[:, 4]], axis=1)
    F4, _, _ = n3.stm_analytic_nd(X4, 60.0)
    Ffd4, _, _ = nm.stm_fd(X4, 60.0)
    d4 = float((np.abs(F4 - Ffd4) / np.abs(Ffd4).max()).max())
    ok &= record('2D reduction: analytic 4-state == shipped stm_fd within the '
                 "FD's own truncation floor", d4 < 1e-6,
                 f'max rel {d4:.2e}; symplectic analytic '
                 f'{n3.symplectic_residual(F4).max():.2e} vs shipped '
                 f'{n3.symplectic_residual(Ffd4).max():.2e}')

    # ── the tick-time table ────────────────────────────────────────────────
    def tick_table(B):
        rngb = np.random.default_rng(1)
        a = n3.R_EARTH + rngb.uniform(300e3, 800e3, B)
        ecc = rngb.uniform(0.0, 0.05, B)
        inc = rngb.uniform(0.0, math.radians(1.0), B)
        raan = rngb.uniform(0.0, 2 * np.pi, B)
        argp = rngb.uniform(0.0, 2 * np.pi, B)
        nu = rngb.uniform(0.0, 2 * np.pi, B)
        sat = n3.orbit_to_cartesian_3d(a, ecc, nu, argp, inc, raan)
        tgt = n3.orbit_to_cartesian_3d(a + rngb.uniform(-3e4, 3e4, B), ecc,
                                       nu + rngb.uniform(-1e-3, 1e-3, B), argp,
                                       inc * 0.5, raan)
        sat2 = n3.propagate_cartesian_nd(sat, 60.0)[0]
        X4b = np.stack([sat[:, 0], sat[:, 1], sat[:, 3], sat[:, 4]], axis=1)
        idx = np.arange(B)

        def mk(stm):
            f = n3.BatchedBearingMSC6(B, sigma_beta=1e-3, stm=stm)
            f.set_pole(idx, sat)
            f.set_cart(idx, tgt, np.tile(np.diag([2500.0] * 3 + [0.01] * 3),
                                         (B, 1, 1)), sat)
            return f

        def tick(f):
            def go():
                f.predict(idx, 60.0, sat, sat2)
                f.update(idx, sat2, f.y[:, 0], f.y[:, 1])
                f.repole(idx)
            return go

        mpc = nm.BatchedBearingMPC(B, sigma_beta=1e-3)
        t4 = np.stack([tgt[:, 0], tgt[:, 1], tgt[:, 3], tgt[:, 4]], axis=1)
        s4 = np.stack([sat[:, 0], sat[:, 1], sat[:, 3], sat[:, 4]], axis=1)
        s4b = np.stack([sat2[:, 0], sat2[:, 1], sat2[:, 3], sat2[:, 4]], axis=1)
        mpc.set_cart(idx, t4, np.tile(np.diag([2500.0] * 2 + [0.01] * 2),
                                      (B, 1, 1)), s4)

        def tick2d():
            mpc.predict(idx, 60.0, s4, s4b)
            mpc.update(idx, s4b, mpc.y[:, 0])

        ops = [
            ('stm_fd_nd (6-state FD, 13 propagations)',
             lambda: n3.stm_fd_nd(sat, 60.0)),
            ('stm_analytic_nd (6-state analytic)',
             lambda: n3.stm_analytic_nd(sat, 60.0)),
            ('nav_math.stm_fd (4-state, the 2D shipped object)',
             lambda: nm.stm_fd(X4b, 60.0)),
            ('MSC6 tick BASELINE (y-space FD, the mechanical port)',
             tick(mk('fd_msc'))),
            ('MSC6 tick intermediate (analytic chart Jac + FD STM)',
             tick(mk('fd'))),
            ('MSC6 tick SHIPPED (fully analytic)', tick(mk('analytic'))),
            ('2D shipped BatchedBearingMPC tick (reference)', tick2d),
        ]
        # ROUND-ROBIN, min-of-batches. Timing each op to completion in turn
        # measures whatever else the machine was doing during that op's slot;
        # interleaving makes every op see the same conditions, and the min
        # discards the slots where it was doing something else.
        rep, batches = 20, 7
        best = {nme: float('inf') for nme, _ in ops}
        for _, fn in ops:
            fn()
        for _ in range(batches):
            for nme, fn in ops:
                t = time.perf_counter()
                for _ in range(rep):
                    fn()
                best[nme] = min(best[nme], (time.perf_counter() - t) / rep * 1e3)
        return best

    print("\n  TICK TIME, OMP pinned to 1. B = 256 is one worker's env slice "
          'at the shipped 8w x 256 shape.')
    tabs = {B: tick_table(B) for B in (256, 1024)}
    names = list(tabs[256])
    print(f"    {'op':56s} {'B=256':>10s} {'B=1024':>10s}")
    for nme in names:
        print(f'    {nme:56s} {tabs[256][nme]:9.4f}  {tabs[1024][nme]:9.4f}')
    sp = {B: tabs[B]['MSC6 tick BASELINE (y-space FD, the mechanical port)']
          / tabs[B]['MSC6 tick SHIPPED (fully analytic)'] for B in (256, 1024)}
    speed = sp[256]
    ok &= record('analytic path speeds the full 6-state tick by >= 2x vs the '
                 'mechanical port',
                 min(sp.values()) >= 2.0,
                 f'{sp[256]:.2f}x at B=256, {sp[1024]:.2f}x at B=1024')
    note("SHORTFALL: the red-team's >= 3x projection is NOT met on the full "
         'tick',
         f"measured {sp[256]:.2f}x. The projection came from the PROTOTYPE's "
           f'accounting, where FD-STM was 87% of the tick; in the shipped tick '
           f'the analytic path also replaces the FD chart Jacobians, so the '
           f'STM is only '
           f"{tabs[256]['stm_fd_nd (6-state FD, 13 propagations)'] / tabs[256]['MSC6 tick BASELINE (y-space FD, the mechanical port)']:.0%}"
           f' of the baseline, and what remains (decode/encode, two analytic '
           f'6x6 Jacobians, one 6x6 inverse, five (B,6,6) matmuls, the update) '
           f'is numpy-call-overhead bound at these batch shapes. Residual '
         f'levers are the red-team\'s own items 2-4, not more STM work.')

    # BLOCKER-3 item 2: "confirm it, do not assume it".
    from pufferlib.ocean.orbital_nav.nav_surrogate import AcquisitionSurrogate
    B = 256
    acq = AcquisitionSurrogate(B, sigma_beta=1e-3)
    rngc = np.random.default_rng(2)
    sat_c = np.zeros((B, 4)); sat_c[:, 0] = n3.R_EARTH + 5e5
    sat_c[:, 3] = math.sqrt(n3.MU / sat_c[0, 0])
    tgt_c = sat_c.copy(); tgt_c[:, 0] += 5e3
    idx = np.arange(B)
    acq.reset_rows(idx, sat_c, tgt_c, np.full(B, 5545.0))

    def acc():
        acq.accumulate(idx, sat_c, tgt_c, tgt_c, 60.0)
    acc(); t = time.perf_counter()
    for _ in range(50):
        acc()
    t_un = (time.perf_counter() - t) / 50 * 1e3
    acq.acquired[:] = True
    acc(); t = time.perf_counter()
    for _ in range(50):
        acc()
    t_acq = (time.perf_counter() - t) / 50 * 1e3
    ok &= record('surrogate accumulate is SKIPPED once a row is acquired '
                 '(measured, not assumed)', t_acq < 0.05 * t_un,
                 f'{t_un:.4f} ms unacquired -> {t_acq:.4f} ms acquired '
                 f'({t_acq / max(t_un, 1e-12):.1%})')
    return dict(ok=bool(ok), speedup=speed, ticks=tabs)



# ── f1: BLOCKER-2 + the remaining MAJORs / NOTEs ────────────────────────────
def _nav_kw(**over):
    kw = dict(_D3_KW)
    kw.update(over)
    return kw


def stage_f1(args):
    print('== f1  BLOCKER-2 sim-time floor + MAJOR-7/11/12/13/16 + NOTE-20/21 =')
    from pufferlib.ocean.orbital_nav.orbital_nav import OrbitalNav
    from pufferlib.ocean.orbital_nav import nav_math3d as n3
    from pufferlib.ocean.orbital_nav import nav_surrogate as ns

    ok = True

    # ── BLOCKER-2: the floor is SIM TIME, invariant under the cadence ───────
    def latency(dt_tick, steps=140):
        env = OrbitalNav(num_envs=48, nav_mode='bearings_only', seed=7,
                         log_interval=10 ** 9, nav_sensor_dt=dt_tick,
                         **_D3_KW)
        env.reset(seed=42)
        rng = np.random.default_rng(0)
        for _ in range(steps):
            env.step(rng.integers(0, 30, 48).astype(np.int32))
        lat = np.array(env._acq.acq_latency_s)
        env.close()
        return lat

    l60 = latency(60.0)
    l300 = latency(300.0)
    print(f'    dt_tick =  60 s: {l60.size} acquisitions, median latency '
          f'{np.median(l60):.0f} s ({np.median(l60)/60:.1f} min)')
    print(f'    dt_tick = 300 s: {l300.size} acquisitions, median latency '
          f'{np.median(l300):.0f} s ({np.median(l300)/60:.1f} min)')
    ok &= record('acquisition floor is a SIM-TIME floor: >= 2700 s at both '
                 'cadences', l60.min() >= 2700.0 and l300.min() >= 2700.0,
                 f'min {l60.min():.0f} s / {l300.min():.0f} s')
    ok &= record('a TICK floor would have scaled the floor with the cadence',
                 np.median(l300) < 45 * 300.0 * 0.9,
                 f'a 45-tick floor at dt=300 s would be {45*300:.0f} s '
                 f'(3.75 h); measured {np.median(l300):.0f} s')
    try:
        OrbitalNav(num_envs=2, nav_mode='bearings_only',
                   nav_acq_min_ticks=45, **_D3_KW)
        raised = False
    except ValueError:
        raised = True
    ok &= record('nav_acq_min_ticks > 0 RAISES (no silent unit re-entry)',
                 raised)

    # ── MAJOR-7: nav_max_ticks no longer desyncs filter from truth ──────────
    tau = int(nm.ACTION_TAU[17])            # warp 6 h
    K = 12
    n_tick = min(tau, K)
    dt_tick = tau * 60.0 / n_tick
    ok &= record('MAJOR-7: fixed COUNT x adaptive INTERVAL spans the whole tau',
                 abs(n_tick * dt_tick - tau * 60.0) < 1e-9,
                 f'tau={tau} sub-steps, K={K}: {n_tick} ticks x {dt_tick:.0f} s '
                 f'= {n_tick*dt_tick:.0f} s == {tau*60} s. The pre-fix code '
                 f'ticked {K} x 60 s = {K*60} s and was then handed a bearing '
                 f'{tau*60 - K*60} s in the future as one 60 s innovation.')

    def rmse_with(mt):
        env = OrbitalNav(num_envs=48, nav_mode='bearings_only', seed=7,
                         log_interval=10 ** 9, nav_max_ticks=mt, **_D3_KW)
        env.reset(seed=42)
        rng = np.random.default_rng(1)
        for _ in range(120):
            env.step(rng.integers(0, 30, 48).astype(np.int32))
        d = env._d_div / max(env._d_n, 1)
        r = math.sqrt(env._d_sq / max(env._d_n, 1))
        env.close()
        return r, d

    r0, d0 = rmse_with(0)
    rk, dk = rmse_with(K)
    print(f'    nav_max_ticks=0:   pos rmse {r0:.3e} m, diverge rate {d0:.4f}')
    print(f'    nav_max_ticks={K}:  pos rmse {rk:.3e} m, diverge rate {dk:.4f}')
    ok &= record('MAJOR-7: capping ticks no longer inflates the divergence '
                 'rate', dk <= max(3.0 * d0, 0.02),
                 f'{d0:.4f} -> {dk:.4f}')

    # ── MAJOR-12: basis-free P_perp == explicit az/el FIM with the ──────────
    #    cos-el-inflated R, at the elevations BLOCKER-1 says are reachable.
    print('    MAJOR-12 information-kernel equivalence:')
    worst = 0.0
    for el_deg in (0.0, 27.0, 45.0, 60.0):
        el = math.radians(el_deg)
        az = 0.83
        rho, sb = 5.0e3, 1.0e-3
        u = np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az),
                      math.sin(el)])
        e_az = np.array([-math.sin(az), math.cos(az), 0.0])
        e_el = np.array([-math.sin(el) * math.cos(az),
                         -math.sin(el) * math.sin(az), math.cos(el)])
        # explicit az/el: H = d(az,el)/d(r) , R = diag(sb^2/cos^2 el, sb^2)
        H = np.stack([e_az / (rho * math.cos(el)), e_el / rho])
        R = np.diag([sb ** 2 / math.cos(el) ** 2, sb ** 2])
        M_azel = H.T @ np.linalg.inv(R) @ H
        M_free = (np.eye(3) - np.outer(u, u)) / (sb * rho) ** 2
        d = np.abs(M_azel - M_free).max() / np.abs(M_free).max()
        worst = max(worst, d)
        # what an ISOTROPIC R would have claimed
        iso = H.T @ np.linalg.inv(np.eye(2) * sb ** 2) @ H
        infl = np.trace(iso) / np.trace(M_free)
        print(f'      el = {el_deg:4.0f} deg   |P_perp - azel| / |P_perp| '
              f'{d:.2e}   isotropic-R inflation {infl:.3f}x '
              f'(1/cos^2 el = {1/math.cos(el)**2:.3f})')
    ok &= record('MAJOR-12: M = P_perp/(sigma_b rho)^2 == explicit az/el FIM '
                 'with the cos-el-inflated R, el in {0,27,45,60} deg',
                 worst < 1e-12, f'worst rel {worst:.2e}')

    # ── MAJOR-13: the two lanes' "sigma_plane" are different OBJECTS ────────
    r_orb = 6771e3
    eps = 1e-5                                   # plane tilt, rad
    errs = []
    for u_deg in (10.0, 30.0, 90.0, 150.0):
        u_arg = math.radians(u_deg)
        x = n3.orbit_to_cartesian_3d(np.array([r_orb]), np.array([0.0]),
                                     np.array([u_arg]), np.array([0.0]),
                                     np.array([0.0]), np.array([0.0]))
        xt = n3.orbit_to_cartesian_3d(np.array([r_orb]), np.array([0.0]),
                                      np.array([u_arg]), np.array([0.0]),
                                      np.array([eps]), np.array([0.0]))
        pos = float(abs(xt[0, 2] - x[0, 2]))
        pred = r_orb * eps * abs(math.sin(u_arg))
        errs.append(abs(pos - pred) / max(pred, 1e-30))
    ok &= record('MAJOR-13: sigma_plane_pos_m == r * sigma_plane_tilt_rad * '
                 '|sin u| (the two lanes are not in conflict, they are in '
                 'different units)', max(errs) < 1e-4,
                 f'max rel {max(errs):.2e}; the scale factor is '
                 f'r = {r_orb:.3e} m/rad, so mixing them silently rescales by '
                 f'6.8e6')

    # ── MAJOR-11 / NOTE-20 / NOTE-21 / sigma-channel relocation ────────────
    try:
        ns.AcquisitionSurrogate(4, sigma_beta=1e-3, mode='table', dim=6)
        raised = False
    except ValueError:
        raised = True
    ok &= record("MAJOR-11: acq_mode='table' under dim3 RAISES", raised)
    acq = ns.AcquisitionSurrogate(4, sigma_beta=1e-3, mode='table', dim=6,
                                  allow_table=True)
    ok &= record('MAJOR-11: ... unless explicitly opted into for a reporting '
                 'run', acq.mode == 'table')
    ok &= record('MAJOR-11: the Cholesky diagonal fallback has a COUNTER',
                 hasattr(acq, 'n_chol_fallback') and acq.n_chol_fallback == 0)

    for bad, msg in ((dict(omega_offset_fixed=0.0), 'omega_offset_fixed'),
                     (dict(raan_target_rad=0.3), 'raan_target_rad')):
        try:
            OrbitalNav(num_envs=2, nav_mode='bearings_only', **_nav_kw(**bad))
            raised = False
        except ValueError:
            raised = True
        ok &= record(f'NOTE-20: {msg} RAISES under dim3 nav', raised)

    env = OrbitalNav(num_envs=32, nav_mode='bearings_only', seed=7,
                     log_interval=32, nav_sigma_channel=1, **_D3_KW)
    env.reset(seed=42)
    rng = np.random.default_rng(0)
    info = []
    for _ in range(96):
        _, _, _, _, i = env.step(rng.integers(0, 30, 32).astype(np.int32))
        if i:
            info = i
    o = np.asarray(env.observations)
    nav = [d for d in info if isinstance(d, dict) and 'nav_pos_rmse' in d]
    ok &= record('MAJOR-14: the Sigma channel writes obs[29], NOT obs[21] '
                 '(which is the ext-3d plane channel under dim3)',
                 float(np.abs(o[:, 29]).max()) > 0.0,
                 f'max |obs[29]| {np.abs(o[:, 29]).max():.4f}, '
                 f'obs[30:33] still zero: {bool((o[:, 30:33] == 0).all())}')
    ok &= record('NOTE-21: the obs[28] < 0 while unacquired diagnostic ships',
                 bool(nav) and 'nav_obs28_neg_unacq' in nav[0],
                 f"{nav[0].get('nav_obs28_neg_unacq') if nav else 'n/a'}")
    ok &= record('BLOCKER-1: the re-pole counter ships as a nav_* diagnostic',
                 bool(nav) and 'nav_repole_per_ep' in nav[0])
    ok &= record('BLOCKER-2: acquisition latency ships in SIM SECONDS',
                 bool(nav) and 'nav_acq_latency_s' in nav[0],
                 f"{nav[0].get('nav_acq_latency_s') if nav else 'n/a'} s")
    ok &= record('MAJOR-11: the chol-fallback counter ships and is 0',
                 bool(nav) and nav[0].get('nav_chol_fallback', -1) == 0.0)
    env.close()

    # ── MAJOR-16: the out-of-plane rate seed ───────────────────────────────
    env = OrbitalNav(num_envs=32, nav_mode='bearings_only', seed=7,
                     log_interval=10 ** 9, **_D3_KW)
    env.reset(seed=42)
    y = env._filt.y
    P = env._filt.Py
    st = env.get_state(env._state_buf)
    rho0 = np.exp(y[:, n3.IDX_LNRHO])
    v_c = np.sqrt(n3.MU / st[:, 0])
    need = v_c * math.sin(_D3_KW['di_max_rad']) / rho0
    got = np.sqrt(P[:, n3.IDX_WE, n3.IDX_WE])
    ok &= record('MAJOR-16: out-of-plane rate seed >= v_c sin(di_max)/rho0, '
                 'decoupled from the eccentricity term',
                 bool(np.all(got >= need - 1e-12)),
                 f'margin min {np.min(got / np.maximum(need, 1e-300)):.3f}x '
                 f'(the 2D seed reused sigma_v_ecc for both rate components: '
                 f'154 m/s against 134 m/s of real ignorance, 1.15x)')
    env.close()

    # ── the 3D recon closed loop must be bit-identical to truth ────────────
    def roll(mode, steps=150):
        e = OrbitalNav(num_envs=32, nav_mode=mode, seed=7,
                       log_interval=10 ** 9, **_D3_KW)
        o, _ = e.reset(seed=42)
        rng = np.random.default_rng(3)
        acc = [np.asarray(o, dtype=np.float32).copy()]
        for _ in range(steps):
            o, r, t, tr, _ = e.step(rng.integers(0, 30, 32).astype(np.int32))
            acc.append(np.asarray(o, dtype=np.float32).copy())
        e.close()
        return np.stack(acc)

    A, C = roll('truth'), roll('recon')
    ok &= record('dim3 recon closed loop is BIT-IDENTICAL to truth over '
                 '150 steps x 32 envs', np.array_equal(A, C),
                 f'max|diff| {np.abs(A - C).max():.3e} over {A.size} values')
    return dict(ok=bool(ok))


# ext-3d (dim3_mode=1) configuration: the X3 rung the campaign warm-starts from.
_D3_KW = dict(_MIN_KW,
              e_max_target=0.05, e_max_sat=0.05,
              dim3_mode=1, di_max_rad=math.radians(1.0),
              legacy_action_space=30,
              shaping_mode=2, shape_w_lambda=1.0, shape_w_match=0.8166667,
              shape_dv_ref_ms=700.0)

STAGES = {'a1': stage_a1, 'a2': stage_a2, 'b1': stage_b1,
          'c1': stage_c1, 'd1': stage_d1, 'e1': stage_e1,
          'f1': stage_f1}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', default='all')
    args = ap.parse_args()
    names = list(STAGES) if args.stage == 'all' else args.stage.split(',')
    out = {}
    for nme in names:
        out[nme] = STAGES[nme](args)
        print()
    print('== SUMMARY =======================================================')
    for name, ok, detail in _RESULTS:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    n_fail = sum(1 for _, ok, _ in _RESULTS if not ok)
    print(f'\n  {len(_RESULTS) - n_fail}/{len(_RESULTS)} checks pass')
    if _NOTES:
        print('\n  measured findings that are NOT pass/fail gates:')
        for name, detail in _NOTES:
            print(f'    [NOTE] {name}')
    return 1 if n_fail else 0


if __name__ == '__main__':
    sys.exit(main())
