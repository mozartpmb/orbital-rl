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


def record(name, ok, detail=''):
    _RESULTS.append((name, bool(ok), detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"   {detail}" if detail else ''))
    return bool(ok)


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
    record('two-fold i_s ambiguity at i_t=51.6 deg (why NOT Omega_s)',
           n_amb > 0.1,
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
    record('NOTE: the red-team\'s example probe (+1000 km ALONG-TRACK) is '
           'NOT a valid leakcheck garbage', len(still) > 0,
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


# ext-3d (dim3_mode=1) configuration: the X3 rung the campaign warm-starts from.
_D3_KW = dict(_MIN_KW,
              e_max_target=0.05, e_max_sat=0.05,
              dim3_mode=1, di_max_rad=math.radians(1.0),
              legacy_action_space=30,
              shaping_mode=2, shape_w_lambda=1.0, shape_w_match=0.8166667,
              shape_dv_ref_ms=700.0)

STAGES = {'a1': stage_a1, 'a2': stage_a2, 'b1': stage_b1,
          'c1': stage_c1}


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
    return 1 if n_fail else 0


if __name__ == '__main__':
    sys.exit(main())
