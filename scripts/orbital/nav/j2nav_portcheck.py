#!/usr/bin/env python3
"""Port check: the PRODUCTION J2 filter reproduces the ext-j2 head-to-head.

`nav_math3d.BatchedBearingMSC6J2` is a port of `MSC6J2Cov(stm_j2='fd')` from
scripts/orbital/extj2/j2_nav_filter_probe.py, which is where the covariance
spec was DECIDED. A port that quietly changed the answer would be worse than no
port, so this re-measures the two arms that matter, with the production
classes, against a J2 truth:

    FIXED   J2 state, two-body covariance  (local `_FixedArm`, the rejected
                                            arm — deliberately not shipped)
    C1-fd   J2 state, J2 covariance (fd)   (nav_math3d.BatchedBearingMSC6J2)

Probe reference (N=128), position median / NEES median / in-band:
    6 h    FIXED 1693.9 m / 1.40 / 66.4%      C1-fd 1632.8 m / 1.02 / 83.6%
    24 h   FIXED 1202.7 m / 11.37 / 21.9%     C1-fd  457.3 m / 1.01 / 87.5%

The load-bearing claim is not the absolute metres — the scenario sampler here
is a re-implementation of the probe's cell, not an import of it, so absolute
NEES and in-band move with N, seed and geometry mix. What must survive the port
is the RELATIONSHIP: at 24 h the two-body covariance leaves NEES orders of
magnitude high AND pays several times in position, because an overconfident P
shrinks the Kalman gain, while the J2 covariance holds NEES to order 1.

Measured here (N=64): FIXED 2145 m / NEES 2078 / 0% in band against C1-fd
457.6 m / NEES 2.13 / 59.4%, a 4.69x position ratio. C1-fd's 24 h position
lands within 0.1% of the probe's 457.3 m, which is the strongest single
agreement available and is what says the ported filter is the same object.

    python3 scripts/orbital/nav/j2nav_portcheck.py --n 48
"""

import argparse
import math
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(_HERE, '..', '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'pufferlib'))

from pufferlib.ocean.orbital_nav import nav_math as nm             # noqa: E402
from pufferlib.ocean.orbital_nav import nav_math3d as n3           # noqa: E402

MU = n3.MU
DT = 60.0


class Elems:
    """Mean-element truth, propagated exactly as the C env does."""

    def __init__(self, a, e, M, omega, inc, raan):
        self.a, self.e, self.M = a.copy(), e.copy(), M.copy()
        self.omega, self.inc, self.raan = omega.copy(), inc.copy(), raan.copy()

    def step(self, dt, j2):
        if not j2:
            self.M = np.mod(self.M + np.sqrt(MU / self.a ** 3) * dt, 2 * np.pi)
            return
        _n, Om, om, Md = n3.j2_secular_rates(self.a, self.e, self.inc)
        eq = (self.inc == 0.0)
        self.omega = np.where(eq, np.mod(self.omega + (om + Om) * dt, 2 * np.pi),
                              np.mod(self.omega + om * dt, 2 * np.pi))
        self.raan = np.where(eq, self.raan, np.mod(self.raan + Om * dt, 2 * np.pi))
        self.M = np.mod(self.M + Md * dt, 2 * np.pi)

    def cart(self):
        th = nm.eccentric_to_true(nm.solve_kepler(self.M, self.e), self.e)
        return n3.orbit_to_cartesian_3d(self.a, self.e, th, self.omega,
                                        self.inc, self.raan)


class _FixedArm(n3.BatchedBearingMSC6):
    """The REJECTED arm, kept here and not in production: J2 in the state,
    two-body in the covariance (the probe's `MSC6J2`).

    It has to be built locally because shipping it would mean shipping a filter
    the head-to-head already rejected. Note what it is NOT: a fully two-body
    filter under a J2 truth diverges outright (measured here at 735 km / NEES
    4.8e9 at 6 h), which is a different and much easier thing to beat. The
    honest comparator keeps J2 in the state so the ONLY difference is the
    covariance — that is the claim being reproduced.
    """

    def predict(self, idx, dt, sat_from, sat_to, **_):
        if idx.size == 0:
            return np.zeros(0, dtype=bool)
        Rp = self.Rp[idx]
        y = self.y[idx]
        x_old = n3.msc6_decode(y, sat_from, Rp)
        Phi, ok, _x_tb = n3.stm_analytic_nd(x_old, dt)     # two-body STM
        x_new, ok2 = n3.propagate_cartesian_j2(x_old, dt)  # J2 state
        ok = ok & ok2
        y_new = n3.msc6_encode(sat_to, x_new, Rp)
        Jold = n3.msc6_decode_jac(y, Rp)
        Jnew = n3.msc6_decode_jac(y_new, Rp)
        with np.errstate(all='ignore'):
            try:
                Gnew = np.linalg.inv(Jnew)
            except np.linalg.LinAlgError:
                Gnew = np.linalg.pinv(Jnew)
            A = Phi @ Jold
            M = A @ self.Py[idx] @ np.swapaxes(A, 1, 2) + self._Q(dt)
            Py = Gnew @ M @ np.swapaxes(Gnew, 1, 2)
        self.y[idx] = y_new
        self.Py[idx] = 0.5 * (Py + np.swapaxes(Py, 1, 2))
        self.sat[idx] = sat_to
        return (ok & np.all(np.isfinite(y_new), axis=1)
                & np.all(np.isfinite(Py), axis=(1, 2)))


def sample(n, rng):
    """The campaign's own cell: LEO 300-800 km, e <= 0.05, i_t ~ U(30,60) deg,
    relative plane <= 1 deg."""
    a_t = n3.R_EARTH + rng.uniform(300e3, 800e3, n)
    e_t = rng.uniform(0.0, 0.05, n)
    inc_t = np.radians(rng.uniform(30.0, 60.0, n))
    raan_t = rng.uniform(0, 2 * np.pi, n)
    om_t = rng.uniform(0, 2 * np.pi, n)
    M_t = rng.uniform(0, 2 * np.pi, n)
    di = np.radians(1.0) * np.sqrt(rng.uniform(0, 1, n))
    ph = rng.uniform(0, 2 * np.pi, n)
    inc_s = inc_t + di * np.cos(ph)
    raan_s = raan_t + di * np.sin(ph) / np.maximum(np.sin(inc_t), 1e-3)
    a_s = a_t + rng.uniform(-3e4, 3e4, n)
    return (Elems(a_s, e_t.copy(), np.mod(M_t + rng.uniform(-1e-3, 1e-3, n),
                                          2 * np.pi),
                  om_t.copy(), inc_s, raan_s),
            Elems(a_t, e_t, M_t, om_t, inc_t, raan_t))


def seed_cov(rng, n, sep0):
    """Post-IOD covariance and a matching error draw."""
    sp = np.maximum(0.02 * sep0, 500.0)
    sv = np.maximum(2e-5 * sep0, 0.5)
    P = np.zeros((n, 6, 6))
    err = np.zeros((n, 6))
    for k in range(3):
        P[:, k, k] = sp ** 2
        P[:, 3 + k, 3 + k] = sv ** 2
        err[:, k] = rng.normal(0, sp)
        err[:, 3 + k] = rng.normal(0, sv)
    return P, err


def run_arm(kind, hours, n, seed, sigma_beta=1e-3, q_a=1e-13):
    rng = np.random.default_rng(seed)
    sat, tgt = sample(n, rng)
    ticks = int(round(max(hours) * 3600.0 / DT))
    marks = {int(round(h * 3600.0 / DT)): h for h in hours}
    sat_c, tgt_c = sat.cart(), tgt.cart()
    sep0 = np.linalg.norm(tgt_c[:, :3] - sat_c[:, :3], axis=1)
    P0, x0err = seed_cov(rng, n, sep0)

    if kind == 'j2cov_fd':
        F = n3.BatchedBearingMSC6J2(n, sigma_beta=sigma_beta, q_a=q_a,
                                    stm_j2='fd')
    elif kind == 'fixed':
        F = _FixedArm(n, sigma_beta=sigma_beta, q_a=q_a, stm='analytic')
    else:
        F = n3.BatchedBearingMSC6(n, sigma_beta=sigma_beta, q_a=q_a,
                                  stm='analytic')
    idx = np.arange(n)
    F.set_pole(idx, sat_c)
    F.set_cart(idx, tgt_c + x0err, P0, sat_c)

    out = {}
    for t in range(1, ticks + 1):
        sat.step(DT, True)                 # truth is J2 in BOTH arms
        tgt.step(DT, True)
        sat_to, tgt_to = sat.cart(), tgt.cart()
        F.predict(idx, DT, sat_c, sat_to)
        d = tgt_to[:, :3] - sat_to[:, :3]
        u_p = np.einsum('nij,nj->ni', F.Rp, d)
        rho = np.linalg.norm(u_p, axis=1)
        az = np.arctan2(u_p[:, 1], u_p[:, 0])
        el = np.arcsin(np.clip(u_p[:, 2] / rho, -1, 1))
        az = az + rng.normal(0, sigma_beta / np.maximum(np.cos(el), 1e-3), n)
        el = el + rng.normal(0, sigma_beta, n)
        F.update(idx, sat_to, az, el)
        F.repole(idx)
        sat_c = sat_to
        if t in marks:
            x, P = F.mean_cov(idx)
            pos = np.linalg.norm(x[:, :3] - tgt_to[:, :3], axis=1)
            vel = np.linalg.norm(x[:, 3:] - tgt_to[:, 3:], axis=1)
            nees = n3.nees_nd(x, P, tgt_to)
            fin = np.isfinite(nees)
            out[marks[t]] = dict(
                pos=float(np.median(pos[np.isfinite(pos)])),
                vel=float(np.median(vel[np.isfinite(vel)])),
                nees=float(np.median(nees[fin])),
                in_band=float(np.mean((nees[fin] >= n3.NEES6_LO)
                                      & (nees[fin] <= n3.NEES6_HI))))
    return out


def inloop_nees(rung, steps=400, seed=42):
    """NEES of the filter inside the REAL training loop, split by acquisition.

    The standalone arms above seed from truth and never acquire, burn or
    re-pole at scale. This runs the actual wrapper. It exists because the two
    disagree: the J2 filter is consistent standalone (NEES ~2 at 24 h) and
    inconsistent in the loop, which no filter-vs-filter check can see.
    """
    sys.path.insert(0, os.path.join(ROOT, 'scripts', 'orbital', 'nav'))
    import eval_relnav3d as ER3
    env = ER3.make_env(rung, 'bearings_only')
    env.reset(seed=seed)
    rng = np.random.default_rng(0)
    n = env.num_agents
    acq_n, hand_n = [], []
    prev = np.zeros(n, dtype=bool)
    for _ in range(steps):
        env.step(rng.integers(0, 30, n).astype(np.int32))
        idx = np.arange(n)
        x, P = env._filt.mean_cov(idx)
        v = n3.nees_nd(x, P, env._prev_tgt)
        acq = env._acq.acquired.copy()
        fresh = acq & ~prev
        fin = np.isfinite(v)
        if (acq & fin).any():
            acq_n += list(v[acq & fin])
        if (fresh & fin).any():
            hand_n += list(v[fresh & fin])
        prev = acq
    env.close()
    med = lambda a: float(np.median(a)) if a else float('nan')
    return med(acq_n), med(hand_n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=48)
    ap.add_argument('--seed', type=int, default=20260814)
    ap.add_argument('--hours', default='6,24')
    ap.add_argument('--skip-inloop', action='store_true')
    a = ap.parse_args()
    hours = [float(h) for h in a.hours.split(',')]

    print(f'== PORT CHECK  N={a.n}  seed={a.seed}  truth = J2 in both arms ==')
    res = {k: run_arm(k, hours, a.n, a.seed) for k in ('fixed', 'j2cov_fd')}
    print(f"{'arm':34} {'arc':>5} {'pos med':>10} {'vel med':>9} "
          f"{'NEES med':>9} {'in band':>8}")
    for k, lab in (('fixed', 'FIXED  J2 state, two-body cov'),
                   ('j2cov_fd', 'C1-fd  J2 truth, J2 cov (fd)')):
        for h in hours:
            r = res[k][h]
            print(f'{lab:34} {h:4.0f}h {r["pos"]:10.1f} {r["vel"]:9.3f} '
                  f'{r["nees"]:9.2f} {r["in_band"]:7.1%}')
    ok = True
    for h in hours:
        f, c = res['fixed'][h], res['j2cov_fd'][h]
        ratio = f['pos'] / max(c['pos'], 1e-9)
        print(f'\n  {h:.0f} h: FIXED/C1-fd position {ratio:.2f}x   '
              f'NEES {f["nees"]:.2f} vs {c["nees"]:.2f}   '
              f'in-band {f["in_band"]:.1%} vs {c["in_band"]:.1%}')
        if h >= 24:
            # The decision this port must preserve: at the 24 h operating point
            # the two-body covariance is BOTH inconsistent and materially worse
            # in position, and the J2 covariance restores both.
            # Thresholds test the CLAIM, not the probe's exact sampler: the
            # two-body covariance is inconsistent by orders of magnitude and
            # materially worse in position; the J2 covariance is order-1
            # consistent. Tightening these to the probe's own 11.37 / 1.01
            # would be fitting the gate to a geometry mix this file does not
            # reproduce and never claimed to.
            good = (f['nees'] > 10.0 and c['nees'] < 5.0
                    and ratio > 1.8 and c['in_band'] > 0.50)
            ok &= good
            print(f"  [{'PASS' if good else 'FAIL'}] 24 h relationship "
                  f"reproduced (FIXED NEES > 10, C1-fd NEES < 5, "
                  f"position ratio > 1.8, C1-fd in-band > 50%)")
    # ── the gate that matters for the campaign ─────────────────────────────
    # A filter that is consistent standalone and inconsistent in the loop would
    # let the campaign produce numbers whose covariance means nothing. Measured
    # 2026-08-14: two-body 0.66 acquired / 1.02 at handoff; J2 25520 / 1818.
    # The campaign must not launch while this is red.
    if not a.skip_inloop:
        print('\n  in-loop NEES (real wrapper, bearings_only, 400 steps):')
        for rung in ('X3', 'J2X'):
            ac, hd = inloop_nees(rung)
            print(f'    {rung:5} acquired {ac:12.2f}   at handoff {hd:10.2f}')
            if rung == 'J2X':
                good = ac < 20.0 and hd < 20.0
                ok &= good
                print(f"    [{'PASS' if good else 'FAIL'}] J2 filter is "
                      f"consistent IN THE LOOP (acquired and handoff NEES < 20)")
    print(f"\n  PORT CHECK: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
