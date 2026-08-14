#!/usr/bin/env python3
"""J2 x NAV gate — does the bearings-only filter need a J2 term?

THE QUESTION. `BatchedBearingMSC6` predicts with two-body dynamics
(`propagate_cartesian_nd` + `stm_analytic_nd`). Under `j2_mode=1` truth the
prediction is wrong by the secular J2 drift. Does two-body + process noise
absorb it, or does the filter diverge?

THE DESIGN. Three arms over identical scenarios and identical measurement
noise, so every difference is dynamics mismatch and nothing else:

    MATCHED     truth two-body,  filter two-body   (the control)
    MISMATCHED  truth J2,        filter two-body   (what ships today)
    FIXED       truth J2,        filter J2         (the cheap fix)

The "cheap fix" is the same closed-form secular rates the C env uses, applied
to the filter's prediction step ONLY — the Cartesian STM is left two-body,
because the J2 correction to the STM is O(J2) ~ 1e-3 relative and the
covariance does not need that precision.

The filter is SEEDED FROM TRUTH with a realistic post-acquisition covariance,
then coasts. That isolates the dynamics mismatch from acquisition: an IOD
failure would otherwise dominate and tell us nothing about J2.

Run:
    PYTHONPATH=<worktree>/pufferlib python3 scripts/orbital/extj2/j2_nav_filter_probe.py
"""

import argparse
import math
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
WT = os.path.abspath(os.path.join(_HERE, '..', '..', '..'))
sys.path.insert(0, os.path.join(WT, 'pufferlib'))

import pufferlib                                                      # noqa: E402
_PL = os.path.abspath(pufferlib.__file__)
if not _PL.startswith(os.path.abspath(WT) + os.sep):
    raise SystemExit(f'REFUSING TO RUN: pufferlib is {_PL}, not under {WT}')

from pufferlib.ocean.orbital_nav import nav_math as nm                # noqa: E402
from pufferlib.ocean.orbital_nav import nav_math3d as n3              # noqa: E402

MU = 3.986004418e14
R_EARTH = 6.371e6
J2_COEF = 1.08262668e-3          # orbital.h, WGS-84
J2_R_EQ = 6.378137e6             # orbital.h, equatorial (NOT R_EARTH)
DT = 60.0                        # nav60 cadence


# ══════════════════════════════════════════════════════════════════════════
# Truth propagation — mirrors orbital.h::propagate_orbit_j2 exactly
# ══════════════════════════════════════════════════════════════════════════
def secular_rates(a, e, inc):
    n = np.sqrt(MU / a ** 3)
    p = a * (1.0 - e * e)
    k = 1.5 * n * J2_COEF * (J2_R_EQ / p) ** 2
    si2 = np.sin(inc) ** 2
    return (n,
            -k * np.cos(inc),
            0.5 * k * (4.0 - 5.0 * si2),
            n + 0.5 * k * np.sqrt(1.0 - e * e) * (2.0 - 3.0 * si2))


def kepler_E(M, e, iters=12):
    M = np.mod(M, 2 * np.pi)
    E = np.where(e < 0.8, M, np.pi)
    for _ in range(iters):
        E = E - (E - e * np.sin(E) - M) / (1.0 - e * np.cos(E))
    return E


def E_to_theta(E, e):
    return 2.0 * np.arctan2(np.sqrt(1.0 + e) * np.sin(E / 2.0),
                            np.sqrt(1.0 - e) * np.cos(E / 2.0))


class Elems:
    """Mean-element state, propagated exactly as the C env does."""

    def __init__(self, a, e, M, omega, inc, raan):
        self.a, self.e, self.M = a.copy(), e.copy(), M.copy()
        self.omega, self.inc, self.raan = omega.copy(), inc.copy(), raan.copy()

    def step(self, dt, j2):
        if not j2:
            n = np.sqrt(MU / self.a ** 3)
            self.M = np.mod(self.M + n * dt, 2 * np.pi)
            return
        n, Om, om, Md = secular_rates(self.a, self.e, self.inc)
        eq = (self.inc == 0.0)
        # the C env's equatorial special case: varpi_dot = Om_dot + om_dot,
        # raan pinned at exactly 0
        self.omega = np.where(eq,
                              np.mod(self.omega + (om + Om) * dt, 2 * np.pi),
                              np.mod(self.omega + om * dt, 2 * np.pi))
        self.raan = np.where(eq, self.raan,
                             np.mod(self.raan + Om * dt, 2 * np.pi))
        self.M = np.mod(self.M + Md * dt, 2 * np.pi)

    def cart(self):
        E = kepler_E(self.M, self.e)
        th = E_to_theta(E, self.e)
        return n3.orbit_to_cartesian_3d(self.a, self.e, th, self.omega,
                                        self.inc, self.raan)


# ══════════════════════════════════════════════════════════════════════════
# The fix: secular J2 in the filter's prediction step
# ══════════════════════════════════════════════════════════════════════════
def propagate_cartesian_j2(X, dt):
    """Two-body f&g, then the secular J2 angle increments, in Cartesian.

    Element round-trip rather than a perturbation of the f&g solution: the
    secular rates are defined on mean elements, so the honest thing is to go
    there, advance the three angles, and come back. ~2x the cost of the
    two-body call and still O(microseconds) per row.

    (a, e, inc) are invariants of secular J2, so only Omega, omega and M move —
    which is why this needs no iteration and no integrator.
    """
    el = n3.cartesian_to_elements_3d(X)
    a, e, inc = el['a'], el['e'], el['inc']
    n, Om, om, Md = secular_rates(a, e, inc)
    eq = (np.sqrt(np.maximum(1.0 - np.cos(inc) ** 2, 0.0)) == 0.0)
    omega = np.where(eq,
                     el['omega'] + (om + Om) * dt,
                     el['omega'] + om * dt)
    raan = np.where(eq, el['raan'], el['raan'] + Om * dt)
    M = el['M'] + Md * dt
    E = kepler_E(M, e)
    th = E_to_theta(E, e)
    Y = n3.orbit_to_cartesian_3d(a, e, th, omega, inc, raan)
    ok = np.all(np.isfinite(Y), axis=1) & (a > 0)
    return Y, ok


class MSC6J2(n3.BatchedBearingMSC6):
    """BatchedBearingMSC6 with a J2-aware prediction step.

    Only `predict` changes, and inside it only the STATE propagation: the
    covariance still uses the two-body analytic STM. The J2 correction to the
    STM is O(J2) ~ 1e-3 relative, far below what a covariance needs, and using
    the two-body Phi keeps the cost and the numerics of the shipped path.
    """

    def predict(self, idx, dt, sat_from, sat_to, **_):
        if idx.size == 0:
            return np.zeros(0, dtype=bool)
        Rp = self.Rp[idx]
        y = self.y[idx]
        x_old = n3.msc6_decode(y, sat_from, Rp)
        Phi, ok, _x_tb = n3.stm_analytic_nd(x_old, dt)   # two-body STM
        x_new, ok2 = propagate_cartesian_j2(x_old, dt)   # J2 state
        ok = ok & ok2
        y_new = n3.msc6_encode(sat_to, x_new, Rp)
        Jold = n3.msc6_decode_jac(y, Rp)
        Jnew = n3.msc6_decode_jac(y_new, Rp)
        with np.errstate(all='ignore'):
            try:
                Gnew = np.linalg.inv(Jnew)
            except np.linalg.LinAlgError:                # pragma: no cover
                Gnew = np.linalg.pinv(Jnew)
            A = Phi @ Jold
            M = A @ self.Py[idx] @ np.swapaxes(A, 1, 2) + self._Q(dt)
            Py = Gnew @ M @ np.swapaxes(Gnew, 1, 2)
        self.y[idx] = y_new
        self.Py[idx] = 0.5 * (Py + np.swapaxes(Py, 1, 2))
        self.sat[idx] = sat_to
        return ok & np.all(np.isfinite(y_new), axis=1) \
            & np.all(np.isfinite(Py), axis=(1, 2))


# ══════════════════════════════════════════════════════════════════════════
# Scenario sampling — the A3b distribution
# ══════════════════════════════════════════════════════════════════════════
def sample(n, rng, i_lo=30.0, i_hi=60.0, di_max_deg=1.0, e_max=0.05):
    """LEO 300-800 km, e <= 0.05, i_t ~ U(30,60) deg, relative plane <= 1 deg,
    chaser/target both drawn as the env draws them (|da| >= 50 km)."""
    a_t = R_EARTH + rng.uniform(300e3, 800e3, n)
    a_s = R_EARTH + rng.uniform(300e3, 800e3, n)
    bad = np.abs(a_s - a_t) < 50e3
    while bad.any():
        a_s[bad] = R_EARTH + rng.uniform(300e3, 800e3, bad.sum())
        bad = np.abs(a_s - a_t) < 50e3
    e_t = rng.uniform(0, e_max, n)
    e_s = rng.uniform(0, e_max, n)
    i_t = np.deg2rad(rng.uniform(i_lo, i_hi, n))
    O_t = rng.uniform(0, 2 * np.pi, n)
    w_t = rng.uniform(0, 2 * np.pi, n)
    w_s = rng.uniform(0, 2 * np.pi, n)
    M_t = rng.uniform(0, 2 * np.pi, n)
    # relative plane: rotate the chaser's plane by delta about a node axis in
    # the target plane (area-uniform disc, the shipped rotation-form sampler)
    delta = np.deg2rad(di_max_deg) * np.sqrt(rng.uniform(0, 1, n))
    ph = rng.uniform(0, 2 * np.pi, n)
    i_s = np.arccos(np.clip(np.cos(delta) * np.cos(i_t)
                            - np.sin(delta) * np.cos(ph) * np.sin(i_t), -1, 1))
    O_s = O_t + np.arctan2(np.sin(delta) * np.sin(ph),
                           np.sin(i_t) * np.cos(delta)
                           + np.cos(i_t) * np.sin(delta) * np.cos(ph))
    # chaser phase near the target's, so the pair is a rendezvous geometry
    M_s = M_t + rng.uniform(-0.02, 0.02, n)
    tgt = Elems(a_t, e_t, M_t, w_t, i_t, O_t)
    sat = Elems(a_s, e_s, M_s, w_s, i_s, O_s)
    return sat, tgt


def seed_cov(rng, n, sep):
    """Post-acquisition covariance. Deliberately NOT tiny: the campaign's real
    IOD certifies epoch errors with a 10.1 km median (N3DNAV_RESULTS), so a
    filter seeded at 1 m would be measuring a regime that never occurs."""
    sp = np.maximum(0.02 * sep, 500.0)          # 2% of separation, >= 500 m
    sv = np.maximum(2e-5 * sep, 0.5)            # ~ sp * n_orbit
    P = np.zeros((n, 6, 6))
    for k in range(3):
        P[:, k, k] = sp ** 2
        P[:, 3 + k, 3 + k] = sv ** 2
    x0err = np.concatenate([rng.normal(0, sp[:, None], (n, 3)),
                            rng.normal(0, sv[:, None], (n, 3))], axis=1)
    return P, x0err


# ══════════════════════════════════════════════════════════════════════════
def run_arm(label, truth_j2, filt_j2, hours, n, seed, sigma_beta, q_a,
            verbose=False):
    rng = np.random.default_rng(seed)
    sat, tgt = sample(n, rng)
    ticks = int(round(max(hours) * 3600.0 / DT))
    marks = {int(round(h * 3600.0 / DT)): h for h in hours}

    sat_c = sat.cart()
    tgt_c = tgt.cart()
    sep0 = np.linalg.norm(tgt_c[:, :3] - sat_c[:, :3], axis=1)
    P0, x0err = seed_cov(rng, n, sep0)

    F = (MSC6J2 if filt_j2 else n3.BatchedBearingMSC6)(
        n, sigma_beta=sigma_beta, q_a=q_a, stm='analytic')
    idx = np.arange(n)
    F.set_pole(idx, sat_c)
    F.set_cart(idx, tgt_c + x0err, P0, sat_c)

    alive = np.ones(n, dtype=bool)
    out = {}
    for t in range(1, ticks + 1):
        sat.step(DT, truth_j2)
        tgt.step(DT, truth_j2)
        sat_to = sat.cart()
        tgt_to = tgt.cart()
        F.predict(idx, DT, sat_c, sat_to)
        # bearing measurement with the shipped anisotropic noise model
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

        x_est, P_est = F.mean_cov(idx)
        alive &= np.all(np.isfinite(x_est), axis=1)

        if t in marks:
            perr = np.linalg.norm(x_est[:, :3] - tgt_to[:, :3], axis=1)
            verr = np.linalg.norm(x_est[:, 3:] - tgt_to[:, 3:], axis=1)
            nees = n3.nees_nd(x_est, P_est, tgt_to)
            good = alive & np.isfinite(nees) & np.isfinite(perr)
            # divergence: NEES above the 6-dof 95% upper bound, or a position
            # error worse than the separation it was meant to resolve
            sep = np.linalg.norm(tgt_to[:, :3] - sat_to[:, :3], axis=1)
            # Three failure modes, reported SEPARATELY. Lumping them hides
            # which one J2 causes: an inconsistent covariance (NEES out of
            # band) is a filter-health failure, an error larger than the
            # separation is a usability failure, and a non-finite state is a
            # numerical failure. Bearings-only is still converging at 1 h, so
            # a nonzero baseline in the first two is expected, not alarming.
            nees_bad = np.zeros(n, dtype=bool); nees_bad[good] = nees[good] > n3.NEES6_HI
            perr_bad = np.zeros(n, dtype=bool); perr_bad[good] = perr[good] > sep[good]
            out[marks[t]] = dict(
                n=int(good.sum()),
                perr=perr[good], verr=verr[good], nees=nees[good],
                div=float(((~good) | nees_bad | perr_bad).mean()),
                nees_bad=float(nees_bad.mean()),
                perr_bad=float(perr_bad.mean()),
                nonfinite=float((~good).mean()),
                sep=sep[good],
                repole=float(np.mean(F.n_repole)),
                label=label)
            if verbose:
                print(f'    {label} @{marks[t]}h: perr p50 '
                      f'{np.median(perr[good]):.1f} m  NEES p50 '
                      f'{np.median(nees[good]):.2f}  div {div.mean():.1%}')
    return out


def ci95(v):
    """Median with a bootstrap 95% CI — error bars, not point estimates."""
    if len(v) == 0:
        return (float('nan'),) * 3
    r = np.random.default_rng(0)
    b = np.median(r.choice(v, (400, len(v))), axis=1)
    return float(np.median(v)), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=256)
    ap.add_argument('--seed', type=int, default=20260814)
    ap.add_argument('--hours', default='1,6,24')
    ap.add_argument('--noise-mult', type=float, default=1.0)
    ap.add_argument('--q-mult', type=float, default=1.0,
                    help='process-noise PSD multiplier (the "absorb it with Q" arm)')
    args = ap.parse_args()
    hours = [float(x) for x in args.hours.split(',')]
    sb = nm.SIGMA_BETA_RAD * args.noise_mult
    qa = nm.Q_ACCEL_PSD_BO * args.q_mult

    print('=== J2 x NAV filter-health gate ===')
    print(f'    N={args.n} scenarios, nav60 cadence, arcs {hours} h')
    print(f'    LEO 300-800 km, e<=0.05, i_t ~ U(30,60) deg, rel plane <= 1 deg')
    print(f'    sigma_beta {sb:.2e} rad, q_a {qa:.2e} m^2/s^3'
          f'{" (x%g)" % args.q_mult if args.q_mult != 1 else ""}')
    print(f'    filter seeded from truth + post-IOD covariance (2% of separation)')
    print()

    # The 'absorb it with Q' question, answered rather than asserted: the
    # mismatched filter is re-run with the process-noise PSD inflated by
    # 1e2 / 1e4 / 1e6. If two-body + Q suffices, one of these recovers the
    # matched arm; if it does not, the margin is the table.
    arms = [('MATCHED    truth 2body, filt 2body        ', 0, 0, 1.0),
            ('MISMATCHED truth J2,    filt 2body        ', 1, 0, 1.0),
            ('MISMATCHED truth J2,    filt 2body, Q x1e2', 1, 0, 1e2),
            ('MISMATCHED truth J2,    filt 2body, Q x1e4', 1, 0, 1e4),
            ('MISMATCHED truth J2,    filt 2body, Q x1e6', 1, 0, 1e6),
            ('FIXED      truth J2,    filt J2           ', 1, 1, 1.0),
            ('FIXED      truth J2,    filt J2,    Q x1e1', 1, 1, 1e1),
            ('FIXED      truth J2,    filt J2,    Q x1e2', 1, 1, 1e2)]
    res = {}
    for label, tj, fj, qm in arms:
        res[label] = run_arm(label, tj, fj, hours, args.n, args.seed, sb,
                             qa * qm)

    print(f'{"arm":42s} {"arc":>4s} {"pos err p50 [95% CI] (m)":>30s} '
          f'{"vel p50":>9s} {"NEES p50 [95% CI]":>24s} {"NEES>band":>9s} '
          f'{"err>sep":>8s} {"nonfin":>7s} {"repole":>7s}')
    for label, _, _, _ in arms:
        for h in hours:
            c = res[label][h]
            pm, plo, phi = ci95(c['perr'])
            vm, _, _ = ci95(c['verr'])
            nmed, nlo, nhi = ci95(c['nees'])
            print(f'{label:42s} {h:3.0f}h {pm:10.1f} [{plo:8.1f},{phi:8.1f}] '
                  f'{vm:8.3f} {nmed:9.2f} [{nlo:6.2f},{nhi:7.2f}] '
                  f'{c["nees_bad"]:8.1%} {c["perr_bad"]:7.1%} '
                  f'{c["nonfinite"]:6.1%} {c["repole"]:7.2f}')
        print()

    print(f'    NEES 6-dof 95% band: [{n3.NEES6_LO:.3f}, {n3.NEES6_HI:.3f}]'
          '  (1.0 = consistent)')
    print()
    print('=== VERDICT INPUTS (ratios vs the MATCHED control) ===')
    base = arms[0][0]
    for h in hours:
        a = res[base][h]
        pa, na = np.median(a['perr']), np.median(a['nees'])
        print(f'  --- {h:.0f} h arc ---   matched: pos {pa:.1f} m, NEES {na:.2f}, '
              f'NEES>band {a["nees_bad"]:.1%}')
        for label, _, _, _ in arms[1:]:
            c = res[label][h]
            pc, nc = np.median(c['perr']), np.median(c['nees'])
            print(f'      {label:42s} pos {pc:10.1f} m ({pc/max(pa,1e-9):7.2f}x)  '
                  f'NEES {nc:10.2f} ({nc/max(na,1e-9):8.2f}x)  '
                  f'NEES>band {c["nees_bad"]:6.1%}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
