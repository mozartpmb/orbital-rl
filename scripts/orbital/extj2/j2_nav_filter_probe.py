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
# Candidate 1 — the J2 term in the COVARIANCE propagation
# ══════════════════════════════════════════════════════════════════════════
def stm_fd_j2(X, dt, h_pos=1.0, h_vel=1e-3):
    """Central-difference STM of `propagate_cartesian_j2`.

    Correct by construction, 12 extra propagations per tick. This is the
    REFERENCE the cheap analytic form is scored against, not a shippable path.
    """
    n = X.shape[0]
    Y, ok = propagate_cartesian_j2(X, dt)
    F = np.empty((n, 6, 6))
    h = np.array([h_pos] * 3 + [h_vel] * 3)
    for j in range(6):
        Xp = X.copy(); Xp[:, j] += h[j]
        Xm = X.copy(); Xm[:, j] -= h[j]
        Yp, _ = propagate_cartesian_j2(Xp, dt)
        Ym, _ = propagate_cartesian_j2(Xm, dt)
        F[:, :, j] = (Yp - Ym) / (2.0 * h[j])
    return F, ok, Y


def _dq_dx(X):
    """Closed-form partials of the rate arguments (a, i) w.r.t. the Cartesian
    state. e is DELIBERATELY omitted — see `stm_analytic_j2`."""
    r = X[:, :3]; v = X[:, 3:]
    rn = np.linalg.norm(r, axis=1); vn2 = np.einsum('ni,ni->n', v, v)
    a = 1.0 / (2.0 / rn - vn2 / MU)
    da = np.concatenate([(2.0 * a**2 / rn**3)[:, None] * r,
                         (2.0 * a**2 / MU)[:, None] * v], axis=1)
    h = np.cross(r, v); hn = np.linalg.norm(h, axis=1)
    hh = h / hn[:, None]
    inc = np.arccos(np.clip(hh[:, 2], -1.0, 1.0))
    si = np.maximum(np.sin(inc), 1e-9)
    z = np.zeros_like(hh); z[:, 2] = 1.0
    g = -(z - hh[:, 2:3] * hh) / (hn * si)[:, None]      # di/dh
    di = np.concatenate([np.cross(v, g), np.cross(g, r)], axis=1)
    return a, inc, da, di


def stm_analytic_j2(X, dt):
    """Phi_J2 = Phi_2body + dt * (angle-column) x (rate-partial) outer products.

    In ELEMENT space the secular map is trivial — (a, e, i) are invariant and
    the three angles gain constant rates — so its Jacobian is
    I + dt * d(rates)/d(a,e,i) in the (Omega, omega, M) rows. Pushing that
    through coe2rv gives, exactly,

        dPhi = dt * [ dG/dOmega (x) dOmdot/dq + dG/domega (x) domdot/dq
                                              + dG/dM     (x) dMdot_J2/dq ] dq/dx

    and every factor is closed form:
        dG/dOmega = (z_hat x r,  z_hat x v)      (rotation about z)
        dG/domega = (h_hat x r,  h_hat x v)      (rotation about h)
        dG/dM     = (v/n,       -mu r/(r^3 n))   (motion along the orbit)
        dOmdot/da = -3.5 Omdot/a ,  dOmdot/di = +k sin i , etc.

    THREE OUTER PRODUCTS, and MEASURED TO BE INSUFFICIENT — kept in the tree
    because the negative result is the useful part.

    It removes only ~59% (p05 41%) of the two-body STM's error and it
    OVERSHOOTS (|model| max 4.82e-1 vs |truth| max 3.41e-1). The partials
    themselves are right: `_dq_dx` agrees with finite differences to 1.2e-9,
    and the e-route is provably negligible here (dOmdot/de = Omdot*4e/(1-e^2)
    is 5e-5 of the a-route at e <= 0.05). What breaks it is that the exact
    chain has the form

        dPhi = [dG_J2 - dG_2body] (dangles/dx + dt d(rates)/dq dq/dx)
             + dG_J2 dt d(rates - n e_M)/dq dq/dx

    and the two pieces are individually ~150 while their difference is 0.34 —
    a 440:1 cancellation. Truncating to the second piece keeps a residual of
    the same order as the quantity being modelled. Getting it right needs the
    full coe2rv/rv2coe Jacobian chain, at which point the finite-difference
    form is simpler AND correct. See J2_RECON_NOTES.md.
    """
    Phi, ok, _ = n3.stm_analytic_nd(X, dt)
    Y, ok2 = propagate_cartesian_j2(X, dt)
    a, inc, da_dx, di_dx = _dq_dx(X)

    e = np.zeros_like(a)                      # e-partials omitted by design
    n_mm = np.sqrt(MU / a ** 3)
    p = a * (1.0 - e * e)
    k = 1.5 * n_mm * J2_COEF * (J2_R_EQ / p) ** 2
    ci, si = np.cos(inc), np.sin(inc)
    Om_d = -k * ci
    om_d = 0.5 * k * (4.0 - 5.0 * si * si)
    Mj_d = 0.5 * k * np.sqrt(1.0 - e * e) * (2.0 - 3.0 * si * si)   # J2 part only

    dOm_da = -3.5 * Om_d / a
    dOm_di = k * si
    dom_da = -3.5 * om_d / a
    dom_di = -2.5 * k * np.sin(2.0 * inc)
    dMj_da = -3.5 * Mj_d / a
    dMj_di = -1.5 * k * np.sin(2.0 * inc)

    rN, vN = Y[:, :3], Y[:, 3:]
    rnN = np.linalg.norm(rN, axis=1)
    hN = np.cross(rN, vN); hhN = hN / np.linalg.norm(hN, axis=1)[:, None]
    z = np.zeros_like(rN); z[:, 2] = 1.0
    colOm = np.concatenate([np.cross(z, rN), np.cross(z, vN)], axis=1)
    colom = np.concatenate([np.cross(hhN, rN), np.cross(hhN, vN)], axis=1)
    nN = np.sqrt(MU / a ** 3)
    colM = np.concatenate([vN / nN[:, None],
                           (-MU / (rnN ** 3 * nN))[:, None] * rN], axis=1)

    rowOm = dOm_da[:, None] * da_dx + dOm_di[:, None] * di_dx
    rowom = dom_da[:, None] * da_dx + dom_di[:, None] * di_dx
    rowM = dMj_da[:, None] * da_dx + dMj_di[:, None] * di_dx

    dPhi = dt * (colOm[:, :, None] * rowOm[:, None, :]
                 + colom[:, :, None] * rowom[:, None, :]
                 + colM[:, :, None] * rowM[:, None, :])
    return Phi + dPhi, ok & ok2, Y


class MSC6J2Cov(MSC6J2):
    """C1: J2 state propagation AND a J2-aware covariance propagation."""

    def __init__(self, *a, stm_j2='analytic', **kw):
        super().__init__(*a, **kw)
        self.stm_j2 = stm_j2

    def predict(self, idx, dt, sat_from, sat_to, **_):
        if idx.size == 0:
            return np.zeros(0, dtype=bool)
        Rp = self.Rp[idx]
        y = self.y[idx]
        x_old = n3.msc6_decode(y, sat_from, Rp)
        if self.stm_j2 == 'fd':
            Phi, ok, x_new = stm_fd_j2(x_old, dt)
        else:
            Phi, ok, x_new = stm_analytic_j2(x_old, dt)
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


class MSC6J2Struct(MSC6J2):
    """C3: J2 state, two-body STM, STRUCTURED inflation.

    The neglected covariance term is a rank-2 object: the secular bias feeds
    the node direction (via dOmdot/da) and the along-track direction (via
    dMdot/da), driven by the state's own semi-major-axis uncertainty. So
    inflate exactly those two directions by exactly that amount,

        dP = dt^2 * sigma_a^2 [ (dOmdot/da)^2 colOm colOm^T
                              + (dMdot_J2/da)^2 colM colM^T ]

    with sigma_a^2 = (da/dx)^T P (da/dx) read off the current covariance. Two
    outer products and no i-partials — cheaper than C1, and it is the honest
    middle option rather than a scalar fudge, because it inflates only the
    subspace the bias actually occupies.
    """

    def predict(self, idx, dt, sat_from, sat_to, **_):
        if idx.size == 0:
            return np.zeros(0, dtype=bool)
        Rp = self.Rp[idx]
        y = self.y[idx]
        x_old = n3.msc6_decode(y, sat_from, Rp)
        Phi, ok, _ = n3.stm_analytic_nd(x_old, dt)
        x_new, ok2 = propagate_cartesian_j2(x_old, dt)
        y_new = n3.msc6_encode(sat_to, x_new, Rp)
        Jold = n3.msc6_decode_jac(y, Rp)
        Jnew = n3.msc6_decode_jac(y_new, Rp)

        a, inc, da_dx, _ = _dq_dx(x_old)
        n_mm = np.sqrt(MU / a ** 3)
        k = 1.5 * n_mm * J2_COEF * (J2_R_EQ / a) ** 2
        ci, si = np.cos(inc), np.sin(inc)
        dOm_da = -3.5 * (-k * ci) / a
        dMj_da = -3.5 * (0.5 * k * (2.0 - 3.0 * si * si)) / a
        rN, vN = x_new[:, :3], x_new[:, 3:]
        rnN = np.linalg.norm(rN, axis=1)
        z = np.zeros_like(rN); z[:, 2] = 1.0
        colOm = np.concatenate([np.cross(z, rN), np.cross(z, vN)], axis=1)
        colM = np.concatenate([vN / n_mm[:, None],
                               (-MU / (rnN ** 3 * n_mm))[:, None] * rN], axis=1)
        with np.errstate(all='ignore'):
            try:
                Gnew = np.linalg.inv(Jnew)
            except np.linalg.LinAlgError:                # pragma: no cover
                Gnew = np.linalg.pinv(Jnew)
            A = Phi @ Jold
            Pcart = Jold @ self.Py[idx] @ np.swapaxes(Jold, 1, 2)
            sig_a2 = np.einsum('ni,nij,nj->n', da_dx, Pcart, da_dx)
            dP = (dt ** 2) * sig_a2[:, None, None] * (
                (dOm_da ** 2)[:, None, None] * colOm[:, :, None] * colOm[:, None, :]
                + (dMj_da ** 2)[:, None, None] * colM[:, :, None] * colM[:, None, :])
            M = A @ self.Py[idx] @ np.swapaxes(A, 1, 2) + self._Q(dt) + dP
            Py = Gnew @ M @ np.swapaxes(Gnew, 1, 2)
        self.y[idx] = y_new
        self.Py[idx] = 0.5 * (Py + np.swapaxes(Py, 1, 2))
        self.sat[idx] = sat_to
        return ok & ok2 & np.all(np.isfinite(y_new), axis=1) \
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
def make_filter(kind, n, sigma_beta, q_a):
    if kind == '2body':
        return n3.BatchedBearingMSC6(n, sigma_beta=sigma_beta, q_a=q_a, stm='analytic')
    if kind == 'j2':
        return MSC6J2(n, sigma_beta=sigma_beta, q_a=q_a, stm='analytic')
    if kind == 'j2cov_an':
        return MSC6J2Cov(n, sigma_beta=sigma_beta, q_a=q_a, stm='analytic',
                         stm_j2='analytic')
    if kind == 'j2cov_fd':
        return MSC6J2Cov(n, sigma_beta=sigma_beta, q_a=q_a, stm='analytic',
                         stm_j2='fd')
    if kind == 'j2struct':
        return MSC6J2Struct(n, sigma_beta=sigma_beta, q_a=q_a, stm='analytic')
    raise ValueError(kind)


def run_arm(label, truth_j2, filt_j2, hours, n, seed, sigma_beta, q_a,
            verbose=False, kind=None, reinit_h=0.0):
    rng = np.random.default_rng(seed)
    sat, tgt = sample(n, rng)
    ticks = int(round(max(hours) * 3600.0 / DT))
    marks = {int(round(h * 3600.0 / DT)): h for h in hours}

    sat_c = sat.cart()
    tgt_c = tgt.cart()
    sep0 = np.linalg.norm(tgt_c[:, :3] - sat_c[:, :3], axis=1)
    P0, x0err = seed_cov(rng, n, sep0)

    if kind is None:
        kind = 'j2' if filt_j2 else '2body'
    F = make_filter(kind, n, sigma_beta, q_a)
    reinit_ticks = int(round(reinit_h * 3600.0 / DT)) if reinit_h > 0 else 0
    spike = []          # position-error jump caused by each re-init
    sig_jump = []       # sigma_pos jump, for the acquisition-gate question
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

        # C2: periodic covariance re-initialisation. A real system re-acquires
        # rather than reading truth, so the prior is rebuilt from the CURRENT
        # estimate's range, exactly as the seed was.
        if reinit_ticks and t % reinit_ticks == 0 and t < ticks:
            rho_est = np.linalg.norm(x_est[:, :3] - sat_to[:, :3], axis=1)
            sp = np.maximum(0.02 * rho_est, 500.0)
            sv = np.maximum(2e-5 * rho_est, 0.5)
            e_before = np.linalg.norm(x_est[:, :3] - tgt_to[:, :3], axis=1)
            s_before = np.sqrt(np.maximum(
                np.trace(P_est[:, :3, :3], axis1=1, axis2=2), 0.0))
            Pn = np.zeros((n, 6, 6))
            for kk in range(3):
                Pn[:, kk, kk] = sp ** 2
                Pn[:, 3 + kk, 3 + kk] = sv ** 2
            F.set_cart(idx, x_est, Pn, sat_to)
            x2, P2 = F.mean_cov(idx)
            e_after = np.linalg.norm(x2[:, :3] - tgt_to[:, :3], axis=1)
            s_after = np.sqrt(np.maximum(
                np.trace(P2[:, :3, :3], axis1=1, axis2=2), 0.0))
            spike.append(float(np.median(e_after - e_before)))
            sig_jump.append(float(np.median(s_after / np.maximum(s_before, 1e-9))))
            x_est, P_est = x2, P2

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
                reinit_spike=float(np.median(spike)) if spike else 0.0,
                reinit_sig_jump=float(np.median(sig_jump)) if sig_jump else 1.0,
                n_reinit=len(spike),
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


def tick_cost(kind, n, sigma_beta, q_a, reps=40):
    """Wall time per predict() call, the only part any candidate changes."""
    import time as _t
    rng = np.random.default_rng(7)
    sat, tgt = sample(n, rng)
    sat_c, tgt_c = sat.cart(), tgt.cart()
    sep = np.linalg.norm(tgt_c[:, :3] - sat_c[:, :3], axis=1)
    P0, err = seed_cov(rng, n, sep)
    F = make_filter(kind, n, sigma_beta, q_a)
    idx = np.arange(n)
    F.set_pole(idx, sat_c); F.set_cart(idx, tgt_c + err, P0, sat_c)
    F.predict(idx, DT, sat_c, sat_c)                    # warm
    t0 = _t.perf_counter()
    for _ in range(reps):
        F.predict(idx, DT, sat_c, sat_c)
    return (_t.perf_counter() - t0) / reps * 1e3        # ms per tick


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=128)
    ap.add_argument('--seed', type=int, default=20260814)
    ap.add_argument('--hours', default='6,24')
    ap.add_argument('--noise-mult', type=float, default=1.0)
    ap.add_argument('--stage', default='candidates',
                    help="'base' = the original 3-arm + Q table; "
                         "'candidates' = the C1/C2/C3 head-to-head")
    args = ap.parse_args()
    hours = [float(x) for x in args.hours.split(',')]
    sb = nm.SIGMA_BETA_RAD * args.noise_mult
    qa = nm.Q_ACCEL_PSD_BO

    print('=== J2 x NAV: covariance-consistency candidates ===')
    print(f'    N={args.n}, nav60, arcs {hours} h, LEO 300-800 km, e<=0.05,')
    print(f'    i_t ~ U(30,60) deg, rel plane <= 1 deg; truth is J2 in every arm')
    print(f'    except MATCHED. Filter seeded from truth + post-IOD covariance.')
    print()

    # ── first: is the cheap analytic STM correction right? ────────────────
    rng = np.random.default_rng(11)
    sat, tgt = sample(64, rng)
    X = tgt.cart()
    Ffd, _, _ = stm_fd_j2(X, DT)
    Fan, _, _ = stm_analytic_j2(X, DT)
    F2b, _, _ = n3.stm_analytic_nd(X, DT)
    d_j2 = np.abs(Ffd - F2b)
    d_an = np.abs(Ffd - Fan)
    # score the CORRECTION, not the matrix: how much of the two-body error
    # does the analytic form actually remove?
    frac = 1.0 - d_an.max(axis=(1, 2)) / np.maximum(d_j2.max(axis=(1, 2)), 1e-300)
    print('--- C1 sanity: analytic dPhi vs the finite-difference reference ---')
    print(f'    |Phi_fdJ2 - Phi_2body| max        = {d_j2.max():.3e}  '
          f'(the term being modelled)')
    print(f'    |Phi_fdJ2 - Phi_analyticJ2| max   = {d_an.max():.3e}')
    print(f'    fraction of the two-body STM error removed: p50 {np.median(frac):.4f}, '
          f'p05 {np.percentile(frac, 5):.4f}')
    print()

    if args.stage == 'base':
        arms = [('MATCHED    truth 2body, filt 2body        ', 0, 0, None, 0.0),
                ('MISMATCHED truth J2,    filt 2body        ', 1, 0, None, 0.0),
                ('FIXED      truth J2,    filt J2           ', 1, 1, None, 0.0)]
    else:
        arms = [
            ('MATCHED    2body/2body  (control)         ', 0, 0, '2body',    0.0),
            ('FIXED      J2 state, 2body cov            ', 1, 1, 'j2',       0.0),
            ('C1-an      J2 state, J2 cov (analytic)    ', 1, 1, 'j2cov_an', 0.0),
            ('C1-fd      J2 state, J2 cov (fin-diff ref)', 1, 1, 'j2cov_fd', 0.0),
            ('C2-T4      J2 state, re-init every 4 h    ', 1, 1, 'j2',       4.0),
            ('C2-T8      J2 state, re-init every 8 h    ', 1, 1, 'j2',       8.0),
            ('C2-T12     J2 state, re-init every 12 h   ', 1, 1, 'j2',      12.0),
            ('C3         J2 state, structured inflation ', 1, 1, 'j2struct', 0.0),
        ]

    res = {}
    for label, tj, fj, kind, rh in arms:
        res[label] = run_arm(label, tj, fj, hours, args.n, args.seed, sb, qa,
                             kind=kind, reinit_h=rh)

    print(f'{"arm":44s} {"arc":>4s} {"pos p50 [95% CI] (m)":>28s} {"vel p50":>9s} '
          f'{"NEES p50 [95% CI]":>24s} {"in band":>8s} {"repole":>7s}')
    for label, _, _, _, _ in arms:
        for h in hours:
            c = res[label][h]
            pm, plo, phi = ci95(c['perr'])
            vm, _, _ = ci95(c['verr'])
            nmed, nlo, nhi = ci95(c['nees'])
            print(f'{label:44s} {h:3.0f}h {pm:9.1f} [{plo:8.1f},{phi:8.1f}] '
                  f'{vm:8.3f} {nmed:9.2f} [{nlo:6.2f},{nhi:7.2f}] '
                  f'{1.0 - c["nees_bad"]:7.1%} {c["repole"]:7.2f}')
        print()

    print('--- C2 transients (the cost of throwing the covariance away) ---')
    for label, _, _, kind, rh in arms:
        if rh <= 0:
            continue
        c = res[label][hours[-1]]
        print(f'  {label:44s} n_reinit {c["n_reinit"]:2d}  '
              f'median position-error jump at re-init {c["reinit_spike"]:+9.1f} m  '
              f'sigma_pos inflation {c["reinit_sig_jump"]:6.2f}x')
    print()

    print('--- tick cost (predict() only; the sole part any candidate changes) ---')
    base = None
    for kind in ('2body', 'j2', 'j2cov_an', 'j2cov_fd', 'j2struct'):
        ms = tick_cost(kind, args.n, sb, qa)
        if base is None:
            base = ms
        print(f'  {kind:12s} {ms:8.3f} ms/tick   {ms/base:6.2f}x the two-body path')
    print()

    print('=== 24 h SUMMARY (the operating point) ===')
    h = hours[-1]
    ctl = res[arms[0][0]][h]
    print(f'  {"arm":44s} {"pos":>10s} {"vs ctl":>8s} {"NEES":>10s} {"in band":>8s}')
    for label, _, _, _, _ in arms:
        c = res[label][h]
        print(f'  {label:44s} {np.median(c["perr"]):9.1f}m '
              f'{np.median(c["perr"])/np.median(ctl["perr"]):7.2f}x '
              f'{np.median(c["nees"]):9.2f} {1.0 - c["nees_bad"]:7.1%}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
