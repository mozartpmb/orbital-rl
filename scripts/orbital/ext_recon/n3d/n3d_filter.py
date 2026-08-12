#!/usr/bin/env python3
"""N3D-A §1/§3/§4 — 3D bearings-only relative-navigation filter PROTOTYPE.

The go/no-go datum for a combined 3D + angles-only ("3D-nav") capability.
Mirrors `ext_bo_filter.py` (NAV-G, 2D) arm for arm so the tables are directly
comparable, and reuses its architecture verbatim where the lift is trivial.

Arms
----
  RB3-EKF        range + (az, el), Cartesian 6-state EKF          [baseline]
  BO3-EKF        bearings-only Cartesian 6-state EKF, blind init  [pathology]
  BO3-MSC-o      modified-spherical 6-state, ORACLE epoch range   [steady state]
  BO3-BLS-MSC    RECOMMENDED. Plane-seeded angles-only batch acquisition handed
                 to the modified-spherical recursive filter.  The acquisition
                 lattice is the SHIPPED 2D lattice (range x 3 tangential x 3
                 radial) with the out-of-plane velocity seeded at zero; the
                 out-of-plane POSITION comes free from the measured elevation
                 because the seed sits exactly on the LOS ray.
  BO3-BLS-FULL   same, but with the out-of-plane velocity lattice expanded 3x
                 (27 nodes/range instead of 9).  Cost control: does the cheap
                 route lose anything?

State parameterisation, y = [az, el, w_a, w_e, rho_dot/rho, ln rho]
    * measurement is EXACTLY (y0, y1) -> H = [e1; e2], zero linearisation error
      in the update, which is what kills premature covariance collapse;
    * the weak direction (range) is isolated in a single component, ln rho;
    * at zero relative inclination el == w_e == 0 identically and the filter
      collapses onto the shipped 4-state modified-polar filter.
Rejected alternatives are argued in the memo (Cartesian-6: measured here;
ROE/relative elements: linearisation is invalid at rho/r ~ 2 and e <= 0.30).

Run:  python3 n3d_filter.py [--quick] [--seeds N] [--scenarios N1,N3]
Outputs: web_data/results/n3d_filter.csv, n3d_filter_conv.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from n3d_core import (MU, R_EARTH, SIGMA_BETA_RAD, SIGMA_RHO_M,  # noqa: E402
                      Q_ACCEL_PSD_BO, NEES6_LO, NEES6_HI, azel_jac,
                      elements_of, los_azel, make_geometry, msc_decode,
                      msc_enc_jac, msc_encode, _msc_dy, period,
                      pin_blas_threads, pole_frame, process_noise6, propagate3,
                      roll_truth, stm_fd3, wrap_pi)

OUT = "/Users/pete/space_training/web_data/results/n3d_filter.csv"
OUT_CONV = "/Users/pete/space_training/web_data/results/n3d_filter_conv.csv"

R_LO, R_HI = 0.9 * R_EARTH, 60.0 * R_EARTH
SETTLE_FRAC = 0.75
A_LEO = R_EARTH + 400e3
P_LEO = period(A_LEO)
_RHO_FLOOR = 1.0


# ── measurement ─────────────────────────────────────────────────────────────
def observe(S, G, W, rng, sigma_beta=SIGMA_BETA_RAD):
    """Noisy (az, el) with ISOTROPIC angular noise on the sphere.

    An angular error of sigma_beta in a uniformly random direction perturbs
    elevation by sigma_beta and azimuth by sigma_beta/cos(el) — the same
    convention the CRLB uses, so filter and bound are like for like.
    """
    az, el = los_azel(S, G, W)
    n = len(az)
    e_n = rng.normal(0.0, sigma_beta, n)
    a_n = rng.normal(0.0, sigma_beta, n) / np.maximum(np.cos(el), 1e-6)
    return wrap_pi(az + a_n), wrap_pi(el + e_n)


def range_prior_intervals(rc, u, r_min, r_max, rho_floor=100.0):
    """Feasible range set for a target constrained to the shell [r_min, r_max].

    DIMENSION-AGNOSTIC: |r_c + rho*u|^2 = R^2 is the same scalar quadratic in
    3D as in 2D (b0 = r_c.u, disc = b0^2 - |r_c|^2 + R^2), so the shipped 2D
    analytic prior lifts verbatim — annulus becomes spherical shell, ray stays
    a ray.  Still bimodal whenever the ray pierces the inner sphere.
    """
    rc = np.asarray(rc, dtype=np.float64)[:3]
    u = np.asarray(u, dtype=np.float64)[:3]
    rc2 = float(rc @ rc)
    b0 = float(rc @ u)
    disc_out = b0 * b0 - rc2 + r_max * r_max
    if disc_out <= 0.0:
        return [(rho_floor, rho_floor * 10.0)]
    rho_out = -b0 + math.sqrt(disc_out)
    if rho_out <= rho_floor:
        return [(rho_floor, max(2.0 * rho_floor, rho_out))]
    disc_in = b0 * b0 - rc2 + r_min * r_min
    if disc_in <= 0.0:
        return [(rho_floor, rho_out)]
    sq = math.sqrt(disc_in)
    lo_in, hi_in = -b0 - sq, -b0 + sq
    out = []
    if lo_in > rho_floor:
        out.append((rho_floor, min(lo_in, rho_out)))
    if hi_in < rho_out:
        out.append((max(hi_in, rho_floor), rho_out))
    return out or [(rho_floor, rho_out)]


def los_unit(sat, az, el, W):
    ca, sa, ce, se = math.cos(az), math.sin(az), math.cos(el), math.sin(el)
    return np.array([ce * ca, ce * sa, se]) @ W


# ── Cartesian EKFs ──────────────────────────────────────────────────────────
class Cart6EKF:
    """6-state Cartesian EKF on the target's absolute state.

    `with_range=True` adds the range row (the RB3 baseline); otherwise the
    measurement is (az, el) only.
    """

    def __init__(self, W, with_range=False, sigma_beta=SIGMA_BETA_RAD,
                 sigma_rho=SIGMA_RHO_M, q_a=Q_ACCEL_PSD_BO):
        self.W = W
        self.with_range = with_range
        self.sb = sigma_beta
        self.sr = sigma_rho
        self.q_a = q_a
        self.x = None
        self.P = None
        self.alive = True
        self.name = "RB3-EKF" if with_range else "BO3-EKF"

    def set(self, x, P):
        self.x = np.asarray(x, dtype=float).reshape(6)
        self.P = np.asarray(P, dtype=float).reshape(6, 6)
        return self

    def predict(self, dt):
        if not self.alive:
            return
        F, ok, Y = stm_fd3(self.x[None, :], dt, iters=12, warm_iters=6)
        if not ok[0]:
            self.alive = False
            return
        self.x = Y[0]
        P = F[0] @ self.P @ F[0].T + process_noise6(dt, self.q_a)
        self.P = 0.5 * (P + P.T)
        if not np.all(np.isfinite(self.x)) or not np.all(np.isfinite(self.P)):
            self.alive = False

    def update(self, sat, az, el, rho=None):
        if not self.alive:
            return
        H = azel_jac(sat[None, :], self.x[None, :], self.W)[0]
        pa, pe = los_azel(sat, self.x, self.W)
        nu = np.array([wrap_pi(az - pa), wrap_pi(el - pe)])
        ce = max(math.cos(pe), 1e-6)
        R = np.diag([(self.sb / ce) ** 2, self.sb ** 2])
        if self.with_range:
            d = self.x[:3] - sat[:3]
            rr = max(float(np.linalg.norm(d)), _RHO_FLOOR)
            Hr = np.zeros((1, 6))
            Hr[0, :3] = d / rr
            H = np.vstack([H, Hr])
            nu = np.append(nu, rho - rr)
            R = np.diag([(self.sb / ce) ** 2, self.sb ** 2, self.sr ** 2])
        S = H @ self.P @ H.T + R
        try:
            K = self.P @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            self.alive = False
            return
        self.x = self.x + K @ nu
        IKH = np.eye(6) - K @ H
        P = IKH @ self.P @ IKH.T + K @ R @ K.T
        self.P = 0.5 * (P + P.T)

    def mean_cov(self):
        return self.x, self.P


class MSCFilter:
    """Bearings-only EKF in 3D modified spherical coordinates (the pole frame).

    Update is exactly linear (H = [e1; e2]); all nonlinearity lives in the
    transition, evaluated as encode o exact-two-body-flow o decode with a
    central-difference Jacobian.  13 batched propagations per predict
    (1 nominal + 12 perturbations) vs 9 for the shipped 4-state filter.
    """

    name = "BO3-MSC"

    def __init__(self, W, sigma_beta=SIGMA_BETA_RAD, q_a=Q_ACCEL_PSD_BO):
        self.W = W
        self.sb = sigma_beta
        self.q_a = q_a
        self.y = None
        self.Py = None
        self.sat = None
        self.alive = True

    def set_cart(self, x, P, sat):
        self.sat = np.asarray(sat, dtype=float).reshape(6)
        self.y = msc_encode(self.sat[None, :], np.asarray(x)[None, :], self.W)[0]
        G = msc_enc_jac(self.sat[None, :], np.asarray(x)[None, :], self.W)[0]
        Py = G @ np.asarray(P) @ G.T
        self.Py = 0.5 * (Py + Py.T)
        return self

    def _trans(self, y, sat_from, sat_to, dt, iters=12):
        xt = msc_decode(y, sat_from, self.W)
        out = propagate3(xt, dt, iters=iters)[0]
        return msc_encode(sat_to, out, self.W)

    def predict(self, dt, sat_from, sat_to):
        if not self.alive:
            return
        sf = np.atleast_2d(sat_from)
        st = np.atleast_2d(sat_to)
        y = self.y[None, :]
        rho = max(math.exp(min(self.y[5], 25.0)), _RHO_FLOOR)
        h = np.array([1.0, 1.0, 1e-3, 1e-3, 1e-3, 1.0]) / rho
        y0 = self._trans(y, sf, st, dt)[0]
        Y2 = np.repeat(y, 12, axis=0)
        for j in range(6):
            Y2[2 * j, j] += h[j]
            Y2[2 * j + 1, j] -= h[j]
        Wp = self._trans(Y2, np.repeat(sf, 12, axis=0),
                         np.repeat(st, 12, axis=0), dt)
        F = np.empty((6, 6))
        for j in range(6):
            F[:, j] = _msc_dy(Wp[2 * j].copy(), Wp[2 * j + 1]) / (2.0 * h[j])
        xt = msc_decode(y0[None, :], st, self.W)
        Gj = msc_enc_jac(st, xt, self.W)[0]
        Qy = Gj @ process_noise6(dt, self.q_a) @ Gj.T
        self.y = y0
        Py = F @ self.Py @ F.T + Qy
        self.Py = 0.5 * (Py + Py.T)
        self.sat = np.asarray(sat_to, dtype=float).reshape(6)
        if not np.all(np.isfinite(self.y)) or not np.all(np.isfinite(self.Py)):
            self.alive = False

    def update(self, sat, az, el):
        if not self.alive:
            return
        self.sat = np.asarray(sat, dtype=float).reshape(6)
        H = np.zeros((2, 6))
        H[0, 0] = H[1, 1] = 1.0
        nu = np.array([wrap_pi(az - self.y[0]), wrap_pi(el - self.y[1])])
        ce = max(math.cos(self.y[1]), 1e-6)
        R = np.diag([(self.sb / ce) ** 2, self.sb ** 2])
        S = self.Py[:2, :2] + R
        try:
            K = self.Py[:, :2] @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            self.alive = False
            return
        self.y = self.y + K @ nu
        self.y[0] = wrap_pi(self.y[0])
        self.y[1] = wrap_pi(self.y[1])
        IKH = np.eye(6) - K @ H
        Py = IKH @ self.Py @ IKH.T + K @ R @ K.T
        self.Py = 0.5 * (Py + Py.T)

    def mean_cov(self):
        sat = self.sat[None, :]
        x = msc_decode(self.y[None, :], sat, self.W)
        G = msc_enc_jac(sat, x, self.W)[0]
        try:
            Gi = np.linalg.inv(G)
        except np.linalg.LinAlgError:
            return x[0], np.eye(6) * 1e12
        P = Gi @ self.Py @ Gi.T
        return x[0], 0.5 * (P + P.T)


# ── angles-only batch acquisition (3D) ──────────────────────────────────────
def _cost_batch(S, AZ, EL, X0, W, sigma_beta, dt):
    """Sum of squared normalised (az, el) residuals for M candidates at once.

    Forward propagation only (no STM), so this is ~13x cheaper than a
    Gauss-Newton iteration and can be run on the whole grid.
    """
    X = np.array(X0, dtype=float)
    M = X.shape[0]
    cost = np.zeros(M)
    bad = np.zeros(M, dtype=bool)
    for k in range(len(S)):
        if k > 0:
            X, ok = propagate3(X, dt, iters=8)
            bad |= ~ok
        r = np.linalg.norm(X[:, :3], axis=1)
        bad |= (r < R_LO) | (r > R_HI) | ~np.isfinite(r)
        az, el = los_azel(np.repeat(S[k][None, :], M, axis=0), X, W)
        ce = np.maximum(np.cos(el), 1e-6)
        ra = wrap_pi(AZ[k] - az) * ce
        re = wrap_pi(EL[k] - el)
        cost += (ra * ra + re * re) / sigma_beta ** 2
    cost[bad] = np.inf
    return cost


def _gn_batch(S, AZ, EL, X0, W, sigma_beta, dt, iters=12, lam0=1e-2):
    """Levenberg-damped Gauss-Newton on the target epoch 6-state, M starts.

    Batched over starts; per-start damping, accept/reject and trust region are
    applied with masks so no start's control flow can stall the others.
    Returns (X, cost, N) with N the information matrices.
    """
    M = X0.shape[0]
    X = np.array(X0, dtype=float)
    lam = np.full(M, lam0)
    bestX = X.copy()
    bestC = np.full(M, np.inf)
    bestN = np.tile(np.eye(6), (M, 1, 1))
    n = len(S)
    for _ in range(iters):
        Phi = np.tile(np.eye(6), (M, 1, 1))
        Xt = X.copy()
        A = np.zeros((M, 2 * n, 6))
        b = np.zeros((M, 2 * n))
        bad = np.zeros(M, dtype=bool)
        for k in range(n):
            if k > 0:
                Fk, ok, Xt = stm_fd3(Xt, dt, iters=8, warm_iters=4)
                Phi = Fk @ Phi
                bad |= ~ok
            rr = np.linalg.norm(Xt[:, :3], axis=1)
            bad |= (rr < R_LO) | (rr > R_HI) | ~np.isfinite(rr)
            Sk = np.repeat(S[k][None, :], M, axis=0)
            az, el = los_azel(Sk, Xt, W)
            ce = np.maximum(np.cos(el), 1e-6)
            H = azel_jac(Sk, Xt, W) @ Phi                     # (M,2,6)
            H[:, 0, :] *= ce[:, None]
            A[:, 2 * k, :] = H[:, 0, :] / sigma_beta
            A[:, 2 * k + 1, :] = H[:, 1, :] / sigma_beta
            b[:, 2 * k] = wrap_pi(AZ[k] - az) * ce / sigma_beta
            b[:, 2 * k + 1] = wrap_pi(EL[k] - el) / sigma_beta
        cost = np.einsum('mi,mi->m', b, b)
        cost[bad] = np.inf
        N = np.swapaxes(A, 1, 2) @ A
        g = np.einsum('mij,mi->mj', A, b)
        imp = cost < bestC
        bestX[imp] = X[imp]
        bestC[imp] = cost[imp]
        bestN[imp] = N[imp]
        lam = np.where(imp, np.maximum(lam * 0.3, 1e-8),
                       np.minimum(lam * 10.0, 1e12))
        # Rejected rows retreat to their best iterate and re-damp WITHOUT
        # taking the stale step computed at the rejected point (the batched
        # equivalent of the scalar version's `continue`).
        X = np.where(imp[:, None], X, bestX)
        D = np.zeros_like(N)
        di = np.arange(6)
        D[:, di, di] = np.maximum(N[:, di, di], 1e-12)
        try:
            step = np.linalg.solve(N + lam[:, None, None] * D
                                   + 1e-9 * np.eye(6), g[:, :, None])[:, :, 0]
        except np.linalg.LinAlgError:
            break
        step = np.where(np.isfinite(step), step, 0.0)
        # trust region: never move epoch position by > half the separation,
        # nor velocity by > half the local circular speed
        rho_now = np.linalg.norm(X[:, :3] - S[0][None, :3], axis=1)
        sp = np.linalg.norm(step[:, :3], axis=1)
        f = np.minimum(1.0, 0.5 * rho_now / np.maximum(sp, 1e-30))
        step[:, :3] *= f[:, None]
        vc = np.sqrt(MU / np.maximum(np.linalg.norm(X[:, :3], axis=1), R_LO))
        sv = np.linalg.norm(step[:, 3:], axis=1)
        fv = np.minimum(1.0, 0.5 * vc / np.maximum(sv, 1e-30))
        step[:, 3:] *= fv[:, None]
        X = X + step * imp[:, None]
    return bestX, bestC, bestN


def _seed_states(sat, u, rhos, v_lat, W):
    """Candidate epoch states on the LOS ray at ranges `rhos`.

    Position comes FREE and exactly from the measured (az, el) — the seed sits
    on the ray, so the out-of-plane POSITION carries the elevation information
    at zero search cost.  Only the velocity needs a lattice, and it is built in
    the chaser's own orbital triad (tangential, radial, normal) because the
    envelope caps the relative inclination at ~1 deg, i.e. |v_normal| <=
    0.018 v_c.
    """
    P = sat[None, :3] + rhos[:, None] * u[None, :]
    r = np.linalg.norm(P, axis=1)
    vc = np.sqrt(MU / np.maximum(r, R_LO))
    rad = P / r[:, None]
    nor = np.repeat(W[2][None, :], len(rhos), axis=0)          # chaser normal
    tan = np.cross(nor, rad)
    tan /= np.linalg.norm(tan, axis=1)[:, None]
    out = []
    for ft, fr, fn in v_lat:
        V = (ft * vc)[:, None] * tan + (fr * vc)[:, None] * rad \
            + (fn * vc)[:, None] * nor
        out.append(np.concatenate([P, V], axis=1))
    return np.concatenate(out, axis=0)


V_LAT_2D = [(ft, fr, 0.0) for ft in (0.80, 1.00, 1.20)
            for fr in (-0.30, 0.0, 0.30)]
V_LAT_3D = [(ft, fr, fn) for ft in (0.80, 1.00, 1.20)
            for fr in (-0.30, 0.0, 0.30) for fn in (-0.02, 0.0, 0.02)]


def bls_acquire(times, S, AZ, EL, W, r_min, r_max, sigma_beta, dt,
                window=45, grid_ratio=1.15, grid_max=160, n_bins=8,
                iters=12, cov_inflate=4.0, v_lat=None, growth=1.6,
                max_windows=6):
    """3D angles-only acquisition. Returns (x0, P0, k_acq, diag) or (None,...).

    Same three gates as NAV-G: chi-square on the residual, an AMBIGUITY MARGIN
    (best cost at a range differing by >1.2x must be >= 16 worse), and
    sigma_LOS/rho <= 0.20.  Arc grown x1.6 until all pass.
    """
    v_lat = V_LAT_2D if v_lat is None else v_lat
    n = len(times)
    w = min(window, n)
    stats = dict(windows=0, nodes=0, props=0)
    prev = None                       # warm start carried across window growth
    for _ in range(max_windows):
        stats['windows'] += 1
        u0 = los_unit(S[0], AZ[0], EL[0], W)
        ivs = range_prior_intervals(S[0], u0, r_min, r_max)
        spans = [(lo, hi, math.log(hi / lo)) for lo, hi in ivs if hi > lo]
        tot = sum(s[2] for s in spans) or 1.0
        rhos = []
        for lo, hi, sp in spans:
            m = max(4, min(grid_max, int(round(sp / math.log(grid_ratio)))))
            m = max(4, int(round(grid_max * sp / tot)))
            rhos.append(np.exp(np.linspace(math.log(lo), math.log(hi), m)))
        rhos = np.concatenate(rhos)
        X0 = _seed_states(S[0], u0, rhos, v_lat, W)
        stats['nodes'] += X0.shape[0]
        c = _cost_batch(S[:w], AZ[:w], EL[:w], X0, W, sigma_beta, dt)
        stats['props'] += X0.shape[0] * w
        rr = np.tile(rhos, len(v_lat))
        # binned multi-start: cheapest node in each of n_bins equal log-range
        # bins (a global top-k systematically misses the true basin — NAV-G §3)
        lr = np.log(np.maximum(rr, 1.0))
        edges = np.linspace(lr.min(), lr.max() + 1e-12, n_bins + 1)
        starts = []
        for i in range(n_bins):
            m = (lr >= edges[i]) & (lr < edges[i + 1]) & np.isfinite(c)
            if m.any():
                starts.append(int(np.flatnonzero(m)[np.argmin(c[m])]))
        if not starts and prev is None:
            w = min(n, int(round(w * growth)))
            continue
        Xstart = X0[starts] if starts else np.zeros((0, 6))
        if prev is not None:
            Xstart = np.vstack([Xstart, prev[None, :]])
        Xs, Cs, Ns = _gn_batch(S[:w], AZ[:w], EL[:w], Xstart, W,
                               sigma_beta, dt, iters=iters)
        order = np.argsort(Cs)
        i0 = int(order[0])
        best = Xs[i0]
        rho_b = float(np.linalg.norm(best[:3] - S[0][:3]))
        dof = 2 * w - 6
        chi_ok = Cs[i0] <= dof + 3.0 * math.sqrt(2.0 * dof)
        # ambiguity margin
        amb = np.inf
        for j in order[1:]:
            rj = float(np.linalg.norm(Xs[j][:3] - S[0][:3]))
            if rj > 0 and (rj / rho_b > 1.2 or rho_b / rj > 1.2):
                amb = Cs[j] - Cs[i0]
                break
        amb_ok = amb >= 16.0
        try:
            P = cov_inflate * np.linalg.inv(Ns[i0] + 1e-12 * np.eye(6))
            u = (best[:3] - S[0][:3]) / max(rho_b, 1.0)
            sig_los = math.sqrt(max(u @ P[:3, :3] @ u, 0.0))
        except np.linalg.LinAlgError:
            P, sig_los = None, np.inf
        cov_ok = np.isfinite(sig_los) and sig_los / max(rho_b, 1.0) <= 0.20
        if np.isfinite(Cs[i0]):
            prev = best.copy()
        if chi_ok and amb_ok and cov_ok and P is not None:
            stats.update(chi2=float(Cs[i0]), dof=dof, amb=float(amb),
                         sig_los=float(sig_los), rho=rho_b, w=w)
            return best, P, w - 1, stats
        if w >= n:
            break
        w = min(n, max(w + 1, int(round(w * growth))))
    stats.update(chi2=float('nan'), dof=0, amb=float('nan'),
                 sig_los=float('nan'), rho=float('nan'), w=w)
    return None, None, None, stats


# ── scenarios ───────────────────────────────────────────────────────────────
def scenarios():
    """Five geometries. N0 is the coplanar CONTROL (exact 2D degeneracy);
    di is set to the largest value the separation can physically carry, capped
    at the ext-3d V-ladder envelope (<= 1 deg)."""
    def cap(rho, di_deg):
        return min(A_LEO * math.sin(math.radians(di_deg)) / rho, 0.95)
    return [
        dict(name='N0_10km_coplanar', rho=10e3, da=2e3, orbits=3.0, di=0.0,
             dtheta=None, burns=(), sig_v=100.0,
             r_min=R_EARTH + 300e3, r_max=R_EARTH + 800e3),
        dict(name='N1_10km_di0p04', rho=10e3, da=2e3, orbits=3.0, di=0.04,
             dtheta=None, burns=(), sig_v=100.0,
             r_min=R_EARTH + 300e3, r_max=R_EARTH + 800e3),
        dict(name='N2_5km_burns_di0p02', rho=5e3, da=400.0, orbits=2.0,
             di=0.02, dtheta=None, burns=((20, 1.0), (60, -1.0)), sig_v=100.0,
             r_min=R_EARTH + 300e3, r_max=R_EARTH + 800e3),
        # paired coplanar control at the tight box: isolates the 3D geometry
        # from every other difference (same da, same burn schedule, same arc)
        dict(name='N2b_5km_burns_coplanar', rho=5e3, da=400.0, orbits=2.0,
             di=0.0, dtheta=None, burns=((20, 1.0), (60, -1.0)), sig_v=100.0,
             r_min=R_EARTH + 300e3, r_max=R_EARTH + 800e3),
        dict(name='N3_180deg_di0p75', rho=13.5e6, da=30e3, orbits=1.5,
             di=0.75, dtheta=math.pi, burns=(), sig_v=200.0,
             r_min=R_EARTH + 300e3, r_max=R_EARTH + 800e3),
        dict(name='N3b_180deg_coplanar', rho=13.5e6, da=30e3, orbits=1.5,
             di=0.0, dtheta=math.pi, burns=(), sig_v=200.0,
             r_min=R_EARTH + 300e3, r_max=R_EARTH + 800e3),
        dict(name='N4_300km_di1p0', rho=300e3, da=60e3, orbits=2.0, di=1.0,
             dtheta=None, burns=(), sig_v=100.0,
             r_min=R_EARTH + 300e3, r_max=R_EARTH + 800e3),
        dict(name='N4b_300km_coplanar', rho=300e3, da=60e3, orbits=2.0, di=0.0,
             dtheta=None, burns=(), sig_v=100.0,
             r_min=R_EARTH + 300e3, r_max=R_EARTH + 800e3),
    ], cap


def build(sc, cap, dt=60.0):
    f_oop = cap(sc['rho'], sc['di']) if sc['di'] > 0 else 0.0
    n_steps = int(round(sc['orbits'] * P_LEO / dt))
    sat0, tgt0, di = make_geometry(A_LEO, 0.001, sc['da'], sc['rho'], f_oop,
                                   dtheta=sc['dtheta'])
    t, S, G = roll_truth(sat0, tgt0, dt, n_steps, sc['burns'])
    return t, S, G, di


# ── one arm, one geometry, one noise seed ───────────────────────────────────
def run_arm(kind, t, S, G, W, sc, rng, dt=60.0):
    n = len(t)
    AZ, EL = observe(S, G, W, rng)
    RHO = np.linalg.norm(G[:, :3] - S[:, :3], axis=1) \
        + rng.normal(0.0, SIGMA_RHO_M, n)
    k0 = 0
    diag = dict(acq_min=0.0, acq_fail=0, chi2=float('nan'),
                nodes=0, props=0, windows=0, acq_s=0.0,
                acq_err_m=float('nan'), acq_err_hand_m=float('nan'))

    if kind in ('BO3-BLS-MSC', 'BO3-BLS-FULL'):
        t0 = time.time()
        vl = V_LAT_3D if kind == 'BO3-BLS-FULL' else V_LAT_2D
        x0, P0, k0, st = bls_acquire(t, S, AZ, EL, W, sc['r_min'], sc['r_max'],
                                     SIGMA_BETA_RAD, dt, v_lat=vl)
        diag.update(acq_s=time.time() - t0, nodes=st['nodes'],
                    props=st['props'], windows=st['windows'],
                    chi2=st.get('chi2', float('nan')))
        if x0 is None:
            diag['acq_fail'] = 1
            return None, diag
        diag['acq_min'] = t[k0] / 60.0
        # The batch solves for the state at t0; advance it (and its covariance)
        # to the END of the acquisition window before the handoff. Skipping
        # this hands the recursive filter a k0-steps-stale mean with a tight
        # covariance -- a silent, catastrophic, gate-passing failure.
        diag['acq_err_m'] = float(np.linalg.norm(x0[:3] - G[0, :3]))
        xb, Pb = x0, P0
        for j in range(1, k0 + 1):
            Fk, ok, Yk = stm_fd3(xb[None, :], dt, iters=12, warm_iters=6)
            if not ok[0]:
                diag['acq_fail'] = 1
                return None, diag
            xb = Yk[0]
            Pb = Fk[0] @ Pb @ Fk[0].T + process_noise6(dt, Q_ACCEL_PSD_BO)
            Pb = 0.5 * (Pb + Pb.T)
        diag['acq_err_hand_m'] = float(np.linalg.norm(xb[:3] - G[k0, :3]))
        flt = MSCFilter(W).set_cart(xb, Pb, S[k0])
    elif kind == 'BO3-MSC-o':
        rho_t = float(np.linalg.norm(G[0, :3] - S[0, :3]))
        u0 = los_unit(S[0], AZ[0], EL[0], W)
        rho0 = rho_t * (1.0 + 0.30 * rng.normal())
        x0, P0 = _oracle_init(S[0], u0, abs(rho0), 0.30 * rho_t, sc['sig_v'], W)
        flt = MSCFilter(W).set_cart(x0, P0, S[0])
    else:
        u0 = los_unit(S[0], AZ[0], EL[0], W)
        ivs = range_prior_intervals(S[0], u0, sc['r_min'], sc['r_max'])
        lo = min(a for a, _ in ivs)
        hi = max(b for _, b in ivs)
        rho0 = math.sqrt(lo * hi)
        if kind == 'RB3-EKF':
            rho0 = RHO[0]
            sig = SIGMA_RHO_M
        else:
            sig = 0.5 * (hi - lo)
        x0, P0 = _oracle_init(S[0], u0, rho0, sig, sc['sig_v'], W)
        flt = Cart6EKF(W, with_range=(kind == 'RB3-EKF')).set(x0, P0)

    errs, nees, ok_n, trace = [], [], 0, []
    for k in range(k0, n):
        if k > k0:
            if isinstance(flt, MSCFilter):
                flt.predict(dt, S[k - 1][None, :], S[k][None, :])
            else:
                flt.predict(dt)
        if not flt.alive:
            return None, diag
        if isinstance(flt, MSCFilter):
            flt.update(S[k], AZ[k], EL[k])
        else:
            flt.update(S[k], AZ[k], EL[k], RHO[k])
        x, P = flt.mean_cov()
        e = x - G[k]
        pe = float(np.linalg.norm(e[:3]))
        ve = float(np.linalg.norm(e[3:]))
        errs.append((pe, ve))
        try:
            nv = float(e @ np.linalg.solve(P, e)) / 6.0
        except np.linalg.LinAlgError:
            nv = float('nan')
        nees.append(nv)
        ok_n += int(NEES6_LO <= nv <= NEES6_HI)
        trace.append((t[k] / 60.0, pe, ve, nv))
    if not errs:
        return None, diag
    errs = np.array(errs)
    m = int(SETTLE_FRAC * len(errs))
    x, P = flt.mean_cov()
    el_t = elements_of(G[-1][None, :])
    el_e = elements_of(x[None, :])
    plane_err = math.degrees(math.acos(
        min(1.0, abs(float(el_t['hhat'][0] @ el_e['hhat'][0])))))
    res = dict(pos_rmse=float(np.sqrt(np.mean(errs[m:, 0] ** 2))),
               vel_rmse=float(np.sqrt(np.mean(errs[m:, 1] ** 2))),
               pos_final=float(errs[-1, 0]), vel_final=float(errs[-1, 1]),
               nees=float(np.nanmedian(np.array(nees)[m:])),
               nees_in=ok_n / max(len(nees), 1),
               plane_err_deg=plane_err,
               di_err_deg=math.degrees(abs(el_t['inc'][0] - el_e['inc'][0])),
               a_err_m=float(abs(el_t['a'][0] - el_e['a'][0])),
               e_err=float(abs(el_t['e'][0] - el_e['e'][0])))
    return (res, trace), diag


def _oracle_init(sat, u, rho0, sig_rho, sig_v_ecc, W):
    p = sat[:3] + rho0 * u
    r = float(np.linalg.norm(p))
    vc = math.sqrt(MU / r)
    rad = p / r
    nor = W[2]
    tan = np.cross(nor, rad)
    tan /= np.linalg.norm(tan)
    v = vc * tan
    across = (rho0 * SIGMA_BETA_RAD) ** 2
    Ppos = (sig_rho ** 2) * np.outer(u, u) + across * (np.eye(3) - np.outer(u, u))
    sig_vv = math.hypot(sig_v_ecc, 0.5 * vc * abs(sig_rho * float(p @ u)) / r / r)
    P = np.zeros((6, 6))
    P[:3, :3] = Ppos
    P[3:, 3:] = np.eye(3) * sig_vv ** 2
    return np.concatenate([p, v]), P


ARMS = ['RB3-EKF', 'BO3-EKF', 'BO3-MSC-o', 'BO3-BLS-MSC', 'BO3-BLS-FULL']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--seeds', type=int, default=0)
    ap.add_argument('--scenarios', default='')
    ap.add_argument('--arms', default='')
    a = ap.parse_args()
    pin_blas_threads()
    nseed = a.seeds or (4 if a.quick else 12)
    scs, cap = scenarios()
    if a.scenarios:
        keep = set(a.scenarios.split(','))
        scs = [s for s in scs if s['name'].split('_')[0] in keep
               or s['name'] in keep]
    arms = a.arms.split(',') if a.arms else ARMS

    rows, conv = [], []
    for sc in scs:
        t, S, G, di = build(sc, cap)
        W = pole_frame(S[0])
        rho = np.linalg.norm(G[:, :3] - S[:, :3], axis=1)
        zs = np.abs(((G[:, :3] - S[:, :3]) @ W.T)[:, 2])
        print(f"\n=== {sc['name']}  di={math.degrees(di):.5f} deg  "
              f"rho0={rho[0]/1e3:.3f} km  rho_mean={rho.mean()/1e3:.3f} km  "
              f"z_max={zs.max()/1e3:.3f} km  arc={t[-1]/60:.0f} min  "
              f"n={len(t)}", flush=True)
        for kind in arms:
            acc, fails, accs, t0 = [], 0, [], time.time()
            for s in range(nseed):
                rng = np.random.default_rng(1000 + 97 * s)
                out, diag = run_arm(kind, t, S, G, W, sc, rng)
                if out is None:
                    fails += 1
                    continue
                res, trace = out
                acc.append(res)
                accs.append(diag)
                if s == 0:
                    for tr in trace[::5]:
                        conv.append(dict(scenario=sc['name'], arm=kind,
                                         t_min=tr[0], pos_err=tr[1],
                                         vel_err=tr[2], nees=tr[3]))
            if not acc:
                print(f"  {kind:14s} ALL FAILED ({fails}/{nseed})")
                rows.append(dict(scenario=sc['name'], arm=kind, seeds=nseed,
                                 fails=fails, di_deg=math.degrees(di),
                                 rho0_m=float(rho[0]),
                                 rho_mean_m=float(rho.mean()),
                                 zmax_m=float(zs.max()),
                                 arc_min=float(t[-1] / 60), n_obs=len(t)))
                continue
            med = lambda k: float(np.median([r[k] for r in acc]))  # noqa: E731
            r = dict(scenario=sc['name'], arm=kind, seeds=nseed, fails=fails,
                     di_deg=math.degrees(di), rho0_m=float(rho[0]),
                     rho_mean_m=float(rho.mean()), zmax_m=float(zs.max()),
                     arc_min=float(t[-1] / 60), n_obs=len(t),
                     pos_rmse_m=med('pos_rmse'), vel_rmse_ms=med('vel_rmse'),
                     pos_final_m=med('pos_final'),
                     vel_final_ms=med('vel_final'),
                     nees=med('nees'), nees_in=med('nees_in'),
                     plane_err_deg=med('plane_err_deg'),
                     inc_err_deg=med('di_err_deg'),
                     a_err_m=med('a_err_m'), e_err=med('e_err'),
                     acq_min=float(np.median([d['acq_min'] for d in accs])),
                     acq_fail=int(sum(d['acq_fail'] for d in accs)) + fails
                     if kind.startswith('BO3-BLS') else 0,
                     acq_s=float(np.median([d['acq_s'] for d in accs])),
                     acq_nodes=float(np.median([d['nodes'] for d in accs])),
                     acq_err_m=float(np.median([d['acq_err_m'] for d in accs])),
                     acq_err_hand_m=float(
                         np.median([d['acq_err_hand_m'] for d in accs])),
                     wall_s=round(time.time() - t0, 1))
            rows.append(r)
            print(f"  {kind:14s} pos {r['pos_rmse_m']:11.4g} m  "
                  f"vel {r['vel_rmse_ms']:10.4g} m/s  NEES {r['nees']:9.3g} "
                  f"(in {r['nees_in']:.2f})  plane {r['plane_err_deg']:9.4g} deg"
                  f"  acq {r['acq_min']:6.1f} min  fail {fails}  "
                  f"[{r['wall_s']}s]", flush=True)

    with open(OUT, 'w', newline='') as fh:
        keys = sorted({k for r in rows for k in r})
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    with open(OUT_CONV, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(conv[0].keys()))
        w.writeheader()
        w.writerows(conv)
    print(f"\nwrote {OUT} ({len(rows)} rows), {OUT_CONV} ({len(conv)} rows)")


if __name__ == '__main__':
    main()
