#!/usr/bin/env python3
"""N3D-A §1 (cost) — what does a 6-state 3D bearings-only filter cost at 1024
envs, measured against the shipped 4-state modified-polar filter on the same
machine in the same process?

The shipped ext-nav wrapper runs `nav_math.BatchedBearingMPC` (4-state, 9
batched propagations per predict) at a measured 1.35-2x the range+bearing EKF
tick.  The 3D lift is 6-state: 13 batched propagations per predict, 6x6 matrix
chains instead of 4x4, and a 2x2 (instead of 1x1) innovation solve.  Naive
scaling says 13/9 x (6/4)^2 ~ 3.25x.  This measures it.

Everything here is a batched numpy port of the scalar prototype in
`n3d_filter.py`; the scalar version stays the reference.

Run:  python3 n3d_cost.py [--batch 1024] [--reps 20]
Output: web_data/results/n3d_cost.csv
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
sys.path.insert(0, "/Users/pete/space_training/pufferlib")
from n3d_core import (MU, R_EARTH, SIGMA_BETA_RAD, Q_ACCEL_PSD_BO,  # noqa: E402
                      coe2rv, msc_decode, msc_encode, _msc_dy, pin_blas_threads,
                      pole_frame, process_noise6, propagate3, stm_fd3, wrap_pi)

OUT = "/Users/pete/space_training/web_data/results/n3d_cost.csv"


class BatchedMSC6:
    """Batched 3D modified-spherical bearings-only filter (N rows at once).

    Shapes: y (N,6), Py (N,6,6), sat (N,6).  Frame W is (N,3,3) — one pole
    frame per env, since each env has its own chaser orbit.
    """

    def __init__(self, n, sigma_beta=SIGMA_BETA_RAD, q_a=Q_ACCEL_PSD_BO):
        self.n = n
        self.sb = sigma_beta
        self.q_a = q_a
        self.y = np.zeros((n, 6))
        self.Py = np.tile(np.eye(6), (n, 1, 1))
        self.sat = np.zeros((n, 6))
        self.W = np.tile(np.eye(3), (n, 1, 1))
        self._I6 = np.eye(6)

    # frame-aware encode/decode (per-row W)
    @staticmethod
    def _enc(sat, tgt, W):
        d = np.einsum('nij,nj->ni', W, tgt[:, :3] - sat[:, :3])
        dd = np.einsum('nij,nj->ni', W, tgt[:, 3:] - sat[:, 3:])
        rho2 = np.maximum(np.sum(d * d, axis=1), 1.0)
        rho = np.sqrt(rho2)
        x, y, z = d[:, 0], d[:, 1], d[:, 2]
        vx, vy, vz = dd[:, 0], dd[:, 1], dd[:, 2]
        rxy2 = np.maximum(x * x + y * y, 1e-30)
        az = np.arctan2(y, x)
        el = np.arcsin(np.clip(z / rho, -1.0, 1.0))
        s = (x * vx + y * vy + z * vz) / rho2
        azdot = (x * vy - y * vx) / rxy2
        ce = np.maximum(np.cos(el), 1e-12)
        eldot = (vz / rho - (z / rho) * s) / ce
        return np.stack([az, el, azdot * ce, eldot, s, np.log(rho)], axis=1)

    @staticmethod
    def _dec(y, sat, W):
        az, el, wa, we, s, L = (y[:, 0], y[:, 1], y[:, 2], y[:, 3],
                                y[:, 4], y[:, 5])
        rho = np.maximum(np.exp(np.minimum(L, 25.0)), 1.0)
        ca, sa, ce, se = np.cos(az), np.sin(az), np.cos(el), np.sin(el)
        u = np.stack([ce * ca, ce * sa, se], axis=1)
        e_az = np.stack([-sa, ca, np.zeros_like(sa)], axis=1)
        e_el = np.stack([-se * ca, -se * sa, ce], axis=1)
        d = rho[:, None] * u
        dd = (s * rho)[:, None] * u + rho[:, None] * (wa[:, None] * e_az
                                                      + we[:, None] * e_el)
        R = sat[:, :3] + np.einsum('nji,nj->ni', W, d)
        V = sat[:, 3:] + np.einsum('nji,nj->ni', W, dd)
        return np.concatenate([R, V], axis=1)

    def _enc_jac(self, sat, xt, W):
        n = xt.shape[0]
        G = np.empty((n, 6, 6))
        h = (1.0, 1.0, 1.0, 1e-3, 1e-3, 1e-3)
        for j in range(6):
            xp = xt.copy(); xp[:, j] += h[j]
            xm = xt.copy(); xm[:, j] -= h[j]
            G[:, :, j] = _msc_dy(self._enc(sat, xp, W),
                                 self._enc(sat, xm, W)) / (2.0 * h[j])
        return G

    def predict(self, dt, sat_from, sat_to):
        """13 batched propagations (1 nominal + 12 central differences)."""
        n = self.n
        y = self.y
        rho = np.maximum(np.exp(np.minimum(y[:, 5], 25.0)), 1.0)
        h = np.stack([1.0 / rho, 1.0 / rho, 1e-3 / rho,
                      1e-3 / rho, 1e-3 / rho, 1.0 / rho], axis=1)
        x0 = self._dec(y, sat_from, self.W)
        X1, ok, dE = propagate3(x0, dt, want_dE=True)
        y0 = self._enc(sat_to, X1, self.W)
        Y2 = np.empty((2 * n, 6))
        W2 = np.concatenate([self.W, self.W])
        sf2 = np.concatenate([sat_from, sat_from])
        st2 = np.concatenate([sat_to, sat_to])
        dE2 = np.concatenate([dE, dE])
        F = np.empty((n, 6, 6))
        for j in range(6):
            Y2[:n] = y; Y2[:n, j] += h[:, j]
            Y2[n:] = y; Y2[n:, j] -= h[:, j]
            Xp = self._dec(Y2, sf2, W2)
            Xq = propagate3(Xp, dt, iters=2, dE0=dE2)[0]
            Yq = self._enc(st2, Xq, W2)
            F[:, :, j] = _msc_dy(Yq[:n], Yq[n:]) / (2.0 * h[:, j:j + 1])
        xt = self._dec(y0, sat_to, self.W)
        G = self._enc_jac(sat_to, xt, self.W)
        Qy = G @ process_noise6(dt, self.q_a) @ np.swapaxes(G, 1, 2)
        Py = F @ self.Py @ np.swapaxes(F, 1, 2) + Qy
        self.y = y0
        self.Py = 0.5 * (Py + np.swapaxes(Py, 1, 2))
        self.sat = sat_to
        return ok

    def update(self, sat, az, el):
        """Linear update, H = [e1; e2] -> a 2x2 solve per row."""
        self.sat = sat
        y, Py = self.y, self.Py
        nu = np.stack([wrap_pi(az - y[:, 0]), wrap_pi(el - y[:, 1])], axis=1)
        ce = np.maximum(np.cos(y[:, 1]), 1e-6)
        R = np.zeros((self.n, 2, 2))
        R[:, 0, 0] = (self.sb / ce) ** 2
        R[:, 1, 1] = self.sb ** 2
        S = Py[:, :2, :2] + R
        det = S[:, 0, 0] * S[:, 1, 1] - S[:, 0, 1] * S[:, 1, 0]
        det = np.where(np.abs(det) < 1e-300, 1e-300, det)
        Si = np.empty_like(S)
        Si[:, 0, 0] = S[:, 1, 1] / det
        Si[:, 1, 1] = S[:, 0, 0] / det
        Si[:, 0, 1] = -S[:, 0, 1] / det
        Si[:, 1, 0] = -S[:, 1, 0] / det
        K = Py[:, :, :2] @ Si                                  # (n,6,2)
        self.y = y + np.einsum('nij,nj->ni', K, nu)
        self.y[:, 0] = wrap_pi(self.y[:, 0])
        self.y[:, 1] = wrap_pi(self.y[:, 1])
        IKH = np.tile(self._I6, (self.n, 1, 1))
        IKH[:, :, :2] -= K
        Pn = (IKH @ Py @ np.swapaxes(IKH, 1, 2)
              + K @ R @ np.swapaxes(K, 1, 2))
        self.Py = 0.5 * (Pn + np.swapaxes(Pn, 1, 2))


def make_batch(n, seed=5):
    rng = np.random.default_rng(seed)
    a = R_EARTH + rng.uniform(400e3, 800e3, n)
    S = np.empty((n, 6))
    G = np.empty((n, 6))
    W = np.empty((n, 3, 3))
    for i in range(n):
        S[i] = coe2rv(a[i], 0.001, rng.uniform(0.0, 0.02),
                      rng.uniform(0, 6.28), 0.0, rng.uniform(-3.14, 3.14))
        G[i] = coe2rv(a[i] - 2e3, 0.001, rng.uniform(0.0, 0.02),
                      rng.uniform(0, 6.28), 0.0, rng.uniform(-3.14, 3.14))
        W[i] = pole_frame(S[i])
    return S, G, W


def timeit(fn, reps):
    fn()
    t = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        t.append(time.perf_counter() - t0)
    return float(np.median(t)) * 1e3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--batch', type=int, default=1024)
    ap.add_argument('--reps', type=int, default=20)
    a = ap.parse_args()
    pin_blas_threads()
    n, dt = a.batch, 60.0
    S, G, W = make_batch(n)
    S2 = propagate3(S, dt)[0]
    rows = []

    def rec(op, ms, note=''):
        rows.append(dict(op=op, batch=n, ms=ms, us_per_env=ms * 1e3 / n,
                         ticks_per_s=1e3 / ms, note=note))
        print(f"  {op:44s} {ms:8.3f} ms  {ms*1e3/n:7.3f} us/env  {note}")

    print(f"3D 6-state (this work), B={n}")
    rec('propagate3 f&g (6 Newton iters)', timeit(lambda: propagate3(S, dt), a.reps))
    rec('stm_fd3 (12N+1 propagate)', timeit(lambda: stm_fd3(S, dt), a.reps))
    f6 = BatchedMSC6(n)
    f6.W = W
    f6.sat = S
    f6.y = BatchedMSC6._enc(S, G, W)
    f6.Py = np.tile(np.eye(6) * 1e2, (n, 1, 1))
    az = f6.y[:, 0] + 1e-3
    el = f6.y[:, 1] + 1e-3
    rec('MSC6 predict (13 prop + FPF^T + Q)',
        timeit(lambda: f6.predict(dt, S, S2), a.reps))
    rec('MSC6 update (2x2 solve + Joseph)',
        timeit(lambda: f6.update(S, az, el), a.reps))
    tick6 = timeit(lambda: (f6.predict(dt, S, S2), f6.update(S, az, el)), a.reps)
    rec('FULL 3D sensor tick (predict+update)', tick6)

    print(f"\nshipped 2D 4-state reference (nav_math), B={n}")
    try:
        from pufferlib.ocean.orbital_nav import nav_math as nm
        S4 = np.stack([S[:, 0], S[:, 1], S[:, 3], S[:, 4]], axis=1)
        G4 = np.stack([G[:, 0], G[:, 1], G[:, 3], G[:, 4]], axis=1)
        S4b = nm.propagate_cartesian(S4, dt)[0]
        idx = np.arange(n)
        rec('nav_math propagate_cartesian (2D)',
            timeit(lambda: nm.propagate_cartesian(S4, dt), a.reps))
        rec('nav_math stm_fd (2D, 8N+1)', timeit(lambda: nm.stm_fd(S4, dt), a.reps))
        f4 = nm.BatchedBearingMPC(n)
        f4.set_polar(idx, nm.msc_encode(S4, G4),
                     np.tile(np.eye(4) * 1e2, (n, 1, 1)), S4)
        b4 = nm.msc_encode(S4, G4)[:, 0] + 1e-3
        rec('BatchedBearingMPC predict (2D, 9 prop)',
            timeit(lambda: f4.predict(idx, dt, S4, S4b), a.reps))
        rec('BatchedBearingMPC update (2D)',
            timeit(lambda: f4.update(idx, S4, b4), a.reps))
        tick4 = timeit(lambda: (f4.predict(idx, dt, S4, S4b),
                                f4.update(idx, S4, b4)), a.reps)
        rec('FULL 2D sensor tick (predict+update)', tick4)
        ekf = nm.BatchedRangeBearingEKF(n)
        ekf.set_state(idx, G4, np.tile(np.eye(4) * 1e2, (n, 1, 1)))
        rho = np.linalg.norm(G4[:, :2] - S4[:, :2], axis=1)
        beta = np.arctan2(G4[:, 1] - S4[:, 1], G4[:, 0] - S4[:, 0])
        tickrb = timeit(lambda: (ekf.predict(idx, dt),
                                 ekf.update(idx, S4, rho, beta)), a.reps)
        rec('FULL 2D range+bearing EKF tick', tickrb)
        print(f"\n  3D/2D-MPC tick ratio      {tick6/tick4:6.2f}x")
        print(f"  3D/2D-RB-EKF tick ratio   {tick6/tickrb:6.2f}x")
        rows.append(dict(op='ratio_3d_over_2d_mpc', batch=n, ms=tick6 / tick4,
                         us_per_env=float('nan'), ticks_per_s=float('nan'),
                         note='dimensionless'))
        rows.append(dict(op='ratio_3d_over_2d_rbekf', batch=n,
                         ms=tick6 / tickrb, us_per_env=float('nan'),
                         ticks_per_s=float('nan'), note='dimensionless'))
    except Exception as exc:                                  # pragma: no cover
        print(f"  (2D reference unavailable: {exc})")

    with open(OUT, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {OUT}")


if __name__ == '__main__':
    main()
