"""N3D-C probe 2 — the 6-state port of the nav filter kernel: correctness + cost.

READ-ONLY. Two questions:

  1. Does `nav_math.propagate_cartesian` generalise to 3-vectors by index change
     alone? Validated against the INDEPENDENT universal-variable oracle
     (`ext_recon/orbital_math3d.propagate_universal`) and against the C env's own
     Kepler propagation, over the dt values a warp sub-propagation actually uses.
  2. What does the 6-state filter kernel cost per tick at B=1024, against the
     shipped 4-state numbers in NAV-H section 2.2 (1.353 ms/tick)?

Output: web_data/results/n3d_prop6_validation.csv
        web_data/results/n3d_filter6_cost.csv
"""
import csv
import os
import sys
import time

import numpy as np

sys.path.insert(0, '/Users/pete/space_training/pufferlib')
sys.path.insert(0, '/Users/pete/space_training/scripts/orbital/ext_recon')
import orbital_math3d as o3  # noqa: E402
from pufferlib.ocean.orbital_nav import nav_math as nm  # noqa: E402

MU = 3.986004418e14
R_E = 6.371e6
SQ_MU = np.sqrt(MU)
OUT = '/Users/pete/space_training/web_data/results'
D = 3  # spatial dimension of the port


def propagate_cartesian_nd(X, dt, iters=6, dE0=None, want_dE=False, d=D):
    """`nav_math.propagate_cartesian`, dimension-generic.

    The ONLY changes vs the shipped 2D routine are (a) the position/velocity
    slices, (b) `hypot` -> `norm`. The Lagrange f&g coefficients are scalars
    that multiply the whole vector, and the eccentric-anomaly-difference Newton
    iteration is identical, so the 3D port introduces no new math to validate
    beyond the slicing.
    """
    X = np.asarray(X, dtype=np.float64)
    r0v, v0v = X[..., :d], X[..., d:]
    with np.errstate(all='ignore'):
        r0 = np.sqrt(np.sum(r0v * r0v, axis=-1))
        v2 = np.sum(v0v * v0v, axis=-1)
        inv_a = 2.0 / r0 - v2 / MU
        ok0 = np.isfinite(inv_a) & (inv_a > 1e-30) & (r0 > 1.0)
        inv_a = np.where(ok0, inv_a, 1.0 / 7.0e6)
        r0s = np.where(ok0, r0, 7.0e6)
        sig0 = np.where(ok0, np.sum(r0v * v0v, axis=-1) / SQ_MU, 0.0)

        a = 1.0 / inv_a
        sqa = np.sqrt(a)
        a_sqa = a * sqa
        n = SQ_MU / a_sqa
        target = SQ_MU * dt
        dE = (n * dt) if dE0 is None else dE0
        for _ in range(iters):
            s, c = np.sin(dE), np.cos(dE)
            F = (a_sqa * (dE - s) + sig0 * a * (1.0 - c) + r0s * sqa * s) - target
            dF = a_sqa * (1.0 - c) + sig0 * a * s + r0s * sqa * c
            dE = dE - F / dF

        s, c = np.sin(dE), np.cos(dE)
        f = 1.0 - (a / r0s) * (1.0 - c)
        g = dt - (a_sqa / SQ_MU) * (dE - s)
        rv = f[..., None] * r0v + g[..., None] * v0v
        r = np.sqrt(np.sum(rv * rv, axis=-1))
        fdot = -(SQ_MU * sqa / (r * r0s)) * s
        gdot = 1.0 - (a / r) * (1.0 - c)
        vv = fdot[..., None] * r0v + gdot[..., None] * v0v

    ok = ok0 & np.isfinite(rv).all(-1) & np.isfinite(vv).all(-1)
    Y = np.where(ok[..., None], np.concatenate([rv, vv], axis=-1), X)
    if want_dE:
        return Y, ok, dE
    return Y, ok


# ── 3D modified-spherical encode/decode (the 6-state generalisation) ─────────
_LOG_RHO_MAX, _RHO_FLOOR = 25.0, 1.0


def msc3_encode(sat, tgt, pole):
    """[beta, eps, betadot*cos eps, epsdot, rhodot/rho, ln rho].

    `pole` is the per-episode spherical-frame polar axis (unit 3-vector). The
    measurement (beta, eps) is components 0 and 1 EXACTLY, so H = [e1; e2] and
    the update stays linear — the property that removes the premature-collapse
    mode NAV-G measured for a Cartesian bearings-only EKF.
    """
    d = tgt[..., :3] - sat[..., :3]
    dv = tgt[..., 3:] - sat[..., 3:]
    e3 = pole / np.linalg.norm(pole, axis=-1, keepdims=True)
    # in-plane basis of the frame
    tmp = np.zeros_like(e3); tmp[..., 0] = 1.0
    alt = np.zeros_like(e3); alt[..., 1] = 1.0
    tmp = np.where(np.abs(e3[..., :1]) > 0.9, alt, tmp)
    e1 = np.cross(tmp, e3); e1 /= np.linalg.norm(e1, axis=-1, keepdims=True)
    e2 = np.cross(e3, e1)
    x = np.sum(d * e1, -1); y = np.sum(d * e2, -1); z = np.sum(d * e3, -1)
    vx = np.sum(dv * e1, -1); vy = np.sum(dv * e2, -1); vz = np.sum(dv * e3, -1)
    rho2 = np.maximum(x * x + y * y + z * z, _RHO_FLOOR ** 2)
    rho = np.sqrt(rho2)
    rxy2 = np.maximum(x * x + y * y, 1e-12)
    out = np.empty(x.shape + (6,))
    out[..., 0] = np.arctan2(y, x)
    out[..., 1] = np.arcsin(np.clip(z / rho, -1.0, 1.0))
    out[..., 2] = (x * vy - y * vx) / rxy2 * np.sqrt(rxy2) / rho     # betadot*cos eps
    out[..., 3] = (vz * rho2 - z * (x * vx + y * vy + z * vz)) / (rho2 * np.sqrt(rxy2))
    out[..., 4] = (x * vx + y * vy + z * vz) / rho2
    out[..., 5] = np.log(rho)
    return out


def bench(fn, *a, reps=20, **k):
    fn(*a, **k)
    t0 = time.perf_counter()
    for _ in range(reps):
        fn(*a, **k)
    return (time.perf_counter() - t0) / reps * 1e3   # ms


def draw(n, rng, a_lo=6.671e6, a_hi=1.4e7, e_hi=0.30, i_hi=np.radians(1.0)):
    a = rng.uniform(a_lo, a_hi, n)
    e = rng.uniform(0.0, e_hi, n)
    inc = rng.uniform(0.0, i_hi, n)
    raan = rng.uniform(0, 2 * np.pi, n)
    argp = rng.uniform(0, 2 * np.pi, n)
    nu = rng.uniform(0, 2 * np.pi, n)
    rv = np.array([o3.coe2rv(a[i], e[i], inc[i], raan[i], argp[i], nu[i])
                   for i in range(n)])
    return rv


def main():
    rng = np.random.default_rng(20260812)
    rows = []

    # ── 1. correctness vs the independent universal-variable oracle ─────────
    N = 300
    rv = draw(N, rng)
    for dt in (60.0, 300.0, 1800.0, 3600.0, 10800.0, 21600.0):
        Y, ok = propagate_cartesian_nd(rv, dt)
        Z = np.array([o3.propagate_universal(rv[i], dt) for i in range(N)])
        dp = np.linalg.norm(Y[:, :3] - Z[:, :3], axis=1)
        dvv = np.linalg.norm(Y[:, 3:] - Z[:, 3:], axis=1)
        rows.append(dict(test='fg6_vs_universal_oracle', dt_s=dt, n=N,
                         max_pos_err_m=float(dp.max()),
                         max_vel_err_ms=float(dvv.max()),
                         all_ok=int(ok.all())))
        print(f"dt={dt:8.0f}s  max|dr| {dp.max():.3e} m   max|dv| {dvv.max():.3e} m/s")

    # equatorial rows must reproduce the shipped 2D routine bit-for-bit
    rv2 = draw(N, rng, i_hi=0.0)
    Y6, _ = propagate_cartesian_nd(rv2, 60.0)
    X4 = np.stack([rv2[:, 0], rv2[:, 1], rv2[:, 3], rv2[:, 4]], 1)
    Y4, _ = nm.propagate_cartesian(X4, 60.0)
    d24 = np.abs(np.stack([Y6[:, 0], Y6[:, 1], Y6[:, 3], Y6[:, 4]], 1) - Y4)
    rows.append(dict(test='fg6_equatorial_vs_shipped_2d', dt_s=60.0, n=N,
                     max_pos_err_m=float(d24[:, :2].max()),
                     max_vel_err_ms=float(d24[:, 2:].max()),
                     all_ok=int(d24.max() == 0.0)))
    print(f"equatorial 6D vs shipped 2D: max abs diff {d24.max():.3e} "
          f"(bit-exact: {d24.max() == 0.0})")

    # round-trip through the 3D MSC encode
    pole = np.tile(np.array([0.0, 0.0, 1.0]), (N, 1))
    sat = draw(N, rng)
    y = msc3_encode(sat, rv, pole)
    rows.append(dict(test='msc3_finite', dt_s=0.0, n=N,
                     max_pos_err_m=float(np.abs(y).max()),
                     max_vel_err_ms=0.0, all_ok=int(np.isfinite(y).all())))

    # ── 2. cost ─────────────────────────────────────────────────────────────
    cost = []
    for B in (256, 1024, 4096):
        X6 = draw(min(B, 600), rng)
        X6 = np.repeat(X6, int(np.ceil(B / X6.shape[0])), 0)[:B]
        X4 = np.stack([X6[:, 0], X6[:, 1], X6[:, 3], X6[:, 4]], 1)
        t6 = bench(propagate_cartesian_nd, X6, 60.0)
        t4 = bench(nm.propagate_cartesian, X4, 60.0)
        # FD Jacobian: 2*d_state propagations, warm-started
        def fd6(X):
            Y, ok, dE = propagate_cartesian_nd(X, 60.0, want_dE=True)
            Z = np.empty((2 * X.shape[0], 6))
            dE2 = np.concatenate([dE, dE])
            F = np.empty((X.shape[0], 6, 6))
            h = (1.0, 1.0, 1.0, 1e-3, 1e-3, 1e-3)
            for j in range(6):
                Z[:X.shape[0]] = X; Z[:X.shape[0], j] += h[j]
                Z[X.shape[0]:] = X; Z[X.shape[0]:, j] -= h[j]
                W, _ = propagate_cartesian_nd(Z, 60.0, iters=2, dE0=dE2)
                F[:, :, j] = (W[:X.shape[0]] - W[X.shape[0]:]) / (2 * h[j])
            return F
        t_fd6 = bench(fd6, X6, reps=10)
        t_fd4 = bench(nm.stm_fd, X4, 60.0, reps=10)
        P6 = np.tile(np.eye(6), (B, 1, 1)); F6 = np.tile(np.eye(6), (B, 1, 1))
        P4 = np.tile(np.eye(4), (B, 1, 1)); F4 = np.tile(np.eye(4), (B, 1, 1))
        t_m6 = bench(lambda: F6 @ P6 @ np.swapaxes(F6, 1, 2), reps=30)
        t_m4 = bench(lambda: F4 @ P4 @ np.swapaxes(F4, 1, 2), reps=30)
        cost.append(dict(B=B, prop6_ms=t6, prop4_ms=t4, fdstm6_ms=t_fd6,
                         fdstm4_ms=t_fd4, cov6_ms=t_m6, cov4_ms=t_m4,
                         tick6_ms=t_fd6 + 2 * t_m6, tick4_ms=t_fd4 + 2 * t_m4,
                         ratio_6_over_4=(t_fd6 + 2 * t_m6) / (t_fd4 + 2 * t_m4)))
        print(f"B={B:5d}  prop 6D {t6:.3f} / 4D {t4:.3f} ms | FD-STM 6D {t_fd6:.3f} "
              f"/ 4D {t_fd4:.3f} ms | cov 6x6 {t_m6:.3f} / 4x4 {t_m4:.3f} ms "
              f"| tick ratio {cost[-1]['ratio_6_over_4']:.2f}x")

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, 'n3d_prop6_validation.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    with open(os.path.join(OUT, 'n3d_filter6_cost.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(cost[0].keys())); w.writeheader(); w.writerows(cost)
    print('\nwrote n3d_prop6_validation.csv, n3d_filter6_cost.csv')


if __name__ == '__main__':
    main()
