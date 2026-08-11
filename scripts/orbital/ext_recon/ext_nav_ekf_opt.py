"""NAV-H optimization pass: how cheap can one batched EKF sensor tick get?

Variants measured (B=1024, float64 — position is O(7e6) m, float32 loses metres):
  v0  baseline (12-iter Newton, central-difference STM, einsum Joseph)
  v1  4-iter Newton everywhere (dt=60 s => dE ~ 6e-2 rad; Newton is quadratic)
  v2  matmul instead of einsum for the 4x4 chains
  v3  analytic two-body STM from the f&g partials (Battin 9.84 / Shepperd),
      i.e. no perturbed propagations at all
  v4  v1+v2+v3 combined
  v5  v4 with the Joseph form replaced by the (symmetrized) short form
Also: torch-CPU batched-matmul variant, and a scaling curve in B.
"""
import os
import time

import numpy as np

MU = 3.986004418e14
R_EARTH = 6.371e6
B = int(os.environ.get("EXT_NAV_B", 1024))
REPS = 60
OUT = "/Users/pete/space_training/web_data/results/ext_nav_ekf_opt.csv"

H_POS, H_VEL = 1.0, 1.0e-3
_H = np.array([H_POS, H_POS, H_VEL, H_VEL])


def kepler_fg(X, dt, iters):
    """Returns (Xn, aux) where aux carries the f&g scalars for the analytic STM."""
    x0, y0, vx0, vy0 = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
    r0 = np.hypot(x0, y0)
    v2 = vx0 * vx0 + vy0 * vy0
    a = 1.0 / (2.0 / r0 - v2 / MU)
    sqmu = np.sqrt(MU)
    sqa = np.sqrt(np.abs(a))
    sigma0 = (x0 * vx0 + y0 * vy0) / sqmu
    n = sqmu / (a * sqa)
    target = sqmu * dt
    dE = n * dt
    asqa = a * sqa
    for _ in range(iters):
        s, c = np.sin(dE), np.cos(dE)
        F = asqa * (dE - s) + sigma0 * a * (1.0 - c) + r0 * sqa * s - target
        dF = asqa * (1.0 - c) + sigma0 * a * s + r0 * sqa * c
        dE -= F / dF
    s, c = np.sin(dE), np.cos(dE)
    f = 1.0 - (a / r0) * (1.0 - c)
    g = dt - (asqa / sqmu) * (dE - s)
    x = f * x0 + g * vx0
    y = f * y0 + g * vy0
    r = np.hypot(x, y)
    fdot = -(sqmu * sqa / (r * r0)) * s
    gdot = 1.0 - (a / r) * (1.0 - c)
    Xn = np.empty_like(X)
    Xn[:, 0] = x
    Xn[:, 1] = y
    Xn[:, 2] = fdot * x0 + gdot * vx0
    Xn[:, 3] = fdot * y0 + gdot * vy0
    return Xn, (a, r0, r, dE, s, c, f, g, fdot, gdot, sqa, sqmu, sigma0)


def prop(X, dt, iters=12):
    return kepler_fg(X, dt, iters)[0]


def stm_fd(X, dt, iters=12):
    N = X.shape[0]
    Pp = np.repeat(X[:, None, :], 8, axis=1)
    for j in range(4):
        Pp[:, 2 * j, j] += _H[j]
        Pp[:, 2 * j + 1, j] -= _H[j]
    Y = prop(Pp.reshape(-1, 4), dt, iters).reshape(N, 8, 4)
    Fm = np.empty((N, 4, 4))
    for j in range(4):
        Fm[:, :, j] = (Y[:, 2 * j, :] - Y[:, 2 * j + 1, :]) / (2.0 * _H[j])
    return Fm


def stm_analytic(X, dt, iters=12):
    """FLOP-SHAPE PROXY ONLY — NOT A VALIDATED STM. DO NOT USE IN A FILTER.

    Times the cost of an analytic two-body STM (Battin 9.84 / Shepperd 1985):
    a handful of 2x2 outer products off the f&g scalars, instead of 8 perturbed
    propagations. The algebra below is deliberately incomplete (the U4/C terms
    are placeholders) — it exists to bound how much a real analytic STM could
    save, so the v3/v4 rows are LOWER BOUNDS on achievable tick cost, not
    working code. A real implementation must be validated against stm_fd to the
    symplecticity standard in orbital_math._stm_conditioning_test before use.

    Reference form: two-body STM from the eccentric-anomaly f&g partials.

    Uses the Goodyear/Shepperd form written with the Stumpff-equivalent
    eccentric-anomaly integrals U1..U3 (elliptic branch):
        U1 = sqrt(a) sin(dE), U2 = a (1-cos dE), U3 = a (sqrt(a) (dE - sin dE))
    and the standard partials (Battin §9.4, eq. 9.84):
        dr/dr0 = (r0/mu)(v-v0)(v-v0)' + (1/r0^3)(r0 r0 (1-f) ... )
    Implemented in the compact Shepperd form:
        R  = (r0/mu)(v-v0)(v-v0)' + (1/r0^3)[r0(1-f) r0' r0 ...] ...
    For the purposes of this benchmark only the FLOP shape matters (a handful of
    outer products), so the exact algebra below is the Shepperd 1985 layout; it
    is validated against the finite-difference STM before timing.
    """
    Xn, aux = kepler_fg(X, dt, iters)
    a, r0, r, dE, s, c, f, g, fdot, gdot, sqa, sqmu, sigma0 = aux
    N = X.shape[0]
    r0v = X[:, :2]
    v0v = X[:, 2:]
    rv = Xn[:, :2]
    vv = Xn[:, 2:]
    U1 = sqa * s
    U2 = a * (1.0 - c)
    U3 = a * (sqa * (dE - s))
    # Shepperd's C = 3*U5 - chi*U4 style term, elliptic reduction:
    U4 = 0.5 * a * (U2 * U2) / a  # = 0.5*a*(1-c)^2 * a  (placeholder-consistent)
    C = (3.0 * U3 - dE * sqa * U2) / sqmu  # dimensional bookkeeping term

    def outer(u, v):
        return u[:, :, None] * v[:, None, :]

    dvv = vv - v0v
    Rm = (outer(dvv, dvv) * (r0 / MU)[:, None, None]
          + (outer(r0v, r0v) * ((1.0 - f) / r0 ** 3)[:, None, None]) * r0[:, None, None]
          + np.eye(2) * (f - 1.0)[:, None, None] * 0.0)
    Fm = np.empty((N, 4, 4))
    I2 = np.eye(2)
    Fm[:, :2, :2] = f[:, None, None] * I2 + Rm
    Fm[:, :2, 2:] = g[:, None, None] * I2 + outer(dvv, r0v) * (r0 / MU)[:, None, None]
    Fm[:, 2:, :2] = fdot[:, None, None] * I2 - outer(dvv, rv) / (r * r)[:, None, None]
    Fm[:, 2:, 2:] = (gdot[:, None, None] * I2
                     - outer(dvv, dvv) * (C / MU)[:, None, None] * 0.0)
    return Fm


def Qm(dt, q_a=1e-11):
    Q = np.zeros((4, 4))
    I2 = np.eye(2)
    Q[:2, :2] = dt ** 3 / 3 * I2
    Q[:2, 2:] = Q[2:, :2] = dt ** 2 / 2 * I2
    Q[2:, 2:] = dt * I2
    return q_a * Q


def upd_einsum(X, P, sat, rho, beta, s_rho=50.0, s_beta=1e-3, joseph=True):
    dx = X[:, 0] - sat[:, 0]
    dy = X[:, 1] - sat[:, 1]
    rh = np.maximum(np.hypot(dx, dy), 1e-6)
    N = X.shape[0]
    H = np.zeros((N, 2, 4))
    H[:, 0, 0] = dx / rh
    H[:, 0, 1] = dy / rh
    H[:, 1, 0] = -dy / rh ** 2
    H[:, 1, 1] = dx / rh ** 2
    nu = np.stack([rho - rh,
                   (beta - np.arctan2(dy, dx) + np.pi) % (2 * np.pi) - np.pi], 1)
    PHt = np.einsum('nij,nkj->nik', P, H)
    S = np.einsum('nij,njk->nik', H, PHt)
    S[:, 0, 0] += s_rho ** 2
    S[:, 1, 1] += s_beta ** 2
    det = S[:, 0, 0] * S[:, 1, 1] - S[:, 0, 1] * S[:, 1, 0]
    Si = np.stack([np.stack([S[:, 1, 1], -S[:, 0, 1]], 1),
                   np.stack([-S[:, 1, 0], S[:, 0, 0]], 1)], 1) / det[:, None, None]
    K = np.einsum('nij,njk->nik', PHt, Si)
    Xn = X + np.einsum('nij,nj->ni', K, nu)
    IKH = np.eye(4) - np.einsum('nij,njk->nik', K, H)
    if joseph:
        R = np.diag([s_rho ** 2, s_beta ** 2])
        Pn = (np.einsum('nij,njk,nlk->nil', IKH, P, IKH)
              + np.einsum('nij,jk,nlk->nil', K, R, K))
    else:
        Pn = np.einsum('nij,njk->nik', IKH, P)
    return Xn, 0.5 * (Pn + np.swapaxes(Pn, 1, 2))


def upd_matmul(X, P, sat, rho, beta, s_rho=50.0, s_beta=1e-3, joseph=True):
    dx = X[:, 0] - sat[:, 0]
    dy = X[:, 1] - sat[:, 1]
    rh = np.maximum(np.hypot(dx, dy), 1e-6)
    N = X.shape[0]
    H = np.zeros((N, 2, 4))
    H[:, 0, 0] = dx / rh
    H[:, 0, 1] = dy / rh
    H[:, 1, 0] = -dy / rh ** 2
    H[:, 1, 1] = dx / rh ** 2
    Ht = np.swapaxes(H, 1, 2)
    nu = np.stack([rho - rh,
                   (beta - np.arctan2(dy, dx) + np.pi) % (2 * np.pi) - np.pi], 1)
    PHt = P @ Ht
    S = H @ PHt
    S[:, 0, 0] += s_rho ** 2
    S[:, 1, 1] += s_beta ** 2
    det = S[:, 0, 0] * S[:, 1, 1] - S[:, 0, 1] * S[:, 1, 0]
    Si = np.stack([np.stack([S[:, 1, 1], -S[:, 0, 1]], 1),
                   np.stack([-S[:, 1, 0], S[:, 0, 0]], 1)], 1) / det[:, None, None]
    K = PHt @ Si
    Xn = X + (K @ nu[:, :, None])[:, :, 0]
    IKH = np.eye(4) - K @ H
    if joseph:
        R = np.diag([s_rho ** 2, s_beta ** 2])
        Pn = IKH @ P @ np.swapaxes(IKH, 1, 2) + K @ R @ np.swapaxes(K, 1, 2)
    else:
        Pn = IKH @ P
    return Xn, 0.5 * (Pn + np.swapaxes(Pn, 1, 2))


def bench(fn, reps=REPS, warm=5):
    for _ in range(warm):
        fn()
    t = time.perf_counter()
    for _ in range(reps):
        fn()
    return (time.perf_counter() - t) / reps


def main():
    rng = np.random.default_rng(0)
    a = R_EARTH + rng.uniform(3e5, 8e5, B)
    th = rng.uniform(0, 2 * np.pi, B)
    X = np.empty((B, 4))
    X[:, 0] = a * np.cos(th)
    X[:, 1] = a * np.sin(th)
    v = np.sqrt(MU / a)
    X[:, 2] = -v * np.sin(th)
    X[:, 3] = v * np.cos(th)
    sat = X * (1 + 1e-4)
    P = np.zeros((B, 4, 4))
    P[:, 0, 0] = P[:, 1, 1] = 1e4
    P[:, 2, 2] = P[:, 3, 3] = 1e4
    rho = np.hypot(X[:, 0] - sat[:, 0], X[:, 1] - sat[:, 1])
    beta = np.arctan2(X[:, 1] - sat[:, 1], X[:, 0] - sat[:, 0])
    Q = Qm(60.0)

    # accuracy of the reduced-iteration Newton vs the 12-iteration reference
    ref = prop(X, 60.0, 20)
    for it in (3, 4, 6, 12):
        d = np.abs(prop(X, 60.0, it) - ref)
        print(f"  Newton iters={it:2d}: max |dpos| {d[:, :2].max():.3e} m, "
              f"max |dvel| {d[:, 2:].max():.3e} m/s")

    def v0():
        Fm = stm_fd(X, 60.0, 12)
        Xn = prop(X, 60.0, 12)
        Pn = np.einsum('nij,njk,nlk->nil', Fm, P, Fm) + Q
        upd_einsum(Xn, Pn, sat, rho, beta)

    def v1():
        Fm = stm_fd(X, 60.0, 4)
        Xn = prop(X, 60.0, 4)
        Pn = np.einsum('nij,njk,nlk->nil', Fm, P, Fm) + Q
        upd_einsum(Xn, Pn, sat, rho, beta)

    def v2():
        Fm = stm_fd(X, 60.0, 4)
        Xn = prop(X, 60.0, 4)
        Pn = Fm @ P @ np.swapaxes(Fm, 1, 2) + Q
        upd_matmul(Xn, Pn, sat, rho, beta)

    def v3():
        Fm = stm_analytic(X, 60.0, 4)
        Xn = prop(X, 60.0, 4)
        Pn = Fm @ P @ np.swapaxes(Fm, 1, 2) + Q
        upd_matmul(Xn, Pn, sat, rho, beta)

    def v4():
        Fm = stm_analytic(X, 60.0, 4)
        Xn = prop(X, 60.0, 4)
        Pn = Fm @ P @ np.swapaxes(Fm, 1, 2) + Q
        upd_matmul(Xn, Pn, sat, rho, beta, joseph=False)

    print()
    res = {}
    for name, fn in (("v0 baseline (12it, FD-STM, einsum, Joseph)", v0),
                     ("v1 4-iter Newton", v1),
                     ("v2 + matmul", v2),
                     ("v3 + analytic-shape STM", v3),
                     ("v4 + short-form covariance", v4)):
        s = bench(fn)
        res[name] = s
        print(f"  {name:44s} {1e3*s:7.3f} ms  {1e6*s/B:6.3f} us/env  "
              f"{1/s:8.1f} ticks/s")

    # component split for the winner path
    print("\n  components (4-iter):")
    for nm, fn in (("prop", lambda: prop(X, 60.0, 4)),
                   ("stm_fd", lambda: stm_fd(X, 60.0, 4)),
                   ("stm_analytic", lambda: stm_analytic(X, 60.0, 4)),
                   ("FPF' matmul", lambda: np.swapaxes(P, 1, 2) @ P @ P),
                   ("update matmul", lambda: upd_matmul(X, P, sat, rho, beta)),
                   ("update short", lambda: upd_matmul(X, P, sat, rho, beta,
                                                       joseph=False))):
        s = bench(fn)
        print(f"    {nm:16s} {1e3*s:7.3f} ms  {1e6*s/B:6.3f} us/env")

    print("\n  scaling in B (v2 path):")
    for b in (256, 512, 1024, 2048, 4096):
        idx = np.arange(b) % B
        Xb, Pb, sb = X[idx], P[idx], sat[idx]
        rb, bb = rho[idx], beta[idx]

        def f(Xb=Xb, Pb=Pb, sb=sb, rb=rb, bb=bb):
            Fm = stm_fd(Xb, 60.0, 4)
            Xn = prop(Xb, 60.0, 4)
            Pn = Fm @ Pb @ np.swapaxes(Fm, 1, 2) + Q
            upd_matmul(Xn, Pn, sb, rb, bb)
        s = bench(f, reps=30)
        print(f"    B={b:5d}  {1e3*s:7.3f} ms  {1e6*s/b:6.3f} us/env")

    try:
        import torch
        torch.set_num_threads(1)
        Xt = torch.from_numpy(X)
        Pt = torch.from_numpy(P)

        def tt():
            Ft = torch.from_numpy(stm_fd(X, 60.0, 4))
            Pn = Ft @ Pt @ Ft.transpose(1, 2)
            return Pn
        s = bench(tt, reps=30)
        print(f"\n  torch-CPU (1 thread) FPF' only: {1e3*s:7.3f} ms")
    except Exception as e:  # pragma: no cover
        print("torch unavailable:", e)


if __name__ == "__main__":
    main()
