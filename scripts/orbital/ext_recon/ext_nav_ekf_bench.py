"""NAV-H microbenchmark: batched numpy EKF cost vs the C env's throughput.

Read-only recon. Measures, at B = 1024 envs (the shipped [env] num_envs):
  1. batched Lagrange f&g Kepler propagation (the filter's mean map)
  2. batched STM by central differences (8 perturbed propagations)
  3. batched predict (F P F^T + Q)
  4. batched update (2x2 innovation, Joseph form)
  5. full predict+update sensor tick
  6. batched observation decode (Kepler solve for the target anomaly) + re-encode
     — the cost of recovering truth from obs instead of adding a C getter
  7. the same tick with a torch/CPU backend for comparison

Outputs a CSV to web_data/results/ext_nav_ekf_bench.csv.
Run from /Users/pete/space_training/pufferlib (or anywhere; no env import here).
"""
import csv
import os
import time

import numpy as np

MU = 3.986004418e14
R_EARTH = 6.371e6
B = int(os.environ.get("EXT_NAV_B", 1024))
REPS = int(os.environ.get("EXT_NAV_REPS", 60))
OUT = "/Users/pete/space_training/web_data/results/ext_nav_ekf_bench.csv"


# ── batched two-body f&g propagation (vectorized orbital_math.propagate_cartesian)
def propagate_batch(X, dt, iters=12):
    """X: (N,4) [x,y,vx,vy]; dt: scalar or (N,). Fixed-iteration Newton on dE."""
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
        s = np.sin(dE)
        c = np.cos(dE)
        F = asqa * (dE - s) + sigma0 * a * (1.0 - c) + r0 * sqa * s - target
        dF = asqa * (1.0 - c) + sigma0 * a * s + r0 * sqa * c
        dE -= F / dF
    s = np.sin(dE)
    c = np.cos(dE)
    f = 1.0 - (a / r0) * (1.0 - c)
    g = dt - (asqa / sqmu) * (dE - s)
    x = f * x0 + g * vx0
    y = f * y0 + g * vy0
    r = np.hypot(x, y)
    fdot = -(sqmu * sqa / (r * r0)) * s
    gdot = 1.0 - (a / r) * (1.0 - c)
    out = np.empty_like(X)
    out[:, 0] = x
    out[:, 1] = y
    out[:, 2] = fdot * x0 + gdot * vx0
    out[:, 3] = fdot * y0 + gdot * vy0
    return out


H_POS, H_VEL = 1.0, 1.0e-3
_H = np.array([H_POS, H_POS, H_VEL, H_VEL])


def stm_batch(X, dt):
    """(N,4,4) central-difference STM: one propagate call over 8N states."""
    N = X.shape[0]
    P = np.repeat(X[:, None, :], 8, axis=1)          # (N,8,4)
    for j in range(4):
        P[:, 2 * j, j] += _H[j]
        P[:, 2 * j + 1, j] -= _H[j]
    Y = propagate_batch(P.reshape(-1, 4), dt).reshape(N, 8, 4)
    Fm = np.empty((N, 4, 4))
    for j in range(4):
        Fm[:, :, j] = (Y[:, 2 * j, :] - Y[:, 2 * j + 1, :]) / (2.0 * _H[j])
    return Fm


def process_noise(dt, q_a):
    I2 = np.eye(2)
    Q = np.zeros((4, 4))
    Q[:2, :2] = (dt ** 3 / 3.0) * I2
    Q[:2, 2:] = (dt ** 2 / 2.0) * I2
    Q[2:, :2] = (dt ** 2 / 2.0) * I2
    Q[2:, 2:] = dt * I2
    return q_a * Q


def predict_batch(X, P, dt, q_a=1e-11):
    Fm = stm_batch(X, dt)
    Xn = propagate_batch(X, dt)
    Pn = np.einsum('nij,njk,nlk->nil', Fm, P, Fm) + process_noise(dt, q_a)
    Pn = 0.5 * (Pn + np.swapaxes(Pn, 1, 2))
    return Xn, Pn


def update_batch(X, P, sat, rho, beta, s_rho=50.0, s_beta=1e-3):
    dx = X[:, 0] - sat[:, 0]
    dy = X[:, 1] - sat[:, 1]
    rh = np.hypot(dx, dy)
    rh = np.maximum(rh, 1e-6)
    bh = np.arctan2(dy, dx)
    N = X.shape[0]
    H = np.zeros((N, 2, 4))
    H[:, 0, 0] = dx / rh
    H[:, 0, 1] = dy / rh
    H[:, 1, 0] = -dy / (rh * rh)
    H[:, 1, 1] = dx / (rh * rh)
    nu = np.empty((N, 2))
    nu[:, 0] = rho - rh
    nu[:, 1] = (beta - bh + np.pi) % (2 * np.pi) - np.pi
    PHt = np.einsum('nij,nkj->nik', P, H)            # (N,4,2)
    S = np.einsum('nij,njk->nik', H, PHt)            # (N,2,2)
    S[:, 0, 0] += s_rho ** 2
    S[:, 1, 1] += s_beta ** 2
    det = S[:, 0, 0] * S[:, 1, 1] - S[:, 0, 1] * S[:, 1, 0]
    Si = np.empty_like(S)
    Si[:, 0, 0] = S[:, 1, 1] / det
    Si[:, 1, 1] = S[:, 0, 0] / det
    Si[:, 0, 1] = -S[:, 0, 1] / det
    Si[:, 1, 0] = -S[:, 1, 0] / det
    K = np.einsum('nij,njk->nik', PHt, Si)           # (N,4,2)
    Xn = X + np.einsum('nij,nj->ni', K, nu)
    IKH = np.eye(4) - np.einsum('nij,njk->nik', K, H)
    R = np.zeros((2, 2))
    R[0, 0] = s_rho ** 2
    R[1, 1] = s_beta ** 2
    Pn = (np.einsum('nij,njk,nlk->nil', IKH, P, IKH)
          + np.einsum('nij,jk,nlk->nil', K, R, K))
    Pn = 0.5 * (Pn + np.swapaxes(Pn, 1, 2))
    return Xn, Pn


# ── batched obs decode / re-encode (the no-C-getter path) ────────────────────
def solve_kepler_batch(M, e, iters=6):
    E = M.copy()
    for _ in range(iters):
        E = E - (E - e * np.sin(E) - M) / (1.0 - e * np.cos(E))
    return E


def decode_reencode(obs, alt=1.6e6, lvlh=6.371e6):
    """Invert fill_observations() under phase_obs_mode=1 and rebuild the target
    slots (the batched analogue of eval_relnav.recover_states_t3 + build_obs_t3)."""
    a_s = obs[:, 0] * alt + R_EARTH
    e_s = obs[:, 1]
    th_s = np.arctan2(obs[:, 2], obs[:, 3])
    om_s = np.arctan2(obs[:, 9], obs[:, 10])
    # true -> eccentric -> mean
    E_s = 2.0 * np.arctan2(np.sqrt(1 - e_s) * np.sin(th_s / 2),
                           np.sqrt(1 + e_s) * np.cos(th_s / 2))
    M_s = E_s - e_s * np.sin(E_s)
    a_t = obs[:, 7] * alt + R_EARTH
    e_t = obs[:, 8]
    om_t = np.arctan2(obs[:, 11], obs[:, 12])
    dlam = np.arctan2(obs[:, 13], obs[:, 14])
    M_t = (M_s + om_s - dlam) - om_t
    E_t = solve_kepler_batch(np.mod(M_t, 2 * np.pi), e_t)
    th_t = 2.0 * np.arctan2(np.sqrt(1 + e_t) * np.sin(E_t / 2),
                            np.sqrt(1 - e_t) * np.cos(E_t / 2))
    # elements -> cartesian, both bodies
    out = np.empty((obs.shape[0], 8))
    for k, (a, e, th, om) in enumerate(((a_s, e_s, th_s, om_s),
                                        (a_t, e_t, th_t, om_t))):
        p = a * (1 - e * e)
        r = p / (1 + e * np.cos(th))
        u = th + om
        cu, su = np.cos(u), np.sin(u)
        h = np.sqrt(MU * p)
        vr = MU / h * e * np.sin(th)
        vt = h / r
        out[:, 4 * k + 0] = r * cu
        out[:, 4 * k + 1] = r * su
        out[:, 4 * k + 2] = vr * cu - vt * su
        out[:, 4 * k + 3] = vr * su + vt * cu
    # re-encode the target-derived slots from (possibly estimated) elements
    o2 = obs.copy()
    o2[:, 7] = (a_t - R_EARTH) / alt
    o2[:, 8] = e_t
    o2[:, 11] = np.sin(om_t)
    o2[:, 12] = np.cos(om_t)
    lam = (M_s + om_s) - (M_t + om_t)
    o2[:, 13] = np.sin(lam)
    o2[:, 14] = np.cos(lam)
    o2[:, 16] = np.cos(om_s - om_t)
    dx = out[:, 0] - out[:, 4]
    dy = out[:, 1] - out[:, 5]
    o2[:, 33] = dx / lvlh
    o2[:, 34] = dy / lvlh
    o2[:, 35] = (out[:, 2] - out[:, 6]) / 1e3
    o2[:, 36] = (out[:, 3] - out[:, 7]) / 1e3
    return o2, out


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
    r = a
    X = np.empty((B, 4))
    X[:, 0] = r * np.cos(th)
    X[:, 1] = r * np.sin(th)
    v = np.sqrt(MU / r)
    X[:, 2] = -v * np.sin(th)
    X[:, 3] = v * np.cos(th)
    sat = X + rng.normal(0, 5e4, (B, 4)) * np.array([1, 1, 1e-4, 1e-4])
    P0 = np.zeros((B, 4, 4))
    P0[:, 0, 0] = P0[:, 1, 1] = 1e4
    P0[:, 2, 2] = P0[:, 3, 3] = 1e4
    rho = np.hypot(X[:, 0] - sat[:, 0], X[:, 1] - sat[:, 1])
    beta = np.arctan2(X[:, 1] - sat[:, 1], X[:, 0] - sat[:, 0])
    obs = rng.uniform(-1, 1, (B, 38))
    obs[:, 1] = obs[:, 8] = 0.03
    obs[:, 0] = obs[:, 7] = 0.3

    rows = []

    def add(name, s, note=""):
        rows.append(dict(op=name, batch=B, ms=1e3 * s,
                         us_per_env=1e6 * s / B,
                         ticks_per_s=1.0 / s, note=note))
        print(f"  {name:34s} {1e3*s:8.3f} ms/call  {1e6*s/B:8.3f} us/env  "
              f"{1.0/s:10.1f} calls/s  {note}")

    print(f"batched numpy EKF microbenchmark, B={B}, reps={REPS}, "
          f"numpy {np.__version__}")
    add("propagate f&g (12 Newton iters)", bench(lambda: propagate_batch(X, 60.0)))
    add("stm central-diff (8N propagate)", bench(lambda: stm_batch(X, 60.0)))
    add("predict (stm + prop + FPF'+Q)", bench(lambda: predict_batch(X, P0, 60.0)))
    add("update (2x2 solve + Joseph)",
        bench(lambda: update_batch(X, P0, sat, rho, beta)))

    def tick():
        Xn, Pn = predict_batch(X, P0, 60.0)
        update_batch(Xn, Pn, sat, rho, beta)
    t_tick = bench(tick)
    add("FULL sensor tick (predict+update)", t_tick)

    add("obs decode + re-encode (no C getter)",
        bench(lambda: decode_reencode(obs)))

    # predict with a 5-iteration Newton (dt=60 s is a tiny dE; convergence is fast)
    add("propagate f&g (4 Newton iters)",
        bench(lambda: propagate_batch(X, 60.0, iters=4)))

    def tick_fast():
        Fm = stm_batch(X, 60.0)
        Xn = propagate_batch(X, 60.0, iters=4)
        Pn = np.einsum('nij,njk,nlk->nil', Fm, P0, Fm)
        update_batch(Xn, Pn, sat, rho, beta)
    add("tick w/ 4-iter Newton", bench(tick_fast))

    # analytic-free alternative: reuse one STM for all envs (same dt, similar a)
    def tick_sharedF():
        Fm = stm_batch(X[:1], 60.0)[0]
        Xn = propagate_batch(X, 60.0)
        Pn = np.einsum('ij,njk,lk->nil', Fm, P0, Fm)
        update_batch(Xn, Pn, sat, rho, beta)
    add("tick w/ SHARED STM (upper bound only)", bench(tick_sharedF),
        "not physically valid; floor on achievable cost")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {OUT}")

    # ── projection ──────────────────────────────────────────────────────────
    print("\nSPS projection (single worker, B=1024):")
    env_substep_sps = 2.9e6          # measured, all-warp-1h, this machine
    for tau, label in ((1, "burn/coast (tau=1)"), (5, "warp-5min"),
                       (60, "warp-1h"), (360, "warp-6h")):
        for ncad, cname in ((tau, "60 s cadence"), (min(tau, 12), "capped 12/decision")):
            filt = ncad * t_tick
            envt = tau * B / env_substep_sps
            print(f"  {label:20s} {cname:20s} env {1e3*envt:7.2f} ms  "
                  f"filter {1e3*filt:8.2f} ms  slowdown x{(envt+filt)/envt:6.2f}")


if __name__ == "__main__":
    main()
