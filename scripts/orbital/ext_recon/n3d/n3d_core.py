#!/usr/bin/env python3
"""N3D-A core — batched 3D two-body math, LOS az/el measurement model, and
modified-spherical-coordinate (MSC) state encoding for a 3D bearings-only
relative-navigation filter.

Relationship to the shipped stack
---------------------------------
* `pufferlib/ocean/orbital_nav/nav_math.py` — batched 2D f&g-in-dE propagator +
  4-state modified-polar filter. THIS FILE IS ITS 3D LIFT: the Lagrange f&g
  solution in the eccentric-anomaly difference is dimension-agnostic (only the
  scalars r0, v^2, sigma0 change from 2-component to 3-component sums), so the
  propagator is literally the same algebra on 3-vectors. That is deliberate:
  the 3D filter must reduce to the shipped 2D filter EXACTLY at zero relative
  inclination, and that reduction is a regression test (see `--selftest`).
* `scripts/orbital/ext_recon/orbital_math3d.py` — the INDEPENDENT 3D oracle
  (universal variables + Stumpff, 56/56 validated). Used here only as the
  cross-check reference; never as an implementation dependency in the hot loop.

Frame choice (the one real design decision, justified in the memo)
------------------------------------------------------------------
LOS angles are measured as (azimuth, elevation) in an INERTIALLY FIXED,
EPISODE-FROZEN orthonormal frame whose +z axis is the CHASER's orbit normal at
epoch, h_hat_c(t0).  Consequences, all load-bearing:

  1. `az` is exactly the 2D inertial bearing beta of the shipped filter, and
     `el` is exactly the out-of-plane LOS angle.  At zero relative inclination
     el == 0 identically and the 6-state MSC filter collapses onto the shipped
     4-state modified-polar filter, component for component.
  2. The frame is FIXED (frozen at epoch), not co-rotating, so there are no
     frame-rate terms in the transition and no chaser-attitude model is needed.
     The chaser's absolute state is known, so this frame is known.
  3. The az/el gimbal singularity sits at el = +-90 deg, i.e. LOS parallel to
     the chaser's orbit normal.  Reaching it requires the out-of-plane
     component of the separation to dominate the in-plane component, which for
     a rendezvous geometry means |rho_oop| >~ |rho_inplane|.  That is possible
     (see `--selftest` guard) and is handled by an explicit re-pole guard
     rather than being assumed away.

Nothing here imports the C extension. No existing file is modified.
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np

MU = 3.986004418e14
R_EARTH = 6.371e6
SQ_MU = math.sqrt(MU)

SIGMA_BETA_RAD = 1.0e-3          # shipped optical bearing sigma (per axis)
SIGMA_RHO_M = 50.0               # shipped range sigma (RB baseline only)
Q_ACCEL_PSD_BO = 1.0e-13         # NAV-G wide-envelope bearings-only value

_NEWTON_ITERS = 6
H_POS, H_VEL = 1.0, 1.0e-3
_RHO_FLOOR = 1.0
_LOG_RHO_MAX = 25.0

# 95% two-sided chi-square bounds normalised by dof, dof = 6.
NEES6_LO, NEES6_HI = 1.2373 / 6.0, 14.4494 / 6.0


def pin_blas_threads():
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(v, "1")


def wrap_pi(x):
    return (np.asarray(x) + np.pi) % (2.0 * np.pi) - np.pi


# ── batched 3D propagation (f&g in the eccentric-anomaly difference) ─────────
def propagate3(X, dt, iters=_NEWTON_ITERS, dE0=None, want_dE=False):
    """Lagrange f&g propagation of (..., 6) Cartesian states by dt seconds.

    Identical algebra to nav_math.propagate_cartesian, lifted to 3-vectors.
    Returns (Y, ok[, dE]).  Non-elliptical / non-finite rows are returned
    unchanged with ok=False so a caller can reinitialise instead of
    propagating garbage.
    """
    X = np.asarray(X, dtype=np.float64)
    R0 = X[..., :3]
    V0 = X[..., 3:]
    with np.errstate(all='ignore'):
        r0 = np.sqrt(np.sum(R0 * R0, axis=-1))
        v2 = np.sum(V0 * V0, axis=-1)
        inv_a = 2.0 / r0 - v2 / MU
        ok0 = np.isfinite(inv_a) & (inv_a > 1e-30) & (r0 > 1.0)
        inv_a = np.where(ok0, inv_a, 1.0 / 7.0e6)
        r0s = np.where(ok0, r0, 7.0e6)
        sig0 = np.where(ok0, np.sum(R0 * V0, axis=-1) / SQ_MU, 0.0)

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
        Rn = f[..., None] * R0 + g[..., None] * V0
        r = np.sqrt(np.sum(Rn * Rn, axis=-1))
        fdot = -(SQ_MU * sqa / (r * r0s)) * s
        gdot = 1.0 - (a / r) * (1.0 - c)
        Vn = fdot[..., None] * R0 + gdot[..., None] * V0

    Y = np.concatenate([Rn, Vn], axis=-1)
    ok = ok0 & np.all(np.isfinite(Y), axis=-1)
    Y = np.where(ok[..., None], Y, X)
    if want_dE:
        return Y, ok, dE
    return Y, ok


def stm_fd3(X, dt, iters=_NEWTON_ITERS, warm_iters=2):
    """d(propagate3(X, dt))/dX by central differences. (N,6) -> (N,6,6).

    12 warm-started perturbed propagations + 1 nominal = 13 batched
    propagations, vs 9 for the 4-state 2D filter.  Returns (F, ok, Y).
    """
    X = np.asarray(X, dtype=np.float64)
    n = X.shape[0]
    Y, ok, dE = propagate3(X, dt, iters, want_dE=True)
    F = np.empty((n, 6, 6), dtype=np.float64)
    h = (H_POS, H_POS, H_POS, H_VEL, H_VEL, H_VEL)
    Z = np.empty((2 * n, 6), dtype=np.float64)
    dE2 = np.concatenate([dE, dE])
    for j in range(6):
        Z[:n] = X; Z[:n, j] += h[j]
        Z[n:] = X; Z[n:, j] -= h[j]
        W, okw = propagate3(Z, dt, warm_iters, dE0=dE2)
        F[:, :, j] = (W[:n] - W[n:]) / (2.0 * h[j])
        ok &= okw[:n] & okw[n:]
    return F, ok, Y


def process_noise6(dt, q_a):
    """Continuous white-noise-acceleration Q for a 3D double integrator."""
    I3 = np.eye(3)
    Q = np.zeros((6, 6))
    Q[:3, :3] = (dt ** 3 / 3.0) * I3
    Q[:3, 3:] = (dt ** 2 / 2.0) * I3
    Q[3:, :3] = (dt ** 2 / 2.0) * I3
    Q[3:, 3:] = dt * I3
    return q_a * Q


# ── elements (only what the observability decomposition needs) ──────────────
def elements_of(X):
    """(...,6) -> dict of a, e, inc, raan, argp, nu, u (arg of latitude), h_hat.

    Scalar-safe, vectorised over leading axes.  Uses atan2 forms throughout
    (no acos + sign patch), matching the oracle's convention.
    """
    X = np.asarray(X, dtype=np.float64)
    R, V = X[..., :3], X[..., 3:]
    r = np.linalg.norm(R, axis=-1)
    v2 = np.sum(V * V, axis=-1)
    H = np.cross(R, V)
    h = np.linalg.norm(H, axis=-1)
    hhat = H / h[..., None]
    inv_a = 2.0 / r - v2 / MU
    a = 1.0 / inv_a
    Ev = (np.sum(V * V, axis=-1)[..., None] / MU - 1.0 / r[..., None]) * R \
        - (np.sum(R * V, axis=-1)[..., None] / MU) * V
    e = np.linalg.norm(Ev, axis=-1)
    inc = np.arctan2(np.linalg.norm(H[..., :2], axis=-1), H[..., 2])
    Nv = np.stack([-H[..., 1], H[..., 0], np.zeros_like(H[..., 0])], axis=-1)
    nn = np.linalg.norm(Nv, axis=-1)
    raan = np.arctan2(Nv[..., 1], Nv[..., 0])
    # argument of latitude u = angle from node to r, right-handed about h
    nh = np.where(nn[..., None] > 0.0, Nv / np.maximum(nn, 1e-300)[..., None],
                  np.array([1.0, 0.0, 0.0]))
    rh = R / r[..., None]
    u = np.arctan2(np.sum(np.cross(nh, rh) * hhat, axis=-1),
                   np.sum(nh * rh, axis=-1))
    eh = np.where(e[..., None] > 0.0, Ev / np.maximum(e, 1e-300)[..., None], nh)
    argp = np.arctan2(np.sum(np.cross(nh, eh) * hhat, axis=-1),
                      np.sum(nh * eh, axis=-1))
    nu = np.arctan2(np.sum(np.cross(eh, rh) * hhat, axis=-1),
                    np.sum(eh * rh, axis=-1))
    return dict(a=a, e=e, inc=inc, raan=raan, argp=argp, nu=nu, u=u,
                hhat=hhat, h=h, r=r)


def coe2rv(a, e, inc, raan, argp, nu):
    p = a * (1.0 - e * e)
    r = p / (1.0 + e * math.cos(nu))
    sq = math.sqrt(MU / p)
    r_pf = np.array([r * math.cos(nu), r * math.sin(nu), 0.0])
    v_pf = np.array([-sq * math.sin(nu), sq * (e + math.cos(nu)), 0.0])
    cO, sO = math.cos(raan), math.sin(raan)
    ci, si = math.cos(inc), math.sin(inc)
    cw, sw = math.cos(argp), math.sin(argp)
    R = np.array([
        [cO * cw - sO * sw * ci, -cO * sw - sO * cw * ci,  sO * si],
        [sO * cw + cO * sw * ci, -sO * sw + cO * cw * ci, -cO * si],
        [sw * si,                 cw * si,                 ci],
    ])
    return np.concatenate([R @ r_pf, R @ v_pf])


def mean_from_true(nu, e):
    E = 2.0 * math.atan2(math.sqrt(max(1.0 - e, 0.0)) * math.sin(0.5 * nu),
                         math.sqrt(1.0 + e) * math.cos(0.5 * nu))
    return E - e * math.sin(E)


def period(a):
    return 2.0 * math.pi * math.sqrt(a ** 3 / MU)


# ── pole frame + LOS az/el measurement ──────────────────────────────────────
def pole_frame(sat0):
    """Orthonormal W (3x3, rows = basis) with +z = chaser orbit normal at epoch.

    +x is the chaser's radial direction at epoch (an arbitrary but reproducible
    in-plane reference), +y = z x x.  Returns W such that v_frame = W @ v_eci.
    """
    sat0 = np.asarray(sat0, dtype=np.float64).reshape(6)
    z = np.cross(sat0[:3], sat0[3:])
    z = z / np.linalg.norm(z)
    x = sat0[:3] - np.dot(sat0[:3], z) * z
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    return np.stack([x, y, z], axis=0)


def los_azel(sat, tgt, W):
    """(az, el) of the LOS in the pole frame. sat/tgt (...,6) or (...,3)."""
    sat = np.asarray(sat, dtype=np.float64)
    tgt = np.asarray(tgt, dtype=np.float64)
    d = tgt[..., :3] - sat[..., :3]
    dp = d @ W.T
    rho = np.linalg.norm(dp, axis=-1)
    rho = np.maximum(rho, _RHO_FLOOR)
    az = np.arctan2(dp[..., 1], dp[..., 0])
    el = np.arcsin(np.clip(dp[..., 2] / rho, -1.0, 1.0))
    return az, el


def azel_jac(sat, tgt, W):
    """d(az, el)/d(target inertial 6-state).  (...,2,6); velocity columns 0."""
    sat = np.asarray(sat, dtype=np.float64)
    tgt = np.asarray(tgt, dtype=np.float64)
    d = (tgt[..., :3] - sat[..., :3]) @ W.T
    x, y, z = d[..., 0], d[..., 1], d[..., 2]
    rho2 = x * x + y * y + z * z
    rxy2 = np.maximum(x * x + y * y, 1e-30)
    rxy = np.sqrt(rxy2)
    # d(az)/d(d_frame)
    daz = np.stack([-y / rxy2, x / rxy2, np.zeros_like(x)], axis=-1)
    # d(el)/d(d_frame),  el = atan2(z, rxy)
    del_ = np.stack([-x * z / (rho2 * rxy), -y * z / (rho2 * rxy), rxy / rho2],
                    axis=-1)
    # chain through the (constant) frame rotation: d_frame = W (r_t - r_c)
    J = np.zeros(np.broadcast_shapes(daz.shape[:-1], (1,)) + (2, 6))
    J[..., 0, :3] = daz @ W
    J[..., 1, :3] = del_ @ W
    return J


# ── modified spherical coordinates (MSC) ────────────────────────────────────
# y = [az, el, w_a, w_e, s, L]
#   az, el : LOS angles in the pole frame          (THE MEASUREMENT, exactly)
#   w_a    : d(az)/dt * cos(el)  (physical angular rate component)
#   w_e    : d(el)/dt
#   s      : rho_dot / rho
#   L      : ln rho                                (the weak direction, isolated)
def msc_encode(sat, tgt, W):
    sat = np.asarray(sat, dtype=np.float64)
    tgt = np.asarray(tgt, dtype=np.float64)
    d = (tgt[..., :3] - sat[..., :3]) @ W.T
    dd = (tgt[..., 3:] - sat[..., 3:]) @ W.T
    rho2 = np.sum(d * d, axis=-1)
    rho2 = np.where(np.isfinite(rho2) & (rho2 > _RHO_FLOOR ** 2),
                    rho2, _RHO_FLOOR ** 2)
    rho = np.sqrt(rho2)
    x, y, z = d[..., 0], d[..., 1], d[..., 2]
    vx, vy, vz = dd[..., 0], dd[..., 1], dd[..., 2]
    rxy2 = np.maximum(x * x + y * y, 1e-30)
    rxy = np.sqrt(rxy2)
    az = np.arctan2(y, x)
    el = np.arcsin(np.clip(z / rho, -1.0, 1.0))
    s = (x * vx + y * vy + z * vz) / rho2
    azdot = (x * vy - y * vx) / rxy2
    eldot = (vz / rho - (z / rho) * s) / np.maximum(np.cos(el), 1e-12)
    w_a = azdot * np.cos(el)
    return np.stack([az, el, w_a, eldot, s, np.log(rho)], axis=-1)


def msc_decode(y, sat, W):
    y = np.asarray(y, dtype=np.float64)
    sat = np.asarray(sat, dtype=np.float64)
    az, el, w_a, w_e, s, L = (y[..., 0], y[..., 1], y[..., 2],
                              y[..., 3], y[..., 4], y[..., 5])
    L = np.where(np.isfinite(L), L, math.log(_RHO_FLOOR))
    rho = np.maximum(np.exp(np.minimum(L, _LOG_RHO_MAX)), _RHO_FLOOR)
    ca, sa = np.cos(az), np.sin(az)
    ce, se = np.cos(el), np.sin(el)
    u = np.stack([ce * ca, ce * sa, se], axis=-1)
    e_az = np.stack([-sa, ca, np.zeros_like(sa)], axis=-1)
    e_el = np.stack([-se * ca, -se * sa, ce], axis=-1)
    d = rho[..., None] * u
    dd = (s * rho)[..., None] * u + rho[..., None] * (
        w_a[..., None] * e_az + w_e[..., None] * e_el)
    R = sat[..., :3] + d @ W
    V = sat[..., 3:] + dd @ W
    return np.concatenate([R, V], axis=-1)


def _msc_dy(a, b):
    d = a - b
    d[..., 0] = wrap_pi(d[..., 0])
    d[..., 1] = wrap_pi(d[..., 1])
    return d


def msc_enc_jac(sat, xt, W):
    """d(y)/d(target inertial 6-state) by central differences. (N,6,6)."""
    sat = np.atleast_2d(sat)
    xt = np.atleast_2d(xt)
    n = xt.shape[0]
    G = np.empty((n, 6, 6))
    h = (H_POS, H_POS, H_POS, H_VEL, H_VEL, H_VEL)
    for j in range(6):
        xp = xt.copy(); xp[:, j] += h[j]
        xm = xt.copy(); xm[:, j] -= h[j]
        G[:, :, j] = _msc_dy(msc_encode(sat, xp, W),
                             msc_encode(sat, xm, W)) / (2.0 * h[j])
    return G


# ── geometry construction ───────────────────────────────────────────────────
def rotate_state(X, axis, ang):
    """Rotate a 6-state rigidly about `axis` (through the origin) by `ang`."""
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    c, s = math.cos(ang), math.sin(ang)
    K = np.array([[0.0, -axis[2], axis[1]],
                  [axis[2], 0.0, -axis[0]],
                  [-axis[1], axis[0], 0.0]])
    R = np.eye(3) * c + s * K + (1.0 - c) * np.outer(axis, axis)
    X = np.asarray(X, dtype=np.float64).reshape(6)
    return np.concatenate([R @ X[:3], R @ X[3:]])


def make_geometry(a_c, e_c, da, rho, f_oop, node_lead=math.pi / 2.0,
                  theta0=0.0, e_t=None, dtheta=None):
    """Chaser/target epoch states with a controlled in-plane / out-of-plane split.

    `rho` is the TOTAL epoch separation; a fraction `f_oop` of it is carried
    out-of-plane and the remainder along-track.  The chaser sits on
    (a_c + da, e_c) at true anomaly theta0 - dtheta with
    dtheta = rho*sqrt(1-f_oop^2)/a_c (override with `dtheta` for the far-field
    geometries where the small-angle map is invalid).  The target sits on
    (a_c, e_t) at theta0 in the SAME plane, then its plane is rotated by

        di = asin(f_oop * rho / r_t)

    about an axis lying in the reference plane `node_lead` radians behind the
    target, which puts the out-of-plane excursion at its maximum at epoch.

    f_oop = 0 reproduces the coplanar (2D) geometry EXACTLY — that is the
    control arm.  Returns (sat0, tgt0, di).
    """
    e_t = e_c if e_t is None else e_t
    tgt0 = coe2rv(a_c, e_t, 0.0, 0.0, 0.0, theta0)
    rho_ip = rho * math.sqrt(max(1.0 - f_oop * f_oop, 0.0))
    dth = (rho_ip / a_c) if dtheta is None else dtheta
    sat0 = coe2rv(a_c + da, e_c, 0.0, 0.0, 0.0, theta0 - dth)
    if f_oop <= 0.0:
        return sat0, tgt0, 0.0
    r_t = np.linalg.norm(tgt0[:3])
    di = math.asin(min(f_oop * rho / r_t, 0.999))
    ang = theta0 - node_lead
    axis = np.array([math.cos(ang), math.sin(ang), 0.0])
    return sat0, rotate_state(tgt0, axis, di), di


def roll_truth(sat0, tgt0, dt, n_steps, burns=()):
    """Propagate both states. `burns` = ((step_index, dv_prograde_m_s), ...).

    Returns (t[n+1], S[n+1,6], G[n+1,6]).
    """
    S = np.empty((n_steps + 1, 6))
    G = np.empty((n_steps + 1, 6))
    S[0] = sat0
    G[0] = tgt0
    bmap = {int(k): float(v) for k, v in burns}
    s = np.asarray(sat0, dtype=np.float64).reshape(1, 6)
    g = np.asarray(tgt0, dtype=np.float64).reshape(1, 6)
    for k in range(1, n_steps + 1):
        s = propagate3(s, dt, iters=12)[0]
        g = propagate3(g, dt, iters=12)[0]
        if k in bmap:
            v = s[0, 3:]
            s = s.copy()
            s[0, 3:] = v + bmap[k] * v / np.linalg.norm(v)
        S[k] = s[0]
        G[k] = g[0]
    return np.arange(n_steps + 1) * dt, S, G


# ── validation ──────────────────────────────────────────────────────────────
def _selftest(verbose=True):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import orbital_math3d as om3
    NAV = "/Users/pete/space_training/pufferlib"
    sys.path.insert(0, NAV)
    from pufferlib.ocean.orbital_nav import nav_math as nm2

    rng = np.random.default_rng(11)
    fails = []

    # 1. propagate3 vs the independent universal-variable oracle
    worst = 0.0
    for _ in range(60):
        a = R_EARTH + rng.uniform(300e3, 8000e3)
        e = rng.uniform(0.0, 0.30)
        inc = rng.uniform(0.0, 1.2)
        X = coe2rv(a, e, inc, rng.uniform(0, 6.28), rng.uniform(0, 6.28),
                   rng.uniform(-3.14, 3.14))
        for dt in (60.0, 600.0, 3600.0):
            Y = propagate3(X[None, :], dt, iters=24)[0][0]
            Yo = om3.propagate_universal(X, dt)
            err = np.linalg.norm(Y[:3] - Yo[:3]) / np.linalg.norm(Yo[:3])
            worst = max(worst, err)
    fails.append(("propagate3 vs oracle (rel pos)", worst, 1e-11))

    # 2. exact 2D reduction: coplanar 3D == shipped 2D propagator
    a = R_EARTH + 400e3
    X3 = coe2rv(a, 0.02, 0.0, 0.0, 0.3, 0.7)
    X2 = np.array([X3[0], X3[1], X3[3], X3[4]])
    Y3 = propagate3(X3[None, :], 600.0, iters=12)[0][0]
    Y2 = nm2.propagate_cartesian(X2[None, :], 600.0, iters=12)[0][0]
    d = max(abs(Y3[0] - Y2[0]), abs(Y3[1] - Y2[1]),
            abs(Y3[3] - Y2[2]) * 1e3, abs(Y3[4] - Y2[3]) * 1e3)
    fails.append(("coplanar 3D == shipped 2D propagator (m)", d, 1e-7))

    # 3. MSC encode/decode round trip
    sat, tgt, _ = make_geometry(a, 0.001, 2.0e3, 10.0e3, 0.3)
    W = pole_frame(sat)
    y = msc_encode(sat[None, :], tgt[None, :], W)
    xr = msc_decode(y, sat[None, :], W)[0]
    fails.append(("MSC round trip pos (m)", np.linalg.norm(xr[:3] - tgt[:3]), 1e-6))
    fails.append(("MSC round trip vel (m/s)", np.linalg.norm(xr[3:] - tgt[3:]), 1e-9))

    # 4. MSC first two components ARE the measurement
    az, el = los_azel(sat, tgt, W)
    fails.append(("MSC[0] == az", abs(y[0, 0] - az), 1e-14))
    fails.append(("MSC[1] == el", abs(y[0, 1] - el), 1e-14))

    # 5. coplanar case: el identically zero, MSC collapses to modified polar
    satc, tgtc, _ = make_geometry(a, 0.001, 2.0e3, 10.0e3, 0.0)
    Wc = pole_frame(satc)
    yc = msc_encode(satc[None, :], tgtc[None, :], Wc)
    fails.append(("coplanar el (rad)", abs(yc[0, 1]), 1e-15))
    fails.append(("coplanar el_dot (rad/s)", abs(yc[0, 3]), 1e-18))

    # 6. analytic azel Jacobian vs central differences
    J = azel_jac(sat[None, :], tgt[None, :], W)[0]
    Jn = np.zeros((2, 6))
    for j in range(6):
        xp = tgt.copy(); xp[j] += 1.0 if j < 3 else 1e-3
        xm = tgt.copy(); xm[j] -= 1.0 if j < 3 else 1e-3
        ap, ep = los_azel(sat, xp, W)
        am, em = los_azel(sat, xm, W)
        Jn[0, j] = wrap_pi(ap - am) / (2.0 * (1.0 if j < 3 else 1e-3))
        Jn[1, j] = wrap_pi(ep - em) / (2.0 * (1.0 if j < 3 else 1e-3))
    fails.append(("azel Jacobian vs FD (rel)",
                  np.max(np.abs(J - Jn)) / max(np.max(np.abs(Jn)), 1e-300), 1e-6))

    # 7. STM symplecticity
    F = stm_fd3(np.atleast_2d(X3), 600.0, iters=12, warm_iters=6)[0][0]
    Jsym = np.block([[np.zeros((3, 3)), np.eye(3)],
                     [-np.eye(3), np.zeros((3, 3))]])
    # (mixed-unit residual; the 1 m / 1e-3 m/s central-difference truncation
    # floor on a 7e6 m state is ~1e-6, matching the shipped 2D stm_fd)
    fails.append(("STM symplectic residual",
                  np.max(np.abs(F.T @ Jsym @ F - Jsym)), 1e-5))

    ok = True
    for name, val, thr in fails:
        good = np.isfinite(val) and val <= thr
        ok &= bool(good)
        if verbose:
            print(f"  [{'PASS' if good else 'FAIL'}] {name:48s} {val:.3e} <= {thr:.0e}")
    return ok


if __name__ == "__main__":
    pin_blas_threads()
    print("N3D-A core self-test")
    sys.exit(0 if _selftest() else 1)
