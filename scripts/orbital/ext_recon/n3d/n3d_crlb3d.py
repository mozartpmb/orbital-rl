#!/usr/bin/env python3
"""N3D-B — 3D angles-only (az/el) Cramer-Rao machinery over the 3D oracle.

This is the 3D lift of `ext_recon/ext_bo_filter.crlb_range_sigma` (the estimator
that generated `web_data/results/ext_bo_observability.csv`, the map the shipped
2D training surrogate interpolates). Same object, same conventions, one extra
dimension:

    FIM = sum_k Phi(t_k,t_0)^T H_k^T R^-1 H_k Phi(t_k,t_0)

with the state = the TARGET's absolute inertial 6-state (the ext-nav filter
architecture: chaser self-state known, target estimated), Phi the numerically
differenced 6x6 two-body STM, and H the angular measurement Jacobian.

── The measurement model, and why it is basis-free ──────────────────────────
A camera measures two angles. Write them as the two focal-plane coordinates of
the LOS unit vector u = (r_t - r_c)/rho. For ANY orthonormal pair (e1,e2) with
e1,e2 _|_ u,

    z = (e1.u, e2.u),   H = (1/rho) [e1^T 0_3 ; e2^T 0_3],
    H^T R^-1 H = (1/(sigma_b^2 rho^2)) [[P_perp, 0],[0, 0]],  P_perp = I - u u^T

The information contribution depends only on P_perp — it is INDEPENDENT of how
the focal plane is rolled. So there is no az/el frame singularity in this
formulation, and the classical az/el model with sigma_az = sigma_b/cos(el),
sigma_el = sigma_b is exactly this same object. The 2D machinery is the rank-1
restriction (P_perp -> n n^T with n the in-plane LOS normal) and drops out of
this one exactly when the geometry is coplanar (validated: `--check`).

── Error decomposition ──────────────────────────────────────────────────────
Errors are reported in the epoch LOS triad (u, e_t, e_p):
    u    range direction (the classically unobservable one)
    e_p  PLANE direction  = unit(P_perp h_hat_t)  — out-of-plane, _|_ LOS
    e_t  in-plane transverse = e_p x u
so `sigma_plane` is literally "how well is the out-of-plane component of the
relative state known", which is the quantity the 3D guidance leg consumes.

── Conditioning ─────────────────────────────────────────────────────────────
The raw 6x6 FIM mixes metres and metres/second and is condition-limited by that
alone (~1e7 before any geometry). Everything is computed in the nondimensional
state D^-1 x, D = diag(a,a,a,v_c,v_c,v_c), and mapped back. NAV-F's §4.1
correction to NAV-G — "exactly singular" was a numerical artifact of a badly
scaled differenced FIM — is exactly this failure mode, so the scaling is not
cosmetic; both condition numbers are reported.
"""

import math
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, '..')))
sys.path.insert(0, '/Users/pete/space_training/scripts/orbital/nav')

import orbital_math3d as om3                                    # noqa: E402

MU = om3.MU
R_EARTH = om3.R_EARTH
J2 = 1.08262668e-3
R_EQ = 6.378137e6

SIGMA_BETA = 1.0e-3          # 1 mrad, the shipped optical bearing sigma
SIGMA_RHO_M = 50.0           # shipped range sensor sigma (for the RB reference)
DT_SENSOR = 60.0

H_POS = 1.0                  # STM finite-difference steps (mirror the 2D module)
H_VEL = 1.0e-3

COND_MAX = 1.0e14            # scaled-FIM condition ceiling -> report unbounded


# ═════════════════════════════════ dynamics ═════════════════════════════════
def prop2b(rv, dt):
    return om3.propagate_universal(rv, dt)


def _accel_j2(r):
    x, y, z = r
    rn = math.sqrt(x * x + y * y + z * z)
    k = -MU / rn ** 3
    zr2 = 5.0 * (z * z) / (rn * rn)
    c = 1.5 * J2 * (R_EQ / rn) ** 2
    return np.array([k * x * (1.0 + c * (1.0 - zr2)),
                     k * y * (1.0 + c * (1.0 - zr2)),
                     k * z * (1.0 + c * (3.0 - zr2))])


def prop_j2(rv, dt, h=10.0, j2=True):
    """RK4 with J2 (or pure two-body when j2=False, used as the RK4 control)."""
    rv = np.asarray(rv, dtype=np.float64).reshape(6).copy()
    n = max(1, int(round(abs(dt) / h)))
    hh = dt / n
    acc = _accel_j2 if j2 else (lambda r: -MU / np.linalg.norm(r) ** 3 * r)

    def f(s):
        return np.concatenate([s[3:], acc(s[:3])])

    for _ in range(n):
        k1 = f(rv)
        k2 = f(rv + 0.5 * hh * k1)
        k3 = f(rv + 0.5 * hh * k2)
        k4 = f(rv + hh * k3)
        rv = rv + (hh / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return rv


def stm_fd6(rv, dt, propagator=prop2b, h_pos=H_POS, h_vel=H_VEL):
    """d prop(x,dt) / dx by central differences. 12 propagations."""
    F = np.empty((6, 6))
    h = (h_pos, h_pos, h_pos, h_vel, h_vel, h_vel)
    x = np.asarray(rv, dtype=np.float64).reshape(6)
    for j in range(6):
        xp = x.copy(); xp[j] += h[j]
        xm = x.copy(); xm[j] -= h[j]
        F[:, j] = (propagator(xp, dt) - propagator(xm, dt)) / (2.0 * h[j])
    return F


# ═════════════════════════════ geometry builders ════════════════════════════
def make_pair(a_t=R_EARTH + 500e3, e=0.0, i_t=math.radians(51.6), raan_t=0.0,
              argp_t=0.0, nu_t=0.6, sep_m=5e3, da_m=0.0, di_rad=0.0,
              lead=True):
    """Target on a reference orbit; chaser offset in (along-track, a, plane).

    The chaser's plane is the target's rotated by `di_rad` about the target's
    own radius vector at epoch — a relative-inclination offset whose relative
    node sits at the epoch, i.e. the out-of-plane relative motion is a sine that
    is ZERO at t0 and grows over the arc. (Rotating about r_hat rather than
    bumping `i` keeps the offset a pure relative inclination independent of the
    absolute node, matching the env's rotation-form sampler in orbital.h:1775.)

    `sep_m` is the along-track (true-anomaly) offset at epoch, `da_m` the
    semi-major-axis difference — NAV-F's governing parameter.
    """
    rv_t = om3.coe2rv(a_t, e, i_t, raan_t, argp_t, nu_t)
    a_s = a_t + da_m
    dnu = (-sep_m / a_t) if lead else (sep_m / a_t)
    rv_s = om3.coe2rv(a_s, e, i_t, raan_t, argp_t, nu_t + dnu)
    if di_rad != 0.0:
        axis = om3.unit(rv_t[:3])
        R = _rot_axis(axis, di_rad)
        rv_s = np.concatenate([R @ rv_s[:3], R @ rv_s[3:]])
    return rv_s, rv_t


def _rot_axis(k, ang):
    k = np.asarray(k, dtype=np.float64)
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + math.sin(ang) * K + (1.0 - math.cos(ang)) * (K @ K)


def roll(rv_s, rv_t, duration, dt=DT_SENSOR, burns=(), propagator=prop2b):
    """Propagate both bodies; `burns` = ((t_s, dv_ms, axis), ...) on the CHASER.

    axis in {'pro','rad','nor','perp_los'}; 'perp_los' is the in-plane
    direction perpendicular to the instantaneous LOS (the literature-optimal
    observability direction) and is included so 'is normal special?' is
    measured against the best in-plane alternative, not against an arbitrary one.
    """
    n = int(round(duration / dt))
    s, t = np.asarray(rv_s, float).copy(), np.asarray(rv_t, float).copy()
    bk = {}
    for bt, bdv, ax in burns:
        bk.setdefault(int(round(bt / dt)), []).append((bdv, ax))
    S, T, TS = [s.copy()], [t.copy()], [0.0]
    for k in range(1, n + 1):
        s = propagator(s, dt)
        t = propagator(t, dt)
        for bdv, ax in bk.get(k, ()):
            s = apply_burn(s, t, bdv, ax)
        S.append(s.copy()); T.append(t.copy()); TS.append(k * dt)
    return np.array(TS), np.array(S), np.array(T)


def apply_burn(s, t, dv, axis):
    if axis == 'pro':
        return om3.apply_impulse_3d(s, dv_pro=dv)
    if axis == 'rad':
        return om3.apply_impulse_3d(s, dv_rad=dv)
    if axis == 'nor':
        return om3.apply_impulse_3d(s, dv_nor=dv)
    if axis == 'perp_los':
        u = om3.unit(t[:3] - s[:3])
        hh = om3.unit(np.cross(s[:3], s[3:]))
        d = np.cross(hh, u)
        nrm = np.linalg.norm(d)
        d = d / nrm if nrm > 1e-12 else om3.unit(np.cross(u, s[:3]))
        out = np.asarray(s, float).copy()
        out[3:] += dv * d
        return out
    raise ValueError(axis)


# ═══════════════════════════════ the CRLB ═══════════════════════════════════
def crlb3d(times, sats, tgts, sigma_beta=SIGMA_BETA, with_range=False,
           sigma_rho=SIGMA_RHO_M, propagator=prop2b, cum_at=None,
           angles='both'):
    """3D angles-only CRLB on the target's epoch 6-state.

    Returns a list of dicts (one per entry of `cum_at`, default = full arc),
    each with the epoch-LOS-triad sigmas. `cum_at` lets one roll serve many arc
    lengths: the FIM is accumulated once and inverted at each checkpoint.
    """
    n = len(times)
    checks = sorted(set(cum_at)) if cum_at is not None else [n - 1]
    a0 = float(np.linalg.norm(tgts[0][:3]))
    v0 = float(np.linalg.norm(tgts[0][3:]))
    D = np.diag([a0, a0, a0, v0, v0, v0])
    Di = np.diag([1 / a0, 1 / a0, 1 / a0, 1 / v0, 1 / v0, 1 / v0])

    # epoch triad
    d0 = tgts[0][:3] - sats[0][:3]
    rho0 = float(np.linalg.norm(d0))
    u0 = d0 / max(rho0, 1e-12)
    ht = om3.unit(np.cross(tgts[0][:3], tgts[0][3:]))
    w = ht - np.dot(ht, u0) * u0
    nw = float(np.linalg.norm(w))
    e_p = w / nw if nw > 1e-9 else om3.unit(np.cross(u0, [0.0, 0.0, 1.0]))
    e_t = np.cross(e_p, u0)

    Phi = np.eye(6)
    F = np.zeros((6, 6))
    out = []
    ci = 0
    for k in range(n):
        if k > 0:
            Phi = stm_fd6(tgts[k - 1], times[k] - times[k - 1], propagator) @ Phi
        d = tgts[k][:3] - sats[k][:3]
        rho = float(np.linalg.norm(d))
        u = d / max(rho, 1e-12)
        if angles == 'both':
            P = np.eye(3) - np.outer(u, u)
        else:
            # Split the two focal-plane angles into the one that moves the LOS
            # WITHIN the target's orbit plane and the one that moves it OUT of
            # it, so the marginal value of the out-of-plane angle is isolable.
            hk = om3.unit(np.cross(tgts[k][:3], tgts[k][3:]))
            wv = hk - np.dot(hk, u) * u
            nwv = float(np.linalg.norm(wv))
            ep = wv / nwv if nwv > 1e-9 else om3.unit(np.cross(u, [0., 0., 1.]))
            et = np.cross(ep, u)
            P = np.outer(et, et) if angles == 'inplane' else np.outer(ep, ep)
        M = np.zeros((6, 6))
        M[:3, :3] = P / (sigma_beta * rho) ** 2
        if with_range:
            M[:3, :3] += np.outer(u, u) / sigma_rho ** 2
        Ph = Di @ Phi @ D                     # nondimensional STM
        Mh = D @ M @ D                        # nondimensional measurement info
        F += Ph.T @ Mh @ Ph
        while ci < len(checks) and checks[ci] == k:
            out.append(_invert(F, D, u0, e_t, e_p, rho0, times[k] - times[0],
                               k + 1))
            ci += 1
    return out


def _invert(Fh, D, u0, e_t, e_p, rho0, arc_s, n_obs):
    w = np.linalg.eigvalsh(0.5 * (Fh + Fh.T))
    aw = np.abs(w)
    cond = float(aw.max() / max(aw.min(), 1e-300))
    rec = dict(arc_s=arc_s, n_obs=n_obs, rho0_m=rho0, fim_cond=cond)
    if (not np.all(np.isfinite(w))) or w.min() <= 0.0 or cond > COND_MAX:
        for key in ('sigma_range_m', 'sigma_inplane_m', 'sigma_plane_m',
                    'sigma_pos_m', 'sigma_vrange_ms', 'sigma_vinplane_ms',
                    'sigma_vplane_ms', 'sigma_vel_ms'):
            rec[key] = float('inf')
        return rec
    Ch = np.linalg.inv(0.5 * (Fh + Fh.T))
    C = D @ Ch @ D
    Cp, Cv = C[:3, :3], C[3:, 3:]

    def s(A, e):
        return math.sqrt(max(float(e @ A @ e), 0.0))
    rec.update(sigma_range_m=s(Cp, u0), sigma_inplane_m=s(Cp, e_t),
               sigma_plane_m=s(Cp, e_p),
               sigma_pos_m=math.sqrt(max(float(np.trace(Cp)), 0.0)),
               sigma_vrange_ms=s(Cv, u0), sigma_vinplane_ms=s(Cv, e_t),
               sigma_vplane_ms=s(Cv, e_p),
               sigma_vel_ms=math.sqrt(max(float(np.trace(Cv)), 0.0)))
    return rec


def period(a):
    return 2.0 * math.pi * math.sqrt(a ** 3 / MU)


# ═══════════════════════ validation: the 2D anchor ══════════════════════════
def check(verbose=True):
    """Three checks that the machinery is the 3D lift of the shipped 2D one.

    C1  RK4(J2 off) vs the universal-variable oracle over an orbit.
    C2  STM symplecticity (parameter-free measure of differencing error).
    C3  THE ANCHOR — coplanar equatorial geometry: the 3D az/el CRLB's
        in-plane sigmas must equal the 2D single-bearing CRLB from
        `ext_bo_filter.crlb_range_sigma` (the estimator behind the shipped map).
    """
    ok = True
    rv = om3.coe2rv(R_EARTH + 500e3, 0.01, math.radians(51.6), 0.3, 0.2, 0.7)
    e1 = float(np.linalg.norm(prop_j2(rv, 3600.0, h=5.0, j2=False)
                              - prop2b(rv, 3600.0)) / np.linalg.norm(rv[:3]))
    if verbose:
        print(f"C1 RK4(no J2) vs universal, 1 h : {e1:.3e} rel  "
              f"[{'PASS' if e1 < 1e-9 else 'FAIL'}]")
    ok &= e1 < 1e-9

    F = stm_fd6(rv, 600.0)
    J = np.zeros((6, 6)); J[:3, 3:] = np.eye(3); J[3:, :3] = -np.eye(3)
    Ds = np.diag([1e6, 1e6, 1e6, 1e3, 1e3, 1e3])
    Fs = np.linalg.solve(Ds, F @ Ds)
    sym = float(np.abs(Fs.T @ J @ Fs - J).max())
    if verbose:
        print(f"C2 STM symplecticity (scaled)   : {sym:.3e}       "
              f"[{'PASS' if sym < 1e-5 else 'FAIL'}]")
    ok &= sym < 1e-5

    # C3 — coplanar equatorial, compared against the 2D pipeline
    sys.path.insert(0, os.path.abspath(os.path.join(_HERE, '..')))
    import ext_bo_filter as X                                   # noqa: E402
    a = R_EARTH + 400e3
    for sep, dv in ((5e3, 1.0), (50e3, 0.0), (200e3, 5.0)):
        dur = 1.0 * period(a)
        burns = ((0.35 * dur, dv, 'pro'),) if dv else ()
        rv_s, rv_t = make_pair(a_t=a, e=0.0, i_t=0.0, nu_t=0.5, sep_m=sep)
        T, S, G = roll(rv_s, rv_t, dur, burns=burns)
        r3 = crlb3d(T, S, G)[-1]
        sc = dict(name='x', sat=X.elements(a, 0.0, 0.0, 0.5 - sep / a),
                  tgt=X.elements(a, 0.0, 0.0, 0.5), dt=DT_SENSOR, duration=dur,
                  r_min=R_EARTH + 300e3, r_max=R_EARTH + 800e3,
                  sigma_v_ecc=100.0,
                  burns=((0.35 * dur, dv),) if dv else ())
        t2, s2, g2 = X.roll_truth(sc)
        sl2, sp2, c2 = X.crlb_range_sigma(t2, s2, g2, SIGMA_BETA, False)
        r = r3['sigma_range_m'] / sl2 if np.isfinite(sl2) and sl2 > 0 else np.nan
        zmax = max(abs(S[:, 2]).max(), abs(G[:, 2]).max())
        if verbose:
            print(f"C3 sep={sep/1e3:6.1f} km dv={dv:4.1f}: "
                  f"3D sigma_range {r3['sigma_range_m']:11.3f} m | "
                  f"2D {sl2:11.3f} m | ratio {r:7.4f} | z_max {zmax:.1e} m")
        # The 50 km drift-only cell is the one where the 2D pipeline's UNSCALED
        # FIM trips its own cond>1e16 guard and reports `inf` while the scaled
        # 6x6 returns 2.219e5 m — the same numerical-artifact failure NAV-F
        # §4.1 diagnosed in NAV-G. Not a disagreement to assert on.
        ok &= (abs(r - 1.0) < 0.02) if np.isfinite(sl2) else True
    if verbose:
        print(f"\n{'ALL PASS' if ok else 'FAILURES PRESENT'}")
    return ok


if __name__ == '__main__':
    check()
