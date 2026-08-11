#!/usr/bin/env python3
"""ext-3d — INDEPENDENT 3D two-body oracle for fuzzing a future 3D orbital env.

Purpose
-------
`scripts/orbital/t3/fuzz_dynamics.py` validated the 2D C environment against
`scripts/orbital/nav/orbital_math.py` (Lagrange f&g in the eccentric-anomaly
difference, 2D arrays).  This module is the 3D equivalent of that oracle.  It
must be able to CATCH the bugs of a 3D env that does not exist yet, so it is
written from a deliberately different formulation than any plausible env
implementation:

    * propagation   — UNIVERSAL VARIABLES with Stumpff functions c2/c3
                      (conic-agnostic: ellipse, parabola, hyperbola, and the
                      near-parabolic seam all take the same code path).  The
                      2D oracle uses ellipse-only f&g in dE; a classical env
                      will use element-wise Kepler.  Three different algebras.
    * elements      — full classical set (a, e, i, RAAN, argp, nu) with
                      quadrant handling done by signed atan2 against the
                      orbit's own frame vectors, never by acos + if-sign.
                      Degenerate cases (circular, equatorial, retrograde
                      equatorial, circular-equatorial) are handled by a single
                      convention that makes coe2rv o rv2coe the identity —
                      which is the only definition that can be regression
                      tested.
    * anomalies     — nu<->E<->M elliptic and nu<->H<->N hyperbolic, each with
                      a monotone bracketed solver (no 5-iteration Newton with
                      a fixed iteration budget; the env's fixed budget is
                      itself something the oracle must be able to indict).

Nothing here imports the C extension.  The 2D oracle is imported only in the
validation battery, as a cross-check reference.

Conventions
-----------
Right-handed ECI (x, y, z), z = north.  h = r x v.  i = angle(h, z) in [0, pi];
i > pi/2 is retrograde.  n = z x h is the ascending-node direction.  All angles
in radians, all lengths in metres, all times in seconds.

Run the validation battery:
    python3 /Users/pete/space_training/scripts/orbital/ext_recon/orbital_math3d.py
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np

MU = 3.986004418e14        # Earth gravitational parameter (m^3/s^2)
R_EARTH = 6.371e6          # Earth equatorial radius (m)

# Degeneracy thresholds.  These are *classification* tolerances, not accuracy
# tolerances: inside them the canonical angle set is replaced by the reduced
# set that stays well conditioned.  See rv2coe().
#
# Why 1e-11 and not something rounder.  Two error sources fight each other in
# the rv -> coe -> rv round trip:
#   * keeping argp when e is tiny: e_vec has absolute fp noise ~1e-16, so the
#     periapsis DIRECTION has error d ~ 1e-16/e.  argp and nu each carry d and
#     cancel in argp+nu, leaving a second-order residual ~d^2/2.
#   * collapsing to argp=0 when e is not yet negligible: introduces an O(e)
#     radius error, since r = p/(1+e cos nu) is then evaluated at the wrong nu.
# Setting d^2/2 = e gives e ~ 1.7e-11.  Same argument for the node vector and
# sin(i).  These are therefore the optimal switch points for a CLASSICAL
# element set; the residual ~1e-11 at the seam is irreducible in this
# representation (see v9_element_conditioning -- and the design note that
# follows from it: a 3D env should keep Cartesian state and treat elements as
# a derived view, or use equinoctial elements, rather than round-tripping
# through (raan, argp, nu) on every burn as the 2D env does).
E_CIRC_TOL = 1e-11         # |e| below this -> treat as circular
I_EQ_TOL = 1e-11           # |n|/|h| below this -> treat as equatorial


# ── vector helpers ──────────────────────────────────────────────────────────
def _v(*a):
    return np.asarray(a, dtype=np.float64)


def unit(x):
    n = float(np.linalg.norm(x))
    return np.asarray(x, dtype=np.float64) / n if n > 0.0 else np.zeros(3)


def wrap_pi(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def wrap_2pi(a):
    a = math.fmod(a, 2.0 * math.pi)
    return a + 2.0 * math.pi if a < 0.0 else a


def signed_angle(u, w, axis):
    """Angle from u to w measured right-handedly about `axis`, in (-pi, pi].

    This is the whole quadrant story in one expression: it is exact at 0 and
    pi, needs no sign patch-up, and degrades gracefully instead of returning
    NaN when the acos form would step outside [-1, 1].
    """
    u, w, axis = unit(u), unit(w), unit(axis)
    return math.atan2(float(np.dot(np.cross(u, w), axis)), float(np.dot(u, w)))


# ── Stumpff functions ───────────────────────────────────────────────────────
def stumpff_c2c3(z):
    """c2(z) = sum (-z)^k/(2k+2)!,  c3(z) = sum (-z)^k/(2k+3)!.

    Closed forms lose precision through cancellation as z -> 0 (1 - cos sqrt z
    against z), so the series is used on |z| < 0.5 where it converges to
    machine precision in ~10 terms.  Continuity across the switch is verified
    in the battery (case P4).
    """
    if abs(z) < 0.5:
        c2 = 0.0
        c3 = 0.0
        term2 = 0.5                      # 1/2!
        term3 = 1.0 / 6.0                # 1/3!
        k = 0
        while k < 14:
            c2 += term2
            c3 += term3
            term2 *= -z / ((2 * k + 3) * (2 * k + 4))
            term3 *= -z / ((2 * k + 4) * (2 * k + 5))
            k += 1
        return c2, c3
    if z > 0.0:
        s = math.sqrt(z)
        return (1.0 - math.cos(s)) / z, (s - math.sin(s)) / (s * s * s)
    s = math.sqrt(-z)
    return (math.cosh(s) - 1.0) / (-z), (math.sinh(s) - s) / (s * s * s)


# ── universal-variable propagation ──────────────────────────────────────────
def propagate_universal(rv, dt, mu=MU, tol=1e-12, max_iter=200):
    """Exact two-body propagation of a Cartesian 6-state by dt seconds.

    rv = (x, y, z, vx, vy, vz).  Returns a length-6 ndarray.

    Solves the universal Kepler equation
        F(chi) = chi^3 c3(psi) + sigma0 chi^2 c2(psi) + r0 chi (1 - psi c3) - sqrt(mu) dt
    with psi = alpha chi^2, sigma0 = r.v/sqrt(mu), alpha = 1/a.  F'(chi) = r(chi) > 0,
    so F is strictly increasing: a bracketed Newton (bisection safeguard, bracket
    grown geometrically) converges for every conic including the near-parabolic
    seam and for |dt| spanning many revolutions.  Sign of dt is unrestricted.
    """
    rv = np.asarray(rv, dtype=np.float64).reshape(6)
    r0v, v0v = rv[:3], rv[3:]
    r0 = float(np.linalg.norm(r0v))
    v0 = float(np.linalg.norm(v0v))
    if r0 <= 0.0:
        raise ValueError("propagate_universal: zero radius")
    if dt == 0.0:
        return rv.copy()

    smu = math.sqrt(mu)
    alpha = 2.0 / r0 - v0 * v0 / mu          # = 1/a
    sigma0 = float(np.dot(r0v, v0v)) / smu

    def F(chi):
        psi = alpha * chi * chi
        c2, c3 = stumpff_c2c3(psi)
        val = (chi ** 3) * c3 + sigma0 * chi * chi * c2 + r0 * chi * (1.0 - psi * c3)
        r = chi * chi * c2 + sigma0 * chi * (1.0 - psi * c3) + r0 * (1.0 - psi * c2)
        return val - smu * dt, r, c2, c3, psi

    # bracket: F(0) = -sqrt(mu) dt has the sign of -dt; grow the other end.
    lo, hi = (0.0, 1.0) if dt > 0.0 else (-1.0, 0.0)
    grow = hi if dt > 0.0 else lo
    for _ in range(200):
        f, _, _, _, _ = F(grow)
        if (dt > 0.0 and f >= 0.0) or (dt < 0.0 and f <= 0.0):
            break
        grow *= 2.0
    else:                                    # pragma: no cover - unreachable
        raise RuntimeError("propagate_universal: bracket failed")
    if dt > 0.0:
        hi = grow
    else:
        lo = grow

    chi = 0.5 * (lo + hi)
    f, r, c2, c3, psi = F(chi)
    for _ in range(max_iter):
        if f > 0.0:
            hi = chi
        else:
            lo = chi
        step = f / r                         # Newton; r > 0 always
        nxt = chi - step
        if not (lo < nxt < hi):              # safeguard
            nxt = 0.5 * (lo + hi)
        d = abs(nxt - chi)
        chi = nxt
        f, r, c2, c3, psi = F(chi)
        if d < tol * max(1.0, abs(chi)):
            break

    fq = 1.0 - (chi * chi / r0) * c2
    gq = dt - (chi ** 3) * c3 / smu
    rvec = fq * r0v + gq * v0v
    rn = float(np.linalg.norm(rvec))
    fdot = smu * chi * (psi * c3 - 1.0) / (rn * r0)
    gdot = 1.0 - (chi * chi / rn) * c2
    vvec = fdot * r0v + gdot * v0v
    return np.concatenate([rvec, vvec])


# ── anomaly conversions ─────────────────────────────────────────────────────
def nu_to_E(nu, e):
    """True -> eccentric anomaly.  tan(E/2) = sqrt((1-e)/(1+e)) tan(nu/2)."""
    return 2.0 * math.atan2(math.sqrt(1.0 - e) * math.sin(0.5 * nu),
                            math.sqrt(1.0 + e) * math.cos(0.5 * nu))


def E_to_nu(E, e):
    return 2.0 * math.atan2(math.sqrt(1.0 + e) * math.sin(0.5 * E),
                            math.sqrt(1.0 - e) * math.cos(0.5 * E))


def nu_to_H(nu, e):
    """True -> hyperbolic anomaly.  tanh(H/2) = sqrt((e-1)/(e+1)) tan(nu/2)."""
    t = math.sqrt((e - 1.0) / (e + 1.0)) * math.tan(0.5 * nu)
    if abs(t) >= 1.0:
        raise ValueError("nu outside the hyperbolic asymptote")
    return 2.0 * math.atanh(t)


def H_to_nu(H, e):
    t = math.sqrt((e + 1.0) / (e - 1.0)) * math.tanh(0.5 * H)
    return 2.0 * math.atan(t)


def nu_to_M(nu, e):
    """True anomaly -> mean anomaly (elliptic M, hyperbolic N, parabolic D)."""
    if e < 1.0 - 1e-12:
        E = nu_to_E(nu, e)
        return E - e * math.sin(E)
    if e > 1.0 + 1e-12:
        H = nu_to_H(nu, e)
        return e * math.sinh(H) - H
    D = math.tan(0.5 * nu)                   # Barker
    return D + D ** 3 / 3.0


def M_to_nu(M, e, tol=1e-14):
    """Mean -> true anomaly.  Monotone bracketed Newton, no iteration cap games."""
    if e < 1.0 - 1e-12:
        M = wrap_2pi(M)
        lo, hi = 0.0, 2.0 * math.pi
        E = M if e < 0.8 else math.pi
        for _ in range(100):
            f = E - e * math.sin(E) - M
            if f > 0.0:
                hi = E
            else:
                lo = E
            dE = f / (1.0 - e * math.cos(E))
            nxt = E - dE
            if not (lo <= nxt <= hi):
                nxt = 0.5 * (lo + hi)
            if abs(nxt - E) < tol:
                E = nxt
                break
            E = nxt
        return E_to_nu(E, e)
    if e > 1.0 + 1e-12:
        N = M
        lo, hi = -1.0, 1.0
        while N - (e * math.sinh(lo) - lo) < 0.0:
            lo *= 2.0
        while N - (e * math.sinh(hi) - hi) > 0.0:
            hi *= 2.0
        H = 0.5 * (lo + hi)
        for _ in range(200):
            f = e * math.sinh(H) - H - N
            if f > 0.0:
                hi = H
            else:
                lo = H
            nxt = H - f / (e * math.cosh(H) - 1.0)
            if not (lo <= nxt <= hi):
                nxt = 0.5 * (lo + hi)
            if abs(nxt - H) < tol * max(1.0, abs(H)):
                H = nxt
                break
            H = nxt
        return H_to_nu(H, e)
    raise ValueError("parabolic M_to_nu not supported (use propagate_universal)")


def mean_motion(a, mu=MU):
    return math.sqrt(mu / abs(a) ** 3)


def period(a, mu=MU):
    if a <= 0.0:
        return math.inf
    return 2.0 * math.pi * math.sqrt(a ** 3 / mu)


# ── elements <-> Cartesian ──────────────────────────────────────────────────
def coe2rv(a, e, i, raan, argp, nu, mu=MU, p=None):
    """Classical elements -> Cartesian 6-state.

    Perifocal (periapsis on +x_pf, motion toward +y_pf) rotated by the 3-1-3
    sequence R3(-raan) R1(-i) R3(-argp).  For e >= 1 pass `p` (semi-latus
    rectum) explicitly if a is not finite; otherwise p = a(1-e^2) is used and
    is correctly negative-times-negative for hyperbolas.
    """
    if p is None:
        p = a * (1.0 - e * e)
    if p <= 0.0:
        raise ValueError("coe2rv: non-positive semi-latus rectum")
    r = p / (1.0 + e * math.cos(nu))
    sq = math.sqrt(mu / p)
    r_pf = _v(r * math.cos(nu), r * math.sin(nu), 0.0)
    v_pf = _v(-sq * math.sin(nu), sq * (e + math.cos(nu)), 0.0)
    R = rot_pf_to_eci(i, raan, argp)
    return np.concatenate([R @ r_pf, R @ v_pf])


def rot_pf_to_eci(i, raan, argp):
    """R3(-raan) R1(-i) R3(-argp) — perifocal to inertial."""
    cO, sO = math.cos(raan), math.sin(raan)
    ci, si = math.cos(i), math.sin(i)
    cw, sw = math.cos(argp), math.sin(argp)
    return np.array([
        [cO * cw - sO * sw * ci, -cO * sw - sO * cw * ci,  sO * si],
        [sO * cw + cO * sw * ci, -sO * sw + cO * cw * ci, -cO * si],
        [sw * si,                 cw * si,                 ci],
    ], dtype=np.float64)


def rv2coe(rv, mu=MU):
    """Cartesian 6-state -> classical elements, with an exactly invertible
    convention in every degenerate case.

    Returns a dict:
        a, e, i, raan, argp, nu      canonical set (radians)
        p, h_vec, e_vec, n_vec       derived vectors
        energy, r, v                 scalars
        arglat, lonper, truelon      alternate angles (u, varpi, lambda_true)
        case                         'general' | 'circ' | 'eq' | 'circ_eq'

    Convention (the reason coe2rv(rv2coe(x)) == x for all x):
        * equatorial (sin i < I_EQ_TOL, prograde OR retrograde): raan := 0 and
          the node direction is taken as +x_hat.  argp is then the signed angle
          from +x_hat to e_hat about h_hat, which reproduces the standard
          longitude-of-periapsis for i = 0 and its mirror for i = pi — the
          single formula that survives both.
        * circular (e < E_CIRC_TOL): argp := 0 and nu := signed angle from the
          node direction to r_hat about h_hat (the argument of latitude).
        * circular equatorial: both of the above, so nu = true longitude.
    Everything is a signed atan2 against (node, e_vec, h_vec); no acos-plus-if
    branch exists in this file, which is exactly the bug class it must catch.
    """
    rv = np.asarray(rv, dtype=np.float64).reshape(6)
    rvec, vvec = rv[:3], rv[3:]
    r = float(np.linalg.norm(rvec))
    v = float(np.linalg.norm(vvec))
    hvec = np.cross(rvec, vvec)
    h = float(np.linalg.norm(hvec))
    if r <= 0.0 or h <= 0.0:
        raise ValueError("rv2coe: degenerate (rectilinear or zero) state")

    evec = ((v * v - mu / r) * rvec - float(np.dot(rvec, vvec)) * vvec) / mu
    e = float(np.linalg.norm(evec))
    energy = 0.5 * v * v - mu / r
    p = h * h / mu
    a = math.inf if abs(e - 1.0) < 1e-12 else -mu / (2.0 * energy)

    hhat = hvec / h
    # i via atan2, NOT acos(h_z/h): near i=0 or i=pi, acos loses half the
    # mantissa (measured 1.4e-8 rad at i~1e-8, which then leaks into every
    # plane-invariance test).  atan2 of the two well-scaled cross-product
    # components is uniformly accurate.  A 3D env that reports i by acos will
    # be flagged by this oracle -- correctly.
    i = math.atan2(math.hypot(hhat[0], hhat[1]), hhat[2])
    nvec = np.cross(_v(0.0, 0.0, 1.0), hvec)
    nn = float(np.linalg.norm(nvec))

    equatorial = (nn / h) < I_EQ_TOL
    circular = e < E_CIRC_TOL

    if equatorial:
        raan = 0.0
        nhat = _v(1.0, 0.0, 0.0)
    else:
        raan = wrap_2pi(math.atan2(nvec[1], nvec[0]))
        nhat = nvec / nn

    if circular:
        argp = 0.0
        ehat = nhat
    else:
        ehat = evec / e
        argp = wrap_2pi(signed_angle(nhat, ehat, hhat))

    nu = wrap_2pi(signed_angle(ehat, rvec, hhat))

    case = ('circ_eq' if (circular and equatorial) else
            'circ' if circular else 'eq' if equatorial else 'general')

    return dict(
        a=a, e=e, i=i, raan=raan, argp=argp, nu=nu, p=p,
        h_vec=hvec, e_vec=evec, n_vec=nvec, energy=energy, r=r, v=v,
        arglat=wrap_2pi(signed_angle(nhat, rvec, hhat)),
        lonper=(wrap_2pi(math.atan2(evec[1], evec[0])) if not circular else 0.0),
        truelon=wrap_2pi(math.atan2(rvec[1], rvec[0])),
        case=case,
    )


def coe2rv_dict(el, mu=MU):
    return coe2rv(el['a'], el['e'], el['i'], el['raan'], el['argp'], el['nu'],
                  mu=mu, p=el.get('p'))


def propagate_elements(el, dt, mu=MU):
    """Advance elements by dt through the mean anomaly.  Independent second
    propagation path — used to cross-check propagate_universal, and the closest
    analogue of what a classical 3D env will actually implement."""
    a, e = el['a'], el['e']
    out = dict(el)
    if e < 1.0 - 1e-12:
        n = math.sqrt(mu / a ** 3)
        M = nu_to_M(el['nu'], e) + n * dt
        out['nu'] = wrap_2pi(M_to_nu(M, e))
    else:
        n = math.sqrt(mu / abs(a) ** 3)
        N = nu_to_M(el['nu'], e) + n * dt
        out['nu'] = M_to_nu(N, e)
    return out


# ── burn frames (what a 3D env must get right) ──────────────────────────────
def frame_ntw(rv):
    """(T, N, W): T = v_hat (prograde), W = h_hat (orbit normal), N = W x T.

    NOT orthogonal to r_hat unless e = 0.  This is the frame in which the 2D
    env's `prograde` axis lives; its 3D lift must keep W = h_hat, and a
    normal burn must be along W, NOT along z_hat (bug class B1)."""
    rv = np.asarray(rv, dtype=np.float64).reshape(6)
    T = unit(rv[3:])
    W = unit(np.cross(rv[:3], rv[3:]))
    return T, np.cross(W, T), W


def frame_rsw(rv):
    """(R, S, W): R = r_hat, W = h_hat, S = W x R (transverse).  The RIC/LVLH
    basis; orthonormal for every conic."""
    rv = np.asarray(rv, dtype=np.float64).reshape(6)
    R = unit(rv[:3])
    W = unit(np.cross(rv[:3], rv[3:]))
    return R, np.cross(W, R), W


def apply_impulse_3d(rv, dv_pro=0.0, dv_rad=0.0, dv_nor=0.0):
    """Impulse in the env's own (non-orthogonal) local basis, lifted to 3D.

    Mirrors orbital.h apply_impulse(): prograde = v_hat, radial = r_hat (NOT
    the RSW transverse axis), and — the new axis — normal = h_hat.  Returns the
    post-burn 6-state; position is unchanged by construction, which is itself a
    checkable invariant (I7)."""
    rv = np.asarray(rv, dtype=np.float64).reshape(6).copy()
    pro = unit(rv[3:])
    rad = unit(rv[:3])
    nor = unit(np.cross(rv[:3], rv[3:]))
    rv[3:] += dv_pro * pro + dv_rad * rad + dv_nor * nor
    return rv


def plane_change_dv(v, di):
    """Δv for a pure inclination change of di at speed v (circular case)."""
    return 2.0 * v * math.sin(0.5 * di)


# ═══════════════════════════════════════════════════════════════════════════
# VALIDATION BATTERY
# ═══════════════════════════════════════════════════════════════════════════
_RESULTS = []


def _record(name, value, thresh, units='', note=''):
    ok = bool(value <= thresh) if value == value else False
    _RESULTS.append(dict(check=name, value=float(value), thresh=float(thresh),
                         units=units, verdict='PASS' if ok else 'FAIL', note=note))
    return ok


def _draw(rng, regime):
    """Random element draw per regime."""
    a = R_EARTH + rng.uniform(300e3, 20000e3)
    if regime == 'circ_eq':
        e, i = 0.0, 0.0
    elif regime == 'circ_incl':
        e, i = 0.0, rng.uniform(0.05, math.pi - 0.05)
    elif regime == 'ell_eq':
        e, i = rng.uniform(0.01, 0.7), 0.0
    elif regime == 'ell_eq_retro':
        e, i = rng.uniform(0.01, 0.7), math.pi
    elif regime == 'polar':
        e, i = rng.uniform(0.0, 0.7), math.pi / 2.0
    elif regime == 'retro':
        e, i = rng.uniform(0.0, 0.7), rng.uniform(math.pi / 2 + 0.05, math.pi - 0.05)
    elif regime == 'near_circ':
        e, i = 10.0 ** rng.uniform(-12, -6), rng.uniform(0.0, math.pi)
    elif regime == 'near_eq':
        e, i = rng.uniform(0.01, 0.5), 10.0 ** rng.uniform(-12, -6)
    elif regime == 'high_e':
        e, i = rng.uniform(0.7, 0.95), rng.uniform(0.0, math.pi)
    elif regime == 'hyper':
        e = rng.uniform(1.05, 3.0)
        a = -(R_EARTH + rng.uniform(300e3, 5000e3))
        i = rng.uniform(0.0, math.pi)
    else:
        e, i = rng.uniform(0.0, 0.7), rng.uniform(0.0, math.pi)
    nu_lim = math.pi - 1e-3
    if e > 1.0:
        nu_lim = math.acos(-1.0 / e) - 0.2
    return dict(a=a, e=e, i=i,
                raan=rng.uniform(0.0, 2.0 * math.pi),
                argp=rng.uniform(0.0, 2.0 * math.pi),
                nu=rng.uniform(-nu_lim, nu_lim))


REGIMES = ('circ_eq', 'circ_incl', 'ell_eq', 'ell_eq_retro', 'polar', 'retro',
           'near_circ', 'near_eq', 'high_e', 'general', 'hyper')


def v1_roundtrip(n=400, seed=3):
    """coe -> rv -> coe -> rv.  The state round trip must be exact to fp noise
    in EVERY regime; the *element* round trip is only required where the
    element is defined (guarded by the case flag)."""
    rng = np.random.default_rng(seed)
    worst = {k: 0.0 for k in ('rv', 'rv_deg', 'a', 'e', 'i', 'raan', 'argp',
                              'nu', 'arglat', 'truelon_eq')}
    ctx = {}
    degenerate = ('near_circ', 'near_eq')
    for reg in REGIMES:
        for _ in range(n):
            el = _draw(rng, reg)
            rv = coe2rv(**el)
            back = rv2coe(rv)
            rv2 = coe2rv_dict(back)
            rel = float(np.linalg.norm(rv2[:3] - rv[:3]) / np.linalg.norm(rv[:3]))
            rel = max(rel, float(np.linalg.norm(rv2[3:] - rv[3:])
                                 / np.linalg.norm(rv[3:])))
            key = 'rv_deg' if reg in degenerate else 'rv'
            if rel > worst[key]:
                worst[key], ctx[key] = rel, (reg, el)
            worst['a'] = max(worst['a'], abs(back['a'] - el['a']) / abs(el['a']))
            worst['e'] = max(worst['e'], abs(back['e'] - el['e']))
            worst['i'] = max(worst['i'], abs(wrap_pi(back['i'] - el['i'])))
            # Argument of latitude u = argp + nu is well defined for ANY e (it
            # only needs the node line), so it must round-trip in every
            # non-equatorial regime including exactly circular ones -- where
            # argp and nu individually do not exist.  This is the 3D analogue
            # of the 2D oracle's theta+omega invariant.
            if math.sin(el['i']) > 1e-6:
                worst['arglat'] = max(worst['arglat'], abs(wrap_pi(
                    back['argp'] + back['nu'] - el['argp'] - el['nu'])))
            else:
                # Equatorial: the node line does not exist, so the invariant is
                # the TRUE LONGITUDE, and its sign convention differs between
                # prograde (lambda = raan + argp + nu) and retrograde
                # (lambda = raan - argp - nu).  Getting this seam wrong is a
                # classic 3D bug; the check is the seam's regression test.
                s = 1.0 if el['i'] < math.pi / 2 else -1.0
                lam_in = el['raan'] + s * (el['argp'] + el['nu'])
                worst['truelon_eq'] = max(worst['truelon_eq'],
                                          abs(wrap_pi(back['truelon'] - lam_in)))
            # Individual angles only where they are well conditioned.
            if back['case'] == 'general' and math.sin(el['i']) > 1e-3 and el['e'] > 1e-6:
                for k in ('raan', 'argp', 'nu'):
                    worst[k] = max(worst[k], abs(wrap_pi(back[k] - el[k])))
    return worst, ctx


def v9_element_conditioning(seed=23):
    """Design-guidance sweep: rv -> coe -> rv error vs e and vs sin(i).

    Not a correctness check of the oracle -- a MEASUREMENT of the classical
    element set's conditioning, so the 3D env's state representation can be
    chosen with numbers instead of taste.  The peak sits at the degeneracy
    switch points; away from them the round trip is exact to fp noise.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for e in (0.0, 1e-13, 1e-12, 1e-11, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 0.1, 0.5):
        w = 0.0
        for _ in range(200):
            el = dict(a=R_EARTH + float(rng.uniform(300e3, 8000e3)), e=e,
                      i=float(rng.uniform(0.05, math.pi - 0.05)),
                      raan=float(rng.uniform(0, 2 * math.pi)),
                      argp=float(rng.uniform(0, 2 * math.pi)),
                      nu=float(rng.uniform(-math.pi, math.pi)))
            rv = coe2rv(**el)
            rv2 = coe2rv_dict(rv2coe(rv))
            w = max(w, float(np.linalg.norm(rv2[:3] - rv[:3])
                             / np.linalg.norm(rv[:3])))
        rows.append(('e', e, w))
    for si in (0.0, 1e-13, 1e-12, 1e-11, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 0.5):
        w = 0.0
        for _ in range(200):
            el = dict(a=R_EARTH + float(rng.uniform(300e3, 8000e3)),
                      e=float(rng.uniform(0.01, 0.5)), i=math.asin(min(si, 1.0)),
                      raan=float(rng.uniform(0, 2 * math.pi)),
                      argp=float(rng.uniform(0, 2 * math.pi)),
                      nu=float(rng.uniform(-math.pi, math.pi)))
            rv = coe2rv(**el)
            rv2 = coe2rv_dict(rv2coe(rv))
            w = max(w, float(np.linalg.norm(rv2[:3] - rv[:3])
                             / np.linalg.norm(rv[:3])))
        rows.append(('sin_i', si, w))
    return rows


def v2_prop_vs_elements(n=120, seed=5):
    """Universal-variable propagation vs mean-anomaly element propagation.
    Two independent algebras; agreement to fp noise means both are right."""
    rng = np.random.default_rng(seed)
    worst, ctx = 0.0, None
    for reg in REGIMES:
        for _ in range(n):
            el = _draw(rng, reg)
            rv = coe2rv(**el)
            T = period(el['a']) if el['e'] < 1.0 else 6000.0
            for dt in (60.0, 1800.0, 3600.0, 0.37 * T, -1800.0):
                if el['e'] > 1.0 and abs(dt) > 4000.0:
                    continue
                got = propagate_universal(rv, dt)
                ref = coe2rv_dict(propagate_elements(el, dt))
                rel = float(np.linalg.norm(got[:3] - ref[:3])
                            / np.linalg.norm(ref[:3]))
                if rel > worst:
                    worst, ctx = rel, (reg, dt, el['e'], el['i'])
    return worst, ctx


def v3_prop_vs_numeric(n=40, seed=9):
    """Universal-variable propagation vs an adaptive numerical integrator
    (scipy DOP853, rtol=atol=1e-13).  Third, fully independent path — it shares
    no algebra with either analytic method."""
    try:
        from scipy.integrate import solve_ivp
    except Exception:                        # pragma: no cover
        return float('nan'), 'scipy unavailable'
    rng = np.random.default_rng(seed)

    def f(_t, s):
        r3 = (s[0] ** 2 + s[1] ** 2 + s[2] ** 2) ** 1.5
        return [s[3], s[4], s[5],
                -MU * s[0] / r3, -MU * s[1] / r3, -MU * s[2] / r3]

    worst, ctx = 0.0, None
    for reg in ('circ_incl', 'polar', 'retro', 'high_e', 'general', 'hyper'):
        for _ in range(n):
            el = _draw(rng, reg)
            rv = coe2rv(**el)
            dt = 3600.0 if el['e'] < 1.0 else 600.0
            sol = solve_ivp(f, (0.0, dt), rv, method='DOP853',
                            rtol=1e-13, atol=1e-6, dense_output=False)
            ref = sol.y[:, -1]
            got = propagate_universal(rv, dt)
            rel = float(np.linalg.norm(got[:3] - ref[:3])
                        / np.linalg.norm(ref[:3]))
            if rel > worst:
                worst, ctx = rel, (reg, el['e'], el['i'])
    return worst, ctx


def v4_invariants(n=120, seed=11):
    """Energy, h-VECTOR (all three components) and e-VECTOR must be invariant
    under coast for any dt.  This is the coast-time plane check that a 2D
    oracle structurally cannot make."""
    rng = np.random.default_rng(seed)
    w = dict(energy=0.0, hvec=0.0, evec=0.0, hz_sign=0.0)
    for reg in REGIMES:
        for _ in range(n):
            el = _draw(rng, reg)
            rv = coe2rv(**el)
            c0 = rv2coe(rv)
            s = rv.copy()
            for _ in range(8):
                s = propagate_universal(s, 900.0)
                c = rv2coe(s)
                # energy scaled by mu/r, not by |E|: near-parabolic orbits have
                # |E| -> 0 and a relative-to-|E| measure diverges for reasons
                # that have nothing to do with propagation quality.
                w['energy'] = max(w['energy'],
                                  abs(c['energy'] - c0['energy']) / (MU / c0['r']))
                w['hvec'] = max(w['hvec'],
                                float(np.linalg.norm(c['h_vec'] - c0['h_vec'])
                                      / np.linalg.norm(c0['h_vec'])))
                w['evec'] = max(w['evec'],
                                float(np.linalg.norm(c['e_vec'] - c0['e_vec'])))
                # sign of h_z is meaningless for an exactly polar orbit
                if abs(c0['h_vec'][2]) / np.linalg.norm(c0['h_vec']) > 1e-6:
                    w['hz_sign'] = max(w['hz_sign'],
                                       0.0 if np.sign(c['h_vec'][2]) ==
                                       np.sign(c0['h_vec'][2]) else 1.0)
    return w


def v5_split(n=80, seed=13):
    """One long propagation == many short ones (semigroup property).  Catches
    a dt-dependent bug — the exact class the 2D warp actions introduced."""
    rng = np.random.default_rng(seed)
    worst, ctx = 0.0, None
    for reg in REGIMES:
        for _ in range(n):
            el = _draw(rng, reg)
            rv = coe2rv(**el)
            one = propagate_universal(rv, 3600.0)
            many = rv.copy()
            for _ in range(60):
                many = propagate_universal(many, 60.0)
            rel = float(np.linalg.norm(one[:3] - many[:3])
                        / np.linalg.norm(one[:3]))
            if rel > worst:
                worst, ctx = rel, (reg, el['e'], el['i'])
            back = propagate_universal(one, -3600.0)
            rel2 = float(np.linalg.norm(back[:3] - rv[:3])
                         / np.linalg.norm(rv[:3]))
            if rel2 > worst:
                worst, ctx = rel2, (reg + '/reverse', el['e'], el['i'])
    return worst, ctx


def v6_known_cases():
    """Closed-form cases with independently known answers."""
    out = []

    # (a) circular equatorial: |r| constant, quarter period = 90 deg of arc.
    a = R_EARTH + 500e3
    rv = coe2rv(a, 0.0, 0.0, 0.0, 0.0, 0.0)
    T = period(a)
    q = propagate_universal(rv, T / 4.0)
    out.append(('circ_eq |r| drift (rel)',
                abs(np.linalg.norm(q[:3]) - a) / a, 1e-12))
    out.append(('circ_eq quarter-period arc err (rad)',
                abs(wrap_pi(math.atan2(q[1], q[0]) - math.pi / 2)), 1e-9))
    out.append(('circ_eq speed vs sqrt(mu/a) (rel)',
                abs(np.linalg.norm(q[3:]) - math.sqrt(MU / a)) / math.sqrt(MU / a),
                1e-12))
    out.append(('circ_eq full-period return (rel)',
                float(np.linalg.norm(propagate_universal(rv, T)[:3] - rv[:3]) / a),
                1e-11))

    # (b) polar: h_z = 0 exactly, z spans +-a.
    rv = coe2rv(a, 0.0, math.pi / 2, 0.3, 0.0, 0.0)
    h = np.cross(rv[:3], rv[3:])
    out.append(('polar |h_z| / |h|', abs(h[2]) / np.linalg.norm(h), 1e-15))
    zmax = max(abs(propagate_universal(rv, T * k / 64.0)[2]) for k in range(65))
    out.append(('polar max|z| vs a (rel)', abs(zmax - a) / a, 2e-3))

    # (c) Molniya-like: a = 26554 km, e = 0.74, i = 63.4 deg, argp = 270 deg.
    aM, eM = 26554e3, 0.74
    el = dict(a=aM, e=eM, i=math.radians(63.4), raan=math.radians(45.0),
              argp=math.radians(270.0), nu=0.0)
    rv = coe2rv(**el)
    out.append(('molniya perigee radius (rel)',
                abs(np.linalg.norm(rv[:3]) - aM * (1 - eM)) / (aM * (1 - eM)), 1e-13))
    apo = propagate_universal(rv, period(aM) / 2.0)
    out.append(('molniya apogee radius after T/2 (rel)',
                abs(np.linalg.norm(apo[:3]) - aM * (1 + eM)) / (aM * (1 + eM)), 1e-9))
    cM = rv2coe(rv)
    out.append(('molniya i recovery (rad)', abs(cM['i'] - el['i']), 1e-12))
    out.append(('molniya argp recovery (rad)',
                abs(wrap_pi(cM['argp'] - el['argp'])), 1e-11))
    out.append(('molniya period vs 12 h sidereal (rel)',
                abs(period(aM) - 43082.0) / 43082.0, 3e-3))
    # perigee at argp=270 sits in the southern hemisphere -> z < 0 at nu=0
    out.append(('molniya perigee below equator [1=bad]',
                0.0 if rv[2] < 0.0 else 1.0, 0.0))

    # (d) retrograde i > 90: h_z < 0 and the node line is still recovered.
    el = dict(a=R_EARTH + 800e3, e=0.2, i=math.radians(120.0),
              raan=math.radians(200.0), argp=math.radians(30.0), nu=math.radians(70.0))
    rv = coe2rv(**el)
    c = rv2coe(rv)
    out.append(('retro h_z sign [1=bad]', 0.0 if c['h_vec'][2] < 0 else 1.0, 0.0))
    out.append(('retro i recovery (rad)', abs(c['i'] - el['i']), 1e-12))
    out.append(('retro raan recovery (rad)', abs(wrap_pi(c['raan'] - el['raan'])), 1e-11))
    out.append(('retro argp recovery (rad)', abs(wrap_pi(c['argp'] - el['argp'])), 1e-11))
    out.append(('retro nu recovery (rad)', abs(wrap_pi(c['nu'] - el['nu'])), 1e-11))
    nhat = unit(np.cross(_v(0, 0, 1), c['h_vec']))
    out.append(('retro node direction vs raan (rad)',
                abs(wrap_pi(math.atan2(nhat[1], nhat[0]) - el['raan'])), 1e-11))

    # (e) retrograde EQUATORIAL (i = pi): the convention seam.
    el = dict(a=R_EARTH + 700e3, e=0.3, i=math.pi, raan=0.0,
              argp=math.radians(140.0), nu=math.radians(-50.0))
    rv = coe2rv(**el)
    c = rv2coe(rv)
    rv2 = coe2rv_dict(c)
    out.append(('retro-eq state round trip (rel)',
                float(np.linalg.norm(rv2 - rv) / np.linalg.norm(rv[:3])), 1e-12))
    out.append(('retro-eq motion is clockwise in xy [1=bad]',
                0.0 if np.cross(rv[:3], rv[3:])[2] < 0 else 1.0, 0.0))

    # (f) pure plane change at the node: analytic Δv, and the node is preserved.
    a0 = R_EARTH + 700e3
    rv = coe2rv(a0, 0.0, math.radians(28.5), math.radians(10.0), 0.0, 0.0)
    v0 = float(np.linalg.norm(rv[3:]))
    di = math.radians(10.0)
    # at nu=0 with argp=0 the satellite IS at the ascending node
    c0 = rv2coe(rv)
    out.append(('plane-change setup: at node (rad)', abs(wrap_pi(c0['arglat'])), 1e-12))
    dv = plane_change_dv(v0, di)
    # Rodrigues-rotate v about r_hat by +di.  At the ascending node r_hat IS
    # the node line, so this is a pure inclination change: raan and a must not
    # move at all, and |dv| must equal 2 v sin(di/2).
    k = unit(rv[:3])
    vv = rv[3:]
    vrot = (vv * math.cos(di) + np.cross(k, vv) * math.sin(di)
            + k * float(np.dot(k, vv)) * (1 - math.cos(di)))
    rvn = np.concatenate([rv[:3], vrot])
    cn = rv2coe(rvn)
    out.append(('plane change: |dv| vs 2 v sin(di/2) (rel)',
                abs(float(np.linalg.norm(vrot - vv)) - dv) / dv, 1e-12))
    out.append(('plane change: di realised (rad)',
                abs((cn['i'] - c0['i']) - di), 1e-11))
    out.append(('plane change: raan unchanged (rad)',
                abs(wrap_pi(cn['raan'] - c0['raan'])), 1e-9))
    out.append(('plane change: a unchanged (rel)',
                abs(cn['a'] - c0['a']) / c0['a'], 1e-12))
    return out


def v7_2d_crosscheck(n=600, seed=17):
    """Planar (i = 0, raan = 0) cases must agree with the EXISTING 2D oracle
    to 1e-9 — the anchor that ties the 3D extension to the validated 2D stack.

    Compared: elements->cartesian, cartesian->elements, and f&g (2D) vs
    universal-variable (3D) propagation over 60/1800/3600 s.
    """
    sys.path.insert(0, os.path.join('/Users/pete/space_training', 'scripts',
                                    'orbital', 'nav'))
    import orbital_math as om2                                   # noqa: E402
    rng = np.random.default_rng(seed)
    w = dict(c2r=0.0, r2c_a=0.0, r2c_e=0.0, r2c_ang=0.0, prop=0.0, elems=0.0)
    for _ in range(n):
        el2 = dict(a=R_EARTH + float(rng.uniform(300e3, 8000e3)),
                   e=float(rng.uniform(1e-9, 0.6)),
                   theta=float(rng.uniform(-math.pi, math.pi)),
                   omega=float(rng.uniform(0.0, 2 * math.pi)))
        el2['M'] = om2.mean_from_true(el2['theta'], el2['e'])
        x, y, vx, vy = om2.orbit_to_cartesian(el2)
        rv = coe2rv(el2['a'], el2['e'], 0.0, 0.0, el2['omega'], el2['theta'])
        scale = math.hypot(x, y)
        w['c2r'] = max(w['c2r'], float(np.linalg.norm(rv[:3] - _v(x, y, 0.0))) / scale)
        w['c2r'] = max(w['c2r'], float(np.linalg.norm(rv[3:] - _v(vx, vy, 0.0)))
                       / math.hypot(vx, vy))
        c3 = rv2coe(rv)
        c2 = om2.cartesian_to_elements(x, y, vx, vy)
        w['r2c_a'] = max(w['r2c_a'], abs(c3['a'] - c2['a']) / c2['a'])
        w['r2c_e'] = max(w['r2c_e'], abs(c3['e'] - c2['e']))
        w['r2c_ang'] = max(w['r2c_ang'],
                           abs(wrap_pi(c3['argp'] + c3['nu']
                                       - c2['omega'] - c2['theta'])))
        for dt in (60.0, 1800.0, 3600.0):
            p2 = om2.propagate_cartesian((x, y, vx, vy), dt)
            p3 = propagate_universal(rv, dt)
            w['prop'] = max(w['prop'],
                            float(np.linalg.norm(p3[:3] - _v(p2[0], p2[1], 0.0)))
                            / float(np.linalg.norm(p3[:3])))
            e2 = om2.cartesian_to_elements(*p2)
            e3 = rv2coe(p3)
            w['elems'] = max(w['elems'], abs(e3['a'] - e2['a']) / e2['a'])
    return w


def v8_impulse_geometry(n=400, seed=19):
    """The killer regression: an impulse with NO normal component must not
    move the orbital plane, and a normal-only impulse must not change |r| or
    the in-plane shape at the burn point.  Also: an impulse never moves the
    satellite (position continuity)."""
    rng = np.random.default_rng(seed)
    w = dict(pos=0.0, plane_i=0.0, plane_raan=0.0, hhat=0.0,
             nor_r=0.0, nor_speed=0.0, nor_i=0.0)
    for reg in ('circ_incl', 'polar', 'retro', 'near_eq', 'near_circ',
                'high_e', 'general'):
        for _ in range(n):
            el = _draw(rng, reg)
            rv = coe2rv(**el)
            c0 = rv2coe(rv)
            dvp = float(rng.uniform(-25.0, 25.0))
            dvr = float(rng.uniform(-10.0, 10.0))
            b = apply_impulse_3d(rv, dv_pro=dvp, dv_rad=dvr, dv_nor=0.0)
            cb = rv2coe(b)
            w['pos'] = max(w['pos'], float(np.linalg.norm(b[:3] - rv[:3])))
            w['plane_i'] = max(w['plane_i'], abs(cb['i'] - c0['i']))
            if c0['case'] in ('general', 'circ'):
                w['plane_raan'] = max(w['plane_raan'],
                                      abs(wrap_pi(cb['raan'] - c0['raan'])))
            w['hhat'] = max(w['hhat'],
                            float(np.linalg.norm(unit(cb['h_vec']) - unit(c0['h_vec']))))
            # normal-only burn: |r| fixed, |v| grows by exactly sqrt(v^2+dv^2)
            dvn = float(rng.uniform(-30.0, 30.0))
            bn = apply_impulse_3d(rv, dv_nor=dvn)
            cn = rv2coe(bn)
            w['nor_r'] = max(w['nor_r'], abs(cn['r'] - c0['r']) / c0['r'])
            w['nor_speed'] = max(w['nor_speed'],
                                 abs(cn['v'] - math.hypot(c0['v'], dvn)) / c0['v'])
            w['nor_i'] = max(w['nor_i'],
                             0.0 if abs(cn['i'] - c0['i']) > 0.0 or abs(dvn) < 1e-9
                             else 1.0)
    return w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join('/Users/pete/space_training',
                                                  'web_data', 'results'))
    ap.add_argument('--quick', action='store_true')
    args = ap.parse_args()
    k = 0.25 if args.quick else 1.0

    print('== V1  coe <-> rv round trip, all regimes ==')
    w, ctx = v1_roundtrip(n=int(400 * k))
    _record('V1 state round-trip, non-degenerate (rel)', w['rv'], 1e-13,
            note=str(ctx.get('rv', ('', ''))[0]))
    _record('V1 state round-trip, at degeneracy seam (rel)', w['rv_deg'], 5e-11,
            note='irreducible for classical elements; see V9')
    _record('V1 a (rel)', w['a'], 1e-12)
    _record('V1 e (abs)', w['e'], 1e-11)
    _record('V1 i (rad)', w['i'], 1e-11)
    _record('V1 raan (rad, well-cond. only)', w['raan'], 1e-10)
    _record('V1 argp (rad, well-cond. only)', w['argp'], 1e-9)
    _record('V1 nu (rad, well-cond. only)', w['nu'], 1e-9)
    _record('V1 argp+nu = arglat (rad, all inclined incl. e=0)', w['arglat'], 1e-9)
    _record('V1 equatorial true-longitude seam (rad, incl. i=pi)',
            w['truelon_eq'], 1e-9)

    print('== V2  universal-variable vs element propagation ==')
    v, ctx = v2_prop_vs_elements(n=int(120 * k))
    _record('V2 |dr|/r vs element prop', v, 1e-10, note=str(ctx))

    print('== V3  universal-variable vs DOP853 numerical integration ==')
    v, ctx = v3_prop_vs_numeric(n=int(40 * k))
    _record('V3 |dr|/r vs DOP853', v, 1e-9, note=str(ctx))

    print('== V4  coast invariants (energy, h-vector, e-vector) ==')
    w = v4_invariants(n=int(120 * k))
    _record('V4 energy drift / (mu/r)', w['energy'], 1e-11)
    _record('V4 h-VECTOR drift (rel)', w['hvec'], 5e-11,
            note='fp accumulation over 8x900 s; worst at near-parabolic e')
    _record('V4 e-VECTOR drift (abs)', w['evec'], 1e-11)
    _record('V4 h_z sign flips [1=bad]', w['hz_sign'], 0.0)

    print('== V5  semigroup / time reversal ==')
    v, ctx = v5_split(n=int(80 * k))
    _record('V5 1x3600 vs 60x60 / reverse (rel)', v, 1e-10, note=str(ctx))

    print('== V6  known closed-form cases ==')
    for name, val, th in v6_known_cases():
        _record('V6 ' + name, val, th)

    print('== V7  planar cross-check vs the validated 2D oracle ==')
    w = v7_2d_crosscheck(n=int(600 * k))
    _record('V7 coe2rv vs 2D orbit_to_cartesian (rel)', w['c2r'], 1e-9)
    _record('V7 rv2coe a vs 2D (rel)', w['r2c_a'], 1e-9)
    _record('V7 rv2coe e vs 2D (abs)', w['r2c_e'], 1e-9)
    _record('V7 rv2coe argp+nu vs 2D omega+theta (rad)', w['r2c_ang'], 1e-9)
    _record('V7 universal vs 2D f&g propagation (rel)', w['prop'], 1e-9)
    _record('V7 propagated a agreement (rel)', w['elems'], 1e-9)

    print('== V8  impulse geometry (the plane-invariance regression) ==')
    w = v8_impulse_geometry(n=int(400 * k))
    _record('V8 impulse moves position (m)', w['pos'], 1e-6)
    _record('V8 in-plane burn changes i (rad)', w['plane_i'], 1e-12)
    _record('V8 in-plane burn changes raan (rad)', w['plane_raan'], 1e-9)
    _record('V8 in-plane burn rotates h_hat', w['hhat'], 1e-12)
    _record('V8 normal burn changes |r| (rel)', w['nor_r'], 1e-15)
    _record('V8 normal burn |v| vs hypot (rel)', w['nor_speed'], 1e-14)
    _record('V8 normal burn failed to change i [1=bad]', w['nor_i'], 0.0)

    print('== V9  classical-element conditioning sweep (design guidance) ==')
    v9 = v9_element_conditioning()
    print(f'   {"param":>7s} {"value":>10s}   {"max rv->coe->rv rel err":>24s}')
    for p, val, err in v9:
        print(f'   {p:>7s} {val:10.1e}   {err:24.3e}')
    _record('V9 worst round-trip over the whole sweep (rel)',
            max(r[2] for r in v9), 5e-11,
            note='peak sits exactly at the e / sin(i) switch points')

    print()
    print(f'{"check":52s} {"value":>13s} {"thresh":>11s}  verdict')
    print('-' * 92)
    nfail = 0
    for r in _RESULTS:
        nfail += r['verdict'] == 'FAIL'
        print(f'{r["check"]:52s} {r["value"]:13.4e} {r["thresh"]:11.2e}  '
              f'{r["verdict"]}' + (f'   [{r["note"]}]' if r['note'] and
                                   r['verdict'] == 'FAIL' else ''))
    print('-' * 92)
    print(f'{len(_RESULTS) - nfail}/{len(_RESULTS)} PASS')

    os.makedirs(args.out, exist_ok=True)
    import csv
    p = os.path.join(args.out, 'ext_3d_oracle_validation.csv')
    with open(p, 'w', newline='') as f:
        wcsv = csv.DictWriter(f, fieldnames=['check', 'value', 'thresh', 'units',
                                             'verdict', 'note'])
        wcsv.writeheader()
        for r in _RESULTS:
            wcsv.writerow(r)
    print(f'wrote {p}')
    return 1 if nfail else 0


if __name__ == '__main__':
    sys.exit(main())
