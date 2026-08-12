#!/usr/bin/env python3
"""
J2-A core: independent classical-element machinery + secular-J2 mean-element
propagator + a Cowell (numerical, osculating) J2 integrator.

Deliberately independent of orbital.h and of orbital_math3d.py so that every
number produced by the J2-A probes is a cross-check rather than a restatement.

Conventions match the env (3-1-3, prograde, angles in rad).
Constants: MU and R_EARTH from orbital.h; R_EQ is the J2 reference radius
(equatorial, 6378.137 km) which is NOT the env's R_EARTH (mean, 6371 km).
"""
import math

import numpy as np

MU = 3.986004418e14          # orbital.h
R_ENV = 6.371e6              # orbital.h R_EARTH (mean radius) - used for altitudes
R_EQ = 6.378137e6            # WGS-84 equatorial radius - the correct J2 reference
J2 = 1.08263e-3
DT = 60.0
DV_BUDGET = 478.12987798780637   # 15% fuel fraction, Isp 300 s


# ── element <-> cartesian (independent implementation, atan2 everywhere) ────
def kepler_E(M, e):
    M = math.fmod(M, 2 * math.pi)
    if M < 0:
        M += 2 * math.pi
    E = M if e < 0.8 else math.pi
    for _ in range(60):
        dE = (M - E + e * math.sin(E)) / (1 - e * math.cos(E))
        E += dE
        if abs(dE) < 1e-15:
            break
    return E


def E_to_nu(E, e):
    return 2 * math.atan2(math.sqrt(1 + e) * math.sin(E / 2),
                          math.sqrt(1 - e) * math.cos(E / 2))


def nu_to_M(nu, e):
    E = 2 * math.atan2(math.sqrt(1 - e) * math.sin(nu / 2),
                       math.sqrt(1 + e) * math.cos(nu / 2))
    return E - e * math.sin(E)


def coe2rv(a, e, i, raan, argp, nu):
    p = a * (1 - e * e)
    r = p / (1 + e * math.cos(nu))
    rp = np.array([r * math.cos(nu), r * math.sin(nu), 0.0])
    vp = math.sqrt(MU / p) * np.array([-math.sin(nu), e + math.cos(nu), 0.0])
    cO, sO = math.cos(raan), math.sin(raan)
    cw, sw = math.cos(argp), math.sin(argp)
    ci, si = math.cos(i), math.sin(i)
    R = np.array([
        [cO * cw - sO * sw * ci, -cO * sw - sO * cw * ci, sO * si],
        [sO * cw + cO * sw * ci, -sO * sw + cO * cw * ci, -cO * si],
        [sw * si,                 cw * si,                ci],
    ])
    return R @ rp, R @ vp


def rv2coe(r, v):
    """→ (a, e, i, raan, argp, nu). atan2 only; equatorial/circular resolved so
    that coe2rv∘rv2coe is the identity."""
    rn = np.linalg.norm(r)
    vn = np.linalg.norm(v)
    h = np.cross(r, v)
    hn = np.linalg.norm(h)
    a = 1.0 / (2.0 / rn - vn * vn / MU)
    evec = ((vn * vn - MU / rn) * r - np.dot(r, v) * v) / MU
    e = np.linalg.norm(evec)
    hxy = math.hypot(h[0], h[1])
    i = math.atan2(hxy, h[2])
    if hxy > 0:
        raan = math.atan2(h[0], -h[1]) % (2 * math.pi)
        nvec = np.array([-h[1], h[0], 0.0]) / hxy
    else:
        raan = 0.0
        nvec = np.array([1.0, 0.0, 0.0])
    what = h / hn
    mvec = np.cross(what, nvec)
    if e < 1e-12:
        argp = 0.0
        nu = math.atan2(np.dot(r, mvec), np.dot(r, nvec)) % (2 * math.pi)
    else:
        eu = evec / e
        argp = math.atan2(np.dot(evec, mvec), np.dot(evec, nvec)) % (2 * math.pi)
        q = np.cross(what, eu)
        nu = math.atan2(np.dot(r, q), np.dot(r, eu)) % (2 * math.pi)
    return a, e, i, raan, argp, nu


# ── secular J2 mean-element rates (Vallado 9-38..9-40) ─────────────────────
def secular_rates(a, e, i):
    """Returns (n, Omega_dot, argp_dot, M_dot) in rad/s. a,e,i are MEAN."""
    n = math.sqrt(MU / a ** 3)
    p = a * (1 - e * e)
    k = 1.5 * n * J2 * (R_EQ / p) ** 2
    Om = -k * math.cos(i)
    om = 0.5 * k * (4.0 - 5.0 * math.sin(i) ** 2)
    Md = n + 0.5 * k * math.sqrt(1 - e * e) * (2.0 - 3.0 * math.sin(i) ** 2)
    return n, Om, om, Md


class Orb:
    __slots__ = ("a", "e", "i", "raan", "argp", "M", "nu")

    def __init__(self, a, e, i, raan, argp, M):
        self.a, self.e, self.i, self.raan, self.argp, self.M = a, e, i, raan, argp, M
        self.nu = E_to_nu(kepler_E(M, e), e)

    def copy(self):
        o = Orb(self.a, self.e, self.i, self.raan, self.argp, self.M)
        o.nu = self.nu
        return o

    def rv(self):
        return coe2rv(self.a, self.e, self.i, self.raan, self.argp, self.nu)

    def hhat(self):
        si, ci = math.sin(self.i), math.cos(self.i)
        return np.array([si * math.sin(self.raan), -si * math.cos(self.raan), ci])

    def evec(self):
        cO, sO = math.cos(self.raan), math.sin(self.raan)
        cw, sw = math.cos(self.argp), math.sin(self.argp)
        ci, si = math.cos(self.i), math.sin(self.i)
        return self.e * np.array([cO * cw - sO * sw * ci,
                                  sO * cw + cO * sw * ci,
                                  sw * si])


def propagate(o, dt, j2_mode=0):
    """The PROPOSED c-side ordering, in double precision.

    j2_mode=0 reproduces orbital.h::propagate_orbit exactly.
    j2_mode=1 adds the three secular rates; a, e, i are untouched (secular J2
    has zero secular rate on them), so the rates are CONSTANT and the map is
    closed-form at any dt."""
    if j2_mode:
        n, Om, om, Md = secular_rates(o.a, o.e, o.i)
        o.raan = math.fmod(o.raan + Om * dt, 2 * math.pi)
        if o.raan < 0:
            o.raan += 2 * math.pi
        o.argp = math.fmod(o.argp + om * dt, 2 * math.pi)
        if o.argp < 0:
            o.argp += 2 * math.pi
        o.M += Md * dt
    else:
        n = math.sqrt(MU / o.a ** 3)
        o.M += n * dt
    o.M = math.fmod(o.M, 2 * math.pi)
    if o.M < 0:
        o.M += 2 * math.pi
    o.nu = E_to_nu(kepler_E(o.M, o.e), o.e)
    return o


def apply_impulse(o, dv_pro, dv_rad, dv_nor):
    """env apply_impulse geometry: prograde = v̂, radial = r̂ (oblique at e>0),
    normal = ĥ. Returns the new Orb and |Δv| of the assembled vector."""
    r, v = o.rv()
    vh = v / np.linalg.norm(v)
    rh = r / np.linalg.norm(r)
    hh = np.cross(r, v)
    hh = hh / np.linalg.norm(hh)
    dv = dv_pro * vh + dv_rad * rh + dv_nor * hh
    a, e, i, raan, argp, nu = rv2coe(r, v + dv)
    o2 = Orb(a, e, i, raan, argp, nu_to_M(nu, e))
    o2.nu = nu
    return o2, float(np.linalg.norm(dv))


# ── Cowell numerical integrator with J2 (osculating truth) ─────────────────
def j2_accel(state, j2=J2, req=R_EQ):
    x, y, z, vx, vy, vz = state
    r2 = x * x + y * y + z * z
    r = math.sqrt(r2)
    k = 1.5 * j2 * MU * req * req / (r2 * r2 * r)
    zr2 = (z * z) / r2
    ax = -MU * x / (r2 * r) - k * x * (1 - 5 * zr2)
    ay = -MU * y / (r2 * r) - k * y * (1 - 5 * zr2)
    az = -MU * z / (r2 * r) - k * z * (3 - 5 * zr2)
    return np.array([vx, vy, vz, ax, ay, az])


def integrate_j2(r0, v0, t_end, n_out, rtol=1e-12, atol=1e-6, j2=J2):
    from scipy.integrate import solve_ivp
    ts = np.linspace(0.0, t_end, n_out)
    sol = solve_ivp(lambda t, s: j2_accel(s, j2=j2),
                    (0.0, t_end), np.concatenate([r0, v0]),
                    t_eval=ts, method="DOP853", rtol=rtol, atol=atol)
    assert sol.success, sol.message
    return ts, sol.y


# ── target-plane gauge (orbital.h gauge_from_orbit / orb_varpi_gauge) ───────
def gauge_from_orbit(t):
    if t.i == 0.0 and t.raan == 0.0:
        return None
    w = t.hhat()
    n = np.array([-w[1], w[0], 0.0])
    nn = np.linalg.norm(n)
    e1 = n / nn if nn > 1e-14 else np.array([1.0, 0.0, 0.0])
    e3 = w
    e2 = np.cross(e3, e1)
    return np.vstack([e1, e2, e3])


def lambda_gauge(o, G):
    if G is None:
        return o.M + o.argp + o.raan
    r, v = o.rv()
    _, _, _, raan, argp, _ = rv2coe(G @ r, G @ v)
    return o.M + argp + raan


def wrap_pi(x):
    return (x + math.pi) % (2 * math.pi) - math.pi


# ── shaping_mode 2 potential (verbatim algebra from orbital.h) ─────────────
def phi_mode2(sat, tgt, w_lambda=1.0, w_match=0.817, dv_ref=700.0, squash=0):
    G = gauge_from_orbit(tgt)
    dlam = wrap_pi(lambda_gauge(sat, G) - lambda_gauge(tgt, G))
    de = float(np.linalg.norm(sat.evec() - tgt.evec()))
    da_rel = (sat.a - tgt.a) / tgt.a
    v_t = math.sqrt(MU / tgt.a)
    dv_in = 0.5 * v_t * math.sqrt(da_rel * da_rel + de * de)
    dv_pl = 1.0 * v_t * float(np.linalg.norm(sat.hhat() - tgt.hhat()))
    m = (dv_in + dv_pl) / dv_ref
    m = m / (1 + m) if squash else min(1.0, m)
    return -(w_lambda * abs(dlam) / math.pi + w_match * m), dv_in, dv_pl, dlam


def di_rel(sat, tgt):
    hs, ht = sat.hhat(), tgt.hhat()
    return math.atan2(float(np.linalg.norm(np.cross(ht, hs))), float(np.dot(ht, hs)))
