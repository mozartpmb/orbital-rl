"""Shared 3D geometry helpers for the ext-3d red-team probes.

Deliberately independent of orbital.h and of ext_recon/orbital_math3d.py:
written from the textbook 3-1-3 rotation so agreement with either is evidence.
"""
import math

MU = 3.986004418e14
R_EARTH = 6.371e6
BUDGET_DV = 478.13          # V_e * ln(1/(1-0.15)), the shipped fuel budget


def what_from_ie(inc, raan):
    """Unit angular-momentum vector from (i, RAAN), 3-1-3 convention.

    h_hat = R3(-O) R1(-i) zhat = ( sin i sin O, -sin i cos O, cos i )
    """
    si, ci = math.sin(inc), math.cos(inc)
    return (si * math.sin(raan), -si * math.cos(raan), ci)


def ivec_from_ie(inc, raan):
    """The 3d_A '(inclination vector)' i_vec = (sin i cos O, sin i sin O)."""
    si = math.sin(inc)
    return (si * math.cos(raan), si * math.sin(raan))


def ie_from_what(w):
    """Inverse of what_from_ie, full [0, pi] range (atan2, per 3d_E F1)."""
    wx, wy, wz = w
    inc = math.atan2(math.hypot(wx, wy), wz)
    raan = math.atan2(wx, -wy) % (2.0 * math.pi)
    return inc, raan


def plane_angle(w1, w2):
    """True relative inclination = angle between the two orbit planes (rad)."""
    cx = w1[1] * w2[2] - w1[2] * w2[1]
    cy = w1[2] * w2[0] - w1[0] * w2[2]
    cz = w1[0] * w2[1] - w1[1] * w2[0]
    dot = w1[0] * w2[0] + w1[1] * w2[1] + w1[2] * w2[2]
    return math.atan2(math.sqrt(cx * cx + cy * cy + cz * cz), dot)


def rotate_about(v, axis, ang):
    """Rodrigues rotation of v about a unit axis by ang."""
    ax, ay, az = axis
    c, s = math.cos(ang), math.sin(ang)
    dot = ax * v[0] + ay * v[1] + az * v[2]
    cr = (ay * v[2] - az * v[1], az * v[0] - ax * v[2], ax * v[1] - ay * v[0])
    return tuple(v[i] * c + cr[i] * s + (ax, ay, az)[i] * dot * (1.0 - c)
                 for i in range(3))


def coe2rv(a, e, inc, raan, argp, nu, mu=MU):
    """Classical elements -> inertial r, v (3-1-3, matches 3d_A section 2.1)."""
    p = a * (1.0 - e * e)
    r = p / (1.0 + e * math.cos(nu))
    xp, yp = r * math.cos(nu), r * math.sin(nu)
    sq = math.sqrt(mu / p)
    vxp, vyp = -sq * math.sin(nu), sq * (e + math.cos(nu))
    co, so = math.cos(raan), math.sin(raan)
    cw, sw = math.cos(argp), math.sin(argp)
    ci, si = math.cos(inc), math.sin(inc)
    R11 = co * cw - so * sw * ci
    R12 = -co * sw - so * cw * ci
    R21 = so * cw + co * sw * ci
    R22 = -so * sw + co * cw * ci
    R31 = sw * si
    R32 = cw * si
    return ((R11 * xp + R12 * yp, R21 * xp + R22 * yp, R31 * xp + R32 * yp),
            (R11 * vxp + R12 * vyp, R21 * vxp + R22 * vyp, R31 * vxp + R32 * vyp))


def evec_from_rv(r, v, mu=MU):
    """Inertial eccentricity vector e_vec = (v x h)/mu - r_hat."""
    hx = r[1] * v[2] - r[2] * v[1]
    hy = r[2] * v[0] - r[0] * v[2]
    hz = r[0] * v[1] - r[1] * v[0]
    rm = math.sqrt(r[0] ** 2 + r[1] ** 2 + r[2] ** 2)
    ex = (v[1] * hz - v[2] * hy) / mu - r[0] / rm
    ey = (v[2] * hx - v[0] * hz) / mu - r[1] / rm
    ez = (v[0] * hy - v[1] * hx) / mu - r[2] / rm
    return (ex, ey, ez)


def evec_from_elements(e, inc, raan, argp):
    """Inertial e-vector built from elements (the cheap path a C impl would use)."""
    co, so = math.cos(raan), math.sin(raan)
    cw, sw = math.cos(argp), math.sin(argp)
    ci, si = math.cos(inc), math.sin(inc)
    return (e * (co * cw - so * sw * ci),
            e * (so * cw + co * sw * ci),
            e * (sw * si))


def true_to_mean(nu, e):
    E = 2.0 * math.atan2(math.sqrt(1.0 - e) * math.sin(nu / 2.0),
                         math.sqrt(1.0 + e) * math.cos(nu / 2.0))
    return E - e * math.sin(E)


def mean_to_true(M, e):
    E = M
    for _ in range(60):
        f = E - e * math.sin(E) - M
        E -= f / (1.0 - e * math.cos(E))
    return 2.0 * math.atan2(math.sqrt(1.0 + e) * math.sin(E / 2.0),
                            math.sqrt(1.0 - e) * math.cos(E / 2.0))


def rv2coe(r, v, mu=MU):
    """3D inverse per 3d_A section 2.2 (atan2 for i, signed-angle branch table)."""
    rm = math.sqrt(sum(c * c for c in r))
    v2 = sum(c * c for c in v)
    rv = sum(r[i] * v[i] for i in range(3))
    a = 1.0 / (2.0 / rm - v2 / mu)
    hx = r[1] * v[2] - r[2] * v[1]
    hy = r[2] * v[0] - r[0] * v[2]
    hz = r[0] * v[1] - r[1] * v[0]
    hxy = math.hypot(hx, hy)
    hm = math.sqrt(hx * hx + hy * hy + hz * hz)
    inc = math.atan2(hxy, hz)
    vr = rv / rm
    ex = ((v2 - mu / rm) * r[0] - vr * rm * v[0]) / mu
    ey = ((v2 - mu / rm) * r[1] - vr * rm * v[1]) / mu
    ez = ((v2 - mu / rm) * r[2] - vr * rm * v[2]) / mu
    e = math.sqrt(ex * ex + ey * ey + ez * ez)
    if hxy == 0.0:
        raan = 0.0
        if e < 1e-10:
            argp, nu = 0.0, math.atan2(r[1], r[0])
        else:
            argp = math.atan2(ey, ex)
            ct = max(-1.0, min(1.0, (ex * r[0] + ey * r[1]) / (e * rm)))
            nu = math.acos(ct)
            if vr < 0.0:
                nu = 2 * math.pi - nu
    else:
        raan = math.atan2(hx, -hy) % (2 * math.pi)
        nx, ny = -hy / hxy, hx / hxy
        wx, wy, wz = hx / hm, hy / hm, hz / hm
        mx, my, mz = -wz * ny, wz * nx, wx * ny - wy * nx
        if e < 1e-10:
            argp = 0.0
            nu = math.atan2(r[0] * mx + r[1] * my + r[2] * mz, r[0] * nx + r[1] * ny)
        else:
            argp = math.atan2(ex * mx + ey * my + ez * mz, ex * nx + ey * ny) % (2 * math.pi)
            eux, euy, euz = ex / e, ey / e, ez / e
            qx = wy * euz - wz * euy
            qy = wz * eux - wx * euz
            qz = wx * euy - wy * eux
            nu = math.atan2(r[0] * qx + r[1] * qy + r[2] * qz,
                            r[0] * eux + r[1] * euy + r[2] * euz) % (2 * math.pi)
    return a, e, inc, raan, argp % (2 * math.pi), nu % (2 * math.pi)


def wrap_pi(x):
    x = math.fmod(x + math.pi, 2.0 * math.pi)
    if x < 0.0:
        x += 2.0 * math.pi
    return x - math.pi


def pct(vals, q):
    s = sorted(vals)
    if not s:
        return float('nan')
    k = (len(s) - 1) * q
    lo, hi = int(math.floor(k)), int(math.ceil(k))
    return s[lo] if lo == hi else s[lo] + (s[hi] - s[lo]) * (k - lo)
