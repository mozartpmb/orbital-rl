"""Batched 3D orbital math for the `dim3_mode=1` nav wrapper.

Companion to `nav_math.py` (2D). Everything here is float64, batched over the
leading axis, and validated against two independent references:

* the C environment itself (`orbital.h`) — for anything the wrapper has to
  REPRODUCE bit-for-bit or value-for-value, the C spelling is copied, not
  "improved". Two spellings in particular are load-bearing and are called out
  at their sites: `cartesian_to_elements`'s equatorial branch (which fires on
  an EXACT `hxy == 0.0` test, not a tolerance) and its e-vector form.
* `scripts/orbital/ext_recon/orbital_math3d.py` — the independent
  universal-variable oracle, used by `verify_n3dnav.py`.

Nothing here imports the C extension.
"""

import numpy as np

from . import nav_math as nm

MU = nm.MU
R_EARTH = nm.R_EARTH
SQ_MU = nm.SQ_MU

ISP, G0 = 300.0, 9.80665
VE = ISP * G0                     # orbital.h: exhaust velocity, 2941.995 m/s


# ── small vector helpers ─────────────────────────────────────────────────────
def _norm(v):
    return np.sqrt(np.einsum('...i,...i->...', v, v))


def unit(v, eps=1e-300):
    n = _norm(v)
    return v / np.maximum(n, eps)[..., None]


def cross(a, b):
    return np.cross(a, b)


def hhat_from_elements(inc, raan):
    """orbital.h::orb_hhat — 3-1-3 convention, VERBATIM.

    h-hat = (sin i sin RAAN, -sin i cos RAAN, cos i)
    """
    si, ci = np.sin(inc), np.cos(inc)
    out = np.empty(np.shape(inc) + (3,), dtype=np.float64)
    out[..., 0] = si * np.sin(raan)
    out[..., 1] = -si * np.cos(raan)
    out[..., 2] = ci
    return out


def evec_from_elements(e, inc, raan, omega):
    """orbital.h::orb_evec — the ELEMENT route.

    3d_REDTEAM BLOCKER-2 variant V3: the algebraically identical Cartesian
    route (v x h)/mu - r_hat is a different floating-point path and breaks the
    A2 bit-exact anchor on 87.7% of draws. Anything that has to reproduce a C
    observation slot must use this one.
    """
    cO, sO = np.cos(raan), np.sin(raan)
    cw, sw = np.cos(omega), np.sin(omega)
    ci, si = np.cos(inc), np.sin(inc)
    out = np.empty(np.shape(e) + (3,), dtype=np.float64)
    out[..., 0] = e * (cO * cw - sO * sw * ci)
    out[..., 1] = e * (sO * cw + cO * sw * ci)
    out[..., 2] = e * (sw * si)
    return out


def evec_from_cartesian(X):
    """e = (v x h)/mu - r_hat, the CARTESIAN route.

    This is the only route available for an ESTIMATE (which has no elements),
    and it is what MAJOR-15's `recon-cart` gate exists to bound: with the
    getter in place, the ordinary `recon` gate stops exercising this path.
    """
    r, v = X[..., :3], X[..., 3:6]
    h = cross(r, v)
    rn = np.maximum(_norm(r), 1.0)
    return cross(v, h) / MU - r / rn[..., None]


# ── Cartesian -> elements, mirroring orbital.h::cartesian_to_elements ────────
def cartesian_to_elements_3d(X):
    """(N,6) -> dict(a, e, omega, theta, M, inc, raan), all (N,) float64.

    Both branches of the C routine are reproduced, including the EXACT
    `hxy == 0.0` equatorial test. That test is not a tolerance: with
    z = vz = 0 exactly (every 2D-lineage state, and every target at
    i_target_rad = 0), hx and hy are each a difference of two exact zeros, so
    the equatorial branch fires bit-deterministically and the wrapper's
    reconstruction lands on the same branch the env did.
    """
    X = np.asarray(X, dtype=np.float64)
    x, y, z = X[..., 0], X[..., 1], X[..., 2]
    vx, vy, vz = X[..., 3], X[..., 4], X[..., 5]

    hx = y * vz - z * vy
    hy = z * vx - x * vz
    hz = x * vy - y * vx
    hxy = np.sqrt(hx * hx + hy * hy)
    eq = (hxy == 0.0)

    with np.errstate(all='ignore'):
        # ── equatorial branch (verbatim C; the whole 2D lineage) ─────────────
        r2 = np.sqrt(x * x + y * y)
        v2q = vx * vx + vy * vy
        vrq = (x * vx + y * vy) / r2
        a_q = 1.0 / (2.0 / r2 - v2q / MU)
        exq = ((v2q - MU / r2) * x - vrq * r2 * vx) / MU
        eyq = ((v2q - MU / r2) * y - vrq * r2 * vy) / MU
        e_q = np.sqrt(exq * exq + eyq * eyq)
        circ_q = e_q < 1e-10
        om_q = np.where(circ_q, 0.0, np.arctan2(eyq, exq))
        cth = np.clip((exq * x + eyq * y) / np.where(circ_q, 1.0, e_q * r2),
                      -1.0, 1.0)
        thq = np.arccos(cth)
        thq = np.where(vrq < 0.0, 2.0 * np.pi - thq, thq)
        th_q = np.where(circ_q, np.arctan2(y, x), thq)

        # ── inclined branch ─────────────────────────────────────────────────
        r3 = np.sqrt(x * x + y * y + z * z)
        v23 = vx * vx + vy * vy + vz * vz
        rv = x * vx + y * vy + z * vz
        vr3 = rv / r3
        a_i = 1.0 / (2.0 / r3 - v23 / MU)
        hmag = np.sqrt(hx * hx + hy * hy + hz * hz)
        inc_i = np.arctan2(hxy, hz)
        raan_i = np.mod(np.arctan2(hx, -hy), 2.0 * np.pi)

        exi = ((v23 - MU / r3) * x - vr3 * r3 * vx) / MU
        eyi = ((v23 - MU / r3) * y - vr3 * r3 * vy) / MU
        ezi = ((v23 - MU / r3) * z - vr3 * r3 * vz) / MU
        e_i = np.sqrt(exi * exi + eyi * eyi + ezi * ezi)

        hxy_s = np.where(eq, 1.0, hxy)          # guard the equatorial rows
        hmag_s = np.where(hmag > 0.0, hmag, 1.0)
        nx, ny = -hy / hxy_s, hx / hxy_s
        wx, wy, wz = hx / hmag_s, hy / hmag_s, hz / hmag_s
        mx, my, mz = -wz * ny, wz * nx, wx * ny - wy * nx

        circ_i = e_i < 1e-10
        th_circ = np.mod(np.arctan2(x * mx + y * my + z * mz, x * nx + y * ny),
                         2.0 * np.pi)
        w_i = np.mod(np.arctan2(exi * mx + eyi * my + ezi * mz,
                                exi * nx + eyi * ny), 2.0 * np.pi)
        e_s = np.where(circ_i, 1.0, e_i)
        eux, euy, euz = exi / e_s, eyi / e_s, ezi / e_s
        qx = wy * euz - wz * euy
        qy = wz * eux - wx * euz
        qz = wx * euy - wy * eux
        th_ecc = np.mod(np.arctan2(x * qx + y * qy + z * qz,
                                   x * eux + y * euy + z * euz), 2.0 * np.pi)
        om_i = np.where(circ_i, 0.0, w_i)
        th_i = np.where(circ_i, th_circ, th_ecc)

    a = np.where(eq, a_q, a_i)
    e = np.where(eq, e_q, e_i)
    omega = np.where(eq, om_q, om_i)
    theta = np.where(eq, th_q, th_i)
    inc = np.where(eq, 0.0, inc_i)
    raan = np.where(eq, 0.0, raan_i)
    # C's true_to_mean is the CORRECTED map at HEAD and is identical to
    # nav_math.mean_from_true (same half-angle spelling).
    M = np.mod(nm.mean_from_true(theta, e), 2.0 * np.pi)
    return dict(a=a, e=e, omega=omega, theta=theta, M=M, inc=inc, raan=raan,
                equatorial=eq)


def orbit_to_cartesian_3d(a, e, theta, omega, inc, raan):
    """orbital.h::orbit_to_cartesian, including its value-gated 2D fast path."""
    p = a * (1.0 - e * e)
    r = p / (1.0 + e * np.cos(theta))
    h = np.sqrt(MU * np.maximum(p, 1e-9))
    xp = r * np.cos(theta)
    yp = r * np.sin(theta)
    vxp = -(MU / h) * np.sin(theta)
    vyp = (MU / h) * (e + np.cos(theta))

    cO, sO = np.cos(raan), np.sin(raan)
    cw, sw = np.cos(omega), np.sin(omega)
    ci, si = np.cos(inc), np.sin(inc)
    R11 = cO * cw - sO * sw * ci
    R12 = -cO * sw - sO * cw * ci
    R21 = sO * cw + cO * sw * ci
    R22 = -sO * sw + cO * cw * ci
    R31 = sw * si
    R32 = cw * si
    out = np.empty(np.shape(a) + (6,), dtype=np.float64)
    out[..., 0] = R11 * xp + R12 * yp
    out[..., 1] = R21 * xp + R22 * yp
    out[..., 2] = R31 * xp + R32 * yp
    out[..., 3] = R11 * vxp + R12 * vyp
    out[..., 4] = R21 * vxp + R22 * vyp
    out[..., 5] = R31 * vxp + R32 * vyp

    fast = (inc == 0.0) & (raan == 0.0)
    if np.any(fast):
        co, so = np.cos(omega), np.sin(omega)
        out[..., 0] = np.where(fast, co * xp - so * yp, out[..., 0])
        out[..., 1] = np.where(fast, so * xp + co * yp, out[..., 1])
        out[..., 2] = np.where(fast, 0.0, out[..., 2])
        out[..., 3] = np.where(fast, co * vxp - so * vyp, out[..., 3])
        out[..., 4] = np.where(fast, so * vxp + co * vyp, out[..., 4])
        out[..., 5] = np.where(fast, 0.0, out[..., 5])
    return out


# ── target-plane gauge (orbital.h::gauge_from_orbit / orb_varpi_gauge) ───────
class PlaneGauge:
    """Batched twin of the C's `PlaneGauge`.

    The gauge is what makes the mean-longitude phase channel obs[13,14] frame
    invariant. It is built from the TARGET, and in the nav wrapper that means
    the target ESTIMATE — NON-ISSUE-18 measured that substitution safe with a
    closed form, `d(dlam)/d(eps) = tan(di_rel/2) sin(psi - Omega)`, i.e. the
    gauge error vanishes exactly as guidance nulls the plane.
    """

    __slots__ = ('e1', 'e2', 'e3', 'identity')

    def __init__(self, inc, raan):
        self.identity = (inc == 0.0) & (raan == 0.0)
        w = hhat_from_elements(inc, raan)
        nx, ny = -w[..., 1], w[..., 0]
        nn = np.sqrt(nx * nx + ny * ny)
        good = nn > 1e-14
        e1 = np.zeros(w.shape, dtype=np.float64)
        e1[..., 0] = np.where(good, nx / np.where(good, nn, 1.0), 1.0)
        e1[..., 1] = np.where(good, ny / np.where(good, nn, 1.0), 0.0)
        self.e1 = e1
        self.e3 = w
        self.e2 = cross(w, e1)

    def rotate(self, X):
        """Express a (N,6) inertial Cartesian state in the gauge frame."""
        r, v = X[..., :3], X[..., 3:6]
        out = np.empty(X.shape, dtype=np.float64)
        out[..., 0] = np.einsum('...i,...i->...', r, self.e1)
        out[..., 1] = np.einsum('...i,...i->...', r, self.e2)
        out[..., 2] = np.einsum('...i,...i->...', r, self.e3)
        out[..., 3] = np.einsum('...i,...i->...', v, self.e1)
        out[..., 4] = np.einsum('...i,...i->...', v, self.e2)
        out[..., 5] = np.einsum('...i,...i->...', v, self.e3)
        return out

    def varpi(self, X):
        """orb_varpi_gauge: omega + raan of the state expressed in the gauge.

        The C short-circuits identity-gauge rows to `o->omega + o->raan` and
        skips the round trip. This does not: on those rows the basis is
        e1 = x-hat, e2 = y-hat, e3 = z-hat EXACTLY (`nn > 1e-14` fails, so
        e1 := (1,0,0), and e2 = e3 x e1 = (0,1,0)), so `rotate` is the exact
        identity map and the round trip decodes the same state the C's element
        set describes. Doing it uniformly costs one decode and removes the need
        to carry the chaser's bare RAAN anywhere — which orbital.h's CONSUMER
        RULE forbids reading alone, and which the getter deliberately does not
        return.
        """
        el = cartesian_to_elements_3d(self.rotate(X))
        return el['omega'] + el['raan']


def lambda_gauge(gauge, X, M):
    """orb_lambda_gauge = M + varpi_gauge. M is a time coordinate and therefore
    frame invariant, so it is taken from the element set, never re-derived."""
    return M + gauge.varpi(X)


# ── dimension-generic exact two-body propagation ─────────────────────────────
def propagate_cartesian_nd(X, dt, iters=nm._NEWTON_ITERS, dE0=None,
                           want_dE=False):
    """Lagrange f&g propagation of (..., 2d) Cartesian states, d in {2, 3}.

    Structurally identical to `nav_math.propagate_cartesian`: r0 = |r|,
    sigma0 = r.v/sqrt(mu), and f/g/fdot/gdot are SCALARS multiplying whole
    vectors, so the routine is dimension-generic by index change alone. N3D-C
    measured the equatorial rows of this form reproducing the shipped 2D
    routine bit-exactly (max abs diff 0.000e+00 over 300 draws) and the 3D rows
    agreeing with the universal-variable oracle to 3.1e-8 m at dt = 60 s.

    The 2D routine is nonetheless left in place and still used by the 2D
    lineage: the bit-exactness anchor should be STRUCTURAL, not empirical on
    one toolchain.
    """
    X = np.asarray(X, dtype=np.float64)
    d = X.shape[-1] // 2
    r0v, v0v = X[..., :d], X[..., d:]
    with np.errstate(all='ignore'):
        r0 = np.sqrt(np.einsum('...i,...i->...', r0v, r0v))
        v2 = np.einsum('...i,...i->...', v0v, v0v)
        inv_a = 2.0 / r0 - v2 / MU
        ok0 = np.isfinite(inv_a) & (inv_a > 1e-30) & (r0 > 1.0)
        inv_a = np.where(ok0, inv_a, 1.0 / 7.0e6)
        r0s = np.where(ok0, r0, 7.0e6)
        sig0 = np.where(ok0, np.einsum('...i,...i->...', r0v, v0v) / SQ_MU, 0.0)

        a = 1.0 / inv_a
        sqa = np.sqrt(a)
        a_sqa = a * sqa
        n = SQ_MU / a_sqa
        target = SQ_MU * dt
        dE = (n * dt) if dE0 is None else dE0
        for _ in range(iters):
            s, c = np.sin(dE), np.cos(dE)
            F = (a_sqa * (dE - s) + sig0 * a * (1.0 - c)
                 + r0s * sqa * s) - target
            dF = a_sqa * (1.0 - c) + sig0 * a * s + r0s * sqa * c
            dE = dE - F / dF

        s, c = np.sin(dE), np.cos(dE)
        f = 1.0 - (a / r0s) * (1.0 - c)
        g = dt - (a_sqa / SQ_MU) * (dE - s)
        rv = f[..., None] * r0v + g[..., None] * v0v
        r = np.sqrt(np.einsum('...i,...i->...', rv, rv))
        fdot = -(SQ_MU * sqa / (r * r0s)) * s
        gdot = 1.0 - (a / r) * (1.0 - c)
        vv = fdot[..., None] * r0v + gdot[..., None] * v0v

    Y = np.concatenate([rv, vv], axis=-1)
    ok = ok0 & np.isfinite(Y).all(axis=-1)
    Y = np.where(ok[..., None], Y, X)
    if want_dE:
        return Y, ok, dE
    return Y, ok
