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


def stm_fd_nd(X, dt, iters=nm._NEWTON_ITERS, warm_iters=2):
    """Central-difference STM of `propagate_cartesian_nd`. (N,2d) -> (N,2d,2d).

    The 6-state generalisation of `nav_math.stm_fd`, kept as the BASELINE the
    analytic STM is timed and cross-checked against — not as the shipped path.
    n3d_REDTEAM BLOCKER-3: this object is 87% of the 3D filter tick (12 of its
    13 propagations are FD probes) and it appears in BOTH the filter and the
    acquisition surrogate, so one analytic implementation pays twice.
    """
    X = np.asarray(X, dtype=np.float64)
    n, m = X.shape[0], X.shape[1]
    d = m // 2
    Y, ok, dE = propagate_cartesian_nd(X, dt, iters, want_dE=True)
    F = np.empty((n, m, m), dtype=np.float64)
    h = np.array([nm.H_POS] * d + [nm.H_VEL] * d)
    Z = np.empty((2 * n, m), dtype=np.float64)
    dE2 = np.concatenate([dE, dE])
    for j in range(m):
        Z[:n] = X; Z[:n, j] += h[j]
        Z[n:] = X; Z[n:, j] -= h[j]
        W, okw = propagate_cartesian_nd(Z, dt, warm_iters, dE0=dE2)
        F[:, :, j] = (W[:n] - W[n:]) / (2.0 * h[j])
        ok &= okw[:n] & okw[n:]
    return F, ok, Y


def stm_analytic_nd(X, dt, iters=nm._NEWTON_ITERS):
    """ANALYTIC two-body state transition matrix. (N,2d) -> (N,2d,2d), d in {2,3}.

    n3d_REDTEAM BLOCKER-3's first and only zero-science-risk cost lever.

    Method. The flow is r = f r0 + g v0, v = fdot r0 + gdot v0, and the four
    Lagrange coefficients depend on the initial state ONLY through three
    scalars — r0 = |r0|, sigma0 = r0.v0/sqrt(mu), a (vis-viva) — plus the
    eccentric-anomaly difference dE, which Kepler's equation makes an implicit
    function of the same three. So

        dPhi = f I + r0 (grad f)^T + g [0 I] + v0 (grad g)^T      (top rows)

    and every gradient follows by the chain rule from closed-form partials.
    dE's own partials come from the implicit function theorem on

        K(dE) = a^1.5 (dE - sin dE) + sigma0 a (1 - cos dE)
                + r0 sqrt(a) sin dE - sqrt(mu) dt = 0

    whose dK/ddE is exactly sqrt(a) * r — the classical identity, which is what
    makes this cheap.

    Why the dE parameterisation and not the universal-variable one. The
    universal form needs d(U_n)/d(alpha) = (chi U_{n-1} - n U_n)/(2 alpha),
    and at LEO alpha ~ 1.4e-7 with dt = 60 s that numerator is a difference of
    two ~3.2e4 quantities leaving ~12.6, i.e. it throws away 3.4 decimal
    digits before it starts. Every expression below is free of 1/alpha and of
    that cancellation. The cost is that it is elliptical-only — which is
    exactly the shipped envelope, and a hyperbolic ESTIMATE is masked
    `ok = False` and reinitialised by the divergence guard, the same
    convention `propagate_cartesian` already uses.

    Returns (F, ok, Y) with Y the nominal propagation, mirroring `stm_fd`.
    """
    X = np.asarray(X, dtype=np.float64)
    n, m = X.shape[0], X.shape[1]
    d = m // 2
    r0v, v0v = X[:, :d], X[:, d:]

    with np.errstate(all='ignore'):
        Y, ok, dE = propagate_cartesian_nd(X, dt, iters, want_dE=True)

        r0 = np.sqrt(np.einsum('ni,ni->n', r0v, r0v))
        v2 = np.einsum('ni,ni->n', v0v, v0v)
        inv_a = 2.0 / r0 - v2 / MU
        good = np.isfinite(inv_a) & (inv_a > 1e-30) & (r0 > 1.0)
        inv_a = np.where(good, inv_a, 1.0 / 7.0e6)
        r0 = np.where(good, r0, 7.0e6)
        sig0 = np.where(good, np.einsum('ni,ni->n', r0v, v0v) / SQ_MU, 0.0)
        a = 1.0 / inv_a
        sqa = np.sqrt(a)
        a15 = a * sqa

        S = np.sin(dE)
        cE = np.cos(dE)
        Cv = 1.0 - cE                       # C = 1 - cos dE

        r = a + (r0 - a) * cE + sig0 * sqa * S
        r = np.where(r > 1.0, r, 1.0)

        f = 1.0 - (a / r0) * Cv
        g = dt - (a15 / SQ_MU) * (dE - S)
        fd = -(SQ_MU * sqa / (r * r0)) * S
        gd = 1.0 - (a / r) * Cv

        # -- dE partials (implicit function theorem; dK/ddE = sqrt(a) * r) ----
        dKdE = sqa * r
        K_r0 = sqa * S
        K_s0 = a * Cv
        K_a = 1.5 * sqa * (dE - S) + sig0 * Cv + r0 * S / (2.0 * sqa)
        E_r0 = -K_r0 / dKdE
        E_s0 = -K_s0 / dKdE
        E_a = -K_a / dKdE

        # -- explicit partials at FIXED dE ------------------------------------
        r_r0, r_s0 = cE, sqa * S
        r_a = 1.0 - cE + sig0 * S / (2.0 * sqa)
        r_E = -(r0 - a) * S + sig0 * sqa * cE

        f_r0 = (a / (r0 * r0)) * Cv
        f_a = -Cv / r0
        f_E = -(a / r0) * S

        g_a = -(1.5 * sqa / SQ_MU) * (dE - S)
        g_E = -(a15 / SQ_MU) * Cv

        rr0 = r * r0
        k = SQ_MU * sqa * S
        fd_r0 = k * (r_r0 * r0 + r) / (rr0 * rr0)
        fd_s0 = k * r_s0 / (r * rr0)
        fd_a = -SQ_MU * S / (2.0 * sqa * rr0) + k * r_a / (r * rr0)
        fd_E = -SQ_MU * sqa * cE / rr0 + k * r_E / (r * rr0)

        aC_r2 = a * Cv / (r * r)
        gd_r0 = aC_r2 * r_r0
        gd_s0 = aC_r2 * r_s0
        gd_a = -Cv / r + aC_r2 * r_a
        gd_E = -(a / r) * S + aC_r2 * r_E

        # -- total partials w.r.t. (r0, sigma0, a) ----------------------------
        F_r0 = f_r0 + f_E * E_r0
        F_s0 = f_E * E_s0
        F_a = f_a + f_E * E_a
        G_r0 = g_E * E_r0
        G_s0 = g_E * E_s0
        G_a = g_a + g_E * E_a
        FD_r0 = fd_r0 + fd_E * E_r0
        FD_s0 = fd_s0 + fd_E * E_s0
        FD_a = fd_a + fd_E * E_a
        GD_r0 = gd_r0 + gd_E * E_r0
        GD_s0 = gd_s0 + gd_E * E_s0
        GD_a = gd_a + gd_E * E_a

        # -- gradients of the three scalars w.r.t. the 2d-state ---------------
        grad_r0 = np.zeros((n, m))
        grad_r0[:, :d] = r0v / r0[:, None]
        grad_s0 = np.empty((n, m))
        grad_s0[:, :d] = v0v / SQ_MU
        grad_s0[:, d:] = r0v / SQ_MU
        grad_a = np.empty((n, m))
        a2 = a * a
        grad_a[:, :d] = (2.0 * a2 / (r0 ** 3))[:, None] * r0v
        grad_a[:, d:] = (2.0 * a2 / MU)[:, None] * v0v

        # One (n,4,3) @ (n,3,2d) matmul for all four gradients, and one
        # (n,d,2) @ (n,2,2d) per block for the outer-product accumulation.
        # At the training batch shape this routine is numpy-call-overhead
        # bound, not FLOP bound, so collapsing 20 broadcast ops into 3 matmuls
        # is most of the difference between "analytic" and "analytic and fast".
        coef = np.empty((n, 4, 3))
        coef[:, 0, 0] = F_r0;  coef[:, 0, 1] = F_s0;  coef[:, 0, 2] = F_a
        coef[:, 1, 0] = G_r0;  coef[:, 1, 1] = G_s0;  coef[:, 1, 2] = G_a
        coef[:, 2, 0] = FD_r0; coef[:, 2, 1] = FD_s0; coef[:, 2, 2] = FD_a
        coef[:, 3, 0] = GD_r0; coef[:, 3, 1] = GD_s0; coef[:, 3, 2] = GD_a
        basis = np.stack([grad_r0, grad_s0, grad_a], axis=1)      # (n,3,2d)
        grads = coef @ basis                                       # (n,4,2d)

        Phi = np.zeros((n, m, m))
        I = np.eye(d)
        Phi[:, :d, :d] = f[:, None, None] * I
        Phi[:, :d, d:] = g[:, None, None] * I
        Phi[:, d:, :d] = fd[:, None, None] * I
        Phi[:, d:, d:] = gd[:, None, None] * I
        RV = np.stack([r0v, v0v], axis=2)                          # (n,d,2)
        Phi[:, :d, :] += RV @ grads[:, 0:2, :]
        Phi[:, d:, :] += RV @ grads[:, 2:4, :]

    ok = ok & good & np.isfinite(Phi).all(axis=(1, 2))
    Phi = np.where(ok[:, None, None], Phi, np.eye(m))
    return Phi, ok, Y


def symplectic_residual(Phi):
    """max |Phi^T J Phi - J| per row. Parameter-free exactness check.

    Every two-body STM is symplectic identically, so this bounds the STM's
    own error without needing a reference implementation. The shipped FD-STM
    sits at its 1 m / 1e-3 m/s truncation floor (~1e-6); an analytic one should
    be at round-off.
    """
    m = Phi.shape[-1]
    d = m // 2
    J = np.zeros((m, m))
    J[:d, d:] = np.eye(d)
    J[d:, :d] = -np.eye(d)
    R = np.swapaxes(Phi, 1, 2) @ J @ Phi - J
    return np.abs(R).max(axis=(1, 2))


def process_noise_nd(dt, q_a, d=3):
    """Continuous white-noise-acceleration Q for a d-dimensional double
    integrator. `nav_math.process_noise` is the d = 2 special case."""
    I = np.eye(d)
    Q = np.zeros((2 * d, 2 * d))
    Q[:d, :d] = (dt ** 3 / 3.0) * I
    Q[:d, d:] = (dt ** 2 / 2.0) * I
    Q[d:, :d] = (dt ** 2 / 2.0) * I
    Q[d:, d:] = dt * I
    return q_a * Q


# ═══════════════════════════════════════════════════════════════════════════
# 3D modified-spherical coordinates (MSC6) and the pole
# ═══════════════════════════════════════════════════════════════════════════
#
# n3d_REDTEAM BLOCKER-1. The two binding recon documents specify the chart pole
# two different ways, 90 degrees apart, and the chart singularity is at
# |el| = 90:
#
#   N3D-A  1.1  pole = h-hat_c(t0), the chaser orbit normal at epoch
#   N3D-C  3b   pole = unit(u0 x h-hat_t), i.e. a pole IN the orbit plane
#
# N3D-C's is DELETED. The LOS of a close rendezvous is predominantly
# along-track — in that same plane — so it sweeps straight through such a pole
# once per orbit; the red-team measured 90.00 degrees of elevation, exact gimbal
# lock, in the COPLANAR arm, which is the arm N3D-A uses as its 2D-reduction
# regression anchor.
#
# N3D-A's pole is adopted. Its own caveat ("the gimbal guard was specified but
# never exercised, max |el| 27 deg") is understated: that held only because the
# prototype enforced di <= asin(rho/r) and skipped infeasible cells, which is a
# property of a HELD rendezvous, not of an APPROACH. Closed form:
#
#     |el| > 60 deg   <=>   rho_inplane <  0.577 * r * sin(di_rel)
#     |el| > 45 deg   <=>   rho_inplane <  1.000 * r * sin(di_rel)
#
# so at di_rel = 1 deg (the X3 rung) the 45-degree guard arms below
# rho_inplane = 118 km and the 60-degree guard below 68 km, against a 30 km
# success box. The guard therefore fires INSIDE the box at the headline rung
# either way, and lowering the trigger to 45 makes it fire MORE, not less.
# That is the correct trade and it is taken deliberately: 45 bounds the
# azimuth-noise inflation 1/cos^2(el) at 2.0 where 60 allows 4.0, and the
# re-pole is an exact similarity transform, so the cost of firing is a
# round-trip at 1e-12 and a NEES that does not step. The design goal is not to
# keep re-poles out of the success box — that is unreachable — it is to handle
# them cleanly inside it, and to COUNT them.

REPOLE_EL_DEG = 45.0
REPOLE_EL_RAD = np.deg2rad(REPOLE_EL_DEG)

IDX_AZ, IDX_EL, IDX_WA, IDX_WE, IDX_RDOT, IDX_LNRHO = 0, 1, 2, 3, 4, 5
_LOG_RHO_MAX = nm._LOG_RHO_MAX
_RHO_FLOOR = nm._RHO_FLOOR


def pole_frame(sat_cart):
    """(N,6) chaser state -> (N,3,3) rows [e1; e2; e3], the epoch-frozen frame.

    +z = h-hat_c(t0) (the chaser orbit normal AT EPOCH), +x = the chaser radial
    at epoch. Frozen, not co-rotating: an LVLH reference injects a deterministic
    orbital-rate bearing sweep that dominates the signal and couples chaser
    velocity into H.

    At di_rel = 0 this frame makes `el` and `el-dot` identically zero, so the
    6-state collapses onto the shipped 4-state component-for-component. That
    reduction is the cheapest regression test the 3D filter has, and it is
    exactly what N3D-C's in-plane pole destroys.
    """
    r, v = sat_cart[:, :3], sat_cart[:, 3:6]
    e3 = unit(cross(r, v))
    e1 = unit(r)
    e2 = cross(e3, e1)
    return np.stack([e1, e2, e3], axis=1)


def _rel_pole(sat, tgt, Rp):
    """Relative state expressed in the pole frame. (N,3) d, (N,3) dv."""
    d = np.einsum('nij,nj->ni', Rp, tgt[:, :3] - sat[:, :3])
    dv = np.einsum('nij,nj->ni', Rp, tgt[:, 3:6] - sat[:, 3:6])
    return d, dv


def _basis(az, el):
    """(u, e_az, e_el, p) for the given angles. e_az = d(u)/d(az)/cos el."""
    ca, sa = np.cos(az), np.sin(az)
    ce, se = np.cos(el), np.sin(el)
    u = np.stack([ce * ca, ce * sa, se], axis=-1)
    e_az = np.stack([-sa, ca, np.zeros_like(sa)], axis=-1)
    e_el = np.stack([-se * ca, -se * sa, ce], axis=-1)
    p = np.stack([ca, sa, np.zeros_like(ca)], axis=-1)
    return u, e_az, e_el, p


def msc6_encode(sat, tgt, Rp):
    """[az, el, w_a, w_e, rhodot/rho, ln rho] from inertial Cartesian states.

    Guarded exactly like the 2D twin: `log(max(rho, 1 m))` cannot fire for a
    physical geometry and turns a `math domain error` 50M steps from now into a
    bounded number.
    """
    d, dv = _rel_pole(sat, tgt, Rp)
    rho2 = np.maximum(np.einsum('ni,ni->n', d, d), _RHO_FLOOR ** 2)
    rho = np.sqrt(rho2)
    u = d / rho[:, None]
    el = np.arcsin(np.clip(u[:, 2], -1.0, 1.0))
    az = np.arctan2(u[:, 1], u[:, 0])
    _, e_az, e_el, _ = _basis(az, el)
    s = np.einsum('ni,ni->n', u, dv)
    w = (dv - s[:, None] * u) / rho[:, None]
    y = np.empty((sat.shape[0], 6), dtype=np.float64)
    y[:, IDX_AZ] = az
    y[:, IDX_EL] = el
    y[:, IDX_WA] = np.einsum('ni,ni->n', w, e_az)
    y[:, IDX_WE] = np.einsum('ni,ni->n', w, e_el)
    y[:, IDX_RDOT] = s / rho
    y[:, IDX_LNRHO] = np.log(rho)
    return y


def msc6_decode(y, sat, Rp):
    """Inverse of `msc6_encode`. (N,6) inertial Cartesian target state."""
    az, el = y[:, IDX_AZ], y[:, IDX_EL]
    rho = np.exp(np.minimum(y[:, IDX_LNRHO], _LOG_RHO_MAX))
    u, e_az, e_el, _ = _basis(az, el)
    w = y[:, IDX_WA, None] * e_az + y[:, IDX_WE, None] * e_el
    d = rho[:, None] * u
    dv = (rho * y[:, IDX_RDOT])[:, None] * u + rho[:, None] * w
    # back to inertial: rows of Rp are the frame axes, so Rp^T maps back.
    di = np.einsum('nji,nj->ni', Rp, d)
    dvi = np.einsum('nji,nj->ni', Rp, dv)
    return np.concatenate([sat[:, :3] + di, sat[:, 3:6] + dvi], axis=1)


def msc6_decode_jac(y, Rp):
    """ANALYTIC d(x_target)/d(y), (N,6,6), in the inertial frame.

    Replaces the 2 x 6 finite-difference encode-Jacobian evaluations of the
    prototype. Together with `stm_analytic_nd` it turns the MSC transition
    Jacobian into

        F = inv(J_dec(y_new, Rp)) @ Phi_cart @ J_dec(y_old, Rp)

    i.e. ONE propagation + one analytic STM + one 6x6 inverse, in place of the
    13 propagations the FD chain needs.
    """
    n = y.shape[0]
    az, el = y[:, IDX_AZ], y[:, IDX_EL]
    wa, we = y[:, IDX_WA], y[:, IDX_WE]
    rr = y[:, IDX_RDOT]
    rho = np.exp(np.minimum(y[:, IDX_LNRHO], _LOG_RHO_MAX))
    u, e_az, e_el, p = _basis(az, el)
    se = np.sin(el)
    ce = np.cos(el)
    w = wa[:, None] * e_az + we[:, None] * e_el
    dv = (rho * rr)[:, None] * u + rho[:, None] * w

    J = np.zeros((n, 6, 6))
    R = rho[:, None]
    # position block: d = rho * u
    J[:, :3, IDX_AZ] = R * ce[:, None] * e_az
    J[:, :3, IDX_EL] = R * e_el
    J[:, :3, IDX_LNRHO] = R * u
    # velocity block: dv = rho*rr*u + rho*w
    #   d(e_az)/d(az) = -p ; d(e_el)/d(az) = -sin(el) e_az ; d(e_el)/d(el) = -u
    J[:, 3:, IDX_AZ] = ((rho * rr)[:, None] * ce[:, None] * e_az
                        + R * (-wa[:, None] * p
                               - we[:, None] * se[:, None] * e_az))
    J[:, 3:, IDX_EL] = (rho * rr)[:, None] * e_el - R * we[:, None] * u
    J[:, 3:, IDX_WA] = R * e_az
    J[:, 3:, IDX_WE] = R * e_el
    J[:, 3:, IDX_RDOT] = R * u
    J[:, 3:, IDX_LNRHO] = dv
    # rotate both blocks back to inertial (Rp rows are the frame axes)
    Rt = np.swapaxes(Rp, 1, 2)
    J[:, :3, :] = Rt @ J[:, :3, :]
    J[:, 3:, :] = Rt @ J[:, 3:, :]
    return J


def repole_frames(Rp, y):
    """New pole frames for the rows whose |el| exceeds the trigger.

    Minimal rotation: e1' := u (so az' = 0) and e3' := the component of the OLD
    pole orthogonal to u (so el' = 0 exactly). At the 45-degree trigger
    |e3.u| = 0.707, so the orthogonal component has norm 0.707 and the
    construction is well conditioned — it degenerates only at the singularity
    itself, which the trigger exists to prevent reaching.
    """
    az, el = y[:, IDX_AZ], y[:, IDX_EL]
    u_p, _, _, _ = _basis(az, el)
    Rt = np.swapaxes(Rp, 1, 2)
    u = np.einsum('nij,nj->ni', Rt, u_p)          # LOS in inertial
    e3o = Rp[:, 2, :]
    e3 = unit(e3o - np.einsum('ni,ni->n', e3o, u)[:, None] * u)
    e1 = u
    e2 = cross(e3, e1)
    return np.stack([e1, e2, e3], axis=1)


def _inv2x2(S):
    a, b, c, d = S[:, 0, 0], S[:, 0, 1], S[:, 1, 0], S[:, 1, 1]
    det = a * d - b * c
    det = np.where(np.abs(det) < 1e-300, 1e-300, det)
    out = np.empty_like(S)
    out[:, 0, 0] = d / det
    out[:, 0, 1] = -b / det
    out[:, 1, 0] = -c / det
    out[:, 1, 1] = a / det
    return out


class BatchedBearingMSC6:
    """N independent bearings-only EKFs in 3D modified-spherical coordinates.

    The 6-state lift of `nav_math.BatchedBearingMPC`, keeping the property that
    makes NAV-G's design work: the measurement IS (y0, y1) exactly, so
    H = [e1; e2] and the update is LINEAR — which is what removes the premature
    covariance collapse that makes a Cartesian bearings-only EKF confidently
    wrong (measured NEES 3.9-7.9e17, 8/12 seeds divergent, in the 3D prototype).

    Three red-team dispositions are structural here, not options:

    * **Pole = h-hat_c(t0)** (BLOCKER-1). See the module comment above
      `REPOLE_EL_DEG`.
    * **Anisotropic R** (MAJOR-12). Isotropic-on-the-sphere bearing noise is
      `R = diag(sigma_beta^2/cos^2 el, sigma_beta^2)` in az/el. Writing an
      isotropic `R = sigma_beta^2 I` inflates the information by 1/cos^2 el —
      1.26x at 27 deg, 2.0x at 45, 4.0x at 60. N3D-A's prototype never saw past
      27 deg so the error would have been invisible there; BLOCKER-1 measures
      87.58 deg on the campaign's own trajectory.
    * **Analytic transition Jacobian** (BLOCKER-3). `F = Gnew @ Phi @ Jold`,
      one propagation and one 6x6 inverse in place of the 13 propagations of
      the finite-difference chain.
    """

    STATE_DIM = 6
    CART_DIM = 6
    POS_DIM = 3
    IDX_LNRHO = IDX_LNRHO

    def __init__(self, n, sigma_beta=nm.SIGMA_BETA_RAD, q_a=nm.Q_ACCEL_PSD_BO,
                 stm='analytic', repole_el_rad=REPOLE_EL_RAD):
        # 'fd_msc' is the BASELINE, not an option: it is the mechanical 6-state
        # port of the shipped 2D `BatchedBearingMPC.predict` — 13 y-space
        # transitions (decode -> propagate -> encode) for the transition
        # Jacobian plus 12 more encodes for the process-noise Jacobian. It
        # exists so BLOCKER-3's speedup is measured against the thing the
        # campaign would actually have shipped, not against a half-optimised
        # straw man. 'fd' is the intermediate (analytic chart Jacobians,
        # finite-difference Cartesian STM).
        if stm not in ('analytic', 'fd', 'fd_msc'):
            raise ValueError(
                f"stm must be 'analytic'|'fd'|'fd_msc', got {stm!r}")
        self.n = n
        self.sigma_beta = float(sigma_beta)
        self.q_a = float(q_a)
        self.stm = stm
        self.repole_el = float(repole_el_rad)
        self.y = np.zeros((n, 6))
        self.Py = np.tile(np.eye(6), (n, 1, 1))
        self.sat = np.zeros((n, 6))
        self.Rp = np.tile(np.eye(3), (n, 1, 1))
        self._I6 = np.eye(6)
        # Diagnostics the red-team requires to be SURFACED, not inferred.
        self.n_repole = np.zeros(n, dtype=np.int64)     # per-episode counter
        self.n_repole_total = 0

    def _Q(self, dt):
        """Process noise, memoised on dt (the cadence is fixed per run)."""
        q = getattr(self, '_Qc', None)
        if q is None or q[0] != dt:
            self._Qc = (dt, process_noise_nd(dt, self.q_a, 3))
        return self._Qc[1]

    # -- frames ---------------------------------------------------------------
    def set_pole(self, idx, sat_cart):
        if idx.size == 0:
            return
        self.Rp[idx] = pole_frame(sat_cart)
        self.n_repole[idx] = 0

    # -- seeding --------------------------------------------------------------
    def set_cart(self, idx, x, P, sat_cart):
        if idx.size == 0:
            return
        self.sat[idx] = sat_cart
        Rp = self.Rp[idx]
        y = msc6_encode(sat_cart, x, Rp)
        J = msc6_decode_jac(y, Rp)
        G = np.linalg.inv(J)
        Py = G @ P @ np.swapaxes(G, 1, 2)
        self.y[idx] = y
        self.Py[idx] = 0.5 * (Py + np.swapaxes(Py, 1, 2))

    def set_polar(self, idx, y, Py, sat_cart):
        """Seed directly in modified-spherical coordinates — the right object
        for a BLIND start, because the 4-decade range ignorance then lives
        entirely in `ln rho`, where it is bounded by (ln(hi/lo))^2/12."""
        if idx.size == 0:
            return
        self.sat[idx] = sat_cart
        self.y[idx] = y
        self.Py[idx] = 0.5 * (Py + np.swapaxes(Py, 1, 2))

    # -- predict / update -----------------------------------------------------
    def _predict_fd_msc(self, idx, dt, sat_from, sat_to, warm_iters=2):
        """The mechanical 6-state port of `nav_math.BatchedBearingMPC.predict`.

        13 y-space transitions + a 12-encode process-noise Jacobian. Kept only
        as BLOCKER-3's measured baseline.
        """
        Rp = self.Rp[idx]
        y = self.y[idx]
        m = idx.size
        rho = np.maximum(self.rho()[idx], _RHO_FLOOR)
        h = np.stack([1.0 / rho, 1.0 / rho, 1e-3 / rho, 1e-3 / rho,
                      1e-3 / rho, 1.0 / rho], axis=1)

        def trans(yy, sf, st, Rf, iters=nm._NEWTON_ITERS, dE0=None,
                  want_dE=False):
            xt = msc6_decode(yy, sf, Rf)
            out = propagate_cartesian_nd(xt, dt, iters, dE0=dE0,
                                         want_dE=want_dE)
            if want_dE:
                return msc6_encode(st, out[0], Rf), out[2]
            return msc6_encode(st, out[0], Rf)

        y0, dE = trans(y, sat_from, sat_to, Rp, want_dE=True)
        F = np.empty((m, 6, 6))
        Y2 = np.empty((2 * m, 6))
        s_from = np.concatenate([sat_from, sat_from])
        s_to = np.concatenate([sat_to, sat_to])
        R2 = np.concatenate([Rp, Rp])
        dE2 = np.concatenate([dE, dE])
        for j in range(6):
            Y2[:m] = y; Y2[:m, j] += h[:, j]
            Y2[m:] = y; Y2[m:, j] -= h[:, j]
            W = trans(Y2, s_from, s_to, R2, iters=warm_iters, dE0=dE2)
            d = W[:m] - W[m:]
            d[:, IDX_AZ] = nm.wrap_pi(d[:, IDX_AZ])
            F[:, :, j] = d / (2.0 * h[:, j:j + 1])

        xt = msc6_decode(y0, sat_to, Rp)
        G = np.empty((m, 6, 6))
        hh = (1.0, 1.0, 1.0, 1e-3, 1e-3, 1e-3)
        for j in range(6):
            xp = xt.copy(); xp[:, j] += hh[j]
            xm = xt.copy(); xm[:, j] -= hh[j]
            d = msc6_encode(sat_to, xp, Rp) - msc6_encode(sat_to, xm, Rp)
            d[:, IDX_AZ] = nm.wrap_pi(d[:, IDX_AZ])
            G[:, :, j] = d / (2.0 * hh[j])
        Qy = G @ process_noise_nd(dt, self.q_a, 3) @ np.swapaxes(G, 1, 2)
        Py = F @ self.Py[idx] @ np.swapaxes(F, 1, 2) + Qy
        self.y[idx] = y0
        self.Py[idx] = 0.5 * (Py + np.swapaxes(Py, 1, 2))
        self.sat[idx] = sat_to
        return (np.all(np.isfinite(y0), axis=1)
                & np.all(np.isfinite(Py), axis=(1, 2)))

    def predict(self, idx, dt, sat_from, sat_to, **_):
        if idx.size == 0:
            return np.zeros(0, dtype=bool)
        if self.stm == 'fd_msc':
            return self._predict_fd_msc(idx, dt, sat_from, sat_to)
        Rp = self.Rp[idx]
        y = self.y[idx]
        x_old = msc6_decode(y, sat_from, Rp)
        if self.stm == 'analytic':
            Phi, ok, x_new = stm_analytic_nd(x_old, dt)
        else:
            Phi, ok, x_new = stm_fd_nd(x_old, dt)
        y_new = msc6_encode(sat_to, x_new, Rp)
        Jold = msc6_decode_jac(y, Rp)
        Jnew = msc6_decode_jac(y_new, Rp)
        with np.errstate(all='ignore'):
            try:
                Gnew = np.linalg.inv(Jnew)
            except np.linalg.LinAlgError:            # pragma: no cover
                Gnew = np.linalg.pinv(Jnew)
            # Py' = G (A Py A^T + Q) G^T with A = Phi Jold. Written this way
            # the transition matrix F = G A is never formed, which drops the
            # chain from six (n,6,6) matmuls to five.
            A = Phi @ Jold
            M = A @ self.Py[idx] @ np.swapaxes(A, 1, 2) + self._Q(dt)
            Py = Gnew @ M @ np.swapaxes(Gnew, 1, 2)
        self.y[idx] = y_new
        self.Py[idx] = 0.5 * (Py + np.swapaxes(Py, 1, 2))
        self.sat[idx] = sat_to
        ok = ok & np.all(np.isfinite(y_new), axis=1) \
            & np.all(np.isfinite(Py), axis=(1, 2))
        return ok

    def update(self, idx, sat_cart, az, el):
        """Linear update, H = [e1; e2]. Returns per-row NIS (dof 2)."""
        if idx.size == 0:
            return np.zeros(0)
        self.sat[idx] = sat_cart
        y = self.y[idx]
        Py = self.Py[idx]
        # MAJOR-12: isotropic-on-the-sphere noise is anisotropic in az/el.
        ce = np.maximum(np.cos(y[:, IDX_EL]), 1e-8)
        sb2 = self.sigma_beta ** 2
        R = np.zeros((idx.size, 2, 2))
        R[:, 0, 0] = sb2 / (ce * ce)
        R[:, 1, 1] = sb2

        nu = np.stack([nm.wrap_pi(az - y[:, IDX_AZ]), el - y[:, IDX_EL]],
                      axis=1)
        S = Py[:, :2, :2] + R
        Sinv = _inv2x2(S)
        K = Py[:, :, :2] @ Sinv                        # (n,6,2)
        self.y[idx] = y + (K @ nu[:, :, None])[:, :, 0]
        self.y[idx, IDX_AZ] = nm.wrap_pi(self.y[idx, IDX_AZ])
        self.y[idx, IDX_EL] = np.clip(self.y[idx, IDX_EL],
                                      -0.5 * np.pi + 1e-9, 0.5 * np.pi - 1e-9)
        H = np.zeros((2, 6))
        H[0, IDX_AZ] = 1.0
        H[1, IDX_EL] = 1.0
        IKH = self._I6 - K @ H
        Pn = (IKH @ Py @ np.swapaxes(IKH, 1, 2)
              + K @ R @ np.swapaxes(K, 1, 2))
        self.Py[idx] = 0.5 * (Pn + np.swapaxes(Pn, 1, 2))
        return np.einsum('ni,nij,nj->n', nu, Sinv, nu) / 2.0

    # -- BLOCKER-1: the re-pole, as an EXACT similarity transform -------------
    def repole(self, idx=None):
        """Re-pole every row whose |el| exceeds the trigger. Returns the rows.

        `y' = encode(decode(y))` under the new frame, and
        `P' = J P J^T` with `J = inv(Jdec(y', Rp')) @ Jdec(y, Rp)` — the exact
        chain rule of the chart change, not an approximation. Both properties
        the red-team asks for follow and are checked in verify_n3dnav stage d1:
        the Cartesian round trip through the re-pole is the identity to 1e-12,
        and NEES is continuous across it because the underlying Cartesian
        (mean, covariance) pair is literally unchanged.
        """
        if idx is None:
            idx = np.arange(self.n)
        if idx.size == 0:
            return idx[:0]
        hot = idx[np.abs(self.y[idx, IDX_EL]) > self.repole_el]
        if hot.size == 0:
            return hot
        y = self.y[hot]
        Rold = self.Rp[hot]
        sat = self.sat[hot]
        Rnew = repole_frames(Rold, y)
        x = msc6_decode(y, sat, Rold)
        y2 = msc6_encode(sat, x, Rnew)
        Jold = msc6_decode_jac(y, Rold)
        Jnew = msc6_decode_jac(y2, Rnew)
        with np.errstate(all='ignore'):
            T = np.linalg.inv(Jnew) @ Jold
            P2 = T @ self.Py[hot] @ np.swapaxes(T, 1, 2)
        self.y[hot] = y2
        self.Py[hot] = 0.5 * (P2 + np.swapaxes(P2, 1, 2))
        self.Rp[hot] = Rnew
        self.n_repole[hot] += 1
        self.n_repole_total += int(hot.size)
        return hot

    # -- interface ------------------------------------------------------------
    def mean_cov(self, idx):
        if idx.size == 0:
            return np.zeros((0, 6)), np.zeros((0, 6, 6))
        Rp = self.Rp[idx]
        y = self.y[idx]
        x = msc6_decode(y, self.sat[idx], Rp)
        J = msc6_decode_jac(y, Rp)
        P = J @ self.Py[idx] @ np.swapaxes(J, 1, 2)
        return x, 0.5 * (P + np.swapaxes(P, 1, 2))

    def mean_cart(self, idx):
        return msc6_decode(self.y[idx], self.sat[idx], self.Rp[idx])

    def trace(self):
        return np.trace(self.Py, axis1=1, axis2=2)

    def rho(self):
        return np.exp(np.minimum(self.y[:, IDX_LNRHO], _LOG_RHO_MAX))


def nees_nd(x, P, truth):
    """Normalised estimation error squared / dof."""
    if x.shape[0] == 0:
        return np.zeros(0)
    d = x.shape[1]
    err = x - truth
    try:
        sol = np.linalg.solve(P, err[:, :, None])[:, :, 0]
    except np.linalg.LinAlgError:                      # pragma: no cover
        return np.full(x.shape[0], np.nan)
    return np.einsum('ni,ni->n', err, sol) / float(d)


# 95% two-sided chi-square bounds normalised by dof, 6 dof (nav_math has 4).
NEES6_LO, NEES6_HI = 1.2373 / 6.0, 14.4494 / 6.0


class BatchedRangeBearingEKF3D:
    """N independent range + TWO-ANGLE EKFs on the target's inertial 6-state.

    The N1-rb3d control arm. n3d_REDTEAM and N3D-C 5 both call it essential
    rather than optional, and for the same reason T-RB was essential in NAV-F:
    it is the arm in which range is MEASURED, so a maneuver buys the estimator
    nothing. If the bearings-only treatment and this control move together, the
    effect is not observability — it is guidance, or the box, or the warm start.
    Without it a positive result has no attribution.

    Deliberately the SAME sensor as the bearings-only arm plus one row: the two
    angles are az/el in the identical epoch-frozen pole frame, with the same
    isotropic-on-the-sphere noise (MAJOR-12), so the two arms differ in exactly
    one measurement channel and nothing else.
    """

    STATE_DIM = 6
    CART_DIM = 6
    POS_DIM = 3
    IDX_LNRHO = None          # Cartesian filter: no ln-rho slot (MAJOR-9)

    def __init__(self, n, sigma_rho=nm.SIGMA_RHO_M,
                 sigma_beta=nm.SIGMA_BETA_RAD, q_a=nm.Q_ACCEL_PSD_RB,
                 sigma_v0=nm.SIGMA_V0, stm='analytic'):
        self.n = n
        self.sigma_rho = float(sigma_rho)
        self.sigma_beta = float(sigma_beta)
        self.q_a = float(q_a)
        self.sigma_v0 = float(sigma_v0)
        self.stm = stm
        self.x = np.zeros((n, 6))
        self.P = np.tile(np.eye(6) * 1e6, (n, 1, 1))
        self.Rp = np.tile(np.eye(3), (n, 1, 1))
        self._I6 = np.eye(6)
        # Present so the wrapper's shared code paths do not have to branch.
        self.n_repole = np.zeros(n, dtype=np.int64)
        self.n_repole_total = 0

    def _Q(self, dt):
        q = getattr(self, '_Qc', None)
        if q is None or q[0] != dt:
            self._Qc = (dt, process_noise_nd(dt, self.q_a, 3))
        return self._Qc[1]

    def set_pole(self, idx, sat_cart):
        if idx.size == 0:
            return
        self.Rp[idx] = pole_frame(sat_cart)

    def repole(self, idx=None):
        """No chart, no singularity, nothing to re-pole. The pole frame here is
        only the measurement basis, and az/el noise is drawn against it exactly
        as in the bearings-only arm so the two sensors are identical."""
        return np.zeros(0, dtype=np.int64)

    # -- init -----------------------------------------------------------------
    def initialize(self, idx, sat_cart, rho, az, el):
        """Single-measurement init: inverted measurement + circular velocity."""
        if idx.size == 0:
            return
        Rp = self.Rp[idx]
        u_p, e_az, e_el, _ = _basis(az, el)
        u = np.einsum('nji,nj->ni', Rp, u_p)              # pole -> inertial
        p = sat_cart[:, :3] + rho[:, None] * u
        r = np.maximum(_norm(p), 1.0)
        v_c = np.sqrt(MU / r)
        # Prograde circular in the CHASER's plane — the least-committal guess
        # that is still a bound orbit.
        h = unit(cross(sat_cart[:, :3], sat_cart[:, 3:6]))
        v = v_c[:, None] * unit(cross(h, p))
        self.x[idx] = np.concatenate([p, v], axis=1)

        # Measurement-Jacobian-consistent position block, velocity isotropic.
        ce = np.maximum(np.cos(el), 1e-8)
        a_az = np.einsum('nji,nj->ni', Rp, e_az)
        a_el = np.einsum('nji,nj->ni', Rp, e_el)
        s_t = (rho * self.sigma_beta)
        P = np.zeros((idx.size, 6, 6))
        P[:, :3, :3] = (self.sigma_rho ** 2) * u[:, :, None] * u[:, None, :]
        P[:, :3, :3] += (s_t / ce)[:, None, None] ** 2 \
            * a_az[:, :, None] * a_az[:, None, :]
        P[:, :3, :3] += (s_t ** 2)[:, None, None] \
            * a_el[:, :, None] * a_el[:, None, :]
        ii = np.arange(3, 6)
        P[:, ii, ii] = self.sigma_v0 ** 2
        self.P[idx] = P

    def set_cart(self, idx, x, P, sat_cart):
        if idx.size == 0:
            return
        self.x[idx] = x
        self.P[idx] = P

    # -- predict / update -----------------------------------------------------
    def predict(self, idx, dt, sat_from=None, sat_to=None, **_):
        if idx.size == 0:
            return np.zeros(0, dtype=bool)
        X = self.x[idx]
        if self.stm == 'analytic':
            F, ok, Y = stm_analytic_nd(X, dt)
        else:
            F, ok, Y = stm_fd_nd(X, dt)
        P = F @ self.P[idx] @ np.swapaxes(F, 1, 2) + self._Q(dt)
        self.x[idx] = Y
        self.P[idx] = 0.5 * (P + np.swapaxes(P, 1, 2))
        return ok

    def update(self, idx, sat_cart, rho, az, el):
        """Joseph-form update on (range, az, el). Returns per-row NIS / 3."""
        if idx.size == 0:
            return np.zeros(0)
        X = self.x[idx]
        P = self.P[idx]
        Rp = self.Rp[idx]
        d = np.einsum('nij,nj->ni', Rp, X[:, :3] - sat_cart[:, :3])
        rho_h = np.maximum(_norm(d), 1e-6)
        u = d / rho_h[:, None]
        el_h = np.arcsin(np.clip(u[:, 2], -1.0, 1.0))
        az_h = np.arctan2(u[:, 1], u[:, 0])
        _, e_az, e_el, _ = _basis(az_h, el_h)
        ce = np.maximum(np.cos(el_h), 1e-8)

        H = np.zeros((idx.size, 3, 6))
        H[:, 0, :3] = np.einsum('ni,nij->nj', u, Rp)
        H[:, 1, :3] = np.einsum('ni,nij->nj', e_az, Rp) / (rho_h * ce)[:, None]
        H[:, 2, :3] = np.einsum('ni,nij->nj', e_el, Rp) / rho_h[:, None]

        R = np.zeros((idx.size, 3, 3))
        R[:, 0, 0] = max(self.sigma_rho, 1e-6) ** 2
        R[:, 1, 1] = self.sigma_beta ** 2 / (ce * ce)
        R[:, 2, 2] = self.sigma_beta ** 2

        nu = np.stack([rho - rho_h, nm.wrap_pi(az - az_h), el - el_h], axis=1)
        Ht = np.swapaxes(H, 1, 2)
        S = H @ P @ Ht + R
        Sinv = np.linalg.inv(S)
        K = P @ Ht @ Sinv
        self.x[idx] = X + (K @ nu[:, :, None])[:, :, 0]
        IKH = self._I6 - K @ H
        Pn = IKH @ P @ np.swapaxes(IKH, 1, 2) + K @ R @ np.swapaxes(K, 1, 2)
        self.P[idx] = 0.5 * (Pn + np.swapaxes(Pn, 1, 2))
        return np.einsum('ni,nij,nj->n', nu, Sinv, nu) / 3.0

    # -- interface ------------------------------------------------------------
    def mean_cov(self, idx):
        return self.x[idx], self.P[idx]

    def mean_cart(self, idx):
        return self.x[idx]

    def trace(self):
        return np.trace(self.P, axis1=1, axis2=2)


# ═══════════════════════════════════════════════════════════════════════════
# J2 — secular perturbation in the filter (ext-j2 rung)
# ═══════════════════════════════════════════════════════════════════════════
#
# Ported from scripts/orbital/extj2/j2_nav_filter_probe.py, whose head-to-head
# (N=128, nav60, LEO, i_t ~ U(30,60) deg, truth J2 in every arm) DECIDED this
# spec rather than assuming it. At the 24 h operating point, against a matched
# two-body control at 458.0 m / NEES 1.00 / 89.1% in band:
#
#   FIXED   J2 state, two-body covariance   1202.7 m (2.63x)  NEES 11.37  21.9%
#   C1-an   J2 state, analytic J2 cov       1234.5 m (2.70x)  NEES 11.46  18.8%
#   C1-fd   J2 state, FD J2 cov              457.3 m (1.00x)  NEES  1.01  87.5%
#
# The lesson is that the covariance is not a bookkeeping detail. Putting J2 in
# the STATE alone does not merely leave the filter overconfident (NEES 11.4) —
# the overconfident P shrinks the Kalman gain, so it also pays 2.63x in
# POSITION. The analytic O(J2) STM removes only ~59% of the two-body STM's
# error (p05 0.41) and does not fix it. The finite-difference STM restores both
# to three digits, and that is the shipped path.
#
# Cost, measured: predict() goes 0.603 -> 3.072 ms/tick (5.09x), which takes a
# full bearings_only nav step to 217% of baseline. That is the price of the
# only arm that is consistent, and it is paid deliberately.
#
# NOT TAKEN: the element-space covariance optimisation. It is identified but
# unmeasured, and a mid-campaign swap of the covariance path is exactly the
# compound change this project keeps being burned by.

J2_COEF = 1.08262668e-3      # orbital.h, WGS-84
J2_R_EQ = 6.378137e6         # orbital.h, EQUATORIAL radius (not R_EARTH)


def j2_secular_rates(a, e, inc):
    """(n, Omega_dot, omega_dot, M_dot) — mirrors orbital.h::propagate_orbit_j2."""
    n = np.sqrt(MU / a ** 3)
    p = a * (1.0 - e * e)
    k = 1.5 * n * J2_COEF * (J2_R_EQ / p) ** 2
    si2 = np.sin(inc) ** 2
    return (n,
            -k * np.cos(inc),
            0.5 * k * (4.0 - 5.0 * si2),
            n + 0.5 * k * np.sqrt(1.0 - e * e) * (2.0 - 3.0 * si2))


def propagate_cartesian_j2(X, dt):
    """Two-body f&g plus the secular J2 angle increments, in Cartesian.

    An element round trip rather than a perturbation of the f&g solution: the
    secular rates are DEFINED on mean elements, so the honest thing is to go
    there, advance the three angles, and come back. (a, e, inc) are invariants
    of secular J2, so only Omega, omega and M move — no iteration, no
    integrator. The equatorial branch mirrors the C env's own special case
    (varpi_dot = Omega_dot + omega_dot with raan pinned at 0).
    """
    el = cartesian_to_elements_3d(X)
    a, e, inc = el['a'], el['e'], el['inc']
    # Bad-row discipline, identical to `propagate_cartesian_nd`'s: a DIVERGED
    # estimate is routinely hyperbolic (a <= 0), and the secular rates take
    # sqrt(MU/a^3), so an unguarded call returns NaN, poisons the covariance
    # chain and floods the log with RuntimeWarnings. Substitute a harmless
    # LEO-ish orbit in those rows so the arithmetic stays finite, then hand
    # them back masked-out for the divergence guard to reinitialise. The probe
    # never needed this because it seeded from truth and never diverged; the
    # training loop does, every episode, by construction.
    with np.errstate(all='ignore'):
        bad = ~np.isfinite(a) | (a <= 1.0) | ~np.isfinite(e) | (e >= 1.0)
        a_s = np.where(bad, 7.0e6, a)
        e_s = np.where(bad, 0.0, e)
        inc_s = np.where(bad, 0.0, inc)
        _n, Om, om, Md = j2_secular_rates(a_s, e_s, inc_s)
        eq = (np.sin(inc_s) == 0.0)
        omega = np.where(eq, el['omega'] + (om + Om) * dt,
                         el['omega'] + om * dt)
        raan = np.where(eq, el['raan'], el['raan'] + Om * dt)
        M = el['M'] + Md * dt
        th = nm.eccentric_to_true(nm.solve_kepler(M, e_s), e_s)
        Y = orbit_to_cartesian_3d(a_s, e_s, th, omega, inc_s, raan)
    ok = np.all(np.isfinite(Y), axis=1) & ~bad
    Y = np.where(ok[:, None], Y, X)
    return Y, ok


def stm_fd_j2(X, dt, h_pos=nm.H_POS, h_vel=nm.H_VEL):
    """Central-difference STM of `propagate_cartesian_j2`. 12 propagations.

    Correct by construction. The cheap analytic O(J2) alternative was measured
    and rejected: it removes only ~59% of the two-body STM's error and leaves
    NEES at 11.5.
    """
    n = X.shape[0]
    Y, ok = propagate_cartesian_j2(X, dt)
    F = np.empty((n, 6, 6))
    h = np.array([h_pos] * 3 + [h_vel] * 3)
    for j in range(6):
        Xp = X.copy(); Xp[:, j] += h[j]
        Xm = X.copy(); Xm[:, j] -= h[j]
        Yp, okp = propagate_cartesian_j2(Xp, dt)
        Ym, okm = propagate_cartesian_j2(Xm, dt)
        F[:, :, j] = (Yp - Ym) / (2.0 * h[j])
        ok = ok & okp & okm
    return F, ok, Y


class BatchedBearingMSC6J2(BatchedBearingMSC6):
    """MSC6 with secular J2 in BOTH the state and the covariance propagation.

    `MSC6J2Cov(stm_j2='fd')` of the probe, in the production stack. Only
    `predict` differs from the parent, and only in which (Phi, x_new) pair it
    uses — everything else, including the chart, the re-pole and the update, is
    the shipped path unchanged.
    """

    def __init__(self, *a, stm_j2='fd', **kw):
        super().__init__(*a, **kw)
        if stm_j2 not in ('fd', 'analytic'):
            raise ValueError(f"stm_j2 must be 'fd' or 'analytic', got {stm_j2!r}")
        if stm_j2 == 'analytic':
            raise NotImplementedError(
                "the analytic O(J2) STM was measured and REJECTED (NEES 11.46, "
                "2.70x position at 24 h — it removes only ~59% of the two-body "
                "STM's error). Only stm_j2='fd' is shipped.")
        self.stm_j2 = stm_j2

    def predict(self, idx, dt, sat_from, sat_to, **_):
        if idx.size == 0:
            return np.zeros(0, dtype=bool)
        Rp = self.Rp[idx]
        y = self.y[idx]
        x_old = msc6_decode(y, sat_from, Rp)
        Phi, ok, x_new = stm_fd_j2(x_old, dt)
        y_new = msc6_encode(sat_to, x_new, Rp)
        Jold = msc6_decode_jac(y, Rp)
        Jnew = msc6_decode_jac(y_new, Rp)
        with np.errstate(all='ignore'):
            try:
                Gnew = np.linalg.inv(Jnew)
            except np.linalg.LinAlgError:            # pragma: no cover
                Gnew = np.linalg.pinv(Jnew)
            A = Phi @ Jold
            M = A @ self.Py[idx] @ np.swapaxes(A, 1, 2) + self._Q(dt)
            Py = Gnew @ M @ np.swapaxes(Gnew, 1, 2)
        self.y[idx] = y_new
        self.Py[idx] = 0.5 * (Py + np.swapaxes(Py, 1, 2))
        self.sat[idx] = sat_to
        return (ok & np.all(np.isfinite(y_new), axis=1)
                & np.all(np.isfinite(Py), axis=(1, 2)))
