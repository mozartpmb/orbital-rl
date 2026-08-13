"""Real angles-only batch IOD in 3D: 6 unknowns, 2 angles per observation.

The EVAL-side acquisition. Training replays a calibrated surrogate
(`nav_surrogate.AcquisitionSurrogate`) because the batch solver is 17-90x an
entire training step; eval runs this, the real thing, exactly as the 2D lineage
runs `ext_bo_filter.bls_acquire_adaptive`.

Structure is the 2D solver's, lifted per N3D-A section 3:

  * **The analytic range prior lifts verbatim.** `|r_c + rho u|^2 = R^2` is the
    same scalar quadratic in 3D; the annulus becomes a spherical shell and the
    LOS stays a ray. N3D-A verified interval-count agreement 200/200 against
    brute-force ray scans, worst relative support error 2.4e-4.
  * **The lattice does not grow.** Out-of-plane POSITION comes free and exactly
    from the measured elevation, because the seed sits on the LOS ray; the
    out-of-plane VELOCITY is seeded at ZERO. N3D-A measured the 27-node lattice
    producing bit-identical results to the 9-node one at all 8 geometries x 12
    seeds — true `v_oop = v_c sin(di)` is 0-134 m/s across the whole envelope,
    against a lattice step of `0.02 v_c` = 153 m/s, i.e. smaller than one step
    everywhere. So the 3D acquisition is the 2D acquisition plus two
    locally-convergent degrees of freedom, at 1.0x the node count.
  * **Do NOT build a plane-first stage.** Measured dead (N3D-A section 3b): a
    candidate plane normal gives the whole range history in closed form, but
    its conditioning is d(rho)/rho ~ 2931 per radian of tilt at the 5 km box,
    so 1% range needs 3.4e-6 rad of plane while the plane's own bearings-only
    accuracy there is 1.5e-5 rad — 10-20x too coarse.

Two things are genuinely new and both come from n3d_REDTEAM:

  * **The whitened Jacobian is the basis-free one.** With
    `R = diag(sigma_b^2/cos^2 el, sigma_b^2)` (MAJOR-12) the 1/cos(el) in the
    azimuth row cancels against the cos(el) in the azimuth weight, leaving
    `A = [e_az; e_el]^T Rp Phi_pos / (rho sigma_b)` — which is exactly
    `P_perp/(sigma_b rho)^2` in disguise. Writing az/el with an isotropic R
    instead inflates the information by 1/cos^2 el: 2.0x at 45 deg, 4.0x at 60.
  * **The acceptance window is SIM TIME** (BLOCKER-2). `w0` is derived from a
    45-minute floor and the cadence, never written as an observation count.

The red-team MAJOR-2 caveat of the 2D lineage carries over verbatim and must
travel with any number this produces: the chi-square, ambiguity-margin and
covariance gates test the self-consistency of the chosen range basin, NOT the
correctness of the basin choice.
"""

import math
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(_HERE, '..', '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'pufferlib'))

from pufferlib.ocean.orbital_nav import nav_math as nm            # noqa: E402
from pufferlib.ocean.orbital_nav import nav_math3d as n3          # noqa: E402

MU = n3.MU
R_EARTH = n3.R_EARTH
R_LO, R_HI = 0.9 * R_EARTH, 60.0 * R_EARTH     # physically admissible radii


# ── the analytic range prior, in 3D ──────────────────────────────────────────
def range_prior_intervals3(r_c, u, r_min, r_max, rho_floor=100.0):
    """Feasible ranges along the LOS ray, as a list of (lo, hi) intervals.

    p(rho) = r_c + rho u, |p|^2 = R^2 is a quadratic in rho for each shell
    radius. The feasible set is (0, rho_out] minus the chord inside r_min, and
    is BIMODAL whenever the ray pierces the inner sphere — both branches are
    kept here (unlike the training wrapper, which only needs the outer
    envelope to place a blind seed).
    """
    rc2 = float(r_c @ r_c)
    b0 = float(r_c @ u)
    disc_out = b0 * b0 - rc2 + r_max * r_max
    if disc_out <= 0.0:
        return []
    rho_out = -b0 + math.sqrt(disc_out)
    if rho_out <= rho_floor:
        return []
    disc_in = b0 * b0 - rc2 + r_min * r_min
    if disc_in <= 0.0:
        return [(rho_floor, rho_out)]
    sq = math.sqrt(disc_in)
    lo_in, hi_in = -b0 - sq, -b0 + sq
    out = []
    if lo_in > rho_floor:
        out.append((rho_floor, min(lo_in, rho_out)))
    if hi_in < rho_out:
        out.append((max(hi_in, rho_floor), rho_out))
    return [(a, b) for a, b in out if b > a * 1.000001]


def prior_span(intervals):
    if not intervals:
        return 100.0, 1.0e7
    return min(a for a, _ in intervals), max(b for _, b in intervals)


# ── measurement model ────────────────────────────────────────────────────────
def azel_of(Rp, r_sat, r_tgt):
    """(az, el) of the LOS in the epoch-frozen pole frame. Batched over rows."""
    d = np.einsum('ij,...j->...i', Rp, np.atleast_2d(r_tgt) - np.atleast_2d(r_sat))
    rho = np.maximum(np.linalg.norm(d, axis=-1), 1e-6)
    u = d / rho[..., None]
    el = np.arcsin(np.clip(u[..., 2], -1.0, 1.0))
    az = np.arctan2(u[..., 1], u[..., 0])
    return az, el, rho


def _basis(az, el):
    ca, sa = np.cos(az), np.sin(az)
    ce, se = np.cos(el), np.sin(el)
    e_az = np.stack([-sa, ca, np.zeros_like(sa)], axis=-1)
    e_el = np.stack([-se * ca, -se * sa, ce], axis=-1)
    return e_az, e_el


def los_unit(az, el):
    """Measured LOS unit vector in the pole frame."""
    ce = np.cos(el)
    return np.stack([ce * np.cos(az), ce * np.sin(az), np.sin(el)], axis=-1)


# ── the residual: CHART-FREE, not az/el ──────────────────────────────────────
#
# The filter has to carry az/el — the Aidala-Hammel property that makes the
# update linear is exactly "the measurement IS two state components" — and
# BLOCKER-1's re-pole is what keeps that chart usable. The batch solver has no
# such constraint, and it must not inherit the chart's pole, because the pole is
# reachable: BLOCKER-1 measures |el| = 87.58 deg on the campaign's own X3
# trajectory, where cos(el) = 0.042 and an azimuth difference is meaningless.
#
# Measured, on a di = 1 deg / 77 km geometry: with az/el residuals the solver
# converged to a local minimum at cost 1413 against 96.5 at truth, at EVERY
# window, and reported a 77 km epoch error — i.e. it never found the basin. The
# cause is not the Jacobian (the whitened az/el Jacobian is already the
# basis-free one) but the RESIDUAL: near the pole a large `wrap_pi(daz)` times a
# small cos(el) is a discontinuous, multi-modal cost surface.
#
# The chart-free residual is the GEODESIC angular error, expressed in the
# transverse basis at the predicted LOS:
#
#     theta = atan2(|P_perp u_obs|, u_obs . u_hat)      full angle, 0..pi
#     r     = theta * (u_obs.e_az, u_obs.e_el)/|P_perp u_obs| / sigma_beta
#
# no pole, no wrap, and identical to the az/el residual to first order.
#
# The `theta/|P_perp u_obs|` factor is NOT cosmetic — it is what makes
# Gauss-Newton work. The bare transverse projection saturates at 1, so the cost
# surface is FLAT far from the solution and the gradient carries no
# information. Measured on the di = 1 deg / 77 km geometry: from the circular
# grid node at the TRUE range (62 m of position error, 102 m/s of velocity
# error) the saturating residual made GN stall completely — 20 iterations, cost
# unchanged at 2.56e6 against 96.6 at truth. With the angle restored GN
# descends from the same node.
def _sphere_residual(u_obs, u_hat, e_az, e_el):
    ta = np.einsum('...i,...i->...', u_obs, e_az)
    te = np.einsum('...i,...i->...', u_obs, e_el)
    ca = np.clip(np.einsum('...i,...i->...', u_obs, u_hat), -1.0, 1.0)
    n = np.hypot(ta, te)
    th = np.arctan2(n, ca)
    sc = np.where(n > 1e-15, th / np.where(n > 1e-15, n, 1.0), 1.0)
    return ta * sc, te * sc


# ── grid cost, vectorised over CANDIDATES ────────────────────────────────────
def bearing_cost3(times, sats, az_obs, el_obs, X0, sigma_beta, Rp):
    """Sum of squared whitened (az, el) residuals for K candidate epoch states.

    Propagation only — no STM — so it is ~9x cheaper than a Gauss-Newton
    iteration and can be evaluated on a dense range grid. Every candidate is
    propagated in ONE batched call per observation, which is what makes a 3D
    grid affordable at all. +inf for any candidate whose trajectory leaves the
    admissible radius band; that is what keeps the grid from scoring garbage.
    """
    X = np.array(X0, dtype=np.float64, copy=True)
    K = X.shape[0]
    cost = np.zeros(K)
    dead = np.zeros(K, dtype=bool)
    for k in range(len(times)):
        if k > 0:
            X, ok = n3.propagate_cartesian_nd(X, float(times[k] - times[k - 1]))
            dead |= ~ok
        rr = np.linalg.norm(X[:, :3], axis=1)
        dead |= ~np.isfinite(X).all(axis=1) | (rr < R_LO) | (rr > R_HI)
        az, el, rho = azel_of(Rp, sats[k][None, :3], X[:, :3])
        dead |= rho < 1.0
        e_az, e_el = _basis(az, el)
        u_hat = los_unit(az, el)
        u_obs = los_unit(np.float64(az_obs[k]), np.float64(el_obs[k]))
        r_a, r_e = _sphere_residual(u_obs, u_hat, e_az, e_el)
        cost += (r_a ** 2 + r_e ** 2) / sigma_beta ** 2
    return np.where(dead, np.inf, cost)


def _bls_pass3(times, sats, az_obs, el_obs, x0, sigma_beta, Rp, iters,
               lam0=1e-2):
    """Levenberg-damped Gauss-Newton on the 6-state target epoch state.

    In 3D the target has 6 dof and each observation supplies 2 scalars, so 3
    well-separated observations are the algebraic minimum — Gauss's classical
    three-observation problem, which is what the 2D solver's 4-bearing minimum
    was the planar analogue of. In practice the arc must also be long enough
    for the curvature signature that breaks the scale symmetry.

    N3D-B section 3.4: the in-plane angle ALONE can never see the two
    out-of-plane dof, so a one-angle 6-state FIM is structurally singular. The
    second angle is not optional in 3D; it is what makes the state estimable.
    """
    x = np.array(x0, dtype=np.float64)
    lam = lam0
    best = (x.copy(), float('inf'), np.eye(6))
    for _ in range(iters):
        Phi = np.eye(6)
        xt = x.copy()
        rows, res = [], []
        ok = True
        for k in range(len(times)):
            if k > 0:
                dt = float(times[k] - times[k - 1])
                F, okf, Y = n3.stm_analytic_nd(xt[None], dt)
                Phi = F[0] @ Phi
                xt = Y[0]
                if not okf[0]:
                    ok = False
                    break
            rr = float(np.linalg.norm(xt[:3])) if np.all(np.isfinite(xt)) else 0.0
            az, el, rho = azel_of(Rp, sats[k][None, :3], xt[None, :3])
            az, el, rho = float(az[0]), float(el[0]), float(rho[0])
            if not np.isfinite(rho) or rho < 1.0 or rr < R_LO or rr > R_HI:
                ok = False
                break
            e_az, e_el = _basis(np.array(az), np.array(el))
            # WHITENED rows. The 1/cos(el) of the azimuth Jacobian cancels the
            # cos(el) of the azimuth weight under the anisotropic R, leaving the
            # basis-free form (MAJOR-12) — which is also exactly conjugate to
            # the chart-free residual below.
            g_az = np.zeros(6); g_az[:3] = (e_az @ Rp) / (rho * sigma_beta)
            g_el = np.zeros(6); g_el[:3] = (e_el @ Rp) / (rho * sigma_beta)
            rows.append(g_az @ Phi)
            rows.append(g_el @ Phi)
            u_hat = los_unit(np.array(az), np.array(el))
            u_obs = los_unit(np.float64(az_obs[k]), np.float64(el_obs[k]))
            r_a, r_e = _sphere_residual(u_obs, u_hat, e_az, e_el)
            res.append(float(r_a) / sigma_beta)
            res.append(float(r_e) / sigma_beta)
        if not ok:
            lam *= 100.0
            x = 0.5 * (x + best[0])
            if lam > 1e12:
                break
            continue
        A = np.array(rows)
        b = np.array(res)
        cost = float(b @ b)
        N = A.T @ A
        if cost < best[1]:
            best = (x.copy(), cost, N)
            lam = max(lam * 0.3, 1e-8)
        else:
            lam = min(lam * 10.0, 1e12)
            x = best[0].copy()
            continue
        try:
            step = np.linalg.solve(
                N + lam * np.diag(np.diag(N)) + 1e-9 * np.eye(6), A.T @ b)
        except np.linalg.LinAlgError:
            break
        # Trust region on the RANGE direction — the one bad direction — and on
        # velocity, so one step can never throw an iterate hyperbolic.
        rho_now = float(np.linalg.norm(x[:3] - sats[0][:3]))
        sp = float(np.linalg.norm(step[:3]))
        if sp > 0.5 * rho_now:
            step = step * (0.5 * rho_now / sp)
        vc = math.sqrt(MU / max(float(np.linalg.norm(x[:3])), R_LO))
        sv = float(np.linalg.norm(step[3:]))
        if sv > 0.5 * vc:
            step = step.copy()
            step[3:] *= 0.5 * vc / sv
        x = x + step
    return best


def ray_init3(r_c, v_c_vec, u, rho, ft=1.0, fr=0.0, fn=0.0):
    """Candidate epoch state at range `rho` along the LOS.

    Out-of-plane POSITION is exact — the seed sits on the ray, so the elevation
    measurement supplies it for free, exactly as N3D-A says.

    Out-of-plane VELOCITY is NOT free, and this is where N3D-A section 3c is
    wrong for the campaign's own headline rung. It concluded the 27-node lattice
    is bit-identical to the 9-node one and therefore that out-of-plane velocity
    can be seeded at zero. That was measured only on cells its prototype kept —
    and its section 6 caveat 5 says it enforced `di <= asin(rho/r)` and SKIPPED
    the rest. At the X3 rung (di = 1 deg, rho_inplane 5 km) that constraint
    gives di <= 0.042 deg, so the headline cell is one of the skipped ones.

    Measured here on exactly that cell (true separation 77 km, one full orbit
    of 60 s bearings): the zero-out-of-plane-velocity node at the TRUE range is
    62 m from truth in position but 102 m/s out in velocity, and Gauss-Newton
    from it lands 126 km away — outside the basin. `v_oop = v_c sin(di)` is
    134 m/s at di = 1 deg, and the lattice must bracket it. `fn` is that third
    axis, in units of v_c; +/-0.02 v_c = +/-153 m/s at LEO covers the whole
    0-134 m/s span, which is N3D-A's own quoted lattice step.
    """
    p = r_c + rho * u
    rr = max(float(np.linalg.norm(p)), R_LO)
    vc = math.sqrt(MU / rr)
    h = np.cross(r_c, v_c_vec)
    h = h / max(float(np.linalg.norm(h)), 1e-300)
    ur = p / rr
    ut = np.cross(h, ur)
    ut = ut / max(float(np.linalg.norm(ut)), 1e-300)
    return np.concatenate([p, ft * vc * ut + fr * vc * ur + fn * vc * h])


def bls_acquire3(times, sats, az_obs, el_obs, intervals, sigma_beta, Rp,
                 window=90, grid_ratio=1.15, grid_max=160, iters=12,
                 n_bins=8, cov_inflate=4.0,
                 v_tang=(0.80, 1.00, 1.20), v_rad=(-0.30, 0.0, 0.30),
                 v_norm=(-0.02, 0.0, 0.02), extra_starts=()):
    """Dense range grid + binned multi-start Gauss-Newton. 3D twin of
    `ext_bo_filter.bls_acquire`, node for node."""
    w = min(window, len(times))
    t_w = times[:w]
    s_w = sats[:w]
    a_w, e_w = az_obs[:w], el_obs[:w]

    r_c0 = s_w[0][:3]
    v_c0 = s_w[0][3:6]
    ca, sa = math.cos(a_w[0]), math.sin(a_w[0])
    ce, se = math.cos(e_w[0]), math.sin(e_w[0])
    u0 = Rp.T @ np.array([ce * ca, ce * sa, se])

    spans = [(lo, hi, math.log(hi / lo)) for lo, hi in intervals if hi > lo]
    if not spans:
        return None
    cand, rhos = [], []
    for lo, hi, sp in spans:
        k = max(3, min(grid_max, int(math.ceil(sp / math.log(grid_ratio)))))
        for i in range(k):
            rho = lo * (hi / lo) ** ((i + 0.5) / k)
            for ft in v_tang:
                for fr in v_rad:
                    for fn in v_norm:
                        cand.append(ray_init3(r_c0, v_c0, u0, rho, ft, fr, fn))
                        rhos.append(rho)
    if not cand:
        return None
    X0 = np.array(cand)
    costs = bearing_cost3(t_w, s_w, a_w, e_w, X0, sigma_beta, Rp)
    rhos = np.array(rhos)

    # Binned multi-start: split the prior into equal LOG-range bins and refine
    # the cheapest node in each. Taking the globally cheapest nodes instead
    # systematically misses the true basin — a wrong-range node with a lucky
    # velocity out-scores the true range with a circular velocity guess, and a
    # "diverse top-k" then never visits the true decade at all.
    fin = np.isfinite(costs)
    if not fin.any():
        return None
    lo_all, hi_all = rhos.min(), rhos.max()
    span_all = math.log(max(hi_all / max(lo_all, 1e-9), 1.0 + 1e-9))
    nb = max(2, int(n_bins))
    b_idx = np.minimum(nb - 1,
                       (nb * np.log(rhos / lo_all) / span_all).astype(int))
    starts = []
    for b in range(nb):
        m = fin & (b_idx == b)
        if m.any():
            starts.append(X0[np.flatnonzero(m)[int(np.argmin(costs[m]))]])
    starts += [np.asarray(s, dtype=float) for s in extra_starts]
    if not starts:
        return None

    coarse = [_bls_pass3(t_w, s_w, a_w, e_w, x0, sigma_beta, Rp, 3)
              for x0 in starts]
    coarse.sort(key=lambda r: r[1])
    sols = [_bls_pass3(t_w, s_w, a_w, e_w, r[0], sigma_beta, Rp, iters)
            for r in coarse[:3] if np.isfinite(r[1])]
    sols += [r for r in coarse if np.isfinite(r[1])]
    if not sols:
        return None
    sols.sort(key=lambda r: r[1])
    x, c, N = sols[0]

    # Ambiguity margin: the cost of the best solution at a MATERIALLY different
    # range. Short angles-only arcs admit several well-separated orbits that fit
    # comparably, and a covariance test cannot see that — each basin is locally
    # tight — so the solver must report the likelihood-ratio gap.
    rho_1 = float(np.linalg.norm(x[:3] - s_w[0][:3]))
    c2 = float('inf')
    for xs, cs, _ in sols[1:]:
        rs = float(np.linalg.norm(xs[:3] - s_w[0][:3]))
        if max(rs / max(rho_1, 1.0), rho_1 / max(rs, 1.0)) > 1.2:
            c2 = min(c2, cs)
    try:
        P = cov_inflate * np.linalg.inv(N + 1e-9 * np.eye(6))
    except np.linalg.LinAlgError:
        return None
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(P)):
        return None
    return x, 0.5 * (P + P.T), c, w, c2 - c


def bls_acquire_adaptive3(times, sats, az_obs, el_obs, intervals, sigma_beta,
                          Rp, w0=45, growth=1.6, gate=0.20, amb_margin=16.0,
                          **kw):
    """Grow the batch arc until the range is actually determined.

    Gates, all three necessary and none sufficient alone:
      chi-square  cost <= dof + 3 sqrt(2 dof), dof = 2w - 6. Two rows per
                  observation and six unknowns in 3D, against 1 and 4 in 2D.
      ambiguity   likelihood-ratio gap to the best materially-different range.
      covariance  1-sigma LOS range uncertainty <= gate * range.

    `w0` is passed in DERIVED from the 45-minute sim-time floor and the
    cadence (BLOCKER-2), never hard-coded as an observation count.
    """
    n = len(times)
    w = min(int(w0), n)
    prev = None
    last = None
    while True:
        acq = bls_acquire3(times, sats, az_obs, el_obs, intervals, sigma_beta,
                           Rp, window=w,
                           extra_starts=() if prev is None else (prev,), **kw)
        if acq is not None:
            x, P, c, ww, margin = acq
            last = acq[:4]
            prev = x
            dof = max(1, 2 * ww - 6)
            chi_ok = c <= dof + 3.0 * math.sqrt(2.0 * dof)
            rho_v = x[:3] - sats[0][:3]
            rho = max(float(np.linalg.norm(rho_v)), 1.0)
            u = rho_v / rho
            sig = math.sqrt(max(float(u @ P[:3, :3] @ u), 0.0))
            if chi_ok and margin >= amb_margin and sig / rho <= gate:
                return acq[:4] + (True,)
        if w >= n:
            return (last + (False,)) if last is not None else None
        w = min(n, max(w + 1, int(round(w * growth))))
