"""EXT-ANGLES — is the RANGE actually estimable from bearings alone in our regime?

`ext_angles_scale_ambiguity.py` measures the literal object the classical theorem
is about: the bearing history of the naively scaled relative state
x_c0 + k*(x_t0 - x_c0), which under LINEAR relative dynamics is bit-identical to
the truth for every k.  That measurement answers "how badly is the classical
symmetry broken", and the answer is "a lot".

But it OVERSTATES what an estimator faces.  A real estimator handed a 2x range
hypothesis is free to re-fit the other three degrees of freedom (initial bearing
and the two velocity components) to soak up most of the mismatch.  The honest
question is the PROFILE LIKELIHOOD:

    D(k) = sqrt( min over the remaining 3 dof of  sum_i (beta_i(x) - beta_i^true)^2 / sigma^2 )
           subject to |p(t0) - r_c(t0)| = k * rho_0

D(k) is the generalized-likelihood-ratio statistic for the hypothesis "the range
is k times the truth".  D(k) >= 3 means a bearings-only batch estimator rejects
that range at 3 sigma.  The width of the D <= 1 well is the 1-sigma range
uncertainty, and it must agree with the Cramer-Rao bound when the well is
quadratic -- which makes this an independent cross-check on NAV-G's CRLB map
(`ext_bo_observability.py`), computed by a completely different route.

Also here, because it settles a disagreement:

  * NAV-G reports an EXACTLY SINGULAR Fisher information for range at rho <= 10 km
    with no burn.  That pipeline chains a numerically differenced STM across the
    arc, so its conditioning floor is ~1e-16 relative on a matrix whose entries
    span 10 orders of magnitude.  Here the FIM is built by differencing the
    WHOLE-ARC bearing map directly (8 full-arc propagations, no chaining), which
    is far better conditioned, and the numerical floor is measured explicitly
    rather than assumed.

Everything runs on a vectorised f&g propagator validated against the oracle
`orbital_math.propagate_cartesian` at the top of the run.

Run:  python3 ext_angles_observability_profile.py [--quick]
Writes web_data/results/ext_angles_crlb.csv
       web_data/results/ext_angles_profile.csv
       web_data/results/ext_angles_dv_sizing.csv
"""

import argparse
import csv
import math
import os
import sys

import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, "/Users/pete/space_training/scripts/orbital/nav")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import orbital_math as om                                        # noqa: E402
from orbital_math import MU, R_EARTH                             # noqa: E402
import ext_angles_scale_ambiguity as A                           # noqa: E402

RESULTS = "/Users/pete/space_training/web_data/results"
OUT_CRLB = f"{RESULTS}/ext_angles_crlb.csv"
OUT_PROF = f"{RESULTS}/ext_angles_profile.csv"
OUT_DV = f"{RESULTS}/ext_angles_dv_sizing.csv"

SIGMA_BETA = 1.0e-3
DT = 60.0


# ── vectorised exact two-body propagation ────────────────────────────────────
def prop_vec(x0, dts, iters=80, tol=1e-13):
    """Lagrange f&g propagation of ONE state to an ARRAY of times.

    Identical algorithm to orbital_math.propagate_cartesian (Newton on the
    eccentric-anomaly difference), vectorised over dt. Returns (N, 4).
    """
    x, y, vx, vy = float(x0[0]), float(x0[1]), float(x0[2]), float(x0[3])
    r0 = math.hypot(x, y)
    v2 = vx * vx + vy * vy
    a = 1.0 / (2.0 / r0 - v2 / MU)
    if not (a > 0.0) or r0 < 1.0 or not np.isfinite(a):
        return None                       # hyperbolic / degenerate hypothesis
    sqmu = math.sqrt(MU)
    sqa = math.sqrt(a)
    sigma0 = (x * vx + y * vy) / sqmu
    n = sqmu / (a * sqa)

    dts = np.asarray(dts, dtype=float)
    tgt = sqmu * dts
    dE = n * dts
    for _ in range(iters):
        s, c = np.sin(dE), np.cos(dE)
        F = (a * sqa * (dE - s) + sigma0 * a * (1.0 - c) + r0 * sqa * s) - tgt
        dF = a * sqa * (1.0 - c) + sigma0 * a * s + r0 * sqa * c
        step = F / dF
        dE = dE - step
        if np.max(np.abs(step)) < tol:
            break
    s, c = np.sin(dE), np.cos(dE)
    f = 1.0 - (a / r0) * (1.0 - c)
    g = dts - (a * sqa / sqmu) * (dE - s)
    px = f * x + g * vx
    py = f * y + g * vy
    r = np.hypot(px, py)
    fd = -(sqmu * sqa / (r * r0)) * s
    gd = 1.0 - (a / r) * (1.0 - c)
    return np.stack([px, py, fd * x + gd * vx, fd * y + gd * vy], axis=1)


def validate_prop(n=400, seed=7):
    """prop_vec vs the oracle scalar propagator, and the bearing numerical floor."""
    rng = np.random.default_rng(seed)
    worst = 0.0
    for _ in range(n):
        a = R_EARTH + rng.uniform(300e3, 20000e3)
        e = rng.uniform(0.0, 0.5)
        el = A.X.elements(a, e, rng.uniform(0, 6.28), rng.uniform(0, 6.28))
        x0 = om.orbit_to_cartesian(el)
        dts = np.array([60.0, 3600.0, 86400.0, 180000.0])
        pv = prop_vec(x0, dts)
        for i, dt in enumerate(dts):
            ref = np.array(om.propagate_cartesian(x0, float(dt)))
            worst = max(worst, float(np.max(np.abs(pv[i] - ref)) /
                                     max(np.linalg.norm(ref[:2]), 1.0)))
    return worst


def bearing_floor(geo, dur_s):
    """Measured numerical floor of the bearing history, in mrad.

    Same truth state propagated two ways (single-shot dt vs two chained half
    steps). Any measured d_beta below this is numerics, not physics.
    """
    times = np.arange(0.0, dur_s + 0.5 * DT, DT)
    sat = prop_vec(geo['sat_cart'], times)
    tgt = prop_vec(geo['tgt_cart'], times)
    half = prop_vec(geo['tgt_cart'], times / 2.0)
    tgt2 = np.array([prop_vec(half[i], np.array([times[i] / 2.0]))[0]
                     for i in range(len(times))])
    b1 = np.arctan2(tgt[:, 1] - sat[:, 1], tgt[:, 0] - sat[:, 0])
    b2 = np.arctan2(tgt2[:, 1] - sat[:, 1], tgt2[:, 0] - sat[:, 0])
    return float(np.max(np.abs(np.unwrap(b1) - np.unwrap(b2)))) * 1e3


# ── chaser trajectory with an optional burn ──────────────────────────────────
def chaser_path(x0, times, burn):
    if burn is None:
        return prop_vec(x0, times)
    tb, dv, dirn = burn
    pre = times[times <= tb]
    post = times[times > tb]
    out = np.zeros((len(times), 4))
    if len(pre):
        out[:len(pre)] = prop_vec(x0, pre)
    xb = prop_vec(x0, np.array([tb]))[0]
    ux, uy = (xb[0], xb[1]) if dirn == 'radial' else (xb[2], xb[3])
    nrm = math.hypot(ux, uy)
    xbp = (xb[0], xb[1], xb[2] + dv * ux / nrm, xb[3] + dv * uy / nrm)
    if len(post):
        out[len(pre):] = prop_vec(xbp, post - tb)
    return out


def bearings_of(tgt_state, sat_path, times):
    tp = prop_vec(tgt_state, times)
    if tp is None:
        return None
    return np.arctan2(tp[:, 1] - sat_path[:, 1], tp[:, 0] - sat_path[:, 0])


def wrap(v):
    return (v + np.pi) % (2.0 * np.pi) - np.pi


# ── Fisher information / CRLB by direct full-arc differencing ────────────────
def crlb(geo, times, sat_path, sigma_beta=SIGMA_BETA, with_range=False,
         sigma_rho=50.0):
    """sigma of the range along the initial LOS, from the exact arc information.

    H rows are d(measurement)/d(target initial state), obtained by central
    differences of the WHOLE-ARC map -- no STM chaining, so no accumulated
    conditioning loss. Returns (sigma_los_m, cond, sigma_los_over_rho).
    """
    x0 = np.array(geo['tgt_cart'], dtype=float)
    # scale the state so the FIM is dimensionless-ish before inversion
    L = math.hypot(x0[0], x0[1])
    V = math.sqrt(MU / L)
    S = np.diag([L, L, V, V])
    h = np.array([1.0, 1.0, 1e-3, 1e-3])       # m, m/s

    b0 = bearings_of(x0, sat_path, times)
    Jb = np.zeros((len(times), 4))
    Jr = np.zeros((len(times), 4))
    if with_range:
        tp0 = prop_vec(x0, times)
        r0 = np.hypot(tp0[:, 0] - sat_path[:, 0], tp0[:, 1] - sat_path[:, 1])
    for j in range(4):
        xp = x0.copy(); xp[j] += h[j]
        xm = x0.copy(); xm[j] -= h[j]
        bp, bm = bearings_of(xp, sat_path, times), bearings_of(xm, sat_path, times)
        Jb[:, j] = wrap(bp - bm) / (2.0 * h[j])
        if with_range:
            pp, pm = prop_vec(xp, times), prop_vec(xm, times)
            rp = np.hypot(pp[:, 0] - sat_path[:, 0], pp[:, 1] - sat_path[:, 1])
            rm = np.hypot(pm[:, 0] - sat_path[:, 0], pm[:, 1] - sat_path[:, 1])
            Jr[:, j] = (rp - rm) / (2.0 * h[j])

    Jb_s = Jb @ S
    F = (Jb_s.T @ Jb_s) / sigma_beta ** 2
    if with_range:
        Jr_s = Jr @ S
        F = F + (Jr_s.T @ Jr_s) / sigma_rho ** 2
    cond = float(np.linalg.cond(F))
    try:
        C = S @ np.linalg.inv(F) @ S.T
    except np.linalg.LinAlgError:
        return float('inf'), cond, float('inf')
    d = np.array([geo['tgt_cart'][0] - sat_path[0, 0],
                  geo['tgt_cart'][1] - sat_path[0, 1]])
    rho0 = float(np.linalg.norm(d))
    u = d / rho0
    var = float(u @ C[:2, :2] @ u)
    sig = math.sqrt(max(var, 0.0)) if np.isfinite(var) else float('inf')
    return sig, cond, sig / rho0


# ── profile likelihood over the range scale ──────────────────────────────────
def profile_D(geo, times, sat_path, beta_true, k, sigma_beta=SIGMA_BETA):
    """min over (initial bearing, velocity) of the bearing misfit at range k*rho0."""
    rc0 = sat_path[0, :2]
    d0 = np.array(geo['tgt_cart'][:2]) - rc0
    rho0 = float(np.linalg.norm(d0))
    alpha0 = math.atan2(d0[1], d0[0])
    vt = np.array(geo['tgt_cart'][2:])
    vc = sat_path[0, 2:]
    v_scaled = vc + k * (vt - vc)              # the naive scaled hypothesis

    rk = k * rho0

    def resid(u):
        al, vx, vy = u
        p = rc0 + rk * np.array([math.cos(al), math.sin(al)])
        b = bearings_of(np.array([p[0], p[1], vx, vy]), sat_path, times)
        if b is None:
            return np.full(len(times), 1e3)
        return wrap(b - beta_true) / sigma_beta

    starts = [np.array([alpha0, v_scaled[0], v_scaled[1]])]
    p_c = rc0 + rk * np.array([math.cos(alpha0), math.sin(alpha0)])
    rr = float(np.linalg.norm(p_c))
    if rr > 1.0:
        v_circ = math.sqrt(MU / rr) * np.array([-p_c[1] / rr, p_c[0] / rr])
        starts.append(np.array([alpha0, v_circ[0], v_circ[1]]))

    best = float('inf')
    for s0 in starts:
        try:
            sol = least_squares(resid, s0, method='lm', xtol=1e-14,
                                ftol=1e-14, gtol=1e-14, max_nfev=400)
            best = min(best, float(np.linalg.norm(sol.fun)))
        except Exception:
            pass
    return best


def one_config(geo, dur_s, burn, ks, sigma_beta=SIGMA_BETA):
    times = np.arange(0.0, dur_s + 0.5 * DT, DT)
    sat_path = chaser_path(geo['sat_cart'], times, burn)
    beta_true = bearings_of(np.array(geo['tgt_cart']), sat_path, times)
    sig, cond, rel = crlb(geo, times, sat_path, sigma_beta)
    sig_rb, _, rel_rb = crlb(geo, times, sat_path, sigma_beta, with_range=True)
    out = dict(geom=geo['name'], rho0_km=geo['rho0'] / 1e3,
               rho_over_r=geo['rho_over_r'], arc_min=dur_s / 60.0,
               n_obs=len(times), sigma_beta_mrad=sigma_beta * 1e3,
               dv_ms=0.0 if burn is None else burn[1],
               burn_dir='none' if burn is None else burn[2],
               crlb_los_m=sig, crlb_rel=rel, fim_cond=cond,
               crlb_los_rb_m=sig_rb, crlb_rel_rb=rel_rb)
    for k in ks:
        out[f'D_k{k:g}'] = profile_D(geo, times, sat_path, beta_true, k,
                                     sigma_beta)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true')
    args = ap.parse_args()

    err = validate_prop(120 if args.quick else 400)
    print(f"[validate] prop_vec vs oracle propagate_cartesian: "
          f"max relative position error {err:.3e}")
    assert err < 1e-10, "vectorised propagator disagrees with the oracle"

    geos = A.make_geometries()
    for g in geos:
        fl = bearing_floor(g, 6.0 * 3600.0)
        g['beta_floor_mrad'] = fl
        print(f"[floor]  {g['name']:<22} rho0 {g['rho0']/1e3:9.1f} km  "
              f"bearing numerical floor {fl:.3e} mrad "
              f"({fl/1.0:.1e} of the 1 mrad sensor sigma)")

    ks = [0.5, 0.8, 1.25, 2.0]
    arcs_min = [30.0, 90.0, 360.0] if args.quick else [15.0, 30.0, 90.0, 360.0, 1440.0, 3000.0]
    burns = [None, (600.0, 0.1, 'radial'), (600.0, 1.0, 'radial'),
             (600.0, 1.0, 'prograde'), (600.0, 10.0, 'prograde')]

    rows = []
    hdr = (f"{'geom':<22} {'arc_min':>8} {'dv':>6} {'dir':<9} "
           f"{'crlb_los_m':>12} {'crlb/rho':>10} {'cond':>10} "
           f"{'D(0.5)':>9} {'D(0.8)':>9} {'D(1.25)':>9} {'D(2)':>9}")
    print("\n" + hdr)
    for g in geos:
        for arc in arcs_min:
            for b in burns:
                if b is not None and b[0] > 0.5 * arc * 60.0:
                    continue                       # burn must be inside the arc
                r = one_config(g, arc * 60.0, b, ks)
                r['beta_floor_mrad'] = g['beta_floor_mrad']
                rows.append(r)
                print(f"{r['geom']:<22} {r['arc_min']:8.0f} {r['dv_ms']:6.2f} "
                      f"{r['burn_dir']:<9} {r['crlb_los_m']:12.4g} "
                      f"{r['crlb_rel']:10.3g} {r['fim_cond']:10.3g} "
                      f"{r['D_k0.5']:9.4g} {r['D_k0.8']:9.4g} "
                      f"{r['D_k1.25']:9.4g} {r['D_k2']:9.4g}")
        print()

    with open(OUT_PROF, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {OUT_PROF} ({len(rows)} rows)")

    # ── Delta-v sizing: what burn does it take to pin range to 10% in 90 min ──
    print("\n== Delta-v sizing: CRLB(range)/rho within a 90 min arc ==")
    dvs = [0.0, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 25.0]
    srows = []
    print(f"{'geom':<22} {'dir':<9} " + " ".join(f"{d:>9.3g}" for d in dvs))
    for g in geos:
        for dirn in ('radial', 'prograde'):
            line, cells = [], []
            for dv in dvs:
                b = None if dv == 0.0 else (600.0, dv, dirn)
                r = one_config(g, 90.0 * 60.0, b, [2.0])
                srows.append(dict(geom=g['name'], burn_dir=dirn, dv_ms=dv,
                                  arc_min=90.0, rho0_km=g['rho0'] / 1e3,
                                  rho_over_r=g['rho_over_r'],
                                  crlb_los_m=r['crlb_los_m'],
                                  crlb_rel=r['crlb_rel'],
                                  fim_cond=r['fim_cond'],
                                  D_k2=r['D_k2']))
                cells.append(f"{r['crlb_rel']:9.3g}")
            print(f"{g['name']:<22} {dirn:<9} " + " ".join(cells))
    with open(OUT_DV, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(srows[0].keys()))
        w.writeheader()
        w.writerows(srows)
    print(f"wrote {OUT_DV} ({len(srows)} rows)")

    # CRLB csv is a projection of the profile csv; emit for symmetry with NAV-G
    with open(OUT_CRLB, 'w', newline='') as fh:
        keys = ['geom', 'rho0_km', 'rho_over_r', 'arc_min', 'n_obs', 'dv_ms',
                'burn_dir', 'crlb_los_m', 'crlb_rel', 'fim_cond',
                'crlb_los_rb_m', 'crlb_rel_rb', 'beta_floor_mrad']
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {OUT_CRLB} ({len(rows)} rows)")


if __name__ == '__main__':
    main()
