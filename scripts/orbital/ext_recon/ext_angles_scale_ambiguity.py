"""EXT-ANGLES — how fast does the classical angles-only scale ambiguity break in OUR regime?

The classical result (Woffinden & Geller 2007/2009; Grzymisch & Fichter 2014; and
in the terrestrial bearings-only-TMA form, Nardone & Aidala 1981) is a statement
about LINEAR relative dynamics.  If the relative state obeys

    dx/dt = A(t) x        (CW / Hill, Yamanaka-Ankersen, or any homogeneous LTV)

then x(t) = Phi(t,t0) x0 is linear in x0, so k*x0 propagates to k*x(t) for any
scalar k > 0.  A bearing measurement beta = atan2(dy, dx) depends only on the
DIRECTION of the relative position, so it is invariant under x -> k*x.  The whole
ray {k*x0 : k > 0} therefore produces a bit-identical bearing history and range
is *structurally* unobservable.  A chaser maneuver enters the dynamics as an
inhomogeneous forcing term, x(t) = Phi(t,t0)x0 - Phi(t,tb)[0; dv]; the burn term
does NOT scale with k, so the ray is no longer a symmetry and range becomes
observable.  That is the entire content of the classical criterion.

Our regime is different in one specific, quantifiable way: the filter estimates
the TARGET'S ABSOLUTE state on a Keplerian orbit about a known center, and the
separations run from 5 km to 20 000 km, i.e. rho/r from 7e-4 to >2.  The relative
dynamics are not linear at all out there, and even close in the gravity gradient
is only linear to O(rho/r).  Under the exact two-body flow the scaled hypothesis
r_c + k*(r_t - r_c) sits at a different orbital radius, hence a different mean
motion (n ~ a^-3/2), hence a bearing history that separates from the truth.
Nonlinearity alone breaks the ambiguity.  The design question is not "observable
or not" but "how fast, compared to the 1 mrad noise floor and the episode clock".

This script measures exactly that, by forward simulation only -- no STM, no
finite differencing, no matrix conditioning -- so it is immune to the numerical
floor that makes a differenced Fisher matrix look singular at close range
(cf. ext_bo_observability.py, which reports a singular FIM at rho <= 10 km).

Three arms per geometry, same epochs, same chaser trajectory:
  TRUTH   exact two-body f&g propagation of the true target state
  SCALED  exact two-body f&g propagation of  x_c0 + k*(x_t0 - x_c0)
  LINEAR  relative state propagated by the STM about the chaser's reference
          trajectory, with the burn entered as the classical inhomogeneous term

  d_beta_nl(t)  = beta_TRUTH - beta_SCALED     total ambiguity-breaking signal
  d_beta_lin(t) = same, computed in the LINEAR arm
                  == 0 identically when there is no burn (validation of the
                     classical theorem inside our own code)
                  == the Woffinden-Geller maneuver signal when there is
  d_beta_nl - d_beta_lin = the part of observability that Keplerian
                  nonlinearity contributes and the classical theory misses.

Detection metrics against sigma_beta = 1 mrad at 60 s cadence:
  t_1sig / t_3sig  first epoch |d_beta| exceeds 1 / 3 sigma  (single-sample)
  t_D3             first epoch the accumulated Mahalanobis distance
                   D(t) = sqrt(sum_i (d_beta_i/sigma)^2) exceeds 3, i.e. when a
                   BATCH estimator can reject the scaled hypothesis at 3 sigma.
                   This is the operationally correct detection time: a bias far
                   below the per-sample noise floor is still rejected by a long
                   enough arc.

Run:  python3 ext_angles_scale_ambiguity.py [--quick]
Writes web_data/results/ext_angles_scale_ambiguity.csv
       web_data/results/ext_angles_dbeta_trace.csv
       web_data/results/ext_angles_burn_threshold.csv
"""

import argparse
import csv
import math
import os
import sys

import numpy as np

sys.path.insert(0, "/Users/pete/space_training/scripts/orbital/nav")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import orbital_math as om                                        # noqa: E402
from orbital_math import MU, R_EARTH, propagate_cartesian        # noqa: E402
import ext_bo_filter as X                                        # noqa: E402

RESULTS = "/Users/pete/space_training/web_data/results"
OUT_MAIN = f"{RESULTS}/ext_angles_scale_ambiguity.csv"
OUT_TRACE = f"{RESULTS}/ext_angles_dbeta_trace.csv"
OUT_BURN = f"{RESULTS}/ext_angles_burn_threshold.csv"

SIGMA_BETA = 1.0e-3          # rad, the shipped optical bearing sigma
DT = 60.0                    # s, nav cadence
EP_CAP_MIN = 3000.0          # the T3/T4 episode clock, minutes


def wrap_pi(x):
    return (x + math.pi) % (2.0 * math.pi) - math.pi


def bearing(sat, tgt):
    return math.atan2(tgt[1] - sat[1], tgt[0] - sat[0])


def rng_of(sat, tgt):
    return math.hypot(tgt[0] - sat[0], tgt[1] - sat[1])


def period(a):
    return 2.0 * math.pi * math.sqrt(a ** 3 / MU)


# ── geometries ───────────────────────────────────────────────────────────────
def make_geometries():
    """Representative of every shipped envelope, aligned with NAV-G's G1-G6.

    Named so the two reports compose.  Durations are set to the episode clock
    (3000 min = 50 h) rather than a couple of orbits, because the question here
    is what an ENTIRE EPISODE's bearing arc reveals, not what one pass does.
    """
    a_leo = R_EARTH + 400e3
    g = []

    # A1 -- tight terminal box, 5 km.  rho/r = 7.4e-4.  The hard corner: this is
    #       where the linearized dynamics are most nearly exact, so this is where
    #       the classical unobservability theorem most nearly applies.
    tgt = X.elements(a_leo, 0.001, 1.7, 0.0)
    sat = X.elements(a_leo + 400.0, 0.001, 1.7, -5e3 / a_leo)
    g.append(dict(name='A1_leo_5km_box', sat=sat, tgt=tgt, a_ref=a_leo))

    # A2 -- close co-orbital drift, 10 km (NAV-G G1).
    tgt = X.elements(a_leo, 0.001, 0.3, 0.0)
    sat = X.elements(a_leo + 2.0e3, 0.001, 0.3, -10e3 / a_leo)
    g.append(dict(name='A2_leo_10km_drift', sat=sat, tgt=tgt, a_ref=a_leo))

    # A3 -- 100 km, the scale at which conjunction screening and mid-field
    #       guidance live.
    tgt = X.elements(a_leo, 0.005, 0.9, 0.0)
    sat = X.elements(a_leo + 20e3, 0.005, 0.9, -100e3 / a_leo)
    g.append(dict(name='A3_leo_100km', sat=sat, tgt=tgt, a_ref=a_leo))

    # A4 -- mid-field drift-phasing leg, 300 km sep on a 60 km drift orbit
    #       (NAV-G G2).  This is where the T3 policy spends most of its clock.
    tgt = X.elements(a_leo, 0.01, 1.1, 0.0)
    sat = X.elements(a_leo + 60e3, 0.012, 1.1, -300e3 / a_leo)
    g.append(dict(name='A4_leo_300km_drift', sat=sat, tgt=tgt, a_ref=a_leo))

    # A5 -- headline 180-degree phase gap, separation ~2r ~ 13 500 km
    #       (NAV-G G3).  rho/r = 2.
    tgt = X.elements(a_leo, 0.02, 0.7, 0.0)
    sat = X.elements(a_leo + 30e3, 0.02, 0.7, math.pi)
    g.append(dict(name='A5_leo_180deg', sat=sat, tgt=tgt, a_ref=a_leo))

    # A6 -- WL4/M5 wide-eccentric envelope, a = 12 000 km, e_t = 0.30, 90 deg
    #       (NAV-G G4).
    a_w = 12.0e6
    tgt = X.elements(a_w, 0.30, 2.0, 0.0)
    sat = X.elements(a_w + 300e3, 0.26, 2.0, math.pi / 2.0)
    g.append(dict(name='A6_wide_e30_90deg', sat=sat, tgt=tgt, a_ref=a_w))

    for d in g:
        d['sat_cart'] = om.orbit_to_cartesian(d['sat'])
        d['tgt_cart'] = om.orbit_to_cartesian(d['tgt'])
        d['rho0'] = rng_of(d['sat_cart'], d['tgt_cart'])
        d['rho_over_r'] = d['rho0'] / math.hypot(d['sat_cart'][0],
                                                 d['sat_cart'][1])
        d['period_min'] = period(d['a_ref']) / 60.0
    return g


# ── trajectory rollouts ──────────────────────────────────────────────────────
def roll_chaser(x0, times, burn):
    """Exact chaser trajectory. burn = (t_burn, dv_ms, dirn) or None.

    dirn: 'prograde' (along v) or 'radial' (along r).  Both are in the env's
    Discrete-20 action set (radial +-1 m/s were added in T4).
    """
    out = []
    if burn is None:
        for t in times:
            out.append(propagate_cartesian(x0, t) if t > 0 else tuple(x0))
        return out
    tb, dv, dirn = burn
    xb = propagate_cartesian(x0, tb) if tb > 0 else tuple(x0)
    if dirn == 'radial':
        ux, uy = xb[0], xb[1]
    else:
        ux, uy = xb[2], xb[3]
    n = math.hypot(ux, uy)
    xb_post = (xb[0], xb[1], xb[2] + dv * ux / n, xb[3] + dv * uy / n)
    for t in times:
        if t <= tb:
            out.append(propagate_cartesian(x0, t) if t > 0 else tuple(x0))
        else:
            out.append(propagate_cartesian(xb_post, t - tb))
    return out


def roll_target(x0, times):
    return [propagate_cartesian(x0, t) if t > 0 else tuple(x0) for t in times]


def roll_linear(sat0, tgt0, times, burn, k=1.0):
    """Relative motion under LINEARIZED dynamics about the chaser's reference orbit.

    delta_r(t) = [Phi_c(t,t0) * k*delta_x0]_pos  -  [Phi_c(t,tb) * (0,0,dvx,dvy)]_pos

    The first term is the homogeneous solution and is exactly proportional to k
    -- this is the classical scale symmetry.  The second is the maneuver's
    inhomogeneous term and carries no k, which is exactly why a burn breaks the
    symmetry.  Returns the relative POSITION vectors (target minus chaser).
    """
    dx0 = np.array([tgt0[0] - sat0[0], tgt0[1] - sat0[1],
                    tgt0[2] - sat0[2], tgt0[3] - sat0[3]])
    out = []
    if burn is not None:
        tb, dv, dirn = burn
        xb = propagate_cartesian(sat0, tb) if tb > 0 else tuple(sat0)
        if dirn == 'radial':
            ux, uy = xb[0], xb[1]
        else:
            ux, uy = xb[2], xb[3]
        nrm = math.hypot(ux, uy)
        dvvec = np.array([0.0, 0.0, dv * ux / nrm, dv * uy / nrm])
    for t in times:
        if t <= 0.0:
            rel = k * dx0[:2]
        else:
            rel = (om.stm_numerical(sat0, t) @ (k * dx0))[:2]
        if burn is not None and t > tb:
            rel = rel - (om.stm_numerical(xb, t - tb) @ dvvec)[:2]
        out.append(rel)
    return out


# ── metric extraction ────────────────────────────────────────────────────────
def detection_metrics(times, dbeta, sigma=SIGMA_BETA):
    """First-crossing and batch-detection times, minutes. nan if never."""
    ab = np.abs(np.asarray(dbeta))
    tmin = np.asarray(times) / 60.0
    out = {}
    for lab, thr in (('t_1sig_min', sigma), ('t_3sig_min', 3.0 * sigma)):
        idx = np.nonzero(ab > thr)[0]
        out[lab] = float(tmin[idx[0]]) if idx.size else float('nan')
    D = np.sqrt(np.cumsum((ab / sigma) ** 2))
    idx = np.nonzero(D > 3.0)[0]
    out['t_D3_min'] = float(tmin[idx[0]]) if idx.size else float('nan')
    out['D_final'] = float(D[-1])
    for lab, tm in (('dbeta_10min', 10.0), ('dbeta_1h', 60.0),
                    ('dbeta_6h', 360.0), ('dbeta_24h', 1440.0),
                    ('dbeta_50h', 3000.0)):
        j = np.searchsorted(tmin, tm)
        out[lab] = float(ab[min(j, len(ab) - 1)]) if tm <= tmin[-1] else float('nan')
    out['dbeta_max'] = float(ab.max())
    return out


def power_law(times, dbeta, t_lo=600.0, t_hi=6000.0):
    """Fit |d_beta| ~ t^p over an early window; p distinguishes mechanisms.

    Classical maneuver signal grows ~t (ballistic offset) to ~t^2 (along-track
    amplification of a semi-major-axis change).  Pure gravity-gradient
    nonlinearity accumulates through the mean-motion difference, also ~t
    secularly but with a much smaller coefficient.
    """
    t = np.asarray(times)
    ab = np.abs(np.asarray(dbeta))
    m = (t >= t_lo) & (t <= t_hi) & (ab > 1e-14)
    if m.sum() < 8:
        return float('nan')
    p = np.polyfit(np.log(t[m]), np.log(ab[m]), 1)
    return float(p[0])


# ── one geometry x one burn config ───────────────────────────────────────────
def analyse(geo, duration_s, burn, ks, want_linear=True, trace_rows=None,
            trace_k=2.0):
    times = np.arange(0.0, duration_s + 0.5 * DT, DT)
    sat = roll_chaser(geo['sat_cart'], times, burn)
    tgt = roll_target(geo['tgt_cart'], times)
    beta_true = np.array([bearing(sat[i], tgt[i]) for i in range(len(times))])
    rho_true = np.array([rng_of(sat[i], tgt[i]) for i in range(len(times))])

    lin_true = roll_linear(geo['sat_cart'], geo['tgt_cart'], times, burn, 1.0) \
        if want_linear else None

    rows = []
    for k in ks:
        s0 = geo['sat_cart']
        t0 = geo['tgt_cart']
        hyp0 = tuple(s0[j] + k * (t0[j] - s0[j]) for j in range(4))
        # validity of the hypothesis as a physical orbit
        r0 = math.hypot(hyp0[0], hyp0[1])
        v2 = hyp0[2] ** 2 + hyp0[3] ** 2
        a_h = 1.0 / (2.0 / r0 - v2 / MU)
        el = om.cartesian_to_elements(*hyp0)
        rp = el['a'] * (1.0 - el['e']) if el['a'] > 0 else float('nan')
        valid = (a_h > 0) and (rp > R_EARTH)

        hyp = roll_target(hyp0, times)
        beta_h = np.array([bearing(sat[i], hyp[i]) for i in range(len(times))])
        d_nl = np.array([wrap_pi(beta_true[i] - beta_h[i])
                         for i in range(len(times))])

        rec = dict(geom=geo['name'], rho0_km=geo['rho0'] / 1e3,
                   rho_over_r=geo['rho_over_r'],
                   period_min=geo['period_min'],
                   arc_min=duration_s / 60.0,
                   burn_dv_ms=0.0 if burn is None else burn[1],
                   burn_dir='none' if burn is None else burn[2],
                   burn_t_min=float('nan') if burn is None else burn[0] / 60.0,
                   k=k, hyp_valid=int(valid), hyp_a_km=a_h / 1e3,
                   hyp_perigee_alt_km=(rp - R_EARTH) / 1e3 if rp == rp else float('nan'),
                   arm='nonlinear')
        rec.update(detection_metrics(times, d_nl))
        rec['pow_p'] = power_law(times, d_nl)
        rows.append(rec)

        if want_linear:
            lin_k = roll_linear(geo['sat_cart'], geo['tgt_cart'], times, burn, k)
            d_lin = np.array([wrap_pi(math.atan2(lin_true[i][1], lin_true[i][0]) -
                                      math.atan2(lin_k[i][1], lin_k[i][0]))
                              for i in range(len(times))])
            recl = dict(rec)
            recl['arm'] = 'linear_classical'
            recl.update(detection_metrics(times, d_lin))
            recl['pow_p'] = power_law(times, d_lin)
            rows.append(recl)

            recd = dict(rec)
            recd['arm'] = 'nonlinear_minus_linear'
            recd.update(detection_metrics(times, d_nl - d_lin))
            recd['pow_p'] = power_law(times, d_nl - d_lin)
            rows.append(recd)

        if trace_rows is not None and abs(k - trace_k) < 1e-9:
            stride = max(1, len(times) // 400)
            for i in range(0, len(times), stride):
                trace_rows.append(dict(
                    geom=geo['name'],
                    burn_dv_ms=0.0 if burn is None else burn[1],
                    t_min=times[i] / 60.0, k=k,
                    rho_km=rho_true[i] / 1e3,
                    dbeta_nl_mrad=d_nl[i] * 1e3,
                    dbeta_lin_mrad=(wrap_pi(
                        math.atan2(lin_true[i][1], lin_true[i][0]) -
                        math.atan2(lin_k[i][1], lin_k[i][0])) * 1e3)
                    if want_linear else float('nan')))
    return rows


# ── burn-size threshold: how big a Delta-v to break scale in a given window ──
def burn_threshold(geo, window_min, ks=(0.5, 2.0),
                   dvs=(0.0, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 25.0),
                   dirn='prograde'):
    """D(window) vs Delta-v: the observability-maneuver sizing curve.

    D is the accumulated 3-sigma detection statistic for rejecting the scaled
    hypothesis using only bearings.  D = 3 is the rejection threshold, so the
    smallest dv with D >= 3 inside the window is the required maneuver.
    """
    dur = window_min * 60.0
    out = []
    for dv in dvs:
        burn = None if dv == 0.0 else (0.1 * dur, dv, dirn)
        for k in ks:
            r = analyse(geo, dur, burn, [k], want_linear=False)[0]
            out.append(dict(geom=geo['name'], rho0_km=geo['rho0'] / 1e3,
                            rho_over_r=geo['rho_over_r'],
                            window_min=window_min, burn_dir=dirn,
                            dv_ms=dv, k=k, D_final=r['D_final'],
                            t_D3_min=r['t_D3_min'],
                            t_1sig_min=r['t_1sig_min'],
                            dbeta_max_mrad=r['dbeta_max'] * 1e3))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true')
    args = ap.parse_args()

    geos = make_geometries()
    ks = [0.5, 0.9, 1.1, 2.0]
    # Episode clock is 3000 min (T3) / 6000 min (WL4) / 12000 min (M5). Use the
    # T3 headline clock for the LEO family and 1.5 periods minimum.
    arcs = {g['name']: min(EP_CAP_MIN * 60.0, 40.0 * period(g['a_ref']))
            for g in geos}
    if args.quick:
        arcs = {k: min(v, 6.0 * 3600.0) for k, v in arcs.items()}
        ks = [0.5, 2.0]

    rows, trace = [], []
    print(f"{'geom':<22} {'rho0_km':>9} {'rho/r':>8} {'arc_min':>8} "
          f"{'dv':>6} {'k':>5} {'arm':<24} {'t_1sig':>9} {'t_D3':>8} "
          f"{'db_1h_mrad':>11} {'db_50h_mrad':>12}")
    for g in geos:
        dur = arcs[g['name']]
        for burn in (None, (0.1 * dur, 1.0, 'radial'),
                     (0.1 * dur, 10.0, 'prograde')):
            r = analyse(g, dur, burn, ks, want_linear=True,
                        trace_rows=trace if burn is None or burn[1] == 1.0 else None)
            rows.extend(r)
            for x in r:
                if x['k'] not in (0.5, 2.0):
                    continue
                print(f"{x['geom']:<22} {x['rho0_km']:9.1f} "
                      f"{x['rho_over_r']:8.4f} {x['arc_min']:8.0f} "
                      f"{x['burn_dv_ms']:6.1f} {x['k']:5.2f} {x['arm']:<24} "
                      f"{x['t_1sig_min']:9.1f} {x['t_D3_min']:8.1f} "
                      f"{x['dbeta_1h']*1e3:11.4g} {x['dbeta_50h']*1e3:12.4g}")
        print()

    with open(OUT_MAIN, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {OUT_MAIN} ({len(rows)} rows)")

    with open(OUT_TRACE, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(trace[0].keys()))
        w.writeheader()
        w.writerows(trace)
    print(f"wrote {OUT_TRACE} ({len(trace)} rows)")

    # burn sizing at the two windows that matter operationally:
    #   90 min  ~ one LEO orbit, the acquisition budget
    #   360 min ~ 6 h, a comfortable fraction of the 50 h episode
    brows = []
    for g in geos:
        for w_min in (90.0, 360.0):
            for dirn in ('radial', 'prograde'):
                brows.extend(burn_threshold(g, w_min, dirn=dirn))
    with open(OUT_BURN, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(brows[0].keys()))
        w.writeheader()
        w.writerows(brows)
    print(f"wrote {OUT_BURN} ({len(brows)} rows)")

    print("\n== burn sizing: smallest dv reaching D>=3 within the window ==")
    print(f"{'geom':<22} {'window':>7} {'dir':<9} {'k':>5} {'dv_needed_ms':>13}")
    import collections
    best = collections.OrderedDict()
    for r in brows:
        key = (r['geom'], r['window_min'], r['burn_dir'], r['k'])
        if r['D_final'] >= 3.0 and (key not in best or r['dv_ms'] < best[key]):
            best[key] = r['dv_ms']
    for g in geos:
        for w_min in (90.0, 360.0):
            for dirn in ('radial', 'prograde'):
                for k in (0.5, 2.0):
                    key = (g['name'], w_min, dirn, k)
                    v = best.get(key)
                    print(f"{g['name']:<22} {w_min:7.0f} {dirn:<9} {k:5.2f} "
                          f"{(f'{v:.3g}' if v is not None else '>25'):>13}")


if __name__ == '__main__':
    main()
