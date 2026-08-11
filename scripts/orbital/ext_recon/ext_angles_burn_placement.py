"""EXT-ANGLES — where and which way should the observability burn point?

`ext_angles_observability_profile.py` sizes the burn (how many m/s).  This sizes
its GEOMETRY and its TIMING, which are the two things an RL policy could plausibly
learn and the two things an experiment has to be able to detect.

Two sweeps, both on arcs matched to what the shipped tight-box policy actually
flies (`ext_angles_policy_actions.py`: median 5 h dwell below 10 km separation,
median 3 h longest burn-free gap in there):

  DIRECTION   a fixed 1 m/s burn rotated through 360 deg relative to the
              instantaneous line of sight.  The classical prediction is that a
              burn along the LOS is worthless -- it displaces the observer
              towards or away from the target, changing range but not bearing --
              and a burn perpendicular to the LOS is best.  In a co-orbital
              geometry the LOS is along-track, so this predicts RADIAL >> ALONG-
              TRACK, which is the opposite of the direction guidance wants.  That
              tension is the whole reason an observability maneuver can be a
              distinct, learnable behaviour rather than a free by-product.

  TIMING      the same burn slid through the arc.  d(information)/d(burn epoch)
              is the quantity an "is the policy burning where it is informative"
              metric has to correlate against, so it needs to be measured, not
              assumed monotone.

Reported as the CRLB on range along the LOS, normalised by the separation, and
as information gain (sigma_drift/sigma_burn)^2 -- the factor by which one burn
multiplies the Fisher information about range.

Run:  python3 ext_angles_burn_placement.py
Writes web_data/results/ext_angles_burn_geometry.csv
"""

import csv
import math
import os
import sys

import numpy as np

sys.path.insert(0, "/Users/pete/space_training/scripts/orbital/nav")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from orbital_math import MU                                      # noqa: E402
import ext_angles_scale_ambiguity as A                           # noqa: E402
import ext_angles_observability_profile as P                     # noqa: E402

OUT = "/Users/pete/space_training/web_data/results/ext_angles_burn_geometry.csv"
DT = 60.0


def chaser_path_dirn(x0, times, t_burn, dv, ang_from_los, los_hat):
    """Chaser path with a 1-burn at t_burn, pointed ang_from_los off the LOS.

    ang_from_los is measured in the inertial plane from the line-of-sight unit
    vector at the burn epoch, so 0 = straight at the target, pi/2 = transverse.
    """
    pre = times[times <= t_burn]
    post = times[times > t_burn]
    out = np.zeros((len(times), 4))
    if len(pre):
        out[:len(pre)] = P.prop_vec(x0, pre)
    xb = P.prop_vec(x0, np.array([t_burn]))[0]
    c, s = math.cos(ang_from_los), math.sin(ang_from_los)
    ux = c * los_hat[0] - s * los_hat[1]
    uy = s * los_hat[0] + c * los_hat[1]
    xbp = (xb[0], xb[1], xb[2] + dv * ux, xb[3] + dv * uy)
    if len(post):
        out[len(pre):] = P.prop_vec(xbp, post - t_burn)
    return out


def los_at(geo, t):
    s = P.prop_vec(geo['sat_cart'], np.array([t]))[0]
    g = P.prop_vec(geo['tgt_cart'], np.array([t]))[0]
    d = np.array([g[0] - s[0], g[1] - s[1]])
    return d / np.linalg.norm(d)


def main():
    geos = {g['name']: g for g in A.make_geometries()}
    rows = []

    # arcs matched to the measured policy dwell at the tight box
    cases = [('A1_leo_5km_box', 360.0), ('A1_leo_5km_box', 180.0),
             ('A2_leo_10km_drift', 360.0), ('A3_leo_100km', 180.0)]

    print("== DIRECTION sweep: 1 m/s at 25% into the arc, angle from LOS ==")
    print(f"{'geom':<20} {'arc':>5} {'ang_deg':>8} {'crlb/rho':>11} "
          f"{'info gain':>11}")
    for name, arc_min in cases:
        g = geos[name]
        times = np.arange(0.0, arc_min * 60.0 + 0.5 * DT, DT)
        tb = 0.25 * arc_min * 60.0
        u = los_at(g, tb)
        base_path = P.prop_vec(g['sat_cart'], times)
        s_drift, _, rel_drift = P.crlb(g, times, base_path)
        for ang_deg in range(0, 360, 15):
            ang = math.radians(ang_deg)
            path = chaser_path_dirn(g['sat_cart'], times, tb, 1.0, ang, u)
            sig, cond, rel = P.crlb(g, times, path)
            gain = (s_drift / sig) ** 2 if sig > 0 else float('inf')
            rows.append(dict(sweep='direction', geom=name, arc_min=arc_min,
                             rho0_km=g['rho0'] / 1e3, dv_ms=1.0,
                             burn_frac=0.25, ang_from_los_deg=ang_deg,
                             crlb_los_m=sig, crlb_rel=rel,
                             crlb_rel_drift=rel_drift, info_gain=gain,
                             fim_cond=cond))
            if ang_deg % 30 == 0:
                print(f"{name:<20} {arc_min:5.0f} {ang_deg:8d} {rel:11.4g} "
                      f"{gain:11.4g}")
        print(f"{name:<20} {arc_min:5.0f} {'drift':>8} {rel_drift:11.4g} "
              f"{1.0:11.4g}\n")

    print("== TIMING sweep: 1 m/s transverse (90 deg from LOS), burn epoch ==")
    print(f"{'geom':<20} {'arc':>5} {'burn_frac':>10} {'crlb/rho':>11} "
          f"{'info gain':>11}")
    for name, arc_min in cases:
        g = geos[name]
        times = np.arange(0.0, arc_min * 60.0 + 0.5 * DT, DT)
        base_path = P.prop_vec(g['sat_cart'], times)
        s_drift, _, rel_drift = P.crlb(g, times, base_path)
        for frac in [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
            tb = max(60.0, frac * arc_min * 60.0)
            u = los_at(g, tb)
            path = chaser_path_dirn(g['sat_cart'], times, tb, 1.0,
                                    math.pi / 2.0, u)
            sig, cond, rel = P.crlb(g, times, path)
            gain = (s_drift / sig) ** 2 if sig > 0 else float('inf')
            rows.append(dict(sweep='timing', geom=name, arc_min=arc_min,
                             rho0_km=g['rho0'] / 1e3, dv_ms=1.0,
                             burn_frac=frac, ang_from_los_deg=90,
                             crlb_los_m=sig, crlb_rel=rel,
                             crlb_rel_drift=rel_drift, info_gain=gain,
                             fim_cond=cond))
            print(f"{name:<20} {arc_min:5.0f} {frac:10.2f} {rel:11.4g} "
                  f"{gain:11.4g}")
        print()

    with open(OUT, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {OUT} ({len(rows)} rows)")


if __name__ == '__main__':
    main()
