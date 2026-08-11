"""EXT-ANGLES — reconciliation probe: does the semi-major-axis difference dominate observability?

Two independent measurements of the same physics disagreed by ~500x:

  this recon (ext_angles_observability_profile.py)
      rho = 5 km, LEO, 1-orbit bearings-only arc, drift-only
      -> CRLB(range)/rho = 0.75, i.e. a ~75% range error is the 1-sigma bound

  the literature recon's own two-body separability experiment
      rho = 5-10 km, LEO, 1-orbit arc, 40 bearings, delta_a = 0 EXACTLY
      -> fitted  d_theta_rms ~ 0.5 * eps * (rho/r)^2  =>  eps_min ~ 2 sigma (r/rho)^2 / sqrt(N)
      -> at rho = 5 km that is eps_min ~ 380, i.e. hopeless

The hypothesis under test: the discrepancy is not an error in either, it is the
SEMI-MAJOR-AXIS DIFFERENCE.  The literature probe puts the target on the chaser's
own circular orbit (delta_a = 0), a pure constant along-track offset: the two
bodies have identical periods, the separation never changes, and the only signal
is orbit curvature -- the O((rho/r)^2) refit residual.  This recon's geometry has
delta_a = 400 m, so the pair drifts 3*pi*delta_a ~ 3.8 km per orbit, comparable
to the 5 km separation itself, and the geometry sweeps.

This matters operationally, not just as bookkeeping.  Our policy phases on a
drift orbit (|delta_a| ~ 100-340 km through the mission), so it lives nowhere
near delta_a = 0 -- EXCEPT at terminal capture, where nulling the orbit is
precisely what it is trying to do.  If the hypothesis holds, angles-only
observability collapses exactly when the chaser succeeds at matching orbits,
which is the one place the tight success box needs it.

Sweeps delta_a from 0 to 10 km at fixed separation, drift-only and with one
1 m/s burn.

Run:  python3 ext_angles_da_dependence.py
Writes web_data/results/ext_angles_da_dependence.csv
"""

import csv
import math
import os
import sys

import numpy as np

sys.path.insert(0, "/Users/pete/space_training/scripts/orbital/nav")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from orbital_math import MU, R_EARTH                             # noqa: E402
import orbital_math as om                                        # noqa: E402
import ext_angles_scale_ambiguity as A                           # noqa: E402
import ext_angles_observability_profile as P                     # noqa: E402
import ext_angles_box_margin as B                                # noqa: E402

OUT = "/Users/pete/space_training/web_data/results/ext_angles_da_dependence.csv"
DT = 60.0


def geom(sep_m, da_m, a_leo=R_EARTH + 400e3):
    tgt = A.X.elements(a_leo, 0.0, 1.7, 0.0)
    sat = A.X.elements(a_leo + da_m, 0.0, 1.7, -sep_m / a_leo)
    g = dict(name=f'sep{sep_m/1e3:g}km_da{da_m:g}m', sat=sat, tgt=tgt,
             a_ref=a_leo)
    g['sat_cart'] = om.orbit_to_cartesian(sat)
    g['tgt_cart'] = om.orbit_to_cartesian(tgt)
    g['rho0'] = math.hypot(g['tgt_cart'][0] - g['sat_cart'][0],
                           g['tgt_cart'][1] - g['sat_cart'][1])
    g['rho_over_r'] = g['rho0'] / math.hypot(*g['sat_cart'][:2])
    return g


def main():
    das = [0.0, 1.0, 10.0, 100.0, 400.0, 1e3, 3e3, 10e3]
    seps = [5e3, 10e3]
    arcs = [90.0, 360.0]
    rows = []
    print("bearings-only CRLB(range)/rho vs semi-major-axis difference, LEO 400 km")
    print("sigma_beta = 1 mrad, 60 s cadence. da=0 is the literature probe's geometry.\n")
    print(f"{'sep_km':>7} {'arc_min':>8} {'da_m':>8} {'drift_rate_km/orb':>18} "
          f"{'CRLB/rho drift':>15} {'CRLB/rho +1m/s':>15} {'gain':>9}")
    for sep in seps:
        for arc in arcs:
            times = np.arange(0.0, arc * 60.0 + 0.5 * DT, DT)
            for da in das:
                g = geom(sep, da)
                p_d = P.chaser_path(g['sat_cart'], times, None)
                s_d, _, rel_d = P.crlb(g, times, p_d)
                p_b = P.chaser_path(g['sat_cart'], times,
                                    (0.2 * arc * 60.0, 1.0, 'radial'))
                s_b, _, rel_b = P.crlb(g, times, p_b)
                drift = 3.0 * math.pi * da / 1e3      # km of along-track per orbit
                gain = (s_d / s_b) ** 2 if s_b > 0 else float('nan')
                rows.append(dict(sep_km=sep / 1e3, arc_min=arc, da_m=da,
                                 rho0_km=g['rho0'] / 1e3,
                                 rho_over_r=g['rho_over_r'],
                                 drift_km_per_orbit=drift,
                                 crlb_rel_drift=rel_d, crlb_m_drift=s_d,
                                 crlb_rel_burn=rel_b, crlb_m_burn=s_b,
                                 info_gain=gain))
                print(f"{sep/1e3:7.1f} {arc:8.0f} {da:8.4g} {drift:18.4g} "
                      f"{rel_d:15.4g} {rel_b:15.4g} {gain:9.3g}")
            print()

    with open(OUT, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {OUT} ({len(rows)} rows)")

    # Literature-probe prediction for the da = 0 rows, for direct comparison.
    print("\ncomparison at da = 0 (the literature probe's exact geometry):")
    print("  predicted eps_min = 2*sigma*(r/rho)^2/sqrt(N)")
    for r_ in rows:
        if r_['da_m'] != 0.0:
            continue
        N = int(r_['arc_min']) + 1
        pred = 2.0 * 1e-3 / (r_['rho_over_r'] ** 2) / math.sqrt(N)
        print(f"  sep {r_['sep_km']:5.1f} km arc {r_['arc_min']:5.0f} min: "
              f"predicted {pred:10.4g}   measured {r_['crlb_rel_drift']:10.4g}   "
              f"ratio {r_['crlb_rel_drift']/pred:8.3g}")


if __name__ == '__main__':
    main()
