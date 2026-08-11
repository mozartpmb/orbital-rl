"""EXT-ANGLES — where is the crossover between maneuver-driven and nonlinearity-driven range observability?

The whole verdict turns on one number: the separation below which a chaser
maneuver is what makes range observable (the classical Woffinden-Geller regime)
and above which the curvature of the two Keplerian arcs already does it for free.

Measured as the INFORMATION GAIN of one 1 m/s burn,

    G(rho) = ( sigma_range_drift / sigma_range_burn )^2

over a fixed arc, swept in separation.  G >> 1 means the maneuver is what buys
the range; G ~ 1 means the maneuver is irrelevant and the arc alone suffices.
Reported alongside the absolute drift-only sigma so "irrelevant" can be read as
"already good" rather than "hopeless either way".

Run:  python3 ext_angles_crossover.py
Writes web_data/results/ext_angles_crossover.csv
"""

import csv
import math
import os
import sys

import numpy as np

sys.path.insert(0, "/Users/pete/space_training/scripts/orbital/nav")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from orbital_math import R_EARTH                                 # noqa: E402
import ext_angles_observability_profile as P                     # noqa: E402
import ext_angles_box_margin as B                                # noqa: E402

OUT = "/Users/pete/space_training/web_data/results/ext_angles_crossover.csv"
DT = 60.0


def main():
    seps = [2e3, 5e3, 10e3, 20e3, 30e3, 50e3, 100e3, 200e3, 300e3, 500e3,
            1e6, 2e6, 5e6, 1e7]
    arcs = [90.0, 180.0, 360.0]
    rows = []
    print("information gain G of ONE 1 m/s burn placed 20% into the arc")
    print("G = (sigma_drift/sigma_burn)^2 ; G ~ 1 means the burn is irrelevant\n")
    print(f"{'rho_km':>9} {'rho/r':>9} " +
          " ".join(f"{'G@'+str(int(a))+'min':>12} {'sig_drift/rho':>14}"
                   for a in arcs))
    for sep in seps:
        g = B.geom_at(sep)
        cells = []
        for arc in arcs:
            times = np.arange(0.0, arc * 60.0 + 0.5 * DT, DT)
            p_d = P.chaser_path(g['sat_cart'], times, None)
            s_d, _, rel_d = P.crlb(g, times, p_d)
            best = 0.0
            for dirn in ('radial', 'prograde'):
                p_b = P.chaser_path(g['sat_cart'], times,
                                    (0.2 * arc * 60.0, 1.0, dirn))
                s_b, _, rel_b = P.crlb(g, times, p_b)
                best = max(best, (s_d / s_b) ** 2 if s_b > 0 else 0.0)
                rows.append(dict(rho_km=g['rho0'] / 1e3,
                                 rho_over_r=g['rho_over_r'], arc_min=arc,
                                 burn_dir=dirn, dv_ms=1.0,
                                 sigma_drift_m=s_d, sigma_drift_rel=rel_d,
                                 sigma_burn_m=s_b, sigma_burn_rel=rel_b,
                                 info_gain=(s_d / s_b) ** 2 if s_b > 0 else 0.0))
            cells.append(f"{best:12.4g} {rel_d:14.4g}")
        print(f"{g['rho0']/1e3:9.2f} {g['rho_over_r']:9.5f} " + " ".join(cells))

    with open(OUT, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {OUT} ({len(rows)} rows)")


if __name__ == '__main__':
    main()
