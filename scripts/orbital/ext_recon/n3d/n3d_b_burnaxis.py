#!/usr/bin/env python3
"""N3D-B.5 — burn AXIS x burn PLACEMENT at the terminal box, in 3D.

NAV-F §2.5 measured, in 2D, that (a) a maneuver at the very start of the arc is
worth nothing and the information peaks ~20% in, and (b) radial is the WORST
1 m/s direction at multi-hour arcs. 3D adds a third axis — normal — which the
Discrete-30 action set already carries and which 3D guidance must use anyway.

The 3D map (delta-a = 0, burn at 35% of arc) and the 3D tight box (delta-a != 0,
burn at 20%) disagreed about whether normal beats prograde, so placement and
delta-a are swept jointly here rather than asserted.

Writes web_data/results/n3d_burn_axis_placement.csv
Run: python3 n3d_b_burnaxis.py
"""

import csv
import math
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import n3d_crlb3d as K                                          # noqa: E402

OUT = '/Users/pete/space_training/web_data/results/n3d_burn_axis_placement.csv'
A_LEO = K.R_EARTH + 500e3
I_T = 51.6


def sigma(da, di_deg, dv, axis, frac, orbits=1.0, sep=5e3):
    dur = orbits * K.period(A_LEO)
    s0, t0 = K.make_pair(a_t=A_LEO, e=0.0, i_t=math.radians(I_T), nu_t=0.6,
                         sep_m=sep, da_m=da, di_rad=math.radians(di_deg))
    burns = ((frac * dur, dv, axis),) if dv > 0 else ()
    T, S, G = K.roll(s0, t0, dur, burns=burns)
    return K.crlb3d(T, S, G)[-1]['sigma_range_m']


def main():
    n = math.sqrt(K.MU / A_LEO ** 3)
    axes = ['pro', 'rad', 'nor', 'perp_los']
    fracs = [0.0, 0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 0.95]
    rows = []
    for v_tol in (0.0, 1.0, 10.0):
        da = v_tol / (1.5 * n)
        for di in (0.0, 0.0416939):
            base = sigma(da, di, 0.0, '', 0.0)
            print(f"\n-- v_tol {v_tol} m/s (da {da:.1f} m), dI {di:.4f} deg, "
                  f"1 orbit, drift-only sigma_range = {base:.1f} m --")
            print(f"{'frac':>6} | " + " ".join(f"{a:>21}" for a in axes))
            for fr in fracs:
                cells = []
                for a in axes:
                    s = sigma(da, di, 1.0, a, fr)
                    g = (base / s) ** 2 if np.isfinite(base) and s > 0 \
                        else float('inf')
                    cells.append((s, g))
                    rows.append(dict(v_tol_ms=v_tol, da_m=da, di_deg=di,
                                     dv_ms=1.0, dv_axis=a, burn_frac=fr,
                                     orbits=1.0, sigma_drift_m=base,
                                     sigma_range_m=s, gain=g))
                print(f"{fr:6.2f} | " + " ".join(
                    f"{s:9.2f}m x{g:<10.4g}" for s, g in cells))
    with open(OUT, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\nwrote {OUT} ({len(rows)} rows)")


if __name__ == '__main__':
    main()
