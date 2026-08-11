"""NAV-G — bearings-only observability map (filter-independent).

Answers the design question the filter comparison cannot: for a given geometry,
how much range information IS there in an angles-only arc, and what buys more of
it? Everything here is the Cramer-Rao bound from the exact Fisher information

    F = sum_k H_k^T R^-1 H_k,   H_k = (d z / d r_k) * Phi(t_k, t_0)

with Phi accumulated from the same numerically differenced two-body STM the
filters use. No filter, no tuning, no Monte Carlo — this is the information
content of the geometry itself, so it is the right thing to design against and
the right thing to compare a filter to.

Three axes, each with a decision attached:
  separation   does angles-only hold up at the tight terminal box (5-10 km),
               where rho/r ~ 1e-3 and the linearized relative dynamics are
               nearly scale-invariant (the classic unobservability regime)?
  arc length   how long must the chaser stare before the range is determined —
               i.e. what is the acquisition latency budget?
  burn size    the Woffinden-Geller observability maneuver. The policy already
               burns; does it burn ENOUGH, and does the 1 m/s radial action at
               terminal phase carry its weight?

Run:  python3 ext_bo_observability.py
Writes web_data/results/ext_bo_observability.csv
"""

import csv
import math
import os
import sys

import numpy as np

sys.path.insert(0, "/Users/pete/space_training/scripts/orbital/nav")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from orbital_math import MU, R_EARTH                       # noqa: E402
import ext_bo_filter as X                                  # noqa: E402

OUT = "/Users/pete/space_training/web_data/results/ext_bo_observability.csv"
SIG_B = X.SIGMA_BETA
SIG_R = X.SIGMA_RHO_M


def build(a_c, e, sep_m, n_orbits, dv_ms, dt=60.0, burn_frac=0.35):
    """Co-orbital-ish geometry with a given along-track separation at t0."""
    tgt = X.elements(a_c, e, 0.5, 0.0)
    sat = X.elements(a_c, e, 0.5, -sep_m / a_c)
    dur = n_orbits * X.period(a_c)
    burns = ((burn_frac * dur, dv_ms),) if dv_ms else ()
    return dict(name='x', sat=sat, tgt=tgt, dt=dt, duration=dur,
                r_min=R_EARTH + 300e3, r_max=R_EARTH + 800e3,
                sigma_v_ecc=100.0, burns=burns)


def main():
    rows = []
    a_leo = R_EARTH + 400e3
    seps = [5e3, 10e3, 50e3, 200e3, 1e6, 5e6, 1.3e7]
    arcs = [0.25, 0.5, 1.0, 2.0, 3.0]
    dvs = [0.0, 1.0, 5.0, 25.0]

    print(f"{'sep_km':>9} {'orbits':>7} {'dv':>6} | "
          f"{'sigLOS_m':>10} {'sig/rho':>10} {'sigLOS_RB_m':>12} "
          f"{'BO/RB':>8} {'gain_vs_dv0':>12}")
    for sep in seps:
        for narb in arcs:
            base = None
            for dv in dvs:
                sc = build(a_leo, 0.0, sep, narb, dv)
                t, s, g = X.roll_truth(sc)
                sl_b, sp_b, cond_b = X.crlb_range_sigma(t, s, g, SIG_B, False)
                sl_r, _, _ = X.crlb_range_sigma(t, s, g, SIG_B, True, SIG_R)
                rho0 = X.range_of(s[0], g[0])
                if dv == 0.0:
                    base = sl_b
                rows.append(dict(a_km=a_leo / 1e3, e=0.0, sep_km=sep / 1e3,
                                 orbits=narb, dv_ms=dv, n_obs=len(t),
                                 rho0_km=rho0 / 1e3,
                                 sigma_los_bo_m=sl_b,
                                 sigma_los_rb_m=sl_r,
                                 rel_sigma_bo=sl_b / rho0,
                                 bo_over_rb=sl_b / max(sl_r, 1e-9),
                                 gain_vs_no_burn=(base / sl_b) if sl_b > 0 else float('nan'),
                                 fim_cond_bo=cond_b))
                print(f"{sep/1e3:9.1f} {narb:7.2f} {dv:6.1f} | "
                      f"{sl_b:10.2f} {sl_b/rho0:10.2e} {sl_r:12.2f} "
                      f"{sl_b/max(sl_r,1e-9):8.1f} "
                      f"{(base/sl_b if sl_b>0 else float('nan')):12.2f}")
            print()

    # eccentric / wide-envelope spot checks
    print("wide-envelope spot checks (a = 12 000 km)")
    for e in (0.0, 0.30, 0.50):
        for sep in (10e3, 1e6, 1.2e7):
            for dv in (0.0, 25.0):
                sc = build(12.0e6, e, sep, 1.5, dv)
                sc['r_max'] = 1.868e7
                t, s, g = X.roll_truth(sc)
                sl_b, _, cond_b = X.crlb_range_sigma(t, s, g, SIG_B, False)
                sl_r, _, _ = X.crlb_range_sigma(t, s, g, SIG_B, True, SIG_R)
                rho0 = X.range_of(s[0], g[0])
                rows.append(dict(a_km=12000.0, e=e, sep_km=sep / 1e3,
                                 orbits=1.5, dv_ms=dv, n_obs=len(t),
                                 rho0_km=rho0 / 1e3, sigma_los_bo_m=sl_b,
                                 sigma_los_rb_m=sl_r,
                                 rel_sigma_bo=sl_b / rho0,
                                 bo_over_rb=sl_b / max(sl_r, 1e-9),
                                 gain_vs_no_burn=float('nan'),
                                 fim_cond_bo=cond_b))
                print(f"  e={e:.2f} sep={sep/1e3:8.1f} km dv={dv:5.1f} -> "
                      f"sigLOS {sl_b:10.2f} m ({sl_b/rho0:.2e} rel), "
                      f"RB {sl_r:8.2f} m")

    with open(OUT, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {OUT} ({len(rows)} rows)")


if __name__ == '__main__':
    main()
