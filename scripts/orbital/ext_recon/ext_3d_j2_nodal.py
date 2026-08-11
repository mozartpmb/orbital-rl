#!/usr/bin/env python3
"""
ext-3d B3 addendum — differential J2 nodal precession as a *free* plane-change
mechanism, and how much of it is reachable inside the proven episode horizons.

The shipped env is pure two-body, so RAAN is frozen and every plane change must
be bought propulsively. Real operators buy differential RAAN with altitude
(nodal drift rate depends on a, e, i), which is the dominant cheap technique.
This table quantifies what enabling secular J2 rates would put inside reach.

Secular rates (Vallado 9-38..9-40):
  Omega_dot = -1.5 n J2 (Re/p)^2 cos i
  omega_dot =  0.75 n J2 (Re/p)^2 (4 - 5 sin^2 i)
  M_dot     =  n + 0.75 n J2 (Re/p)^2 sqrt(1-e^2)(2 - 3 sin^2 i)
with p = a(1-e^2).

Writes web_data/results/ext_3d_j2_nodal.csv
"""
import csv
import math
import os

MU, R_EQ, J2 = 3.986004418e14, 6.378137e6, 1.08263e-3
R_EARTH = 6.371e6
DT = 60.0
OUT = "/Users/pete/space_training/web_data/results"


def rates(a, e, i):
    n = math.sqrt(MU / a ** 3)
    p = a * (1 - e * e)
    k = 1.5 * n * J2 * (R_EQ / p) ** 2
    Om = -k * math.cos(i)
    om = 0.5 * k * (2.0 - 2.5 * math.sin(i) ** 2)
    return n, Om, om


def main():
    rows = []
    d2r = math.radians
    for alt in (400.0, 800.0, 2000.0, 8000.0, 20200.0):
        a = R_EARTH + alt * 1e3
        vc = math.sqrt(MU / a)
        for i_deg in (0.0, 28.5, 51.6, 63.4, 90.0, 98.0):
            i = d2r(i_deg)
            n, Om, om = rates(a, 0.0, i)
            Om_dps = math.degrees(Om) * 86400.0
            om_dps = math.degrees(om) * 86400.0
            for da_km in (50.0, 100.0, 200.0, 400.0, 600.0):
                a2 = a + da_km * 1e3
                _, Om2, _ = rates(a2, 0.0, i)
                dOm_dps = math.degrees(Om2 - Om) * 86400.0
                # in-plane drift rate that comes with the same da (why it is not free)
                n2 = math.sqrt(MU / a2 ** 3)
                dlam_dps = math.degrees(n2 - n) * 86400.0
                for horizon in (3000, 6000, 12000):
                    days = horizon * DT / 86400.0
                    dOm_avail = abs(dOm_dps) * days
                    # propulsive cost of the RAAN change that drifts for free
                    theta = abs(math.sin(i)) * d2r(dOm_avail)
                    dv_equiv = 2 * vc * math.sin(0.5 * theta)
                    rows.append(dict(
                        alt_km=alt, i_deg=i_deg, da_km=da_km, horizon_steps=horizon,
                        horizon_days=days,
                        RAAN_rate_deg_per_day=Om_dps,
                        argp_rate_deg_per_day=om_dps,
                        dRAAN_rate_deg_per_day=dOm_dps,
                        dlambda_rate_deg_per_day=dlam_dps,
                        dRAAN_available_deg=dOm_avail,
                        theta_equiv_deg=math.degrees(theta),
                        dv_equiv_saved_ms=dv_equiv,
                        frac_of_budget=dv_equiv / 478.12987798780637))
    os.makedirs(OUT, exist_ok=True)
    p = f"{OUT}/ext_3d_j2_nodal.csv"
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print("wrote", p, len(rows), "rows")
    print("\n-- free differential RAAN inside a horizon, i = 51.6 deg --")
    print(f"{'alt':>7} {'da_km':>6} {'horizon':>8} {'days':>6} "
          f"{'dOmdot(deg/d)':>13} {'dOm avail':>10} {'dv saved':>9}")
    for r in rows:
        if r['i_deg'] == 51.6 and r['da_km'] in (100.0, 400.0):
            print(f"{r['alt_km']:7.0f} {r['da_km']:6.0f} {r['horizon_steps']:8d} "
                  f"{r['horizon_days']:6.2f} {r['dRAAN_rate_deg_per_day']:13.4f} "
                  f"{r['dRAAN_available_deg']:10.3f} {r['dv_equiv_saved_ms']:9.1f}")
    print("\n-- absolute nodal / apsidal rates (circular) --")
    seen = set()
    for r in rows:
        k = (r['alt_km'], r['i_deg'])
        if k in seen:
            continue
        seen.add(k)
        print(f"  alt {r['alt_km']:7.0f} km  i={r['i_deg']:5.1f}  "
              f"Omega_dot={r['RAAN_rate_deg_per_day']:8.3f} deg/day  "
              f"omega_dot={r['argp_rate_deg_per_day']:8.3f} deg/day")


if __name__ == "__main__":
    main()
