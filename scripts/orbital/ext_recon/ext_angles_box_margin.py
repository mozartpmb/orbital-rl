"""EXT-ANGLES — does bearings-only nav clear the SUCCESS BOX at the arcs the policy flies?

The observability map says how well range is determined.  The decision the
campaign needs is narrower: at the separations and burn-free coast gaps the
shipped policy actually produces (`ext_angles_policy_actions.py`), is the
resulting navigation uncertainty small compared with the terminal tolerance it
has to hit?

Measured policy behaviour at the tight box (TB5, 5 km / 1 m/s, 30 episodes):
  31.1% of sim time is spent below 10 km separation, median dwell 301 min
  median longest burn-free gap below 10 km = 180 min, p90 = 300 min
  closest approach p50 2.66 km

So the operative question is the CRLB after a 60-300 min bearings-only arc at
2-10 km separation, in metres and m/s, against 5000 m and 1 m/s.

Emits both the range (LOS) sigma and the relative-velocity sigma, drift-only and
with one 1 m/s burn, so the margin can be read directly.

Run:  python3 ext_angles_box_margin.py
Writes web_data/results/ext_angles_box_margin.csv
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

OUT = "/Users/pete/space_training/web_data/results/ext_angles_box_margin.csv"
DT = 60.0
SIG_B = 1.0e-3


def full_crlb(geo, times, sat_path, sigma_beta=SIG_B):
    """(sigma_LOS_range, sigma_pos_total, sigma_vel_total) on the target state."""
    x0 = np.array(geo['tgt_cart'], dtype=float)
    L = math.hypot(x0[0], x0[1])
    V = math.sqrt(MU / L)
    S = np.diag([L, L, V, V])
    h = np.array([1.0, 1.0, 1e-3, 1e-3])
    J = np.zeros((len(times), 4))
    for j in range(4):
        xp = x0.copy(); xp[j] += h[j]
        xm = x0.copy(); xm[j] -= h[j]
        J[:, j] = P.wrap(P.bearings_of(xp, sat_path, times) -
                         P.bearings_of(xm, sat_path, times)) / (2.0 * h[j])
    Js = J @ S
    F = (Js.T @ Js) / sigma_beta ** 2
    try:
        C = S @ np.linalg.inv(F) @ S.T
    except np.linalg.LinAlgError:
        return float('inf'), float('inf'), float('inf')
    d = np.array([x0[0] - sat_path[0, 0], x0[1] - sat_path[0, 1]])
    rho = float(np.linalg.norm(d))
    u = d / rho
    sl = math.sqrt(max(float(u @ C[:2, :2] @ u), 0.0))
    sp = math.sqrt(max(float(np.trace(C[:2, :2])), 0.0))
    sv = math.sqrt(max(float(np.trace(C[2:, 2:])), 0.0))
    return sl, sp, sv


def geom_at(sep_m, a_leo=None):
    """Co-orbital geometry at a chosen along-track separation, near-circular."""
    a_leo = a_leo or (R_EARTH + 400e3)
    tgt = A.X.elements(a_leo, 0.001, 1.7, 0.0)
    sat = A.X.elements(a_leo + 400.0, 0.001, 1.7, -sep_m / a_leo)
    g = dict(name=f'box_{sep_m/1e3:g}km', sat=sat, tgt=tgt, a_ref=a_leo)
    g['sat_cart'] = om.orbit_to_cartesian(sat)
    g['tgt_cart'] = om.orbit_to_cartesian(tgt)
    g['rho0'] = P.math.hypot(g['tgt_cart'][0] - g['sat_cart'][0],
                             g['tgt_cart'][1] - g['sat_cart'][1])
    g['rho_over_r'] = g['rho0'] / math.hypot(*g['sat_cart'][:2])
    return g


def main():
    seps = [2.66e3, 5e3, 10e3, 30e3]        # p50 closest approach, the two boxes
    arcs = [60.0, 120.0, 180.0, 300.0, 600.0]
    rows = []
    print("bearings-only CRLB at the terminal box, sigma_beta = 1 mrad, 60 s cadence")
    print("boxes: TB5 = 5000 m / 1 m/s   T3 headline = 30000 m / 50 m/s\n")
    print(f"{'rho_km':>8} {'arc_min':>8} {'burn':<14} {'sig_rng_m':>11} "
          f"{'sig_pos_m':>11} {'sig_vel_ms':>11} {'rng/5km':>9} {'vel/1ms':>9}")
    for sep in seps:
        g = geom_at(sep)
        for arc in arcs:
            times = np.arange(0.0, arc * 60.0 + 0.5 * DT, DT)
            for lab, burn in (('drift', None),
                              ('1 m/s radial', (0.2 * arc * 60.0, 1.0, 'radial')),
                              ('1 m/s prograde', (0.2 * arc * 60.0, 1.0, 'prograde'))):
                path = P.chaser_path(g['sat_cart'], times, burn)
                sl, sp, sv = full_crlb(g, times, path)
                rows.append(dict(rho_km=g['rho0'] / 1e3, arc_min=arc, burn=lab,
                                 sigma_range_m=sl, sigma_pos_m=sp,
                                 sigma_vel_ms=sv,
                                 frac_of_tb5_pos=sl / 5000.0,
                                 frac_of_tb5_vel=sv / 1.0,
                                 frac_of_t3_pos=sl / 30000.0,
                                 frac_of_t3_vel=sv / 50.0))
                print(f"{g['rho0']/1e3:8.2f} {arc:8.0f} {lab:<14} {sl:11.4g} "
                      f"{sp:11.4g} {sv:11.4g} {sl/5000.0:9.3g} {sv/1.0:9.3g}")
            print()
    with open(OUT, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {OUT} ({len(rows)} rows)")


if __name__ == '__main__':
    main()
