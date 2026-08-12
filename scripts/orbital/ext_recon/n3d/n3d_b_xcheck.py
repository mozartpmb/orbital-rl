#!/usr/bin/env python3
"""N3D-B.0 — cross-check of the two independently-written 3D CRLB pipelines.

Sibling lane N3D-A (`n3d_crlb.py` + `n3d_core.py`) and this lane (`n3d_crlb3d.py`)
each built a 3D angles-only Fisher bound, from different formulations:

                    N3D-A                          N3D-B (this lane)
  measurement    explicit (az, el) about a       basis-free P_perp = I - u u^T
                 pole frame, R = diag(sb^2/      (equivalent to az/el with
                 cos^2 el, sb^2)                 sigma_az = sb/cos el)
  propagation    f&g in dE, 6-state              universal variables + Stumpff
  STM            central difference of THAT      central difference of THAT
                 propagator                      propagator
  scaling        raw FIM                         nondimensionalised by
                                                 diag(a,a,a,v_c,v_c,v_c)
  geometry       f_oop split of a fixed rho,     dI_rel rotation about r_hat,
                 out-of-plane peak AT EPOCH      out-of-plane peak a QUARTER
                                                 ORBIT LATER

Feeding BOTH CRLBs the same truth states isolates the estimator formulation
from the geometry convention, so agreement is evidence rather than a shared
assumption (the T3/3D-E oracle discipline). The geometry-convention difference
is then measured separately as a real effect, not as disagreement.

Writes web_data/results/n3d_crlb_xcheck.csv
Run: python3 n3d_b_xcheck.py
"""

import csv
import math
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import n3d_crlb3d as K                                          # noqa: E402
import n3d_core as A                                            # noqa: E402
import n3d_crlb as AC                                           # noqa: E402

OUT = '/Users/pete/space_training/web_data/results/n3d_crlb_xcheck.csv'


def main():
    rows = []
    a = K.R_EARTH + 500e3
    P = A.period(a)
    n = math.sqrt(K.MU / a ** 3)
    cases = []
    for v_tol in (0.0, 1.0, 50.0):
        da = v_tol / (1.5 * n)
        for f_oop in (0.0, 0.5, 0.9):
            for dv in (0.0, 1.0):
                cases.append((v_tol, da, 5.0e3, f_oop, dv, 1.0))
    print(f"{'v_tol':>6} {'da_m':>9} {'f_oop':>6} {'di_deg':>8} {'dv':>5} | "
          f"{'A sig_rng':>12} {'B sig_rng':>12} {'B/A':>7} | "
          f"{'A sig_vel':>10} {'B sig_vel':>10} {'B/A':>7}")
    for v_tol, da, rho, f_oop, dv, orbits in cases:
        sat0, tgt0, di = A.make_geometry(a, 0.0, da, rho, f_oop)
        nst = int(round(orbits * P / 60.0))
        burns = ((int(round(0.20 * nst)), dv),) if dv > 0 else ()
        t, S, G = A.roll_truth(sat0, tgt0, 60.0, nst, burns=burns)
        ra = AC.crlb(t, S, G)
        rb = K.crlb3d(list(t), list(S), list(G))[-1]
        r_a, r_b = ra['sig_range'], rb['sigma_range_m']
        v_a = ra['sig_vel']
        v_b = math.sqrt(rb['sigma_vrange_ms'] ** 2
                        + rb['sigma_vinplane_ms'] ** 2
                        + rb['sigma_vplane_ms'] ** 2)
        rows.append(dict(v_tol_ms=v_tol, da_m=da, rho_m=rho, f_oop=f_oop,
                         di_deg=math.degrees(di), dv_ms=dv, orbits=orbits,
                         a_sig_range_m=r_a, b_sig_range_m=r_b,
                         ratio_range=r_b / r_a if r_a > 0 else float('nan'),
                         a_sig_vel_ms=v_a, b_sig_vel_ms=v_b,
                         ratio_vel=v_b / v_a if v_a > 0 else float('nan'),
                         a_cond=ra['cond'], b_cond=rb['fim_cond']))
        print(f"{v_tol:6.1f} {da:9.1f} {f_oop:6.2f} {math.degrees(di):8.4f} "
              f"{dv:5.1f} | {r_a:12.2f} {r_b:12.2f} "
              f"{r_b/max(r_a,1e-12):7.4f} | {v_a:10.5f} {v_b:10.5f} "
              f"{v_b/max(v_a,1e-12):7.4f}")

    with open(OUT, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    fin = [r['ratio_range'] for r in rows if np.isfinite(r['ratio_range'])]
    print(f"\nrange-sigma ratio B/A over {len(fin)} matched cells: "
          f"min {min(fin):.4f}  median {float(np.median(fin)):.4f}  "
          f"max {max(fin):.4f}")
    print(f"wrote {OUT} ({len(rows)} rows)")


if __name__ == '__main__':
    main()
