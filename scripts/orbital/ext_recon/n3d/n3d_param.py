#!/usr/bin/env python3
"""N3D-A §1 — state-parameterisation evidence: why NOT relative orbital
elements / linearised relative motion (ROE, CW/Hill, Yamanaka-Ankersen).

The ROE school (Gaias/D'Amico 2014, Sullivan & D'Amico 2017) is the flight-
proven angles-only relative-navigation parameterisation, so rejecting it needs
a measured reason, not a preference.  Two are measured here:

  (a) VALIDITY.  Linearised relative motion is exact only to O(rho/r).  This
      probe propagates the exact two-body relative state and the Clohessy-
      Wiltshire (Hill) linear state from the same initial condition, and
      reports the divergence over the arcs the filter actually runs.
  (b) OBSERVABILITY.  Any HOMOGENEOUS LINEAR relative model has the exact
      scale symmetry x_rel -> k x_rel, so range is structurally unobservable
      in it (Woffinden-Geller; reproduced numerically in NAV-F 2.1, where the
      linear arm's scaled-trajectory bearing difference is identically zero).
      Our entire observability is the O(rho/r) nonlinearity that a linear
      parameterisation deletes by construction.  A first-order ROE filter would
      therefore be exactly unobservable in range; recovering it means carrying
      the second-order terms, i.e. re-introducing the nonlinearity that the
      absolute-Cartesian / MSC formulation carries EXACTLY and for free.

Output: web_data/results/n3d_param_linearity.csv
"""

from __future__ import annotations

import csv
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from n3d_core import (MU, R_EARTH, elements_of, make_geometry,  # noqa: E402
                      period, pin_blas_threads, pole_frame, propagate3,
                      roll_truth)

OUT = "/Users/pete/space_training/web_data/results/n3d_param_linearity.csv"
A_LEO = R_EARTH + 400e3
P_LEO = period(A_LEO)


def lvlh_basis(rv):
    R = rv[:3] / np.linalg.norm(rv[:3])
    Hv = np.cross(rv[:3], rv[3:])
    Wv = Hv / np.linalg.norm(Hv)
    Sv = np.cross(Wv, R)
    return np.stack([R, Sv, Wv], axis=0)          # rows: radial, along, cross


def cw_stm(n, t):
    s, c = math.sin(n * t), math.cos(n * t)
    return np.array([
        [4 - 3 * c, 0, 0, s / n, 2 * (1 - c) / n, 0],
        [6 * (s - n * t), 1, 0, -2 * (1 - c) / n, (4 * s - 3 * n * t) / n, 0],
        [0, 0, c, 0, 0, s / n],
        [3 * n * s, 0, 0, c, 2 * s, 0],
        [-6 * n * (1 - c), 0, 0, -2 * s, 4 * c - 3, 0],
        [0, 0, -n * s, 0, 0, c],
    ])


def main():
    pin_blas_threads()
    geoms = [
        ('TB_5km_di0p02', 5.0e3, 400.0, 2.0, 0.02, None),
        ('G1_10km_di0p04', 10.0e3, 2.0e3, 3.0, 0.04, None),
        ('G2_300km_di1p0', 300.0e3, 60.0e3, 2.0, 1.0, None),
        ('G3_180deg_di0p75', 13.5e6, 30.0e3, 1.5, 0.75, math.pi),
    ]
    dt = 60.0
    rows = []
    print(f"{'geometry':20s} {'rho/r':>8s} {'arc_min':>8s} "
          f"{'CW err @1 orbit':>16s} {'CW err @end':>13s} {'as % of rho':>12s}")
    for name, rho, da, orb, di_deg, dth in geoms:
        f_oop = min(A_LEO * math.sin(math.radians(di_deg)) / rho, 0.95)
        n_steps = int(round(orb * P_LEO / dt))
        sat0, tgt0, di = make_geometry(A_LEO, 0.001, da, rho, f_oop, dtheta=dth)
        t, S, G = roll_truth(sat0, tgt0, dt, n_steps)
        # Fair test: the CHIEF is the TARGET (the ROE convention), n from the
        # target's SEMI-MAJOR AXIS (not its instantaneous radius), and the
        # relative state is expressed in the target's LVLH at every epoch.
        a_t = float(elements_of(G[0][None, :])['a'][0])
        n = math.sqrt(MU / a_t ** 3)
        B = lvlh_basis(G[0])
        # CW's velocity state is the derivative IN THE ROTATING FRAME, so the
        # transport term omega x d must be removed from the inertial relative
        # velocity.  Omitting it is a 4x-the-separation error and it is exactly
        # the trap this probe had to survive (validated at 1 m separation).
        w0 = np.linalg.norm(np.cross(G[0, :3], G[0, 3:])) \
            / float(G[0, :3] @ G[0, :3])
        d0p = B @ (S[0, :3] - G[0, :3])
        d0v = B @ (S[0, 3:] - G[0, 3:]) - np.cross([0.0, 0.0, w0], d0p)
        d0 = np.concatenate([d0p, d0v])
        errs = []
        for k in range(len(t)):
            Bk = lvlh_basis(G[k])
            ex = Bk @ (S[k, :3] - G[k, :3])
            lin = cw_stm(n, t[k]) @ d0
            errs.append(float(np.linalg.norm(ex - lin[:3])))
        errs = np.array(errs)
        k1 = min(int(round(P_LEO / dt)), len(t) - 1)
        rho_m = float(np.median(np.linalg.norm(G[:, :3] - S[:, :3], axis=1)))
        r0 = float(np.linalg.norm(G[0, :3]))
        print(f"{name:20s} {rho/r0:8.4f} {t[-1]/60:8.0f} "
              f"{errs[k1]:16.4g} {errs[-1]:13.4g} "
              f"{100.0*errs[-1]/rho_m:11.4g}%")
        rows.append(dict(geometry=name, rho0_m=rho, rho_over_r=rho / r0,
                         arc_min=t[-1] / 60.0,
                         cw_err_1orbit_m=float(errs[k1]),
                         cw_err_end_m=float(errs[-1]),
                         rho_median_m=rho_m,
                         cw_err_pct_of_rho=100.0 * errs[-1] / rho_m))
    with open(OUT, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {OUT}")


if __name__ == '__main__':
    main()
