#!/usr/bin/env python3
"""N3D-B.2 — dual control at the TIGHT BOX, in 3D.

NAV-F's central result: 2D angles-only observability is governed by the
semi-major-axis difference da, and it collapses exactly as the guidance
objective succeeds (nulling relative velocity to v_tol leaves da = v_tol/(1.5n),
and drift-only sigma_range then grows to 27-52% of the position box). The
question this script answers is whether 3D changes that verdict:

  Q2a  Does a residual relative inclination keep range information alive at the
       terminal, via out-of-plane parallax? (Sweep dI at fixed da, drift-only.)
  Q2b  Does the PLANE channel itself stay observable as guidance succeeds, or
       does it collapse the way range does?
  Q2c  Is the normal burn — which 3D guidance must make anyway — as good an
       observability maneuver as the in-plane ones?
  Q2d  How much of the range information comes from the SECOND angle? (Compare
       the full 2-angle CRLB against in-plane-angle-only and out-of-plane-only.)

The dI grid is anchored on the plane tolerances the success box itself implies
(`web_data/results/ext_3d_box_plane_tol.csv`, LEO 500 km):
    5 km / 1 m/s box  ->  dI <= 0.0417 deg (position) / 0.0075 deg (velocity)

Writes web_data/results/n3d_tightbox_dualcontrol.csv
                        n3d_tightbox_anglesplit.csv
Run: python3 n3d_b_tightbox.py [--quick]
"""

import csv
import math
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import n3d_crlb3d as K                                          # noqa: E402

OUT = '/Users/pete/space_training/web_data/results/n3d_tightbox_dualcontrol.csv'
OUT_A = '/Users/pete/space_training/web_data/results/n3d_tightbox_anglesplit.csv'

A_LEO = K.R_EARTH + 500e3
I_T = 51.6
RHO_BOX_M = 5.0e3            # TB5 position tolerance
V_BOX_MS = 1.0               # TB5 velocity tolerance
DI_POS_TOL_DEG = 0.04169392  # ext_3d_box_plane_tol.csv, LEO 500 km, 5 km/1 m/s
DI_VEL_TOL_DEG = 0.00752253

FIELDS = ['da_m', 'v_tol_implied_ms', 'di_deg', 'dv_ms', 'dv_axis', 'orbits',
          'n_obs', 'rho0_km', 'rho_mean_km', 'rho_max_km', 'oop_amp_m',
          'sigma_range_m', 'sigma_inplane_m', 'sigma_plane_m',
          'sigma_vrange_ms', 'sigma_vinplane_ms', 'sigma_vplane_ms',
          'range_frac_of_box', 'vrange_frac_of_box', 'plane_frac_of_planebox',
          'vplane_frac_of_planebox', 'gain_vs_drift', 'fim_cond']

AFIELDS = ['da_m', 'di_deg', 'dv_ms', 'dv_axis', 'orbits', 'angles',
           'sigma_range_m', 'sigma_plane_m', 'sigma_inplane_m', 'fim_cond']


def _sep_for_rho(rho_target=RHO_BOX_M):
    return rho_target


def run_cell(da, di_deg, dv, axis, arcs, sep=RHO_BOX_M, angles='both',
             burn_frac=0.20, nu_t=0.6):
    """One roll PER ARC, burn at `burn_frac` of THAT arc (NAV-F §2.5 measured
    the information peak at ~20% into the arc; a fixed fraction of the LONGEST
    arc would silently move the burn, or drop it out of the short arcs)."""
    P = K.period(A_LEO)
    recs, rhos, oops = [], [], []
    for o in arcs:
        dur = o * P
        rv_s, rv_t = K.make_pair(a_t=A_LEO, e=0.0, i_t=math.radians(I_T),
                                 nu_t=nu_t, sep_m=sep, da_m=da,
                                 di_rad=math.radians(di_deg))
        burns = ((burn_frac * dur, dv, axis),) if dv > 0.0 else ()
        T, S, G = K.roll(rv_s, rv_t, dur, burns=burns)
        recs.append(K.crlb3d(T, S, G, angles=angles)[-1])
        d = G[:, :3] - S[:, :3]
        rhos.append(np.linalg.norm(d, axis=1))
        ht = np.cross(G[0, :3], G[0, 3:]); ht /= np.linalg.norm(ht)
        oops.append(np.abs(d @ ht))
    return recs, rhos, oops


def main(quick=False):
    t0 = time.time()
    n = math.sqrt(K.MU / A_LEO ** 3)
    # NAV-F's map: nulling relative velocity to v_tol leaves da = v_tol/(1.5 n)
    v_tols = [0.0, 0.1, 0.5, 1.0, 2.0, 10.0, 50.0]
    das = [v / (1.5 * n) for v in v_tols]
    dis = [0.0, DI_VEL_TOL_DEG, DI_POS_TOL_DEG, 0.1, 0.25, 1.0]
    burns = [(0.0, ''), (1.0, 'pro'), (1.0, 'rad'), (1.0, 'nor'),
             (1.0, 'perp_los')]
    arcs = [1.0, 3.0]
    if quick:
        v_tols = [0.0, 1.0, 50.0]; das = [v / (1.5 * n) for v in v_tols]
        dis = [0.0, DI_POS_TOL_DEG, 1.0]
        burns = [(0.0, ''), (1.0, 'nor')]
        arcs = [1.0]

    rows = []
    drift = {}
    print(f"LEO 500 km, rho0 = 5 km, i_t = {I_T} deg, sigma_beta = 1 mrad, "
          f"60 s cadence")
    print(f"box: {RHO_BOX_M/1e3:.0f} km / {V_BOX_MS:.0f} m/s   "
          f"plane box: {DI_POS_TOL_DEG:.4f} deg pos "
          f"({A_LEO*math.radians(DI_POS_TOL_DEG)/1e3:.2f} km) / "
          f"{DI_VEL_TOL_DEG:.4f} deg vel\n")
    hdr = (f"{'v_tol':>6} {'da_m':>9} {'dI_deg':>8} {'dv':>5} {'axis':>9} "
           f"{'orb':>4} | {'sig_rng':>11} {'/box':>7} {'sig_vrng':>9} "
           f"{'/vbox':>7} {'sig_plane':>9} {'/plbox':>7} {'gain':>10}")
    print(hdr)
    for v_tol, da in zip(v_tols, das):
        for di in dis:
            for dv, ax in burns:
                recs, rhos, oops = run_cell(da, di, dv, ax, arcs)
                for o, r, rho, oop in zip(arcs, recs, rhos, oops):
                    pl_box_m = A_LEO * math.radians(DI_POS_TOL_DEG)
                    pl_box_v = math.sqrt(K.MU / A_LEO) * math.radians(
                        DI_VEL_TOL_DEG)
                    key = (round(da, 3), di, o)
                    if dv == 0.0:
                        drift[key] = r['sigma_range_m']
                    base = drift.get(key, float('nan'))
                    gain = ((base / r['sigma_range_m']) ** 2
                            if np.isfinite(base) and r['sigma_range_m'] > 0
                            else float('inf'))
                    row = dict(
                        da_m=da, v_tol_implied_ms=v_tol, di_deg=di, dv_ms=dv,
                        dv_axis=ax or 'none', orbits=o, n_obs=r['n_obs'],
                        rho0_km=r['rho0_m'] / 1e3,
                        rho_mean_km=float(rho.mean()) / 1e3,
                        rho_max_km=float(rho.max()) / 1e3,
                        oop_amp_m=float(oop.max()),
                        sigma_range_m=r['sigma_range_m'],
                        sigma_inplane_m=r['sigma_inplane_m'],
                        sigma_plane_m=r['sigma_plane_m'],
                        sigma_vrange_ms=r['sigma_vrange_ms'],
                        sigma_vinplane_ms=r['sigma_vinplane_ms'],
                        sigma_vplane_ms=r['sigma_vplane_ms'],
                        range_frac_of_box=r['sigma_range_m'] / RHO_BOX_M,
                        vrange_frac_of_box=r['sigma_vrange_ms'] / V_BOX_MS,
                        plane_frac_of_planebox=r['sigma_plane_m'] / pl_box_m,
                        vplane_frac_of_planebox=r['sigma_vplane_ms'] / pl_box_v,
                        gain_vs_drift=gain, fim_cond=r['fim_cond'])
                    rows.append(row)
                    print(f"{v_tol:6.1f} {da:9.1f} {di:8.4f} {dv:5.1f} "
                          f"{(ax or '-'):>9} {o:4.1f} | "
                          f"{r['sigma_range_m']:11.2f} "
                          f"{row['range_frac_of_box']:7.3f} "
                          f"{r['sigma_vrange_ms']:9.4f} "
                          f"{row['vrange_frac_of_box']:7.3f} "
                          f"{r['sigma_plane_m']:9.3f} "
                          f"{row['plane_frac_of_planebox']:7.4f} "
                          f"{gain:10.3g}")
        print()
    _write(OUT, rows, FIELDS)

    # ── Q2d: marginal value of the second (out-of-plane) angle ──────────────
    arows = []
    print("\nangle-split: what does the OUT-OF-PLANE angle buy for RANGE?")
    print(f"{'da_m':>9} {'dI_deg':>8} {'dv':>5} {'axis':>5} {'angles':>9} | "
          f"{'sig_range':>12} {'sig_plane':>10}")
    split_das = [das[0], das[3] if len(das) > 3 else das[-1], das[-1]]
    split_dis = [0.0, DI_POS_TOL_DEG, 0.25, 1.0] if not quick else [0.0, 1.0]
    for da in split_das:
        for di in split_dis:
            for dv, ax in ([(0.0, ''), (1.0, 'nor')] if not quick
                           else [(0.0, '')]):
                for angles in ('both', 'inplane', 'oop'):
                    recs, rhos, oops = run_cell(da, di, dv, ax, [1.0],
                                                angles=angles)
                    r = recs[-1]
                    arows.append(dict(da_m=da, di_deg=di, dv_ms=dv,
                                      dv_axis=ax or 'none', orbits=1.0,
                                      angles=angles,
                                      sigma_range_m=r['sigma_range_m'],
                                      sigma_plane_m=r['sigma_plane_m'],
                                      sigma_inplane_m=r['sigma_inplane_m'],
                                      fim_cond=r['fim_cond']))
                    print(f"{da:9.1f} {di:8.4f} {dv:5.1f} {(ax or '-'):>5} "
                          f"{angles:>9} | {r['sigma_range_m']:12.2f} "
                          f"{r['sigma_plane_m']:10.3f}")
                print()
    _write(OUT_A, arows, AFIELDS)
    print(f"\n{len(rows)} + {len(arows)} rows in {time.time()-t0:.1f} s")


def _write(path, rows, fields):
    with open(path, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, '') for k in fields})
    print(f"wrote {path} ({len(rows)} rows)")


if __name__ == '__main__':
    main(quick='--quick' in sys.argv)
