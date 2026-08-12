#!/usr/bin/env python3
"""N3D-B.1 — the 3D angles-only observability map.

Sweeps separation x arc x dv(magnitude AND axis) x dI_rel x e and reports the
CRLB decomposed into (range, in-plane, plane) at the epoch LOS triad.

The separation/arc/dv axes deliberately MIRROR `ext_bo_observability.py` (the
2D map the shipped training surrogate interpolates), so the 2D-vs-3D question
— "can the existing table be reused with a plane bolt-on, or is a full 3D table
needed?" — is answered cell-for-cell rather than by analogy.

Writes web_data/results/n3d_obs_map.csv  and  n3d_obs_map_wide.csv
Run: python3 n3d_b_map.py [--quick]
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

OUT = '/Users/pete/space_training/web_data/results/n3d_obs_map.csv'
OUT_W = '/Users/pete/space_training/web_data/results/n3d_obs_map_wide.csv'

FIELDS = ['block', 'a_km', 'e', 'i_t_deg', 'sep_km', 'da_m', 'di_deg',
          'dv_ms', 'dv_axis', 'orbits', 'n_obs', 'rho0_km', 'rho_mean_km',
          'oop_amp_m', 'sigma_range_m', 'sigma_inplane_m', 'sigma_plane_m',
          'sigma_pos_m', 'sigma_vrange_ms', 'sigma_vinplane_ms',
          'sigma_vplane_ms', 'rel_sigma_range', 'rel_sigma_plane',
          'plane_over_range', 'plane_over_inplane', 'fim_cond']


def cell(a, e, i_t, sep, da, di_deg, dv, axis, arcs_orbits, burn_frac=0.35,
         nu_t=0.6):
    """One roll PER ARC, with the burn at `burn_frac` of THAT arc.

    (Reusing one long roll for every arc checkpoint is tempting and wrong: it
    places the burn at a fixed fraction of the LONGEST arc, so the short-arc
    rows silently become drift-only. This mirrors `ext_bo_observability.build`,
    which also re-rolls per arc — which is what makes the 2D-vs-3D comparison
    cell-for-cell rather than approximate.)
    """
    P = K.period(a)
    rows = []
    for o in arcs_orbits:
        dur = o * P
        rv_s, rv_t = K.make_pair(a_t=a, e=e, i_t=math.radians(i_t), nu_t=nu_t,
                                 sep_m=sep, da_m=da,
                                 di_rad=math.radians(di_deg))
        burns = ((burn_frac * dur, dv, axis),) if dv > 0.0 else ()
        T, S, G = K.roll(rv_s, rv_t, dur, burns=burns)
        r = K.crlb3d(T, S, G)[-1]
        d = G[:, :3] - S[:, :3]
        rho = np.linalg.norm(d, axis=1)
        ht = np.cross(G[0, :3], G[0, 3:]); ht /= np.linalg.norm(ht)
        rows.append(dict(
            a_km=a / 1e3, e=e, i_t_deg=i_t, sep_km=sep / 1e3, da_m=da,
            di_deg=di_deg, dv_ms=dv, dv_axis=axis or 'none', orbits=o,
            n_obs=r['n_obs'], rho0_km=r['rho0_m'] / 1e3,
            rho_mean_km=float(rho.mean()) / 1e3,
            oop_amp_m=float(np.abs(d @ ht).max()),
            sigma_range_m=r['sigma_range_m'],
            sigma_inplane_m=r['sigma_inplane_m'],
            sigma_plane_m=r['sigma_plane_m'], sigma_pos_m=r['sigma_pos_m'],
            sigma_vrange_ms=r['sigma_vrange_ms'],
            sigma_vinplane_ms=r['sigma_vinplane_ms'],
            sigma_vplane_ms=r['sigma_vplane_ms'],
            rel_sigma_range=r['sigma_range_m'] / max(r['rho0_m'], 1.0),
            rel_sigma_plane=r['sigma_plane_m'] / max(r['rho0_m'], 1.0),
            plane_over_range=r['sigma_plane_m'] / max(r['sigma_range_m'], 1e-12),
            plane_over_inplane=(r['sigma_plane_m']
                                / max(r['sigma_inplane_m'], 1e-12)),
            fim_cond=r['fim_cond']))
    return rows


def main(quick=False):
    t0 = time.time()
    a = K.R_EARTH + 400e3
    seps = [5e3, 10e3, 50e3, 200e3, 1e6, 5e6, 1.3e7]
    dis = [0.0, 0.01, 0.05, 0.25, 1.0]
    dvs = [(0.0, ''), (1.0, 'pro'), (1.0, 'nor'), (5.0, 'pro'), (5.0, 'nor'),
           (25.0, 'pro'), (25.0, 'nor')]
    arcs = [0.25, 0.5, 1.0, 2.0, 3.0]
    if quick:
        seps = [5e3, 200e3, 1.3e7]; dis = [0.0, 0.25]; arcs = [0.5, 1.0]
        dvs = [(0.0, ''), (1.0, 'nor'), (5.0, 'pro')]

    rows = []
    print(f"{'sep_km':>9} {'dI':>6} {'dv':>6} {'ax':>4} {'orb':>5} | "
          f"{'sig_rng':>11} {'sig_inpl':>10} {'sig_plane':>10} "
          f"{'pl/rng':>8} {'rng/rho':>9}")
    for sep in seps:
        for di in dis:
            for dv, ax in dvs:
                for r in cell(a, 0.0, 51.6, sep, 0.0, di, dv, ax, arcs):
                    r['block'] = 'leo'
                    rows.append(r)
                    if r['orbits'] in (1.0, arcs[-1]):
                        print(f"{sep/1e3:9.1f} {di:6.3f} {dv:6.1f} "
                              f"{(ax or '-'):>4} {r['orbits']:5.2f} | "
                              f"{r['sigma_range_m']:11.2f} "
                              f"{r['sigma_inplane_m']:10.3f} "
                              f"{r['sigma_plane_m']:10.3f} "
                              f"{r['plane_over_range']:8.4f} "
                              f"{r['rel_sigma_range']:9.2e}")
        print()
    _write(OUT, rows)

    # ── wide envelope (V5 ladder geometry: a to 8000 km alt, e <= 0.30) ──
    wrows = []
    print("\nwide envelope (a = 12 000 km)")
    warcs = [0.5, 1.0, 1.5]
    wcfg = [(0.0,), (0.30,)] if not quick else [(0.30,)]
    for (ew,) in wcfg:
        for sep in ([10e3, 1e6, 1.2e7] if not quick else [10e3]):
            for di in ([0.0, 0.1, 1.0] if not quick else [1.0]):
                for dv, ax in [(0.0, ''), (25.0, 'pro'), (25.0, 'nor')]:
                    for r in cell(12.0e6, ew, 51.6, sep, 0.0, di, dv, ax,
                                  warcs):
                        r['block'] = 'wide'
                        wrows.append(r)
                        if r['orbits'] == 1.5:
                            print(f"  e={ew:.2f} sep={sep/1e3:8.1f} "
                                  f"dI={di:5.2f} dv={dv:5.1f}{ax:>4} -> "
                                  f"rng {r['sigma_range_m']:11.2f} m "
                                  f"inpl {r['sigma_inplane_m']:9.2f} "
                                  f"plane {r['sigma_plane_m']:9.2f} "
                                  f"(pl/rng {r['plane_over_range']:.4f})")
    _write(OUT_W, wrows)
    print(f"\n{len(rows)} + {len(wrows)} rows in {time.time()-t0:.1f} s")


def _write(path, rows):
    with open(path, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, '') for k in FIELDS})
    print(f"wrote {path} ({len(rows)} rows)")


if __name__ == '__main__':
    main(quick='--quick' in sys.argv)
