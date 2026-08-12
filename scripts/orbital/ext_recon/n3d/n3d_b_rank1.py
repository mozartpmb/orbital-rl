#!/usr/bin/env python3
"""N3D-B.4 — is the 3D angles-only covariance rank-1 along the SCALE FAMILY?

The classical bearings-only ambiguity is the ray x_t -> x_c + k(x_t - x_c): every
member of that family produces an identical bearing history under LINEAR
relative dynamics, and Keplerian nonlinearity only bends it, it does not remove
it. If that family is the dominant error direction, then the whole 6x6 CRLB
covariance is approximately

    Sigma ~ sigma_k^2 g g^T + (measurement floor),   g = x_rel(t_0) (6-vector),
    sigma_k = sigma_range / rho_0   (the relative uncertainty of the scale)

i.e. the error along ANY direction e is just sigma_k |x_rel . e|, floored at the
arc-averaged bearing precision. That would be the single most consequential fact
for the 3D training surrogate: the plane channel needs NO table of its own, only
a projection of one scalar.

This tests it component by component (range / in-plane / plane, position and
velocity) against the exact CRLB, over the tight-box and mid-field geometries.

Writes web_data/results/n3d_rank1_scale_model.csv
Run: python3 n3d_b_rank1.py [--quick]
"""

import csv
import math
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import n3d_crlb3d as K                                          # noqa: E402
import orbital_math3d as om3                                    # noqa: E402

OUT = '/Users/pete/space_training/web_data/results/n3d_rank1_scale_model.csv'
A_LEO = K.R_EARTH + 500e3
I_T = 51.6


def triad(sat0, tgt0):
    d = tgt0[:3] - sat0[:3]
    rho = float(np.linalg.norm(d))
    u = d / rho
    ht = om3.unit(np.cross(tgt0[:3], tgt0[3:]))
    w = ht - np.dot(ht, u) * u
    nw = float(np.linalg.norm(w))
    ep = w / nw if nw > 1e-9 else om3.unit(np.cross(u, [0., 0., 1.]))
    return u, np.cross(ep, u), ep, rho


def main(quick=False):
    n = math.sqrt(K.MU / A_LEO ** 3)
    P = K.period(A_LEO)
    cases = []
    for v_tol in ([0.1, 1.0, 2.0, 10.0, 50.0] if not quick else [1.0, 50.0]):
        for di in ([0.0, 0.0075225, 0.0416939, 0.25, 1.0] if not quick
                   else [0.0, 0.25]):
            for dv, ax in ([(0.0, ''), (1.0, 'nor'), (1.0, 'pro')] if not quick
                           else [(0.0, '')]):
                for sep in ([5e3, 50e3] if not quick else [5e3]):
                    for o in ([1.0, 3.0] if not quick else [1.0]):
                        cases.append((v_tol, di, dv, ax, sep, o))

    rows = []
    print(f"{'v_tol':>6} {'dI':>8} {'dv':>5}{'ax':>5} {'sep':>7} {'orb':>4} | "
          f"{'comp':>9} {'CRLB':>12} {'rank-1 model':>13} {'model/CRLB':>11}")
    for v_tol, di, dv, ax, sep, o in cases:
        da = v_tol / (1.5 * n)
        dur = o * P
        s0, t0 = K.make_pair(a_t=A_LEO, e=0.0, i_t=math.radians(I_T), nu_t=0.6,
                             sep_m=sep, da_m=da, di_rad=math.radians(di))
        burns = ((0.20 * dur, dv, ax),) if dv > 0 else ()
        T, S, G = K.roll(s0, t0, dur, burns=burns)
        r = K.crlb3d(T, S, G)[-1]
        if not np.isfinite(r['sigma_range_m']):
            continue
        u, et, ep, rho0 = triad(S[0], G[0])
        rel_p = G[0, :3] - S[0, :3]
        rel_v = G[0, 3:] - S[0, 3:]
        sig_k = r['sigma_range_m'] / rho0
        rho = np.linalg.norm(G[:, :3] - S[:, :3], axis=1)
        # Transverse (measurement-limited) floor. Bearing information adds as
        # sum_k 1/(rho_k sigma_b)^2, so the correct arc combination is the
        # INFORMATION mean of rho, not its arithmetic mean — the difference is
        # 2x once a drifting geometry makes rho grow over the arc.
        floor = math.sqrt(2.0) * K.SIGMA_BETA / math.sqrt(
            float(np.sum(1.0 / rho ** 2)))
        comps = [('pos_range', u, rel_p, r['sigma_range_m'], floor),
                 ('pos_inplane', et, rel_p, r['sigma_inplane_m'], floor),
                 ('pos_plane', ep, rel_p, r['sigma_plane_m'], floor),
                 ('vel_range', u, rel_v, r['sigma_vrange_ms'], floor * n),
                 ('vel_inplane', et, rel_v, r['sigma_vinplane_ms'], floor * n),
                 ('vel_plane', ep, rel_v, r['sigma_vplane_ms'], floor * n)]
        for name, e, x, meas, fl in comps:
            pred = math.hypot(sig_k * abs(float(x @ e)), fl)
            rows.append(dict(v_tol_ms=v_tol, di_deg=di, dv_ms=dv,
                             dv_axis=ax or 'none', sep_m=sep, orbits=o,
                             component=name, sigma_k=sig_k,
                             proj_m=abs(float(x @ e)), floor=fl,
                             crlb=meas, model=pred,
                             ratio=pred / meas if meas > 0 else float('nan')))
            print(f"{v_tol:6.1f} {di:8.4f} {dv:5.1f}{(ax or '-'):>5} "
                  f"{sep/1e3:7.1f} {o:4.1f} | {name:>9} {meas:12.5g} "
                  f"{pred:13.5g} {pred/max(meas,1e-30):11.3f}")

    with open(OUT, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    print("\n=== rank-1 scale model vs exact CRLB, per component ===")
    print(f"{'component':>12} {'n':>5} {'p05':>8} {'median':>8} {'p95':>8} "
          f"{'within 1.5x':>12}")
    for name in ('pos_range', 'pos_inplane', 'pos_plane', 'vel_range',
                 'vel_inplane', 'vel_plane'):
        v = np.array([r['ratio'] for r in rows
                      if r['component'] == name and np.isfinite(r['ratio'])])
        if v.size == 0:
            continue
        ok = float(np.mean((v > 1 / 1.5) & (v < 1.5)))
        print(f"{name:>12} {v.size:5d} {np.percentile(v,5):8.3f} "
              f"{np.median(v):8.3f} {np.percentile(v,95):8.3f} {ok:12.2%}")
    print(f"\nwrote {OUT} ({len(rows)} rows)")


if __name__ == '__main__':
    main(quick='--quick' in sys.argv)
