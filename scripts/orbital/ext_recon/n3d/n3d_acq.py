#!/usr/bin/env python3
"""N3D-A §3 — acquisition geometry in 3D: is "solve the plane first" a real
shortcut, and does the analytic range prior survive the lift?

Two questions, both answered by construction/measurement rather than assertion.

(1) THE ANALYTIC RANGE PRIOR IS DIMENSION-AGNOSTIC.  |r_c + rho*u|^2 = R^2 is
    the same scalar quadratic in 3D as in 2D; the annulus becomes a spherical
    shell and the LOS stays a ray, so the closed form and its BIMODALITY carry
    over verbatim.  Verified here against a brute-force scan of the ray.

(2) THE PLANE-FIRST ROUTE.  A target on a Keplerian orbit lies in a plane
    through the centre, so for a candidate normal n:

        n . (r_c,k + rho_k u_k) = 0   =>   rho_k = -(n . r_c,k) / (n . u_k)

    i.e. a candidate PLANE hands you the whole range history in closed form,
    with no range grid and no velocity lattice.  That is the strongest possible
    version of "3D escapes the 2D degeneracy".  This probe measures its
    conditioning: perturb the true normal by a tilt eps and record the induced
    relative range error.  The predicted amplification is

        d(rho)/rho / eps  ~  r / rho_oop

    (rho_oop = the OUT-OF-PLANE component of the separation), which diverges in
    the coplanar limit -- exactly the classical D0 = L1.(L2 x L3) == 0
    singularity, recovered as a limit rather than as a special case.

Output: web_data/results/n3d_acq_plane.csv, n3d_acq_prior.csv
Run:  python3 n3d_acq.py
"""

from __future__ import annotations

import csv
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from n3d_core import (MU, R_EARTH, SIGMA_BETA_RAD, elements_of,  # noqa: E402
                      los_azel, make_geometry, period, pin_blas_threads,
                      pole_frame, roll_truth)
import n3d_filter as nf                                        # noqa: E402

OUT_PLANE = "/Users/pete/space_training/web_data/results/n3d_acq_plane.csv"
OUT_PRIOR = "/Users/pete/space_training/web_data/results/n3d_acq_prior.csv"

A_LEO = R_EARTH + 400e3
P_LEO = period(A_LEO)


def prior_check():
    """Closed-form 3D range prior vs a brute-force scan of the LOS ray."""
    rows = []
    r_min, r_max = R_EARTH + 300e3, R_EARTH + 800e3
    rng = np.random.default_rng(3)
    worst = 0.0
    for trial in range(200):
        sat = np.array([A_LEO, 0.0, 0.0, 0.0, math.sqrt(MU / A_LEO), 0.0])
        sat[:3] = sat[:3] @ np.linalg.qr(rng.normal(size=(3, 3)))[0]
        u = rng.normal(size=3)
        u /= np.linalg.norm(u)
        ivs = nf.range_prior_intervals(sat, u, r_min, r_max)
        # brute force: sample the ray and mark admissible rho
        rho = np.linspace(1.0, 4.0 * r_max, 400000)
        rr = np.linalg.norm(sat[None, :3] + rho[:, None] * u[None, :], axis=1)
        adm = (rr >= r_min) & (rr <= r_max)
        if not adm.any():
            continue
        lo_bf, hi_bf = rho[adm].min(), rho[adm].max()
        # closed form must cover the same support
        lo_cf = min(a for a, _ in ivs)
        hi_cf = max(b for _, b in ivs)
        d = max(abs(lo_cf - lo_bf), abs(hi_cf - hi_bf)) / hi_bf
        worst = max(worst, d)
        # count brute-force connected components vs analytic interval count
        idx = np.flatnonzero(adm)
        comps = 1 + int(np.sum(np.diff(idx) > 1))
        rows.append(dict(trial=trial, n_intervals_analytic=len(ivs),
                         n_components_bruteforce=comps,
                         span_ratio=hi_cf / max(lo_cf, 1.0),
                         support_rel_err=d))
    with open(OUT_PRIOR, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    agree = sum(r['n_intervals_analytic'] == r['n_components_bruteforce']
                for r in rows)
    bimodal = sum(r['n_intervals_analytic'] == 2 for r in rows)
    print(f"range prior (3D shell x ray): {len(rows)} random geometries")
    print(f"  interval-count agreement with brute force : {agree}/{len(rows)}")
    print(f"  bimodal cases                             : {bimodal}/{len(rows)}"
          f"  ({100.0*bimodal/len(rows):.0f}%)")
    print(f"  worst relative support error              : {worst:.3e}")
    print(f"  wrote {OUT_PRIOR}")


def plane_conditioning():
    """Range implied by a tilted candidate plane, vs the true range."""
    geoms = [
        ('TB_5km_di0p02', 5.0e3, 400.0, 2.0, 0.02, None),
        ('G1_10km_di0p04', 10.0e3, 2.0e3, 3.0, 0.04, None),
        ('G1_10km_di0p005', 10.0e3, 2.0e3, 3.0, 0.005, None),
        ('G2_300km_di1p0', 300.0e3, 60.0e3, 2.0, 1.0, None),
        ('G3_180deg_di0p75', 13.5e6, 30.0e3, 1.5, 0.75, math.pi),
        ('G1_10km_coplanar', 10.0e3, 2.0e3, 3.0, 0.0, None),
    ]
    dt = 60.0
    rows = []
    print(f"\n{'geometry':20s} {'di_deg':>8s} {'rho_oop':>10s} "
          f"{'A_meas':>10s} {'A_pred=r/z':>11s} {'tilt@100%':>11s} "
          f"{'tilt for 1%':>12s}")
    for name, rho, da, orb, di_deg, dth in geoms:
        f_oop = (min(A_LEO * math.sin(math.radians(di_deg)) / rho, 0.95)
                 if di_deg > 0 else 0.0)
        n_steps = int(round(orb * P_LEO / dt))
        sat0, tgt0, di = make_geometry(A_LEO, 0.001, da, rho, f_oop, dtheta=dth)
        t, S, G = roll_truth(sat0, tgt0, dt, n_steps)
        W = pole_frame(S[0])
        w = min(45, len(t))
        az, el = los_azel(S[:w], G[:w], W)
        U = np.stack([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az),
                      np.sin(el)], axis=1) @ W
        rho_true = np.linalg.norm(G[:w, :3] - S[:w, :3], axis=1)
        nhat = elements_of(G[0][None, :])['hhat'][0]
        zsep = np.abs(((G[:w, :3] - S[:w, :3]) @ W.T)[:, 2])
        # orthonormal tilt basis
        e1 = np.cross(nhat, [0.0, 0.0, 1.0])
        if np.linalg.norm(e1) < 1e-9:
            e1 = np.cross(nhat, [1.0, 0.0, 0.0])
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(nhat, e1)
        rec = []
        for eps in (1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3):
            errs = []
            for ax in (e1, e2, (e1 + e2) / math.sqrt(2.0)):
                nc = nhat * math.cos(eps) + ax * math.sin(eps)
                nc /= np.linalg.norm(nc)
                den = U @ nc
                num = -(S[:w, :3] @ nc)
                with np.errstate(all='ignore'):
                    rho_c = num / den
                bad = ~np.isfinite(rho_c) | (rho_c <= 0.0)
                e = np.abs(rho_c - rho_true) / rho_true
                e[bad] = np.inf
                errs.append(float(np.median(e)))
            rec.append((eps, float(np.median(errs))))
        # local amplification from the smallest tilt that is above fp noise
        A = float('nan')
        for eps, e in rec:
            if np.isfinite(e) and e > 1e-9:
                A = e / eps
                break
        # predicted amplification: d(rho)/rho / eps ~ (r_t + rho)/rho_oop,
        # from rho = -(n.r_c)/(n.u) with n.r_c = -rho_oop.
        r_t = np.linalg.norm(G[:w, :3], axis=1)
        A_pred = float(np.median((r_t + rho_true) / np.maximum(zsep, 1e-9)))
        tilt100 = 1.0 / A if np.isfinite(A) and A > 0 else float('nan')
        tilt1 = 0.01 / A if np.isfinite(A) and A > 0 else float('nan')
        print(f"{name:20s} {math.degrees(di):8.5f} "
              f"{np.median(zsep)/1e3:9.4f}k {A:10.4g} {A_pred:11.4g} "
              f"{tilt100:11.4g} {tilt1:12.4g}")
        for eps, e in rec:
            rows.append(dict(geometry=name, di_deg=math.degrees(di),
                             rho_med_m=float(np.median(rho_true)),
                             zsep_med_m=float(np.median(zsep)),
                             tilt_rad=eps, range_rel_err=e,
                             amplification=A, amp_pred_r_over_z=A_pred,
                             tilt_for_1pct_rad=tilt1))
    with open(OUT_PLANE, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote {OUT_PLANE}")


if __name__ == '__main__':
    pin_blas_threads()
    prior_check()
    plane_conditioning()
