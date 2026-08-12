#!/usr/bin/env python3
"""N3D-A §2 — 3D angles-only observability map (filter-independent CRLB).

Question this answers, from NAV-G §5 caveat:
    "2D coplanar sits exactly on the classical IOD coplanar singularity
     D0 = L1.(L2 x L3) == 0 ... porting to ext-3d should IMPROVE observability
     -- the LOS rays through a known observer determine the orbit plane
     strongly -- but that needs its own probe."

Method.  FIM on the TARGET'S EPOCH 6-STATE from the whole arc:

    F = sum_k Phi(t_k,t_0)^T H_k^T R^-1 H_k Phi(t_k,t_0)

with H_k = d(az,el)/dx (2x6, velocity columns identically zero), Phi chained
from the same central-difference STM the filters use, and R = diag(sigma_b^2 /
cos^2 el, sigma_b^2) so the angular noise is ISOTROPIC ON THE SPHERE (the
correct model for a focal-plane sensor; it also makes the az row degrade
gracefully toward the pole instead of pretending to carry information).

C = F^-1 is then projected onto the directions the design cares about, each
Jacobian taken by central differences on the exact element map:

    range     u0^T C_rr u0                         (epoch LOS direction)
    plane     sqrt(tr(J_h C J_h^T)), J_h = d h_hat/dx     -> radians of tilt
    i, RAAN   marginal sigmas of the two plane angles
    a, e      in-plane shape
    u         argument of latitude (the in-track phase = the weak 2D direction)
    vel       sqrt(tr(C_vv))

Controlled variable: `f_oop`, the fraction of the epoch separation carried
OUT of the chaser's orbit plane.  f_oop = 0 is the 2D-degenerate coplanar
control; f_oop -> 1 is a purely out-of-plane geometry.  Holding total
separation fixed while moving it between in-plane and out-of-plane is the only
honest way to ask "does 3D help", because a rendezvous at separation rho simply
cannot carry a relative inclination larger than ~rho/r.

Outputs: web_data/results/n3d_crlb.csv, n3d_crlb_families.csv
Run:  python3 n3d_crlb.py [--quick]
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from n3d_core import (MU, R_EARTH, SIGMA_BETA_RAD, SIGMA_RHO_M,  # noqa: E402
                      azel_jac, elements_of, los_azel, make_geometry,
                      period, pin_blas_threads, pole_frame, propagate3,
                      roll_truth, stm_fd3, wrap_pi)

OUT = "/Users/pete/space_training/web_data/results/n3d_crlb.csv"
OUT_FAM = "/Users/pete/space_training/web_data/results/n3d_crlb_families.csv"


# ── derived-quantity Jacobians (central differences on the exact maps) ──────
def _derived(x):
    el = elements_of(np.asarray(x, dtype=np.float64).reshape(1, 6))
    return np.array([el['a'][0], el['e'][0], el['inc'][0], el['raan'][0],
                     el['u'][0], el['hhat'][0, 0], el['hhat'][0, 1],
                     el['hhat'][0, 2]])


def derived_jac(x):
    """d[a, e, i, RAAN, u, h_hat(3)] / d(epoch 6-state).  (8,6)."""
    x = np.asarray(x, dtype=np.float64).reshape(6)
    J = np.zeros((8, 6))
    h = (1.0, 1.0, 1.0, 1e-3, 1e-3, 1e-3)
    for j in range(6):
        xp = x.copy(); xp[j] += h[j]
        xm = x.copy(); xm[j] -= h[j]
        d = _derived(xp) - _derived(xm)
        d[2:5] = wrap_pi(d[2:5])
        J[:, j] = d / (2.0 * h[j])
    return J


def crlb(times, S, G, sigma_beta=SIGMA_BETA_RAD, with_range=False,
         sigma_rho=SIGMA_RHO_M, W=None):
    """Whole-arc Fisher information on the target epoch state -> projections."""
    n = len(times)
    if W is None:
        W = pole_frame(S[0])
    dt = times[1] - times[0]
    Fst, ok, _ = stm_fd3(G[:-1], dt, iters=12, warm_iters=6)     # (n-1,6,6)

    az, el = los_azel(S, G, W)
    Hall = azel_jac(S, G, W)                                      # (n,2,6)
    Phi = np.eye(6)
    F = np.zeros((6, 6))
    for k in range(n):
        if k > 0:
            Phi = Fst[k - 1] @ Phi
        H = Hall[k] @ Phi
        c2 = max(math.cos(el[k]) ** 2, 1e-12)
        Rinv = np.diag([c2 / sigma_beta ** 2, 1.0 / sigma_beta ** 2])
        F += H.T @ Rinv @ H
        if with_range:
            d = G[k, :3] - S[k, :3]
            rho = np.linalg.norm(d)
            Hr = np.zeros((1, 6))
            Hr[0, :3] = d / rho
            Hr = Hr @ Phi
            F += (Hr.T @ Hr) / sigma_rho ** 2

    w = np.linalg.eigvalsh(F)
    cond = float(abs(w).max() / max(abs(w).min(), 1e-300))
    out = dict(cond=cond)
    if (not np.all(np.isfinite(w))) or cond > 1e17 or abs(w).min() <= 0.0:
        for kk in ('sig_range', 'sig_pos', 'sig_vel', 'sig_plane', 'sig_i',
                   'sig_raan', 'sig_a', 'sig_e', 'sig_u', 'sig_scale_rho'):
            out[kk] = float('inf')
        out['corr_range_plane'] = float('nan')
        return out
    C = np.linalg.inv(F)
    d0 = G[0, :3] - S[0, :3]
    rho0 = float(np.linalg.norm(d0))
    u0 = d0 / rho0
    out['sig_range'] = math.sqrt(max(u0 @ C[:3, :3] @ u0, 0.0))
    # CONDITIONAL (profile) range sigma along the classical scale family
    # x_t(k) = x_c + k (x_t - x_c): g = dx/dk = relative 6-state.
    g = G[0] - S[0]
    gFg = float(g @ F @ g)
    out['sig_scale_rho'] = rho0 / math.sqrt(gFg) if gFg > 0 else float('inf')
    out['sig_pos'] = math.sqrt(max(np.trace(C[:3, :3]), 0.0))
    out['sig_vel'] = math.sqrt(max(np.trace(C[3:, 3:]), 0.0))
    J = derived_jac(G[0])
    Cd = J @ C @ J.T
    out['sig_a'] = math.sqrt(max(Cd[0, 0], 0.0))
    out['sig_e'] = math.sqrt(max(Cd[1, 1], 0.0))
    out['sig_i'] = math.sqrt(max(Cd[2, 2], 0.0))
    out['sig_raan'] = math.sqrt(max(Cd[3, 3], 0.0))
    out['sig_u'] = math.sqrt(max(Cd[4, 4], 0.0))
    out['sig_plane'] = math.sqrt(max(np.trace(Cd[5:8, 5:8]), 0.0))
    # Is the plane an INDEPENDENT observable, or just another projection of the
    # range ambiguity?  Correlate the epoch-LOS range error against each
    # component of the orbit-normal error.
    M = np.vstack([np.concatenate([u0, np.zeros(3)])[None, :], J[5:8, :]])
    Cm = M @ C @ M.T
    dg = np.sqrt(np.maximum(np.diag(Cm), 1e-300))
    cc = Cm / np.outer(dg, dg)
    out['corr_range_plane'] = float(np.max(np.abs(cc[0, 1:])))
    return out


# ── configurations ──────────────────────────────────────────────────────────
A_LEO = R_EARTH + 400e3
P_LEO = period(A_LEO)


def families(quick=False):
    """Design families.  `di_deg` is swept DIRECTLY (not f_oop), because the
    achievable relative inclination is capped by the separation itself:
    a rendezvous holding at rho cannot carry di > asin(rho/r).  The cap is
    enforced per family, and infeasible cells are skipped rather than silently
    turned into a different geometry (the failure mode that produced ext-3d's
    de_max bug)."""
    f = [
        # terminal box (TB5 geometry, NAV-G G6): 5 km, da = 400 m
        dict(name='TB_5km', rho=5.0e3, da=400.0, orbits=2.0, dtheta=None,
             di=[0.0, 0.002, 0.005, 0.01, 0.02, 0.04]),
        # NAV-G G1: 10 km co-orbital drift, da = 2 km
        dict(name='G1_10km', rho=10.0e3, da=2.0e3, orbits=3.0, dtheta=None,
             di=[0.0, 0.002, 0.005, 0.01, 0.02, 0.04, 0.08]),
        # NAV-G G2 mid-field: 300 km, da = 60 km
        dict(name='G2_300km', rho=300.0e3, da=60.0e3, orbits=2.0, dtheta=None,
             di=[0.0, 0.01, 0.05, 0.1, 0.3, 0.75, 1.0, 2.0]),
        # NAV-G G3 headline 180-degree phase gap
        dict(name='G3_180deg', rho=13.5e6, da=30.0e3, orbits=1.5,
             dtheta=math.pi, di=[0.0, 0.05, 0.1, 0.3, 0.75, 1.0]),
    ]
    if quick:
        f = [f[1], f[3]]
        for x in f:
            x['di'] = x['di'][::2]
    return f


def run(quick=False):
    dt = 60.0
    rows = []
    for fam in families(quick):
        n_steps = int(round(fam['orbits'] * P_LEO / dt))
        for di_deg in fam['di']:
            f_oop = min(A_LEO * math.sin(math.radians(di_deg)) / fam['rho'],
                        0.999)
            if di_deg > 0.0 and f_oop >= 0.999:
                print(f"  skip {fam['name']} di={di_deg} — infeasible "
                      f"(needs f_oop>1 at rho={fam['rho']/1e3:.0f} km)")
                continue
            for dv in [0.0, 1.0]:
                sat0, tgt0, di = make_geometry(
                    A_LEO, 0.001, fam['da'], fam['rho'], f_oop,
                    dtheta=fam['dtheta'])
                burns = ((int(0.20 * n_steps), dv),) if dv else ()
                t, S, G = roll_truth(sat0, tgt0, dt, n_steps, burns)
                rho = np.linalg.norm(G[:, :3] - S[:, :3], axis=1)
                Wf = pole_frame(S[0])
                _, elv = los_azel(S, G, Wf)
                # out-of-plane separation history (metres)
                zsep = np.abs(((G[:, :3] - S[:, :3]) @ Wf.T)[:, 2])
                t0 = time.time()
                bo = crlb(t, S, G, W=Wf)
                rb = crlb(t, S, G, with_range=True, W=Wf)
                r0 = float(np.linalg.norm(G[0, :3]))
                rows.append(dict(
                    family=fam['name'], rho_epoch_m=float(rho[0]),
                    rho_mean_m=float(rho.mean()), rho_over_r=float(rho[0] / r0),
                    f_oop=f_oop, di_deg=math.degrees(di), dv_ms=dv,
                    arc_min=float(t[-1] / 60.0), n_obs=len(t),
                    zsep_max_m=float(zsep.max()), zsep_epoch_m=float(zsep[0]),
                    el_max_deg=float(np.degrees(np.abs(elv).max())),
                    bo_sig_range_m=bo['sig_range'],
                    bo_sig_range_frac=bo['sig_range'] / float(rho[0]),
                    bo_sig_vel_ms=bo['sig_vel'],
                    bo_sig_plane_rad=bo['sig_plane'],
                    bo_sig_plane_deg=math.degrees(bo['sig_plane'])
                    if np.isfinite(bo['sig_plane']) else float('inf'),
                    bo_sig_i_deg=math.degrees(bo['sig_i'])
                    if np.isfinite(bo['sig_i']) else float('inf'),
                    bo_sig_raan_deg=math.degrees(bo['sig_raan'])
                    if np.isfinite(bo['sig_raan']) else float('inf'),
                    bo_sig_a_m=bo['sig_a'], bo_sig_e=bo['sig_e'],
                    bo_sig_u_deg=math.degrees(bo['sig_u'])
                    if np.isfinite(bo['sig_u']) else float('inf'),
                    bo_sig_scale_rho_m=bo['sig_scale_rho'],
                    bo_corr_range_plane=bo['corr_range_plane'],
                    rb_corr_range_plane=rb['corr_range_plane'],
                    bo_cond=bo['cond'],
                    rb_sig_range_m=rb['sig_range'], rb_sig_vel_ms=rb['sig_vel'],
                    rb_sig_plane_deg=math.degrees(rb['sig_plane'])
                    if np.isfinite(rb['sig_plane']) else float('inf'),
                    secs=round(time.time() - t0, 2)))
                r = rows[-1]
                print(f"{fam['name']:10s} di={r['di_deg']:8.5f}d dv={dv:<4} "
                      f"zmax={r['zsep_max_m']/1e3:8.3f}km "
                      f"| BO rho={r['bo_sig_range_m']:11.4g}m "
                      f"({r['bo_sig_range_frac']*100:8.3g}%) "
                      f"v={r['bo_sig_vel_ms']:9.4g} "
                      f"plane={r['bo_sig_plane_deg']:9.4g}d "
                      f"(RB {r['rb_sig_plane_deg']:9.4g}d) "
                      f"corr={r['bo_corr_range_plane']:6.4f}", flush=True)
    with open(OUT, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {OUT} ({len(rows)} rows)")
    return rows


def xcheck_2d():
    """Independent cross-check: at di = 0 the 3D CRLB pipeline must reproduce
    the 2D pipeline of `ext_bo_filter.crlb_range_sigma` (different code, 4-state
    STM, no elevation row).  Agreement here is what licenses every 3D number."""
    sys.path.insert(0, "/Users/pete/space_training/scripts/orbital/ext_recon")
    sys.path.insert(0, "/Users/pete/space_training/scripts/orbital/nav")
    import ext_bo_filter as bo2                                   # noqa: E402
    dt = 60.0
    print(f"\n{'geometry':16s} {'arc_min':>8s} {'3D sig_rho':>12s} "
          f"{'2D sig_rho':>12s} {'ratio':>7s}")
    out = []
    for fam in families():
        n_steps = int(round(fam['orbits'] * P_LEO / dt))
        sat0, tgt0, _ = make_geometry(A_LEO, 0.001, fam['da'], fam['rho'], 0.0,
                                      dtheta=fam['dtheta'])
        t, S, G = roll_truth(sat0, tgt0, dt, n_steps)
        c3 = crlb(t, S, G)
        S2 = [(s[0], s[1], s[3], s[4]) for s in S]
        G2 = [(g[0], g[1], g[3], g[4]) for g in G]
        s2, _, _ = bo2.crlb_range_sigma(list(t), S2, G2, SIGMA_BETA_RAD)
        ratio = c3['sig_range'] / s2 if np.isfinite(s2) and s2 > 0 else float('nan')
        print(f"{fam['name']:16s} {t[-1]/60.0:8.1f} {c3['sig_range']:12.4g} "
              f"{s2:12.4g} {ratio:7.4f}")
        out.append((fam['name'], c3['sig_range'], s2, ratio))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--xcheck', action='store_true')
    a = ap.parse_args()
    pin_blas_threads()
    if a.xcheck:
        xcheck_2d()
        return
    run(a.quick)


if __name__ == '__main__':
    main()
