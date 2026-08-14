#!/usr/bin/env python3
"""Is the crlb_online acquisition surrogate biased by the J2 mismatch?

WHY ASK. `AcqSurrogate._accumulate6` builds the Fisher information of the
realized bearing arc as

    Phi_k = STM(t_k <- t_k-1) Phi_k-1 ,  F += Phi_k^T (H^T R^-1 H)_k Phi_k

and that `STM` is `nav_math3d.stm_analytic_nd` — the TWO-BODY transition. The
measurement kernel `(H^T R^-1 H)_k` is built from the REALIZED geometry, so it
is J2-correct for free; the transition that maps it back to the epoch state is
not. That is a different situation from eccentricity, which the surrogate is
immune to precisely BECAUSE e enters only through the realized geometry.

So the surrogate has a J2 exposure the e-immunity argument does not cover, and
the failure mode is silent: a biased sigma_LOS changes WHEN acquisition is
declared and WHAT covariance is handed to the filter, with nothing in the loop
to flag it. Measured here rather than argued.

Protocol: identical J2 truth trajectories, FIM accumulated two ways —
    shipped   Phi chained with the two-body STM
    J2        Phi chained with the finite-difference STM of the J2 map
sigma_LOS read out with the surrogate's own scaled-FIM estimator, at the
campaign's acquisition floor (45 min) and beyond.

Run:
    PYTHONPATH=<worktree>/pufferlib python3 scripts/orbital/extj2/j2_acq_surrogate_probe.py
"""

import argparse
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
WT = os.path.abspath(os.path.join(_HERE, '..', '..', '..'))
sys.path.insert(0, os.path.join(WT, 'pufferlib'))
sys.path.insert(0, _HERE)

import pufferlib                                                      # noqa: E402
if not os.path.abspath(pufferlib.__file__).startswith(os.path.abspath(WT) + os.sep):
    raise SystemExit('REFUSING TO RUN: pufferlib is not the worktree build')

from pufferlib.ocean.orbital_nav import nav_math as nm                # noqa: E402
from pufferlib.ocean.orbital_nav import nav_math3d as n3              # noqa: E402

import importlib.util                                                 # noqa: E402
_spec = importlib.util.spec_from_file_location(
    'probe', os.path.join(_HERE, 'j2_nav_filter_probe.py'))
_pb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_pb)

MU, DT = _pb.MU, 60.0
GATE = 0.20            # nav_acq_gate, the campaign default
MIN_SEC = 2700.0       # nav_acq_min_sec = 45 min, the campaign default


def sigma_los(FIM, u0, a0, vc0):
    """The surrogate's own estimator: scaled FIM, cond guard, LOS projection.

    Mirrors `AcqSurrogate._sigma_los_online6` — the nondimensionaliser
    D = diag(a,a,a,vc,vc,vc) and the cond > 1e14 guard on the SCALED matrix.
    """
    n = FIM.shape[0]
    d = np.empty((n, 6))
    d[:, :3] = a0[:, None]
    d[:, 3:] = vc0[:, None]
    Fs = FIM * d[:, :, None] * d[:, None, :]
    w = np.linalg.eigvalsh(Fs)
    wmin = np.abs(w).min(axis=1)
    wmax = np.abs(w).max(axis=1)
    bad = (wmin <= 0.0) | (wmax / np.maximum(wmin, 1e-300) > 1e14) \
        | ~np.isfinite(w).all(axis=1)
    Fg = np.where(bad[:, None, None], np.eye(6), Fs)
    C = np.linalg.inv(Fg + 1e-30 * np.eye(6))
    # C is the inverse of the SCALED FIM, so it is in scaled coordinates;
    # the position block returns to metres by multiplying by a0^2 (D is
    # diagonal, D_pos = a0).
    Cpos = C[:, :3, :3] * (a0 ** 2)[:, None, None]
    v = np.einsum('ni,nij,nj->n', u0, Cpos, u0)
    return np.where(bad, np.inf, np.sqrt(np.maximum(v, 0.0)))


def run(n, seed, arcs_h, verbose=False):
    rng = np.random.default_rng(seed)
    sat, tgt = _pb.sample(n, rng)
    sat_c, tgt_c = sat.cart(), tgt.cart()
    sep0 = np.linalg.norm(tgt_c[:, :3] - sat_c[:, :3], axis=1)
    a0 = np.linalg.norm(tgt_c[:, :3], axis=1)
    vc0 = np.sqrt(MU / a0)
    d0 = tgt_c[:, :3] - sat_c[:, :3]
    u0 = d0 / np.linalg.norm(d0, axis=1)[:, None]

    Phi = {k: np.tile(np.eye(6), (n, 1, 1)) for k in ('shipped', 'j2')}
    FIM = {k: np.zeros((n, 6, 6)) for k in ('shipped', 'j2')}
    sb = nm.SIGMA_BETA_RAD

    def kernel(sat_x, tgt_x):
        d = tgt_x[:, :3] - sat_x[:, :3]
        rho = np.maximum(np.linalg.norm(d, axis=1), 1.0)
        u = d / rho[:, None]
        Pp = np.eye(3) - u[:, :, None] * u[:, None, :]
        M = np.zeros((n, 6, 6))
        M[:, :3, :3] = Pp / ((sb * rho) ** 2)[:, None, None]
        return M

    # epoch observation
    K = kernel(sat_c, tgt_c)
    for k in FIM:
        FIM[k] += np.swapaxes(Phi[k], 1, 2) @ K @ Phi[k]

    ticks = int(round(max(arcs_h) * 3600.0 / DT))
    marks = {int(round(h * 3600.0 / DT)): h for h in arcs_h}
    out = {}
    # gate crossing, per arm
    crossed = {k: np.full(n, np.nan) for k in FIM}

    tgt_prev = tgt_c.copy()
    for t in range(1, ticks + 1):
        sat.step(DT, 1)          # truth is J2 in BOTH arms
        tgt.step(DT, 1)
        sat_to, tgt_to = sat.cart(), tgt.cart()
        F2, _, _ = n3.stm_analytic_nd(tgt_prev, DT)          # shipped
        Fj, _, _ = _pb.stm_fd_j2(tgt_prev, DT)               # J2-aware
        Phi['shipped'] = F2 @ Phi['shipped']
        Phi['j2'] = Fj @ Phi['j2']
        K = kernel(sat_to, tgt_to)
        for k in FIM:
            FIM[k] += np.swapaxes(Phi[k], 1, 2) @ K @ Phi[k]
        tgt_prev = tgt_to

        elapsed = t * DT
        if elapsed >= MIN_SEC:
            for k in FIM:
                s = sigma_los(FIM[k], u0, a0, vc0)
                hit = np.isnan(crossed[k]) & (s <= GATE * sep0)
                crossed[k][hit] = elapsed

        if t in marks:
            out[marks[t]] = {k: sigma_los(FIM[k], u0, a0, vc0) for k in FIM}
            out[marks[t]]['sep0'] = sep0
    return out, crossed, sep0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=128)
    ap.add_argument('--seed', type=int, default=20260814)
    ap.add_argument('--arcs', default='0.75,1.5,3,6')
    args = ap.parse_args()
    arcs = [float(x) for x in args.arcs.split(',')]

    print('=== crlb_online acquisition surrogate under J2 truth ===')
    print(f'    N={args.n}, nav60, LEO 300-800 km, i_t ~ U(30,60) deg')
    print(f'    truth is J2 in BOTH arms; only the STM that CHAINS Phi differs')
    print(f'    gate {GATE}, acquisition floor {MIN_SEC/60:.0f} min\n')

    out, crossed, sep0 = run(args.n, args.seed, arcs)

    print(f'  {"arc":>6s} {"sigma_LOS shipped":>22s} {"sigma_LOS J2-aware":>22s} '
          f'{"ratio p50":>10s} {"ratio p95":>10s} {"|log2| > 0.1":>12s}')
    for h in arcs:
        a = out[h]['shipped']; b = out[h]['j2']
        good = np.isfinite(a) & np.isfinite(b) & (b > 0)
        r = a[good] / b[good]
        print(f'  {h*60:5.0f}m {np.median(a[good]):15.1f} m '
              f'{np.median(b[good]):19.1f} m {np.median(r):10.4f} '
              f'{np.percentile(r, 95):10.4f} '
              f'{np.mean(np.abs(np.log2(r)) > 0.1):11.1%}')

    print('\n  --- does the ACQUISITION DECISION move? ---')
    cs, cj = crossed['shipped'], crossed['j2']
    both = ~np.isnan(cs) & ~np.isnan(cj)
    only_s = ~np.isnan(cs) & np.isnan(cj)
    only_j = np.isnan(cs) & ~np.isnan(cj)
    print(f'    acquired by BOTH        {both.sum():4d}/{args.n}')
    print(f'    shipped only            {only_s.sum():4d}   (would acquire on a '
          f'bound the J2-aware FIM does not support)')
    print(f'    J2-aware only           {only_j.sum():4d}')
    if both.any():
        d = (cs[both] - cj[both]) / 60.0
        print(f'    latency delta (shipped - J2-aware), minutes: '
              f'p05 {np.percentile(d, 5):+.2f}  p50 {np.median(d):+.2f}  '
              f'p95 {np.percentile(d, 95):+.2f}  max|.| {np.abs(d).max():.2f}')
        print(f'    fraction with |delta| >= 1 tick (60 s): '
              f'{np.mean(np.abs(d) >= 1.0):.1%}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
