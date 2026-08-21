#!/usr/bin/env python3
"""W1xnav probe: what actually happens to the filter and the acquisition
surrogate across a drift-and-wait day-warp.

FIRST, A FRAMING CORRECTION THIS PROBE EXISTS TO VERIFY.
The task was posed as "~24 h BLIND windows". Under the shipped configuration
that is not what a day-warp is. `_tick3` runs predict -> update ->
acq.accumulate -> acq.ready on EVERY tick, and `nav_max_ticks=120` turns a
tau=1440 action into n = min(1440, 120) = 120 ticks at dt = 1440*60/120 = 720 s.
So a day-warp is a SPARSE-CADENCE window (120 bearings at 12-minute spacing),
not a blind one. At nav_max_ticks=0 it is 1440 ticks at 60 s, i.e. fully
sampled. Nothing in the shipped stack produces a genuinely unmeasured 24 h arc.
Stage `cadence` measures the realised tick count and spacing so this is a
measurement rather than a reading of the source.

WHAT THE REAL RISK IS, GIVEN THAT. Not information starvation but LINEARISATION
and COVARIANCE CONSISTENCY at dt = 720 s per predict step, twelve times the
60 s the filter was tuned at. The J2 filter probe already measured the shipped
FIXED arm (truth J2 / filt J2) at nav60 cadence:
    1 h  NEES  1.27   (band [0.206, 2.408])
    6 h  NEES  1.35
   24 h  NEES 10.91   71.5% of runs out of band
so the covariance is ALREADY ~4.5x outside the top of the consistency band at
24 h with 60 s ticks. Stage `warp` asks the question that matters for W1: what
does it do at 720 s ticks, across an actual day-warp action, and does the gain
recover afterwards.

Stages
    cadence  realised ticks/dt per action row (the framing check)
    warp     NEES + position error + gain health across a day-warp
    arc      NEES vs arc length {6, 12, 24, 48} h at the warp cadence

Light by construction: single process, small N, no training.
"""
import argparse
import os
import sys

import numpy as np

WT = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
sys.path.insert(0, os.path.join(WT, 'pufferlib'))
sys.path.insert(0, os.path.join(WT, 'scripts', 'orbital', 'extj2'))

import pufferlib.ocean.orbital_nav.nav_math3d as n3            # noqa: E402
from pufferlib.ocean.orbital_nav.orbital_nav import OrbitalNav  # noqa: E402
import pufferlib.ocean.orbital_nav.nav_math as nm               # noqa: E402
import t11_cells as T                                           # noqa: E402

BAND_LO, BAND_HI = n3.NEES6_LO, n3.NEES6_HI
DAY_WARP = 30


def w1_env(n, max_ticks=120, seed=11):
    """The W1_driftwait cell exactly as the mixture ships it, nav on."""
    c = dict(T.CELLS)['W1_driftwait']
    kw = T.nav_env_kwargs(
        num_envs=n, nav_mode='bearings_only', cell_mixture_mode=0,
        j2_mode=int(c['j2']), nav_j2_mode=int(c['j2']),
        episode_cap_steps=int(c['cap']), rendezvous_radius_m=c['box_r'],
        rel_vel_tol_ms=c['box_v'], a_min_override=c['a_min'],
        a_max_override=c['a_max'], e_max_target=c['e_max_target'],
        e_max_sat=c['e_max_sat'], de_max=c['de_max'], da_max_m=c['da_max'],
        di_max_rad=c['di_max'], di_min_rad=c['di_min'],
        di_phase_mode=int(c['di_phase']),
        i_target_min_rad=c['i_t_min'], i_target_max_rad=c['i_t_max'],
        fuel_frac_min=c['fuel_min'], fuel_frac_max=c['fuel_max'])
    kw['nav_max_ticks'] = max_ticks
    e = OrbitalNav(**kw)
    e.reset(seed=seed)
    return e


def nees_now(e):
    """Per-row NEES of the live filter against the true target state."""
    _, _, _, tgt_c = e._decode()
    idx = np.arange(e.num_agents)
    x, P = e._filt.mean_cov(idx)
    with np.errstate(all='ignore'):
        v = n3.nees_nd(x, P, np.asarray(tgt_c))
    return v, x, np.asarray(tgt_c)


def pos_err(x, tgt):
    return np.linalg.norm(x[:, :3] - tgt[:, :3], axis=1)


def stage_cadence(args):
    print('== cadence: what a day-warp actually IS under the shipped stack ==')
    import pufferlib.ocean.orbital_nav.nav_math as _nm
    tau = np.asarray(_nm.ACTION_TAU)
    for K in (0, 120):
        print(f'  nav_max_ticks={K}:')
        for row in (0, 16, 17, DAY_WARP):
            t = int(tau[row])
            n_t = t if K <= 0 else min(t, K)
            dt = t * 60.0 / max(n_t, 1)
            print(f'    row {row:2d}  tau {t:5d} min -> {n_t:5d} ticks '
                  f'at dt {dt:7.1f} s   ({t/60.0:5.2f} h of sim time)')
    print('  => a day-warp is SPARSE-CADENCE, never blind: the filter and the')
    print('     surrogate both run every tick (predict/update/accumulate).')


def stage_warp(args):
    print('\n== warp: filter health straight through a day-warp ==')
    e = w1_env(args.n, max_ticks=args.max_ticks, seed=args.seed)
    # settle: coast (row 0) until the surrogate has acquired what it will
    a_coast = np.zeros(e.num_agents, dtype=np.int32)
    for _ in range(args.settle):
        e.step(a_coast)
    acq0 = e._acq.acquired.copy()
    v0, x0, t0 = nees_now(e)
    p0 = pos_err(x0, t0)
    tick0 = int(e._acq.ticks.max())

    a_warp = np.full(e.num_agents, DAY_WARP, dtype=np.int32)
    e.step(a_warp)
    ticks_used = int(e._acq.ticks.max()) - tick0
    acq1 = e._acq.acquired.copy()
    v1, x1, t1 = nees_now(e)
    p1 = pos_err(x1, t1)

    # gain health: one ordinary 60 s tick after the warp
    e.step(a_coast)
    v2, x2, t2 = nees_now(e)
    p2 = pos_err(x2, t2)
    div = e._d_div / max(e._d_n, 1)
    e.close()

    def row(lbl, v, p, m=None):
        m = np.ones(v.shape, bool) if m is None else m
        v = v[m & np.isfinite(v)]; p = p[m[:p.size]] if m.size == p.size else p
        if v.size == 0:
            print(f'  {lbl:26s} (no rows)'); return
        print(f'  {lbl:26s} NEES med {np.median(v):9.2f}  '
              f'p90 {np.percentile(v, 90):10.2f}  '
              f'out-of-band {100.0 * np.mean((v < BAND_LO) | (v > BAND_HI)):5.1f}%  '
              f'pos med {np.median(p):9.0f} m')
    print(f'  acquired BEFORE warp {100.0 * acq0.mean():5.1f}%  ->  '
          f'AFTER warp {100.0 * acq1.mean():5.1f}%   of {args.n} rows')
    print(f'  surrogate ticks consumed by the one day-warp action: {ticks_used}')
    print(f'  NEES 6-dof 95% band [{BAND_LO:.3f}, {BAND_HI:.3f}]')
    print('  -- conditioned on rows ALREADY acquired before the warp '
          '(the honest before/after) --')
    row('before warp  [acq]', v0, p0, acq0)
    row('after warp   [acq]', v1, p1, acq0)
    row('after +1 tick[acq]', v2, p2, acq0)
    print('  -- all rows (unacquired rows carry a diffuse prior by design) --')
    row('before warp  [all]', v0, p0)
    row('after warp   [all]', v1, p1)
    print(f'  divergence rate over the run: {div:.4f}')


def stage_arc(args):
    print('\n== arc: NEES vs arc length at the day-warp cadence ==')
    print(f'  {"arc":>6s}  {"warps":>5s}  {"NEES med":>10s}  {"NEES p90":>10s}  '
          f'{"out-of-band":>11s}  {"pos med":>10s}')
    for hours in args.hours:
        e = w1_env(args.n, max_ticks=args.max_ticks, seed=args.seed)
        a_coast = np.zeros(e.num_agents, dtype=np.int32)
        for _ in range(args.settle):
            e.step(a_coast)
        n_warp = max(1, int(round(hours / 24.0)))
        a_warp = np.full(e.num_agents, DAY_WARP, dtype=np.int32)
        if hours < 24.0:
            # sub-day arcs use the 6 h row (17) so the cadence is the warp's,
            # not the 60 s baseline: tau=360 at K=120 -> dt=180 s
            a_warp = np.full(e.num_agents, 17, dtype=np.int32)
            n_warp = max(1, int(round(hours / 6.0)))
        for _ in range(n_warp):
            e.step(a_warp)
        v, x, t = nees_now(e)
        p = pos_err(x, t)
        v = v[np.isfinite(v)]
        oob = 100.0 * np.mean((v < BAND_LO) | (v > BAND_HI))
        print(f'  {hours:5.1f}h  {n_warp:5d}  {np.median(v):10.2f}  '
              f'{np.percentile(v, 90):10.2f}  {oob:10.1f}%  '
              f'{np.median(p):9.0f} m')
        e.close()


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', default='cadence,warp,arc')
    ap.add_argument('--n', type=int, default=64)
    ap.add_argument('--settle', type=int, default=60)
    ap.add_argument('--max-ticks', type=int, default=120)
    ap.add_argument('--seed', type=int, default=11)
    ap.add_argument('--hours', type=float, nargs='+',
                    default=[6.0, 12.0, 24.0, 48.0])
    a = ap.parse_args()
    np.random.seed(a.seed)
    for s in a.stage.split(','):
        {'cadence': stage_cadence, 'warp': stage_warp, 'arc': stage_arc}[s.strip()](a)
