#!/usr/bin/env python3
"""Where does W1xnav wall-clock actually go?

Step 1 of the C-port task, and its gate: if the filter tick loop is not the
dominant term, the port is not worth its risk and the answer is "stop".

Profiles two configurations that matter:
    w1      the W1_driftwait cell at nav_max_ticks=0 — up to 1440 filter ticks
            per day-warp decision, the worst case the port exists for
    mix     the shipped 7-cell mixture at nav_max_ticks=120

Attribution is by cProfile TOTTIME (self time), rolled into categories, so
numpy time lands on the caller that spent it rather than on `numpy` in the
abstract. Categories are deliberately coarse and named after the port decision
they inform.
"""
import argparse
import cProfile
import os
import pstats
import sys
import time

import numpy as np

WT = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
sys.path.insert(0, os.path.join(WT, 'pufferlib'))
sys.path.insert(0, os.path.join(WT, 'scripts', 'orbital', 'extj2'))

from pufferlib.ocean.orbital_nav.orbital_nav import OrbitalNav  # noqa: E402
import t11_cells as T                                           # noqa: E402

# (category, [substrings matched against "file:line(func)"])
CATS = [
    # The FD J2 STM calls propagate_cartesian_j2 FOURTEEN times per predict
    # (nominal + 6 states x +/-), and each of those runs the full element
    # conversion chain. This is the kernel, split out from the rest of the
    # filter so the port target is unambiguous.
    ('J2 propagation kernel (FD STM)', ['propagate_cartesian_j2', 'stm_fd_j2',
                                        'cartesian_to_elements_3d',
                                        'orbit_to_cartesian_3d', 'solve_kepler',
                                        'j2_secular_rates', 'mean_from_true',
                                        'eccentric_to_true', 'true_to_mean']),
    ('MSC6 chart + EKF update', ['msc6_', '(update)', '(predict)', '(_Q)',
                                 '(mean_cov)', 'repole']),
    ('CRLB surrogate  (accumulate/ready)', ['nav_surrogate.py']),
    ('obs encode/decode', ['nav_encode3d.py', '(_encode)', '(_decode)']),
    ('env C step (binding)', ['binding', '(step)', 'vec_step']),
]


def w1_kwargs(n, max_ticks):
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
    return kw


def mix_kwargs(n, max_ticks):
    kw = T.nav_env_kwargs(num_envs=n, nav_mode='bearings_only',
                          t11_mixture=1, j2_mode=1, nav_j2_mode=1)
    kw['nav_max_ticks'] = max_ticks
    return kw


def build(cfg, n, max_ticks):
    kw = w1_kwargs(n, max_ticks) if cfg == 'w1' else mix_kwargs(n, max_ticks)
    e = OrbitalNav(**kw)
    e.reset(seed=11)
    return e


def actions_for(cfg, e, rng, warp_frac):
    n = e.num_agents
    if cfg == 'w1':
        # drift-and-wait's own action character: mostly day-warps, which is
        # exactly the regime the port targets. A uniform policy would understate
        # the tick load by ~an order of magnitude.
        a = rng.integers(0, 30, n).astype(np.int32)
        m = rng.random(n) < warp_frac
        a[m] = 30
        return a
    return rng.integers(0, 31, n).astype(np.int32)


def run(cfg, n, steps, max_ticks, warp_frac):
    e = build(cfg, n, max_ticks)
    rng = np.random.default_rng(5)
    e.step(actions_for(cfg, e, rng, warp_frac))     # warm
    t0 = time.perf_counter()
    pr = cProfile.Profile()
    pr.enable()
    for _ in range(steps):
        e.step(actions_for(cfg, e, rng, warp_frac))
    pr.disable()
    wall = time.perf_counter() - t0
    e.close()
    return pr, wall


def report(pr, wall, label, n, steps):
    st = pstats.Stats(pr)
    tot = {c: 0.0 for c, _ in CATS}
    other = 0.0
    for (fn, ln, fname), (_cc, _nc, tt, _ct, _cal) in st.stats.items():
        key = f'{os.path.basename(fn)}:{ln}({fname})'
        hit = None
        for cat, pats in CATS:
            if any(p in key for p in pats):
                hit = cat
                break
        if hit:
            tot[hit] += tt
        else:
            other += tt
    prof_total = sum(tot.values()) + other
    print(f'\n== {label}  ({n} envs x {steps} decisions, wall {wall:.1f}s, '
          f'{n*steps/wall:.0f} env-steps/s) ==')
    print(f'  {"category":36s} {"self s":>8s} {"% profiled":>11s}')
    for cat, _ in CATS:
        print(f'  {cat:36s} {tot[cat]:8.2f} {100*tot[cat]/prof_total:10.1f}%')
    print(f'  {"other python":36s} {other:8.2f} {100*other/prof_total:10.1f}%')
    filt = tot['J2 propagation kernel (FD STM)'] + tot['MSC6 chart + EKF update']
    print(f'  ---- FILTER TICK LOOP TOTAL: {filt:.2f}s = '
          f'{100*filt/prof_total:.1f}% of profiled time ----')
    print('  top self-time frames:')
    st.sort_stats('tottime')
    rows = list(st.stats.items())
    rows.sort(key=lambda kv: kv[1][2], reverse=True)
    for (fn, ln, fname), (_cc, nc, tt, _ct, _cal) in rows[:12]:
        print(f'    {tt:7.2f}s {nc:9d} calls  {os.path.basename(fn)}:{ln}({fname})')
    return filt / prof_total


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg', default='w1,mix')
    ap.add_argument('--n', type=int, default=64)
    ap.add_argument('--steps', type=int, default=30)
    ap.add_argument('--w1-max-ticks', type=int, default=0)
    ap.add_argument('--mix-max-ticks', type=int, default=120)
    ap.add_argument('--warp-frac', type=float, default=0.5)
    a = ap.parse_args()
    fr = {}
    for cfg in a.cfg.split(','):
        cfg = cfg.strip()
        mt = a.w1_max_ticks if cfg == 'w1' else a.mix_max_ticks
        pr, wall = run(cfg, a.n, a.steps, mt, a.warp_frac)
        fr[cfg] = report(pr, wall, f'{cfg}  nav_max_ticks={mt}', a.n, a.steps)
    print('\n== port decision ==')
    for k, v in fr.items():
        verdict = ('PORT IS WORTH IT' if v >= 0.5 else
                   'marginal' if v >= 0.3 else 'DO NOT PORT — not the hot path')
        print(f'  {k:4s} filter tick loop = {100*v:5.1f}% of profiled time  -> {verdict}')
