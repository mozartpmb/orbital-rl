"""ATTACK D (part 2) — how much of the task falls to a single burn + warping?

Policy class under test: "pick one prograde/retrograde impulse magnitude, fire
it once, then warp-60 until the episode ends".  It never nulls relative
velocity and never circularises.  If it wins, the 50 m/s velocity box is slack
and the rung is a ballistic-intercept problem, not a rendezvous problem.

For each phase gap we report the success rate of the BEST single burn in the
class (i.e. what an optimal policy of this degenerate form achieves), at
episode_cap_steps = 2000 (legacy) and 3000 (proposed).

Outputs web_data/results/t3_redteam_trivial_sweep.csv
"""
import csv
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rt_common import CAUSES, MU, ObsView, R_EARTH, T3_KW, make_env  # noqa

OUT = '/Users/pete/space_training/web_data/results/t3_redteam_trivial_sweep.csv'

# (action, repeats) -> total retrograde dv
PLANS = []
for act, q in ((13, 1), (15, 1), (15, 2), (4, 1), (4, 2), (5, 1), (5, 2),
               (5, 3), (6, 1), (6, 2), (6, 3), (6, 4), (6, 6)):
    PLANS.append((act, q))
PLANS = sorted(set(PLANS))


def run(gap_deg, act, n_burn, cap, seed, e=0.0, warp=11):
    kw = dict(T3_KW)
    kw['episode_cap_steps'] = cap
    env = make_env(seed=seed, same_orbit_init=1,
                   e_max_target=e, e_max_sat=e,
                   phase_gap_fixed=math.radians(gap_deg), **kw)
    env.reset(seed=seed)
    k = 0
    while k < 200000:
        env.step(np.array([act if k < n_burn else warp], dtype=np.int32))
        k += 1
        if env.terminals[0]:
            steps, cause = env.last_episode_result(0)
            break
    o = ObsView(env.observations[0])
    env.close()
    return cause, steps, k


def main():
    from rt_common import ACTION_DV
    rows = []
    gaps = list(range(0, 181, 10))
    seeds = list(range(8))
    for cap in (2000, 3000):
        for g in gaps:
            for act, n in PLANS:
                dv = abs(ACTION_DV[act][0]) * n
                succ = 0
                for s in seeds:
                    cause, steps, dec = run(g, act, n, cap, s)
                    succ += int(cause == 1)
                rows.append(dict(cap=cap, gap_deg=g, action=act, n_burn=n,
                                 dv=dv, n_seeds=len(seeds), successes=succ,
                                 rate=succ / len(seeds)))
    with open(OUT, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f'wrote {len(rows)} rows -> {OUT}\n')

    print('best single-burn-then-warp policy, success over 8 seeds')
    print(f'{"gap":>5s} | {"cap=2000 best":>26s} | {"cap=3000 best":>26s}')
    for g in gaps:
        line = f'{g:5d} |'
        for cap in (2000, 3000):
            sel = [r for r in rows if r['cap'] == cap and r['gap_deg'] == g]
            best = max(sel, key=lambda r: r['rate'])
            line += f' {best["successes"]}/8 @ dv={best["dv"]:5.0f} m/s     |'
        print(line)

    print('\naggregate: fraction of (gap) cells where some single burn wins '
          '>= 6/8')
    for cap in (2000, 3000):
        n = 0
        for g in gaps:
            sel = [r for r in rows if r['cap'] == cap and r['gap_deg'] == g]
            if max(r['successes'] for r in sel) >= 6:
                n += 1
        print(f'  cap={cap}: {n}/{len(gaps)} gaps')


if __name__ == '__main__':
    main()
