"""BLOCKER probe — the safety-cap penalty makes time-warping irrational until
the policy is already good.

Under gamma = 0.995 PER DECISION with a -10 terminal at the safety cap, the
cost of failing is gamma^N * 10 where N is the number of DECISIONS taken.  A
coast-only policy reaches the cap in 3000 decisions (gamma^3000 = 3e-7, the
-10 is invisible); a warp-60 policy reaches it in 50 decisions (gamma^50 =
0.78, the -10 is worth -7.8).  Warping is therefore a large, immediate,
guaranteed loss for any policy that is not already succeeding -- while it is
the ONLY way to make the +10 visible for a policy that does succeed.

This script measures the resulting break-even success probability at which
warping becomes worth more than doing nothing, for the design as shipped
(cap reward -10) and for the minimal fix (cap reward 0), and sweeps the fix
across the whole warp ladder.

Outputs web_data/results/t3_redteam_warp_barrier.csv
"""
import csv
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rt_common import CAUSES, T3_KW, disc_sum, make_env  # noqa

OUT = '/Users/pete/space_training/web_data/results/t3_redteam_warp_barrier.csv'
GAMMA = 0.995


def episode(gap, plan, seed, cap=3000, cap_reward=-10.0):
    kw = dict(T3_KW)
    kw['episode_cap_steps'] = cap
    env = make_env(seed=seed, same_orbit_init=1, e_max_target=0.0,
                   e_max_sat=0.0, phase_gap_fixed=math.radians(gap), **kw)
    env.reset(seed=seed)
    rews = []
    k = 0
    while True:
        env.step(np.array([plan(k)], dtype=np.int32))
        rews.append(float(env.rewards[0]))
        k += 1
        if env.terminals[0]:
            steps, cause = env.last_episode_result(0)
            break
    env.close()
    if cause == 4:                      # safety cap: swap the terminal reward
        rews[-1] = rews[-1] + 10.0 + cap_reward
    return CAUSES[cause], k, steps, disc_sum(rews, GAMMA), float(np.sum(rews))


def main():
    rows = []
    warps = [(0, 'coast'), (9, 'warp5'), (10, 'warp30'), (11, 'warp60')]
    print(f'{"warp":>7s} {"cap_r":>6s} {"V_fail (do-nothing)":>20s} '
          f'{"V_succ":>9s} {"break-even p*":>14s}')
    for w, wname in warps:
        for cap_r in (-10.0, 0.0):
            # failing policy: pure warp, never burns
            vf = np.mean([episode(90, lambda k: w, s, cap_reward=cap_r)[3]
                          for s in range(6)])
            # succeeding policy of the same warp granularity (10 deg gap,
            # single 5 m/s retro burn -> capture without any closing burn)
            out = [episode(10, lambda k: 4 if k < 1 else w, s, cap_reward=cap_r)
                   for s in range(6)]
            vs = np.mean([o[3] for o in out])
            nsucc = sum(o[0] == 'success' for o in out)
            pstar = (0.0 - vf) / (vs - vf) if vs > vf else float('nan')
            print(f'{wname:>7s} {cap_r:6.0f} {vf:20.3f} {vs:9.3f} '
                  f'{pstar:13.1%}' + ('' if nsucc == 6 else f'  (succ {nsucc}/6)'))
            rows.append(dict(warp=wname, tau=[1, 5, 30, 60][[0, 9, 10, 11].index(w)],
                             cap_reward=cap_r, V_fail=float(vf), V_succ=float(vs),
                             n_success=nsucc, break_even_p=pstar))
    with open(OUT, 'w', newline='') as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    print(f'\nwrote {len(rows)} rows -> {OUT}')


if __name__ == '__main__':
    main()
