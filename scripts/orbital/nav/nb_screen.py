#!/usr/bin/env python3
"""Cheap checkpoint screen for a T5 NB arm, before the expensive native eval.

`checkpoint_interval = 200` epochs gives ~5 checkpoints over a 50M-step arm,
and the T1/T3 precedent is that late-training performance is NOT monotone
(T1 §3 row 3: greedy held-out went 17 -> 19.5 -> 33.5 -> 16.5 -> 19 across
consecutive checkpoints). So screen the last few and evaluate the best.

The screen runs the WRAPPER path (`OrbitalNav(nav_mode='bearings_only')`, the
calibrated acquisition surrogate) rather than the real batch solver: it is ~5x
cheaper, and V4 measured the two agreeing on the canonical checkpoint at
139/200 = 69.5% on both. Use it to rank, never to report.

Usage: nb_screen.py DIR [--eps 100] [--last 3]
"""

import argparse
import glob
import os
import re
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
WT = os.path.abspath(os.path.join(_HERE, '..', '..', '..'))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(WT, 'pufferlib'))

from verify_extnav import T3_KW, rollout                        # noqa: E402
from pufferlib.ocean.orbital_nav.orbital_nav import OrbitalNav  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument('dir')
    p.add_argument('--eps', type=int, default=100)
    p.add_argument('--last', type=int, default=3)
    p.add_argument('--seed', type=int, default=123)
    a = p.parse_args()
    torch.set_num_threads(1)

    cks = sorted(glob.glob(os.path.join(a.dir, '**', 'model_*.pt'),
                           recursive=True),
                 key=lambda f: int(re.findall(r'(\d+)\.pt$', f)[0]))
    if not cks:
        print(f'NO CHECKPOINTS under {a.dir}')
        sys.exit(3)
    cks = cks[-a.last:]
    best, best_r = None, -1.0
    for c in cks:
        env = OrbitalNav(num_envs=1, nav_mode='bearings_only', **T3_KW)
        r = rollout(env, c, a.eps, a.seed, os.path.basename(c), verbose=False)
        print(f'  {os.path.basename(c):40s} {r["success"]}/{r["n_valid"]} = '
              f'{r["rate"]:6.1%}   causes: {r["cause_str"]}', flush=True)
        if r['rate'] > best_r:
            best, best_r = c, r['rate']
    print(f'BEST {best} {best_r:.4f}')


if __name__ == '__main__':
    main()
