#!/usr/bin/env python3
"""Scripted-GNC expert box ladder, with actuation matched to the policies.

THE QUESTION. The published tight-box expert rows (recon_expert_baseline §4.6:
53% at 5 km / 5 m/s, 18% at 5 km / 1 m/s) were measured on **Discrete-16**,
while the tight-box POLICY lineages fly **Discrete-20**. orbital.h:192-194
records that the 10 m/s radial quantum was the binding floor for the tight box
(best |v_rel| 5.02 m/s with 16 actions -> 0.71 m/s with fine radial), so the
expert's cliff may be tracking its own actuator rather than a guidance limit.
If so, the expert-vs-policy gap at TB5 is an artifact of the comparison.

THE ARMS. Each tight row runs n=200 at seeds 42 and 7 (the published rows were
n=100, single seed).

  A  D16 control       -- reproduction check. Must land on 53% / 18% within
                          binomial noise or the harness has drifted and nothing
                          downstream is interpretable.
  B  D20 quantum-only  -- BURN_ACTIONS gains rows 18/19 (+-1 m/s radial),
                          every controller constant untouched. The clean
                          "was it the actuator?" measurement.
  C  D20 + scaled      -- B, plus A_DEADBAND and DA_FLOOR scaled to the box.
                          Bounds the controller-constant confound.
  H  headline control  -- 30 km / 50 m/s at D16 and D20. Must NOT move.

THE SCALING, stated. Both constants are absolute metres fixed at the 30 km
design point: A_DEADBAND = 6 km = 0.20 * box_r, DA_FLOOR = 3 km = 0.10 * box_r.
Arm C keeps those fractions: `min(default, frac * box_r)`, giving 1.0 km and
0.5 km at a 5 km box. The reason to suspect DA_FLOOR specifically is dimensional
rather than empirical -- a parked semi-major-axis offset `da` implies a relative
speed |dv| ~ 0.5 * v * da / a, so at a = 7e6 m (v = 7546 m/s) the unscaled 3 km
floor implies **1.62 m/s** all by itself, already over a 1 m/s tolerance before
any guidance error at all. Scaled to 0.5 km it implies 0.27 m/s, inside it.

    python3 scripts/orbital/t3/expert_boxladder.py --arm B --seed 42
    python3 scripts/orbital/t3/expert_boxladder.py --report
"""

import argparse
import csv
import glob
import math
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(_HERE, '..', '..', '..'))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(ROOT, 'pufferlib'))

import expert_controller as EC                                    # noqa: E402

OUT = os.path.join(ROOT, 'web_data', 'results', 'expert_boxladder')
BOXES = {'tight5': (5000.0, 5.0), 'tight1': (5000.0, 1.0),
         'headline': (30000.0, 50.0)}

ARMS = {
    'A': dict(action_space=16, fine_radial=False, scale_consts=False,
              boxes=('tight5', 'tight1'), note='D16 reproduction control'),
    'B': dict(action_space=20, fine_radial=True, scale_consts=False,
              boxes=('tight5', 'tight1'), note='D20 quantum-only'),
    'C': dict(action_space=20, fine_radial=True, scale_consts=True,
              boxes=('tight5', 'tight1'), note='D20 + box-scaled constants'),
    'H16': dict(action_space=16, fine_radial=False, scale_consts=False,
                boxes=('headline',), note='D16 headline control'),
    'H20': dict(action_space=20, fine_radial=True, scale_consts=False,
                boxes=('headline',), note='D20 headline control'),
}


def run_arm(arm, seed, episodes):
    spec = ARMS[arm]
    os.makedirs(OUT, exist_ok=True)
    for bname in spec['boxes']:
        br, bv = BOXES[bname]
        tag = f'{arm}_{bname}_s{seed}'
        rows = EC.run(episodes=episodes, seed=seed, same_orbit=0, tag=tag,
                      box_r=br, box_v=bv,
                      action_space=spec['action_space'],
                      fine_radial=spec['fine_radial'],
                      scale_consts=spec['scale_consts'],
                      out_csv=os.path.join(OUT, f'{tag}.csv'))
        for r in rows:
            r['arm'], r['box'] = arm, bname
        sr = sum(r['success'] for r in rows) / max(len(rows), 1)
        print(f'{tag:22} {100 * sr:5.1f}%  n={len(rows)}', flush=True)


def _f(v, d=float('nan')):
    try:
        return float(v)
    except (TypeError, ValueError):
        return d


def report():
    rows = []
    for p in sorted(glob.glob(os.path.join(OUT, '*.csv'))):
        tag = os.path.basename(p)[:-4]
        arm, box, sd = tag.split('_')
        for r in csv.DictReader(open(p)):
            r['arm'], r['box'], r['sd'] = arm, box, sd[1:]
            rows.append(r)
    if not rows:
        print('no results')
        return

    def agg(sel):
        s = [r for r in rows if sel(r)]
        if not s:
            return None
        n = len(s)
        ok = [r for r in s if r['success'] == '1']
        dv = [_f(r['dv_used']) for r in ok] or [_f(r['dv_used']) for r in s]
        # Split successes out. A successful episode TERMINATES the moment both
        # tolerances are met, so its recorded pre-step minimum is censored just
        # above the tolerance and the pooled minimum is pinned at box_v by
        # construction. The failures are the uncensored sample, and they are
        # the ones that answer "what floor could the controller not beat?".
        vb = [_f(r['best_vrel_inbox']) for r in s
              if r['best_vrel_inbox'] not in ('', 'nan')]
        vbf = [_f(r['best_vrel_inbox']) for r in s
               if r['best_vrel_inbox'] not in ('', 'nan') and r['success'] != '1']
        ci = 1.96 * math.sqrt(max(len(ok) / n * (1 - len(ok) / n), 0) / n)
        causes = {}
        for r in s:
            causes[r['cause']] = causes.get(r['cause'], 0) + 1
        return dict(n=n, ok=len(ok), rate=len(ok) / n, ci=ci,
                    dv=float(np.median(dv)) if dv else float('nan'),
                    vb_med=float(np.median(vb)) if vb else float('nan'),
                    vb_p10=float(np.percentile(vb, 10)) if vb else float('nan'),
                    vb_min=float(np.min(vb)) if vb else float('nan'),
                    n_vb=len(vb), n_vbf=len(vbf),
                    vbf_med=float(np.median(vbf)) if vbf else float('nan'),
                    vbf_p10=float(np.percentile(vbf, 10)) if vbf else float('nan'),
                    vbf_min=float(np.min(vbf)) if vbf else float('nan'),
                    fine=sum(int(_f(r['n_fine_rad'], 0)) for r in s),
                    inbox=sum(int(_f(r['n_inbox'], 0)) for r in s),
                    inbox1=sum(int(_f(r['n_inbox_tau1'], 0)) for r in s),
                    causes=causes)

    print(f'{"arm":4}{"box":10}{"seed":6}{"n":>5}{"succ":>8}{"+-95%":>8}'
          f'{"dv_med":>8}{"FAILED-ep vrel_inbox med/p10/min":>34}'
          f'{"nf":>5}{"fine":>6}  causes')
    print('-' * 128)
    for arm in ('A', 'B', 'C', 'H16', 'H20'):
        for box in ('tight5', 'tight1', 'headline'):
            for sd in ('42', '7', 'both'):
                def sel(r, a=arm, b=box, s=sd):
                    return (r['arm'] == a and r['box'] == b
                            and (s == 'both' or r['sd'] == s))
                g = agg(sel)
                if not g or (sd == 'both' and g['n'] <= 200):
                    continue
                cz = ', '.join(f'{k} {v}' for k, v in
                               sorted(g['causes'].items(), key=lambda kv: -kv[1]))
                print(f'{arm:4}{box:10}{sd:6}{g["n"]:5d}'
                      f'{100 * g["rate"]:7.1f}%{100 * g["ci"]:7.1f}%'
                      f'{g["dv"]:8.0f}'
                      f'{g["vbf_med"]:13.2f}{g["vbf_p10"]:8.2f}'
                      f'{g["vbf_min"]:8.2f}{g["n_vbf"]:5d}'
                      f'{g["fine"]:6d}  {cz}')
    print('\nvrel_inbox = min |v_rel| over decisions with |dp| < box_r '
          '(episodes that never entered the position box are excluded).')
    print('fine = total emissions of rows 18/19 (fine radial) across the arm.')
    g16 = agg(lambda r: r['arm'] == 'A' and r['box'] == 'tight1')
    gb = agg(lambda r: r['arm'] == 'B' and r['box'] == 'tight1')
    if g16 and gb:
        print(f'\ncadence red-team: decisions inside the position box that were '
              f'tau=1 -- A {100 * g16["inbox1"] / max(g16["inbox"], 1):.1f}%, '
              f'B {100 * gb["inbox1"] / max(gb["inbox"], 1):.1f}%')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', choices=sorted(ARMS))
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--episodes', type=int, default=200)
    ap.add_argument('--report', action='store_true')
    a = ap.parse_args()
    if a.report:
        report()
        return 0
    if not a.arm:
        ap.error('need --arm or --report')
    run_arm(a.arm, a.seed, a.episodes)
    return 0


if __name__ == '__main__':
    sys.exit(main())
