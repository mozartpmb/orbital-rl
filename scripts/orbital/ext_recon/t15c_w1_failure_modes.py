#!/usr/bin/env python3
"""W1 failure-mode breakdown at 45% — is the residual 55% ONE mechanism?

This is the diagnostic that decides whether iteration 3 is the right lever at
all. Iterated DAgger buys a step per refresh and saturates; if the residual
failures are dominated by a STRUCTURAL constraint (the 22000-substep cap, or
the fuel floor) then no amount of better supervision fixes them, and the next
move is to change the cell rather than to teach the policy harder.

Records per episode: terminal cause, decisions, sim-time fraction of cap
consumed, fuel remaining, whether acquisition ever happened, and the final
separation as a multiple of the box radius. The last one is the discriminator:

  * safety_cap failures that end CLOSE to the box  -> ran out of CLOCK while
    converging. Structural: raise the cap.
  * safety_cap failures that end FAR from the box  -> never converging. Not a
    clock problem; more supervision or a different strategy.
  * stranded / near-zero fuel                      -> structural: fuel floor.
  * never acquired                                 -> navigation, not guidance.
"""
import argparse
import os
import sys

import numpy as np
import torch

WT = '/Users/pete/space_training'
sys.path.insert(0, os.path.join(WT, 'scripts', 'orbital', 'ext_recon'))
sys.path.insert(0, os.path.join(WT, 'scripts', 'orbital', 'extj2'))
from t15b_dagger_probe import cell_env, load                     # noqa: E402
import t11_cells as T                                            # noqa: E402

CAUSES = ['none', 'success', 'collision', 'escape', 'safety_cap',
          'stranded', 'hyperbolic', 'gave_up']


def run(ckpt, cell='W1_driftwait', n=32, want_eps=120, seed=31):
    c = T.ALL_CELLS[cell]
    cap = float(c['cap'])
    box = float(c['box_r'])
    e = cell_env(cell, n, seed)
    pol = load(e, ckpt)
    obs, _ = e.reset(seed=seed)
    h = torch.zeros(n, pol.hidden_size)
    cs = torch.zeros(n, pol.hidden_size)
    dec = np.zeros(n, int)
    ever_acq = np.zeros(n, bool)
    rows = []
    guard = 0
    while len(rows) < want_eps and guard < 60000:
        guard += 1
        with torch.no_grad():
            lg, _ = pol.forward_eval(
                torch.from_numpy(np.asarray(obs)).float(),
                {'lstm_h': h, 'lstm_c': cs})
        a = torch.argmax(lg, -1).numpy().astype(np.int32)
        ever_acq |= np.asarray(e._acq.acquired)
        st = e.get_state()
        obs, _, term, _, _ = e.step(a)
        dec += 1
        for i in np.flatnonzero(term):
            steps, cause = e.last_episode_result(i)
            s = st[i]
            # relative separation at the final recorded state, and fuel
            sat = np.asarray(s[5:8], float) if False else None
            rows.append(dict(
                cause=CAUSES[int(cause)] if int(cause) < len(CAUSES) else str(cause),
                dec=int(dec[i]), substeps=int(steps),
                cap_frac=float(steps) / cap,
                fuel=float(s[14]),
                acq=bool(ever_acq[i])))
            dec[i] = 0; ever_acq[i] = False
            h[i] = 0; cs[i] = 0
    e.close()
    return rows, cap, box


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default=f'{WT}/models/t3/t15b_dagger_final.pt')
    ap.add_argument('--eps', type=int, default=120)
    ap.add_argument('--n', type=int, default=32)
    a = ap.parse_args()

    rows, cap, box = run(a.ckpt, want_eps=a.eps, n=a.n)
    n = len(rows)
    causes = {}
    for r in rows:
        causes[r['cause']] = causes.get(r['cause'], 0) + 1
    succ = causes.get('success', 0)
    print(f'== W1 failure modes, {os.path.basename(a.ckpt)}, {n} episodes ==')
    print(f'  success {succ}/{n} = {succ/n:.1%}   (battery reference 45.0%)')
    print(f'\n  {"cause":12s} {"n":>4s} {"share":>7s} {"of FAILURES":>12s}')
    fails = n - succ
    for k in sorted(causes, key=lambda k: -causes[k]):
        fs = '' if k == 'success' else f'{causes[k]/max(fails,1):11.1%}'
        print(f'  {k:12s} {causes[k]:4d} {causes[k]/n:6.1%} {fs:>12s}')

    F = [r for r in rows if r['cause'] != 'success']
    S = [r for r in rows if r['cause'] == 'success']
    if F:
        print(f'\n  FAILED episodes (n={len(F)}):')
        print(f'    cap consumed  median {np.median([r["cap_frac"] for r in F]):.3f}  '
              f'p90 {np.percentile([r["cap_frac"] for r in F],90):.3f}')
        print(f'    fuel left     median {np.median([r["fuel"] for r in F]):.4f}  '
              f'p10 {np.percentile([r["fuel"] for r in F],10):.4f}')
        print(f'    ever acquired {np.mean([r["acq"] for r in F]):.1%}')
        print(f'    decisions     median {np.median([r["dec"] for r in F]):.0f}')
    if S:
        print(f'\n  SUCCESS episodes (n={len(S)}):')
        print(f'    cap consumed  median {np.median([r["cap_frac"] for r in S]):.3f}')
        print(f'    fuel left     median {np.median([r["fuel"] for r in S]):.4f}')
        print(f'    decisions     median {np.median([r["dec"] for r in S]):.0f}')

    print('\n== reading ==')
    if fails:
        cap_share = causes.get('safety_cap', 0) / fails
        fuel_share = (causes.get('stranded', 0)) / fails
        med_cap = np.median([r['cap_frac'] for r in F])
        med_fuel = np.median([r['fuel'] for r in F])
        print(f'  safety_cap is {cap_share:.1%} of failures; stranded {fuel_share:.1%}')
        if cap_share > 0.8 and med_cap > 0.95 and med_fuel > 0.05:
            print('  -> ONE MECHANISM: the CLOCK. Failures burn the whole cap with')
            print('     fuel still aboard. That is STRUCTURAL — raise the W1 cap (or')
            print('     shorten the required profile); another DAgger iteration')
            print('     teaches a policy that is already trying to do the right thing.')
        elif fuel_share > 0.3 or med_fuel < 0.02:
            print('  -> ONE MECHANISM: FUEL. Structural — raise the W1 fuel floor.')
        elif np.mean([r['acq'] for r in F]) < 0.5:
            print('  -> ONE MECHANISM: ACQUISITION. Navigation, not guidance;')
            print('     supervision on guidance states will not fix it.')
        else:
            print('  -> MIXED failure modes: no single structural lever dominates,')
            print('     which is the case where another supervision iteration is')
            print('     still the most direct move.')
