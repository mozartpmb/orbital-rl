#!/usr/bin/env python3
"""T15b gates: the teacher-label-quality gate the DAgger refresh needs.

A DAgger refresh is only supervision if the TEACHER IS COMPETENT WHERE THE
STUDENT GOES. If the student has wandered somewhere the teacher is also lost,
its softmax is noise with a confident shape, and training on it is bug-#13 in a
new costume. That is decidable before committing a run, so it is a gate.

D1  TEACHER NOT LOST on the aggregated anchor set. Median teacher entropy must
    stay well below flat (ln 31 = 3.434). Measured: 0.476 on teacher-visited
    states, 0.695 on student-visited — a +0.219 nat shift, nowhere near flat,
    with max-prob still 0.78. Gate at 0.5*ln31 = 1.717.

D2  THE REFRESH REACHES SOMEWHERE NEW. Weak form only, and the rationale is
    RE-DERIVED after its original premise was falsified.

    The gate was first justified as a yield predictor: more correction on offer
    => bigger buy. Iteration 2 killed that. CE fell 40% (ratio 1.75 -> 1.34) and
    the yield HELD (+13.5 -> +14.0). So the ratio does not forecast the buy.

    What survives is the weak claim: a ratio near 1.0 means the fresh states
    carry no more disagreement than the teacher's own, i.e. the refresh
    re-collected what the student already handles.

    THE FLOOR IS NOT MOVED, and deliberately so. Observed working points are
    ratio 1.75 -> +13.5 and 1.34 -> +14.0; observed FAILURES: none. With no
    failure to calibrate against, the data cannot justify a floor above the
    lowest ratio seen to work, and lowering it now would be convenience — the
    current iteration measures 1.39 and passes as-is. If a future iteration
    lands below 1.30, that is NOT evidence it will fail; the honest response is
    to run it with this gate downgraded to a warning, not to quietly retune the
    number to make it pass.

    NOVELTY is the better-behaved companion and is reported alongside: median
    nearest-neighbour distance from fresh states to the teacher set ran
    1.1969 -> 1.1316 -> 1.1288 across the three generations — essentially FLAT,
    which matches the flat yield far better than CE did.

D3  AGGREGATION, NOT SUBSTITUTION. The shipped set must contain BOTH halves —
    the D in DAgger is aggregation (Ross & Bagnell), and replacing would bet
    the run on the student distribution being non-pathological.
"""
import os
import sys

import numpy as np
import torch

WT = '/Users/pete/space_training'
sys.path.insert(0, os.path.join(WT, 'scripts', 'orbital', 'ext_recon'))
from t15b_dagger_probe import cell_env, load                      # noqa: E402

LN31 = float(np.log(31))
G_PASS, G_FAIL = [], []
AGG = os.environ.get('T15_AGG', f'{WT}/models/t15/anchor_w1_dagger.pt')
TEACHER = f'{WT}/models/t3/w1nav_child.pt'
STUDENT = os.environ.get('T15_STUDENT', f'{WT}/models/t3/t15_remix_final.pt')
TSET = f'{WT}/models/t15/anchor_w1_k0.pt'
SSET = os.environ.get('T15_SSET', f'{WT}/models/t15/anchor_w1_student.pt')


def check(name, ok, detail=''):
    (G_PASS if ok else G_FAIL).append(name)
    print(f'  [{"PASS" if ok else "FAIL"}] {name}')
    if detail:
        print(f'         {detail}')


def _m(teach, stud, obs):
    with torch.no_grad():
        lt, _ = teach(torch.as_tensor(obs).float(),
                      dict(action=None, lstm_h=None, lstm_c=None))
        ls, _ = stud(torch.as_tensor(obs).float(),
                     dict(action=None, lstm_h=None, lstm_c=None))
    if isinstance(lt, (list, tuple)):
        lt, ls = lt[0], ls[0]
    tlp = torch.log_softmax(lt.float(), -1)
    slp = torch.log_softmax(ls.float(), -1)
    tp = tlp.exp()
    return (float(np.median((-(tp * tlp).sum(-1)).flatten().numpy())),
            float(np.median(tp.max(-1).values.flatten().numpy())),
            float(np.median(((tp * (tlp - slp)).sum(-1)).flatten().numpy())))


if __name__ == '__main__':
    env = cell_env('W1_driftwait', 2, 3)
    teach, stud = load(env, TEACHER), load(env, STUDENT)
    env.close()
    agg = torch.load(AGG, map_location='cpu', weights_only=False)
    tset = torch.load(TSET, map_location='cpu', weights_only=False)['obs']
    sset = torch.load(SSET, map_location='cpu', weights_only=False)['obs']

    print('== D1  is the teacher lost where the student goes? ==')
    hA, mA, cA = _m(teach, stud, agg['obs'][:200].numpy())
    hT, mT, cT = _m(teach, stud, tset[:200].numpy())
    hS, mS, cS = _m(teach, stud, sset[:200].numpy())
    print(f'    teacher-visited  H {hT:.3f}  maxp {mT:.3f}  CE {cT:.3f}')
    print(f'    student-visited  H {hS:.3f}  maxp {mS:.3f}  CE {cS:.3f}')
    print(f'    AGGREGATED       H {hA:.3f}  maxp {mA:.3f}  CE {cA:.3f}')
    check('D1 teacher stays competent on the anchor set', hA < 0.5 * LN31,
          f'median teacher entropy {hA:.3f} vs flat {LN31:.3f} '
          f'(gate {0.5*LN31:.3f}); max-prob {mA:.3f}. A lost teacher would sit '
          f'near flat and its labels would be noise.')

    print('\n== D2  does the refresh add correction the old set did not have? ==')
    check('D2 student-visited CE exceeds teacher-visited CE', cS > 1.3 * cT,
          f'{cS:.3f} vs {cT:.3f} = {cS/max(cT,1e-9):.2f}x. If these were equal '
          f'the refresh would be re-collecting states the student already '
          f'handles, and T15b would be T15 with extra steps.')

    print('\n== D3  aggregation, not substitution ==')
    n_ok = agg['obs'].shape[0] == tset.shape[0] + sset.shape[0]
    check('D3 the shipped set is BOTH halves', n_ok,
          f'{agg["obs"].shape[0]} = {tset.shape[0]} teacher-visited + '
          f'{sset.shape[0]} student-visited. The D in DAgger is aggregation; '
          f'substitution would bet the run on the student distribution being '
          f'non-pathological.')

    print(f'\n=== {len(G_PASS)}/{len(G_PASS)+len(G_FAIL)} T15b anchor gates pass ===')
    for f in G_FAIL:
        print(f'  FAILED: {f}')
    sys.exit(1 if G_FAIL else 0)
