#!/usr/bin/env python3
"""Red-team (ii): is the TIGHT defense anchor's state coverage AGING?

TIGHT drifted 90.5 -> 85.5 across T15 while carrying a defense anchor that was
supposed to hold it. Two very different explanations:

  (a) the anchor is working and 61% opposing gradient share simply outweighs
      it — a WEIGHT problem, fix by raising lambda or lowering W1's share;
  (b) the anchor's STATES are stale. The set was collected from
      `t13b_anchor_final`; if the current policy visits TIGHT states that set
      does not cover, the anchor is defending geometry the policy has left, and
      raising lambda would only pull harder toward the wrong place.

These have opposite fixes, so the difference is measured rather than guessed.

The discriminator: CE(teacher || current policy) evaluated on the OLD anchor
states versus on FRESHLY COLLECTED states from the current policy. If the fresh
states show much larger divergence, the policy has moved somewhere the anchor
does not see — (b). If they are comparable, the anchor sees what the policy
does — (a).
"""
import os
import sys

import numpy as np
import torch

WT = '/Users/pete/space_training'
sys.path.insert(0, os.path.join(WT, 'scripts', 'orbital', 'ext_recon'))
from t15b_dagger_probe import (cell_env, load, collect_student_states,  # noqa: E402
                               stats)

TEACH = f'{WT}/models/t3/t13b_anchor_final.pt'      # the defense teacher
STUD = f'{WT}/models/t3/t15_remix_final.pt'
OLD_SET = f'{WT}/models/t15/anchor_tight_k0.pt'

if __name__ == '__main__':
    print('== TIGHT defense anchor: are its states still where the policy goes? ==')
    W, lab, ep_len, ep_ok = collect_student_states(
        'TIGHT_5k1', STUD, n=32, want_windows=400, seed=23)
    print(f'  student on TIGHT: success {ep_ok.mean():.1%}, '
          f'decisions/ep median {np.median(ep_len):.0f}')

    env = cell_env('TIGHT_5k1', 2, 3)
    teach = load(env, TEACH)
    stud = load(env, STUD)
    env.close()

    old = torch.load(OLD_SET, map_location='cpu', weights_only=False)['obs']
    print('\n  CE(defense teacher || current policy):')
    _, _, c_old = stats(teach, stud, old[:200].numpy(), 'OLD anchor states (t13b-visited)')
    _, _, c_new = stats(teach, stud, W[:200], 'FRESH states (t15-visited)')
    if (~lab).any():
        stats(teach, stud, W[~lab][:200], 'FRESH states (FAILED eps only)')

    r = float(np.median(c_new)) / max(float(np.median(c_old)), 1e-12)
    print(f'\n  fresh/old CE ratio = {r:.2f}')
    if r > 3.0:
        print('  -> (b) THE ANCHOR STATES ARE STALE. The policy has moved to TIGHT')
        print('     geometry the anchor set does not cover, so the defense is')
        print('     holding ground the policy already left. Fix = REFRESH the')
        print('     defense set too, not raise lambda.')
    elif r > 1.5:
        print('  -> partial aging; refresh is worth doing but is not the whole story.')
    else:
        print('  -> (a) states are still representative. TIGHT drift is a WEIGHT')
        print('     problem (61% opposing share), not stale coverage: fix with')
        print('     lambda / sampling weight, and refreshing would buy nothing.')
