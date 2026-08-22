#!/usr/bin/env python3
"""Root-vs-harness compatibility gate — the one W1xnav's stage 0 did not have.

WHAT HAPPENED. The W1xnav campaign gated the filter (L2), the acquisition
signal (W1A/W1B), the C kernel (fuzz + mutation battery) and the env anchors —
and then warm-started from `j2wait_W1_driftwait.pt`, a 96.0% specialist trained
in the NARROW normalizer family (obs_alt_scale_m 1.6e6), into a harness running
WIDE (8.0e6). Its own home cell scored 0/200. Nothing gated that, because the
floor row that would have shown it was ADVISORY. Bug-#15's class, root-family
instance: a campaign specifies its CELL completely and its ROOT not at all.

THE STANDING PATTERN, in two checks of very different cost.

R1  FAMILY FINGERPRINT — static, ~0.1 s, no env, no eval.
    The wide/narrow families differ by a factor of 5 on the alt-scaled
    observation columns (0, 7, 17, 20), so a network trained against 5x
    SMALLER inputs carries ~5x LARGER weights there. That is directly
    measurable from the checkpoint:

        j2wait_W1_driftwait.pt   (narrow)          0.5004
        w1nav_root_wide.pt       (wide/transplant) 2.4235   <- 4.84x
        t11_generalist_rungB.pt  (wide)            2.1756
        t11t_tight_child.pt      (wide)            2.0359

    4.84 against a theoretical 5.00. The mismatch was detectable from the file
    alone, before constructing an env, and would have cost 0.1 s instead of a
    launch.

    HONEST LIMIT: the threshold is calibrated on four checkpoints of one
    architecture. It is a fast SCREEN, not the authority. Pass `--reference` to
    replace the absolute threshold with a ratio-of-ratios against a checkpoint
    known to be in the target family, which is robust to architecture drift.

R2  HOME-CELL FLOOR THROUGH THIS HARNESS — authoritative, one short eval.
    A root with a published home-cell score must reproduce it through the
    campaign's OWN harness, as a HARD gate. This is the check that generalises:
    it catches normalizer family, head size, action-space width, cell
    misparameterisation and checkpoint corruption without knowing which of them
    went wrong, because it asks the only question that matters — does this root
    still work where it is known to work.

Usage (a warm-start campaign should run BOTH in stage 0):
    python3 root_gate.py --root models/t3/w1nav_root_wide.pt \\
        --cell W1_driftwait --nav-mode truth --expect 96 --tol 15 --episodes 40
"""
import argparse
import os
import re
import subprocess
import sys

import numpy as np
import torch

WT = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
EVAL = os.path.join(WT, 'scripts', 'orbital', 'extj2', 't11_eval.py')

# transplant groups, from rescale_ckpt_normalizers.py
ALT_COLS = (0, 7, 17, 20)
LVLH_COLS = (24, 33, 34)
# log-midpoint of the measured narrow (0.50) and wide (2.04-2.42) clusters
FAMILY_SPLIT = 1.10

G_PASS, G_FAIL = [], []


def check(name, ok, detail=''):
    (G_PASS if ok else G_FAIL).append(name)
    print(f'  [{"PASS" if ok else "FAIL"}] {name}')
    if detail:
        print(f'         {detail}')


def fingerprint(path):
    """alt-column / other-column weight-norm ratio of the obs encoder."""
    sd = torch.load(path, map_location='cpu', weights_only=True)
    keys = [k for k, v in sd.items()
            if getattr(v, 'ndim', 0) == 2 and v.shape[1] == 38]
    if not keys:
        return None, None
    W = sd[keys[0]].double().numpy()
    other = [j for j in range(38) if j not in ALT_COLS + LVLH_COLS]
    return (float(np.linalg.norm(W[:, list(ALT_COLS)])
                  / max(np.linalg.norm(W[:, other]), 1e-30)), keys[0])


def gate_r1(a):
    print('\n== R1  root normalizer FAMILY (static fingerprint) ==')
    r, key = fingerprint(a.root)
    if r is None:
        check('R1 encoder found in the checkpoint', False,
              'no (*, 38) weight matrix — cannot fingerprint')
        return
    want_wide = a.obs_alt_scale >= 4.0e6
    if a.reference:
        rr, _ = fingerprint(a.reference)
        ratio = r / max(rr, 1e-30)
        ok = 0.5 <= ratio <= 2.0
        check('R1 root is in the same family as the reference', ok,
              f'root {r:.4f} vs reference {rr:.4f} -> ratio {ratio:.3f} '
              f'(same family ~1.0; one transplant apart ~5.0 or ~0.2)')
    else:
        is_wide = r > FAMILY_SPLIT
        ok = (is_wide == want_wide)
        check(f'R1 root fingerprint matches the harness family '
              f'({"wide" if want_wide else "narrow"})', ok,
              f'ratio {r:.4f} -> looks {"WIDE" if is_wide else "NARROW"}; '
              f'harness obs_alt_scale_m = {a.obs_alt_scale:.3g}. '
              f'Calibration: narrow 0.50, wide 2.04-2.42, split {FAMILY_SPLIT}. '
              f'Screen only — R2 is the authority.'
              + ('' if ok else '  FIX: transplant with '
                 'scripts/orbital/extj2/rescale_ckpt_normalizers.py'))


def gate_r2(a):
    print(f'\n== R2  home-cell floor through THIS harness '
          f'({a.cell}, {a.nav_mode}) ==')
    env = dict(os.environ, OMP_NUM_THREADS='1',
               PYTHONPATH=os.path.join(WT, 'pufferlib'))
    cmd = [sys.executable, EVAL, '--ckpt', a.root, '--cell', a.cell,
           '--nav-mode', a.nav_mode, '--episodes', str(a.episodes),
           '--seed', str(a.seed), '--label', 'root_gate']
    if a.filter_impl != 'py':
        cmd += ['--filter-impl', a.filter_impl]
    r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    m = re.search(r'success (\d+)/(\d+) = *([0-9.]+)%', r.stdout)
    if not m:
        check('R2 the root evaluates at all', False,
              (r.stdout[-300:] + r.stderr[-300:]).strip())
        return
    got = float(m.group(3))
    lo = a.expect - a.tol
    check(f'R2 root scores >= {lo:.0f}% on its own home cell '
          f'(published {a.expect:.0f}%)', got >= lo,
          f'measured {got:.1f}% ({m.group(1)}/{m.group(2)}). A root that '
          f'cannot reproduce its own published number through this harness is '
          f'mis-paired with it, and training 50M steps from it would '
          f'manufacture a false bootstrap failure that looks exactly like a '
          f'real one.')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--cell', required=True)
    ap.add_argument('--nav-mode', default='truth')
    ap.add_argument('--expect', type=float, required=True,
                    help="the root's published score on this cell, in percent")
    ap.add_argument('--tol', type=float, default=15.0)
    ap.add_argument('--episodes', type=int, default=40)
    ap.add_argument('--seed', type=int, default=123)
    ap.add_argument('--obs-alt-scale', type=float, default=8.0e6)
    ap.add_argument('--reference', default=None,
                    help='checkpoint known to be in the harness family')
    ap.add_argument('--filter-impl', default='py')
    ap.add_argument('--stage', default='r1,r2')
    a = ap.parse_args()
    for s in a.stage.split(','):
        {'r1': gate_r1, 'r2': gate_r2}[s.strip()](a)
    print(f'\n=== {len(G_PASS)}/{len(G_PASS)+len(G_FAIL)} root gates pass ===')
    for f in G_FAIL:
        print(f'  FAILED: {f}')
    sys.exit(1 if G_FAIL else 0)
