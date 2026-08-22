#!/usr/bin/env python3
"""T15 multi-anchor gates.

T1a/T1b  BIT-INERTNESS. `anchor_specs` unset must reproduce the pre-T15 tree
         exactly. Same discipline and same determinism control as T13b's
         A1a/A1b and T14's C1a/C1b — pufferl leaves torch/numpy/random UNSEEDED
         (all three seed calls are commented out), so a weight-hash comparison
         is meaningless until a sitecustomize seed is injected, and T1a proves
         the instrument before T1b claims anything with it.

T2       BOTH TEACHERS LOAD, AND EACH IS ZERO AGAINST ITSELF ON ITS OWN SET.
         CE(teacher || teacher) == 0 is the check that the term is a divergence
         rather than an arbitrary scalar. For the DEFENSE teacher this also
         means the anchor contributes no gradient at step 0 (it IS the init,
         CLEAR-style). For the ACQUISITION teacher it must be NONZERO against
         the ROOT — that is the entire point: a W1 anchor that already agreed
         with the root would be teaching nothing.

T3       PER-TEACHER LAMBDA SCHEDULES are distinct and land where intended.
"""
import os
import shutil
import subprocess
import sys

import numpy as np
import torch

WT = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
sys.path.insert(0, os.path.join(WT, 'pufferlib'))
sys.path.insert(0, os.path.join(WT, 'scripts', 'orbital', 'extj2'))
import t13b_anchor_gates as A                                    # noqa: E402

G_PASS, G_FAIL = [], []
TOUCHED = ['pufferlib/pufferlib/pufferl.py',
           'pufferlib/pufferlib/config/ocean/orbital_nav.ini',
           'pufferlib/pufferlib/ocean/orbital/t11_cells.py',
           'pufferlib/pufferlib/ocean/orbital/orbital.py']
W1_DATA = os.path.join(WT, 'models/t15/anchor_w1_k0.pt')
TG_DATA = os.path.join(WT, 'models/t15/anchor_tight_k0.pt')
ROOT = os.path.join(WT, 'models/t3/t13b_anchor_final.pt')
W1_TEACH = os.path.join(WT, 'models/t3/w1nav_child.pt')


def check(name, ok, detail=''):
    (G_PASS if ok else G_FAIL).append(name)
    print(f'  [{"PASS" if ok else "FAIL"}] {name}')
    if detail:
        print(f'         {detail}')


def gate_bitid():
    print('\n== T1  anchor_specs unset is BIT-IDENTICAL to the pre-T15 tree ==')
    sd = A._make_seed_dir()
    bak = os.path.join(sd, 'bak'); os.makedirs(bak, exist_ok=True)
    paths = [os.path.join(WT, r) for r in TOUCHED]
    for r, p in zip(TOUCHED, paths):
        shutil.copy2(p, os.path.join(bak, os.path.basename(r)))
    for d in ('T15_B1', 'T15_B2', 'T15_OFF'):
        shutil.rmtree(os.path.join(WT, 'pufferlib', 'experiments_t15', d),
                      ignore_errors=True)
    try:
        for r, p in zip(TOUCHED, paths):
            base = subprocess.run(['git', 'show', f'origin/main:{r}'], cwd=WT,
                                  capture_output=True, text=True).stdout
            open(p, 'w').write(base)
        b1 = A.run_train('experiments_t15/T15_B1', seed_dir=sd)
        b2 = A.run_train('experiments_t15/T15_B2', seed_dir=sd)
        h1 = A.ckpt_hash(b1[-1]) if b1 else None
        h2 = A.ckpt_hash(b2[-1]) if b2 else None
    finally:
        for r, p in zip(TOUCHED, paths):
            shutil.copy2(os.path.join(bak, os.path.basename(r)), p)
    check('T1a instrument is deterministic (pristine vs pristine)',
          h1 is not None and h1 == h2, f'{h1} vs {h2}')
    if h1 is None or h1 != h2:
        check('T1b anchor_specs unset reproduces the pre-T15 tree', False,
              'SKIPPED — instrument not deterministic')
        return
    o = A.run_train('experiments_t15/T15_OFF', seed_dir=sd)
    ho = A.ckpt_hash(o[-1]) if o else None
    check('T1b anchor_specs unset reproduces the pre-T15 tree bit-for-bit',
          ho is not None and ho == h1, f'pristine {h1} vs T15-off {ho}')


def _ce(net_a, net_b, obs, n=8):
    mb = obs[:n]
    with torch.no_grad():
        la, _ = net_a(mb, dict(action=None, lstm_h=None, lstm_c=None))
        lb, _ = net_b(mb, dict(action=None, lstm_h=None, lstm_c=None))
    if isinstance(la, (list, tuple)):
        la, lb = la[0], lb[0]
    alp = torch.log_softmax(la.float(), -1)
    blp = torch.log_softmax(lb.float(), -1)
    return float((alp.exp() * (alp - blp)).sum(-1).mean())


def gate_teachers():
    print('\n== T2  both teachers load; defense==0 at init, acquisition!=0 ==')
    import copy
    from pufferlib.models import Default, LSTMWrapper
    from pufferlib.ocean.orbital_nav.orbital_nav import OrbitalNav
    import t11_cells as T
    env = OrbitalNav(**T.nav_env_kwargs(num_envs=1, nav_mode='bearings_only',
                                        t11_mixture=1, j2_mode=1, nav_j2_mode=1))
    root = LSTMWrapper(env, Default(env))
    root.load_state_dict(torch.load(ROOT, map_location='cpu', weights_only=True))
    root.eval()
    w1t = copy.deepcopy(root)
    w1t.load_state_dict(torch.load(W1_TEACH, map_location='cpu', weights_only=True))
    w1t.eval()
    tg = copy.deepcopy(root)      # defense teacher IS the root
    env.close()

    ow1 = torch.load(W1_DATA, map_location='cpu', weights_only=False)['obs']
    otg = torch.load(TG_DATA, map_location='cpu', weights_only=False)['obs']

    d_self = _ce(tg, root, otg)
    check('T2a DEFENSE teacher is zero against the root on its own set',
          abs(d_self) < 1e-6,
          f'CE = {d_self:.3e} — the anchor IS the init (CLEAR), so it adds no '
          f'gradient at step 0, which is the correct initial condition')
    a_self = _ce(w1t, w1t, ow1)
    check('T2b ACQUISITION teacher is zero against itself', abs(a_self) < 1e-6,
          f'CE = {a_self:.3e} — confirms the term is a divergence')
    a_root = _ce(w1t, root, ow1)
    check('T2c ACQUISITION teacher DISAGREES with the root (it has something '
          'to teach)', a_root > 0.05,
          f'CE(w1nav_child || t13b_root) = {a_root:.4f} on W1 states. The root '
          f'scores 0.0/25 on W1; an anchor that already agreed with it would be '
          f'teaching nothing and the run would be a 6/7 with extra steps.')


def gate_lambda():
    print('\n== T3  per-teacher lambda schedules ==')
    import pufferlib.pufferl as P

    class F:
        global_step = 0
    f = F()
    acq = dict(lam0=0.30, lam1=0.05, decay=3e7)
    dfn = dict(lam0=0.02, lam1=0.02, decay=0.0)
    pts = [(0, 0.30), (1.5e7, 0.175), (3e7, 0.05), (1e8, 0.05)]
    ok = True
    for gs, want in pts:
        f.global_step = gs
        got = P.PuffeRL._anchor_lambda_now(f, acq)
        ok &= abs(got - want) < 1e-9
        print(f'    acquisition @ {gs/1e6:6.1f}M -> {got:.4f} (want {want:.4f})')
    f.global_step = 5e7
    dv = P.PuffeRL._anchor_lambda_now(f, dfn)
    print(f'    defense     @  50.0M -> {dv:.4f} (constant)')
    check('T3 acquisition weans 0.30->0.05 by 30M; defense stays 0.02',
          ok and abs(dv - 0.02) < 1e-12,
          'one global lambda cannot serve both regimes: 0.02 is too quiet to '
          'escape f_k(0)=0, 0.30 would pin TIGHT to the root')


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', default='t2,t3,t1')
    a = ap.parse_args()
    for s in a.stage.split(','):
        {'t1': gate_bitid, 't2': gate_teachers, 't3': gate_lambda}[s.strip()]()
    print(f'\n=== {len(G_PASS)}/{len(G_PASS)+len(G_FAIL)} T15 gates pass ===')
    for f in G_FAIL:
        print(f'  FAILED: {f}')
    sys.exit(1 if G_FAIL else 0)
