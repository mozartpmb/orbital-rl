#!/usr/bin/env python3
"""T13b anchor gates.

A1  ANCHOR-OFF IS BIT-IDENTICAL TO THE PRE-ANCHOR TRAINER.
    Not "inert", not "shouldn't matter" — identical weights after a fixed-seed
    run, against the pristine `pufferl.py` from origin/main. This is the gate
    that lets every existing lineage keep its meaning, and it is why the anchor
    path (including its RNG draw) sits behind a single guard: a stray
    `torch.randint` outside that guard would silently reorder every subsequent
    `torch.multinomial` minibatch draw and change training for a reason that has
    nothing to do with the anchor.

A2  THE ANCHOR LOSS IS ZERO AGAINST ITSELF. CE(anchor || anchor) == 0 to
    float tolerance when the live policy IS the anchor — the sanity check that
    the term is a divergence and not an arbitrary scalar. At warm start from the
    root this also means the anchor contributes no gradient on step 0, which is
    the correct initial condition for a CLEAR-style anchor.

A3  THE DATASET MATCHES THE TRAINER'S WINDOW CONVENTION. shape (N, horizon,
    obs_dim) with horizon == bptt_horizon. A mismatched horizon would silently
    anchor on windows the trainer never evaluates that way.

Run from the worktree:
    python3 scripts/orbital/extj2/t13b_anchor_gates.py --stage a2,a3
    python3 scripts/orbital/extj2/t13b_anchor_gates.py --stage a1   # slow
"""
import argparse
import hashlib
import os
import subprocess
import time
import sys

import numpy as np
import torch

WT = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
sys.path.insert(0, os.path.join(WT, 'pufferlib'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

G_PASS, G_FAIL = [], []


def check(name, ok, detail=''):
    (G_PASS if ok else G_FAIL).append(name)
    print(f'  [{"PASS" if ok else "FAIL"}] {name}')
    if detail:
        print(f'         {detail}')


def ckpt_hash(path):
    sd = torch.load(path, map_location='cpu', weights_only=True)
    h = hashlib.md5()
    for k in sorted(sd):
        v = sd[k]
        h.update(k.encode())
        h.update(np.ascontiguousarray(v.detach().cpu().numpy()).tobytes())
    return h.hexdigest()


TRAIN_ARGS = [
    '--train.device', 'cpu', '--train.total-timesteps', '32768',
    '--train.seed', '424242', '--train.batch-size', '16384',
    '--train.bptt-horizon', '64', '--train.minibatch-size', '4096',
    '--vec.backend', 'Serial', '--vec.num-envs', '1', '--vec.num-workers', '1',
    '--env.num-envs', '256',
    '--env.num-debris-min', '0', '--env.num-debris-max', '0',
    '--env.same-orbit-init', '0', '--env.init-phase-gap-max', '3.14159',
    '--env.valid-init-only', '1', '--env.gave-up-action', 'terminate',
    '--env.obs-alt-scale-m', '8.0e6', '--env.lvlh-scale-m', '6.371e6',
    '--env.shaping-mode', '2', '--env.shape-w-lambda', '1.0',
    '--env.shape-w-match', '0.8166667', '--env.shape-dv-ref-ms', '700.0',
    '--env.shape-gamma', '1.0', '--env.phase-gap-mode', '1',
    '--env.phase-obs-mode', '1', '--env.cap-terminal-reward', '0.0',
    '--env.dim3-mode', '1', '--env.j2-mode', '1', '--env.nav-j2-mode', '1',
    '--env.lvlh-frame-mode', '1', '--env.legacy-action-space', '31',
    '--env.nav-mode', 'bearings_only', '--env.nav-max-ticks', '120',
    '--env.t11-mixture', '2',
]


def run_train(datadir, extra=(), seed_dir=None):
    """Run a short training and return its checkpoints.

    Two things this has to work around, both discovered the hard way:

    1. `pufferl.__init__` has `random.seed`, `np.random.seed` AND
       `torch.manual_seed` ALL COMMENTED OUT. Training is therefore
       nondeterministic run-to-run no matter what `--train.seed` says, so a
       naive weight-hash comparison can never pass. `seed_dir` injects a
       `sitecustomize.py` that seeds all three at interpreter start, which is
       the only way to get determinism without editing the trainer (and editing
       it would change the very thing under test).

    2. The trainer HANGS IN TEARDOWN after writing its final checkpoint. So we
       poll for the checkpoint and kill the process rather than waiting on it.
    """
    import signal as _sig
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               MKL_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1')
    pp = os.path.join(WT, 'pufferlib')
    env['PYTHONPATH'] = f'{seed_dir}:{pp}' if seed_dir else pp
    outdir = os.path.join(WT, 'pufferlib', datadir)
    cmd = [sys.executable, '-m', 'pufferlib.pufferl', 'train',
           'puffer_orbital_nav', '--train.data-dir', datadir] + TRAIN_ARGS + list(extra)
    proc = subprocess.Popen(cmd, cwd=pp, env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True,
                            preexec_fn=os.setsid)
    cks, waited = [], 0
    while waited < 1800:
        cks = [os.path.join(r, f) for r, _, fs in os.walk(outdir) for f in fs
               if f.startswith('model_') and f.endswith('.pt')]
        if cks or proc.poll() is not None:
            break
        time.sleep(5); waited += 5
    if cks:
        time.sleep(3)   # let the write settle before hashing
    try:
        os.killpg(os.getpgid(proc.pid), _sig.SIGTERM)
    except Exception:
        pass
    try:
        proc.wait(timeout=30)
    except Exception:
        try:
            os.killpg(os.getpgid(proc.pid), _sig.SIGKILL)
        except Exception:
            pass
    if not cks:
        print((proc.stdout.read() or '')[-2500:])
    return sorted(cks)


def _make_seed_dir():
    import tempfile
    d = tempfile.mkdtemp(prefix='t13b_seed_')
    with open(os.path.join(d, 'sitecustomize.py'), 'w') as f:
        f.write('import random, numpy, torch\n'
                'random.seed(424242)\n'
                'numpy.random.seed(424242)\n'
                'torch.manual_seed(424242)\n'
                'torch.use_deterministic_algorithms(False)\n')
    return d


def gate_a1():
    """Pristine trainer vs anchor-off trainer, with a determinism control.

    The control is not optional. `pufferl` does not seed torch (see run_train),
    so without externally seeding it, base-vs-base already differs and the
    comparison measures nothing. A1a establishes that the instrument can detect
    equality at all; only then does A1b mean anything.
    """
    print('\n== A1  anchor-off is BIT-IDENTICAL to the pre-anchor trainer ==')
    import shutil
    sd = _make_seed_dir()
    puf = os.path.join(WT, 'pufferlib', 'pufferlib', 'pufferl.py')
    ini = os.path.join(WT, 'pufferlib', 'pufferlib', 'config', 'ocean',
                       'orbital_nav.ini')
    # Keep a copy on DISK, not just in memory: an earlier revision held the
    # modified files only in local variables, so killing the gate mid-run would
    # have left the pristine versions in place and silently discarded the work.
    bak = os.path.join(sd, 'bak'); os.makedirs(bak, exist_ok=True)
    shutil.copy2(puf, os.path.join(bak, 'pufferl.py'))
    shutil.copy2(ini, os.path.join(bak, 'orbital_nav.ini'))
    for d in ('GATE_B1', 'GATE_B2', 'GATE_OFF'):
        shutil.rmtree(os.path.join(WT, 'pufferlib', 'experiments_t13b', d),
                      ignore_errors=True)
    try:
        for rel, dst in (('pufferlib/pufferlib/pufferl.py', puf),
                         ('pufferlib/pufferlib/config/ocean/orbital_nav.ini', ini)):
            base = subprocess.run(['git', 'show', f'origin/main:{rel}'],
                                  cwd=WT, capture_output=True, text=True).stdout
            open(dst, 'w').write(base)
        b1 = run_train('experiments_t13b/GATE_B1', seed_dir=sd)
        b2 = run_train('experiments_t13b/GATE_B2', seed_dir=sd)
        h1 = ckpt_hash(b1[-1]) if b1 else None
        h2 = ckpt_hash(b2[-1]) if b2 else None
    finally:
        shutil.copy2(os.path.join(bak, 'pufferl.py'), puf)
        shutil.copy2(os.path.join(bak, 'orbital_nav.ini'), ini)

    check('A1a the instrument is deterministic (pristine vs pristine)',
          h1 is not None and h1 == h2,
          f'base#1 {h1} vs base#2 {h2} — pufferl leaves torch/numpy/random '
          f'UNSEEDED, so this only holds with the injected sitecustomize seed')
    if h1 is None or h1 != h2:
        check('A1b anchor_lambda=0 reproduces the pre-anchor trainer', False,
              'SKIPPED — instrument not deterministic, so the comparison '
              'would be meaningless either way')
        return
    a = run_train('experiments_t13b/GATE_OFF', ['--train.anchor-lambda', '0.0'],
                  seed_dir=sd)
    ha = ckpt_hash(a[-1]) if a else None
    check('A1b anchor_lambda=0 reproduces the pre-anchor trainer bit-for-bit',
          ha is not None and ha == h1, f'pristine {h1} vs anchor-off {ha}')


def gate_a2():
    """CE(anchor || anchor) == 0 when the live policy is the anchor."""
    print('\n== A2  the anchor term is a divergence (zero against itself) ==')
    from pufferlib.models import Default, LSTMWrapper
    from pufferlib.ocean.orbital_nav.orbital_nav import OrbitalNav
    import t11_cells as T
    root = '/Users/pete/space_training/models/t3/t11t_tight_child.pt'
    data = os.path.join(WT, 'models/t13b/t11b_anchor_tight.pt')
    obs = torch.load(data, map_location='cpu', weights_only=False)['obs']
    env = OrbitalNav(**T.nav_env_kwargs(num_envs=1, nav_mode='bearings_only',
                                        t11_mixture=1, j2_mode=1, nav_j2_mode=1))
    pol = LSTMWrapper(env, Default(env))
    pol.load_state_dict(torch.load(root, map_location='cpu', weights_only=True))
    pol.eval()
    import copy
    anc = copy.deepcopy(pol)
    mb = obs[:8]
    with torch.no_grad():
        la, _ = anc(mb, dict(action=None, lstm_h=None, lstm_c=None))
        lt, _ = pol(mb, dict(action=None, lstm_h=None, lstm_c=None))
    if isinstance(la, (list, tuple)):
        la, lt = la[0], lt[0]
    alp = torch.log_softmax(la.float(), -1)
    tlp = torch.log_softmax(lt.float(), -1)
    ce = (alp.exp() * (alp - tlp)).sum(-1).mean().item()
    env.close()
    check('A2 CE(anchor||anchor) == 0 at warm start', abs(ce) < 1e-6,
          f'CE = {ce:.3e} (so the anchor adds no gradient at step 0, the '
          f'correct initial condition for a CLEAR-style anchor)')


def gate_a3():
    print('\n== A3  dataset matches the trainer window convention ==')
    data = os.path.join(WT, 'models/t13b/t11b_anchor_tight.pt')
    blob = torch.load(data, map_location='cpu', weights_only=False)
    obs, hz = blob['obs'], int(blob.get('horizon', -1))
    import configparser
    cp = configparser.ConfigParser(inline_comment_prefixes=('#', ';'))
    cp.read(os.path.join(WT, 'pufferlib/pufferlib/config/default.ini'))
    bptt = int(cp['train']['bptt_horizon'])
    check('A3 anchor windows are (N, bptt_horizon, obs_dim)',
          obs.ndim == 3 and obs.shape[1] == hz == bptt,
          f'shape {tuple(obs.shape)}; dataset horizon {hz}; ini bptt_horizon '
          f'{bptt}; cell {blob.get("cell")}; from {blob.get("ckpt")}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', default='a2,a3')
    a = ap.parse_args()
    for s in a.stage.split(','):
        {'a1': gate_a1, 'a2': gate_a2, 'a3': gate_a3}[s.strip()]()
    print(f'\n=== {len(G_PASS)}/{len(G_PASS) + len(G_FAIL)} anchor gates pass ===')
    for f in G_FAIL:
        print(f'  FAILED: {f}')
    sys.exit(1 if G_FAIL else 0)
