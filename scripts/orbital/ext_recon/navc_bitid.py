#!/usr/bin/env python3
"""C1: nav_filter_impl='py' is BIT-IDENTICAL to the pre-port tree.

The port adds a dispatch to `stm_fd_j2`, a `set_filter_impl` call in
`OrbitalNav.__init__`, and an ini key. None of that changes the 'py'
arithmetic — but "none of that changes the arithmetic" is exactly the sentence
that has been wrong twice in this project, so it is measured against the
pristine tree rather than asserted.

Reuses the anchor battery's machinery deliberately, including its determinism
control: `pufferl` leaves random/np.random/torch UNSEEDED (all three seed calls
are commented out), so a weight-hash comparison is meaningless until a
sitecustomize seed is injected. C1a proves the instrument can detect equality
before C1b claims it — without that, a PASS would be indistinguishable from
two runs that happened to agree.
"""
import os
import shutil
import subprocess
import sys

WT = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
sys.path.insert(0, os.path.join(WT, 'scripts', 'orbital', 'extj2'))

import t13b_anchor_gates as A                                    # noqa: E402

G_PASS, G_FAIL = [], []


def check(name, ok, detail=''):
    (G_PASS if ok else G_FAIL).append(name)
    print(f'  [{"PASS" if ok else "FAIL"}] {name}')
    if detail:
        print(f'         {detail}')


TOUCHED = [
    'pufferlib/pufferlib/ocean/orbital_nav/nav_math3d.py',
    'pufferlib/pufferlib/ocean/orbital_nav/orbital_nav.py',
    'pufferlib/pufferlib/config/ocean/orbital_nav.ini',
]


def main():
    print('== C1  nav_filter_impl="py" vs the pre-port tree ==')
    sd = A._make_seed_dir()
    bak = os.path.join(sd, 'bak')
    os.makedirs(bak, exist_ok=True)
    abspaths = [os.path.join(WT, r) for r in TOUCHED]
    for r, p in zip(TOUCHED, abspaths):
        shutil.copy2(p, os.path.join(bak, os.path.basename(r)))
    for d in ('NAVC_B1', 'NAVC_B2', 'NAVC_PY'):
        shutil.rmtree(os.path.join(WT, 'pufferlib', 'experiments_navc', d),
                      ignore_errors=True)
    try:
        for r, p in zip(TOUCHED, abspaths):
            base = subprocess.run(['git', 'show', f'origin/main:{r}'], cwd=WT,
                                  capture_output=True, text=True).stdout
            open(p, 'w').write(base)
        b1 = A.run_train('experiments_navc/NAVC_B1', seed_dir=sd)
        b2 = A.run_train('experiments_navc/NAVC_B2', seed_dir=sd)
        h1 = A.ckpt_hash(b1[-1]) if b1 else None
        h2 = A.ckpt_hash(b2[-1]) if b2 else None
    finally:
        for r, p in zip(TOUCHED, abspaths):
            shutil.copy2(os.path.join(bak, os.path.basename(r)), p)

    check('C1a the instrument is deterministic (pristine vs pristine)',
          h1 is not None and h1 == h2,
          f'base#1 {h1} vs base#2 {h2}')
    if h1 is None or h1 != h2:
        check('C1b nav_filter_impl="py" reproduces the pre-port tree', False,
              'SKIPPED — instrument not deterministic')
    else:
        p = A.run_train('experiments_navc/NAVC_PY',
                        ['--env.nav-filter-impl', 'py'], seed_dir=sd)
        hp = A.ckpt_hash(p[-1]) if p else None
        check('C1b nav_filter_impl="py" reproduces the pre-port tree bit-for-bit',
              hp is not None and hp == h1, f'pristine {h1} vs py {hp}')
    print(f'\n=== {len(G_PASS)}/{len(G_PASS)+len(G_FAIL)} bit-identity gates pass ===')
    for f in G_FAIL:
        print(f'  FAILED: {f}')
    return 1 if G_FAIL else 0


if __name__ == '__main__':
    sys.exit(main())
