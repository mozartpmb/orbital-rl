#!/usr/bin/env python3
"""Run PufferLib's CLI against THIS WORKTREE's pufferlib, not the editable install.

`pip install -e` registered pufferlib at /Users/pete/space_training/pufferlib —
the MAIN checkout — so the `puffer` console script trains the main checkout's C
extension and reads the main checkout's orbital.ini. That is silently wrong for
an ext-3d run: the new kwargs do not exist there, so the flags come back as
"unrecognized arguments", and had they matched a stale name the run would have
looked fine while testing the wrong binary.

Usage is identical to `puffer`:
    python3 scripts/orbital/ext3d/puffer_wt.py train puffer_orbital --train...
"""
import os
import sys

WT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  '..', '..', '..', 'pufferlib'))
sys.path.insert(0, WT)

import pufferlib                                                   # noqa: E402

if not os.path.realpath(pufferlib.__file__).startswith(os.path.realpath(WT)):
    raise SystemExit(f"refusing to run: pufferlib resolved to "
                     f"{pufferlib.__file__}, expected under {WT}")

from pufferlib.ocean.orbital.orbital import TRAJ_FLOATS            # noqa: E402
assert TRAJ_FLOATS == 100, f"stale build: TRAJ_FLOATS={TRAJ_FLOATS}, expected 100"

from pufferlib.pufferl import main                                 # noqa: E402

if __name__ == '__main__':
    main()
