#!/usr/bin/env python3
"""ext-3d anchor A2, part 2 — the float32 REWARD form of the bit-exact anchor.

`a2_bitexact.c` proves Φ(mode 2) == Φ(mode 1) at Δi = 0 in double. This proves
the property survives the cast the trainer actually sees: identical seeds and
identical action sequences under

    (A) shaping_mode=1, dim3_mode=0                    — the 2D lineage
    (B) shaping_mode=2, dim3_mode=1, di_max_rad=-1.0   — 3D code path, Δi = 0

must produce bitwise-identical rewards, terminals, and obs slots 0-20 / 33-37.
Slots 21-32 are the 3D block and are *expected* to differ from the (all-zero)
body slots — they are reported separately, not as a failure.

The red-team measured the naive spelling breaking here at ~1 reward in 10^4,
which is invisible in a spot check and fatal over a 200-episode anchor.

Run (from the pufferlib dir):
    python3 ../scripts/orbital/ext3d/a2_reward_f32.py --steps 1000 --envs 64
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', '..', '..', 'pufferlib'))
from pufferlib.ocean.orbital.orbital import Orbital  # noqa: E402

T3 = dict(
    num_debris_min=0, num_debris_max=0,
    e_max_target=0.05, e_max_sat=0.05, init_phase_gap_max=3.14159265358979,
    valid_init_only=1, gave_up_action="terminate",
    shape_gamma=1.0, phase_gap_mode=1, phase_obs_mode=1,
    episode_cap_steps=3000, cap_terminal_reward=0.0,
    shape_w_lambda=1.0, shape_w_match=0.8166667, shape_dv_ref_ms=700.0,
    legacy_action_space=30,
)


def rollout(kw, n_envs, steps, seed, actions):
    env = Orbital(num_envs=n_envs, seed=seed, **kw)
    env.reset(seed=seed)
    R = np.zeros((steps, n_envs), dtype=np.float32)
    D = np.zeros((steps, n_envs), dtype=np.uint8)
    O = np.zeros((steps, n_envs, 38), dtype=np.float32)
    for t in range(steps):
        obs, rew, term, trunc, _ = env.step(actions[t])
        R[t] = rew
        D[t] = term
        O[t] = obs
    env.close()
    return R, D, O


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--steps', type=int, default=1000)
    p.add_argument('--envs', type=int, default=64)
    p.add_argument('--seed', type=int, default=4242)
    a = p.parse_args()

    rng = np.random.default_rng(a.seed)
    # Include the new normal + combined actions: they are all no-ops on the
    # PLANE at Δi = 0 only if the normal axis reduces correctly, so exercising
    # them here is deliberate... except that a normal burn legitimately tilts
    # the plane and breaks the anchor by design (3d_REDTEAM m7). So the anchor
    # run uses the legacy in-plane/warp action set, as specced.
    acts = rng.integers(0, 20, size=(a.steps, a.envs)).astype(np.int32)

    A = dict(T3); A.update(shaping_mode=1, dim3_mode=0)
    B = dict(T3); B.update(shaping_mode=2, dim3_mode=1, di_max_rad=-1.0)

    Ra, Da, Oa = rollout(A, a.envs, a.steps, a.seed, acts)
    Rb, Db, Ob = rollout(B, a.envs, a.steps, a.seed, acts)

    n = Ra.size
    rmm = int((Ra.view(np.uint32) != Rb.view(np.uint32)).sum())
    dmm = int((Da != Db).sum())
    keep = list(range(0, 21)) + list(range(33, 38))
    omm = int((Oa[:, :, keep].view(np.uint32)
               != Ob[:, :, keep].view(np.uint32)).sum())
    blk = Ob[:, :, 21:33]

    print(f"steps={a.steps} envs={a.envs} -> {n} one-step rewards")
    print(f"  reward  f32 bitwise mismatches : {rmm} / {n}  ({100.0*rmm/n:.6f}%)")
    print(f"  terminal        mismatches     : {dmm} / {n}")
    print(f"  obs[0-20,33-37] f32 mismatches : {omm} / {Oa[:, :, keep].size}")
    print(f"  max |Δreward|                  : {np.abs(Ra - Rb).max():.3e}")
    print(f"  3D block obs[21-32] range      : [{blk.min():+.4f}, {blk.max():+.4f}]"
          f"  (differs from the zero body slots BY DESIGN)")
    ok = (rmm == 0 and dmm == 0 and omm == 0)
    print("VERDICT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
