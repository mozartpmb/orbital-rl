#!/usr/bin/env python3
"""T11 — can a NARROW-family checkpoint be transplanted EXACTLY into WIDE?

GEN_MATRIX's largest effect is the normalizer barrier: narrow-trained encoders
in wide normalizers collide (25-33% of episodes), wide-trained encoders in
narrow normalizers idle (100% timeout), and neither recovers under perfect
truth. The read concludes the families must be unified before any mixture.

But the barrier may be an ARTIFACT OF THE PARAMETERISATION rather than a
learned incompatibility. `Default.encode_observations` applies exactly one
`nn.Linear` to the raw observation vector. So if a normalizer change multiplies
obs channel j by c_j, dividing `encoder.weight[:, j]` by c_j leaves the encoder
OUTPUT unchanged — and therefore the LSTM state, the value head and the action
logits unchanged. The policy would be behaviourally identical on any scenario
expressible in both families.

Channels that move between the families (everything else is dimensionless or
normalised by a quantity both families share):

    obs[0], obs[7]        / obs_alt_scale_m          1.6e6 -> 8.0e6
    obs[17], obs[20]      / (R_EARTH + obs_alt_scale_m)  7.971e6 -> 14.371e6
    obs[24], obs[33,34]   / lvlh_scale_m             6.371e6 -> 1.5e7

obs[21,22,23] use di/de scales that are per-rung, and obs[15] is a dimensionless
clock whose UNIT differs when episode_cap_steps differs — both are held equal
here so the test isolates the normalizer question.

Run:
    PYTHONPATH=<worktree>/pufferlib python3 scripts/orbital/ext_recon/t11_transplant_probe.py
"""

import math
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
WT = os.path.abspath(os.path.join(_HERE, '..', '..', '..'))
sys.path.insert(0, os.path.join(WT, 'pufferlib'))
sys.path.insert(0, os.path.join(WT, 'scripts', 'orbital', 'nav'))

import pufferlib                                                     # noqa: E402
if not os.path.abspath(pufferlib.__file__).startswith(os.path.abspath(WT) + os.sep):
    raise SystemExit('REFUSING TO RUN: pufferlib is not the worktree build')

import importlib.util                                                # noqa: E402
_spec = importlib.util.spec_from_file_location(
    'ev', os.path.join(WT, 'scripts', 'orbital', 'nav', 'eval_relnav3d.py'))
ev = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ev)

R_EARTH = 6.371e6
NARROW_ALT, WIDE_ALT = 1.6e6, 8.0e6
NARROW_LVLH, WIDE_LVLH = 6.371e6, 1.5e7


def transplant(src, dst, alt_from=NARROW_ALT, alt_to=WIDE_ALT,
               lvlh_from=NARROW_LVLH, lvlh_to=WIDE_LVLH):
    """Rescale the encoder's input columns so the network is invariant."""
    sd = torch.load(src, map_location='cpu', weights_only=True)
    # The encoder is Sequential(Linear(38,128), GELU) -> 'policy.encoder.0.weight'.
    # Match on the tensor that consumes the raw 38-dim observation, not on a name.
    key = None
    for k, v in sd.items():
        if 'encoder' in k and v.dim() == 2 and v.shape[1] == 38:
            key = k
            break
    if key is None:
        raise SystemExit(f'no encoder.weight in {src}: {list(sd)[:8]}')
    W = sd[key].clone()

    dist_from, dist_to = R_EARTH + alt_from, R_EARTH + alt_to
    # obs_new = obs_old * f  =>  weight column must be divided by f
    groups = {
        (0, 7): alt_from / alt_to,
        (17, 20): dist_from / dist_to,
        (24, 33, 34): lvlh_from / lvlh_to,
    }
    for cols, f in groups.items():
        for c in cols:
            W[:, c] = W[:, c] / f
    sd[key] = W
    torch.save(sd, dst)
    return {c: f for cols, f in groups.items() for c in cols}


def main():
    ck = os.path.join(WT, 'models/t3/extj2_A3b_j2_box5k1.pt')
    if not os.path.exists(ck):
        raise SystemExit(f'missing {ck}')
    out = '/tmp/t11_A3b_wide.pt'
    fac = transplant(ck, out)
    print('=== T11 transplant probe: narrow -> wide, exactly ===')
    print(f'    src {os.path.basename(ck)}')
    print('    obs channel scale factors applied to encoder.weight columns:')
    for c in sorted(fac):
        print(f'      obs[{c:2d}]  obs_new/obs_old = {fac[c]:.6f}  '
              f'=> weight column x {1/fac[c]:.4f}')

    # The SAME physical task, stated in both families. Identical bands, boxes,
    # di/de scales and cap: only the normalizers move.
    COMMON = dict(
        num_debris_min=0, num_debris_max=0,
        e_max_target=0.05, e_max_sat=0.05, same_orbit_init=0,
        init_phase_gap_max=3.14159, valid_init_only=1,
        gave_up_action='terminate', max_valid_init_attempts=4096,
        rendezvous_radius_m=30000.0, rel_vel_tol_ms=50.0,
        shaping_mode=2, shape_w_lambda=1.0, shape_w_match=0.8166667,
        shape_dv_ref_ms=700.0, shape_gamma=1.0,
        phase_gap_mode=1, phase_obs_mode=1,
        episode_cap_steps=3000, cap_terminal_reward=0.0,
        dim3_mode=1, di_max_rad=0.017453, legacy_action_space=30,
        a_min_override=6.671e6, a_max_override=7.171e6,
    )
    NARROW_KW = dict(COMMON, obs_alt_scale_m=NARROW_ALT, lvlh_scale_m=NARROW_LVLH)
    WIDE_KW = dict(COMMON, obs_alt_scale_m=WIDE_ALT, lvlh_scale_m=WIDE_LVLH)

    N = 100
    print(f'\n  X3-loose task, truth mode, {N} eps, seed 123:')
    ev.rollout._mask_rows = None
    a = ev.rollout(ev.make_env('X3', 'truth', **NARROW_KW), ck, N, 123, 'orig/narrow')
    ev.rollout._mask_rows = None
    b = ev.rollout(ev.make_env('X3', 'truth', **WIDE_KW), ck, N, 123, 'orig/WIDE (no transplant)')
    ev.rollout._mask_rows = None
    c = ev.rollout(ev.make_env('X3', 'truth', **WIDE_KW), out, N, 123, 'TRANSPLANT/WIDE')

    print(f'\n  {"arm":34s} {"success":>10s} {"md5":>14s}')
    for nm_, r in (('original @ narrow (home)', a),
                   ('original @ wide (the barrier)', b),
                   ('TRANSPLANTED @ wide', c)):
        print(f'  {nm_:34s} {r["success"]:4d}/{r["n_valid"]:<4d} {r["md5"][:12]:>14s}')
    print(f'\n  transplant reproduces home: '
          f'{"YES" if c["md5"] == a["md5"] else "no (md5 differs)"}   '
          f'({c["success"]}/{c["n_valid"]} vs {a["success"]}/{a["n_valid"]})')
    print(f'  the barrier is real without it: original loses '
          f'{a["success"] - b["success"]} episodes moving to wide')
    return 0


if __name__ == '__main__':
    sys.exit(main())
