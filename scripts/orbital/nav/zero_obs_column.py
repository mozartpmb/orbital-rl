#!/usr/bin/env python3
"""Zero one observation column of a checkpoint's first encoder layer.

Why this exists. The T-BO+Sigma arm lights obs[21], which is one of the twelve
slots (obs[21..32]) that are identically zero in every nav/T3 config because
`num_debris = 0` hard-zeroes three body blocks in `fill_observations`. Those
columns of `policy.encoder.0.weight` therefore received EXACTLY ZERO gradient
during all prior training and still hold their random initialisation. Feeding a
live signal through them perturbs the warm-started policy for reasons that have
nothing to do with the signal's content — red-team NON-ISSUE-9 measured the
argmax flipping on 0/836 canonical decisions at magnitude 0.1, 6/836 at 0.5,
and 26/836 (3.11%) at 1.0.

Zeroing the column makes the channel provably inert at t=0: the warm start is
then bit-identical in behaviour to the un-instrumented one, and any subsequent
difference is attributable to the channel being LEARNED rather than to a random
projection of it. Without this, T-BO+Sigma is confounded at step 0 and a
positive result would be uninterpretable.

Usage:
    zero_obs_column.py IN.pt OUT.pt [--col 21] [--layer policy.encoder.0.weight]
"""

import argparse

import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument('src')
    p.add_argument('dst')
    p.add_argument('--col', type=int, default=21)
    p.add_argument('--layer', default='policy.encoder.0.weight')
    a = p.parse_args()

    sd = torch.load(a.src, map_location='cpu', weights_only=True)
    if a.layer not in sd:
        raise SystemExit(f'{a.layer!r} not in checkpoint; keys: {list(sd)[:8]}')
    W = sd[a.layer]
    if W.dim() != 2:
        raise SystemExit(f'{a.layer} is not a matrix: {tuple(W.shape)}')
    if not (0 <= a.col < W.shape[1]):
        raise SystemExit(f'col {a.col} out of range for {tuple(W.shape)}')
    before = W[:, a.col].abs().max().item()
    norm_before = W.norm().item()
    W[:, a.col] = 0.0
    sd[a.layer] = W
    torch.save(sd, a.dst)
    print(f'{a.layer}{tuple(W.shape)}  column {a.col}: '
          f'max|w| {before:.6f} -> 0.0   ‖W‖ {norm_before:.6f} -> '
          f'{W.norm().item():.6f}')
    print(f'wrote {a.dst}')


if __name__ == '__main__':
    main()
