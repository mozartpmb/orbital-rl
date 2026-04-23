"""Expand a Phase 2 orbital ckpt (obs_dim=29) to Phase 3 (obs_dim=33).

Phase 3 inserts 4 new obs at indices [13-16] (dθ_phase sin/cos, target.θ sin/cos).
The body block base shifted 13+k*4 → 17+k*4, so encoder input columns remap:
    new_cols[0:13]   ← old_cols[0:13]
    new_cols[13:17]  ← 0 (new inputs, zero-init)
    new_cols[17:33]  ← old_cols[13:29]

Usage:
    python scripts/orbital/expand_ckpt_p2_to_p3.py <phase2_ckpt.pt> <out_ckpt.pt>
"""

import sys
import torch


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    src, dst = sys.argv[1], sys.argv[2]

    sd = torch.load(src, map_location='cpu', weights_only=True)
    w_old = sd['policy.encoder.0.weight']
    assert w_old.shape == (128, 29), f"expected (128, 29), got {tuple(w_old.shape)}"

    w_new = torch.zeros(128, 33, dtype=w_old.dtype)
    w_new[:, 0:13]  = w_old[:, 0:13]
    w_new[:, 17:33] = w_old[:, 13:29]
    # columns 13:17 stay zero — new obs channels see no signal until retrained

    sd['policy.encoder.0.weight'] = w_new
    torch.save(sd, dst)
    print(f"encoder weight: {tuple(w_old.shape)} -> {tuple(w_new.shape)}")
    print(f"saved {dst}")


if __name__ == '__main__':
    main()
