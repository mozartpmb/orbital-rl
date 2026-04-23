"""Expand action head from Discrete(7) → Discrete(9) by inserting two fine actions.

Old action ordering (7):
    0 coast, 1 pro 10, 2 pro 25, 3 retro 10, 4 retro 25, 5 rad+, 6 rad-

New action ordering (9):
    0 coast, 1 pro 5 (new), 2 pro 10, 3 pro 25,
             4 retro 5 (new), 5 retro 10, 6 retro 25, 7 rad+, 8 rad-

Map: old→new = [0, 2, 3, 5, 6, 7, 8]; new rows 1,4 zero-init.

Usage:
    python scripts/orbital/expand_ckpt_actions_7_to_9.py <in.pt> <out.pt>
"""

import sys
import torch


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    src, dst = sys.argv[1], sys.argv[2]

    sd = torch.load(src, map_location='cpu', weights_only=True)
    w_old = sd['policy.decoder.weight']
    b_old = sd['policy.decoder.bias']
    assert w_old.shape == (7, 128), f"expected (7, 128), got {tuple(w_old.shape)}"

    old_to_new = [0, 2, 3, 5, 6, 7, 8]
    w_new = torch.zeros(9, 128, dtype=w_old.dtype)
    b_new = torch.zeros(9,      dtype=b_old.dtype)
    for old_i, new_i in enumerate(old_to_new):
        w_new[new_i] = w_old[old_i]
        b_new[new_i] = b_old[old_i]
    # Rows 1 (pro 5) and 4 (retro 5) remain zero — unbiased exploration at start.

    sd['policy.decoder.weight'] = w_new
    sd['policy.decoder.bias']   = b_new
    torch.save(sd, dst)
    print(f"decoder: weight {tuple(w_old.shape)} -> {tuple(w_new.shape)}, bias ({b_old.shape[0]},) -> ({b_new.shape[0]},)")
    print(f"saved {dst}")


if __name__ == '__main__':
    main()
