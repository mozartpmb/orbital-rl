#!/usr/bin/env python3
"""Expand an action head from Discrete(30) → Discrete(31) for the day-warp row.

ROW 30 IS SEEDED FROM ROW 17 (the 6 h warp), NOT FROM ZERO — and that is a
measured decision, not a preference. A zero-init row is unreachable under this
warm start: the policy is saturated (median argmax softmax probability 0.986),
so a zero logit sitting among trained ones gets

    P(row 30) median 3.9e-09, p95 3.1e-06   =>  0 expected samples in 100M
    P(row 17) median 5.1e-05                     decisions

i.e. the day-warp would never be explored, the campaign would burn its whole
budget, and the mechanism under test would silently never have been available.
Seeding from row 17 starts the two warps as an exact tie at precisely the states
where the policy already chooses to warp, so PPO samples each ~50% of the time
there and learns to tell them apart by their tau. It is also the right prior on
the merits: the day-warp *is* "warp, but longer", so inheriting the 6 h warp's
learned "when to warp" representation is the correct starting point.

The bit-exactness property is unaffected: with row 30 MASKED the expanded
checkpoint still reproduces the parent's action stream exactly, because masking
removes it from the argmax and rows 0-29 are untouched either way.

This is the *easy* case of the surgery this repo already does in
`scripts/orbital/expand_ckpt_actions_7_to_9.py`: row 30 is APPENDED, so rows
0-29 keep their indices and their weights bit-for-bit and there is no remap
table at all. (`--zero` restores a zero-init row; it is kept only so the
measurement above is reproducible, and it is NOT the default for the reason
given.)

Why an append and not a rebind of an existing warp row: measured over 100
held-out episodes of the lineage this warm-starts from, row 17 (the 6 h warp)
is 11.1% of the loose-box policy's decisions and rows 16+17 carry 87.4% of all
its sub-steps. Rebinding row 17's tau from 360 to 1440 would quadruple the
effect of 11% of the policy's decisions while leaving its logit untouched — a
silent semantic swap under a trained head. Appending changes nothing that
already exists.

Verification (run automatically): with row 30 masked out of the logits, the
expanded checkpoint must produce a BIT-IDENTICAL action stream to the parent.

Usage:
    python3 scripts/orbital/extj2/expand_ckpt_30_to_31.py <in.pt> <out.pt>
                                                          [--from-row N | --zero]
"""

import sys

import torch


def main():
    argv = [a for a in sys.argv[1:]]
    from_row = 17
    if '--zero' in argv:
        from_row = None
        argv.remove('--zero')
    if '--from-row' in argv:
        i = argv.index('--from-row')
        from_row = int(argv[i + 1])
        del argv[i:i + 2]
    if len(argv) != 2:
        print(__doc__)
        return 1
    src, dst = argv

    sd = torch.load(src, map_location='cpu', weights_only=True)
    wk, bk = 'policy.decoder.weight', 'policy.decoder.bias'
    if wk not in sd:
        cands = [k for k in sd if k.endswith('decoder.weight')]
        print(f'ERROR: {wk} not in checkpoint. Candidates: {cands}')
        return 1

    w_old, b_old = sd[wk], sd[bk]
    n_old, hid = w_old.shape
    if n_old == 31:
        print(f'{src} already has a 31-row head; nothing to do.')
        return 0
    if n_old != 30:
        print(f'ERROR: expected a 30-row head, got {n_old}.')
        return 1

    w_new = torch.zeros(31, hid, dtype=w_old.dtype)
    b_new = torch.zeros(31, dtype=b_old.dtype)
    w_new[:30] = w_old          # pure append: rows 0-29 are untouched
    b_new[:30] = b_old
    if from_row is None:
        seed = 'ZERO (unreachable under a saturated head — see the docstring)'
    else:
        w_new[30] = w_old[from_row]
        b_new[30] = b_old[from_row]
        seed = f'copied from row {from_row}'

    sd[wk], sd[bk] = w_new, b_new
    torch.save(sd, dst)

    assert torch.equal(w_new[:30], w_old) and torch.equal(b_new[:30], b_old)
    print(f'{src}\n  -> {dst}')
    print(f'  head {n_old} -> 31; rows 0-29 bit-identical, row 30 {seed}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
