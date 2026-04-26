"""Phase 4.5 Ablation C: dual surgery on a Phase 3 D7 checkpoint.

Expands a checkpoint trained with 33-dim observations and Discrete(7) actions
to the Phase 4 obs/action shape (38-dim, Discrete(10)). Two surgeries:

1. Encoder: pad input columns (33 -> 38). New columns init to zero so step-0
   behavior is identical for the LVLH inputs (which the policy has not yet
   learned to use).

2. Decoder: pad output rows (7 -> 10). New rows are zero-init weight + a
   user-tunable bias (-1.0 default per spec ~3% softmax prior on warp).

Saves a state_dict that loads cleanly into the Phase 4 Default+LSTMWrapper.

Usage:
  python expand_ckpt_actions_d7_to_d10.py SRC DST [--warp-bias -1.0]
"""
import argparse
import torch


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("src", help="source D7+33-dim ckpt")
    ap.add_argument("dst", help="destination D10+38-dim ckpt")
    ap.add_argument("--old-actions", type=int, default=7)
    ap.add_argument("--new-actions", type=int, default=10)
    ap.add_argument("--old-obs", type=int, default=33)
    ap.add_argument("--new-obs", type=int, default=38)
    ap.add_argument("--warp-bias", type=float, default=-1.0,
                    help="bias for new (warp) action logits; -1.0 ~3% prior")
    ap.add_argument("--encoder-key", default="policy.encoder.0.weight")
    ap.add_argument("--decoder-w-key", default="policy.decoder.weight")
    ap.add_argument("--decoder-b-key", default="policy.decoder.bias")
    args = ap.parse_args()

    sd = torch.load(args.src, weights_only=True, map_location="cpu")

    # ---- Encoder: pad input columns ----
    e_w = sd[args.encoder_key]
    assert e_w.shape[1] == args.old_obs, \
        f"encoder col mismatch: got {tuple(e_w.shape)} expected (*, {args.old_obs})"
    new_e_w = torch.zeros(e_w.shape[0], args.new_obs, dtype=e_w.dtype)
    new_e_w[:, :args.old_obs] = e_w
    sd[args.encoder_key] = new_e_w

    # ---- Decoder: pad output rows ----
    d_w = sd[args.decoder_w_key]
    d_b = sd[args.decoder_b_key]
    assert d_w.shape[0] == args.old_actions, \
        f"decoder row mismatch: got {tuple(d_w.shape)} expected ({args.old_actions}, *)"
    new_d_w = torch.zeros(args.new_actions, d_w.shape[1], dtype=d_w.dtype)
    new_d_w[:args.old_actions] = d_w
    new_d_b = torch.full((args.new_actions,), args.warp_bias, dtype=d_b.dtype)
    new_d_b[:args.old_actions] = d_b
    sd[args.decoder_w_key] = new_d_w
    sd[args.decoder_b_key] = new_d_b

    torch.save(sd, args.dst)
    print(f"Wrote {args.dst}")
    print(f"  encoder {args.old_obs} -> {args.new_obs} (zero pad)")
    print(f"  decoder {args.old_actions} -> {args.new_actions} (zero weight, bias={args.warp_bias})")


if __name__ == "__main__":
    main()
