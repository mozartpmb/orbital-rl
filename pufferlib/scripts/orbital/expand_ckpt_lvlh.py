"""Expand a 33-dim-obs checkpoint to 38-dim by zero-padding the encoder input.

Phase 4 R1 needs warm-start from Stage 2 (33-dim) but LVLH obs space is 38-dim.
Zero-init on the 5 new columns preserves exact 33-dim policy behavior at step 0.

Usage:
    python3 scripts/orbital/expand_ckpt_lvlh.py \
        experiments/puffer_orbital_q1pj876v/model_puffer_orbital_000115.pt \
        experiments/puffer_orbital_q1pj876v/model_000115_lvlh38.pt
"""
import argparse, sys, torch

def main():
    p = argparse.ArgumentParser()
    p.add_argument("src")
    p.add_argument("dst")
    p.add_argument("--old-dim", type=int, default=33)
    p.add_argument("--new-dim", type=int, default=38)
    p.add_argument("--encoder-key", default="policy.encoder.0.weight")
    args = p.parse_args()

    sd = torch.load(args.src, map_location="cpu", weights_only=True)
    if args.encoder_key not in sd:
        print(f"ERROR: {args.encoder_key} not in state_dict. Keys:")
        for k in sd.keys(): print("  ", k, tuple(sd[k].shape) if hasattr(sd[k],'shape') else '')
        sys.exit(1)

    w = sd[args.encoder_key]
    hidden_size, in_dim = w.shape
    if in_dim != args.old_dim:
        print(f"ERROR: encoder input dim {in_dim} != expected {args.old_dim}")
        sys.exit(1)

    pad = torch.zeros(hidden_size, args.new_dim - args.old_dim, dtype=w.dtype)
    sd[args.encoder_key] = torch.cat([w, pad], dim=1)
    print(f"Expanded {args.encoder_key}: {tuple(w.shape)} -> {tuple(sd[args.encoder_key].shape)}")

    torch.save(sd, args.dst)
    print(f"Saved: {args.dst}")

if __name__ == "__main__":
    main()
