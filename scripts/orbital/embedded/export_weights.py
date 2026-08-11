#!/usr/bin/env python3
"""Export a trained orbital-rendezvous PufferLib policy to a C header.

Emits:
  policy_weights.h  — static const float arrays, C99 hex float literals
                      (exact bit round-trip; no decimal parsing ambiguity)
  policy_manifest.json — shapes, layer order, activations, provenance hashes

Architecture recovered from the checkpoint (NOT assumed — every shape below is
printed from the state_dict and asserted against the emitted arrays):

    obs[38] --> Linear(38,128) --> GELU(exact/erf) --> LSTMCell(128,128)
                                                          |
                                        +-----------------+-----------------+
                                        |                                   |
                                 Linear(128,16)                       Linear(128,1)
                                    logits                                value

The PufferLib `Default` policy applies NO observation normalization: its
encode_observations() only does view(B,-1).float() before the encoder Linear.
All normalization lives inside the C environment's fill_observations().
LSTMWrapper.forward_eval() runs the torch LSTMCell (not nn.LSTM); the two share
parameters by assignment (cell.weight_ih = lstm.weight_ih_l0, ...), so the
checkpoint stores both copies. We verify they are bit-identical and export once.

Usage:
    python3 scripts/orbital/embedded/export_weights.py \
        --ckpt models/t3/seed42_L2_headline.pt \
        --out-dir scripts/orbital/embedded
"""

import argparse
import hashlib
import json
import os
import struct
import sys
from datetime import datetime, timezone

import numpy as np
import torch

OBS_DIM_EXPECT = 38
HIDDEN_EXPECT = 128

# (state_dict key, C symbol, expected shape as a function of (obs, hidden, act))
LAYERS = [
    ("policy.encoder.0.weight", "POLICY_ENC_W", lambda o, h, a: (h, o)),
    ("policy.encoder.0.bias",   "POLICY_ENC_B", lambda o, h, a: (h,)),
    ("lstm.weight_ih_l0",       "POLICY_LSTM_W_IH", lambda o, h, a: (4 * h, h)),
    ("lstm.weight_hh_l0",       "POLICY_LSTM_W_HH", lambda o, h, a: (4 * h, h)),
    ("lstm.bias_ih_l0",         "POLICY_LSTM_B_IH", lambda o, h, a: (4 * h,)),
    ("lstm.bias_hh_l0",         "POLICY_LSTM_B_HH", lambda o, h, a: (4 * h,)),
    ("policy.decoder.weight",   "POLICY_DEC_W", lambda o, h, a: (a, h)),
    ("policy.decoder.bias",     "POLICY_DEC_B", lambda o, h, a: (a,)),
    ("policy.value.weight",     "POLICY_VAL_W", lambda o, h, a: (1, h)),
    ("policy.value.bias",       "POLICY_VAL_B", lambda o, h, a: (1,)),
]

# Parameters that must be bit-identical duplicates of an exported layer.
TIED = {
    "cell.weight_ih": "lstm.weight_ih_l0",
    "cell.weight_hh": "lstm.weight_hh_l0",
    "cell.bias_ih":   "lstm.bias_ih_l0",
    "cell.bias_hh":   "lstm.bias_hh_l0",
}


def f32_hex(x: float) -> str:
    """C99 hex float literal for an exactly-representable float32 value.

    Round-trips bit-exactly through any conforming C compiler, unlike decimal
    (%.9g is exact for IEEE754 binary32 in practice but relies on the compiler's
    correctly-rounded decimal->binary conversion).
    """
    f = float(np.float32(x))
    if f != f:
        return "NAN"
    if f in (float("inf"), float("-inf")):
        return "INFINITY" if f > 0 else "-INFINITY"
    if f == 0.0:
        return "-0.0f" if struct.pack(">f", f)[0] & 0x80 else "0.0f"
    return f.hex() + "f"


def emit_array(name: str, arr: np.ndarray, shape, fmt: str, per_line: int = 6):
    arr = np.ascontiguousarray(arr, dtype=np.float32).reshape(-1)
    lines = [
        f"/* {name}: shape {tuple(shape)}, row-major, {arr.size} floats */",
        f"static const float {name}[{arr.size}] = {{",
    ]
    if fmt == "hex":
        vals = [f32_hex(v) for v in arr]
    else:
        vals = [f"{np.float32(v):.9g}f" for v in arr]
    for i in range(0, len(vals), per_line):
        lines.append("    " + ", ".join(vals[i:i + per_line]) + ",")
    lines.append("};")
    return "\n".join(lines)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default="models/t3/seed42_L2_headline.pt")
    ap.add_argument("--out-dir", default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--format", choices=["hex", "decimal"], default="hex",
                    help="float literal format (hex = exact C99 round-trip)")
    ap.add_argument("--print-only", action="store_true",
                    help="print the state_dict shapes and exit (no files written)")
    args = ap.parse_args()

    ckpt = os.path.abspath(args.ckpt)
    sd = torch.load(ckpt, map_location="cpu", weights_only=True)

    print(f"checkpoint: {ckpt}")
    print(f"{'key':<26} {'shape':<14} dtype")
    total = 0
    for k, v in sd.items():
        print(f"{k:<26} {str(tuple(v.shape)):<14} {v.dtype}")
        total += v.numel()
    print(f"stored tensors: {total} elements (includes the tied LSTM duplicate)")

    # --- recover dimensions from the checkpoint, do not assume -----------------
    hidden = sd["policy.encoder.0.weight"].shape[0]
    obs_dim = sd["policy.encoder.0.weight"].shape[1]
    n_act = sd["policy.decoder.weight"].shape[0]
    print(f"\nrecovered dims: obs={obs_dim} hidden={hidden} actions={n_act}")
    if obs_dim != OBS_DIM_EXPECT or hidden != HIDDEN_EXPECT:
        print(f"  NOTE: differs from the T3 canonical ({OBS_DIM_EXPECT}, {HIDDEN_EXPECT})")

    # --- tie check -------------------------------------------------------------
    for dup, src in TIED.items():
        if dup not in sd:
            print(f"  WARN: tied key {dup} absent from checkpoint")
            continue
        if not torch.equal(sd[dup], sd[src]):
            raise SystemExit(
                f"FATAL: {dup} is not bit-identical to {src}. LSTMWrapper ties them "
                "by assignment; a divergence means the checkpoint was produced by a "
                "different wrapper and the export would silently use the wrong weights."
            )
    print("tie check: cell.* == lstm.* (bit-identical) OK")

    unique = sum(sd[k].numel() for k, _, _ in LAYERS)
    print(f"unique parameters: {unique}")

    if args.print_only:
        return

    # --- emit ------------------------------------------------------------------
    os.makedirs(args.out_dir, exist_ok=True)
    hdr_path = os.path.join(args.out_dir, "policy_weights.h")
    man_path = os.path.join(args.out_dir, "policy_manifest.json")

    ckpt_sha = sha256_file(ckpt)
    blocks, manifest_layers = [], []
    weight_hash = hashlib.sha256()

    for key, sym, shape_fn in LAYERS:
        t = sd[key]
        want = shape_fn(obs_dim, hidden, n_act)
        if tuple(t.shape) != want:
            raise SystemExit(f"FATAL: {key} shape {tuple(t.shape)} != expected {want}")
        if t.dtype != torch.float32:
            raise SystemExit(f"FATAL: {key} dtype {t.dtype} != float32")
        arr = t.detach().cpu().numpy().astype(np.float32)
        weight_hash.update(np.ascontiguousarray(arr).tobytes())
        blocks.append(emit_array(sym, arr, want, args.format))
        manifest_layers.append({
            "state_dict_key": key,
            "c_symbol": sym,
            "shape": list(want),
            "numel": int(arr.size),
            "layout": "row_major",
            "dtype": "float32",
            "abs_max": float(np.abs(arr).max()),
        })

    wsha = weight_hash.hexdigest()
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    header = f"""/* policy_weights.h — AUTO-GENERATED, DO NOT EDIT.
 *
 * Generated  : {stamp}
 * Generator  : scripts/orbital/embedded/export_weights.py
 * Checkpoint : {os.path.relpath(ckpt, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(ckpt)))))}
 * ckpt sha256: {ckpt_sha}
 * wts  sha256: {wsha}   (concatenated row-major float32 bytes, layer order below)
 *
 * Layer order / activations (see policy_manifest.json for the machine copy):
 *   0  encoder   Linear({obs_dim} -> {hidden})   act: GELU (exact, erf form)
 *   1  recurrent LSTMCell({hidden} -> {hidden})  gate order i,f,g,o (torch layout)
 *   2  decoder   Linear({hidden} -> {n_act})     act: none (raw logits, argmax)
 *   3  value     Linear({hidden} -> 1)           act: none (unused for control)
 *
 * No observation normalization is applied inside the policy: PufferLib's
 * Default.encode_observations() is view().float() -> Linear. Normalization is
 * the environment's job (orbital.h::fill_observations).
 *
 * Float literals are C99 hexadecimal ({args.format} format selected at export)
 * so the compiled constants are bit-identical to the checkpoint tensors.
 */
#ifndef ORBITAL_POLICY_WEIGHTS_H
#define ORBITAL_POLICY_WEIGHTS_H

#define POLICY_WEIGHTS_OBS_DIM   {obs_dim}
#define POLICY_WEIGHTS_HIDDEN    {hidden}
#define POLICY_WEIGHTS_ACTIONS   {n_act}
#define POLICY_WEIGHTS_CKPT_SHA  "{ckpt_sha}"
#define POLICY_WEIGHTS_BLOB_SHA  "{wsha}"
#define POLICY_WEIGHTS_STAMP     "{stamp}"

"""
    footer = "\n#endif /* ORBITAL_POLICY_WEIGHTS_H */\n"

    with open(hdr_path, "w") as fh:
        fh.write(header)
        fh.write("\n\n".join(blocks))
        fh.write(footer)

    manifest = {
        "generated_utc": stamp,
        "generator": "scripts/orbital/embedded/export_weights.py",
        "checkpoint_path": ckpt,
        "checkpoint_sha256": ckpt_sha,
        "weights_blob_sha256": wsha,
        "torch_version": torch.__version__,
        "obs_dim": int(obs_dim),
        "hidden_size": int(hidden),
        "num_actions": int(n_act),
        "unique_parameters": int(unique),
        "obs_normalization_in_policy": False,
        "obs_normalization_note": (
            "PufferLib Default.encode_observations() = view(B,-1).float() -> Linear. "
            "All scaling is done by the C env in fill_observations()."),
        "graph": [
            {"stage": "encoder", "op": "linear", "in": int(obs_dim), "out": int(hidden),
             "weight": "POLICY_ENC_W", "bias": "POLICY_ENC_B", "activation": "gelu_exact_erf"},
            {"stage": "recurrent", "op": "lstm_cell", "in": int(hidden), "out": int(hidden),
             "weight_ih": "POLICY_LSTM_W_IH", "weight_hh": "POLICY_LSTM_W_HH",
             "bias_ih": "POLICY_LSTM_B_IH", "bias_hh": "POLICY_LSTM_B_HH",
             "gate_order": ["i", "f", "g", "o"],
             "gate_activations": {"i": "sigmoid", "f": "sigmoid", "g": "tanh", "o": "sigmoid"},
             "equations": ["c' = f*c + i*g", "h' = o*tanh(c')"],
             "bias_convention": "torch: b_ih and b_hh both added (redundant but must match)"},
            {"stage": "decoder", "op": "linear", "in": int(hidden), "out": int(n_act),
             "weight": "POLICY_DEC_W", "bias": "POLICY_DEC_B", "activation": "none",
             "note": "greedy control = argmax(logits), ties broken by lowest index"},
            {"stage": "value", "op": "linear", "in": int(hidden), "out": 1,
             "weight": "POLICY_VAL_W", "bias": "POLICY_VAL_B", "activation": "none",
             "note": "critic head; not used by the flight control path"},
        ],
        "layers": manifest_layers,
    }
    with open(man_path, "w") as fh:
        json.dump(manifest, fh, indent=2)
        fh.write("\n")

    print(f"\nwrote {hdr_path} ({os.path.getsize(hdr_path)/1e6:.2f} MB)")
    print(f"wrote {man_path}")
    print(f"ckpt sha256 {ckpt_sha}")
    print(f"blob sha256 {wsha}")


if __name__ == "__main__":
    sys.exit(main())
