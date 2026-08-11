#!/usr/bin/env python3
"""Bit-level parity test: torch reference vs the C policy implementation.

Drives both implementations over observation sequences harvested from real
environment episodes (harvest_obs.py) and reports, per build variant:

  * action agreement rate            (target: 100.00%)
  * max / p99 / mean |logit diff|    (target: float32 accumulation noise)
  * max |value diff|
  * max |LSTM h,c diff| when the harvest carries recorded state, which is the
    direct measurement of whether error compounds across recurrent carry
  * decision margin statistics — how far the top-1 logit sits above top-2, i.e.
    how much numerical headroom the action decision actually has

The C side is exercised through its real public API (ctypes -> libpolicy),
including the episode-boundary reset, so the recurrent state carry under test is
the same code path the flight loop uses. Error-path behaviour (NULL pointers,
uninitialised state, non-finite observations) is unit-tested separately.

Usage:
    make -C scripts/orbital/embedded shim
    python3 scripts/orbital/embedded/test_parity.py \
        --harvest /tmp/emb_harvest_s123.npz /tmp/emb_harvest_s777.npz
    python3 scripts/orbital/embedded/test_parity.py --variants   # all GELU builds
"""

import argparse
import ctypes
import math
import os
import platform
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OBS_DIM = 38
HIDDEN = 128
ACTIONS = 16
SHLIB_EXT = "dylib" if platform.system() == "Darwin" else "so"

POLICY_OK = 0
POLICY_ERR_NULL = -1
POLICY_ERR_UNINIT = -2
POLICY_ERR_OBS_NONFINITE = -3
POLICY_ERR_DIVERGED = -4


class PolicyState(ctypes.Structure):
    _fields_ = [
        ("h", ctypes.c_float * HIDDEN),
        ("c", ctypes.c_float * HIDDEN),
        ("magic", ctypes.c_uint32),
        ("steps", ctypes.c_uint32),
    ]


def load_lib(path):
    lib = ctypes.CDLL(path)
    lib.policy_reset.argtypes = [ctypes.POINTER(PolicyState)]
    lib.policy_reset.restype = None
    lib.policy_infer.argtypes = [
        ctypes.POINTER(PolicyState),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_int),
    ]
    lib.policy_infer.restype = ctypes.c_int
    lib.policy_act.argtypes = [ctypes.POINTER(PolicyState), ctypes.POINTER(ctypes.c_float)]
    lib.policy_act.restype = ctypes.c_int
    for fn in ("policy_ckpt_sha256", "policy_weights_blob_sha256", "policy_build_variant"):
        getattr(lib, fn).argtypes = []
        getattr(lib, fn).restype = ctypes.c_char_p
    return lib


def replay(lib, obs, is_start):
    """Run the C policy over the observation sequence, resetting at episode starts."""
    n = obs.shape[0]
    logits = np.zeros((n, ACTIONS), dtype=np.float32)
    values = np.zeros(n, dtype=np.float32)
    actions = np.zeros(n, dtype=np.int32)
    h_out = np.zeros((n, HIDDEN), dtype=np.float32)
    c_out = np.zeros((n, HIDDEN), dtype=np.float32)

    st = PolicyState()
    lib.policy_reset(ctypes.byref(st))

    lg_buf = (ctypes.c_float * ACTIONS)()
    val = ctypes.c_float()
    act = ctypes.c_int()
    obs_c = np.ascontiguousarray(obs, dtype=np.float32)

    for t in range(n):
        if is_start[t]:
            lib.policy_reset(ctypes.byref(st))
        p_obs = obs_c[t].ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        rc = lib.policy_infer(ctypes.byref(st), p_obs, lg_buf,
                              ctypes.byref(val), ctypes.byref(act))
        if rc != POLICY_OK:
            raise RuntimeError(f"policy_infer returned {rc} at index {t}")
        logits[t] = np.frombuffer(lg_buf, dtype=np.float32, count=ACTIONS)
        values[t] = val.value
        actions[t] = act.value
        h_out[t] = np.frombuffer(st.h, dtype=np.float32, count=HIDDEN)
        c_out[t] = np.frombuffer(st.c, dtype=np.float32, count=HIDDEN)
    return logits, values, actions, h_out, c_out


def margins(logits):
    """Top-1 minus top-2 logit per row — the numerical headroom of the argmax."""
    part = np.partition(logits, -2, axis=1)
    return part[:, -1] - part[:, -2]


def report_variant(name, lib_path, datasets, verbose=False):
    lib = load_lib(lib_path)
    variant = lib.policy_build_variant().decode()
    sha = lib.policy_ckpt_sha256().decode()

    tot = 0
    n_agree = 0
    max_dl = 0.0
    all_dl = []
    max_dv = 0.0
    max_dh = 0.0
    max_dc = 0.0
    disagree_rows = []
    margin_at_disagree = []
    all_margin = []
    per_step_dl = {}

    for tag, d in datasets:
        obs = d["obs"]
        ref_logits = d["logits"]
        ref_actions = d["action"]
        ref_value = d["value"]
        is_start = d["is_episode_start"]
        step_idx = d["step"]

        c_logits, c_values, c_actions, c_h, c_c = replay(lib, obs, is_start)

        dl = np.abs(c_logits.astype(np.float64) - ref_logits.astype(np.float64))
        dv = np.abs(c_values.astype(np.float64) - ref_value.astype(np.float64))
        agree = (c_actions == ref_actions)

        tot += obs.shape[0]
        n_agree += int(agree.sum())
        max_dl = max(max_dl, float(dl.max()))
        max_dv = max(max_dv, float(dv.max()))
        all_dl.append(dl.reshape(-1))
        m = margins(ref_logits.astype(np.float64))
        all_margin.append(m)
        bad = np.flatnonzero(~agree)
        for i in bad[:20]:
            disagree_rows.append((tag, int(d["episode"][i]), int(step_idx[i]),
                                  int(ref_actions[i]), int(c_actions[i]),
                                  float(m[i]), float(dl[i].max())))
        margin_at_disagree.extend(m[bad].tolist())

        if "lstm_h" in d:
            dh = np.abs(c_h.astype(np.float64) - d["lstm_h"].astype(np.float64))
            dc = np.abs(c_c.astype(np.float64) - d["lstm_c"].astype(np.float64))
            max_dh = max(max_dh, float(dh.max()))
            max_dc = max(max_dc, float(dc.max()))

        rowmax = dl.max(axis=1)
        for s, v in zip(step_idx, rowmax):
            per_step_dl[int(s)] = max(per_step_dl.get(int(s), 0.0), float(v))

    all_dl = np.concatenate(all_dl)
    all_margin = np.concatenate(all_margin)
    agree_pct = 100.0 * n_agree / tot

    print(f"\n--- {name}  [{variant}] ---")
    print(f"  library            : {os.path.relpath(lib_path, HERE)}")
    print(f"  ckpt sha256        : {sha}")
    print(f"  obs vectors        : {tot}")
    print(f"  action agreement   : {n_agree}/{tot} = {agree_pct:.4f}%")
    print(f"  |logit diff| max   : {max_dl:.3e}")
    print(f"  |logit diff| p99   : {np.percentile(all_dl, 99):.3e}")
    print(f"  |logit diff| mean  : {all_dl.mean():.3e}")
    print(f"  |value diff| max   : {max_dv:.3e}")
    if max_dh or max_dc:
        print(f"  |lstm h diff| max  : {max_dh:.3e}")
        print(f"  |lstm c diff| max  : {max_dc:.3e}")
    print(f"  decision margin    : min {all_margin.min():.4f}  "
          f"p1 {np.percentile(all_margin, 1):.4f}  median {np.median(all_margin):.4f}")
    ratio = all_margin.min() / max_dl if max_dl > 0 else float("inf")
    print(f"  margin / max diff  : {ratio:.3e}x  (headroom before a flip is possible)")

    if verbose and per_step_dl:
        ks = sorted(per_step_dl)
        show = [k for k in ks if k in (0, 1, 2, 5, 10, 20, 40, 80, max(ks))]
        print("  max |logit diff| by within-episode step index (LSTM carry depth):")
        for k in show:
            print(f"      step {k:>4} : {per_step_dl[k]:.3e}")

    if disagree_rows:
        print(f"  DISAGREEMENTS ({len(disagree_rows)} shown, margin at disagreement "
              f"min {min(margin_at_disagree):.3e}):")
        for r in disagree_rows:
            print(f"      {r[0]} ep{r[1]} step{r[2]}: torch={r[3]} c={r[4]} "
                  f"margin={r[5]:.3e} maxdiff={r[6]:.3e}")

    return dict(name=name, variant=variant, n=tot, agree=n_agree,
                agree_pct=agree_pct, max_logit_diff=max_dl,
                p99_logit_diff=float(np.percentile(all_dl, 99)),
                mean_logit_diff=float(all_dl.mean()),
                max_value_diff=max_dv, max_h_diff=max_dh, max_c_diff=max_dc,
                min_margin=float(all_margin.min()))


def error_path_tests(lib_path):
    """Unit tests for the defensive contract in policy.h."""
    lib = load_lib(lib_path)
    st = PolicyState()
    obs = (ctypes.c_float * OBS_DIM)(*([0.1] * OBS_DIM))
    results = []

    # 1. uninitialised state is rejected (zeroed struct has no magic)
    ctypes.memset(ctypes.byref(st), 0, ctypes.sizeof(st))
    rc = lib.policy_infer(ctypes.byref(st), obs, None, None, None)
    results.append(("uninitialised state rejected", rc == POLICY_ERR_UNINIT, rc))

    # 2. NULL observation rejected
    lib.policy_reset(ctypes.byref(st))
    rc = lib.policy_infer(ctypes.byref(st), None, None, None, None)
    results.append(("NULL obs rejected", rc == POLICY_ERR_NULL, rc))

    # 3. NULL state rejected
    rc = lib.policy_infer(None, obs, None, None, None)
    results.append(("NULL state rejected", rc == POLICY_ERR_NULL, rc))

    # 4. NaN observation rejected AND recurrent state left untouched
    lib.policy_reset(ctypes.byref(st))
    act = ctypes.c_int()
    lib.policy_infer(ctypes.byref(st), obs, None, None, ctypes.byref(act))
    h_before = np.frombuffer(st.h, dtype=np.float32, count=HIDDEN).copy()
    c_before = np.frombuffer(st.c, dtype=np.float32, count=HIDDEN).copy()
    steps_before = st.steps
    bad = (ctypes.c_float * OBS_DIM)(*([0.1] * OBS_DIM))
    bad[7] = float("nan")
    rc = lib.policy_infer(ctypes.byref(st), bad, None, None, None)
    h_after = np.frombuffer(st.h, dtype=np.float32, count=HIDDEN)
    c_after = np.frombuffer(st.c, dtype=np.float32, count=HIDDEN)
    untouched = (np.array_equal(h_before, h_after)
                 and np.array_equal(c_before, c_after)
                 and st.steps == steps_before)
    results.append(("NaN obs rejected", rc == POLICY_ERR_OBS_NONFINITE, rc))
    results.append(("state untouched after rejected frame", untouched, int(untouched)))

    # 5. Inf observation rejected
    bad2 = (ctypes.c_float * OBS_DIM)(*([0.1] * OBS_DIM))
    bad2[0] = float("inf")
    rc = lib.policy_infer(ctypes.byref(st), bad2, None, None, None)
    results.append(("Inf obs rejected", rc == POLICY_ERR_OBS_NONFINITE, rc))

    # 6. determinism: same state + same obs => bit-identical logits
    lib.policy_reset(ctypes.byref(st))
    lg1 = (ctypes.c_float * ACTIONS)()
    lib.policy_infer(ctypes.byref(st), obs, lg1, None, None)
    a1 = np.frombuffer(lg1, dtype=np.float32, count=ACTIONS).copy()
    lib.policy_reset(ctypes.byref(st))
    lg2 = (ctypes.c_float * ACTIONS)()
    lib.policy_infer(ctypes.byref(st), obs, lg2, None, None)
    a2 = np.frombuffer(lg2, dtype=np.float32, count=ACTIONS).copy()
    results.append(("deterministic repeat (bit-identical)",
                    np.array_equal(a1, a2), 0))

    # 7. reset clears the recurrent state exactly
    lib.policy_infer(ctypes.byref(st), obs, None, None, None)
    lib.policy_reset(ctypes.byref(st))
    zeroed = (not np.any(np.frombuffer(st.h, dtype=np.float32, count=HIDDEN))
              and not np.any(np.frombuffer(st.c, dtype=np.float32, count=HIDDEN))
              and st.steps == 0)
    results.append(("policy_reset zeroes h, c, steps", zeroed, 0))

    # 8. policy_act mirrors policy_infer's argmax
    lib.policy_reset(ctypes.byref(st))
    lg = (ctypes.c_float * ACTIONS)()
    a_ref = ctypes.c_int()
    lib.policy_infer(ctypes.byref(st), obs, lg, None, ctypes.byref(a_ref))
    lib.policy_reset(ctypes.byref(st))
    a_act = lib.policy_act(ctypes.byref(st), obs)
    results.append(("policy_act == policy_infer argmax", a_act == a_ref.value, a_act))

    print("\n--- error-path / contract unit tests ---")
    ok = True
    for name, passed, code in results:
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}" +
              ("" if passed else f"  (got {code})"))
        ok = ok and passed
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--harvest", nargs="+",
                    default=["/tmp/emb_harvest_s123.npz", "/tmp/emb_harvest_s777.npz"])
    ap.add_argument("--lib", default=os.path.join(HERE, f"libpolicy.{SHLIB_EXT}"))
    ap.add_argument("--variants", action="store_true",
                    help="test every GELU / accumulation build variant")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    datasets = []
    for p in args.harvest:
        if not os.path.exists(p):
            raise SystemExit(f"missing harvest file {p} — run harvest_obs.py first")
        d = np.load(p, allow_pickle=False)
        datasets.append((os.path.basename(p), d))

    total = sum(d["obs"].shape[0] for _, d in datasets)
    print("=== torch vs C parity ===")
    print(f"harvests: {', '.join(t for t, _ in datasets)}")
    print(f"total obs vectors: {total}")
    if total < 2000:
        print(f"WARNING: fewer than 2000 obs vectors ({total})")

    libs = [("default build", args.lib)]
    if args.variants:
        subprocess.run(["make", "-s", "-C", HERE, "variants"], check=True)
        libs = [
            ("GELU exact erff (default)", os.path.join(HERE, f"libpolicy_erf.{SHLIB_EXT}")),
            ("GELU erf poly (expf only)", os.path.join(HERE, f"libpolicy_poly.{SHLIB_EXT}")),
            ("GELU tanh approx", os.path.join(HERE, f"libpolicy_tanh.{SHLIB_EXT}")),
            ("double accumulation", os.path.join(HERE, f"libpolicy_accdouble.{SHLIB_EXT}")),
        ]

    summaries = [report_variant(n, p, datasets, args.verbose) for n, p in libs]
    contract_ok = error_path_tests(libs[0][1])

    print("\n=== summary ===")
    print(f"{'build':<30} {'agree':>16} {'max|dlogit|':>13} {'p99':>11} {'max|dh|':>11}")
    for s in summaries:
        print(f"{s['name']:<30} {s['agree']}/{s['n']:<8} "
              f"{s['agree_pct']:>6.2f}% {s['max_logit_diff']:>12.3e} "
              f"{s['p99_logit_diff']:>11.3e} {s['max_h_diff']:>11.3e}")

    primary = summaries[0]
    ok = (primary["agree"] == primary["n"]) and contract_ok
    print(f"\nRESULT: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
