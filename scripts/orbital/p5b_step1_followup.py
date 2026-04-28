"""Phase 5b Step 1 follow-up: ep-length distribution + action distribution
conditional on success vs cap-failure."""
import argparse, glob, os, numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("log_dir")
args = ap.parse_args()

files = sorted(glob.glob(os.path.join(args.log_dir, "ep_*.npz")))
print(f"=== {os.path.basename(args.log_dir)}: {len(files)} eps ===\n")

succ_lens, fail_lens = [], []
succ_actions = np.zeros(10, dtype=np.int64)
fail_actions = np.zeros(10, dtype=np.int64)
succ_dv, fail_dv = [], []

for f in files:
    d = np.load(f)
    er = float(d["episode_reward"][0])
    actions = d["action"].astype(int)
    delta_v = d["delta_v"]
    is_success = er > 0
    # ep length: find last nonzero or use full length
    n = actions.shape[0]
    ep_len = n
    if is_success:
        succ_lens.append(ep_len)
        succ_dv.append(float(np.sum(delta_v)))
        for a in actions:
            if 0 <= a < 10: succ_actions[a] += 1
    else:
        fail_lens.append(ep_len)
        fail_dv.append(float(np.sum(delta_v)))
        for a in actions:
            if 0 <= a < 10: fail_actions[a] += 1

succ_lens = np.array(succ_lens); fail_lens = np.array(fail_lens)
print(f"--- Episode length distribution ---")
print(f"  Successes (n={len(succ_lens)}): min={succ_lens.min()} med={int(np.median(succ_lens))} mean={int(succ_lens.mean())} max={succ_lens.max()}")
print(f"  Failures  (n={len(fail_lens)}): min={fail_lens.min()} med={int(np.median(fail_lens))} mean={int(fail_lens.mean())} max={fail_lens.max()}")
# How many failures are at the cap?
cap_fail = int(np.sum(fail_lens >= 1995))
print(f"  Failures at cap (≥1995):     {cap_fail}/{len(fail_lens)} ({100*cap_fail/max(len(fail_lens),1):.1f}%)")
print(f"  Failures terminating early:  {len(fail_lens) - cap_fail}/{len(fail_lens)}")
if cap_fail < len(fail_lens):
    early = fail_lens[fail_lens < 1995]
    print(f"    Early-fail length distribution: {sorted(early.tolist())}")

print(f"\n--- Action distribution ---")
names = ["coast", "prograde+", "prograde++", "retro+", "retro++", "normal+", "normal-", "radial+", "radial-", "warp"]
succ_total = succ_actions.sum() if succ_actions.sum() > 0 else 1
fail_total = fail_actions.sum() if fail_actions.sum() > 0 else 1
print(f"  {'action':<14} {'success %':>10} {'cap-fail %':>11}")
for a in range(10):
    print(f"  {a} {names[a]:<11} {100*succ_actions[a]/succ_total:>9.2f}% {100*fail_actions[a]/fail_total:>10.2f}%")

print(f"\n--- Δv usage ---")
print(f"  Successes: median Δv={int(np.median(succ_dv))} m/s, mean={int(np.mean(succ_dv))}")
if fail_dv:
    print(f"  Failures:  median Δv={int(np.median(fail_dv))} m/s, mean={int(np.mean(fail_dv))}")
