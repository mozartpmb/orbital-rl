"""Phase 5b Step 1 §3: R-vs-G diagnostic.
Distinguishes 'agent isn't trying' (G — gradient absence) from
'agent is trying but can't reach success' (R — reachability)."""
import argparse, glob, os, numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("dirs", nargs="+")
args = ap.parse_args()

for log_dir in args.dirs:
    files = sorted(glob.glob(os.path.join(log_dir, "ep_*.npz")))
    label = os.path.basename(log_dir.rstrip("/"))
    print(f"\n=== {label}: {len(files)} eps ===")

    total_dv_per_ep = []
    burn_frac_per_ep = []
    action_counts = np.zeros(10, dtype=np.int64)

    for f in files:
        d = np.load(f)
        actions = d["action"].astype(int)
        delta_v = d["delta_v"]
        # Total |Δv| for the episode
        total_dv_per_ep.append(float(np.sum(delta_v)))
        # Burn frequency: actions 1-8 are burns; 0 = coast, 9 = warp
        is_burn = (actions >= 1) & (actions <= 8)
        burn_frac_per_ep.append(float(np.mean(is_burn)))
        for a in actions:
            if 0 <= a < 10:
                action_counts[a] += 1

    total_dv = np.array(total_dv_per_ep)
    burn_frac = np.array(burn_frac_per_ep)
    total_steps = action_counts.sum()
    print(f"  Mean total Δv per ep:   {np.mean(total_dv):.1f} m/s  (median {np.median(total_dv):.1f})")
    print(f"  Mean burn frequency:    {100*np.mean(burn_frac):.2f}%  (median {100*np.median(burn_frac):.2f}%)")
    print(f"  Action distribution (% of all steps):")
    names = ["coast", "prograde+", "prograde++", "retro+", "retro++", "normal+", "normal-", "radial+", "radial-", "warp"]
    for a, n in enumerate(action_counts):
        print(f"    {a} ({names[a]:>10}): {100*n/total_steps:6.2f}%")
