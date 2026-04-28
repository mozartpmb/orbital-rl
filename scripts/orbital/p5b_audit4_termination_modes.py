"""Audit 4: classify how each episode terminated (success/collision/escape/stranded/cap)."""
import argparse, glob, os, numpy as np

R_EARTH = 6.371e6
MU = 3.986004418e14
MAX_STEPS = 2000

ap = argparse.ArgumentParser()
ap.add_argument("log_dir")
args = ap.parse_args()

files = sorted(glob.glob(os.path.join(args.log_dir, "ep_*.npz")))
print(f"Found {len(files)} episodes in {args.log_dir}")

counts = dict(success=0, collision=0, escape=0, stranded=0, cap=0, unknown=0)

for f in files:
    d = np.load(f)
    er = float(d["episode_reward"][0])
    sx, sy = float(d["sat_x"][-1]), float(d["sat_y"][-1])
    svx, svy = float(d["sat_vx"][-1]), float(d["sat_vy"][-1])
    sat_a = float(d["sat_a"][-1])
    fuel = float(d["fuel"][-1])
    ep_len = int(d["sat_x"].shape[0])
    r = (sx**2 + sy**2) ** 0.5
    v = (svx**2 + svy**2) ** 0.5

    if er > 0:
        counts["success"] += 1
        continue
    # specific energy
    energy = 0.5 * v * v - MU / max(r, 1.0)
    # body collision check (Earth bodies[0] hard radius = R_EARTH)
    if r < R_EARTH:
        counts["collision"] += 1
    elif energy >= 0 or sat_a <= 0:
        counts["escape"] += 1
    elif fuel <= 1e-4:
        counts["stranded"] += 1
    elif ep_len >= MAX_STEPS - 5:
        counts["cap"] += 1
    else:
        counts["unknown"] += 1

total = sum(counts.values())
print(f"\nTermination histogram (n={total}):")
for k, v in counts.items():
    print(f"  {k:>10}: {v:4d}  ({100*v/total:5.1f}%)")
