"""Phase 5b Step 1 — analyze the 15.5% cap-timeout tail.
Two questions:
  1. Are failures clustered on hard (phase_gap × e_target) combos, or uniform?
  2. Is closest-approach near the tolerance window (brittleness) or far (capability gap)?
"""
import argparse, glob, math, os, numpy as np

R_EARTH = 6.371e6
MU = 3.986004418e14
MAX_STEPS = 2000

ap = argparse.ArgumentParser()
ap.add_argument("log_dir")
args = ap.parse_args()

files = sorted(glob.glob(os.path.join(args.log_dir, "ep_*.npz")))

results = []
for f in files:
    d = np.load(f)
    er = float(d["episode_reward"][0])
    is_success = er > 0

    # Initial phase gap (sat θ − target θ)
    sat_x0, sat_y0 = float(d["sat_x"][0]), float(d["sat_y"][0])
    tgt_x0, tgt_y0 = float(d["target_x"][0]), float(d["target_y"][0])
    sat_a0 = float(d["sat_a"][0])
    tgt_e = float(d["target_e"][0])

    sat_inertial = math.atan2(sat_y0, sat_x0)
    tgt_inertial = math.atan2(tgt_y0, tgt_x0)
    phase_gap = (tgt_inertial - sat_inertial + math.pi) % (2 * math.pi) - math.pi

    # Closest approach over the whole trajectory (in distance and rel-velocity)
    sx, sy = d["sat_x"], d["sat_y"]
    tx, ty = d["target_x"], d["target_y"]
    svx, svy = d["sat_vx"], d["sat_vy"]
    tvx, tvy = d["target_vx"], d["target_vy"]
    n = sx.shape[0]
    # Skip step 0 zero entries from any logging artifacts; mask to nonzero positions
    mask = (sx != 0) | (sy != 0)
    if not mask.any():
        continue
    dist = np.sqrt((sx - tx) ** 2 + (sy - ty) ** 2)
    rel_vel = np.sqrt((svx - tvx) ** 2 + (svy - tvy) ** 2)
    dist_m = np.where(mask, dist, np.inf)
    relv_m = np.where(mask, rel_vel, np.inf)
    min_dist = float(dist_m.min())
    min_relv = float(relv_m.min())
    # The "tolerance window" closest: position AND velocity simultaneously near tol
    # For each step, compute "miss factor" = max(dist/30km, relv/50m/s); the min of that
    # tells us how close the agent got to satisfying both.
    miss = np.maximum(dist_m / 30000.0, relv_m / 50.0)
    min_miss = float(miss.min())

    results.append({
        "f": os.path.basename(f),
        "is_success": is_success,
        "phase_gap_deg": math.degrees(phase_gap),
        "abs_phase_gap_deg": abs(math.degrees(phase_gap)),
        "tgt_e": tgt_e,
        "sat_a_km": sat_a0 / 1000,
        "min_dist_km": min_dist / 1000,
        "min_relv_mps": min_relv,
        "min_miss_factor": min_miss,
    })

success = [r for r in results if r["is_success"]]
fail = [r for r in results if not r["is_success"]]

print(f"=== {os.path.basename(args.log_dir)} ===")
print(f"  total eps: {len(results)}, success: {len(success)}, fail: {len(fail)}")

def stats(arr, label, fmt="{:.2f}"):
    a = np.array(arr)
    print(f"    {label:>22}: min={fmt.format(a.min())}  med={fmt.format(np.median(a))}  mean={fmt.format(a.mean())}  max={fmt.format(a.max())}")

print(f"\n  --- SUCCESS episodes (n={len(success)}) ---")
stats([r["abs_phase_gap_deg"] for r in success], "abs phase_gap (deg)", "{:5.1f}")
stats([r["tgt_e"] for r in success], "target.e", "{:.4f}")
stats([r["min_miss_factor"] for r in success], "min miss factor", "{:.3f}")

print(f"\n  --- FAILURE episodes (n={len(fail)}) ---")
stats([r["abs_phase_gap_deg"] for r in fail], "abs phase_gap (deg)", "{:5.1f}")
stats([r["tgt_e"] for r in fail], "target.e", "{:.4f}")
stats([r["min_dist_km"] for r in fail], "min dist (km)", "{:6.2f}")
stats([r["min_relv_mps"] for r in fail], "min rel-vel (m/s)", "{:6.2f}")
stats([r["min_miss_factor"] for r in fail], "min miss factor", "{:.3f}")

# Brittleness assessment: how many failures had min_miss < 1.5? (Just outside the tolerance.)
fail_miss = np.array([r["min_miss_factor"] for r in fail])
brittle_close = int(np.sum(fail_miss < 1.5))
brittle_very_close = int(np.sum(fail_miss < 1.1))
far = int(np.sum(fail_miss >= 3.0))
print(f"\n  --- Brittleness vs capability gap ---")
print(f"    Failures with min_miss < 1.1 (very close):  {brittle_very_close}/{len(fail)}  ({100*brittle_very_close/len(fail):.1f}%)")
print(f"    Failures with min_miss < 1.5 (close):       {brittle_close}/{len(fail)}  ({100*brittle_close/len(fail):.1f}%)")
print(f"    Failures with min_miss >= 3.0 (far):        {far}/{len(fail)}  ({100*far/len(fail):.1f}%)")

# 2D bin: phase_gap x eccentricity. Phase gap bins: 0-60, 60-120, 120-180. e bins: <0.025, 0.025-0.05.
print(f"\n  --- 2D failure rate by (phase_gap, e_target) ---")
phase_bins = [(0, 60), (60, 120), (120, 180)]
e_bins = [(0.0, 0.025), (0.025, 0.05)]
print(f"    {'phase\\e':>14} | " + "  ".join(f"{eb[0]:.3f}-{eb[1]:.3f}" for eb in e_bins))
for pb in phase_bins:
    row = [f"{pb[0]}-{pb[1]}°"]
    for eb in e_bins:
        in_bin = [r for r in results if pb[0] <= r["abs_phase_gap_deg"] < pb[1] and eb[0] <= r["tgt_e"] < eb[1]]
        if not in_bin:
            row.append("  --  ")
        else:
            n = len(in_bin)
            sr = sum(1 for r in in_bin if r["is_success"])
            row.append(f"{sr}/{n}={100*sr/n:5.1f}%")
    print(f"    {row[0]:>14} | " + "  ".join(row[1:]))
