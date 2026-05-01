"""Bonus stats for the web frontend / blog post.

Aggregates the multi-seed eval logs into:
  - action distribution per e_max (which actions does the agent use?)
  - Δv-vs-Hohmann ratio per e_max (fuel efficiency)
  - failure-mode breakdown per e_max
  - trajectory length distribution per e_max
  - per-(seed, e_max) success rates (already in CSV, restated as JSON)

Outputs JSON files under web_data/results/ for the frontend to consume.
"""
import glob, json, math, os
from collections import Counter, defaultdict
import numpy as np

MU = 3.986004418e14
R_EARTH = 6.371e6
EARTH_KEEPOUT = R_EARTH + 200e3
MAX_STEPS = 2000
DV_BUDGET = 478.1


def hohmann(r1, r2):
    a_t = 0.5 * (r1 + r2)
    v1c = math.sqrt(MU/r1); v2c = math.sqrt(MU/r2)
    v1t = math.sqrt(MU * (2.0/r1 - 1.0/a_t)); v2t = math.sqrt(MU * (2.0/r2 - 1.0/a_t))
    return abs(v1t - v1c) + abs(v2c - v2t)


def classify(d, mask, active):
    er = float(d["episode_reward"][0])
    if er > 0: return "success"
    last = int(active[-1])
    sx, sy = float(d["sat_x"][last]), float(d["sat_y"][last])
    svx, svy = float(d["sat_vx"][last]), float(d["sat_vy"][last])
    r_last = math.sqrt(sx*sx + sy*sy)
    v2 = svx*svx + svy*svy
    E = 0.5*v2 - MU/max(r_last, 1.0)
    fuel = float(d["fuel"][last])
    sa = float(d["sat_a"][last]); se = float(d["sat_e"][last])
    rp = sa*(1-se) if sa > 0 else -1.0
    r_all = np.sqrt(d["sat_x"][active].astype(np.float64)**2 +
                    d["sat_y"][active].astype(np.float64)**2)
    if float(r_all.min()) < R_EARTH or (rp > 0 and rp < R_EARTH):
        return "collision"
    if E >= 0: return "escape"
    if fuel <= 1e-6: return "stranded"
    if len(active) >= MAX_STEPS - 1: return "safety_cap"
    return "other"


ACTION_NAMES = ["coast", "prograde_5", "prograde_10", "prograde_25",
                "retrograde_5", "retrograde_10", "retrograde_25",
                "radial_out", "radial_in", "warp_5min"]


def aggregate_dir(npz_dir):
    files = sorted(glob.glob(f"{npz_dir}/ep_*.npz"))
    actions = Counter()
    dv_ratios = []     # actual_dv / hohmann
    dv_actual = []
    lengths = {"success": [], "fail": []}
    modes = Counter()
    for f in files:
        d = np.load(f)
        mask = (d["sat_x"] != 0) | (d["sat_y"] != 0)
        if mask.sum() < 1: continue
        active = np.where(mask)[0]
        n = len(active)
        mode = classify(d, mask, active)
        modes[mode] += 1
        # action distribution among active steps
        for a in d["action"][active]:
            actions[int(a)] += 1
        # length
        bucket = "success" if mode == "success" else "fail"
        lengths[bucket].append(n)
        # dv ratio (success only, where hohmann is meaningful)
        if mode == "success":
            total_dv = float(np.abs(d["delta_v"][active]).sum())
            dv_actual.append(total_dv)
            sat_a0 = float(d["sat_a"][0]); tgt_a0 = float(d["target_a"][0])
            h = hohmann(sat_a0, tgt_a0)
            if h > 1e-3:
                dv_ratios.append(total_dv / h)
    return {
        "n_episodes": len(files),
        "termination_modes": dict(modes),
        "action_distribution": {ACTION_NAMES[k]: v for k, v in actions.items()},
        "action_total": sum(actions.values()),
        "lengths_summary": {
            "success": {
                "n": len(lengths["success"]),
                "mean": float(np.mean(lengths["success"])) if lengths["success"] else None,
                "p50": float(np.percentile(lengths["success"], 50)) if lengths["success"] else None,
                "p90": float(np.percentile(lengths["success"], 90)) if lengths["success"] else None,
            },
            "failure": {
                "n": len(lengths["fail"]),
                "mean": float(np.mean(lengths["fail"])) if lengths["fail"] else None,
                "p50": float(np.percentile(lengths["fail"], 50)) if lengths["fail"] else None,
            },
        },
        "fuel_efficiency": {
            "n_success": len(dv_ratios),
            "actual_dv_mean": float(np.mean(dv_actual)) if dv_actual else None,
            "actual_dv_p50": float(np.percentile(dv_actual, 50)) if dv_actual else None,
            "ratio_to_hohmann_mean": float(np.mean(dv_ratios)) if dv_ratios else None,
            "ratio_to_hohmann_p50": float(np.percentile(dv_ratios, 50)) if dv_ratios else None,
            "fuel_budget_ms": DV_BUDGET,
        },
    }


def main():
    out = {}
    seed_dirs = {
        "phase5e_seed42": "puffer_orbital_177765503091",
        "phase5e_seedA":  "puffer_orbital_177765655537",
        "phase5e_seedB":  "puffer_orbital_177765658166",
        "phase5e_seedC":  "puffer_orbital_177765658729",
        "phase5e_seedD":  "puffer_orbital_177765659007",
    }
    for seed_label, dir_id in seed_dirs.items():
        out[seed_label] = {}
        for e in ["0.05", "0.10", "0.20", "0.30", "0.50", "0.70"]:
            src = f"/tmp/p5closure_escan_{dir_id}_e{e}"
            if not os.path.isdir(src): continue
            stats = aggregate_dir(src)
            out[seed_label][f"e_{e}"] = stats

    os.makedirs("web_data/results", exist_ok=True)
    with open("web_data/results/bonus_stats.json", "w") as fh:
        json.dump(out, fh, indent=2)
    print("Wrote web_data/results/bonus_stats.json")

    # Pretty summary
    print("\n=== Action distribution at e=0.20 (seed 42) ===")
    s = out["phase5e_seed42"]["e_0.20"]
    total = s["action_total"]
    for a, c in sorted(s["action_distribution"].items(), key=lambda kv: -kv[1]):
        print(f"  {a:>14}: {c:>7} ({100*c/total:.1f}%)")
    print(f"\n=== Termination modes at e=0.20 (seed 42) ===")
    for m, c in s["termination_modes"].items():
        print(f"  {m:>11}: {c}")
    print(f"\n=== Fuel efficiency at e=0.20 (seed 42, successes only) ===")
    fe = s["fuel_efficiency"]
    print(f"  median Δv: {fe['actual_dv_p50']:.1f} m/s (budget={fe['fuel_budget_ms']:.0f})")
    print(f"  median Δv/Hohmann: {fe['ratio_to_hohmann_p50']:.2f}x")


if __name__ == "__main__":
    main()
