"""Phase 5 verification I2: scan every web-data trajectory for sub-keepout perigees.

For every ep_*.json under /Users/pete/space_training/web_data/runs/, compute:
  * sat orbit perigee  = sat_a_m  * (1 - sat_e)
  * target orbit perigee = target_a_m * (1 - target_e)
  * sat initial Cartesian radius   from steps[0].x, y
  * target initial Cartesian radius from steps[0].target_x, target_y

Categorize each axis (sat / target) independently:
  * Pass     — both perigee AND initial radius >= EARTH_KEEPOUT (6571 km)
  * Marginal — min(perigee, init_radius) in [R_EARTH, EARTH_KEEPOUT)  (i.e. 6371-6571 km)
  * Failed   — min(perigee, init_radius) <  R_EARTH (6371 km, sub-surface)

Aggregate per directory + per axis. Print 5 worst-case files.
"""

import glob
import json
import math
import os
import sys

R_EARTH       = 6_371_000.0   # m
ALT_MIN       = 200_000.0     # m
EARTH_KEEPOUT = R_EARTH + ALT_MIN  # 6.571e6 m


def classify(min_radius_m):
    """Return 'Pass' / 'Marginal' / 'Failed' for the worst point on this axis."""
    if min_radius_m >= EARTH_KEEPOUT:
        return "Pass"
    if min_radius_m >= R_EARTH:
        return "Marginal"
    return "Failed"


def worse(a, b):
    """Return the worse of two status strings (Failed > Marginal > Pass)."""
    order = {"Pass": 0, "Marginal": 1, "Failed": 2}
    return a if order[a] >= order[b] else b


def scan_one(path):
    with open(path) as f:
        j = json.load(f)
    init = j["initial"]
    sat_a = float(init["sat_a_m"])
    sat_e = float(init["sat_e"])
    tgt_a = float(init["target_a_m"])
    tgt_e = float(init["target_e"])

    sat_rp = sat_a * (1.0 - sat_e)
    tgt_rp = tgt_a * (1.0 - tgt_e)

    s0 = j["steps"][0]
    r_sat0 = math.hypot(float(s0["x"]),        float(s0["y"]))
    r_tgt0 = math.hypot(float(s0["target_x"]), float(s0["target_y"]))

    sat_min = min(sat_rp, r_sat0)
    tgt_min = min(tgt_rp, r_tgt0)

    sat_status = classify(sat_min)
    tgt_status = classify(tgt_min)
    overall    = worse(sat_status, tgt_status)

    md = j.get("metadata", {})
    env_cfg = md.get("env_config", {}) or {}
    return {
        "path": path,
        "sat_rp_km":  sat_rp / 1000.0,
        "tgt_rp_km":  tgt_rp / 1000.0,
        "r_sat0_km":  r_sat0 / 1000.0,
        "r_tgt0_km":  r_tgt0 / 1000.0,
        "sat_min_km": sat_min / 1000.0,
        "tgt_min_km": tgt_min / 1000.0,
        "sat_status": sat_status,
        "tgt_status": tgt_status,
        "overall":    overall,
        "phase":             md.get("phase", "?"),
        "valid_init_only":   env_cfg.get("valid_init_only", "?"),
        "e_max_target":      env_cfg.get("e_max_target", "?"),
        "e_max_sat":         env_cfg.get("e_max_sat", "?"),
        "same_orbit_init":   env_cfg.get("same_orbit_init", "?"),
        "checkpoint":        md.get("checkpoint", "?"),
    }


def summarize(results, label):
    n = len(results)
    if n == 0:
        return
    sat_counts = {"Pass": 0, "Marginal": 0, "Failed": 0}
    tgt_counts = {"Pass": 0, "Marginal": 0, "Failed": 0}
    overall_counts = {"Pass": 0, "Marginal": 0, "Failed": 0}
    for r in results:
        sat_counts[r["sat_status"]] += 1
        tgt_counts[r["tgt_status"]] += 1
        overall_counts[r["overall"]] += 1
    sat_min  = min(r["sat_min_km"] for r in results)
    tgt_min  = min(r["tgt_min_km"] for r in results)
    sat_max  = max(r["sat_min_km"] for r in results)
    tgt_max  = max(r["tgt_min_km"] for r in results)
    vio_set  = sorted({r["valid_init_only"] for r in results})
    phase_set = sorted({r["phase"] for r in results})
    print(f"\n=== {label} (n={n}) ===")
    print(f"  metadata.phase:           {phase_set}")
    print(f"  metadata.valid_init_only: {vio_set}")
    print(f"  Sat-axis    : Pass={sat_counts['Pass']}  Marginal={sat_counts['Marginal']}  Failed={sat_counts['Failed']}")
    print(f"  Target-axis : Pass={tgt_counts['Pass']}  Marginal={tgt_counts['Marginal']}  Failed={tgt_counts['Failed']}")
    print(f"  Overall     : Pass={overall_counts['Pass']}  Marginal={overall_counts['Marginal']}  Failed={overall_counts['Failed']}")
    print(f"  Sat-axis min_radius range  : {sat_min:.1f} .. {sat_max:.1f} km  (keepout={EARTH_KEEPOUT/1000:.1f}, surface={R_EARTH/1000:.1f})")
    print(f"  Target-axis min_radius rng : {tgt_min:.1f} .. {tgt_max:.1f} km")


def main():
    runs_root = "/Users/pete/space_training/web_data/runs"
    dirs = sorted([d for d in glob.glob(runs_root + "/*") if os.path.isdir(d)])
    all_results = []
    by_dir = {}
    for d in dirs:
        files = sorted(glob.glob(os.path.join(d, "ep_*.json")))
        results = []
        for f in files:
            try:
                results.append(scan_one(f))
            except Exception as ex:
                print(f"WARN failed to parse {f}: {ex}", file=sys.stderr)
        by_dir[d] = results
        all_results.extend(results)

    print("=" * 70)
    print("Phase 5 verification I2 — perigee scan over web_data/runs/")
    print("=" * 70)
    print(f"R_EARTH       = {R_EARTH/1000:.1f} km")
    print(f"EARTH_KEEPOUT = {EARTH_KEEPOUT/1000:.1f} km")
    print(f"Total files   = {len(all_results)}")

    for d in dirs:
        summarize(by_dir[d], os.path.basename(d.rstrip("/")))

    summarize(all_results, "ALL")

    # Worst 10 files (by min radius across both axes)
    print("\n=== 10 worst-case files (lowest min radius, sat or target) ===")
    def key(r):
        return min(r["sat_min_km"], r["tgt_min_km"])
    worst = sorted(all_results, key=key)[:10]
    for r in worst:
        which = "sat" if r["sat_min_km"] <= r["tgt_min_km"] else "tgt"
        print(f"  {os.path.relpath(r['path'], runs_root)}  "
              f"sat_min={r['sat_min_km']:.1f} tgt_min={r['tgt_min_km']:.1f} "
              f"(worst={which})  status={r['overall']}  phase={r['phase']}  "
              f"vio={r['valid_init_only']}  e_max={r['e_max_target']}")

    # Files where overall status is anything but Pass
    nonpass = [r for r in all_results if r["overall"] != "Pass"]
    print(f"\n=== {len(nonpass)} files with non-Pass overall status ===")
    for r in nonpass[:50]:
        print(f"  {os.path.relpath(r['path'], runs_root)}  "
              f"sat={r['sat_status']}({r['sat_min_km']:.1f}) "
              f"tgt={r['tgt_status']}({r['tgt_min_km']:.1f}) "
              f"phase={r['phase']}")


if __name__ == "__main__":
    main()
