"""Phase 5 wrap-up W2: capability surface at fixed eccentricities.

Per-cell axes:
  - e_fixed   ∈ {0.0, 0.05, 0.10, 0.20, 0.50, 0.70}   (sat & target both)
  - phase_gap ∈ {30°, 90°, 150°, 180°}
  - relation  ∈ {fully_random, same_orbit_init}

For each cell, altitude band is auto-set so that the orbit is physically valid
(perigee >= EARTH_KEEPOUT). For e=0.0 use trained LEO band.

Eps per cell × 5 training seeds; output per-seed CSVs.

Subprocess-based (one eval_checkpoint.py per cell) — single-env eval is bound
by Python overhead per episode, so launching multiple cells in parallel
across seeds gives the best wall time.
"""
import argparse, csv, math, os, subprocess, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed

R_EARTH = 6.371e6
EARTH_KEEPOUT = R_EARTH + 200e3

# 5 Phase 5e seeds
SEED_CKPTS = {
    "seed42":    "experiments/puffer_orbital_177765503091/model_puffer_orbital_000325.pt",
    "seedA":     "experiments/puffer_orbital_177765655537/model_puffer_orbital_000375.pt",
    "seedB":     "experiments/puffer_orbital_177765658166/model_puffer_orbital_000250.pt",
    "seedC":     "experiments/puffer_orbital_177765658729/model_puffer_orbital_000350.pt",
    "seedD":     "experiments/puffer_orbital_177765659007/model_puffer_orbital_000275.pt",
}

E_VALUES = [0.0, 0.05, 0.10, 0.20, 0.50, 0.70]
PHASE_DEG_VALUES = [30, 90, 150, 180]
RELATIONS = ["fully_random", "same_orbit_init"]


def alt_band_for_e(e):
    """Pick (a_min, a_max) so orbit at this eccentricity has valid perigee.
    For e=0, use trained LEO band (300-800km).
    For higher e, expand band so a*(1-e) ∈ [EARTH_KEEPOUT, EARTH_KEEPOUT * 1.6]."""
    if e <= 0.001:
        return (-1.0, -1.0)  # use default LEO
    rp_min = EARTH_KEEPOUT
    rp_max = EARTH_KEEPOUT * 1.7  # ~factor of 1.7 above keepout for some altitude diversity
    a_min = rp_min / (1.0 - e)
    a_max = rp_max / (1.0 - e)
    return (a_min, a_max)


def run_cell(args):
    seed_label, ckpt, e, phase_deg, relation, episodes, out_root = args
    a_min, a_max = alt_band_for_e(e)
    same_orbit = 1 if relation == "same_orbit_init" else 0
    phase_rad = math.radians(phase_deg)
    cell_id = f"{seed_label}_e{e}_p{phase_deg}_{relation}"
    out_dir = f"{out_root}/{cell_id}"
    cmd = [
        "python3", "scripts/orbital/eval_checkpoint.py", ckpt,
        "--episodes", str(episodes),
        "--e-max-target", str(max(e, 0.05)),
        "--e-max-sat", str(max(e, 0.05)),
        "--init-phase-gap-max", "3.14159",
        "--valid-init-only", "1",
        "--e-target-fixed", str(e),
        "--e-sat-fixed", str(e),
        "--phase-gap-fixed", f"{phase_rad:.5f}",
        "--same-orbit-init", str(same_orbit),
        "--out-dir", out_dir,
        "--seed", "42",
    ]
    if a_min > 0:
        cmd += ["--a-min-override", str(a_min), "--a-max-override", str(a_max)]
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd="pufferlib")
    dt = time.time() - t0
    out = proc.stdout
    succ_n = total = -1
    for line in out.splitlines():
        if "Success rate" in line:
            try:
                frac = line.split()[-2]  # e.g. "92/100"
                succ_n, total = map(int, frac.split("/"))
            except Exception:
                pass
            break
    return {
        "seed": seed_label, "e": e, "phase_deg": phase_deg, "relation": relation,
        "success_n": succ_n, "total": total, "wall_s": round(dt, 1),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--out-csv", default="web_data/results/phase5_capability_surface.csv")
    p.add_argument("--scratch", default="/tmp/p5wrap_surface")
    p.add_argument("--seeds", default=",".join(SEED_CKPTS.keys()))
    args = p.parse_args()

    cells = []
    for seed in args.seeds.split(","):
        if seed not in SEED_CKPTS:
            print(f"unknown seed {seed}"); continue
        for e in E_VALUES:
            for phase_deg in PHASE_DEG_VALUES:
                for relation in RELATIONS:
                    cells.append((seed, SEED_CKPTS[seed], e, phase_deg, relation,
                                   args.episodes, args.scratch))

    print(f"running {len(cells)} cells × {args.episodes} eps × {args.workers} parallel")
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    rows = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(run_cell, c) for c in cells]
        for i, fut in enumerate(as_completed(futures)):
            row = fut.result()
            rows.append(row)
            print(f"  [{i+1}/{len(cells)}] {row['seed']:>6} e={row['e']:.2f} "
                  f"p={row['phase_deg']:>3}° {row['relation']:>16}: "
                  f"{row['success_n']}/{row['total']} ({row['wall_s']}s)")
    print(f"\nTotal wall: {time.time()-t0:.0f}s")

    # Write CSV
    with open(args.out_csv, "w") as fh:
        w = csv.writer(fh)
        w.writerow(["seed", "e_fixed", "phase_gap_deg", "relation",
                     "success_n", "total", "wall_s"])
        for r in sorted(rows, key=lambda x: (x["seed"], x["e"], x["phase_deg"], x["relation"])):
            w.writerow([r["seed"], r["e"], r["phase_deg"], r["relation"],
                        r["success_n"], r["total"], r["wall_s"]])
    print(f"Wrote {args.out_csv}")


if __name__ == "__main__":
    main()
