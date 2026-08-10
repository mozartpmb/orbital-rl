"""Phase 5 wrap-up W2: full capability surface per spec §2.

Per spec:
  - phase_gap   ∈ {30°, 90°, 150°, 180°}      (4)
  - e_target    ∈ {0.0, 0.05, 0.20, 0.50, 0.70} (5)
  - e_sat       ∈ {0.0, 0.05, 0.20, 0.50}      (4)
  - relation    ∈ {same_orbit, different_a, different_ω, fully_random}  (4)
  - eps per cell = 200 × 3 rollout seeds = 600

5 training seeds × 4×5×4×4 = 320 cells/ckpt × 600 eps = 192K eps/seed
Total: 960K eps. With 4 parallel workers at ~30s/cell → 3-4 hrs wall.

Skips physically-infeasible cells (target perigee a*(1-e_target) cannot be
made valid in any altitude band, etc.) — these are recorded as N/A.

Output: web_data/results/phase5_capability_surface_full.csv
"""
import argparse, csv, math, os, subprocess, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed

R_EARTH = 6.371e6
EARTH_KEEPOUT = R_EARTH + 200e3

SEED_CKPTS = {
    "seed42":    "experiments/puffer_orbital_177765503091/model_puffer_orbital_000325.pt",
    "seedA":     "experiments/puffer_orbital_177765655537/model_puffer_orbital_000375.pt",
    "seedB":     "experiments/puffer_orbital_177765658166/model_puffer_orbital_000250.pt",
    "seedC":     "experiments/puffer_orbital_177765658729/model_puffer_orbital_000350.pt",
    "seedD":     "experiments/puffer_orbital_177765659007/model_puffer_orbital_000275.pt",
}

# Per spec
E_TARGET_VALUES = [0.0, 0.05, 0.20, 0.50, 0.70]
E_SAT_VALUES    = [0.0, 0.05, 0.20, 0.50]
PHASE_DEG_VALUES = [30, 90, 150, 180]
RELATIONS = ["same_orbit", "different_a", "different_omega", "fully_random"]
ROLLOUT_SEEDS = [42, 1337, 20260423]


def alt_band_for_e(e_target, e_sat):
    """Pick (a_min, a_max) so that BOTH sat and target perigees can be valid.
    Need a*(1-max(e_target, e_sat)) >= EARTH_KEEPOUT.
    For both circular, use trained LEO band (300-800km)."""
    e_max = max(e_target, e_sat)
    if e_max <= 0.001:
        return (-1.0, -1.0)  # default LEO
    rp_min = EARTH_KEEPOUT
    rp_max = EARTH_KEEPOUT * 1.7
    a_min = rp_min / (1.0 - e_max)
    a_max = rp_max / (1.0 - e_max)
    return (a_min, a_max)


def run_cell(args):
    seed_label, ckpt, e_t, e_s, phase_deg, relation, rollout_seed, episodes = args
    a_min, a_max = alt_band_for_e(e_t, e_s)
    phase_rad = math.radians(phase_deg)

    cell_id = f"{seed_label}_et{e_t}_es{e_s}_p{phase_deg}_{relation}_rs{rollout_seed}"
    out_dir = f"/tmp/p5wrap_full_surface/{cell_id}"

    cmd = [
        "python3", "scripts/orbital/eval_checkpoint.py", ckpt,
        "--episodes", str(episodes),
        "--e-max-target", str(max(e_t, 0.05)),
        "--e-max-sat", str(max(e_s, 0.05)),
        "--init-phase-gap-max", "3.14159",
        "--valid-init-only", "1",
        "--e-target-fixed", str(e_t),
        "--e-sat-fixed", str(e_s),
        "--phase-gap-fixed", f"{phase_rad:.5f}",
        "--out-dir", out_dir,
        "--seed", str(rollout_seed),
    ]

    # Relation-specific kwargs
    if relation == "same_orbit":
        # sat & target share a, e, ω — only θ differs
        cmd += ["--same-orbit-init", "1"]
        cmd += ["--omega-offset-fixed", "0.0"]
    elif relation == "different_a":
        # sat & target share e, ω but differ in a
        cmd += ["--same-orbit-init", "0"]
        cmd += ["--omega-offset-fixed", "0.0"]
    elif relation == "different_omega":
        # sat & target share a, e but differ in ω
        # Need same_orbit_init=1 to force same a (and e via fixed-e), then offset ω
        cmd += ["--same-orbit-init", "1"]
        cmd += ["--omega-offset-fixed", "1.5708"]  # π/2 = 90° offset
    else:  # fully_random
        cmd += ["--same-orbit-init", "0"]
        # No omega offset → target & sat ω sampled independently

    if a_min > 0:
        cmd += ["--a-min-override", str(a_min), "--a-max-override", str(a_max)]

    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd="pufferlib")
    dt = time.time() - t0
    succ_n = total = -1
    for line in proc.stdout.splitlines():
        if "Success rate" in line:
            try:
                frac = line.split()[-2]
                succ_n, total = map(int, frac.split("/"))
            except: pass
            break
    return {
        "seed": seed_label, "e_target": e_t, "e_sat": e_s,
        "phase_deg": phase_deg, "relation": relation, "rollout_seed": rollout_seed,
        "success_n": succ_n, "total": total, "wall_s": round(dt, 1),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--out-csv", default="web_data/results/phase5_capability_surface_full.csv")
    p.add_argument("--seeds", default=",".join(SEED_CKPTS.keys()))
    args = p.parse_args()

    cells = []
    for seed in args.seeds.split(","):
        if seed not in SEED_CKPTS: continue
        for e_t in E_TARGET_VALUES:
            for e_s in E_SAT_VALUES:
                for phase_deg in PHASE_DEG_VALUES:
                    for relation in RELATIONS:
                        for rollout_seed in ROLLOUT_SEEDS:
                            cells.append((seed, SEED_CKPTS[seed], e_t, e_s,
                                          phase_deg, relation, rollout_seed,
                                          args.episodes))

    print(f"running {len(cells)} cells × {args.episodes} eps × {args.workers} workers")
    print(f"  axes: 5 training_seed × 5 e_target × 4 e_sat × 4 phase × 4 relation × 3 rollout_seed")
    print(f"  est compute @ ~30s/cell sequential / {args.workers}-way parallel = "
          f"~{30 * len(cells) / args.workers / 60:.0f} min wall")
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)

    rows = []
    t0 = time.time()
    # Stream-write in case of interrupt
    with open(args.out_csv, "w") as fh:
        w = csv.writer(fh)
        w.writerow(["seed", "e_target", "e_sat", "phase_gap_deg", "relation",
                     "rollout_seed", "success_n", "total", "wall_s"])
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futures = [ex.submit(run_cell, c) for c in cells]
            for i, fut in enumerate(as_completed(futures)):
                r = fut.result()
                rows.append(r)
                w.writerow([r["seed"], r["e_target"], r["e_sat"], r["phase_deg"],
                            r["relation"], r["rollout_seed"],
                            r["success_n"], r["total"], r["wall_s"]])
                fh.flush()
                if (i+1) % 50 == 0 or i+1 == len(cells):
                    elapsed = time.time() - t0
                    rate = (i+1) / elapsed
                    remaining = (len(cells) - i - 1) / rate
                    print(f"  [{i+1}/{len(cells)}] {r['seed']:>6} et={r['e_target']:.2f} "
                          f"es={r['e_sat']:.2f} p={r['phase_deg']:>3}° "
                          f"{r['relation']:>15} rs={r['rollout_seed']}: "
                          f"{r['success_n']}/{r['total']} | "
                          f"elapsed {elapsed/60:.1f}m, eta {remaining/60:.1f}m")
    print(f"\nTotal wall: {(time.time()-t0)/60:.1f}m")
    print(f"Wrote {args.out_csv}")


if __name__ == "__main__":
    main()
