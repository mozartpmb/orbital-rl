"""Phase 5.5 Probe 1 — capability decomposition.

Surface eval of canonical Phase 5b ckpt on a 4 alt × 6 e × 4 phase × 2 relation grid,
with LEO obs scales (per stochastic-probe finding: scaled obs at MEO/GEO destroys the
policy). Diagonal e_target = e_sat per spec §1.2 Reduction A, plus 8 off-diagonal cells
at LEO and GEO (Reduction B).

Per cell: 200 episodes × 3 rollout seeds = 600 eps.
Total: 192 + 8 = 200 cells × 600 eps = 120K eps.

Output: web_data/results/p5_5_probe1_decompose.csv
"""
import argparse, csv, math, os, subprocess, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed

ROOT = "/Users/pete/space_training"
PUFFER = f"{ROOT}/pufferlib"
CKPT = "experiments/puffer_orbital_177750405236/model_puffer_orbital_000350.pt"

# 4 altitude bands. (a_min, a_max) in meters.
ALT_BANDS = {
    "LEO":      (6.671e6, 7.171e6),
    "MEO_low":  (6.671e6, 1.2e7),
    "MEO":      (6.671e6, 2.6e7),
    "GEO":      (6.671e6, 4.25e7),
}

# Diagonal e values per spec §1.2.
E_VALUES = [0.0, 0.05, 0.10, 0.20, 0.30, 0.50]
PHASE_DEG_VALUES = [30, 90, 150, 180]
RELATIONS = ["same_orbit", "fully_random"]  # spec §1.2 axis
ROLLOUT_SEEDS = [42, 1337, 31415]

# Off-diagonal sentinels per Reduction B (4 each at LEO + GEO).
OFF_DIAGONAL = [
    # (e_target, e_sat) — sample asymmetric combinations
    (0.30, 0.0),
    (0.0,  0.30),
    (0.50, 0.05),
    (0.05, 0.50),
]


def build_cells():
    cells = []
    # Diagonal: 4 alt × 6 e × 4 phase × 2 relation = 192
    for alt_label, (a_min, a_max) in ALT_BANDS.items():
        for e in E_VALUES:
            for phase_deg in PHASE_DEG_VALUES:
                for relation in RELATIONS:
                    cells.append(("diag", alt_label, a_min, a_max, e, e, phase_deg, relation))
    # Off-diagonal: 4 pairs at LEO + 4 at GEO = 8
    for alt_label in ["LEO", "GEO"]:
        a_min, a_max = ALT_BANDS[alt_label]
        for e_t, e_s in OFF_DIAGONAL:
            # Use phase=90 + fully_random as representative
            cells.append(("off_diag", alt_label, a_min, a_max, e_t, e_s, 90, "fully_random"))
    return cells


def run_cell(args):
    (kind, alt_label, a_min, a_max, e_t, e_s, phase_deg,
     relation, rollout_seed, episodes) = args
    phase_rad = math.radians(phase_deg)
    cell_id = (f"{kind}_{alt_label}_et{e_t:.2f}_es{e_s:.2f}_p{phase_deg}_"
               f"{relation}_rs{rollout_seed}")
    out_dir = f"/tmp/p5_5_probe1/{cell_id}"

    cmd = [
        "python3", "scripts/orbital/eval_checkpoint.py", CKPT,
        "--episodes", str(episodes),
        "--e-max-target", str(max(e_t, 0.05)),
        "--e-max-sat",    str(max(e_s, 0.05)),
        "--init-phase-gap-max", "3.14159",
        "--valid-init-only", "1",
        "--e-target-fixed", str(e_t),
        "--e-sat-fixed",    str(e_s),
        "--phase-gap-fixed", f"{phase_rad:.5f}",
        "--a-min-override", str(a_min),
        "--a-max-override", str(a_max),
        # LEO obs scales (per stochastic-probe finding)
        "--obs-alt-scale-m", "1.6e6",
        "--lvlh-scale-m", "6.371e6",
        "--phi-orbit-scale-k", "0.001",
        "--legacy-action-space", "10",
        "--out-dir", out_dir,
        "--seed", str(rollout_seed),
    ]
    if relation == "same_orbit":
        cmd += ["--same-orbit-init", "1", "--omega-offset-fixed", "0.0"]
    else:  # fully_random
        cmd += ["--same-orbit-init", "0"]

    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=PUFFER)
    dt = time.time() - t0
    succ_n = total = -1
    for line in proc.stdout.splitlines():
        if "Success rate" in line:
            try:
                frac = line.split()[-2]
                succ_n, total = map(int, frac.split("/"))
            except Exception:
                pass
            break
    return {
        "kind": kind, "alt": alt_label, "e_target": e_t, "e_sat": e_s,
        "phase_deg": phase_deg, "relation": relation, "rollout_seed": rollout_seed,
        "success_n": succ_n, "total": total, "wall_s": round(dt, 1),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--out-csv", default="web_data/results/p5_5_probe1_decompose.csv")
    args = p.parse_args()

    cells_def = build_cells()
    cells = []
    for c in cells_def:
        for rs in ROLLOUT_SEEDS:
            cells.append(c + (rs, args.episodes))

    out_csv = os.path.join(ROOT, args.out_csv)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    print(f"Cells: {len(cells_def)} unique × {len(ROLLOUT_SEEDS)} seeds = {len(cells)} runs")
    print(f"Output: {out_csv}")

    t0 = time.time()
    with open(out_csv, "w") as fh:
        w = csv.writer(fh)
        w.writerow(["kind", "alt", "e_target", "e_sat", "phase_deg", "relation",
                    "rollout_seed", "success_n", "total", "wall_s"])
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futures = [ex.submit(run_cell, c) for c in cells]
            for i, fut in enumerate(as_completed(futures)):
                r = fut.result()
                w.writerow([r["kind"], r["alt"], r["e_target"], r["e_sat"],
                            r["phase_deg"], r["relation"], r["rollout_seed"],
                            r["success_n"], r["total"], r["wall_s"]])
                fh.flush()
                if (i+1) % 50 == 0 or i+1 == len(cells):
                    el = time.time() - t0
                    eta = el / (i+1) * (len(cells) - i - 1)
                    print(f"  [{i+1}/{len(cells)}] elapsed {el/60:.1f}m, eta {eta/60:.1f}m")
    print(f"Done. Wall {(time.time()-t0)/60:.1f}m. CSV: {out_csv}")


if __name__ == "__main__":
    main()
