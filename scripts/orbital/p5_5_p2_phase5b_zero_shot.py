"""Phase 5.5 P2 — Phase 5b canonical ckpt zero-shot at high altitude.

Per Phase 5.5 spec §2.2: eval the Phase 5b Stage 4.0 canonical ckpt across
five cells (LEO baseline, MEO low-e, MEO eccentric, GEO low-e, GEO eccentric)
to measure how much altitude-generalization (if any) exists in the LEO-trained
recipe. Pre-committed as informational, not deciding.

Subprocess-spawns eval_checkpoint.py per cell with 200 episodes × 3 rollout
seeds = 600 eps per cell. Two sweeps:
  Sweep A: obs_alt_scale_m=1.6e6 (LEO default; obs saturate at high alt).
  Sweep B: obs_alt_scale_m=4.2e7 (GEO scale; LEO obs look small).

Output: /tmp/p5_5_p2_results.csv with per-cell success rates.

Run:
    python3 scripts/orbital/p5_5_p2_phase5b_zero_shot.py
"""
import csv
import math
import os
import re
import subprocess
import sys
import time

R_EARTH = 6.371e6

CKPT = "/Users/pete/space_training/pufferlib/experiments/puffer_orbital_177750405236/model_puffer_orbital_000350.pt"
EVAL_SCRIPT = "/Users/pete/space_training/pufferlib/scripts/orbital/eval_checkpoint.py"

# (cell_id, alt_min_m, alt_max_m, e_max, label)
# alt_min/max = -1 means use env default LEO 300-800 km
CELLS = [
    ("LEO_baseline",   -1.0,        -1.0,        0.05, "LEO 300-800km / e≤0.05 (control)"),
    ("MEO_low_e",      10000e3,    11000e3,     0.05, "a∈[10-11Mm] / e≤0.05"),
    ("MEO_eccentric",  10000e3,    11000e3,     0.30, "a∈[10-11Mm] / e≤0.30"),
    ("GEO_low_e",      42000e3,    42500e3,     0.05, "a∈[42.0-42.5Mm] / e≤0.05"),
    ("GEO_eccentric",  42000e3,    42500e3,     0.30, "a∈[42.0-42.5Mm] / e≤0.30"),
]

ROLLOUT_SEEDS = [42, 1337, 31415]
N_EPISODES = 200

# Two obs/Φ scale sweeps
SCALE_SWEEPS = [
    ("LEO_default",  1.6e6, 0.001),
    ("GEO_default",  4.2e7, 0.001),
]


def run_one_cell(cell_id, alt_min, alt_max, e_max, rollout_seed,
                  obs_alt_scale, phi_K):
    """Subprocess-spawn eval_checkpoint.py for one cell + seed."""
    out_dir = f"/tmp/p5_5_p2/{cell_id}_seed{rollout_seed}_scale_{obs_alt_scale:.0e}_K{phi_K}"
    os.makedirs(out_dir, exist_ok=True)
    cmd = [
        "python3", EVAL_SCRIPT, CKPT,
        "--episodes", str(N_EPISODES),
        "--seed", str(rollout_seed),
        "--e-max-target", str(e_max),
        "--e-max-sat", str(e_max),
        "--init-phase-gap-max", "3.14159",
        "--valid-init-only", "1",
        "--max-valid-init-attempts", "4096",
        "--gave-up-action", "terminate",
        "--obs-alt-scale-m", f"{obs_alt_scale}",
        "--phi-orbit-scale-k", f"{phi_K}",
        "--out-dir", out_dir,
    ]
    if alt_min >= R_EARTH:
        cmd += ["--a-min-override", f"{alt_min}",
                "--a-max-override", f"{alt_max}"]
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    wall = time.time() - t0
    if r.returncode != 0:
        return None, wall, r.stderr[:200]
    # Parse "Success rate:    N/M (X.X%)"
    m = re.search(r"Success rate:\s+(\d+)/(\d+)\s+\((\d+\.?\d*)%\)", r.stdout)
    if not m:
        return None, wall, r.stdout[-300:]
    success_n = int(m.group(1))
    total = int(m.group(2))
    return success_n, total, wall, None


def main():
    print("=" * 90)
    print("Phase 5.5 P2 — Phase 5b canonical ckpt zero-shot at altitude")
    print("=" * 90)
    print(f"Checkpoint: {CKPT}")
    print(f"Episodes per cell: {N_EPISODES} × {len(ROLLOUT_SEEDS)} rollout seeds")
    print()

    os.makedirs("/tmp/p5_5_p2", exist_ok=True)
    csv_path = "/tmp/p5_5_p2_results.csv"

    with open(csv_path, "w") as fh:
        w = csv.writer(fh)
        w.writerow([
            "cell_id", "alt_min_m", "alt_max_m", "e_max",
            "obs_alt_scale_m", "phi_orbit_scale_k",
            "rollout_seed", "success_n", "total", "success_pct",
            "wall_s", "error"
        ])

        print(f"{'cell':<16} {'alt scale':<14} {'K':>6} {'seed':>6} "
              f"{'success':>8} {'pct':>7} {'wall_s':>7}")
        print("-" * 90)

        for scale_label, obs_scale, phi_K in SCALE_SWEEPS:
            for cell_id, alt_min, alt_max, e_max, desc in CELLS:
                # Skip the "rescaled" sweep for LEO baseline (would change V3-anchored numbers)
                if cell_id == "LEO_baseline" and scale_label != "LEO_default":
                    continue
                seed_results = []
                for rs in ROLLOUT_SEEDS:
                    out = run_one_cell(cell_id, alt_min, alt_max, e_max, rs,
                                         obs_scale, phi_K)
                    if len(out) == 3:
                        sn, total, wall = out
                        err = None
                    else:
                        sn, total, wall, err = out
                    if sn is None:
                        sn, total, pct = 0, N_EPISODES, 0.0
                        if err is None:
                            err = "parse_failed"
                    else:
                        pct = 100.0 * sn / total
                    seed_results.append((rs, sn, total, pct, wall, err))
                    w.writerow([cell_id, alt_min, alt_max, e_max,
                                 obs_scale, phi_K, rs, sn, total, pct, wall,
                                 err if err else ""])
                    fh.flush()
                    print(
                        f"{cell_id:<16} {scale_label:<14} {phi_K:>6.3f} {rs:>6} "
                        f"{sn}/{total:<5} {pct:>6.1f}% {wall:>6.1f}s"
                        + (f" ERR: {err[:50]}" if err else "")
                    )
                # Mean across seeds
                pcts = [r[3] for r in seed_results if r[1] is not None]
                if pcts:
                    mean_pct = sum(pcts) / len(pcts)
                    print(
                        f"{cell_id:<16} {scale_label:<14} {phi_K:>6.3f} "
                        f"{'MEAN':>6} {'':>8} {mean_pct:>6.1f}%"
                    )
                print()

    print(f"\nResults written to {csv_path}")


if __name__ == "__main__":
    main()
