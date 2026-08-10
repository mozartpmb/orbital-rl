"""T1 — success-box scan: headline regeneration + terminal-criterion tightening.

Evaluates the five committed Phase 5e canonical checkpoints (models/phase5e/)
at the headline conditions (LEO 300-800 km, both e ~ U(0, 0.05), phase gap ±π,
valid_init_only=1) under three success boxes:

  30 km / 50 m/s  — the historical far-field criterion (headline regeneration)
   5 km /  1 m/s  — defensible far-field rendezvous (review Tier-1 target)
   1 km / 0.5 m/s — unambiguous rendezvous

Records BOTH classifiers (legacy reward>0 and physical terminal-branch) plus
the terminal-cause histogram. Zero-shot: measures how much of the 30 km / 50 m/s
performance is grazing vs. genuinely closing, before any retraining.

Output: web_data/results/successbox_scan.csv
"""
import csv, os, subprocess, time
from concurrent.futures import ProcessPoolExecutor, as_completed

ROOT = "/Users/pete/space_training"
PUFFER = f"{ROOT}/pufferlib"

CKPTS = {
    "seed42": "../models/phase5e/seed42_stage4_best.pt",
    "seedA":  "../models/phase5e/seedA_stage4_best.pt",
    "seedB":  "../models/phase5e/seedB_stage4_best.pt",
    "seedC":  "../models/phase5e/seedC_stage4_best.pt",
    "seedD":  "../models/phase5e/seedD_stage4_best.pt",
}

BOXES = [
    ("30km_50ms", 30000.0, 50.0),
    ("5km_1ms",    5000.0,  1.0),
    ("1km_0.5ms",  1000.0,  0.5),
]

EPISODES = 200
ROLLOUT_SEED = 42


def run_cell(args):
    label, ckpt, box_label, radius_m, vel_ms = args
    out_dir = f"/tmp/t1_boxscan/{label}_{box_label}"
    cmd = [
        "python3", "scripts/orbital/eval_checkpoint.py", ckpt,
        "--episodes", str(EPISODES),
        "--e-max-target", "0.05", "--e-max-sat", "0.05",
        "--init-phase-gap-max", "3.14159",
        "--valid-init-only", "1",
        "--legacy-action-space", "10",
        "--rendezvous-radius-m", str(radius_m),
        "--rel-vel-tol-ms", str(vel_ms),
        "--out-dir", out_dir,
        "--seed", str(ROLLOUT_SEED),
    ]
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=PUFFER)
    dt = time.time() - t0
    succ_n = total = phys_n = phys_total = -1
    causes = ""
    for line in proc.stdout.splitlines():
        if "Success rate" in line:
            try:
                succ_n, total = map(int, line.split()[-2].split("/"))
            except Exception:
                pass
        elif "Physical success" in line:
            try:
                phys_n, phys_total = map(int, line.split()[2].split("/"))
            except Exception:
                pass
        elif "Terminal causes" in line:
            causes = line.split(":", 1)[1].strip()
    return {
        "ckpt": label, "box": box_label,
        "radius_m": radius_m, "vel_ms": vel_ms,
        "success_n": succ_n, "total": total,
        "phys_n": phys_n, "phys_total": phys_total,
        "causes": causes, "wall_s": round(dt, 1),
    }


def main():
    cells = [(lbl, ck, bl, r, v)
             for lbl, ck in CKPTS.items()
             for bl, r, v in BOXES]
    out_csv = os.path.join(ROOT, "web_data/results/successbox_scan.csv")
    print(f"{len(cells)} runs ({len(CKPTS)} ckpts x {len(BOXES)} boxes x {EPISODES} eps)")

    t0 = time.time()
    with open(out_csv, "w") as fh:
        w = csv.writer(fh)
        w.writerow(["ckpt", "box", "radius_m", "vel_ms",
                    "success_n", "total", "phys_n", "phys_total",
                    "causes", "wall_s"])
        with ProcessPoolExecutor(max_workers=4) as ex:
            futures = [ex.submit(run_cell, c) for c in cells]
            for i, fut in enumerate(as_completed(futures)):
                r = fut.result()
                w.writerow([r["ckpt"], r["box"], r["radius_m"], r["vel_ms"],
                            r["success_n"], r["total"], r["phys_n"],
                            r["phys_total"], r["causes"], r["wall_s"]])
                fh.flush()
                print(f"  [{i+1}/{len(cells)}] {r['ckpt']} {r['box']}: "
                      f"phys {r['phys_n']}/{r['phys_total']} ({r['wall_s']}s)")
    print(f"Done. Wall {(time.time()-t0)/60:.1f}m. CSV: {out_csv}")


if __name__ == "__main__":
    main()
