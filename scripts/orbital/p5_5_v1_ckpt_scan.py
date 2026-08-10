"""V1 — full ckpt scan for Probe 2 + F-B+ runs.

For F-B, F-C, F-B+ × 3 seeds × all saved ckpts, eval at Stage 5.5.1 conditions
and LEO regression conditions. Identifies peak ckpt + Pareto-optimal ckpt per
(protocol, seed).
"""
import csv
import glob
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

ROOT = "/Users/pete/space_training"
PUFFER = f"{ROOT}/pufferlib"
OUT_CSV = "/tmp/p5_5_v1_scan.csv"

RUNS = {
    ("FB",     42):       "puffer_orbital_177938770127",
    ("FB",     31415):    "puffer_orbital_177938778683",
    ("FB",     20260423): "puffer_orbital_177938788103",
    ("FC",     42):       "puffer_orbital_177938807595",
    ("FC",     31415):    "puffer_orbital_177938823952",
    ("FC",     20260423): "puffer_orbital_177938839214",
    ("FBplus", 42):       "puffer_orbital_177939089240",
    ("FBplus", 31415):    "puffer_orbital_177939096896",
    ("FBplus", 20260423): "puffer_orbital_177939105753",
}

CONDITIONS = {
    "stage5_5_1": ("6.671e6", "8.5e6"),
    "leo":        ("6.671e6", "7.171e6"),
}

EPISODES = 200
SEED = 42


def run_one(args):
    protocol, train_seed, ckpt, epoch, cond_label, a_min, a_max = args
    out_dir = f"/tmp/p5_5_v1_scan/{protocol}_s{train_seed}_e{epoch:03d}_{cond_label}"
    cmd = [
        "python3", "scripts/orbital/eval_checkpoint.py", ckpt,
        "--episodes", str(EPISODES),
        "--e-max-target", "0.05", "--e-max-sat", "0.05",
        "--init-phase-gap-max", "3.14159",
        "--valid-init-only", "1",
        "--a-min-override", a_min, "--a-max-override", a_max,
        "--legacy-action-space", "10",
        "--out-dir", out_dir,
        "--seed", str(SEED),
    ]
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
    return (protocol, train_seed, epoch, cond_label, succ_n, total, round(dt, 1))


def main():
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    tasks = []
    for (proto, seed), dirname in RUNS.items():
        ckpt_dir = f"{PUFFER}/experiments/{dirname}"
        ckpts = sorted(glob.glob(f"{ckpt_dir}/model_puffer_orbital_*.pt"))
        for ckpt in ckpts:
            epoch = int(os.path.basename(ckpt).split("_")[-1].replace(".pt", ""))
            for cond_label, (a_min, a_max) in CONDITIONS.items():
                tasks.append((proto, seed, ckpt, epoch, cond_label, a_min, a_max))

    print(f"Total evals: {len(tasks)} ({EPISODES} eps each)")
    print(f"Workers: {workers}")

    t0 = time.time()
    with open(OUT_CSV, "w") as fh:
        w = csv.writer(fh)
        w.writerow(["protocol", "train_seed", "epoch", "condition", "success_n", "total", "wall_s"])
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(run_one, t) for t in tasks]
            for i, fut in enumerate(as_completed(futs)):
                r = fut.result()
                w.writerow(r)
                fh.flush()
                if (i+1) % 50 == 0 or i+1 == len(tasks):
                    el = time.time() - t0
                    eta = el / (i+1) * (len(tasks) - i - 1)
                    print(f"  [{i+1}/{len(tasks)}] elapsed {el/60:.1f}m, eta {eta/60:.1f}m")
    print(f"DONE wall {(time.time()-t0)/60:.1f}m. CSV: {OUT_CSV}")


if __name__ == "__main__":
    main()
