"""LEO-altitude probe at fixed e values to characterize in-distribution capability.

Recipe was trained at altitudes 300-800km with e_max_sat = 0.05. This probe
measures per-condition capability at the trained altitude band, for e values
where the altitude band can sustain valid perigees (e ≤ ~0.05 at most cells).

Outputs `web_data/results/phase5_leo_probe.csv` with columns:
  seed, e_fixed, phase_gap_deg, success_n, total
"""
import csv, math, os, subprocess, time
from concurrent.futures import ProcessPoolExecutor, as_completed

R_EARTH = 6.371e6
EARTH_KEEPOUT = R_EARTH + 200e3

SEED_CKPTS = {
    "seed42": "experiments/puffer_orbital_177765503091/model_puffer_orbital_000325.pt",
    "seedA":  "experiments/puffer_orbital_177765655537/model_puffer_orbital_000375.pt",
    "seedB":  "experiments/puffer_orbital_177765658166/model_puffer_orbital_000250.pt",
    "seedC":  "experiments/puffer_orbital_177765658729/model_puffer_orbital_000350.pt",
    "seedD":  "experiments/puffer_orbital_177765659007/model_puffer_orbital_000275.pt",
}

# At LEO (300-800km, a in [6.671e6, 7.171e6]), max valid e is 1 - 6.571e6/a.
# At a=7.171e6, that's 0.0837. So e=0.10+ won't have many valid samples.
E_VALUES = [0.0, 0.01, 0.025, 0.05, 0.075, 0.10]
PHASE_DEG_VALUES = [30, 90, 150, 180]


def run_cell(args):
    seed_label, ckpt, e, phase_deg, episodes = args
    phase_rad = math.radians(phase_deg)
    cell_id = f"leo_{seed_label}_e{e}_p{phase_deg}"
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
        "--same-orbit-init", "0",  # fully random within LEO
        "--out-dir", f"/tmp/p5wrap_leo/{cell_id}",
        "--seed", "42",
        # No alt override — use default LEO 300-800km
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd="pufferlib")
    succ = total = -1
    for line in proc.stdout.splitlines():
        if "Success rate" in line:
            try:
                frac = line.split()[-2]
                succ, total = map(int, frac.split("/"))
            except: pass
            break
    return {"seed": seed_label, "e": e, "phase_deg": phase_deg,
            "success_n": succ, "total": total}


def main():
    cells = []
    for seed in SEED_CKPTS:
        for e in E_VALUES:
            for phase_deg in PHASE_DEG_VALUES:
                cells.append((seed, SEED_CKPTS[seed], e, phase_deg, 100))

    print(f"running {len(cells)} cells (LEO-only, fully_random)")
    rows = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=4) as ex:
        futures = [ex.submit(run_cell, c) for c in cells]
        for i, fut in enumerate(as_completed(futures)):
            r = fut.result()
            rows.append(r)
            print(f"  [{i+1}/{len(cells)}] {r['seed']:>6} e={r['e']:.3f} "
                  f"p={r['phase_deg']:>3}°: {r['success_n']}/{r['total']}")
    print(f"\nTotal wall: {time.time()-t0:.0f}s")

    out = "web_data/results/phase5_leo_probe.csv"
    with open(out, "w") as f:
        w = csv.writer(f)
        w.writerow(["seed", "e_fixed", "phase_gap_deg", "success_n", "total"])
        for r in sorted(rows, key=lambda x: (x["seed"], x["e"], x["phase_deg"])):
            w.writerow([r["seed"], r["e"], r["phase_deg"], r["success_n"], r["total"]])
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
