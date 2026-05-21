"""Phase 5.5 P4 — Warp duration adequacy at GEO.

Run the Phase 5b canonical ckpt at GEO altitudes, log the action histogram
across an episode, and count how many warp-5min actions get chained. Per
the spec §2.4 verdict thresholds:
  < 10 warps: current 5-min warp is fine at GEO
  10-50:     marginal; flag 30-min or 1-hour warp
  > 50:      structurally inadequate; longer warps needed

Also reports max consecutive warps, time to first non-warp action, total
sim time covered, and termination mode.

Run:
    python3 scripts/orbital/p5_5_p4_warp_adequacy.py
"""
import math
import os
import subprocess
import sys

import numpy as np


# Action names matching orbital.h:58-69
ACTION_NAMES = [
    "coast",      # 0
    "pro+5",      # 1
    "pro+10",     # 2
    "pro+25",     # 3
    "retro-5",    # 4
    "retro-10",   # 5
    "retro-25",   # 6
    "rad+10",     # 7
    "rad-10",     # 8
    "warp5min",   # 9
]
WARP_ACTION = 9
DT = 60.0
WARP_DT = 5 * DT  # 300 s per warp action


def main():
    # Use P2's GEO_low_e cell with the Phase 5b canonical ckpt.
    ckpt = "/Users/pete/space_training/pufferlib/experiments/puffer_orbital_177750405236/model_puffer_orbital_000350.pt"
    out_dir = "/tmp/p5_5_p4"
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 90)
    print("Phase 5.5 P4 — Warp duration adequacy at GEO")
    print("=" * 90)
    print(f"Checkpoint: {ckpt}")
    print(f"Action set: {ACTION_NAMES}")
    print(f"DT = {DT} s, WARP_DT = {WARP_DT} s ({WARP_DT/60:.0f} min)")
    print()

    # 5 episodes to get a sample of behaviors
    cmd = [
        "python3",
        "/Users/pete/space_training/pufferlib/scripts/orbital/eval_checkpoint.py",
        ckpt,
        "--episodes", "5",
        "--seed", "42",
        "--e-max-target", "0.05",
        "--e-max-sat", "0.05",
        "--init-phase-gap-max", "3.14159",
        "--valid-init-only", "1",
        "--max-valid-init-attempts", "4096",
        "--gave-up-action", "terminate",
        "--obs-alt-scale-m", "4.2e7",
        "--phi-orbit-scale-k", "0.001",
        "--a-min-override", "42000e3".replace("e", "e+0") if False else "42000000.0",
        "--a-max-override", "42500000.0",
        "--out-dir", out_dir,
    ]
    print("Running 5 episodes at GEO low-e...")
    print(" ".join(cmd))
    print()
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    if r.returncode != 0:
        print("ERROR:", r.stderr[:500])
        sys.exit(1)
    # Print summary tail
    for line in r.stdout.splitlines()[-15:]:
        print(line)
    print()

    # Per-episode action histogram + warp analysis
    import glob
    files = sorted(glob.glob(f"{out_dir}/ep_*.npz"))
    print(f"Analyzing {len(files)} trajectories...")
    print(f"{'episode':>8} {'steps':>6} {'reward':>8} "
          f"{'n_warps':>8} {'max_consec':>11} {'first_burn':>11} "
          f"{'sim_hr':>7} {'unique_actions':>15} {'term_a':>7}")
    print("-" * 100)

    total_warps = []
    for f in files:
        d = np.load(f)
        actions = d["action"].astype(int)
        n_steps = len(actions)
        ep_reward = float(d["episode_reward"][0])
        # Episode termination action — the last action before terminal
        term_action = ACTION_NAMES[actions[-1]] if n_steps else "?"
        n_warps = int(np.sum(actions == WARP_ACTION))
        # Max consecutive warp run
        max_consec = 0
        cur = 0
        for a in actions:
            if a == WARP_ACTION:
                cur += 1
                max_consec = max(max_consec, cur)
            else:
                cur = 0
        # First non-warp non-coast action step
        first_burn = -1
        for i, a in enumerate(actions):
            if a != WARP_ACTION and a != 0:
                first_burn = i
                break
        # Sim time covered (warps advance 5× faster than other steps)
        sim_seconds = sum(WARP_DT if a == WARP_ACTION else DT for a in actions)
        sim_hours = sim_seconds / 3600
        unique_actions = len(set(actions))
        ep_idx = int(d["episode_id"][0])
        total_warps.append(n_warps)
        print(
            f"{ep_idx:>8} {n_steps:>6} {ep_reward:>8.2f} "
            f"{n_warps:>8} {max_consec:>11} {first_burn:>11} "
            f"{sim_hours:>7.1f} {unique_actions:>15} {term_action:>7}"
        )

    print()
    print("--- Verdict per spec §2.4 ---")
    if not total_warps:
        print("No data.")
    else:
        med = int(np.median(total_warps))
        mn = min(total_warps)
        mx = max(total_warps)
        print(f"Warp counts per episode: min={mn} median={med} max={mx}")
        if mx < 10:
            verdict = "FINE — current 5-min warp is adequate at GEO"
        elif mx < 50:
            verdict = "MARGINAL — consider adding 30-min or 1-hour warp"
        else:
            verdict = "STRUCTURALLY INADEQUATE — must add longer warps before training"
        print(f"Verdict: {verdict}")

    print()
    print("Note: Phase 5b ckpt was trained at LEO and shows 0% success at GEO.")
    print("These episodes likely terminate via safety-cap (2000 steps) or escape.")
    print("Episode reward < 0 typically means terminal failure (collision/escape/stranded).")
    print("Reward = 0 with last action != 0 suggests the gave_up_action=terminate fired")
    print("(i.e., doomed init at cap exhaust — should be rare at GEO given P1 E1 results).")


if __name__ == "__main__":
    main()
