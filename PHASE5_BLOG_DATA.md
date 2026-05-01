# Phase 5 — Blog & Web Frontend Data Catalog

**Date:** 2026-05-01 · what's been generated for the web frontend / blog post, and where it lives.

---

## Headline numbers

| metric | value | source |
|---|---|---|
| Phase 5 deliverable | working agent at e ≤ 0.50 | — |
| Multi-seed at e=0.20 | **90.2% ± 2.0%** | `web_data/results/multiseed_escan.csv` |
| Multi-seed at e=0.50 | **83.4% ± 1.7%** | same |
| Multi-seed at e=0.70 (Molniya) | 63.0% ± 2.3% | same |
| Phase 4 conditions multi-seed | 91.4% ± 5.0% | `web_data/results/phase4_multiseed.csv` |
| Action distribution (any e_max) | ~94% time-warp, ~3% burns | `web_data/results/bonus_stats.json` |
| Median Δv vs Hohmann | 2.4-3.2× (across e_max) | same |

---

## Files generated under `web_data/`

### `web_data/results/` — aggregate stats

- `multiseed_escan.csv` — per-(seed, e_max) success counts. 5 seeds × 6 e_max levels × 200 eps. The headline capability surface table.
- `phase4_multiseed.csv` — Phase 4 conditions multi-seed. Confirms generalization tax.
- `seed42_training_progression.csv` — held-out e=0.20 success% at epochs 5/25/50/100/150/200/250/325. Substitute for wandb training curves where wandb wasn't enabled.
- `bonus_stats.json` — comprehensive per-(seed, e_max) action distribution, termination modes, fuel efficiency, episode lengths. The "what does the agent do?" data.
- `highlights_index.json` — index of curated highlight episodes per e_max with their summary scores.

### `web_data/runs/` — per-episode JSON trajectories

- `phase5e_seed42_e0.05/` — 50 episodes at low e (the Phase 5b deliverable regime)
- `phase5e_seed42_e0.20/` — 50 episodes at the canonical Phase 5d/5e closure regime
- `phase5e_seed42_e0.50/` — 50 episodes at the spec stretch goal
- `phase5e_seed42_e0.70/` — 50 episodes (incl. 19 failures) at Molniya territory
- `phase5e_progression/ep{005,025,050,100,150,200,250,325}/` — 10 episodes per checkpoint, same condition, illustrating how the policy evolves during Stage 4.0 retraining
- `highlights/` — curated picks per e_max (most-fuel-efficient success, near-miss failure, "patient" successes that exploit phasing)

JSON schema per episode: see `scripts/orbital/export_web_data.py`. Each file is self-contained (initial conditions, env config, target spec, per-step state + action + reward, body list).

### `plots/` — static PNG plots

- `p5e_capability_surface.png` — multi-seed mean ± std vs e_max (the headline chart)
- `p5e_training_progression.png` — seed 42's V-shaped recovery curve
- `p5e_action_distribution.png` — bar chart showing 94% warp / 3% burns
- `p5e_fuel_efficiency.png` — histogram of Δv/Hohmann ratio at e=0.20

---

## Cool data calls — possible blog narratives

### "The agent learned to wait, not burn."

- 94.2% of actions are time-warps (5min each)
- ~3% are small (5-10 m/s) burns; agent rarely uses 25 m/s burns (<0.1%)
- 0.7% radial burns — agent learned that radial is inefficient for orbit-shape changes
- This emerged purely from PPO + a fuel-budget environment; no demonstration data, no expert curriculum

### "The recipe was always capable; the test was wrong."

- Phase 5b/5c/5d spent ~50 hours of compute investigating "why doesn't the agent extend to e=0.20"
- Phase 5e Block I revealed: 64% of e_max=0.20 *initial states* had perigee < R_EARTH (sub-surface). Physically unrecoverable.
- The same Stage 4.0 ckpt, evaluated under `valid_init_only=1` rejection sampling, jumps from 16% → 93.5% at e=0.20. **No retraining.**
- The "wall" was a curriculum-sampling bug. One conditional in c_reset.

### "The training curve has a V, not an arrow."

- Seed 42 starts Stage 4.0 at 42% (warm-start from Stage 1.0).
- Drops to 30% at epoch 50 as the policy adjusts to random sat init.
- Recovers and climbs to 90% by epoch 325.
- This is `seed42_training_progression.csv`. The web frontend should plot this as the primary "see how it learned" widget.

### "Same recipe, three eccentricity regimes, graceful degradation."

| e_max | description | mean success |
|---|---|---|
| 0.05 | mostly circular | 93.7% |
| 0.20 | "moderate eccentricity" (think geo transfer arc) | 90.2% |
| 0.50 | "highly eccentric" (Molniya-class geometry) | 83.4% |
| 0.70 | extreme | 63.0% |

The recipe trained at **e=0.05** generalizes to e=0.70 at 63%. No high-e fine-tuning. **The capability surface chart `p5e_capability_surface.png` is the cleanest visual representation of the deliverable.**

### "Beating Hohmann by exploiting eccentricity."

- 0.71-0.80× of circular-Hohmann Δv on the most fuel-efficient successful episodes
- Even at e=0.70, the median actual Δv (250 m/s) is only 2.44× circular-Hohmann's analytic minimum
- Highlight `e_0.05_efficient`, `e_0.20_efficient`, `e_0.50_efficient`, `e_0.70_efficient` show the cleanest examples
- Each is a self-contained JSON trajectory ready for the web frontend's "this is what success looks like" widget

### "Failure modes by eccentricity."

Per-(seed, e_max) termination-mode breakdown is in `bonus_stats.json`. Summary at e=0.20 (seed 42):
- success: 184/200 (92%)
- safety_cap (timeout): 15
- collision: 1

At e=0.70:
- success: 129/200 (64.5%)
- safety_cap: most failures
- collision: rare under valid_init_only

### "What is a 'patient' rendezvous trajectory?"

The highlights include "patient" picks where the agent burns once early, then warps for many orbital periods until natural phasing brings the target close. This emerges from training; the curriculum doesn't reward patience explicitly. See `web_data/runs/highlights/e_0.70_patient_*.json`.

---

## What's missing / deferred

- **wandb training curves: now available for canonical seed 42 retrain** (`web_data/results/wandb_curves/stage_{1_0,4_0}_seed42.json` + `plots/p5e_wandb_curves.png`). The canonical retrain reproduces the Phase 5e seed 42 result (88% at e=0.20 vs 92% on the non-wandb run — within seed-noise band). Stage 1.0: 307 sampled rows; Stage 4.0: 383 sampled rows. Includes `environment/perf`, `losses/entropy`, `losses/explained_variance`, `environment/episode_return`, `agent_steps`, `learning_rate`, `SPS`. The other 4 multi-seed runs do *not* have wandb curves; the held-out-eval `seed42_training_progression.csv` is the substitute for those.
- **Capability surface across (phase × e × ω-relation).** Spec asked for a 64-256 cell grid; deferred due to compute time (single-env eval).
- **Original Phase 4 / Phase 5b episode dumps.** The `web_data/runs/` is currently Phase 5e-only. Adding Phase 4 baseline + Phase 5b deliverable trajectories would enable "before/after by phase" narratives. ~1 hour of additional eval if desired.
- **Animated GIFs of trajectories.** Out of scope; web frontend should render natively from JSON.

---

## Reproduction recipe

To regenerate from scratch:

```bash
# Multi-seed retrain (5 seeds, ~12 min each, parallelizable)
bash scripts/orbital/p5e_curriculum.sh all

# Multi-seed eccentricity scan (30 evals, ~30 min)
bash scripts/orbital/p5closure_multiseed_escan.sh

# Phase 4 multi-seed (5 evals, ~5 min)
bash scripts/orbital/p5closure_phase4_multiseed.sh

# Bonus stats (action dist, term modes, fuel eff)
python3 scripts/orbital/p5closure_bonus_stats.py

# Highlights
python3 scripts/orbital/p5closure_highlights.py

# Plots
python3 scripts/orbital/p5closure_plots.py

# Per-episode JSON exports for web frontend
python3 scripts/orbital/export_web_data.py --src-dir <eval_dir> \
    --out-dir web_data/runs/<label> --phase <label> \
    --checkpoint <ckpt> --e-max-target <e> --e-max-sat <e> \
    --same-orbit-init 0 --valid-init-only 1
```

---

*Phase 5 closure data ready. Web frontend implementation can begin.*
