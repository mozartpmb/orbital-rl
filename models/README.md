# Phase 5e Canonical Model Weights

Tracked checkpoints (~8 MB total). The full training runs (3.6 GB of intermediate ckpts every 5 epochs × 5 seeds) are in `pufferlib/experiments/` and gitignored — the canonical ones below are sufficient to reproduce all Phase 5e claims.

## Multi-seed deliverable

5 seeds, all reach 86-92% at e=0.20 under `valid_init_only=1`. Curriculum: Stage 1.0 (e=0.05, same_orbit_init=1) → Stage 4.0 (e=0.05, random sat init).

| File | Source dir | Best epoch | e=0.20 success | Notes |
|---|---|---|---|---|
| `phase5e/seed42_stage4_best.pt` | puffer_orbital_177765503091 | 325 | 92.0% | seed 42, no wandb |
| `phase5e/seedA_stage4_best.pt` | puffer_orbital_177765655537 | 375 | 91.0% | parallel seed (one of 1337/20260423/31415/2718, mapping lost to a race condition) |
| `phase5e/seedB_stage4_best.pt` | puffer_orbital_177765658166 | 250 | 86.5% | parallel seed |
| `phase5e/seedC_stage4_best.pt` | puffer_orbital_177765658729 | 350 | 91.5% | parallel seed |
| `phase5e/seedD_stage4_best.pt` | puffer_orbital_177765659007 | 275 | 90.0% | parallel seed |

Multi-seed mean: **90.2 ± 2.0%** at e=0.20.

## Canonical (wandb-tracked) retrain

Same recipe as `seed42_stage4_best.pt`, retrained from scratch on 2026-05-01 with `--wandb` enabled to produce the training curves under `web_data/results/wandb_curves/`:

| File | Source dir | Best epoch | e=0.20 success | wandb run |
|---|---|---|---|---|
| `phase5e/canonical_seed42_stage4_best.pt` | puffer_orbital_qlpmgyko | 200 | 88.0% | qlpmgyko ("deep-forest-2") |

The wandb-canonical ckpt is the one whose training curves are plotted in `plots/p5e_wandb_curves.png`.

## Stage 1.0 warm-start

| File | Source dir | Best epoch | Notes |
|---|---|---|---|
| `phase5e/seed42_stage1_warmstart.pt` | puffer_orbital_177765458150 | 300 | Re-use as `--load-model-path` to skip Stage 1.0 when re-running Stage 4.0 ablations |

## Training-progression checkpoints (seed 42)

For the "agent learns over training" web frontend widget. Same source dir as `seed42_stage4_best.pt` (177765503091), evaluated at e=0.20 with valid_init_only=1:

| Epoch | Held-out e=0.20 | File |
|---|---|---|
| 5 | 42% | `phase5e/seed42_progression/ep005.pt` |
| 25 | 36% | `seed42_progression/ep025.pt` |
| 50 | 30% (warm-start adjustment dip) | `seed42_progression/ep050.pt` |
| 100 | 62% | `seed42_progression/ep100.pt` |
| 150 | 70% | `seed42_progression/ep150.pt` |
| 200 | 78% | `seed42_progression/ep200.pt` |
| 250 | 84% | `seed42_progression/ep250.pt` |
| 325 | 90% | `seed42_progression/ep325.pt` |

These show the V-shape recovery: Stage 1.0 warm-start gives initial ~42%, dips during early Stage 4.0 retraining as the policy adjusts to random sat init, then climbs steadily to the deliverable 90%+.

## Loading

```python
import torch
from pufferlib.ocean.orbital.orbital import Orbital
from pufferlib.models import Default, LSTMWrapper

env = Orbital(num_envs=1, num_debris_min=0, num_debris_max=0,
              e_max_target=0.20, e_max_sat=0.20, init_phase_gap_max=3.14159,
              valid_init_only=1)
policy = LSTMWrapper(env, Default(env))
policy.load_state_dict(torch.load("models/phase5e/seed42_stage4_best.pt",
                                   map_location="cpu", weights_only=True))
policy.eval()
```

## Reproduction

```bash
# Train from scratch (5 seeds, ~12 min each):
bash scripts/orbital/p5e_curriculum.sh all

# Or eval an existing ckpt:
cd pufferlib && python3 scripts/orbital/eval_checkpoint.py \
    ../models/phase5e/seed42_stage4_best.pt \
    --episodes 200 --e-max-target 0.20 --e-max-sat 0.20 \
    --init-phase-gap-max 3.14159 --valid-init-only 1
```
