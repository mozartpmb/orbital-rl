# MODELS.md — Canonical Checkpoint Registry

Logical-name → on-disk-path → published-result mapping for the orbital RL project. Checkpoint files are not committed to git (they're large `.pt` binaries) **except the five Phase 5e canonical weights in `models/phase5e/`**; this file is the durable record so future work can find them and reproduce results.

Last updated: 2026-08-10 (corrections pass — see §"2026-08-10 corrections" below).

> **⚠️ 2026-08-10 corrections — read before quoting any number from this file:**
>
> 1. **The raw-data-backed headline is 93.7% mean, 88.0–97.5% range** (5 Phase 5e
>    seeds × 200 eps, deterministic, LEO 300–800 km, both e ~ U(0, 0.05), phase gap
>    ±π; `web_data/results/multiseed_escan.csv` e_max=0.05 row, reproduced bit-exact
>    post-classifier-fix in `web_data/results/successbox_scan.csv`). The Phase 5b
>    "96.4% multi-seed" below has **no raw eval output anywhere in the repo** (the
>    collapsed seeds' runs were never preserved) — treat it as reported, not verified.
> 2. **The "2 of 5 seeds collapse" bimodality belongs to the retired Phase 5b
>    recipe.** The superseding Phase 5e `valid_init_only=1` recipe produced 5/5
>    working seeds (88.0–97.5%). Do not pair the collapse stat with current numbers.
> 3. **The Phase 5e seed-42 row below was wrong** (dir/epoch cross-wired). MD5-settled:
>    seed 42 = `puffer_orbital_177765503091/model_puffer_orbital_000325.pt`
>    (= `models/phase5e/seed42_stage4_best.pt`); `177765655537` is seedA at epoch 375.
>    `models/README.md` was correct; the table below is corrected in place.
> 4. **Terminal criterion context for every number here:** success = 30 km position
>    AND 50 m/s relative velocity vs the propagated target. The 2026-08-10 success-box
>    scan (`successbox_scan.csv`) shows the policy zero-shots **2.0%** at 5 km / 1 m/s
>    and **0.0%** at 1 km / 0.5 m/s — the deliverable is a far-field/terminal-approach
>    policy, not a capture policy. Its 10-action head cannot reach the ±1/±2 m/s fine
>    burns (M3) that velocity-nulling below its 5 m/s burn quantum would need.
> 5. **All "e_max ≥ 0.10" rows measure the same task** (LEO geometry caps realized
>    e at ≈ 0.084; realized mean ≈ 0.028) and pre-env-fix rows at e_max ≥ 0.5 are
>    additionally contaminated by 256-cap doomed inits. See
>    `PHASE5_PRE_CLOSURE_MECHANISM_FINDINGS.md` for the Φ-clamp leak retraction
>    (any MEO/GEO `fully_random` "capability" published before 2026-08-10 is an
>    eval artifact; corrected surface in `p5_5_probe1_decompose_v2.csv`).

---

## Phase 5b deliverable (the working LEO low-e specialist)

Phase 5b's "two-stage curriculum" (Stage 1.0 → Stage 4.0 directly) is the shippable Phase 5 deliverable. At LEO 300-800 km / e ≤ 0.05 fully random / init_phase_gap_max = π, the recipe achieves **96.4% multi-seed mean** (3 of 5 training seeds succeed at 94-98%; 2 of 5 collapse to 2-16% — the recipe is bimodal at the edge).

### Stage 4.0 canonical ckpts (5-seed retrain)

| Logical name | Training seed | Eval seed-42 success | Ckpt path |
|---|---|---|---|
| **phase5b_canonical_31415** (headline) | 31415 | 97.7% | `pufferlib/experiments/puffer_orbital_177750405236/model_puffer_orbital_000350.pt` |
| phase5b_canonical_42 | 42 | 95.8% | `pufferlib/experiments/puffer_orbital_177750198246/model_puffer_orbital_000350.pt` |
| phase5b_canonical_20260423 | 20260423 | 95.7% | `pufferlib/experiments/puffer_orbital_177750301624/model_puffer_orbital_000175.pt` |
| phase5b_collapsed_seed_a | (1337 or 2718) | 2-16% | training runs not preserved at specific paths |
| phase5b_collapsed_seed_b | (1337 or 2718) | 2-16% | training runs not preserved at specific paths |

**V3 reproduction (2026-05-12)**: the seed-31415 ckpt re-evaluates at **98.0%** in the env-fix-landed code (commit `4b41cdc`). This serves as the backward-compat anchor for any subsequent env changes.

### Stage 4.1 ckpts (e_max=0.10, 65% multi-seed; partial extension that didn't reach Stage 4.2)

| Logical name | Training seed | Eval success | Ckpt path |
|---|---|---|---|
| phase5b_stage_4_1_seed_42 | 42 | ~65% | `pufferlib/experiments/puffer_orbital_177750529227/model_puffer_orbital_000300.pt` |

---

## Phase 5e canonical ckpts (`valid_init_only=1` retrain, contaminated headlines)

These were trained with `valid_init_only=1` to filter doomed inits but evaluated against headline numbers that the verification investigation later found to be contaminated by the 256-attempt-cap exhaustion bug (now fixed via env-fix F1). Pass-only success at high-e LEO is much higher than the published headlines (e.g., 97.1% at e=0.70 LEO vs published 71.7%).

| Logical name | Training seed | Headline (contaminated) | Pass-only (post env-fix) | Ckpt path |
|---|---|---|---|---|
| **phase5e_canonical_42** | 42 | 92.0% @ e_max=0.20 (5-seed mean is 90.2%; skewed dist — realized e ≈ 0.028) | V1 measured 90.5% @ e=0.70 LEO with cap=4096 | `pufferlib/experiments/puffer_orbital_177765503091/model_puffer_orbital_000325.pt` (= `models/phase5e/seed42_stage4_best.pt`, MD5-verified) |
| phase5e_seed_unknown_1 | ? | — | — | `pufferlib/experiments/puffer_orbital_177765658166/model_puffer_orbital_*.pt` |
| phase5e_seed_unknown_2 | ? | — | — | `pufferlib/experiments/puffer_orbital_177765658729/model_puffer_orbital_*.pt` |
| phase5e_seed_unknown_3 | ? | — | — | `pufferlib/experiments/puffer_orbital_177765659007/model_puffer_orbital_*.pt` |
| phase5e_canonical_wandb | 42 | (alt mirror) | — | `models/phase5e/canonical_seed42_stage4_best.pt` (if present) |

**Note on lost seed→dir mapping**: 4 of the 5 Phase 5e seed dirs lost their explicit seed identity due to a race condition in the parallel orchestrator at training time. Results from these dirs are valid in aggregate but cannot be matched back to specific seeds.

---

## Web data trajectory archives

Trajectories from canonical ckpt evaluations, exported via `scripts/orbital/export_web_data.py`:

| Path | Content | Status |
|---|---|---|
| `web_data/runs/phase5e_seed42_e0.05/` | Phase 5e seed 42 @ e=0.05 LEO | Clean |
| `web_data/runs/phase5e_seed42_e0.20/` | Phase 5e seed 42 @ e=0.20 LEO | Clean |
| `web_data/runs/phase5e_seed42_e0.50/` | Phase 5e seed 42 @ e=0.50 LEO | **5 of 49 contaminated** (pre-F1 cap exhaust) |
| `web_data/runs/phase5e_seed42_e0.70/` | Phase 5e seed 42 @ e=0.70 LEO | **12 of 46 contaminated** (pre-F1 cap exhaust) |
| `web_data/runs/phase5e_progression/` | Phase 5e training progression V-curve | Clean |
| `web_data/runs/phase5_env_fix_v4_contamination_flags.json` | Sidecar flagging the 17 contaminated trajectories | — |

---

## Useful eval/diagnostic scripts

| Script | Purpose |
|---|---|
| `pufferlib/scripts/orbital/eval_checkpoint.py` | Standard eval harness. Full env-kwarg support post env-fix. |
| `scripts/orbital/p5verify_perigee_scan.py` | Audits perigees in trajectory JSON files. |
| `scripts/orbital/p5wrap_surface_full.py` | Multi-cell capability surface eval (uses subprocess `eval_checkpoint.py`). |
| `scripts/orbital/p5wrap_surface_aggregate.py` | Aggregates per-cell surface results. |
| `scripts/orbital/p5e_e1_lambert.py` | Lambert reachability check (Block I E1). |
| `scripts/orbital/p5e_e2_kepler_precision.py` | Kepler propagation precision (Block I E2). |
| `scripts/orbital/p5e_e3_round_trip.py` | Cartesian↔elements round-trip (Block I E3). |
| `scripts/orbital/p5e_e5_phi_calib.py` | Φ_orbit calibration (Block I E5). |
| `scripts/orbital/p5e_e6_action_effect.py` | Action-effect peri-vs-apo asymmetry (Block I E6). |
| `scripts/orbital/p5b_step1_cap_tail_analysis.py` | Cap-tail timeout-failure analysis. |
| `scripts/orbital/p5b_step1_phi_traj.py` | Per-step Φ trajectory plot. |

---

## Post-`phase5-5-env-mods` requirement: `--legacy-action-space 10`

After commit landing the `phase5-5-env-mods` tag (M1 LVLH scaling + M2 longer warps + M3 sub-5 m/s actions), the env's action space is `Discrete(16)` but the Phase 5b/5e canonical ckpts all have a 10-dim policy logits head. **To evaluate any pre-env-mods ckpt, you must pass `--legacy-action-space 10` to `eval_checkpoint.py`** — without it, `torch.load_state_dict` will raise a `size mismatch for policy.decoder.weight: copying a param with shape torch.Size([10, 128]) from checkpoint, the shape in current model is torch.Size([16, 128]).`

This flag coerces `env.single_action_space = Discrete(10)` before policy construction, so the policy head sizes to 10. The env's `c_step` still accepts integers in `[0, NUM_ACTIONS)`; a 10-dim argmax produces ints 0-9, which are the legacy actions.

V5 multi-seed reproduction post-env-mods (2026-05-15):
- seed 31415: 98.0%
- seed 42: 95.0%
- seed 20260423: 96.5%
- **Mean: 96.5%** (published Phase 5b: 96.4%)

The published Phase 5b numbers reproduce. Backward compat is preserved.

## Conventions

- **Experiment directory naming**: `puffer_orbital_{wandb_run_id}/` or `puffer_orbital_{int(100*time.time())}/` if no wandb. The run ID is opaque; resolve to seed via wandb config or curriculum logs.
- **Ckpt naming**: `model_puffer_orbital_{epoch:06d}.pt`. Checkpoints save every `--train.checkpoint-interval N` epochs (default 200; curriculum uses 5).
- **Best-ckpt selection**: `scan_best_ckpt()` in `scripts/orbital/p5b_curriculum.sh:70-92` samples every 25 epochs + final, picks the highest `eval_checkpoint.py` success rate.

---

## How to add an entry

When new canonical ckpts are produced (e.g., Phase 5.5 stages), add a section here with: training seed, ckpt path, eval conditions, headline number(s), and any caveats (collapsed seed, contamination, etc.). Keep this file the durable source of truth for "which ckpt represents what."
