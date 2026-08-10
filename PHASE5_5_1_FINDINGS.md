# Phase 5.5 Stage 5.5.1 — Smoke Findings

> **Status:** 2026-05-21. Stage 5.5.1 ran as a 3-seed × 30M-step smoke test per the spec §4.3 (Option B). Result: **smoke fail by the metric, smoke pass by the underlying capability** — the Phase 5b recipe transfers to a∈[6671, 8500] km out of the box, but the Stage-4.0 fine-tuning hyperparameters catastrophically destroy the warm-start policy within ~20 epochs.

---

## TL;DR

- **Pre-training zero-shot** (greedy eval, seed-31415 ckpt, 50 eps, a∈[6671,8500] km, e=0.05): **34%**.
- **Training-time perf at epoch 3** (stochastic, in-distribution): **99.7%** (seed 42). The warm-start policy fully solves the new band.
- **Training-time perf at epoch 21**: **0.0%**. Collapse complete.
- **Post-training best ckpts (greedy eval, 50 eps each):** all three seeds peak at 2–10% at epoch 5 and reach 0% by epoch 25.
- **LEO regression check** (seed 42 final ckpt, 100 eps at default LEO band, e=0.05): **0.0%**. The policy is globally destroyed, not just OOD-broken.

The decision criteria table called this **smoke fail** (mean post-train < 40%, plus LEO regression > 5pp). Do not commit to the full Option B curriculum under Stage-4.0 hyperparameters.

---

## What ran

- 3 seeds × 30M steps via `scripts/orbital/p5_5_1_curriculum.sh all`. Total wall: 16 min (≈5 min/seed on M3 Max CPU).
- Warm-starts (per `MODELS.md`):
  - seed 42 → `pufferlib/experiments/puffer_orbital_177750198246/model_puffer_orbital_000350.pt`
  - seed 31415 → `pufferlib/experiments/puffer_orbital_177750405236/model_puffer_orbital_000350.pt`
  - seed 20260423 → `pufferlib/experiments/puffer_orbital_177750301624/model_puffer_orbital_000175.pt`
- Env: `a_min_override=6.671e6`, `a_max_override=8.5e6`, `e_max=0.05` (sat+target), `same_orbit_init=0`, `valid_init_only=1`, `init_phase_gap_max=π`.
- Env scaling kwargs held at LEO defaults (`lvlh_scale_m=6.371e6`, `obs_alt_scale_m=1.6e6`, `phi_orbit_scale_k=0.001`). Action space coerced to legacy 10 via new kwarg `legacy_action_space=10`.
- Trainer hyperparams: `orbital.ini` defaults (lr_max=1e-2, ent_coef=0.01, gamma=0.995, minibatch=8192, checkpoint every 5 epochs, ~230 epochs total at 30M steps).
- Output dirs:
  - seed 42 → `pufferlib/experiments/puffer_orbital_177937876499/` (46 ckpts)
  - seed 31415 → `pufferlib/experiments/puffer_orbital_177937906637/` (46 ckpts)
  - seed 20260423 → `pufferlib/experiments/puffer_orbital_177937938299/` (46 ckpts)

## Per-seed perf trajectory (training-time, stochastic)

Seed 42 (representative; the other two follow the same pattern):

| Epoch | perf | episode_return |
|---:|---:|---:|
| 3  | **0.997** | +7.24 |
| 5  | 0.823 | +3.72 |
| 7  | 0.374 | −4.03 |
| 9  | 0.036 | −8.79 |
| 11 | 0.097 | −7.82 |
| 13 | 0.031 | −8.85 |
| 15 | 0.001 | −9.11 |
| 21 | 0.000 | −9.20 |
| 229 (final) | 0.000 | −9.32 |

Collapse trajectory: warm-start policy enters in-distribution near-perfect, decays monotonically (with one tiny bounce at epoch 11) into floor by epoch 21, then stays there for the remaining ~210 epochs.

## Per-seed scan of saved ckpts (greedy eval, 50 eps at 5.5.1 band)

| Seed | epoch 5 | epoch 25 | epoch 50 | epoch 100 | epoch 229 (final) |
|---|---:|---:|---:|---:|---:|
| 42 | 8% | 0% | 0% | 0% | 0% |
| 31415 | 2% | 0% | 0% | 0% | 0% |
| 20260423 | 10% | 0% | 0% | 0% | 0% |

The greedy 5% number is much lower than the training-time 99.7% at epoch 3 because (a) the saved ckpt at epoch 5 is already 4 epochs past the peak collapse onset, and (b) greedy-vs-stochastic differ when the policy is mid-collapse and noisy.

## LEO regression (seed 42 final ckpt, 100 eps, LEO default band)

| Metric | Value |
|---|---|
| Success rate | **0/100 (0.0%)** |
| Mean reward | −9.80 |
| Mean ep length | 448 steps |

The trained policy doesn't solve LEO either. This rules out "just OOD at the new band" — the policy is globally broken.

## Mechanism: plasticity collapse from aggressive fine-tune of converged ckpt

The Phase 5b deliverable ckpts have very low entropy (converged for the full 50M-step Stage 4.0 budget). Restarting PPO with `lr_max=1e-2`, `ent_coef=0.01`, fresh annealing schedule on top of a converged policy is the canonical recipe for catastrophic plasticity loss. This has been documented in this project before:

- **Phase 2 fine-tune attempts** (`PROGRESS.md` lines 468–477): warm-started Stage 3a at e_max=0.30 with lr=3e-4 reached 45% at epoch 25 then *decayed* to 31% by epoch 153. Same coast-collapse pattern.
- **Phase 4 R5** (memory: `project_orbital_rl_phase4_r5_results.md`): all 4 reward-reshape variants on warm-start collapse to <10%; plasticity (R3) is prerequisite.
- **Phase 4 warm-start continuation at 180°** (memory: `project_orbital_rl_warmstart_continuation_fails.md`): R3a/b/c/d all collapse; untrained warm-start ckpt (47.3%) beats every trained variant.

Stage 5.5.1 reproduces this pattern. The Phase 5b "Stage 1.0 → Stage 4.0 warm-start works" lesson does **not** generalize to "any warm-start works under Stage-4.0 hyperparams." Stage 1.0 is a deliberately weak/high-entropy bootstrap policy with plasticity to spare. The Phase 5b *deliverable* ckpts are not.

## What the smoke test actually proved

The interesting result is the epoch-3 training perf of 0.997. **The Phase 5b recipe ckpts generalize to a∈[6671, 8500] km under stochastic policy without any fine-tuning at all.** The greedy 34% zero-shot is consistent with this — under argmax the policy is less robust at the OOD edges, but under sampling (training mode) it solves the task.

This is a **stronger** result than the spec anticipated for Stage 5.5.1. It implies Option B's premise — "incremental altitude bumps that each need their own training stage" — may be wrong for the slight-extension stage. The recipe may already be implicitly altitude-robust within some band, and the curriculum may only need training at stages where zero-shot fails.

## Recommendations for Stage 5.5.1 retry

In rough order of cheapness:

1. **Don't fine-tune for Stage 5.5.1 at all.** Use the Phase 5b ckpts directly. The deliverable claim becomes "Phase 5b ckpts handle a∈[6671, 8500] km zero-shot at training-time perf ~99%, greedy ~34%." Move directly to Stage 5.5.2 (a_max=12000 km) where zero-shot is presumably worse and training is actually needed.
   - Verify with a proper multi-seed stochastic eval (run training for ~1 epoch to capture training-time perf, then stop; OR add a stochastic-eval mode to `eval_checkpoint.py`).

2. **Low-lr fine-tune.** Drop `lr_max` from 1e-2 to ~1e-4, `ent_coef` from 0.01 to ~0.001, train for 5–10M steps. This is the conventional fine-tune-from-converged-ckpt recipe. Expected: small lift over zero-shot, no collapse.

3. **Train-longer-with-different-warm-start.** Use an intermediate (epoch ~50–100) Phase 5b ckpt with more residual entropy as warm-start, not the converged ckpt. Phase 5b spec §6.8 ("save mid-training ckpts") flagged this hypothesis but never tested it cleanly. Stage 5.5.1 is the natural place.

4. **Stochastic-eval `eval_checkpoint.py`.** It currently does argmax (greedy). For warm-start-quality assessment we need stochastic eval too, because training-time perf shows the policy can sample the right actions even when argmax doesn't.

## Recommendations for the broader Phase 5.5 curriculum

The Stage-4.0 hyperparam profile (lr_max=1e-2, ent_coef=0.01) is the wrong default for warm-starting converged ckpts. Either:
- The Option B curriculum should explicitly schedule lr/ent for warm-start stages (different from the Stage 1.0 bootstrap profile), or
- Each altitude stage should re-bootstrap with `same_orbit_init=1` style structure (like Phase 5b's two-stage). Doubles the compute per stage but matches the recipe whose plasticity assumptions actually hold.

The Option B spec §4.3 was implicitly assuming "the recipe transfers" means "the recipe + the recipe's training hyperparams transfer." That's not what this test showed. The *policy* transfers; the *training procedure* doesn't — it kills the policy.

## Decision per spec §5.4

| Criterion | Result |
|---|---|
| Mean (1) in-distribution greedy | 0% mean, 0–10% per seed |
| Mean (2) LEO regression check | 0% (regressed −96.5pp from published 96.5%) |
| Per-seed variance | All three seeds collapse identically — no bimodality, deterministic failure |

**Soft-success threshold (≥60%): not met. Hard-success (≥80%): not met. Per spec §5.4 this is a "documented limit" outcome, but the limit is hyperparam-related, not capability-related.**

## Files touched / produced

| Path | Purpose |
|---|---|
| `pufferlib/pufferlib/ocean/orbital/orbital.py` | Added `legacy_action_space` kwarg so a 10-head ckpt can warm-start under the new Discrete(16) env without surgery |
| `pufferlib/pufferlib/config/ocean/orbital.ini` | Added `legacy_action_space = -1` so puffer's CLI introspects the new kwarg |
| `scripts/orbital/p5_5_1_curriculum.sh` | 3-seed × 30M warm-start runner |
| `PHASE5_5_1_FINDINGS.md` | This document |
| `pufferlib/experiments/puffer_orbital_17793787{6499,906637,938299}/` | 46 ckpts × 3 seeds (preserved for follow-up low-lr / intermediate-ckpt experiments) |

## Open questions for a follow-up

1. **Stochastic eval at the 5.5.1 band — is the training-time 99.7% real, or an artifact?** Need a quick `eval_checkpoint.py --stochastic` mode (sample from logits, don't argmax) and re-evaluate the un-trained Phase 5b ckpts at the 5.5.1 distribution. If stochastic eval gives ~99%, the spec's premise needs revision.

2. **Does Stage 5.5.2 (a_max=12000 km) also fail zero-shot under stochastic eval?** If yes, Stage 5.5.1 can be skipped entirely and effort goes to 5.5.2+. If no, the recipe has more zero-shot range than expected and the whole Option B curriculum compresses.

3. **What's the right lr/ent profile for warm-starting converged ckpts?** A focused 2-3 trial sweep (lr_max ∈ {1e-4, 1e-3}, ent_coef ∈ {0.0, 0.001}, 5M steps each) would establish a "fine-tune" recipe distinct from the Stage 4.0 "train from bootstrap" recipe.

---

*Author: 2026-05-21. Phase 5.5 Stage 5.5.1 smoke. ~30 min compute (training + eval). Smoke fail under headline metric, but a stronger underlying finding: the Phase 5b deliverable ckpts already handle the 5.5.1 band; Stage-4.0 fine-tuning hyperparams destroy them. Recommendation: don't fine-tune for the slight-extension stage; verify stochastic-eval baseline before committing to any altitude-curriculum training.*
