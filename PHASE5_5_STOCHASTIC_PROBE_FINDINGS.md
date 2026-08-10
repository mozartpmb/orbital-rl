# Phase 5.5 — Stochastic Eval Probe Findings

> **Status:** 2026-05-21. Probe complete per `phase5-5-stochastic-eval-probe.md`. ~40 min compute (LEO smoke + 42-cell matrix). The probe's hypothesis — that training-time 99.7% perf would reproduce under post-training stochastic eval — is **falsified**. Stochastic ≈ greedy at every cell. The Stage 5.5.1 training-time 99.7% was an artifact. The curriculum design from `phase5-5-altitude-expansion-spec.md` §4.3 holds. A secondary, unanticipated finding: scaling `obs_alt_scale_m` / `lvlh_scale_m` to the eval altitude band catastrophically degrades the policy compared to leaving them at LEO defaults — surprising and informative.

---

## TL;DR

| Cell | Conditions | Greedy mean (n=3 seeds × 200 eps) | Stochastic mean (n=3 × 200) | Δ stoch − greedy |
|---|---|---:|---:|---:|
| Control (LEO) | a∈[6.671, 7.171] Mm, e≤0.05 | **98.3%** | **99.5%** | +1.2 |
| Stage 5.5.1 | a∈[6.671, 8.5] Mm, e≤0.05 | 30.0% | 34.2% | +4.2 |
| Stage 5.5.2 (scaled obs) | a∈[6.671, 12] Mm, e≤0.10, scales=1.2e7 | 0.2% | 0.2% | 0 |
| Stage 5.5.3A (LEO scales) | a∈[6.671, 26] Mm, e≤0.30, scales=LEO | 22.3% | 22.3% | 0 |
| Stage 5.5.3B (scaled obs) | a∈[6.671, 26] Mm, e≤0.30, scales=2.6e7 | 0.0% | 0.0% | 0 |
| Stage 5.5.4A (LEO scales) | a∈[6.671, 42.5] Mm, e≤0.50, scales=LEO | **49.7%** | **49.7%** | 0 |
| Stage 5.5.4B (scaled obs) | a∈[6.671, 42.5] Mm, e≤0.50, scales=4.2e7 | 0.0% | 0.0% | 0 |

(per-seed table in §3.)

**Outcome: C** per spec §4.3 — Stage 5.5.1's training-time 99.7% was an artifact. Stochastic-vs-greedy gap is universally < 5pp.

**Secondary outcome** (not anticipated by spec): the Option A "hold obs scales at LEO defaults" massively outperforms Option B "scale to altitude band" at MEO/GEO. 5.5.3A = 22% vs 5.5.3B = 0%; 5.5.4A = 50% vs 5.5.4B = 0%. The recipe is sensitive to its training obs distribution; rescaling for "cleaner observations" at OOD altitudes breaks the policy.

---

## 1. Infrastructure

`--stochastic` flag added to `pufferlib/scripts/orbital/eval_checkpoint.py`:
- Samples actions from `softmax(logits)` instead of `argmax`.
- Seeds `torch.manual_seed(seed)` for reproducibility.
- ~12 lines including the CLI option.

LEO smoke check passed (spec §2.3): Phase 5b seed-31415 ckpt at LEO defaults, stochastic, 200 eps × seed=42 = **99.0%** (vs published greedy 96.4-98.3%). Implementation correct.

## 2. The probe matrix

Seven cells (5 curriculum stages + 2 obs-scaling variants) × 2 modes × 3 rollout seeds × 200 eps = 8400 episodes total. Ckpt: canonical Phase 5b seed-31415 (`puffer_orbital_177750405236/model_puffer_orbital_000350.pt`). All cells use `valid_init_only=1`, `init_phase_gap_max=π`, `legacy_action_space=10`.

## 3. Per-seed results

| Cell | Mode | seed 42 | seed 1337 | seed 31415 | mean |
|---|---|---:|---:|---:|---:|
| Control | greedy | 98.0% | 98.5% | 98.5% | **98.3%** |
| Control | stochastic | 99.0% | 100.0% | 99.5% | **99.5%** |
| 5.5.1 | greedy | 27.5% | 32.0% | 30.5% | **30.0%** |
| 5.5.1 | stochastic | 33.5% | 34.5% | 34.5% | **34.2%** |
| 5.5.2 | greedy | 0.0% | 0.5% | 0.0% | 0.2% |
| 5.5.2 | stochastic | 0.5% | 0.0% | 0.0% | 0.2% |
| 5.5.3A | greedy | 23.0% | 22.0% | 22.0% | **22.3%** |
| 5.5.3A | stochastic | 23.0% | 22.0% | 22.0% | **22.3%** |
| 5.5.3B | greedy | 0.0% | 0.0% | 0.0% | 0.0% |
| 5.5.3B | stochastic | 0.0% | 0.0% | 0.0% | 0.0% |
| 5.5.4A | greedy | 54.5% | 48.5% | 46.0% | **49.7%** |
| 5.5.4A | stochastic | 54.5% | 48.5% | 46.0% | **49.7%** |
| 5.5.4B | greedy | 0.0% | 0.0% | 0.0% | 0.0% |
| 5.5.4B | stochastic | 0.0% | 0.0% | 0.0% | 0.0% |

Per-seed variance is small (≤6pp) at all cells. No bimodality across rollout seeds.

Raw CSV: `/tmp/p5_5_stoch_probe_results.csv`.

## 4. The 99.7% mystery, resolved

Stage 5.5.1 training-time at epoch 3 reported `perf=0.997`. Stable post-training stochastic eval at the same conditions gives **34.2%**. The gap is real and large. Plausible explanations:

1. **`perf` is not held-out success rate.** Training-time `perf` may average across all timesteps with a different denominator (e.g., per-step shaping reward normalized to [0, 1], or a rolling window biased by short successful episodes). Even though the C env exposes the success flag, the PPO trainer's logger may be reporting something else.
2. **Training-time stochastic policy at epoch 3 differs from the saved ckpt at epoch 5.** The optimizer has already moved past the warm-start ckpt; PPO's clip + entropy regularization at epoch 3 might be momentarily producing different-distribution actions than what gets saved.
3. **Episode-length bias in the early-collapse window.** If `perf` is `mean(reward > 0)` over a fixed step budget, and ckpt-5's policy is just starting to take longer-to-fail episodes, the denominator shifts.

The 99.7% does NOT reflect a reproducible capability gap, and shouldn't be cited as evidence of any kind of stochastic-policy advantage.

This is the **eighth instance** of the metric-vs-implementation pattern flagged in `PHASE5_5_PRE_FINDINGS.md` §6.3 and the Phase 5 retrospective. Lesson reaffirmed: any named quantity must be verified against its implementation. `perf` ≠ greedy success rate ≠ stochastic success rate.

## 5. Secondary finding: eval-time obs scaling destroys policy at MEO/GEO

The spec §3.2 anticipated obs-saturation could be an issue at MEO/GEO and queued both Option A (LEO scales — saturate) and Option B (scale to altitude band — match obs distribution) for 5.5.3 and 5.5.4. The expectation was: B should be better because observations are in-range.

Measured: **A massively outperforms B**.

- 5.5.3A (LEO scales, saturated obs): **22.3% mean**.
- 5.5.3B (band-scaled obs): **0.0% mean**.
- 5.5.4A (LEO scales): **49.7% mean**.
- 5.5.4B (band-scaled obs): **0.0% mean**.

Interpretation: the Phase 5b policy was trained against `obs_alt_scale_m=1.6e6` and `lvlh_scale_m=6.371e6`. Even though those produce saturated/clipped observations at MEO/GEO, the policy has learned to operate against *that specific distribution*. Rescaling at eval time means the policy sees the same physical configuration with completely different obs values, which is a hard OOD shift — strictly worse than living with saturation.

**Implication for Phase 5.5 curriculum:** training stages at MEO/GEO need to either (a) keep LEO obs scales (accept saturation but obs distribution matches policy initialization) or (b) gradually shift obs scales during training so the policy retrains its obs interpretation. Option (a) is simpler; option (b) is what the env-mods spec §1.1 implicitly assumed but never validated.

This finding also explains a puzzle from the pre-experiments: P2 zero-shot at MEO/GEO got 0% with BOTH `obs_alt_scale_m=1.6e6` (LEO) AND `obs_alt_scale_m=4.2e7` (GEO). At the time we said "no shortcut, training needed at altitude." But the LEO-scale-and-saturate result here shows non-trivial capability at MEO/GEO that P2 missed because P2 likely had bugs in its altitude bounds plumbing or different conditions. Worth re-examining the P2 cells with this hindsight.

Specifically the **5.5.4A 49.7%** is a real surprise. With `a∈[6.671, 42.5] Mm` and `e≤0.50` (sampled uniformly + filtered by `valid_init_only=1`), the policy achieves ~50% success WITHOUT any training. Most successful episodes are likely landing in the LEO-portion of the random altitude range (~14% of uniform random a's fall in the original LEO band [6.671, 7.171] Mm), but that alone explains ~14pp; the other ~36pp of successes must be at higher altitudes with LEO-scale-and-saturate observations.

This means the Phase 5b recipe has more latent capability across altitudes than P2's zero-shot suggested — but only with LEO obs scales. **Worth a follow-up sub-probe: at fixed eccentricity, characterize the recipe's success rate vs altitude band, holding obs scales at LEO defaults.**

## 6. Outcome per spec §4

**Outcome C: Stage 5.5.1's 99.7% was an artifact.** Stochastic eval doesn't reveal hidden capability; the greedy-vs-stochastic gap is < 5pp at every cell. The Phase 5.5 curriculum design from `phase5-5-altitude-expansion-spec.md` §4.3 (Option B, 4 stages of altitude expansion) holds in its structure, but with two adjustments:

1. **Drop the implicit assumption that training-time `perf` ≈ greedy success rate.** Going forward, only report greedy/stochastic numbers from `eval_checkpoint.py` with explicit episode counts and CIs. Training-time `perf` is a diagnostic, not a deliverable metric.

2. **Curriculum stages should keep `obs_alt_scale_m` / `lvlh_scale_m` at LEO defaults during training, OR include explicit obs-rescaling schedules.** Naively scaling to altitude band at eval time (Option B in this probe) destroys performance. The env-mods spec §1.1 framing was wrong in this respect; the M1 LVLH scaling is necessary to *avoid Box-bound clipping*, but its application during training needs care.

## 7. Implication for Phase 5.5.x curriculum

The curriculum's premise — that each altitude tier needs its own training stage — is unchanged. But the recipe baselines for each stage are now clearer:

| Stage | Pre-training capability (greedy mean, n=600) |
|---|---:|
| 5.5.1 (a≤8.5 Mm, e≤0.05) | 30% — clear training target |
| 5.5.2 (a≤12 Mm, e≤0.10, scaled obs) | 0.2% — needs curriculum bootstrap |
| 5.5.2 (a≤12 Mm, e≤0.10, LEO obs) | not measured yet; suggested follow-up |
| 5.5.3 (a≤26 Mm, e≤0.30, LEO obs) | 22% — partial baseline |
| 5.5.4 (a≤42.5 Mm, e≤0.50, LEO obs) | 50% — large fraction of envelope already handled |

The 5.5.4A 50% number is unexpectedly high and complicates the curriculum's framing. The deliverable might not be "trains the recipe through 4 stages" but "the recipe already handles 50% of the full envelope and we need to lift specific weak cells." A clean follow-up: break the 5.5.4A 50% down by per-condition surface (similar to Phase 5 `web_data/results/phase5_capability_surface_agg.csv`) to identify *which conditions fail* and design training to address those, not the full curriculum.

## 8. Remaining follow-ups (per spec §5)

§5.1 doesn't apply (curriculum doesn't compress — outcome was C, not A/B).

§5.2 (fine-tune protocol investigation) **does apply**, and the Stage 5.5.1 smoke already gave us free data: aggressive Stage-4.0 hyperparams catastrophically collapse a converged warm-start ckpt. The follow-up is the gentle (F-B) and intermediate-ckpt (F-C) protocols. Worth scheduling as a separate spec/run.

§5.3 doesn't apply (outcome wasn't D — Phase 5b-e numbers don't need re-measurement under stochastic).

**New follow-up not in spec §5:** characterize the 5.5.4A 50% across the per-condition surface (altitude × eccentricity × phase × relation). Identifies the gaps to attack, may reframe the curriculum to a "targeted training" model rather than a "tier-by-tier" model.

## 9. Files touched

| Path | Purpose |
|---|---|
| `pufferlib/scripts/orbital/eval_checkpoint.py` | Added `--stochastic` flag (samples from softmax), seeded `torch.manual_seed(seed)` for reproducibility |
| `PHASE5_5_STOCHASTIC_PROBE_FINDINGS.md` | This document |
| `/tmp/p5_5_stoch_probe.sh` | The probe runner (transient) |
| `/tmp/p5_5_stoch_probe_results.csv` | Raw per-cell-per-seed-per-mode results (transient — copy if needed) |
| `/tmp/p5_5_stoch_probe/<cell>_<mode>_s<seed>/*.npz` | 8400 trajectory dumps (transient) |

---

*Author: 2026-05-21. Phase 5.5 stochastic eval probe. ~40 min compute, 8400 episodes. Outcome C — the 99.7% training-time perf was an artifact, stochastic ≈ greedy. Curriculum design holds in structure. Secondary finding: eval-time obs rescaling crushes the policy at MEO/GEO; 5.5.4A reaches a surprising 50% with LEO obs scales, deserving a per-condition follow-up.*
