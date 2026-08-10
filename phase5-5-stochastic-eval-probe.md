# Phase 5.5 — Stochastic Eval Probe (Pre-curriculum)

> **Status:** 2026-05-21. Stage 5.5.1 smoke (`PHASE5_5_1_FINDINGS.md`) revealed a 65pp gap between training-time stochastic perf (99.7%) and greedy eval (34%) on the same ckpt at the same conditions. This means Phase 5b's published numbers may significantly undercount actual policy capability. Before committing to the multi-stage altitude curriculum, measure the recipe's stochastic-mode capability across the planned curriculum stages. The result may compress the curriculum dramatically. ~2-3 hours total work, mostly infrastructure + eval compute.

---

## 0. Why this probe exists

Stage 5.5.1 produced an unexpected finding: training-time perf at epoch 3 was 99.7% on the warm-started Phase 5b ckpt at the new altitude band (a∈[6671, 8500] km). Greedy eval at the same conditions was 34%. The 65pp gap is the stochastic-vs-argmax difference at OOD edges.

This raises a sharp question: **what does the Phase 5b recipe actually do at Phase 5.5's planned curriculum stages, measured under stochastic policy?**

If the answer is "much better than greedy eval suggested," Phase 5.5's curriculum compresses — possibly to zero stages of training at LEO+, only at MEO/GEO. If the answer is "still drops off sharply between stages," the curriculum is needed and the next question is how to fine-tune without plasticity collapse.

The probe is cheap (~2-3 hours total) and decisive for curriculum design.

---

## 1. What we know vs what we need to measure

### 1.1 Known (greedy eval, from prior work)

- Phase 5b at LEO e=0.05: 96.4% multi-seed
- Phase 5b ckpt zero-shot at MEO/GEO (any e): 0% (P2 from pre-experiments)
- Phase 5b ckpt zero-shot at Stage 5.5.1 (a∈[6671, 8500] km, e=0.05): 34%

### 1.2 Known (stochastic, from Stage 5.5.1 training-time)

- Phase 5b ckpt stochastic perf at Stage 5.5.1: 99.7% (epoch 3 training, n unclear, single training seed)

### 1.3 What's not measured

- Phase 5b ckpt stochastic perf at LEO baseline (control)
- Phase 5b ckpt stochastic perf at Stage 5.5.2 conditions (a∈[6671, 12000] km, e=0.10)
- Phase 5b ckpt stochastic perf at Stage 5.5.3 conditions (a∈[6671, 26000] km, e=0.30)
- Phase 5b ckpt stochastic perf at Stage 5.5.4 conditions (a∈[6671, 42500] km, e=0.50)

The gap matters because the curriculum stages assume zero-shot fails between them. If stochastic eval shows capability extending through multiple stages, the curriculum is over-staged.

---

## 2. Infrastructure: stochastic-eval mode

### 2.1 The change

Add `--stochastic` flag to `eval_checkpoint.py`. When enabled, sample actions from the policy's softmax distribution instead of argmax.

```python
# In the eval loop where the action is selected:
if args.stochastic:
    # Sample from softmax distribution
    probs = torch.softmax(logits, dim=-1)
    action = torch.multinomial(probs, num_samples=1)
else:
    # Argmax (current behavior)
    action = logits.argmax(dim=-1)
```

Default `--stochastic=False` preserves current behavior. Adding this is ~10 lines.

### 2.2 Reporting

Stochastic eval has higher variance per episode. To get a stable measurement, run 200 episodes (same as standard) and report:
- Success rate (point estimate)
- Bootstrap 95% CI (resample with replacement, 1000 iterations)
- Mean / std of reward across episodes

The CI matters because a single 200-episode stochastic run could land at ±5pp easily.

### 2.3 Validation

Smoke check: run `--stochastic` on the canonical Phase 5b ckpt at LEO e=0.05. Expect success rate near the published 96.4% (might be slightly higher due to sampling exploration helping at edges). If it's wildly different (e.g., 60% or 100%), the implementation has a bug.

---

## 3. The probe: Phase 5b ckpt across all 4 stages, stochastic and greedy

### 3.1 What to run

For the canonical Phase 5b ckpt (`puffer_orbital_177750405236/model_puffer_orbital_000350.pt`, seed 31415), eval at the following five conditions:

| Cell | a_min | a_max | e_max | Notes |
|---|---|---|---|---|
| Control (LEO baseline) | 6.671e6 | 7.171e6 | 0.05 | Reproduces 96.4% under greedy; baseline check |
| Stage 5.5.1 | 6.671e6 | 8.5e6 | 0.05 | Already measured (greedy 34%, training-time 99.7%) |
| Stage 5.5.2 | 6.671e6 | 1.2e7 | 0.10 | First MEO extension |
| Stage 5.5.3 | 6.671e6 | 2.6e7 | 0.30 | Full MEO |
| Stage 5.5.4 | 6.671e6 | 4.25e7 | 0.50 | Full LEO+MEO+GEO |

Per cell, run both modes:
- Greedy: 200 episodes × 3 rollout seeds = 600 episodes
- Stochastic: 200 episodes × 3 rollout seeds = 600 episodes

Total: 5 cells × 2 modes × 600 episodes = 6000 episodes. ~30-60 min compute.

### 3.2 Env configuration per cell

Stage 5.5.2+ needs the env-mods kwargs set for the altitude:

| Cell | `obs_alt_scale_m` | `lvlh_scale_m` | `phi_orbit_scale_k` |
|---|---|---|---|
| Control / 5.5.1 | 1.6e6 (default) | 6.371e6 (default) | 0.001 (default) |
| 5.5.2 (MEO low) | 1.2e7 | 1.2e7 | 0.01 |
| 5.5.3 (full MEO) | 2.6e7 | 2.6e7 | 0.02 |
| 5.5.4 (LEO+MEO+GEO) | 4.2e7 | 4.2e7 | 0.05 |

These are reasonable defaults per the pre-experiment E5 findings. Worth flagging: the obs_alt_scale_m and lvlh_scale_m values here are *eval-only* — the Phase 5b ckpt was trained against `obs_alt_scale_m=1.6e6` and `lvlh_scale_m=6.371e6`. Changing them at eval time changes the obs distribution the policy sees.

**This is a real ambiguity:** for cells beyond LEO, what's the right obs scale at eval time?
- Option A: hold scales at LEO defaults (1.6e6, 6.371e6). The policy sees obs in the distribution it was trained on, but obs[33-34] saturate at MEO+ per E4.
- Option B: scale them to the eval altitude band. The policy sees obs in the right range, but the distribution differs from training.

For this probe, **run both options** for Stage 5.5.3 and 5.5.4 (the cells where saturation matters). For 5.5.1 and 5.5.2 (where LEO scales don't saturate too badly), just use Option A.

Adds ~2400 more episodes (Stage 5.5.3 and 5.5.4 with Option B × greedy + stochastic × 600). Total ~8400 episodes, ~1 hour compute.

### 3.3 Decision criteria

Per stage:
- **Stochastic ≥ 80%, greedy ≥ 60%**: zero-shot capability is solid; no training needed at this stage.
- **Stochastic 50-80%, greedy < 50%**: capability is present but brittle; *gentle fine-tuning* (per Stage 5.5.1's Option 2 recommendation) might extend; or use the ckpt as-is and tolerate the greedy gap.
- **Stochastic < 50%**: capability genuinely runs out; training is needed at this stage.

These thresholds are heuristic. The result will inform but not strictly dictate the curriculum design.

---

## 4. What the probe answers

The probe answers one question: **at what altitude band does the Phase 5b ckpt stop having useful stochastic capability?**

Three plausible outcomes:

### 4.1 Outcome A: Recipe extends all the way to Stage 5.5.4

Stochastic eval ≥ 80% at all stages, including 5.5.4. This would mean the recipe is implicitly altitude-invariant under sampling, and the entire Phase 5.5 curriculum compresses to zero training. The deliverable becomes "Phase 5b ckpt + stochastic eval handles the full envelope."

This is the most surprising possible outcome and would significantly reframe Phase 5.5.

### 4.2 Outcome B: Recipe extends through some intermediate stage

Stochastic eval ≥ 80% at 5.5.1, 5.5.2 (or 5.5.3) but drops at higher altitudes. Curriculum is needed only for the higher-altitude stages. The first 1-2 stages can be skipped.

This is the most likely outcome based on prior data. P2 showed 0% greedy at MEO/GEO, so MEO+ might still be far OOD under stochastic.

### 4.3 Outcome C: Stage 5.5.1's 99.7% was an artifact

Stochastic eval at 5.5.1 doesn't reproduce the 99.7%. Possibilities: the epoch-3 training-time perf was sample-luck (small n), or there's a real difference between PPO's training-time policy (which is mid-update) and a stable post-training stochastic eval.

If Outcome C, the Phase 5b ckpt is more LEO-specific than Stage 5.5.1's smoke suggested, and the curriculum stays at 4 stages.

### 4.4 Outcome D: Stage 5.5.1's 99.7% reproduces, but the gap is consistent across all stages

Stochastic eval is significantly higher than greedy at every stage by a similar magnitude (e.g., +30-60pp). This would mean the greedy-vs-stochastic gap is a *structural* property of how the recipe handles OOD conditions, not specific to Stage 5.5.1. The implication: greedy eval has been systematically undercounting capability throughout Phase 5b-e.

If Outcome D, several earlier Phase 5b-e numbers should be re-measured under stochastic for accurate characterization.

---

## 5. Optional follow-ups (depending on outcome)

### 5.1 If Outcome A or B — write a revised curriculum

Phase 5.5's curriculum stages compress to whatever's actually needed. Update `phase5-5-altitude-expansion-spec.md` to reflect the new scope.

### 5.2 If Outcome B or C — investigate fine-tune protocol

The plasticity collapse from Stage 5.5.1 (epoch 3 → epoch 21 from 99.7% → 0%) is a known recurring issue in this project. If the curriculum needs even one stage of training at altitude, the fine-tune protocol matters.

Quick investigation: at the first stage where stochastic capability isn't sufficient, try three fine-tune protocols on a single seed:
- F-A: Aggressive (current Stage-4.0 defaults: lr_max=1e-2, ent_coef=0.01). This is what Stage 5.5.1 used; expect collapse.
- F-B: Gentle (lr_max=1e-4, ent_coef=0.001, ~5M steps). Conventional fine-tune-from-converged.
- F-C: Mid-training warm-start (Phase 5b ckpt at ~half-training, with more residual entropy). Tests the "intermediate ckpt as warm-start" hypothesis from PHASE5b §6.8.

~3 hours compute. Picks the right fine-tune protocol for subsequent curriculum work.

### 5.3 If Outcome D — re-measure Phase 5b-e numbers under stochastic

The published Phase 5b 96.4% and Phase 5e capability numbers may all undercount. Re-measure at a representative subset of conditions under stochastic eval to characterize the gap.

This is sub-scope for Phase 5.5 (not blocking the curriculum), but worth noting in the PHASE5 retrospective as an eighth instance of the metric-vs-implementation pattern.

---

## 6. Sequencing

In order:

1. **Add `--stochastic` flag to `eval_checkpoint.py`** (~30 min engineering).
2. **Smoke validate at LEO baseline** (~5 min compute). Stochastic eval on Phase 5b ckpt at LEO should reproduce 96-98%.
3. **Run the 5-stage probe greedy + stochastic** (~1 hour compute total).
4. **Analyze and produce findings doc** (~30 min).
5. **Decision: which outcome (A/B/C/D), what's next.**

Total ~2-3 hours wall time.

---

## 7. Pre-committed acknowledgments

### 7.1 Surprise A — The stochastic eval at LEO baseline doesn't reproduce 96.4%

Plausible scenarios:
- Stochastic eval introduces enough exploration variance to lift the number (96.4% → ~98%, fine).
- It substantially lowers the number (e.g., 70%). This would mean the `--stochastic` implementation has a bug, OR the published 96.4% under argmax is itself an over-statement of what the policy "reliably does."

If the second: investigate, don't proceed until the LEO baseline reproduces.

### 7.2 Surprise B — Stage 5.5.1 stochastic eval doesn't reproduce 99.7%

The training-time 99.7% was a single training seed at one epoch. A stable stochastic eval (200 episodes × 3 rollout seeds) might give something different. Plausible range: 70-99%, with significant variance.

If the eval gives a wide variance (CI ≥ 10pp), report it honestly. The training-time 99.7% may have been favorable-sample-luck.

### 7.3 Surprise C — Stochastic and greedy give similar numbers at all stages

If stochastic eval is within 5pp of greedy at every stage, the Stage 5.5.1 99.7% is the artifact, not the greedy 34%. The original curriculum design holds.

This is fine. The probe ran cheaply and confirmed the curriculum's premise.

### 7.4 Surprise D — Wildly different patterns across the 3 rollout seeds

The 3 rollout seeds × 200 eps give us bootstrap confidence in the mean. If they're wildly bimodal (e.g., one seed at 95%, two at 30%), report the bimodality — it's information about the recipe's stability under stochastic policy.

This wouldn't necessarily change the conclusion (the mean is what matters for curriculum design), but it's worth documenting.

---

## 8. Output

A single document `PHASE5_5_STOCHASTIC_PROBE_FINDINGS.md` with:
- The 5-stage × 2-mode results table (mean, CI, n_seeds, n_episodes)
- Identified outcome (A/B/C/D from §4)
- Implication for Phase 5.5 curriculum (compress, hold, or rework)
- If applicable: which sub-investigation (§5.1, §5.2, §5.3) follows

Then update `phase5-5-altitude-expansion-spec.md` per the outcome.

---

## 9. What this probe is NOT

- **Not a Phase 5.5 curriculum stage.** No training. Eval only.
- **Not a fine-tune protocol investigation.** That's §5.2 if needed.
- **Not a complete re-measurement of all Phase 5 numbers under stochastic.** Only the cells that affect Phase 5.5's curriculum design.
- **Not a substitute for greedy eval.** Greedy remains the "what would deployment do?" metric. Stochastic is the "what can the policy actually do?" metric. Both are useful, in different contexts.

---

*Author: 2026-05-21. Phase 5.5 stochastic-eval probe. Three to five hours total work. Decisive for Phase 5.5 curriculum design — outcome determines whether the curriculum is 4 stages, 2 stages, or 0 stages of training. Follows Stage 5.5.1 smoke findings.*
