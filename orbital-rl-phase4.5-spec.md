# Orbital RL — Phase 4.5 Spec: Attribution Ablations

> **Status:** 2026-04-25. Bridge between Phase 4 (79.6% multi-seed @ 180° rendezvous, R4 curriculum) and Phase 5 (unified agent across phase gap × eccentricity). Total compute budget: ~3 hours on M3 Max.

---

## 0. TL;DR

Phase 4 shipped the R4 curriculum at 79.6% multi-seed mean @ 180° on 2/3 seeds. The winning recipe contains R1 (LVLH obs), R2 (gated shaping at low weights), R4 (warp action), and curriculum scheduling. The deep dive's claim that "only R4 worked" is overstated — R1 and R2 are baked into the recipe but their *marginal contribution under curriculum* was never measured. Phase 4.5 closes those holes with three cheap ablations before Phase 5 extends the recipe to harder difficulty regimes.

If we don't run Phase 4.5 first, any failure in Phase 5 (e.g., the recipe breaks at high eccentricity) cannot be cleanly attributed — we won't know whether LVLH, shaping, warp, or curriculum is the binding ingredient.

---

## 1. Why this exists (justification for not skipping)

### 1.1 The unresolved attribution

PHASE4_FINDINGS.md frames the result as "only R4 (curriculum + warp) worked; R1, R2, R3 did not." This conflates two distinct findings:

- **True:** R1, R2, and R3 *standalone at 180°* did not lift the baseline meaningfully, and R3 actively regressed it.
- **Unmeasured:** R1 and R2 *under R4 curriculum* may or may not be contributing. They are present in the winning config (`OBS_DIM=38`, `BETA_SHAPE=1.0`, `w_orbit=w_phase=w_vel=0.01`) but never isolated.

Three pieces of evidence motivate the doubt:

1. The untrained Stage 2 LVLH zero-pad checkpoint evaluates at **47.3%** at 180° before any LVLH-specific training — close to R0's 52.7%. That tells us LVLH is not actively harmful, but does not tell us whether it contributes when properly trained.
2. R4 from-scratch at 180° (no curriculum) returned 0.0% in 8.8M steps. The curriculum unlocks reward discovery; warp alone doesn't. But: the curriculum was always run with LVLH and shaping baked in.
3. The Stage 1 seed-1337 failure (0% @ 30° on Discrete(10), vs Phase 3's 99% on Discrete(9)) is a 22pp regression from adding one no-op action. That's a much larger effect than action-space-expansion theory would predict, and it's the most concrete actionable problem in the current recipe.

### 1.2 Why this matters for Phase 5

Phase 5 will extend the curriculum to combined (phase gap × eccentricity) difficulty. If at any point Phase 5 stalls, the diagnostic question "which ingredient broke?" is only answerable if we know what each ingredient was contributing in Phase 4. Otherwise we're back to debugging in the dark, which is exactly the failure mode that ate Phase 3.

### 1.3 Why now and not later

Three runs at ~30 minutes each on M3 Max. Total compute is small relative to a single Phase 5 stage. Doing this before Phase 5 means we enter Phase 5 with a clean attribution table, not after Phase 5 closes when we'd be re-running already-relevant ablations under new conditions.

---

## 2. The three ablations

Each ablation is a single training run on seed 42, using the R4 curriculum (30° → 90° → 180°) with one ingredient stripped or altered. Seed 42 is sufficient for first-pass attribution; if any ablation produces a result more than ±10pp from the baseline, the question is settled and a multi-seed confirm can follow.

### 2.1 Ablation A — R4 curriculum without LVLH

**Hypothesis.** LVLH obs is contributing meaningfully under curriculum, even though it didn't move the standalone R1 number. If LVLH is doing real work, this ablation will land below 79.6% by a measurable margin.

**Implementation.**
- Branch from current `orbital.h`/`orbital.py`.
- Set `OBS_DIM = 33` (revert from 38).
- Strip the LVLH-frame block from `fill_observations` (lines 475–511 in `orbital.h`).
- Verify Python obs space updates: `Box(low=-inf, high=inf, shape=(33,), dtype=np.float32)`.
- No other config changes.

**Run.** Full curriculum, seed 42, no warm-start across this change (obs dim differs from Phase 4 checkpoints, so per Principle D this is from-scratch at Stage 1):

```bash
# Stage 1 (30°, fresh, 10M)
puffer train puffer_orbital --train.seed 42 --train.total-timesteps 10_000_000 \
  --train.device cpu --env.init-phase-gap-max 0.524

# Stage 2 (90°, warm-start from Stage 1, 15M)
puffer train puffer_orbital --train.seed 42 --train.total-timesteps 15_000_000 \
  --train.device cpu --env.init-phase-gap-max 1.5708 \
  --load-model-path experiments/<stage1_runid>/model_puffer_orbital_000077.pt

# Stage 3 (180°, warm-start from Stage 2, 15M)
puffer train puffer_orbital --train.seed 42 --train.total-timesteps 15_000_000 \
  --train.device cpu --env.init-phase-gap-max 3.14159 \
  --load-model-path experiments/<stage2_runid>/model_puffer_orbital_000115.pt
```

**Expected outcomes and interpretation.**

| Stage 3 result | Interpretation |
|---|---|
| ≥ 75% (within seed noise of 81%) | LVLH did not contribute. Recipe simplifies — drop LVLH for Phase 5. |
| 60–75% | LVLH contributed marginally. Worth keeping but not load-bearing. |
| < 60% | LVLH was load-bearing under curriculum, even though standalone R1 was within noise. Important finding — keep, and update the writeup. |

**Risks.**
- Stage 1 fresh fails on seed 42 (Phase 4 showed seed 42 succeeds at 77% with LVLH). If it fails, that itself is a finding (LVLH may help bootstrap), and we re-run on a different seed.
- Re-running Stage 1 fresh introduces variance; we should compare against Phase 4 Stage 1 seed 42 number (77.0%) not just the final 81.3%.

### 2.2 Ablation B — R4 curriculum without gated shaping

**Hypothesis.** The NHR gated shaping at low weights (β=1.0, w=0.01) contributes to learning under curriculum even though R5 collapsed it on warm-start. If shaping helps, this ablation will land below 79.6%.

**Implementation.**
- Set `BETA_SHAPE = 0.0` in `orbital.h` (or set `w_orbit = w_phase = w_vel = 0.0`).
- Keep all other Phase 4 R4 settings (LVLH, warp action, NHR clamp on terminals).
- Verify: shaping reward addition in `c_step` produces zero before `phi_prev` update.

**Run.** Same curriculum as Ablation A, seed 42, fresh from scratch at Stage 1. Code is identical except for the BETA_SHAPE value, so technically warm-starting from Phase 4's Stage 2 is permissible, but for clean comparison run from scratch.

**Expected outcomes and interpretation.**

| Stage 3 result | Interpretation |
|---|---|
| ≥ 75% | Shaping is inert under curriculum. Recipe simplifies — Phase 5 can use sparse-only with `R_terminal` plus the exp-decay bonus. |
| 60–75% | Shaping is helping, but not critical. Keep at low weights. |
| < 60% | Shaping is load-bearing. Keep, and consider whether tighter eps thresholds would help further. |

**Risks.**
- The exp-decay terminal bonus (`50·exp(-d/5km) + 50·exp(-vr/10m/s)`) was introduced as part of R2's terminal reward. Decide whether to also strip the bonus or only the per-step shaping. **Recommendation: strip both the per-step Φ-diff shaping and the exp-decay terminal bonus together** — they were a package, and isolating just one loses the cleanness of the ablation. If the run regresses, a follow-up can isolate which sub-component mattered.

### 2.3 Ablation C — Discrete(9)→Discrete(10) action-table zero-pad

**Hypothesis.** The Stage 1 seed-1337 failure (0% at 30°) is caused by Discrete(10)'s extra logit creating a bootstrap pathology that some random initializations can't escape. Warm-starting from Phase 3's Discrete(9) Stage 1 policy (99% at 30°) with the warp-action logit zero-initialized should recover most of the 22pp regression.

**Implementation.**

This is a more involved surgery than A or B. Steps:

1. Locate Phase 3's Discrete(9) Stage 1 checkpoint with 99% performance (PROGRESS.md references this; pull the exact run ID).
2. Write a one-shot conversion script (`scripts/orbital/expand_action_d9_to_d10.py`):

```python
# Pseudocode
state = torch.load(d9_ckpt_path, weights_only=True)

# Action head: actor's final linear layer
# Shape changes from [9, hidden] → [10, hidden]
# Bias changes from [9] → [10]
old_w = state['policy.actor.weight']           # [9, 128]
old_b = state['policy.actor.bias']              # [9]

new_w = torch.zeros(10, old_w.shape[1])
new_w[:9] = old_w
# Row 9 = warp action: zero-init weight (small softmax prior)
# Already zero from torch.zeros

new_b = torch.zeros(10)
new_b[:9] = old_b
# new_b[9] = 0  →  warp logit prior matches whatever bias-0 means in the old softmax
# Optional: bias warp slightly negative (e.g., -1.0) to start with ~3% prior instead of ~9%

state['policy.actor.weight'] = new_w
state['policy.actor.bias'] = new_b
torch.save(state, expanded_ckpt_path)
```

3. **Important:** also need to handle the obs side. If the source D9 ckpt was trained with 33-dim obs and we want to run under Phase 4's 38-dim LVLH-extended obs, we also need the encoder zero-pad surgery from R1. So this experiment combines two surgeries: action-head expand + encoder input expand.

4. Continue training under Phase 4's full R4 curriculum config from Stage 2 onward (the surgery target *is* a Stage 1 policy, so it skips Stage 1 entirely):

```bash
# Skip Stage 1 — surgery starts us with a 99% Stage 1 baseline.
# Run Stage 2 (90°, 15M) directly:
puffer train puffer_orbital --train.seed <S> --train.total-timesteps 15_000_000 \
  --train.device cpu --env.init-phase-gap-max 1.5708 \
  --load-model-path <surgery_ckpt>

# Then Stage 3 as usual.
```

**Expected outcomes and interpretation.**

| Stage 3 result | Interpretation |
|---|---|
| ≥ 80% reliably across 3 seeds | Surgery is the right Stage 1 fix. Phase 5 can drop "Stage 1 retry rate ~1/3" as a known issue. |
| Comparable to native curriculum | Surgery is an alternative path but doesn't dominate. Document and move on. |
| Worse than native curriculum | Action-table surgery is brittle in ways the literature suggests it might be. Skip it and live with the seed-sensitivity. |

**Risks.**

- The bias initialization for the new warp logit affects the prior probability. Bias=0 gives a uniform-ish prior (~10% on the warp action under softmax), which is close to Phase 4's behavior at Stage 1 init but applied to a policy that has *already learned* the other 9 actions. This could destabilize the existing strategy.
- **Recommendation:** start with bias = -1.0 on the warp logit (~3% prior), giving the existing policy room to keep its current strategy while the warp action's value is learned. If that doesn't recover Stage 2/3 performance, try bias = 0 and bias = -2.
- Encoder surgery + action-head surgery in the same checkpoint is a stack of two interventions. If the result regresses, attribution is muddled. **Mitigation:** do encoder surgery alone first as a sanity check (load it, eval at 30° without further training, confirm it gets ~99% — that's R1's zero-pad behavior verified). Then add the action-head surgery on top.

### 2.4 Ablation D — DAPO clip-higher under R4 curriculum

**Hypothesis.** The Phase 4 R3 evaluation tested asymmetric clip in two regimes that don't apply to the working recipe: warm-start onto a committed (entropy-0.16) policy, and from-scratch at 180°. Under the curriculum, DAPO's mechanism (asymmetric upper clip allows initially-rare good actions to escalate confidence faster) plausibly applies during the Stage 2 → Stage 3 transition, where the policy has to extend the phasing strategy from 90° to 180° and the right action sequence is by definition rare under the Stage 2 distribution. This is a single-component test — the rest of R3 (L2-init, LayerNorm, adaptive-KL, TEC) stays out.

**Why this and not the others.** Of R3's five components:

- **DAPO clip-higher** has a clean theoretical case under curriculum: the policy needs to escalate confidence in increasingly hard-to-discover action sequences, and symmetric clip caps that escalation at +20% per update.
- **L2-init** regularizes toward the warm-start weights of *each* curriculum stage — counterproductive when the whole point of each stage is to learn beyond the previous one.
- **LayerNorm** broke warm-start completely in R3b. Cannot be added mid-curriculum.
- **Adaptive-KL** is partly redundant with `anneal_lr=True` already in the recipe.
- **TEC** kept entropy up but did not help discovery in R3-FS; the curriculum already addresses discovery.

DAPO is the only one whose mechanism cleanly maps to a thing the curriculum can plausibly need. Limit Phase 4.5 to it.

**Implementation.** Single config change to `orbital.ini`:

```ini
clip_coef_low  = 0.2
clip_coef_high = 0.28
```

Verify the DAPO codepath is intact in `pufferl.py` (lines 346–350, 434–437 from the deep dive). It was implemented for R3 and not removed when R3 was stripped — should be a config-flag flip, not a code change.

**Run.** Single seed, warm-start from Phase 4's Stage 2 checkpoint (seed 42) to isolate the question to the hard-stage transition. Skip Stage 1 entirely (the bootstrap question is settled by Phase 4 — DAPO doesn't help discovery). Run Stage 3 only.

```bash
# Stage 3 only, warm-started from Phase 4 Stage 2, with DAPO clip-higher.
# Phase 4 Stage 2 ckpt: experiments/puffer_orbital_h2ccoyi1/model_puffer_orbital_000115.pt
puffer train puffer_orbital --train.seed 42 --train.total-timesteps 15_000_000 \
  --train.device cpu --env.init-phase-gap-max 3.14159 \
  --load-model-path experiments/puffer_orbital_h2ccoyi1/model_puffer_orbital_000115.pt
```

Compare directly to Phase 4 Stage 3 seed 42 result (94.1% rolling train, 81.3% multi-rollout eval).

**Expected outcomes and interpretation.**

| Stage 3 result | Interpretation |
|---|---|
| ≥ 86% (delta ≥ +5pp on rolling-train) | DAPO contributes under curriculum. Add to Phase 5 recipe. Trigger follow-up testing of L2-init under curriculum. |
| 79–86% (within seed noise of baseline) | DAPO is inert under curriculum. Don't add. R3 components are dead for this project. |
| < 79% | DAPO is harmful even under curriculum. The mechanism that killed R3 on warm-start is more general than the Phase 4 evidence suggested. Definitively dead. |

**Risks.**

- Stage 2 → Stage 3 with a config change is itself a small distribution shift on a committed policy. Phase 4's R5 results suggest committed-policy + new-clip-shape is exactly the failure mode that destroyed R5 v1–v4. **Mitigation:** if entropy crashes within the first 1M steps (mirroring R5's collapse arc), kill the run early. The signature was peak 60–68% then crash to <10% with entropy < 0.01 — same trajectory shape would invalidate the test.
- ε_high = 0.28 is what DAPO recommends; we tested it in R3c without success. If this run succeeds, that's evidence the Phase 4 R3c failure was a different binding constraint (the surrounding L2-init / adaptive-KL machinery) rather than the clip asymmetry itself. If both fail, it's the same constraint.
- Single seed is sufficient for binary attribution. If the result is borderline (say 84%), running seed 20260423 to disambiguate is a 30-minute follow-up.

**What this does *not* test.**

- DAPO under from-scratch curriculum (Stage 1 → Stage 2 → Stage 3 fresh). That would be a separate run; first-pass test on Stage 3 only is enough to settle the binary question. If Stage 3 results are positive, the from-scratch curriculum follow-up is worthwhile.
- Combined R3 components. Explicitly out of scope.

---

## 3. Eval protocol (applies to all four)

For each ablation:

1. After each stage, run `eval_checkpoint.py` at the stage's phase gap (50 episodes argmax, fixed eval seed).
2. After Stage 3, run the standard rendezvous eval: 50 episodes argmax, 3 rollout seeds (so 150 total episodes), report mean and per-rollout breakdown.
3. **Critical:** verify `--e-max-target 0.0` to match training. The bug from Phase 4 entry must not recur.
4. Log to wandb under group `phase4.5-attribution`. Run names: `ablation_a_no_lvlh_seed42`, `ablation_b_no_shaping_seed42`, `ablation_c_action_surgery_seed{42,1337,20260423}`.

Diagnostic logging to keep:
- True success rate per checkpoint (every 10 epochs).
- Entropy per update (rolling mean over 10 updates).
- Mid-episode abort rate per checkpoint (if reachable from existing infra).
- For Ablation C only: per-action-class usage rate, especially warp-action usage.

---

## 4. Decision matrix after Phase 4.5

The output of Phase 4.5 is a 4-row table:

| Ablation | Result | Recipe change for Phase 5 |
|---|---|---|
| A — no LVLH | _____ | Keep LVLH / drop LVLH |
| B — no shaping | _____ | Keep shaping / drop shaping (and exp-decay bonus) |
| C — action surgery | _____ | Use surgery for Stage 1 init / use fresh Stage 1 with retry rate |
| D — DAPO under curriculum | _____ | Add DAPO to recipe / leave R3 dead |

Phase 5 enters with a recipe that contains exactly the ingredients Phase 4.5 confirms are contributing. If LVLH and shaping both drop out, Phase 5 starts with the simplest recipe (warp + curriculum + exp-decay terminal bonus only), which is also the easiest to extend to multi-body in Phase 6.

---

## 5. What this is *not*

- **Not a re-run of Phase 4 with different seeds.** The seed-42-only design is intentional: first-pass attribution is enough to settle the binary question of "does this ingredient contribute," and full multi-seed confirmation can wait until Phase 5 final results.
- **Not a chance to re-attempt the full R3 plasticity stack.** Ablation D tests DAPO clip-higher in isolation under curriculum — a single component of R3 with a clean theoretical case for the curriculum regime. The other R3 components (L2-init, LayerNorm, adaptive-KL, TEC) stay out of Phase 4.5 because their priors are weaker. If Ablation D succeeds, that's the trigger for testing the others in a follow-up.
- **Not a curriculum-design exploration.** Phase 4.5 holds the curriculum schedule (30° → 90° → 180°) fixed and varies only the ingredients. Curriculum design is Phase 5's problem.

---

## 6. Sequencing within Phase 4.5

Run order matters slightly because of compute economy:

1. **Ablation A (no LVLH)** first. It's the cheapest and tells us whether the obs-space change has been doing work.
2. **Ablation B (no shaping)** second. Same compute as A. Together with A, gives us the two "is this ingredient inert" answers.
3. **Ablation D (DAPO under curriculum)** third. Stage 3 only, warm-start from Phase 4 Stage 2 — fastest of the four (~25 min vs ~40 min for full curriculum). Slot here so it runs alongside the A and B writeups.
4. **Ablation C (action surgery)** fourth. More involved (two surgeries to chain, three seeds for confirmation). If A and B both come back as "inert," the recipe simplifies enough that C becomes more interesting (the only complex ingredient is then the surgery itself).

Total wall time: ~7–9 hours of compute (A ~40min, B ~40min, D ~25min, C ~3hr including 3 seeds), spread over an afternoon and evening.

---

## 7. Output artifact

A short markdown writeup (~1 page) appended to PHASE4_FINDINGS.md as a §9 "Phase 4.5 attribution":

- Three ablation rows with results.
- One sentence per row of interpretation.
- Final "Phase 5 recipe" section listing exactly what's in/out.

This is the closing piece for the Phase 4 narrative and the input for Phase 5 planning. It also fills a real hole in the eventual blog post — the attribution paragraph that lets the reader know we earned the result rather than getting lucky on a coupled experiment.

---

*Author: 2026-04-25. Pre-Phase 5 attribution closure. Successor doc: Phase 5 spec.*
