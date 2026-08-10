# Phase 5b Step 1 — Cap-Tail Analysis

**Date:** 2026-04-28 · Follow-up on `PHASE5b_STEP1_FINDINGS.md` · ~5 min post-hoc.

## Question

Step 1's seed-42 eval at Stage 1.0 lands 84.5% success / 15.5% safety_cap. Are those 31 cap failures clustered on hard (phase × e) combos, or distributed uniformly? And: is the agent grazing the tolerance window (brittleness — could be fixed with looser tolerance) or far away (capability gap — needs more training/recipe work)?

Tool: `scripts/orbital/p5b_step1_cap_tail_analysis.py`. Computes per-episode min distance, min relative velocity, min "miss factor" = `max(dist/30km, rel_vel/50m/s)` (1.0 = exactly at threshold; >1 = outside).

## Results

### Brittleness vs capability gap

| Failure category (n=31) | Count | % |
|---|---|---|
| Very close (miss factor < 1.1) | 1 | 3.2% |
| Close (< 1.5) | 3 | 9.7% |
| **Far (≥ 3.0)** | **25** | **80.6%** |

Failure stats: median min-distance **11,051 km**, median min-rel-velocity **12,066 m/s**, median miss factor **368×**. Loosening the tolerance window wouldn't fix this — the agent is never close.

**Verdict: 80% capability gap, 10% borderline, 3% true brittleness.**

### (Phase × eccentricity) success grid (seed 42, 200 eps)

| abs phase gap ＼ target.e | 0 – 0.025 | 0.025 – 0.05 |
|---|---|---|
| 0 – 60° | 32/33 = **97.0%** | 40/41 = **97.6%** |
| 60 – 120° | 31/39 = 79.5% | 25/30 = 83.3% |
| **120 – 180°** | **15/31 = 48.4%** | **26/26 = 100.0%** |

Two findings:
1. **Phase gap is the dominant difficulty axis** — small phase ≈ 97%, large phase varies 48-100%.
2. **At large phase, eccentricity inverts the difficulty.** Low-e is hardest (48%); high-e is fully solved (100%). Counter to the simple "harder = both axes max" intuition.

## Mechanism hypothesis: natural phasing on eccentric orbits

Under `same_orbit_init=1`, sat and target start on the *same* orbit, only θ differs. Phasing requires changing the time at which sat reaches a given point on the orbit. Two strategies:

- **Transfer-orbit phasing (circular case).** Raise or lower semi-major axis temporarily → enter a faster/slower drifting orbit → wait for relative phase to shift → return to original orbit. Costs fuel for two burns and requires episode time.
- **Natural phasing (eccentric case).** On an eccentric orbit, angular velocity is non-uniform: ~`v_circ × (1−e)/(1+e)` slower at apoapsis, faster at periapsis. The agent can simply *wait* through apoapsis to let the target close the phase gap. No fuel cost beyond minor station-keeping.

The data is consistent with this: the agent has emergently learned to exploit eccentricity as a phasing mechanism, which is why high-e + high-phase cases are *easier* than low-e + high-phase. On near-circular orbits the agent must execute a transfer-orbit maneuver that apparently it doesn't always nail.

This is a sophisticated emergent strategy — a free positive finding from Step 1.

## Implications for Stage 1.x scaling

1. **Expanding e_max (0.05 → 0.10 → 0.20+) should help, not hurt the headline number.** The harder corner is `(high-phase, low-e)`, which doesn't change as e_max expands; the new `(high-phase, high-e)` cells should benefit from the natural-phasing strategy.

2. **The 48% (high-phase, low-e) bottleneck is a Phase 4 territory issue resurfacing.** With near-circular orbits, the agent must do transfer-orbit phasing, which Phase 4's Stage 3 also struggled with on the long-phase end. Mitigations to test in Phase 5b proper:
   - **Raise MAX_STEPS** from 2000 to 3000 — gives transfer-orbit phasing room to complete a full transfer + drift + return. Cheap (config knob).
   - **Curriculum reweighting** to oversample low-e/high-phase episodes during training. Adds a sampling bias kwarg.
   - **Phase 4 ckpt warm-start** for the low-e regime specifically — Phase 4 Stage 3 reached 80% at e=0 with phase gap up to π. A hybrid ckpt initialization (random sat init + warm-from-Phase-4) might compose. Untested.
   - **Action-space audit** — radial actions are barely used (radial- 0.34%, radial+ 0%). Maybe action-discretization is the bottleneck for transfer maneuvers.

3. **Track natural-phasing skill across Stage 2-4 transitions.** When sat.ω, sat.a become random (Stages 2-4), the agent loses the same-orbit guarantee that enables natural phasing. Watch whether this strategy persists or has to be re-learned.

## What this isn't

- Not multi-seed. Single-seed (42) post-hoc on existing data; the (phase × e) bin counts are noisy. Re-run on seeds 1337 and 20260423 to confirm the pattern would tighten the result.
- Not a Stage 1.1 prediction. The natural-phasing inversion may shift as e_max scales; we expect it to persist but haven't tested.

---

## Follow-up findings (2026-04-28, +30 min)

Three more probes refine the cap-tail story.

### A. Step 1 ckpt at Phase 4 conditions

Stage 1.0 ckpt (`puffer_orbital_177741559081/model_puffer_orbital_000150.pt`) evaluated at Phase 4's task — `e_max_target=0`, `e_max_sat=0`, `same_orbit_init=0`, full π phase gap, 50 eps × 3 rollout seeds:

| Rollout seed | Step 1 @ Phase 4 conditions | Phase 4 baseline |
|---|---|---|
| 42 | 36.0% | 81.3% (R4 ckpt) |
| 1337 | 14.0% | 86.0% |
| 20260423 | 24.0% | 80.0% |
| **mean** | **24.7%** | **79.6%** |

**Generality cost: -55pp.** Step 1 is a Stage 1 specialist (sat and target on the SAME orbit, only θ differs). Phase 4's task — sat circular at one altitude, target circular at a different altitude — requires an a-transfer (Hohmann-like maneuver) that Step 1 never trains on. The Step 1 policy is OOD for this and reverts to ineffective coasting.

This isn't a regression in "shipping" terms — Phase 4 ckpt is preserved as the canonical solver for the Phase 4 task. But it does mean Stage 1's policy doesn't subsume Phase 4 capability. Phase 5b's later stages (Stage 3 = transfer rendezvous; Stage 4 = fully general) will need to re-acquire that skill.

### B. Episode length distribution for cap failures

| Group | min | median | mean | max |
|---|---|---|---|---|
| Successes (n=169) | 4 | **197** | 270 | 1883 |
| Failures (n=31) | 2000 | 2000 | 2000 | 2000 |

**100% of failures hit MAX_STEPS=2000 exactly. None terminate early.** Successful episodes finish fast — median 197 steps (~2 orbital periods). Raising MAX_STEPS would in principle let the agent keep trying, but only matters if the agent is actually trying.

### C. Action distribution conditional on success vs cap-failure

| Action | Success episodes | Cap-failure episodes |
|---|---|---|
| 0 coast | 3.6% | **30.9%** (8.5× more) |
| 1 prograde+ | 6.0% | 0.4% |
| 4 retro++ | 5.6% | 0.3% |
| 8 radial- | 0.7% | 0.06% |
| 9 warp | 84.1% | 68.3% |
| **Total burn** | **12.3%** | **0.7%** (17× less) |

| Group | Δv used (median) | Δv used (mean) |
|---|---|---|
| Successes | 165 m/s | 175 m/s |
| **Failures** | **0 m/s** | 80 m/s |

**The agent literally doesn't burn on its failure cases.** Median fuel usage on failures is zero — the policy's argmax is "coast or warp" for the entire 2000-step episode. This isn't a policy that's *trying* and missing — it's a policy that has *stopped trying.*

### Refined cap-tail mechanism

Combining (B) and (C): the failure mode isn't "agent runs out of episode time mid-maneuver" — it's "agent decides not to maneuver, then runs out the clock." That's qualitatively different from a capability gap. Both interpretations land in the same decision-matrix bucket (the recipe doesn't generalize to those states), but the mechanism — **policy passivity on under-sampled states** — has different mitigation implications than "the maneuver is too hard / too long."

The (high-phase, near-circular) corner is exactly the under-sampled region: with target.e ~ Uniform(0, 0.05), only ~10% of episodes have e<0.005 (truly near-circular), and most of those don't simultaneously land at high phase gap.

### Sharpened Stage 1.x scaling implication

If we expand e_max from 0.05 → 0.10 → 0.20, the under-sampling of near-circular cases gets *worse*, not better:

| e_max | Fraction of eps with e < 0.005 |
|---|---|
| 0.05 | 10% |
| 0.10 | 5% |
| 0.20 | 2.5% |
| 0.50 | 1% |

Naïve Stage 1.x scaling will make the (high-phase, low-e) corner failure rate larger, not smaller. Three ways to fix:

1. **Stratified e sampling.** Guarantee a fixed fraction (~30%) of episodes have e ∈ [0, 0.005] regardless of e_max. Cheap env-side change.
2. **Curriculum reweighting (DORAEMON-style).** Track per-bin success rate, oversample failing bins. More involved (requires a stats hook + a sampling-distribution kwarg).
3. **Hybrid task mixture.** Mix in a fraction of Phase-4-style episodes (different-a, circular) so the policy continues to practice transfer-orbit maneuvers. Cheap if `same_orbit_init` becomes a per-episode random flag rather than a global setting.

Option 1 + Option 3 in combination would directly address the cap-tail mechanism and likely close most of the (high-phase, low-e) failure rate. Worth implementing before the Stage 1.x e_max expansion.

---

*Cap-fail mechanism is "agent doesn't try" (median Δv = 0), not "agent tries and misses." Stems from under-sampled (high-phase, low-e) corner. Stage 1.x e expansion will make this worse without sampling-distribution reform.*

---

## Mechanism probes — under-sampled OOD vs learned-to-not-try (2026-04-28, +30 min)

The previous follow-up established that failures show median Δv = 0. Three more probes test whether this is **P1: under-sampled OOD** (policy never explored these states) or **P2: learned-to-not-try** (policy explored, then mode-collapsed to passivity).

### Probe 1 — Φ_orbit excursion: success vs failure

For each of seed-42's 200 eps, compute per-step Φ_orbit (=`|Δa|/SUCCESS_TOL_A + ||Δē||`) and the excursion (max − init):

| Group | Excursion <0.1 (chaser stays put) | Excursion ≥1.0 (genuine transfer) |
|---|---|---|
| Successes (n=169) | 0/169 (0%) | 166/169 (98%) |
| Failures (n=31) | **20/31 (65%)** | 7/31 (22%) |

Median success excursion: **3.83**. Median failure excursion: **0.0**. Successes uniformly use orbit-transfer maneuvers; the majority of failures don't transfer at all. Confirms median-Δv-zero finding via an independent metric.

### Probe 2 — Action distribution across training (corner states only)

Eval each saved ckpt (epochs 10, 30, …, 150) at the narrow corner condition `(init_phase=π, e_max=0.01, same_orbit_init=1)`, 50 eps:

| Epoch | Corner success | Coast % | Burn % | Warp % | Median Δv |
|---|---|---|---|---|---|
| 10 | 4% | 16% | 0.1% | 84% | 0 |
| 30 | 14% | 37% | 1.4% | 62% | 2 |
| 50 | 22% | 11% | 2.9% | 87% | 257 |
| **70** | **48%** | 5.7% | **4.1%** | 90% | **312** |
| 90 | 50% | 13% | 3.0% | 84% | 150 |
| 110 | 48% | 22% | 2.0% | 76% | 130 |
| 130 | 68% | 29% | 3.4% | 68% | 220 |
| **150** | **68%** | **32%** | 3.7% | 65% | 207 |

Patterns:
- Burn % peaks at epoch 70 (4.1%), then declines and stabilizes at ~3.5%.
- Coast % is non-monotonic: minimum at epoch 70 (5.7%), then climbs to 32% at end.
- Median Δv peaks at epoch 70 (312 m/s), then declines to 207.
- **Yet success rate continues climbing (48% at epoch 70 → 68% at epoch 150).**

The policy did explore corner states — epoch 70 shows clear active-maneuvering signature. As training progresses past epoch 70, burn frequency and fuel usage *decrease* while success rate *increases*. The policy converged toward fewer-but-more-precise burns, not toward giving up.

**Verdict: not pure P1 (the corner WAS explored) and not pure P2 (the policy got better at the corner, not worse).**

### Probe 3 — Sampled-action eval at the corner

Same ckpt (epoch 150), same corner condition, 100 eps each:

| Action selection | Success rate |
|---|---|
| **Argmax** | **62%** |
| **Sampled** | **1%** |

Sampled is **dramatically worse** than argmax. This is the opposite of the "argmax wrong, sampling would solve it" pattern. The policy has learned a sharply-peaked argmax distribution that produces precise burn timing; stochastic deviations destroy the maneuver.

### Refined mechanism: P3 — argmax brittleness via efficiency convergence

Combining the three probes, the cap-tail mechanism is neither under-exploration nor mode collapse — it's **convergence to a sharply-peaked argmax that doesn't cover every corner state.**

The trajectory through training:
1. **Epochs 10–70 (exploration):** policy actively maneuvers, burn rate climbs, Δv usage climbs, success rate climbs. Standard PPO exploration.
2. **Epochs 70–150 (efficiency refinement):** PPO drives the policy toward fewer, more precise burns. Coast share grows, burn share shrinks slightly, Δv usage drops. Success rate continues climbing because efficient maneuvers are more effective than over-burning.
3. **Tail behavior:** the efficient policy's argmax sequence is correct on ~68% of corner states. For the remaining ~32%, the precise burn timing is slightly off and argmax falls back to coast. The policy *can* burn — it just doesn't, because for these specific initial states, the value function doesn't see a clear improvement option. Stochastic exploration during training would reach success on these states; argmax doesn't.

This is consistent with an entropy-coefficient that's too low to maintain breadth at the corner. With `ent_coef=0.01`, the policy converges sharply on the dominant maneuver template and trims off the long-tail variants needed for edge cases.

### Sharpened mitigation strategy for Stage 1.x

The three earlier mitigations (stratified e sampling, curriculum reweighting, hybrid task mixture) all assumed the corner was under-explored or under-represented. The probes reverse this — the corner *was* explored. New mitigation list, ordered by expected leverage:

1. **Higher entropy floor specifically at corner states** — adaptive entropy or per-state entropy regularization. Discourages sharp argmax convergence on the long-tail. Requires a per-state entropy hook.
2. **Curriculum reweighting (DORAEMON-style) on failing bins** — directly oversample the corner during late training. Forces PPO to keep refining there instead of spending compute on already-solved states. Cheaper than per-state entropy.
3. **Two-pass training** — train to convergence (Phase 4-style schedule), then a second pass with raised `ent_coef` (e.g., 0.03) and corner-biased sampling. Should re-broaden the argmax distribution.
4. **Policy distillation from epoch 70 ckpt** — the active-maneuvering policy at epoch 70 had broader argmax behavior. Distill back to a final policy that retains some of that breadth. Most invasive; defer unless 1-3 don't work.

The earlier "stratified e sampling" suggestion is still useful for Stage 1.x scaling (the under-sampling problem at higher e_max is real), but it doesn't address the cap-tail mechanism alone. Pair it with one of the entropy/reweighting mitigations.

### What this isn't

- Not multi-seed. Single-seed (42) probes; the 68% / 62% / 1% numbers are noisy at n=100. Multi-seed re-runs would tighten error bars but the qualitative pattern (argmax >> sampled at the corner) is robust.
- Not a Stage 2-4 prediction. P3-flavored argmax brittleness might appear at Stages 2/3/4 too, but the mechanism details depend on the specific state distribution at each stage.

---

*Final cap-tail mechanism: argmax brittleness from efficiency convergence (P3), not under-exploration (P1) or mode collapse (P2). Mitigation focus shifts from "increase exposure" to "preserve argmax breadth."*

---

## Train-longer experiment (2026-04-28, +30 min) — overturns P3

The P3 verdict above predicted that more training wouldn't help — the policy had supposedly already converged to an over-sharp argmax that no amount of additional PPO updates would fix. **This prediction is wrong.**

### Setup

Continued Stage 1.0 from the epoch 150 checkpoint (`puffer_orbital_177741559081/model_puffer_orbital_000150.pt`) for an additional 20M steps under identical conditions: same env (`init_phase_gap_max=π, e_max_target=0.05, e_max_sat=0.05, same_orbit_init=1`), same seed (42), same hyperparameters, `--train.checkpoint-interval 10`. New run id: `puffer_orbital_177742224544`.

### Corner success rate vs additional training

Eval condition: corner = `(init_phase=π, e_max_target=0.01, e_max_sat=0.01, same_orbit_init=1)`, 100 eps argmax, seed 42.

| Cumulative training steps | Corner success rate |
|---|---|
| 20M (original final ckpt) | 62% |
| +6.5M (~26M cumulative) | **80%** |
| +13M (~33M cumulative) | **96%** |
| +20M (~40M cumulative) | **100%** |

The corner failures didn't bottom out at 62% — they monotonically dropped to zero with more training.

### Full Stage 1.0 multi-rollout at the extended final ckpt

200 eps argmax × 3 rollout seeds, identical to the original §5.1 protocol:

| Rollout seed | Original (epoch 150) | Extended (cumulative ~40M) |
|---|---|---|
| 42 | 84.5% | **100.0%** (200/200) |
| 1337 | 83.5% | **99.0%** (198/200) |
| 20260423 | 77.0% | **100.0%** (200/200) |
| **mean** | **81.7%** | **99.7%** |

+18pp on the headline number. The "15.5% safety_cap tail" we spent three rounds of probes investigating just disappears with more training.

### What overturns

The P3 verdict claimed:
1. The policy explored corner states (epoch 70 burn 4.1%, Δv 312)
2. PPO refined toward sharp argmax that fails on edge cases
3. Mitigation requires "preserve argmax breadth" interventions (adaptive entropy, curriculum reweighting, two-pass training, distillation)

What actually happened: the sharpening at epoch 150 was real, but it wasn't terminal. PPO continued refining the long-tail past epoch 150, expanding the argmax's coverage of corner states. The "sharp argmax that fails on edge cases" was a snapshot of mid-convergence, not steady-state.

The probe-3 finding (argmax 62%, sampled 1%) still holds at epoch 150 — sampling deviates from the precise sequence. But the *mechanism* I attributed it to (PPO over-converging) was wrong. The right mechanism: **at epoch 150 the long-tail of corner states was simply not yet trained on enough**. By epoch 303 (cumulative), they were.

### Mirage absence

Training-time peak at extend epoch 150: 99.7-99.8%. Held-out eval: 99.7% mean. **Train/eval gap = 0.1pp.**

The PHASE5a A2 mirage was a difficulty-cliff artifact — variance manifested when learning was infeasible at the recipe's edge. Once the policy is genuinely converged on a feasible task, training-time `perf` and held-out eval agree closely. Conversely, train/eval gaps remain a useful warning signal that something is still being learned (or failing to be learned) in the policy.

### Updated cap-tail verdict

**Mechanism: under-converged training, not P1 / P2 / P3.** The earlier mechanism probes were observing a snapshot of an in-progress convergence and over-interpreted it as a structural property.

The right mitigation is the simplest one available: **train longer**. The Phase 4 schedule (10M Stage 1, 15M Stage 2, 15M Stage 3) was calibrated for the easier Phase 4 task. Phase 5b's tasks are harder per-stage, and need more steps to fully converge.

### Implications for Phase 5b proper

The mitigation list shrinks dramatically:

1. **Stratified e sampling** — still useful for Stage 1.x scaling at e_max ≥ 0.10 (under-sampling at high e_max remains a real concern), but no longer needed to fix the cap tail at e_max=0.05.
2. **Curriculum reweighting (DORAEMON-style)** — defer. Train-longer alone resolves the corner failure rate; reweighting is now a "marginal improvement" question, not a "make it work" question.
3. **Adaptive entropy / two-pass training / policy distillation** — drop entirely from Stage 1 mitigation list. P3 was the wrong frame.

**New Stage 1.x compute estimate: 40M steps per sub-stage** (vs the originally-budgeted 15M warm-start per stage). This is more compute but less engineering, which is the trade-off worth taking. ~1.5× compute for a much simpler recipe.

### What this teaches about the project's debugging discipline

The cap-tail probes (Φ_orbit excursion, action distribution across training, sampled vs argmax) were each individually correct measurements. The P3 conclusion overshot because we extrapolated from a single training duration without testing the train-longer null hypothesis first.

Lesson for future analyses: **before declaring a structural failure mode, run the train-longer baseline.** It's the cheapest control experiment in this project (~3 min wall) and resolves a wide class of "is this a recipe bug or just under-trained?" questions in one shot. The cost of skipping it was three layers of mechanism speculation that the data immediately overturned.

---

*Final-final cap-tail verdict: under-converged training. ~40M steps suffice to drive Stage 1.0 success to 99.7%. The P1/P2/P3 framework was a snapshot mistake, not a structural finding.*

---

## Post-extend characterization (2026-04-29)

Two follow-up evals on the extended (40M cumulative) Stage 1.0 ckpt: a fresh (phase × e) grid breakdown, and a Phase 4 conditions test. Settle whether the gains are even across the difficulty surface and whether they transfer outside the training distribution.

### Question 1 — Has the (phase × e) grid filled in evenly?

200-ep eval at Stage 1.0 conditions (`init_phase=π, e_max=0.05, same_orbit_init=1`), seed 42, broken down by initial phase gap and target eccentricity:

| abs phase gap ＼ target.e | 0 – 0.025 | 0.025 – 0.05 |
|---|---|---|
| 0 – 60° | **100.0%** (33/33) | **100.0%** (41/41) |
| 60 – 120° | **100.0%** (39/39) | **100.0%** (30/30) |
| **120 – 180°** | **100.0%** (31/31) | **100.0%** (26/26) |

Compare to the same grid at 20M training:

| abs phase gap ＼ target.e | 0 – 0.025 | 0.025 – 0.05 |
|---|---|---|
| 0 – 60° | 97.0% | 97.6% |
| 60 – 120° | 79.5% | 83.3% |
| **120 – 180°** | **48.4%** | 100.0% |

The (high-phase, low-e) corner — 48% at 20M — is at 100% at 40M. Every other cell is now 100% as well. **The grid filled in uniformly with extra training; no structural corner remains.**

The earlier "natural-phasing-on-eccentric-orbits saves the high-e corner" finding still holds — at 20M, the high-phase × high-e cell already converged because the agent could exploit eccentricity for free phasing. The low-e corner needed full transfer-orbit competence, which takes longer to refine. By 40M both are solved.

Min miss factor for the 200 successes: median 1.07, mean 2.33, max 20.8. A handful of episodes succeed close to the tolerance threshold (miss ≈ 1) — there's a marginal tail in the *tightness* of success even though all 200 succeed. Worth noting if Phase 5b later considers tightening tolerances.

### Question 2 — Did extended training change Phase 4 conditions performance?

Eval the extended ckpt at Phase 4's task — `e_max_target=0`, `e_max_sat=0`, `same_orbit_init=0`, full π phase gap, 50 eps × 3 rollout seeds:

| Rollout seed | Original Step 1 ckpt (20M) | Extended ckpt (40M) | Phase 4 baseline |
|---|---|---|---|
| 42 | 36.0% | **74.0%** | 81.3% |
| 1337 | 14.0% | **66.0%** | 86.0% |
| 20260423 | 24.0% | **58.0%** | 80.0% |
| **mean** | **24.7%** | **66.0%** | **79.6%** |

**Extended training closed +41pp of the generalization gap** (24.7% → 66%) without ever training on Phase 4 conditions. The remaining −14pp shortfall (66% vs Phase 4's 79.6%) is consistent with the structural difference: Phase 4's task requires a-transfer maneuvers (sat.a ≠ target.a), and Stage 1's `same_orbit_init=1` constraint means sat.a always equals target.a during training.

The partial OOD generalization mechanism: random sat.e ∈ [0, 0.05] means roughly 20% of training episodes have sat.e < 0.01 (very near circular). Those subsidize the "near-circular chaser" capability. Combined with broader maneuvering experience from longer training, the extended policy handles Phase 4-style tasks decently — even though a-transfer specifically remains OOD.

### Updated lesson

Both findings push the same direction: **longer training delivers both depth (in-distribution corner closure) and breadth (OOD generalization) simultaneously.** This is contrary to the standard prior that compute-heavy training overfits to the training distribution at the cost of generalization. In this regime — well-shaped task, good recipe — extra steps fill in difficulty corners *and* transfer to nearby tasks.

Implication for Phase 5b proper: stage transitions might be less surgical than originally planned. If Stage 1 (40M, same-orbit) already gets 66% on Phase 4-style transfer rendezvous, Stage 3 (transfer rendezvous) may converge fast as a continuation rather than from-scratch.

---

*Grid is uniformly 100% at 40M; +41pp OOD generalization to Phase 4 conditions as a free side effect. Longer training is the load-bearing intervention; per-stage budget for Phase 5b proper is 40M.*

---

## α vs β test: Stage 4 OOD + 60M training (2026-04-29)

The post-extend OOD result (66% at Phase 4 conditions, +41pp from longer training) raised two competing hypotheses for what drives the OOD generalization:

- **α (near-circular subsample):** ~20% of training eps have sat.e < 0.01 (very near circular). Those subsidize the "near-circular chaser" capability, which Phase 4 conditions reuse directly. Predicts: only sat-circular-style OOD conditions transfer; fully-random conditions (Stage 4) fail.
- **β (broad skill composition):** the policy learned phasing/transfer/burn-timing skills that compose for unseen task variations. Predicts: multiple OOD axes (Stage 2 ω-mismatch, Stage 3 a-mismatch, Stage 4 everything-random) all show partial transfer.

Two probes test this: (1) eval the 40M ckpt at Stage 4 conditions (everything random, e_max=0.05) — high score there favors β; (2) train another 20M (60M cumulative) and re-eval at all OOD conditions — if OOD continues climbing, β is real *and* unbounded; if it plateaus or regresses, the skill ceiling has structure.

### 40M ckpt — broader OOD characterization

| Condition | What's OOD | Mean (3 rollout seeds × 50 eps) |
|---|---|---|
| Stage 1.0 (in-dist) | — | **99.7%** |
| Phase 4 (sat circ, target circ, diff a) | a-transfer | 66.0% |
| Phase 4 + eccentric target | a-transfer + target.e | 62.7% |
| **Stage 4 (sat & target all random)** | a-transfer + ω-mismatch + sat.e | **58.7%** |

Stage 4 at 58.7% is **far above α's prediction of ~0%**. Pure α would require Stage 4 to fail completely (since the chaser isn't circular and there's no near-circular subsample to leverage). The fact that fully-random conditions still get 58.7% is direct evidence that broad skill composition (β) is happening.

The slight drop from Phase 4 (66.0%) → Phase 4+eccentric (62.7%) → Stage 4 (58.7%) is consistent: each additional OOD axis costs ~3-4pp.

### 60M ckpt — does broader generalization continue with more training?

Same OOD eval suite on the 60M ckpt (extended from 40M with another 20M):

| Condition | 40M mean | 60M mean | Δ |
|---|---|---|---|
| Stage 1.0 (in-dist) | 99.7% | 99.3% (50/49/50) | -0.4 (flat) |
| **Phase 4 (a-transfer, e=0)** | **66.0%** | **50.7%** (60/42/50) | **-15.3pp** |
| Stage 4 (everything random, e=0.05) | 58.7% | 58.7% (46/66/64) | 0.0 (flat) |

The Phase 4 drop is roughly 1.5-2σ given per-seed std ~9pp on 50-ep evals — borderline statistical significance, not pure noise. Even if part is variance, the *direction* is opposite to "more training → more breadth."

In-distribution Stage 1.0 plateaued (already converged at 40M). Stage 4 stayed at exactly the same mean despite different per-seed numbers (54/56/66 → 46/66/64) — the noise canceled to zero net change.

### Refined α vs β verdict: neither cleanly

The data supports **β with bounded ceiling**:

1. β is real: Stage 4 transfer at 58.7% is incompatible with α's prediction. The policy did learn composable skills, not just a near-circular template.
2. β doesn't strengthen monotonically: extra training past 40M gave 0pp on Stage 4 and -15pp on Phase 4. The breadth ceiling is roughly hit by 40M for this single-stage recipe.
3. There's a sweet spot for OOD: 40M is the peak; 60M trades some breadth for slightly more in-distribution specialty (though in-dist itself is already at ceiling).

### Implications for Phase 5b proper

The four-stage curriculum is still appropriate, but with two refinements:

1. **Per-stage budget around 30-50M, not "as long as possible."** Past in-distribution convergence, additional training mildly *erodes* OOD generalization. The 40M sweet spot for Stage 1 likely has analogues at later stages — train to convergence + a bit, then transition.

2. **Track OOD during training, not just in-distribution.** The Phase 4 OOD eval at intermediate ckpts would have revealed the 40M sweet spot directly. For Phase 5b proper, eval-during-training should include both in-distribution AND a sampled OOD test set. The peak-OOD ckpt is the right Stage transition point, not the peak-in-distribution ckpt.

3. **Stage 2/3/4 will still need their own training,** but starting from a 40M Stage 1 warm-start gives meaningful free progress (Stage 4 already at 58.7% before any direct training). The four stages may converge faster as continuations than as cold starts.

4. **The "Stage 1 ckpt as universal warm-start" hypothesis is partially validated.** At 40M, Stage 1 is solid for Stages 2-4 boot. Past 60M, that universality erodes mildly. There's an optimal handoff window.

### Methodological note

The α vs β framing was a binary categorization that the data refused to fit cleanly. β is the better description, but it's β with structure — bounded breadth ceiling, sweet spot at convergence, mild specialization past it. **Most "clean" mechanism categorizations in this project have refined into structured intermediate verdicts on contact with data.** The discipline of running the train-longer baseline before declaring structural failure modes (from the previous addendum) generalizes here too: testing at multiple training durations reveals peak/plateau structure that single-snapshot evals miss.

---

*β with bounded ceiling: skills compose broadly (Stage 4 at 58.7%) but breadth peaks at ~40M and erodes mildly past convergence. Phase 5b proper: per-stage budget 30-50M, eval-during-training should include OOD checks, four-stage curriculum is still right but stages converge faster from Stage 1 warm-start.*
