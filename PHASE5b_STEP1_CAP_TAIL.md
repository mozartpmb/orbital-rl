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
