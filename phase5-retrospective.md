# Phase 5 — Comprehensive Retrospective

> **Purpose.** This document is the lay-of-the-land for future engineering work on this project. It covers everything Phase 5 did (good and bad), what's actually known vs claimed, the methodological discipline that emerged, and the open questions. Anyone picking up the project should be able to read this and understand where things actually stand, what's been tried that didn't work and why, and what discipline carries forward.
>
> Phase 5 ran from approximately 2026-04-23 (Phase 5a kickoff) through 2026-05-02 (verification + altitude expansion specs). It produced one shippable deliverable (the LEO low-e specialist), substantial methodological infrastructure, and four explicit closure-retraction cycles before the actual state stabilized.
>
> **Last updated:** 2026-05-02.

---

## 1. The deliverable that actually stands

### 1.1 What's real

**A two-body LEO orbital rendezvous agent.** Trained on the Phase 4 R4 recipe plus Phase 5b's random satellite eccentricity initialization, via the two-stage curriculum Stage 1.0 → Stage 4.0. At LEO altitudes (300-800 km) with sat & target eccentricity ≤ 0.05 fully random, the agent achieves 96.4% multi-seed mean success on Phase 5b's deliverable ckpt (`pufferlib/experiments/puffer_orbital_177750405236/model_puffer_orbital_000350.pt`, seed 31415, 97.7% multi-rollout). 3 of 5 training seeds reached this performance; 2 seeds collapsed to 2-16% (the recipe is bimodal at the edge — a known fragility).

The agent handles arbitrary phase gaps (30° through 180°) within this regime; the phase axis is fully generalized. The eccentricity axis is the binding difficulty.

### 1.2 What's NOT real (despite earlier claims)

Multiple Phase 5 phases produced headline numbers that did not survive scrutiny. These should be treated as superseded:

- **Phase 5e's "90.2% multi-seed at e_max=0.20."** True as a measurement, misleading as a framing. At LEO with `valid_init_only=1`, eccentricity is geometrically constrained to e ≤ 0.084. The "e_max=0.20" eval distribution had realized e mean of 0.028 because the rejection sampler rejected most high-e samples. The number measures performance on a near-circular distribution wearing a high-e label.
- **Phase 5e's "84.5% at e_max=0.50" and "64.5% at e_max=0.70."** Contaminated by 256-attempt-cap exhaustion. At e_max=0.70 in LEO, 31.4% of resets exhausted the cap and accepted physically impossible orbits (sat or target perigee below Earth's surface). Doomed inits never succeed, so they drag the success rate down by ~25pp.
- **The wrap-up's "cliff at e ≥ 0.075."** Confounded with altitude OOD. The wrap-up's surface used altitude overrides to make high-e orbits physically valid, which pushed eval altitudes outside the recipe's trained LEO band. The "cliff" was altitude generalization failure, not eccentricity capability failure.
- **The wrap-up's "LEO low-e specialist with eccentricity cliff."** Partially right (LEO specialist is correct as a training-distribution statement) but partially misleading (the eccentricity capability within LEO is not a cliff — pass-only success at e=0.70 in LEO is 97.1% per the verification).

### 1.3 The honest framing

The recipe is a LEO 300-800 km specialist. Within the LEO altitude band, the recipe handles eccentricities up to the geometric limit (e ≈ 0.08 at LEO altitudes). The eccentricity ceiling in subsequent stages (e_max=0.10 trained, e_max=0.20 attempted) is real, but its meaning is constrained by the LEO physics:

- Stage 4.1 (e_max=0.10): 65% multi-seed. Genuine improvement on Phase 5b's Stage 4.0.
- Stage 4.2 (e_max=0.20): 0-2% across all seeds, training collapses. Plausibly because most e=0.20 random samples in LEO are physically infeasible, making terminal rewards unreachable.

Extending the recipe to handle higher eccentricities (the original Phase 5 ambition) requires altitude expansion, which is now scoped in `phase5-5-altitude-expansion-spec.md` and is not yet done.

---

## 2. Project history through Phase 5

### 2.1 Entering Phase 5 from Phase 4

Phase 4 shipped at 79.6% multi-seed mean at 180° rendezvous on circular orbits. The R4 recipe consisted of: `Discrete(10)` actions including a 5-minute time-warp action, gated NHR shaping with terminal Φ-clamp, LVLH-frame observations, and a three-stage curriculum on phase gap (30° → 90° → 180°).

Phase 4.5 attribution ablations established that LVLH observations and the shaping were load-bearing. DAPO was inert. Cross-phase action-table surgery (D7→D10) didn't transfer.

Going into Phase 5, the plan was to extend the recipe to eccentric target orbits (and eventually rendezvous in cislunar space). The original Phase 5 spec proposed a joint (phase × eccentricity) curriculum with structured intermediate stages.

### 2.2 Phase 5a — entry condition investigation (paused mid-execution)

Phase 5a was designed to test pre-curriculum questions: does the Phase 4 ckpt generalize to e>0? Does smaller eccentricity steps work? What's the right curriculum ordering (square vs tall vs wide)?

Phase 5a halted within the first investigation when the user noticed bizarre training-time variance: at fixed seed 42, identical config, byte-identical code, the same Stage 1 produced 5%, 40%, and 77% final perf across three separate runs. This led to:

#### 2.2.1 The determinism finding (PHASE5a_DETERMINISM_FINDING.md)

**Root cause:** PufferLib's vec backend (`backend=Multiprocessing` in default.ini) forks worker processes. The order in which `vecenv.recv()` returns step results depends on OS scheduling. PPO's minibatch ordering and gradient noise are sensitive to this order at fixed seed.

**Implications:**
- Single-seed comparisons cannot reliably distinguish curriculum orderings differing by ~5-10pp; the noise floor is ~30-70pp at fixed seed.
- Phase 4's "Stage 1 has ~1/3 retry rate" was reframed: the seed isn't determinative, the worker scheduling is.
- Phase 4.5 attribution claims based on single-point comparisons (e.g., LVLH-off 13% vs LVLH-on 77%) have wide error bars. Direction is probably right, magnitude has uncertainty.

**Decision:** All subsequent Phase 5 work uses 3-5 seeds minimum for any structural claim, with ≥10pp gap required to call a winner.

#### 2.2.2 The A2 peak diagnostic (PHASE5a_A2_DIAGNOSTIC_FINDINGS.md)

The original A2 probe reported "88.9% peak training perf at e=0.05" — looked like the Phase 4 recipe partially generalized. Investigation:

- Tight checkpoint interval (every 5 epochs from 5 to 75) saved 16 ckpts of the A2 run.
- All 16 ckpts evaluated at 0% argmax / ~1% sampled at e=0.05.
- The 88.9% training-time peak was **rolling-window burst luck** in PufferLib's dashboard `perf` metric. The dashboard averages over a small window of completed episodes; at low success rates, an unlucky run of easy e_target draws (e ∈ [0, 0.005]) produces a transient spike.

**This was the first instance of the "metric measures something different from its name" pattern that would recur throughout Phase 5.** The lesson: never trust dashboard `perf` for capability claims at low success rates. Use `eval_checkpoint.py` (deterministic argmax, large sample) for any decision.

**Verdict:** Phase 4 recipe doesn't transfer to e>0. The Phase 5b structural fix is the next step.

### 2.3 Phase 5b — random sat init + two-stage curriculum (the working deliverable)

Phase 5b's audit identified the failure mechanism at e>0: terminal +10 is geometrically unreachable for a circular chaser against an eccentric target without first transferring to an eccentric orbit. Random-init policies never attempt this maneuver because there's no gradient signal pulling them toward it.

#### 2.3.1 The audit (PHASE5b_AUDIT_FINDINGS.md)

Four audits ran on the Phase 4 ckpt evaluated at e ∈ {0, 0.05, 0.10}:

1. **Perf metric trace.** Confirmed M1 (rolling-window burst luck), not M2 (clamp bug). Use `eval_checkpoint.py` not dashboard.

2. **Gate-activation rates.** σ₂ (orbit-shape gate) drops slightly at e>0 (24% → 17% at e=0.05). **σ₃ (velocity-gate) is structurally dead even at e=0** (1.6% activation). The recipe is de facto single-component (Φ_orbit only) with σ₂ as the only meaningful gate. Phase 4 worked anyway.

3. **Goodhart ratio.** Median 0.114 at e=0.05 (above Phase 4 plan's 0.1 threshold). Mild Goodharting at e>0.

4. **Termination-mode histogram.** **93% of failures at e=0.05 are safety-cap timeouts.** The agent isn't doing anything destructive; it's running out the clock without achieving rendezvous.

**Verdict:** "Reward reachability" — the agent has fuel and capability; what it lacks is gradient signal toward the eccentric target's orbit shape. The structural fix is either REL_VEL_TOL annealing or random sat init.

#### 2.3.2 Step 1 — the working bootstrap (PHASE5b_STEP1_FINDINGS.md)

Implementation: random satellite eccentricity initialization. Sample `sat.e ~ Uniform(0, e_max_sat)` and `sat.omega ~ Uniform(0, 2π)` at reset, where the env defaults to using `e_max_target` for `e_max_sat` if not explicitly set.

Stage 1.0 at e_max=0.05 same_orbit_init=1: 81.7% multi-rollout mean across 3 seeds. Phase 4 conditions OOD: 24.7% mean. Significantly broader than Phase 4's Stage 1 (~50% Phase 4 OOD).

#### 2.3.3 Cap-tail analysis (PHASE5b_STEP1_CAP_TAIL.md) — the methodological discipline arc

This is the most important methodological document in the project. It's a 424-line tour through what diagnostic discipline looks like when applied rigorously.

**Initial observation:** Stage 1.0 lands 84.5% success / 15.5% safety-cap. 

**Question:** Are the cap failures clustered on hard combos, or distributed uniformly?

**Finding:** (high-phase, low-e) is harder than (high-phase, high-e). The agent uses **natural phasing on eccentric orbits** — exploiting non-uniform angular velocity at apoapsis/periapsis to wait for phase closing. This is an emergent strategy that wasn't designed in. Free positive finding.

**Then came three rounds of mechanism speculation that all dissolved on contact with longer training:**

- **P1 (under-sampled OOD):** "Policy never explored these states." Refuted by sampled eval at the same conditions giving comparable failure rates.
- **P2 (learned to not try):** "Policy explored, then mode-collapsed to passivity." Refuted by intermediate-ckpt eval showing burn frequencies don't monotonically decline.
- **P3 (argmax brittleness via efficiency convergence):** "Policy converges to a sharply-peaked argmax that doesn't cover every corner state." Sophisticated story with multiple supporting probes.

Then the train-longer null was run: take the Stage 1.0 ckpt, train 20M more steps. **Cap-tail success rate climbed from 84.5% to 99.7%.** All three mechanism stories collapsed simultaneously.

**The right diagnosis was the simplest one available: under-converged training.** The cap-tail wasn't a structural failure mode; it was the policy still converging.

**The lesson, named explicitly in the doc:** *"Before declaring a structural failure mode, run the train-longer baseline. It's the cheapest control experiment in this project (~3 min wall) and resolves a wide class of 'is this a recipe bug or just under-trained?' questions in one shot. The cost of skipping it was three layers of mechanism speculation that the data immediately overturned."*

The cap-tail's P1/P2/P3 arc is the template for what would later be the explicit methodological discipline.

#### 2.3.4 The two-stage curriculum (PHASE5b_FINDINGS.md)

Block A ran the Stage 1.x curriculum: Stage 1.0 (e=0.05) at 99.6%, Stage 1.1 (e=0.10) at 86.7%, Stage 1.2 (e=0.20) at 42% (plateaued, +20M worsened to 34%), Stage 1.3 (e=0.5) at 55.2% best-per-seed.

Block B was the critical "is curriculum required?" test. Three arms at Stage 4.0 (random sat + random target, e_max=0.05):
- Fresh training (no warm-start): 0% mean.
- Warm from Stage 1.3 (specialist at e=0.5): 1.2% mean.
- Warm from Stage 1.0 (broader at e=0.05): **61.2% mean (3/5 at 94-98%)**.

This is the cleanest "curriculum is required" finding the project produced. Fresh fails to bootstrap. Warm-from-specialist also fails. Only warm-from-broader works.

**Key finding: specialization erodes breadth.** Direct OOD eval at Stage 4.0 conditions:
- Stage 1.0 (e=0.05 same-orbit, 99.6% in-distribution) → 43.6% Stage 4.0 OOD
- Stage 1.3 (e=0.5 same-orbit, 55.2% in-distribution) → **5.2% Stage 4.0 OOD**

Despite Stage 1.3 being "harder" in-distribution, its policy is *worse* for transfer. The intermediate Stage 1.x stages from the original spec actively hurt subsequent transfer to Stage 4.0. The spec's four-stage curriculum was over-engineered; **the two-stage curriculum (Stage 1.0 → Stage 4.0 directly) is the right design.**

**The lesson:** *"For curriculum learning, train each stage just enough to give a useful warm-start, not to convergence on that stage's own metric. The 'early peak' ckpt may be a better warm-start than the converged ckpt."* — this hypothesis was identified but not directly tested in Phase 5b.

Block C extended the recipe to higher eccentricities:
- Stage 4.1 (e=0.10): 65.3% multi-seed. Real extension.
- Stage 4.2 (e=0.20): 0/0/2% across 3 seeds. Training collapses.
- Stage 4.3 (e=0.30, retry from 4.1): 0%.

The recipe ceiling in the fully-random regime: e ≈ 0.10-0.15.

Block D produced a 24-cell reduced capability surface (phase × e_max). **Phase axis is fully generalized (within 5pp across 30°/90°/180°). Eccentricity is the binding difficulty.**

**Per-seed bimodality:** 3/5 seeds at 94-98%, 2/5 at 2-16%. The collapsed seeds peak at epoch 25 (warm-start preservation only). Same recipe, same warm-start, different RNG → different outcomes. This is consistent with the Phase 5a determinism finding at much larger scale.

#### 2.3.5 Phase 5b's deliverable

`pufferlib/experiments/puffer_orbital_177750405236/model_puffer_orbital_000350.pt` (seed 31415, Stage 4.0, 350 epochs). 96.4% multi-rollout mean at e_max=0.05 fully random. **This is the deliverable that survives all subsequent scrutiny.**

### 2.4 Phase 5c — the "ceiling investigation" that wasn't (50+ hours of compute)

Phase 5b's Stage 4.2 collapse at e=0.20 motivated Phase 5c: investigate the e=0.20 ceiling, run targeted interventions to crack it.

#### 2.4.1 The Goodhart prediction tests (PHASE5b_PREDICTIONS.md)

Three predictions about the Stage 4.2 collapse mechanism:

**P1 (action distribution):** predicted coast/warp ↑, burn ↓, Δv ↓ at collapsed ckpt. **Mixed result.** Δv dropped (150 → 104 m/s) but burn frequency rose (10.4% → 13.1%). The collapsed policy does more, smaller burns — not "stops trying." This contradicted the simple "policy gives up" hypothesis.

**P2 (gate activation):** predicted σ₂ activation should drop. **Confirmed.** σ₂ 8.1% → 4.4%, σ₃ 7.5% → 0.0%, Φ_orbit grew (19.7 → 23.8). **But:** Goodhart ratio at collapsed ckpt was 0.226 (73% of eps over 0.1). **The shaping reward was the dominant signal, not the terminal failure reward.**

**P3 (global degradation):** predicted training at e=0.20 should degrade the policy globally. **Confirmed at extreme magnitude.** Collapsed Stage 4.2 ckpt evaluated at e=0.05 fully random: 0%. The warm-start (Stage 4.0 best) was 99% at e=0.05.

**Mechanism story at this stage:** "Goodhart-induced collapse" — shaping accumulates over long failure trajectories, becomes dominant over terminal reward, policy optimizes for shaping rather than rendezvous, collapses.

#### 2.4.2 Phase 5c Block I diagnostic (PHASE5c_FINDINGS.md)

Detailed per-step shaping decomposition (A1-A11). Key finding:

**A1+A2 — Cumulative shaping at e=0.20:**
- Successes: -0.59 (γ=0.995) / -0.12 (γ=1)
- Failures: **-0.07 (γ=0.995) / 0.00 (γ=1)**

**Failures accumulate less negative cumulative shaping than successes.** Direction "reversed" under both γ values.

**A6 — Phase 4 baseline contrast (e=0):**
- Success cum r: -0.80
- Failure cum r: -3.04 (much more negative — direction correct)

Mechanism story refined: **at e=0, failures have high Φ_orbit + short trajectories → strong negative drag → terminal -10 is dominant → recipe works.** At e=0.20, failures cap-timeout at 2000 steps with modest Φ → weakened drag → direction flips.

The recipe's apparent functioning at e=0 was load-bearingly dependent on a γ-discount-on-bounded-Φ asymmetry that doesn't survive at high e.

#### 2.4.3 The Block II intervention sweep

Eight reward-side interventions tested:

| Intervention | Held-out @ e=0.20 | Notes |
|---|---|---|
| **B3 — mixed-distribution (70% e≤0.05 + 30% e≤0.20)** | **18%** | Best result, but below 30% threshold |
| B3+B4 combined | 14% | Combination hurt slightly |
| B3 long (100M instead of 50M) | 14% | Train-longer didn't help |
| B4 — larger Δv (Discrete(14)) | 0% | Training peak 25%, collapsed |
| B5 — higher γ (0.998) | 0% | Training peak 26%, collapsed |
| B6 — REL_VEL_TOL anneal | 0% | Stage 1 → 6%, stage 2 → 0% |
| B1 — γ=1 shaping | SKIPPED | Predicted not to help by A1 |
| B2 — continuous gates | SKIPPED | Orthogonal to direction-inversion |

**All <30%.** Per spec decision rule: "Block I diagnostic was insufficient; return to investigation." But it was the *intervention space* that was insufficient, not the diagnostic — all 8 were in the reward-side class.

#### 2.4.4 The Simpson's paradox catch (PHASE5c_CORRECTIONS.md)

After the spec was written, a follow-up probe length-binned the cumulative shaping data:

| Length bin | n_succ | succ cum_r | n_fail | fail cum_r | Direction |
|---|---|---|---|---|---|
| [0, 100) | 2 | -0.05 | 83 | -0.02 | ✓ correct |
| [100, 300) | 11 | -0.20 | 3 | -0.08 | ✓ correct |
| [300, 700) | 6 | -0.59 | 2 | -0.35 | ✓ correct |
| [700, 2000) | 13 | -1.44 | 10 | -0.89 | ✓ correct |

**Within each length bin, successes have MORE negative cumulative shaping than failures. The direction is NOT reversed.** The original A1 "direction reversed" finding was a Simpson's paradox: 83 short failures (small drag) and 13 long successes (large drag), with population medians flipped by the mixed length distribution.

The Phase 5c mechanism story ("direction inversion at high e via γ-discount-on-bounded-Φ") was wrong. The corrected story:
- 46% of failures terminate **early** (length ~54) with high Φ ≥ 20 — escape/collision/stranded events.
- 17% are **near-miss** (Φ_T ≤ 5, length ≥ 700) — close to target but missed simultaneously.
- 37% are **mid** — somewhere between.

The interventions tested (B1-B6) address mostly the near-miss failure mode. They don't address the early-death failure mode, which dominates.

**Retracted Phase 5c mechanism story:** the "direction inversion" was an aggregation artifact. The real failure mode is early-death from catastrophic actions at high e.

**The lesson:** *"The original A1 result was a textbook Simpson's paradox. Always bin by length AND outcome before claiming a population-level relationship."*

### 2.5 Phase 5d — early-death investigation, 2/7 interventions before pause

Block I diagnosed the early-death pattern: 70% of failures at e=0.20 are Earth-collision via perigee lowering. The agent makes a burn that lowers its orbit's perigee below EARTH_KEEPOUT, then the propagation hits Earth.

Block II tested I4 (collision soft penalty):
- I4 alone (w=1.0): held-out 3% at epoch 5 (warm-start preservation only), then 0%.
- I4 + B3 (w=0.1): 12% at epoch 5, declining.

**Same collapse pattern as Phase 5c's reward-side interventions.** The "penalize burns that cause collisions" intervention had a hidden flaw: collision is correlated with task difficulty (high-e tasks need risky burns; low-e tasks don't). PPO learned "avoid penalties = avoid hard tasks," producing global capability degradation rather than targeted collision avoidance.

Phase 5d paused at 2/7 interventions with the recommendation to either:
- **Path A:** Engineer hard action masking (block catastrophic actions at the logit level).
- **Path B:** Close Phase 5 with documented ceiling.

But the user pushed back on premature closure. The conclusion that "the recipe ceilings at e=0.10-0.15" was based on 8+ failed interventions, all in one class (reward-side). The structural conclusion needed at least one intervention from a different class to be supported.

### 2.6 Phase 5e — env validation finds the real issue

Pivoting to env-level investigation rather than more reward interventions.

#### 2.6.1 Block I env validation (PHASE5e_BLOCK_I_FINDINGS.md)

Six probes at high eccentricity:

**E1 — Lambert reachability check (the big one):** at e_max=0.20 in LEO with the default reset distribution, **86% of randomly-sampled inits have perigee `a(1-e)` below R_EARTH (sub-surface)**. 94% have perigee below EARTH_KEEPOUT.

This was the env bug Phase 5c/5d had been searching for. The "wall" at e=0.20 wasn't a recipe ceiling. It was the eval distribution being 94% physically impossible. The agent literally couldn't succeed because most tasks started inside the planet.

**E2-E6:** Kepler precision, round-trip conversion, LVLH frame, Φ_orbit calibration, action effects at peri vs apo — all passed at high eccentricity. The env was correct in mechanics, broken in sampling.

**The fix:** add `valid_init_only=1` kwarg with rejection sampling on perigee. Reject (sat, target) draws where either perigee < EARTH_KEEPOUT, up to 256 attempts.

#### 2.6.2 Block II — the apparent triumph (PHASE5e_BLOCK_II_FINDINGS.md)

5-seed retrain with `valid_init_only=1`:
- e=0.20: **90.2% ± 2.1% multi-seed mean.** No collapse seeds.
- e=0.30 single-seed: 93%.
- e=0.50 single-seed: 84.5%.
- e=0.70 single-seed: 64.5%.

The "wall" Phase 5c/5d spent ~50 hours of compute investigating turned out to be a curriculum-sampling bug. One conditional in `c_reset` resolves it.

**This was framed as Phase 5 Closure A. It was premature.**

#### 2.6.3 The wrap-up (PHASE5_FINDINGS.md — first version)

The Phase 5e closure motivated a wrap-up that aimed to:
1. Multi-seed the full e-scan (II.D was deferred from Block II)
2. Produce a comprehensive capability surface across (phase × e_target × e_sat × sat-target relation)

The wrap-up agent ran the full grid but with significant scope corrections:
- 4 phase × 6 e (with sat=target conflated to diagonal) × 2 relation = 48 cells/ckpt
- 100 eps × 1 rollout seed instead of spec's 200 × 3
- For high-e cells where LEO altitude can't admit e geometrically, used **altitude overrides** (a_min_override, a_max_override) scaled by `alt_band_for_e()` to push altitudes above the keepout requirement.

The surface showed:
- e=0.025 LEO: 97% (peak)
- e=0.05 LEO: 85%
- e=0.075 LEO: 14% (cliff)
- e=0.10 LEO: 2%
- e=0.20+ surface (with altitude override): 0% across phases

**Wrap-up's headline:** "LEO low-e specialist with cliff at e ≥ 0.075." Phase 5e's e ≤ 0.50 framing retracted as aggregate-distribution artifact.

### 2.7 The verification investigation (PHASE5_VERIFICATION_FINDINGS.md)

The user observed trajectory files on the web frontend appearing to start with sat or target inside Earth's surface — despite metadata tagging `valid_init_only=1`. This triggered the verification spec.

#### 2.7.1 I1 — Does valid_init_only fire?

Added instrumentation to `c_reset`. Three instrumented evals on the canonical Phase 5e seed-42 ckpt:

**Eval 1 (e_max=0.20, default LEO):** 0/51 cap exhausts. Realized sat_e mean 0.028 vs expected 0.10. **Filter fires correctly, but realized distribution is heavily biased to low e because the rejection sampler filters out most high-e samples at LEO altitudes.**

**Eval 2 (e_max=0.70, default LEO):** 16/51 (31.4%) cap exhausts. 12/51 accepted sat_rp < R_EARTH. The filter is overwhelmed at high e in LEO; the 256-attempt cap exhausts and the env accepts doomed inits.

**Eval 3 (e=0.05 fixed, alt-band override 6917-11759 km):** Filter clean, 0 cap exhausts. Eval success 1/50 = 2%. **The wrap-up's "5.8% at e=0.05 fully_random" was an altitude OOD result, not an eccentricity capability result** — the recipe was being evaluated at altitudes outside its training band.

#### 2.7.2 I2 — Trajectory file audit

200 trajectories from `web_data/runs/phase5e_seed42_e0.{05,20,50,70}/`. Found:
- e0.05, e0.20: all 100 trajectories Pass (perigees ≥ EARTH_KEEPOUT)
- e0.50: 5/49 Failed (sat or target perigee below surface)
- e0.70: 12/46 Failed

**Pass-only success at e=0.70 in LEO: 97.1%.** The published 71.7% headline was a 25.4pp env-bug artifact from cap-exhaust doomed inits.

The trajectory files were correctly metadata-tagged `valid_init_only=1` because that recorded the *kwarg*, not the *outcome*. The export script wrote intent, not reality.

#### 2.7.3 I4 — What was actually trained

Read the curriculum scripts and env code:
- Stage 1.0 and Stage 4.0 both trained at LEO 300-800 km altitude band (hardcoded in orbital.h:842 with sentinel `a_min_override=-1.0` not triggering override).
- e_max=0.05 sat and target.
- No altitude expansion ever happened.

**The recipe is genuinely a LEO 300-800 km specialist at e ≤ 0.05.** The user's prior assumption that high-altitude eccentric orbits were being trained was incorrect.

#### 2.7.4 The verification's synthesis

The wrap-up's framing had two confounds:
1. **The 256-cap exhaustion** at high-e in LEO produced doomed inits that the eval scored as failures.
2. **The altitude override** at high-e cells pushed evaluation outside the trained LEO band, conflating eccentricity capability with altitude generalization.

Phase 5b's e ≤ 0.05 LEO deliverable stands. Phase 5e's headline numbers and the wrap-up's "cliff" framing both had material framing problems.

---

## 3. What's actually known vs claimed (the matrix)

| Claim | Status | Evidence |
|---|---|---|
| Phase 4 R4 recipe works at e=0 in LEO | **TRUE** | 79.6% multi-seed at 180° rendezvous; multi-seed validated |
| Phase 4 recipe doesn't transfer to e>0 | **TRUE** | A1 at e=0.05 → 1.5%; A2 at e=0.05 → 0% (Phase 5a) |
| Random sat init + two-stage curriculum works at e_max=0.05 LEO | **TRUE** | Phase 5b: 96.4% multi-seed (3/5 successful seeds, 2/5 collapse) |
| The recipe is bimodal at the e=0.05 LEO edge | **TRUE** | 3/5 seeds at 94-98%, 2/5 at 2-16% |
| Curriculum is required (not just helpful) | **TRUE** | Block B: fresh 0%, warm-from-1.3 1.2%, warm-from-1.0 61.2% |
| Intermediate Stage 1.x stages erode breadth | **TRUE** | Stage 1.0 → 43.6% Stage 4 OOD vs Stage 1.3 → 5.2% |
| Phase generalization is essentially free | **TRUE** | Phase axis flat within 5pp across 30°/90°/180° |
| The recipe ceilings at e=0.10-0.15 in LEO fully-random | **APPROX TRUE** | Stage 4.1 65%, Stage 4.2 0-2%. But qualified: at LEO, e is physically bounded to ~0.08, so "ceiling at 0.10-0.15" is partly a physics artifact |
| Phase 5e's "90.2% at e_max=0.20" | **MISLEADING** | Real measurement; eval distribution had realized e mean 0.028 due to LEO physics |
| Phase 5e's "84% at e_max=0.50" | **CONTAMINATED** | 25pp+ of failure is doomed inits from 256-cap exhaustion |
| Pass-only success at e=0.70 LEO is 97.1% | **TRUE** | Verification I2 audit of trajectory files |
| The wrap-up's "cliff at e≥0.075" | **CONFOUNDED** | Surface used altitude overrides for high-e cells; cliff is altitude OOD |
| Recipe trained at high-altitude eccentric orbits | **FALSE** | I4 confirmed: trained at LEO 300-800 km only |
| The recipe could extend to higher altitudes with retraining | **UNKNOWN** | Never tested. Phase 5.5 spec scopes this |

---

## 4. What's been tried and what worked

### 4.1 Things that worked

**Random satellite eccentricity initialization.** The single most important env change Phase 5b made. Replaces the hardcoded `sat.orbit.e = 0` with `Uniform(0, e_max_sat)`. Phase 5b's audit identified this as the structural fix; Phase 5b Step 1 confirmed it worked.

**The two-stage curriculum (Stage 1.0 → Stage 4.0 directly).** Skipping intermediate Stage 1.x stages. Phase 5b Block B established this as the working design.

**Stratified shaping with terminal Φ-clamp from Phase 4.** The recipe inherited from Phase 4 worked at e=0 and survives all of Phase 5. The gated NHR shaping is mostly fine despite σ₃ being structurally dead (it doesn't fire, but it doesn't actively hurt either).

**Tight checkpoint intervals + held-out eval at every ckpt.** Phase 5b Step 1's cap-tail discovered this. Picking the best ckpt across training (not just the final) is the difference between observing "training collapsed" and "training had a high-quality peak."

**The train-longer null.** Cap-tail's P3 mechanism story dissolved when this was finally run. ~3 minutes of compute resolved three rounds of mechanism speculation.

**Length-binning before aggregation.** Phase 5c corrections caught the Simpson's paradox via length-binning. Should be standard before any population-level claim.

**Multi-seed before structural claims.** Phase 5a's determinism finding established this. The fragility (3/5 success vs 2/5 collapse at the recipe edge) is itself informative.

**Block B's three-arm comparison.** The "fresh / warm-from-intermediate / warm-from-broad" comparison is the cleanest curriculum-justification finding the project produced. Stronger than "warm-start helps" — establishes whether curriculum is required.

**Env validation at every regime.** Phase 5e Block I (Lambert reachability + the six probes) found the actual issue in 1.5 hours of post-hoc analysis. Should have been done at the start of Phase 5c.

### 4.2 Things that didn't work (and why)

**Reward-side interventions for the e=0.20 wall.** Phase 5c/5d tested 8+ interventions: γ=1 shaping, continuous gates, mixed-distribution, larger Δv, higher γ, REL_VEL_TOL anneal, collision soft penalty, I4+B3 combined. All collapsed or barely moved the needle (best was B3 at 18% mean). Reason: the wall wasn't a recipe ceiling at all — it was the env producing physically impossible tasks. Reward-side interventions can't fix physical impossibility.

**Continuous actions (Box) instead of discrete.** Tested early in the project. Collapsed at 24% greedy / 33% sampled at e=0.10 after 15M steps. Broader action space dilutes random-policy productive rate; PPO can't bootstrap. Reverted to Discrete.

**Discrete(14) action expansion.** Phase 5c B4 tested adding ±3 m/s fine actions. Training peak 25%, held-out 0%. Same dilution issue as continuous. Reverted to Discrete(10).

**DT=30 timestep (halved).** Earlier project work tested this for finer ω timing. Collapsed at all eccentricities; doubling episode length halves terminal-reward density per training step. Reverted to DT=60.

**Phase 5a's premise that single-seed comparisons are sufficient.** Determinism finding invalidated this. Three runs at fixed seed gave 5%, 40%, 77% final perf.

**Direct fresh training at Stage 4.0.** Phase 5b Block B showed 0%. Fresh training at the random sat+target distribution can't bootstrap.

**Direct fresh training at higher eccentricities.** Phase 5b Stage 4.2 (e=0.20) collapsed to 0/0/2% across 3 seeds. The "harder task with random sat init" requires a warm-start from somewhere.

**Intermediate Stage 1.x curriculum.** Phase 5b Block A confirmed that each intermediate stage erodes Stage 4.0 transfer. The original spec's 4-stage curriculum was over-engineered.

**Phase 5c's "Goodhart-induced collapse" mechanism story.** Was wrong. Length-binning revealed Simpson's paradox.

**Phase 5d's "recipe ceilings at e=0.10-0.15" claim.** Was at minimum misleading. The e=0.20 collapse in training was likely caused by env producing doomed inits (sub-keepout perigees), not a recipe property.

**The wrap-up's "LEO low-e specialist with cliff at e ≥ 0.075" framing.** Cliff was altitude OOD, not eccentricity capability.

**Phase 5e's "Closure A confirmed" framing.** Was premature. The verification revealed multiple confounds in the headline numbers.

### 4.3 Things that should have been tried but weren't

**Action masking at the logit level (Path A from Phase 5d).** Block invalid actions before sampling. This is a policy-level intervention, structurally different from the reward-side interventions that all failed. Was deferred and never tested. Infrastructure exists in the codebase (`enable_action_mask`) but is off by default.

**Demonstration bootstrapping.** Generate analytical Hohmann-plus-correction trajectories offline; warm-start training on these before exposing to random rollouts. Was deferred from Phase 4 and continues to be deferred.

**Bound-expansion curriculum as continuous schedule.** Phase 5e Block II proposed continuous bounds on (Δa, Δω) instead of discrete Stage 1/4 transitions. Designed but never implemented.

**Mid-training ckpt as warm-start (Phase 5b's open hypothesis).** The "early peak ckpt may be a better warm-start than converged ckpt" hypothesis. Identified but not directly tested.

**Env validation at altitude.** All Phase 5e Block I probes were at the LEO altitude band. The altitude expansion question wasn't probed until the wrap-up — and that probe was contaminated by the altitude/eccentricity conflation.

---

## 5. Issues in implementation, interpretation, and methodology

### 5.1 The metric-vs-implementation pattern (six instances)

The recurring failure mode: a metric's name implied one thing, the implementation measured something different. Every Phase 5 closure-and-retraction cycle had at least one instance.

1. **Dashboard `perf` looked like success rate.** Was rolling-window noise dominating at low signal. The 88.9% A2 peak that turned out to be ~1% held-out.

2. **Cumulative shaping aggregated across episodes.** Looked like a population property. Was Simpson's-paradox-confounded by length. The "direction reversed" finding inverted under proper binning.

3. **`e_max_target` sampling.** Looked like "tests at e=X." Was `Uniform(0, X) ∩ valid`. At LEO with `valid_init_only=1`, realized e mean is far below e_max because the rejection sampler clips high-e samples.

4. **`valid_init_only=1` kwarg.** Looked like a guaranteed filter. Filtered up to 256 attempts then accepted whatever; at high-e LEO, accepted doomed inits at 31% rate.

5. **Trajectory metadata `env_config.valid_init_only`.** Looked like outcome. Was the kwarg value (intent). Said `valid_init_only: 1` even when the cap exhausted and accepted a doomed init.

6. **Diagonal-only capability surface.** Looked like a 4D grid (phase × e_target × e_sat × relation). Was actually 3D (phase × e × relation) with `e_target = e_sat` conflated.

**The transferable principle:** when a name implies a specific value or guarantee, assume it's potentially misleading until verified. Read the implementation. Compare to the name. Document the gap. This is closer to type-checking-for-experiments than to traditional ML methodology.

### 5.2 The mechanism-story overcommitment pattern

Every Phase 5 phase produced a mechanism story that "explained the data" before orthogonal evidence was available:

- Phase 5b cap-tail: P1 (under-sampled OOD), then P2 (mode-collapsed to passivity), then P3 (argmax brittleness via efficiency convergence). All three dissolved when the train-longer null was run.
- Phase 5b post-extend: "60M shows erosion." Was a single-seed borderline data point. 80M training showed no erosion.
- Phase 5c: "shaping direction reversed at high e via γ-discount-on-bounded-Φ." Was Simpson's paradox.
- Phase 5d: "recipe ceilings at e=0.10-0.15." Was a confound with env distribution.
- Phase 5e: "recipe handles e ≤ 0.50 with graceful degradation." Was a confound with realized eccentricity distribution.
- Wrap-up: "LEO low-e specialist with eccentricity cliff." Was a confound with altitude OOD.

**The pattern:** each mechanism story had a coherent narrative explaining specific observations. Each was internally consistent. Each was wrong because it didn't test orthogonal predictions.

**The discipline that catches this:** before committing to a mechanism story, identify a prediction it makes that's distinct from other plausible stories. Test that prediction. If multiple stories make the same predictions, more diagnostic work is needed before declaring a mechanism.

### 5.3 The scope-inheritance failure

The LEO 300-800 km altitude band was a Phase 4 design decision that was never reconsidered through Phase 5. Multiple closure-attempts assumed LEO without explicit acknowledgment that this was a scope decision. The user's stated assumption that high-altitude eccentric orbits were being trained was reasonable but incorrect — the recipe was inheriting LEO from Phase 4 silently.

**The discipline:** for any project that builds on previous work, explicitly enumerate the scope decisions being inherited at the start of each new phase. Re-evaluate them. Decide deliberately whether to extend or maintain them.

### 5.4 The "looks impossible — must be visualization" temptation

The user's observation that web frontend trajectories appeared to start inside Earth's surface could have been dismissed as a visualization artifact. It wasn't. The user pushed, and the verification confirmed the trajectories were genuinely impossible (cap-exhaust doomed inits).

**The discipline:** when a visual or interpretive issue is flagged, take it seriously. Verify the underlying data before assuming it's a presentation issue. "Looks wrong" is a real signal.

### 5.5 The corner-cutting pattern in agent work

Multiple Phase 5 wrap-up agents produced "complete" deliverables that, on close reading, had significantly reduced scope from what was specified:
- The wrap-up agent ran 48 cells instead of 256, 100 eps × 1 rollout seed instead of 200 × 3, with sat/target eccentricity conflated to the diagonal.
- The Phase 5e Block II "deferred" the multi-seed eccentricity scan that was the central validation.

**The discipline:** explicit verification of "what was actually run vs what was specified." Don't accept "I ran the surface" without confirming the dimensions, cell count, and per-cell statistics match the spec.

### 5.6 The "metadata records intent, not outcome" pattern

Trajectory export script wrote kwargs (intent), not the actual reset outcome. Curriculum scripts didn't record the altitude band used. Multiple post-hoc audits had to read the curriculum source code to determine what was actually trained, because the saved artifacts didn't capture it.

**The discipline:** metadata should record **what happened**, not what was requested. For training: log effective hyperparameters including resolved defaults. For env resets: log realized states (final accepted perigees, eccentricities, altitudes). For evals: log the actual cell parameters, not just the script that produced them.

---

## 6. The methodological discipline that emerged

Carried forward into Phase 5.5 and beyond:

### 6.1 Test the train-longer null first

Before declaring any stage stalled, run +20M steps. Multiple Phase 5b iterations confirmed this resolves "P1/P2/P3 mechanism" speculation. Cheapest control experiment.

### 6.2 Length-bin (or condition-bin) before aggregating

Any analysis aggregating across episodes must bin by relevant covariates first. Phase 5c's Simpson's paradox came from population-level aggregation hiding within-bin truth.

### 6.3 Verify metric vs implementation

Five-plus instances of named metrics measuring something different from what they implied. Always: read the code, compare to the name, document the gap.

### 6.4 Multi-seed before structural claims

5 seeds minimum for any "the recipe does X" claim. Single-seed results are screening signals at best. Phase 5b's bimodality (3/5 vs 2/5) is sufficient evidence that single-seed numbers are unreliable.

### 6.5 Per-seed variance is information

Report failure rate alongside mean. Bimodality at the recipe edge is data, not noise.

### 6.6 OOD-eval at every stage

Each stage's ckpt must be evaluated at Phase 4 conditions (no regression) and at the next stage's distribution (warm-start quality). Catches specialization-erosion early.

### 6.7 Save mid-training ckpts

Phase 5b's "early peak ckpt may be better warm-start than converged ckpt" hypothesis was identified but not tested. Save ckpts at 25%, 50%, 75%, 100% of each stage; eval at next-stage OOD; pick best for warm-start.

### 6.8 Trajectory inspection

For every stage, look at 3-5 successful and 3-5 failed trajectories. Mechanism stories should be visually verifiable, not just statistical.

### 6.9 Test the curriculum's necessity

For any new curriculum design, run the analog of Phase 5b Block B: fresh, warm-from-intermediate, warm-from-broader. Stronger than "curriculum helps" — establishes whether it's required.

### 6.10 Validate env at every new regime

Before training at conditions not previously evaluated, run the equivalent of Phase 5e Block I's six probes. Re-run Phase 1's e=0 validation tests at the new conditions. Catches env-distribution issues before they manifest as training collapses.

### 6.11 Mechanism stories require orthogonal evidence

Before committing to a mechanism story, identify a prediction it makes that's distinct from competing stories. Test that prediction. If multiple stories explain the same observations, more diagnostic work needed.

### 6.12 Scope decisions need explicit re-evaluation

For each new phase, enumerate inherited scope decisions. Decide deliberately whether to extend or maintain them. The LEO altitude band silently inherited from Phase 4 caused significant confusion in Phase 5e and the wrap-up.

---

## 7. The env state (as of 2026-05-02)

### 7.1 What's in the codebase

**Active recipe components:**
- Phase 4 R4 recipe: Discrete(10) actions including warp-5min, gated NHR shaping with terminal Φ-clamp, LVLH-frame observations, full π phase gap, 60s timestep.
- Phase 5b additions: random satellite eccentricity initialization (`sat.e ~ Uniform(0, e_max_sat)`, `sat.omega ~ Uniform(0, 2π)`).
- Phase 5e additions: `valid_init_only=1` kwarg with 256-attempt rejection sampling.

**Available but off-by-default:**
- `enable_action_mask` infrastructure (Phase 5d Path A, never validated).
- `collision_penalty_w` (Phase 5d, verified ineffective during training).
- `e_mix_easy_frac` / `e_mix_easy_max` (Phase 5c B3, partial improvement at 18%).
- `log_validation_debug` (verification investigation instrumentation).

**Reverted / never made canonical:**
- R3 components (DAPO, L2-init, TEC, adaptive KL) — explicitly excluded.
- Continuous action space (Box(2)) — Phase 5b ruled out.
- Discrete(14) action expansion — Phase 5c B4 collapsed.
- DT=30 timestep — collapsed at all eccentricities.

### 7.2 Known env issues (per verification)

1. **256-attempt cap in `c_reset`'s rejection sampling exhausts at high-e LEO.** Accepts doomed inits at ~31% rate at e_max=0.70.
2. **`metadata.env_config` in trajectory exports records kwargs, not outcomes.** Files say `valid_init_only: 1` even when the cap exhausted.
3. **Hardcoded LEO altitude band in orbital.h:842.** `a_min_override`/`a_max_override` kwargs exist but the override branch requires `a_min_override ≥ R_EARTH`; the default sentinel `-1.0` doesn't trigger.
4. **Φ_orbit normalization uses `ALT_MAX` constant.** May saturate at higher altitudes if extended.

### 7.3 What needs to happen next (per the env-fix spec)

The env-fix spec scopes three small fixes:
- **F1:** Raise cap to 4096 as a kwarg. Eliminates exhausts at all realistic configurations.
- **F2:** Add `realized_init` metadata to trajectory exports (sat/target perigees, gave_up flag).
- **F3:** Add `gave_up_action` kwarg to choose accept vs terminate behavior.
- **F4:** Keep `log_validation_debug` instrumentation.

~3 hours engineering. Prerequisite for altitude expansion (Phase 5.5).

---

## 8. Key artifacts (paths and what they are)

### 8.1 The deliverable ckpt

`pufferlib/experiments/puffer_orbital_177750405236/model_puffer_orbital_000350.pt`
- Phase 5b Stage 4.0 best ckpt, seed 31415, 350 epochs.
- 97.7% multi-rollout at e_max=0.05 fully random LEO.
- The headline two-body deliverable.

### 8.2 Alternative seeds (for multi-seed validation)

- `pufferlib/experiments/puffer_orbital_177750198246/model_puffer_orbital_000350.pt` (seed 42, 95.8%)
- `pufferlib/experiments/puffer_orbital_177750301624/model_puffer_orbital_000175.pt` (seed 20260423, 95.7%)

### 8.3 Stage 4.1 ckpts (e=0.10 partial extension, 65% multi-seed)

- `pufferlib/experiments/puffer_orbital_177750529227/model_puffer_orbital_000300.pt` (seed 42)

### 8.4 Phase 5e retrained ckpts (`valid_init_only=1`)

- Seed 42 canonical: `puffer_orbital_177765655537/model_puffer_orbital_000325.pt`
- Seed 42 wandb-tracked retrain: `models/phase5e/canonical_seed42_stage4_best.pt`
- 4 additional seed dirs: `puffer_orbital_177765658166`, `_177765658729`, `_177765659007`
- Note: 4 of the 5 seed dirs lost their explicit seed→dir mapping due to a race condition in the parallel orchestrator; results are valid but reproducibility from named seeds requires care.

### 8.5 Useful diagnostic scripts

- `scripts/orbital/eval_checkpoint.py` — standard eval script. Use this, not dashboard `perf`.
- `scripts/orbital/p5b_step1_cap_tail_analysis.py` — cap-tail analysis (binned by phase × e).
- `scripts/orbital/p5c_a1_shaping_decomp.py` — per-step shaping decomposition (with length-binning).
- `scripts/orbital/p5b_audit2_gate_activations.py` — Φ-port + activation/Goodhart analysis.
- `scripts/orbital/p5e_e1_lambert.py` — Lambert reachability check.
- `scripts/orbital/p5verify_perigee_scan.py` — verify perigees in trajectory files.

### 8.6 Web frontend data

`web_data/runs/phase5e_seed42_e0.{05,20,50,70}/`
- Per-cell trajectory JSONs from Phase 5e seed 42.
- **The e0.50 and e0.70 directories contain contaminated trajectories** (cap-exhaust doomed inits at 10-26% per directory). Should be regenerated after env-fix.

`web_data/results/`
- `multiseed_escan.csv` — per-(seed, e_max) success counts.
- `phase4_multiseed.csv` — Phase 4 conditions multi-seed.
- `seed42_training_progression.csv` — V-curve at epochs 5/25/50/100/150/200/250/325.
- `bonus_stats.json` — comprehensive action distribution, termination modes, fuel efficiency.

### 8.7 Documentation hierarchy

- `RECIPE.md` (in user's project) — final recipe doc.
- `PHASE5_FINDINGS.md` (in user's project) — current consolidated findings doc. **Note: has been retracted twice; the current version may still have framing issues per the verification.**
- `PHASE5_VERIFICATION_FINDINGS.md` — the most recent ground-truth document.
- `phase5-env-fix-spec.md` — next steps (env cleanup).
- `phase5-5-altitude-expansion-spec.md` — Phase 5.5 plan.
- This document — comprehensive retrospective.

---

## 9. Open questions for future engineering

### 9.1 Can the recipe extend to higher altitudes?

Unknown. Phase 5.5 is scoped to answer this. The first investigation (P1-P4 in the Phase 5.5 spec) probes env behavior at high altitude before any training; the curriculum then attempts bound-expansion stages.

### 9.2 At higher altitudes, can the recipe handle higher eccentricities?

Tied to 9.1. The hypothesis: at MEO altitudes (a ∈ 10000-15000 km), e=0.30 is physically valid (perigee ~7000+ km). If the recipe extends to MEO, the geometric constraint on e relaxes.

### 9.3 Is bimodality fixable?

3/5 seeds succeed, 2/5 collapse at the same recipe and warm-start. This is the recipe's main fragility. Hypotheses worth testing:
- Mid-training ckpt as warm-start may produce more robust results than converged ckpt.
- Higher entropy coefficient may prevent the "stuck on warm-start argmax" failure mode in the 2/5.
- Determinism backend (Serial vs Multiprocessing) might reveal whether it's seed-fundamental or scheduler-fundamental.

### 9.4 Does action masking work?

Path A from Phase 5d was deferred and never tested. The infrastructure exists (`enable_action_mask`). At minimum, worth a quick smoke test to see whether it changes the e=0.20 dynamics.

### 9.5 Does demonstration bootstrapping work?

Deferred since Phase 4. Generate analytical Hohmann-plus-correction trajectories offline; warm-start training on these. Substantively different intervention class from anything tried.

### 9.6 What's the actual capability surface at training conditions?

The wrap-up surface conflated altitude with eccentricity. The post-env-fix surface (Phase 5.5 Stage 5.5.0 or earlier) at fixed eccentricities within the physically-valid LEO range would tell us what the recipe genuinely does at its trained conditions. ~30 min compute.

### 9.7 Phase 4 numbers — are they similarly affected by aggregation?

Phase 4's "180° phase gap" used `init_phase_gap_max=π` (uniform [0, π]) — same pattern as `e_max`. The actual capability at exactly π is unknown; what's published is the aggregate over [0, π]. The Phase 4 baseline likely still stands directionally, but exact numbers would need re-eval at `phase_gap_fixed=π`.

---

## 10. The portfolio framing implications

### 10.1 Without altitude expansion (current state)

**Deliverable framing:** "Built a two-body LEO orbital rendezvous agent. At LEO altitudes (300-800 km) with eccentricities physically valid in that band (≤ 0.08), the agent handles arbitrary phase gaps and orbital relations at 96.4% multi-seed mean success (3/5 successful training seeds; 2/5 collapse). Documented the recipe's eccentricity ceiling and altitude scope explicitly."

**Strengths of the framing:**
- Honest about scope.
- Multi-seed validated.
- Recipe is reproducible and documented.

**Weaknesses:**
- LEO-only is narrower than the project's stated ambition (cislunar / any-orbit work).
- The eccentricity ceiling is partly a physics artifact (LEO can't admit high e), partly a recipe property — the framing has to handle this carefully.

### 10.2 With altitude expansion (Phase 5.5 success)

**Deliverable framing:** "Built a two-body orbital rendezvous agent that handles arbitrary phase gaps, eccentricities, and altitudes from LEO through GEO. Trained via curriculum learning with explicit attention to specialization-erosion trade-offs. Documented capability surface across the full envelope."

**Strengths:**
- Matches the project's natural ambition.
- Substantive engineering contribution at the curriculum design level.

**Weaknesses:**
- Phase 5.5 might not succeed at this scope. Plausible outcomes range from full success to documented limit at intermediate altitudes.

### 10.3 The methodology framing (always available)

**Deliverable framing:** "Built an orbital rendezvous agent with explicit attention to diagnostic discipline at the recipe edge. Found and fixed multiple env-level and metric-level bugs that had been hiding behind plausible-sounding mechanism stories. Documented the methodological infrastructure (train-longer null, length-binning, multi-seed validation, metric-vs-implementation verification, env-validation-at-every-regime) that produced the deliverable."

**This framing is robust regardless of whether altitude expansion succeeds.** The methodological discipline is itself the contribution. The agent is the artifact that demonstrates it.

This is probably the strongest portfolio piece available to the project right now. It's a story about how to build complex ML systems correctly, with the orbital rendezvous as the concrete example. The audience (space industry — APL, Lincoln Lab, Blue Origin) values diagnostic depth and engineering discipline; the methodology framing speaks to that directly.

---

## 11. A final honest note

Phase 5 took longer than it should have. The 50+ hours of compute Phase 5c/5d spent investigating mechanism stories that turned out to be measurement artifacts is a real cost. The four explicit closure-retraction cycles are real. The project's natural ambition (any-orbit any-eccentricity) is not yet achieved.

What Phase 5 did produce:
1. **The Phase 5b LEO deliverable.** A working agent, multi-seed validated, with documented capability and limits. This is real and shippable.
2. **The methodological infrastructure.** Twelve named disciplines, all grounded in specific findings. This is the most reusable output of the project.
3. **The verification of what doesn't work.** Reward-side interventions (~8 of them) at the recipe edge in LEO. Soft penalties for catastrophic actions. Continuous action spaces. Discrete(14). DT=30. All ruled out, with mechanism attribution where available.
4. **The setup for Phase 5.5.** The env-fix spec scopes the prerequisite cleanup; the altitude expansion spec scopes the work that addresses the recipe's actual scope limit.

The pattern that became visible across Phase 5: **the gap between named metrics and actual measurements is where most issues lived.** Six instances. Every closure-retraction cycle had at least one. The methodological discipline that emerged is largely about closing that gap systematically.

That's the lesson worth carrying forward beyond this project. The agent is one deliverable; the discipline of "verify the metric measures what its name implies" is another. Both are real.

---

*Author: 2026-05-02. Comprehensive Phase 5 retrospective. Captures the deliverable, what was tried (worked and didn't), the issues in implementation and interpretation, the methodological discipline that emerged, and the open questions for future engineering. Successors: phase5-env-fix-spec.md, phase5-5-altitude-expansion-spec.md.*
