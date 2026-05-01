# Phase 5c — Cracking the High-Eccentricity Ceiling

> **Status:** 2026-04-30. After Phase 5b shipped a working agent at e ≤ 0.10 and the Goodhart-induced collapse at e ≥ 0.20 was diagnosed. Phase 5c's goal is to extend the recipe past e = 0.20, ideally to e ≥ 0.50, because high-eccentricity capability is a structural prerequisite for Phase 6 (multi-body) work. This spec lays out a deep diagnostic suite plus a wide intervention search, organized to maximize information per compute.

---

## 0. Why high-eccentricity matters (load-bearing motivation)

A multi-body transfer rendezvous is fundamentally an eccentric-orbit problem. Three structural reasons:

1. **Transfer orbits are eccentric.** Earth-Moon Hohmann transfer has e ≈ 0.97. Earth-Mars has e ≈ 0.21. Multi-body transit *requires* the agent to fly eccentric trajectories.
2. **Multi-body orbits aren't Keplerian.** Halo orbits, Lissajous orbits, ballistic capture — the instantaneous osculating orbit has e that varies continuously, often passing through high-e transients.
3. **Recovery from off-nominal states.** Multi-body perturbations push the agent into eccentric states constantly. An agent that's brittle at e = 0.20 will be brittle at every flyby and every perturbation.

If the recipe can't handle e ≥ 0.20 robustly in two-body, attempting Phase 6 multi-body work would be premature. Phase 5c is the gate.

---

## 1. The Phase 5b finding we're starting from

Stage 4.2 (e_max = 0.20, fully random sat + target) collapses uniformly across 8 seeds. The collapse is mechanistically diagnosed as Goodhart-induced policy degeneration: cumulative shaping reward grows during training (1.50 → 2.18) while Φ_orbit grows (chaser drifts further from target), σ₂ activation drops (8.1% → 4.4%), and the policy globally degrades — both at the training distribution (e = 0.20: 4% → 0%) and at the warm-start's distribution (e = 0.05: 99% → 0%).

The mechanism is identified at first-order; the *exact* component generating the wrong-direction gradient isn't fully isolated. Phase 5c starts by closing that gap before committing to fixes.

### The numbers don't fully add up

A signal worth flagging: the writeup's "cumulative shaping 2.18 / Goodhart ratio 0.23" is a number that, by my analysis, is suspiciously close to the bias floor expected from γ = 0.995 discounting alone on Φ_orbit ≈ 24 over 2000 steps. **The shaping might not be doing wrong-direction work; the γ-discount on the magnitude of |Φ| might be the dominant bias term.** This is testable (§3.A1) and changes the fix substantially — γ-discount bias is a different problem than gate-failure Goodhart.

---

## 2. Phase 5c structure

Three blocks, sequential. Each block is gated on the previous:

- **Block I — Deep diagnostic.** Ten analytics that fully characterize the collapse mechanism. ~3-4 hours total compute. Output: refined mechanism story with high confidence about which interventions are likely to work.
- **Block II — Cheap intervention sweep.** Six low-cost interventions targeting different mechanism candidates, run in parallel where possible. ~6-10 hours compute. Output: which interventions move the needle.
- **Block III — Targeted deep work.** The 1-2 most promising interventions from Block II, expanded with multi-seed validation and capability surface eval at high e. ~10-15 hours compute. Output: a working agent at e ≥ 0.20, ideally e ≥ 0.50.

If Block I reveals the mechanism is unambiguous (e.g., γ-discount bias dominates), Block II compresses to just the relevant intervention. If Block I is ambiguous, Block II runs the full sweep.

---

## 3. Block I — Deep diagnostic

### 3.A1 — Per-step shaping decomposition

For a representative collapsed-policy episode at Stage 4.2 conditions, log per-step contributions from each component:
- `r_shape_Φorbit` = β·W_ORBIT·(γ^τ·Φ_orbit(s') − Φ_orbit(s))
- `r_shape_Φphase` = β·W_PHASE·(γ^τ·σ₂·Φ_phase(s') − σ₂·Φ_phase(s))
- `r_shape_Φvel` = β·W_VEL·(γ^τ·σ₃·Φ_vel(s') − σ₃·Φ_vel(s))
- `r_shape_clamp` = terminal Φ-clamp adjustment

**Plot each across the episode.** Sum each across the episode. Identify where the positive cumulative reward comes from.

**Decision rule:**
- If r_shape_Φorbit dominates and is mostly negative-per-step but cumulative-positive only via γ-discount accounting on growing |Φ|: this is **γ-discount bias** (Wiewiora 2003 effect). Fix: rescale W_ORBIT or use γ = 1 for the shaping (Ng-Harada-Russell shows this is valid).
- If r_shape_Φorbit dominates and is positively-signed per step (the agent finds states where Φ decreases briefly): this is **active gradient hacking**. Fix: gate-redesign or different Φ formulation.
- If r_shape_clamp dominates: this is **Φ-clamp pathology**. Fix: revisit clamp interaction with high-Φ failures.

Cost: ~30 min instrumenting + analysis on existing collapsed ckpts.

### 3.A2 — Counterfactual trajectory shaping

Take known-successful trajectories from the Stage 4.0 ckpt's 14% success rate at e = 0.20. Replay each through the shaping calculation. Compute cumulative shaping for successes vs. failures.

**The smoking-gun test.** If failed trajectories accumulate more cumulative shaping than successful ones, the shaping signal is *reversed* — and PPO is correctly maximizing what it's told to maximize, just toward the wrong target.

If successful trajectories accumulate more shaping than failed ones, the shaping is *correct* but training fails for a different reason (gradient noise, value confusion, exploration).

Cost: ~45 min. Need to extract the right trajectories from logs and run them through the Φ port.

### 3.A3 — Value-function landscape

At the collapsed Stage 4.2 ckpt, sample a grid of states across (Φ_orbit, Φ_phase, sat-target relative geometry). For each state, compute V(s) under the trained value head.

**Plot V(s) across the grid.** If V is *higher* at high-Φ_orbit states than at low-Φ_orbit states, the value function has learned to expect more return when far from the goal. This is the Wiewiora bias signature in the value function itself.

If V is correctly ordered (low Φ → high V), the value function is fine but the policy update is broken.

Cost: ~30 min. Implement state grid sampler, run V eval on existing collapsed ckpt.

### 3.A4 — Action-conditional value

Same grid as A3, evaluate Q(s, a) for each (s, a) pair using the actor-critic. At high-Φ states, is "coast" higher-value than burns? Is "warp" higher than burns?

If burns are uniformly lower-value than coast/warp at high-Φ, the policy correctly converges to coast/warp under its own value model. The value model is wrong; the policy is rational. Fix is value-function-targeted.

If burns are higher-value than coast at some states but the policy still picks coast: the policy is irrationally suboptimal. Fix is policy-targeted (entropy floor, exploration bonuses).

Cost: ~30 min. Extends A3.

### 3.A5 — Trajectory-level success-correlation of Φ

Across all Stage 4.0 ckpt episodes at e = 0.20: bin episodes by terminal Φ_orbit and compute success rate per bin.

**The hypothesis to test:** is Φ_orbit a valid signal for success at e = 0.20?

Expected if shaping is well-formed: success rate ↓ as Φ_orbit ↑.
Concerning if: success rate doesn't track Φ_orbit, or peaks at intermediate values.

This is the "is the proxy actually a proxy" check. If Φ_orbit doesn't correlate with success at e = 0.20, *no* fix to the gates or weights makes Φ_orbit-based shaping work — we'd need a different Φ.

Cost: ~20 min. Re-runs analysis on existing eval logs.

### 3.A6 — Same diagnostics on Phase 4 baseline

Run A1, A2, A5 on the Phase 4 ckpt (which works) at its native conditions (e = 0). Compare signatures to the e = 0.20 case.

**The diagnostic-by-contrast.** If A1 looks similar at e = 0 and e = 0.20, the shaping mechanism is the same; the issue is regime-specific. If A1 looks qualitatively different, the shaping mechanism *changes* at high e — and we've identified the regime boundary.

Cost: ~30 min. Re-uses A1/A2/A5 instrumentation on Phase 4 ckpts.

### 3.A7 — LVLH-frame degradation at high e

Log LVLH observation components during eval at e = 0.05 vs. e = 0.20. Look for:
- Saturation (components hitting normalization limits)
- Sign discontinuities (frame flipping)
- High-frequency oscillation that wasn't present at low e

If LVLH degrades at high e, the policy's input is partly noise — and no shaping fix repairs the underlying state observation problem.

Cost: ~30 min. Add observation logging hook during eval.

### 3.A8 — Action saturation analysis

In Stage 4.0 ckpt's successful episodes at e = 0.20, what fraction of burns are at the maximum (±25 m/s)?

If most successful burns saturate the action space, the agent is bottlenecked by action expressiveness — would benefit from larger Δv actions.

If burns are spread across the action space, action expressiveness isn't the limit.

Cost: ~15 min. Extends existing action distribution analysis.

### 3.A9 — Comparison to Lambert-optimal

For 10 e = 0.20 task instances, compute the analytical Lambert solution offline (initial state + target state → required Δv). Compare to the agent's actual Δv on the cases it solved.

If agent uses 5× the optimal Δv: the agent is highly suboptimal, and there's room for improvement.
If agent uses ≤ 1.5× optimal: agent is near-optimal, and the bottleneck is task difficulty, not policy quality.

Cost: ~1 hour. Need to write/find Lambert solver.

### 3.A10 — Effective horizon analysis

Compute typical successful-trajectory length at e = 0.20. Compare to γ = 0.995 effective horizon (1/(1−γ) = 200 steps).

If successful trajectories are ≥ 800 steps and effective horizon is 200, the agent literally can't see the reward through the discount. This was Phase 4's R2 finding; may be re-emerging at higher difficulty.

Fix candidates: γ → 0.998 (effective horizon 500), γ → 0.999 (1000), or N-step return adjustment.

Cost: ~15 min.

### 3.A11 — Bonus: phase × eccentricity correlation in failures

In Stage 4.2 collapsed-ckpt eval, partition failures by initial (phase_gap, e_target) and check for clustering. If failures concentrate at specific (phase, e) combinations, those are the "structurally hard" cases that the recipe genuinely can't handle. If failures are uniform across (phase, e), the recipe is broken everywhere at high e.

This is the cap-tail analysis at the high-e regime, applied to the *collapsed* policy rather than the working one. Comparing to Step 1's cap-tail (which found a (high-phase, low-e) clustering at low e) is informative.

Cost: ~15 min. Reuses existing cap-tail analyzer.

### Block I deliverable

A single document `PHASE5c_BLOCK_I_FINDINGS.md` covering all 11 analytics, with:
- The refined mechanism story (one of: γ-discount bias dominates / gate-failure Goodhart / value-function bias / observation degradation / horizon mismatch / mixed)
- Pre-committed Block II intervention priority based on the verdict

---

## 4. Block II — Cheap intervention sweep

Six interventions targeting different mechanism candidates. Run in parallel where compute allows. Each is ≤ 1.5 hour training + ~20 min eval.

### 4.B1 — γ = 1 for shaping (only)

If A1 reveals γ-discount bias dominates: zero out the discount on the shaping component while keeping γ = 0.995 for the value function. PB shaping with γ = 1 is the original Ng-Harada-Russell formulation and is provably bias-free.

Implementation: `r_shape = β·(Φ(s') − Φ(s))` instead of `β·(γ^τ·Φ(s') − Φ(s))`. Single line change in env reward.

Train Stage 4.2 from Stage 4.0 warm-start, eval at e = 0.20.

**Strong candidate if A1 verdict is "γ-discount bias."**

### 4.B2 — Continuous gates

Replace threshold gates (σ₂ = 1 if Φ_orbit < EPS_ORBIT else 0) with smooth sigmoids:
σ₂_continuous = sigmoid((EPS_ORBIT − Φ_orbit) / temperature)

Removes the gate cliff that creates discontinuities in the gradient. Hygiene improvement regardless of mechanism.

**Implementation: ~2 hours.** Sigmoid-replace the threshold logic. Choose temperature: temperature = 0.5·EPS_ORBIT is a reasonable default.

Train Stage 4.2 from warm-start.

### 4.B3 — Mixed-distribution training

Train Stage 4.2 with e ~ Mixture(0.7·Uniform(0, 0.05) + 0.3·Uniform(0, 0.20)) instead of e ~ Uniform(0, 0.20).

The 70% e ≤ 0.05 episodes provide clean gradient signal that maintains the warm-start's capability. The 30% e ≤ 0.20 episodes provide exposure without overwhelming the gradient.

**Implementation: ~30 min.** Modify env reset to sample from mixture distribution.

This is the most theoretically-motivated intervention regardless of mechanism — it directly addresses the "training-distribution shift overwhelms the warm-start" failure pattern that the data is consistent with.

### 4.B4 — Larger Δv actions

Add ±50 and ±100 m/s prograde/retrograde actions to the Discrete(10) action space. Discrete(14) total.

**Implementation: ~1 hour.** Extend action table; warm-start surgery from Discrete(10) to Discrete(14) (zero-init the new rows, similar to Phase 4 D7→D9 surgery).

Train Stage 4.2 from the surgically-extended warm-start.

If A8 verdict is "successful burns saturate at ±25," this is highly likely to help.

### 4.B5 — Higher γ

γ = 0.998 (effective horizon 500) or γ = 0.999 (1000). Train Stage 4.2 from warm-start.

If A10 verdict is "successful trajectories ≥ 800 steps," this is likely necessary.

**Implementation: ~10 min.** Config change.

### 4.B6 — REL_VEL_TOL annealing

Start training at REL_VEL_TOL = 200 m/s, anneal to 50 m/s over 25M steps.

Lowers the success bar early, providing more frequent terminal +10 reward to anchor the value function.

**Implementation: ~1 hour.** Add annealing schedule to env config.

This is the long-deferred Phase 5b Step 1 intervention; testing it directly settles whether it would have helped earlier.

### Block II deliverable

A document `PHASE5c_BLOCK_II_FINDINGS.md` reporting:
- Per-intervention success rate at e = 0.20 (single seed, 50-ep eval)
- Cross-intervention compatibility flags (which fixes work better when combined)
- Block III prioritization

**Decision rule for Block III:**
- If any single intervention reaches ≥ 50% at e = 0.20: that's the candidate for Block III multi-seed.
- If 2-3 interventions reach 30-50%: combine them (e.g., mixed-distribution + γ = 0.999 + larger Δv) and test the combination.
- If all interventions stay below 30%: Block I diagnostic was insufficient; return to mechanism investigation.

---

## 5. Block III — Targeted deep work

The 1-2 most promising Block II interventions, validated with multi-seed protocol.

### 5.C1 — Multi-seed validation

5 training seeds × the chosen intervention. Reach in-distribution convergence at e = 0.20 with the new recipe. Report mean ± std.

### 5.C2 — Eccentricity expansion

If Block III's recipe lands at e = 0.20 cleanly, expand to e = 0.30, 0.50, 0.70 (Molniya). Each is a sub-stage; train to convergence per-stage; eval at all preceding e levels for OOD characterization.

This re-runs the Phase 5b Stage 4.x progression but with the fixed recipe. The expectation: if the recipe is fixed, eccentricity expansion should be smooth (consistent with the Phase 5b Step 1 finding that random sat init enables natural-phasing strategies that are *more* powerful at higher e).

### 5.C3 — Capability surface eval

Re-run the Phase 5b Block D capability surface at the final Phase 5c ckpt. Compare to Phase 5b's surface.

### 5.C4 — Phase 4 condition check

Eval the final Phase 5c ckpt at Phase 4 conditions (sat circular, target circular, different a). Verify the high-e training hasn't destroyed low-e capability. Per the post-extend lesson, this check should be standard at every stage.

### Block III deliverable

`PHASE5c_FINDINGS.md` final report covering:
- Mechanism verdict (refined from Block I)
- Working recipe at e ≥ 0.20 (target: e ≥ 0.50)
- Capability surface
- Recipe diff from Phase 5b (what changed and why)

---

## 6. Compute budget

| Block | Activity | Compute |
|---|---|---|
| I | 11 analytics, mostly post-hoc on existing data | ~4 hours |
| II | 6 interventions × ~1.5 hr training + ~20 min eval | ~10 hours |
| III | 5-seed multi-seed at chosen intervention + e expansion | ~15 hours |
| **Total** | | **~30 hours** |

Comparable to Phase 5b's full budget. Realistically 1-2 weeks of elapsed time with parallel runs.

---

## 7. What's locked, what's open

### Locked

- Random sat init + same_orbit_init logic from Phase 5b Step 1
- 5-seed multi-seed protocol at headline stages (Phase 5a addendum)
- Train-longer null first when stages stall (Phase 5b Step 1 lesson)
- No structural claims from <2σ data (Phase 5b post-extend lesson)
- Capability surface eval as final deliverable
- Phase 4 condition eval at every major stage

### Open (Block I will determine)

- Whether γ-discount bias is the dominant Goodhart driver (changes which intervention runs first)
- Whether action-space expressiveness is a co-factor (changes whether Discrete(14) surgery is required)
- Whether LVLH degrades at high e (changes whether observation reformulation is needed)
- Whether the Phase 4 baseline shows the same signatures (changes whether the issue is regime-specific or recipe-specific)

### Out of scope

- New algorithmic interventions outside what Block II proposes (R3 components stay dead unless Block I demands them)
- Continuous action space (Phase 5b ruled out)
- Debris (still on user hold)
- Multi-body (Phase 6)

---

## 8. The deeper bet

Phase 5c is a bet that the e ≥ 0.20 ceiling is fixable, not fundamental. The Phase 5b Goodhart finding identifies a specific failure mode (gated-shaping degeneration with cumulative-shaping inflation) that admits multiple plausible fixes. The diagnostic-then-intervention structure here is designed to identify *which* fix without burning compute on all of them.

If Block I reveals the mechanism is something we haven't anticipated (e.g., the value function has fundamentally collapsed in a way that no per-step intervention can repair), Phase 5c may not produce a working high-e agent. In that case, the deliverable becomes "we identified a recipe-level failure that can't be patched without recipe redesign," and Phase 6 starts with a broader recipe rethink.

This is acceptable as an outcome. **The discipline: don't pretend a fix worked if Block II's interventions don't actually move the needle.** Prior phases of the project have caught themselves overinterpreting borderline data; Phase 5c needs to maintain that discipline at higher stakes.

---

## 9. Pre-committed thresholds

For each block, pre-commit to what counts as success:

### Block I success
A single mechanism story that explains: cumulative shaping growing, Φ_orbit growing, gates dropping, policy degrading globally. The story should make a falsifiable prediction about which Block II intervention will work.

### Block II success
At least one intervention reaches ≥ 50% at e = 0.20 (single seed). At least 2 interventions reach ≥ 30%. Or: a combination of interventions reaches ≥ 50%.

### Block III success
Multi-seed (5 seeds) at e = 0.20 reaches ≥ 70% mean. Eccentricity expansion to e = 0.50 reaches ≥ 50% mean. Capability surface shows graceful degradation across all axes.

### Stretch goal
e = 0.70 (Molniya) reaches ≥ 30%. This validates the recipe across the full eccentricity range relevant for multi-body work.

### Conditions to stop and revise
- Block II: zero interventions reach ≥ 30%. Block I diagnostic was insufficient; return to investigation rather than pushing forward.
- Block III: multi-seed variance is so high (>30pp std) that "average performance" is meaningless. The recipe is too fragile; need to address robustness before scaling.

---

## 10. Sequencing

In order:

1. **Block I (week 1).** Run all 11 analytics. Synthesize verdict. Estimated 1 day of analysis on existing data plus diagnostic infrastructure setup.
2. **Block II (week 2).** Run interventions in priority order from Block I's verdict. Parallelize where compute allows (multiple training runs concurrent). Estimated 3-5 days.
3. **Block III (week 3-4).** Multi-seed validation of best intervention. Eccentricity expansion. Capability surface. Estimated 1 week.
4. **Writeup (week 4).** PHASE5c_FINDINGS.md, recipe documentation, Phase 6 transition spec.

Total elapsed: ~3-4 weeks. Most of that is training; analysis and writing fit in compute-blocked time.

---

## 11. Phase 6 implications

If Phase 5c succeeds (working agent at e ≥ 0.50):

- Phase 6 multi-body work proceeds with confidence that the two-body recipe handles the high-e regime that multi-body inevitably involves.
- The recipe's e-handling lessons transfer (gate design, γ choice, action space expressiveness, distribution mixing).
- The capability surface methodology generalizes to multi-body's higher-dimensional task space.

If Phase 5c partially succeeds (working at e = 0.20 or 0.30, not 0.50):

- Phase 6 multi-body work is constrained to the eccentricity regime the recipe handles. Some interesting multi-body scenarios (Earth-Moon Hohmann, Mars transfer) become harder.
- The portfolio piece is still strong but with a documented eccentricity ceiling for two-body work.

If Phase 5c fails (no intervention crosses e = 0.20):

- Multi-body work is premature. Phase 6 starts with a recipe redesign rather than environment expansion.
- Phase 5c's failure becomes a meaningful negative finding about curriculum-learning-with-PB-shaping limits at challenging task regimes.

All three outcomes are publishable. The discipline: ship what the data supports.

---

*Author: 2026-04-30. Phase 5c spec, drafted after Phase 5b's Goodhart-induced collapse identification. Successor: PHASE5c_FINDINGS.md after Blocks I-III complete.*
