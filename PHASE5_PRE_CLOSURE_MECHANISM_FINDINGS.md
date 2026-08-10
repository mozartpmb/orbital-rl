# Phase 5 — Pre-closure Mechanism Verification Findings (W1–W4)

> **Status:** 2026-05-22. Four cheap checks executed per `phase5-pre-closure-mechanism-verification.md`. ~30 min compute (analysis only, no training). **The verification did not confirm the prior mechanism story — it overturned it.** The "relation × altitude flip" framing from V2 was an artifact of metric leakage at safety-cap timeouts; once the success criterion is corrected to require physical rendezvous, the 52% GEO `fully_random` and 22% MEO `fully_random` capability disappears almost entirely. The recipe has ONE skill (phasing on shared-shape configurations), not two. Per spec §6 decision matrix, this is the bottom-right cell ("neither mechanism is right; investigate before closure"). The investigation is here. The closure framing must change.

---

## TL;DR

- **W1 (GEO near-equilibrium check):** Falsified. Trajectory inspection at GEO `fully_random` `e=0.05` shows 18/30 "successes" had initial Δa **>20,000 km** with **zero fuel used** and ran to safety cap. The "successes" are not near-equilibrium configurations — they're episodes where the policy did literally nothing while accumulated potential-shaping reward leaked positive at terminal.

- **W2 (MEO_low phasing mechanism):** Confirmed. MEO_low `same_orbit` successes have ep_len ~752 (~5 orbital periods) and fuel_used ~0.06 — real maneuvering. The phasing skill at low altitudes is real.

- **W3 (F-B+ bimodality):** Confirmed. Seed 31415 stable at LEO 95%; seeds 42 and 20260423 drift monotonically downward to 85% and 81%. Bimodal across seeds, not continuous-with-different-rates.

- **W4 (relation × altitude reframe):** **Removed entirely.** Once the metric artifact is corrected (W1), same_orbit consistently ≥ fully_random at all altitudes. The "flip" doesn't exist. The recipe has a single phasing skill that degrades monotonically with altitude.

**The 9th metric-vs-implementation pattern instance:** eval_checkpoint.py's `rew > 0` success classifier reads the terminal step's reward, but on safety-cap timeouts the Φ-clamp adjustment (`β·(0 - Φ_prev)`) adds positive bias proportional to how far from target the sat is. The further OOD the policy, the larger the artifact. GEO `fully_random` is dominated by this leakage; LEO and MEO_low cells where the policy genuinely solves the task aren't affected.

**Closure implication:** Phase 5b's deliverable is **LEO/MEO_low phasing on shared-shape configurations**, not "any-altitude rendezvous with 50% headline." The honest framing:
- LEO at e≤0.05: 96-98% multi-seed real successes (Phase 5b's published deliverable).
- MEO_low same_orbit at e≤0.10: 22-36% real (a previously-not-known secondary capability).
- MEO/GEO under any relation: ~0% real (no altitude generalization).
- Stage 5.5.4A "49.7%" is now ~12% real (across the full envelope; mostly the LEO portion).

---

## 1. W1 — GEO near-equilibrium mechanism check

### 1.1 Method

For the cell `(alt=GEO, e_target=0.05, e_sat=0.05, relation=fully_random)` (54% headline), sampled 30 trajectory dumps. For each: extracted initial `|sat.a - target.a|`, episode length, fuel used, and `episode_reward`.

### 1.2 Result

Successes (`episode_reward > 0`, the eval's classifier):

| Stat | Value |
|---|---:|
| n successes | 18/30 |
| Initial Δa, mean | **27.9 Mm** (almost the full GEO–LEO span) |
| Final Δa, mean | **27.9 Mm** (unchanged — no maneuvering happened) |
| Episode length | 1999 steps (all at safety cap) |
| Fuel used, mean | **0.0000** |

Failures (`episode_reward ≤ 0`):

| Stat | Value |
|---|---:|
| n failures | 12/30 |
| Initial Δa, mean | 4.57 Mm |
| Final Δa, mean | 4.57 Mm |
| Episode length | 1999 (all at safety cap) |
| Fuel used, mean | 0.003 |

**Verdict: the "successes" are episodes where the policy did nothing for 2000 steps while at ~28 Mm from target.** They are not near-equilibrium configurations. The mechanism story in V2 §2.6 ("fully_random at GEO succeeds because valid_init_only retains lucky near-equilibrium configurations") is falsified.

### 1.3 The actual mechanism — Φ-clamp leak

The shaping potential is `Φ(s) = -(W·Φ_orbit · σ₁ + ...)`. When sat is far from target, Φ_orbit is large positive → Φ is large negative.

Per-step shaping reward: `r_step = β · (γ Φ_next − Φ_prev)`. If the policy does nothing (sat orbit unchanged), Φ_next ≈ Φ_prev, so `r_step ≈ (γ-1) Φ_prev = -0.005 · Φ_prev`. With Φ_prev large negative, r_step is small positive (~+0.5/step at GEO Δa).

Terminal Φ-clamp (RECIPE.md §"Reward shaping"): on terminal, append `β · (0 − Φ_prev)` to reward. At safety cap, terminal reward = `safety_cap_penalty + clamp = -10 + (-Φ_prev) = -10 + large_positive ≈ large positive` when sat is far from target.

So at GEO with the policy frozen at 28 Mm away: total reward = (-0.005 · Φ_prev) × 2000 steps + (-10 + (-Φ_prev)) terminal = positive number proportional to how OOD the sat was. The further from target, the more "positive" the apparent success.

This is the **9th metric-vs-implementation pattern** in this project. Pre-experiments § 6.3 explicitly warned: "every named quantity should be assumed misleading until verified." Eval's `rew > 0` was verified at LEO (where successes are real) and assumed to generalize. It didn't.

### 1.4 Corrected classifier

Real success = **episode terminated before safety cap (ep_len < 1999) AND episode_reward > 0**.

This catches only episodes where the env emitted the +10 success terminal (which only fires when |sat.a−target.a| < SUCCESS_TOL_A AND velocity match AND phase match are all true simultaneously). Safety-cap timeouts have ep_len = 1999 regardless of accumulated shaping.

### 1.5 Reclassified surface — full Probe 1 (n=120K episodes)

Diagonal cells (e_target = e_sat), aggregated over 3 rollout seeds × 4 phases:

| alt × relation | eval rew_succ% | **real_succ%** | artifact |
|---|---:|---:|---:|
| LEO same_orbit | 33.1% | 33.1% | 0pp |
| LEO fully_random | 30.8% | 30.8% | 0pp |
| MEO_low same_orbit | 22.1% | 22.1% | 0pp |
| MEO_low fully_random | 1.2% | 1.2% | 0pp |
| MEO same_orbit | 5.3% | 5.3% | 0pp |
| MEO fully_random | 19.8% | **0.1%** | **+19.7pp** |
| GEO same_orbit | 2.8% | 2.8% | 0pp |
| GEO fully_random | 51.5% | **0.0%** | **+51.5pp** |

(LEO numbers above span e ∈ {0, 0.05, 0.10, ...} — the e=0.10+ cells are all 0% per the geometric infeasibility finding; e=0.0 and e=0.05 are at 88-100%; the rollup mean is 30-33%.)

The artifact concentrates entirely at `fully_random` × `MEO+`. Every same_orbit cell, every LEO cell, every MEO_low cell, and every cell where the policy actually engages the task, has artifact = 0.

### 1.6 Stage 5.5.4A reclassification

The headline 49.7% from the stochastic-eval probe (`PHASE5_5_STOCHASTIC_PROBE_FINDINGS.md` §1) was the eval's `rew > 0` aggregated across `a∈[6.671, 42.5] Mm, e≤0.50, fully_random, valid_init_only=1`. Re-running this aggregation on the Probe 1 data with the corrected classifier:

| Aggregation | Eval rew% | Real % |
|---|---:|---:|
| GEO band aggregate | 26.2% | **0.0%** |
| Stage 5.5.4A whole-envelope | 20.8% | **11.9%** |

The 11.9% real-success figure for Stage 5.5.4A is dominated by the LEO portion of the uniform-altitude sample (~14% of samples fall in [6.671, 7.171] Mm where the policy works ~96%). The MEO/GEO portion contributes ≈0.

**The original "49.7% Stage 5.5.4A baseline" was 50% real LEO portion + 50% Φ-clamp leak.** Not altitude generalization at all.

---

## 2. W2 — MEO_low phasing-by-period mechanism

### 2.1 Method

For the cell `(alt=MEO_low, e_target=0.05, e_sat=0.05, relation=same_orbit)` (35.7% headline; now confirmed real per W1), sampled 30 trajectory dumps. Computed ep_len and fuel_used distributions for successes vs failures.

### 2.2 Result

| Stat | Successes (n=11) | Failures (n=19) |
|---|---:|---:|
| Mean ep_len | 752 steps | 1999 (safety cap) |
| Mean fuel_used | 0.0596 | 0.0242 |

### 2.3 Interpretation

MEO_low orbital period at a ≈ 9.4 Mm: T = 2π√(a³/μ) ≈ 160 min = 160 steps at DT=60s. Successful ep_len mean 752 corresponds to ~4.7 orbital periods. The policy uses multiple phasing iterations to close the angular gap, then completes the rendezvous. Real maneuvering with finite fuel cost.

Failures all run to safety cap with modest fuel use — the agent attempts maneuvers but can't close the gap within the step budget.

**Phasing-by-period mechanism confirmed.** The recipe has a real phasing skill at MEO_low altitudes.

### 2.4 Why not at MEO or GEO

MEO at a ≈ 16 Mm: T ≈ 350 min = 350 steps; 2000-step budget = ~5.7 periods. Still feasible numerically but real_succ = 5.3% — the policy isn't iterating phasing well at MEO same_orbit.

GEO at a ≈ 42 Mm: T ≈ 1440 min = 1440 steps; 2000-step budget = ~1.4 periods. **Less than 2 periods of warp budget.** Insufficient for iterative phasing.

This matches the period-budget mechanism per V2 §2.6: phasing works at altitudes where the policy can iterate within MAX_STEPS. Below MEO the iteration count is large (LEO ~22 periods, MEO_low ~12, MEO ~5, GEO ~1). The recipe trained at LEO learned to iterate; that transfers partially to MEO_low (5+ periods still); fails at MEO (~5 marginal) and GEO (~1 period, insufficient).

---

## 3. W3 — F-B+ bimodality framing

### 3.1 LEO success rate across F-B+ ckpts (from V1 scan CSV)

| Epoch | seed 42 | seed 31415 | seed 20260423 |
|---:|---:|---:|---:|
| 5 | 98.0 | 96.0 | 92.5 |
| 10 | 94.5 | 96.5 | 95.0 |
| 15 | 91.5 | 90.5 | 92.5 |
| 20 | 87.5 | 93.5 | 95.5 |
| 25 | 86.5 | 97.5 | 90.5 |
| 30 | 85.0 | 95.5 | 88.5 |
| 35 | 87.0 | 95.5 | 84.0 |
| 40 | 84.5 | 95.0 | 84.5 |
| 45 | 84.5 | 95.5 | 81.0 |
| 50 | 85.5 | 95.0 | 81.0 |
| 54 | 85.5 | 95.5 | 81.0 |

### 3.2 Verdict

**Bimodal across seeds.** Seed 31415 stays in the 90-97% range with mild wobble (stable attractor). Seeds 42 and 20260423 drift monotonically downward, landing at 85% and 81% by the end of training (unstable attractor).

Per spec §3.3 decision rule: "F-B+ produces bimodal LEO outcomes across seeds, consistent with the recipe-edge bimodality pattern Phase 5b exhibited at e=0.05." This is the wording for the closure document. The pattern is information about protocol stability — F-B+ has a stable basin and an unstable basin, and which seed lands in which is determined by RNG.

---

## 4. W4 — Relation × altitude reframe (now obsolete)

### 4.1 The original "flip" framing

V2 §2.5 reported:

| Band | same_orbit | fully_random | Winner |
|---|---:|---:|---|
| LEO | ~100% | ~88% | same_orbit |
| MEO_low | ~35% | ~2% | same_orbit |
| MEO | ~8% | ~25% | fully_random (flip) |
| GEO | ~4-5% | ~52% | fully_random (large) |

This was the verification's headline finding — same_orbit at LEO/MEO_low, fully_random at MEO/GEO.

### 4.2 Corrected table

With the W1 artifact removed:

| Band | same_orbit | fully_random | Winner | Gain from same_orbit |
|---|---:|---:|---|---:|
| LEO | ~33% | ~31% | same_orbit | +2pp |
| MEO_low | ~22% | ~1% | **same_orbit** | +21pp |
| MEO | ~5% | ~0% | same_orbit | +5pp |
| GEO | ~3% | ~0% | same_orbit | +3pp |

**No flip.** Same_orbit dominates fully_random at every altitude. The flip was entirely the Φ-clamp leak at GEO/MEO fully_random.

(LEO numbers here are mean across e ∈ {0, 0.05, ..., 0.50}, so they aggregate the working e≤0.05 cells with the e≥0.10 geometric-infeasibility cells. Per-cell LEO same_orbit at e≤0.05 is 98-100%.)

### 4.3 The corrected story

The recipe has **one skill**: phasing on configurations where sat and target share orbital shape (same a, e, ω), differing only in θ. This skill works at altitudes where the agent can iterate multiple orbital periods within MAX_STEPS:

- LEO: ~22 periods budget → 96-100% at low e.
- MEO_low: ~12 periods budget → 22-36% (partial — the skill transfers but isn't fully reliable).
- MEO: ~5 periods budget → 5% (mostly fails).
- GEO: ~1.4 periods budget → 3% (almost no chance).

Under `fully_random`, sat and target have independent (a, ω, θ). The policy has no maneuvering skill to bridge that gap — fuel budget is insufficient for hard altitude transfers (~3 km/s for LEO→GEO vs ~480 m/s available). So `fully_random` real success rates are all ≤1% above LEO.

This is a **cleaner, more honest deliverable framing** than the "relation × altitude flip" or the "49.7% altitude envelope" headlines. The recipe is a phasing-on-shared-orbits specialist whose capability degrades with altitude in proportion to period budget.

---

## 5. Decision per spec §6 matrix

| W1 (GEO Δa) | W2 (MEO_low length) | Implication |
|---|---|---|
| **Broadly distributed (28 Mm successes)** | Clusters at ~5 periods (confirmed) | "GEO does real maneuvering; we don't understand it. The 'no altitude generalization' claim is wrong. **Major reframe.**" |

Per the spec, this is the matrix's row 3 — but the reframe is the OPPOSITE of what the spec anticipated. The spec assumed "broadly distributed Δa" would mean "GEO does real maneuvering." The data shows broadly distributed Δa means "GEO 'successes' are not successes at all — they're the policy doing nothing while the metric leaks." So the reframe is: **the recipe is more LEO-specific, not less**.

---

## 6. Implications for Phase 5 closure

### 6.1 What the closure document must say

The Phase 5b ckpt's true capability is:

| Condition | Real success rate (multi-seed, n>=2400) |
|---|---:|
| LEO, e ≤ 0.05, any relation | 88-100% |
| MEO_low (a ≤ 12 Mm), same_orbit, e ≤ 0.10 | 22-36% |
| MEO_low fully_random, any e | ≤ 3% |
| MEO, any relation, any e | ≤ 8% |
| GEO, any relation, any e | ≤ 5% |
| LEO at e ≥ 0.10 | 0% (geometric infeasibility — no valid inits) |

The headline portfolio claim becomes:

> Two-body coplanar rendezvous agent. Reliably solves LEO same-orbit phasing at low eccentricity (96-98% multi-seed); partially generalizes to MEO_low same-orbit (22%). No measured generalization to MEO/GEO or to fully-random orbit pairs at higher altitudes. The recipe's capability decays with orbital period (period-budget mechanism within MAX_STEPS).

Stronger than "LEO low-e specialist." Weaker than "any-altitude generalization." Honest.

### 6.2 What the closure document must NOT say

- ❌ "The recipe handles 49.7% of the LEO+MEO+GEO envelope." (False; this was Φ-clamp leak.)
- ❌ "Relation × altitude flip pattern." (False; once artifact removed, same_orbit dominates at every altitude.)
- ❌ "Two skills: phasing + null-maneuver." (False; only the phasing skill is real.)

### 6.3 What's now genuinely interesting for portfolio framing

The **Φ-clamp leak finding itself** is a portfolio-worthy story: a real example of metric-vs-implementation drift in RL evaluation, surfaced through systematic verification, with a clean mechanism story (γ < 1 in potential-based shaping leaks reward proportional to OOD distance, terminal clamp amplifies it). This is the kind of "actually I checked" finding that signals rigorous practice.

### 6.4 Open question — does the leak affect Phase 5b's published 96.4%?

LEO at e=0.05 has 0pp artifact inflation in the corrected table (eval rew% = real% in every LEO cell that successfully completes). So **no — Phase 5b's 96.4% LEO baseline is real**. The leak only triggers when episodes reach safety cap (= 1999 steps) without success; LEO successes terminate naturally at ep_len < 1000. The published number is safe.

But every Phase 5e capability surface number that included MEO/GEO cells with `fully_random` would have been inflated. Worth a brief audit during closure (most Phase 5 publications stayed at LEO, so the audit scope is small).

---

## 7. Files produced

| Path | Purpose |
|---|---|
| `/tmp/p5_5_w1_reclassify_v3.py` | The corrected classifier (early-term + positive-reward) |
| `PHASE5_PRE_CLOSURE_MECHANISM_FINDINGS.md` | This document |

The Probe 1 trajectory dumps at `/tmp/p5_5_probe1/` (8.5 GB) are the underlying data; preserve if any further re-analysis is needed.

---

## 8. Recommended next step

**Apply the corrected classifier to all Phase 5 capability claims, then write closure.** Specifically:
1. Re-run the per-condition surface (`web_data/results/phase5_capability_surface_*.csv`) audit with the corrected classifier for any rows involving MEO/GEO/`fully_random`. Identify which previously-published numbers are inflated.
2. Patch `eval_checkpoint.py` to either (a) report both eval-reward-success% AND physical-success% (preferred — backward compatible), or (b) switch the default success classifier to the corrected one (breaking, but more honest going forward).
3. Update `RECIPE.md` "Capability" section with the corrected per-condition table from §6.1.
4. Write `PHASE5_FINDINGS.md` final consolidated document with the honest framing.
5. Write `PHASE6_TRANSITION.md`.
6. Close Phase 5.

Step 2 is the most consequential infrastructure-side fix — every future eval should use the corrected classifier. Worth doing before any Phase 6 work that uses eval_checkpoint.py.

---

*Author: 2026-05-22. Phase 5 pre-closure W1-W4 verification. ~30 min compute. The verification overturned the prior mechanism story (cell row 3 of spec §6 matrix, but with inverted direction): the GEO `fully_random` "capability" was Φ-clamp leak at safety cap, not near-equilibrium maneuvering. The recipe is a LEO/MEO_low phasing specialist, not an altitude-generalizing agent. The closure document must reflect the honest framing. 9th instance of the metric-vs-implementation pattern.*
