# Phase 5b Prediction Tests — Goodhart-Induced Collapse Identified

**Date:** 2026-04-30 · ~30 min compute · diagnoses the Stage 4.2 ceiling mechanism.

## What was tested

Three predictions about the Stage 4.2 collapse + a bonus zero-shot scan, designed to probe whether the failure is "policy stops trying" (passive) or "policy actively Goodharts" (active).

---

## Bonus — zero-shot Stage 4.0 ckpts at e=0.20

For each of 3 successful Stage 4.0 training seeds, scan all saved ckpts (every 25 epochs) at e=0.20 conditions:

| Stage 4.0 seed | Best Stage 4.0 ckpt @ e=0.20 (no e=0.20 training) | Best epoch |
|---|---|---|
| 42 | 14.0% | 275 |
| 20260423 | 14.0% | 300 |
| 31415 | 16.0% | 275 |

**No Stage 4.0 ckpt exceeds 16% at e=0.20.** Combined with 0/8 trained Stage 4.2 results, the e=0.20 ceiling is robust across both pre-training (zero-shot) and post-training (collapse) views. **Ceiling claim holds.**

---

## Prediction 1 — action distribution at peak vs collapsed

Predicted: coast/warp share ↑, burn share ↓, Δv ↓ at the collapsed ckpt.

50-ep eval at Stage 4.2 conditions on the same seed-7919 run:

| Metric | Ep 25 (peak, 4%) | Ep 200 (collapsed, 0%) |
|---|---|---|
| Mean total Δv per ep | 150 m/s | **104 m/s** ↓ as predicted |
| Burn frequency (mean) | 10.4% | **13.1%** ↑ opposite of prediction |
| Coast frequency | 7.9% | 6.0% |
| Warp frequency | 90.9% | 93.1% |

**Mixed result.** Δv drops as predicted, but burn frequency *rises*. The collapsed policy is doing *more, smaller burns*, not "stopping." This contradicts the simple "policy gives up" hypothesis from earlier writeups.

Note: median burn frequency is similar (1.30% vs 1.07%). The mean-vs-median gap suggests bimodality — most episodes mostly coast/warp, but a minority burn aggressively. The "more burns at lower Δv" pattern in the mean reflects shifted aggression, not uniform passivity.

This rules out one mechanism (passive coasting) but doesn't yet identify the active mechanism.

---

## Prediction 2 — σ gate activation drops

Predicted: σ₂ activation should drop as policy stops attempting orbit-shape changes.

| Gate metric | Ep 25 (peak) | Ep 200 (collapsed) |
|---|---|---|
| σ₂ activation rate | 8.1% | **4.4%** ↓ |
| σ₃ activation rate | 7.5% | **0.0%** ↓ |
| Φ_orbit (orbit mismatch potential) | 19.7 | **23.8** ↑ |
| Mean Φ_total | -0.20 | -0.24 |
| Cumulative shaping reward (per ep) | 1.50 | **2.18** ↑ |
| Goodhart ratio | mean 0.16 / 63% over 0.1 | mean 0.23 / **73% over 0.1** |

**Confirmed and informative.** σ activation drops as predicted, but the *reason* exposes the mechanism:

- Φ_orbit *grows* during training (19.7 → 23.8) — sat orbit drifts further from target.
- σ₂ requires Φ_orbit < EPS_ORBIT (=2.0). Φ_orbit ≈ 24 means σ₂ rarely fires.
- σ₃ requires σ₂ active AND Φ_phase < EPS_PHASE. With σ₂ near-dead, σ₃ goes to 0.
- The gated cascade is fully inactive at the end of training.
- **Shaping reward is positive (cumulative ~2.18) and growing across training.** Without σ₂/σ₃ gating, the un-gated Φ_orbit term dominates — and it's incentivizing whatever-the-agent-does that grows shaping accumulation.

Goodhart ratio at ep200 is 0.226 with 73% of eps exceeding the 0.1 threshold (vs 63% at ep25). **The shaping reward is the dominant signal**, not the terminal failure reward.

---

## Prediction 3 — collapsed Stage 4.2 ckpt at e=0.05

Predicted: training at e=0.20 should globally degrade the policy, including at the warm-start's training distribution (e=0.05).

| Ckpt | Eval condition | Success |
|---|---|---|
| Stage 4.0 best (seed 31415, ep 350) | e=0.05 fully random | **99%** |
| Stage 4.2 final (collapsed, same seed) | e=0.05 fully random | **0%** |

**Confirmed at extreme magnitude.** The collapsed Stage 4.2 ckpt is *catastrophically* worse than its warm-start at the warm-start's own training distribution.

This isn't "specialization" — specialization would mean the policy is over-tuned to e=0.20 (and worse at e=0.05 because of distribution shift). What we see is **global destruction**: the policy is worse at e=0.20 (4% vs 14% zero-shot) AND worse at e=0.05 (0% vs 99%). Both directions degraded.

---

## Unified mechanism story: Goodhart-induced policy collapse at e=0.20

Combining all three prediction results plus the Φ-port instrumentation:

1. **Warm-start state:** Stage 4.0 ckpt at Stage 4.2 conditions has Φ_orbit ≈ 11-20 (target-eccentric mismatch is large), σ₂ activation ~8%, scattered terminal successes (~14% baseline at e=0.20).
2. **Training begins:** PPO updates the policy. Some early episodes succeed (epoch 25 sees 4% peak).
3. **Shaping reward dynamics dominate:** at e=0.20, the gated cascade rarely activates fully. The dominant component is **the un-gated Φ_orbit · σ_1 = Φ_orbit · 1 = Φ_orbit term**. With β=1 and W_ORBIT=0.01, that's `r_shape ≈ -0.01 · (γ^τ · Φ_orbit_curr − Φ_orbit_prev)`.
4. **Wrong-direction gradient:** the policy can *increase* shaping reward by *increasing* Φ_orbit_prev > γ^τ · Φ_orbit_curr (i.e., previously orbit was further away and it's now slightly closer). Within the constraints of the env physics, the policy can find a "drift away then a tiny bit closer" pattern that accumulates positive shaping while never converging on success.
5. **Φ_orbit drifts upward:** the agent settles into an orbit even further from the target (19.7 → 23.8 across training). Gates remain inactive.
6. **Terminal reward never fires:** without σ₂/σ₃ gates, the gated multi-stage potential never sees the "approaching success" condition. Successful trajectories become rarer (4% → 0%).
7. **Policy globally degrades:** the same trajectory pattern fails at e=0.05 too (Stage 4.2 ckpt at e=0.05 = 0%, despite warm-start being 99%).

This is the **classic Goodhart pattern** — exactly what the Phase 4 plan §3.2 was designed to prevent via the gated cascade and Φ-terminal-clamp. The audits earlier showed gates working at e≤0.10. **At e≥0.20, the gates fail to activate, the shaping degenerates, and the policy Goodharts.**

---

## What this means for the e=0.20 ceiling

The ceiling is not "the recipe can't produce success at e=0.20." The recipe's *zero-shot* (no-train) capability at e=0.20 is 14-16%. The *trained* capability is 0-4%. **Training actively makes things worse.**

The mechanism is not opaque or fundamental — it's an identifiable shaping-design issue that the Phase 4 audit predicted (gates need to activate to produce useful gradient) and the Phase 5b audit confirmed (σ₃ structurally weak at high e). Phase 5b never closed the loop on this issue because Step 1's warm-start fix at e=0.05 was sufficient and we deferred σ₃ retune.

**Phase 6 fix candidates** (in order of likely effectiveness):

1. **Retune EPS_ORBIT and EPS_PHASE for high-e regime.** Currently EPS_ORBIT=2 means σ₂ activates only when sat orbit nearly matches target shape. At e=0.20, "nearly matches" is rare during training — the gate is too tight. Raising EPS_ORBIT to 5 or 10 lets σ₂ activate during training trajectories.
2. **Or: replace gated cascade with continuous form.** `σ₂ · σ₃ · Φ_components` with σ as smooth sigmoids of (EPS - Φ) instead of strict thresholds. Removes the cliff that creates wrong-direction gradient.
3. **Or: add Φ_terminal-clamp scrutiny at e=0.20.** Verify the clamp is working at terminal events; if not, re-implement.
4. **REL_VEL_TOL annealing** (deferred from Phase 5b Step 1's spec). Currently REL_VEL_TOL=50 m/s; at e=0.20 the natural relative velocity range is 100s of m/s. Wider tolerance early, narrow as training progresses.

Each is a small targeted intervention. Phase 6 should test (1) first — it's the cheapest and most likely effective.

---

## Lessons

The prediction-test discipline worked here:

- Prediction 1 was wrong about "stopping" but its falsification (more burns, less Δv) pushed toward the right mechanism (active wrong-direction).
- Prediction 2 confirmed gate failure but, more importantly, exposed the Goodhart pattern in the cumulative-shaping data.
- Prediction 3 was the smoking gun: cross-distribution destruction is incompatible with "specialization" framing.

This is the kind of mechanism work that the cap-tail probes did at Step 1 — testable predictions that produce informative falsifications. **The "Goodhart-induced collapse at e=0.20" finding is the strongest mechanism story Phase 5b has produced.**

It also means the e=0.20 ceiling is *fixable*, not fundamental. Phase 6 has a concrete repair target.

---

*Three predictions tested, two confirmed, one informatively falsified. The Stage 4.2 collapse mechanism is identified: Goodhart-induced policy degeneration when the gated NHR shaping fails to activate at high e. Specific fix candidates exist; Phase 6 has a concrete repair target.*
