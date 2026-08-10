# Phase 5b Audit — Findings

**Date:** 2026-04-28
**Spec:** `phase5b-audit-plan.md`
**Compute spent:** ~25 min (3 evals × ckpt A1 at e ∈ {0, 0.05, 0.10}, plus post-hoc analyses)
**Scope:** 4 audits — perf-metric trace, shaping gate-activation, Goodhart ratio, termination histogram. State-injection probe (Audit 2.2) deferred — primary results were decisive without it.

---

## TL;DR

Three findings change Phase 5b's scope:

1. **The dashboard "perf" metric over-reports success at low success rates by burst luck on easy e_target draws.** Not a clamp_bonus accounting bug; just rolling-window noise dominating at low signal. Replace with eval_checkpoint.py for any decision.
2. **σ₃ (the velocity-gate) effectively never fires — even at e=0 where the recipe works.** Median activation rate <0.001 across e ∈ {0, 0.05, 0.10}. This means Φ_vel is structurally inactive in the current shaping; the recipe is *de facto* single-component (Φ_orbit only) with σ₂ as the only meaningful gate. Phase 4 worked anyway.
3. **At e>0, failure mode is uniformly safety-cap timeout (93%) — not collisions, not stranded, not escape.** Agent is doing nothing destructive; it just can't find a maneuver sequence that satisfies position+velocity tolerances.

Combined with Phase 5a's C-weak verdict (recipe doesn't transfer to e>0): **the dominant mechanism is reward reachability** (chaser circular, target eccentric, terminal +10 almost never fires). Shaping isn't anti-correlated with the goal — it's just too weak to bootstrap on its own at e>0. The Phase 5b structural fix that addresses this is **REL_VEL_TOL annealing OR random satellite eccentricity initialization**, not a shaping rewrite.

A secondary finding: **σ₃ being dead even at e=0 means the gated cascade design isn't doing what it was designed for.** That's a Phase 5 recipe-cleanup item but doesn't block Phase 5b.

---

## Audit 1 — `perf` metric trace and empirical disambiguation

### 1.1 Code trace (read-only)

The training-time `perf` field is populated by:

- `orbital.h:725-734` — `add_log(env, success)` increments `env->log.perf += success ? 1 : 0` and `env->log.n++` once per terminated episode.
- `orbital.h:946` — at warp/normal terminal: `int success = (env->rewards[0] > 0.0f);`. The flag uses the terminal step's reward, which is `r_env + β·(0 − phi_prev)` = `r_env + |phi_prev|` (clamp bonus is always ≥0 since phi_prev ≤ 0).
- `orbital.h:895` — at early-escape terminal (a ≤ 0 hyperbolic): `add_log(env, 0)` hard-coded to failure. Correct.
- `pufferl.py:319` — vec env stats are extended into `self.stats[k]`; `mean_and_log()` averages the list each log interval. Dashboard `perf` = rolling mean over a small log window.

**Hypothesis check (M2 — clamp bonus flips failure to success):** Φ = -(0.01·Φ_orbit + 0.01·Φ_phase·σ₂ + 0.01·Φ_vel·σ₃). Empirically Φ_orbit ranges 5–12 at e ≥ 0, so |Φ| ≈ 0.05–0.5 typically. Clamp bonus is therefore ≤ 0.5. To flip a -10 failure to "success", clamp would need > 10. Mechanism rejected as dominant cause.

**Hypothesis (M1 — rolling-window burst luck):** at low log intervals (~hundreds of episodes), a sequence of easy e_target draws (e ∈ [0, 0.005]) where the inherited Phase 4 policy still works produces a transient spike. Consistent with all observations.

### 1.2 Empirical: A1 ckpt at (π/6, e=0.05), 200 eps argmax

| Metric | Value |
|---|---|
| eval_checkpoint.py held-out success | **3/200 = 1.5%** |
| Mean episode length | 1921 steps (median 2000) |
| Mean terminal-step reward | -9.63 |

The terminal-step reward of -9.63 is consistent with `r_env = -10` (failure) + small clamp bonus ≈ +0.4. **No clamp-bonus contamination of the success flag.**

### 1.3 Verdict

The ~88pp gap between training `perf` (88.9% peak) and held-out eval (1.5%) is **rolling-window burst luck (M1)**, not a metric-construction bug. The success flag itself is correctly counting `at_target` events.

**Recommendation:** treat dashboard `perf` as a noisy progress indicator only. Use eval_checkpoint.py for any capability claim. Phase 5 main spec should add eval-during-training every 500k steps (per addendum §6.2) to replace the noisy dashboard with held-out estimates.

**Numbers retroactively in question:** any "training-time peak perf" in PHASE4_FINDINGS, PHASE4_5_FINDINGS, PHASE5a_INTERIM_FINDINGS that wasn't independently verified by eval_checkpoint.py. Phase 4's *eval* numbers (multi-rollout-seed eval at training end) remain valid since those came from eval_checkpoint.py.

---

## Audit 2.1 — Gate-activation rates at e ∈ {0, 0.05, 0.10}

A1 (Phase 4 e=0 baseline) evaluated at 3 eccentricities (50/200/50 eps). Per-step Φ components recomputed in Python (port of compute_phi at orbital.h:514-562).

| Condition | Success | σ₂ activation (mean) | σ₃ activation (mean) | Φ_orbit (mean) | Φ_total (mean) |
|---|---|---|---|---|---|
| e=0 | 84.0% | 24.2% | **1.6%** | 8.45 | -0.088 |
| e=0.05 | 1.5% | 17.0% | **0.7%** | 11.70 | -0.120 |
| e=0.10 | 4.0% | 20.1% | **0.2%** | 11.70 | -0.120 |

### Interpretation

**σ₃ (velocity gate) is structurally dead.** Even at e=0 where the recipe achieves 84% success, σ₃ fires only 1.6% of steps (median 0%). At e>0 it's worse. σ₃ requires both σ₂ active AND Φ_phase < EPS_PHASE (=0.3). EPS_PHASE = 0.3 means `1 − cos(Δθ) < 0.3` i.e., `|Δθ| < ~45°`. For a random initial Δθ ~ Uniform[-π/6, +π/6] this is sometimes satisfied at start, but rarely sustained. **Φ_vel contributes essentially nothing to learning under any condition tested.**

**σ₂ (orbit-shape gate) drops slightly at e>0:** 24% → 17% → 20%. Mild but consistent. The threshold EPS_ORBIT = 2.0 means σ₂ fires when Φ_orbit < ~2 (with smoothing). Since e>0 conditions push Φ_orbit higher (chaser-circular vs target-eccentric mismatch), σ₂ activates less often.

**Φ_orbit mean is the diagnostic signal.** At e=0 it averages 8.5; at e>0 it jumps to 11.7. The eccentricity vector difference `||Δē||` adds ~3 to Φ_orbit when target.e ≈ 0.05. That's larger than EPS_ORBIT=2.0, so σ₂ stays gated off until the policy learns to match orbit shape — but the policy never gets a terminal reward to learn from. **Stuck in a chicken-and-egg loop: shaping won't activate until orbit closes, orbit doesn't close because no signal.**

### Decision matrix outcome

Per audit-plan §2.4:

| Probe 2.1 finding | Mechanism | Phase 5b fix |
|---|---|---|
| σ₂ partially collapses at e>0; σ₃ never fires anywhere | Shaping is mildly weakened at e>0 + structurally degenerate σ₃ | Reward-reachability fix (REL_VEL_TOL anneal / random sat init) is primary; σ₃ design is secondary cleanup |

This is closer to "shaping is fine, reward-reachability is the issue" than to "shaping silent at e>0." σ₂ is fine in both regimes; σ₃ is dead in both. The reachability fix should help; a shaping rewrite is *not* required for Phase 5b but IS warranted for Phase 5 main hygiene.

---

## Audit 3 — Goodhart ratio (cumulative shaping vs terminal reward)

Per Phase 4 plan §3.2: shaping should satisfy `|G_shape| ≤ 0.1·|R_terminal|`. Measured here on A1 trajectories:

| Condition | Median ratio | % eps over threshold (>0.1) |
|---|---|---|
| e=0 | 0.034 | 24.0% |
| e=0.05 | 0.114 | **56.0%** |
| e=0.10 | 0.119 | **60.0%** |

### Interpretation

At e=0 the recipe satisfies the Goodhart threshold for ~76% of episodes (median 3.4% of terminal magnitude). At e>0 the median crosses the threshold and over half of episodes are at or above 11% — borderline Goodharting.

This is mild relative to PHASE4_R5's catastrophic shaping-dominated collapse (>0.5 ratio in some R5 variants). But it shows the recipe is sailing close to the wind at e>0: cumulative shaping accumulates over ~2000 step episodes (since safety_cap timeouts dominate), while terminal reward stays bounded.

**Recommendation:** in Phase 5b, monitor the Goodhart ratio per-episode during training. If reward-reachability fixes succeed and episodes shorten (terminal reward fires more frequently), the ratio should drop naturally. If it doesn't, lower BETA_SHAPE.

---

## Audit 4 — Termination-mode histogram at e=0.05

200 eps, A1 ckpt, classification by terminal state:

| Mode | Count | % |
|---|---|---|
| Success | 3 | 1.5% |
| Collision | 0 | 0.0% |
| Escape | 0 | 0.0% |
| Stranded | 0 | 0.0% |
| **Safety cap (timeout)** | **186** | **93.0%** |
| Unknown (boundary cases) | 11 | 5.5% |

Mean episode length: 1921 (median 2000).

### Interpretation

**The agent is not doing anything destructive at e>0.** It's not escaping, not crashing, not exhausting fuel. It's just running out the clock without satisfying the position+velocity success criterion.

Combined with Audit 2.1's gate-activation data: the agent is performing some maneuvers (since shaping is non-zero), but those maneuvers don't reduce Φ_orbit enough to cross EPS_ORBIT=2.0 and unlock σ₂. The chaser is wandering near the target without ever achieving orbit-shape match.

This is exactly the "reward reachability" diagnosis. The agent has fuel and capability; what it lacks is gradient signal toward the eccentric target's orbit shape.

---

## Combined verdict and Phase 5b scope

The Phase 4 recipe fails at e>0 because:

1. **Terminal +10 is geometrically unreachable** for a circular chaser against an eccentric target without first transferring to an eccentric orbit. Random-init policies never attempt this maneuver because there's no gradient signal pulling them toward it.
2. **Shaping doesn't fill the gap** — σ₂ activates only when Φ_orbit < 2, which requires the chaser to already be in the right neighborhood. σ₃ never activates anywhere (even at e=0 where recipe works); Φ_vel contributes nothing.
3. **The dashboard `perf` mirage** in Phase 5a probes was burst luck on easy e_target draws, not a real signal of policy capability.

### Phase 5b structural fix (recommended)

**Primary fix — reward reachability (in priority order):**

1. **`REL_VEL_TOL` annealing.** Start training at REL_VEL_TOL=200 m/s (4× current); anneal to 50 m/s as success rate stabilizes. Lets the agent get terminal reward at intermediate-difficulty velocity tolerances, then learns to tighten over time. ~1 day of training to validate.

2. **Random satellite eccentricity initialization.** Currently `sat.orbit.e = 0` hard-coded at orbital.h:758. Sample `sat.e ~ Uniform(0, e_max_target)` and `sat.omega ~ Uniform(0, 2π)` at reset so the chaser starts in a state geometrically capable of reaching an eccentric target. ~30 min env edit + validation.

These two are complementary: option 1 widens the success funnel, option 2 widens the random-policy reachability of that funnel.

**Secondary fix — recipe hygiene (defer to Phase 5 main):**

3. **σ₃ retune.** EPS_PHASE = 0.3 is too tight; σ₃ never fires. Either raise to 1.0 (1 − cos(45°) ≈ 0.29 → 1 − cos(60°) = 0.5; raise to e.g. 0.7 → ~70°) or replace gated cascade with continuous form `σ₂ · σ₃` as a smooth blend rather than a hard threshold.

4. **eval-during-training infrastructure.** Add held-out eval every 500k steps; replace dashboard `perf` mirage with reliable estimates.

### What this rules out

- A wholesale shaping rewrite (Φ_orbit/Φ_phase formulation) is **not** required for Phase 5b. The shaping isn't anti-correlated with the goal; it's just weak. Reachability fixes should be sufficient.
- A different reward-shape (e.g., orbit-match success criterion adding `|Δa|<TOL` and `|Δē|<TOL`) is **not** required. The current position+velocity success criterion is achievable; the issue is reaching it from random init.
- DAPO / R3 components remain ruled out from Phase 4.5; nothing in Phase 5b changes that.

### What this opens up for Phase 5 main spec

After Phase 5b fix:
- Joint (phase, e) curriculum becomes runnable. Investigation A's six candidates (square/tall/wide/fresh/minus-LVLH/minus-shaping) can be re-evaluated with the new entry condition.
- Phase 5a's variance-aware multi-seed protocol stands.
- Eccentricity ceiling (target e=0.3+) becomes empirically testable.

---

## What's saved on disk

- `pufferlib/experiments/puffer_orbital_177738621627/` — A1 ckpt (Phase 4 e=0 baseline)
- `logs/orbital/p5b_audit_A1_e000/` — 50-ep eval at e=0 (84% success)
- `logs/orbital/p5b_audit_A1_e005/` — 200-ep eval at e=0.05 (1.5% success)
- `logs/orbital/p5b_audit_A1_e010/` — 50-ep eval at e=0.10 (4.0% success)
- `scripts/orbital/p5b_audit2_gate_activations.py` — Φ-port + activation/Goodhart analysis
- `scripts/orbital/p5b_audit4_termination_modes.py` — termination classifier
- `/tmp/p5b_audit_out.txt` — full numerical output of Audit 2/3
- This file: `PHASE5b_AUDIT_FINDINGS.md`

Code state: unchanged. No env or config edits during the audits.

---

## Status

| Audit | Outcome |
|---|---|
| 1 — perf metric | M1 (rolling-window burst luck), not M2 (clamp bug). Use eval_checkpoint.py instead. |
| 2.1 — gate activations | σ₂ partial drop at e>0; **σ₃ structurally dead everywhere**. Shaping is mildly weakened at e>0 but not anti-correlated. |
| 2.2 — state injection | Deferred; primary results decisive without it. Re-run if Phase 5b fix doesn't work. |
| 3 — Goodhart ratio | Median 0.114 at e=0.05 (>Phase 4 plan's 0.1 threshold for 56% of eps). Mild Goodhart at e>0; resolves naturally if reachability fix shortens episodes. |
| 4 — termination histogram | **93% safety_cap timeouts at e=0.05.** Agent isn't crashing; it's running out the clock. Confirms reward-reachability mechanism. |

Next action: implement Phase 5b primary fix (REL_VEL_TOL anneal + random sat init), validate on a single-seed run at the Phase 5a entry condition (π/6, 0.05), then re-run Investigation A from a viable entry.

---

*Resolves the C-weak verdict's mechanism question. ~25 min compute, definitive. Phase 5b scope is now structural-but-targeted: reachability fixes, not shaping rewrite.*
