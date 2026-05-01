# Phase 5b Audit Plan — Training Metric and Shaping Diagnostics

> **Status:** 2026-04-28. Pre-Phase-5b audits, run before any further training. Two read-only investigations to determine (1) what the training-time `perf` metric actually measures, and (2) whether gated NHR shaping produces usable gradient signal at e>0. Total wall time: ~2 hours, mostly code reading.

---

## Why these audits, before any more training

Phase 5a's A2 diagnostic surfaced an 88pp gap between training-time `perf` (88.9% peak) and held-out eval (1% sampled). This invalidates several training-time numbers in earlier phases as inputs to design decisions, and the C-weak verdict on the Phase 4 recipe at e>0 raises a separate question: is the recipe producing *any* useful gradient signal at e>0, or are we training on a mis-specified objective?

Both are answerable by code reading and trivial empirical checks, before committing further compute. The cost of *not* doing them is that Phase 5b's structural changes get applied to a recipe whose existing components may be the actual problem, producing the same Goodhart-then-mirage pattern under a different name.

---

## Audit 1 — What does the training-time `perf` metric measure?

### 1.1 The question

The dashboard `perf` field reported 88.9% peak during A2 training. Held-out 100-episode eval at the same condition gave 1% sampled / 0% argmax. A 88pp gap between the two means the dashboard is measuring something other than terminal-rendezvous success rate, or measuring it in a way that systematically inflates at low success rates. We need to know which.

### 1.2 Three candidate mechanisms (from the diagnostic post-mortem)

- **M1 — Rolling-window burst luck.** Dashboard averages over a small rolling window of completed episodes; stochastic-action burst on a sequence of easy task draws spikes the metric.
- **M2 — Reward-vs-success divergence.** Dashboard measures positive-total-return rather than terminal-success-only. Shaping-positive trajectories that fail to rendezvous get counted.
- **M3 — Termination-bias.** Dashboard averages over `terminated` episodes preferentially, biasing toward early-terminating success cases vs late-truncating timeouts.

These aren't mutually exclusive.

### 1.3 What to read

Trace the codepath that populates the dashboard's `perf` field. Likely starting points (verify against actual paths):

- `pufferlib/pufferlib/cleanrl.py` or `pufferl.py` — main training loop. Look for where `perf` or `success_rate` is computed and logged.
- `pufferlib/pufferlib/ocean/orbital/binding.c` — env returns to Python. Look for what's exposed as the per-episode "success" signal.
- `pufferlib/pufferlib/ocean/orbital/orbital.h` — env step function. Look for what flag in the trajectory record corresponds to "success."

For each location, document:

1. **Definition.** What event triggers a "success" count? Terminal +10? Any positive-reward episode? Episode that didn't end in the four failure modes (collision/escape/stranded/safety-cap)?
2. **Aggregation.** How is the metric computed across episodes? Rolling mean? Cumulative? Window size?
3. **Termination-handling.** Are truncated (timed-out) episodes counted? At what value? Are they excluded?

### 1.4 What to verify empirically

After tracing the code, run a 30-second empirical check:

```python
# Pseudocode — adapt to the actual env API
env = make_env(num_envs=64, init_phase_gap_max=0.524, e_max_target=0.05)
policy = load_ckpt('A1_ckpt')

success_terminal = 0       # terminal-success count by hard-coded definition
success_dashboard = 0      # whatever the dashboard's `perf` would count
total_episodes = 0

for step in range(50_000):  # ~50 episode-lengths
    actions = policy.sample(obs)
    obs, rewards, terms, truncs, infos = env.step(actions)
    for env_i in range(64):
        if terms[env_i] or truncs[env_i]:
            total_episodes += 1
            # whatever the C env reports as "success"
            if infos[env_i]['terminal_reward'] > 0:
                success_terminal += 1
            # whatever the dashboard counts (this is what we need to read out of the codepath)
            if dashboard_success_logic(env_i, rewards, terms, truncs):
                success_dashboard += 1

print(f"Terminal: {success_terminal/total_episodes:.1%}")
print(f"Dashboard: {success_dashboard/total_episodes:.1%}")
```

If the two numbers differ by more than 5pp, the dashboard isn't measuring what we thought, and the magnitude of difference characterizes the bug.

### 1.5 Decision matrix

| Code reads | Empirical check |
|---|---|
| Dashboard counts terminal +10 only, rolling window short | Run on A1 at e=0.05; if dashboard shows >10% while terminal shows <2%, M1 confirmed. Fix: replace `perf` with eval_checkpoint.py for any decision; document the dashboard caveat. |
| Dashboard counts positive total return | M2 confirmed regardless of empirical. Fix: rewrite dashboard `perf` to count terminal-success only, OR introduce a separate `success_rate` metric and ignore `perf` for capability claims. |
| Dashboard counts non-failure-mode episodes (i.e., truncations as successes) | M3 confirmed. Fix: subtract truncations from numerator. |
| All three present in compounded form | The dashboard is non-actionable. Replace with a clean terminal-success metric. |

The fix for any of these is small (a few lines in the training loop). The expensive part was running the failed experiments before knowing.

### 1.6 Output

A short document `PHASE5b_AUDIT1_PERF_METRIC.md`:
- The three definitions (terminal success, positive return, non-failure)
- Which the dashboard actually used
- Empirical confirmation: Δ between dashboard and held-out eval on A1 at e=0.05
- Recommended fix
- Implications for which previously-reported numbers in the project are revisable

---

## Audit 2 — Is the gated NHR shaping producing useful gradient at e>0?

### 2.1 The question

The C-weak verdict says the Phase 4 recipe doesn't transfer to e>0. But "doesn't transfer" has multiple plausible mechanisms:

- The success criterion is unreachable from random init at e>0 (chaser starts circular, target eccentric, terminal reward almost never fires). This is reward-reachability — option 2/option 1 in the diagnostic findings.
- The shaping is producing gradient signal *away from* success at e>0 — i.e., the gates that worked for circular targets are now punishing necessary maneuvers for eccentric ones. This is reward-mis-specification, and it's a different problem with different fixes.

We need to know which (or both). Because if shaping is producing wrong-direction gradient, fixing reward-reachability alone won't help — we'd just be giving useful gradient to a mis-specified objective.

### 2.2 What to read

Trace the gated shaping codepath in `pufferlib/pufferlib/ocean/orbital/orbital.h`:

- `Φ_orbit` computation: how is "orbit match" measured for non-circular targets? Specifically, the eccentricity-vector formulation `||Δē||` — does it correctly handle the case where target.e > 0?
- Gate `σ₂`: at what threshold does it activate? Was the threshold tuned for target.e = 0?
- `Φ_phase` and gate `σ₃`: same questions.
- `Φ_vel`: relative velocity in what frame? With what reference?

Open question we want to specifically answer: **for a random-policy trajectory with target.e=0.05, does Φ ever decrease in a way that correlates with proximity to success?** If Φ is constant or anti-correlated with success, the shaping is producing useless or wrong-direction gradient.

### 2.3 What to verify empirically

Two cheap probes:

**Probe 2a — Gate activation rates.**

Modify trajectory logging to record (Φ_orbit, σ₂, Φ_phase, σ₃, Φ_vel) at every step. Run a random-policy episode at:
- (π/6, e=0): the regime where shaping demonstrably works
- (π/6, e=0.05): the regime where the recipe fails
- (π/6, e=0.10): mid-range

For each, compute over the trajectory:
- Mean and std of Φ_orbit (and components)
- Activation rate of σ₂ (% of steps where σ₂ > 0.5)
- Activation rate of σ₃

Compare across the three eccentricity regimes. If gate activation rates collapse (e.g., σ₂ rarely activates at e=0.05), shaping is silent at e>0 and we have a recipe-level problem.

**Probe 2b — Gradient direction.**

For a trajectory that ends in success and one that ends in failure (both at e=0.05), compute Σ(Φ_curr - Φ_prev) over the episode. The sign tells us whether shaping accumulates positive or negative reward toward success. If success trajectories accumulate negative shaping reward (or failure trajectories accumulate positive), the shaping is anti-correlated with the actual goal at e>0.

This requires a successful e=0.05 trajectory. If we don't have one (because the recipe doesn't produce them), we can construct one analytically: a known-good Hohmann + apsidal-correction trajectory for e=0.05, computed offline, replayed step-by-step through the env to extract per-step Φ values. This is ~2 hours of script-writing for a definitive answer.

### 2.4 Decision matrix

| Probe 2a result | Probe 2b result | Mechanism | Phase 5b fix |
|---|---|---|---|
| Gate activations comparable at e=0 and e=0.05 | Φ-diff sums positive on success traj | Shaping is fine at e>0; reward-reachability is the issue. | Option 1 (REL_VEL_TOL anneal) and/or option 2 (random sat init). Don't touch shaping. |
| Gate activations collapse at e=0.05 | (regardless) | Shaping is silent at e>0; gradient signal absent. | Retune gate thresholds for e>0, or replace gates with smoother continuous form. Reward-reachability fixes alone won't help. |
| Gate activations comparable | Φ-diff sums *negative* on success traj | Shaping is mis-specified at e>0 — gradient points away from goal. | Rewrite Φ_orbit/Φ_phase formulation for non-circular targets. Largest scope. |
| All four cells in some hybrid | | Multi-mechanism. | Phase 5b becomes more involved; need follow-up before recipe is fixable. |

### 2.5 Output

A short document `PHASE5b_AUDIT2_SHAPING_DIAGNOSTIC.md`:
- Gate activation rates at e ∈ {0, 0.05, 0.10}
- Φ-diff sum for success vs failure trajectories at e=0.05 (using analytic-Hohmann if no learned successes available)
- Mechanism verdict
- Recommended fix (recipe-preserving vs recipe-modifying)

---

## Sequencing

Audit 1 first, then Audit 2. Reasoning:

- Audit 1 takes ~30 min and resolves the most urgent question (can we trust *any* training-time number).
- Audit 2 takes ~2 hours, partly because if the analytic-Hohmann trajectory needs to be constructed, that's the expensive part.
- Some of Audit 2's setup (trajectory logging additions, eval harness modifications) can be done while the Audit 1 empirical check runs.

Total: ~2.5 hours wall-time. Lowest information-per-compute experiment is the trace-the-code part of Audit 1; everything else is decision-blocking.

---

## What this is *not*

- **Not Phase 5b itself.** Phase 5b is the structural-change phase that follows once we know what's actually broken. Audits inform that scope.
- **Not a re-test of Phase 4.5 attribution claims.** Even if Audit 1 finds the dashboard metric was unreliable, we're not re-running Phase 4.5 — those claims become "directionally probably right but with an explicit caveat about the metric."
- **Not a redesign of the env or recipe.** Read-only investigations + cheap empirical probes. No env code changes during the audits.

---

## Risk register

| Risk | Likelihood | Mitigation |
|---|---|---|
| Audit 1's codepath trace turns out to be PufferLib internals not easily inspectable. | Medium | Worst case, instrument the env to log per-episode terminal status to wandb directly, and compare to dashboard `perf`. Bypasses internal tracing. |
| Audit 2's gate-activation probe requires env code changes (not strictly read-only). | High | Adding logging hooks is a small env change, acceptable. Protect against state contamination by branching the change off main. |
| The analytic-Hohmann construction for Probe 2b takes longer than 2 hours. | Medium | If so, defer Probe 2b — Probe 2a alone (gate activations) is informative enough for first-pass diagnosis. |
| Audit 1 finds the dashboard is fine; Audit 2 finds the shaping is fine. | Low | Then the C-weak verdict is purely reward-reachability. Phase 5b is option 1 + option 2, simple recipe extension. |

---

## After the audits

Phase 5b spec gets drafted with audit results baked in. The key decisions for Phase 5b that the audits inform:

1. Should training-time `perf` be replaced with held-out eval-during-training? (Audit 1)
2. Does the recipe's shaping need rework, or just reward-reachability fixes? (Audit 2)
3. What's the smallest set of changes that produces useful gradient at e>0? (Both)

These three decisions shape Phase 5b's scope by 2-4×, similar to how the A2 diagnostic shaped Phase 5a's scope.

---

*Author: 2026-04-28. Pre-Phase-5b read-only audits. Successor: Phase 5b spec, drafted after audit findings.*
