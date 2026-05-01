# Phase 5 Closure + Web Frontend Prep

> **Status:** 2026-05-01. Phase 5e produced a working two-body rendezvous agent (90.2% multi-seed at e=0.20, partial extension to e=0.70). Before declaring Phase 5 closed and pivoting to Phase 6 (multi-body), this spec captures the work needed to: (1) finalize Phase 5's deliverable validation, (2) preserve the data the web frontend will need, (3) consolidate the recipe and the project narrative. Total ~10-15 hours of work, mostly engineering and analysis rather than training.

---

## 0. Why this spec exists

Two distinct concerns are at risk of being lost if Phase 5 closes without addressing them:

**Concern 1 — The Phase 5e deliverable's high-e capability is single-seed.** II.D from the Phase 5e Block II findings deferred the multi-seed e-scan (5 seeds × 6 e_max levels = 6000 episodes). The 84.5% at e=0.50 and 64.5% at e=0.70 are screening signals from one seed. Single-seed numbers don't qualify as shippable given the project's history of single-seed surprises. Closure A confidence depends on resolving this.

**Concern 2 — Web frontend data preservation is downstream of training infrastructure that's about to go cold.** The training runs are done; ckpts are on disk; wandb logs are accessible. As Phase 6 starts, the temptation will be to focus forward and treat data preservation as "later." Each piece of data that isn't exported now becomes harder to retrieve later (storage costs, deprovisioning, organizational drift). The web frontend's value depends on data that's currently captured but not in the form the frontend needs.

This spec addresses both concerns before Phase 5 formally closes.

---

## 1. The Phase 5 deliverable — what we have

After Phase 5e, the deliverable is:

**Recipe:** Phase 4 components (Discrete(10) actions including warp-5min, gated NHR shaping with terminal Φ-clamp, LVLH-frame observations) + Phase 5b additions (random satellite eccentricity init, two-stage curriculum Stage 1.0 → Stage 4.0) + Phase 5e fix (`valid_init_only=1` rejection sampling).

**Capability:**
- e=0.05 fully random: 97% (5-seed multi-seed)
- e=0.20 fully random: 90.2% ± 2.1% (5-seed multi-seed)
- e=0.30 fully random: 93% (single-seed; multi-seed pending)
- e=0.50 fully random: 84.5% (single-seed; multi-seed pending)
- e=0.70 fully random: 64.5% (single-seed; multi-seed pending)
- Phase 4 conditions (e=0): 92% (4.4pp generalization tax from Phase 5b's 96.4%)

**Open:** multi-seed at e ≥ 0.30, capability surface across (phase × e × sat-target relation) at high e.

---

## 2. Block A — Final validation (compute ~1-2 hours)

### 2.A1 — Multi-seed e-scan

Per Phase 5e Block II.D: 5 seeds × 6 e_max levels × 200 eps each.

Use the 5 Phase 5e seed checkpoints. For each, run held-out eval at each of e_max ∈ {0.05, 0.10, 0.20, 0.30, 0.50, 0.70}. 200 episodes per (seed × e_max) cell.

**Output:** a per-(seed, e_max) success matrix. Aggregate as mean ± std per e_max level.

**Success criteria for closure:**
- e=0.20: ≥ 80% multi-seed mean (current single-figure: 90.2% mean)
- e=0.30: ≥ 70% multi-seed mean
- e=0.50: ≥ 50% multi-seed mean (consistent with the spec stretch)
- e=0.70: report whatever it lands at; this is genuinely stretch territory

**Cost:** 6000 evals at ~100k SPS ≈ 30-60 min compute.

### 2.A2 — Capability surface eval at the headline ckpt

Pick one canonical Phase 5e ckpt (e.g., seed 42's epoch 325, the one with 92% at e=0.20). Run a capability surface eval:

| Axis | Values |
|---|---|
| `phase_gap` | 30°, 90°, 150°, 180° |
| `e_target` | 0, 0.05, 0.20, 0.50 |
| `e_sat` | 0, 0.05, 0.20, 0.50 |
| `sat_target_relation` | same_orbit, different_a, different_ω, fully_random |

256 cells. 200 eps × 3 rollout seeds = 600 eps per cell. Total ~150K episodes. At ~100k SPS, ~25 min.

**Output:** heatmap + cell-by-cell stats. Look for:
- Graceful degradation across all axes (not cliffs)
- Phase axis flat (consistent with Phase 5b finding that phase generalizes)
- Eccentricity axis as the dominant difficulty driver
- Whether the same_orbit corner has different behavior than fully_random at high e

This is the headline visual artifact for the web frontend's "Phase 5 deliverable" page.

### 2.A3 — Phase 4 conditions multi-seed

Per the post-extend lesson: at every major stage, eval at Phase 4 conditions to verify no low-e capability regression. Phase 5e showed -4.4pp at single-seed; multi-seed should confirm this is consistent, not a single-seed artifact.

5 seeds × 200 eps each at Phase 4 conditions (e=0, sat circular, target circular at different a, no `valid_init_only`).

**Cost:** 1000 episodes ≈ 5 min.

**Output:** mean ± std at Phase 4 conditions. Document the generalization tax.

---

## 3. Block B — Data audit and export pipeline (~3-4 hours engineering)

### 3.B1 — Trajectory log format audit

Check existing `.npz` files from `eval_checkpoint.py` against the web frontend's data schema requirements (from `orbital-rl-web-spec.md`):

Required per-step fields:
- `step`, `sim_time`
- `sat_x`, `sat_y`, `sat_vx`, `sat_vy`
- `sat_a`, `sat_e`, `sat_theta`, `sat_omega`
- `fuel`, `action`, `reward`, `delta_v`
- `target_x`, `target_y`, `target_a`, `target_e`, `target_omega`, `target_theta`
- `min_conj_dist` (for debris cases; can be NaN otherwise)

Required per-episode metadata:
- `episode_id`, `success` (bool), `termination_mode` (success/escape/collision/stranded/cap)
- `total_dv` (sum of |Δv| across burns)
- `num_steps` (terminal step)
- `init_e_sat`, `init_e_target`, `init_phase_gap` (sampled values)
- `init_sat_a`, `init_target_a`, `init_sat_omega`, `init_target_omega`
- `e_max_target`, `e_max_sat`, `same_orbit_init`, `valid_init_only` (env config used)

**Action:**

1. Open one canonical existing `.npz` (Phase 5e Stage 4.0 best ckpt eval). List its fields.
2. Compare against the required list. Note gaps.
3. For each gap, decide: (a) can be computed post-hoc from existing fields, (b) needs to be added to the env's logging, (c) needs to be derived from training metadata.

If category (b) gaps exist, modify `binding.c` / `orbital.h` trajectory record to log them, then re-run eval on key ckpts to regenerate. This is the most expensive case; bound the scope by only fixing what's actually missing.

**Cost:** ~30 min audit + ≤ 2 hours of any needed re-logging.

### 3.B2 — Build `export_web_data.py`

Converts `.npz` trajectory files into the JSON format the web frontend expects. Schema from the web spec:

```json
{
  "episode_id": 42,
  "metadata": {
    "checkpoint": "puffer_orbital_177765655537/model_puffer_orbital_000325.pt",
    "phase": "phase_5e_stage_4_0",
    "success": true,
    "termination_mode": "success",
    "total_dv": 282.3,
    "hohmann_dv_estimate": 90.1,
    "fuel_efficiency": 0.319,
    "num_steps": 43,
    "init_phase_gap_deg": 87.3,
    "init_e_target": 0.18,
    "init_e_sat": 0.12
  },
  "initial": {
    "sat_a_km": 6771,
    "sat_e": 0.12,
    "sat_omega_rad": 1.04,
    "target_a_km": 7171,
    "target_e": 0.18,
    "target_omega_rad": 2.31
  },
  "steps": [
    {"t": 60, "x": 6771000, "y": 0, "a": 6771000, "e": 0.001, "theta": 0.05,
     "fuel": 0.98, "action": 1, "dv": 10, "min_debris_dist": null}
    // ... per step
  ],
  "bodies": [
    {"type": "earth", "x": 0, "y": 0, "hard_r": 6371000, "keepout_r": 6571000}
  ],
  "target": {"a": 7171000, "e": 0.18, "omega": 2.31}
}
```

**Implementation:**
- Read `.npz` with `np.load`
- Compute per-episode metadata (sum Δv for total_dv, etc.)
- Hohmann Δv estimate: simple closed-form for circular endpoints; treat eccentric cases as an approximation
- Optional downsampling for long episodes (every 2nd step) to keep JSON size reasonable
- Output one JSON per episode under `web_data/runs/`

**Cost:** ~2 hours engineering. Test on 5-10 episodes from the canonical Phase 5e ckpt.

### 3.B3 — Training curve export

WandB has reward, success rate, episode length curves for every Phase 5 training run. The web frontend wants standalone JSON files (no wandb API at frontend runtime).

**Implementation:**
- Use wandb's API to fetch each run's metrics
- Per run, output a JSON file with `{step: int, reward: float, success_rate: float, episode_length: float}` arrays
- Save under `web_data/curves/`
- Annotate key runs with metadata: which phase, which seed, what conditions

Specifically, export curves for:
- Phase 4 baseline runs (~3 representative seeds)
- Phase 5b Stage 1.0 multi-seed (5 seeds)
- Phase 5b Stage 4.0 multi-seed (5 seeds)
- Phase 5e Stage 4.0 multi-seed under `valid_init_only=1` (5 seeds)
- (Optional) selected Phase 5c/5d failed-intervention runs for the "ceiling investigation" narrative

**Cost:** ~1 hour engineering, mostly wandb API + JSON serialization.

### 3.B4 — Intermediate-checkpoint eval for "training progression"

For one canonical Phase 5e seed (e.g., seed 42), eval at intermediate checkpoints to capture the *learning trajectory* the web frontend needs for its "before/after" comparisons.

Checkpoints: epoch 25, 50, 100, 150, 200, 250, 325 (best). 50 episodes each at the canonical eval condition (e=0.20 fully random).

For each checkpoint, save:
- Held-out success rate at that epoch
- 5-10 representative trajectory `.npz` files showing what the policy at that epoch looks like

This produces the data for the "agent learns over training" narrative. Without this, the web frontend can show "trained policy works" but not "here's how it got there."

**Cost:** ~30 min eval + ~30 min trajectory selection.

---

## 4. Block C — Curated demo episodes (~2-3 hours)

The web frontend's run explorer needs hand-picked example trajectories. Picking them isn't automatic; it requires human judgment about what's interesting.

### 4.C1 — Selection criteria

Per the original web spec, curate 3-5 trajectories per milestone covering:
- Clean textbook-like success (the "this is what success looks like")
- Interesting strategic episode (e.g., natural-phasing exploitation at high e)
- Failure case (edge case, illustrative of what's hard)
- Before/after comparison (early-training vs final-ckpt on the same scenario)

### 4.C2 — Milestones to curate

Phase 5 milestones for the web frontend:

1. **Phase 4: Circular rendezvous specialist** (working baseline)
   - 1 textbook Hohmann-like success
   - 1 long-phasing success showing patience
   - 1 failure (likely a phase=180° case that didn't converge)

2. **Phase 5b Stage 4.0: Eccentric-target rendezvous at low e**
   - 1 success showing natural-phasing strategy at moderate e
   - 1 success at near-circular config (tests low-e robustness)
   - 1 failure (likely at the e=0.05 hard corner)

3. **Phase 5e Stage 4.0 with valid_init_only**: The deliverable
   - 1 success at e=0.20 with phase=180°
   - 1 success at e=0.50 (zero-shot transfer demonstration)
   - 1 success at e=0.70 (Molniya-class)
   - 1 failure for capability-surface honesty

4. **Bonus: training progression (one canonical seed)**
   - epoch 25 trajectory (early training, likely fails)
   - epoch 100 trajectory (mid training)
   - epoch 325 trajectory (final, success)

### 4.C3 — Trajectory inspection method

For each candidate episode, look at:
- Initial conditions (phase gap, eccentricities, ω alignment)
- Trajectory shape (Hohmann-like vs phasing orbit vs natural-phasing exploitation)
- Fuel usage pattern
- Termination mode and reason

Pick episodes that *teach the viewer something*, not just "high success rate." A boring 99% success episode is less informative than a 50% success that demonstrates an interesting strategy.

**Cost:** ~2-3 hours of trajectory viewing + selection. Save selected episodes to `web_data/runs/curated/` with metadata.

---

## 5. Block D — Recipe documentation (~2 hours)

Single document that describes the final Phase 5 recipe completely. Future-self / portfolio readers need this to understand what was actually built.

### 5.D1 — `RECIPE.md`

Structure:

1. **Environment**: action space, observation space, episode termination, success criterion, reward structure (gated NHR shaping with formulas), env config knobs
2. **Curriculum**: Stage 1.0 → Stage 4.0, hyperparameters per stage, training duration to convergence
3. **Recipe-specific decisions and why**: 
   - Why Discrete(10) with warp-5min
   - Why LVLH-frame observations
   - Why gated NHR (not flat shaping or PB-only)
   - Why random sat init + same_orbit_init Stage 1
   - Why `valid_init_only=1` (Phase 5e finding)
4. **Hyperparameters**: full PPO config, all env kwargs
5. **Capability surface**: the headline numbers from Block A
6. **Known limits**: 
   - 4.4pp generalization tax at Phase 4 conditions
   - Single-seed only at e ≥ 0.30 (until Block A multi-seed)
   - e=0.70 partial (64.5%, multi-seed pending)
   - Action discretization marginal at e ≥ 0.50 (Phase 5e Block I E6)

### 5.D2 — `PHASE5_FINDINGS.md` (consolidated)

Single document covering Phase 5b through 5e. Honest narrative:

- Phase 5b: shipped working agent at e ≤ 0.10 with two-stage curriculum (this stands)
- Phase 5c: investigated ceiling at e ≥ 0.20, ran 8 reward-side interventions (B1-B6 + I4 variants), produced multiple mechanism stories that turned out to be Simpson's-paradox-confounded or otherwise misattributed
- Phase 5d: continued ceiling investigation, identified early-death failure mode dominance, tested additional reward-side interventions (collision penalty), all collapsed
- Phase 5e Block I: found `valid_init_only` issue (~64% of e=0.20 raw inits had sub-keepout perigee), 1.5 hours of post-hoc env validation
- Phase 5e Block II: with the env fix, 5/5 seeds reach 86-92% at e=0.20, capability surface extends well beyond original Phase 5 targets

The methodological lesson, named explicitly: when a system you've validated under condition X starts misbehaving under condition Y, validate the system under condition Y before designing interventions. Phase 1's e=0 validation tests should have been re-run at e>0 before Phase 5c committed to mechanism investigations.

This isn't a self-flagellation document. It's an honest accounting of how the project actually went, with the corresponding methodological takeaways.

---

## 6. Block E — Sequencing

In order:

1. **Block A.A1** (multi-seed e-scan) — 30-60 min compute. Resolves the multi-seed deliverable question.
2. **Block A.A2** (capability surface) — 25 min compute. Produces the headline visual artifact.
3. **Block A.A3** (Phase 4 conditions multi-seed) — 5 min compute. Confirms generalization tax is consistent.
4. **Block B.B1** (trajectory format audit) — 30 min. Identifies what's missing in `.npz` schema.
5. **Block B.B4** (intermediate-checkpoint eval) — 1 hour, parallelizable with B1. Captures training progression data while infrastructure is warm.
6. **Block B.B2** (`export_web_data.py`) — 2 hours engineering. Converts `.npz` to JSON.
7. **Block B.B3** (training curve export) — 1 hour engineering. Wandb → JSON.
8. **Block C** (curated demos) — 2-3 hours. Trajectory inspection and selection.
9. **Block D** (recipe + findings docs) — 2 hours writing.
10. **Phase 5 closure** — formal close.

Total ~10-15 hours wall, plus ~1.5 hours compute. Spread over ~1-2 weeks elapsed.

The sequence puts compute-blocking work first (Block A) so analysis blocks (B-D) can run while compute is busy. Block B's audit (B1) happens early to identify any logging gaps before re-runs become expensive.

---

## 7. What this spec is NOT

- **Not the web frontend itself.** The frontend implementation (per `orbital-rl-web-spec.md`) is post-Phase-5-closure work. This spec is data preservation + closure, not frontend engineering.
- **Not Phase 6 planning.** Phase 6 starts after Phase 5 closes. This spec is the gate.
- **Not a re-investigation.** The Phase 5e findings stand. This spec validates them more thoroughly and preserves the data, but doesn't re-litigate the recipe.
- **Not portfolio-narrative writing.** The eventual blog post / tweet thread / portfolio piece is downstream of `PHASE5_FINDINGS.md`. The findings doc is technical; the portfolio is narrative.

---

## 8. Pre-committed decision points

**If Block A.A1 reveals high-e variance is much larger than expected (e.g., e=0.50 single-seed at 84% but multi-seed is 50% ± 25pp):**
The single-seed numbers were overconfident. Phase 5 still ships at the strong e ≤ 0.30 multi-seed result, with e ≥ 0.50 marked as "high variance, see surface." Don't fudge the numbers.

**If Block A.A2 capability surface shows cliffs or corners that the aggregate multi-seed numbers hid:**
Document the surface honestly. The "graceful degradation" claim might need to be revised to "graceful in some axes, has corners in others." Truth over narrative.

**If Block B.B1 reveals significant trajectory-logging gaps:**
Re-run eval on key ckpts with extended logging. Bound the scope: only re-run for the ckpts that need to feed the web frontend (Phase 5e best, Phase 5b Stage 4.0 best, Phase 4 baseline best, intermediate checkpoints from one seed).

**If Block C reveals the recipe doesn't actually produce many "interesting" trajectories at high e (e.g., the agent converges to similar maneuver templates regardless of conditions):**
That's still an honest finding. The web frontend can show consistent strategy as a feature (the agent's policy is robust) rather than diverse strategy as a feature (the agent finds creative solutions). Either narrative is defensible.

---

## 9. Phase 6 readiness items

While Phase 5 closure work runs, two refactors enable Phase 6 cleanly:

### 9.E1 — Trajectory logging extensibility (~2 hours)

Current trajectory record assumes 2D Kepler with sat + target + (optional) debris. Phase 6 multi-body needs N gravitating bodies with positions. Generalize the trajectory record to support arbitrary body lists.

Concretely: replace the fixed `body_x[MAX_BODIES]`, `body_y[MAX_BODIES]` arrays with a parameterized list. Phase 5 still works (uses 1-2 bodies); Phase 6 works (uses 3+ bodies).

### 9.E2 — Curriculum scheduler abstraction (~1 hour)

Replace hard-coded stage transitions in the training scripts with a Python class wrapping the bash orchestrator. Phase 6 reuses with multi-body env_kwargs.

The Phase 5 scripts (`p5e_curriculum.sh` etc.) do this work in bash. Wrapping in Python with a config-driven schedule is a small refactor that pays off in Phase 6.

These are concurrent with the closure work; not blocking.

---

## 10. Closure conditions

Phase 5 is formally closed when:

1. Block A multi-seed and capability surface complete. Results consolidated into `PHASE5_FINDINGS.md`.
2. Block B trajectory data exported to `web_data/` in JSON format covering: 50+ canonical Phase 5e episodes, 20+ Phase 5b episodes, 10+ Phase 4 baseline episodes, training-progression episodes from one seed.
3. Block B training curves exported for all multi-seed runs.
4. Block C curated demos selected and tagged.
5. Block D `RECIPE.md` and `PHASE5_FINDINGS.md` complete.
6. Phase 6 readiness refactors (E1, E2) complete.

Once these conditions hold, Phase 6 can start without leaving Phase 5 data orphaned.

---

*Author: 2026-05-01. Phase 5 closure + web frontend prep, after Phase 5e Block II's deliverable confirmation. Successor: PHASE5_FINDINGS.md + Phase 6 transition spec.*
