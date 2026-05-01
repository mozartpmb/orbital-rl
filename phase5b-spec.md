# Phase 5b — Full Spec

> **Status:** 2026-04-29. Draft after Phase 5b Step 1 + cap-tail diagnostics + post-extend characterization. Establishes the curriculum, training protocol, evaluation methodology, and Phase 6 readiness items for completing the two-body transfer agent.

---

## 0. TL;DR

Phase 5b builds a single agent that handles **rendezvous from any orbit to any other orbit, with whatever phase and eccentricity, provided it is physically possible with given fuel** — the task framing the Phase 5b Step 1 redefinition committed to. The recipe is locked: random sat init, gated NHR shaping, LVLH obs, Discrete(10) actions, warp-5min, Phase 4 hyperparameters. The infrastructure is locked: tight checkpointing, eval-during-training (in-distribution + OOD), held-out eval-gated stage transitions. The methodology is locked: 5 seeds at headline stages, train-longer as the default response to stalls, no structural conclusions from one borderline data point.

The four-stage curriculum from Step 1's spec is reduced to a primary two-stage path (Stage 1 same-orbit → Stage 4 everything-random) with contingency branches if Stage 4 stalls. This reflects the empirical finding that Stage 1 already produces 58.7% partial transfer to Stage 4. Stage 2 (different ω only) and Stage 3 (different a only) become diagnostic fallback stages, not mandatory.

The deliverable is the trained agent plus a capability surface across `(phase_gap, e_target, e_sat, sat_target_orbit_relation)`. Target: ≥80% multi-seed at e_max=0.50 on the headline (Stage 4 fully general) eval, with capability surface showing graceful degradation rather than cliffs.

---

## 1. The reframed task and why this spec is shaped the way it is

Phase 4 was a circular-rendezvous specialist that scored 79.6% multi-seed on (sat circular, target circular at different a, full phase gap). Phase 5b Step 1 redefined the env to support random sat eccentricity, restored bootstrap on same-orbit eccentric rendezvous, and produced a 99.7% Stage 1.0 result that partially transfers to harder OOD conditions.

The underlying lesson from the project's path here: **the rendezvous task is broader than any single phase has trained on, and the recipe's transferability across the broader task is empirically generous but bounded.** Phase 5b's job is to characterize and extend that transferability across a defined capability surface.

Three principles that thread through every section of this spec, derived from project history:

**Principle I — Train to convergence, not to step count.** The cap-tail probes that produced the P3 verdict were undone by 20 more minutes of training. Step 1's "20M for Stage 1" was wrong; 40M was right. Per-stage compute should be gated on held-out eval thresholds, not on pre-budgeted step counts.

**Principle II — Borderline data does not justify structural claims.** The post-extend "60M shows erosion at Phase 4 conditions" reading was partially-supported and partially-noise; the previous review correctly stripped it. Phase 5b's evaluation will produce many borderline-significant comparisons. The discipline: report effect size and uncertainty, refuse to build narratives from <2σ effects.

**Principle III — Test the simplest hypothesis first.** Train-longer is always the first response to a stall. Recipe changes are second. Curriculum reorganization is third. New algorithmic interventions (R3-style components) stay dead unless very strong evidence demands them.

---

## 2. The locked recipe

Inherited from Phase 4, validated through Phase 5b Step 1:

### 2.1 Environment

| Component | Value | Source |
|---|---|---|
| Action space | Discrete(10): coast, prograde ±5/±10/±25 m/s, retro ±5/±10/±25 m/s, radial in/out, warp-5min (τ=5) | Phase 4 R4 |
| Observation | 38-dim with LVLH-frame relative state | Phase 5b Step 1 |
| Shaping | Gated NHR multi-stage: Φ_orbit + σ₂·Φ_phase + σ₃·Φ_vel; β=1.0, w_*=0.01 each; terminal Φ-clamp | Phase 4 R4 |
| Success | dist < 30 km AND rel_vel < 50 m/s | Phase 4 |
| Episode cap | 2000 steps (33 hr orbital time) | Phase 4 |
| Timestep | 60 s | Phase 4, validated against DT=30 alternatives |
| sat init | random sat.e ∈ [0, e_max_sat], random sat.ω; sat.a per same_orbit_init constraint | Phase 5b Step 1 |
| same_orbit_init | per-stage (Stage 1: 1; Stage 4: 0; intermediate stages variable) | Phase 5b Step 1 |

### 2.2 Hyperparameters (Phase 4)

| Parameter | Value |
|---|---|
| Algorithm | PPO with default actor-critic (no LSTM) |
| `gamma` | 0.995 |
| `lr` | 0.01 |
| `ent_coef` | 0.01 |
| `clip_range` | 0.2, 0.2 (symmetric, no DAPO) |
| `n_epochs` | 10 |
| `minibatch_size` | 8192 |
| `num_envs` | 1024 |
| `anneal_lr` | True |

### 2.3 What's explicitly NOT in the recipe

- **R3 components** (LayerNorm, L2-init, DAPO, adaptive-KL, target-entropy controller). Phase 4.5 ruled them out; Phase 5b Step 1 confirmed the recipe works without them.
- **REL_VEL_TOL annealing.** Phase 5b Step 1 confirmed not needed for Stage 1.0. Reserved as a contingency if a later stage needs it.
- **Curriculum reweighting / stratified sampling.** Train-longer suffices for the cap-tail; reweighting is a "marginal improvement" question, not "make it work" question.
- **Per-state entropy regularization, two-pass training, distillation.** P3 was wrong; these don't apply.
- **Continuous actions, finer Δv discretization, longer warp.** Phase 4.5's ruling stands.

---

## 3. The curriculum

The four-stage decomposition from the Step 1 spec is reduced to a **primary two-stage path** with contingency branches.

### 3.1 The primary path

```
Stage 1.0  (same orbit, e_max=0.05)        — bootstrap baseline, validated by Step 1
Stage 1.1  (same orbit, e_max=0.10)        — eccentricity expansion
Stage 1.2  (same orbit, e_max=0.20)        — continued expansion
Stage 1.3  (same orbit, e_max=0.50)        — high-eccentricity territory
Stage 1.4  (same orbit, e_max=0.70)        — Molniya-class stretch
Stage 4.0  (everything random, e_max=0.05) — full task at low difficulty
Stage 4.1  (everything random, e_max=0.10)
Stage 4.2  (everything random, e_max=0.20)
Stage 4.3  (everything random, e_max=0.30) — original Phase 5 plan target
Stage 4.4  (everything random, e_max=0.50) — full task at hard difficulty
```

Each stage warm-starts from the previous stage's converged ckpt. Stage 1.x progresses through eccentricity at fixed same-orbit constraint; Stage 4.x removes the constraint and expands eccentricity again.

The Stage 4.x sub-stages may need a smaller eccentricity ceiling than 1.x — different-orbit transfers at high eccentricity are physically expensive (Δv-wise). Hold the floor at e_max=0.30 for Stage 4 with stretch to 0.50.

### 3.2 Why the curriculum is shaped this way

Two structural decisions worth flagging:

**Decision A — Stage 1 fully expands in eccentricity before Stage 4 starts.** Alternative: alternate Stage 1.0 → Stage 4.0 → Stage 1.1 → Stage 4.1, etc. Reasoning for the chosen sequence: Stage 1 is faster per-stage (same-orbit constraint = simpler task = faster convergence), so expanding through e first lets us validate the recipe scales in eccentricity before adding the cross-orbit complexity. If e=0.5 fails in Stage 1, we know to debug eccentricity before adding orbit transfer.

**Decision B — Stage 2 and Stage 3 are NOT in the primary path.** The Step 1 spec proposed Stage 1 (same orbit) → Stage 2 (different ω) → Stage 3 (different a) → Stage 4 (everything random). The post-extend characterization showed Stage 1 already gets 58.7% on Stage 4 conditions, suggesting the intermediate stages may be unnecessary scaffolding. Phase 5b primary path skips them.

**Stage 2 and Stage 3 are reserved as diagnostic fallback stages.** If Stage 4.0 stalls below threshold from Stage 1 warm-start, we add them back to isolate which constraint relaxation broke things. Spec for that contingency is in §5.4.

### 3.3 Stage transition criteria

A stage is considered converged and ready to transition when:

- Held-out eval (200 eps × 3 rollout seeds = 600 episodes) reaches ≥ 80% mean success at the stage's training condition, OR
- Held-out eval rolling mean over the last 10 evals stops improving (≤ 1pp gain over 10 consecutive evals at 200K-step intervals)

The 80% threshold is the primary gate. The plateau detection is the safety net for stages that converge below 80% — if learning has plateaued, we don't keep training fruitlessly. Stage transitions on plateau-below-threshold trigger a contingency review (§5).

### 3.4 Compute budget

Step 1's evidence: Stage 1.0 needed ~40M steps to reach 99.7%. Each subsequent stage may need similar or more. Realistic per-stage budget:

- Stage 1.0: 40M (validated)
- Stage 1.1-1.3: 30-50M each (extrapolated)
- Stage 1.4 (e=0.7 stretch): up to 80M (uncharted; high-e might need more)
- Stage 4.0: 50M (warm-started from Stage 1, but with constraint relaxation)
- Stage 4.1-4.4: 30-60M each

Rough total per seed: 5×40M (Stage 1.0-1.4) + 4×40M (Stage 4.0-4.3) = ~360M steps per seed. Stretch to 4.4 adds 60M. With 5 seeds, ~2B total steps. At ~100k SPS, ~5 hours per seed for the full curriculum, ~25 hours total compute.

This is the realistic full-curriculum compute. We don't pre-budget per-stage step counts; we run until convergence per §3.3.

---

## 4. Multi-seed protocol

Per the Phase 5a addendum (variance-aware multi-seed), updated by Step 1's evidence that variance shrinks at convergence.

### 4.1 Seed allocation per stage

| Stage | Training seeds | Rollout seeds (per ckpt eval) |
|---|---|---|
| Stage 1.0 | 5 (headline result) | 3 |
| Stage 1.1, 1.2 | 3 (representative) | 3 |
| Stage 1.3, 1.4 | 5 (high-e validation) | 3 |
| Stage 4.0 | 5 (transition validation) | 3 |
| Stage 4.1, 4.2 | 3 | 3 |
| Stage 4.3 | 5 (headline result) | 3 |
| Stage 4.4 (stretch) | 5 if reached | 3 |

Headline stages (1.0, 1.3, 1.4, 4.0, 4.3) get full 5-seed treatment. Intermediate stages get 3 seeds — enough to catch obvious failures without 5x-ing every comparison.

The seeds: `42, 1337, 20260423, 2718, 31415` (per the addendum).

### 4.2 What the seeds buy us

At convergence, Step 1 measured 3.3pp std across rollout seeds, but only 1 training seed was tested. The 5-seed training protocol tells us:

- Inter-seed std at convergence (expected: ≤ 5pp based on Step 1 evidence)
- Whether any seed fails to converge (Step 1 had no failures; Phase 4 had ~1/3 retry rate, but at a less-converged regime)
- Whether different seeds converge to qualitatively different policies (worth logging trajectories from each seed)

### 4.3 Reporting protocol

Every reported number is mean ± std across 5 training seeds, OR all 5 per-seed numbers explicitly. Don't quote single-seed numbers as the headline.

For comparisons (e.g., Stage 4 warm-start vs from-scratch), use: `mean(A) − mean(B) > 1.5 × pooled_std(A, B)` to declare a winner. If within threshold, "no clear difference, both work" is the honest read.

---

## 5. The warm-start vs from-scratch question

The post-extend characterization showed Stage 1 → Stage 4 transferability of 58.7%. We don't know if this transfer is *useful* (Stage 4 trained from scratch might also reach 80%, just slower) or merely *visible* (warm-start gives a head start that disappears at convergence).

This is a real research question and Phase 5b will answer it cleanly.

### 5.1 The Stage 4 comparison run

After Stage 1 fully converges, run two Stage 4.0 paths in parallel:

- **Stage 4.0 warm:** warm-start from the best Stage 1.x ckpt, train to convergence
- **Stage 4.0 fresh:** train from scratch, same conditions, same total compute

Compare:

| Comparison | What it tells us |
|---|---|
| `time_to_convergence(warm) < time_to_convergence(fresh)` | Warm-start helps speed; curriculum is justified for compute efficiency |
| `eval(warm at convergence) > eval(fresh at convergence)` | Warm-start helps capability ceiling; curriculum is justified for ultimate performance |
| `eval(warm) ≈ eval(fresh) and time_to_convergence(warm) ≈ time_to_convergence(fresh)` | Curriculum is unnecessary; Stage 4 fresh is fine |
| `eval(warm) < eval(fresh)` | Negative transfer — warm-start *hurts*. Investigate before continuing |

This is a meaningful experiment on its own. The result informs not just Phase 5b but the broader question of when curriculum learning helps vs. is overhead.

### 5.2 If warm-start helps

Use it for Stage 4.x and report the curriculum justification with empirical support.

### 5.3 If warm-start doesn't help

Stage 4 fresh becomes the primary path. Stage 1 still has independent value (the same-orbit specialist is a deliverable in its own right) but isn't a prerequisite for Stage 4.

### 5.4 If Stage 4 stalls regardless of warm-start

Reintroduce Stage 2 (different ω only, sat.a = target.a) and Stage 3 (different a only, sat.ω = target.ω) as intermediate stages. Each isolates one constraint relaxation. If Stage 4 stalls but Stages 2 and 3 individually reach threshold, the failure is in the joint relaxation, not either constraint individually. That's a substantive finding.

---

## 6. Evaluation infrastructure

### 6.1 Eval-during-training

Required infrastructure (per the audit's recommendation, validated by Step 1):

- Eval every 200K steps during the first 5M of any stage; every 500K thereafter
- Eval condition matches training condition (same e_max, same constraint flags)
- 50 episodes × 3 rollout seeds = 150 episodes per eval
- Argmax actions
- Log to wandb under `held_out_success` (distinct from dashboard `perf`)
- **Mirage guard:** if `dashboard_perf - held_out_success > 10pp` over 3 consecutive evals, log a warning

The first 5M cadence (200K) catches early peaks; the 500K cadence afterward is for tracking convergence. ~40 evals per stage × 150 episodes = 6000 episodes of eval per stage per training seed. Cheap given current SPS.

### 6.2 OOD eval-during-training (informational)

In addition to in-distribution eval, run OOD evals at the same cadence. The OOD conditions for each stage:

- During Stage 1.x: eval at Stage 4.0 conditions (everything random, e_max=0.05)
- During Stage 4.x: eval at Stage 1.0 conditions (same orbit, e_max=0.05)

These are *informational only* — they tell us how the policy is generalizing, but they do NOT gate stage transitions. The previous critique correctly noted that the "transition at peak-OOD" reading was overinterpreted from one comparison.

### 6.3 Final capability surface eval

After the full curriculum lands its final Stage 4 ckpt (or Stage 1.4 ckpt for the same-orbit specialist), eval on a capability surface grid:

| Axis | Values |
|---|---|
| `phase_gap` | 30°, 90°, 150°, 180° |
| `e_target` | 0, 0.05, 0.20, 0.50 |
| `e_sat` | 0, 0.05, 0.20, 0.50 |
| `sat_target_orbit_relation` | same_orbit, different_a, different_ω, fully_random |

Full grid is 4×4×4×4 = 256 cells. Per-cell: 200 episodes × 3 rollout seeds = 600 episodes. Total: 153,600 eval episodes per ckpt.

At ~100K SPS, full surface is ~25 minutes per ckpt eval. Trivial.

### 6.4 Reduced grids for intermediate stages

Full 256-cell grids are overkill for intermediate stages. Use a 4-cell representative slice:
- (90° phase, e=0.05, same orbit)
- (180° phase, e=0.05, same orbit)
- (90° phase, e=0.05, fully random)
- (180° phase, e=0.05, fully random)

These map "easy phase × same orbit" → "hard phase × full random." Track these four numbers across every stage's training to see the capability evolve.

---

## 7. Specific failure modes and contingencies

For each plausible stall pattern, pre-commit to the response.

### 7.1 Stage 1.x stalls at high eccentricity (e ≥ 0.30)

**Diagnostic:** is failure mode "agent doesn't try" (cap-tail style, median Δv = 0) or "agent tries and misses" (capability gap)?

| Failure mode | Response |
|---|---|
| Doesn't try | Train longer (per Principle III). If still stalls after 80M, consider stratified e sampling biased toward high e. |
| Tries and misses | Probe the gated shaping at high e (analogous to the Phase 5b audit's σ₃ probe). If σ₃ is dead at high e, retune EPS_PHASE. |

### 7.2 Stage 4.0 stalls (warm-started from Stage 1)

**Diagnostic:** has Stage 4 partially-OOD-transferred, or did warm-start fail entirely?

| Stage 4.0 status | Response |
|---|---|
| ≤ 30% (no real transfer) | Investigate — does the env behave correctly at sat.a ≠ target.a? Run a sanity-check Phase-4-style training fresh. |
| 30-50% (partial, slow growth) | Train longer (default response). Add Stage 2/Stage 3 as fallback stages if Stage 4 hasn't reached threshold by 100M cumulative. |
| 50-70% (partial, plateaued) | Add Stage 2 and Stage 3 explicitly. The constraint relaxation is breaking; isolate which one. |
| ≥ 70% (works) | Standard primary path; continue Stage 4.x. |

### 7.3 Stage 4 fresh outperforms Stage 4 warm

**This would be the most interesting failure mode.** It means the Stage 1 specialty is anti-transferring (negative transfer). Investigate:

- Is the Stage 1 ckpt initializing the policy in a "don't burn" mode that anti-correlates with Stage 4's optimal policy?
- Does freezing some layers of the Stage 1 policy and only fine-tuning others recover Stage 4 capability?
- Is there a smaller "useful subset" of the Stage 1 ckpt to warm-start from?

If clean negative transfer is established, **drop the curriculum and train Stage 4 from scratch**. The same-orbit specialist becomes a separate deliverable, not a stepping stone.

### 7.4 Specific seed fails to converge

Phase 4 had a 1/3 retry rate at Stage 1 fresh. Step 1 saw all seeds converge at Stage 1.0 (random sat init recipe). If Phase 5b sees occasional seed failures:

- Single seed failure at one stage: retry that seed with a different random initialization (within-seed). Not a recipe issue.
- Multiple seed failures at one stage: recipe issue. Investigate.

Pre-commit: ≤ 1/5 seed failure rate per stage is acceptable. > 1/5 triggers diagnosis.

---

## 8. Phase 6 readiness items

Done concurrently with Phase 5b training (mostly during compute-blocked time). These are refactor items, not new investigations.

### 8.1 Curriculum scheduler abstraction

Replace hand-coded stage transitions with a config-driven scheduler:

```python
# Pseudocode
curriculum = [
    {"name": "stage_1_0", "env_kwargs": {"e_max_sat": 0.05, "e_max_target": 0.05, "same_orbit_init": 1, "init_phase_gap_max": pi}},
    {"name": "stage_1_1", "env_kwargs": {"e_max_sat": 0.10, "e_max_target": 0.10, "same_orbit_init": 1, "init_phase_gap_max": pi}},
    # ... etc
]

for stage in curriculum:
    train_until_converged(stage["env_kwargs"], threshold=0.80, max_steps=120_000_000)
    save_ckpt(stage["name"])
```

Phase 6 reuses this with multi-body env_kwargs.

### 8.2 Trajectory logging extensibility

Current trajectory log captures sat + target + debris positions. Phase 6 needs N gravitating bodies. Refactor to:

```c
typedef struct {
    int n_bodies;
    BodyState bodies[MAX_BODIES];
    SatelliteState sat;
    int action;
    float reward;
    // ...
} TrajectoryRecord;
```

Where `MAX_BODIES` is configurable. Phase 5b doesn't need this for two-body work, but doing the refactor now (when the codebase is fresh) is cheaper than Phase 6.

### 8.3 Dynamics interface abstraction

Currently Kepler propagation is called inline in `c_step`. Refactor to:

```c
typedef struct {
    void (*propagate)(WorldState* w, double dt);
    void (*reset)(WorldState* w, EnvConfig* cfg);
} DynamicsBackend;

DynamicsBackend kepler_2body = {.propagate = kepler_propagate_all, .reset = kepler_reset};
// Future: DynamicsBackend rk4_nbody, ...
```

Phase 5b uses `kepler_2body` exclusively. Phase 6 swaps in `rk4_nbody` or similar.

### 8.4 Eval pipeline parameterization

Already partially done — `--init-phase-gap-max`, `--e-max-target`, `--e-max-sat`, `--same-orbit-init` flags exist. Extend to support arbitrary env-kwarg passing for forward compatibility.

---

## 9. What this spec is NOT

- **Not a multi-body extension.** Phase 5b stays in two-body. Phase 6 is multi-body.
- **Not a debris re-introduction.** User hold remains. Debris is a Phase 6+ question.
- **Not an attempt to revisit the 50%-at-180° wall from Phase 4.** That wall was at e=0; Phase 5b's eccentric-target work is downstream of and separate from it.
- **Not a search for new algorithmic interventions.** R3 components stay dead unless very strong evidence demands them.
- **Not a portfolio-narrative document.** Blog/tweet writeups are post-Phase-5b.

---

## 10. Sequencing

Phase 5b proceeds in roughly this order, with each block gated on the previous:

**Block A — Stage 1.x curriculum (compute estimate: ~50M × 5 seeds × 5 sub-stages = ~1.5B steps, ~15 hours)**
1. Stage 1.0 multi-seed (5 seeds, 40M each)
2. Stage 1.1, 1.2 (3 seeds each, 30-50M each)
3. Stage 1.3 multi-seed (5 seeds, ~50M each)
4. Stage 1.4 stretch (5 seeds if Stage 1.3 lands cleanly, ~80M each)
5. Capability surface eval at Stage 1.x final ckpt (informational)

**Block B — Warm-start vs fresh comparison (compute: ~50M × 5 seeds × 2 paths = 500M steps, ~5 hours)**
1. Stage 4.0 warm (5 seeds, warm from Stage 1.x best ckpt)
2. Stage 4.0 fresh (5 seeds, from scratch)
3. Compare per §5

**Block C — Stage 4.x curriculum (compute estimate: ~40M × 5 seeds × 4 sub-stages = ~800M steps, ~8 hours)**
1. Stage 4.0 best path (warm or fresh) confirmed multi-seed
2. Stage 4.1, 4.2 (3 seeds each)
3. Stage 4.3 multi-seed (5 seeds)
4. Stage 4.4 stretch if Stage 4.3 lands

**Block D — Capability surface eval (compute: ~25 min per ckpt × 5 seeds × 1 final = ~2 hours)**
1. Full 256-cell grid eval at final Stage 4.x ckpt
2. Per-seed grids
3. Aggregate report

**Block E — Phase 6 readiness refactors (concurrent with training time)**
1. Curriculum scheduler abstraction
2. Trajectory logging extensibility
3. Dynamics interface abstraction
4. Eval pipeline parameterization

**Block F — Writeup (post-everything)**
1. Findings document
2. Phase 5b → Phase 6 transition spec

Total compute: roughly 25-30 hours over 1-2 weeks. Engineering time: ~1 week including refactors.

---

## 11. Success criteria

Phase 5b is complete when one of the following holds:

- **Soft success:** Stage 4.3 (everything random, e=0.30) reaches ≥ 70% multi-seed mean. Capability surface shows graceful degradation across all axes.
- **Hard success:** Stage 4.4 (everything random, e=0.50) reaches ≥ 80% multi-seed mean. Capability surface shows graceful degradation including high-e cells.

Failure modes that conclude Phase 5b without claim of success but with documented findings:

- Stage 1.x stalls at e ≥ 0.30: Phase 5b ships as "same-orbit eccentric specialist up to e_max", with explicit note on the eccentricity ceiling.
- Stage 4 stalls regardless of warm-start: Phase 5b ships as "Stage 1 same-orbit specialist", with documented finding that orbit-shape transfer is structurally harder than eccentricity within fixed orbit shape.
- Stage 4 negative transfer found: Phase 5b ships two specialists (Stage 1 and Stage 4 fresh), with documented finding that warm-start anti-transfers.

Each of these "failures" is publishable in its own right. The discipline: don't fudge the curriculum to claim success; ship what the data supports.

---

## 12. The one open question worth flagging

The spec assumes the recipe scales to e_max = 0.5 within Stage 1 (same orbit). This is empirically untested. If Stage 1.3 (e=0.5) stalls, the entire Stage 4 plan is at risk because the Stage 4 curriculum assumes Stage 1 covers e ≤ 0.5.

The cheap pre-flight check: train Stage 1.0 → Stage 1.3 directly (skip 1.1, 1.2) on one seed, see if it converges. If yes, the recipe scales and the full curriculum is realistic. If no, hold the eccentricity ceiling at e_max = 0.20 or 0.30 for Stage 4 work, and treat e = 0.50 as a stretch goal that may not land.

This is one extra ~80M run (~1 hour) before the full Block A starts. Cheap risk-reduction.

---

*Author: 2026-04-29. Phase 5b proper, drafted after Step 1 + cap-tail + post-extend findings. Successor: Phase 5b findings document, drafted after Block A and Block B land.*
