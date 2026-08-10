# Phase 6 Transition

**Status:** Phase 5 closed 2026-05-01. This is the bridge document.

---

## What Phase 5 ships

- A 2D coplanar orbital rendezvous agent that handles random LEO tasks (300-800 km) at e ≤ 0.05, multi-seed validated 5/5 at ≥84%.
- The recipe (`RECIPE.md`) — reproducible from `bash scripts/orbital/p5e_curriculum.sh all`.
- The deliverable data (`web_data/`) — 200+ trajectory JSONs, capability surface CSV, LEO probe CSV, training curves, plots.
- The findings (`PHASE5_FINDINGS.md`) — honest narrative including two retracted mechanism stories.
- The model weights (`models/phase5e/`) — 8 MB committed, sufficient to reproduce all claims.

What Phase 5 does *not* ship: high-eccentricity capability, multi-altitude capability, multi-body, debris, 3D, continuous thrust. These are explicit non-goals or Phase 6 candidates.

---

## What Phase 6 might do

The natural progressions, in order of difficulty:

### 6.A — Multi-altitude curriculum (LEO → MEO → GEO)

The Phase 5e wrap-up surface revealed the recipe's altitude dependence. A natural Phase 6 starts by varying the altitude band during training. Should produce a recipe that generalizes 300-3000 km altitudes at low e. Not a fundamentally new direction; just a curriculum extension.

**Compute estimate:** ~2-3 days × 5 seeds. Probably tractable.

### 6.B — High-e curriculum at LEO

The Phase 5d/5e attempts at e_max=0.20 used reward-side interventions; none worked. Phase 6 could try the *task-distribution-level* interventions deferred from Phase 5e Block III:

- Bound-expansion curriculum (sample sat near target's already-valid orbit, expanding bounds smoothly)
- Action masking + retraining (Phase 5d Path A infrastructure exists)
- Demonstration bootstrapping (Lambert-optimal solutions as initial behavior cloning data)

Each is genuinely orthogonal to what Phase 5 tested. Honest expectation: at least one of these helps; possibly all three are needed.

**Compute estimate:** ~3-5 days × multi-seed.

### 6.C — Multi-body (cislunar)

The Phase 1 design doc's Earth-Moon scenario. Replaces Kepler with n-body integration (RK4/RK45). Time warp becomes essential. Brings real-world relevance: Artemis-era cislunar transfers.

**Implementation cost:** Significant. Env rewrite. Phase 5 readiness items (W5.1, W5.2 — trajectory logging extensibility, curriculum scheduler abstraction) help here but are not done.

**Compute estimate:** ~1 week env rewrite + ~2 weeks training. Big chunk.

### 6.D — Realistic conjunction data

Real TLE / CDM data from Space-Track.org. Trains the policy on actual debris fields. Not a curriculum question; an env-data question. Probably valuable for portfolio framing.

**Cost:** ~1 week to set up data pipeline. Training cost similar to Phase 5.

---

## Phase 5 readiness items NOT done

The Phase 5 closure spec listed three Phase 6 readiness refactors. Status:

### W5.1 — Trajectory logging extensibility (NOT DONE)

Generalize the C trajectory record from "sat + target + (debris)" to "N gravitating bodies." Phase 5 still uses 1-2 bodies; Phase 6 multi-body needs 3+.

**Why deferred:** Phase 5 closure cost-benefit pushed back closure work. Phase 6 should do this as the first env-rewrite item. Estimated 2-3 hours.

### W5.2 — Curriculum scheduler abstraction (NOT DONE)

Replace bash orchestration in `p5e_curriculum.sh` with a Python class. Configurable schedule. Phase 6 reuses with multi-body env_kwargs.

**Why deferred:** Same reason. Estimated 1-2 hours.

### W5.3 — Eval pipeline parameterization (PARTIALLY DONE)

The Phase 5 wrap-up's `p5wrap_surface_eval.py` and `p5wrap_leo_probe.py` are subprocess-driven cell sweeps. Each spawns N parallel `eval_checkpoint.py` subprocesses with cell-specific kwargs. This works but isn't a unified tool.

A proper Phase 6 eval pipeline would use a single config file specifying cells, run them in-process (sharing model load), and aggregate. Tractable.

**Why deferred:** Not blocking. Current scripts work for Phase 5; Phase 6 can refactor.

---

## Open questions Phase 5 didn't resolve

### Q1 — Does action masking + bound-expansion crack high-e at LEO?

Phase 5d's Path A action masking infrastructure exists (`enable_action_mask` kwarg, 48-dim obs, Default policy mask consumer — all off by default but verified). Phase 5e Block III's bound-expansion curriculum was designed but not implemented. The combined intervention class wasn't tested.

**Plausible:** at least one helps. The Phase 5e narrative calls these "Phase 6 candidates if multi-body work surfaces issues that the e=0.70 surface reveals."

### Q2 — Is there a fundamental ceiling on PPO + gated NHR for high-eccentricity rendezvous?

Or is it always a curriculum / distribution / observation-encoding problem? Phase 5's diagnostic depth doesn't answer this. Worth keeping in mind but not investigating directly until Phase 6 has a tangible problem to study.

### Q3 — How does the recipe transfer to actual mission scenarios?

Real LEO rendezvous (e.g., ISS approach) is at altitudes 400-450 km, e ≈ 0, phase variable. The recipe should handle this directly — well within trained distribution. But a real-data eval (TLE-based scenarios from Space-Track) would be the strongest portfolio claim.

---

## Recommendation

If Phase 6 starts immediately:

1. Do W5.1 + W5.2 first (~3-5 hours). Sets up clean infrastructure for whatever Phase 6 chooses.
2. Pick **6.A multi-altitude** if the goal is breadth (LEO → MEO → GEO at low e). Easy win.
3. Pick **6.B high-e curriculum** if the goal is depth (cracking the Phase 5 ceiling). More uncertain but more interesting.
4. Pick **6.C multi-body** if the goal is the most-valuable portfolio piece (cislunar = APL/Lincoln Lab relevance). Big investment.
5. Pick **6.D realistic data** if the goal is highest portfolio impact for least training compute.

Phase 6 spec writing should pick *one* of these as the primary direction. The Phase 5 readiness items get done en route.

---

*Phase 5 closed. Phase 6 unblocked. The recipe is documented; the data is preserved; the methodology is named.*
