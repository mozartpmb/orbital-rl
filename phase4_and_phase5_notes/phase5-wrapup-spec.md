# Phase 5 Final Wrap-up Spec

> **Status:** 2026-05-01. Phase 5e shipped a working agent with multi-seed validation along the eccentricity axis. The previous closure spec covered the broader landscape; this final wrap-up focuses specifically on what's needed before declaring Phase 5 done. Adjustments after the e_max-vs-fixed-eccentricity question: capability surface needs fixed-eccentricity per cell, not uniform-up-to-bounds sampling. Total ~12-15 hours work, mostly engineering and analysis, ~10 hours compute.

---

## 0. What this spec is for

The Phase 5e Block II findings + the previous closure spec + the blog data catalog have produced most of what's needed. Three things remain unresolved:

1. **The e_max sampling caveat.** Reported numbers (90.2% at e_max=0.20, 83.4% at e_max=0.50) are aggregates over uniform distributions, not per-eccentricity capability. Headline framing implies the latter.
2. **The capability surface is single-seed and single-axis.** The full (phase × e_target × e_sat × sat-target relation) surface across multiple seeds has been planned but not run.
3. **The findings consolidation and recipe documentation.** Phase 5b through 5e haven't been written up as a single coherent narrative.

This spec addresses all three plus the residual web-frontend data needs that depend on the surface result.

---

## 1. Block W1 — Fix the eval methodology (engineering, ~1 hour)

Before any new training or eval runs, fix the env's eccentricity sampling so we can measure per-eccentricity capability rather than per-distribution.

### W1.1 — Add fixed-eccentricity env kwargs

Modify `orbital.h` reset logic to support exact-value sampling:

```c
// New env kwargs (default behavior preserves existing semantics)
float e_target_fixed;     // if >= 0, use exactly this value; if < 0, uniform up to e_max_target
float e_sat_fixed;        // same pattern
float phase_gap_fixed;    // optional, for phase-axis surface cells
float omega_offset_fixed; // optional, for ω-relation cells
```

Reset logic:

```c
target.e = (e_target_fixed >= 0) ? e_target_fixed : uniform(0, e_max_target);
sat.e = (e_sat_fixed >= 0) ? e_sat_fixed : uniform(0, e_max_sat);
```

Default values preserve existing behavior. New code paths only activate when explicit fixed values are provided.

### W1.2 — Plumb through binding.c, orbital.py, eval_checkpoint.py

Add the four new kwargs to the Python wrapper. Add CLI flags to `eval_checkpoint.py`:

```
--e-target-fixed <float>   # default -1 (uniform sampling preserved)
--e-sat-fixed <float>      # default -1
--phase-gap-fixed <float>  # default -1
--omega-offset-fixed <float>  # default -1
```

### W1.3 — Validation

After the changes:

1. Run a control eval with all defaults — should match prior Phase 5e numbers exactly. Confirms the addition didn't break existing semantics.
2. Run `--e-target-fixed 0.0 --e-sat-fixed 0.0 --phase-gap-fixed 3.14159` on the Phase 5e seed 42 ckpt. Should approximately match the Phase 5e Block II.C result for Phase 4 conditions (~92%).
3. Run `--e-target-fixed 0.50 --e-sat-fixed 0.50` on the same ckpt. This is the first measurement of *actual* e=0.50 capability rather than e_max=0.50 capability.

The third measurement is the first interesting data point. If it's substantially below 84.5% (e.g., 60%), the previous headline numbers were optimistic; if it's near 84.5%, the recipe truly handles e=0.50 well.

**Cost:** ~1 hour engineering + ~5 min eval validation.

---

## 2. Block W2 — Capability surface, properly measured (~10 hours compute)

### W2.1 — Surface design

Now that we can fix eccentricities per cell, the surface measures *per-condition* capability:

| Axis | Values | Sampling |
|---|---|---|
| `phase_gap` | 30°, 90°, 150°, 180° | Fixed per cell |
| `e_target` | 0.0, 0.05, 0.20, 0.50, 0.70 | Fixed per cell |
| `e_sat` | 0.0, 0.05, 0.20, 0.50 | Fixed per cell |
| `sat_target_relation` | same_orbit, different_a, different_ω, fully_random | Per-cell config |

4 × 5 × 4 × 4 = **320 cells** per checkpoint.

Per cell: 200 episodes × 3 rollout seeds = 600 episodes.

Per checkpoint total: 192,000 episodes.

5 training seeds × 192,000 = 960,000 episodes. At ~100k SPS effective for single-env eval, ~10 hours compute, parallelizable.

### W2.2 — Reduced grid (alternative if 10 hours is too much)

If the full grid is overscoped, a reduced version covers the most informative cells:

- 4 phase × 5 e_target × 4 e_sat × 1 relation (`fully_random`) = 80 cells per ckpt
- 5 seeds × 80 cells × 600 eps = 240K episodes ≈ 2.5 hours

This drops the `sat_target_relation` axis (which is largely redundant with the other axes — fully_random subsumes the others as a generalization). The headline heatmap is `phase × e_target × e_sat` aggregated to 2D for visualization.

I'd recommend the **full 320-cell grid** if compute permits. The relation axis might reveal interesting structure (e.g., same_orbit_init at e=0.50 might be a different regime than fully_random). If it's redundant, the surface confirms it; if it's not, we've found something.

### W2.3 — Eval execution

For each (training seed, cell) combination:

1. Load the ckpt.
2. Set the 4 fixed kwargs from the cell.
3. Run 600 episodes (200 × 3 rollout seeds).
4. Record success rate, mean total Δv, mean episode length.

Output: a CSV per training seed with columns `{phase_gap, e_target, e_sat, relation, success, total_dv_mean, episode_length_mean}`.

### W2.4 — Aggregation and visualization

Across the 5 training seeds, compute mean ± std per cell. Output:

- `phase5_capability_surface.csv`: per-cell aggregate stats (mean, std, n_seeds)
- `plots/phase5_capability_surface.png`: heatmap visualization. Probably 4 sub-panels (one per phase_gap), each showing e_target × e_sat as a 2D heatmap, possibly with the `relation` axis as faceted columns.
- Identify cells with high inter-seed variance (>5pp std) — these are the regime boundaries where the recipe is fragile.
- Identify cells with mean success below 50% — these are the "documented limit" boundary.

### W2.5 — Decision criteria

Pre-committed responses to plausible Block W2 surprises:

**Surprise A — Surface is uniformly graceful.** Heatmap shows smooth degradation across all axes; no cliffs. The deliverable holds at the headline numbers and the eventual writeup gets the cleanest possible visual.

**Surprise B — Cliffs at specific (phase, e) combinations.** E.g., (phase=180°, e_target=0.50, e_sat=0.0) might be much harder than other 0.50 cells. Document the cliffs honestly. The deliverable still holds in aggregate but with explicit corner caveats.

**Surprise C — Per-condition numbers are substantially below e_max numbers.** E.g., e_target=0.50 fixed gives 50% rather than the 83.4% reported as e_max=0.50. The previous headline numbers were optimistic; rewrite the deliverable framing to use per-condition values rather than aggregates.

**Surprise D — Inter-seed variance much higher than e_max scan suggested.** Some cells have 30pp std across seeds. The recipe is fragile in those regimes; flag honestly.

In all four cases, the discipline is the same: report what's true. If the deliverable is weaker than previously reported, say so; if it's stronger, say so; if it has structure, document it.

---

## 3. Block W3 — Web frontend data finalization (~3-4 hours)

The blog data catalog produced most of what's needed. Three remaining items.

### W3.1 — Per-cell trajectory exports

For the capability surface to be informative on the web frontend, sample trajectories from a few per-cell evals — particularly the boundary cells (cliffs, high variance) and the "interesting" cells (Molniya territory at e=0.70 phase=180°).

For each of ~10 representative cells across the surface:
- 3-5 trajectory JSONs per cell, mix of success and failure where applicable
- Cell metadata in the JSON

Save under `web_data/runs/surface/<cell_id>/`.

**Cost:** ~30 min selection + ~30 min export. Trajectories already produced during W2; this is selection and conversion.

### W3.2 — Update `web_data/results/` with surface

Add to `web_data/results/`:
- `phase5_capability_surface.csv` (from W2.4)
- `phase5_capability_summary.json`: top-level stats (worst cell, best cell, median per e level, etc.)

Update `web_data/results/multiseed_escan.csv` with a note that "e_max" sampling differs from fixed-e measurement, with reference to the surface CSV.

### W3.3 — Plot updates

Update `plots/p5e_capability_surface.png` to use the fixed-e surface data, or add `plots/phase5_capability_surface_fixed.png` as the new headline visual.

The 1D e_max scan plot stays — it's still useful — but becomes a secondary visual rather than the headline.

---

## 4. Block W4 — Findings consolidation (~3 hours writing)

### W4.1 — `PHASE5_FINDINGS.md`

Single comprehensive document covering Phase 5b through Phase 5e and the wrap-up. Sections:

1. **Headline result.** What was built. Reference to the surface heatmap.
2. **Phase 5b: working agent at e ≤ 0.10.** What the two-stage curriculum produced. Multi-seed validated.
3. **Phase 5c/5d: ceiling investigation.** Honest accounting of the ~50 hours spent on what turned out to be a misattributed mechanism. The Simpson's paradox catch, the early-death framing, the 8 reward-side interventions, the eventual recipe-ceiling claim that proved wrong.
4. **Phase 5e: env validation found the issue.** The Lambert reachability check; 64% of e_max=0.20 inits were physically unrecoverable. One conditional in `c_reset` fixed it.
5. **Wrap-up: capability surface and recipe.** Per-cell results, recipe documentation, where the deliverable holds and where it has limits.
6. **Methodological lessons.** Three named principles:
   - Test the train-longer null before structural claims (Phase 5b cap-tail).
   - Length-bin (or condition-bin) before aggregating any quantity that depends on the binning dimension (Phase 5c Simpson's correction).
   - When metric names imply specific values, verify whether the metric measures that value or something more aggregated (Phase 5e env validation, e_max-vs-fixed-e in W1).
7. **What the project transferable methodology produced.** The deliverable plus the methodological discipline are a single output, not two separate things. The hard parts of building this agent were not the algorithm choices — they were the diagnostic discipline that caught (and corrected) successive mechanism stories.

This document is technical, not portfolio-narrative. It's for someone (future-self or future maintainer) who wants to understand what was actually done.

### W4.2 — `RECIPE.md`

Standalone document describing the final Phase 5 recipe. Sections:

1. **Environment.** Action space (Discrete(10) including warp-5min), observation (38-dim with LVLH), success criterion (dist < 30km, rel_vel < 50 m/s), termination (success/escape/collision/stranded/safety_cap), env config (random sat init, valid_init_only, eccentricity bounds).
2. **Reward.** Gated NHR shaping with terminal Φ-clamp. Formula. Component weights.
3. **Curriculum.** Stage 1.0 (same_orbit_init, e=0.05) → Stage 4.0 (fully random, e=0.05). Stage transition criterion. Multi-seed protocol.
4. **Hyperparameters.** Full PPO config. All env kwargs.
5. **Capability.** From the surface. Per-eccentricity capability with multi-seed mean ± std.
6. **Limits.** Where the recipe degrades. e=0.70 zone, action discretization at high e, the 5pp generalization tax at Phase 4 conditions.
7. **What's not in the recipe.** R3 components (DAPO, etc.) — explicitly excluded. REL_VEL_TOL annealing — tested, didn't help. Continuous actions — ruled out by Phase 5b.

This is the document a future self or maintainer reads to reproduce or extend the work.

### W4.3 — Phase 6 transition stub

A short document `PHASE6_TRANSITION.md` summarizing:

- What Phase 5 ships (the agent + the data + the recipe)
- What Phase 6 might do (multi-body env, eccentric multi-body transfers, cislunar scenarios)
- The Phase 6 readiness items that were done (or should be done now if not)
- Open questions that the Phase 5 work didn't resolve (e.g., does action masking + bound-expansion crack e=0.70 to higher than 64%? Reserved for if Phase 6 doesn't displace this)

This is short — maybe 1 page. It's the bridge document between Phase 5's deliverable and whatever comes next.

---

## 5. Block W5 — Phase 6 readiness refactors (~3 hours, concurrent)

These items are still on the table from earlier specs. Done concurrently with W2-W4 if compute is busy.

### W5.1 — Trajectory logging extensibility

Generalize the C trajectory record from "sat + target + (debris)" to "N gravitating bodies." Phase 5 still uses 1-2 bodies; Phase 6 needs 3+.

### W5.2 — Curriculum scheduler abstraction

Replace bash orchestration in `p5e_curriculum.sh` with a Python class. Configurable schedule. Phase 6 reuses with multi-body env_kwargs.

### W5.3 — Eval pipeline parameterization

The eval scripts have grown organically. Consolidate into a single tool that takes a config (or kwargs) for arbitrary cell specifications. Used by both Phase 5 W2 and any future phases.

These aren't blocking and don't need to ship before Phase 5 closes, but they're "easier now than later" items.

---

## 6. Sequencing

In order:

1. **W1** (env mod for fixed-e). ~1 hour. Single most-important item; everything downstream depends on it.
2. **W1.3 validation runs.** ~5 min. Confirms the env mod works.
3. **W2.1-W2.3** (capability surface execution). ~10 hours compute, parallelizable across seeds. Run while writing W4.
4. **W2.4** (aggregation and viz). ~1 hour after compute completes.
5. **W3** (web data updates). ~3-4 hours, can overlap with W2 compute time.
6. **W4** (findings docs). ~3 hours, mostly writing during W2 compute.
7. **W5** (refactors, concurrent throughout). ~3 hours.
8. **Phase 5 closure.** Formally close once the surface is run, the docs are written, and the data is in place.

Total wall time ~1-2 weeks elapsed depending on writing pace and compute scheduling.

---

## 7. What this spec is NOT

- **Not a re-investigation.** Phase 5e's deliverable stands. The surface measures it more carefully; doesn't undo it.
- **Not Phase 6.** Phase 6 is downstream of closure.
- **Not the web frontend implementation.** That's downstream of W3's data prep.
- **Not the portfolio piece.** That's downstream of W4's findings doc.
- **Not a complete intervention exhaustion.** Action masking, demonstration bootstrapping, bound-expansion curriculum — these remain plausible Phase 6 follow-ups if multi-body work surfaces issues that the e=0.70 surface reveals.

---

## 8. Closure conditions

Phase 5 is formally closed when:

1. **W1**: env modification done, validated; default behavior preserves prior numbers.
2. **W2**: capability surface multi-seed measured at fixed eccentricities; heatmap produced; per-cell stats in `phase5_capability_surface.csv`.
3. **W3**: web data updated with surface results and representative trajectories.
4. **W4**: `PHASE5_FINDINGS.md`, `RECIPE.md`, `PHASE6_TRANSITION.md` written.
5. **W5**: Phase 6 readiness refactors done (or explicitly deferred with reasoning).

When all five hold, Phase 5 closes. Phase 6 starts (or doesn't, depending on what comes next).

---

## 9. Pre-committed acknowledgments

Per the project's pattern, expect a surprise. Specifically:

**Plausible surprise — Per-condition surface reveals previously-reported numbers were aggregates that were higher than the per-condition values.** E.g., e_max=0.50 at 83.4% might be 70% at e=0.50 fixed when isolated. If this happens, **the closure writeup honestly reports per-condition values** as the headline, with the e_max numbers as historical context. Don't fudge the closure framing; if the data says weaker, write weaker.

**Less plausible but worth flagging — Per-condition surface reveals the recipe is *better* than the e_max numbers suggested.** This would happen if low-e cases within an e_max=0.50 distribution were dragging the aggregate down (which would be weird but not impossible). If so, the closure writeup is stronger; report what's true.

**Methodologically — The W1 fix to env eccentricity sampling reveals issues in older Phase 4-5 reported numbers.** The aggregate-vs-condition gap exists throughout the project. Phase 4's "180° phase gap" is also `init_phase_gap_max=π` (uniform up to π), not "exactly π." If we want to be rigorous, we'd re-eval Phase 4 ckpts with `phase_gap_fixed=π` to get the true 180° number. **This is out of scope** for Phase 5 closure. Phase 4's number stands as reported with the methodological caveat documented in `PHASE5_FINDINGS.md`.

The discipline carries forward to anything Phase 6 measures: per-condition values are the right framing; aggregate-over-distribution values are caveatted.

---

*Author: 2026-05-01. Phase 5 final wrap-up after the e_max-vs-fixed-e methodological correction. Successor: PHASE5_FINDINGS.md + Phase 6 transition.*
