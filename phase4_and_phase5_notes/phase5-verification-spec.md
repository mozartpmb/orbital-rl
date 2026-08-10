# Phase 5 Verification — Investigation Spec

> **For a fresh agent.** Phase 5 has had multiple closure-and-retraction cycles. The latest closure (PHASE5_FINDINGS.md, dated 2026-05-01) declared the recipe a "LEO low-eccentricity specialist" after the wrap-up surface showed cliffs at e ≥ 0.075. But the user is observing trajectories on the web frontend that appear to have initial states inside Earth's surface — suggesting `valid_init_only=1` may not be doing what its code says it does. This spec runs four targeted investigations to determine what's actually true before any more compute is spent on closure.

---

## 0. Context you need

### 0.1 The project in one paragraph

A 2D coplanar orbital rendezvous RL agent built on PufferLib Ocean, training PPO+LSTM on a custom C environment with Kepler dynamics. The task: chaser satellite must rendezvous with a target satellite given random initial orbits. The recipe (from Phase 4 + Phase 5b + Phase 5e fix) is `Discrete(10)` actions including a 5-minute warp action, gated NHR shaping with terminal Φ-clamp, LVLH-frame observations, two-stage curriculum (Stage 1.0 same_orbit_init → Stage 4.0 fully random), trained at `e_max=0.05` in LEO altitudes (300-800 km). Phase 5e added `valid_init_only=1` rejection sampling to filter physically infeasible inits.

### 0.2 The specific situation now

Phase 5 has retracted multiple mechanism stories:
- Phase 5b post-extend "60M shows erosion" — single-seed overcommitment, retracted
- Phase 5c "shaping direction reversed" — Simpson's paradox, retracted
- Phase 5e "recipe ceilings at e=0.10" — was a `valid_init_only` env bug, retracted
- Wrap-up "recipe handles e ≤ 0.50" — aggregate-vs-fixed-e framing, retracted

Most recent state: PHASE5_FINDINGS.md claims the recipe is a LEO low-e specialist (peak ~97% at fixed e=0.025; ~85% at fixed e=0.05; cliff at e ≥ 0.075). The wrap-up surface used altitude overrides for high-e cells, which conflated altitude generalization with eccentricity capability.

**The trigger for this spec:** the user is browsing trajectory files on the web frontend (claimed by the visualizer agent to be post-5e, i.e. produced under `valid_init_only=1`). Many of the trajectories appear to show satellite or target starting in orbits whose perigee is below Earth's surface. If true, `valid_init_only=1` is not filtering as expected, and the post-5b headline numbers (Phase 5e's 90.2% at e_max=0.20, the wrap-up's surface) need re-examination.

### 0.3 What `valid_init_only=1` is supposed to do

Per the C code in `pufferlib/ocean/orbital/orbital.h::c_reset`:

```c
int valid_init_attempts = 0;
valid_init_resample:
valid_init_attempts++;
// ... sample sat.{a, e, ω, M}, target.{a, e, ω, M}, phase_gap, etc ...
if (env->valid_init_only && valid_init_attempts < 256) {
    double sat_rp = env->sat.orbit.a * (1.0 - env->sat.orbit.e);
    double tgt_rp = env->target.a * (1.0 - env->target.e);
    if (sat_rp < EARTH_KEEPOUT || tgt_rp < EARTH_KEEPOUT) {
        goto valid_init_resample;
    }
}
```

Mechanically: rejection sampling on the orbit's perigee. Bounded at 256 attempts before accepting whatever sample is on hand. `EARTH_KEEPOUT = R_EARTH + 200km = 6.571e6 m` (approximately).

**What this should produce:** every accepted init has both sat.a*(1-sat.e) ≥ 6.571e6 and target.a*(1-target.e) ≥ 6.571e6. Filtering on orbit perigee, not initial position. So an orbit whose perigee is above keepout but whose chaser starts at apogee would pass. An orbit whose perigee is below keepout would not.

### 0.4 The likely possibilities

1. **`valid_init_only=1` isn't being plumbed through to the C env.** The Python wrapper or eval scripts might silently default it to 0. The code path runs only when env->valid_init_only is truthy.
2. **The 256-attempt limit is being exceeded routinely.** At conditions where most samples are invalid, 256 attempts may not be enough.
3. **The trajectories user is viewing are pre-Phase-5e** despite the visualizer agent's claim. Mistagged or path-confused.
4. **The render/visualizer is misleading** — initial position appears "in planet" but is actually on a valid orbit at apogee, with perigee just below visual threshold. (Worth checking but the user already pushed on this.)
5. **The check is correct but a separate code path bypasses it.** Some eval scripts may construct initial states directly rather than going through `c_reset`.
6. **Some other bug entirely.** New finding.

### 0.5 What you should NOT do

- **Don't kick off new training runs.** This investigation is post-hoc analysis only.
- **Don't draw conclusions about Phase 5 deliverable until all four investigations are complete.** The pattern in this project has been "single-piece-of-evidence overcommitment"; resist it.
- **Don't trust the wrap-up surface re-run** that may be in flight. If it's still running, let it finish but don't analyze it until `valid_init_only` is verified.
- **Don't write any retraction or closure document yet.** The deliverable framing depends on what these investigations find.

---

## 1. Investigation I1 — Does `valid_init_only=1` actually fire?

### 1.1 Goal

Verify mechanically that when an eval is configured with `valid_init_only=1`, the C env's rejection-sampling code path runs and produces inits with sat & target perigees ≥ EARTH_KEEPOUT.

### 1.2 Method

**Step A — Add instrumentation to `c_reset`.**

In `pufferlib/ocean/orbital/orbital.h::c_reset`, add a printf that logs:
- `valid_init_only` flag value
- Number of rejection attempts
- Final accepted sat perigee, target perigee
- Whether the 256-attempt limit was hit (i.e., final accepted state failed the check)

Wrap in a debug compile flag or just set a global counter to print every Nth reset (e.g., every 100th). The goal is to see what's happening, not to flood logs.

```c
// in c_reset, after the rejection loop
if (env->log_validation_debug) {
    double sat_rp = env->sat.orbit.a * (1.0 - env->sat.orbit.e);
    double tgt_rp = env->target.a * (1.0 - env->target.e);
    int gave_up = (env->valid_init_only && valid_init_attempts >= 256);
    printf("RESET: valid_init=%d attempts=%d sat_rp=%.3e tgt_rp=%.3e gave_up=%d\n",
           env->valid_init_only, valid_init_attempts, sat_rp, tgt_rp, gave_up);
}
```

Add `log_validation_debug` as a kwarg in binding.c and orbital.py. Default 0.

**Step B — Run one eval with debug instrumentation.**

Pick the canonical Phase 5e seed-42 ckpt. Run a small eval (50 episodes) with:
- `valid_init_only=1`
- `e_max_target=0.20`
- `e_max_sat=0.20`
- LEO altitude band (default `a_min`, `a_max`)
- `log_validation_debug=1`

This is the configuration that produced the reported "90.2% multi-seed at e=0.20" Phase 5e number.

**Step C — Inspect the output.**

For each reset, verify:
1. `valid_init=1` (the kwarg is reaching the C env)
2. `attempts > 0` (the loop is running)
3. `sat_rp ≥ 6.571e6` and `tgt_rp ≥ 6.571e6` for *most* resets
4. `gave_up=0` for *most* resets (256 attempts shouldn't be exhausting often)

Aggregate stats:
- What fraction of resets accept on first attempt vs require multiple?
- What fraction hit the 256 limit and accept a doomed init?
- For the doomed-init cases, what's the actual perigee?

### 1.3 Decision rule

**If valid_init=0 in the printf:** the kwarg isn't being plumbed through. Find where it's being silently dropped (Python wrapper, binding.c, env init). Fix and re-run all post-5e evals. Major retraction territory.

**If valid_init=1 but attempts=0 always:** the loop isn't being hit, possibly a control-flow bug in the goto. Investigate the C code. Probably also a major issue.

**If valid_init=1 and attempts > 0 and most accept with valid perigees:** the filter is working. Move to I2 to investigate the trajectory files directly.

**If valid_init=1 and the 256-limit is being hit on >5% of resets:** the filter is working but at extreme conditions can't find valid samples. Doomed inits are being accepted. This explains some apparent in-planet trajectories.

### 1.4 Cost

~20 minutes. Code mod is small; eval is fast; analysis is direct.

---

## 2. Investigation I2 — What's actually in the trajectory files?

### 2.1 Goal

For trajectories the user is viewing on the web frontend that appear to start in-planet, verify directly: does the JSON file's initial state have a perigee below EARTH_KEEPOUT? Or below R_EARTH (the actual surface)?

### 2.2 Method

**Step A — Identify a sample.**

The web frontend presumably has trajectory files under `web_data/runs/`. Pick 20 trajectories that the user has identified as "looks impossible" (or sample broadly across the post-5e directories). Note which ckpt produced each, which env config (the JSON metadata should include this).

**Step B — Compute initial perigees.**

For each trajectory JSON:
- Read the `initial` block: `sat_a_km`, `sat_e`, `target_a_km`, `target_e`
- Compute sat perigee = `sat_a_km * (1 - sat_e)` (in km; convert if needed)
- Compute target perigee
- Compare to `R_EARTH = 6371 km` (surface) and `EARTH_KEEPOUT = 6571 km` (keepout)

**Step C — Categorize.**

For each trajectory:
- Pass: both perigees ≥ EARTH_KEEPOUT (filter worked correctly)
- Marginal: one or both perigees ∈ [R_EARTH, EARTH_KEEPOUT) (above surface but below keepout — atmosphere/decay zone)
- Failed: one or both perigees < R_EARTH (below surface)

**Step D — Cross-reference with metadata.**

For each trajectory:
- What's the file's `valid_init_only` metadata field?
- What's the `phase` (which Phase 5 stage produced it)?
- What's the `e_max_target` and `e_max_sat`?

The user thinks they're all post-5e. Verify this directly.

### 2.3 Decision rule

**If most "looks impossible" trajectories actually have both perigees ≥ EARTH_KEEPOUT:** the filter worked; the visualization is misleading the user. The trajectories *look* like they're in the planet because of how 2D projections render eccentric orbits, but mathematically they're fine. No env bug; the web frontend's rendering needs work.

**If many trajectories have perigees below R_EARTH:** the filter didn't work for these specific files. Cross-reference with metadata to see if they're actually pre-Phase-5e (mis-tagged) or post-5e (real bug). If post-5e, this is a real `valid_init_only` issue and connects to I1.

**If some are in the marginal band [R_EARTH, EARTH_KEEPOUT):** the filter is targeting the right thing (keepout) but the visualization shows surface; mostly a visual issue, but the marginal band is technically "atmosphere/decay" which is also unrecoverable.

**If the metadata doesn't include `valid_init_only` field at all:** the export script wasn't recording this. Trajectory provenance is unclear. This is itself a bug to fix.

### 2.4 Cost

~30 minutes. Mostly scripting + sampling.

---

## 3. Investigation I3 — Does Phase 5b's pre-fix deliverable still hold?

### 3.1 Goal

Phase 5b's headline result (96.4% multi-seed at e_max=0.05 fully random) was achieved before `valid_init_only` existed. At e_max=0.05 in LEO altitudes, almost all randomly-sampled inits are physically valid (perigee ≥ keepout) without any filter. So the Phase 5b result should be invariant to `valid_init_only`.

If it is — Phase 5b stands as a deliverable independent of the `valid_init_only` debate. If it's not — something deeper is going on.

### 3.2 Method

Pick the canonical Phase 5e seed-42 Stage 4.0 ckpt. Run two evals, identical except for `valid_init_only`:

**Eval A:** `valid_init_only=0`, `e_max_target=0.05`, `e_max_sat=0.05`, LEO altitudes, 200 episodes × 3 rollout seeds = 600 episodes. Record success rate.

**Eval B:** `valid_init_only=1`, same config, same episode count. Record success rate.

### 3.3 Decision rule

**If Eval A and Eval B are within ~3pp of each other:** confirms the filter has minimal effect at e_max=0.05 (as expected — almost all samples are physically valid even without filter). Phase 5b's deliverable is robust.

**If they differ substantially (>5pp):** something's odd. Possibilities: the filter is doing something else than expected at low e, or there's a different env behavior depending on the flag.

**If both are dramatically lower than Phase 5b's claimed 96.4%:** the ckpt may not actually be the Phase 5b deliverable, or the eval pipeline has changed in a way that changes results. Also a major issue.

### 3.4 Cost

~15 minutes (~5 min per eval + analysis).

---

## 4. Investigation I4 — What was actually trained?

### 4.1 Goal

Document the env kwargs used during Stage 1.0 and Stage 4.0 training, explicitly. The user assumed (reasonably) that the recipe trained on high-altitude eccentric orbits; the agent's analysis suggests it didn't. Verify which is true.

### 4.2 Method

**Step A — Read the curriculum scripts.**

Find `scripts/orbital/p5e_curriculum.sh` (or its predecessor). For Stage 1.0 and Stage 4.0, extract the env kwargs:
- `a_min`, `a_max` (altitude bounds)
- `e_max_target`, `e_max_sat`
- `same_orbit_init`
- `valid_init_only`
- `init_phase_gap_max`
- Any other relevant kwargs

**Step B — Read the orbital.ini config.**

Find `pufferlib/config/ocean/orbital.ini`. Document the default values for the kwargs above.

**Step C — Read the binding.c kwargs handler.**

In `pufferlib/ocean/orbital/binding.c`, find the kwargs-parsing section. Document defaults if not specified in script or config.

**Step D — Reconstruct the actual training distribution.**

Combine: `script kwarg > config kwarg > binding.c default`. For each curriculum stage, list:
- Sat altitude range trained on
- Sat eccentricity range trained on
- Target altitude range trained on
- Target eccentricity range trained on
- Phase gap range trained on
- Other relevant axes

### 4.3 Decision rule

This is descriptive, not pass/fail. The output is a clear table of "what the recipe was trained on."

If the user's intuition matches (high-altitude eccentric orbits): the wrap-up's "LEO specialist" framing is wrong; the recipe should generalize beyond LEO and the surface results are surprising.

If the user's intuition doesn't match (recipe is LEO-only as the wrap-up claims): the LEO-specialist framing is correct; extension to MEO/HEO requires retraining; the wrap-up holds in this respect.

### 4.4 Cost

~30 minutes. Mostly reading.

---

## 5. Sequencing

In order:

1. **I1 first** (~20 min). Most consequential single check.
2. **I4 in parallel with I1** (~30 min). Independent question, can run while I1 is happening.
3. **I2 after I1** (~30 min). Builds on I1's findings.
4. **I3 last** (~15 min). Sanity check on the Phase 5b deliverable.

Total ~70-90 minutes of work. No new training. Mostly reading code, adding small instrumentation, and direct analysis.

---

## 6. What to produce

A single document `PHASE5_VERIFICATION_FINDINGS.md` with sections per investigation. For each:
- What was checked
- What was found (concrete numbers, not narratives)
- Verdict on the relevant decision rule
- Implication for Phase 5 closure framing

Plus a top-level synthesis section that answers, given all four investigation results:

- Does `valid_init_only=1` actually filter as designed?
- Is Phase 5b's e ≤ 0.05 deliverable still solid?
- Is the wrap-up's "LEO low-e specialist" framing accurate, or is it confounded by env bugs and altitude/eccentricity conflation?
- What needs to be retracted, what stands?
- What's the cleanest next step (re-run Phase 5e evals with fixed filter? Extend training to higher altitudes? Both?)

---

## 7. Pre-committed acknowledgments

The user has explicitly stated they're frustrated with Phase 5's pattern of retract-and-retry. Three closure-and-retraction cycles have occurred. This investigation may surface a fourth.

If it does: report what's true. The honest framing is:

- Phase 5b's e ≤ 0.05 result probably stands (it doesn't depend on `valid_init_only`)
- Phase 5e and the wrap-up may both have issues that need to be redone with a working filter
- The recipe is genuinely a LEO specialist *because that's what was trained*, not because of some emergent property
- High-altitude / high-eccentricity capability would require retraining at those conditions

Don't try to rescue prior framings. The user wants accurate information about where they actually stand, not a face-saving narrative.

---

## 8. What this spec is NOT

- **Not a new closure spec.** Closure follows the verification, not precedes it.
- **Not a redo of Phase 5e or the wrap-up.** Those are the *subjects* of the investigation, not deliverables to be repeated.
- **Not a Phase 6 plan.** Phase 6 starts after we know what Phase 5 actually delivered.
- **Not a retraining spec.** No new training runs as part of this investigation.

---

## 9. After verification

Based on what I1-I4 find, the user will decide:

- If `valid_init_only` is broken: fix it, re-run Phase 5e evals, redo the surface. Probably 1-2 days of work.
- If `valid_init_only` is fine but trajectories are pre-fix: data hygiene issue. Re-export trajectories with proper tagging.
- If everything is fine but the wrap-up's framing is over-narrowed: relax the claims, write a more generous (but accurate) deliverable doc.
- If the recipe is genuinely a LEO specialist as the wrap-up claims and the user wants higher-altitude capability: plan a Phase 5.5 or Phase 6a that extends the training distribution.

The user will make this decision after seeing PHASE5_VERIFICATION_FINDINGS.md. Don't pre-empt it.

---

*Author: 2026-05-01. Phase 5 verification spec, after multiple closure retractions and the user's observation that post-5e trajectories appear to start inside Earth's surface. Investigation only — no new training, no new framings, no closure document. Just figure out what's actually true.*
