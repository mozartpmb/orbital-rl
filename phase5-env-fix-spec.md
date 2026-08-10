# Phase 5 — Env Cleanup Spec (Pre-altitude-expansion)

> **Status:** 2026-05-02. The Phase 5 verification investigation (PHASE5_VERIFICATION_FINDINGS.md) identified two real env bugs: (1) `valid_init_only=1`'s 256-attempt cap exhausts at high-e in LEO and accepts doomed inits, contaminating ~25pp of measured success at e=0.70; (2) the trajectory export script's metadata records the kwarg, not the outcome. This spec scopes the small fixes that need to happen regardless of what comes next. ~3 hours of work, no compute beyond verification. These fixes are prerequisites for both the LEO closure path AND for altitude expansion (where the same bug class would recur).

---

## 0. Why this spec exists separately from altitude expansion

Two reasons:

1. **The bugs need fixing whether or not we pursue altitude expansion.** Anyone running future evals needs an env that doesn't silently accept physically impossible inits. Anyone reading trajectory metadata needs metadata that records what actually happened.

2. **The fix is small and verifiable.** Keeping it separate from altitude expansion means we get clean validation of "the env is now correct" before stacking new training work on top.

Don't combine this with altitude expansion. Land env-fix first; verify; then build on it.

---

## 1. What's known broken

### 1.1 The 256-attempt cap (verified)

In `pufferlib/ocean/orbital/orbital.h::c_reset`, the rejection-sampling loop has a hardcoded cap at 256 attempts:

```c
int valid_init_attempts = 0;
valid_init_resample:
valid_init_attempts++;
// ... sample sat & target ...
if (env->valid_init_only && valid_init_attempts < 256) {
    double sat_rp = env->sat.orbit.a * (1.0 - env->sat.orbit.e);
    double tgt_rp = env->target.a * (1.0 - env->target.e);
    if (sat_rp < EARTH_KEEPOUT || tgt_rp < EARTH_KEEPOUT) {
        goto valid_init_resample;
    }
}
```

When the cap exhausts, the env accepts whatever was last drawn. Per I1's instrumentation, at e_max=0.70 in LEO this happens on 31.4% of resets. Per I2's audit of `web_data/runs/phase5e_seed42_e0.70/`, 12 of 46 trajectories (26%) have sat or target perigee below Earth's surface — they're cap-exhaust artifacts that the env accepted as initial states.

### 1.2 Metadata records inputs, not outcomes (verified)

`scripts/orbital/export_web_data.py` writes `metadata.env_config.valid_init_only` from the CLI flag passed to the eval run. This is the *intended* setting, not the *realized* sampling outcome. A trajectory file says `valid_init_only: 1` even when the cap exhausted and accepted a doomed init.

### 1.3 What's not broken (per verification)

- The plumbing of `valid_init_only` is correct end-to-end: kwarg → Python → binding → C struct → c_reset.
- The OR-check correctly filters both sat and target.
- At low-e LEO, the filter has minimal effect (~5pp at e_max=0.05), and Phase 5b's e ≤ 0.05 deliverable is robust.

So the env code is mostly correct; only the cap and the metadata recording are wrong.

---

## 2. Scope of fixes

Three small fixes plus instrumentation. Total ~3 hours engineering.

### 2.1 F1 — Configurable rejection-sampling cap

Replace hardcoded `256` with a kwarg `max_valid_init_attempts` (default 4096). At 4096, even the e_max=0.70 LEO regime (where ~88% of samples are physically invalid, so per-attempt acceptance is ~12%) has a cap-exhaust rate of `(0.88)^4096 ≈ 0` — essentially never gives up.

```c
// orbital.h c_reset:
int valid_init_attempts = 0;
valid_init_resample:
valid_init_attempts++;
// ... sample ...
if (env->valid_init_only && valid_init_attempts < env->max_valid_init_attempts) {
    // ... check ...
}
// after loop:
env->last_init_attempts = valid_init_attempts;
env->last_init_gave_up = (env->valid_init_only && valid_init_attempts >= env->max_valid_init_attempts);
```

Add `max_valid_init_attempts` and the two outcome fields to the env struct. Plumb the cap through binding.c, orbital.py, eval_checkpoint.py. Default 4096 in orbital.ini.

**Performance check:** at e_max=0.70 LEO, expected cost is ~145 attempts (per I1 Eval 2 mean). At 4096 cap with realistic eval distributions, mean attempts will rarely exceed ~500, so the eval cost goes from ~31% cap-exhausts to <0.01% with maybe 2× higher attempts on average. Still well under 1ms per reset.

### 2.2 F2 — Outcome metadata in trajectory exports

`export_web_data.py` should record:
- `last_init_attempts` (how many attempts the rejection sampler took)
- `last_init_gave_up` (whether the cap exhausted)
- `realized_sat_perigee_m`, `realized_target_perigee_m` (the actual perigees of the accepted orbit)

These need to be exposed from the C env via the existing `vec_get_trajectory` mechanism (or a new accessor if needed). Then `export_web_data.py` reads them and writes to `metadata.realized_init`.

The existing `metadata.env_config` field stays — it records intent. The new `metadata.realized_init` field records outcome. Consumers (web frontend, downstream analysis) can filter on either.

**Acceptance criterion:** for the same seed, an eval run produces trajectory files where `metadata.realized_init.sat_perigee_m * (1 - sat_e) ≥ EARTH_KEEPOUT` for every file with `gave_up: 0`, and the `gave_up: 1` cases (if any) are rare and explicit.

### 2.3 F3 — Gave-up handling at the env level

When the cap exhausts, the env should choose between two behaviors based on a kwarg:
- `gave_up_action="accept"` (current behavior): accept the last sample; record `last_init_gave_up=1`.
- `gave_up_action="terminate"` (new option): immediately terminate the episode with reward 0 (not -10) and a special termination flag. The agent gets no learning signal from the doomed init; it's filtered out at the population level.

For most evals, "terminate" is the right behavior — it cleanly excludes doomed inits from success-rate calculations. For training, "accept" preserves backward compatibility with Phase 5b/5e ckpts (which trained on whatever the sampler produced, including occasional cap-exhausts at low rates).

**Default:** `gave_up_action="accept"` for backward compat. New eval scripts pass `gave_up_action="terminate"` by default.

This isn't strictly necessary if F1 sets the cap high enough that exhausts never happen. But it's cheap to add and protects against future configurations where the cap could still be hit.

### 2.4 F4 — Instrumentation kept on

The `log_validation_debug` kwarg added by the verification investigation is generally useful. Keep it. Make sure `eval_checkpoint.py` exposes it. Anyone running future evals can use it to verify the filter is doing what they expect.

Default off; flag-on doesn't change behavior, just emits stderr lines per reset.

---

## 3. Validation protocol

Before declaring the env-fix done, verify:

### 3.1 V1 — Cap raise eliminates exhausts

Run instrumented eval (`log_validation_debug=1`) at e_max=0.70 in LEO with `max_valid_init_attempts=4096`. 200 episodes. Verify `last_init_gave_up=0` for all 200 episodes (or document the rate if any gave up).

If any cases give up at 4096 attempts: investigate. Either the cap needs to be higher, or the configuration is genuinely degenerate (e.g., e_max=0.99 LEO would have ~0% acceptance rate even at 100K attempts).

### 3.2 V2 — Metadata records realized outcomes

Run 50-episode eval with `valid_init_only=1`, `max_valid_init_attempts=4096`. Open the output JSON files. For each:
- `metadata.realized_init.sat_perigee_m / 1000 ≥ 6571` (km, EARTH_KEEPOUT)
- `metadata.realized_init.target_perigee_m / 1000 ≥ 6571`
- `metadata.realized_init.gave_up == 0`

Spot-check 5 files manually. Confirm the perigees match what the trajectory shows.

### 3.3 V3 — Phase 5b deliverable still reproduces

Run 200-episode eval at e_max=0.05 LEO on the canonical Phase 5b ckpt with the new code. Compare to the published Phase 5b 96.4% number. Should be within ~3pp (since e=0.05 LEO has minimal cap-exhaust contamination).

If it's substantially different (>5pp): investigate whether F1/F3 changed something subtler than expected. The fix shouldn't change behavior at conditions where the cap wasn't being hit.

### 3.4 V4 — Web frontend trajectory contamination

For the existing `web_data/runs/phase5e_seed42_e0.{50,70}/` trajectories, the realized perigees should be added retroactively (compute them from the existing JSON `initial` fields) and the contaminated trajectories flagged. New evals can be added with the cap raise.

This is post-hoc analysis on existing data, ~30 min.

---

## 4. What this spec is NOT

- **Not a re-eval of Phase 5e numbers at the corrected env.** That's a separate decision: do we want to know what the recipe does in LEO with no doomed inits? Probably yes for closing Phase 5b's deliverable cleanly, but it's a follow-up to env-fix, not part of it.
- **Not altitude expansion.** Phase 5.5 is the next spec; this is prerequisite cleanup.
- **Not retraining.** No new training. Existing ckpts are fine to keep.
- **Not a closure document.** Phase 5 closure happens after altitude expansion, not after env-fix.

---

## 5. Sequencing

In order:

1. **F1** (cap raise + outcome fields). ~1 hour engineering.
2. **F2** (metadata in exports). ~45 min engineering.
3. **F3** (gave-up termination option). ~30 min engineering.
4. **F4** (verify validation logging works). ~10 min.
5. **V1-V4** (validation runs). ~45 min.
6. Land in main branch. Tag `phase5-env-fix`.

Total ~3-4 hours wall.

---

## 6. After env-fix

The env is now correct. Two paths:

**Path I — Re-eval Phase 5b deliverable cleanly.**
With the fixed env, run a clean 5-seed e-scan at LEO eccentricities (fixed e ∈ {0.0, 0.025, 0.05}, the physically-valid LEO range). Multi-seed validates the deliverable. ~1 hour compute. Gives an accurate picture of "what Phase 5b actually delivers."

**Path II — Move directly to altitude expansion.**
Phase 5.5 spec (separate document). Curriculum design for altitude × eccentricity expansion.

Path I is cheap and grounds the deliverable framing. I'd run it before Path II to have a baseline that altitude expansion can be measured against.

---

## 7. Why this doesn't need pre-committed acknowledgments

This spec is mechanical. Three named fixes, four validation checks, ~3 hours of work. There's no mechanism story to be wrong about, no aggregation to be confounded, no curriculum to be mis-designed. The risk is just engineering: the fix doesn't work first try, or breaks something else. Standard development risk, not project-pattern risk.

The pre-committed acknowledgments in the next spec (altitude expansion) are heavier because that work is structurally similar to the work that produced multiple retractions.

---

*Author: 2026-05-02. Phase 5 env-fix spec. Three fixes (cap raise, outcome metadata, gave-up handling) plus instrumentation. ~3 hours. Prerequisite for any subsequent eval or altitude expansion work.*
