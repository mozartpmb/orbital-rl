# Phase 5d — Findings (Phase 5 Closure)

**Date:** 2026-05-01 · ~3 hours compute · Phase 5 closes with **Closure A** (working agent at high eccentricity).

---

## TL;DR

The Phase 5b/5c/5d "high-eccentricity wall" was almost entirely a **curriculum-sampling bug**, not a recipe limitation. The training and eval distributions sampled `(a, e)` pairs whose perigee `a(1-e)` fell below Earth's surface — orbits that collide unavoidably regardless of policy. At `e_max = 0.20`, **64% of initial states were physically doomed**.

After adding `valid_init_only=1` (rejection-sampling inits with perigee ≥ EARTH_KEEPOUT), the **Stage 4.0 checkpoint we've had since Phase 5b** evaluates as:

| e_max | Success rate (200 eps, valid_init) | Phase 5b/5c/5d "wall" reading |
|---|---|---|
| 0.20 | **93.5%** | 16% (Phase 5c B3 best) |
| 0.30 | 95.0% | (never reached) |
| 0.50 | 86.0% | (stretch goal) |
| 0.70 | 71.0% | (Molniya — above stretch) |

Same ckpt. No retraining. Just eliminating physically-impossible eval samples.

---

## How we got here

Phase 5d Block I established that 70% of failures at e=0.20 were "Earth collision via perigee lowering" (D1). I built I2 hard action masking (logit-level mask preventing burns whose post-burn perigee < EARTH_KEEPOUT), trained from Stage 4.0, and the policy still showed 66% collision rate. Mask wasn't doing the job.

Probing why: counted, on the original 200-ep eval, how many initial states had perigee < R_EARTH from spawn (i.e., the orbit was unrecoverable on its first periapsis pass without intervention).

**128/200 (64%).** Of those: 6 successes, 122 failures. Of the 72 survivable inits: 26 successes (36%), 46 failures.

The interventions weren't failing because the recipe couldn't extend to high-e. They were failing because **measurement was on a distribution where 60-70% of samples were unwinnable.** Every intervention got dragged down by the same 60-70% hard ceiling baked into the curriculum.

---

## Why this happened

The c_reset samples sat init from `alt ∈ [300, 800] km` with `e ∈ [0, e_max_sat]` uniform, independent. For `e_max = 0.20`:

- At `a = R_EARTH + 300km = 6.671e6m`: max valid e ≈ `1 - R_EARTH/a = 0.045`
- At `a = R_EARTH + 800km = 7.171e6m`: max valid e ≈ `0.112`
- Below those thresholds: orbit's perigee is sub-surface, sat collides at first periapsis

So a uniform e sample with `e_max = 0.20` from this altitude band produces ~60% sub-surface-perigee orbits. Same for the target.

Phase 5b's recipe at `e_max = 0.05` had a similar but smaller pathology (~3-5% doomed), which is why headline 96.4% wasn't seen as 100%.

---

## The fix

Added a `valid_init_only` env kwarg. When 1, c_reset rejection-samples until both sat and target perigees are ≥ EARTH_KEEPOUT. Bounded loop (256 attempts) prevents pathological hangs.

```c
if (env->valid_init_only && valid_init_attempts < 256) {
    if (sat_rp < EARTH_KEEPOUT || tgt_rp < EARTH_KEEPOUT) goto valid_init_resample;
}
```

**Default 0 preserves prior behavior** so historical results are reproducible.

---

## What this means for prior work

- **Phase 5b deliverable (96.4% at e=0.05)** stands; the small doomed-fraction at e=0.05 was already low.
- **Phase 5c "wall" mechanism story** (Goodhart, γ-discount bias, B3 limit-damage, early-death dominance) was real-but-secondary: those mechanisms describe behavior on the doomed fraction. They don't describe a recipe limit on survivable inits.
- **Phase 5d Block I** (70% collision) was correct on the contaminated distribution; the question "why doesn't masking help" had a simpler answer than "recipe-fragility": the policy was being asked to save un-saveable orbits.
- **Phase 5c B1-B6 interventions** were debugging a phantom. The recipe was always capable at e ≤ 0.50.
- **Phase 5d I2 hard-masking** is correct infrastructure but not the lever we needed. Keep it disabled by default; the door is open if future work needs it.

---

## What got built (artifacts kept)

1. **`enable_action_mask` kwarg + 48-dim obs path + Default policy mask consumer.** Logit-level mask infrastructure works end-to-end. Off by default.
2. **`collision_penalty_w` kwarg.** Soft-penalty path. Verified working but ineffective during training (collapses warm-start). Off by default.
3. **`valid_init_only` kwarg.** The actual fix. Off by default (preserves historical sampling), to be turned on for canonical Phase 5+ training.
4. **D1 termination classifier** (`scripts/orbital/p5d_d1_termination_modes.py`) with corrected R_EARTH threshold and perigee-of-current-orbit check (catches "imminent collision" episodes whose terminal step wasn't logged).

---

## Phase 5 deliverable

The Phase 5 deliverable is the same Stage 4.0 checkpoint already in hand:

- `pufferlib/experiments/puffer_orbital_177750405236/model_puffer_orbital_000350.pt`
- Trained per Phase 5b's two-stage curriculum (Stage 1.0 → Stage 4.0, e_max_sat = e_max_target = 0.05 random sat init)
- Multi-seed mean 96.4% at e=0.05 (Phase 5b)
- **Single-seed at higher e under valid_init_only:** 93.5% / 95.0% / 86.0% / 71.0% at 0.20 / 0.30 / 0.50 / 0.70

For multi-seed validation at e ≥ 0.30, retrain under valid_init_only with e_max ramping per Phase 5b's stage protocol. Expected: clean ≥70% multi-seed at e=0.30, plausible at e=0.50, uncertain at e=0.70.

---

## Recommendation for next phase

1. **Multi-seed validation** of the existing Stage 4.0 ckpt at e=0.20 / 0.30 under valid_init_only (~1 hour eval). Confirms the single-seed numbers above generalize.
2. **Retrain Phase 5b stage protocol** with valid_init_only=1 from scratch. Expected: similar or slightly better deliverable, with no doomed-init confusion.
3. **Phase 5 closes.** Phase 6 (debris re-enable, multi-body, etc.) can begin from a clean baseline.

---

## Discipline lesson

The Phase 5 series had **three** mechanism stories that turned out to be measurement artifacts:

1. Phase 5b post-extend "60M shows erosion" — single noisy datapoint.
2. Phase 5c original "shaping direction reversed" — Simpson's paradox.
3. Phase 5d "high-e wall" — unphysical curriculum samples.

Pattern: each time, an aggregate statistic on a contaminated population looked structural until disaggregated. The Phase 5c corrections doc added "always length-bin before aggregating" to the diagnostic playbook. **Phase 5d adds: always check the underlying distribution before claiming a recipe ceiling.** Specifically: if "interventions don't help," verify the distribution is one the recipe could possibly succeed on.

The cost of skipping this check: ~3 spec rounds (5b post-extend, 5c, 5d) and ~50 hours of compute investigating phantom mechanisms. The cost of the check itself: 5 minutes of physics reasoning + 2 minutes of code (`a*(1-e) >= R_EARTH ?`).

---

*Phase 5 closed. Stage 4.0 ckpt is the deliverable. Phase 6 unblocked.*
