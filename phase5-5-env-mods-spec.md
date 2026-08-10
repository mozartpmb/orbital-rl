# Phase 5.5 — Env Mods Spec (Pre-curriculum)

> **Status:** 2026-05-15. Phase 5.5 pre-experiments (`PHASE5_5_PRE_FINDINGS.md`) surfaced four env-level issues that must be addressed before any altitude-curriculum training. This spec scopes the targeted fixes. Pattern mirrors `phase5-env-fix-spec.md`: named fixes (M1-M4), backward-compat validation (V1-V5), sequencing, pre-committed surprises. ~1-2 days engineering, ~3 hours validation compute. No new training as part of this work.

---

## 0. Why this spec exists separately

Three reasons to keep env mods separate from Phase 5.5 curriculum:

1. **Backward-compat risk is real.** The env mods touch the LVLH observation block (load-bearing in Phase 4-5b), the action table (Phase 4.5 found cross-phase action surgery doesn't transfer), and the Φ_orbit shaping (load-bearing). Each change needs explicit verification that Phase 5b's deliverable still reproduces.

2. **Scope creep would obscure attribution.** Per Phase 5.5 spec §7.6, piling env changes onto active training would confound "did the curriculum work?" with "did the env mod work?" Land env mods first, validate, then start curriculum.

3. **Same methodological pattern that landed env-fix cleanly.** The env-fix spec (`phase5-env-fix-spec.md`) shipped without surprises because it was scoped tightly to mechanical fixes with explicit validation. This spec follows the same pattern.

---

## 1. What's known broken

### 1.1 LVLH spatial observations saturate at MEO/GEO (E4)

Per pre-experiments, the LVLH block at `obs[33-37]` uses `R_EARTH ≈ 6.37e6` as the spatial normalizer in `orbital.h:585-586`:

```c
obs[33] = (float)(dx_l / R_EARTH);   /* relative position x in LVLH */
obs[34] = (float)(dy_l / R_EARTH);   /* relative position y in LVLH */
```

Measured worst-case |obs[33]| values from E4:
- LEO e=0.00: 2.157 (just past Box bound of 2.0)
- LEO e=0.70: 3.431
- MEO e=0.70: 6.583
- MEO_high e=0.70: 13.459
- GEO e=0.70: 20.306

The env-fix's `obs_alt_scale_m` kwarg only rescales position/distance obs at indices 0, 7, 17-32. The LVLH block was untouched. This is a meaningful gap with two consequences:

1. **Mandatory for MEO/GEO training.** Saturated observations train poorly; the policy sees clipped/squashed values that don't represent the underlying geometry.

2. **Potentially relevant for Phase 5b LEO bimodality.** Even at LEO with e=0.70, |obs[33]| = 3.43 exceeds Box bound. This may contribute to the 2/5 collapsed-seed pattern at the recipe edge.

### 1.2 5-minute warp is structurally inadequate at GEO (P4)

The current `WARP_TAU = 5` (5 minutes per warp action) gives:
- LEO orbital period: 90 min → 18 warps/period
- MEO orbital period: ~270 min → 54 warps/period
- GEO orbital period: 1440 min → **288 warps/period**

Per P4 measurements at GEO: the LEO ckpt chains 1500+ warp actions per episode without ever taking a non-coast burn. The agent literally cannot make meaningful temporal progress at GEO orbital periods.

This is structural: at any altitude where one orbit takes >50 warps to traverse, the agent's "wait for clear phasing window" strategy becomes infeasible.

### 1.3 Action set Δv is wrong for MEO/GEO (E6/P3)

Δa per Δv scales as 2a²·Δv/μ. At GEO (a=42 Mm), the smallest action (5 m/s prograde) moves orbit by **138 km — 14× the rendezvous tolerance** of 10 km. Single-burn fine-tuning is impossible at GEO with the current action set.

Bonus finding: at LEO, the 5 m/s action only moves orbit by 9 km — **below** SUCCESS_TOL_A = 10 km. A mild undershoot that may contribute to LEO fine-tuning friction.

### 1.4 Φ_orbit shaping K default breaks gating at MEO+ (E5)

Backward-compat at LEO with K=0.001 and obs_alt_scale_m=1.6e6 preserves legacy SUCCESS_TOL_A=10km behavior. But at MEO+ even with obs_alt_scale_m=4.2e7, Φ_orbit reaches 100+, well past σ₂ gate threshold of 2.0. The gated NHR shaping degenerates to single-component.

This is **config, not code** — the kwarg already exists from env-fix. The fix is in curriculum scripts, not env code. But it's documented here for completeness.

### 1.5 What's NOT broken (per probes)

- Kepler precision through GEO: PASS at all altitudes (E2).
- Cartesian↔elements round-trip through GEO: PASS (E3).
- Lambert reachability and altitude geometry: PASS (E1).
- The `valid_init_only` filter: works correctly (verified in env-fix work).

So the env mechanics are correct. What needs work is the observation normalization, the action table, and the warp duration.

---

## 2. Scope of fixes

Four named fixes (M1-M4) plus configuration changes. ~1-2 days engineering.

### 2.1 M1 — LVLH observation scaling

**The fix:** replace `R_EARTH` in the LVLH block normalization with a configurable altitude scale.

Options considered:

**Option M1a — Use existing `obs_alt_scale_m` for both blocks (unified).**
```c
/* orbital.h:585-586 */
obs[33] = (float)(dx_l / env->obs_alt_scale_m);
obs[34] = (float)(dy_l / env->obs_alt_scale_m);
```
Pros: single kwarg controls all observation scaling; matches the kwarg's name implication.
Cons: changes Phase 5b/5e LEO obs distribution — at LEO with default `obs_alt_scale_m=1.6e6`, current LVLH normalization is `R_EARTH=6.37e6`, so `dx_l / R_EARTH` becomes `dx_l / 1.6e6` (4× larger). **Breaks Phase 5b backward compat.**

**Option M1b — Add separate `lvlh_scale_m` kwarg.**
```c
/* orbital.h:585-586 */
obs[33] = (float)(dx_l / env->lvlh_scale_m);
obs[34] = (float)(dy_l / env->lvlh_scale_m);
```
Default `lvlh_scale_m = R_EARTH ≈ 6.37e6` preserves Phase 5b/5e behavior byte-identically. For MEO/GEO training, pass larger value (e.g., `lvlh_scale_m=4.2e7`).

Pros: clean backward compat; explicit kwarg names what it does.
Cons: two scale kwargs to keep in sync.

**Recommendation: M1b** (separate kwarg with backward-compat default). The unification option is tempting but breaks Phase 5b. Better to add a kwarg with a clear name than silently change observation distributions.

**Naming:** `lvlh_scale_m` is fine. Alternatives considered: `lvlh_obs_scale_m` (more descriptive), `relative_scale_m` (less coupled to LVLH-specific implementation). The first is shortest and the existing kwarg is `obs_alt_scale_m`, so `lvlh_scale_m` is consistent.

**Struct addition:**
```c
/* orbital.h env struct, near obs_alt_scale_m */
double lvlh_scale_m;  /* spatial normalizer for LVLH obs[33-34]; default R_EARTH */
```

**Plumbing:** binding.c unpacks; orbital.py adds kwarg; orbital.ini default = 6.371e6.

**Backward-compat verification:** V1 below.

### 2.2 M2 — Longer warp actions

**The fix:** add additional warp actions at higher action indices. Preserve existing Discrete(10) behavior so Phase 5b/5e ckpts still work.

Current action table (Discrete(10)):
```
0: coast (no burn, 1 sub-step)
1-7: prograde/retrograde/radial burns (1 sub-step each)
8: prograde 25 m/s (1 sub-step)
9: warp 5 min (5 sub-steps)
```

Proposed extension to Discrete(12):
```
0-9: unchanged (Phase 5b/5e backward compat preserved)
10: warp 30 min (30 sub-steps)
11: warp 1 hr (60 sub-steps)
```

GEO orbital period coverage:
- warp-5min: 288 actions/period
- warp-30min: 48 actions/period
- warp-1hr: 24 actions/period

24 actions per orbital period is the same order of magnitude as Phase 5b's LEO regime (18 warps/period). The recipe's temporal abstraction patterns should transfer.

**Why two warp actions, not one:** the agent needs intermediate granularity. warp-1hr alone forces 1-hour quantization on phase windows. warp-30min gives the agent ability to fine-tune timing within an hour. Adding both is cheap (~2 lines each).

**Why not warp-6hr:** considered but rejected for v1. At GEO with warp-1hr × 24 = 24 hours = 1 orbital period, the agent can coast a full orbit in one decision. Adding warp-6hr (4 actions/period at GEO) would let the agent skip multiple orbits, which is useful for very-long-phasing strategies but probably not necessary for v1. Can add later if curriculum reveals a need.

**Implementation:**

In `orbital.h`, extend the action table:
```c
#define ACTION_COUNT 12  /* was 10 */

static const float ACTION_DV[ACTION_COUNT] = {
    /* 0-9 unchanged from Phase 4-5b */
    0.0f, 10.0f, -10.0f, ...,
    /* M2: longer warps, no Δv */
    0.0f,  /* action 10: warp 30 min */
    0.0f,  /* action 11: warp 1 hr */
};

static const int ACTION_WARP_TAU[ACTION_COUNT] = {
    1, 1, 1, 1, 1, 1, 1, 1, 1, 5,
    /* M2: */
    30,  /* action 10: 30 sub-steps */
    60,  /* action 11: 60 sub-steps */
};
```

In `c_step`, warp dispatch reads `ACTION_WARP_TAU[action]` for sub-step count. Existing logic already handles `tau > 1` via the warp branch; just need to extend the action range it accepts.

**Action space change in `orbital.py`:**
```python
self.single_action_space = gym.spaces.Discrete(12)  # was 10
```

**Backward-compat for Phase 5b/5e ckpts:** the existing policy networks have a logits head of size 10. Adding actions 10-11 requires either:
- **Weight surgery:** extend the logits head to size 12, init the new entries with a low constant (e.g., -10 logit so they're never sampled). Existing weights unchanged at indices 0-9.
- **Wrapper layer:** during eval of Phase 5b/5e ckpts, slice the logits to indices 0-9 before sampling.

The second is cleaner because it doesn't modify ckpt files. Add a `legacy_action_space=10` kwarg to `eval_checkpoint.py` that masks indices 10-11.

**Decision:** wrapper-layer masking at eval time. Avoids ckpt surgery.

### 2.3 M3 — Sub-5 m/s actions

**The fix:** add finer-granularity Δv actions to the table.

Δa per Δv at altitude bands (from E6):
- LEO 5 m/s: 9 km (below tolerance 10 km — undershoots)
- MEO 5 m/s: 22 km (2× tolerance)
- MEO_high 5 m/s: 69 km (7× tolerance)
- GEO 5 m/s: 138 km (14× tolerance)

For GEO fine-tuning, we'd want ~0.5-1 m/s actions to produce ~10-30 km Δa.

Proposed extension to Discrete(16):
```
0-11: unchanged from M2 (Phase 5b/5e backward compat + M2 warps)
12: prograde +1 m/s
13: prograde -1 m/s (retrograde 1)
14: prograde +2 m/s
15: prograde -2 m/s
```

This gives sub-LEO-tolerance actions at GEO (1 m/s at GEO = 28 km Δa, larger than current 5 m/s at LEO = 9 km but smaller than 5 m/s at GEO = 138 km).

**Concerns:**

1. **Phase 4.5 found cross-phase action surgery doesn't transfer.** Adding actions that the policy hasn't trained on means a Phase 5b ckpt starts at suboptimal behavior on the new actions. This is fine for *new training* (the policy learns to use them); it's a problem only if we expect zero-shot transfer.

2. **The "broader action space dilutes random exploration" concern from Phase 5b/c.** Tested empirically:
   - Discrete(11) and Box(2) regressed Phase 5b at e_max=0.05 LEO (15M steps).
   - The mechanism: at LEO, 4/7 productive actions vs 4/11 (or 2/box) → less productive exploration per random sample.

   Does this argument apply to Discrete(16)? The four new actions (12-15) are sub-5m/s, which produce smaller Δa than even the smallest current action. Most are arguably "less useful" at LEO (where 5 m/s is already on the edge of tolerance). So the productive-rate at LEO might drop.

   Counter-argument: at MEO/GEO the productive rate is *higher* with sub-5m/s actions because the existing actions all overshoot. The trade-off is regime-dependent.

   **Decision: do it, but flag this as a real risk and add backward-compat verification.** The risk is that Phase 5b on Discrete(16) regresses from 96.4%. Mitigation: V2 below checks this.

3. **Should sub-5 m/s actions include radial?** At GEO, even radial 10 m/s might overshoot. But adding more actions amplifies the productive-rate dilution risk. **Decision: prograde/retrograde only for v1.** Add radial sub-actions later if curriculum reveals a need.

**Implementation:** extend ACTION_DV table, extend ACTION_WARP_TAU table (all 1's for non-warp actions), update Discrete(16) in orbital.py.

**Wrapper-layer masking** (same as M2): `eval_checkpoint.py` masks logits >= 10 (or >= 12 if M2 is also active) for Phase 5b/5e ckpts.

### 2.4 M4 — MAX_STEPS expansion (conditional)

**The need:** at GEO with 2000-step cap × 60s = 33 hr ≈ 1.4 orbital periods. Even with M2's warp-1hr action, single-orbital-period maneuvers might brush against the cap.

Per P4 measurements: the LEO ckpt at GEO ran 2000 steps with 1588 warps, covering 139 hours of sim time. With M2's warp-1hr (60 sub-steps each), the same trajectory would be ~88 warps covering similar sim time, well under 2000 steps.

**Verdict:** M4 may not be needed if M2 lands. Defer to a conditional check post-M2.

**Re-evaluation criterion:** after M2 lands, repeat the P4 measurement. If GEO episodes consistently hit the 2000-step cap before reaching meaningful capability, expand MAX_STEPS to 5000.

**For this spec:** document as conditional. Don't implement M4 until P4 re-measurement shows it's needed.

### 2.5 Configuration changes (no code)

Update `orbital.ini`:
```ini
# Phase 5.5 env mods (defaults preserve Phase 5b/5e LEO backward compat)
lvlh_scale_m = 6.371e6    # M1: LVLH normalizer; default = R_EARTH
# (existing) max_valid_init_attempts = 4096
# (existing) gave_up_action = accept
# (existing) obs_alt_scale_m = 1.6e6
# (existing) phi_orbit_scale_k = 0.001
```

Curriculum scripts for MEO/GEO training will need to set:
- `lvlh_scale_m=4.2e7` (or appropriate to altitude band)
- `obs_alt_scale_m=4.2e7`
- `phi_orbit_scale_k=0.05` (per E5 finding)

These go in the future curriculum scripts, not the ini defaults.

---

## 3. Validation protocol

Before declaring env mods done, V1-V5 must all pass. Pattern mirrors env-fix's V1-V4 with an additional backward-compat check for action-table changes.

### 3.1 V1 — LVLH backward compat

**Goal:** verify default `lvlh_scale_m=6.371e6` preserves Phase 5b/5e LEO observation distribution.

**Method:** Run 200-episode eval on Phase 5b deliverable ckpt at e=0.05 LEO with all default kwargs.

**Acceptance:** success rate within ±3pp of published 96.4-97.7%. (Spec target: 95-99%.)

If V1 fails: `lvlh_scale_m` default is wrong or M1 implementation has a bug. Don't proceed.

### 3.2 V2 — Action-space backward compat (M2 + M3 stack)

**Goal:** verify that adding actions 10-15 doesn't break Phase 5b's LEO behavior. The new actions should never be sampled by Phase 5b ckpts (which have logits-head of size 10), but the env must accept Discrete(16) and pass through correctly.

**Method:** Run 200-episode eval on Phase 5b deliverable ckpt with `--legacy-action-space 10` flag enabled (wrapper masks logits >= 10).

**Acceptance:** success rate within ±3pp of V1. If V2 differs substantially from V1, the wrapper masking has a bug.

Additionally, smoke-test: load Phase 5b ckpt without legacy masking (the policy outputs Discrete(10) logits; the env expects Discrete(16)). Eval should produce a clean error or work transparently via numpy broadcasting. The error case is fine; it means we know we need explicit masking.

### 3.3 V3 — New actions function at altitude

**Goal:** smoke test that the new actions (10-15) produce expected physics.

**Method:** Single trajectory with deterministic action sequence at GEO:
- Step 0-100: action 0 (coast)
- Step 100: action 12 (prograde +1 m/s)
- Steps 101-200: action 11 (warp-1hr)
- Etc.

Verify:
- Action 10 advances sim time by 30 minutes per call.
- Action 11 advances sim time by 60 minutes per call.
- Action 12 produces ~28 km Δa at GEO (predicted from vis-viva).
- Action 15 produces ~-56 km Δa at GEO.

**Acceptance:** physics matches predictions within 5%. No NaN, no crashes.

### 3.4 V4 — Φ_orbit gating at MEO with config (no code change)

**Goal:** verify that the existing `phi_orbit_scale_k` kwarg, set appropriately for MEO, produces operational gating.

**Method:** Eval Phase 5b ckpt at MEO altitudes with `obs_alt_scale_m=1.2e7, phi_orbit_scale_k=0.01`. Don't expect high success (ckpt is LEO-trained), but verify Φ_orbit stays tractable (max < 50 per episode).

**Acceptance:** Φ_orbit max < 50 across 50 episodes. Gating doesn't degenerate to permanently-closed.

### 3.5 V5 — Multi-seed Phase 5b backward compat (the headline check)

**Goal:** confirm Phase 5b's deliverable still produces multi-seed at 96.4% on the env-modded code.

**Method:** 5-seed multi-rollout eval at e=0.05 LEO with default kwargs (V1-pattern, but multi-seed). Use the 3 Phase 5b ckpts (seeds 31415, 42, 20260423) plus the 2 Phase 5e ckpts that work at e=0.05 LEO.

**Acceptance:** multi-seed mean within ±5pp of published 96.4%. (Spec target: 91-99%.)

If V5 fails: the env mods broke something subtle that V1-V4 didn't catch. Investigate before proceeding.

---

## 4. Sequencing

In order:

1. **M1 — LVLH scaling** (~1 hour engineering)
2. **V1 — LVLH backward compat** (~5 min compute)
3. **M2 — Warp actions** (~2 hours engineering, includes wrapper masking)
4. **M3 — Sub-5 m/s actions** (~1 hour engineering)
5. **V2 — Action space backward compat** (~5 min compute)
6. **V3 — New actions function at altitude** (~10 min compute + analysis)
7. **V4 — Φ_orbit gating verification** (~5 min compute)
8. **V5 — Multi-seed backward compat** (~30 min compute)
9. **Update orbital.ini** (defaults)
10. **Document forward-looking caveats**

If V1, V2, or V5 fail: pause, investigate, don't merge.

Total ~5-6 hours engineering + ~1 hour compute. Smaller than env-fix because the changes are more localized.

---

## 5. What this spec is NOT

- **Not Phase 5.5 curriculum.** That's the next spec to be written (or revised, since `phase5-5-altitude-expansion-spec.md` exists but predates these pre-experiment findings).
- **Not new training.** Existing ckpts are validated, not re-trained.
- **Not a complete env redesign.** Targeted fixes for known issues. Other things may surface during curriculum work; address them then.
- **Not action masking (Path A from Phase 5d).** That's a policy-level intervention. Different scope.
- **Not closure of Phase 5.** Closure follows curriculum completion.

---

## 6. Pre-committed acknowledgments

Phase 5's pattern is real (multiple closure-and-retraction cycles). For env mods specifically:

### 6.1 Surprise A — V1 fails (LVLH default doesn't preserve LEO)

This would mean `lvlh_scale_m=R_EARTH` doesn't byte-match the prior `R_EARTH` hardcode. Possibilities:
- The prior code uses a slightly different value (e.g., `R_EARTH * 1.001`).
- The plumbing has a bug (default not applied correctly).

**Response:** investigate. Don't proceed until V1 passes within ±1pp.

### 6.2 Surprise B — V2 fails (action space change breaks Phase 5b)

This would mean Phase 5b's policy somehow ends up sampling indices >= 10 despite the wrapper masking. Possibilities:
- The wrapper isn't applied (CLI flag not threaded).
- The policy's logits head was already larger than 10 (unlikely but possible).

**Response:** investigate. Don't proceed until V2 passes.

### 6.3 Surprise C — V5 fails (multi-seed regression)

Plausible scenarios:
- One of M1/M2/M3 has a subtle backward-compat issue that V1-V4 didn't catch.
- The Phase 5b bimodality (3/5 success, 2/5 collapse) makes multi-seed variance high; V5 falls outside ±5pp from natural seed noise alone.

**Response:** if the failure is "one seed flipped from success to collapse" (so 3/5 → 2/5), this is plausibly just seed noise — try 5 more seeds. If multiple seeds regress, the env mods are at fault.

### 6.4 Surprise D — Action-space dilution effect appears

This is the explicit risk in §2.3 #2. Phase 5b with Discrete(16) might regress at LEO because broader action space dilutes random-policy productive rate (the Phase 5b/c lesson).

**Detection:** if V5 multi-seed regresses by 5-15pp from Phase 5b's published numbers but no other validation fails.

**Response:** this is a real architectural finding worth documenting. Possible mitigations:
- Use action masking during training at LEO (mask indices 10-15 at low altitudes).
- Curriculum that starts with Discrete(10), expands to Discrete(16) at higher altitudes.
- Accept regression as the cost of altitude expansion.

Document and decide; don't fudge.

### 6.5 Surprise E — Φ_orbit K needs different default

E5 found K=0.05 works for within-GEO-band training, K=0.001 preserves LEO. The curriculum scripts will set this per-stage. But if E5's predictions don't match V4's measurements (e.g., Φ_orbit at MEO with K=0.01 actually goes higher than predicted), the per-stage K values need recalibration.

**Response:** measure Φ_orbit empirically in V4 across MEO/GEO. Adjust curriculum-script defaults to match measurements, not predictions.

### 6.6 The discipline at every step

Each M_x change gets:
- Backward-compat check (does Phase 5b still work?)
- Forward-looking check (does the new functionality work as predicted at altitude?)
- Documentation of decisions (why this design over alternatives)

No "ship it and see if Phase 5.5 curriculum exposes problems" attitude. The env-fix shipped cleanly because validation was thorough; this spec should too.

---

## 7. After env mods land

The path is:

1. Env mods + V1-V5 (this spec). ~1-2 days.
2. **Update `phase5-5-altitude-expansion-spec.md`** to reflect the new env state. Specifically: replace any "env modifications probably needed" references with "env mods landed in `phase5-5-env-mods-spec.md`; proceed to Stage 5.5.0."
3. **Stage 5.5.0** — multi-seed re-validate Phase 5b at LEO on the env-modded code. Anchors absolute backward-compat baseline.
4. **Stage 5.5.1+** per altitude expansion spec. Curriculum runs on validated env.

The env mods enable the curriculum. Without them, Phase 5.5 would either fail (Phase 5c/5d-style mystery collapses) or produce another retract-and-retry cycle.

---

## 8. Files touched

| File | Lines (estimate) | Purpose |
|---|---|---|
| `pufferlib/pufferlib/ocean/orbital/orbital.h` | ~15 lines | M1 lvlh_scale_m struct field + fill_observations; M2-M3 action table; M2 c_step warp dispatch |
| `pufferlib/pufferlib/ocean/orbital/binding.c` | ~5 lines | M1 kwarg unpack |
| `pufferlib/pufferlib/ocean/orbital/orbital.py` | ~10 lines | M1 kwarg; Discrete(16) action space; M2-M3 documentation |
| `pufferlib/scripts/orbital/eval_checkpoint.py` | ~10 lines | --legacy-action-space flag (wrapper masking) |
| `pufferlib/pufferlib/config/ocean/orbital.ini` | ~3 lines | lvlh_scale_m default |
| `scripts/orbital/p5_5_v3_new_actions.py` | new file (~80 lines) | V3 deterministic-action smoke test |
| `MODELS.md` | minor update | Note env-modded ckpts vs pre-env-mods |

Total: ~125 lines of code + a new smoke test script.

---

## 9. Reproduction recipe

After implementing M1-M3:

```bash
cd /Users/pete/space_training

# Build C extension
cd pufferlib && python3 setup.py build_ext --inplace --force && cd ..

# V1 — LVLH backward compat
python3 pufferlib/scripts/orbital/eval_checkpoint.py \
  pufferlib/experiments/puffer_orbital_177750405236/model_puffer_orbital_000350.pt \
  --episodes 200 --e-max-target 0.05 --e-max-sat 0.05 \
  --valid-init-only 1 --init-phase-gap-max 3.14159 \
  --seed 42
# Expect success ≥ 95%

# V2 — Action space backward compat (legacy masking)
python3 pufferlib/scripts/orbital/eval_checkpoint.py \
  pufferlib/experiments/puffer_orbital_177750405236/model_puffer_orbital_000350.pt \
  --episodes 200 --e-max-target 0.05 --e-max-sat 0.05 \
  --valid-init-only 1 --init-phase-gap-max 3.14159 \
  --legacy-action-space 10 --seed 42
# Expect success within ±3pp of V1

# V3 — Deterministic action smoke at GEO
python3 scripts/orbital/p5_5_v3_new_actions.py
# Expect: action 10 advances 30min sim time; action 11 advances 1hr;
# actions 12/14 produce predicted Δa at GEO

# V4 — Phi_orbit gating verification at MEO
python3 pufferlib/scripts/orbital/eval_checkpoint.py \
  pufferlib/experiments/puffer_orbital_177750405236/model_puffer_orbital_000350.pt \
  --episodes 50 --e-max-target 0.20 --e-max-sat 0.20 \
  --valid-init-only 1 --init-phase-gap-max 3.14159 \
  --a-min-override 1.0e7 --a-max-override 1.3e7 \
  --obs-alt-scale-m 1.2e7 --phi-orbit-scale-k 0.01 \
  --log-phi-orbit 1 --seed 42
# Expect: max(Phi_orbit) < 50 across all 50 episodes (gate operational)

# V5 — Multi-seed backward compat (the headline check)
for SEED_DIR in puffer_orbital_177750405236 puffer_orbital_177750198246 puffer_orbital_177750301624; do
  python3 pufferlib/scripts/orbital/eval_checkpoint.py \
    pufferlib/experiments/$SEED_DIR/model_puffer_orbital_000350.pt \
    --episodes 200 --e-max-target 0.05 --e-max-sat 0.05 \
    --valid-init-only 1 --init-phase-gap-max 3.14159 \
    --seed 42
done
# Expect: multi-seed mean ≥ 91% (published 96.4% ± 5pp)
```

---

## 10. Commit plan (pending validation)

Single commit on `main`:

```
Phase 5.5 env mods (M1-M3 + V1-V5)

M1: LVLH spatial obs configurable via lvlh_scale_m kwarg. Default
    6.371e6 (R_EARTH) preserves Phase 5b/5e LEO backward compat.
    Set to ~4.2e7 for GEO-inclusive training.

M2: Two new warp actions at indices 10-11. Action 10 = warp 30 min
    (30 sub-steps), action 11 = warp 1 hr (60 sub-steps). Action space
    expanded Discrete(10) → Discrete(12). Existing actions 0-9
    unchanged.

M3: Four new sub-5 m/s prograde/retrograde actions at indices 12-15
    (±1 m/s, ±2 m/s). Action space expanded Discrete(12) → Discrete(16).
    Enables fine-grained control at MEO/GEO where 5 m/s overshoots.

Eval pipeline: --legacy-action-space 10 flag for backward-compat eval
of Phase 5b/5e ckpts. Masks logits >= 10 so the policy can't sample
the new actions.

Validation:
  V1: LVLH backward compat at LEO: 96.X% (published 96.4-97.7%)
  V2: Action-space backward compat with masking: 96.X%
  V3: New actions function at GEO (deterministic smoke test passes)
  V4: Φ_orbit gating verified at MEO with K=0.01: max < 50
  V5: Multi-seed backward compat: 9X.X% mean (published 96.4%)

Closes pre-experiments findings #1, #2, #3. Phi_orbit K (finding #4)
remains as configuration per altitude stage.
```

Followed by tag `phase5-5-env-mods`.

---

## 11. Open questions for the curriculum follow-up

These are intentionally NOT addressed in this spec; they belong in the (revised) Phase 5.5 curriculum work:

1. **What's the right per-stage `phi_orbit_scale_k`?** E5 found K=0.05 works for within-GEO-band; K=0.001 preserves LEO. The curriculum needs a schedule.

2. **Does adding actions 10-15 cause measurable LEO regression in re-trained policies?** V5 tests Phase 5b ckpts (which can't use 10-15 anyway via masking). New training from scratch with Discrete(16) might regress vs Discrete(10) due to the productive-rate dilution issue.

3. **Should the M4 MAX_STEPS expansion happen?** Conditional on post-M2 P4 re-measurement.

4. **Does the Phase 5b ckpt have any latent altitude generalization that wasn't visible because of OOD observations?** With M1's fixed LVLH scaling, the Phase 5b ckpt at MEO/GEO might still hit 0% (no surprise) or might show some weak signal. Worth a quick re-measurement after M1 lands as a curiosity check.

---

*Author: 2026-05-15. Phase 5.5 env-mods spec. Four named fixes (M1 LVLH scaling, M2 longer warps, M3 sub-5 m/s actions, M4 conditional). Five validation checks (V1-V5). ~5-6 hours engineering + ~1 hour compute. Prerequisite for Phase 5.5 curriculum work. Follows pattern of phase5-env-fix-spec.md.*
