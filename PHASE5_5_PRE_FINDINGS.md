# Phase 5.5 Pre-Experiments Findings — Pre-flight + P1-P4

> **Status:** 2026-05-15. Pre-experiment phase complete. Per the Phase 5.5 spec §2.5, this document records env behavior at high altitude and the existing Phase 5b ckpt's transfer behavior. Per spec §9.1, the findings here gate any subsequent curriculum work — substantial env mods are required before Stage 5.5.0 onwards.
>
> **Spec source:** `phase5-5-altitude-expansion-spec.md`
> **Plan source:** `/Users/pete/.claude/plans/let-s-think-really-hard-stateless-dream.md` (Plan 1 of N)
> **Prior context:** env-fix landed (commit `4b41cdc`); see `PHASE5_ENV_FIX_IMPLEMENTATION.md`.

---

## TL;DR

Six probes (P1: E1-E6) plus zero-shot eval (P2) plus warp adequacy (P4). Verdict:

- **Env is numerically correct at all altitudes through GEO** (E2, E3, E1).
- **LVLH spatial observations saturate badly at MEO/GEO** (E4). |obs[33]| reaches ~20 at GEO (vs Box bound 2.0). The B.1 `obs_alt_scale_m` from env-fix rescales the *position* obs[0/7/17-32] but does NOT touch the LVLH block.
- **Φ_orbit shaping with default K=0.001 produces gate-killing magnitudes at MEO/GEO** (E5). Within-altitude-band Δa needs K≥0.005 to keep Φ tractable; cross-altitude training needs K≥0.05.
- **Action set's smallest Δv (5 m/s) overshoots at GEO** (E6/P3): 14× rendezvous tolerance per single burn. Sub-5 m/s actions needed for MEO+.
- **Phase 5b ckpt shows zero altitude generalization** (P2): 0/600 at every MEO/GEO cell across 3 rollout seeds × 2 obs-scale settings. LEO baseline reproduces at 98.3% (pipeline integrity confirmed).
- **5-minute warp action is structurally inadequate at GEO** (P4): the LEO ckpt chains 1500+ warp actions per episode, never takes a non-coast burn. GEO orbital period (24 hr) overwhelms the 5-min warp cadence.

**Net assessment:** Phase 5.5 cannot proceed directly from Stage 5.5.0 (re-validate Phase 5b at LEO) to Stage 5.5.1 (altitude extension) without env modifications first. The "altitude expansion" framing per the spec needs to be extended to "altitude expansion preceded by targeted env mods." Both spec §3.x (env modifications probably needed) and spec §7.6 (scope creep risk) anticipated this; the findings make the prerequisites concrete.

Recommended sequencing (revised):
1. Stage 5.5.0 — multi-seed re-validate Phase 5b at LEO in env-fix code (single plan; ~2 hrs compute). Anchors backward compat.
2. **Env-mods follow-up spec** — new doc scoping the 4 prerequisites (action set, warp duration, LVLH normalization, Φ_orbit K default). ~1-2 days engineering + tests.
3. Stage 5.5.1 onwards — proceed with curriculum stages on the env-modded code.

---

## 1. Pre-flight

### 1.1 Env-fix verified
Commit `4b41cdc` (2026-05-12) landed F1-F4 + B.1-B.4. V3 reproduction confirmed Phase 5b @ 98.0% at e=0.05 LEO with default kwargs. No drift from published 96.4-97.7%.

### 1.2 Canonical ckpts registered
`MODELS.md` written 2026-05-15. Three Phase 5b ckpts (seeds 31415, 42, 20260423) and four Phase 5e ckpts confirmed present on disk. The seed-31415 ckpt at `puffer_orbital_177750405236/model_puffer_orbital_000350.pt` is the canonical Phase 5b reference and used throughout these probes.

### 1.3 Block I probe templates inventoried
- `scripts/orbital/p5e_e1_lambert.py` — Lambert reachability (~334 lines, full solver)
- `scripts/orbital/p5e_e2_kepler_precision.py` — Newton-Raphson precision
- `scripts/orbital/p5e_e3_round_trip.py` — Cartesian↔elements roundtrip
- `scripts/orbital/p5e_e5_phi_calib.py` — Φ_orbit formula
- `scripts/orbital/p5e_e6_action_effect.py` — Δa per Δv analytical
- (No E4 standalone — written fresh for this probe.)

---

## 2. P1 Block I extended to high altitude

Altitude cells: LEO (a=6.87 Mm), MEO_low (7.47 Mm), MEO (12.37 Mm), MEO_high (26.57 Mm, GPS), GEO (42.16 Mm).

### 2.1 E1 — Lambert reachability

Script: `scripts/orbital/p5_5_p1_e1_lambert_alt.py`. Output: `/tmp/p5_5_p1_e1.log`.

Lambert solver validates against Hohmann 400→2000 km at 1.5% error (170° transfer proxy, avoiding degenerate 180° case). Solver is altitude-agnostic and works at all bands. Key per-band Lambert stats (median, p10, fraction under fuel budget):

| Band     | e_max | doomed_KO | hohm>bud | lamb_med (m/s) | lamb_p10 (m/s) | lamb<bud |
|----------|------:|----------:|---------:|---------------:|---------------:|---------:|
| LEO      | 0.05  |     41.0% |     0.0% |           6517 |           1759 |     1.0% |
| LEO (valid_init) | 0.05 | 0.0% |  0.0% |           7151 |           1968 |     1.0% |
| LEO      | 0.50  |     99.0% |     0.0% |           6951 |           2690 |     0.0% |
| MEO_low  | 0.05  |      0.0% |     0.5% |           6058 |           1919 |     1.0% |
| MEO      | 0.05  |      0.0% |     0.0% |           4812 |           1494 |     2.0% |
| MEO_high | 0.05  |      0.0% |     0.0% |           3335 |            991 |     4.0% |
| GEO      | 0.05  |      0.0% |     0.0% |           2525 |            789 |     5.0% |
| GEO      | 0.50  |      0.0% |     0.0% |           3026 |            845 |     2.0% |

**Findings:**
- At MEO and above, random sampling rarely produces doomed orbits (geometric constraint relaxes).
- Lambert Δv per task decreases with altitude (3335 m/s LEO → 2525 m/s GEO median) — counterintuitive but consistent: at higher altitudes the slower orbital velocities mean smaller absolute Δv to change orbit shape.
- **But Lambert is still well above the 478 m/s fuel budget across all altitudes.** Lambert 2-impulse is a rough upper bound; the agent's actual policy is multi-burn with phasing, which can be much cheaper. This is informational, not a feasibility test.

**Verdict:** Lambert solver works at all altitudes. Env is correct on this axis.

### 2.2 E2 — Kepler precision

Script: `scripts/orbital/p5_5_p1_e2_kepler_alt.py`. Output: `/tmp/p5_5_p1_e2.log`.

Propagated each (alt, e) cell for 100 orbital periods at DT=60s. At GEO this is 100 × 24 hr = 100 days of sim time, 143570 steps. Max drift in (a, e, ω) per cell:

| Band      | e=0.05 max\|Δa\| | e=0.70 max\|Δa\| | All Δe, Δω |
|-----------|------------------:|------------------:|-------------:|
| LEO       | 9.3e-9            | 3.5e-8            | < 1e-14       |
| MEO       | 2.1e-8            | 5.8e-8            | < 1e-14       |
| MEO_high  | 4.1e-8            | 1.2e-7            | < 1e-14       |
| GEO       | 7.5e-8            | 2.0e-7            | < 1e-14       |

**Verdict:** PASS at all cells (pass criterion: < 1e-6). Worst case drift at GEO/e=0.70 over 100 periods = 0.2 microns. Kepler propagation is numerically robust through GEO. Eccentricity and ω drifts are at floating-point precision (~1e-15).

### 2.3 E3 — Cartesian↔elements round-trip

Script: `scripts/orbital/p5_5_p1_e3_roundtrip_alt.py`. Output: `/tmp/p5_5_p1_e3.log`.

100 random orbits per cell. Worst-case round-trip Δa at GEO/e=0.70: 1.04e-7 m. Δe, Δω, Δθ all at floating-point precision.

**Verdict:** PASS. Cartesian↔elements is altitude-robust through GEO. The vis-viva subtraction (1/a = 2/r - v²/μ) doesn't lose precision at high altitude despite 1/a being small.

### 2.4 E4 — LVLH frame magnitudes (NEW probe)

Script: `scripts/orbital/p5_5_p1_e4_lvlh_alt.py`. Output: `/tmp/p5_5_p1_e4.log`.

Mirrors the LVLH-frame computation from `orbital.h:560-589`. Each cell sampled 100 random (sat.ω, sat.M, target.ω, target.M) configurations with sat & target at same a (mirrors Stage 1 same_orbit_init regime). Worst-case |obs| per cell:

| Band      | e    | n_tgt    | v_circ | \|obs33\| | \|obs34\| | \|obs35\| | \|obs36\| | obs37 |
|-----------|-----:|---------:|-------:|----------:|----------:|----------:|----------:|------:|
| LEO       | 0.00 | 1.11e-3  | 7617   | **2.157** | 1.078     | 0.000     | 0.000     | 1.109 |
| LEO       | 0.50 | 1.11e-3  | 7617   | **3.125** | 1.585     | 1.722     | **2.117** | 1.109 |
| LEO       | 0.70 | 1.11e-3  | 7617   | **3.431** | 1.805     | **2.783** | **3.330** | 1.109 |
| MEO       | 0.20 | 4.59e-4  | 5676   | **4.440** | **2.320** | 0.574     | 0.771     | 0.459 |
| MEO       | 0.70 | 4.59e-4  | 5676   | **6.583** | **3.279** | **2.681** | **2.907** | 0.459 |
| MEO_high  | 0.70 | 1.46e-4  | 3873   | **13.459**| **7.033** | **2.147** | **3.000** | 0.146 |
| GEO       | 0.00 | 7.29e-5  | 3075   | **13.226**| **6.617** | 0.000     | 0.000     | 0.073 |
| GEO       | 0.70 | 7.29e-5  | 3075   | **20.306**| **11.139**| **2.056** | **2.622** | 0.073 |

(Bolded values exceed the gymnasium Box(-2, 2) bound.)

**Critical finding:** the LVLH spatial obs[33-34] use `dx_l / R_EARTH` normalization in orbital.h:585-586. R_EARTH ≈ 6.37 Mm. At GEO (a=42.16 Mm), the relative position between two satellites on opposite sides of the orbit reaches ~84 Mm — divided by R_EARTH gives obs ≈ 13.2. The B.1 `obs_alt_scale_m` kwarg (added in env-fix) only rescales position/distance obs at indices 0/7/17-32; it does **not** touch the LVLH block at obs[33-37].

Also note **even at LEO with e=0.70**, |obs33| = 3.43, already past Box bound. This explains some Phase 5b/5e fragility at the high-e LEO edge: the LVLH obs were mildly OOD even there.

**Velocity components obs[35-36] are bounded better** because v_circ_t scales with altitude (sqrt(MU/a)) so the relative-velocity normalization is altitude-invariant in spirit. At e=0.70 they still hit ~3.0 (50% saturation).

**Verdict:** **LVLH spatial normalization needs altitude scaling.** A fix is to replace `R_EARTH` in orbital.h:585-586 with `env->obs_alt_scale_m` (or `R_EARTH + env->obs_alt_scale_m`) so the LVLH block sees the same scale as the position obs. This is mandatory before training at MEO/GEO; possibly also worth doing for high-e LEO to clean up the Phase 5b/5e bimodality.

### 2.5 E5 — Φ_orbit calibration with K-variations

Script: `scripts/orbital/p5_5_p1_e5_phi_calib_alt.py`. Output: `/tmp/p5_5_p1_e5.log` (partial; full table in script output).

For each altitude band + e_max + obs_alt_scale_m + K combination, compute worst-case Φ_orbit (max Δa within band × Δē at e_max). Selected results showing the K sensitivity at the GEO band (a≈42 Mm, half-band 500 km):

| Band | Δa_max | e_max | obs_scale  | K     | tol_eff | Φ_orbit | gate |
|------|-------:|------:|------------|------:|--------:|--------:|-----:|
| GEO  | 1000 km | 0.05 | LEO_default (1.6e6) | 0.001 | 10 km  | 100.07  | off  |
| GEO  | 1000 km | 0.05 | GEO_default (4.2e7) | 0.001 | 42 km  | 23.88   | off  |
| GEO  | 1000 km | 0.05 | GEO_default (4.2e7) | 0.005 | 210 km | 4.83    | off  |
| GEO  | 1000 km | 0.05 | GEO_default (4.2e7) | 0.010 | 420 km | 2.45    | off  |
| GEO  | 1000 km | 0.05 | GEO_default (4.2e7) | 0.050 | 2.1 Mm | **0.55**| **ON** |
| GEO  | 1000 km | 0.05 | GEO_default (4.2e7) | 0.100 | 4.2 Mm | **0.31**| **ON** |

**Findings:**
- Backward-compat at LEO with default kwargs is preserved (K=0.001 + obs_scale=1.6e6 → tol_eff=10 km, matches legacy SUCCESS_TOL_A).
- **K=0.001 (current default) is too small for within-GEO-band training:** Φ_orbit max = 100.07, σ₂ gate (threshold 2.0) never opens. The gated NHR shaping degenerates to single-component (Φ_orbit only), similar to the σ₃ "structurally dead" pattern Phase 5b documented at LEO.
- **K=0.05 with obs_alt_scale_m=4.2e7** keeps Φ_orbit ≤ ~1 at within-GEO worst case — gate operational.
- **For cross-altitude training** (LEO→GEO transfers, Δa_max ~35 Mm): K=0.05 gives Φ_orbit ≈ 17, still tractable.

**Verdict:** Default K=0.001 is correct for backward-compat at LEO but **breaks for any high-altitude training**. The follow-up training spec needs to use K ≈ 0.01-0.05 depending on the altitude domain. This is a config decision (no code change needed — the kwarg is already plumbed).

### 2.6 E6/P3 — Action effect calibration

Script: `scripts/orbital/p5_5_p1_e6_action_effect_alt.py`. Output: `/tmp/p5_5_p1_e6.log`.

Δa produced by each Δv action at e=0 (circular), per altitude band:

| Band      | a (Mm) | v_circ | pro+5 (km) | pro+10 (km) | pro+25 (km) | undershoot | overshoot |
|-----------|-------:|-------:|-----------:|------------:|------------:|-----------:|----------:|
| LEO       |   6.87 |   7617 |        9.0 |        18.1 |        45.5 | **YES**    | ok        |
| MEO_low   |   7.47 |   7304 |       10.2 |        20.5 |        51.6 | ok         | ok        |
| MEO       |  12.37 |   5676 |       21.8 |        43.8 |       110.2 | ok         | ok        |
| MEO_high  |  26.57 |   3873 |       68.8 |       138.1 |       348.6 | ok         | ok        |
| **GEO**   |  42.16 |   3075 |    **137.7** |    **276.4** |    **699.7** | ok       | **YES**   |

Peri-vs-apo asymmetry at e=0.50 is consistent ~3× at all altitudes (altitude-invariant, expected from r=a(1-e)/(1+e cos θ) geometry).

**Findings:**
- At LEO, the smallest action (5 m/s prograde) moves orbit by only 9 km — **below** SUCCESS_TOL_A = 10 km. This is a mild undershoot that explains some LEO fine-tuning difficulties.
- At GEO, the smallest action moves orbit by 138 km — **14× the rendezvous tolerance**. Single-burn fine-tuning at GEO is impossible with the current action set.
- The 25 m/s action at GEO moves orbit by **700 km** (70× tolerance), so adding *larger* actions makes the problem worse.
- The right direction is **smaller** actions for MEO/GEO: candidates are 0.5, 1.0, 2.0 m/s prograde/retrograde.

**Verdict per spec §2.3:** **Δv action set redesign needed for MEO/GEO.** This is a recipe-side change with surgery considerations (Phase 4.5 found cross-phase action-table surgery didn't transfer cleanly). Two paths:
1. Add new actions to the end of the table (e.g., actions 10-13 = 0.5, 1.0, 2.0, ±2.0 m/s). Existing Phase 5b ckpts wouldn't see them and would behave as before; new training could use them. Backward-compat preserved.
2. Replace the existing 5/10/25 m/s actions with a continuous scale. Bigger change; need to rerun Phase 5b validation.

**Recommendation:** path 1 (additive). Defer to follow-up env-mods spec.

---

## 3. P2 — Phase 5b ckpt zero-shot at altitude

Script: `scripts/orbital/p5_5_p2_phase5b_zero_shot.py`. Output: `/tmp/p5_5_p2.log` and `/tmp/p5_5_p2_results.csv`.

Eval canonical Phase 5b ckpt (`puffer_orbital_177750405236/model_puffer_orbital_000350.pt`, seed 31415) at 5 cells × 3 rollout seeds × 2 obs-scale sweeps (LEO default 1.6e6, GEO default 4.2e7). 200 episodes per cell-seed = 600 eps per cell × scale combination.

| Cell           | obs scale     | seed 42 | seed 1337 | seed 31415 | MEAN |
|----------------|---------------|--------:|----------:|-----------:|-----:|
| LEO_baseline   | LEO (1.6e6)   |   98.0% |     98.5% |      98.5% | **98.3%** |
| MEO_low_e      | LEO (1.6e6)   |    0.0% |      0.0% |       0.0% | 0.0% |
| MEO_eccentric  | LEO (1.6e6)   |    0.0% |      0.0% |       0.0% | 0.0% |
| GEO_low_e      | LEO (1.6e6)   |    0.0% |      0.0% |       0.0% | 0.0% |
| GEO_eccentric  | LEO (1.6e6)   |    0.0% |      0.0% |       0.0% | 0.0% |
| MEO_low_e      | GEO (4.2e7)   |    0.0% |      0.0% |       0.0% | 0.0% |
| MEO_eccentric  | GEO (4.2e7)   |    0.0% |      0.0% |       0.0% | 0.0% |
| GEO_low_e      | GEO (4.2e7)   |    0.0% |      0.0% |       0.0% | 0.0% |
| GEO_eccentric  | GEO (4.2e7)   |    0.0% |      0.0% |       0.0% | 0.0% |

**Findings:**
- **LEO baseline 98.3% across 3 seeds** — pipeline integrity confirmed; matches V3 result (98.0%) and published Phase 5b (96.4-97.7%) within ±2pp.
- **Every MEO/GEO cell is 0.0% across all seeds × both obs-scale settings** — zero altitude generalization.
- The rescaled-obs sweep (GEO default obs_alt_scale_m=4.2e7) didn't help. This makes sense: the ckpt was trained against the LEO-normalized observation distribution; rescaling produces a different OOD distribution, but still OOD. No "fix it just by rescaling" shortcut.

**Verdict per spec §2.2:** **The Phase 5b recipe is strictly LEO-specific.** Achieving any MEO/GEO capability requires training at those altitudes — no shortcut. Confirms the spec's basic premise.

---

## 4. P4 — Warp duration adequacy at GEO

Script: `scripts/orbital/p5_5_p4_warp_adequacy.py`. Output: `/tmp/p5_5_p4.log`.

5 episodes of Phase 5b ckpt at GEO low-e cell. Action histogram per episode:

| Episode | steps | reward | n_warps | max_consec_warps | first_burn | sim_hrs | unique_actions | term_action |
|--------:|------:|-------:|--------:|-----------------:|-----------:|--------:|---------------:|------------:|
|       1 |  2000 | -9.94  |  **1588** | **970**       |      **-1**|   139.2 |              2 | warp5min    |
|       2 |  2000 | -9.97  |    16   |      16          |     -1     |    34.4 |              2 | coast       |
|       3 |  2000 | -9.97  |   786   |     275          |     -1     |    85.7 |              2 | coast       |
|       4 |  2000 | -9.95  |  1572   |     225          |     -1     |   138.1 |              2 | warp5min    |
|       5 |  2000 | -9.88  |     6   |       6          |     -1     |    33.7 |              2 | coast       |

`first_burn = -1` means the agent NEVER takes any action other than coast or warp5min across the whole episode. Unique actions = 2 (only 0 and 9 used).

**Findings:**
- **The LEO-trained agent at GEO is helpless.** It chains warp5min and coast actions but never burns. Max 1588 warps per episode.
- Spec §2.4 verdict: "**STRUCTURALLY INADEQUATE**" — current 5-min warp is far too short. Even chained 1000+ times, the agent covers ~83 hours of sim time = ~3.5 GEO orbits, but its observation space is dominated by saturated LVLH spatial obs (per E4) and OOD altitude obs (per E5/B.1 saturation), so it can't form a useful policy.
- The reward of -9.94 to -9.97 is the safety-cap timeout (no terminal success, no collision/escape — just ran out of steps).

**Verdict per spec §2.4:** **Add longer warp actions before any MEO/GEO training.** Candidates: warp-30min (action 10, τ=30 sub-steps × 60s = 30 min), warp-1hr (τ=60), warp-6hr (τ=360). For GEO orbital period (24 hr), warp-1hr × 24 covers one period.

**Note on MAX_STEPS:** at GEO, 2000-step cap × 60s = 33 hr ≈ 1.4 GEO orbits. Even with longer warps, episode lifetime needs to extend (spec §3.5 already flagged this). MAX_STEPS = 5000 or 10000 plus expanded warp actions.

---

## 5. Synthesis — env mods required before Stage 5.5.1

Based on the six probes + zero-shot + warp adequacy:

| Issue | Severity | Required fix | Code site |
|-------|----------|--------------|-----------|
| **Action set too coarse at MEO/GEO** | Mandatory | Add sub-5 m/s prograde/retrograde actions; possibly add larger prograde for GEO transfer | `orbital.h:55-69` (ACTION_DV table) |
| **Warp 5 min inadequate at GEO** | Mandatory | Add warp-1hr action (τ=60); possibly warp-6hr (τ=360) | `orbital.h:56-57` (WARP_TAU), c_step logic |
| **LVLH spatial obs saturate at MEO+** | Mandatory | Replace `R_EARTH` with `env->obs_alt_scale_m` (or new kwarg) in LVLH normalization | `orbital.h:585-586` |
| **Φ_orbit K=0.001 default breaks gate at MEO+** | Configuration only | Use K≈0.01-0.05 in training configs at higher altitudes | `pufferlib/config/ocean/orbital.ini` (per-stage) |
| **MAX_STEPS=2000 too short at GEO** | Recommended | Expand to 5000-10000 OR use longer warps to cover same sim time fewer steps | `orbital.h:24` (MAX_STEPS) |
| **Phase 5b ckpt is LEO-specific** | Expected — not a fix | Training must happen at higher altitudes (no zero-shot shortcut) | N/A |

### 5.1 Prioritization

**Tier 1 (mandatory before Stage 5.5.1):**
1. **LVLH spatial obs scaling** — single one-line code fix; saturated observations make training impossible.
2. **Warp action(s) longer than 5 min** — small code change adding entries to the action table + c_step warp dispatch.
3. **Sub-5 m/s prograde/retrograde actions** — extends action table; needs validation that LEO-trained Phase 5b ckpts still work with the extended table (additive, so backward compat preserved if existing actions retain their indices).

**Tier 2 (configuration, no code):**
4. **Φ_orbit K scheduled per altitude stage** — set in curriculum scripts, not env code.

**Tier 3 (probably needed, can defer):**
5. **MAX_STEPS expansion** — only if longer warps don't make the 2000-step cap sufficient.

### 5.2 Estimated env-mods scope

The Tier 1 fixes are ~1 day engineering + smoke testing:
- LVLH scaling: 2 lines in `fill_observations`, plumb new kwarg if needed (1-2 hours).
- Longer warp actions: extend ACTION_DV table + WARP_TAU lookup, ~4 hours.
- Sub-5 m/s actions: extend ACTION_DV table at higher indices, ~2 hours.
- Smoke test at all altitude bands: ~2 hours.
- V3-style backward-compat check (Phase 5b reproduces): ~1 hour.

Tier 2 is just curriculum.sh edits.

This is a **follow-up env-mods spec** (call it `phase5-5-env-mods-spec.md`), to be written and landed before Stage 5.5.1 (per spec §7.6 scope-creep guidance — don't pile this onto Phase 5.5 itself).

### 5.3 Effect on Phase 5.5 sequencing

Original spec sequencing:
1. Pre-flight (done).
2. Pre-experiments (done; this doc).
3. Stage 5.5.0 → 5.5.1 → ... → 5.5.4.

Revised sequencing:
1. Pre-flight (done).
2. Pre-experiments (done; this doc).
3. **Stage 5.5.0 — re-validate Phase 5b at LEO multi-seed.** Anchors the env-fix backward-compat baseline. ~2 hrs compute.
4. **Env-mods spec + implementation (NEW).** Tier 1 fixes + backward-compat check. ~1-2 days.
5. Stage 5.5.1 → ... → 5.5.4 on the env-modded code.

The pause point (per Plan 1's design) becomes: write the env-mods spec next.

---

## 6. Open questions

1. **Does adding sub-5 m/s actions cause training collapse at LEO via the Phase 4.5 D7→D10 surgery pattern?** Phase 5b's recipe is sensitive to action-table changes. Mitigation: backward-compat test ensures Phase 5b ckpts still hit 96-98% with new actions added at higher indices (untouched indices 0-9 should produce byte-identical behavior).

2. **What's the right Φ_orbit normalization for cross-altitude transfers (LEO↔GEO)?** Within-band K=0.05 works; cross-band Δa can be ~35 Mm. K=0.05 gives Φ_orbit ≈ 17 at the cross-altitude worst case — still tractable but on the high side. May want K=0.1 for cross-altitude curriculum stages.

3. **Should the new warp actions be discrete (warp-1hr, warp-6hr) or continuous (one warp action with a magnitude parameter)?** Discrete matches the existing Discrete(10) → Discrete(12-14) extension philosophy and Phase 5b's lesson that continuous action spaces dilute random-policy productive rate.

4. **Will MAX_STEPS=5000 be sufficient at GEO, or do we need 10000?** Depends on warp count. With warp-1hr (12× faster than warp-5min), the 1588-warp episode would become 132 warp-1hr actions covering 132 hours — well under 2000 steps. So MAX_STEPS expansion might not be needed if warp-1hr lands. Re-probe after env-mods.

5. **What altitude band should Stage 5.5.0 use?** Plan says LEO (mirror Phase 5b). After env mods, the Stage 5.5.0 baseline should re-run at LEO with the new env to anchor backward compat AT the new code (not just at the pre-env-mods code). This becomes the V3 of the env-mods.

---

## 7. Compute consumed

| Probe | Wall time | Notes |
|-------|-----------|-------|
| E1 Lambert at altitudes | ~15 min | 50 cells × 100 Lambert + 200 sample tasks each |
| E2 Kepler precision | ~30 sec | 25 cells × 143570 steps max |
| E3 round-trip | ~5 sec | 25 cells × 100 orbits |
| E4 LVLH | ~3 sec | 25 cells × 100 random configs |
| E5 Φ_orbit calib | ~1 sec | Pure math scan |
| E6 action effect | ~1 sec | Pure math |
| P2 zero-shot | ~15 min | 9 cells × 3 seeds × 200 eps; CPU |
| P4 warp adequacy | ~3 min | 5 GEO episodes |
| **TOTAL** | **~35 min** | Well under spec's ~4 hr budget |

---

## 8. Files produced

### New probe scripts
- `/Users/pete/space_training/scripts/orbital/p5_5_p1_e1_lambert_alt.py`
- `/Users/pete/space_training/scripts/orbital/p5_5_p1_e2_kepler_alt.py`
- `/Users/pete/space_training/scripts/orbital/p5_5_p1_e3_roundtrip_alt.py`
- `/Users/pete/space_training/scripts/orbital/p5_5_p1_e4_lvlh_alt.py`
- `/Users/pete/space_training/scripts/orbital/p5_5_p1_e5_phi_calib_alt.py`
- `/Users/pete/space_training/scripts/orbital/p5_5_p1_e6_action_effect_alt.py`
- `/Users/pete/space_training/scripts/orbital/p5_5_p2_phase5b_zero_shot.py`
- `/Users/pete/space_training/scripts/orbital/p5_5_p4_warp_adequacy.py`

### Findings & registries
- `/Users/pete/space_training/MODELS.md` — canonical ckpt registry
- `/Users/pete/space_training/PHASE5_5_PRE_FINDINGS.md` — this document

### Raw outputs (transient)
- `/tmp/p5_5_p1_e{1,2,3,4,5,6}.log`
- `/tmp/p5_5_p2.log`, `/tmp/p5_5_p2_results.csv`, `/tmp/p5_5_p2/<cell>/ep_*.npz`
- `/tmp/p5_5_p4.log`, `/tmp/p5_5_p4/ep_*.npz`

---

## 9. Next plan

Per the user-preferred staged-experiments → analysis → proposal pattern AND the Phase 5.5 spec §9.1 (pause if pre-experiments reveal env issues that change scope), the next plan should:

1. **Write `phase5-5-env-mods-spec.md`** scoping the Tier 1 fixes. Mirrors `phase5-env-fix-spec.md`'s structure (named fixes, validation protocol, sequencing, pre-committed surprises).
2. **Stage 5.5.0 multi-seed re-validation** — done as a small pre-step before env mods, to anchor the absolute backward-compat baseline before any code changes land.
3. **Implement env-mods + validate** following the same V1-V4 + backward-compat pattern as the env-fix.
4. Only then proceed to Stage 5.5.1+.

This sequencing reflects the Phase 5b-e methodological discipline at §6.10 ("validate env at every new regime"): the probes here found 4 real issues; landing the fixes before training keeps Phase 5.5 from being another "50 hours of compute investigating env bugs" episode.

---

*Author: 2026-05-15. Phase 5.5 Plan 1 of N complete. ~35 min total compute. Six probes + zero-shot + warp adequacy. Four mandatory env mods identified (LVLH scaling, longer warp, sub-5 m/s actions, Φ_orbit K config). Phase 5b ckpt confirmed strictly LEO-specific (0/600 at all MEO/GEO cells). Recommended next: write `phase5-5-env-mods-spec.md` before proceeding to Stage 5.5.1 curriculum.*
