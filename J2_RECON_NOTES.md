# ext-j2 recon — filter health under J2, plane-change feasibility, seed plumbing

Branch `ext-j2-rung`, 2026-08-14. Worktree only, no training launched, MAIN
untouched. Records: `scripts/orbital/extj2/j2_nav_filter_probe.txt`,
`scripts/orbital/extj2/j2_plane_change.txt`.

---

## 1. Filter health under J2 — the J2 × nav gate

**Verdict: J2 × nav REQUIRES a filter change. Two-body + process noise does not
absorb it, not at any Q. The cheap fix suffices out to ~6 h arcs and leaves a
covariance-consistency residual at 24 h.**

`scripts/orbital/extj2/j2_nav_filter_probe.py`. Three-arm design over identical
scenarios and identical measurement noise, N=256, nav60 cadence, LEO 300–800 km,
e ≤ 0.05, i_t ~ U(30°,60°), relative plane ≤ 1°. The filter is **seeded from
truth** with a post-IOD covariance (2% of separation) so the measurement is
dynamics mismatch, not acquisition failure. Medians with bootstrap 95% CIs.

| arm | arc | pos err p50 [95% CI] | vel p50 | NEES p50 [95% CI] | NEES>band |
|---|---|---|---|---|---|
| MATCHED (2body/2body) | 1 h | 6679 m [5980, 7831] | 7.48 m/s | 1.31 [1.13, 1.48] | 25.4% |
| MATCHED | 6 h | 1358 m [1162, 1471] | 1.40 | 1.07 [0.89, 1.21] | 10.9% |
| MATCHED | 24 h | 455 m [398, 534] | 0.48 | 0.90 [0.83, 1.04] | 8.6% |
| **MISMATCHED (J2/2body)** | 1 h | **13392 m** (2.01×) | 14.37 | **21.0** (16×) | **97.7%** |
| **MISMATCHED** | 6 h | **38097 m** (28.1×) | 38.76 | **3316** (3099×) | **100%** |
| **MISMATCHED** | 24 h | **190614 m** (419×) | 212.1 | **6.2e6** (6.8e6×) | **100%** |
| FIXED (J2/J2) | 1 h | 6782 m (1.02×) | 7.68 | 1.27 (0.98×) | 25.8% |
| FIXED | 6 h | 1473 m (1.08×) | 1.52 | 1.35 (1.26×) | 25.0% |
| FIXED | 24 h | 1074 m (2.36×) | 1.18 | **10.91** (12.1×) | **71.5%** |

NEES 6-dof 95% band [0.206, 2.408]; 1.0 = consistent. Divergence by non-finite
state was **0.0% everywhere** — the mismatched filter does not blow up, it
confidently converges to the wrong answer, which is worse.

### Can process noise absorb it? No, and the reason is structural

| Q multiplier | 6 h pos err | 24 h pos err | 24 h NEES |
|---|---|---|---|
| ×1 | 38097 m (28.1×) | 190614 m (419×) | 6.2e6 |
| ×10² | 38097 m (28.1×) | 190565 m (419×) | 5.5e6 |
| ×10⁴ | 38097 m (28.1×) | 186295 m (410×) | 1.6e6 |
| ×10⁶ | 37842 m (27.9×) | 148566 m (327×) | 4.0e5 |

**Six orders of magnitude of Q moves the 24 h error by 22% and leaves it at
327× the control.** That is the signature of a *deterministic secular bias*
rather than a random walk: inflating Q widens the covariance but never corrects
the mean, so the filter stays wrong and merely becomes less certain about it.
NEES stays ≥ 4e5, i.e. ~10⁵× outside the band. There is no Q that buys
consistency.

Sanity check on the magnitude: over 6 h the target's uncorrected J2 drift is
~18 km along-track (Ṁ correction) plus ~106 km cross-track (ΔΩ ≈ 1.25° at
i = 45°, displacement ≈ a·ΔΩ·sin i). The filter's 38 km median error is it
partially tracking that through bearings. The mismatch is not subtle.

### The cheap fix, and its one limitation

`MSC6J2` subclasses `BatchedBearingMSC6` and overrides **only** the state
propagation in `predict`: element round-trip, apply the same closed-form secular
Ω̇/ω̇/Ṁ the C env uses, come back. The covariance keeps the two-body analytic
STM (the J2 correction to Φ is O(J2) ~ 1e-3 relative). ~25 lines.

- **1 h and 6 h: fully restored.** 1.02× / 1.08× position error, NEES 1.27 /
  1.35 — inside the band, statistically indistinguishable from the matched
  control.
- **24 h: position restored, covariance is not.** 1074 m (2.36× the control,
  operationally fine against a 5 km box) but NEES 10.91 with 71.5% out of band —
  the filter is ~11× overconfident. **Q inflation does not repair this either**
  (×10² gives NEES 9.97, 68.4%), because the residual is again a bias: the
  two-body STM mispropagates the covariance over 1440 steps.

**Recommendation for the J2 × nav arm.** Ship the secular-J2 prediction — it is
mandatory, not optional; without it the filter is 100% inconsistent by 1 h. For
arcs beyond ~6 h either add the J2 term to the STM or re-initialise
periodically. Note the rung's own episodes run 1119–1666 sub-steps (18.6–27.8 h)
median, so **24 h is the operating point, not a corner case** — the covariance
residual is in scope for that campaign and should be budgeted.

---

## 2. J2-assisted plane change — feasibility

`scripts/orbital/extj2/j2_plane_change.c`, 4/4 checks, measured with the
**shipped `propagate_orbit_j2`** and compared against closed forms written
longhand.

### 2.1 The exact law, and the first half of the catch

For two orbits at common inclination i whose RAANs differ by ΔΩ:

> **Δi_rel = 2·asin( sin i · sin(ΔΩ/2) )** — verified against the propagator to
> **2.1e−16** relative at i ∈ {30,45,60}°, ΔΩ ∈ {1,10,90}°.

Two consequences:

- **Saturation.** Δi_rel → 2i as ΔΩ → 180°. No amount of drift buys more than
  twice the inclination. (At i = 45°, max 90°; at i = 5°, max 10°; at i = 0,
  zero — the pure-gauge case.)
- **The sin 2i law.** Rate = |ΔΩ̇|·sin i for small angles, and ΔΩ̇ ∝ cos i, so the
  product ∝ sin 2i: **zero at the equator (Ω is gauge) and zero at the pole
  (Ω̇ = 0), maximal at 45°.**

### 2.2 Achievable Δi_rel per day, as a function of i and δa

This is the number the question asked for — not Ω̇, the realized relative-plane
rate. Measured over 30 days at LEO-500:

| i | δa = −100 km | −300 km | −600 km |
|---|---|---|---|
| 15° | 0.1011 °/d | 0.3247 | 0.7235 |
| 30° | 0.1751 | 0.5623 | 1.2531 |
| **45°** | **0.2022** | **0.6493** | **1.4469** |
| 60° | 0.1751 | 0.5623 | 1.2531 |

(ΔΩ itself is larger — e.g. 0.2859 °/d at i=45°/−100 km — but only the
`× sin i` projection becomes relative plane.)

The propagator matches the **exact** rate difference
Ω̇(a+δa) − Ω̇(a) to **5.3e−11**. The convenient first-order rule
`(7/4)·k·sin(2i)·|δa|/a` is **3.3% low at 100 km, 9.6% at 300 km, 18.9% at
600 km** — it is first order in δa/a and δa = 600 km is already 8.7% of a.
Use the exact difference for anything quantitative.

### 2.3 Drift-and-wait vs direct burning (i = 45°, the best case)

Round-trip dip Δv = `v_c·|δa|/a` and is **independent of the plane angle
bought** — that is the entire mechanism. Direct = `2·v_c·sin(Δi/2)` scales
linearly. Budget 478.1 m/s.

| Δi | δa | dip Δv | % budget | direct Δv | ratio | drift | sub-steps | fits? |
|---|---|---|---|---|---|---|---|---|
| 1° | −100 km | 110.9 | 23.2% | 132.9 | 1.20× | 5.11 d | 7363 | cap raise |
| 1° | −300 km | 332.6 | 69.6% | 132.9 | 0.40× | 1.70 d | 2454 | **yes** |
| 1° | −600 km | 665.1 | 139% | 132.9 | 0.20× | 0.85 d | 1227 | NO (fuel) |
| 3° | −100 km | 110.9 | 23.2% | 398.8 | **3.60×** | 15.34 d | 22089 | NO (cap) |
| 3° | −300 km | 332.6 | 69.6% | 398.8 | 1.20× | 5.11 d | 7363 | cap raise |
| 5° | −100 km | 110.9 | 23.2% | 664.5 | **5.99×** | 25.57 d | 36815 | NO (cap) |
| 5° | −300 km | 332.6 | 69.6% | 664.5 | 2.00× | 8.52 d | 12272 | NO (cap) |

**δa = −600 km is off the table entirely** — the round trip alone is 139% of
budget.

### 2.4 The ≥3× band, and a result worth noticing

Condition `direct/dip ≥ 3` ⟺ `|δa|/a ≤ (2/3)·sin(Δi/2)`. Taking the largest
δa that satisfies it:

| Δi | max δa | dip Δv | ratio | drift | sub-steps |
|---|---|---|---|---|---|
| 1° | 40 km | 44.3 m/s | 3.00× | 12.79 d | 18420 |
| 3° | 120 km | 132.9 | 3.00× | 12.79 d | 18421 |
| 5° | 200 km | 221.5 | 3.00× | 12.80 d | 18425 |
| 10° | 399 km | 442.6 | 3.00× | 12.81 d | 18443 |

**The drift time at the 3× threshold is a constant ~12.8 days, independent of
Δi.** Because δa_max ∝ sin(Δi/2) ∝ Δi and rate ∝ δa, the ratio `Δi/rate`
cancels. The 3× fuel win always costs the same wall clock; only the dip Δv
scales (and at Δi = 10° it is 93% of budget, so ~10° is the fuel ceiling for a
3× win).

### 2.5 The second half of the catch: 1-D control, 2-D error

`cos Δi_rel = cos i_s cos i_t + sin i_s sin i_t cos ΔΩ` is minimised at ΔΩ = 0,
where it gives Δi_rel = |i_s − i_t| exactly. **|Δi| is a hard floor.** Measured
over 30 d, δa = −300 km, i_t = 45°:

| initial error | Δi_rel(t₀) | **min Δi_rel** | floor \|Δi\| | t_min |
|---|---|---|---|---|
| pure NODE (ΔΩ = 1.414°) | 1.0000° | **0.0002°** | 0.0000° | 1.54 d |
| pure INCLINATION (Δi = 1°) | 1.0000° | **1.0000°** | 1.0000° | 0.00 d |
| MIXED (45° in the error plane) | 1.0031° | **0.7071°** | 0.7071° | 1.19 d |

Drift nulls the node component completely and **cannot touch the inclination
component at all**. For the env's uniform-phase disc sampler the
drift-correctable fraction of a random relative-plane error averages
E|cos φ| = 2/π = **63.7%**; the rest still needs a burn.

### 2.6 Verdict

**Yes, a ≥3× band exists and is fuel-feasible up to Δi ≈ 10°, but it costs a
fixed ~12.8 days ≈ 18,400 sub-steps — 3.1× the current 6000 cap and 1.5×
`MAX_STEPS` (12000).** With a practical δa = −100 km dip (23% of budget) the
same win is 3.60× at 3° for 15.3 days (22,089 sub-steps).

Env changes needed, and **nothing else**:

| change | from | to | note |
|---|---|---|---|
| `episode_cap_steps` | 6000 | ~22000 | 15.4 d |
| `MAX_STEPS` | 12000 | ~22000 | compile-time; traj buffer 4.9 → 9.0 MB/env, virtual unless `log_enabled` |
| one `ACTION_TAU` row | — | τ = 1440 (24 h) | credit horizon only |

The warp row does **not** reduce the sub-step count — the cap counts sub-steps
and warps sub-step internally — it buys the credit horizon: at τ = 360 a 12.8-day
drift is **51 warp decisions (γⁿ = 0.774)**; at τ = 1440 it is **13 decisions
(γⁿ = 0.938)**. No new burn rows, no obs change, no shaping change.

---

## 3. Seed parameterization

`J2_SEED` (default 42) flows into `--train.seed`, the data dir, the wandb group,
the tag, and the trainer log path. Seed 42 produces **no suffix**, so the
already-reviewed default behaviour is byte-identical; other seeds get `_s<N>`.

Floors and refs are eval-only and seed-independent, so they are written once
under the shared JSON dir and **skipped** on later seeds — except
`*_floor_chain`, which *is* seed-dependent (it evaluates the previous stage's
child) and therefore carries the suffix.

Verified by dry-run with a stubbed trainer, seeds 42 then 7:

```
experiments_extj2/A2_j2trained      A2_j2trained_s7
                  A3a_j2_box10k10   A3a_j2_box10k10_s7
web_data/results/extj2_rung/
  A3a_j2_box10k10_floor_ref.json          <- written once, SKIPPED on seed 7
  A3a_j2_box10k10_floor_chain.json / _s7.json   <- per-seed (chained child)
  A3a_j2_box10k10_native.json      / _s7.json
  s2_A2_native.json                / _s7.json
```

No data-dir or wandb-group collisions; the flatline tripwire label is
seed-suffixed too.
