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

---

## 4. Covariance consistency under J2 — the binding decision

Follow-up to §1's open item: `MSC6J2` restores position at 24 h but leaves the
covariance ~11× overconfident, and 24 h is the rung's operating point. Four
candidates, same protocol, N=128, truth J2 in every arm but the control.
Record: `scripts/orbital/extj2/j2_nav_candidates.txt`.

### 4.1 Head-to-head at 24 h (the operating point)

| arm | pos p50 | vs control | NEES p50 | in band | tick cost |
|---|---|---|---|---|---|
| MATCHED 2body/2body (control) | 458.0 m | 1.00× | 1.00 | 89.1% | 1.00× |
| FIXED — J2 state, 2body cov | 1202.7 m | 2.63× | **11.37** | 21.9% | 1.36× |
| C1-an — analytic ΔΦ | 1234.5 m | 2.70× | **11.46** | 18.8% | 1.70× |
| **C1-fd — FD J2 STM** | **457.3 m** | **1.00×** | **1.01** | **87.5%** | **5.09×** |
| C2-T4 — re-init every 4 h | 1842.0 m | 4.02× | 0.93 | 88.3% | 1.36× |
| C2-T8 — re-init every 8 h | 1288.3 m | 2.81× | 1.30 | 73.4% | 1.36× |
| C2-T12 — re-init every 12 h | 1159.6 m | 2.53× | 1.79 | 63.3% | 1.36× |
| C3 — structured inflation | 1202.7 m | 2.63× | **11.34** | 21.9% | 1.70× |

At 6 h: C1-fd is 1.02 NEES / 83.6% in band vs the control's 1.02 / 82.8%;
every other arm sits at 1.34–1.40 NEES / ~66%.

**C1-fd is the only candidate that restores BOTH.** It reproduces the control
to three digits on position (457.3 vs 458.0 m) and on NEES (1.01 vs 1.00). The
covariance error was not a nuisance term — it was feeding back through the
Kalman gain and costing 2.6× in position error as well.

### 4.2 Why the two cheap options failed, and it is the same reason

**C1-an (analytic ΔΦ) — 59% effective, and it overshoots.** The partials are
not the problem: `_dq_dx` matches finite differences to 1.2e−9, and the
e-route is provably negligible (∂Ω̇/∂e = Ω̇·4e/(1−e²) is 5e−5 of the a-route at
e ≤ 0.05). The problem is a **440:1 cancellation**. The exact chain is

```
ΔΦ = [∂G_J2 − ∂G_2body]·(∂angles/∂x + dt·∂rates/∂q·∂q/∂x)
   +  ∂G_J2·dt·∂(rates − n·e_M)/∂q·∂q/∂x
```

and the two pieces are individually ~150 while their difference is 0.34.
Truncating to the second leaves a residual of the same order as the quantity
being modelled — measured |model| max 4.82e−1 against |truth| max 3.41e−1, i.e.
it overshoots. Kept in the tree with the derivation in its docstring, because
the negative result is what stops someone re-deriving it.

**C3 (structured inflation) — indistinguishable from doing nothing** (NEES
11.34 vs FIXED's 11.37). Inflating the node and along-track directions by the
right *magnitude* does not help because the defect is not an additive rank-2
term: it is a **multiplicative** error in the transition, so it compounds
across 1440 ticks in directions that rotate with the orbit. An additive
correction applied in the instantaneous node/along-track frame cannot track
that. This is the same lesson as the Q sweep in §1: the bias is structural, and
no amount of inflation — scalar or structured — repairs a wrong Φ.

**C2 (periodic re-init) — buys consistency by destroying information.** It does
fix NEES (T4: 0.93, 88.3% in band) but the position error goes the wrong way:
1842 m, **4.02× the control and 1.5× worse than doing nothing**. The measured
transient explains it: the position-error jump at re-init is **+0.0 m** (the
mean is untouched, as it must be) while σ_pos inflates **150–307×**. Nothing is
corrected; the filter simply forgets what it knew and must re-converge, and at
T=4 h it never gets to. The T sweep is monotone in the obvious direction —
shorter T buys consistency and costs accuracy — with no setting that gets both.

### 4.3 Cost in context

`predict()` is **24.9%** of a full `OrbitalNav(bearings_only)` step (0.441 ms of
1.771 ms at N=64). So the 5.09× on predict is:

| filter | predict | full nav step |
|---|---|---|
| two-body (today) | 0.441 ms | 100% |
| J2 state only | 0.626 ms | **110%** |
| J2 state + FD J2 covariance | 2.517 ms | **217%** |

**The full fix roughly doubles the nav step**, which on the nav lineage's ~99 min
per 50M arm is ~3.6 h. Real but affordable, and it buys a 2.6× position
improvement alongside the consistency.

### 4.4 Binding recommendation for the campaign spec

**Ship `MSC6J2Cov(stm_j2='fd')` — J2 in the state propagation AND a
finite-difference J2 STM in the covariance propagation — for every J2 × nav
arm.** It is the only measured option that restores both the estimate and its
covariance to the matched control (457 m / NEES 1.01 vs 458 m / 1.00), and the
covariance is not cosmetic: the two-body-STM arms pay 2.6× in position error
because an overconfident covariance shrinks the Kalman gain. Reject the cheap
options on evidence, not taste — the analytic ΔΦ is 59% effective and
overshoots because of a 440:1 cancellation in the exact chain, structured
inflation is statistically identical to doing nothing (NEES 11.34 vs 11.37)
because the defect is multiplicative rather than additive, and periodic re-init
fixes NEES only by discarding information, landing at 4.02× the control's
position error with a 150–307× σ excursion at each re-init. The price is
2.17× a nav step (predict is 24.9% of it), ~3.6 h per 50M arm. **One
optimization is identified but unmeasured**: propagating the covariance in
element space, where the secular-J2 Jacobian is exactly `I + dt·∂rates/∂(a,e,i)`,
would be exact and cheaper than 12 finite-difference propagations — it needs
the coe2rv/rv2coe Jacobian chain that C1-an tried to shortcut. Do not attempt it
mid-campaign; measure it as its own item if the 2.17× ever binds.

### 4.5 Interaction with nav60 and with the acquisition surrogate

**nav60 cadence: no interaction.** Every candidate acts inside `predict(dt)`
and is exact in dt; the choice is cadence-independent. C2 is the exception in
principle — its re-init period is wall-clock, not tick-count — but it is
rejected anyway.

**The acquisition surrogate: exposed in principle, immune in practice AT THIS
RUNG — and the reason is the arc length, not the e-immunity argument.**
`AcqSurrogate._accumulate6` chains `Phi_k = stm_analytic_nd(...) @ Phi_{k-1}`
— the two-body transition — while the measurement kernel is built from the
realized geometry. So the geometry is J2-correct for free (that IS why the
surrogate is e-immune: e enters only through realized geometry) but the
*transition* is not, and that exposure is genuinely different in kind.

Measured (`scripts/orbital/extj2/j2_acq_surrogate.txt`), identical J2 truth,
only the chaining STM differs:

| arc | σ_LOS shipped | σ_LOS J2-aware | ratio p50 | ratio p95 | \|log2\| > 0.1 |
|---|---|---|---|---|---|
| **45 min (the floor)** | 18377.0 m | 18378.5 m | **1.0002** | 1.0441 | 3.1% |
| 90 min | 3827.7 m | 3836.3 m | 0.9997 | 1.0252 | 5.5% |
| 3 h | 1492.4 m | 1491.8 m | 0.9923 | 1.0140 | 5.5% |
| 6 h | 960.9 m | 1046.5 m | **0.9674** | 1.0229 | **32.8%** |

And the decision itself: **128/128 acquired by both arms, latency delta
identically 0.00 min, 0.0% differing by even one 60 s tick.**

So the shipped surrogate needs **no change** for the J2 × nav campaign as
configured, because `nav_acq_min_sec = 2700` and rung 1 measured acquisition at
exactly that 46-min floor — at which the bias is 0.02%. But the immunity is a
property of the short arc: by 6 h the shipped surrogate runs **3.3% optimistic**
at the median with a third of scenarios past 7% error, and optimistic is the
dangerous direction (it declares acquisition on a bound the information does not
support). **Re-measure this if any future rung allows a long blind window or a
re-acquisition after an extended coast.** That boundary belongs in the spec.
