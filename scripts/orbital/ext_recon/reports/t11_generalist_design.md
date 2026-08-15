# T11 — the generalist rung: design, measurements, and self red-team

**Goal (user's words):** one policy with generalized J2 × wide-e × bearings-only
capability, fuel efficiency quantified, budget sampled per episode so scenarios
widen.

**Design input:** `GEN_MATRIX.md` (main, d4a4690) — 42 cells × 6 flagship
checkpoints. Its read: off-diagonal means 6–42%, and the #1 barrier is the
narrow/wide normalizer split (mutually ~0%, unchanged under truth).

**Status:** design only. No training launched; MAIN untouched. All numbers below
are measured in this worktree; raw output in
`scripts/orbital/ext_recon/reports/t11_measurements.txt`, probes in
`scripts/orbital/ext_recon/t11_{family,transplant,nav_transplant,fuel}_probe.py`.

---

## 0. Executive summary

1. **The design conflict is real, and then it dissolves.** The narrow family
   physically cannot hold the user's wide-e requirement (measured ceiling:
   realized e p90 **0.112** vs E3's **0.257**). So the matrix's "restate the
   E-ladder in narrow normalizers" is unimplementable.
2. **But the normalizer barrier is not a learned incompatibility — it is a
   parameterisation artifact, and it is exactly removable.** The encoder's first
   op is a single `nn.Linear` on the raw obs vector, so rescaling 7 weight
   columns moves any checkpoint between families with a **bit-identical action
   stream**. Verified on a truth lineage (91/100 → 91/100, md5 `e81c41eb204c`)
   and a bearings-only lineage (60/60 → 60/60, md5 `724f345ba8ee`), against
   1/100 and 0/60 without the transplant.
3. **So: commit to the WIDE family (for representability) AND keep A3b-j2 (the
   best generalist, 41.8% off-diagonal) as the warm root — by transplanting it.**
   The matrix's two recommendations were only in conflict because the transplant
   was not known.
4. **The "family" is two independent knobs, not one.** `obs_alt_scale_m`
   (bound by the e-requirement → wide) and `lvlh_scale_m` (bound by the
   tight-box requirement → narrow) can be chosen separately.
5. **Three red-team findings change the brief's parameters**, all measured:
   the proposed fuel floor U(0.08,…) makes **46–59% of episodes Δv-infeasible**
   and must be raised to ~0.113; the existing success reward's fuel bonus is
   normalized by a **compile-time constant** and would silently become a
   reward multiplier under sampling; and a per-cell `episode_cap_steps` is
   required or the **clock channel obs[15] is destroyed** for short cells.

---

## 1. THE FAMILY QUESTION — resolved by measurement

### 1.1 What obs[0] is, and what "representable" means

`obs[0] = (a − R_EARTH) / obs_alt_scale_m`, written **raw** (no clamp) against a
declared `Box(-2, 2)`. A band is representable iff `|obs[0]|` stays inside the
space the trainer was told about.

| | narrow (1.6e6) | wide (8.0e6) |
|---|---|---|
| \|obs0\| ≤ 1 ⟹ | a ≤ 7.971e6 | a ≤ 14.371e6 |
| \|obs0\| ≤ 2 ⟹ | a ≤ 9.571e6 | a ≤ 22.7e6 |

### 1.2 The physical coupling the brief flagged

`c_reset` rejection-samples until perigee `a(1−e) ≥ 6.571e6`, so **e is capped
by altitude**: `e_max(a) = 1 − 6.571e6/a`. Measured ceilings:

| band top | e_max at apogee | \|obs0\| narrow | \|obs0\| wide |
|---|---|---|---|
| X3 7.171e6 | 0.084 | 0.50 | 0.10 |
| E1 7.871e6 | 0.165 | 0.94 | 0.19 |
| **narrow limit 7.971e6** | **0.176** | **1.00** | 0.20 |
| E2 9.871e6 | 0.334 | **2.17** | 0.44 |
| E3 14.371e6 | 0.543 | **4.92** | 1.00 |

### 1.3 THE ANSWER (4096 resets/row)

Best realized eccentricity achievable with narrow `|obs0| p99 ≤ 1`:

| band | e_max set | realized e_t p50 | **p90** | p99 | max | \|obs0\| p99 |
|---|---|---|---|---|---|---|
| X3 6.671–7.171 | 0.30 | 0.025 | 0.058 | 0.075 | 0.083 | 0.50 |
| E1 6.671–7.871 | 0.30 | 0.045 | 0.104 | 0.137 | 0.161 | 0.93 |
| **N1max 6.671–7.971** | **0.30** | **0.048** | **0.112** | 0.148 | 0.173 | **1.00** |
| E2 6.671–9.871 | 0.30 | 0.093 | 0.222 | 0.287 | 0.300 | 2.17 ✗ |
| E3 6.671–14.371 | 0.30 | 0.114 | **0.256** | 0.296 | 0.300 | 4.92 ✗ |

**Narrow tops out at realized e p90 = 0.112** — above E1 (0.081) but only 68% of
E2 (0.166) and **44% of E3 (0.257)**. Reaching E3-class eccentricity requires
`|obs0|` up to 4.92, i.e. **2.5× outside the declared observation space**.

`gave_up` was **0.0000 in every row** — the perigee sampler never exhausts; the
knob is not infeasibility, it is representability.

> **Verdict: the user's wide-e requirement forces the WIDE family.** The
> matrix's "restate the E-ladder in narrow normalizers" cannot be implemented at
> E2/E3. The brief's prior is confirmed with data.

A second, independent reason: pushing narrow to `|obs0| ≈ 1–2` to chase e would
put the A3b-j2 encoder 2–4× outside its *experienced* range (X3 tops at 0.50) —
which is precisely GEN_MATRIX Signature A's failure mechanism. **The narrow warm
start's value is contingent on staying narrow, so "narrow + high e" destroys the
very transfer that motivated choosing narrow.**

---

## 2. THE TRANSPLANT — the barrier is removable, exactly

GEN_MATRIX calls the normalizer split "the largest single effect in the matrix"
(99 pp, vs 51 pp for adding J2). **It is an artifact of the parameterisation.**

`Default.encode_observations` is `Sequential(Linear(38,128), GELU)` applied to
the *raw* obs vector. If a normalizer change multiplies obs channel *j* by
`c_j`, then dividing `encoder.0.weight[:, j]` by `c_j` leaves the pre-activation
— and therefore the GELU output, the LSTM state, the value head and the logits —
**unchanged**.

Channels that move between families (all others are dimensionless or normalized
by a quantity both families share):

| channel | divisor | narrow → wide | weight column × |
|---|---|---|---|
| obs[0], obs[7] | `obs_alt_scale_m` | 1.6e6 → 8.0e6 | 5.0000 |
| obs[17], obs[20] | `R_EARTH + obs_alt_scale_m` | 7.971e6 → 14.371e6 | 1.8029 |
| obs[24], obs[33], obs[34] | `lvlh_scale_m` | 6.371e6 → 1.5e7 | 2.3544 |

### Verified

| lineage | mode | @ home (narrow) | @ wide, no transplant | @ wide, **transplanted** |
|---|---|---|---|---|
| A3b-j2 (truth) | truth | 91/100 `e81c41eb204c` | **1/100** | **91/100 `e81c41eb204c`** |
| J2BO-nav (nav) | bearings-only, real IOD | 60/60 `724f345ba8ee` | **0/60** | **60/60 `724f345ba8ee`** |

**Bit-identical action streams.** The transplant is not "recovers most of it" —
it is the same policy.

### Caveats, stated

- Exact only for channels whose *scaling* changes. `obs[21,22,23]` use per-rung
  `di_scale`/`de_scale`; hold them equal or extend the transplant to them.
- `obs[15]` (clock) is dimensionless but its **unit** is `episode_cap_steps`.
  A linear transplant cannot fix a cap change — see red-team (1).
- It preserves behaviour only on scenarios expressible in *both* families;
  outside the overlap the policy is simply untrained, which is what a warm start
  is anyway.
- It is a *warm-start* tool. Once training resumes, the rescaled columns are
  free to move; nothing is frozen.

**Implementation:** `scripts/orbital/ext_recon/t11_transplant_probe.py` contains
`transplant(src, dst, alt_from, alt_to, lvlh_from, lvlh_to)`. Promote it to
`scripts/orbital/extj2/rescale_ckpt_normalizers.py` at campaign-prep time, with
the md5-identity check as its self-test.

---

## 3. THE FAMILY IS TWO KNOBS

`obs_alt_scale_m` and `lvlh_scale_m` are independent kwargs. The matrix's
"narrow/wide family" framing bundles them, but the two requirements bind
different knobs:

| requirement | binds | needs |
|---|---|---|
| wide eccentricity (E2/E3 bands) | `obs_alt_scale_m` | **8.0e6 (wide)** |
| tight boxes (5 km / 1 m/s) | `lvlh_scale_m` | **6.371e6 (narrow)** — a 5 km box is 3.3e−4 obs units at 1.5e7 vs 7.8e−4 at 6.371e6 |

**Recommended generalist normalizers: `obs_alt_scale_m = 8.0e6`,
`lvlh_scale_m = 6.371e6`.** This combination has never been trained, so it is a
new pairing — but the transplant reaches it from *either* parent exactly, and it
is the only pairing that represents both requirements at once.

---

## 4. RUNG PLAN

Common to both rungs: `dim3_mode=1`, `j2_mode` per cell, `nav_j2_mode` matched
to `j2_mode` (the wrapper refuses the mismatched pairing), `lvlh_frame_mode=1`,
obs[29,30] live under J2 and 0 otherwise, obs[31,32] reserved and zero,
31-row head, `shape_w_match=0.8166667` / `shape_dv_ref_ms=700` (see red-team 5).

### Rung A — J2 + inclined targets in the wide family

*What is new is only the normalizer pairing; the J2 kwargs, the inclined
sampler and `MSC6J2Cov` all exist and are validated.*

| item | value |
|---|---|
| warm root | `n3dnav_e_E3` (wide, nav, e-capable), **transplanted** `lvlh 1.5e7 → 6.371e6`, head 30 → 31 |
| cells | E0-band and E1-band, `j2_mode=1`, `i_t ~ U(30°,60°)`, `di_max 1°`, box 30 km/50 m/s |
| nav | bearings-only, `nav_j2_mode=1` (MSC6J2Cov, the ext-j2 head-to-head's only consistent arm) |
| cap | 3000 |
| steps | 50M |
| gate | ≥ the J2BO-nav headline (94%) at the J2X cell, and ≥ E1's own 97% at the E1 cell |

Rung A exists to prove the *pairing* before the mixture depends on it. If it
fails, the mixture's normalizer choice is wrong and everything downstream moves.

### Rung B — the generalist mixture

| warm root | Rung A child |
|---|---|
| head | 31 (day-warp present) |
| steps | 200M (see §6) |

**Cells and rehearsal weights.** Weights follow GEN_MATRIX's nesting result:
subset regimes are *free* from their supersets (tight ⊃ loose at 99%, J2 ⊃
two-body at 91%), so they get no weight of their own.

| # | cell | cap | weight | why this weight |
|---|---|---|---|---|
| 1 | E0/X3-band, J2 on, inclined | 3000 | 0.15 | the base regime; subsumes loose/two-body |
| 2 | E1 band, J2 on, inclined | 3000 | 0.15 | e entry point |
| 3 | E2 band, J2 on, inclined | 4500 | 0.15 | |
| 4 | E3 band, J2 on, inclined | 6000 | 0.15 | the e requirement's top |
| 5 | **W1 node-dominant 2–5°, day-warp** | 22000 | **0.20** | *disjoint skill*: GEN_MATRIX Signature B — no 30-head policy can express it at all, and W1-driftwait is simultaneously the weakest generalist (7.8% off-diag). Needs real weight, not transfer. |
| 6 | tight box 5 km/1 m/s, J2 on | 3000 | 0.10 | see red-team (4) — include at low weight, gated |
| 7 | LEO→MEO long-range transfer | 12000 | 0.10 | the widest `obs[0]` excursion; guards the top of the wide range |

Two-body cells: **weight 0**, per the nesting result. Loose-box cells: **weight
0**, subsumed by cell 6.

**Fuel budget sampled per episode:** `fuel_frac ~ U(0.113, 0.20)` →
**353–656 m/s** (see red-team 3 for why the floor is 0.113 and not 0.08).

---

## 5. SELF RED-TEAM

### (1) Mixture-of-caps — the sharpest implementation problem

`Orbital(num_envs=N)` takes **one** kwarg set, so a vec-env cannot hold per-env
caps. Three options:

| option | verdict |
|---|---|
| one cap = max (22000) for all cells | **rejected.** `obs[15] = (cap − step)/cap` is the clock. At cap 22000 a 3000-substep X3 episode occupies `t_frac ∈ [0.86, 1.0]` — the clock is compressed 7.3× and nearly constant. T3 measured the clock as load-bearing ("even the 99.2% expert fails clock-blind"). This silently deletes it for 5 of 7 cells. |
| separate vec-envs round-robined by the trainer | `pufferl` trains one env spec; not supported without trainer surgery. |
| **C-side per-episode cell sampler** | **recommended.** `c_reset` draws the cell (bands, flags, box, cap) from a table, and `episode_cap_steps` becomes per-episode state rather than a constant. |

**This is the single largest new implementation item in T11** and it is not
optional: without it, either the clock dies or the mixture cannot exist. It also
subsumes the fuel draw (same per-episode mechanism) and makes red-team (2)'s
regularizer cell-appropriate.

**Stall math per cell.** Φ's range is `W_λ + W_m = 1.8167`; success pays
`10·γⁿ`; stalling wins iff `n > 340` decisions. Measured decisions/episode:
X3 ~18, TB5 ~55, W1 **46 with the day-warp** but **544 without**. So the stall
threshold is satisfied in every cell *provided the day-warp exists and is used* —
and violated in W1 if it is not. The cap-reward-0 choice is safe here, but it is
safe *because* of the 31-head, not independently of it.

### (2) The day-warp in near-circular narrow-gap cells

W1 showed policies that own row 30 can mis-spend it. In a 3000-substep cell one
day-warp is **1440 substeps = 48% of the cap**; two overshoot it entirely. So a
mis-fired day-warp in a short cell is self-punishing within the same episode —
the cap is the regularizer, and Φ's λ term registers the overshoot immediately.

But that is only true if the cap is **cell-appropriate**: at cap 22000 a day-warp
is 6.5% and nearly free. This is a second, independent argument for red-team (1)'s
per-episode cap. **Prep gate:** action histogram per cell; day-warp usage must
concentrate in cells 5 and 7. If it appears above a few percent in cells 1–3,
the regularizer is not biting.

### (3) Fuel sampling under `valid_init_only` — **the brief's floor is wrong**

`valid_init_only` rejects on **perigee only**; fuel never enters the rejection
sampler. So a low budget draw does **not** raise `gave_up` — it produces
feasible-looking but **unsolvable** episodes, which is worse than the W3/W4
class because nothing counts it.

Measured required Δv (the same linearized estimate `obs[28]` uses):

| cell | dv_req p50 | p90 | p99 | infeasible @245 | @353 | @478 | @656 |
|---|---|---|---|---|---|---|---|
| X3/E0 | 263 | 352 | 413 | **59.2%** | 9.8% | 0.0% | 0.0% |
| E1 | 240 | 302 | 339 | **46.0%** | 0.2% | 0.0% | 0.0% |
| E2 | 247 | 312 | 375 | **50.8%** | 2.2% | 0.0% | 0.0% |
| E3 | 248 | 335 | 413 | **51.5%** | 6.6% | 0.0% | 0.0% |
| J2X | 262 | 352 | 418 | **58.6%** | 9.8% | 0.0% | 0.0% |
| **W1 node-dom** | **632** | 810 | 899 | 100% | 99.7% | 88.2% | 44.4% |

> **The brief's `U(0.08, 0.20)` would make 46–59% of ordinary episodes
> unsolvable.** Set the floor at **`fuel_frac = 0.113` (353 m/s)**, where the
> infeasible mass is 0.2–9.8%. Recommended: **`U(0.113, 0.20)` → 353–656 m/s.**

**And a trap in the same measurement:** the W1 cell reads 88% "infeasible" at
478 m/s *by construction* — `obs[28]` prices the **direct** plane change, which
is exactly what drift-and-wait exists to avoid. **A feasibility filter must never
be applied to cell 5**, or it would delete precisely the scenarios the skill is
for.

**A second, larger fuel finding.** The success reward already contains a fuel
bonus: `reward = 10·(0.5 + 0.5·fuel_remaining)` with
`initial_fuel = dry·FUEL_FRAC/(1−FUEL_FRAC)` — a **compile-time constant**
(`orbital.h:1745`, and again in `add_log`:1851). Under per-episode sampling and
*unchanged* normalization:

- a 0.08 draw can never exceed `fuel_remaining = 0.087/0.176 = 0.49` ⟹ its
  success is worth at most **7.5**, never 10;
- a 0.20 draw reaches 1.0 after spending 29% of its tank ⟹ **10 for free**.

Since the policy *observes* its fuel (obs[6]), it would learn to prefer
high-budget episodes rather than to be efficient — **the sampled budget becomes a
reward multiplier.** Fix: make `initial_fuel` the episode's drawn value in all
three sites. Then the bonus is scale-free and scarcity pressure emerges as
intended (a fixed maneuver costs a larger *fraction* of a small tank), which is
exactly the user's stated design.

### (4) Tight-box cells in wide normalizers are unproven — include, gated

TB5 has only ever been trained at `lvlh_scale_m = 6.371e6`. §3's recommended
pairing keeps `lvlh_scale_m` narrow precisely so the tight box keeps its
resolution, so the *unproven* part is only `obs_alt_scale_m = 8e6` — which
affects obs[0,7,17,20], none of which is the terminal geometry.

**Verdict: include at weight 0.10, gated.** Prep must run a zero-shot check of
the transplanted TB5 checkpoint at the new pairing (expect ≈98%, its published
score, since the transplant is exact). If that gate fails, the pairing is wrong
and cell 6 is deferred — not the whole rung.

### (5) Shaping-weight conflict — there isn't one

The two lineages are `w_match=0.35, dv_ref=300` (T3/TB3D) and
`w_match=0.8166667, dv_ref=700` (X3/E/J2). Their **gradients are identical**:

```
0.35/300      = 1.167e-3 per m/s
0.8166667/700 = 1.167e-3 per m/s
```

They differ only in where `min(1, ·)` saturates — 300 m/s vs 700 m/s — and hence
in the term's maximum magnitude (0.35 vs 0.8167).

**Pick `w_match = 0.8166667`, `dv_ref = 700`**, on the measured Δv distribution:
required Δv has p90 302–810 m/s across the mixture's cells. With `dv_ref = 300`
**every cell saturates on essentially every episode** and the match term becomes
a constant — no gradient at all. At 700 only W1's top ~1% (p99 899) saturates.
This is also the value the E, X3 and J2 lineages already use, so it is the
warm root's native reward.

### (6) Not asked, but it bites: `Orbital.close()` segfaults

GEN_MATRIX flags a state-dependent double-free in `binding.vec_close` after
multi-episode rollouts. A mixture eval sweeps many cells; **prep must keep the
one-process-per-cell discipline** `gen_matrix.py` uses, or fix the C first.
Flagged, not fixed — this task was design-only.

---

## 6. BUDGET

SPS classes measured in this lineage (all contended, so upper bounds on time):

| cell type | SPS class | driver |
|---|---|---|
| truth, cap 3000 | ~100K dec/s | pure C |
| bearings-only, cap 3000 | ~8.4K dec/s | Python EKF + IOD per tick |
| bearings-only, cap 22000 | ~6.5K dec/s | more substeps per decision |

| rung | steps | mix | wall clock |
|---|---|---|---|
| A | 50M | all bearings-only, cap 3000 | ~1.7 h |
| B | 200M | ~70% bearings-only, caps 3000–22000 | **~7–9 h** |
| evals | 200 eps × ~14 cells | mixed | ~2 h |
| **total** | | | **~11–13 h** |

Rung B at 200M rather than 50M: the mixture has 7 cells, so 50M is ~7M steps per
cell — against the 50M each specialist got for *one* cell. 200M gives ~28M per
cell, still under-trained relative to the specialists, which is the honest cost
of a generalist and should be stated in the result rather than hidden.

---

## 7. WHAT PREP MUST BUILD (ordered)

1. **C: per-episode cell sampler** — cell table + per-episode `episode_cap_steps`,
   bands, flags, box. Red-team (1); blocks everything else.
2. **C: per-episode fuel fraction** — `fuel_frac_min/max` kwargs, and
   `initial_fuel` from the drawn value in *all three* sites
   (`orbital.h:1745`, `:1851`, `:2197`). Red-team (3).
3. **Tool: `rescale_ckpt_normalizers.py`** — promote the probe, md5-identity
   self-test.
4. **Gates:** transplant md5-identity per lineage; zero-shot TB5 at the new
   pairing; infeasible-mass ≤ 10% per cell at the fuel floor; day-warp usage
   histogram per cell; the standard anchor battery (both new C kwargs
   default-off and bitwise-inert).
5. Rung A, then Rung B.

## 8. WHERE THIS DISAGREES WITH GEN_MATRIX

| GEN_MATRIX says | T11 finds |
|---|---|
| "restate the E-ladder in narrow normalizers … a prerequisite, not a mixture weight" | **Unimplementable** — narrow caps realized e at p90 0.112 vs E3's 0.257 |
| the normalizer split is "the largest single effect in the matrix" (99 pp) | **True as measured, but removable exactly** — a 7-column weight rescale restores bit-identical behaviour |
| "start from A3b-j2 … restate the E-ladder in narrow" | Keep A3b-j2's *skill*, but move it to **wide** by transplant; the two recommendations only conflicted because the transplant was unknown |
| (treats narrow/wide as one axis) | It is **two** — `obs_alt_scale_m` and `lvlh_scale_m` bind different requirements and should be chosen separately |

Everything else in GEN_MATRIX is used as-is: the nesting result (drop subset
cells), the day-warp's disjointness (weight 0.20), and A3b-j2's off-diagonal
lead as the reason to root there.
