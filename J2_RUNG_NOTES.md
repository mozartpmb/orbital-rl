# ext-j2 rung — sampler design, pre-launch measurements, self red-team

Branch `ext-j2-rung`, 2026-08-13, branched from `origin/main` (which already
carries the merged `ext-j2` physics). **No training was launched.** Everything
below is eval-only or static analysis, run against the **worktree** build at
`/Users/pete/space_training-j2/pufferlib`; the MAIN checkout was never touched
or rebuilt.

Console record: `scripts/orbital/extj2/j2_rung_anchors.txt`.

---

## 1. The sampler

### 1.1 What was added

| kwarg | default | meaning |
|---|---|---|
| `i_target_min_rad` / `i_target_max_rad` | −1 / −1 (off) | `i_t ~ U(min, max)` per episode, overriding `i_target_rad` |
| `raan_target_sample` | 0 | `Ω_t = raan_target_rad + U(0, 2π)` per episode |
| `lvlh_frame_mode` | 0 (legacy) | obs[33-36] frame — see §3.2 |

**Both sampler branches consume zero `rand()` draws when off**, so the anchors
hold by *RNG-stream identity*, not merely by value (gate S4b: 2000 resets,
chaser a/M and target ω bitwise identical).

### 1.2 The two knobs are not the same kind of knob

This is the part worth being explicit about, because conflating them is the
trap the coordinator flagged.

**`i_t` is task variation.** The J2 plane channel is
`1.75·J2·(R_EQ/p)²·sin 2i` per radian of phase closed — identically **zero** at
i = 0 and i = 90°, maximal at 45°. Sampling `i_t` is the *only* thing that
makes the plane channel exist.

**`Ω_t` is gauge.** J2's potential is axisymmetric about ẑ, so rotating the
entire scene about ẑ maps solutions to solutions exactly. Sampling Ω_t adds no
task content whatsoever. Measured, not asserted: the differential nodal rate
`|Ω̇_s − Ω̇_t|` has median 0.4233 °/day for Ω_t < 180° and 0.4450 °/day for
Ω_t ≥ 180° — a **5.0% split**, i.e. flat (gate S5b). Its role is as an
**SO(2)-about-ẑ leak detector**, the reduced form of the ext-3d SO(3) frame
gate, exactly as `j2_A_design` §4.1 anticipated.

### 1.3 Band choice: 30–60°, and why not the design's U(20°, 100°)

The design suggests `U(20°, 100°)`. That band **straddles 90°, where the
channel is identically zero** — a large fraction of the sampled population
would have nothing to learn, diluting the arm. The C env now warns when a band
straddles 90°.

30–60° keeps `sin 2i ∈ [0.866, 1.0]` (channel within 13% of maximum everywhere)
and keeps **63.43°, the critical inclination where ω̇ = 0**, *outside* the band,
so a second unrelated degeneracy is not mixed in.

`J2_BAND` is overridable; §2.2 gives the measured control-ceiling tradeoff.

---

## 2. Measured results

### 2.1 The non-inertness gate (S5) — and where the signal actually comes from

2000 draws at the rung config, `|Ω̇_s − Ω̇_t|` in °/day:

| p05 | p25 | p50 | p75 | p95 | max | exactly-zero draws |
|---|---|---|---|---|---|---|
| 0.1329 | 0.2684 | **0.4357** | 0.6650 | 1.0452 | 1.5291 | **0 / 2000** |

**The decomposition is the finding, and it is not what the design predicts.**
`Ω̇ = −k(a,e)·cos i` depends on **both** a and i, and the chaser and target
differ in both — the transfer task itself puts them at different altitudes:

| mechanism | median | mean | closed form |
|---|---|---|---|
| Δa only (planes cloned) | 0.4379 | 0.4935 | 0.4551 |
| Δi only (orbits cloned) | 0.0347 | 0.0368 | 0.0392 |

**The altitude difference dominates by 13×.** `di_max` is a 7% correction on
top. Both components match their own closed forms to 8% and 6% respectively.

Operationally: the population injects a mean **0.692° of relative inclination
per 3000-sub-step (50 h) episode** — **183% of the 0.3775° free-plane zone** at
the 30 km / 50 m/s box, ≈ 92 m/s, 19% of the 478 m/s budget.

### 2.2 Zero-shot survey (200 eps, seed 123, X3 canonical checkpoint)

**J2 gap vs box tightness**, each box with its *own* parent checkpoint so the
control sits at ceiling, `i_t = 0` so the plane channel is **provably inert**:

| box | ckpt | j2=0 | j2=1 | gap |
|---|---|---|---|---|
| 30 km / 50 m/s | `seed42_X3_3d_di1deg` | 200/200 = 100% | 200/200 = 100% | **0.0 pp** |
| 10 km / 10 m/s | `seed42_TB3D_box10k10` | 199/200 = 99.5% | 113/200 = 56.5% | **−43.0 pp** |
| 5 km / 1 m/s | `seed42_TB3D_box5k1` | 194/200 = 97.0% | **0/200 = 0%** | **−97.0 pp** |

**Inclination band sweep at the loose box** (`lvlh_frame_mode=1`, Ω fixed):

| band | j2=0 | j2=1 | J2 gap |
|---|---|---|---|
| U(5,10) | 200/200 = 100% | — | — |
| U(20,30) | 200/200 = 100% | 200/200 = 100% | 0.0 pp |
| U(20,45) | 200/200 = 100% | 191/200 = 95.5% | −4.5 pp |
| U(25,50) | 199/200 = 99.5% | 186/200 = 93.0% | −6.5 pp |
| **U(30,60)** | **180/200 = 90.0%** | **159/200 = 79.5%** | **−10.5 pp** |
| U(60,80) | 11/200 = 5.5% | — | — |
| U(80,89) | 1/200 = 0.5% | — | — |

**Two conclusions that should shape the launch decision:**

1. **At the loose box, J2 is a non-event at an equatorial target (0.0 pp) and
   reaches at most −10.5 pp through the inclination-dependent plane channel.**
   That is a small target for a 50M-step arm. `j2_A_design` §4.4 predicted
   A1 = 55–85%; we measure 79.5%, at the top of the range, i.e. the policy
   already absorbs most of it.
2. **At tight boxes the effect is decisive (−43 to −97 pp) and is NOT the plane
   channel** — it is present at `i_t = 0` where the plane channel is provably
   inert. The failure mode is **always `safety_cap`** (median episode length
   exactly 3000 = the cap): the policy never crashes, it just never closes the
   box against a secular drift it cannot null.

The design's own §4.4 note anticipated this — *"at the current envelope the
policy has no reason to bother"* — and recommended tightening the budget or the
box. The measurement puts numbers on it. Stage 3 of the campaign is the
tight-box arm and is **not** in the default stage list (one new variable per
campaign).

### 2.3 The inclination cliff above ~60° is an obs defect, not a physics limit

100% at ≤30°, 90% at 30–60°, **5.5% at 60–80%**, 0.5% at 80–89° — all at
`j2_mode = 0`, so J2 has nothing to do with it. Diagnosis: `obs[18]` is the
Earth-conjunction bearing, and Earth is static at the origin, so
`body_angle = atan2(0,0) = 0` and the channel reduces to **`atan2(s_y, s_x)` —
the chaser's equatorially-projected longitude**. For a circular orbit at
inclination i, `tan λ_eq = tan u · cos i`, so λ_eq departs from the true
argument of latitude by up to ~9.7° at i = 45° and ~35° at i = 80°. The
degradation tracks that curve. Reported, **not fixed** — see §3.3.

---

## 3. Three bugs the sampler exposed

Target-plane randomization is exactly the operation that produced this
project's 11th metric-vs-implementation bug, so every realized-vs-knob metric
was re-measured rather than assumed to carry over. Two of the three are fixed.

### 3.1 FIXED — the inertial-ϖ correction was wrong off the pinned gauge

The T5 W3/W4 root-cause fix subtracts the chaser's **absolute** RAAN to
preserve the drawn inertial periapsis longitude. But the `de_max` disc is drawn
in the **target's** node-relative 2-vector, so what must be preserved is
ϖ_s − ϖ_t and the correction is `(raan_s − raan_t)`. Subtracting `raan_s` alone
is right only when `raan_t == 0` — true of every shipped lineage *precisely
because the target plane was pinned*.

| cell | realized max \|Δē\| vs a 0.020 knob | draws over |
|---|---|---|
| di_max off, i_t = 0 (pure legacy) | 0.019999 | 0 / 20000 |
| di_max 1°, i_t = 0 (shipped X3) | 0.020009 | 1 / 20000 |
| di_max 1°, i_t = 45° **fixed** | 0.020179 | 35 / 20000 (0.2%) |
| di_max 1°, i_t **sampled** — before fix | **0.111037 (5.55×)** | **11128 / 20000 (55.6%)** |
| di_max 1°, i_t **sampled** — after fix | 0.020183 | 35 / 20000 (0.2%) |

This re-opened the exact W3/W4 failure the block was written to close (the
e-match leg alone exceeding the whole Δv budget). Bit-exact for every shipped
lineage: `raan_t` is exactly 0.0 there and `x − 0.0 == x`.

The residual 0.2% at 1.009× is **pre-existing at any i_t > 0** (the di rotation
acts on ē after the disc is drawn, growing it by ~|e|·δ) and is not introduced
here.

### 3.2 FIXED (behind a default-off flag) — obs[33-36] is not LVLH at i_t > 0

The block rotates the **inertial x,y** offset by the in-plane angle ω+θ. That
is the LVLH frame at `i_t = Ω_t = 0` and nothing else: the equatorial
projection of r̂_t is `(cos u, sin u·cos i)` at Ω_t = 0, not `(cos u, sin u)`,
and Δz is dropped outright.

| measurement | value |
|---|---|
| legacy vs true-LVLH over the sampled band, mean max\|Δ\| | **1.5068 obs units** |
| worst | **4.3662** |
| worst relative | **238.7%** |
| pure ẑ-rotation moves obs[34] by | **4.114** |

These are `Box(-2, 2)` channels and they are the policy's *primary rendezvous
observation*. **Measured cost, zero-shot at U(30,60), j2=0:**

| frame | success | causes |
|---|---|---|
| legacy | 113/200 = **56.5%** | 36 collisions, 8 cap, 43 stranded |
| corrected | 180/200 = **90.0%** | 0 collisions, 19 cap, 1 stranded |

**The broken frame costs 33.5 pp and manufactures failure modes that do not
otherwise occur.** Without the fix, stage 1's "J2 gap" would have read
56.5 → 51.0 = −5.5 pp: the confound would have **halved the apparent J2 effect
while tripling the noise**.

`lvlh_frame_mode = 1` uses R̂ = r̂_t (3D), Ĉ = ĥ_t, T̂ = Ĉ × R̂. It is **free**:

- bitwise identical to the legacy block at i = Ω = 0 (0/16000 slots differ);
- all three checkpoint anchors reproduce with **identical action-stream md5s**
  (`f8a2388f0992` / `68b267bed369` / `003105f29898`);
- restores SO(2)-about-ẑ invariance (2.0e−13 vs 4.11).

### 3.3 NOT FIXED — obs[17-20] is the chaser's absolute inertial longitude

`obs[18]` reduces to `atan2(s_y, s_x)` for the static Earth body (§2.3).
Pre-existing, inert while the plane was pinned, live the instant
`raan_target_sample = 1`.

**Not fixed here**, deliberately: it would change the conjunction block's
semantics, which the **legacy 26/200 anchor runs with debris** and depends on.
The consequence is a design recommendation instead — **run every training arm
at `raan_target_sample = 0`.** Ω_t is gauge (§1.2), so sampling it buys no task
content while decorrelating a channel the warm-start policy trained with.
Measured cost at U(30,60), j2=1: **79.5% → 69.0%, −10.5 pp for nothing.** It
runs as one diagnostic eval cell in stage 1 and nowhere else.

---

## 4. Self red-team

### (a) Can Φ(mode 2) see the J2 drift, and does target-plane precession break telescoping?

**Telescoping: intact, structurally.** Φ is a function of state (both element
sets), not of time. PBRS policy-invariance requires exactly that; J2 changes
the *transition* function, not Φ's status as a potential. With
`shape_gamma = 1` the episode total telescopes to Φ_T − Φ_0 **regardless of
path**, so the only way to "farm" is to end in a high-Φ state — which is the
objective. Measured do-nothing leak over a full cap: max farmable gain
**+0.00556**, 0.056% of the +10 terminal.

**Goalpost movement: real, bounded, and 22× from mattering.** Under J2 with
i_s ≠ i_t, Δi_rel *grows* during the drift with no action, so Φ's plane term
falls while its λ term rises. Whether a drift slice can go negative is a pure
ratio, independent of the box:

```
plane loss per radian of λ = W_m · (dv_pl rate) / dv_ref
                           = 0.8167 · 12.44 / 700   = 0.01451
λ gain    per radian of λ  = W_λ / π                = 0.31831
                                             ratio  = 21.9×
```

(12.44 m/s per radian = 0.0936°/rad × 132.9 m/s per degree of plane change at
LEO.) It would take a **273 m/s-per-radian** plane channel to flip a drift
slice; the actual channel is 12.4. Confirmed by direct measurement, not just
algebra: the drift leg stays monotone **36/36** under J2 (min slice +0.01249 vs
the two-body +0.01512), **zero sign flips**, and the worst adverse step is
**unchanged at −0.0354** — still set by the burn quantum. The per-slice numbers
also match the algebra: predicted λ gain 0.0270/slice vs measured max 0.02712;
predicted plane loss 0.00123/slice vs measured 0.00102.

**Verdict: (a) is not a threat at this rung.** It would become one only if
`shape_dv_ref_ms` were cut ~20× or the channel were driven ~20× harder.

### (b) Does the success box's plane term stay well-defined when both planes precess?

**There is no plane term in the box.** `check_termination` (orbital.h:1670-1677)
converts both orbits to Cartesian and tests

```c
dist_to_target < env->rendezvous_radius_m  &&  rel_vel < env->rel_vel_tol_ms
```

— a plain 3D position/relative-velocity test on the propagated states. No
elements, no gauge, no Ω, nothing that precession can make ambiguous. It is
frame-invariant by construction and precession-proof.

The "0.3775° free-plane zone" quoted throughout is a **derived** quantity (the
largest Δi_rel that still admits a position inside the box), not a coded one.
So (b) is a non-issue — but with one caveat that is *not* about well-definedness:
the box is evaluated on **mean elements** under `j2_mode=1`. At 30 km / 50 m/s
that is benign (83 m / 0.094 m/s of mean-vs-osculating relative error per orbit
at 5 km separation). At **5 km / 1 m/s the velocity term is 9.4% of tolerance
per orbit**, so a tight-box J2 arm must not claim osculating-grade terminal
fidelity. Every eval label carries a `MEAN ELEMENTS` tag for this reason.

### (c) Is 50M enough when the J2 timescale is deg/day and an episode is hours?

**The premise is wrong — the drift is not small on an episode timescale.**

| quantity | value |
|---|---|
| median \|Ω̇_s − Ω̇_t\| over the sampled population | 0.436 °/day |
| episode cap | 3000 sub-steps = 50 h = 2.083 d |
| accrued ΔΩ over a full cap | 0.908° |
| → relative inclination injected (×sin i_t) | **0.692°** (gate-measured mean) |
| free-plane zone at the 30 km / 50 m/s box | 0.3775° |
| **ratio** | **1.83×** |
| Δv equivalent | 92 m/s = **19% of the 478 m/s budget** |
| measured *realized* median episode length | 1119–1666 sub-steps → 0.34–0.51° = **90–135%** of the zone |

So the per-episode effect is **order-unity in the units that bind**, not a
rounding term. It is also directly observable: `obs[21,22]` carry δı⃗ in the
chaser RTN frame normalised by `max(di_max, 0.25°) = 1°`, so 0.69° of accrued
relative inclination reads as ~0.69 obs units — plainly visible inside
`Box(-2, 2)`.

**Budget arithmetic.** 50M env steps at a measured ~16 decisions/episode
(1627 decisions / 100 episodes) is ~3.1M episodes. Credit horizon: γ = 0.995
per decision over ~16 decisions gives 0.995¹⁶ = 0.92 — the terminal is not
discount-hidden. Wall clock: three historical 50M plain-`puffer_orbital` runs
in this repo completed ckpt 5 → 382 in **7–8 minutes**; the nav lineage's 50M
arms took ~99 min because of the Python-side EKF, which this rung does not use.

**So the honest answer to (c) has two halves.** The signal is learnable — it is
large, observable, and inside the credit horizon. But **at the loose box there
is only ≤10.5 pp of headroom to demonstrate it in**, and 50M steps of PPO on a
10 pp target is a weak experiment regardless of how learnable the signal is.
The tight box has 43–97 pp of headroom and is where the demonstration lives.

**One further risk specific to stage 3 that (c) surfaces.** At the tight box the
J2-blind policy fails **200/200 with `safety_cap`**, median episode length
exactly the cap. With `cap_terminal_reward = 0.0` and `shape_gamma = 1`, a
capped episode still collects Φ_T − Φ_0 and pays no terminal penalty, so if the
warm start never samples a success the +10 is never seen — the T3 red-team #1
flatline mechanism in new dress. Stage 3 should be treated as *at risk of not
bootstrapping* and may need an intermediate box (10 km / 10 m/s, where the
J2-blind floor is 56.5% rather than 0%) as a rung between them.

---

## 5. The tight-box ladder (stages 3a / 3b)

Added after the survey, on the §4/§5(c) finding that the loose box is the
warm-up and the tight-box recovery is the headline.

| stage | box | warm start | floor rows run in-campaign |
|---|---|---|---|
| 2 | 30 km / 50 m/s | `seed42_X3_3d_di1deg` | — |
| **3a** | **10 km / 10 m/s** | **stage-2 child** | chain floor + `seed42_TB3D_box10k10` @ j2=1 (survey: 113/200) |
| **3b** | **5 km / 1 m/s** | **stage-3a child** | chain floor + `seed42_TB3D_box5k1` @ j2=1 (survey: 0/200) |

Each rung warm-starts from the previous rung's child, so the policy walks the
box down rather than bootstrapping from a 0/200 floor in one jump. Two floor
rows per rung, run **before** training: the *chain* floor (the arm's own warm
start at this box under J2 — the number it must actually beat, and the
tripwire's reference) and the *reference* floor (the box's published parent at
i_t = 0, reproducing the survey row so the campaign is self-contained).

`shape_w_match` stays **0.8166667** through 2 → 3a → 3b: the chain's parent is
the stage-2 child, which is X3 lineage, and an arm must inherit its *own*
parent's value. The TB3D ladder's 0.35 appears only in the `_floor_ref` rows,
which evaluate the TB3D parents under their own training config.

### 5.1 Bootstrap tripwire

`scripts/orbital/extj2/j2_flatline_check.py`. With `cap_terminal_reward = 0`
and `shape_gamma = 1` a capped episode pays nothing and still collects the
telescoped Φ_T − Φ_0, so if the warm start never samples a success the +10 is
never seen — the T3 red-team #1 mechanism. At 5 km / 1 m/s the J2-blind policy
fails **200/200 on `safety_cap`**, so this is live.

The checker parses the trainer's own dashboard frames (`Steps` then `perf`,
once per frame; verified against a real 384-frame log) and emits one RESULT
line: **FLATLINE** if `perf` never exceeded `floor + 5 pp` after 20M steps,
else OK, plus INCOMPLETE / UNREADABLE / SKIPPED for degenerate inputs. All four
paths tested. **A FLATLINE verdict never aborts the campaign** — it is a
finding, named in the log rather than discovered in the eval two stages later,
and the campaign completes and reports it.

### 5.2 Claim labelling

Every `j2_mode=1` row emits a mean-element claim line into the progress log,
with the caveat quoted **at the strength that box earns** rather than borrowed
from the tightest one. The relative-state error is 0.094 m/s per orbit at 5 km
separation, so as a fraction of the box's velocity tolerance:

| box | slip per orbit | wording |
|---|---|---|
| 30 km / 50 m/s | **0.19%** | mean-element claim |
| 10 km / 10 m/s | **0.94%** | mean-element claim |
| 5 km / 1 m/s | **9.40%** | *…and* "meets the box in mean elements, **FULL STOP** — never osculating-grade rendezvous" |

### 5.3 ETA

Measured on this machine, not estimated:

| component | wall |
|---|---|
| stage 0 (C gates 1 s + sampler gates 2 s + Python ladder 18 s) | **~21 s** |
| stage 1 (8 eval cells) | **~10 s** |
| one 50M `puffer_orbital` arm | **~8–15 min** (3 historical runs: ckpt 5→382 in 7–8 min) |
| each eval cell | 0–3 s |

**Full `0,1,2,3a,3b`: ~25–50 min**, dominated by the three 50M arms. The nav
lineage's ~99 min per 50M arm was the Python-side EKF, which this rung does not
use.

---

## 6. Recommendations

1. **Run every training arm at `lvlh_frame_mode = 1`.** It is bit-exact for
   every shipped checkpoint and the confound it removes is 33.5 pp.
2. **Run every training arm at `raan_target_sample = 0`.** Ω_t is gauge;
   sampling it costs 10.5 pp for zero task content. Keep it as a diagnostic.
3. **Reconsider the loose box for stage 2.** ≤10.5 pp of headroom. If the goal
   is a capability demonstration rather than a fidelity statement, the
   10 km / 10 m/s box (−43 pp J2 gap, 99.5% control) is the better rung and
   still only one variable away from the shipped TB3D lineage.
4. **If stage 2 stays at the loose box**, use band `U(30,60)` for maximum
   headroom (10.5 pp, 90% control) or `U(25,50)` for a cleaner control
   (6.5 pp, 99.5% control). `J2_BAND` selects it.
5. **Do not report tight-box J2 results as osculating-grade.** Mean elements.
