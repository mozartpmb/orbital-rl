# ext-j2 — secular mean-element J2

Branch `ext-j2`, 2026-08-13. Binding spec:
`scripts/orbital/ext_recon/reports/j2_A_design.md`. Scope was deliberately
narrow: **physics + kwarg + obs + validation. No training, no sampler change,
no merge.**

Everything below was measured on the **worktree build**
(`/Users/pete/space_training-j2/pufferlib`), never the main checkout. Full
console record: `scripts/orbital/extj2/j2_anchors.txt`.

---

## 1. What was implemented

### 1.1 The propagator

`propagate_orbit_j2(Orbit*, dt, j2_mode)` in
`pufferlib/pufferlib/ocean/orbital/orbital.h`, ~25 lines, three call sites
(bodies / target / satellite in the `c_step` sub-step loop).

```
n = sqrt(MU/a³)    p = a(1−e²)    k = 1.5·n·J2·(R_EQ/p)²
Ω̇ = −k·cos i
ω̇ = +0.5·k·(4 − 5 sin²i)
Ṁ =  n + 0.5·k·√(1−e²)·(2 − 3 sin²i)
ȧ = ė = i̇ = 0
```

`j2_mode = 0` reaches the verbatim legacy `propagate_orbit` by **branching**,
never by adding a zero term. That is what makes the anchors bit-exact rather
than float-close, and it is verified as such (J-G0: 0 double mismatches in
20,000 draws spanning 300–8000 km, e ≤ 0.30, i ≤ 60°, every warp τ).

Two constants added:

| constant | value | why |
|---|---|---|
| `J2_COEF` | `1.08262668e-3` | WGS-84. The memo quoted `1.08263e-3`; the extra digits are a 3.1e−6 relative change, below every measurement floor in the design's own oracle table. That 3.07e−6 residual is exactly what J-G1 measures against the recon table — it is the constant delta and nothing else. |
| `J2_R_EQ` | `6.378137e6` | **Equatorial** radius, deliberately not `R_EARTH = 6.371e6` (the *mean* radius the env uses for altitudes and the keepout). Using the env constant would bias every rate by (6371/6378.137)² = −0.22%. The 7 km inconsistency in the altitude bookkeeping is pre-existing and is not laundered into the dynamics. |

**The equatorial special case is load-bearing, not tidiness.** At `i == 0`, Ω is
gauge but Ω̇ = −k is maximal; the physical rate is ϖ̇ = Ω̇ + ω̇ = +k. Both rates
fold into `omega` and `raan` stays exactly `0.0`. Get this wrong in either
available direction and something breaks silently:

- propagate Ω normally → `target.raan` leaves 0.0 after one step →
  `gauge_from_orbit`'s `identity` fast path disengages **mid-episode** and λ
  switches from the bit-exact `M+ω` form to the cartesian round trip;
- "skip Ω̇ because Ω is gauge" while keeping ω̇ → ϖ̇ = 2k, wrong by exactly 2×.

Both are in the mutation matrix (J-G7) and both fire.

### 1.2 Surface

`j2_mode` (int, default 0), plumbed `orbital.py` → `binding.c::my_init` →
`orbital.ini` → `eval_checkpoint.py` (`--j2-mode`), following `dim3_mode`
exactly. `OrbitalNav` inherits it through `**kwargs`.

Preconditions asserted in `my_init`:

| condition | behaviour | why |
|---|---|---|
| `j2_mode=1` requires `dim3_mode=1` | **hard `ValueError`** | at `dim3_mode=0` every orbit is exactly equatorial, so J2 would apply a real ϖ̇ = +k precession to the whole 2D lineage's ω — unanchored by any 2D checkpoint or test |
| `j2_mode=1` requires `num_debris=0` | **hard `ValueError`** | inherited from `dim3_mode`: the 3D/J2 obs block occupies body slots 21–32 |
| `i_target_rad = 0` under `j2_mode=1` | **stderr warning** | see §4.1 — this is a deviation from the design |

### 1.3 Observations

```
obs[29] = cos i_sat      written only under j2_mode = 1, else 0.0f
obs[30] = cos i_target   ″
obs[31] = 0.0f           reserved (Ω̇ channel, deliberately deferred)
obs[32] = 0.0f           reserved, no assignment
```

Two slots, not four. J2 is axisymmetric, so Ω remains gauge and `cos i` is the
only new physical information the equator adds; feeding Ω would reintroduce the
rotation-variance the target-plane gauge exists to remove. Obs dim stays 38.

Gated on `j2_mode` (not on `dim3_mode`) so the shipped checkpoints stay
bit-exact: columns 29–32 are random-init and zero-gradient in every trained
policy (`n3d_REDTEAM` MAJOR-14 / NON-ISSUE-9), so a nonzero write at
`j2_mode = 0` would be a silent perturbation of a trained encoder.

`obs[31] = Ω̇_s/|Ω̇|_ref` is the quantity the go-around decision keys on, and the
design recommends deferring it to a second arm so the attribution of any J2
result stays clean. Deferred, per that recommendation.

---

## 2. Validation

`scripts/orbital/extj2/j2_gates.c` — **22/22**, standalone C, calls the shipped
header functions directly (the `a2_bitexact.c` / `v4_gates.c` pattern).
`scripts/orbital/extj2/verify_extj2.py` — **14/14**, exercises the built
extension end to end. It refuses to run unless the imported `pufferlib` *and*
the binding `.so` both live under the worktree, because the `puffer` console
script and a bare import both resolve to the MAIN checkout.

### 2.1 Rates — expected vs measured

| check | expected | measured | verdict |
|---|---|---|---|
| recon table, 5 cells (400 km × {28.5, 51.6, 97.4}°, 8000 km, 20200 km), Ω̇ / ω̇ / (Ṁ−n)/n | agreement to the J2-constant delta | worst rel err **3.07e−6** in every cell | PASS |
| sun-synchronous rate, i = 98.6°, 800 km | +0.98565 °/day (360°/365.2422 d) | **+0.98873** °/day, 0.31% | PASS |
| invert for i_SSO at 800 km | ~98.6° | **98.5730°** | PASS |
| ISS-like nodal regression, 51.6°, 400 km | ≈ −5 °/day | **−5.0208** °/day | PASS |
| ω̇ at the critical inclination 63.4349° | 0 | **+0.000e+00** °/day (exactly) | PASS |
| ω̇ 1° off the critical inclination | O(0.1) °/day | **0.2784** °/day | PASS (teeth) |

### 2.2 Bit-exactness at `j2_mode = 0`

| check | expected | measured | verdict |
|---|---|---|---|
| J-A1: `j2_mode=0` vs `propagate_orbit`, bitwise on (a, e, M, θ, ω, Ω) | 0 | **0 / 20000** | PASS |
| obs[29–32] over 120 steps × 32 envs at `j2_mode=0` | identically 0 | max abs **0.000e+00** | PASS |
| legacy anchor (Phase-5e ckpt, corrected dynamics, seed 42) | 26/200, success=26 collision=1 safety_cap=142 stranded=31 | **26/200**, same causes, md5 **f8a2388f0992** | PASS |
| T3 canonical (`models/t3/seed42_L2_headline.pt`, seed 123) | 100/100 | **100/100**, md5 **68b267bed369** | PASS |
| X3 3D (`models/t3/seed42_X3_3d_di1deg.pt`, seed 123) | 100/100 | **100/100**, md5 **003105f29898** | PASS |

The legacy and T3 md5s are **identical to the values recorded in
`scripts/orbital/nav/n3dnav_anchors.txt` before this branch existed** — so the
anchor is action-stream identity, not just a matching score.

### 2.3 Warp exactness

| τ | \|Δpos\| j2=1 (X3 band) | \|Δpos\| j2=0 | max \|Δangle\| j2=1 |
|---|---|---|---|
| 1 | 0 | 0 | 0 |
| 5 | 3.38e−8 m | 2.46e−8 m | 1.78e−15 rad |
| 30 | 1.47e−7 | 8.59e−8 | 1.33e−14 |
| 60 | 2.75e−7 | 1.12e−7 | 2.67e−14 |
| 180 | 8.50e−7 | 2.13e−7 | 7.99e−14 |
| 360 | 1.98e−6 | 4.21e−7 | 1.58e−13 |
| 3000 | 1.22e−5 | 3.37e−6 | 1.32e−12 |

Through the **real env**, `verify_extj2.py` a2 shows warp action 17 (τ=360) is
**bitwise identical** to 360 coast decisions across the whole 36-element state
vector — because `c_step` sub-steps warps, so both paths execute the same
arithmetic. The table above tests the *property* that makes sub-stepping legal
(the map is closed-form and additive in dt), not a path the env takes.

See §4.2: the design's absolute 1e−6 m threshold is restated.

### 2.4 Physics sanity at `j2_mode = 1`

| case | Δa | Δe | Δi | worst \|E − (−μ/2a)\|/\|E\| | ΔΩ over 100 h |
|---|---|---|---|---|---|
| circular LEO-500, i=51.6° | 0 (bitwise) | 0 | 0 | 1.03e−15 | −19.87° |
| eccentric LEO-500, e=0.02 | 0 | 0 | 0 | 1.28e−15 | −19.89° |
| retrograde LEO-800, i=97.4°, e=0.05 | 0 | 0 | 0 | 1.21e−15 | +3.57° |
| WIDE-8000, i=45°, e=0.30 | 0 | 0 | 0 | 1.88e−15 | −2.06° |

A circular orbit keeps `e == 0.0` **bitwise** for 6000 steps. Through the real
`c_step` (3000 coast sub-steps): a and e bitwise unchanged, Δi = 6.4e−15°, and
Ω̇/ω̇/Ṁ match the closed form to **2.7e−12** relative.

Equatorial closure: `raan == 0.0` bitwise at every one of 6000 steps, and
ϖ̇ = 1.552402e−6 rad/s vs the closed-form Ω̇+ω̇ = 1.552402e−6 (rel 1.4e−13); the
wrong 2k answer would be 3.104804e−6.

### 2.5 Shaping — the design's 36/36 matrix, reproduced against the shipped C

The `3d_C §3.1` reference maneuver re-scripted in C and flown through the
shipped `compute_phi` / `apply_impulse` / `propagate_orbit_j2`:

**shaping_mode 2** (w_m 0.8167, dv_ref 700)

| leg | ΔΦ j2=0 | ΔΦ j2=1 | Δ | design's Δ |
|---|---|---|---|---|
| L1a coast-to-node | 0.0000 | 0.0002 | +0.0002 | +0.0002 |
| L1b plane crank ×6 | 0.1236 | 0.1237 | +0.0001 | +0.0001 |
| L2 drift-open ×5 | −0.1735 | −0.1739 | −0.0004 | −0.0004 |
| L3 drift (1110 steps) | 0.9913 | 0.9464 | **−0.0449** | −0.0450 |
| L4 drift-close ×5 | −0.0317 | −0.0325 | −0.0008 | −0.0008 |
| **TOTAL** | **0.9098** | **0.8639** | **−0.0458 (−5.0%)** | −0.0458 |
| worst adverse step | −0.0354 | −0.0354 | **0.0000** | 0.0000 |
| Δi_rel after crank / drift | 0.2037° / 0.2037° | 0.2028° / **0.4898°** | **+0.2860°** | +0.286° |

- **Drift-leg monotonicity: PRESERVED. 36/36** 30-step slices strictly positive
  under J2, min **+0.01249**, max +0.02610; two-body min +0.01512, max +0.02712.
  **Zero sign flips.**
- **Worst adverse step unchanged at −0.0354** = `W_m·25/dv_ref`, still set by the
  burn quantum. Margin vs the +10 terminal: 283×.

**shaping_mode 1** (w_m 0.35, dv_ref 300) — not in the design, run because the
task asked for it. Monotonicity likewise **36/36**, min +0.00964 (two-body
+0.01375); worst adverse step also −0.0354; total ΔΦ delta only −0.0026,
because mode 1 has no plane term and therefore cannot see the 0.286° of
injected relative inclination at all. That is the cleanest available statement
of *why* J2 is a task change only under mode 2.

### 2.6 Do-nothing leak — shipped vs recon reference

δa = 0 exactly, δλ = 0, coast a full cap:

| alt km | i_t | Δi | ΔΦ total (shipped / recon) | max farmable gain (shipped / recon) | Δ(dv_pl) |
|---|---|---|---|---|---|
| 500 | 0.0° | 1° | −0.000132 / −0.000132 | +0.000000 / +0.000000 | **+0.0000** |
| 500 | 28.5° | 1° | +0.000972 / +0.000975 | +0.000972 / +0.000975 | −7.72 m/s |
| 500 | 51.6° | 1° | +0.005364 / +0.005370 | +0.005558 / +0.005564 | −13.17 m/s |
| 500 | 97.4° | 1° | −0.032906 / −0.032918 | +0.000000 / +0.000000 | +25.16 m/s |
| 500 | 51.6° | 0° | −0.000000 / −0.000000 | +0.000000 / +0.000000 | +0.0000 |
| 8000 | 51.6° | 1° | +0.000424 / +0.000425 | +0.000424 / +0.000425 | −0.96 m/s |
| 8000 | 97.4° | 1° | −0.000727 / −0.000727 | +0.000000 / +0.000000 | +0.48 m/s |

Worst deviation from the recon reference across all 7 cells: **1.23e−5**.
Worst farmable gain over a full cap: **+0.00556**, inside the restated 0.006
gate. **J-A5 inertness guard**: at i_t = 0, Δ(dv_pl) = **+0.000000000 m/s** —
exactly inert, as predicted.

### 2.7 Mutation matrix — 14 seeded errors, 0 escaped

An invariant list is worthless until you show it fires. Checks: C1 SSO rate,
C2 i_SSO inversion, C3 ISS Ω̇, C4 ω̇ zero at i_crit, C5 rates vs longhand closed
form at LEO/e=0.02, C6 same at WIDE-8000/i=45/e=0.30, C7 equatorial closure.

| seeded error | caught by |
|---|---|
| Ω̇ sign flipped | C1 C2 C3 C5 C6 C7 |
| Ω̇ factor 0.75 not 1.5 | C1 C2 C3 C5 C6 C7 |
| Ω̇ uses sin i not cos i | C1 C2 C3 C5 C7 |
| ω̇ sign flipped | C5 C6 C7 |
| ω̇ factor 2× | C5 C6 C7 |
| ω̇ uses the Ṁ bracket | C4 C5 C6 C7 |
| Ṁ correction omitted | C5 C6 |
| Ṁ correction sign flipped | C5 C6 |
| Ṁ uses the ω̇ bracket | C5 C6 |
| `R_EARTH` instead of `R_EQ` | C5 C6 C7 |
| `a` instead of `p` in (R/p)² | C5 C6 C7 |
| `√(1−e²)` missing from Ṁ | C5 C6 |
| equatorial: propagate Ω normally | C7 |
| equatorial: skip Ω̇, keep ω̇ (2×) | C7 |

**How to read it honestly.** C1–C4 and C7 are *value* checks against published
constants, independent of how the rate is written. C5/C6 are *consistency*
checks against a longhand transcription of the same equations. Every Ṁ mutation
is caught **only** by C5/C6 — no published constant in this battery constrains
Ṁ on its own. The independent constraint on Ṁ is J-G1's recon-table row
((Ṁ−n)/n agrees with `j2a_core.py`, which shares no code with `orbital.h`, to
3.07e−6). The design's stronger Cowell orbit-averaging oracle (§3.1) was **not**
ported — see §3.

The design's two documented blind spots (`R_ENV` vs `R_EQ` scoring only
2.6e−3 against a 2.5e−3 floor; `p` vs `a` invisible at low e) **do not apply
here**, because the comparand is analytic and has no noise floor: a 0.22% error
is a 2000× violation of a 1e−6 tolerance rather than a marginal one. The
`p`-vs-`a` mutation is nonetheless still checked at e = 0.30 as the design
requires, because at low e it is only caught by margin, not by kind.

---

## 3. Deliberately left out

| item | why |
|---|---|
| **Short-period, m-daily, long-period osculating terms; J2², J3+; drag; third-body; SRP** | Keeping only the secular rates is what buys the closed-form warp property, and warps are load-bearing for this project (§1.4 of the design). |
| **Any mean↔osculating conversion** | Stated, not corrected. Under `j2_mode=1` the env's state **is** the mean element set — at reset, at a burn, and at the success test. A burn is applied to the mean state through the osculating Gauss response, an O(J2) inconsistency. Quantified in the design: relative-state error 83 m / 0.094 m/s per orbit at a 5 km separation (1.66% of separation per orbit; the 29.7 km absolute divergence is common-mode along-track and cancels). Benign where the success classifier looks — but at the 5 km / 1 m/s box the *velocity* term is 9.4% of tolerance per orbit, so **a tight-box J2 rung must not also claim osculating-grade terminal fidelity**. `a_mean − a_osc` at LEO-500 is +5.6 km, larger than the 5 km box; that is a definitional offset, not an error. |
| **`i_target_rad` sampler** (design §4.1 recommends sampling per-episode under `j2_mode=1`) | Out of the assigned scope, which was physics + kwarg + obs + validation with **no training**. `i_target_rad` already exists as a runtime kwarg, so a training arm can set it without further env changes; a *sampler* is a behavioural change that needs its own gates and belongs with the J2-X3 arms. **A training run that leaves `i_target_rad = 0` gets a provably inert J2 plane channel** — the C env warns on stderr at construction, which is the guard that makes the §2.2 inert-knob failure loud instead of silent. |
| **`obs[31] = Ω̇_s/\|Ω̇\|_ref`** | The design recommends deferring it to a second arm so attribution stays clean. Slot left at 0.0. |
| **Cowell orbit-averaging oracle + the 5-cell fuzz matrix** (design §3) | Not in the assigned scope. `j2a_core.py` already contains the integrator, and the recon already ran the protocol and published its mutation table; this branch's independent cross-check is the recon table itself (agreement 3.07e−6) plus published constants. The gap this leaves is stated in §2.7: Ṁ has one independent constraint here, not two. |
| **Re-thresholding the 3D invariants I2/I3/I5 to secular rates under `j2_mode=1`** (design §4.5 calls this "the largest test-side cost, and the one most likely to be under-budgeted") | **Not done.** Under J2 the plane *does* rotate and the apse line *does* precess during a coast, so `ext_invariants3d.py`'s ĥ-constancy (I2/I3) and e-vector-constancy (I5) checks will now fire **by design** at `j2_mode=1`. They are untouched and remain correct at `j2_mode=0`. Anyone running that battery under J2 must convert those three from *constancy* checks to *rate* checks first. This is the single largest known piece of remaining work. |
| **Any training, any merge** | Out of scope by instruction. |

---

## 4. Where the design was wrong or underspecified

### 4.1 §4.1 contradicts §4.2 on `i_target_rad`

§4.1 says to **assert** `j2_mode=1 ⇒ i_target_rad > 0` at init. But §4.2 defines
anchors **J-A3** (equatorial closure) and **J-A5** (inertness guard) at exactly
`j2_mode=1, i_t=0`. A hard assert makes both anchors unrunnable.

**Done instead:** stderr warning, not an error. The inertness is a *training*
mistake (the channel is provably inert, §2.2), not a *correctness* violation,
and the anchors that prove it is inert must be able to construct the env. The
warning names the section and says "fine for the J-A3/J-A5 anchors; wrong for
training", so the failure mode is still loud.

### 4.2 J-A4's "≤ 1e−6 m" is the wrong quantity and the wrong number

The threshold comes from a probe that measured 8e−9…2.3e−8 m. A worst-of-200
sample across an altitude band at τ=360 lands at **1.98e−6 m** — exceeded by
sampling, not by a defect. It is also scale-dependent: the position residual is
the angle residual times the orbit radius, so a metre bound silently tightens
by 2× between LEO and 8000 km on identical arithmetic.

**Done instead:** gated on the scale-free form — **angle residual ≤ 1e−12 rad**
(measured 1.59e−13, ~114 ulp of 2π over 360 accumulations) **and within 10× of
the two-body path's own residual on identical draws** (measured 4.69×, which is
what you get from 3 accumulating angles plus 2 extra `fmod` calls instead of 1).
The metre figure is still printed and flagged `[NOTE]`. Context: `c_step` never
calls the propagator with τ·DT, so this residual is not in the env's execution
path at all; through the real env a τ=360 warp is **bitwise** equal to 360
coast decisions.

### 4.3 §2.2's do-nothing gate reads two-sided but its own table is one-sided

§2.2 restates the gate as "`|do-nothing ΔΦ| ≤ 0.006` over a full cap" while the
same section's table lists **−0.03292** at i_t = 97.4°. Taken literally the gate
fails on the design's own data.

**Done instead:** gated on the **farmable** direction only —
`max_t [Φ(t) − Φ(0)] ≤ 0.006` (measured worst **+0.00556**) — with the signed
total reported alongside. The negative branch is a physical drift the agent
cannot reverse or re-harvest, so it is not a leak. This matches the design's
*intent* (the surrounding prose is about farming) rather than its letter.

### 4.4 §2.2's "any alt, i_t = 0 → 0.00000" is 5-dp rounding, not zero

The underlying CSV has ΔΦ = **−0.000132** at i_t = 0, Δi = 1°. Reproduced
exactly. The quantity that *is* exactly zero — and the one J-A5 actually claims
— is **Δ(dv_pl) = 0**, because at an equatorial target ĥ_t is constant so
Δi_rel = i_s cannot move. ΔΦ still drifts through the λ term, since i_s ≠ i_t
means different Ṁ. Gated on Δ(dv_pl), reported for ΔΦ.

### 4.5 §4.1's pseudo-code calls `wrap_2pi`, which did not exist

`orbital.h` had only `wrap_pi`, and it is defined ~700 lines *after* the
propagator. Added `wrap_2pi` immediately before `propagate_orbit_j2`. It is used
only on the J2 path — the legacy propagator open-codes the same two lines on M
and must stay byte-identical.

### 4.6 §4.3's obs assignment is correct; §4.1's `J2 = 1.08263e-3` was loosened

The design's own correction to `3d_C §2` (obs[28] is occupied; use 29–32) is
right and was followed. The J2 constant was taken to WGS-84 precision
(`1.08262668e-3`) per the implementation instruction rather than the memo's
6-digit value; the resulting 3.07e−6 offset from the recon tables is visible in
every J-G1 cell and is *exactly* that constant delta, which is a useful
incidental confirmation that nothing else differs.

### 4.7 Not a design error, but worth recording: the harness stack limit

`sizeof(Orbital)` is ~4.2 MB (`TrajectoryRecord traj_log[MAX_STEPS]` with
`MAX_STEPS = 12000`). Two of them on the stack overflows the default 8 MB macOS
stack and segfaults **before the first `printf` flushes**, which reads as "the
binary produces no output" rather than as a stack overflow. Every `Orbital` in
`j2_gates.c` therefore has static storage.

---

## 5. Reproducing

```bash
cd /Users/pete/space_training-j2/pufferlib && python3 setup.py build_ext --inplace --force

cd /Users/pete/space_training-j2
cc -O2 -flto -lm -I pufferlib/pufferlib/ocean/orbital \
   scripts/orbital/extj2/j2_gates.c -o /tmp/j2gates && /tmp/j2gates

PYTHONPATH=/Users/pete/space_training-j2/pufferlib \
  python3 scripts/orbital/extj2/verify_extj2.py --stage all --eps 100
```

Never use the `puffer` console script or a bare `import pufferlib` — both
resolve to the MAIN checkout. `verify_extj2.py` refuses to run if they do.
