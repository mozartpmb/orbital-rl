All probes complete. Report below.

---

# RED-TEAM: 3D-nav design (combined `dim3_mode=1` + angles-only `OrbitalNav`)

**Verdict: PROCEED WITH CHANGES — but no training run may launch until B1–B4 are closed.**
The encode/decode layer and the C getter may be implemented today (B4's gate covers
them). The filter, the surrogate and the schedule may not be frozen as specified:
one of the two binding recon documents specifies a chart pole that is **exactly
singular in the campaign's own control arm**, and the cheapest proposed cost lever
**silently multiplies the acquisition latency that is the campaign's headline
metric** by 13×.

Probes (all new, read-only, nothing in the repo modified, nothing committed except
this report): `/Users/pete/.claude/jobs/55d3d276/tmp/rt{1_decode,2_gauge,3_gaugeproper,4_cost,5_cadence,6_pole}.py`.
Every number below is measured by those probes against the C source read verbatim
(`orbital.h`, `binding.c`, `orbital_nav.py`, `nav_math.py`, `nav_surrogate.py`) or
against the independent universal-variable oracle `orbital_math3d.py`. Where I
reproduce a recon claim I say so; where I contradict one I say that too.

**Three findings reframe the design:**

1. **The gauge worry is dead and the pole worry is alive.** The estimated-plane
   gauge is provably safe (§N1, closed form, measured). The *filter chart* pole is
   not: N3D-C §3b's pole is at 90.00° elevation — full gimbal lock — in the
   coplanar arm that N3D-A uses as its 2D-reduction regression anchor.
2. **The acquisition floor is a tick COUNT, and every cost lever divides ticks.**
   The two are the same knob. N3D-C §6b's own `K=12` proposal turns a 45-minute
   acquisition floor into a 9.7-hour one against a 22.2-hour episode.
3. **The cost projection is anchored on an arm the 3D plan does not contain.**
   N3D-C §3d's "~80 min per 50M rung" is derived from the 2D nav60 **range+bearing**
   baseline. The 3D headline rung is bearings-only-plus-surrogate, whose per-tick
   cost I measure at 1.45× the range+bearing tick at the training batch shape.

---

## BLOCKER-1 (surfaces 1/3) — The MSC chart pole is specified two different ways, and *both* specifications hit the singularity on the campaign's own trajectories.

**Attack.** N3D-A §1.2 and N3D-C §3b both specify the 6-state modified-spherical
chart. They do not specify the same pole:

| document | pole | stated rationale |
|---|---|---|
| N3D-A §1.1 | `ĥ_c(t₀)` — chaser orbit normal at epoch | "`el ≡ 0` exactly at Δi=0 ⇒ the 6-state filter collapses onto the shipped 4-state filter component-for-component. That reduction is a regression test, not a claim." |
| N3D-C §3b | `p̂ = unit(u⃗₀ × ĥ_t)` — **"pole in the orbit plane"** | "ε₀ = 0 by construction… Do not pole at ẑ: a purely cross-track separation is reachable late in a plane-matching episode." |

These are 90° apart. Only one can be right, and the chart singularity is at
`el = ±90°`.

**Measurement (`rt6_pole.py`, one full orbit, |el| against each pole):**

| geometry | N3D-A pole `ĥ_c` | N3D-C §3b pole `u₀×ĥ_t` |
|---|---|---|
| **coplanar 5 km (the 2D-reduction anchor)** | **0.00°** | **90.00° — exact gimbal lock** |
| coplanar 10 km | 0.00° | 90.00° — lock |
| Δi=0.02°, 5 km (tight box) | 25.30° | 64.70° — guard fires |
| Δi=0.0417°, 5 km (box plane tol) | 44.58° | 45.42° |
| **Δi=1.0°, ρ_inplane=5 km (X3 rung)** | **87.58° — guard fires** | 2.42° |
| Δi=1.0°, 300 km | 21.50° | 68.50° — guard fires |

Two conclusions, both load-bearing:

**(a) N3D-C §3b's pole is wrong and must be deleted.** It lies *in* the orbit
plane, and the LOS of a close rendezvous is predominantly *along-track* — i.e. in
that same plane — so the LOS sweeps straight through the pole once per orbit. In
the coplanar control arm it does not merely approach the singularity, it *is* the
singularity (90.00°), which destroys N3D-A's `el ≡ 0` reduction anchor — the single
cheapest regression test the 3D filter has.

**(b) N3D-A's pole is right, but its own §6 caveat 6 is understated.** N3D-A
reports "max |el| 27°… the gimbal guard was specified but never exercised."
That is true *of its prototype cells*, because §6 caveat 5 enforces
`Δi ≤ asin(ρ/r)` and "skips infeasible cells". That constraint is a property of a
*held* rendezvous, not of an *approach*. Closed form:

```
|el| > 60°   ⟺   ρ_inplane  <  0.577 · r · sin(Δi_rel)
```

- at Δi_rel = 1.0° (the X3 rung's `di_max`): ρ_inplane < **68 km**
- at Δi_rel = 0.254° (the largest plane error the 30 km T3 box admits): < **17.3 km**

The T3 success box is 30 km. **The guard therefore fires inside the success box at
the headline rung**, on trajectories the prototype excluded by construction. It is
on the critical path and it is unvalidated.

### FIX (required before any filter code is written)

1. **Adopt N3D-A's pole `ĥ_c(t₀)`. Delete N3D-C §3b's pole from the spec.**
2. Lower the re-pole trigger from `|el| > 60°` to **`|el| > 45°`**, which bounds the
   az-noise inflation `1/cos²el` at 2.0 (at 60° it is 4.0 — see MAJOR-8).
3. Implement the re-pole as an **exact similarity transform** on `(y, P_y)` and
   gate it: (i) Cartesian round-trip through the re-pole must be the identity to
   1e-12; (ii) NEES must be continuous across the re-pole (no step in the
   in-bounds fraction); (iii) re-pole count per episode emitted as a `nav_*`
   diagnostic.
4. Exercise it before training, at `Δi = 1°` with `ρ_inplane` swept through 68 km —
   the regime the prototype skipped.

---

## BLOCKER-2 (surface 6) — The acquisition floor is a tick COUNT. Every cost lever divides ticks. They are the same knob, and the campaign's headline metric is acquisition latency.

**Attack.** `nav_acq_min_ticks = 45` (`orbital_nav.py:104`, `nav_surrogate.py:106`)
is an observation **count**, chosen to reproduce NAV-G's `w₀ = 45` — which is also a
count, and which N3D-A §3c re-confirms for the 3D solver ("window must start at 45
obs, not 90"). Separately, N3D-C §3d proposes tick-capping as cost lever (i),
"`K=12` cuts ticks 6.6×". Nobody composed the two.

**Measurement (`rt5_cadence.py`, on the measured 3D action mix
`n3d_tau30_action_mix.csv`, τ̄ = 79.61 reproduced exactly, P(τ≥180) = 0.260):**

| lever | ticks/decision | cost × | sim-time per tick | **45-tick floor becomes** |
|---|---|---|---|---|
| nav60 as shipped (dt = 60 s) | 79.61 | 1.000 | 1.0 min | **0.75 h** |
| dt ≤ 300 s (`nav300`, the only validated stretch) | 16.35 | 0.205 | 4.9 min | **3.65 h** |
| dt ≤ 600 s (unvalidated) | 8.44 | 0.106 | 9.4 min | 7.07 h |
| **K = 12 (N3D-C §6b's own proposal)** | 6.15 | 0.077 | 12.9 min | **9.71 h** |
| K = 6 | 3.34 | 0.042 | 23.8 min | 17.88 h |

Episode sim-time at the measured mix is **22.2 h** (1,330 nav ticks at dt=60 s).
At `K=12` the acquisition floor is 44% of the episode; at `K=6` it is 81%. The
policy would essentially never acquire, and the failure would present as
*"bearings-only does not work in 3D"* — a false negative on the campaign's central
claim, produced entirely by a cost knob.

**Second half of the same defect: the latency comparison is already confounded even
at nav60.** N3D-C §5's pre-registered hypothesis is *"3D-nav acquires with
materially less latency and less Δv than 2D-nav"*. The 3D policy packs **2.45×**
more sim-time into each decision than the 2D one (τ̄ 79.61 vs 32.50). A latency
measured in *decisions* would show 3D winning by 2.45× from the τ mix alone, with
zero contribution from observability.

### FIX

1. **Make the acquisition floor a sim-TIME floor (45 min) plus the information gate
   (`σ_LOS/ρ ≤ 0.20`), never a tick count.** Re-derive it from `dt_tick` on every
   change. The same applies to the real solver's `w₀` at eval.
2. **Pre-register acquisition latency in sim seconds** (equivalently nav ticks at a
   fixed dt), never in decisions, and report the τ distribution alongside it.
3. **Training and eval must run the identical cadence.** A cadence stretched for
   training cost with eval left at 60 s is a train/eval sensor-model mismatch that
   invalidates every arm-vs-arm delta.
4. If a stretch is taken, take **dt ≤ 300 s only** — the 2D `nav300` arm was
   validated at 99.5% @10× noise; dt = 1800 s has no precedent and N3D-C §6b says
   so itself.

---

## BLOCKER-3 (surface 6) — The cost projection is anchored on an arm the 3D plan does not contain. Realistic budget is 5.5–8 h per 50M bearings-only rung, not 80 min.

**Attack.** N3D-C §3d projects "~10K SPS, ~80 min per 50M rung (2D nav60: 50.4K
SPS, 16.5 min)". 50.4K is the 2D **range+bearing** nav60 arm (`EXTNAV_RESULTS.md`
reports 25.6–27.3K for the shipped N-rungs; `nav_REDTEAM` quotes 61K for the same
reference). The 3D headline rung is **bearings-only + `crlb_online` surrogate**,
which is a different cost class — the same substitution error `nav_REDTEAM`
BLOCKER-1 caught NAV-G making.

**Measurement (`rt4_cost.py`, at the actual training batch shape B=256 = one
worker's env slice, OMP pinned to 1, real `nav_math` / `nav_surrogate` objects):**

| op (2D, B=256) | ms |
|---|---|
| `BatchedBearingMPC.predict` | 0.717 |
| `BatchedBearingMPC.update` | 0.083 |
| truth sub-propagate (2n states) | 0.104 |
| `AcquisitionSurrogate.accumulate` | **0.406** |
| **2D nav tick, acquired** | **0.904** |
| **2D nav tick, unacquired** | **1.310** |

The surrogate's FIM accumulation alone is **45% of the acquired tick** — it is not a
rounding term, and in 3D its `stm_fd` goes 6-state (2.35× at B=256 from
`n3d_filter6_cost.csv`).

Per decision, per worker: 2D `0.904 × 32.50 = 29.4 ms`; 3D
`0.904 × 2.25 × 79.61 = 161.8 ms` ⇒ **5.51× on the nav portion** (independently
reproducing N3D-C's 5.5×, but from the bearings-only tick rather than the EKF one).

**Projection**, using N3D-C's own combined nav+env multiplier of 4.9× and letting
`f` be the fraction of 2D wall that scales:

| 2D baseline | 50M as-is | f=0.6 | f=0.8 | f=1.0 |
|---|---|---|---|---|
| 8.5K SPS (bearings-only, the operative one) | 1.63 h | **5.46 h** | **6.73 h** | **8.01 h** |
| 25K SPS (`EXTNAV_RESULTS` N-rungs) | 0.56 h | 1.86 h | 2.29 h | 2.72 h |
| 50.4K SPS (N3D-C's anchor) | 0.28 h | 0.92 h | 1.14 h | 1.35 h |

**5.5–8.0 h per 50M bearings-only rung.** The plan (N3D-C §5) has N2-bo3d fresh × 3
seeds + warm, plus N1-rb3d ⇒ **28–40 h before any wide rung**, against a stated
~6 h/arm ceiling.

**Aggravator: τ̄ is policy-dependent and rises during the run.** A uniform-random
Discrete-30 policy has τ̄ = 22.0 (computed from `ACTION_TAU`); the converged X3
policy has 79.61. Cost therefore grows ~3.6× over a run, so an epoch-1 SPS-based
ETA under-predicts by that factor.

### FIX (ordered; only the first carries no science risk)

1. **Analytic 3D STM (Shepperd/Goodyear) — do this first.** FD-STM is **87%** of the
   3D tick (2.250 of 2.566 ms at B=1024, `n3d_filter6_cost.csv`), 12 of 13
   propagations are FD probes, and *the same object appears in both the filter and
   the surrogate*, so one implementation pays twice. Expected 3–4× on the nav
   portion at zero model change. This alone lands the rung near 2 h.
2. Verify the surrogate's `accumulate` is skipped entirely once a row is acquired
   (it is gated on `~self.acquired`, but at 0.406 ms/tick confirm it, don't assume).
3. Only then consider a cadence stretch — and only under BLOCKER-2's floor fix, at
   dt ≤ 300 s, with eval re-run at the same cadence.
4. Budget from the **converged** τ mix, not from observed early SPS.

---

## BLOCKER-4 (surface 4) — The 19-slot rebuild has seven new truth-leak channels and not one shipped diagnostic would notice.

**Attack.** Under `dim3_mode=1` the target-derived slot set (derived independently
from `fill_observations`, `orbital.h:913-1209`) is

```
{7, 8, 11, 12, 13, 14, 16,  21, 22, 23, 24, 25, 26, 28,  33, 34, 35, 36, 37}
```

— 19 slots, confirming N3D-C §1. The shipped wrapper rebuilds twelve
(`nm.TARGET_SLOTS_T3`, `nav_math.py:74`). The seven new ones are precisely the 3D
task's state:

| slot | C source | what leaks if it is not rebuilt |
|---|---|---|
| 21, 22 | `:1113-1114` | the **true** relative-inclination vector δı⃗ and the relative-node phase — the burn-timing signal |
| 23 | `:1120` | true Δē·ĥ_t |
| 24, 25 | `:1130-1131` | true cross-track LVLH position and velocity |
| **26** | `:1143` | **the true plane-change Δv-to-go** `v_c·‖ĥ_s − ĥ_t‖/dv_ref` |
| 28 | `:1145` | true feasibility margin |

Miss any one and the policy flies the plane leg on **truth** while the write-up says
angles-only. Every shipped `nav_*` diagnostic (`nav_pos_rmse`, `nav_nees_med`,
`nav_clip_rate`, `nav_diverge_rate`, `nav_sigma_los_over_rho`) measures the
*filter*, not the *observation*, so all of them stay green. This is the exact shape
of the project's 11 prior metric-vs-implementation failures, in the one place where
it would invalidate the headline claim outright.

### FIX — one cheap pre-registered gate, run at every rung and in CI before training

**`leakcheck`**: replace the target estimate with a fixed wrong state (e.g. `+1000 km`
along-track) and assert
(a) **all 19** slots change, and
(b) the complement `{0-6, 9, 10, 15, 17-20, 27, 29-32}` does **not** change.

(a) catches a missed rebuild; (b) catches an over-broad one (overwriting `obs[15]`
hands the policy a fake deadline; overwriting `obs[27]`, the chaser-only Δv ledger,
corrupts a chaser-truth channel). Neither direction is detectable any other way.

---

## MAJOR-5 (surface 1) — RAAN alone is sufficient at `i_t = 0` and insufficient the moment it is not. Return `ĥ_s`, not `Ω_s`.

**Measured (`rt1_decode.py`, 300 draws, C algebra ported verbatim):** at
`i_t = Ω_t = 0`,

```
obs[21] = i_s·cos u / di_scale        obs[22] = −i_s·sin u / di_scale
```

to **1.06e-15**, with `u = ω_s + θ_s`. The projection into the chaser's own rotating
RTN frame annihilates the inertial node longitude exactly. **So obs carries
(Δi_rel, u) and Ω_s is absent by construction — N3D-C §2's central claim is
confirmed, and now with the closed form.**

**Sufficiency at the shipped config: YES.** With `i_t = 0`, `Δi_rel = i_s` exactly
(measured above), so `(a, e, θ, ω)` from obs `+ i_s` from obs[21,22] `+ Ω_s` from the
getter completes the chaser 6-state. Nothing else is missing.

I also reproduced N3D-C's `e_t` breakdown analytically rather than trusting it:
setting `Ω_s := 0` rotates *both* decoded bodies by `−Ω_s` (a rigid rotation, exactly
harmless at `e_t = 0`, matching their 0.22 m); at `e_t > 0` the decoded target's `M`
is offset by `−Ω_s` while `M → position` is nonlinear, giving an error of order
`2·a_t·e_t ≈ 6.8e5 m` at `e_t = 0.05` — against their measured **6.1e5 m**. The
mechanism is confirmed, not just the number.

**Insufficiency the moment `i_target_rad ≠ 0`** (an existing kwarg, and any tilted-
target rung): recovering `i_s` from `Δi_rel` and `Ω_s` requires solving
`P sin i_s + Q cos i_s = cos Δi_rel`, which is two-valued. Scan over 200 draws at
`i_t = 51.6°`: **30% admit ≥2 inclination roots.** RAAN-only then needs a 2-D
nonlinear solve with a discrete branch choice, in the hot path.

**FIX.** The getter N3D-C already proposes returns the full element sets
(`(num_envs, 15)` float64). *Use* `i_s` and `Ω_s` from it; never re-derive `i_s` from
obs[21,22]. Identical cost, no branch, no solve. Keep the obs-only decoder as the
**gate** N3D-C §2 describes (agree to < 1 m at `e_target = 0`), not as a dependency.

---

## MAJOR-6 (surface 1) — Getter mechanics: use the out-array pattern, and `-flto` makes the bit-exactness re-run mandatory rather than merely prudent.

Read of `binding.c`: the three existing custom methods are pure reads, take no
RNG, and mutate no state. A `vec_get_state` in the same shape is clean on RNG
stream, buffer layout and env state — **no anchor is at risk from the getter's
semantics.**

Two mechanical points:

1. **Out-array, not allocate.** `vec_get_trajectory` already establishes the
   pattern (`binding.c:108-112`: contiguous float32 check, caller-owned buffer). A
   per-step allocating getter at `1024 × 15 × 8 B = 123 kB` per call per worker is
   avoidable churn on the hot path. Mirror `vec_get_trajectory`, not
   `vec_get_episode_init_info`.
2. **The build is `-O2 -flto`** (`pufferlib/setup.py:108-112`). Link-time
   optimisation means adding a translation-unit-level function can change
   cross-TU inlining and therefore FP code generation in `fill_observations` and
   `compute_phi`. The probability is low; the cost of being wrong is the whole
   bit-exact lineage. N3D-C §5 anchor 4 already says to re-run
   `ext_rt3d_a2_bitexact.csv` and the 425,477-row T3 canonical trajectory after any
   `binding.c` change — **this is why**, and it also applies to any toolchain or
   flag change, not only to source edits. Add legacy 26/200 and the T3 canonical
   200/200 md5 action-stream replays (`verify_extnav.py` stage v1) to the same
   gate.

---

## MAJOR-7 — `nav_max_ticks` is a live filter/truth desync at Discrete-30 (26.0% of decisions), and its natural fix collides with BLOCKER-2.

Confirmed in code. `orbital_nav.py:443-444` caps τ, then the sub-propagation loop
(`:464-491`) ticks that many times at `dt = 60 s` while the C env advances the full
τ — and `:487-490` pins `sc[last] = sat_c[last]` to the env's *current* epoch. So on
a capped decision the filter predicts `K·60 s` and is then handed a bearing of the
**true current** geometry: at τ=360, K=60 that is a 5-hour state jump presented as
one 60 s innovation. The filter diverges, `_guard` reinitialises it, and the only
visible symptom is an elevated `nav_diverge_rate` — not an obvious bug.

NAV-H §2.4 called this "inert for every shipped config"; that was true at
Discrete-16 and is **false at Discrete-30**, where τ=180/360 is 26.0% of decisions
(`n3d_tau30_action_mix.csv`).

**FIX.** Either set `nav_max_ticks = 0` and pay BLOCKER-3, or adopt fixed
measurement count with adaptive interval (`n_ticks = min(τ, K)`,
`dt_tick = τ·60/n_ticks`) — **but only together with BLOCKER-2's floor fix**, or the
cure silently rescales acquisition latency by the same factor it saves.

---

## MAJOR-8 — `ACTION_TAU` / `ACTION_DV_MAG` are length 20 against `NUM_ACTIONS 30`. Hard crash on step 1, and the Δv-conditioning is wrong on 23.6% of burn decisions if fixed carelessly.

Confirmed: `nav_math.py:40,50` are 20 long; `orbital.h:75` is `NUM_ACTIONS 30`.
`nm.ACTION_TAU[np.array([26])]` → `IndexError`.

Extend both from `orbital.h:78-137,166-175`. The trap in the extension:
`ACTION_DV_MAG[26..29] = hypot(25, 25) = 35.355`, **not 50** — `apply_impulse`
(`orbital.h:785-818`) takes the norm of the *assembled* vector and `p̂ ⟂ n̂` exactly.
Rows 20-25 are `{1, 1, 10, 10, 25, 25}`, all τ=1. Getting this wrong misstates the
surrogate's Δv conditioning — the term `nav_surrogate.py`'s docstring calls
"load-bearing, not cosmetic" — by 1.41× on the 23.6% of the measured mix that is
actions 26-29.

---

## MAJOR-9 — Every 2-component slice in the wrapper is a silent 3D bug. One of them fails totally and invisibly.

Enumerated from source: `_decode`, `_guard`, `_diag`, `_sigma_los_over_rho`,
`_apply_action_ablation`, `_init_rows`, `_tick`, `self._pair` `(2n,4)`,
`self._prev_sat/_prev_tgt` `(n,4)`; and in `nav_math`: `range_prior_intervals`
(`sat_cart[:,0]**2 + sat_cart[:,1]**2`, `ux,uy = cos β, sin β`), `msc_encode` /
`msc_decode`, `process_noise` (a 2-D double integrator), `cartesian_to_elements`,
`orbit_to_cartesian`, `nees`.

Most are mechanical. **One is not.** In the 3D MSC state
`y = [az, el, ω_a, ω_e, ρ̇/ρ, ln ρ]`, `ln ρ` moves from index **3 to 5**. The
divergence guard reads it at index 3 (`orbital_nav.py:421`):

```python
rho = np.exp(np.minimum(self._filt.y[:, 3], 25.0))
bad |= (rho < RHO_MIN_M) | (rho > RHO_MAX_M)
```

Under a 6-state, `y[:,3]` is `ω_e` — an angular rate of order 1e-4 — so
`exp(ω_e) ≈ 1.0 m < RHO_MIN_M = 1.0`… and every row is flagged bad on the boundary,
reinitialising the filter **every single step, forever**. The run trains, the loss
curve looks plausible, `nav_pos_rmse` is merely bad rather than absurd, and the
experiment measures nothing.

**FIX.** Replace positional indices with named constants (`IDX_AZ, IDX_EL, IDX_WA,
IDX_WE, IDX_RDOT, IDX_LNRHO`) in the 6-state port, and add a unit test asserting
`_guard` flags **0** rows on a healthy batch. `_apply_action_ablation` additionally
computes ρ in 2-D, which under-reports the true separation by the cross-track term
(1.62 km at the 5 km box per N3D-A §3b) and fires the interlock in the wrong states.

---

## MAJOR-10 — The fine-burn ablation set is wrong for 3D, and the dual-control treatment is confounded with it.

`_FINE = [12,13,14,15,18,19]` (`orbital_nav.py:311`) remains correct for the
*in-plane* fine burns under Discrete-30 (indices 0-19 are frozen). But actions
**20/21 are normal ±1 m/s** — also fine burns, and by N3D-B §3.3 the *observability
treatment* in 3D (1 m/s normal beats 1 m/s prograde by 2.5–8× in information at
every terminal cell, and is 5.5× more robust to mis-placement). A `T-BO−act` arm
that blocks only `[12..15,18,19]` leaves the treatment axis wide open, and its
result is uninterpretable.

**FIX.** Split into `_FINE_INPLANE = [12,13,14,15,18,19]`,
`_FINE_NORMAL = [20,21]`, `_NORMAL_ALL = [20..25]`, `_COMBINED = [26..29]`; state
per arm which set is blocked. And carry NAV-F's lesson: `T-BO−act`'s −4.5 pp
appeared **in truth mode too** (94.0% vs 97.0%), which is what proved fine burns
guidance-critical rather than info-critical. A `T-BO−normal` arm therefore needs a
truth control at the same box, or the plane leg's guidance cost will be misread as
an information result. N3D-B §3.3 states the same thing from the other side:
*"the detection metric must be counterfactual information gain per burn, not
action-mix; action-mix is invalid in 3D by construction."*

---

## MAJOR-11 (surfaces 3/5) — The plane must never enter the surrogate as a separate scalar, and the Cholesky diagonal fallback silently destroys the structure that *is* the claim.

**The correlation is the whole point.** N3D-B measured `corr(range, plane) =
0.999–1.000` at Δi>0 and **exactly 0.0000** at Δi=0, and validated the rank-1 model
`Σ ≈ σ_k²·x_rel x_relᵀ + floor²·P_⊥` component-by-component over 1800 cells (median
model/CRLB 0.987–1.004). `crlb_online` gets this for free: `Σ = ratio²·Φ F⁻¹ Φᵀ`
(`nav_surrogate.py:427-439`) carries the realized arc's full correlation structure.

**Two ways to lose it, both already latent in the shipped code:**

1. **`table` mode.** N3D-B §4.3's "plane bolt-on, one line" emits `sigma_plane_pos`
   and `sigma_plane_vel` as independent scalars. The existing 2D table branch
   (`nav_surrogate.py:397-405`) builds the position block as a correct rank-1 outer
   product but the **velocity block as a pure diagonal** (`Sig[2,2]=Sig[3,3]=sig_v²`).
   In 3D that is exactly the channel N3D-B §5 names *the new binding constraint* —
   out-of-plane velocity, `σ_vplane = σ_k·v_c·sin Δi_rel` (predicted 1.903 vs
   measured 1.902 m/s), i.e. rank-1 along `x_rel`, not diagonal. Drawing it
   independently hands the policy averaging gain the real filter never delivers.
   **FIX: make `crlb_online` the enforced default under `dim3_mode` (raise on
   `mode='table'` unless explicitly opted into for a reporting run); if table mode
   ships, build the full 6×6 rank-1 Σ, never a diagonal.**
2. **`_chol`'s fallback** (`nav_surrogate.py:441-456`) walks a jitter ladder and then
   **falls back to the diagonal of Σ**. A rank-1-dominated 6×6 is near-singular *by
   construction* — that is the observability structure — so this is the fallback
   most likely to fire in 3D, and it converts an anisotropic scale-family error into
   a near-isotropic one with no error, no warning and no counter.
   **FIX: count fallbacks, emit as a `nav_*` diagnostic, gate the campaign on 0.**

**Corollary for gates (surface 5): do not add a plane term to any nav-side gate.**
Two independent failure modes, from N3D-B's own map:

- *Certify-garbage.* At Δi_rel = 0 the plane channel decouples completely
  (`corr = 0.0000`) and is measurement-limited **irrespective of the range error**:
  at 5 km, ΔI=0, `σ_plane = 0.73 m` while `σ_range = 681,895 m`
  (`plane_over_range` spans 3e-6). "The plane converged ⇒ the solution is good" is
  therefore false *precisely at the terminal condition guidance drives toward* —
  reproducing `nav_REDTEAM` MAJOR-2's 2,951 km certification in a fresh channel.
- *Double-count.* At Δi_rel > 0, `corr → 1.000`, so an RSS gate
  `√(σ_range² + σ_plane²)/ρ` counts one degree of freedom twice: conservative in
  direction (≤ √2) but it makes the 3D gate non-comparable to the 2D gate NB1 was
  measured against, which is the entire basis of the arm-vs-arm claim.

**Keep NAV-G's gate exactly as shipped: `σ_LOS/ρ ≤ 0.20` on the range channel
only.** N3D-B §2.1 already measured the plane gate as non-binding (`N ≥ 50`,
geometry-independent, range always binds first), so adding one buys nothing and
costs comparability.

---

## MAJOR-12 (surface 3) — The surrogate's `H` must be the basis-free `P_⊥` form, and it needs an equivalence test, because BLOCKER-1 shows `el` reaches the regime where the difference bites.

Isotropic-on-the-sphere bearing noise is `R = diag(σ_β²/cos²el, σ_β²)` in az/el
(N3D-A §1.1), which is exactly N3D-B §1's basis-free
`HᵀR⁻¹H = (σ_β ρ)⁻²·[P_⊥ 0; 0 0]`, `P_⊥ = I − uuᵀ`. An implementation that writes
az/el with an isotropic `R = σ_β² I` inflates the information by `1/cos²el`:
**1.26× at el = 27°, 2.0× at 45°, 4.0× at 60°**. N3D-A's prototype never saw past
27°, so the error would have been invisible there — but BLOCKER-1 measures 87.58° on
the campaign's own trajectory.

**FIX.** Implement `M = P_⊥/(σ_β ρ)²`; add a test that it equals the explicit az/el
FIM with the cos-el-inflated `R` to 1e-12 at `el ∈ {0, 27, 45, 60}°`. Adopt N3D-B
§4.1's two other mandates while you are there: nondimensionalise by
`D = diag(a,a,a,v_c,v_c,v_c)` (not cosmetic — it is the fix for the NAV-G/NAV-F
"exactly singular" disagreement), and tighten the singular guard to `cond > 1e14`
on the **scaled** FIM. N3D-B §1 found the shipped raw `cond > 1e16` guard letting a
numerically singular cell through and writing `sigma_los_bo_m = 0.0` into
`ext_bo_observability.csv` — the CSV asserts **perfect** range observability in one
of the least observable cells in the table. Runtime is defended
(`ObservabilityMap.__init__` maps `v <= 0 → _INF_REL`, confirmed at
`nav_surrogate.py:153-155`), but any human or script reading the CSV is misled.

---

## MAJOR-13 — The two recon lanes report two different quantities both named "σ_plane". A gate or box built by copying numbers across them is a unit error.

- **N3D-A §2.2** reports the plane as an **orbit-plane tilt angle** (degrees):
  "G1 drift: 2.66e-5° at Δi=0 → 1.81e-3° at Δi=0.08°, a 68× degradation."
- **N3D-B §2.1/§3.2** reports the plane as **out-of-plane position** (metres):
  "5 km: 0.73 m at ΔI=0 → 3.34 m at ΔI=1.0°", against a "plane position box 4.98 km".

Both are correct; they are different objects, and the apparent contradiction
("degrades 68×" vs "always at the measurement floor") dissolves once you notice the
units. This matters concretely because **obs[21,22] is fed by A's quantity** (the
tilt, at gain 0.66–0.81 deg/deg per N3D-C §1) **while the success box is stated in
B's** (a position). Mixing them silently rescales by `r ≈ 6.77e6 m/rad`.

**FIX.** Name them distinctly in all downstream code and reporting —
`sigma_plane_tilt_rad` and `sigma_plane_pos_m` — and assert the relation
`sigma_plane_pos ≈ r · sigma_plane_tilt · |sin u|` in one test rather than trusting
either report's prose.

---

## MAJOR-14 — Slot-dependent clamp, and the definitive obs layout.

**Clamp.** `obs_clamp2` (±2, also NaN-trapping, `orbital.h:907`) is applied to
slots **21-28 only**; 7/8/11-16/33-37 are written raw. The wrapper's single
`nav_clip = 4.0` (`orbital_nav.py:120,268-274`) must become **±2 on 21-28, 4.0
elsewhere**, or `recon ≠ truth` whenever the C clamp fires. Reset-time hit rate is
0/512, but 3d_REDTEAM m3 measured 3.24 mid-episode at the widest rung.

**Definitive layout (surface 4):**

| slots | contents under `dim3_mode=1` + nav |
|---|---|
| 0-6, 9, 10, 15, 17-20, 27 | chaser-truth / Earth-conjunction / clock. **Never written by the wrapper.** |
| 7, 8, 11-14, 16, 33-37 | target-derived, rebuilt from the estimate (the existing 12, minus nothing) |
| **21-28** | the C's ext-3d block, **rebuilt from the estimate**, clamped at ±2 |
| **29-32** | **stay 0.0f.** Reserved. |

**Drop the Σ-channel entirely.** NAV-F measured `T-BO+Σ` statistically identical to
`T-BO` on every metric — native 97.0%/98.5%, β adjusted −0.032 (z −0.62), the same
blind-burn suppression (0.29× vs 0.37×), the same Δv (244.5 vs 243.0 m/s) — so it
has no claim on a slot, and deleting it deletes the obs[21] collision (N3D-C §6c)
outright rather than relocating it. If a later arm wants it, use **obs[29]** and
retarget `scripts/orbital/nav/zero_obs_column.py` from column 21 to 29.

One warm-start subtlety the 2D lineage does not have: `seed42_X3_3d_di1deg.pt` was
trained with **obs[21-28] live**, so those encoder columns are *trained*, not at
random init. `NON-ISSUE-9`'s magnitude discipline (≤0.1, or zero the column at load)
now applies to **29-32 and only 29-32**, which were hard-zeroed by
`orbital.h:1146-1149`. Flag the 29/30/31/32 split to the J2 lane before either
ships — `orbital.h:156` reserves the same block.

---

## MAJOR-15 — With the getter, `recon` stops testing the Cartesian route the filter actually flies. Add a second gate.

The C's identity-gauge branch triggers on `inc == 0.0 && raan == 0.0` **exactly**
(`gauge_from_orbit`, `orbital.h:740`), and `cartesian_to_elements`'s equatorial
branch on `hxy == 0.0` exactly (`:625`). A target estimate carried as a Cartesian
6-vector can never decode to exactly zero, so it always takes the general
round-trip branch. To hit N3D-C §5's `≤ 1 ulp` recon gate, the wrapper must
therefore take `(i_t, Ω_t)` from the getter as exact doubles — which means the gate
no longer exercises the Cartesian→element path that `bearings_only` runs on.

**FIX.** Add **`recon-cart`** as a second gate: round-trip the *truth* target
through Cartesian exactly as the filter does, re-encode, and require agreement on
all 19 slots to a **measured, stated** tolerance (not bit-exact, and not asserted).
Without it, the Cartesian-route encode error is first seen mixed with estimation
noise, where it is unattributable — which is how the last eleven of these ended up
undetected.

---

## MAJOR-16 — The out-of-plane velocity seed has 1.15× of margin and scales the wrong way.

`_sigma_v_ecc = max(7.7e3·max(e_max, 0.02), 100)` (`orbital_nav.py:197`) = **154 m/s**
at `e_max = 0`, and `_init_rows` (`:371`) uses it for *both* MSC rate components.
The true out-of-plane velocity ignorance is `v_c·sin(Δi)` = **134 m/s** at
`di_max = 1°`, LEO (N3D-A §3c measures the same 0–134 m/s span). Margin **1.15×**.
At `di_max = 2°` it is 268 m/s and the seeded covariance is over-confident by ~3× on
a channel with no measurement of its own at epoch — the classic bearings-only
divergence trigger.

**FIX.** Seed the out-of-plane rate explicitly from `v_c·sin(di_max)/ρ₀`, decoupled
from the eccentricity term, and assert `sig_rate_oop ≥ v_c·sin(di_max)/ρ₀` at init.

---

## MAJOR-17 — Condition the surrogate on Δi_rel from **truth**, never from obs after `_encode`.

N3D-B §4.3 item 3 specifies reading realized `Δi_rel` "from the C env's own δı⃗
(obs slots 21/22 decode)". In `_nav_step`, `_encode` (`orbital_nav.py:505`)
overwrites the observation **in place**. Reading 21/22 after that point conditions
the surrogate's own error model on the estimate it just produced — a self-referential
loop that raises nothing and produces a plausible-looking number.

Conditioning the *noise process* on truth is legitimate and is exactly what the
shipped 2D surrogate already does with realized separation and realized Δv; the
defect is only the read site. **FIX: capture Δi_rel from the truth decode (or the
getter) at the top of `_nav_step`, before `_encode`, alongside the existing
truth-side conditioning variables.**

---

## NON-ISSUE-18 (surface 2) — The estimated-plane gauge is safe. Both halves of the concern are refuted, with a closed form.

**Attack.** The truth-side shaping (`compute_phi` mode 2, `orbital.h:1282-1312`)
builds the gauge from the **truth** target; the policy would observe obs[13,14]
built from the **estimated** target plane. Does that (a) bias the policy near plane
convergence, or (b) blow up as estimated `ĥ ≈ target ĥ`?

**Measurement (`rt3_gaugeproper.py`, C algebra ported verbatim including both
branch tests, 200 draws per cell; the estimate is perturbed by a pure cross-track
rotation about `r̂` — the plane-error mode — so the in-plane part is isolated out):**

| Δi_rel | plane err | median \|Δ(Δλ)\| | gain (deg/deg) | tan(Δi_rel/2) |
|---|---|---|---|---|
| 0.0075° | 1e-3° | 4.66e-8° | **0.0000** | 0.00007 |
| 0.0417° | 1e-3° | 2.65e-7° | 0.0003 | 0.00036 |
| 0.25° | 1e-3° | 1.57e-6° | 0.0016 | 0.00218 |
| **1.0°** | 1e-3° | 6.07e-6° | **0.0061** | 0.00873 |

This reproduces N3D-C §3's "≤ 0.006 deg/deg per 1° of plane tilt" independently, and
supplies the mechanism they did not: the closed form is

```
d(Δλ)/dε  =  tan(Δi_rel / 2) · sin(ψ − Ω)
```

**(a) No convergence bias — the opposite.** The gain is proportional to
`tan(Δi_rel/2)`, so it *vanishes* exactly as guidance nulls the plane. The regime
where the policy is most sensitive is the regime where the gauge error is smallest.

**(b) No singularity at `ĥ_est ≈ ĥ_t`.** The `1/sin i` singularity is real but it
lives in `Ω` and `ω` *separately* (`δΩ = ε sin(ψ−Ω)/sin i`) and cancels identically
in `ϖ = ω + Ω`, which is the only combination `orb_varpi_gauge` exposes
(`orbital.h:771`). The gauge's own ill-conditioned axis is `e1` — the estimated
node on the ECI equator, whose azimuth is essentially random when `i_t,est ~ 1e-8` —
but a rotation about `ĥ_t` shifts *both* bodies' `ϖ_gauge` by the same amount and
cancels in the difference, **provided the target estimate's ω is derived from the
same node**, which it is when the estimate is decoded from Cartesian rather than
carried as elements. Verified numerically: the gauge branch change (identity →
round-trip) is continuous, at 4.7e-8° for a 1e-3° plane error.

**Two caveats to carry, or this result will be over-claimed:**

1. A *general* (not pure cross-track) estimate error moves Δλ at gain ≈ **0.5**.
   That is ordinary along-track target-position error, present identically in 2D,
   **not** a gauge effect. Do not let it be attributed to the gauge.
2. The plane-estimate error does not vanish from the observation — it lands on
   **obs[21,22] at gain ≈0.7 : 1** (N3D-C §1, and the mechanism is the same δı⃗
   projection I confirm in MAJOR-5). obs[13,14] is immune *because it is a
   difference*; obs[21,22] is not a difference and is the fragile pair.

---

## NON-ISSUE-19 (surface 3) — `el ≈ 0` is not a degeneracy. The 3D surrogate-vs-real divergence risks are elsewhere.

`R = diag(σ_β²/cos²el, σ_β²)` is regular at `el = 0` (`cos el = 1`); the singularity
is at ±90° and is BLOCKER-1's problem. N3D-B §3.4 measured that at `el ≡ 0` the
out-of-plane channel is still estimable — it is the *in-plane angle alone* that is
rank-deficient, which is why "the second angle is not optional in 3D; it is what
makes the 6-state estimable at all." And the 2D-reduction anchor positively
*requires* `el ≡ 0`.

The real 3D surrogate-vs-real divergence surfaces, ranked:
1. the `1/cos²el` R-inflation (MAJOR-12) — now material because of BLOCKER-1;
2. the pole and the re-pole (BLOCKER-1);
3. the rank-1 structure being lost to table mode or the `_chol` fallback (MAJOR-11);
4. inherited from 2D and *not* to be fixed mid-campaign: the gate projects the
   covariance on `u0`, the **epoch** LOS, and compares against `sep0`, the **epoch**
   separation, while ρ can change by orders of magnitude over the arc
   (`nav_surrogate.py:330,353`). State it; changing it would break comparability
   with NB1.

---

## NOTE-20 — `obs[11,12,16]` are safe at the shipped config and break under two existing kwargs.

`omega_tgt = (e_tgt > 0.0) ? U(0,2π) : 0.0` (`orbital.h:1688-1690`), so at
`e_max_target = 0` the target's ω is exactly 0 and the estimate-side
`ϖ̂_t = atan2(ê_ty, ê_tx) = 0` matches — N3D-C §5's rung-1 recon gate is achievable,
and the degeneracy N3D-C §1 warns about does not fire. Two exceptions:

- `omega_offset_fixed > -10` sets `target.omega = sat.omega + offset` **even at
  `e_t = 0`** (`orbital.h:1738-1741`), at which point obs[11,12,16] carry a value no
  estimate can recover and the recon gate fails for a reason that has nothing to do
  with the port;
- at `raan_target_rad ≠ 0`, `target.omega` is node-referenced while `atan2(ê)` is
  inertial.

Both are eval-only hooks today. **Assert `omega_offset_fixed <= -10` and
`raan_target_rad == 0` in the nav wrapper under `dim3_mode`**, rather than
discovering it in a wide rung.

---

## NOTE-21 — `obs[28]` is the one channel whose failure mode looks plausible.

`obs[28] = (dv_rem − dv_pl − dv_in)/dv_ref` (`orbital.h:1145`) mixes chaser truth
with `â_t` under a 4-decade blind range prior. It is the only slot where a diverged
estimate produces a *confidently wrong feasibility margin* rather than an obviously
broken number. Ship N3D-C §7's diagnostic: fraction of decisions with
`obs[28] < 0` while unacquired. It is cheap and nothing else covers it.

---

## NOTE-22 — Calibration in the plane channel is unmeasured, and it is the stated gating dependency.

N3D-B §4.4 is explicit: N3D-A produced timing (`n3d_filter6_cost.csv`) and
propagation fidelity (`n3d_prop6_validation.csv`), **not** 3D settled-error
decomposed by channel. The prediction `ratio_plane ≈ 1.0–1.5` (against 2.9–5.5 for
range) is falsifiable and cheap, and a measured `> 2` means a filter defect, not an
information limit. Require the **(range, in-plane, plane)** triad over ≥6 geometry
classes × ≥24 noise seeds, **including normal-burn cells** — 3d_E §5's coverage
ablation showed that dropping normal-burn policies made two seeded bug classes
undetectable. A scalar RMSE makes the new channel unmeasurable, which is the exact
failure this campaign exists to avoid.

---

## NOTE-23 — Two numbers to carry into the write-up unaltered.

1. **Do not quote 2.72× as an information result.** N3D-A §4.2 flags it itself: the
   CRLB gain at that cell is 1.21×, so ~2.2× of the 2.72× is acquisition/filter
   behaviour.
2. **N3D-A and N3D-B do not contradict each other on the plane channel**, despite
   the headlines reading that way ("not an independent channel" vs "the plane
   channel is free"). Both are true simultaneously: the plane is *never binding*
   (B: worst case 0.0038 of the position box) **and** *not independent of range*
   (A: `corr → 1.000` wherever Δi_rel > 0). An implementer who reads only one lane
   will build the wrong gate. See MAJOR-11 and MAJOR-13.

---

## Consolidated disposition

| # | severity | finding | change required |
|---|---|---|---|
| 1 | **BLOCKER** | MSC chart pole specified two ways; N3D-C §3b's is at 90.00° el (gimbal lock) in the coplanar anchor arm, N3D-A's hits 87.58° at Δi=1°/ρ_inplane=5 km — guard fires inside the T3 success box | Adopt `ĥ_c(t₀)`; delete N3D-C §3b; re-pole at \|el\|>45°; validate the similarity transform (round-trip 1e-12 + NEES continuity + per-episode counter) at Δi=1°, ρ_inplane<68 km |
| 2 | **BLOCKER** | Acquisition floor is a tick COUNT; every cost lever divides ticks. K=12 ⇒ 0.75 h floor becomes 9.71 h against a 22.2 h episode | Floor becomes sim-TIME (45 min) + the σ_LOS/ρ gate; identical cadence at train and eval; pre-register latency in sim seconds, never decisions (3D packs 2.45× more sim-time per decision) |
| 3 | **BLOCKER** | Cost anchored on the range+bearing arm, not bearings-only. Measured 2D tick 0.904/1.310 ms at B=256; 5.51× in 3D ⇒ **5.5–8.0 h per 50M rung**, 28–40 h for the plan | Analytic 3D STM first (FD-STM is 87% of the tick, shared by filter and surrogate, ~3–4× for zero model change); budget from the converged τ mix (τ̄ rises 22.0→79.61 during a run) |
| 4 | **BLOCKER** | 19-slot rebuild has 7 new leak channels (21,22,23,24,25,**26**,28); obs[26] is the true plane-change Δv-to-go; no shipped diagnostic would notice | `leakcheck` gate: garbage the estimate, assert all 19 slots move and the complement does not. In CI, every rung, before training |
| 5 | MAJOR | Ω_s sufficient at i_t=0 (verified 1.06e-15; `obs[21,22] ≡ (i_s cos u, −i_s sin u)/di_scale`), insufficient at i_t≠0 (30% of draws admit ≥2 inclination roots) | Getter returns full element sets; use i_s and Ω_s from it; obs-only decode stays a gate, not a dependency |
| 6 | MAJOR | Getter mechanics; `-O2 -flto` makes anchor re-runs mandatory | Out-array pattern (mirror `vec_get_trajectory`); re-run A2 0/200k, legacy 26/200, T3 canonical md5 replays after any binding.c or toolchain change |
| 7 | MAJOR | `nav_max_ticks` desyncs filter from truth on 26.0% of Discrete-30 decisions (5 h jump as one 60 s innovation) | `nav_max_ticks=0`, or fixed count + adaptive interval — **only with BLOCKER-2's floor fix** |
| 8 | MAJOR | `ACTION_TAU`/`ACTION_DV_MAG` length 20 vs `NUM_ACTIONS 30` — `IndexError` on step 1 | Extend to 30; `ACTION_DV_MAG[26..29] = 35.355` not 50 (1.41× error on 23.6% of burn decisions otherwise) |
| 9 | MAJOR | Every 2-component slice is a 3D bug; `_guard` reads `y[:,3]` which becomes `ω_e` under the 6-state ⇒ filter reinitialises every step, silently, forever | Named indices (`IDX_LNRHO`); unit test that `_guard` flags 0 rows on a healthy batch; 3-D ρ in `_apply_action_ablation` |
| 10 | MAJOR | `_FINE` omits normal ±1 (actions 20,21) — the 3D observability treatment — so `T-BO−act` is uninterpretable | Split `_FINE_INPLANE`/`_FINE_NORMAL`/`_NORMAL_ALL`/`_COMBINED`; `T-BO−normal` needs a truth control at the same box |
| 11 | MAJOR | Plane as a separate scalar destroys `corr→1.000`; table mode's velocity block is already diagonal; `_chol` falls back to diagonal with no counter | Enforce `crlb_online` under dim3; full 6×6 rank-1 Σ if table ships; count `_chol` fallbacks and gate on 0. **No plane term in any gate** (certifies garbage at Δi=0; double-counts at Δi>0) |
| 12 | MAJOR | `H` must be basis-free `P_⊥`; isotropic-R az/el inflates information 1.26×@27°, 4×@60° — material because of BLOCKER-1 | `M = P_⊥/(σ_β ρ)²` + equivalence test at el ∈ {0,27,45,60}°; nondimensionalise by `D`; guard `cond > 1e14` on the **scaled** FIM |
| 13 | MAJOR | Two lanes both say "σ_plane": N3D-A means tilt (deg), N3D-B means position (m). obs[21,22] is fed by the first, the box is stated in the second | Rename to `sigma_plane_tilt_rad` / `sigma_plane_pos_m`; assert the `r·sin u` relation in a test |
| 14 | MAJOR | `nav_clip=4.0` is uniform; C clamps 21-28 at ±2 only | Slot-dependent clamp. Layout: 21-28 rebuilt+clamped, 29-32 stay 0.0, **Σ-channel dropped** (NAV-F: T-BO+Σ ≡ T-BO). 21-28 are *trained* in the 3D warm-start; only 29-32 are random-init |
| 15 | MAJOR | With the getter, `recon` no longer exercises the Cartesian→element path `bearings_only` flies | Add `recon-cart`: truth target round-tripped through Cartesian, 19-slot agreement to a measured stated tolerance |
| 16 | MAJOR | Out-of-plane velocity seed 154 m/s vs 134 m/s of ignorance — 1.15× margin, over-confident at di_max ≥ 2° | Seed the out-of-plane rate from `v_c·sin(di_max)/ρ₀`, decoupled from `_sigma_v_ecc` |
| 17 | MAJOR | Δi_rel conditioning read from obs[21,22] **after** `_encode` is a self-referential loop | Capture Δi_rel from the truth decode at the top of `_nav_step` |
| 18 | NON-ISSUE | Estimated-plane gauge | Gain = `tan(Δi_rel/2)·sin(ψ−Ω)`, measured 0.0061 deg/deg at Δi=1°, 0.0000 at 0.0075°. No convergence bias (it vanishes there), no `ĥ_est ≈ ĥ_t` singularity (it cancels in ϖ). Carry the two caveats |
| 19 | NON-ISSUE | `el ≈ 0` degeneracy | Regular; the 2D-reduction anchor requires it. Divergence risk is the R-inflation, the pole, and the rank-1 structure |
| 20 | NOTE | obs[11,12,16] break under `omega_offset_fixed` / `raan_target_rad` | Assert both off in the nav wrapper under dim3 |
| 21 | NOTE | obs[28] is the only "plausibly wrong" channel | Ship the `obs[28] < 0 while unacquired` diagnostic |
| 22 | NOTE | 3D CRLB→filter plane calibration unmeasured; it is the stated gating dependency | (range, in-plane, plane) triad, ≥6 classes × ≥24 seeds, normal-burn cells non-negotiable |
| 23 | NOTE | Write-up hygiene | 2.72× is not an information result (CRLB gain 1.21×); N3D-A and N3D-B agree about the plane despite the headlines |

**4 BLOCKER, 13 MAJOR, 2 NON-ISSUE, 4 NOTE.**

**Implementation may start** on the C getter (MAJOR-5/6 shape) and the 19-slot
encode/decode layer, because BLOCKER-4's `leakcheck` gate and MAJOR-15's
`recon-cart` gate bound that work. **No training run may launch** until BLOCKER-1
(pole decided, re-pole validated), BLOCKER-2 (floor converted to sim-time, cadence
matched train/eval), BLOCKER-3 (analytic STM or an accepted 6–8 h/rung schedule)
and BLOCKER-4 (leakcheck green) are closed. MAJOR-8 and MAJOR-9 are crash/silent-
failure class and must land in the first commit that touches `nav_math`.
