# 3D-A — Element-set & conversion design memo (ext-3d)

**Recommendation: extend classical in the struct; go singularity-free in the *consumers*.** Not equinoctial. Evidence below is measured, not asserted; probes at `/Users/pete/space_training/scripts/orbital/ext_recon/ext_3d_conv_probe.c` (C, 5 flag settings) and `/Users/pete/space_training/scripts/orbital/ext_recon/ext_3d_budget_probe.py` → `/Users/pete/space_training/web_data/results/ext_3d_budget.csv`.

---

## 1. Element-set decision

### 1.1 The measurement that settles it

The question "classical is singular at e≈0 and i≈0, both in-distribution" conflates two different things: singular **representation** and singular **information**. Probe P6 separates them — element error induced by a 1 m / 1 mm/s Cartesian perturbation (= the float32 traj log, the EKF estimate, obs quantization), median over 20k draws, e ~ U[0.01, 0.30]:

| inc [rad] | med \|δΩ\| | med \|δω\| | med \|δϖ\| | med \|δλ\| | med \|δŵ\| |
|---|---|---|---|---|---|
| 5e-1 | 1.35e-7 | 7.52e-7 | 7.36e-7 | 1.34e-7 | 1.11e-7 |
| 1e-2 | 6.78e-6 | 6.97e-6 | 7.32e-7 | 1.34e-7 | 1.12e-7 |
| 1e-4 | 6.68e-4 | 6.68e-4 | 7.36e-7 | 1.34e-7 | 1.12e-7 |
| 1e-6 | **6.75e-2** | **6.75e-2** | 7.38e-7 | 1.33e-7 | 1.12e-7 |

Ω and ω individually degrade as 1/sin i — at i = 1e-6 rad, one metre of position error moves Ω by **3.9°**. The combinations ϖ = Ω+ω, λ = M+ω+Ω, and ŵ = ĥ are **flat across seven decades of inclination**. That is exactly the property equinoctial elements are bought for — and it is already available inside the classical set, for free, by never reading Ω or ω alone. (Note the exact structural parallel to the fix this project already made in 2D: λ = M+ω replaced true anomaly for precisely this reason.)

A pure round-trip (P2, no perturbation) recovers Ω to 8.9e-16 rad even at i = 1e-12, because ĥ's direction is preserved to relative precision. So the classical inverse map is not *numerically* broken at small i; only its Ω/ω *outputs* are physically meaningless there. That is a consumer discipline problem, not a representation problem.

### 1.2 Weighed comparison

| | A. classical + (i, Ω) | B. equinoctial (p,f,g,h,k,L) | **C. RECOMMENDED: A + combination-only consumers** |
|---|---|---|---|
| `propagate_orbit` | **untouched** (i,Ω,ω are constants of two-body motion) | **replaced** — generalized Kepler λ = K + g cos K − f sin K, new Newton solve | untouched |
| conversion bug surface | 4-branch table + quadrant rules | branchless, but 2 competing sign conventions (retrograde factor I=±1) and a p,q basis nobody here has validated | 4-branch table, but branch choice is provably invisible downstream (§5, test A6) |
| conditioning at i≈0, e≈0 | Ω, ω garbage; ϖ, λ, ŵ clean (P6) | uniformly clean | uniformly clean **for every quantity that reaches obs/shaping** |
| 2D anchor bit-exactness | **achieved, 0/200000, all 4 flag settings** (P1/P1b/P4) | impossible — different Kepler algebra | achieved |
| downstream blast radius | 2 new struct fields; consumers edited in place | rewrites every consumer of `o->a/e/omega/theta/M`: `fill_observations`, `compute_phi`, `check_termination`, `c_reset`, `write_traj_record`, `preview_perigee`, `orbital.c`, plus 5 Python ports (`nav/orbital_math.py`, `nav/ekf.py`, `nav/eval_relnav.py`, `t3/orb_model.py`, `t3/expert_controller.py`, `t3/fuzz_dynamics.py`, `t3/redteam/rt_common.py`) | same as A |
| est. LOC | ~300 net-new / ~60 modified | ~800+, no legacy anchor | ~300 / ~60 |
| checkpoint lineage | 2D lineage bit-preserved; 3D lineage warm-startable from a 2D ckpt (§6) | full lineage break, no regression anchor survives | same as A |
| residual risk | ill-conditioned Ω/ω leak into a consumer | brand-new propagator in the hottest, most-previously-bugged code path | leak is caught by a mechanical test (A6) |

**The decisive asymmetry:** this project has been killed twice by a single inverted anomaly conversion. Option B *replaces the propagator* — the one function that survived both bugs untouched and is validated at the ±1-ULP level by 618k sub-steps of oracle fuzzing. Option C keeps it byte-identical and confines all new risk to `cartesian_to_elements`, which is fuzzable against the sibling agent's independent universal-variable oracle (`scripts/orbital/ext_recon/orbital_math3d.py` — deliberately different algebra). Take C.

The sibling recon's mutation-test harness (`ext_invariants3d.py`) already models the target architecture as "classical elements + mean-anomaly propagation + `cartesian_to_elements` after each burn". Consistent.

### 1.3 Exact struct delta

```c
/* orbital.h — Orbit gains TWO doubles. Zero-init (Orbital env = {0}) ⇒ legacy. */
typedef struct {
    double a;       /* semi-major axis (m)                                  UNCHANGED */
    double e;       /* eccentricity                                          UNCHANGED */
    double M;       /* mean anomaly (rad)                                    UNCHANGED */
    double theta;   /* true anomaly (rad)                                    UNCHANGED */
    double omega;   /* argument of periapsis (rad), FROM THE ASCENDING NODE.
                     * Identical to the legacy meaning whenever raan == 0.   UNCHANGED */
    double inc;     /* NEW: inclination (rad), 0 = equatorial prograde       */
    double raan;    /* NEW: RAAN Ω (rad), 0 when equatorial                  */
} Orbit;
```

Derived quantities — **the only forms any consumer may use**:

```c
static inline double orb_varpi (const Orbit* o) { return o->omega + o->raan; }          /* ϖ */
static inline double orb_lambda(const Orbit* o) { return o->M + o->omega + o->raan; }   /* λ */
static inline void   orb_ivec  (const Orbit* o, double* ix, double* iy) {               /* ī */
    double si = sin(o->inc); *ix = si*cos(o->raan); *iy = si*sin(o->raan); }
static inline void   orb_what  (const Orbit* o, double* wx, double* wy, double* wz) {   /* ŵ */
    double si = sin(o->inc); *wx = si*sin(o->raan); *wy = -si*cos(o->raan); *wz = cos(o->inc); }
```
Note `ī = (−ŵ_y, +ŵ_x)` exactly — the inclination vector the prompt guessed is a relabeling of the unit angular-momentum vector, and both are computable straight from `h̄` with no trig and no branch. `tan(i/2)` is *not* needed: our i range is bounded well away from π, and `sin i` keeps the exact identity Δv_plane = v·|Δŵ| (§3.4).

Memory: `Body` +16 B ×16 = +256 B; `Orbital` +~64 B. `traj_log` (12000 × 352 B) still dominates. No serialization of `Orbit` anywhere — checked.

---

## 2. Exact formulas the C code needs

### 2.1 Perifocal → ECI (313 rotation)

R = R₃(−Ω)·R₁(−i)·R₃(−ω), applied to the **unchanged** perifocal block:

```
R11 =  cosΩ cosω − sinΩ sinω cos i     R12 = −cosΩ sinω − sinΩ cosω cos i
R21 =  sinΩ cosω + cosΩ sinω cos i     R22 = −sinΩ sinω + cosΩ cosω cos i
R31 =  sinω sin i                      R32 =  cosω sin i
(x,y,z) = R·(xp,yp,0);   (vx,vy,vz) = R·(vxp,vyp,0)
```
At i=Ω=0 this collapses to exactly `co*xp − so*yp`, `so*xp + co*yp`, z=0 — **measured bit-exact vs the legacy two lines, 0/200000 mismatches at `-O0`, `-O2 -flto`, `-O3 -ffp-contract=fast`, `-O2 -ffp-contract=off`** (P1). Still ship the value-gated fast path (`if (o->inc == 0.0 && o->raan == 0.0) { …legacy statements…; *z=0; *vz=0; return; }`): it makes the anchor *structural* rather than empirical-on-one-toolchain, and it is free speed for the 2D lineage.

### 2.2 3D `cartesian_to_elements`

```c
r  = sqrt(x*x + y*y + z*z);          /* z*z == 0.0 ⇒ bit-identical to legacy */
v2 = vx*vx + vy*vy + vz*vz;
rv = x*vx + y*vy + z*vz;   vr = rv / r;
a  = 1.0 / (2.0/r - v2/MU);                                   /* unchanged */

hx = y*vz - z*vy;  hy = z*vx - x*vz;  hz = x*vy - y*vx;
hxy  = sqrt(hx*hx + hy*hy);           hmag = sqrt(hx*hx + hy*hy + hz*hz);
inc  = atan2(hxy, hz);                                  /* NOT acos(hz/h) — trap T1 */

/* KEEP THE LEGACY SPELLING: (v2 - MU/r)*x - vr*r*vx, NOT ... - rv*vx  (trap T8) */
ex = ((v2 - MU/r)*x - vr*r*vx)/MU;   ey = …*y - vr*r*vy;   ez = …*z - vr*r*vz;
e  = sqrt(ex*ex + ey*ey + ez*ez);

if (hxy == 0.0) {                       /* EQUATORIAL — legacy statements verbatim */
    raan = 0.0;
    if (e < 1e-10) { omega = 0.0; theta = atan2(y, x); }
    else { omega = atan2(ey, ex);
           ct = clamp((ex*x + ey*y)/(e*r), -1, 1);
           theta = acos(ct); if (vr < 0.0) theta = 2π - theta; }
} else {                                /* INCLINED */
    raan = atan2(hx, -hy);  if (raan < 0) raan += 2π;    /* n̄ = ẑ×h̄ = (−hy, hx, 0) */
    nx = -hy/hxy;  ny = hx/hxy;                          /* n̂, nz = 0 */
    wx = hx/hmag;  wy = hy/hmag;  wz = hz/hmag;          /* ŵ */
    mx = -wz*ny;   my = wz*nx;    mz = wx*ny - wy*nx;    /* m̂ = ŵ × n̂ */
    if (e < 1e-10) { omega = 0.0;
                     theta = atan2(x*mx+y*my+z*mz, x*nx+y*ny); }   /* arg. of latitude */
    else { omega = atan2(ex*mx+ey*my+ez*mz, ex*nx+ey*ny); if (omega<0) omega += 2π;
           eu = ē/e;  q̂ = ŵ × eu;
           theta = atan2(x*qx+y*qy+z*qz, x*eux+y*euy+z*euz); if (theta<0) theta += 2π; }
}
M = true_to_mean(theta, e);  if (M < 0) M += 2π;                     /* UNCHANGED */
```

`hxy == 0.0` is an exact test, not a tolerance: with z = vz = 0, `hx = y*0 − 0*vy = 0` and `hy = 0*vx − x*0 = 0` identically, so the equatorial branch fires bit-deterministically — this is what closes the 2D invariant (§5).

### 2.3 Burn frame (3D)

```
prograde  p̂ = v̄/|v̄|
radial    r̂ = r̄/|r̄|                    (NOT orthogonal to p̂ at e>0 — pre-existing, keep)
normal    n̂ = h̄/|h̄| = (r̄×v̄)/|r̄×v̄|     (+normal = +ĥ = orbit north)
Δv̄ = dv_pro·p̂ + dv_rad·r̂ + dv_nor·n̂      ← in-plane terms summed FIRST, normal last
```
Fuel/Tsiolkovsky block, clamp logic, `total_dv_used`: **unchanged**.

Analytic geometry checks (P3, 20k draws, +10 m/s normal from an equatorial orbit):
- new Ω = inertial longitude of the burn point: max error **4.44e-16 rad**
- new i = atan(dv_n/v_t): max error **1.62e-6 rad** (that residual is the e ≤ 0.05 model error, not code error)

### 2.4 Functions that survive **unchanged**

`solve_kepler`, `eccentric_to_true`, `true_to_mean`, **`propagate_orbit`** (i, Ω, ω are two-body constants — no code touches them), the Tsiolkovsky block inside `apply_impulse`, `add_log`, `wrap_pi`, the fuel-clamp path, `c_render` (a z-projection is an acceptable 2D view).

Changed-but-mechanical: `preview_perigee` (3-component), `body_position`, `write_traj_record`, `c_close`.

### 2.5 λ-continuity through a normal burn (P7, e ~ U[0.01,0.30], i₀ ~ U[0,0.1], dv_n = 10 m/s)

| quantity | median | p99 |
|---|---|---|
| \|Δλ\| (M+ω+Ω) | 1.67e-5 rad | 6.52e-5 rad (= 0.0037°) |
| \|Δϖ\| (ω+Ω) | 2.00e-5 rad | 1.08e-4 rad |
| \|Δu\| (θ+ω+Ω) | 1.67e-5 rad | 6.52e-5 rad |

The T3 red-team cleared in-plane λ-jump farming at a worst case of 0.167°/burn. Normal burns are **45× smaller** — **normal burns cannot farm the phase potential.** Log this number in the red-team ledger before training.

Caveat measured in P3: at e → 0, ϖ is ill-conditioned (worst \|Δϖ\| = 0.31 rad when e ~ U(0,0.05) includes e ≈ 0). That is the *pre-existing* e≈0 singularity, unchanged by 3D; λ is unaffected. Consumers must prefer λ over ϖ wherever both work.

---

## 3. Consumer changes (this is where the design actually lives)

**Invariant (enforce by review + test A6): no obs channel, shaping term, sampling constraint, or termination test may read `o->omega` or `o->raan` alone. Only `orb_lambda`, `orb_varpi`, the 3-vector ē, and ŵ / ī.**

### 3.1 Observation vector — zero growth

Measured against the live build (`num_debris_min=max=0`, T3 config, 400 steps × 8 envs): obs slots **[21..32] are identically zero — 12 free floats.** obs[17..20] is the live Earth block (do not touch). Proposed 3D block, written only when `enable_3d=1`:

| slot | channel | normalizer |
|---|---|---|
| 21, 22 | δī = ī_s − ī_t (relative inclination vector) | `di_obs_scale` (default 0.05 rad) |
| 23 | \|Δŵ\| (exact plane-change Δv / v) | `di_obs_scale` |
| 24, 25 | sin, cos of u_rel = angle from r̂_s to the relative node n̂_rel ∝ ŵ_s × ŵ_t | — |
| 26, 27 | LVLH cross-track z, ż (completes obs[33-36]) | `lvlh_scale_m`, v_circ_t |
| 28, 29 | ī_t (absolute target plane — context for J2/future) | `di_obs_scale` |
| 30–32 | reserved 0.0 | — |

`OBS_DIM` stays 38; the action-mask arithmetic (a known landmine — writing past obs[37] with the mask disabled corrupts the neighbouring env's slice) is untouched. Alternative (append at obs[38+], grow OBS_DIM) is rejected: it moves the mask block and costs the warm-start property in §6 for no benefit.

Also required under `enable_3d=1`:
- obs[9-12]: sin/cos ω_s, ω_t → **sin/cos ϖ_s, ϖ_t** (raw ω is a 3.9°-per-metre noise channel at small i — P6).
- obs[16]: cos(ω_s−ω_t) → **cos(ϖ_s−ϖ_t)**.
- obs[13,14]: Δλ must be Δ(M+ω+Ω).
- obs[33-37]: LVLH triad must be built from r̂_t and ŵ_t, not from the 2D angle `θ_t + ω_t`; the frame-rotation term must use ω̄ = h̄_t/r_t² rather than the mean motion n (the circular approximation is pre-existing at e>0 — do **not** silently fix it in the 2D path).

### 3.2 Shaping potential

```
Φ = −[ W_λ·|Δλ|/π + W_m·min(1, Δv_match/DV_REF) ]
Δv_match = hypot( ½·v_t·hypot(Δa/a_t, |Δē₃|) , v_t·|Δŵ| )
```
`Δē₃` = full 3-vector eccentricity-vector difference (**not** the 2D (e cosω, e sinω) pair — that projects two orbits into mismatched perifocal frames once the planes differ: trap T12). One combined term, not a separate weight: a Δv metric is a Δv metric, so the per-burn shaping gradient of a 10 m/s normal burn automatically equals that of a 10 m/s prograde burn (both move Δv_match by 10 m/s; verified analytically — da = 2a²v·dv/μ = 17.65 km at LEO ⇒ ½·v·Δa/a = 10.0 m/s).

**Saturation is the live risk.** `min(1,·)` at DV_REF=300 was already 1.8% saturated at L2 and 80.7% at L3. Adding a plane term in quadrature raises it. Gate per rung: measure the saturation fraction over 10k resets, require < 5%, else raise `shape_dv_ref_ms` (500 for di_max ≥ 0.5°, 700 for ≥ 1.5°). Keep `shape_w_plane` / `shape_dv_ref_plane_ms` as a documented split-term ablation arm if a rung stalls.

### 3.3 Termination — no new kwargs, one lethal trap

The success box is already a Cartesian position/velocity test, so it generalizes to 3D by construction — **provided every distance picks up z**. Omitting z anywhere turns the 30 km sphere into a 30 km *cylinder* and awards success to trajectories kilometres out of plane. This is the single most dangerous omission in the whole extension (trap T17). Affected: rendezvous distance, `rel_vel`, collision loop, `min_conj_dist`, escape energy v².

### 3.4 Plane-change economics — the number that shapes the curriculum

Δv_plane = 2v·sin(α/2) = v·|Δŵ| exactly. Budget = −V_e·ln(1−0.15) = **478.1 m/s**.

| envelope | v_c [m/s] | Δi at *full* budget | Δi at residual (478 − 189 expert in-plane) | Δv @ 1° |
|---|---|---|---|---|
| LEO 400 km | 7673 | 3.571° | **2.159°** | 133.9 |
| LEO 800 km | 7456 | 3.675° | 2.222° | 130.1 |
| WL4 4000 km | 6200 | 4.420° | 2.672° | 108.2 |
| WL4 8000 km | 5267 | 5.203° | 3.146° | 91.9 |
| M5 MEO 20,200 km | 3873 | 7.077° | **4.278°** | 67.6 |

**The whole fuel budget buys 3.6° of LEO plane change.** Any "3D" claim is a *few-degree* claim. Say so up front; it is a physics fact, not a limitation of the method (and it is exactly why real missions launch into plane).

Apogee discount (burn at apoapsis, v_apo/v_c = √((1−e)/(1+e))): 1° costs 132.5 m/s at e=0, 97.2 at e=0.30, **76.5 at e=0.50 (−42%)**. At the wide/MEO envelopes this makes "raise apoapsis → plane-change at apoapsis → circularize" a genuinely optimal emergent strategy, which is a strong demo result.

Timing: LEO period 95.5 min = 96 sub-steps; relative-node crossings every 48 sub-steps; the 3000-sub-step cap gives **63 node crossings** — ample.

---

## 4. Runtime kwargs and the `c_reset` sampling design

All default to legacy. Mirrors the existing `de_max`/`e_*_fixed` structure exactly.

| kwarg | default | meaning |
|---|---|---|
| `enable_3d` | 0 | master gate: 3D obs block, 3D shaping term, 3D sampling, 3D LVLH. 0 ⇒ every value gate takes the legacy path |
| `i_max_target` | 0.0 | target inclination ~ U(0, i_max_target) rad (absolute; free — costs nothing) |
| `i_target_fixed` | −1.0 | ≥ 0 ⇒ exact target i (surface-eval cell) |
| `raan_target_fixed` | −99.0 | > −10 ⇒ exact target Ω |
| **`di_max`** | −1.0 (off) | **the de_max analogue: ī_sat = ī_target + area-uniform disc(di_max)** — bounds the *relative* inclination vector. Off ⇒ chaser plane = target plane exactly |
| `di_fixed`, `di_phase_fixed` | −1.0, −99.0 | exact \|δī\| and its node phase, for per-condition evals |
| `di_obs_scale` | 0.05 | obs normalizer for slots 21-23, 28-29 |
| `shape_w_plane`, `shape_dv_ref_plane_ms` | 0.0, 300 | split-term ablation arm only |
| `legacy_action_space` | (existing) | raise the validated upper bound **20 → 26** |

**Why `di_max` and not `i_max_sat` (B2, 200k draws, LEO v):** with i_s, i_t ~ U(0, i_max) and *independent* Ω — i.e. the naive analogue of the retired `e_max_sat` —

| i_max | E\|δī\| | p50 Δv | p90 Δv | frac > budget | frac > residual |
|---|---|---|---|---|---|
| 0.5° | 0.0063 | 46.2 | 82.3 | 0.000 | 0.000 |
| 1.0° | 0.0126 | 92.2 | 164.2 | 0.000 | 0.000 |
| 2.0° | 0.0253 | 184.3 | 329.4 | 0.003 | **0.167** |
| 5.0° | 0.0632 | 461.5 | 821.7 | **0.475** | 0.752 |
| 51.6° (ISS) | 0.623 | 4623 | 7941 | 0.989 | 0.995 |

This is the `de_max` lesson verbatim: an "i_max = 5°" curriculum is 47.5% *unconditionally infeasible*. Bound the relative vector instead (B3, area-uniform disc, mirrors the `de_max` code):

| di_max | p90 \|δi\| | p90 Δv | % budget | burns @10 m/s | @25 m/s | frac > residual |
|---|---|---|---|---|---|---|
| 0.05° | 0.0475° | 6.3 | 1.3% | 1 | 1 | 0.000 |
| 0.10° | 0.0949° | 12.6 | 2.6% | 2 | 1 | 0.000 |
| 0.25° | 0.2372° | 31.4 | 6.6% | 4 | 2 | 0.000 |
| 0.50° | 0.4743° | 62.8 | 13.1% | 7 | 3 | 0.000 |
| 1.00° | 0.9492° | 125.7 | 26.3% | 13 | 6 | 0.000 |
| 1.50° | 1.4221° | 188.4 | 39.4% | 19 | 8 | 0.000 |
| 2.00° | 1.8985° | 251.5 | 52.6% | 26 | 11 | 0.000 |
| 3.00° | 2.8470° | 377.1 | 78.9% | 38 | 16 | **0.472** |

**Proposed ladder: P0 di_max=0 (anchor) → P1 0.1° → P2 0.25° → P3 0.5° → P4 1.0° → P5 2.0°.** P5 is the honest ceiling at LEO; 3.0° is 47% infeasible. Absolute `i_max_target` can be widened freely and independently (it costs nothing) once di_max is fixed — that is what makes the demo look like "polar / ISS-inclination operations" without being infeasible.

### `c_reset` insertion (after the e/ω block, before the M/phase block)

```c
/* ── 3D: target plane ── */
double i_t = 0.0, O_t = 0.0;
if (env->enable_3d) {
    i_t = (env->i_target_fixed >= 0.0) ? env->i_target_fixed
                                       : urand() * env->i_max_target;
    O_t = (env->raan_target_fixed > -10.0) ? env->raan_target_fixed
        : ((i_t > 0.0) ? urand() * TWO_PI : 0.0);
}
env->target.inc = i_t;  env->target.raan = O_t;

/* ── 3D: chaser plane — bound the RELATIVE inclination VECTOR (de_max twin) ── */
env->sat.orbit.inc = i_t;  env->sat.orbit.raan = O_t;      /* di_max off ⇒ coplanar */
if (env->enable_3d && env->di_max >= 0.0 && !env->same_orbit_init) {
    double r_di  = (env->di_fixed >= 0.0) ? env->di_fixed
                                          : env->di_max * sqrt(urand());
    double ph_di = (env->di_phase_fixed > -10.0) ? env->di_phase_fixed : urand()*TWO_PI;
    double ix = sin(i_t)*cos(O_t) + r_di*cos(ph_di);
    double iy = sin(i_t)*sin(O_t) + r_di*sin(ph_di);
    double si = sqrt(ix*ix + iy*iy);  if (si > 1.0) si = 1.0;      /* clamp — trap T18 */
    env->sat.orbit.inc  = asin(si);
    env->sat.orbit.raan = (si > 1e-12) ? atan2(iy, ix) : 0.0;
}
```
`valid_init_only` is unaffected (perigee = a(1−e) is inclination-independent). **`phase_gap_mode == 1` must change** `tgt_M += sat.omega − target.omega` → `tgt_M += orb_varpi(&sat.orbit) − orb_varpi(&target)`, or the physical-gap knob silently reverts to the ANOM-4 "inert knob" failure (trap T10).

### Action space

Append **only** at indices 20-25 (the DO-NOT-RENUMBER rule): normal ±1, ±10, ±25 m/s (`{0,0,±1}`, `{0,0,±10}`, `{0,0,±25}`, τ=1). `NUM_ACTIONS` 20 → 26; `legacy_action_space` upper bound 20 → 26; exposed default stays 20 so every existing command and checkpoint is untouched. ±25 is required: closing 2° needs 26 burns at 10 m/s but 11 at 25 m/s.

---

## 5. The 2D-compat anchor

### 5.1 Closure argument (state it in the code comment; it is what makes the anchor provable)

1. With `enable_3d = 0`, `c_reset` writes `inc = raan = 0.0` exactly, for the chaser, the target and every body.
2. `propagate_orbit` never touches `inc`/`raan`.
3. `orbit_to_cartesian` value-gates on `inc == 0.0 && raan == 0.0` and executes the legacy statements verbatim, emitting `z = vz = 0.0`.
4. `apply_impulse` with `dv_nor == 0.0` and an equatorial orbit takes the legacy in-plane branch verbatim.
5. `cartesian_to_elements` receiving `z = vz = 0` computes `hx = hy = 0` **identically** (each is a difference of two exact zeros), so `hxy == 0.0` and the equatorial branch — which contains the legacy statements verbatim — fires deterministically, restoring `inc = raan = 0.0`.
6. Therefore the invariant "every `Orbit` is exactly equatorial" is closed under reset, propagation and burns ⇒ every value gate takes the legacy path ⇒ the trajectory is bit-identical.

Two spellings must be preserved verbatim inside the equatorial branch or the anchor drops from *bit-exact* to *float-noise*:
- e-vector as `((v2 − MU/r)*x − vr*r*vx)/MU`. Rewriting to `(r̄·v̄)v̄` costs **11175/200000 non-bit-exact**, worst \|ΔM\| = 3.24e-14 rad = 0.23 µm (P1b). Physically nothing; anchor-wise everything.
- θ via `acos` + `vr<0` sign. `atan2(r̄·q̂, r̄·ê)` is 4 orders more accurate near apsides (max err **1.01e-12 vs 1.07e-8 rad**, i.e. 7 µm vs 7.5 cm along-track at 7000 km — P5) — so use atan2 in the **inclined** branch, where no anchor constrains it, and keep acos in the equatorial branch.

### 5.2 Anchor-test spec

| id | test | pass criterion | evidence today |
|---|---|---|---|
| **A0** | debug assert in `c_step`: `enable_3d==0 ⇒ inc==0.0 && raan==0.0` for sat, target, all bodies | never fires | — |
| **A1** | unit, C, in `orbital.c`: forward + inverse, 200k draws, i=Ω=0, 3D vs legacy | **0 mismatches**, bitwise | ✅ 0/200000 at 4 flag settings (P1, P1b) |
| **A2** | chained: 200 eps × 400 {burn, propagate} 2D vs 3D@i=0 | 0/200 divergent, bitwise | ✅ 0/200 (P4) |
| **A3** | **env-level (the real anchor)**: fixed seed, fixed action sequence, 8 envs × 5000 steps, dump obs/rewards/terminals; new build `enable_3d=0` vs pre-change build | float32 arrays bitwise identical | to run |
| **A4** | checkpoint anchors re-run on the new build: legacy ckpt 26/200 bit-exact; T3 canonical 200/200 | unchanged | to run |
| **A5** | **3D-mode reduction**: `enable_3d=1, i_max_target=0, di_max=0` | obs[0-20],[33-37] float32-identical to `enable_3d=0`; slots 21-32 all zero; same success rate | to run |
| **A6** | **branch invariance**: sweep i through the `hxy==0` boundary and through 1e-8…1e-2; assert (a, e, λ, ϖ, ŵ) continuous while Ω, ω are permitted to jump | \|Δλ\|, \|Δϖ\| < 1e-9 rad across the boundary | partially — P6 shows the mechanism |
| **A7** | analytic geometry: normal burn ⇒ Ω = burn longitude; i = atan(dv/v_t); λ continuity | 4.4e-16 rad; 1.6e-6 rad; p99 6.5e-5 rad | ✅ (P3, P7) |
| **A8** | oracle fuzz at i>0: port `fuzz_dynamics.py`'s 13-check battery against `ext_recon/orbital_math3d.py` (universal-variable, independent algebra) | all checks at the float32 noise floor | sibling agent has the oracle + mutation-tested invariant battery |

A3 + A4 are non-negotiable gates before any 3D training starts. A5 is what licenses the warm-start in §6.

---

## 6. Checkpoint lineage — a free rung-0

Under the obs-repurposing design, a 2D T4 checkpoint is a **valid warm start for the 3D curriculum at di_max = 0**: obs[0-20] and [33-37] keep their semantics, obs[21-32] go from constant-zero to constant-zero (δī = 0 when coplanar), and the environment is rotation-invariant so an inclined-but-coplanar scenario is dynamically identical to the equatorial one. Expand the Discrete-20 head to Discrete-26 with near-zero-init logits for actions 20-25 (precedent: `scripts/orbital/expand_ckpt_actions_7_to_9.py`, and the Phase 4 R1 zero-pad warm-start). A5 is the falsification test for this claim — if the warm-started ckpt does not reproduce its 2D success rate at `enable_3d=1, di_max=0`, something in the 3D path is not reducing and the ladder must not start.

Caveat from the project's own history: warm-start + reward-reshape on a committed policy has failed repeatedly (Phase 4 R5, R3 variants). The plane term *adds* a potential component, which is a reshape. Run the ladder both ways at P1 (warm-started vs fresh) and pick on measurement, not preference.

---

## 7. Trap list (every quadrant/sign trap, with the test that catches it)

| # | trap | catches it |
|---|---|---|
| T1 | `i = acos(hz/h)` — precision loss at i≈0/π and domain error when \|hz/h\|>1 by rounding. Use `atan2(hxy, hz)` | A6 sweep; P2 recovers i to 6e-16 relative at i=1e-12 |
| T2 | `Ω = acos(nx/n)` + "if ny<0: 2π−Ω". Use `atan2(hx, −hy)`; never form n̄ | A7; unit test all four Ω quadrants |
| T3 | `ω = acos(n̂·ê)` + "if e_z<0: 2π−ω" — undefined at i→0, inverted for i>π/2 in several texts. Use `atan2(ē·m̂, ē·n̂)`, m̂ = ŵ×n̂ | unit test ω in 4 quadrants at i=1e-3 and i=2.0 |
| T4 | θ via acos + vr-sign loses 8 digits at apsides (1.07e-8 rad) | P5; keep acos only in the equatorial (anchor) branch |
| T5 | **equatorial branch is prograde-only**: `ω := atan2(ey,ex)` assumes hz>0; retrograde needs `atan2(−ey,ex)` and i=π | assert hz>0 under `enable_3d=0`; explicit retrograde unit test or documented exclusion |
| T6 | circular-**inclined** branch: must set ω=0 and θ := argument of latitude `atan2(r̄·m̂, r̄·n̂)`. Forgetting it leaves θ derived from a garbage ê | e=0, i=0.5 rad round-trip |
| T7 | circular-**equatorial**: θ := `atan2(y,x)` (legacy); both branches must set M by the same route | A1 |
| T8 | e-vector operation order `vr*r*v` vs `(r̄·v̄)v` | A1 (11175/200000 diff measured) |
| T9 | division guards: `hxy==0` (branch), `hmag`, `e`, `r`, `v_mag` | fuzz |
| T10 | `phase_gap_mode=1` still uses ω_s−ω_t ⇒ the physical-gap knob silently goes inert (the ANOM-4 failure that unstaged every e>0 curriculum) | measure realized Δλ vs requested gap over 10k resets, KS vs the request |
| T11 | obs[9-12]/obs[16] left on raw ω ⇒ a channel with 3.9° of noise per metre of state error at i=1e-6 | P6; A6 |
| T12 | `compute_phi` keeps the 2D (e cosω, e sinω) pair ⇒ Δē compared in mismatched perifocal frames once planes differ | shaping replica cross-check (the red-team's Python Φ replica, matched to 3.6e-9 — port it to 3D) |
| T13 | LVLH triad from the 2D angle θ_t+ω_t; frame-rotation term from mean motion n instead of h_t/r_t² | A5 (must be identical at di_max=0) + oracle cross-check at i>0 |
| T14 | normal sign convention: +normal must be +ĥ | A7 (Ω = burn longitude) |
| T15 | p̂ and r̂ are not orthogonal (pre-existing); don't combine axes in one action | design rule |
| T16 | escape energy `E = ½v² − μ/r` missing vz | fuzz check B/C |
| T17 | **any distance test missing z** ⇒ the 30 km success sphere becomes a cylinder and cross-track misses score as successes | dedicated test: place the chaser at (0,0,z) offset inside the in-plane box, assert *failure* |
| T18 | `asin(\|ī\|)` domain when the disc pushes \|ī\| > 1 | clamp + reset-distribution test |
| T19 | wrap conventions: ω,Ω ∈ [0,2π); Δλ via `wrap_pi` to [−π,π). Mixing them is how the ANOM-1 sign bug happened | replica cross-check |
| T20 | float32 traj log: `sat_z` at \|z\| ~ 1e5 m quantizes at ~8 mm — the fuzz noise model must scale per-column, not assume the x/y magnitude | A8 |

---

## 8. Function-by-function delta plan

**`pufferlib/pufferlib/ocean/orbital/orbital.h`** (~235 net-new / ~60 modified)
| function | delta |
|---|---|
| `Orbit` | +2 doubles (`inc`, `raan`) |
| `solve_kepler`, `eccentric_to_true`, `true_to_mean`, `propagate_orbit` | **untouched** |
| new: `orb_varpi`, `orb_lambda`, `orb_ivec`, `orb_what`, `rel_node_dir` | +30 |
| `orbit_to_cartesian` | signature +z,+vz; value-gated legacy fast path + generic 313 (+25) |
| `cartesian_to_elements` | signature +z,+vz; branch table §2.2 (+45) — **the risk concentrate** |
| `apply_impulse` | 3D basis, normal axis, legacy-order gate (+12); fuel block untouched |
| `preview_perigee` | 3-component (+8) |
| `fill_observations` | 3D block 21-32, ϖ channels, 3D LVLH, all under `enable_3d` (+45) |
| `compute_phi` | 3D ē, plane term in quadrature, λ = M+ω+Ω (+20) |
| `check_termination` | z in every distance/energy (+12) |
| `write_traj_record` | +sat_z/vz, target_z/vz, sat_inc/raan, target_inc/raan (+10) |
| `c_reset` | §4 sampling block; `phase_gap_mode` ϖ fix (+45) |
| `Orbital` struct | +9 kwargs (+12) |
| `ACTION_DV`/`ACTION_TAU` | +6 rows at 20-25; `NUM_ACTIONS` 26 |

**`binding.c`** (~20): +9 `unpack` lines in `my_init`; `TRAJ_FLOATS` 86 → 94 with the 8 new floats **appended at the end** (every existing column index stays stable); `fill_traj_row` +8.

**`orbital.py`** (~25): +9 kwargs with legacy defaults + pass-through; `TRAJ_COLS` += the 8 appended names (`TRAJ_FLOATS` is derived, `_traj_buf` auto-sizes); `legacy_action_space` bound 20 → 26; docstring.

**`orbital.c`** (~120): promote A1/A2/A6/A7 into the standalone test main().

**Downstream Python that reads the 38-dim obs / traj columns and needs a 3D port before its harness is valid again** — all currently 2D by construction: `scripts/orbital/nav/orbital_math.py`, `nav/ekf.py`, `nav/eval_relnav.py`, `t3/orb_model.py`, `t3/expert_controller.py`, `t3/fuzz_dynamics.py`, `t3/redteam/rt_common.py`. Note `nav/orbital_math.py:60` still carries the **pre-fix inverted `true_to_mean`** deliberately (documented as inert for its harness) — anything reusing that module for 3D M-recovery inherits the original project-killing bug. Flag loudly; the 3D oracle must come from `ext_recon/orbital_math3d.py`, not from a copy of `nav/orbital_math.py`.

---

## 9. Artifacts

- `/Users/pete/space_training/scripts/orbital/ext_recon/ext_3d_conv_probe.c` — P1/P1b/P2/P3/P4/P5/P6/P7; build `cc -O2 -flto -lm ext_3d_conv_probe.c -o /tmp/ext3d && /tmp/ext3d`
- `/Users/pete/space_training/scripts/orbital/ext_recon/ext_3d_budget_probe.py` — B1–B5
- `/Users/pete/space_training/web_data/results/ext_3d_budget.csv`

No repo files modified; nothing committed.