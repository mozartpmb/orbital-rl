## 3D-B — Plane-change Δv geometry, feasible envelope, curriculum ladder

Pure astrodynamics + MC. No env changes, no training, nothing committed.

**Scripts** (`/Users/pete/space_training/scripts/orbital/ext_recon/`): `ext_3d_planechange_tables.py`, `ext_3d_joint_feasibility.py`, `ext_3d_crank_placement.py`, `ext_3d_j2_nodal.py`, `ext_3d_ladder.py`
**CSVs** (`/Users/pete/space_training/web_data/results/`): `ext_3d_dv_planechange.csv`, `ext_3d_dv_envelope.csv`, `ext_3d_dv_combined.csv`, `ext_3d_dv_threeburn.csv`, `ext_3d_dv_raan.csv`, `ext_3d_dv_crankplacement.csv`, `ext_3d_dv_marginal.csv`, `ext_3d_dv_knob.csv`, `ext_3d_dv_feasibility.csv`, `ext_3d_dv_dimax.csv`, `ext_3d_j2_nodal.csv`, `ext_3d_ladder.csv`
Method is a direct extension of `scripts/orbital/t3/joint_feasibility.py` (same Hohmann + e-vector-match + drift-phasing-surcharge + horizon model, budget 478.13 m/s), so the 2D columns reproduce the T3 screen exactly (e.g. L2 99.1%, WL4 97.5%, M5 92.8% — matching §8.4's 93.0%).

---

### 1. Plane-change budgets and the trainable Δi envelope

**Δv = 2 v sin(Δi/2), circular speed** (`ext_3d_dv_planechange.csv`, m/s):

| Δi | 400 km | 800 km | 2000 km | 8000 km | 20200 km |
|---|---|---|---|---|---|
| 0.10° | 13.4 | 13.0 | 12.0 | 9.2 | 6.8 |
| 0.25° | 33.5 | 32.5 | 30.1 | 23.0 | 16.9 |
| 0.50° | 67.0 | 65.1 | 60.2 | 46.0 | 33.8 |
| 0.75° | 100.4 | 97.6 | 90.3 | 68.9 | 50.7 |
| 1.00° | 133.9 | 130.1 | 120.4 | 91.9 | 67.6 |
| 2.00° | 267.8 | 260.2 | 240.9 | 183.8 | 135.2 |
| 3.00° | 401.7 | 390.3 | 361.3 | 275.7 | 202.8 |
| 5.00° | 669.3 | 650.4 | 602.0 | 459.4 | 337.9 |

**Envelope after paying for phasing + a/e match** (`ext_3d_dv_envelope.csv`; reserves are the *measured* in-plane spends of the shipped policies, `t3_headline_characterization.csv` / `t3_joint_feasibility.csv`):

| in-plane reserve | 400 km | 800 km | 2000 km | 8000 km | 20200 km |
|---|---|---|---|---|---|
| 0 (whole budget) | 3.571° | 3.675° | 3.971° | 5.203° | 7.077° |
| 110 (e=0 Hohmann p50) | 2.749° | 2.829° | 3.057° | 4.006° | 5.448° |
| 235 (LEO headline Δv p50) | 1.816° | 1.869° | 2.019° | 2.645° | 3.597° |
| 325 (LEO headline Δv **p90**) | 1.144° | 1.177° | 1.271° | 1.666° | 2.265° |
| 386 (WL4 analytic p90) | 0.688° | 0.708° | 0.765° | 1.002° | 1.363° |
| 439 (M5 analytic p90) | 0.292° | 0.301° | 0.325° | 0.426° | 0.579° |

**Headline number: the trainable relative-plane envelope is O(1°), not O(10°).** The whole plane-change literature lives at 10–90°; at a 478 m/s budget the entire 3D task is a *small-angle* plane problem. That is the honest framing for the write-up.

**di_max for ~99% joint feasibility** (`ext_3d_dv_dimax.csv`, 4000 samples/cell; criterion = plane leg costs ≤1pp of that band's own 2D screen, since M5/WL4 baselines are already <100%):

| band (2D screen) | di_max, sequential (−1pp) | (−5pp) | di_max, combined-burn (−1pp) |
|---|---|---|---|
| L1/WL1 500–800, e=0, same-orbit (100.0%) | ≥3.0° (Δv cap 3.5°) | ≥3.0° | ≥3.0° |
| L2 300–800, e≤0.05 (99.1%) | 0.40° | 0.90° | **1.20°** |
| WL3 300–2000, e≤0.15 (100.0%) | 0.75° | 1.20° | **1.75°** |
| WL4 300–8000, e≤0.30 (97.5%) | 0.30° | 0.90° | **1.20°** |
| M5 300–20200, e≤0.50 (92.8%) | 0.10° | 0.90° | **0.75°** |

Sequential = never combine the plane burn with an in-plane burn (conservative). Combined = the largest single in-plane impulse is co-located with a node. The truth is between; the trained policy already runs at 1.46–1.58× the two-impulse bound, i.e. it does find efficient multi-impulse structure, and the analytic screen has historically *under*-predicted measured success by 2.5–6.5pp (M5: screen 93.0% → measured 99.5%; WL4: 97.5% → 100%).

**Recommendation: one knob, `di_max = 0.75°`, constant from the headline rung upward.** Plane tax is then 62–92 m/s at p90 and screen cost 0.5–3.7pp per rung. Notice di_max is nearly altitude-invariant — v falls with altitude but the in-plane job grows — which is why one number works across the whole ladder.

**Combined-maneuver saving (needed to get the shaping lower bound right).** The exact single-impulse law is
```
Δv_comb = sqrt(v1² + v2² − 2 v1 v2 cos θ) ≡ hypot( |v2−v1| , 2·sqrt(v1 v2)·sin(θ/2) )
```
— an **identity**, not an approximation: in-plane magnitude change and plane rotation add *in quadrature* with the geometric-mean speed. Consequences (`ext_3d_dv_combined.csv`, `ext_3d_dv_marginal.csv`):
- Max saving vs sequential = **29.3%**, attained when Δv_plane = Δv_inplane; falls to 8.6% at a 10:1 ratio either way. Ceiling is √2, i.e. sum/hypot ≤ 1.414.
- Realistic LEO cases: 400→800 km with Δi=0.5° saves 19.6%; Δi=1° saves 26.8%; 400→2400 km with Δi=0.1° saves only 1.2%.
- **Marginal** cost of the plane leg on top of an in-plane impulse D is `hypot(D,P) − D ≈ P²/2D`. At 400 km, Δi=0.5°: 67.0 m/s standalone → 33.6 on top of a 50 m/s burn → 20.3 on top of 100 → 7.4 on top of 300. This is the single biggest lever in the 3D task.
- Optimal split of θ across the two Hohmann burns is ≈48/52, *not* "all at apoapsis," because marginal rates equalize. And the split is geometrically free: a Hohmann departs and arrives 180° apart, and the two nodes are 180° apart, so a transfer departing at one node arrives at the other.

**Shaping directive:** Φ's Δv_match term must combine the plane error in quadrature, not additively:
```
Δv_match_3D = hypot( ½·v_t·hypot(Δa/a_t, |Δē|) ,  2·v_t·sin(θ/2) )
```
Using the sum over-penalizes plane error by up to 41% and re-creates exactly the failure mode that sank T1/T3 (a potential that fights the cheap strategy). Keep the single `min(1, ·/DV_REF)` saturation; at di_max 0.75° with DV_REF=700 the plane term adds ≤100 m/s so saturation stays low (WL4-equivalent 2D saturation was already the reason for DV_REF=700). Red-team gate: re-run the audit's worst-adverse-step measurement on the 3D Φ (2D reference: −0.041 vs terminal ±10, 24× margin).

---

### 2. Raise-crank-lower: outside every feasible rung, by 5–11×

`ext_3d_dv_threeburn.csv`. Three-burn = raise apoapsis to r₂, crank at apoapsis, lower back.

- Crossover Δi where three-burn beats direct, as a function of the raise: **38.94°** at r₂→r₁ (infimum over all raise sizes — this is the classical threshold, and it falls out of the first-order condition `sin(Δi/2) > 1/3`, reproduced numerically to 38.94°), rising monotonically to 39.33° (r₂=800 km), 43.22° (8000 km), 45.53° (20200 km), 48.67° (r₂/r₁=57, the raise-to-infinity limit). **Below 38.94° no apoapsis raise of any size pays.**
- Δv at the crossover is **10.7–13.2× the whole 478 m/s budget**.
- Largest Δi purchasable with the *entire* budget and nothing else to do: 3.571° (400 km) → 7.077° (20200 km) direct; the three-burn version is strictly worse everywhere (2.02° at the LEO-headline ceiling, and 0.000° for any wider ceiling — the raise alone busts the budget).

**Verdict: the raise-crank-lower trade cannot emerge in any rung this env can afford. Do not design for it, do not shape for it, and do not claim it.**

What *is* inside the envelope and worth designing for — two distinct skills, both physically available and both worth isolating with diagnostic probes:

1. **Node/apsis placement.** Cranking at the cheaper of the two nodes saves 3.2% at e=0.05, 9.5% at e=0.15, **18.4% at e=0.30, 28.7% at e=0.50** (median over random node/ω geometry, `ext_3d_dv_crankplacement.csv`). Full apoapsis placement (v_apo/v_peri = 0.538 at e=0.3, 0.333 at e=0.5) is only available when the node happens to sit near apoapsis; the median best-of-two-nodes speed at e=0.5/400 km is 6529 m/s vs v_apo 4430 and v_circ 7673.
2. **Burn combining.** The P²/2D marginal law above — worth 50–80% of the plane leg whenever a ≥100 m/s in-plane burn exists.

**Rung that maximizes the chance either emerges: I5 (e≤0.30, de_max 0.08, da_max 600 km, 300–8000 km, cap 6000, di_max 0.75°)** — high e makes placement worth 18%, and da_max 600 km guarantees a ≥200 m/s Hohmann burn to combine with. To *attribute* rather than hope, add two non-gating probes: (a) high-e/small-da (isolates placement, no burn to hide in), (b) low-e/large-da (isolates combining). Measure realized burn argument-of-latitude vs the relative node in the trajectory log.

---

### 3. RAAN: bound it jointly, via the relative inclination vector

**Recommendation: one knob, `di_max` = radius of a uniform disc on Δi_vec = (Δi, ΔΩ·sin i). Do not expose a separate ΔΩ bound.** Numbers (`ext_3d_dv_raan.csv`, `ext_3d_dv_knob.csv`):

- Plane-change Δv depends on **nothing but** θ = |Δi_vec|, the angle between the planes. The decomposition sets only *where* the relative node sits (u_node = atan2(ΔΩ sin i, Δi)) — which is the interesting geometry and should be sampled uniformly via the disc's phase angle, not bounded.
- A separate ΔΩ knob is not a difficulty knob: the cost of 1° of ΔΩ varies **11.4×** across inclination (sin 98° / sin 5° = 0.990/0.0872) and is exactly **zero and degenerate at i=0**. At 400 km, 1° of ΔΩ costs 133 m/s at i=90° but 11.7 m/s at i=5°.
- A box (|Δi|≤K, |ΔΩ|≤K) has max θ = √2·K and its *mean* θ swings 0.500°→0.765° (**1.53×**) as i goes 0→90° at K=1°; its p99 Δv is 1.15–1.32× the disc's, depending on the i range sampled. The disc is exactly i-invariant (mean θ = 2K/3, max = K) at every inclination.
- **Absolute inclination is free in two-body** — i affects no Δv anywhere. So expose `i_min/i_max` as a separate, cost-free realism/coverage knob and open it fully (0–98°) at the very first 3D rung.
- Implementation nit: sample the disc, then apply the target-plane rotation *exactly* (rotate by θ about the axis at u_node) rather than adding small-angle components — at θ ≤ 3° the linearization error is <0.05%, but exactness costs nothing and keeps the knob's meaning unambiguous under red-team.

**Second-order finding worth a stretch rung (`ext_3d_j2_nodal.csv`):** the env is pure two-body, so ΔΩ is frozen and every plane change must be bought propulsively — which removes the technique real operators actually use. Secular J2 rates are analytic and cheap (`Ω̇ = −1.5 n J2 (Re/p)² cos i`), and differential nodal drift is *inside our horizons*: at 400 km, i=51.6°, Δa=100 km → 0.251°/day → **0.52° free ΔΩ in a 3000-step episode (2.08 days), worth 55 m/s**; Δa=400 km over a 6000-step episode → **3.81°, worth 400 m/s**. It dies with altitude (8000 km: 0.07°; 20200 km: 0.005°). Absolute rates at 400 km: Ω̇ = −5.02°/day at i=51.6°, ω̇ = +1.88°/day. Risk to flag if adopted: ω̇ of 1.5–8°/day at LEO rotates the e-vector and moves λ = M+ω, so it perturbs both the potential and the e-match — this must be a runtime flag, default off, with its own red-team pass, not a quiet dynamics change.

---

### 4. Curriculum ladder (`ext_3d_ladder.csv`)

Mirrors the proven L/WL/M pattern: fresh per-lineage re-ladder, obs scales constant *within* a lineage, gate between rungs, held-out seed-123 evals at the rung's own config.

| rung | lineage | role | i range | di_max | e_max | de_max | da_max | band (km) | cap (steps/h) | warm from | 2D screen | 3D screen (seq) | (comb) | Δv_plane p90 | Δv_tot p90 | gate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **I0** | regression | 2D bit-exact anchor, **no training** | 0 | 0 | 0 | — | — | 500–800 | 3000 / 50 | n/a | 100.0% | 100.0% | 100.0% | 0 | 82 | legacy 26/200 ckpt bit-exact + T3 canonical 200/200 |
| **I1** | I (8e6/1.5e7/700) | 3D frame, zero plane error — WL1 twin | 0–98° | 0 | 0 | — | — | 500–800 | 3000 / 50 | fresh | 100.0% | 100.0% | 100.0% | 0 | 82 | 200/200, **must equal WL1** |
| **I2** | I | first plane error, circular | 0–98° | 0.50° | 0 | — | — | 500–800 | 3000 / 50 | I1 | 100.0% | 100.0% | 100.0% | 62.1 | 138 | ≥190/200 |
| **I3** | I | **3D headline** (L2 twin + plane) | 0–98° | 0.75° | 0.05 | — | — | 300–800 | 3000 / 50 | I2 | 99.2% | 96.6% | 99.1% | 91.8 | 425 | ≥190/200, 3 seeds |
| **I4** | I | wide-e + plane (WL3 twin) | 0–98° | 0.75° | 0.15 | 0.06 | 400 | 300–2000 | 3000 / 50 | I3 | 100.0% | 99.5% | 100.0% | 85.5 | 399 | ≥190/200 |
| **I5** | I | widest jointly feasible (WL4 twin) | 0–98° | 0.75° | 0.30 | 0.08 | 600 | 300–8000 | 6000 / 100 | I4 | 98.0% | 94.3% | 97.5% | 72.9 | 439 | ≥185/200, then 3–4 seeds |
| **MI5** | MI (2.1e7/4e7/700) | stretch: MEO 3D (M5 twin) | 0–98° | 0.75° | 0.50 | 0.10 | 1000 | 300–20200 | 12000 / 200 | fresh MI1–MI4 re-ladder | 92.9% | 89.5% | 92.8% | 61.8 | 486 | report **vs the screen**, not vs 100% |
| I5+ | I | edge probe, **not a gate** | 0–98° | 1.50° | 0.30 | 0.08 | 600 | 300–8000 | 6000 / 100 | I5 | 98.0% | 87.2% | 96.4% | 145.8 | 498 | exploratory |

Screens are 6000-sample MC, seed 4242. "3D screen (seq)" is the conservative no-combining model; "(comb)" assumes the largest in-plane impulse is node-co-located. Historical calibration says measured lands at or above the seq screen (M5 93.0%→99.5%).

**Eval-conditions table** (all rungs carry the T3 base flags unchanged: `--shaping-mode 1 --shape-gamma 1.0 --phase-gap-mode 1 --phase-obs-mode 1 --cap-terminal-reward 0.0 --valid-init-only 1 --init-phase-gap-max 3.14159 --legacy-action-space <N> --seed 123 --episodes 200`; proposed new kwargs in bold):

| rung | rung-specific eval flags |
|---|---|
| I0 | `--plane-mode 0` (legacy path; assert bit-exactness against `models/t3/seed42_L2_headline.pt` and the legacy 26/200 anchor) |
| I1 | **`--plane-mode 1 --i-min-deg 0 --i-max-deg 98 --di-max-deg 0`** `--same-orbit-init 1 --e-max-target 0 --e-max-sat 0 --a-min-override 6.871e6 --a-max-override 7.171e6 --episode-cap-steps 3000 --obs-alt-scale-m 8e6 --lvlh-scale-m 1.5e7 --shape-dv-ref-ms 700` |
| I2 | I1 flags with **`--di-max-deg 0.75`**→ set 0.5 |
| I3 | **`--di-max-deg 0.75`** `--e-max-target 0.05 --e-max-sat 0.05 --a-min-override 6.671e6 --a-max-override 7.171e6 --episode-cap-steps 3000` (+ I-lineage scales) |
| I4 | **`--di-max-deg 0.75`** `--e-max-target 0.15 --de-max 0.06 --da-max-m 400e3 --a-min-override 6.671e6 --a-max-override 8.371e6 --episode-cap-steps 3000` |
| I5 | **`--di-max-deg 0.75`** `--e-max-target 0.30 --de-max 0.08 --da-max-m 600e3 --a-min-override 6.671e6 --a-max-override 14.371e6 --episode-cap-steps 6000` |
| MI5 | **`--di-max-deg 0.75`** `--e-max-target 0.50 --de-max 0.10 --da-max-m 1000e3 --a-min-override 6.671e6 --a-max-override 26.571e6 --episode-cap-steps 12000 --obs-alt-scale-m 2.1e7 --lvlh-scale-m 4e7 --shape-dv-ref-ms 700` |

**Three quantities the ladder designer must not get wrong** (these are mine to hand over, they constrain 3D-A's action/obs design):

1. **Normal-burn quantum.** Plane residual θ alone fills the success box at θ = 0.254° (30 km box @400 km), 0.120° (@8000 km), 0.065° (@20200 km) — position always binds, not velocity. A normal impulse of q m/s moves θ by q/v: at 400 km, 1 m/s = 0.0075°, 5 = 0.037°, **25 = 0.187° (73% of the whole box)**. At 20200 km, 25 m/s = 0.370°, i.e. **5.7× the box**. Recommend normal actions **{±1, ±5, ±25} m/s → Discrete-26**; ±25 is the workhorse (0.75° at LEO = 100 m/s = 4 burns), ±1 is mandatory for MI5 and for any tight-box follow-up. Note the plane residual *consumes* box budget shared with the in-plane miss — keep the 30 km/50 m/s box for the whole I-ladder and defer tightening.
2. **Node-timing granularity.** An out-of-plane impulse at argument-of-latitude offset δu from the node contributes cos δu usefully and injects sin δu of orthogonal inclination-vector error. Per-step along-track motion at 400 km: τ=1 → 3.90°/step (best |δu| 1.95°, 0.06% loss, 0.034× orthogonal), τ=5 → 19.5°/step (9.74°, 1.44%, 0.169×), **τ=60 (warp-1hr) → 233.7°/step — unusable**. Budget ≈ **+15% on the plane leg at 5-min granularity, +3% at 60 s.** The 2D policy spends 96.8% of its actions on warps, so "step down to ≤5-min granularity near the node" is a genuinely new behavior the ladder must elicit; verify it is *reachable* before training by extending `scripts/orbital/t3/expert_controller.py` to 3D and demanding the same 100/100 it achieved at the T3 config. This is the single highest-risk item in the 3D extension and it is a control-granularity risk, not a Δv risk.
3. **Node availability is cheap in time.** Two node crossings per orbit → a plane burn can always be scheduled inside the existing multi-rev drift-phasing loiter, so the node wait is free except for plans shorter than half a period (that assumption is coded explicitly in `ext_3d_joint_feasibility.py`; without it the L1 screen drops 100.0%→98.6% purely as a horizon artifact). Residual coupling: requiring departure at a node quantizes departure time to P/2, which is ~4° of phase at Δa=100 km — absorbed by tuning the phasing-orbit a, already inside the surcharge scan.

**Caveats.** All feasibility figures are analytic screens (impulsive, Keplerian, two-impulse-plus-surcharge in-plane model, best-of-two-nodes plane model) — they bound the task, they do not predict the policy; the 2D history says they under-predict by 2.5–6.5pp. The "combined" column assumes node/burn co-location the policy is not guaranteed to find. J2 numbers describe an env change that does not exist yet. Nothing here has been run through the C env; the first real gate is I0's bit-exactness check.