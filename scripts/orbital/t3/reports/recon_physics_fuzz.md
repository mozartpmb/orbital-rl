All checks complete. Final canonical run: 980 episodes / 618k sub-steps, artifacts persisted.

# T3 — Adversarial physics fuzz of the CORRECTED orbital env

## Verdict

**No further dynamics bugs found.** The corrected two-body/impulse/fuel/propagation core reproduces an independently-written oracle to the float32 logging noise floor across 618k sub-steps. Every hard-physics check PASSES with ≥15× margin to threshold.

**Four non-dynamics defects were found and characterized** (not fixed, per instruction). Three of them are directly load-bearing for why nothing trains post-fix: the env's phase observation and phase shaping are computed in **true anomaly**, a per-body coordinate, not in the physically meaningful inertial angle. Under corrected dynamics this makes obs[13]/obs[14] and Φ_phase a *coordinate artifact* rather than a phase signal.

## Artifacts

| Path | Contents |
|---|---|
| `/Users/pete/space_training/scripts/orbital/t3/fuzz_dynamics.py` | Re-runnable harness, 20 check families (A,B,C,D,E,F,G,H,I,J,K,L,L2,M,N,O,P,Q,R,S) |
| `/Users/pete/space_training/web_data/results/t3_fuzz_dynamics.csv` | Per-check n / mean / p50 / p99 / max / threshold / verdict |
| `/Users/pete/space_training/web_data/results/t3_fuzz_coverage.csv` | Per-cell episode counts + terminal-cause histogram |
| `/Users/pete/space_training/web_data/results/t3_fuzz_dynamics_stdout.txt` | Full canonical-run stdout incl. anomaly contexts |

Re-run: `cd /Users/pete/space_training/pufferlib && python3 ../scripts/orbital/t3/fuzz_dynamics.py --episodes-per-cell 20 --cumulative-per-cell 3` (~15 s).

## Method

Drove the real C env (`num_envs=1`, seeded, kwargs as in `eval_checkpoint.py`) with 7 adversarial action policies × 7 init scenarios × 20 episodes = **980 episodes, 617,999 propagated sub-steps, 67,058 agent decisions, 40,540 burns, 19,315 warp decisions**. Pulled ground truth via `binding.vec_get_trajectory` and re-simulated the identical impulse sequence in `scripts/orbital/nav/orbital_math.py` (`propagate_cartesian`, Lagrange f&g in Cartesian — shares **no** code path with the env's element-wise Kepler propagator; in particular it never calls `true_to_mean`).

Scenarios: `S0_circular` (e=0), `S1_headline` (e_max_target=e_max_sat=0.05), `S2_sameorbit`, `S3_hi_e` (0.10), `S4_hi_e_sameorb`, `S5_fixed_e10` (`e_*_fixed=0.10`), `S6_debris_path` (4–8 debris, physics-coverage only — **all training/eval scenarios are debris-free** per the standing no-debris directive). All with `valid_init_only=1`, `init_phase_gap_max=π`, alt 300–800 km.

Policies: `uniform16`, `burn_heavy`, `warp_burn` (burn→warp-1hr), `theta90` (obs-gated: burn only when |sin θ_sat|>0.84 — the exact regime the old bug corrupted), `periapsis` (burn only near θ≈0), `burst` (3–12 consecutive identical burns), `coast_warp` (warp-heavy, reaches the 2000 sub-step cap).

Terminal-cause coverage: stranded 566, safety_cap 227, collision 187, hyperbolic 0, escape 0, success 0 (random policies never rendezvous).

Comparisons are against an explicit float32-quantization noise model (log is float32; env runs double). Thresholds are ≥8× the modelled floor.

## PASS table (hard dynamics)

| Check | n | p50 | p99 | max | thresh | verdict |
|---|---:|---:|---:|---:|---:|---|
| **L** `true_to_mean` vs correct inverse, e≤0.15 | 20 000 | — | — | **0.000e+00 rad** (exact) | 1e-12 | PASS |
| **L** θ→M→Kepler→θ round trip | 20 000 | — | — | 1.78e-15 rad | 1e-9 | PASS |
| **A** chaser step-local re-sim \|Δpos\| (m) | 617 999 | 1.87e-1 | 4.78e-1 | **6.78e-1** | 10 | PASS |
| **A** …\|Δvel\| (m/s) | 617 999 | 1.90e-4 | 4.79e-4 | 9.46e-4 | 2e-2 | PASS |
| **A** …burn sub-steps only (m) | 40 540 | 1.85e-1 | 4.79e-1 | 6.57e-1 | 10 | PASS |
| **A** …**burns @ \|sin θ\|>0.7, e∈[0.02,0.12]** (m) | 12 088 | 1.86e-1 | 4.78e-1 | 6.57e-1 | 10 | PASS |
| **G** target step-local re-sim \|Δpos\| (m) | 617 999 | 1.87e-1 | 4.78e-1 | 6.67e-1 | 10 | PASS |
| **B** coast ΔE / σ_E (z-score) | 577 459 | 0.22 | 0.79 | 1.28 | 20 | PASS |
| **B** coast Δh / σ_h | 577 459 | 0.11 | 0.39 | 0.64 | 20 | PASS |
| **C** burn ΔE residual vs `v·Δv + ½Δv²` / σ_E | 40 540 | 0.22 | 0.79 | 1.25 | 20 | PASS |
| **C** burn Δh residual vs `x·Δv_y − y·Δv_x` / σ_h | 40 540 | 0.11 | 0.40 | 0.61 | 20 | PASS |
| **D** Tsiolkovsky Δm residual (kg) | 40 540 | 1.58e-6 | 1.34e-5 | 1.66e-5 | 1e-3 | PASS |
| **E** \|dv − table\| unclamped (m/s) | 39 901 | 0 | 0 | **0 (exact)** | 1e-4 | PASS |
| **E** fuel-clamped dv overshoot | 639 | 0 | 0 | 0 | 1e-4 | PASS |
| **F** sub-steps/decision − `ACTION_TAU` | 67 058 | 0 | 0 | **0 (exact)** | 0.5 | PASS |
| **F** dv on warp/coast actions | 19 315 | 0 | 0 | **0 (exact)** | 1e-9 | PASS |
| **J** \|a_cart − a_log\| (m) | 618 979 | 2.82e-1 | 9.56e-1 | 1.56 | 5 | PASS |
| **J** \|(θ+ω)_cart − (θ+ω)_log\| (rad) | 618 979 | 6.76e-8 | 2.91e-7 | 4.84e-7 | 1e-5 | PASS |
| **N** \|Δ(θ+ω)\| across an impulse (deg) — must be 0 | 40 540 | 3.6e-6 | 1.6e-5 | 2.6e-5 | 1e-3 | PASS |
| **O** cumulative whole-episode re-sim, ÷ ULP envelope | 98 | 0.17 | 0.58 | 0.76 | 8 | PASS |
| **H** obs[0..20], obs[33..37] vs ground truth (25 slots) | 67 058 ea | ≤5e-8 | ≤3e-7 | **≤4.8e-7** | 3e-6 | PASS |
| **H** unused body slots obs[21..32] == 0 | 809 472 | 0 | 0 | 0 | 1e-9 | PASS |
| **I** sign(physical gap) == sign(LVLH along-track) | 65 198 | 0 bad | — | 0 bad | 0 | PASS |
| **K** terminal cause consistent with terminal state | 980 | 0 bad | — | 0 bad | 0 | PASS |
| **M** reward == β(γ^τ Φ′ − Φ), independently recomputed | 67 058 | 5.7e-8 | 4.6e-7 | 8.6e-6 | 2e-5 | PASS |

Notes on the numbers: the ~0.19 m median step-local position error is *exactly* the float32 log quantization (0.5·2⁻²³·7e6 ≈ 0.42 m per endpoint, ~0.2 m RMS after the propagation), and it is **flat across burn vs. coast, across e, and across θ** — the signature of pure logging noise, not a physics residual. For contrast, the old `true_to_mean` bug produced ~700 km here. Check O's whole-episode integration (up to 2000 chained sub-steps + 100s of impulses, no re-seeding from the log) stays *inside* the divergence envelope produced by perturbing the seed state by ±1 float32 ULP (max ratio 0.76), so there is no systematic secular drift either.

`compute_phi` was independently transcribed in Python and check M confirms the reward stream matches to 8.6e-6 — this also confirms the shaping-ordering fix (Φ update after `check_termination`) and the Φ-clamp-leak fix behave as documented.

## Anomalies found

### ANOM-1 (HIGH, semantic, not numeric) — phase observation & Φ_phase use true anomaly, not inertial angle

`fill_observations` (orbital.h:571-574) sets `dtheta_phase = sat.orbit.theta − target.theta`, and `compute_phi` (orbital.h:723-724) sets `phi_phase = 1 − cos(sat.theta − target.theta)`. True anomaly is measured from *each body's own periapsis*. The physical along-track separation is the inertial-angle gap `(θ_s+ω_s) − (θ_t+ω_t)`.

Measured (980 episodes):

| | n | p50 | p99 | max |
|---|---:|---:|---:|---:|
| \|true-anomaly gap − physical gap\|, ω_s == ω_t | 1 799 | 0.00° | 0.00° | 5.1e-14° |
| \|true-anomaly gap − physical gap\|, ω_s ≠ ω_t | 63 399 | **64.1°** | 177.1° | **180.0°** |
| sign(obs[13]) disagrees with sign(physical gap) | 63 399 | — | — | **24 907 / 63 399 = 39.3 %** |

Control: `sign(physical gap) == sign(LVLH obs[34] along-track)` is correct in **65 198 / 65 198** samples. So the LVLH block (obs[33..37]) is right and obs[13]/obs[14] is the broken pair. **On 39% of steps the agent's phase observation points the wrong way**, while obs[34] points the right way — the network sees two contradictory phase channels.

Reproduce: any config with `e_max_sat>0` or `e_max_target>0` and `same_orbit_init=0`, e.g. `S1_headline`. Worst case at ctx `dw = π` exactly: physical gap +18.5°, true-anomaly gap −161.5°.

### ANOM-2 (HIGH, semantic) — the (ω, θ) split is ill-conditioned, so every burn teleports θ

An impulse cannot move the satellite, so `θ+ω` is invariant across a burn (check N confirms: max 2.6e-5°, pure float32 noise). But `cartesian_to_elements` recomputes ω from the eccentricity vector, which at small e is tiny and rotates wildly; θ absorbs the compensating jump.

| \|Δθ\| across one impulse (deg) | n | p50 | p99 | max |
|---|---:|---:|---:|---:|
| all burns | 40 540 | **1.70°** | 81.9° | **179.9°** |
| burns with e_pre < 0.005 | 2 718 | **14.8°** | 171.8° | 179.9° |

So obs[2], obs[3], obs[13], obs[14] and Φ_phase all jump discontinuously — up to 180° — on a burn that changes the physical state by ~10 m/s and 0 m. Worst case: `act=3 (+25 m/s), e_pre=0.00108 → e_post=0.0055, Δθ=179.92°, Δω=179.92°`.

This is a **direct successor to the true_to_mean bug in kind but not in magnitude**: the old bug corrupted the *dynamics*; this one corrupts only the *observation and the shaping potential*. Both are triggered by the same call site (`cartesian_to_elements` after every impulse) and both are invisible to a policy that burns only at apsides.

### ANOM-3 (HIGH, shaping) — quantified: ~40% of the phase-shaping signal is a coordinate artifact

Combining ANOM-1+2: per decision I recomputed the shaping delta the env pays, `Δ_true = W_PHASE(γ^τ·Φ_phase(s′)·σ₂ − Φ_phase(s)·σ₂)`, against the same expression with the inertial-angle phase, `Δ_inert`.

| | n | mean | p50 | p99 | max |
|---|---:|---:|---:|---:|---:|
| σ₂ gate open rate (all decisions) | 67 058 | **0.107** | — | — | — |
| \|Δ_true − Δ_inert\| (all decisions) | 67 058 | 1.54e-4 | ~0 | 4.46e-3 | 1.97e-2 |
| \|Δ_true − Δ_inert\|, **gate open** | 7 190 | **7.75e-4** | 7.7e-5 | 1.20e-2 | 1.97e-2 |
| \|Δ_true\| actually paid, **gate open** | 7 190 | **1.95e-3** | 4.1e-4 | 1.80e-2 | 1.99e-2 |
| ratio artifact / \|paid\|, gate open | 7 186 | 33.2 | **0.264** | 69.8 | 6.7e4 |

**Mean artifact magnitude is 40% of the mean phase-shaping magnitude actually paid** (7.75e-4 vs 1.95e-3); on the median gate-open decision, 26%. This is not potential-based-shaping-safe noise — it is a state-dependent term that a policy can farm: at low e, a single 1 m/s burn re-labels θ by tens of degrees and collects up to ±0.02 of phase shaping with zero physical progress.

Also recorded for the shaping audit: **the σ₂ gate is open on only 10.7% of decisions** (opens when `|Δa|/10 km + |Δē| < 2`, i.e. `|Δa| ≲ 20 km`). This confirms the mechanism hypothesis in the brief: the drift-orbit leg that is the *only* correct way to change phase under corrected dynamics requires `|Δa|` of 100–340 km, during which σ₂ is closed (no phase reward at all) and Φ_orbit actively penalizes the drift.

### ANOM-4 (HIGH, experiment-design) — `phase_gap_fixed` / `init_phase_gap_max` do not control the physical phase gap

`c_reset` sets `target.M = sat.M + phase_gap` — a **mean-anomaly** offset in each body's own perifocal frame. With independent ω the physical separation is randomized. Requesting a 30° gap (check R, 512 resets each):

| init config | physical gap mean | physical gap **sd** | \|phys\| p5 | \|phys\| p95 | true-anom gap mean ± sd |
|---|---:|---:|---:|---:|---:|
| e=0 both (circular) | −30.00° | **0.00°** | 30.0° | 30.0° | −30.00 ± 0.00° |
| e=0.05 fixed, independent ω | −15.87° | **111.87°** | 7.5° | **174.0°** | −30.04 ± 2.10° |
| e=0.05 fixed, `same_orbit_init=1` | −30.03° | **2.11°** | 27.2° | 33.1° | −30.03 ± 2.11° |

So in the headline config (`e_max_sat=e_max_target=0.05`, `same_orbit_init=0`) **every phase-gap-labelled eval cell is actually a uniform-random physical gap**. This retroactively confounds any Phase-3/4/5 curriculum stage or surface-eval cell that varied `init_phase_gap_max`/`phase_gap_fixed` while e>0 with independent ω — the 30°/90°/180° curriculum stages were only genuinely staged when `same_orbit_init=1` or e=0. (The Phase-3 stages did use `same_orbit_init`, so those are clean; the Phase-5 fixed-e surface cells are not.)

### ANOM-5 (LOW) — `solve_kepler` does not converge above e≈0.8

5 Newton iterations, initial guess `E=M` for e<0.8 / `E=π` for e≥0.8, early-out at |dE|<1e-12. Max Kepler residual in M:

| e | 0.0–0.70 | 0.80 | 0.85 | 0.90 | 0.95 |
|---|---:|---:|---:|---:|---:|
| max \|residual\| (rad) | ≤8.9e-16 | 1.11e-5 | 9.0e-5 | 6.4e-4 | 3.4e-3 |
| ≈ along-track error at LEO (m) | ≤6e-9 | 77 | 630 | 4 472 | 23 620 |

Harmless in the operational domain (e ≤ 0.10 → 6 nm). Becomes a real error source only if e ≥ 0.8 is ever used — note a LEO→GEO transfer ellipse has e ≈ 0.73 (just inside the good region), and Phase 5 previously ran `e_max=0.70` cells (also inside). Flagging as a tripwire for any future high-e / cislunar work, not a current bug.

### ANOM-6 (LOW) — discrete 60 s termination sampling: bounded, effectively harmless for Earth; debris collision check is inert

Termination samples radius only at 60 s boundaries; the chaser sweeps 3.75° of true anomaly (p99 4.22°, max 4.90°) per sub-step.

- Undetected **sub-surface** arcs (perigee passage below R_EARTH between two above-surface samples): **0 / 617 999**.
- Undetected **sub-keepout** arcs (below R_EARTH+200 km): **2 / 617 999**, both in the same episode (`a=7 037 310 m, e=0.0663, r_min=6 570 986 m` vs keepout 6 571 000 m — a 14 m dip). Keepout is not a termination condition anyway, so this has no behavioural effect.
- Max unsampled dip below the sampled minimum radius: **430 m** (bounded by `a·e·(Δθ)²/2 ≈ 1.6 km` at e=0.10). Earth tunneling is structurally impossible at LEO with these arc lengths.
- **Debris:** closest approach anywhere in 99 219 debris-scenario samples was **3.07 km**; only 1 sample entered the 5 km keepout, and `DEBRIS_HARD_R = 1 m` was never remotely approached. With ≤15 km/s relative speed sampled every 60 s (≈900 km spatial sampling) against a 1 m radius, the debris hard-collision check has a detection probability ~1e-6 — it is effectively **non-functional**. (Not currently exercised: debris is disabled for all training/eval.)

### ANOM-7 (LOW, logging only — found by code reading, confirmed empirically)

1. **Safety-cap terminal record is never written.** `write_traj_record` early-returns on `step >= MAX_STEPS` (orbital.h:872), so a `TERM_SAFETY_CAP` episode's exported trajectory ends at row 1999 / `sim_time = 119 940 s`, one sub-step before the terminal. Confirmed: all-warp-1hr episode → 34 decisions, `env.step=2000`, 2000 rows, last row `sim_time=119940`. Affects trajectory exports only, not learning.
2. **Hyperbolic early-exit clobbers the previous row.** In `c_step` the `sat.orbit.a <= 0` branch (orbital.h:1250-1265) calls `write_traj_record` *before* any propagation, so it writes to `traj_log[env->step]` — the index already holding the previous sub-step's state — with the post-burn hyperbolic state, which `orbit_to_cartesian` evaluates on a conic with `a<0`. Unreachable at LEO with ≤25 m/s burns (needs ~3.1 km/s), and indeed 0 hyperbolic terminals in 980 episodes.
3. **`env->step` / `MAX_STEPS` count sub-steps, not agent decisions** (contrary to the task brief's "MAX_STEPS=2000 agent decisions"). Every episode has a hard wall-clock horizon of 2000 × 60 s = **33.33 h regardless of action mix**. Time-warp buys decision economy, not sim time: 34 consecutive `warp-1hr` actions exhaust the entire episode. Not a bug, but a load-bearing constraint (see below).

## Feasibility corollary (check S, analytic, a = 550 km alt)

Under corrected dynamics phase can only be changed by a drift orbit, `dφ/dt = −1.5(Δa/a)·n`, round-trip `Δv ≈ v·|Δa|/a`, against the hard 33.3 h clock and 478 m/s budget:

| \|Δa\| km | drift °/h | h to close 180° | round-trip Δv | fits 33.3 h | fits 478 m/s |
|---:|---:|---:|---:|---|---|
| 50 | 2.45 | 73.4 | 55 | **no** | yes |
| 100 | 4.90 | 36.7 | 110 | **no** | yes |
| 150 | 7.35 | 24.5 | 165 | yes | yes |
| 200 | 9.80 | 18.4 | 219 | yes | yes |
| 300 | 14.71 | 12.2 | 329 | yes | yes |
| 400 | 19.61 | 9.2 | 439 | yes | yes |
| 500 | 24.51 | 7.3 | 548 | yes | **no** |

The feasible band for a 180° gap is **|Δa| ≈ 150–400 km**, costing 165–439 m/s of a 478 m/s budget — i.e. 35–92% of total fuel spent purely on phasing, before any of it is spent matching `a_target` (which in the headline config differs from `a_init` by ≥50 km by construction). The window is narrow, and *the entire drift leg is spent with σ₂ closed and Φ_orbit penalizing at −W_ORBIT·|Δa|/10 km = −0.15 to −0.40 per step*. This is a concrete, quantified restatement of the leading explanation for post-fix training failure, and it is now backed by verified-correct physics.

## Summary of what changed vs. what did not

- The `true_to_mean` fix is **exact** (bit-identical to the correct inverse over 20k draws) and the burn→propagate path it feeds is clean at every θ and e in the operational domain, including the θ≈90°/270° regime the old bug corrupted (12 088 targeted samples, max error 0.66 m = logging noise).
- Nothing else in propagation, impulse application, fuel, warp, termination, or the 25 verified observation slots is wrong.
- The remaining defects are all in the **coordinate choice for phase** (ANOM-1/2/3/4). They were *masked* by the old bug — which let burns move phase directly with Δa≈0 so σ₂ stayed open — and are now exposed. They are the highest-value targets for the shaping/observation redesign in task #14, and ANOM-4 additionally means several historical eval cells need re-labelling.