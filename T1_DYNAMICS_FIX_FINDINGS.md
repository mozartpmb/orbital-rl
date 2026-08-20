# T1 — Corrected-Dynamics Findings: the true_to_mean() Fix and What Survives It

> **STATUS UPDATE 2026-08-11 — RECOVERY COMPLETE. See `T3_RECOVERY_CAMPAIGN.md`.**
> The shaping redesign proposed in §6 was superseded by a deeper diagnosis (five
> compounding signal defects, incl. a reward-side warp barrier the §6 options would not
> have touched). Under the T3 fix stack the headline task is solved at **100.0% held-out
> by 5/5 seeds** (2,900 episodes, zero failures), extended to e ≤ 0.15 (L3, 100%), with
> Lambert/EKF validation lawful. The §3 recovery-attempt table below remains the accurate
> record of what did NOT work and why.
>
> **Original status (2026-08-10):** The inverted `true_to_mean()` anomaly conversion
> (commit `f55d9cb`) is fixed and verified. Under corrected dynamics the canonical policy
> drops 97.5% → 13.0%, fresh bootstrap fails at 2× budget, and the best recovery achieved
> today is **33.5% greedy** (gentle re-adaptation of the legacy policy, epoch-25 peak).
> The recipe's published learnability and fuel-efficiency were both partly artifacts of
> the bug.

---

## 1. The bug

`true_to_mean()` (orbital.h:340) applied the forward E→θ half-angle map a second time instead
of its inverse — `sqrt(1−e)`/`sqrt(1+e)` swapped. Reached only from `cartesian_to_elements()`,
i.e. after every burn, it set the chaser's mean anomaly with error ≈ 2e·sin(θ).

**Measured** (independent exact f&g propagator as reference):
- Pre-fix: 10 m/s prograde burn at θ≈90°, e≈0.05 → **~707 km** position error one step later.
  Coast steps exact to 0.1 m (float32-obs floor). Post-fix: **0.4 m** over 25 burns.
- On logged headline trajectories: **median 24.5° of net free along-track phase per episode**
  (p90 63.6°, max 105°), sign selectable by burn location.

**Found twice independently on the same day:**
1. The T2 relative-navigation port audit (formula inspection while porting the obs math).
2. The T1 Lambert baseline — against the TIME-MATCHED two-impulse optimum the tail was
   physically impossible (46.5% of episodes below the fixed-time optimum, 15.0% below
   HALF of it, p10 0.368 — no policy legitimately halves a fixed-time two-impulse
   optimum), which forced the trace-down. (The 0.31× median often quoted here was vs
   the IMMEDIATE comparator, which forces departure at t0 and which a policy allowed
   to wait legitimately beats — the corrected, teleport-free policy still scores
   0.331× against it. The tail, not the median, was the impossibility.)

**Why five months of work never saw it:** burns at apsides have sinθ ≈ 0, where the error
vanishes to first order — and Hohmann-style policies burn at apsides. All 8 Phase-1 physics
validation tests exercised pure propagation and Δv budgets, never post-burn phase.
`p5e_e3_round_trip.py` round-tripped Cartesian↔elements without exercising M.

## 2. What the bug was doing to the results

- **Eval:** phasing is the dominant cost of this task; the bug paid the policy ~25°/episode of
  it for free. All published success rates describe the bug-assisted task.
- **Fuel claims:** "2.49× Hohmann" (and the Lambert ratios ≤ 1) describe a policy that could
  buy phase without propellant. Invalid as fuel-efficiency statements.
- **Training:** the bug subsidized exploration — any random off-apsis burn teleported phase,
  so a random policy stumbled into the sparse success box often enough to bootstrap.

## 3. Recovery attempts under corrected dynamics (all seed 42, LEO headline conditions)

| # | Protocol | Result |
|---|---|---|
| 0 | Canonical Phase 5e ckpt, zero-shot re-eval | **13.0%** (26/200); causes: 142 cap, 31 stranded, 1 collision |
| 1 | Fresh Stage-1 bootstrap, 40M (published budget) | **0.7%** rolling at epoch 306 — flatline, never took off |
| 2 | Fresh Stage-1 bootstrap, 80M (train-longer null) | Flatline again; Stage-4 from it: 0/200 |
| 3 | Direct re-adapt of canonical ckpt (lr 5e-4, ent 0.005, 30M) | Rolling 59%; **greedy held-out peak 33.5% (epoch 25)**, non-monotonic after (17→19.5→33.5→16.5→19) |
| 3a | Stochastic eval of the 33.5% ckpt | 27.0% — stochastic ≈ greedy; the rolling 59% was on-policy optimism, no hidden capability |
| 3b | Crystallization: continue from peak at lr 1e-4, 15M | 27.5% immediately, then monotonic decline — no lift |
| 3c | Curriculum rebuild: re-adapt Stage-1 warm-start (lr 5e-4, 20M) → standard Stage-4 (default profile, 50M) | Stage-1 warm-start quality 4/100; Stage-4 collapsed to 0 within 25 epochs — the 5.5.1 aggressive-profile pattern again |

**Interim corrected-dynamics checkpoint:**
`pufferlib/experiments/puffer_orbital_178640242476/model_puffer_orbital_000025.pt` — 33.5%
greedy / 27.0% stochastic, 200 eps, headline conditions.

## 4. Mechanism: why fresh training fails now

Under correct two-body physics, same-orbit phase can only be changed by opening a drift orbit
(temporarily raising |Δa|) and waiting. The shaping potential's Φ_orbit component **rewards
|Δa| → 0**, i.e. it actively penalizes the only physically available phasing strategy during
the leg where it matters. Pre-fix, this tension was invisible: the policy could keep Δa ≈ 0
and let off-apsis burn-teleports do the phasing — reward design and dynamics bug were
synergistic. The pre-fix policy did also learn genuine ±200 km drift orbits (T1 taxonomy), so
its skill partially transfers (13% zero-shot, 33.5% re-adapted), but the full published
performance does not.

## 5. What still stands, unchanged

- The environment engineering (C env, ~500k SPS), the analytic-propagation choice (coast
  dynamics were always exact), the eval/verification infrastructure, and everything fixed in
  commits `b9f0ef5`/`57f1bef` (classifier, Φ-clamp leak, logging).
- The T2 relative-navigation stack: filter consistency (NEES 0.917 / NIS 0.947) is
  independent of the policy; the closed-loop table needs a re-run against any new checkpoint.
- The T1 failure taxonomy and Lambert solver as *tools*; their numbers describe the pre-fix
  policy and are labeled as such.
- Pre-fix results as statements about the pre-fix environment, reproducible at commit
  `b9f0ef5` and earlier.

## 6. Proposed next interventions (in order)

1. **Shaping revision for the phasing leg** — the direct fix for §4. Options, cheapest first:
   (a) gate Φ_orbit off (σ₁=0) while |Δθ| gap is large, so the drift-orbit leg isn't punished;
   (b) replace Φ_orbit's |Δa|→0 target with |Δa − Δa_phasing*(Δθ)| where Δa_phasing* is the
   period-difference needed to close the gap within the horizon (physics-informed potential);
   (c) pure Φ_phase + Φ_vel, drop Φ_orbit entirely at Stage 1. Each is a ~5-line C change +
   40M-step smoke (~7 min each).
2. **Re-adapt + revised shaping** — warm from the 33.5% interim ckpt under the revised
   potential; likely the fastest path back to a headline-grade corrected policy.
3. **Multi-seed whatever works**, then re-run the Lambert baseline (~4 min) and relnav
   harness (~70 s) against the new canonical, and re-issue the capability tables.
4. The tightened success box (5 km / 1 m/s, Discrete-16 fine burns) layers on top once the
   standard box reproduces.

---

*Author: 2026-08-10. Companion commits: f55d9cb (fix), 78e5bb1 (Lambert/independent
discovery), defc932 (relnav/port audit), a3fbe33 + 0c98b2f (recovery runners). Recovery run
dirs: 178639401160 (fresh 40M), 178640277373/178640381331 (fresh 80M + s4),
178640242476 (direct re-adapt), 178640463481 (crystallization), 178640484881/178640519911
(curriculum rebuild).*
