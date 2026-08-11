# T3 — Corrected-Dynamics Recovery Campaign

> **Status: IN PROGRESS (started 2026-08-10, evening).** Goal: a policy that performs the
> rendezvous task **fully under correct physics** — wide phase gaps (±180°) and as wide an
> eccentricity range as the environment geometry allows — with every claim backed by raw
> eval output. This document is the campaign's living record: hypotheses first, then
> evidence, then decisions, in order. Nothing here is final until marked ✅.

Companion docs: `T1_DYNAMICS_FIX_FINDINGS.md` (the bug, the fix, the exhausted quick
recoveries), `MODELS.md` (checkpoint registry — all pre-fix numbers labeled as such).

---

## 0. Starting position (2026-08-10, all verified)

| Fact | Value | Source |
|---|---|---|
| Env physics | **Correct** post `f55d9cb`; coast was always exact; post-burn error 707 km → 0.4 m | T1 findings §1 |
| Canonical pre-fix ckpt, corrected dynamics | 13.0% (26/200) | T1 findings §3 |
| Best corrected-dynamics ckpt | 33.5% greedy / 27.0% stochastic (re-adapt epoch 25) | `puffer_orbital_178640242476/model_..._000025.pt` |
| Fresh Stage-1 bootstrap | Flatlines at 40M and 80M steps | T1 findings §3 |
| Crystallization / curriculum rebuild | Both fail | T1 findings §3 |

## 1. Central hypothesis (H1): the shaping potential opposes the correct strategy

Reading `compute_phi()` (orbital.h:707–753) precisely — sharper than T1 §4 stated:

```
Φ = −(0.01·φ_orbit·σ₁ + 0.01·φ_phase·σ₂ + 0.01·φ_vel·σ₃)      σ₁ = 1 always
φ_orbit = |Δa|/10 km + |Δē|
σ₂      = sigmoid((2.0 − φ_orbit)/0.2)      ← opens only when |Δa| ≲ 20 km
σ₃      = σ₂ · sigmoid((0.3 − φ_phase)/0.03)
```

Under correct two-body physics, phase can only be changed by opening a drift orbit
(|Δa| ~ 100–340 km) and waiting. But at |Δa| = 100 km, σ₂ ≈ 0, so:

1. **Phase progress earns nothing during the drift leg** — the φ_phase term is gated off
   exactly when phase is being closed.
2. **The drift itself is penalized** — φ_orbit grows 10 per 100 km of Δa, a potential drop
   of ~0.1–0.34 the agent must pay to enter the leg and only recovers on exit
   (a long reward valley with a gate cliff at each end).
3. **The stage ordering (orbit → phase → vel) is backwards.** The physical maneuver
   sequence is phase → orbit → vel: you phase *with* a mismatched orbit, then match
   orbits at arrival.

The pre-fix bug made this invisible: burn-teleports changed phase while Δa stayed ≈ 0,
i.e. inside the open-gate region. Reward design and dynamics bug were **synergistic** —
which is why the recipe looked well-tuned for five months.

**H1 prediction:** a potential that is monotone along the *correct* maneuver sequence
(rewarding drift-orbit phasing progress, not punishing it) restores learnability.

## 2. Secondary hypotheses

- **H2 (discount horizon):** at γ = 0.995 the +10 terminal is worth 10·γ^N ≈ 3.5 at
  N = 210 decisions (180° gap via 200 km drift, warp-5min only — the legacy action set's
  only warp) and ~0 at the 2000-step cap. The Discrete(16) long warps (30 min / 1 hr)
  compress the same wall-clock drift leg into ~18–35 decisions, putting the terminal
  reward *fully inside* the discount horizon. The long warps (M2) were never exercised
  by any training run. Prediction: Discrete(16) + revised shaping ≫ Discrete(10) + revised shaping.
- **H3 (φ_phase measures the wrong angle):** `compute_phi` uses θ_sat − θ_target (true
  anomaly difference), not true-longitude difference (θ+ω). If ω's differ (they change on
  every burn via `cartesian_to_elements`), the shaping's alignment point is offset from
  the physical rendezvous point by ω_s − ω_t. Being audited (recon agent 2).
- **H4 (feasibility):** the task is solvable in-action-space within MAX_STEPS=2000 and the
  478 m/s budget by a *scripted classical controller* (drift-orbit phasing + two-impulse
  closure + terminal nulling). If H4 holds, RL failure is purely a learning-signal
  problem (H1/H2), and the expert's trajectories are available as imitation data.
  If H4 fails, the binding constraint (horizon / burn quanta / box) gets relaxed first.

## 3. Campaign structure

**Recon (running)** — workflow `wf_d424c270-3a4`, six parallel agents:

| # | Agent | Question | Deliverable |
|---|---|---|---|
| 1 | physics-fuzz | Any residual dynamics bugs post-fix? | `scripts/orbital/t3/fuzz_dynamics.py`, PASS/FAIL per invariant |
| 2 | shaping-audit | Quantify the H1 valley; settle H3; compare redesigns S-A/S-B/S-C | `scripts/orbital/t3/shaping_trace.py`, per-leg shaping table, one recommendation |
| 3 | feasibility | H2 math: steps/discount/Δv tables; e-extension ladder; γ recommendation | `scripts/orbital/t3/feasibility_tables.py`, config recommendation |
| 4 | expert-baseline | H4: scripted GNC controller success rate at headline conditions | `scripts/orbital/t3/expert_controller.py`, `web_data/results/expert_baseline.csv` |
| 5 | lit: RL practice | Karpathy / Irpan / PPO-details / shaping / curriculum distilled vs our failure signature | ranked recommendations + anti-checklist |
| 6 | lit: rendezvous | What has worked elsewhere for RL phasing; classical phasing design numbers | annotated bibliography + transferable patterns |

**Shaping redesign candidates** (one will be chosen from recon evidence, implemented
behind a new `shaping_mode` env kwarg so legacy evals stay reproducible):

- **S-A — phase-first gate reorder:** Φ_phase ungated (σ=1); Φ_orbit gated *on* only when
  the phase gap is small; Φ_vel gated on both. Minimal change, keeps existing terms.
- **S-B — time-to-go potential:** Φ = −w·t̂_go from remaining gap + current drift rate.
  Smooth, no gates, but zero gradient for *opening* the drift orbit.
- **S-C — guidance-target potential:** replace φ_orbit's target Δa→0 with
  Δa → Δa*(Δθ) = clamp(−k·wrap(Δθ), ±Δa_max): the potential rewards being *on the
  correct drift orbit for the remaining gap*, ramping smoothly to Δa*=0 as the gap
  closes. Densest signal (rewards the opening burns themselves), no cliffs; this is a
  classical phasing guidance law expressed as a potential.

**Then:** implement chosen mode → 7-min smokes → training arms (fresh curriculum on
Discrete-16 / re-adapt the 33.5% ckpt / BC-from-expert if needed) → multi-seed winner →
re-run Lambert + relnav + e/phase sweeps → registry + docs.

**Guardrails** (lessons from this project's own history):
- Every reported number comes from `eval_checkpoint.py`'s *Physical success* line
  (terminal-cause classifier), 200 held-out episodes, greedy, seed disclosed.
- One variable at a time between arms; every arm's exact command logged here.
- Legacy shaping stays the default kwarg so pre-fix evals remain bit-reproducible.
- Clean atomic commits; anything experimental stays in `scripts/orbital/t3/`.

## 4. Recon findings (2026-08-10, six parallel agents; full reports in `scripts/orbital/t3/reports/`)

The six reports were produced independently and converge. Every quantitative claim below
was measured against the real C env (not derived), and the raw CSVs are in
`web_data/results/t3_*.csv`.

### 4.1 The physics is clean — the fix is complete ✅

980 episodes / 618k sub-steps of adversarial fuzzing against the independent f&g oracle
(shares no code with the env's propagator): **every hard-dynamics invariant PASSES at the
float32 logging noise floor** — step-local re-sim max error 0.68 m (old bug: ~700 km),
energy/momentum residuals at z < 1.3 vs threshold 20, `true_to_mean` now *bit-exact*
against the correct inverse over 20k draws, whole-episode chained re-sim inside the ±1-ULP
divergence envelope. Targeted stress of the exact regime the old bug corrupted (burns at
|sin θ| > 0.7, e ∈ [0.02, 0.12], n = 12,088) is clean. **H4 physics-side: no more dynamics
bugs.** (`recon_physics_fuzz.md`)

### 4.2 The task is feasible — decisively (H4 ✅)

A **scripted classical GNC controller** (`scripts/orbital/t3/expert_controller.py`;
obs-only, no ground-truth access, only the env's 16 discrete actions) scores
**99.2% (496/500, 4 seeds)** at headline conditions under corrected dynamics, at a median
189 m/s = 40% of the fuel budget (0.99× its own analytic Hohmann+phasing plan — burn
quantization costs ~1%). All failures are safety-cap timeouts. Success is flat across
phase gap (98–100% in every 45° bin) and eccentricity bin. Analytic MC over 200k headline
draws agrees: **99.0% of episodes are jointly Δv- and horizon-feasible.**
Post-fix RL failure is therefore **purely a learning-signal problem**.
(`recon_expert_baseline.md`, `recon_feasibility.md`)

Load-bearing premise correction: **MAX_STEPS = 2000 counts 60-s sim sub-steps, not
decisions** — every episode has a hard 33.3 h wall clock regardless of warping; warps buy
*decisions*, not time. The fuel-optimal mission uses ~93% of that clock (p90 1884/2000 —
6% margin). The expert's action mix: 42.8% warps, 40.8% coast, 16.4% burns.

### 4.3 Why nothing trains: the shaping is quantitatively adversarial (H1 ✅, worse than hypothesized)

Measured on real trajectories (legacy shaping, γ_shape = γ^τ as shipped):

| finding | measured value |
|---|---|
| σ₂ phase-gate open during the expert's maneuver | **0.2% of sim steps (median episode)** — gate opens at \|Δa\| ≲ 26 km; every horizon-viable drift orbit needs ≥ 106 km |
| entry fee to open a 133 km drift orbit | −0.11 potential = **5.6× the entire available phase reward** (W_PHASE cap = 0.02) |
| net shaping over a *globally optimal* 180° episode | **+0.02** (γ_shape=1 view) — the signal is flat where the task lives |
| do-nothing "stall income" (γ^τ leak, per episode) | **+1.78** — coast/warp-forever harvests (1−γ)Σ\|Φ\| ∝ \|Δa\|·time |
| park-far-and-warp adversary (burn away, then wait) | **+2.59** — 2.2× more shaping than the optimal expert earns |
| drift *down* (the catch-up direction) vs stalling | **−4.63** (and 3/8 seeds reenter from a 300 km floor) |
| "circularize at 180° gap and freeze" local optimum | pays **+0.18 immediately**, then Δθ frozen forever → guaranteed −10; the shaping gradient points *into* an absorbing failure state |

Also: γ^τ shaping vs flat-γ GAE means the shaping is **not potential-based in the MDP the
optimizer actually solves** (Ng–Harada–Russell invariance formally broken), independent of
the gate problem.

### 4.4 Three latent coordinate bugs, all exposed by the dynamics fix (H3 ✅ and then some)

All from the same root: phase is represented in **true anomaly**, a per-body coordinate
measured from each orbit's own (ill-conditioned at e≈0) periapsis:

1. **ω-misdirection** — `Φ_phase = 0` places the chaser a **median 9,800 km** from the
   actual rendezvous point at headline conditions (ω's are independent uniforms when e>0).
   `obs[13]`'s *sign* disagrees with the physical gap on **39.3% of steps** while the LVLH
   channel obs[34] is simultaneously correct — the network is fed two contradictory phase
   channels.
2. **Burn-teleport of the phase coordinate** — `cartesian_to_elements` re-derives (ω, θ)
   after every burn; at e ≈ 0 a **1 m/s burn moves the shaping/obs phase coordinate by a
   median 86.5°** (θ+ω stays exact — pure representation artifact; direct successor to
   `true_to_mean` in kind). ~40% of the paid phase-shaping signal is this artifact.
3. **`init_phase_gap_max` is inert at e > 0** — it offsets *mean anomaly* per-perifocal-
   frame; with independent ω the realized physical gap is uniform ±180° **regardless of
   the knob**. Every phase-gap curriculum stage and surface-eval cell run at e > 0 since
   Phase 5 was actually the full ±180° task. (Phase 3's ladder ran at e=0 / same-orbit —
   genuinely staged.)

### 4.5 Literature verdicts (`recon_lit_rl_practice.md`, `recon_lit_rendezvous.md`)

- **Classical GNC solved our exact problem**: *target semi-major-axis augmentation*
  (Naasz 2002; Holt/Baresi/Armellin, Astrodynamics 2024; arXiv 2606.12108) — bias the
  commanded `a` by the phase error so element-driving controllers phase automatically.
  Our `φ_orbit` is the unaugmented version those papers say fails to converge in phase.
- **Nobody has published end-to-end deep RL closing ±180° co-orbital gaps over tens of
  revolutions with impulses** — the field is CW/terminal-approach dominated (≤3 orbits,
  ≤30 km). We are off the map; both the failure analysis and any success are reportable.
- Ng et al. 1999: best potential ≈ V*; γ_shape must match the optimizer's γ or be 1.
- Florensa et al.: distance-based shaping provably traps far-from-goal tasks in local
  optima (our F-2, measured). Reverse/start-state curriculum is the structural fix.
- If pure PPO stalls: **do not BC-init→PPO** (destroyed by first PG updates; twice
  empirically refuted in this project already). The working recipes are DAPG-style demo
  terms in the loss, JSRL guide roll-ins, or Kickstarting KL — all compatible with our
  99.2% scripted expert as the guide.
- LaFarge et al. 2020 (the best-documented astro-RL success): trained 60 agents and
  selected — seed variance is *known-normal* in this field; our seed-1337 failures are
  expected, multi-seed reporting is mandatory.

## 5. Design decision (adopted 2026-08-11, pending red-team)

**One principle: make the reward and the coordinates tell the truth about the physics,
change nothing else until that's proven.** Concretely, five runtime-flagged env changes
(legacy behavior stays the default everywhere so all historical evals remain reproducible):

| # | kwarg (default = legacy) | change when enabled |
|---|---|---|
| 1 | `shaping_mode=1` | **S-R3 "phase-time" potential** (the audit's measured winner): Φ = −[W_λ·\|Δλ\|/π + W_m·min(1, Δv_match/DV_REF)] with **mean longitude** λ = M+ω, Δv_match = ½·v_t·hypot(Δa/a_t, \|Δē\|); W_λ=1.0, W_m=0.35, DV_REF=300 m/s; **no gates**. Measured: drift leg +0.993 of +1.0 total, worst adverse step −0.041 (24× margin vs terminal), do-nothing exactly 0, park-far adversary +0.13 (was +2.59). |
| 2 | `shape_gamma=1.0` | γ_shape = 1 in the shaping delta (replaces γ^τ): kills the stall income exactly, restores telescoping. NHR clamp stays at physical terminals, stays *off* at the safety cap (bounded partial credit for near-miss timeouts is aligned; deviation from strict PBRS is deliberate and documented). |
| 3 | `phase_gap_mode=1` | `init_phase_gap` controls the **physical mean-longitude gap** (adds ω_s−ω_t to the target's M offset) — makes staged curricula real for e > 0. |
| 4 | `phase_obs_mode=1` | obs[13,14] = sin/cos(Δλ), obs[15,16] = sin/cos(λ_t) — the phase channels become teleport-free and sign-correct. Breaking for old checkpoints; from-scratch runs only. |
| 5 | `episode_cap_steps=3000` | runtime episode cap (compile buffer raised to 3000); default 2000 = legacy. Lifts the p90 headline margin from 6% to ~50%. |

**Training configuration:** Discrete(16) as shipped (warp-1hr gives 84.3% terminal-reward
visibility vs 28.9% legacy and ~150 orders of magnitude of exploration exponent; do NOT
train the legacy 10-logit head — its 2.4% credit window makes the 33.5% re-adapt plateau
*expected*). Trainer untouched: γ=0.995 flat per decision (this is what makes warps
attractive — semi-MDP γ^τ in GAE would collapse every action set to ~0.03 terminal
visibility), Muon/lr/entropy as shipped. One variable at a time from there.

**Campaign ladder (gate between rungs: ≥60% multi-seed, 3 seeds):**
Karpathy battery first (overfit-one-scenario must hit ~100%; do-nothing shaping must
measure ≈0 via `probe_shaping_leak.py` re-run), then L0 (e=0, ±30° *real* gaps) → L1
(e=0, ±180°) → L2 (= headline: e≤0.05, ±180°) → 5-seed final. Imitation arms (DAPG demo
term / JSRL roll-in from the expert) held in reserve — deployed only if pure PPO stalls,
as a separately-ablated arm. The wide-e goal (L3–L5: band climb with `de_max`/`da_max`
kwargs, longer warps, scaled caps) starts only after L2 reproduces.

### 5.1 Adversarial red-team of the design (2026-08-11) — verdict: SHIP WITH CHANGES ✅ (all adopted)

An independent Opus red-team attacked the §5 design with measured probes
(`scripts/orbital/t3/redteam/`, `web_data/results/t3_redteam_*.csv`). Its Python Φ replica
matched the C shaping to 3.6e-9, so its counterfactuals are exact. Findings and dispositions:

| severity | finding | disposition |
|---|---|---|
| **BLOCKER** | **The −10 timeout is the flatline mechanism the recon missed.** Under flat per-decision γ, warp-1hr-to-cap = −7.8 while coast-to-cap = −0.0 (γ³⁰⁰⁰≈0): warps only break even once success ≥ 46.5%, so PPO suppresses warps — the only granularity where the new shaping clears the entropy floor (3.5× at warp-1hr vs 0.29× at warp-5min on a 133 km drift) and the only route to terminal visibility — before the first success is ever sampled. | `cap_terminal_reward` kwarg; **0.0 for all T3 runs** (break-even → 0%). Legacy −10 default. |
| MAJOR | `phase_obs_mode=1` silently broke the expert's obs decoder (12/12 → 0/12) — would have poisoned every imitation arm. | Decoder fixed in both replica copies; `expert_controller.py --t3` regression: **100/100** at the exact T3 training config. |
| MAJOR | The clock is unobservable: bit-identical observations measured 600 vs 1800 sub-steps from the cap; even the 99.2% expert fails clock-blind. | obs[15] := remaining-time fraction, obs[16] := cos(ω_s−ω_t) (both replaced channels are rotation-only, zero task information). |
| MAJOR | Trajectory log always-on: 352 B × every sub-step, 1.01 GB at 1024 envs. | `log_enabled` gated on `traj_log_dir`. |
| MINOR | NHR clamp pays +0.54 mean / +1.21 max for dying *far* vs near. | Clamp-nowhere under `shaping_mode=1` (progress credit = Φ_T − Φ₀ for every episode, no refunds). |
| MINOR | Ballistic intercept (one burn + warps, never nulling) wins 7/19 gap cells at cap 3000 — covers most of ±30°. | L0 small-gap rung dropped as a *claim*; all gates report per-gap-bin success + residual rel-vel at capture. |
| MINOR | `min(1, Δv_match/300)` is 80.7% saturated at L3 inits. | L0–L2 clean (0/0/1.8%); raise `shape_dv_ref_ms`≈700 from L3 on. |
| cleared | λ-jump farming (the bug class that sank the project twice): worst 0.167°/burn, drift 262× more Δv-efficient, telescoping exact to 4.7e-8. `phase_gap_mode=1`: 0.000° error, KS ≤ 0.054 vs uniform, correct under rejection sampling. Warp-through-box capture fires correctly mid-warp (16/16). | — |

Sign convention (documented per red-team): `init_phase_gap`/`phase_gap_fixed` place the
**target ahead**: realized Δλ = λ_s − λ_t = **−gap**. The `>= 0` sentinel on
`phase_gap_fixed` means fixed negative gaps can't be requested; uniform-± sampling is
unaffected.

## 6. Training campaign log

All evals: `eval_checkpoint.py` *Physical success* line, greedy, 200 episodes unless noted.
T3 env flags for every arm: `--env.shaping-mode 1 --env.shape-gamma 1.0
--env.phase-gap-mode 1 --env.phase-obs-mode 1 --env.episode-cap-steps 3000
--env.cap-terminal-reward 0.0 --env.valid-init-only 1`. Trainer: shipped profile
(γ=0.995, λ=0.90, ent 0.01, Muon), Discrete(16), fresh nets.

| # | arm | config | result |
|---|---|---|---|
| S1 | Karpathy overfit-one | fixed 30° gap, e=0, same-orbit, 10M | **perf 1.000**, return ≈ 9.4 — pipeline sound (pre-red-team build; cap −10 didn't bite at this scale) |
| L1 | full gap, circular | same-orbit init, e=0, gap ±180° (physical), a ∈ 500–800 km, 50M fresh, seed 42 | rolling 1.000; **held-out greedy 200/200 (100.0%)**, seed 123, causes: success=200. Ckpt `puffer_orbital_178642016215/model_..._000382.pt`. First RL success under corrected dynamics — full gap distribution, zero failures. |
| L2 | **headline** | e≤0.05 both (independent ω), ±180°, LEO 300–800, 50M warm from L1, seed 42 | **200/200 (100.0%)** held-out greedy (seed 123); **200/200 stochastic**; **200/200 at legacy cap 2000**; **500/500 (seed 777)**. Ckpt `puffer_orbital_178642097817/model_..._000382.pt` — **the corrected-dynamics headline checkpoint** (pre-fix bug-assisted best: 97.5%). |

**Seed-42 headline characterization** (500 eps, `t3_headline_characterization.csv`):
100.0% in every gap bin (0–45/45–90/90–135/135–180°) and every e bin. Δv p50 = 235 m/s
(p90 325, max 445 of 478 budget), 1.58× the shape-only two-impulse lower bound (scripted
expert: 1.30× — the policy trades ~25% fuel for ~2× faster missions: steps p50 417–972 vs
the expert's ~1900). Capture at the box edge: distance p50 29.2 km, residual |v_rel| p50
34.8 m/s (box: 30 km / 50 m/s) — honest label: far-field rendezvous with partial velocity
nulling, not terminal capture (the 10 km/10 m/s tightened-box goal remains future work,
task #8).

**Physics-consistency (Lambert re-run, 200 eps, `t3_lambert_corrected.csv`):**
policy/two-impulse **time-matched median 1.14×** — the pre-fix impossibility signature
(0.31×, policy "beating" the optimum via burn-teleports) is resolved; ratios vs
*immediate* (0.33×) and *free-departure* (0.35×) two-impulse plans are below 1
legitimately (multi-impulse drift phasing classically undercuts two-impulse plans under
a deadline).

**Multi-seed batch 1 (seeds 7 and 1337 — the historical 0%-flatline seed):**
all four runs at **200/200 (100.0%)** held-out — both L1 dirs (`…176815`, `…177125`) and
both L2 dirs (`…258109`, `…259027`) evaluated independently. *Attribution caveat,
recorded honestly:* the two ladders ran concurrently and raced on latest-dir capture, so
each L2's warm-start source is ambiguous between the two (both-perfect) L1 checkpoints;
launch-order timestamp analysis gives L1 s7=`176815` / s1337=`177125`, L2 s7=`258109` /
s1337=`259027` (dir IDs are launch-time centisecond stamps; 3.1 s and 9.2 s ID gaps match
the observed launch offsets). Every run's own `--train.seed` is unambiguous from its
command line. Ladder script hardened (sequential + no-new-dir guard); batch 2
(20260423, 31415) runs sequentially.

**Zero-shot eccentricity curve, T3 canonical (fixed e both, 100 eps/cell, seed 123):**
e=0.00 → **100%**, 0.02 → **100%**, 0.04 → **100%**, 0.05 → 98% (training-distribution
edge), 0.06 → 71%, 0.07 → 56% (out-of-distribution rolloff; the pre-fix curve was
already at 14% by e=0.075), e=0.083 → *not measurable in-band* (perigee floor leaves a
~5 km sliver of the 300–800 km band; all inits gave-up). Wide-e capability beyond this
is the L3+ ladder's job (band climb + `de_max`/`da_max` sampling), not zero-shot.

**Relative-navigation re-validation (T3 port of the T2 EKF harness, 200 eps):**
**truth 100.0% = EKF-nominal 100.0%** at nav cadence 60 s (also 100.0% at 300 s); NEES
0.944 / NIS 0.951 in-bounds, settled position RMSE 7.5 m (pre-fix T2: 0.897 / ~570 m).
New finding worth reporting: the corrected policy's 1-hour warps mean a harness that
ties sensor updates to decision epochs starves the filter (1-hr blackouts → divergence
→ 50.5% closed-loop); decoupling nav rate from guidance rate (`--sensor-dt 60`) is both
the realistic model and the fix. Degraded-sensor curve at nav-60: 3×→100.0%, 10×→99.5%,
30×→96.0%, 100×→81.5%, 300×→61.0%. `t3_relnav_corrected.csv`, plots in `plots/relnav_t3*/`.

**Multi-seed batch 2 (sequential, clean attribution):** seed 20260423 — L1 `…360522`
200/200, L2 `…431536` 200/200; seed 31415 — L1 `…505036` 200/200, L2 `…576174` 200/200.

## 7. Final results — headline task ✅ SOLVED (2026-08-11)

**Under corrected dynamics, the headline task (LEO 300–800 km, e ≤ 0.05 both with
independent ω, physical phase gap ±180°, 30 km / 50 m/s box, 478 m/s budget) is solved
at 100.0% held-out by 5/5 training seeds:**

| train seed | L1 (e=0, ±180°) | L2 headline | ckpt (durable) |
|---|---|---|---|
| 42 | 200/200 | **200/200** + 200/200 stochastic + 200/200 cap-2000 + 500/500 s777 | `models/t3/seed42_L2_headline.pt` |
| 7 † | 200/200 | 200/200 | `models/t3/batch1_L2_a_258109.pt` |
| 1337 † (historical 0% seed) | 200/200 | 200/200 | `models/t3/batch1_L2_b_259027.pt` |
| 20260423 | 200/200 | 200/200 | `models/t3/seed20260423_L2_headline.pt` |
| 31415 | 200/200 | 200/200 | `models/t3/seed31415_L2_headline.pt` |

† batch-1 attribution by launch-stamp analysis (§6); every dir independently evaluated.

**2,900 held-out episodes, zero failures of any kind** (no timeout, collision, stranding,
or escape). Context: pre-fix bug-assisted best = 97.5% single-seed / 93.7% 5-seed mean;
post-fix pre-T3 plateau = 33.5%; scripted classical expert = 99.2%.

Validation stack, all lawful: Lambert time-matched 1.14× (was impossible 0.31×);
EKF closed-loop truth 100.0% = EKF-nominal 100.0% (NEES 0.944/NIS 0.951); zero-shot
e-curve 100/100/100/98/71/56 at e = 0→0.07; learned maneuver structure = classical
drift-orbit phasing (showcase: `plots/t3/ep_0000057.png`, 179.4° gap at 175 m/s).

### 7.1 Wide-eccentricity extension ✅ SOLVED (2026-08-11, seed 42)

**L3 (LEO-scale lineage, warm from the headline canonical):** 300–2000 km, e ≤ 0.15,
de_max 0.06, da_max 400 km → **200/200 (100.0%)** held-out, and **retains 200/200 at the
original headline**. Ckpt `models/t3/seed42_L3_widee.pt` (`…670436`).

**WL1→WL4 (wide-envelope lineage, fresh re-ladder under 8000-km obs scales
`obs_alt_scale_m=8e6, lvlh_scale_m=1.5e7, dv_ref=700` — obs semantics must be constant
within a lineage, so this family is separate from the LEO canonical):**

| rung | conditions | held-out | dir |
|---|---|---|---|
| WL1 | e=0, same-orbit, ±180°, 500–800 km, cap 3000, fresh | 200/200 | `…770764` |
| WL2 | e≤0.05 both, ±180°, 300–800 km | 200/200 | `…839460` |
| WL3 | e≤0.15, de 0.06, da 400 km, 300–2000 km | 200/200 | `…912348` |
| **WL4** | **e≤0.30, de 0.08, da 600 km, 300–8000 km, cap 6000 (100 h)** | **200/200** | `…988768` |

WL4 realized distribution (`t3_wl4_characterization.csv`): e_target p50 **0.140**, p90
0.261, max 0.293 (e_sat max 0.364) — genuinely eccentric orbits, not a nominal knob
(the old project's realized e mean was ≈ 0.028 behind an e_max=0.20 label). 100% in
every gap bin; Δv p50 270 m/s = 1.46× the two-impulse shape bound; capture residual
\|v_rel\| p50 22.2 m/s (improved velocity nulling vs the LEO policy's 34.8).
Ckpts `models/t3/seed42_WL{1..4}_wide.pt`. Honest scope notes: wide-envelope is
single-seed so far (the recipe's seed-robustness is established at LEO, 5/5); `de_max`
bounds the e-vector *mismatch* — free e-vector rotations at these e's are physically
outside the 478 m/s budget (recon feasibility §3.2), so this is the affordable-and-
interesting task, stated as such. MEO/GEO rungs (L5) would need 3 h/6 h warp actions
(Discrete-18) and are future work.

**The user-stated campaign goal — "rendezvous at wide range of eccentricity and phase
difference, working fully" — is met:** ±180° phase gaps at e up to 0.30 across
300–8000 km, 100% held-out, under verified-correct physics, with lawful comparators.
