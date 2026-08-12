# NAV-F — the dual-control experiment at the tight terminal box

**Question.** At the tight box, angles-only observability collapses *exactly as
the guidance objective succeeds*: nulling relative velocity to `v_tol` leaves
`δa = v_tol/(1.5n)`, and drift-only bearings-only range error scales inversely
with δa. At TB5 (5 km / 1 m/s) that is a 2589 m σ_range against a 5000 m box —
52% of tolerance — while one 1 m/s burn buys ×1.4e4 information
(`nav_F_observability.md` verdict table). Does training on bearings-only
estimates therefore produce **information-seeking burns**?

**Why this box and not the headline box.** At the T3 headline box
(30 km / 50 m/s) drift-only nav error is 1.8% of tolerance, so nav error can
never cost the agent reward and the experiment is guaranteed null for trivial
reasons. NAV-F §3.4 calls running at the wrong box "the single highest-risk
design error available."

**Why presence-of-burn is not the metric.** NAV-F §3.1 measured the TB5 policy
already burning on ~29% of decisions inside 10 km for pure guidance reasons.
Only placement, timing, direction and counterfactual information can detect the
effect.

## Setup

All arms: 50M steps on `puffer_orbital_nav`, trained at **TB5**
(`rendezvous_radius_m=5000, rel_vel_tol_ms=1`), LEO headline distribution,
**Discrete-20**, nav60 cadence, 8 workers × 256 envs, `caffeinate -is`.

All arms warm-start from **the same file**,
`models/t3/extnav_TB5_warmstart_col21zero.pt` — the main checkout's
`seed42_TB5_box5k1.pt` with encoder column 21 zeroed. obs[21..32] are
identically zero in every nav/T3 config, so those encoder columns received zero
gradient and still held random init; zeroing column 21 makes the T-BO+Σ channel
provably inert at t=0, and using the same file for every arm makes all four
behaviourally identical at t=0.

All arms evaluated **under bearings-only with the real batch acquisition**
(`bls_acquire_adaptive`), 200 episodes, held-out seed 123, at **both** boxes,
plus a truth control at the same box. Episode *i* is the same scenario across
arms (identical env seeding), which is what makes the paired Δv comparison valid.

### Primary metric — counterfactual information gain

At every decision epoch, accumulate the exact bearings-only Fisher information
over a 4 h arc (1 h before the epoch, 3 h after) under (a) a standardised 1 m/s
prograde probe burn applied **at the epoch** and (b) coast, and take
`G_avail = (σ_coast/σ_burn)²` — the information a burn *would* buy in that
state. Then regress the binary "did the policy burn here" on `log10 G_avail`,
univariate and **adjusted** for (ρ, |δa|, time-to-go, Δv-left).

Burn placement is load-bearing, not a detail: a probe at the *arc start*
measures G ≈ 1.02 for every state — that is NAV-F §2.5's own result (1.06 at
burn_frac 0.0 vs 1758 at 0.20), because a burn at the arc start merely redefines
the initial state. An implementation that gets this wrong returns a confident,
entirely artifactual null.

Validated against NAV-F §2.3's published table (1 m/s, 90 min, co-orbital):

| separation | this implementation | NAV-F |
|---|---|---|
| 2 km | 8.33e4 | 1.1e5 |
| 5 km | 1.88e4 | 3.4e4 |
| 10 km | 5.81e3 | 1.1e4 |
| 30 km | 7.32e2 | 1.81e3 |
| 300 km | 1.44 | 1.56 |
| 1000 km | 0.951 | 1.00 |

Agreement within 1.3–2.5× over six decades; the residual is expected — the
report takes best-of-{radial, prograde} and this uses prograde only.

Statistics are computed on **steady state only** (post-acquisition);
acquisition-phase burns are a separate population (NAV-G: 44–114 min) and would
otherwise contaminate the estimate.

## Arms

| arm | TB5 native | TB5 truth | TB4 native | primary β (adj, inside 10 km) | verdict |
|---|---|---|---|---|---|
| **T-truth** (control, eval-only) | 102/200 = 51.0% | 191/200 = 95.5% | 101/200 = 50.5% | **+0.016 (z +0.21)** | null, as designed |
| **T-RB** (control, trained on range+bearing EKF estimates) | 100/200 = 50.0% | 196/200 = 98.0% | 99/200 = 49.5% | **−0.103 (z −1.34)** | null — estimates per se teach nothing |
| **T-BO** (trained on bearings-only estimates) | **194/200 = 97.0%** | 194/200 = 97.0% | **198/200 = 99.0%** | **−0.022 (z −0.41)** | capability +46 pp; info-seeking null |
| **T-BO+Σ** (T-BO + explicit σ_LOS/ρ channel) | 194/200 = 97.0% | 197/200 = 98.5% | 197/200 = 98.5% | **−0.032 (z −0.62)** | σ channel adds nothing |
| **T-BO−act** (T-BO, fine burns blocked < 10 km) | 185/200 = 92.5% | 188/200 = 94.0% | 196/200 = 98.0% | **−0.434 (z −8.25)** | fine burns are guidance-critical, not info-critical |

### T-truth — the control and the zero-shot floor

The shipped TB5 checkpoint, never trained on an estimate.

| metric | TB5 | TB4 |
|---|---|---|
| native (bearings-only, real acquisition) | **102/200 = 51.0%** | 101/200 = 50.5% |
| causes | collision 48, stranded 47, cap 3 | collision 54, stranded 43, cap 2 |
| truth | 191/200 = 95.5% | — |
| never acquiring | 56/200 | 64/200 |
| timeout rate (pre-registered confound) | 1.5% | 1.0% |
| Δv before acquisition | median 138.0 m/s | 122.5 m/s |
| burn rate blind ÷ acquired | 1.75× | 1.66× |
| total Δv/episode | median 383.5 m/s | 351.0 m/s |

**Counterfactual regression (TB5, steady state):**

| pool | n | β univariate | z | β adjusted | z |
|---|---|---|---|---|---|
| inside 10 km | 692 | −0.169 | −2.45 | **+0.016** | **+0.21** |
| all steady state | 6931 | −0.288 | −12.14 | −0.110 | −2.78 |

- `G_avail` median **at burns 1.01**, **at coasts 3.24** — the truth-trained
  policy's burns land in *low*-information states. Median `log10 G_avail` inside
  10 km is 2.563 (G ≈ 366), so the information is there; the policy is simply
  not responding to it.
- Direction: median **119.8°** from LOS, 23.4% in the 45–135° (near-perpendicular,
  informative) band, 26.7% fine burns.
- Δlog₁₀ tr(P) over 3 decisions: after burn **−0.303**, after coast **−0.531**.
- Action mix inside 10 km: warp-1h 66.9%, retro-2 12.7%, pro-2 12.1%,
  radial-out-10 3.2% — reproducing NAV-F §3.1's measured mix, and confirming
  actions 18/19 (radial ±1 m/s) remain unused.

This is the intended control result: **after adjusting for state, the
truth-trained policy shows no information-seeking whatsoever.**

### T-RB — the load-bearing control

50M steps warm-started from the shared file, trained on **range+bearing EKF
estimates** (fully observable — the estimate is always good during training),
evaluated under bearings-only like every arm. If "training on an estimated
state" were itself the active ingredient, this arm would improve. It does not,
on any metric:

| metric | T-RB TB5 | T-truth TB5 |
|---|---|---|
| native | 100/200 = **50.0%** | 51.0% |
| causes | collision 52, stranded 44, cap 4 | collision 48, stranded 47, cap 3 |
| truth | 196/200 = 98.0% | 95.5% |
| burn rate blind ÷ acquired | 1.82× (0.870/0.477) | 1.75× |
| total Δv/episode | median 375.5 m/s | 383.5 m/s |
| β adjusted, inside 10 km | −0.103 (z −1.34) | +0.016 (z +0.21) |
| `G_avail` at burns / coasts | 1.02 / 4.06 | 1.01 / 3.24 |

Every number is within noise of the untrained control. **Whatever T-BO does is
attributable to the bearings-only observability structure being present during
training, not to estimate noise, filter dynamics, or the extra 50M steps.**

### T-BO — the headline arm

50M steps trained on bearings-only estimates (CRLB-conditioned surrogate at
training, real `bls_acquire_adaptive` at eval).

| metric | TB5 | TB4 |
|---|---|---|
| native | **194/200 = 97.0%** | **198/200 = 99.0%** |
| truth (same ckpt) | 194/200 = 97.0% — **zero truth-tax** | 200/200 = 100.0% |
| causes | success 194, cap 6 | success 198, cap 2 |
| collisions / strandings | **0 / 0** | 0 / 0 |
| Δv before acquisition | median **0.0 m/s** (p90 25) | 0.0 m/s (p90 25) |
| burn rate blind ÷ acquired | **0.37×** (0.185/0.494) | 0.31× |
| total Δv/episode | median **243.0 m/s** | 236.5 m/s |
| β adjusted, inside 10 km | −0.022 (z −0.41) | −0.058 (z −0.76) |
| `G_avail` at burns / coasts | 1.11 / 18.4 | 1.02 / 1.76 |
| Δlog₁₀ tr(P)/3 dec: burn / coast | −0.31 / −0.50 | −0.36 / −0.74 |

Versus the controls: **+46 pp native capability, 37% less Δv, and the
blind-behaviour ratio inverts** — controls burn 1.7–1.8× *more* while blind
(thrashing on a bad estimate); T-BO burns 3× *less* while blind, spends
median zero fuel before acquisition, then flies the transfer on a good
estimate. The 10-episode smoke's β = +0.436 did not survive 200 episodes:
the pre-registered counterfactual regression is null after adjustment, and
significantly *negative* unadjusted (−0.24, z −5.3) — burns land in
low-information states, coasts in high-information ones.

### T-BO+Σ — explicit uncertainty channel

T-BO plus obs[21] = log-scaled σ_LOS/ρ (encoder column zeroed at warm-start so
the channel is provably inert at t=0). Result: statistically identical to T-BO
everywhere — native 97.0%/98.5%, β adjusted −0.032 (z −0.62), same blind-burn
suppression (0.29×), same Δv (244.5 m/s). **The policy already extracts
everything it needs from the estimated-state stream; explicit uncertainty is
not load-bearing.**

### T-BO−act — fine burns blocked below 10 km

Env-variant dynamics: actions 12–15 (±1/±2 m/s) unavailable inside 10 km
separation, forcing the close-range repertoire onto coarse burns. Tests whether
the controls' failure was really a missing *information* skill or a missing
*guidance* repertoire.

| metric | TB5 | TB4 |
|---|---|---|
| native | 185/200 = **92.5%** | 196/200 = 98.0% |
| truth (same ckpt) | 188/200 = **94.0%** | 200/200 = 100.0% |
| causes (native) | success 185, **cap 15** | success 196, cap 3, stranded 1 |
| total Δv/episode | median 261.0 m/s (highest of BO arms) | 252.0 m/s |
| β adjusted, inside 10 km | **−0.434 (z −8.25)** | −0.195 (z −2.76) |

Two reads, both clean: (1) the −4.5 pp deficit at TB5 appears in **truth mode
too** (94.0% vs 97.0%) with timeout as the failure mode — so fine burns are a
*guidance* requirement of the 1 m/s velocity box, not an information
requirement; (2) the strongly negative adjusted β shows that when only coarse
burns exist, burning is even more anticorrelated with information gain —
the opposite of what a dual-control policy would produce.

## Verdict

**Observability-in-the-loop training closes the capability gap; dual control is
affirmatively absent — and the physics says that is the correct policy.**

1. **Capability.** Training with bearings-only estimation in the loop takes the
   tight-box task from 51% → **97.0%** (TB5) and 99.0% (TB4), with zero
   collisions, zero strandings, zero truth-tax, and 37% less Δv than the
   controls. The T-RB control pins the attribution: estimates alone teach
   nothing (50.0%); the observability structure is the active ingredient.
2. **No information-seeking burns.** The pre-registered counterfactual metric
   is null after covariate adjustment in every BO arm (|z| < 1). Unadjusted,
   it is significantly negative: burns happen in low-`G_avail` states, coasts
   in high-`G_avail` states.
3. **The null is physically right.** At these ranges the filter gains
   information faster by coasting (Δlog₁₀ tr(P) −0.50/3 decisions) than
   through a burn (−0.31): a maneuver buys observability geometry but resets
   drift-information accumulation and injects process noise. `G_avail` at the
   states where the policy actually burns is ≈ 1. An agent that probed would
   be wasting fuel and information; the trained agent correctly does not.
4. **The learned skill is passive information management**: suppress burns 3×
   while blind (controls *raise* them 1.75×), spend zero fuel before
   acquisition (controls spend ~130 m/s), wait out the ~105 min acquisition
   latency, then fly the transfer on a converged estimate.
5. **Fine burns are guidance-critical, not info-critical** (T-BO−act): removing
   them costs 4.5 pp even with perfect truth state, via cap-timeout at the
   1 m/s velocity box.

Caveats: single training seed (42) per arm; both boxes share eval seed 123
(paired scenarios across arms — valid for arm-vs-arm deltas, and consistent
with every earlier NAV table); `G_avail` uses a prograde-only probe
(1.3–2.5× conservative vs best-direction).

Checkpoints: `models/t3/extnav_navf_{T_BO,T_BOS,T_BOACT}.pt`; per-arm JSON in
`web_data/results/navf/`; wandb groups `t5-navf-T_{BO,BOS,BOACT}` in
`orbital-rl`.
