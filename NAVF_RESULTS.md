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

_(remaining arms fill in as they land)_
