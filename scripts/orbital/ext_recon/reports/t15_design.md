# T15 — the 7/7 re-mix: design, evidence, red-team

**Recommendation: design (A′) — warm from `t13b_anchor_final` (six skills at
95–100% in the T15 harness), teach W1 with an ACQUISITION anchor to
`w1nav_child`, defend TIGHT with the measured CLEAR-style anchor, K=0 global.
With one escalation pre-registered, because the acquisition regime is the part
we have never measured.**

---

## 0. Two claims checked before designing anything

**The K=0 claim was right for the wrong reason, and the reason matters.**
`nav_max_ticks=120` does NOT bind only on day-warps. It binds on every row with
tau > 120 min:

| row | tau | K=120 | K=0 |
|---|---|---|---|
| 16 | 180 min (3 h) | 120 ticks @ 90 s | 180 ticks @ 60 s |
| 17 | 360 min (6 h) | 120 ticks @ 180 s | 360 ticks @ 60 s |
| 30 | 1440 min (24 h) | 120 ticks @ 720 s | 1440 ticks @ 60 s |

28 of 31 rows are unaffected, but rows 16/17 are general warps every cell emits
(~26% of decisions at Discrete-30 per MAJOR-7). So K=0 *does* change the regime
for non-W1 cells, and every published number for the six skills was measured at
K=120.

**That made the six-skill floors unknown in the T15 harness — my own R2 gate's
exact scenario — so they were measured, not assumed:**

| cell | K=120 | K=0 |
|---|---|---|
| E0_j2 | 97.5 | 97.5 |
| E1_j2 | 100.0 | 100.0 |
| E2_j2 | 95.0 | **100.0** |
| E3_j2 | 95.0 | 95.0 |
| LONGRANGE | 100.0 | 100.0 |
| TIGHT_5k1 | 92.5 | **95.0** |

K=0 global is safe and marginally better. **No per-cell override is required.**

**And a finding that outruns W1.** The surrogate optimism measured on W1 is not
a W1 property — it is a *tick-cadence* property, and it applies to every cell:

| cell | K=120 | K=0 |
|---|---|---|
| E0_j2, 6 h arc | **+68.8%** optimistic | +12.5% |
| E3_j2, 6 h arc | **+66.7%** | +8.3% |

Every cell in the T11/T13/T13b lineage trained on a substantially fictional
acquisition signal — and still reached 92.5–100%. That is worth stating plainly
rather than burying: the six skills are real (they are measured in truth mode
too), but the *signal they trained against* was optimistic, and K=0 is the first
time this lineage trains on an honest one. It is a strict improvement, and it
also means T15 is not merely a re-mix — it is the first run of the whole family
on a validated acquisition signal.

---

## 1. The floors that decide the design

Bearings-only, C filter, seed 123 (W1 at 25 eps, others 40):

| cell | `t13b_anchor_final` | `w1nav_child` |
|---|---|---|
| E0_j2 | 97.5 | 25.0 |
| E1_j2 | 100.0 | 27.5 |
| E2_j2 | 95.0 | 15.0 |
| E3_j2 | 95.0 | **0.0** |
| LONGRANGE | 100.0 | 17.5 |
| TIGHT_5k1 | 92.5 | **0.0** |
| W1_driftwait | **0.0** | 96.0 |

The bootstrap law bites at *exactly zero* (Ray Interference's f_k(0)=0 saddle:
no reward-bearing data, no gradient). So the design question is not "how many
skills are missing" but **how many cells sit at exactly zero, and how hard is
each to lift**:

- **(A)** root `t13b`: **one** zero — W1.
- **(B)** root `w1nav_child`: **two** zeros — E3 and TIGHT — plus four cells at
  15–27.5%, which is the regime T13 already proved re-acquires at 1e-3
  (E2 29→98.5, E3 6.5→93.5).

(B) is better than it was framed — four of its six are not at the saddle. But it
still asks for two bootstraps instead of one, and it discards a root that
already holds six skills at 95–100% *in this exact harness* in order to rebuild
them. **(A) is the smaller bet.**

---

## 2. The real objection to (A), and why it is not fatal

The synthesis is unambiguous about state distribution:

> student-driven, **mandatory** … DAgger O(T·ε) vs offline BC's O(T²·ε) — at
> T=3000–22000 offline BC on teacher trajectories is the worst possible regime

Our anchor is offline replay-BC on a frozen teacher-state set. For **defense**
that is sound and I argued why in T13b: the student *starts as* the teacher, so
covariate shift at t=0 is exactly zero and grows only as fast as the anchor is
failing. **For acquisition that argument evaporates** — the student starts at
0.0/25 on W1, so the shift is large from step one, and W1 has the longest
episodes in the project (22000 cap, ~8 days). Design (A) therefore asks the
anchor to do the one job it has never done, on the single worst cell for it.

**Why it is still the right bet.** The anchor's job here is not to *learn the
W1 policy from offline data* — that is the O(T²·ε) claim. Its job is to get W1
off exactly zero. Once W1 succeeds even occasionally, the mixture's own reward
takes over and the covariate-shift argument reverts to the T13b case. The
saddle is an escape problem, not an imitation problem, and a dense signal that
is merely *approximately* right is sufficient to escape it.

**Pre-registered escalation, decided now rather than at 3 a.m.** If W1 has not
left zero by 25M steps, the offline regime was insufficient, and the answer is
DAgger-style refresh — rebuild the W1 anchor set from the *current* student with
`w1nav_child` labelling it, every N epochs. This was pre-registered in T13b's
red-team (i) as the remedy for exactly this effect. Cost measured below; it is
enabled by a flag, not a redesign.

---

## 3. Per-teacher lambda — why multi-teacher is load-bearing, not convenience

T15 asks one mechanism to do two jobs that the literature runs at
order-of-magnitude different weights:

| job | cell | teacher | lambda | evidence |
|---|---|---|---|---|
| ACQUISITION (kickstarting) | W1 | `w1nav_child` | **0.30 → 0.05** over 30M | Schmitt 2018 ~0.5 decayed; unmeasured here |
| DEFENSE (CLEAR) | TIGHT | `t13b_anchor_final` (= the root) | **0.02 constant** | T13b: tight 92.5 → 92.5 exactly |

A single global lambda cannot serve both: at 0.02 the acquisition teacher is too
quiet to escape the saddle; at 0.30 the defense teacher pins TIGHT to the root
and forbids the improvement the mixture might buy. Hence `anchor_specs`.

The defense teacher is the ROOT ITSELF — CLEAR's "the anchor is the init", which
is why it needs no decay and why CE(anchor‖anchor)=0 at step 0.

---

## 4. Red-team

**(i) The kickstart regime is unmeasured *here*.** True, and it is the
single largest risk. Mitigations: lambda is staged rather than guessed; the
escalation is pre-registered; and the failure is *cheap and legible* — a run
that ends at 6/7 has cost compute and produced the same artifact we already
have, not a corrupted one.

**(ii) W1's step share will exceed its episode share.** W1's cap is 22000
against 3000 for E0/E1/TIGHT. The synthesis warns that step-weighted draws let
long cells dominate ("22000-step cells outweigh the tight cell ~7x per
episode"). Weight is per-EPISODE here, so W1 at 0.15 buys far more than 15% of
gradient. Proposed weights below are set on the measured step share, not the
episode share.

**(iii) The defense anchor covers TIGHT only.** T13b's evidence is TIGHT-only,
and TIGHT is the cell that has collapsed before. Adding W1's five neighbours to
the defense set would be untested scope creep; the probe battery is what catches
a surprise elsewhere.

**(iv) Anchoring TIGHT to the root caps TIGHT at the root.** By construction at
lambda 0.02 this was measured NOT to bind (T13b tight ended exactly at root,
neither above nor below). If T15 wants TIGHT above 95 that is a different run.

**(v) Everything here trains on a signal the lineage has never used (K=0).**
The six-skill floors were re-measured at K=0 and hold, so the root transfers.
But the *training dynamics* at K=0 are new for the six cells. The probe battery
must watch them, not just W1.

---

## 5. Mixture

Starting from the consolidation weights that worked in T13b, with W1 added and
the wide cells shaved proportionally. Set on measured step share (§6).

## 6. Gates

Stage 0, in order: kernel rebuild → C-port fuzz + mutation battery →
`navc_preflight` (kwarg lint + C reachability) → **`root_gate` R1+R2 on every
checkpoint involved** (the first campaign to carry it: root, and both anchor
teachers, each against its own home cell) → anchors → T11 gates → multi-anchor
bit-inertness (T15-A1a/A1b) → 2M smoke with both anchors live.

---

## 7. Smoke evidence (2M, both anchors live, K=0, C kernel)

| cell | root floor (K=0) | after 2M |
|---|---|---|
| E0_j2 | 97.5 | **97.5** |
| E3_j2 | 95.0 | **95.0** |
| TIGHT_5k1 | 95.0 | 90.0 (−5pp, ≈1.5σ at n=40) |
| W1_driftwait | **0.0** | **4.0** |

**W1 left exactly zero inside 2M steps.** That is the mechanism the entire design
rests on — the f_k(0)=0 saddle escaped by a dense signal — demonstrated rather
than argued, and it is the first direct evidence in this project that an anchor
can serve ACQUISITION and not only defense.

The anchor CE fell 10.23 → 7.38 → 5.50 across the smoke, i.e. the student is
converging toward the W1 teacher while the six skills hold.

**Lambda calibration, corrected by measurement.** At λ_acq = 0.05 the anchor
term is 0.05 × 5.50 = 0.275 against a measured `policy_loss` of 0.017 — a 16×
dominance, well above the synthesis's "~1.0 relative to the policy loss". The
naive reading says lower it to ~0.003. The measurement says do not: at 0.05 the
six skills are intact after 2M and W1 has moved, which is the outcome the ratio
was a proxy for. **λ_acq = 0.05 → 0.01 over 30M ships.** TIGHT's −5pp is inside
noise but is the one number the probe battery must watch, because TIGHT is the
cell that has collapsed before.

## 8. ETA

Measured 10.8K SPS (C kernel, K=0, T15 mixture, both anchors):

| | |
|---|---|
| 100M train | 2.6 h |
| 150M train | 3.9 h |
| batteries (7 cells × native+truth, W1 at FULL eps) | ~1.5–2 h |
| **total** | **~4.5 h (100M) / ~6 h (150M)** |

Recommend **150M**: W1 is being acquired from zero, and T13's lesson was that
an under-trained wide cell reads as a capacity failure when it is a budget one.

## 9. Gate results (all green as prepped)

| gate | result |
|---|---|
| root_gate R1+R2 — `t13b_anchor_final` on TIGHT | PASS (wide 2.0400; 92.5% vs published 95) |
| root_gate R1+R2 — `w1nav_child` on W1 | PASS (wide 2.3066; 96.0% vs published 96) |
| T2a defense teacher zero against root at init | PASS (CE 0.000e+00) |
| T2b acquisition teacher zero against itself | PASS (CE 0.000e+00) |
| T2c acquisition teacher disagrees with root | PASS (CE **10.23**) |
| T3 per-teacher lambda schedules | PASS |
| T1a/T1b `anchor_specs` unset bit-identical | PASS |
| G10a/b/c T15 mixture weights, all seven drawn | (stage 0) |
| C-port fuzz + mutation, navc_preflight | (stage 0, inherited) |
| 2M smoke, both anchors | PASS (10.8K SPS, 0 tracebacks) |

---

# T15b — the pre-registered DAgger refresh: design and evidence

## Is the refresh even well-posed?

The escalation says "rebuild the W1 anchor from the student's distribution,
labelled by the teacher". That is supervision only if **the teacher is competent
where the student goes**. If the student has wandered somewhere the teacher is
also lost, its softmax is noise with a confident shape — bug-#13 in a new
costume. Measured before committing anything (600 windows, student =
`t15_remix_final`, teacher = `w1nav_child`):

| states | teacher entropy | max-prob | CE(teacher‖student) |
|---|---|---|---|
| teacher-visited (T15 set) | 0.476 | 0.875 | 0.703 |
| student-visited (all) | 0.695 | 0.779 | **1.636** |
| student-visited, SUCCESS eps | 0.714 | 0.779 | 1.289 |
| student-visited, FAILED eps | 0.698 | 0.778 | 1.650 |

**The teacher is not lost.** Entropy shifts +0.219 nats, nowhere near flat
(ln 31 = 3.434), and it stays committed (max-prob 0.78). The refresh is
well-posed, and it carries 2.3× more correction than the old set — which is the
entire point of DAgger: the correction lives where the student actually goes.

## Decisions taken

**State mix: AGGREGATE, not substitute.** 900 teacher-visited + 900
student-visited = 1800 windows. The D in DAgger is *aggregation* (Ross &
Bagnell); substituting would bet the run on the student distribution being
non-pathological, and aggregation costs nothing because the student already
largely agrees on teacher states (CE 0.703).

**λ_acq = 0.20, CONSTANT.** Aggregated-set CE ≈ 1.17, so λ×CE = 0.234 — matching
T15's *working* contribution of 0.275 (λ 0.05 × CE 5.50) against a policy loss
of 0.017. Constant, no wean: T15's W1 plateaued *post-wean*, so the wean is a
suspect and there is no reason to repeat it while the skill is still climbing.

## Red-team

**(i) DAgger oscillation / failure-scoping — not needed, and measurably so.**
The student-visited distribution is **already 96.7% failed-episode states**,
because failures run 359 decisions against 46 for successes: an 8× length
asymmetry that self-scopes. CE is also naturally *lower* on success states
(1.289 vs 1.650), so the anchor already corrects least where the student is
working. Explicit failure-scoping would move 96.7% → 100% while deleting exactly
the 3.3% that represents working behaviour. Rejected on evidence.

**(ii) Does the defense anchor need refreshing? NO — and the reason matters.**
CE(defense teacher ‖ current policy) on old anchor states 0.035, on freshly
collected states 0.042: ratio **1.19**, so the states are still representative
and a refresh would buy nothing. But the same measurement found something
sharper: **the TIGHT anchor is saturated.** Per-step agreement is already 0.04
while closed-loop performance differs by ~10pp (85.5 vs the teacher's 95). That
is the synthesis's "the tight skill partly lives in recurrent state that
per-step KL does not supervise", measured — small per-step deviations compound
over 278–623 decisions. **Raising λ_def has almost no headroom; only TIGHT's
reward share can recover it**, which is why TIGHT keeps 34.8% of steps.

**(iii) Re-mix vs W1-heavy interleave: re-mix, with weight compensation.**
An interleave would give W1 undiluted gradient, but alternating phases is
exactly the sequential-training seesaw this project has measured twice
(Progress & Compress needed EWC for it). The real problem interleaving was
meant to solve is better fixed directly: **W1's step share self-attenuates with
competence.** Its decisions/episode fell 536 → 177 as it reached 31.5%, so the
T15 weights that bought 61.4% of gradient now buy 34.5% — a fixed weight
delivers *decreasing* acquisition pressure exactly as the skill starts working.
That is a built-in brake and a candidate co-explanation for the 31.5 plateau.
Fix: raise W1's sampling weight rather than change the training shape.

## Mixture (t11_mixture=4), re-solved at the new root

| cell | weight | step share |
|---|---|---|
| W1_driftwait | 0.27 | **44.4%** |
| TIGHT_5k1 | 0.06 | **34.8%** |
| E3_j2 | 0.15 | 6.1% |
| E2_j2 | 0.16 | 5.2% |
| E0_j2 | 0.16 | 4.2% |
| E1_j2 | 0.12 | 3.1% |
| LONGRANGE | 0.08 | 2.2% |

Projected W1 share as it improves: 65.6% (32%) → 47.7% (75%) → 25.2% (100%).

## Gates

| gate | result |
|---|---|
| **dry-run `T15B_STAGES=none` under /bin/bash** | **rc=0** (caught 2 defects first) |
| D1 teacher not lost on the anchor set | PASS (H 0.476 vs flat 3.434) |
| D2 refresh adds correction | PASS (1.78× the old set) |
| D3 aggregation not substitution | PASS (1800 = 900+900) |
| T11 battery | 18/18 |
| T15 multi-anchor (T2/T3) | 4/4 |
| root_gate R1+R2 × 4 checkpoint/cell pairs | 2/2 each |

## ETA

75M at 10.8K SPS ≈ 1.9 h train + ~1.5–2 h batteries ≈ **4 h floor**; realistically
5–6 h, because W1 (the expensive cell at K=0) holds 44.4% of steps.

---

# T15c — DAgger iteration 2, and the question of whether to run it

## The residual 55% is ONE mechanism, and it is the clock

120 episodes on `t15b_dagger_final`, W1 cell, K=0:

| | |
|---|---|
| success | 57/120 = 47.5% (battery ref 45.0) |
| **safety_cap** | **63/120 — 100.0% of all failures** |
| collision / escape / stranded / gave_up | **0** |

And the two populations are cleanly bimodal:

| | successes | failures |
|---|---|---|
| decisions | 44 | 649 |
| cap consumed | 0.201 | **1.000** (p90 1.000) |
| fuel left | 0.1631 | **0.1688** |
| ever acquired | — | **100%** |

Failures **acquire the target, keep their fuel, and run out of clock.** Nothing
is being done wrong in the sense a controller can be blamed for; the policy has
one strategy that works fast when the geometry cooperates (20% of the cap) and
no fallback when it does not.

## But the cap is NOT a free structural knob — measured

The obvious reading is "raise the cap". It is wrong, and cheaply falsified —
same checkpoint, cap swept at eval:

| cap | W1 |
|---|---|
| 22000 | 43.3% |
| 33000 | **8.3%** |
| 44000 | **8.3%** |

More time makes it *worse*, because obs[15] is `(cap − step)/cap`. Raising the
cap rescales the clock the policy paces against, so a strategy welded to the old
normalization mis-times itself. **Any cap change requires retraining, not
re-evaluation** — which moves "raise the cap" from a cheap structural fix to a
full campaign with an OOD warm start.

## The iteration curve is bending, and here is the number

The supervision available to each refresh is CE(teacher ‖ student) on freshly
visited states:

| iteration | CE on student states | yield |
|---|---|---|
| 1 (31.5 → 45.0) | 1.636 | **+13.5 pp** |
| 2 (45.0 → ?) | **0.976** | +8 pp if yield tracks CE |

The D2 gate — "does the refresh add correction the old set lacked?" — is itself
converging on its own threshold: **1.78× at iteration 1, 1.47× at iteration 2,
against a 1.3× floor.** At iteration 3 it plausibly refuses the run, which is
the mechanism reporting its own exhaustion rather than a human deciding it.

The teacher is still competent where the student goes (H 0.782 vs flat 3.434),
so this is not label collapse. It is convergence: the student now agrees with
the teacher on most of what the teacher can express per-step, and the residual
lives in the part a per-step KL cannot reach — the same recurrent-state limit
already measured on TIGHT (per-step CE 0.035 while closed-loop differs 10pp).

## Decisions for iteration 2

- **Two-way aggregate, not three.** Teacher-visited (900) + newest student
  (900). Aggregation exists to prevent distribution collapse and the
  teacher-visited half already provides that; the T15-era student half is two
  generations old, shares the identical failure mode (100% safety_cap), and
  would dilute an already-scarce fresh correction by a third for no benefit.
- **λ_acq 0.20 constant** — unchanged; the contribution now runs lower simply
  because CE is lower, which is the correct behaviour for a saturating anchor.
- **Weights re-solved (t11_mixture=5).** W1's decisions/episode moved again and
  NON-monotonically: 536 (@0%) → 177 (@31.5%) → **307 (@45%)** — it fell as
  short successes appeared, then rose as failures lengthened (359 → 622). At
  the T15b weights that had pushed TIGHT's share from an intended 34.8% down to
  26.2%, and TIGHT's recovery depends entirely on reward share because its
  anchor is saturated. Restored: W1 45.9%, TIGHT 34.3%, wides 19.7%.

## Governance: the menu, with my reading

**The curve is bending, and I would not run iteration 3 after this one.**

1. **Run iteration 2** (this prep). Expected W1 ≈ 53%, six untouched, ~4–6 h.
   Buys a real step and, more valuably, a third point on the yield curve that
   turns "bending" from two points into a trend.
2. **Accept the shelf + mode table.** Already the honest deployment shape, and
   W1 has a 96% specialist. The generalist's W1 number is a *bonus*, not the
   product.
3. **Structural W1 change.** The failure mode says the cell, not the policy, is
   the binding constraint — but the cap sweep says that costs a retrain, and the
   fuel data says fuel is not the lever (0.169 left at failure). A structural
   change means redesigning the W1 cell's time budget and re-deriving its
   specialist, i.e. a new arc, not an iteration.

My recommendation is **(1) then (2)**: run iteration 2 because it is prepped,
cheap, and produces the decisive third data point; then stop iterating and
ship the mode table, unless iteration 2 beats +8pp materially — which would
falsify the bending read and justify continuing.

## Gates

| gate | result |
|---|---|
| **dry-run `T15C_STAGES=none` under /bin/bash** | **rc=0** |
| D1 teacher not lost | PASS (H 0.476 agg; 0.709 student-visited) |
| D2 refresh adds correction | PASS but **narrowing: 1.47× vs 1.30 floor** |
| D3 aggregation not substitution | PASS (1800 = 900+900) |
| T11 battery | 18/18 |
| root_gate R1+R2 (W1, TIGHT) | 2/2 each |

---

# T15d — iteration 3, and the arc's stopping rule

## My CE-yield model was wrong, and the falsification is the useful part

I predicted iteration 2 would buy ~+8pp because the available correction had
halved. It bought **+14.0**, matching iteration 1's +13.5. **CE magnitude does
not forecast yield.** Three generations:

| generation | CE ratio | novelty (NN to teacher set) | yield |
|---|---|---|---|
| iter1 @31.5 | 1.75 | 1.1969 | +13.5 |
| iter2 @45.0 | 1.34 | 1.1316 | **+14.0** |
| iter3 @59.0 | 1.39 | 1.1288 | ? |

**Novelty is flat where CE was not**, and flat novelty matches flat yield. The
mechanism that fits: each refresh reaches a genuinely new region of roughly
constant size and converts a roughly constant slice of it — constant returns,
not diminishing ones, for as long as new region remains.

## D2, re-derived from measurement — and NOT moved

The gate's original justification (more correction ⇒ bigger buy) is falsified.
What survives is the weak claim: ratio ≈ 1.0 means the refresh re-collected what
the student already handles.

**The floor stays at 1.30.** Observed working points are 1.75 → +13.5 and
1.34 → +14.0; observed failures: **none**. With no failure to calibrate against,
the data cannot justify a floor above the lowest ratio seen to work, and
lowering it now would be convenience rather than measurement — this iteration
measures **1.39 and passes as-is**. If a future iteration lands below 1.30 that
is *not* evidence it will fail; the honest response is to run it with the gate
downgraded to a warning, not to retune the number until it passes.

## Failure modes at 59%: no fragmentation, but the bend is now predictable

Still **100.0% safety_cap**, fuel intact (0.1745), acquisition 100%. What moved
is the headroom:

| root | W1 | success cap p50 | p90 | max |
|---|---|---|---|---|
| t15b @45 | 47.5% | 0.201 | 0.511 | 0.824 |
| t15c @59 | 56.7% | 0.321 | **0.698** | **0.983** |

Each iteration converts progressively **slower** geometries — successes now
*touch* the cap. The conversion frontier advances ~0.19/iteration, so it reaches
1.0 after roughly **iteration 4**. That is where the curve must bend, and it is
a *leading* indicator: visible one iteration before the yield collapses.

## Stopping rule — counter-proposal

Accept the proposed rule and **add a leading trigger**:

> Stop at the first of: (a) yield < +7pp, (b) W1 ≥ 85, or
> **(c) success cap-usage p90 ≥ 0.95.**

(a) and (b) are lagging — they cost a full 6-hour run to observe. (c) is
measurable from the *current* checkpoint in ~7 minutes and predicts the bend
before the run that would have hit it. Two independent estimates agree on where
this lands: the headroom extrapolation exhausts after ~iteration 4, and the
trajectory 31.5 → 45 → 59 → ~73 → ~85 reaches the (b) threshold at the same
point, against a specialist ceiling of 91.

## Mechanics

- **Two-way aggregation retained** — the 59% failure modes did not fragment, so
  the argument that decided it at iteration 2 is unchanged.
- **Mixture 5 reused, weights unchanged** — and re-verified on the right
  statistic. dec/ep median swung 307 → 63, which is an artifact: the episode
  distribution is bimodal (success 42, fail 608) and the median flips mode as
  success crosses 50%. Total steps depend on the **mean**, which moved 347 → 290,
  at which the shipped weights give W1 44.5 / TIGHT 35.3 / wides 20.2 against
  targets 45.9 / 34.3 / 19.7. No re-solve needed.
- Gates: dry-run **rc=0**, D1/D2/D3 3/3, T11 18/18, root_gate 2/2 on both cells.
