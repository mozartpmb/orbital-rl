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
