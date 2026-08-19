# Campaign index — T5 → T11

> **Status:** current as of 2026-08-18. This is the routing table for eleven
> campaigns spread across nine documents. Start here.
>
> Read in this order for the short version: **T11** (the generalist, and the
> transplant discovery), **GEN_MATRIX** (why a generalist was buildable),
> **N3DNAV** (the navigation flagship), **J2_RESULTS** (three campaigns in one
> file). Checkpoint registry: `MODELS.md`. Predecessor: `T5_EXTENSIONS_CAMPAIGN.md`.

Three facts this index exists to carry, because no single campaign doc can:

1. **T6 (the TB-3D tight-box ladder) has no document of its own.** Its results
   live only in `scripts/orbital/t3/t6_tb3d.sh` and in the lineage tables of
   `N3DNAV_RESULTS.md`. The entry below is the only prose description of it.
2. **Two campaigns have their prep in one file and their result in another** —
   M10 (`M10_NOTES.md` → `N3DNAV_RESULTS.md`) and j2wait (`J2WAIT_NOTES.md` →
   `J2_RESULTS.md` §3). `J2_RESULTS.md` contains **three** campaigns.
3. **The bug ledger is sparse, not contiguous.** The repo numbers the 6th, 8th,
   9th, 11th, 12th, 13th and #15. **There is no 10th and no 14th** — do not
   write one. The strongest unnumbered catch is M10's, described below.

---

## 1. T5 extensions — embedded C, full 3D, angles-only nav (`ext-3d`, `ext-nav`)

2026-08-11 → 08-12 · **`T5_EXTENSIONS_CAMPAIGN.md`**, **`EXTNAV_RESULTS.md`**

Three parallel extensions. Embedded C port at 100.0000% action agreement over
7,048 obs vectors and 200/200 fully-C closed loop (p50 50.5 µs/step). V-ladder
wide-3D **200/200, 200/200, 200/200, 199/200** (V2–V5) at e ≤ 0.30, 300–8,000 km,
ΔI ≤ 0.75°, ±180°. Bearings-only NB arms **98.0 / 99.0 / 99.5 / 98.5%**, pooled
fresh 594/600 = 99.0%, **0.0 pp truth tax**.

**Contributed the 11th metric-vs-implementation bug:** `de_max` was inert under
`dim3_mode` — the e-disc is drawn node-relative and the plane rotation
randomised RAAN, destroying the bound in inertial space (realized mismatch 4.5–8×
the knob). Caught by a control experiment (`di_max=0` → 199/200), fixed in two
lines by preserving inertial ϖ. Also **replication #2 of the bootstrap /
value-collapse mechanism** (W4: textbook negative transfer under ~57% infeasible
mass, the trained child worse than its untrained parent).

## 2. NAV-F — is there dual control at the tight box? (`ext-nav`)

2026-08-12 · **`NAVF_RESULTS.md`**

Five arms × 200 eps at TB5 (5 km / 1 m/s): **T-BO 194/200 = 97.0% native against
T-truth 51.0% and T-RB 50.0%** — observability in the loop closes the gap, and
**dual control is affirmatively absent** (adjusted β null, |z| < 1, in every BO
arm). T-RB is the load-bearing null: estimates per se teach nothing. T-BO−act at
92.5% native / 94.0% truth (β_adj −0.434, z −8.25) shows the fine burns are
**guidance-critical, not information-critical** — the anti-dual-control
signature.

## 3. T6 — the TB-3D tight-box ladder *(no doc of its own)*

2026-08-12 · commits `ca722d8`, `1889358` · script `scripts/orbital/t3/t6_tb3d.sh`

3D (ΔI ≤ 1.0°) × tightening box, warm from X3, seed 42: **199/200 @ 10 km/10 m/s,
199/200 @ 5 km/2 m/s, 194/200 = 97.0% @ 5 km/1 m/s** — matching the 2D result
(95.5%) despite the out-of-plane tolerance sitting on the 1 m/s normal-burn
actuation floor. Its checkpoints are the parent lineage for everything in §4.

## 4. n3dnav — 3D + bearings-only navigation (`ext-3dnav`)

2026-08-12 → 08-15 · **`N3DNAV_RESULTS.md`**

The navigation flagship. **Rung 1: 200/200 = 100.0% on real batch IOD from a
58.0% floor**, zero truth-tax both directions, 3/3 seeds. **Rung 2 (tight box):
196/200 = 98.0%** from a 46.0% floor, pooled 589/600 = 98.2% across 3 seeds.
**Rung 3 (e-ladder): 95.0–100%** bearings-only to realized e p90 0.257.

Opened with a red-team ledger of **4 BLOCKER / 13 MAJOR / 2 NON-ISSUE / 4 NOTE**
(`n3d_REDTEAM.md`), all BLOCKERs closed before any training ran. Contributed the
**negative-transfer control, measured twice**: N1-rb3d is lossless in-mode
(99–100%) but scores **13.0% / 18.0% flown blind — below its untrained floor**
(46.0% / 53.5%). Training on always-reliable estimates actively unlearns
blind-window caution. Also established "navigation is over-solved, guidance
binds": in-box estimator velocity error 0.001 m/s against a 1 m/s tolerance.

## 5. ext-j2 — the propagator (`ext-j2`)

2026-08-13 · **`J2_DESIGN_NOTES.md`**

Secular mean-element J2 behind `j2_mode` (default 0). **C gate battery 22/22 plus
a 14/14 Python ladder**, scope deliberately narrow: physics, kwarg, obs,
validation — no training.

## 6. ext-j2 rung — tight-box recovery under J2 (`ext-j2-rung`)

2026-08-13 → 08-14 · **`J2_RESULTS.md` §1**, `J2_RUNG_NOTES.md`, `J2_RECON_NOTES.md`

**0/200 → 198/200 = 99.0% at 5 km / 1 m/s** under J2 (mean elements); multi-seed
**595/600 = 99.2%**, 0.5 pp spread.

**Contributed the 12th bug:** the inertial-ϖ correction — itself the fix for the
11th — subtracted the chaser's *absolute* RAAN, correct only in the pinned-node
gauge every prior lineage happened to use. Under target-plane sampling the e-disc
knob silently became **5.55× its setting**. Also caught that **the LVLH
observation is only LVLH at equatorial targets**: frame error to **238.7%** on
the primary rendezvous channels at 30–60° inclination. Fixing it alone moves
zero-shot 56.5% → 90.0%; without it the measured J2 gap would have read −5.5 pp
instead of −10.5 pp, *halving the apparent effect while tripling the noise*.
One defect was **found and deliberately not fixed**: `obs[17-20]` is the chaser's
absolute inertial longitude, capping inclined-target capability above ~60°;
consequence is a design rule (`raan_target_sample=0`) at a measured cost of
79.5% → 69.0%.

## 7. j2nav — J2 × angles-only navigation

2026-08-14 → 08-15 · **`J2_RESULTS.md` §2**

**192/200 = 96.0% bearings-only with real batch IOD in the J2-perturbed inclined
world, from a 32.0% floor**; truth 200/200, so the 4 pp gap is pure navigation.

**Contributed the 13th bug:** `OrbitalNav` **sub-propagated truth states two-body
under a J2 env** — the filter tracked a fiction with perfect self-consistency.
The fix restored **in-loop NEES from 25,520 to 1.20–1.37**. The generalizable
lesson is the one worth carrying: *standalone filter-vs-filter validation is
structurally blind to a harness that is wrong the same way for both arms.*
Honest cost stated rather than buried: X3-home retention drops to **83.0%,
−17 pp** — real forgetting.

## 8. j2wait — drift-and-wait

2026-08-15 · **`J2WAIT_NOTES.md`** (prep) → **`J2_RESULTS.md` §3** (result)

Does a policy trade time for fuel? **192/200 = 96.0% from a 0/200 floor** at
node-dominant plane gaps, with **51.3% of plane alignment bought from J2
precession** — 412 m/s against a 500 m/s direct reference that *exceeds the
478 m/s budget*. 8.1-day episodes, 7 day-warps each.

**Blocker caught by measurement before launch: a zero-init appended row is
unreachable.** The warm start is saturated (median argmax softmax 0.986), so a
zero-init row 30 draws **P = 3.9e−9 → 0 expected samples in 100M decisions** —
the mechanism under test would silently never have been available, producing a
null indistinguishable from a real one. Seeded from row 17 instead. Established
the **append-don't-rebind** rule (rebinding row 17 rejected on evidence: 11.1% of
decisions, 87.4% of substeps) and the **head-mismatch discipline** — mask a row
at sampling, *never* shrink a trained head — later cited by M10, GEN_MATRIX and
T11. Honest cost: home rung collapses to 46.0%.

## 9. M10 — the normal-axis ablation

2026-08-15 · **`M10_NOTES.md`** (prep) → **`N3DNAV_RESULTS.md` §MAJOR-10** (result)

The fine normal axis (rows 20/21, ±1 m/s) is **guidance-critical, not
informational**: masked zero-shot at TB5-3D drops 98.0% → 51.0% native and
99.5% → 56.0% truth, with cross-track residual 2.7–3.1 m/s against a 1 m/s box
*given perfect information*. Closes the n3d_REDTEAM ledger entirely.

**The strongest unnumbered instrument catch in the project.** The pre-existing
`nav_block_fine_below_m` interlock **silently no-ops in truth mode** (`step()`
returns before `_nav_alloc`, so the block array is never allocated): truth 12/12
with the interlock versus 4/12 with a real ablation. Using it would have
fabricated a "no truth deficit" reading *in the one arm where the
guidance-vs-information question is decided* — MAJOR-10 would have been closed
with precisely the misreading it was written to prevent, inverted. Replaced with
an unconditional `nav_ablate_rows`. Also **replication #3 of the bootstrap
mechanism**: trained-ablated arms collapse identically under bearings-only *and*
truth (rolling perf 0.000 / 0.003).

## 10. T11 — the generalist (`ext-t11`)

2026-08-15 → 08-18 · **`T11_GENERALIST.md`**

**One policy at 97–100% per cell across J2 × the full eccentricity range ×
bearings-only × long-range, in 5 of 7 mixture cells**, 200 eps/cell at held-out
seed 123 with real batch IOD, uniform ~1.28× fuel efficiency, and graded
budget-awareness (88–97% on a 353 m/s lean tank → 99–100% on a 656 m/s rich one).

**THE TRANSPLANT DISCOVERY.** GEN_MATRIX's 99-pp normalizer barrier is a
*parameterization artifact, removable exactly*. `Default.encode_observations` is
`Sequential(Linear(38,128), GELU)` on the **raw** obs vector, so if a normalizer
multiplies channel *j* by *c_j*, dividing `encoder.0.weight[:, j]` by *c_j*
leaves the pre-activation — and hence the GELU output, LSTM state, value head and
logits — unchanged. Seven columns move between families. Verified **bit-identically**:
A3b-j2 home 91/100 md5 `e81c41eb204c`; wide without transplant **1/100**; wide
**transplanted 91/100, md5 `e81c41eb204c`**. Not "recovers most of it" — *the same
policy*. As `T11_GENERALIST.md` puts it: **the matrix's two recommendations were
only in conflict because the transplant was not known.**

**Contributed bug #15** — *a single-cell rung must specify its cell as completely
as the mixture sampler does*: `di_max` / fuel-sampling / cap / band fell to ini
defaults, so 50M steps of plane-free training ran against an eval cell drawing 1°
planes. Rolling perf 0.998, honest — for the wrong task; eval 50.0%. Plus
**MAJOR-17** (the OOP seed's fixed ceiling bit whenever ρ₀ < 1326 m; 16.4% of
seeded rows sat exactly on it, and the predicted threshold reproduced the observed
rate to the digit) and **MAJOR-17b**, explicitly *bug-#15's class*: **the mixture
range prior excluded 39.1% of true target radii** — two episodes in five
navigated with the target outside the prior's universe, silently, because nothing
asserted on it.

The two zeros (W1_driftwait, TIGHT_5k1) are **the bootstrap mechanism, measured
for the fifth and sixth time**: a cell whose effective warm root is incompetent
never bootstraps under sparse terminal reward in-mixture, and rehearsal weight is
not a substitute for a competent root.

---

## Standalone measurement deliverables

| doc | headline | contributed |
|---|---|---|
| **`FUEL_AUDIT.md`** (08-14) | 4 lineages × 200 eps: box-credited in-plane Δv ratios **1.55 / 1.81 / 1.79 / 1.41**; **no lineage clears the bar for a fuel intervention** | Two corrections that would each have raised a false alarm: T1's teleport diagnostic is planar (reported 125°/ep on TB5-3D; **0.0010°** taken in the orbit plane), and J2-A3b's 3.35°/ep is physics. Introduced the **box-credited Lambert** comparator |
| **`CALIB_TRIAD.md`** (08-15) | The (range, in-plane, plane) triad **passes**: p68\|z\| 0.949–1.154 across all 36 cells | The **per-channel licence** for the 13th-bug fix — global NEES 25,520 → 1.20 did not by itself license a per-channel claim. Instrument correction: verdict moved off `std(z)` (which read 1294 on a cell with nominal coverage) onto p68 + P(\|z\|>3). FINDING: tail *frequency* at E3, 8.08% beyond 400 km vs 0.27% nominal |
| **`GEN_MATRIX.md`** (08-15) | 42 cells + 20-cell control: **the observation normalizer family is a harder barrier than any dynamics difference** — 99 pp vs 51 pp for adding J2 | The decisive control (e-E3 at *perfect truth state* still 0.0%, identical statistics); **two opposite death modes** (narrow-in-wide destructive, wide-in-narrow inert); **both dynamics barriers one-directional** (tight ⊃ loose, J2 ⊃ two-body). Superseded in part by T11's transplant |
| **`HOLD_EVAL.md`** (08-18) | Does the tight box **hold** after capture? | Success is an instant, not a state — see the doc |

## The bug ledger, by catching campaign

| # | what it was | caught by |
|---|---|---|
| 9th | `rew > 0` success classifier reads the Φ-clamp leak at safety cap | Phase 5 pre-closure |
| **11th** | `de_max` inert under `dim3_mode`; realized e-bound 4.5–8× the knob | T5 wide-3D ladder |
| **12th** | The inertial-ϖ fix subtracted *absolute* RAAN → e-disc knob 5.55× its setting | ext-j2 rung prep |
| **13th** | `OrbitalNav` sub-propagated truth two-body under a J2 env; NEES 25,520 → 1.20 | j2nav prep |
| — | *no 14th exists in this repo* | — |
| **#15** | A single-cell rung must specify its cell as completely as the mixture sampler | T11 rung A |
| #15-class | Mixture range prior excluded 39.1% of true target radii | T11 (MAJOR-17b) |

**Same class, never numbered** — worth knowing they exist: LVLH-only-at-equatorial
(238.7% frame error, ext-j2 rung) · `e_max` knob inert at the nav altitude band
(n3dnav-E prep) · zero-init appended row unexplorable, P = 3.9e−9 (j2wait) ·
`nav_block` truth-mode no-op (M10) · MAJOR-17's OOP ceiling (T11) · G6b's shared-
kwargs `TypeError` that meant four gates never ran at all (T11) · `Orbital.close()`
segfault, **found and not fixed** (GEN_MATRIX).

**Bootstrap / value-collapse — the project's most-replicated mechanism, six times:**
T3 flatline → W4 (T5 wide ladder) → R5 (reward-reshaping a committed policy) →
MAJOR-10 (trained-ablated arms) → T11's two zeros. *A cell whose effective warm
root is incompetent never bootstraps under sparse terminal reward.*
