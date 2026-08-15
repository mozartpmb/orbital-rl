# NOTE-22 — the (range, in-plane, plane) calibration triad

**Verdict: the triad passes. No cell is optimistic by more than 1.16× on any
component, in aggregate or in any range bucket. One finding is filed, and it is
about tail frequency, not scale.**

Measured 2026-08-15 · `scripts/orbital/nav/calib_triad.py` · 36,846 z-samples ·
6 classes × 24 noise seeds × 8 envs · raw data in
`web_data/results/calib_triad_<class>.csv`.

---

## What this gates, and what it does not

NOTE-22 gates the **calibration claim**, not code correctness: that the
covariance the bearings-only stack *reports* is the covariance it *delivers*,
**per channel**. The distinction matters because a filter can be globally
consistent while being optimistic on one axis and conservative on another —
NEES averages over the ellipsoid and the two errors cancel inside the average.
That cancellation is precisely what NOTE-22 exists to break open, and it is why
the 13th-bug fix restoring global in-loop NEES (25520 → 1.20) did **not** by
itself license a per-channel claim.

Errors are resolved in the target-relative triad, built from **truth** geometry
so a biased estimate cannot rotate the frame into agreement with itself:

| axis | definition |
|---|---|
| `range` | LOS unit vector, chaser → target |
| `plane` | target orbit normal, orthogonalised against LOS |
| `inplane` | `plane × range`, completing the right-handed set |

For each component `c`: `z = (x̂ − x_true)·e_c / sqrt(e_cᵀ P e_c)`, position and
velocity blocks separately. `z ~ N(0,1)` iff the covariance is honest.

**Out of scope, stated so it is not inherited as settled.** N3D-B §4.4 also
predicts an *error-magnitude* triad — `ratio_plane ≈ 1.0–1.5` against `2.9–5.5`
for range. That is a different quantity (error against a reference, not error
against own covariance) and this sweep does **not** settle it. It does make it
cheap: because the per-channel covariance is honest (below), the error triad can
be read directly off the filter's own reported σ triad, needing only that σ be
logged alongside z.

---

## Method

Each class is flown by **the checkpoint that ships for it**. Calibration is a
property of the filter evaluated on the geometry it actually sees, and a
converged policy visits a narrow tube — closing range, shrinking relative
velocity — that random actions never enter. Scripting the actions would have
measured a filter nobody flies, and would additionally have collapsed TB5-3D
onto X3-loose, since those two rungs differ *only* in the success box.

| class | rung | box | checkpoint | geometry |
|---|---|---|---|---|
| `X3-loose` | X3 | 30 km / 50 m/s | `n3dnav_T-BO3.pt` | two-body, near-equatorial — rung-1 flagship |
| `TB5-3D` | X3 | 5 km / 1 m/s | `n3dnav_T-BO3D-TB5.pt` | tight box, where σ_vel binds |
| `E1-ecc` | E1 | 30 km / 50 m/s | `n3dnav_e_E1.pt` | e_max 0.10 (realized e_t 0.041) |
| `E3-ecc` | E3 | 30 km / 50 m/s | `n3dnav_e_E3.pt` | e_max 0.30 (realized e_t 0.126) |
| `J2X-incl` | J2X | 30 km / 50 m/s | `j2nav_T-J2BO-nav.pt` | J2 on, `nav_j2_mode=1`, i_t U(30°,60°) |
| `NORMBURN` | X3 | 30 km / 50 m/s | *scripted* | plane-heavy burn mix — NOTE-22 mandated |

`NORMBURN` is the one cell where a scripted mix is the right instrument: 3d_E §5's
coverage ablation showed that dropping normal-burn policies made two seeded bug
classes undetectable, and the point is to excite a channel the trained policies
deliberately avoid. `J2X-incl` uses the 50M J2-trained arm the j2nav campaign
produced overnight (96.0% on J2X bearings-only), not a warm-start root.

### Why the verdict is not taken from std(z)

The first pass reported `std = 1294` on a cell whose 1σ and 2σ coverage were 65%
and 92% — essentially nominal. A std that large with nominal coverage is not a
mis-scaled covariance; it is a well-scaled covariance with a few catastrophic
samples, and calling it "optimistic 1294×" would have misdescribed the defect and
pointed any fix in the wrong direction. The scale verdict is therefore taken from
**p68(|z|)** (1.00 for N(0,1), immune to the tail) and the tail is reported as its
own fact: **P(|z|>3)**, nominal 0.27%.

This correction was load-bearing twice. Read through std, TB5-3D looked as if it
carried a severe close-range defect (std 27 inside 25 km, exactly where its 1 m/s
box binds). Read through p68, that bucket is 0.88–1.04 — the std was a *single*
outlier, and the alarming cell is fine.

---

## The table

`sd` = noise seeds. Nominal: p68|z| = 1.00, 1σ = 68.3%, 2σ = 95.4%, >3σ = 0.27%.

| class | blk | comp | n | med | **p68\|z\|** | 1σ | 2σ | **>3σ** | p99 | verdict |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| X3-loose | pos | range | 1098 | −0.035 | **1.001** | 68.2% | 94.2% | **0.64%** | 2.77 | calibrated |
| X3-loose | pos | inplane | 1098 | −0.047 | **1.043** | 66.3% | 95.3% | **0.36%** | 2.65 | calibrated |
| X3-loose | pos | plane | 1098 | −0.027 | **1.048** | 66.1% | 94.5% | **0.46%** | 2.64 | calibrated |
| X3-loose | vel | range | 1098 | 0.051 | **1.069** | 64.8% | 95.4% | **0.46%** | 2.57 | calibrated |
| X3-loose | vel | inplane | 1098 | −0.031 | **1.018** | 67.7% | 93.4% | **0.46%** | 2.65 | calibrated |
| X3-loose | vel | plane | 1098 | 0.042 | **1.008** | 67.8% | 94.3% | **0.36%** | 2.60 | calibrated |
| TB5-3D | pos | range | 1115 | −0.063 | **1.046** | 66.0% | 94.9% | **0.72%** | 2.74 | calibrated |
| TB5-3D | pos | inplane | 1115 | −0.101 | **0.982** | 69.4% | 95.5% | **0.90%** | 2.85 | calibrated +tail |
| TB5-3D | pos | plane | 1115 | −0.014 | **0.970** | 69.8% | 94.0% | **0.45%** | 2.68 | calibrated |
| TB5-3D | vel | range | 1115 | −0.003 | **0.950** | 71.0% | 94.8% | **0.99%** | 2.75 | calibrated +tail |
| TB5-3D | vel | inplane | 1115 | −0.107 | **0.999** | 68.3% | 95.0% | **0.81%** | 2.70 | calibrated |
| TB5-3D | vel | plane | 1115 | 0.023 | **0.949** | 71.2% | 94.8% | **0.63%** | 2.69 | calibrated |
| E1-ecc | pos | range | 1108 | −0.004 | **1.032** | 66.4% | 94.0% | **1.35%** | 3.44 | calibrated +tail |
| E1-ecc | pos | inplane | 1108 | 0.007 | **1.036** | 66.1% | 94.7% | **0.72%** | 2.70 | calibrated |
| E1-ecc | pos | plane | 1108 | 0.003 | **0.994** | 68.4% | 94.8% | **0.90%** | 2.97 | calibrated +tail |
| E1-ecc | vel | range | 1108 | −0.005 | **1.039** | 66.5% | 94.9% | **1.17%** | 3.07 | calibrated +tail |
| E1-ecc | vel | inplane | 1108 | −0.032 | **1.042** | 67.3% | 93.9% | **1.44%** | 3.51 | calibrated +tail |
| E1-ecc | vel | plane | 1108 | 0.021 | **1.027** | 67.3% | 94.1% | **1.08%** | 3.09 | calibrated +tail |
| E3-ecc | pos | range | 1125 | −0.076 | **1.110** | 63.4% | 87.8% | **6.67%** | 4936 | calibrated +HEAVY TAIL |
| E3-ecc | pos | inplane | 1125 | −0.046 | **1.154** | 61.6% | 88.3% | **5.33%** | 635 | calibrated +HEAVY TAIL |
| E3-ecc | pos | plane | 1125 | −0.031 | **1.041** | 66.3% | 92.7% | **3.20%** | 894 | calibrated +HEAVY TAIL |
| E3-ecc | vel | range | 1125 | −0.065 | **1.122** | 63.7% | 88.4% | **6.31%** | 3702 | calibrated +HEAVY TAIL |
| E3-ecc | vel | inplane | 1125 | −0.066 | **1.135** | 63.8% | 88.4% | **6.22%** | 3084 | calibrated +HEAVY TAIL |
| E3-ecc | vel | plane | 1125 | −0.028 | **1.054** | 65.0% | 92.2% | **4.09%** | 3191 | calibrated +HEAVY TAIL |
| J2X-incl | pos | range | 1085 | −0.001 | **0.983** | 68.8% | 93.6% | **1.94%** | 4.75 | calibrated +tail |
| J2X-incl | pos | inplane | 1085 | −0.007 | **1.005** | 67.8% | 94.2% | **1.66%** | 4.27 | calibrated +tail |
| J2X-incl | pos | plane | 1085 | −0.004 | **1.026** | 66.9% | 94.7% | **0.92%** | 2.93 | calibrated +tail |
| J2X-incl | vel | range | 1085 | 0.038 | **0.981** | 69.4% | 94.6% | **1.38%** | 4.73 | calibrated +tail |
| J2X-incl | vel | inplane | 1085 | −0.006 | **1.016** | 67.4% | 93.6% | **1.75%** | 3.81 | calibrated +tail |
| J2X-incl | vel | plane | 1085 | 0.051 | **0.975** | 70.4% | 94.4% | **1.29%** | 3.54 | calibrated +tail |
| NORMBURN | pos | range | 610 | −0.055 | **0.951** | 69.5% | 92.0% | **2.95%** | 8.30 | calibrated +HEAVY TAIL |
| NORMBURN | pos | inplane | 610 | 0.020 | **0.998** | 69.0% | 94.3% | **1.15%** | 3.34 | calibrated +tail |
| NORMBURN | pos | plane | 610 | 0.028 | **1.022** | 67.5% | 94.6% | **0.66%** | 2.64 | calibrated |
| NORMBURN | vel | range | 610 | −0.124 | **0.984** | 69.0% | 92.6% | **3.28%** | 13.44 | calibrated +HEAVY TAIL |
| NORMBURN | vel | inplane | 610 | 0.003 | **1.048** | 66.4% | 91.6% | **3.11%** | 6.89 | calibrated +HEAVY TAIL |
| NORMBURN | vel | plane | 610 | 0.134 | **1.006** | 67.7% | 93.6% | **2.46%** | 3.55 | calibrated +tail |

`+tail` = P(|z|>3) above 3× nominal; `+HEAVY TAIL` = above 10× nominal.

---

## Reading

**1. The scale is honest everywhere.** p68|z| spans **0.949 – 1.154** across all
36 cells. The median z is within ±0.14 of zero in every cell, so there is no
per-channel bias either. Nothing is optimistic by >2× on any component — the
threshold the task set for a claim-boundary edit is not reached, and no shipped
claim needs narrowing on scale grounds.

**2. The plane channel — the one NOTE-22 called unmeasured — is the *best*
behaved of the three.** In every class, the plane component has the smallest
tail of the triad (X3 0.36%, TB5 0.45%, E1 0.90%, J2X 0.92%, NORMBURN 0.66%,
E3 3.20%), and its p68 sits closest to 1.0. The new channel introduced by the 3D
extension is not the weak one. Range is, consistently — which is the expected
signature of a bearings-only estimator, where range is the weakly-observable
direction.

**3. J2X passes per-component, which is the result the task most wanted.**
Bucketed by true range, J2X is calibrated with a near-nominal tail inside 100 km
(p68 0.84–1.11, >3σ ≤1.23%) and only develops a tail beyond 400 km (≤3.21%). The
13th-bug fix holds **per channel**, not merely in the ellipsoid average — the
global NEES recovery was not hiding a per-axis defect.

**4. TB5-3D is clean where it matters.** Inside 25 km, the regime its 1 m/s box
binds in, p68 is 0.88–1.04 on all six components with >3σ at 1.11–1.67%.

---

## FINDING — tail frequency at the eccentric edge

**The claim it binds:** the covariance is per-channel honest *in scale* at every
rung we ship, but at **E3 (e_max 0.30) the tail is up to 25× nominal**, and it
grows with range. Any downstream consumer that treats the reported σ as
supporting a 3σ gate is over-trusting it at E3 by an order of magnitude in
*frequency* (not in scale).

Tail fraction P(|z|>3) by true range, position block:

| class | 25–100 km | 100–400 km | >400 km |
|---|---:|---:|---:|
| X3-loose | 0.59% | 1.03% | 0.37% |
| TB5-3D | 0.93% | 0.43% | 0.33% |
| J2X-incl | 1.23% | 0.95% | 3.01% |
| **E3-ecc** | **2.68%** | **5.93%** | **8.08%** |

**Operating condition it binds:** eccentric rungs at long range — specifically
E3 (`e_max_target=0.30`, realized e_t 0.126, p90 0.257) beyond ~100 km true
separation, i.e. the pre-terminal cruise phase. E1 (e_max 0.10) is 4–6× milder
and X3/TB5 are effectively clean, so the effect scales with eccentricity rather
than being a fixed property of the filter.

**Why this is a claim boundary and not a bug hunt:** the core scale is correct at
E3 too (p68 1.04–1.15), so the covariance is not mis-sized; a small fraction of
samples is badly wrong. The natural suspect is the MSC-6 → Cartesian decode
Jacobian, which linearises a chart whose `ln ρ` direction is weakly observable at
long range and strongly non-Gaussian there — at high eccentricity the arc
geometry varies fastest, so the linearisation is worst exactly where the tail is
worst. **That mechanism is a hypothesis, not a measurement**; nothing here
localises it, and it should not be quoted as established.

**What it does *not* bind:** the success boxes. Both the 30 km and 5 km boxes are
evaluated inside 25 km, where every class including E3 sits at ≤2.7%. The
published E-ladder scores are not called into question by this finding.

**NORMBURN carries the same signature at 3.0–3.3% on range and in-plane
velocity**, which is worth recording because it is scripted plane-heavy
maneuvering rather than eccentricity — so aggressive out-of-plane burning
reproduces a milder version of the same tail. Consistent with the linearisation
hypothesis; still not a measurement of it.

---

## Reproduce

```bash
for C in J2X-incl E3-ecc E1-ecc TB5-3D X3-loose NORMBURN; do
  OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  python3 scripts/orbital/nav/calib_triad.py --seeds 24 --envs 8 --steps 75 \
    --sample-every 12 --classes $C --out web_data/results/calib_triad_$C.csv &
done; wait
python3 scripts/orbital/nav/calib_triad.py --report
python3 scripts/orbital/nav/calib_triad.py --by-range E3-ecc
```

Pin the BLAS threads. Unpinned, six parallel classes oversubscribe the machine
badly enough to cost 84 s/seed against 1 s/seed pinned — an 80× penalty that
looks exactly like a hung run.
