# Phase 5 — Pre-closure Mechanism Verification Spec

> **Status:** 2026-05-22. The V1+V2+V3 verification (`PHASE5_5_VERIFICATION_FINDINGS.md`) confirmed the original probe's "ship Phase 5b ckpt + close Phase 5" recommendation, but the supporting mechanism story rests on speculation rather than direct measurement. Four cheap analysis checks (~30 min total, no new compute) verify the mechanism before closure. After these checks, formal Phase 5 closure documents follow. Pattern continues: small focused work resolves ambiguity before commitment.

---

## 0. Why this spec exists

The verification produced a coherent mechanism story for the recipe's capability surface:

- The 49.7% Stage 5.5.4A baseline decomposes to ~14pp LEO-portion + ~30pp "near-equilibrium configurations at MEO/GEO under fully_random" + ~0pp altitude generalization.
- The relation × altitude flip happens because same_orbit_init at MEO/GEO fails (period-limited phasing within step budget), while fully_random at MEO/GEO succeeds on lucky near-equilibrium configurations.
- F-B+'s LEO drift on 2/3 seeds is "sometimes drifts" rather than bimodal.

Each of these statements is plausible but **not directly measured**. They're inferences from per-cell aggregate success rates. Before writing the closure document around these claims, verify them.

This is the final-final pre-closure step. ~30 minutes total. After this, Phase 5 closes.

---

## 1. Check W1 — Near-equilibrium mechanism at GEO fully_random

### 1.1 The claim being verified

V2 §2.7 decomposes the 49.7% baseline with this implicit claim: "fully_random at MEO/GEO succeeds because valid_init_only retains lucky near-equilibrium configurations." For GEO specifically, the writeup says "successful GEO fully_random episodes are sampling-luck-induced near-equilibrium configurations."

This is the load-bearing mechanism for the "the recipe doesn't generalize to altitudes" claim. If the mechanism is wrong, the deliverable framing should change.

### 1.2 The check

From existing trajectory dumps for GEO fully_random successful episodes, compute the **initial Δa = |sat.a - target.a|** at episode start. Aggregate.

Specifically, for the cell `(alt=GEO, e_target=0.05, e_sat=0.05, relation=fully_random)` — the cell at 54% success — compute the distribution of initial Δa for successful episodes vs failed episodes.

### 1.3 Decision rule

- **If successful episodes have initial Δa < 100 km (median):** the near-equilibrium mechanism holds. Successes are configurations that were already close at start.
- **If successful episodes have initial Δa broadly distributed across [0, 35 Mm]:** the policy is doing real maneuvering at GEO that we don't understand. Mechanism story needs revision.
- **If successful Δa is between (median ~1-5 Mm):** intermediate case. The policy is doing some maneuvering but not full-range. Worth documenting honestly.

### 1.4 Cost

~10 min. Pull from existing Probe 1 trajectory dumps.

---

## 2. Check W2 — Phasing-by-period mechanism at MEO_low same_orbit

### 2.1 The claim being verified

V2 §2.6 explains the relation × altitude flip with this claim: "same_orbit_init at MEO_low succeeds because orbital periods are short enough (~3 hr) to iterate phasing windows within the 2000-step budget; at MEO/GEO, periods are too long."

This is testable directly: if successful MEO_low same_orbit episodes cluster around 1-2 orbital periods worth of steps, the phasing-iteration mechanism is confirmed.

### 2.2 The check

For the cell `(alt=MEO_low, e_target=0.05, e_sat=0.05, relation=same_orbit)` — the cell at ~35.7% success — compute:

- **Episode length distribution for successful vs failed episodes** (mean, median, 5th/95th percentiles).
- **Length in units of orbital periods.** MEO_low orbital period at a≈9.4 Mm is ~159 minutes = 159 steps at DT=60s. So 1-2 periods ≈ 159-318 steps.

### 2.3 Decision rule

- **If successful episodes cluster at 100-400 steps (1-2.5 periods):** phasing-by-period mechanism holds. The policy uses the period-iterability to close phase gaps.
- **If successful episodes are at the safety cap (~2000 steps):** the policy isn't iterating phasing, it's doing something else (or successes are also luck-driven).
- **If successful episodes are very short (<50 steps):** the policy isn't phasing at all; successes are immediate.

### 2.4 Cost

~10 min. Pull from existing trajectory dumps.

---

## 3. Check W3 — V3 bimodality framing

### 3.1 The claim being verified

V3 reports F-B+ as "sometimes drifts LEO" based on per-seed final LEO values: 95.5%, 85.5%, 81.0%. The writeup interprets this as a continuous drift phenomenon rather than discrete bimodality.

Project history suggests bimodality. Phase 5b at the recipe edge: 3/5 vs 2/5. F-C across all seeds: catastrophic vs moderate. Maybe F-B+ is the same pattern: 1/3 stable, 2/3 drift.

### 3.2 The check

For each F-B+ training seed (42, 31415, 20260423), plot the LEO success rate vs epoch across all 11 saved checkpoints. Look for:

- **Bimodal:** one seed stable across all epochs; other two start drifting at some epoch and continue.
- **Continuous:** all seeds drift smoothly but at different rates.
- **Other:** some other pattern.

This is qualitatively informative: bimodality means the protocol is unstable in a specific way (RNG determines which attractor); continuous drift means the protocol is uniformly slightly-too-aggressive.

### 3.3 Decision rule

This is descriptive, not pass/fail. The output is one sentence in the closure document:

- **Bimodal:** "F-B+ produces bimodal LEO outcomes across seeds, consistent with the recipe-edge bimodality pattern Phase 5b exhibited at e=0.05."
- **Continuous:** "F-B+ produces seed-dependent LEO drift, with all seeds drifting but at different rates."

The first is information about protocol stability that should carry forward to Phase 6 hyperparameter design. The second is information about the protocol being slightly miscalibrated.

### 3.4 Cost

~5 min. Existing V1 CSV data plotted.

---

## 4. Check W4 — Sharpen the relation × altitude flip framing

### 4.1 The claim being verified

V2 §2.5 shows the relation × altitude flip:

| Band | same_orbit | fully_random | Winner |
|---|---:|---:|---|
| LEO | ~100% | ~88% | same_orbit (close) |
| MEO_low | ~35% | ~2% | same_orbit (large gap) |
| MEO | ~8% | ~25% | fully_random (flip) |
| GEO | ~4-5% | ~52% | fully_random (large gap) |

The writeup calls this a "flip." But same_orbit_init is intrinsically easier than fully_random (sat and target share orbit shape — only θ must be matched, not (a, e, ω)). The interesting finding isn't that same_orbit "wins" at LEO — it's that the win disappears at MEO+.

### 4.2 The check

Reframe the table in terms of "easiness gain from same_orbit_init":

| Band | same_orbit | fully_random | Gain from same_orbit |
|---|---:|---:|---:|
| LEO | ~100% | ~88% | +12pp |
| MEO_low | ~35% | ~2% | +33pp |
| MEO | ~8% | ~25% | **-17pp (inverted)** |
| GEO | ~4-5% | ~52% | **-47pp (heavily inverted)** |

At LEO/MEO_low, same_orbit_init's intrinsic easiness translates to policy success (the policy can leverage shared-shape configurations). At MEO+, same_orbit_init becomes the **failing mode** — the policy can't solve same_orbit at high altitude, even though it's intrinsically the easier task structure.

### 4.3 The interpretation

The recipe has *two* skills, not one:
- A **phasing skill** that works on same_orbit configurations at altitudes where the orbital period is short enough to iterate within the step budget (LEO, MEO_low).
- A **null-maneuver skill** that succeeds when valid_init_only happens to retain near-equilibrium configurations (fully_random at MEO, GEO).

Neither is a true altitude generalization. Both are altitude-band-specific exploitations of distribution structure.

### 4.4 Decision rule

This is reframing, not new measurement. The output is a paragraph in the closure document that replaces the "flip" language with the "intrinsic easiness vs policy ability" framing. Sharper, more honest, more informative.

### 4.5 Cost

~5 min. Writing.

---

## 5. Sequencing

In order:

1. **W1** (GEO near-equilibrium): ~10 min. Most consequential for the deliverable framing.
2. **W2** (MEO_low phasing-by-period): ~10 min. Verifies the relation × altitude mechanism.
3. **W3** (bimodality framing): ~5 min. One sentence depends on this.
4. **W4** (relation × altitude reframing): ~5 min. Writing.
5. **Analysis + integration into closure document**: ~30 min.

Total ~1 hour wall time, ~30 minutes actual work. No new compute.

---

## 6. Decision matrix

After W1-W4 complete, possible outcomes:

| W1 (GEO Δa) | W2 (MEO_low length) | Implication |
|---|---|---|
| Small initial Δa | Clusters at 1-2 periods | Both mechanisms confirmed. Ship closure with the verified two-skill framing. |
| Small Δa | Lengths at safety cap | Near-equilibrium works at GEO; phasing-by-period doesn't explain MEO_low. Look for alternative mechanism. |
| Broadly distributed | Clusters at 1-2 periods | GEO does real maneuvering; we don't understand it. The "no altitude generalization" claim is wrong. Major reframe. |
| Broadly distributed | Lengths at safety cap | Neither mechanism is right. The 49.7% has some other explanation. Investigate before closure. |

The first row is the most likely outcome based on the writeup's intuitions. The other three would substantially change the closure framing.

---

## 7. What this spec is NOT

- **Not new training.** Pure analysis on existing trajectory dumps.
- **Not new probes.** Verifies the interpretation, doesn't surface new findings.
- **Not a curriculum revision.** Whatever the outcome, Phase 5 still closes with the Phase 5b ckpt as deliverable.
- **Not the closure document itself.** Closure follows verification.

---

## 8. After W1-W4

Three possible next steps:

### 8.1 All mechanisms verified

Write Phase 5 closure documents:
- `PHASE5_FINDINGS.md` — consolidated, with the verified two-skill framing.
- `RECIPE.md` — final recipe with the per-(alt, relation) capability characterization.
- `PHASE6_TRANSITION.md` — short bridge document.

Phase 5 closes. Phase 6 begins.

### 8.2 Mechanism partially verified

Adjust the framing for whichever mechanism didn't verify. Closure still happens; the explanation just changes. Document the unknown ("we don't have a complete mechanism for [thing]") honestly.

### 8.3 Mechanism contradicted

If W1 shows GEO does real maneuvering (large Δa successful episodes), the "no altitude generalization" claim is wrong. The recipe is more capable than V2 framing suggested. Worth a focused follow-up to understand what's actually happening at GEO before closure.

This is unlikely but possible. The verification is cheap; running it before closure protects against shipping a framing that doesn't match reality.

---

## 9. Resource summary

| Check | Time | Compute | Type |
|---|---|---|---|
| W1 (GEO Δa analysis) | 10 min | None | Analysis on existing dumps |
| W2 (MEO_low length analysis) | 10 min | None | Analysis on existing dumps |
| W3 (bimodality plotting) | 5 min | None | Plot from V1 CSV |
| W4 (relation flip reframe) | 5 min | None | Writing |
| Integration into closure | 30 min | None | Writing |
| **Total** | **~1 hour** | **None** | Analysis + writing |

Smaller than any prior Phase 5 spec. The cost is essentially writing time; the value is verified mechanism before formal closure.

---

## 10. The discipline this spec applies

Project history shows that closure narratives written before mechanism verification tend to need retraction (Phase 5c's Goodhart story, Phase 5e's "graceful degradation" story, the wrap-up's "cliff at e≥0.075"). Each was a coherent-sounding narrative that hadn't been directly verified. Each was wrong in specific ways that the data eventually surfaced.

W1-W4 are the final discipline pass: before writing the Phase 5 closure narrative, verify the mechanism stories it'll rely on. If the verification confirms, the closure is robust. If it contradicts, the closure changes. Either way, the closure isn't shipped on speculation.

This is the 13th methodological discipline to add to the retrospective: **before writing a closure narrative that depends on mechanism stories, directly verify those mechanism stories from existing data.** Cheap, decisive, prevents narrative-shaped retractions.

---

*Author: 2026-05-22. Phase 5 pre-closure mechanism verification. ~1 hour total work, no new compute. Four cheap checks against existing data. After this, formal Phase 5 closure documents follow. Last methodological pass before Phase 6.*
