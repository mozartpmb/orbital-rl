# Phase 5.5 — Probe 1+2 Verification Spec

> **Status:** 2026-05-21. The pre-curriculum probes (`PHASE5_5_PRE_CURRICULUM_PROBE_FINDINGS.md`) concluded with strong recommendations: ship Phase 5b ckpt + close Phase 5. Before accepting that, three concerns warrant cheap verification. Each addresses a specific gap in the underlying evidence. ~1 hour total work. If conclusions hold, ship with stronger evidence; if they flip, Phase 5.5 changes direction.

---

## 0. Why this spec exists

The probe findings are competent but the conclusions outrun the evidence in three specific places:

1. **Probe 2 scanned only 4 ckpts** out of ~54 epochs of training (~7% coverage). The spec's discipline (`phase5-5-altitude-expansion-spec.md` §6.7) was "save mid-training ckpts and eval at next-stage OOD; pick best for warm-start." Phase 5b's cap-tail finding showed peak ckpts can be at unexpected epochs. F-B's actual peak may have been missed.

2. **The MEO valley was accepted as a real recipe limitation** without realized-altitude/eccentricity decomposition. valid_init_only filters with high-e+LEO combinations are known to produce realized distributions very different from nominal. The MEO valley may be a sampling artifact, not a capability finding.

3. **F-B was tested at only one hyperparameter point** (lr_max=1e-4, ent_coef=0.001). The original spec §7.1 pre-committed Surprise A response: try F-B+ (lr_max=5e-4, ent_coef=0.005) before concluding "gentle hyperparams don't lift." That response was never executed.

These three checks together take ~1 hour and decisively resolve the conclusion.

---

## 1. Check V1 — Full Probe 2 ckpt scan

### 1.1 The question

Did F-B / F-C actually peak between the 4 scanned epochs {10, 25, 40, 54}?

Cap-tail finding from Phase 5b: peak ckpts can be at any epoch. Stage 5.5.1's smoke peaked at epoch 5 (already past collapse onset). The probe's scan at {10, 25, 40, 54} misses epochs 1-9 entirely. If F-B's warm-start preservation has a transient peak at epochs 3-7, the "no lift" conclusion is wrong.

### 1.2 Method

For each saved Probe 2 ckpt (F-B and F-C × 3 seeds × ~54 epochs each = ~324 ckpts):
- Eval at Stage 5.5.1 conditions (a∈[6.671, 8.5] Mm, e≤0.05, fully_random, valid_init_only=1, LEO obs scales, legacy_action_space=10).
- 200 episodes × 1 rollout seed per ckpt = 200 eps each. (Lower than the 600 used elsewhere because the goal is to find the peak epoch, not multi-seed-validate a final number.)

Total: ~324 ckpts × 200 eps = ~65K episodes. At ~100k SPS effective, ~10-15 min.

Plus LEO regression check at each ckpt: 200 eps × LEO conditions = ~65K more episodes. ~10-15 min.

Total ~25-30 min compute.

### 1.3 Output

For each protocol × seed, a curve of greedy success rate vs epoch at Stage 5.5.1 conditions AND LEO conditions. Two curves per (protocol, seed) = 6 plots.

For each protocol, identify:
- Peak ckpt at Stage 5.5.1 conditions (epoch, success rate, LEO regression at that epoch)
- Pareto-optimal ckpt: maximizes Stage 5.5.1 success subject to LEO regression < 5pp

### 1.4 Decision rule

- **If F-B's peak (Pareto-optimal) is within 3pp of zero-shot baseline (30%):** confirms "F-B doesn't lift." Original conclusion holds.
- **If F-B's peak is meaningfully higher than zero-shot (e.g., 40%+) at LEO-preserving epoch:** the 4-epoch scan missed the peak. F-B does lift; the original conclusion is wrong; Phase 5.5 has a viable fine-tune protocol.
- **If F-C has Pareto-optimal ckpt with both lift and LEO preservation:** F-C is salvageable; the catastrophic-on-one-seed pattern was averaged-away signal, not noise.

### 1.5 Cost

~30 min compute. Single agent run, no decisions needed during the scan.

---

## 2. Check V2 — MEO valley realized-distribution decomposition

### 2.1 The question

Is the MEO valley (11.6-12.5% mean success) a real recipe limitation, or a sampling artifact of valid_init_only filtering?

### 2.2 Background

At MEO altitudes (a∈[6.671, 12] Mm) with e_target_fixed=0.30, the perigee constraint requires `a ≥ 6.571 / 0.70 = 9.387 Mm`. Most of the LEO band (a < 7.171 Mm) gets rejected — so realized altitudes concentrate in the 9.4-12 Mm sub-band.

But what about MEO_low at e_target_fixed=0.05? Perigee constraint: `a ≥ 6.571 / 0.95 = 6.917 Mm`. Most of the [6.671, 12] Mm range is valid. So MEO_low at e=0.05 should have a near-uniform realized distribution.

The valley at MEO_low (11.6%) is more interesting than the valley at MEO (12.5%) because MEO_low at low-e doesn't have a strong filtering bias. If MEO_low at e=0.05 still gets 11-19% (per §1.2), then there's a real recipe limitation at MEO altitudes — not just a sampling artifact.

But this is a hypothesis. Direct check: what does the realized altitude distribution actually look like per MEO cell?

### 2.3 Method

The env-fix metadata includes `realized_init` per trajectory. Pull this from the Probe 1 CSV (`web_data/results/p5_5_probe1_decompose.csv`).

For each (altitude band × e_target × e_sat × phase × relation) cell, report:
- Realized sat.a mean and 5th/95th percentiles
- Realized target.a mean and percentiles
- Realized sat.e mean and percentiles
- Realized target.e mean and percentiles
- gave_up rate

### 2.4 Output

Two analyses:

1. **Per-(alt, e) realized-distribution table.** Confirms whether MEO_low at low-e has uniform realized altitude (recipe failing at MEO) or biased toward sub-band (sampling artifact).

2. **Per-(alt, e) success rate vs realized-altitude scatter.** Across all cells in the surface, plot success rate (y-axis) vs realized median altitude (x-axis). Look for whether the MEO valley really exists in realized-altitude space, or whether it's an artifact of nominal-altitude binning.

### 2.5 Decision rule

- **If MEO_low at e=0.05 has realized altitudes spanning [6.671, 12] Mm with success 11-19%:** real recipe limitation at MEO. Original interpretation holds.
- **If MEO_low at e=0.05 has realized altitudes concentrated in a narrow band (e.g., 9-10 Mm):** sampling artifact. The MEO valley may not exist; what looks like "MEO performance" is actually "performance at narrow realized altitudes." Re-interpretation needed.
- **If the success-vs-realized-altitude scatter is monotonic:** there's no valley, just a smooth degradation with altitude. The "MEO valley" was a binning artifact.

### 2.6 Cost

~15 minutes. The CSV already exists; this is analysis only.

---

## 3. Check V3 — F-B+ hyperparameter probe

### 3.1 The question

Is F-B's failure-to-lift a property of the recipe (plastic-locked converged policy) or of the specific hyperparameters tested (under-tuned)?

The original spec §7.1 pre-committed: if F-B doesn't lift, try F-B+ (lr_max=5e-4, ent_coef=0.005). This 5× larger learning rate and 5× larger entropy is "less gentle" but still much smaller than Stage-4.0 defaults. It tests whether the policy has unutilized plasticity.

### 3.2 Method

3 seeds (42, 31415, 20260423) × 7M training steps with:
- lr_max=5e-4
- ent_coef=0.005
- Stage 5.5.1 conditions: a∈[6.671, 8.5] Mm, e≤0.05, fully_random, valid_init_only=1
- LEO obs scales
- legacy_action_space=10
- Warm from Phase 5b converged ckpts (same as F-B)

Save ckpts every 5 epochs (so all are scanned by V1's protocol).

Eval the full ckpt scan at Stage 5.5.1 conditions AND LEO conditions. Same protocol as V1.

### 3.3 Output

For F-B+, the same curves as V1: greedy success vs epoch at Stage 5.5.1 and LEO. Plus the Pareto-optimal ckpt identification.

### 3.4 Decision rule

- **F-B+ peaks at Stage 5.5.1 ≤ 35% with LEO regression ≤ 5pp:** confirms "fine-tune doesn't lift this recipe." Conclusion of "ship Phase 5b ckpt" strengthens.
- **F-B+ peaks at Stage 5.5.1 ≥ 40% with LEO regression ≤ 5pp:** F-B was under-tuned. The recipe does have plasticity, just needed less-gentle gradients. Phase 5.5 has a viable fine-tune protocol; sweep more hyperparams to find optimal.
- **F-B+ peaks but with LEO regression > 10pp:** the capability-vs-stability trade-off is real. May need a different intervention class (KL constraint, partial freeze) to preserve LEO while lifting Stage 5.5.1.
- **F-B+ collapses like F-A:** there's a hyperparameter cliff between F-B (gentle, no lift) and F-A (aggressive, collapse). The recipe is fundamentally fragile to PPO updates on the converged ckpt.

### 3.5 Cost

~15 min compute (3 seeds × 7M steps × ~5 min each) + V1-style ckpt scan (~20 min if not bundled with V1's scan; less if bundled).

---

## 4. Sequencing

In order:

1. **V2 — MEO realized-distribution decomposition** (~15 min analysis). Cheapest, can run while V1+V3 are training.
2. **V3 — F-B+ training** (~15 min compute). Kicks off first because it produces ckpts that need scanning.
3. **V1 — Full ckpt scan** (~30 min compute). Scans both the existing F-B/F-C ckpts AND V3's new F-B+ ckpts.
4. **Analysis + writeup** (~30 min).

Total ~1.5 hours wall, ~1 hour compute.

---

## 5. Decision matrix

After V1, V2, V3 complete, the combined evidence maps to specific paths:

| V1 (F-B peak vs zero-shot) | V2 (MEO valley) | V3 (F-B+ peak) | Implication |
|---|---|---|---|
| No lift (within 3pp) | Real (decomposition confirms) | No lift | Ship Phase 5b ckpt; close Phase 5. Original probe conclusion confirmed. |
| No lift | Artifact (decomposition shows narrow bands) | No lift | Ship Phase 5b ckpt; rewrite the surface interpretation to remove "MEO valley" framing. |
| No lift | Real | Lifts (≥40%) | Fine-tune protocol works at "less gentle" hyperparams. Curriculum is viable; sweep F-B+ variants. |
| Lifts (≥40%) | Either | Either | F-B's peak was missed in the 4-epoch scan. Conclusion of "no fine-tune works" was wrong. Curriculum is viable. |
| F-C salvageable | Either | Either | F-C with selective ckpt selection might be the protocol. Worth more exploration. |

The most consequential combinations are the bottom three rows — any of them flips the "ship the ckpt and close" recommendation.

---

## 6. What this spec is NOT

- **Not a re-running of Probe 1.** The surface results stand; V2 is a re-analysis of existing data.
- **Not new probes.** V3 tests the pre-committed Surprise A response from the original probe spec. V1 enforces the discipline the original probe should have followed.
- **Not a curriculum spec.** Decisions about curriculum design follow from the verification results.
- **Not exhaustive hyperparameter sweep.** F-B+ is one variant. If it lifts, more variants are worth exploring; that's follow-up, not this spec.

---

## 7. Pre-committed acknowledgments

### 7.1 Surprise A — V1 ckpt scan reveals F-B peak is much earlier than expected (e.g., epoch 3)

This would mean ~7% of the F-B run produced the best ckpt and the rest was monotonic decay. Like Stage 5.5.1 in miniature. Implication: the policy preserves some capability at the very early epochs of fine-tuning, then loses it. This is itself a useful finding — fine-tunes should checkpoint very early and stop.

### 7.2 Surprise B — V2 decomposition shows realized distributions don't match nominal anywhere

If valid_init_only is filtering more aggressively than expected at all cells, the entire surface needs reinterpretation. The 49.7% Stage 5.5.4A baseline might be measuring something quite different from what its label suggests. Sixth instance of metric-vs-implementation, recurring.

### 7.3 Surprise C — F-B+ lifts but LEO regresses

The capability-vs-stability trade-off. Need to decide whether the lift is worth the LEO regression. If LEO drops from 98% to 92% but Stage 5.5.1 goes 30% → 50%, that's a net 14pp shift in capability that's worth considering.

### 7.4 Surprise D — F-B+ collapses

A 5× lr / 5× ent jump pushes into Stage-4.0-style collapse. The hyperparameter window for productive fine-tuning is narrow. Implication: fine-tuning is feasible but requires careful hyperparameter design, possibly with cosine lr schedule or other tricks. Documents a limit and motivates the next investigation.

### 7.5 The discipline

V1's full ckpt scan enforces the spec's own §6.7 discipline. V2's realized-distribution check applies the metric-vs-implementation discipline. V3's F-B+ executes the original spec's pre-committed Surprise A response. Each is a discipline that should have been applied in the original probes; this verification is paying that bill.

---

## 8. Output

A single document `PHASE5_5_VERIFICATION_FINDINGS.md` with:
- V1 results (per-protocol-per-seed Pareto-optimal ckpt at Stage 5.5.1 + LEO)
- V2 results (per-cell realized-distribution table + scatter plot)
- V3 results (F-B+ peaks + LEO regression)
- Decision matrix → recommendation

Then either:
- Confirm original conclusion: write Phase 5 closure.
- Flip original conclusion: write revised Phase 5.5 spec.

---

## 9. Resource summary

| Check | Time | Compute | Type |
|---|---|---|---|
| V1 (full ckpt scan) | 30 min | ~65K eps × 2 conditions | Eval |
| V2 (realized-distribution) | 15 min | None | Analysis only |
| V3 (F-B+ probe) | 30 min | ~21M steps + ~65K eval eps | Train + eval |
| Analysis + writeup | 30 min | None | Writing |
| **Total** | **~1.5 hours** | **~1 hour compute** | Mix |

Smaller than any prior Phase 5.5 probe. The cost-to-decisiveness ratio is very high.

---

*Author: 2026-05-21. Phase 5.5 Probe 1+2 verification spec. Three targeted checks (full ckpt scan, realized-distribution decomposition, F-B+ hyperparameter probe) before accepting the "ship Phase 5b ckpt" recommendation. ~1 hour total work. Pattern continues: small focused probes before commitment.*
