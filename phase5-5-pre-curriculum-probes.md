# Phase 5.5 — Pre-curriculum Probes (Decomposition + Fine-tune)

> **Status:** 2026-05-21. Two probes before committing to any Phase 5.5 curriculum training. Probe 1 (capability decomposition) characterizes the 49.7% latent capability that the stochastic-eval probe revealed at Stage 5.5.4A conditions. Probe 2 (fine-tune protocol) tests three fine-tune approaches at the smallest stage where training is genuinely needed. Together: ~3-4 hours work, decisive for what Phase 5.5's actual training scope is. Pattern matches prior probes: small focused work surfaces real findings before committing compute.

---

## 0. Why this spec exists

The stochastic-eval probe (`PHASE5_5_STOCHASTIC_PROBE_FINDINGS.md`) produced two findings that together change the Phase 5.5 framing:

1. **Phase 5b achieves 49.7% multi-seed at the full LEO+MEO+GEO envelope (Stage 5.5.4A conditions, LEO obs scales).** The recipe has substantial latent altitude capability that the pre-experiments missed because P2 used different conditions. ~36pp of the 50% is at altitudes the recipe was never trained on.

2. **Stage 5.5.1 demonstrated catastrophic plasticity collapse** when fine-tuning a converged Phase 5b ckpt with Stage-4.0 hyperparameters (lr_max=1e-2, ent_coef=0.01). This will recur at every training stage unless the fine-tune protocol is fixed.

These two findings pull in different directions for curriculum design:
- The 49.7% suggests the curriculum may compress (less training needed than planned).
- The plasticity collapse suggests any training that *is* needed has to use a different protocol.

Neither can be addressed by committing to a curriculum and seeing what happens — the Phase 5.5.1 result already showed that pattern produces catastrophic failures, not informative ones. Better: two focused probes that resolve both questions before more training.

---

## 1. Probe 1 — Capability decomposition of the 49.7% baseline

### 1.1 The question

The Phase 5b ckpt achieves 49.7% at Stage 5.5.4A (a∈[6.671, 42.5] Mm, e∈[0, 0.50], fully random, valid_init_only=1, LEO obs scales). Which cells of the underlying surface contribute to this 50%, and which fail?

Three hypotheses, mutually exclusive predictions:

- **H1 (eccentricity-bias):** The 50% concentrates at low eccentricities across all altitudes. Failing cells are high-eccentricity, regardless of altitude. Implication: train at higher eccentricities; altitude isn't the binding axis.

- **H2 (coast-and-wait):** The 50% concentrates on tasks where the optimal strategy is wait-for-phasing-window. Failing cells require active maneuvering. The recipe's altitude-invariant skill is the warp action, not the burn actions. Implication: target training at tasks that need maneuvering at altitude.

- **H3 (genuine altitude generalization):** The 50% is uniformly distributed across altitudes; the recipe really does transfer altitude-invariantly for a non-trivial fraction of conditions. Failing cells are specific weak corners. Implication: curriculum compresses to targeted training at those corners.

Each implies a different Phase 5.5 design. Probe 1 distinguishes them.

### 1.2 Method

Run the existing capability surface eval infrastructure on the canonical Phase 5b ckpt (`puffer_orbital_177750405236/model_puffer_orbital_000350.pt`) at fine-grained (altitude × eccentricity × phase × relation) cells.

**Axes:**

| Axis | Values |
|---|---|
| `a_max_target` | LEO (7.171 Mm), MEO_low (12 Mm), MEO (26 Mm), GEO (42.5 Mm) |
| `e_target` fixed | 0.0, 0.05, 0.10, 0.20, 0.30, 0.50 |
| `e_sat` fixed | 0.0, 0.05, 0.10, 0.20, 0.30, 0.50 |
| `phase_gap_fixed` | 30°, 90°, 150°, 180° |
| `sat_target_relation` | same_orbit_init=0 (fully_random), same_orbit_init=1 |

**Per-axis defaults:** all cells run with LEO obs scales (`obs_alt_scale_m=1.6e6`, `lvlh_scale_m=6.371e6`) since the stochastic probe established these are the right eval-time scales. `valid_init_only=1`. Greedy mode (per stochastic probe: stochastic ≈ greedy, no need to run both).

**For each altitude band, scale `a_min` accordingly:**
- LEO: a∈[6.671, 7.171] Mm
- MEO_low: a∈[6.671, 12] Mm
- MEO: a∈[6.671, 26] Mm
- GEO: a∈[6.671, 42.5] Mm

Note: cells where the (a, e) combination is physically infeasible (e.g., MEO+e=0.50 + LEO range) get filtered by `valid_init_only=1`, leaving realized samples that ARE valid. The realized-eccentricity distribution per cell may differ from the nominal fixed values. Per the env-fix metadata, this is recorded; the decomposition should report realized vs nominal.

**Reduced grid for scope:** the full 4×6×6×4×2 = 1152 cells would be prohibitive. Two reductions:

- **Reduction A — diagonal e_target = e_sat:** drops e_target × e_sat from 36 to 6 cells. Total: 4×6×4×2 = 192 cells. (~30 min compute at 200 eps × 3 rollout seeds = 600 eps/cell.)
- **Reduction B — sample-off-diagonal sparsely:** add 4 off-diagonal cells (e.g., e_target=0.30, e_sat=0.0; e_target=0.0, e_sat=0.30; etc.) at LEO and GEO to confirm the diagonal approximation. ~8 extra cells. Total: 200 cells, ~30 min compute.

Use Reduction B (the off-diagonal sample tells us if diagonal-approximation is hiding something — analogous to the wrap-up's diagonal-only mistake the project caught earlier).

### 1.3 Output

A CSV with per-cell stats (mean ± std across 3 rollout seeds, n=600 eps). Visualizations:

- **Heatmap 1 (altitude × eccentricity):** for each phase × relation cell, show success rate as a heatmap. Look for axis-aligned structure (rows or columns of failure).
- **Heatmap 2 (phase × relation):** for each (alt, e) cell, show phase × relation. Look for off-diagonal patterns.
- **Histogram of realized eccentricities** per cell: confirms the cell's fixed-e setting reflects what was actually sampled (after valid_init_only filtering).

### 1.4 Decision criteria — which hypothesis is supported

After the surface:

- **H1 supported if:** holding altitude constant, success drops sharply as e_target/e_sat increases. AND holding eccentricity constant, success is relatively flat across altitudes.
- **H2 supported if:** success correlates with "easy phase configurations" (low phase_gap, same_orbit_init), independent of altitude or eccentricity. Coast-and-wait tasks would show this signature.
- **H3 supported if:** success is roughly uniform across altitudes and eccentricities, with specific corner failures that don't align with single axes.

If none of these cleanly fit: there may be a multi-axis interaction worth investigating, OR the 50% is composed of multiple distinct failure modes. Document what's found.

### 1.5 Pre-committed surprises

**Surprise A — The 49.7% reproduces but doesn't decompose cleanly.** The cell-level numbers are noisy enough that no axis dominates. Possible cause: 600 eps/cell with valid_init_only filtering produces small realized-sample populations at some cells; per-cell statistical power is low.

Response: report cells with their realized n. Flag low-n cells. Possibly re-run targeted cells with more episodes (e.g., 2000 each) to tighten CIs.

**Surprise B — The 49.7% doesn't reproduce.** Running 200 cells with 600 eps each is a much larger sample than the stochastic probe's single-cell measurement. The 49.7% might land at 35% or 60% with more data.

Response: report what's measured. If the number is materially different, investigate (could be eval-pipeline variability, could be that Stage 5.5.4A's broad altitude range has variance that 600 eps undersampled).

**Surprise C — A specific altitude band performs much better than expected.** E.g., MEO_low at low-e gets 80%+ success. This would mean the recipe extends partially to MEO without training, beyond what's in the 49.7% aggregate.

Response: this is the most interesting outcome for curriculum design. Document and incorporate into the curriculum revision.

### 1.6 Cost

~30-60 minutes compute. 200 cells × 600 episodes = 120K episodes. At ~100k SPS effective, ~20 minutes; with overhead, ~30-60 minutes wall.

---

## 2. Probe 2 — Fine-tune protocol investigation

### 2.1 The question

Stage 5.5.1 demonstrated that Stage-4.0 hyperparameters (lr_max=1e-2, ent_coef=0.01) catastrophically destroy a converged Phase 5b ckpt within 20 epochs. This will recur at every training stage if the protocol isn't fixed.

Three fine-tune protocols, mutually exclusive predictions about which lifts capability without collapse:

- **F-A (aggressive, baseline):** Stage-4.0 hyperparams. lr_max=1e-2, ent_coef=0.01, 30M steps. Phase 5.5.1 already ran this; expect collapse. Re-running for clean comparison.

- **F-B (gentle):** Conventional fine-tune-from-converged. lr_max=1e-4, ent_coef=0.001, 5-10M steps. Expected: small lift without collapse.

- **F-C (mid-training warm-start):** Use a Phase 5b ckpt at ~50% training (more residual entropy) as warm-start, apply F-B gentle hyperparams. Tests the "intermediate ckpt as warm-start" hypothesis from PHASE5b §6.8 that was identified but never tested.

### 2.2 Choice of training conditions

Run all three protocols at the first altitude band where Phase 5b's zero-shot capability is meaningfully degraded but not zero. From Probe 1's outputs (or from the stochastic probe's known data), this is likely Stage 5.5.1 (a≤8.5 Mm, e≤0.05, 30% baseline).

Stage 5.5.1 is the right choice because:
- Pre-training capability is known (~30% greedy).
- The aggressive protocol already failed there (Stage 5.5.1 smoke).
- The altitude extension is modest (no obs-scale issues, no warp adequacy issues from M2).
- Compute is small (~5M steps per protocol = ~3 min each on M3 Max).

### 2.3 Method

For each of F-A, F-B, F-C:
- Same warm-start ckpt source (canonical Phase 5b seed-31415 for F-A and F-B; an intermediate ckpt at ~epoch 175 of the same training run for F-C — i.e., midpoint of Stage 4.0's 350-epoch run).
- Same env config: a∈[6.671, 8.5] Mm, e≤0.05, valid_init_only=1, LEO obs scales, init_phase_gap_max=π, legacy_action_space=10.
- 3 seeds per protocol (42, 1337, 31415).
- 5M-10M training steps per protocol (the same range that revealed Stage 5.5.1's collapse pattern within 20 epochs).
- Save ckpts every 5 epochs.
- Track training-time `perf` for diagnostics, but report greedy held-out eval (200 eps × 3 rollout seeds per ckpt) as the deliverable metric.

### 2.4 Decision criteria

Per protocol, eval the **best ckpt across all checkpoints** (not just final) at Stage 5.5.1 conditions:

- **F-A:** expect collapse to <5% within 30M steps. If it doesn't collapse, that's a surprise worth investigating.
- **F-B:** target ≥40% mean (10pp lift over 30% zero-shot baseline). Anything ≥35% counts as "didn't collapse + lifted some."
- **F-C:** target ≥40% mean. Compare to F-B.

Also report **LEO regression check** for each protocol's best ckpt (eval at LEO conditions, e=0.05). Phase 5b's 96.4% baseline should be preserved — significant LEO regression (>5pp drop) is a Pyrrhic victory.

### 2.5 What the result tells us

- **If F-B works (lift + no LEO regression):** the fine-tune protocol is the solution. Phase 5.5's curriculum stages can use F-B hyperparameters across the board.
- **If F-B works partially (some lift but LEO regression):** there's a fundamental capability-vs-stability trade-off. Document; explore whether F-C helps.
- **If F-C beats F-B:** mid-training warm-starts are the right approach. This has broader implications: Phase 5b's "save converged ckpt as deliverable" is fine for benchmarking but Phase 5.5's curriculum should warm-start from mid-training ckpts.
- **If neither F-B nor F-C lifts capability:** the converged Phase 5b policy is genuinely too plastic-collapsed to fine-tune. The path forward becomes harder: either retrain Phase 5b with explicit plasticity preservation, or use Phase 5b ckpts directly without further training.

### 2.6 Pre-committed surprises

**Surprise A — F-B doesn't collapse but doesn't lift either.** Hyperparameters are gentle enough that the policy doesn't degrade, but also doesn't learn anything new. The policy is "frozen" by the gentle gradients.

Response: try slightly more aggressive variant (F-B+: lr_max=5e-4, ent_coef=0.005). One additional 3-seed run, ~10 min compute. If still flat, F-B as designed is too gentle.

**Surprise B — F-A this time doesn't collapse.** Run-to-run variance, or some plumbing change since the Stage 5.5.1 smoke. Report the discrepancy honestly and investigate before drawing conclusions.

**Surprise C — F-C is worse than F-B.** The mid-training ckpt has more residual capacity, but also is at a less-optimal point on the loss landscape, and gentle fine-tuning isn't strong enough to overcome that. Documents the hypothesis was wrong; F-B is the right approach.

**Surprise D — All three protocols destroy LEO baseline.** Any altitude fine-tuning at all causes LEO regression because the policy is single-task and forgets when retrained. This would be a real architectural finding: Phase 5b's recipe doesn't support fine-tuning at all without forgetting.

Response: document. The path forward in this scenario is either: (a) use Phase 5b ckpts directly + accept the 49.7% baseline as the deliverable, or (b) restart with a different recipe that supports continual learning.

### 2.7 Cost

~30-45 minutes compute. 3 protocols × 3 seeds × ~5-10M steps × ~5 min per 30M-step run on M3 Max = ~15-25 minutes for training, plus eval (200 eps × 3 rollout seeds × ~10 ckpts per protocol = 1800 eps, ~1 min each, ~30 min).

Total ~45-75 min including analysis. Well under an hour.

---

## 3. Sequencing

In order:

1. **Probe 1 — Capability decomposition** (~30-60 min compute).
2. **Analysis of Probe 1 outputs** (~30 min). Identifies which hypothesis (H1/H2/H3) is supported.
3. **Probe 2 — Fine-tune protocol** (~30-45 min compute). Runs F-A/F-B/F-C at Stage 5.5.1.
4. **Analysis of Probe 2 outputs** (~30 min). Identifies which protocol works.
5. **Decide on Phase 5.5 curriculum revision** based on combined findings.

Total ~2.5-3.5 hours wall time, including analysis.

---

## 4. What the combined probes determine

After both probes complete, Phase 5.5's curriculum design has answers to:

1. **What's the recipe's actual baseline across the full envelope?** (Probe 1's per-cell surface)
2. **Which conditions need training, which don't?** (Probe 1's failing cells)
3. **What fine-tune protocol preserves capability while extending it?** (Probe 2's working protocol, if any)
4. **Can we train at all without LEO regression?** (Probe 2's LEO regression check)

Combinations of these answers map to specific curriculum designs:

- **Probe 1 = H1 + Probe 2 F-B works:** train at higher eccentricities, no altitude curriculum needed. Phase 5.5 compresses to 1-2 stages on eccentricity.
- **Probe 1 = H3 + Probe 2 F-B works:** targeted training at specific failing cells; small focused stages.
- **Probe 1 = H2 + Probe 2 F-B/C works:** training needs to target maneuvering tasks specifically; possibly include task-distribution modifications.
- **Any Probe 1 + Probe 2 fails:** can't train without breaking the policy. Phase 5.5 ships Phase 5b's ckpt with the 49.7% baseline as the deliverable.

The last case is real and worth contemplating. If the converged Phase 5b ckpt can't be fine-tuned, then the deliverable is "we have a ckpt that handles 50% of the full envelope; that's what ships." That's still a substantively stronger deliverable than the previous "LEO low-e specialist" framing.

---

## 5. Output

A single document `PHASE5_5_PRE_CURRICULUM_PROBE_FINDINGS.md` covering both probes:

- **§1:** Probe 1 results table + heatmaps + hypothesis verdict
- **§2:** Probe 2 results table (per-protocol best-ckpt success + LEO regression) + protocol verdict
- **§3:** Combined implication for Phase 5.5 curriculum
- **§4:** Recommended next step (revised curriculum spec, or "ship Phase 5b ckpt directly")

Plus update `phase5-5-altitude-expansion-spec.md` to reflect the new state (or supersede it with the new design).

---

## 6. What this spec is NOT

- **Not Phase 5.5 curriculum training.** No multi-stage training, no >10M-step runs.
- **Not a re-verification of env mods.** The env mods are validated; probes use them as-is.
- **Not a replacement for `phase5-5-altitude-expansion-spec.md`.** That spec's structure is still relevant; the probes refine its scope.
- **Not a complete fine-tune protocol exhaustion.** Three variants are tested. If none work, more variants could be tried (cosine lr schedule, KL constraints, etc.), but that's follow-up if needed.

---

## 7. Pre-committed acknowledgments

### 7.1 Surprise A — Probes are inconclusive

Probe 1 surface could have noisy cells; Probe 2 could be borderline (e.g., F-B gives 30% lift but with 5pp LEO regression). The "decisive" framing might overstate what 3-4 hours of probing can tell us.

Response: report what's found. If neither probe gives clean answers, document the ambiguity and identify the next probe needed. Don't fabricate certainty.

### 7.2 Surprise B — A finding emerges that no probe was designed for

Stage 5.5.1's smoke surfaced the 99.7% mystery that the stochastic probe was then designed to investigate. The current probes might surface a new mystery (e.g., the realized-eccentricity distribution at MEO is weird, or F-C produces a wildly bimodal seed pattern).

Response: document the new finding. Decide whether it's an artifact (false alarm) or substantive (deserves its own probe). Don't ignore unexpected findings to fit the spec's narrative.

### 7.3 Surprise C — Realized-eccentricity at MEO+ doesn't match nominal

valid_init_only=1 with high e_max at MEO+ might still reject substantial fractions, leaving realized samples concentrated at lower e. Per the env-fix metadata (which now records realized perigees), this can be measured directly per cell.

Response: report realized vs nominal per cell. If they differ substantially (>20pp), note that "fixed e=0.30" at MEO is actually "realized e mean ≈ 0.18" or similar, and interpret accordingly.

### 7.4 The discipline continues

Each probe gets:
- Explicit decision criteria pre-committed (not invented after the data).
- Cross-checks (e.g., realized-vs-nominal eccentricity, LEO regression for fine-tune).
- Trajectory inspection on 3-5 representative cells (visual mechanism verification).
- Multi-seed (3+) where applicable.
- Per-seed variance reported alongside means.

No "we found the mechanism" narratives without orthogonal evidence.

---

## 8. After the probes

Three possible next steps:

### 8.1 Revised Phase 5.5 curriculum spec

If the probes reveal a clearer curriculum design (e.g., "train at high-e only" or "target three specific failing cells"), write a new spec that supersedes `phase5-5-altitude-expansion-spec.md` Option B. The new spec inherits the methodology discipline but with a tighter scope.

### 8.2 Phase 5b ckpt is the deliverable

If Probe 2 shows no fine-tune protocol works without breaking the ckpt, Phase 5.5's deliverable is "Phase 5b ckpt + the 49.7% multi-seed baseline at full envelope." Document the recipe, characterize the capability, write Phase 5 closure. Phase 6 starts.

### 8.3 New direction needed

If both probes reveal that the recipe is more limited than thought (e.g., the 49.7% doesn't reproduce on a larger sample, and no fine-tune helps), the next step is a deeper architectural reconsideration. Phase 5b shipped as-is; subsequent altitude work requires a different recipe foundation.

---

## 9. Resource summary

| Item | Time | Compute | Type |
|---|---|---|---|
| Probe 1 (decomposition) | 30-60 min | ~120K episodes | Eval only |
| Probe 1 analysis | 30 min | — | Writing |
| Probe 2 (fine-tune) | 30-45 min | 3 × 3 seeds × ~7M steps | Training + eval |
| Probe 2 analysis | 30 min | — | Writing |
| Combined writeup | 30 min | — | Writing |
| **Total** | **2.5-3.5 hours** | **Modest** | Mix |

Smaller scope than any prior Phase 5 sub-investigation. The cost-to-decisiveness ratio is good.

---

*Author: 2026-05-21. Phase 5.5 pre-curriculum probes. Two focused probes (capability decomposition + fine-tune protocol) before committing to Phase 5.5 curriculum training. ~3 hours work. Decisive for what Phase 5.5's actual scope is.*
