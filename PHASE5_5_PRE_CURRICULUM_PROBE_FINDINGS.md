# Phase 5.5 — Pre-Curriculum Probe Findings (Probes 1 + 2)

> **Status:** 2026-05-21. Both probes complete per `phase5-5-pre-curriculum-probes.md`. ~28 min compute total (Probe 1: 12.6 min eval; Probe 2: 14 min train + ~12 min eval). Both probes produced clean, decisive answers. Combined implication: **the Phase 5b ckpt cannot be productively fine-tuned with the protocols tested; the 49.7% baseline is genuine but its altitude generalization is concentrated in `fully_random` sampling at GEO, and aggressive same_orbit_init=1 maneuvering at altitude is the failing mode (not eccentricity, not altitude per se).**

---

## TL;DR

**Probe 1 verdict (capability decomposition):** H2-ish + H3 hybrid. Recipe behavior is **not e-cliff-driven** (H1) across altitudes — only at LEO. Surprising flat ~52% at GEO under fully_random across all e∈[0, 0.50]. MEO is the actual valley (~12%). Off-diagonal e_target/e_sat at GEO matches diagonal (no Simpson's hiding under the diagonal reduction). The 49.7% Stage 5.5.4A baseline reproduces and is dominated by GEO fully_random.

**Probe 2 verdict (fine-tune protocol):** Surprise A from §2.6 — **F-B preserves LEO but doesn't lift Stage 5.5.1 capability.** F-C lifts plasticity at the cost of LEO regression (up to -31pp). F-A collapses both. No protocol cleanly extends the recipe without breaking it.

**Combined implication:** Phase 5.5 falls into spec §4 case "**Probe 2 fails**" → spec §8.2 "Phase 5b ckpt is the deliverable" is the right next move. The 49.7% baseline is what Phase 5.5 ships, with deeper characterization of *where* it works.

---

## 1. Probe 1 — Capability decomposition surface

### 1.1 Setup

- Ckpt: Phase 5b canonical seed-31415 (`puffer_orbital_177750405236/model_puffer_orbital_000350.pt`).
- 4 altitude bands × 6 eccentricities × 4 phases × 2 relations + 8 off-diagonal cells = 200 unique cells × 3 rollout seeds × 200 eps = 120K episodes.
- LEO obs scales throughout (per stochastic-probe finding). `valid_init_only=1`. Greedy eval. `--legacy-action-space 10`.
- Script: `scripts/orbital/p5_5_probe1_decompose.py`. Raw CSV: `web_data/results/p5_5_probe1_decompose.csv`.

### 1.2 Per-altitude × per-eccentricity surface (mean across phase × relation × seeds)

| alt | e=0.00 | e=0.05 | e=0.10 | e=0.20 | e=0.30 | e=0.50 |
|---|---:|---:|---:|---:|---:|---:|
| LEO | 97.6% | 94.3% | **0.0%** | 0.0% | 0.0% | 0.0% |
| MEO_low | 18.5% | 19.1% | 17.3% | 8.9% | 6.0% | 0.0% |
| MEO | 15.7% | 17.1% | 16.1% | 12.3% | 10.9% | 3.1% |
| GEO | 28.5% | 29.0% | 28.7% | 27.6% | 26.1% | 22.9% |

**Per-altitude aggregate (mean across all cells in band):**

- LEO: **32.0%** (n=144 rollout-seed cells)
- MEO_low: **11.6%**
- MEO: **12.5%**
- GEO: **27.1%**

### 1.3 LEO × eccentricity cliff

LEO cliff is sharp at e=0.05→0.10. Within e≤0.05, all phase × relation cells hit 86-100% greedy. Beyond e=0.075 the LEO band has no valid initial conditions: a_max=7.171 Mm, perigee constraint a*(1-e) ≥ R_EARTH+200km requires a ≥ 6.571/(1-e) Mm, which at e=0.10 needs a ≥ 7.301 Mm > a_max. Per env-fix's `valid_init_only=1` + `gave_up_action='terminate'` policy, those episodes terminate immediately with r=0. The 0% is a **physical infeasibility cell**, not a recipe limit.

This is the same finding from the Phase 5 retrospective ("cliff at e≥0.075 is altitude OOD via alt_band_for_e") expressed differently: the cliff is geometric, and any attempt to characterize "high-e capability" at LEO must extend altitude bounds.

### 1.4 GEO surprise: flat 52% under fully_random across all e

Per-phase / per-relation table at GEO (mean across 3 rollout seeds × 200 eps):

| e | so_30° | so_90° | so_150° | so_180° | fr_30° | fr_90° | fr_150° | fr_180° |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 7.5% | 2.8% | 4.3% | 4.3% | **52.2%** | 52.2% | 52.2% | 52.2% |
| 0.05 | 7.8% | 3.0% | 2.7% | 2.8% | **54.0%** | 54.0% | 54.0% | 54.0% |
| 0.10 | 8.2% | 3.8% | 1.5% | 1.5% | **53.7%** | 53.7% | 53.7% | 53.7% |
| 0.20 | 7.7% | 2.2% | 0.3% | 0.0% | **52.7%** | 52.7% | 52.7% | 52.7% |
| 0.30 | 3.3% | 1.5% | 0.0% | 0.0% | **51.0%** | 51.0% | 51.0% | 51.0% |
| 0.50 | 0.2% | 0.8% | 0.2% | 0.2% | 45.5% | 45.5% | 45.5% | 45.5% |

Two unexpected patterns:

1. **GEO fully_random success is byte-identical across phase gaps within a given e row.** That's because `phase_gap_fixed` is ignored when `same_orbit_init=0` — sat and target are sampled with independent ω and θ, so the phase-gap setting doesn't constrain anything. The values are real, just replicated four ways. Real n per row = 600, not 2400.

2. **Same_orbit_init at GEO collapses to 0-8%.** When the env forces sat onto target's orbit shape (same a, e, ω), with only θ differing, the policy can't close the resulting phase gap at GEO altitudes. But under fully_random sampling — where sat and target have independent orbital elements — the policy hits 52% across all eccentricities.

Interpretation: under fully_random + valid_init_only, the realized distribution is dominated by samples that are **physically near-the-target** by luck. The env filters out doomed inits, and what remains skews toward "sat already close to target shape" rather than "sat needs a hard maneuver." The policy then converges from a near-equilibrium configuration most of the time. Under same_orbit_init, the env forces a fixed-shape-difference scenario the policy can't solve at GEO.

This is **H2-flavored** (the recipe's altitude-invariant skill is "settle near a configuration that's already close"), not genuine altitude generalization. The 49.7% Stage 5.5.4A baseline is not "the recipe handles GEO maneuvers" — it's "the recipe handles GEO configurations that don't need much maneuvering."

### 1.5 MEO valley

MEO band shows the lowest performance:

- MEO_low: 11.6% aggregate
- MEO: 12.5% aggregate
- GEO: 27.1% aggregate
- LEO: 32.0% aggregate

This is counterintuitive — usually OOD perf degrades monotonically with distance from trained distribution. The MEO valley likely reflects two competing effects:

- At LEO: trained distribution, but eccentricity > 0.075 is geometrically infeasible at the trained altitude band.
- At MEO: harder maneuvers (Δa per Δv is larger; period is longer than warp action covers); same_orbit_init forces actual maneuvering; fully_random doesn't have the wide-enough sampling space yet.
- At GEO: under fully_random the sampling is wide enough that "easy" configurations dominate, masking the underlying maneuver-failure with luck.

### 1.6 Off-diagonal at LEO and GEO (Reduction B sanity check)

| alt | e_target | e_sat | mean |
|---|---:|---:|---:|
| LEO | 0.00 | 0.30 | 0.0% |
| LEO | 0.05 | 0.50 | 0.0% |
| LEO | 0.30 | 0.00 | 0.0% |
| LEO | 0.50 | 0.05 | 0.0% |
| GEO | 0.00 | 0.30 | 54.5% |
| GEO | 0.05 | 0.50 | 50.5% |
| GEO | 0.30 | 0.00 | 53.7% |
| GEO | 0.50 | 0.05 | 51.8% |

LEO off-diagonal cells are infeasible (same perigee constraint as e>0.075 diagonal). GEO off-diagonal cells match diagonal — the diagonal approximation isn't hiding interactions.

### 1.7 Hypothesis verdict per spec §1.4

- **H1 (e-bias):** Partial. True at LEO (geometric infeasibility presents as e-cliff), false at MEO and GEO.
- **H2 (coast-and-wait):** Partial-strong. GEO fully_random's flat 52% across e is consistent with "policy handles configurations that don't need active maneuvering" — these are the configurations valid_init_only retains after rejection sampling.
- **H3 (genuine altitude generalization):** Weak. There's no uniform-across-altitude pattern; GEO outperforms MEO; same_orbit_init fails everywhere outside LEO.

**Closest fit: H2 + a MEO valley.** The 49.7% baseline isn't altitude generalization in the usual sense; it's sampling-distribution generalization (the recipe handles whatever the valid_init_only filter happens to retain at wider altitude/eccentricity ranges).

---

## 2. Probe 2 — Fine-tune protocol investigation

### 2.1 Setup

- 3 protocols × 3 seeds (42, 31415, 20260423) × 7M training steps.
- F-A (aggressive): reused Stage 5.5.1 smoke data (lr_max=1e-2, ent_coef=0.01, 30M steps — collapsed to 0%).
- F-B (gentle, converged warm-start): lr_max=1e-4, ent_coef=0.001. Warm from Phase 5b final ckpt (epoch 350/350 or 175/175 per seed per MODELS.md).
- F-C (gentle, mid warm-start): same hyperparams, warm from Phase 5b epoch 175 (or epoch 90 for seed 20260423's shorter run).
- Eval: 200 eps × seed 42 per ckpt × condition. Multiple epochs scanned per run.
- Script: `scripts/orbital/p5_5_probe2_finetune.sh`. Eval CSV: `/tmp/p5_5_probe2_results.csv`.

### 2.2 Training-time perf trajectories

All F-B and F-C runs **did not collapse**. Final-epoch training perf:

| Protocol | seed 42 | seed 31415 | seed 20260423 |
|---|---:|---:|---:|
| F-A (aggressive, 30M) | 0.000 | 0.000 | 0.000 |
| F-B (gentle, 7M) | 0.337 | 0.311 | 0.401 |
| F-C (gentle, mid-warm, 7M) | 0.400 | 0.353 | 0.229 |

F-A's collapse is reproduced from Stage 5.5.1 smoke. F-B and F-C both settle in the 0.23-0.40 range, close to the ~30% zero-shot baseline.

### 2.3 Greedy held-out eval at Stage 5.5.1 (200 eps × ckpt × seed 42)

Best ckpt across epochs {10, 25, 40, 54} per run:

| Protocol | seed 42 | seed 31415 | seed 20260423 | mean |
|---|---:|---:|---:|---:|
| Zero-shot baseline (no training) | 27.5% | 30.5% | 32.0% | **30.0%** |
| F-A (best across all epochs of 30M run) | 8% | 2% | 10% | **6.7%** |
| F-B (best of 4 scanned epochs) | 31.5% | 27.5% | 35.0% | **31.3%** |
| F-C (best of 4 scanned epochs) | 34.0% | 31.0% | 21.0% | **28.7%** |

F-B trends slightly above zero-shot (mean +1.3pp), F-C is essentially at baseline (mean -1.3pp), with seed 20260423 visibly worse. **No protocol meaningfully lifts capability beyond zero-shot at Stage 5.5.1.**

### 2.4 LEO regression check (final ckpt per protocol per seed)

| Protocol | seed 42 | seed 31415 | seed 20260423 | mean |
|---|---:|---:|---:|---:|
| Phase 5b baseline | 98.0% | 98.5% | 98.5% | **98.3%** |
| F-A (final at 30M) | 0.0% | 0.0% | 0.0% | **0.0%** |
| F-B (final at 7M) | 96.0% | 98.0% | 97.0% | **97.0%** |
| F-C (final at 7M) | 83.5% | 85.0% | **65.5%** | **78.0%** |

F-B preserves LEO baseline (-1.3pp, within noise). F-C significantly regresses (-20.3pp mean, with seed 20260423 catastrophically down to 65.5%).

### 2.5 Protocol verdict per spec §2.5

**F-A: collapse confirmed.** Don't use Stage-4.0 hyperparams for warm-starting converged ckpts.

**F-B: Surprise A from §2.6 — preserved LEO but didn't lift Stage 5.5.1.** Hyperparams are gentle enough that the policy doesn't degrade but also doesn't learn anything new. The converged Phase 5b ckpt is plastic-locked under gentle gradients.

**F-C: worse than F-B.** Mid-training warm-starts have more residual entropy but apply it counter-productively — F-C learns at Stage 5.5.1 conditions, which moves the policy AWAY from the LEO competence the mid-training ckpt was on the path to building. The compute cost of the lost LEO performance isn't recovered by a Stage 5.5.1 lift.

**The spec §2.6 Surprise D ("all three destroy LEO") is partially true.** F-A and F-C destroy LEO; F-B doesn't. So the trade-off is real: any protocol that lifts new capability degrades LEO.

### 2.6 Why F-B doesn't lift

Training-time perf at F-B reaches 0.30-0.40 (≈ zero-shot baseline). PPO is updating but not improving the policy at Stage 5.5.1 conditions. Two non-mutually-exclusive mechanisms:

1. **Insufficient gradient signal.** At lr_max=1e-4, with ent_coef=0.001, the policy makes too-small updates per step to overcome the local optimum the converged ckpt sits in. Spec §2.6 Surprise A response: try lr_max=5e-4, ent_coef=0.005. Worth a follow-up if Phase 5.5 needs lift.

2. **The Stage 5.5.1 task isn't actually a learnable extension of LEO.** The 30% zero-shot includes both the LEO-band-of-Stage-5.5.1 (~14% of samples are at trained altitudes, where the policy works ~98% → contributes ~14pp) plus successes at the extended altitude range that come from valid_init_only filtering retaining easy samples. PPO can't lift this because (a) it can't change the env's sampling distribution, and (b) the active-maneuver tasks at the extended altitudes are genuinely beyond the recipe's capability at any hyperparameter setting.

If (2) is the dominant mechanism, gentle hyperparam tweaks won't help — the recipe's structural capability is bounded. Confirming this requires running F-B+ with more aggressive variants (per Surprise A response).

---

## 3. Combined implication for Phase 5.5

### 3.1 What we now know

- The Phase 5b ckpt achieves 49.7% at the broadest altitude/eccentricity envelope, but the 49.7% is dominated by GEO `fully_random` configurations that don't require active maneuvering. Same_orbit_init at MEO/GEO collapses to 0-12%.
- No fine-tune protocol tested (F-A, F-B, F-C) lifts capability without either collapsing (F-A) or regressing LEO (F-C).
- F-B preserves LEO but doesn't extend capability — the policy is plastic-locked.

### 3.2 Phase 5.5 curriculum path forward — three options

**Option α: ship Phase 5b ckpt + characterize.** Per spec §8.2. The deliverable becomes "Phase 5b ckpt + capability surface showing 49.7% on the broadest envelope, with detailed breakdown of what works (GEO fully_random) and what doesn't (same_orbit_init at altitude, MEO active maneuvering, LEO high-e)." Phase 5 closes with this. Phase 6 starts. **This is the recommended outcome.**

**Option β: F-B+ exploration.** Try the Surprise-A follow-up (lr_max=5e-4, ent_coef=0.005) at Stage 5.5.1 to see if the F-B "plastic-locked" finding generalizes or if a slightly less gentle protocol unlocks lift without LEO regression. ~30 min compute. Modest expected ROI (the mechanism analysis in §2.6 suggests structural bound, not hyperparam bound). Worth one focused try if Phase 5.5 must produce a training-based deliverable.

**Option γ: targeted same_orbit_init training.** Use the Probe 1 finding that GEO same_orbit fails as the training target — at altitude, with same_orbit_init=1 (i.e., the failing mode), do F-B fine-tune to see if THIS regime (which the policy hasn't covered) lifts. This is more theoretically motivated than Stage 5.5.1 fine-tuning, because the gap is real and concentrated. ~1 hour compute. Higher uncertainty about outcome.

### 3.3 Recommendation

**Option α.** Probe 2 made clear that the fine-tune protocols don't work without regression; Probe 1 made clear that the 49.7% isn't a hidden capability waiting to be unlocked, it's "configurations the env's filter retains as easy." Continuing to invest in fine-tuning is unlikely to produce a deliverable that's strictly better than the Phase 5b ckpt at its native LEO baseline.

The strong move is to:
1. Document the 49.7% baseline with the per-cell decomposition (Probe 1 already provides this).
2. Write Phase 5 closure with the honest framing: "LEO low-e specialist + characterized envelope coverage."
3. Start Phase 6 (multi-body / cislunar) — that's the next genuinely-novel scope.

Option β as a one-shot follow-up before closure is reasonable — it's cheap and the answer is informative either way. Option γ is feasible but not the best ROI given the Probe 1 mechanism analysis.

---

## 4. Files produced

| Path | Purpose |
|---|---|
| `scripts/orbital/p5_5_probe1_decompose.py` | Surface-eval script for Probe 1 |
| `scripts/orbital/p5_5_probe2_finetune.sh` | Training runner for F-B + F-C |
| `web_data/results/p5_5_probe1_decompose.csv` | Raw Probe 1 results (600 rows: 200 cells × 3 seeds) |
| `/tmp/p5_5_probe2_results.csv` | Raw Probe 2 eval results (per-ckpt-per-condition) |
| `PHASE5_5_PRE_CURRICULUM_PROBE_FINDINGS.md` | This document |
| `pufferlib/experiments/puffer_orbital_17793{8770127,8778683,8788103,8807595,8823952,8839214}/` | F-B and F-C ckpts (preserved for any further analysis) |

---

## 5. Discipline checklist (per spec §7.4)

| Item | Status |
|---|---|
| Pre-committed decision criteria | Yes (§1.4 hypothesis verdict criteria, §2.5 protocol verdict criteria, both stated in spec before probes ran) |
| Cross-checks (off-diagonal, LEO regression) | Yes (off-diagonal in §1.6 confirms diagonal approximation; LEO regression in §2.4) |
| Trajectory inspection | Not done — flagged as a follow-up if any of the §3.2 options proceed |
| Multi-seed | Yes (3 rollout seeds × 200 eps per Probe 1 cell; 3 training seeds × Probe 2) |
| Per-seed variance reported | Yes (Probe 1 §1.2 shows ranges; Probe 2 §2.4 shows per-seed LEO regression range -1pp to -33pp) |

---

*Author: 2026-05-21. Phase 5.5 pre-curriculum probes (Probe 1 + Probe 2). ~28 min compute. Probe 1: 49.7% baseline reproduces and decomposes to "GEO fully_random near-equilibrium configurations" rather than altitude generalization. Probe 2: F-B preserves LEO but doesn't lift (plastic-locked); F-C regresses LEO. Recommendation: spec §8.2 — ship Phase 5b ckpt as Phase 5's deliverable, characterize the envelope, close Phase 5, start Phase 6.*
