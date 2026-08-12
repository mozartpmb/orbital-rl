# T5 — Extension Campaigns: Embedded C, Full 3D, Angles-Only Navigation

> **Status: COMPLETE (2026-08-11 → 08-12).** Three parallel extensions of the corrected-
> dynamics stack, run under the T3 discipline (recon → measured design → adversarial
> red-team → flagged changes with bit-exact anchors → independent-oracle fuzz →
> curriculum with held-out evals). `main` remains frozen at tag
> `campaign-2026-08-11-complete`; the extensions live on branches **ext-3d** and
> **ext-nav** (both pushed), with the embedded port additive on `main`.
> Recon/red-team reports: `scripts/orbital/ext_recon/reports/` (9 reports + 2 red-teams
> + curriculum-literature sweep). wandb: `orbital-rl` project, groups `t5-*`.

## 1. T5-EMB — embedded C port of the flight policy ✅ (on `main`)

PyTorch prototype → allocation-free C (encoder+LSTM+head, 139k params): **100.0000%
action agreement** over 7,048 obs vectors (max logit diff 5.2e-5 = 55 ULP); fully-C
env+policy binary scores **200/200** with an action stream md5-identical to torch;
p50 **50.5 µs** single-core inference = **~10⁶× real-time margin** vs the 60 s decision
epoch; worst-case stack 3,840 B, no allocation, ASan/UBSan clean; `make poly` needs only
expf/tanhf. `scripts/orbital/embedded/`.

## 2. T5-3D — full 3D orbits, plane changes ✅ (branch `ext-3d`)

Design (post red-team): classical elements + singularity-free consumer combinations
(NOT equinoctial — conditioning measured across 7 decades of inclination); shaping
mode 2 = mean-longitude phase (target-plane gauge) + L1 Δv ledger with the derived
plane coefficient (1.0·v·‖ĥ_s−ĥ_t‖); rotation-form relative-inclination sampler;
obs repurposed into dead slots 21–32 (all validated decoders preserved); Discrete-30
(normal ±1/10/25 + 4 combined tangential+normal rows). Anchors: legacy eval bit-exact,
T3 canonical trajectories **bitwise identical over 425,477 rows**, 3D fuzz 18/18 vs the
independent universal-variable oracle, Φ-mode-2 ≡ mode-1 at Δi=0 to 0/200,000.

**Results (held-out 200 eps, seed 123):**
- **X3 (LEO, ±180°, e≤0.05, ΔI_rel≤1.0°): 3/3 seeds at 200/200** — 100% success in
  every ΔI bin, 100% plane-action usage beyond the 0.3775° free-plane zone, and the
  policy prefers the *combined* burns (the vector-addition-optimal maneuver).
- **V-ladder (wide envelope): V2 200/200 → V3 200/200 → V4 200/200 → V5 199/200** —
  V5 = e≤0.30 (realized max 0.298), 300–8,000 km, ΔI≤0.75°, ±180°, 100-h episodes;
  sole failure a cap timeout at ΔI=0.08° (a Δv/time corner, not a plane failure).

**The instructive failure (documented in full):** the first wide-3D ladder scored
70%/7% (W3/W4). Diagnosis by control experiment (`di_max=0` → 199/200) found the
**11th metric-vs-implementation bug** of this project's history: `de_max` was inert
under 3D — the e-disc is drawn node-relative and the plane rotation randomized RAAN,
destroying the bound in inertial space (realized mismatch 4.5–8× the knob → the e-match
leg alone exceeded the whole 478 m/s budget in 28–54% of episodes). Two-line fix
(preserve inertial ϖ through the rotation), verified to knob-exact; the corrected
ladder then passed outright. W4 additionally demonstrated textbook value-collapse under
~57% infeasible mass (trained child worse than its untrained parent — negative
transfer/CICS, see the literature report), with `cap_terminal_reward=0` making
run-out-the-clock genuinely reward-optimal below the p < R_f/(R_s+R_f) threshold.
`scripts/orbital/ext3d/diag/`, `scripts/orbital/ext_recon/reports/lit_infeasible_curricula.md`.

Deferred, documented: coast-to-node macro-action (τ≤5 cranking already 1.00× optimal at
LEO), J2 (obs slots reserved), MEO-3D lineage.

## 3. T5-NAV — angles-only navigation, policy trained on estimates ✅ (branch `ext-nav`)

Stack: `OrbitalNav(Orbital)` worker-side wrapper (zero C changes); batched
modified-polar recursive filter in the loop; **acquisition surrogate at training**
(CRLB-conditioned on realized Δv — the red-team measured real batch IOD at 17–90× the
training step) with the **real batch solver at eval** (surrogate validated end-to-end:
both paths give the canonical the same 69.5% zero-shot). Sensor cadence decoupled from
decisions (welding them: 61/100 — the T4 finding reproduced in-wrapper).

**Results (bearings-only, real acquisition at eval, 200 eps each):**

| arm | native | truth-tax | blind÷acquired burn rate |
|---|---|---|---|
| canonical zero-shot | 69.5% | — | 1.94× |
| NB1-warm | 98.0% | 0.0 pp | 0.26× |
| NB1-fresh-42 | 99.0% | 0.0 pp | 0.41× |
| NB1-fresh-7 | 99.5% | 0.0 pp | 0.52× |
| NB1-fresh-1337 | 98.5% | 0.0 pp | 0.42× |

Pooled fresh **594/600 = 99.0%**; all arms **790/800 = 98.75%**. Failure causes went
from `38 collisions + 22 stranded` to ≤3 benign cap timeouts per arm; every arm reaches
acquisition with a full tank (median pre-acquisition Δv 75 → 0 m/s).

**The honest scientific finding:** estimate-training **inverted the blind-burn
pathology** (burn-while-blind ratio 1.94× → 0.26–0.52×) but did **not** learn the
predicted 1 m/s observability burn — acquisition latency is unchanged (~105 min). The
policies learned the complementary strategy: *coast through the blind window, then
maneuver* — correct for this envelope, where bearings accrue during a coast anyway.
The canonical's 30-point gap was a **guidance pathology (fuel spent on wrong
information), not a navigation deficiency**. Also: all three fresh seeds bootstrapped
(1337 included) — the historical fresh-seed fragility is a property of reward/curriculum
context, not observation structure. Unfixed and stated: batch-IOD acceptance gates still
certify multi-1,000-km epoch errors (self-consistency ≠ correctness); filter consistency
was never the problem. The NAV-F dual-control experiment (maneuver-for-observability at
the tight box, where drift-only nav error is 52% of tolerance) remains designed-not-run:
arms T-truth/T-RB/T-BO(+Σ/−act) with counterfactual-information-per-burn as the metric.
`EXTNAV_RESULTS.md` (branch), `scripts/orbital/ext_recon/reports/nav_F_observability.md`.

## 4. Registry

| artifact | where |
|---|---|
| Embedded port | `main`: `scripts/orbital/embedded/` |
| 3D ckpts: X3 ×3 seeds, V2–V5 | branch `ext-3d`: `models/t3/seed*_X3_3d_di1deg.pt`, `seed42_V{2..5}_wide3d.pt` |
| Nav ckpts: 4 NB arms | branch `ext-nav`: `models/t3/extnav_nb1_*.pt` |
| Diagnosis + fix | branch `ext-3d`: `scripts/orbital/ext3d/diag/`, orbital.h de_max fix commit |
| All recon/red-team/lit reports | `main`: `scripts/orbital/ext_recon/reports/` |

Merging the branches to `main` is deliberately left as a user decision — both are
fast-forward-clean from the tag and fully pushed.
