# MODELS.md — Canonical Checkpoint Registry

Logical-name → on-disk-path → published-result mapping for the orbital RL project. **All 65 `models/t3/*.pt` and all `models/t11/*.pt` are now tracked in git** (an earlier revision of this line said checkpoints were untracked except the five Phase 5e weights — that has not been true since the T5 era). This file remains the durable record of what each one is, what it came from, and what it scored.

Last updated: 2026-08-18 (T5–T11 registry added — see §"T5–T11" below). Campaign narrative and the bug ledger live in **`CAMPAIGNS.md`**.

> **⚠️ 2026-08-10 corrections — read before quoting any number from this file:**
>
> 1. **The raw-data-backed headline is 93.7% mean, 88.0–97.5% range** (5 Phase 5e
>    seeds × 200 eps, deterministic, LEO 300–800 km, both e ~ U(0, 0.05), phase gap
>    ±π; `web_data/results/multiseed_escan.csv` e_max=0.05 row, reproduced bit-exact
>    post-classifier-fix in `web_data/results/successbox_scan.csv`). The Phase 5b
>    "96.4% multi-seed" below has **no raw eval output anywhere in the repo** (the
>    collapsed seeds' runs were never preserved) — treat it as reported, not verified.
> 2. **The "2 of 5 seeds collapse" bimodality belongs to the retired Phase 5b
>    recipe.** The superseding Phase 5e `valid_init_only=1` recipe produced 5/5
>    working seeds (88.0–97.5%). Do not pair the collapse stat with current numbers.
> 3. **The Phase 5e seed-42 row below was wrong** (dir/epoch cross-wired). MD5-settled:
>    seed 42 = `puffer_orbital_177765503091/model_puffer_orbital_000325.pt`
>    (= `models/phase5e/seed42_stage4_best.pt`); `177765655537` is seedA at epoch 375.
>    `models/README.md` was correct; the table below is corrected in place.
> 4. **Terminal criterion context for every number here:** success = 30 km position
>    AND 50 m/s relative velocity vs the propagated target. The 2026-08-10 success-box
>    scan (`successbox_scan.csv`) shows the policy zero-shots **2.0%** at 5 km / 1 m/s
>    and **0.0%** at 1 km / 0.5 m/s — the deliverable is a far-field/terminal-approach
>    policy, not a capture policy. Its 10-action head cannot reach the ±1/±2 m/s fine
>    burns (M3) that velocity-nulling below its 5 m/s burn quantum would need.
> 5. **All "e_max ≥ 0.10" rows measure the same task** (LEO geometry caps realized
>    e at ≈ 0.084; realized mean ≈ 0.028) and pre-env-fix rows at e_max ≥ 0.5 are
>    additionally contaminated by 256-cap doomed inits. See
>    `PHASE5_PRE_CLOSURE_MECHANISM_FINDINGS.md` for the Φ-clamp leak retraction
>    (any MEO/GEO `fully_random` "capability" published before 2026-08-10 is an
>    eval artifact; corrected surface in `p5_5_probe1_decompose_v2.csv`).
> 6. **DYNAMICS FIX (f55d9cb, later on 2026-08-10) — supersedes everything above
>    for forward claims.** `true_to_mean()` was inverted: every burn teleported the
>    chaser along-track by ≈ 2e·sin(θ) (median 24.5° of free phase per episode).
>    ALL checkpoints in this file were trained on those dynamics and every number
>    in this file describes them. Under corrected dynamics the canonical seed-42
>    ckpt scores **13.0%**; the best corrected-dynamics checkpoint so far is the
>    interim re-adapt `puffer_orbital_178640242476/model_puffer_orbital_000025.pt`
>    at **33.5%** greedy. Fresh bootstrap fails at 2× budget (shaping fights
>    drift-orbit phasing — see `T1_DYNAMICS_FIX_FINDINGS.md` §4). Do not quote any
>    success/Δv figure from this registry without stating "pre-fix dynamics".

---

## T3 corrected-dynamics recovery (2026-08-11) — CURRENT CANONICAL

> Everything in this section is measured under **corrected dynamics** (post `f55d9cb`)
> with the T3 env config, and supersedes correction-banner item #6 for forward claims.
> Full campaign record: `T3_RECOVERY_CAMPAIGN.md`; recon reports in
> `scripts/orbital/t3/reports/`.

**T3 env config (required to reproduce — all are runtime kwargs, legacy defaults differ):**
`shaping_mode=1, shape_gamma=1.0, phase_gap_mode=1, phase_obs_mode=1,
episode_cap_steps=3000, cap_terminal_reward=0.0, valid_init_only=1`.
Checkpoints have **Discrete(16) heads** — do NOT pass `--legacy-action-space`.
`phase_obs_mode=1` checkpoints are incompatible with pre-T3 checkpoints (obs semantics).

**Headline (LEO 300–800 km, e ≤ 0.05 both, independent ω, physical gap ±180°, 30 km/50 m/s box):**

| Logical name | Train seed | Held-out success | Ckpt (durable copy) | Experiment dir |
|---|---|---|---|---|
| **t3_canonical_seed42** | 42 | **100.0%** — 200/200 greedy s123, 200/200 stochastic, 200/200 at legacy cap 2000, 500/500 s777 | `models/t3/seed42_L2_headline.pt` (MD5 10a8ba37…) | `puffer_orbital_178642097817/model_..._000382.pt` |
| t3_seed42_L1 (e=0 rung) | 42 | 200/200 at L1 conditions | `models/t3/seed42_L1_final.pt` | `puffer_orbital_178642016215/model_..._000382.pt` |
| t3_batch1_L2_a / _b | 7 / 1337† | 200/200 each (headline) | `models/t3/batch1_L2_{a,b}_*.pt` | `…258109` / `…259027` |
| t3_batch1_L1_a / _b | 7 / 1337† | 200/200 each (L1 conditions) | `models/t3/batch1_L1_{a,b}_*.pt` | `…176815` / `…177125` |
| t3_seed20260423 | 20260423 | 200/200 (headline) | `models/t3/seed20260423_L2_headline.pt` | `…431536` (L1 `…360522` 200/200) |
| t3_seed31415 | 31415 | 200/200 (headline) | `models/t3/seed31415_L2_headline.pt` | `…576174` (L1 `…505036` 200/200) |

**FIVE-SEED FINAL: 5/5 training seeds at 100.0% held-out on the headline task —
2,900 held-out episodes, zero failures of any kind.** (Pre-fix bug-assisted:
93.7% 5-seed mean. Seed 1337 — historically a 0% flatline seed — now 100.0%.)

† Batch-1 ladders ran concurrently and raced dir attribution; seed↔dir mapping above is
from launch-time dir-ID stamp analysis (campaign doc §6), each run's `--train.seed` is
certain from its command line, and **every one of the four dirs was independently
evaluated at 200/200**, so the multi-seed claim is attribution-independent. Batch 2
(20260423, 31415) ran sequentially with clean attribution.

**Wide-eccentricity family (seed 42, 2026-08-11):**

| Logical name | Conditions (held-out 200/200 each) | Ckpt |
|---|---|---|
| **t3_widee_L3** | e≤0.15, 300–2000 km, ±180° (LEO obs scales; also 200/200 at headline) | `models/t3/seed42_L3_widee.pt` (`…670436`) |
| **t3_wide_WL4** | **e≤0.30, 300–8000 km, ±180°, cap 6000** (wide obs scales: `obs_alt_scale_m=8e6, lvlh_scale_m=1.5e7` — REQUIRED at eval; incompatible with LEO-scale ckpts) | `models/t3/seed42_WL4_wide.pt` (`…988768`) |
| t3_wide_WL1–WL3 | lineage rungs, 200/200 each | `models/t3/seed42_WL{1,2,3}_wide.pt` |

WL4 realized e_target p50 0.140 / p90 0.261 / max 0.293. de_max bounds e-vector
mismatch (0.08 at WL4).

**T4 follow-ups (2026-08-11, campaign doc §8). Env now Discrete-20 (3h/6h warps +
radial ±1 m/s; exposed default still Discrete(16)); eval 20-head ckpts with
`--legacy-action-space 20`. wandb: wandb.ai/mozartpmb_training/orbital-rl.**

| Logical name | Conditions (held-out, seed 123) | Result | Ckpt |
|---|---|---|---|
| t3_wide (multi-seed) | WL1–WL4 × seeds 42/7/1337/20260423 | **3,199/3,200 (99.97%)**; WL4: 800/800 | `models/t3/seed{7,1337,20260423}_WL4_wide.pt` |
| **t4_tb3_box10k10** | headline @ **10 km/10 m/s**, D20 | **200/200** | `models/t3/seed42_TB3_box10k10.pt` |
| t4_tb4_box5k2 | headline @ 5 km/2 m/s, D20 | 200/200 | `models/t3/seed42_TB4_box5k2.pt` |
| **t4_tb5_box5k1** | headline @ **5 km/1 m/s**, D20 | **191/200 (95.5%)**, 9 cap timeouts; capture p50 0.91 m/s | `models/t3/seed42_TB5_box5k1.pt` |
| **t4_m5_meo** | **300–20,200 km, e≤0.50**, de 0.10, ±180°, cap 12000, D20, scales 2.1e7/4e7 (REQUIRED at eval) | **199/200 (99.5%)**; realized e_t p50 0.216/max 0.500 | `models/t3/seed42_M5_meo.pt` (M4: `seed42_M4_meo.pt`) |

EKF at wide envelope: nominal = truth = 200/200 (`t4_relnav_wl4.csv`).

**Supporting numbers (seed-42 canonical):** Δv p50 235 m/s = 1.58× shape-only two-impulse
lower bound (`t3_headline_characterization.csv`); Lambert time-matched ratio p50 1.14×
(pre-fix impossibility 0.31× resolved, `t3_lambert_corrected.csv`); EKF closed-loop
truth 100.0% = EKF-nominal 100.0% at 60 s nav cadence (`t3_relnav_corrected.csv`).
Capture at box edge (29.2 km / 34.8 m/s p50) — far-field rendezvous, not terminal capture.
Classical scripted-GNC reference: 99.2% (500 eps) / 100 of 100 at T3 config
(`scripts/orbital/t3/expert_controller.py --t3`).

---

## T5–T11 — extensions, navigation, J2, and the generalist (2026-08-11 → 08-18)

> Campaign narrative, the bug ledger and doc routing: **`CAMPAIGNS.md`**.
> Every number below is 200 held-out episodes at **eval seed 123** unless stated.
> `lineage` names the warm parent; **derived** rows are produced by a tool
> (column-zeroing, head expansion, normalizer rescale), not by training.
>
> **Action heads differ across this era and are not interchangeable** — a
> checkpoint only loads into a model of its own head size. Verified by loading
> each file: `extnav_nb1_*` = 16, `extnav_TB5_* / extnav_navf_*` = 20, all
> `n3dnav_* / extj2_* / j2nav_*` = 30, `j2wait_* / t11_*` = 31. Encoder is
> `(128, 38)` in every one.
>
> **Normalizer family is part of the config, not a detail.** narrow =
> `obs_alt_scale_m 1.6e6 / lvlh_scale_m 6.371e6`; wide = `8e6 / 1.5e7`. A
> checkpoint run under the wrong family scores ~0 (`GEN_MATRIX.md`) — though
> `T11_GENERALIST.md` later showed the barrier is an exactly removable
> reparameterization, not a capability gap.

### Navigation — 3D + bearings-only (`N3DNAV_RESULTS.md`)

Config: `scripts/orbital/nav/n3dnav_campaign.sh`, `n3dnav_tb_campaign.sh`,
`n3dnav_e_campaign.sh`. X3 base = `dim3_mode 1, di_max_rad 0.017453 (1°),
e_max 0.05, obs 1.6e6/6.371e6, shaping_mode 2, dv_ref 700, cap 3000, D30`,
50M steps/arm, train seed 42 unless noted.

| Logical name | Lineage | Config pointer | Result (seed 123) | Ckpt |
|---|---|---|---|---|
| n3dnav_warm_X3 | **derived** from `seed42_X3_3d_di1deg.pt`, cols 29–32 zeroed | campaign §"THE WARM START" | bit-identical to parent; as floor **58.0%** blind | `models/t3/n3dnav_warm_X3.pt` |
| n3dnav_T-truth3d | **byte-identical copy** of `n3dnav_warm_X3.pt` (eval relabel, no training) | `n3dnav_campaign.sh` stage 1 | **116/200 = 58.0%** blind; truth 200/200 | `models/t3/n3dnav_T-truth3d.pt` |
| n3dnav_N1-rb3d | `n3dnav_warm_X3.pt` | stage 2, `nav_mode rb_ekf` | 200/200 in-mode; **cross-mode blind 60.5%** | `models/t3/n3dnav_N1-rb3d.pt` |
| **n3dnav_T-BO3** | `n3dnav_warm_X3.pt` | stage 3, `nav_mode bearings_only` | **200/200 = 100.0%** real batch IOD; truth 200/200 | `models/t3/n3dnav_T-BO3.pt` |
| n3dnav_T-BO3-X3-s7 / -s1337 | `n3dnav_warm_X3.pt` | `n3dnav_tb_campaign.sh` stage 4, seeds 7 / 1337 | **200/200 each**, native and truth → rung-1 3/3 seeds | `models/t3/n3dnav_T-BO3-X3-s{7,1337}.pt` |
| n3dnav_warm_TB5 | **derived** from `seed42_TB3D_box5k1.pt`, cols 29–32 zeroed | `n3dnav_tb_campaign.sh` §WARM STARTS, box 5 km/1 m/s, `w_match 0.35` | as floor **46.0%** blind (truth 97.0%) | `models/t3/n3dnav_warm_TB5.pt` |
| n3dnav_N1-rb3d-TB5 | `n3dnav_warm_TB5.pt` | tb stage 2, `rb_ekf` | 99.0% in-mode; **blind 13.0%** — below its own floor | `models/t3/n3dnav_N1-rb3d-TB5.pt` |
| **n3dnav_T-BO3D-TB5** | `n3dnav_warm_TB5.pt` | tb stage 3, `bearings_only`, box 5 km/1 m/s | **196/200 = 98.0%**; truth 199/200 | `models/t3/n3dnav_T-BO3D-TB5.pt` |
| n3dnav_T-BO3D-TB5-s7 / -s1337 | `n3dnav_warm_TB5.pt` | tb stage 5, seeds 7 / 1337 | 98.0% / 98.5%; pooled **589/600 = 98.2%** | `models/t3/n3dnav_T-BO3D-TB5-s{7,1337}.pt` |
| n3dnav_warm_E0 | **derived** from `seed42_V2_wide3d.pt`, cols 29–32 zeroed | `n3dnav_e_campaign.sh` `WS_E`, **wide** 8e6/1.5e7, `nav_r_min_m 6.571e6` | eval-only floor: E0 70.5% → E3 5.5% blind | `models/t3/n3dnav_warm_E0.pt` |
| n3dnav_e_E0 | `n3dnav_warm_E0.pt` | e-campaign rung E0, e_max 0.05 | **200/200 = 100%**; truth 200/200 | `models/t3/n3dnav_e_E0.pt` |
| n3dnav_e_E1 | `n3dnav_e_E0.pt` | rung E1, e_max 0.10, a 6.671–7.871e6 | **198/200 = 99.0%** | `models/t3/n3dnav_e_E1.pt` |
| n3dnav_e_E2 | `n3dnav_e_E1.pt` | rung E2, e_max 0.20, cap 4500 | **191/200 = 95.5%** | `models/t3/n3dnav_e_E2.pt` |
| **n3dnav_e_E3** | `n3dnav_e_E2.pt` | rung E3, e_max 0.30, a → 14.371e6, cap 6000 | **190/200 = 95.0%**; realized e_t p90 0.257 | `models/t3/n3dnav_e_E3.pt` |

### 2D bearings-only + NAV-F (`EXTNAV_RESULTS.md`, `NAVF_RESULTS.md`)

Config: `scripts/orbital/nav/nb_campaign.sh`, `navf_campaign.sh`; 2D
(`dim3_mode 0`) nav-ini defaults, `shaping_mode 1 / w_match 0.35 / dv_ref 300`,
50M steps, train seed 42 unless noted.

| Logical name | Lineage | Config pointer | Result (seed 123) | Ckpt |
|---|---|---|---|---|
| extnav_nb1_warm | `seed42_L2_headline.pt` | `nb_campaign.sh` arm `nb1_warm`, D16 | **196/200 = 98.0%**; truth 200/200, **0.0 pp tax** | `models/t3/extnav_nb1_warm.pt` |
| extnav_nb1_fresh_42 / _7 / _1337 | **fresh** | arms `nb1_fresh_*` | 99.0 / 99.5 / 98.5%; pooled fresh **594/600 = 99.0%** | `models/t3/extnav_nb1_fresh_{42,7,1337}.pt` |
| extnav_TB5_warmstart_col21zero | **derived** from `seed42_TB5_box5k1.pt`, col 21 zeroed | shared NAV-F warm start, D20 | none of its own; makes all four arms identical at t=0 | `models/t3/extnav_TB5_warmstart_col21zero.pt` |
| **extnav_navf_T_BO** | `extnav_TB5_warmstart_col21zero.pt` | `navf_campaign.sh` stage 1, box 5 km/1 m/s, D20 | **194/200 = 97.0%**; truth 97.0% (zero tax) | `models/t3/extnav_navf_T_BO.pt` |
| extnav_navf_T_BOS | same | stage 2, `+nav_sigma_channel 1` | 194/200 = 97.0%; β_adj −0.032 (z −0.62) | `models/t3/extnav_navf_T_BOS.pt` |
| extnav_navf_T_BOACT | same | stage 3, `+nav_block_fine_below_m 10000` | 185/200 = 92.5%; **β_adj −0.434 (z −8.25)** | `models/t3/extnav_navf_T_BOACT.pt` |

### J2 (`J2_RESULTS.md`, `J2_RUNG_NOTES.md`)

Config: `scripts/orbital/extj2/j2_rung_campaign.sh`. X3 base **plus**
`j2_mode 1, i_target 30–60°, raan_target_sample 0, lvlh_frame_mode 1`; trainer
`puffer_orbital` (truth state, nav out); 50M/rung; D30; train seed 42 unless
noted. **All `j2_mode=1` numbers are MEAN-ELEMENT claims.**

| Logical name | Lineage | Config pointer | Result (seed 123) | Ckpt |
|---|---|---|---|---|
| extj2_A2_j2trained | `seed42_X3_3d_di1deg.pt` | stage 2, box 30 km/50 m/s | **200/200** native; retention j2=0 200/200 | `models/t3/extj2_A2_j2trained.pt` |
| extj2_A3a_j2_box10k10 | A2 child | stage 3a, box 10 km/10 m/s | **200/200**; chain floor 75.5% | `models/t3/extj2_A3a_j2_box10k10.pt` |
| **extj2_A3b_j2_box5k1** | A3a child | stage 3b, box **5 km/1 m/s** | **198/200 = 99.0%**; floor **0/200** | `models/t3/extj2_A3b_j2_box5k1.pt` |
| extj2_A3b_j2_box5k1_s7 / _s1337 | full 3-rung chain re-trained per seed | `J2_SEED=7` / `1337` | 99.5% / 99.0%; pooled **595/600 = 99.2%** | `models/t3/extj2_A3b_j2_box5k1_s{7,1337}.pt` |
| **j2nav_T-J2BO-nav** (`_final` is byte-identical) | `n3dnav_T-BO3.pt` (unmodified) | `scripts/orbital/nav/j2nav_campaign.sh` stage 2, `nav_j2_mode 1` (MSC6J2Cov, `stm_j2='fd'`) | **192/200 = 96.0%** blind (floor 32.0%); truth 200/200; **X3 home 83.0%, −17 pp** | `models/t3/j2nav_T-J2BO-nav{,_final}.pt` |

### Drift-and-wait and the generalist (`J2_RESULTS.md` §3, `T11_GENERALIST.md`)

| Logical name | Lineage | Config pointer | Result (seed 123) | Ckpt |
|---|---|---|---|---|
| j2wait_warm_A2 | **derived** from `extj2_A2_j2trained.pt`, D30→D31, **row 30 seeded from row 17** (zero-init measured vacuous, P = 3.9e−9) | `scripts/orbital/extj2/expand_ckpt_30_to_31.py` | floor **0/200** at the node-dominant cell; home rung 200/200 | `models/t3/j2wait_warm_A2.pt` |
| **j2wait_W1_driftwait** | `j2wait_warm_A2.pt` | `j2wait_campaign.sh` stage 2 — `di_min/max_rad 2°/5°`, `di_phase_mode 1`, **cap 22000**, D31, truth state | **192/200 = 96.0%** (floor 0/200); day-warp masked 71.0%; **home rung 46.0%** | `models/t3/j2wait_W1_driftwait.pt` |
| t11_rungA_j2wide | transplanted A3b root in `models/t11/` † | `scripts/orbital/extj2/t11_campaign.sh` stage 2, single cell, 50M | E0_j2 **100.0%**, E1_j2 **99.5%** | `models/t3/t11_rungA_j2wide.pt` |
| **t11_generalist_rungB** | `t11_rungA_j2wide.pt` | t11 stage 3, `--env.t11-mixture 1`, 7-cell weighted mixture, **200M**, fuel U(0.113, 0.20), D31 | **E0 97.0 / E1 100.0 / E2 99.5 / E3 97.5 / LONGRANGE 99.0%**; **W1_driftwait 0.0%, TIGHT_5k1 0.0%** | `models/t3/t11_generalist_rungB.pt` |

† `T11_GENERALIST.md` states the rung-A root is `extj2_A3b_j2_box5k1`
transplanted into the wide family (`models/t11/t11_root_a3b_wide.pt`). The
launch-time `T11_ROOT_A` value is not recorded in the repo and the script default
names a different candidate; the stage-1 floors on disk corroborate the doc
(a3b_wide E0 92.5% / E1 93.5% versus j2bonav 88.5% / 88.0%). Documented, not
machine-confirmed.

**Two byte-identical duplicate pairs**, flagged so nobody treats them as
independent evidence: `n3dnav_T-truth3d.pt` ≡ `n3dnav_warm_X3.pt`, and
`j2nav_T-J2BO-nav.pt` ≡ `j2nav_T-J2BO-nav_final.pt` (different docs cite
different names for the same weights).

---

## Phase 5b deliverable (the working LEO low-e specialist)

Phase 5b's "two-stage curriculum" (Stage 1.0 → Stage 4.0 directly) is the shippable Phase 5 deliverable. At LEO 300-800 km / e ≤ 0.05 fully random / init_phase_gap_max = π, the recipe achieves **96.4% multi-seed mean** (3 of 5 training seeds succeed at 94-98%; 2 of 5 collapse to 2-16% — the recipe is bimodal at the edge).

### Stage 4.0 canonical ckpts (5-seed retrain)

| Logical name | Training seed | Eval seed-42 success | Ckpt path |
|---|---|---|---|
| **phase5b_canonical_31415** (headline) | 31415 | 97.7% | `pufferlib/experiments/puffer_orbital_177750405236/model_puffer_orbital_000350.pt` |
| phase5b_canonical_42 | 42 | 95.8% | `pufferlib/experiments/puffer_orbital_177750198246/model_puffer_orbital_000350.pt` |
| phase5b_canonical_20260423 | 20260423 | 95.7% | `pufferlib/experiments/puffer_orbital_177750301624/model_puffer_orbital_000175.pt` |
| phase5b_collapsed_seed_a | (1337 or 2718) | 2-16% | training runs not preserved at specific paths |
| phase5b_collapsed_seed_b | (1337 or 2718) | 2-16% | training runs not preserved at specific paths |

**V3 reproduction (2026-05-12)**: the seed-31415 ckpt re-evaluates at **98.0%** in the env-fix-landed code (commit `4b41cdc`). This serves as the backward-compat anchor for any subsequent env changes.

### Stage 4.1 ckpts (e_max=0.10, 65% multi-seed; partial extension that didn't reach Stage 4.2)

| Logical name | Training seed | Eval success | Ckpt path |
|---|---|---|---|
| phase5b_stage_4_1_seed_42 | 42 | ~65% | `pufferlib/experiments/puffer_orbital_177750529227/model_puffer_orbital_000300.pt` |

---

## Phase 5e canonical ckpts (`valid_init_only=1` retrain, contaminated headlines)

These were trained with `valid_init_only=1` to filter doomed inits but evaluated against headline numbers that the verification investigation later found to be contaminated by the 256-attempt-cap exhaustion bug (now fixed via env-fix F1). Pass-only success at high-e LEO is much higher than the published headlines (e.g., 97.1% at e=0.70 LEO vs published 71.7%).

| Logical name | Training seed | Headline (contaminated) | Pass-only (post env-fix) | Ckpt path |
|---|---|---|---|---|
| **phase5e_canonical_42** | 42 | 92.0% @ e_max=0.20 (5-seed mean is 90.2%; skewed dist — realized e ≈ 0.028) | V1 measured 90.5% @ e=0.70 LEO with cap=4096 | `pufferlib/experiments/puffer_orbital_177765503091/model_puffer_orbital_000325.pt` (= `models/phase5e/seed42_stage4_best.pt`, MD5-verified) |
| phase5e_seed_unknown_1 | ? | — | — | `pufferlib/experiments/puffer_orbital_177765658166/model_puffer_orbital_*.pt` |
| phase5e_seed_unknown_2 | ? | — | — | `pufferlib/experiments/puffer_orbital_177765658729/model_puffer_orbital_*.pt` |
| phase5e_seed_unknown_3 | ? | — | — | `pufferlib/experiments/puffer_orbital_177765659007/model_puffer_orbital_*.pt` |
| phase5e_canonical_wandb | 42 | (alt mirror) | — | `models/phase5e/canonical_seed42_stage4_best.pt` (if present) |

**Note on lost seed→dir mapping**: 4 of the 5 Phase 5e seed dirs lost their explicit seed identity due to a race condition in the parallel orchestrator at training time. Results from these dirs are valid in aggregate but cannot be matched back to specific seeds.

---

## Web data trajectory archives

Trajectories from canonical ckpt evaluations, exported via `scripts/orbital/export_web_data.py`:

| Path | Content | Status |
|---|---|---|
| `web_data/runs/phase5e_seed42_e0.05/` | Phase 5e seed 42 @ e=0.05 LEO | Clean |
| `web_data/runs/phase5e_seed42_e0.20/` | Phase 5e seed 42 @ e=0.20 LEO | Clean |
| `web_data/runs/phase5e_seed42_e0.50/` | Phase 5e seed 42 @ e=0.50 LEO | **5 of 49 contaminated** (pre-F1 cap exhaust) |
| `web_data/runs/phase5e_seed42_e0.70/` | Phase 5e seed 42 @ e=0.70 LEO | **12 of 46 contaminated** (pre-F1 cap exhaust) |
| `web_data/runs/phase5e_progression/` | Phase 5e training progression V-curve | Clean |
| `web_data/runs/phase5_env_fix_v4_contamination_flags.json` | Sidecar flagging the 17 contaminated trajectories | — |

---

## Useful eval/diagnostic scripts

| Script | Purpose |
|---|---|
| `pufferlib/scripts/orbital/eval_checkpoint.py` | Standard eval harness. Full env-kwarg support post env-fix. |
| `scripts/orbital/p5verify_perigee_scan.py` | Audits perigees in trajectory JSON files. |
| `scripts/orbital/p5wrap_surface_full.py` | Multi-cell capability surface eval (uses subprocess `eval_checkpoint.py`). |
| `scripts/orbital/p5wrap_surface_aggregate.py` | Aggregates per-cell surface results. |
| `scripts/orbital/p5e_e1_lambert.py` | Lambert reachability check (Block I E1). |
| `scripts/orbital/p5e_e2_kepler_precision.py` | Kepler propagation precision (Block I E2). |
| `scripts/orbital/p5e_e3_round_trip.py` | Cartesian↔elements round-trip (Block I E3). |
| `scripts/orbital/p5e_e5_phi_calib.py` | Φ_orbit calibration (Block I E5). |
| `scripts/orbital/p5e_e6_action_effect.py` | Action-effect peri-vs-apo asymmetry (Block I E6). |
| `scripts/orbital/p5b_step1_cap_tail_analysis.py` | Cap-tail timeout-failure analysis. |
| `scripts/orbital/p5b_step1_phi_traj.py` | Per-step Φ trajectory plot. |

---

## Post-`phase5-5-env-mods` requirement: `--legacy-action-space 10`

After commit landing the `phase5-5-env-mods` tag (M1 LVLH scaling + M2 longer warps + M3 sub-5 m/s actions), the env's action space is `Discrete(16)` but the Phase 5b/5e canonical ckpts all have a 10-dim policy logits head. **To evaluate any pre-env-mods ckpt, you must pass `--legacy-action-space 10` to `eval_checkpoint.py`** — without it, `torch.load_state_dict` will raise a `size mismatch for policy.decoder.weight: copying a param with shape torch.Size([10, 128]) from checkpoint, the shape in current model is torch.Size([16, 128]).`

This flag coerces `env.single_action_space = Discrete(10)` before policy construction, so the policy head sizes to 10. The env's `c_step` still accepts integers in `[0, NUM_ACTIONS)`; a 10-dim argmax produces ints 0-9, which are the legacy actions.

V5 multi-seed reproduction post-env-mods (2026-05-15):
- seed 31415: 98.0%
- seed 42: 95.0%
- seed 20260423: 96.5%
- **Mean: 96.5%** (published Phase 5b: 96.4%)

The published Phase 5b numbers reproduce. Backward compat is preserved.

## Conventions

- **Experiment directory naming**: `puffer_orbital_{wandb_run_id}/` or `puffer_orbital_{int(100*time.time())}/` if no wandb. The run ID is opaque; resolve to seed via wandb config or curriculum logs.
- **Ckpt naming**: `model_puffer_orbital_{epoch:06d}.pt`. Checkpoints save every `--train.checkpoint-interval N` epochs (default 200; curriculum uses 5).
- **Best-ckpt selection**: `scan_best_ckpt()` in `scripts/orbital/p5b_curriculum.sh:70-92` samples every 25 epochs + final, picks the highest `eval_checkpoint.py` success rate.

---

## How to add an entry

When new canonical ckpts are produced (e.g., Phase 5.5 stages), add a section here with: training seed, ckpt path, eval conditions, headline number(s), and any caveats (collapsed seed, contamination, etc.). Keep this file the durable source of truth for "which ckpt represents what."
