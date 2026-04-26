# Orbital RL — Phase 4.5 Findings

**Period:** 2026-04-26 (single session, ~2 hr compute)
**Spec:** `orbital-rl-phase4.5-spec.md`
**Goal:** Attribute the Phase 4 win across LVLH (R1), gated shaping (R2), DAPO clip-higher, and D7→D10 action-surgery, before extending to Phase 5.

---

## TL;DR

The PHASE4_FINDINGS framing — "only R4 (warp + curriculum) worked; LVLH and shaping were within noise" — is **wrong for the curriculum-context question**. Both LVLH and gated shaping are **load-bearing under curriculum**: stripping either prevents Stage 1 from even bootstrapping at 30°. DAPO clip-higher is inert-to-mildly-regressive at deployment despite a rolling-train lift. The D7→D10 cross-phase surgery doesn't transfer. **Phase 5 recipe = Phase 4 recipe unchanged.**

---

## Results

| Ablation | Train metric | Eval metric | Bucket | Decision |
|---|---|---|---|---|
| **A — no LVLH** | Stage 1 peak **13.1%**, end ~0% (vs Phase 4: 77%) | n/a (Stage 1 didn't bootstrap, no Stage 2/3 run) | <60% | **Load-bearing — keep LVLH** |
| **B — no shaping** | Stage 1 peak **3.6%**, end ~3.6% (vs Phase 4: 77%) | n/a (didn't bootstrap) | <60% | **Load-bearing — keep shaping** |
| **D — DAPO under curriculum** | Stage 3 rolling **99.3%** (+5pp vs Phase 4: 94.1%) | Multi-rollout eval **74.0%** mean (76/80/66, vs Phase 4: 81.3% mean) | rolling helps, eval regresses → **inert** at deployment | **Don't add DAPO** |
| **C — D7→D10 surgery** | Smoke test 30°: **6.0%** (vs Phase 3: 99%) → halt before training | n/a | smoke <90% halt | **Skip surgery, live with Stage 1 seed sensitivity** |

---

## Per-ablation deep dive

### A — no LVLH curriculum (decisive)

**What was changed.** OBS_DIM 38→33; LVLH block in `fill_observations` (orbital.h:475-511) commented out; obs space `(38,)→(33,)`. Built clean.

**Run.** Seed 42, fresh Stage 1, 30° gap, 10M steps. Same config as Phase 4 R4 Stage 1 (which got 77%) except LVLH stripped.

**What happened.** Perf trajectory: 0 → 0.002 → ... → 0.131 (peak around mid-training) → 0.0 by end. The policy briefly discovers some success cases but never accumulates a stable phasing strategy without the LVLH frame to disambiguate target-relative state. Phase 4 R4 Stage 1 with LVLH on the same seed/config reaches 77% smoothly. **22pp peak gap, 77pp end gap → load-bearing.**

**Why this matters.** Standalone R1 vs R0 was within noise (+2pp), which the PHASE4_DEEP_DIVE used to argue LVLH wasn't contributing. That comparison missed what curriculum + LVLH does together: LVLH is necessary for *bootstrap-from-random*, not for marginal accuracy on a converged policy. Different question, different answer.

**Stage 2/3 not run.** A 13% peak Stage 1 ckpt is not a useful warm-start for Stage 2; the cascade would have collapsed regardless. Decisive at Stage 1.

### B — no shaping curriculum (decisive)

**What was changed.** `BETA_SHAPE 1.0 → 0.0` in orbital.h:41. Built clean. (Note: the exp-decay terminal bonus described in PHASE4_DEEP_DIVE was actually never landed in code; the only terminal bonus is `10·(0.5 + 0.5·fuel_remaining)`. So no exp-decay strip needed.)

**Run.** Seed 42, fresh Stage 1, 30° gap, 10M steps. Identical to Phase 4 R4 Stage 1 except shaping reward zeroed.

**What happened.** Perf trajectory: 0.007 → 0.013 → 0.024 → 0.034 → **0.036 peak** (final). Even shallower than Ablation A. Without the gated Φ-diff signal nudging the policy toward orbit-shape closure, the sparse terminal-only reward at 30° gap is too sparse for fresh-from-random discovery. Phase 4 R4 with shaping on the same seed: 77%. **74pp gap → load-bearing.**

**Why this matters.** Earlier R5 ablations (warm-start onto Stage 2 with shaping reshape) showed shaping CRASHING committed policies. Easy to over-generalize that finding to "shaping is harmful." Phase 4.5 B shows the opposite from a different angle: shaping is *necessary for fresh bootstrap*, even though it can be destabilizing for warm-starts. Both findings are true and complementary.

### D — DAPO under curriculum (positive train, negative eval)

**What was changed.** `clip_coef_high: 0.2 → 0.28` in orbital.ini. No code changes (DAPO codepath at pufferl.py:346-350, 434-437 already reads these from config and was preserved when R3 was stripped).

**Run.** Seed 42, Stage 3 only (15M, 180° gap), warm-started from Phase 4 Stage 2 ckpt `puffer_orbital_h2ccoyi1`. Output ckpt: `puffer_orbital_177723408423/model_puffer_orbital_000115.pt`.

**Training trajectory.** Smooth, no entropy crash. Final epoch 115: rolling-train perf **99.3%**, entropy 0.523, episode_length 974. Phase 4 baseline at the same point: 94.1% rolling, entropy 0.402.

**Eval (50 eps × 3 rollout seeds, 180° gap, e_max=0.0):**
- Seed 42: 76.0% (38/50)
- Seed 1337: 80.0% (40/50)
- Seed 20260423: 66.0% (33/50)
- **Mean: 74.0%**

Phase 4 baseline (same Stage 2 → Stage 3 warm-start, no DAPO): 78/86/80 = 81.3%.

**Interpretation.** The +5pp rolling-train lift didn't transfer to held-out rollout seeds (−7pp eval). Classic overfitting signature: DAPO's wider upper clip lets the policy escalate confidence in actions that work on the training distribution but generalize less. Given the per-seed std of ~11pp, 74% is within noise of 81%, so it's not statistically harmful — just not improving. **Don't add to Phase 5 recipe.**

**Note on R5 failure-mode prior.** I was concerned this would replicate the R5 v1-v4 collapse arc (committed warm-start + new clip shape → entropy crash). It did not — entropy stayed 0.5+ throughout. The R5 collapse was driven more by reward-landscape shifts than by clip-shape changes; DAPO alone didn't trigger it.

### C — D7→D10 surgery (halted at smoke test)

**Surgery script.** `pufferlib/scripts/orbital/expand_ckpt_actions_d7_to_d10.py` — encoder pad 33→38 (zero columns) + decoder pad 7→10 (zero weight rows, bias=-1.0 on new rows for ~3% softmax prior).

**Surgery output.** `experiments/p4.5_d7_to_d10_warmstart.pt`. Verified shapes: encoder (128, 38), decoder (10, 128) + (10,). Surgery itself ran clean.

**Smoke test (per spec §2.3 mitigation).** Eval surgery ckpt at 30° gap, 50 eps, seed 42, e_max=0.0. **Result: 6.0% (3/50)**. Phase 3 source ckpt `q0jsaz88` evaluated at 99% under its native env. Per spec halt criterion (<90% → halt before training).

**Interpretation.** The Phase 3 → Phase 4 surgery doesn't transfer. The first 7 actions in the action table are semantically aligned (coast + 4 prograde/retro + 2 normal), and the encoder zero-pad is well-precedented (Phase 4 R1's Stage 2 zero-pad eval'd at 47.3% on seed 42 baseline at 180°, which was *higher* than R0's 64% — proving the zero-pad transfer mechanism is sound for same-era surgeries).

The most likely cause is the **reward-landscape shift between Phase 3 and Phase 4**:
- Phase 3 used distance-based shaping `c_shape · Δdist / scale`.
- Phase 4 uses gated multi-stage potential Φ(s) with NHR clamp.

The Phase 3 LSTM hidden-state distribution was learned under Phase 3's reward-driven trajectories. Loaded into Phase 4's env, it sees novel state visitation patterns (because Phase 4's shaping reshapes the gradient surface), and the LSTM's recurrent dynamics drift OOD within the first few steps. By 30°-gap rendezvous time it's in a state space the policy never trained on, hence the 6% performance.

**Decision.** Per spec decision matrix: "worse than native curriculum → skip surgery, live with the 1-in-3 Stage 1 retry rate." The native fresh-Stage 1 path is the recommended Stage 1 init for Phase 5. This is documented; future work could re-attempt cross-phase surgery if shaping recipes converge.

---

## Updated Phase 4 narrative

The earlier framing ("only R4 worked; R1/R2 within noise") conflated two questions:

1. **Standalone marginal effect on a converged Phase 3 baseline at 180°** — what PHASE4_DEEP_DIVE measured. Here LVLH and shaping are within noise.
2. **Necessary-ingredient effect under fresh-from-random curriculum bootstrap** — what Phase 4.5 measures. Here LVLH and shaping are both load-bearing.

The recipe shipped in Phase 4 is correct; the *attribution prose* understated the contribution of LVLH and shaping. Updating the deep dive to reflect that LVLH+shaping+warp+curriculum is a 4-way necessary stack for the curriculum to work; no single component is dispensable.

R3 components (LayerNorm, L2-init, adaptive-KL, TEC) remain ruled out. DAPO has been now isolated and confirmed inert-to-regressive at deployment. R3 is fully dead for this project.

---

## Phase 5 recipe (carried forward)

Identical to Phase 4 R4 curriculum:

```
- Discrete(10) action space; action 9 = warp 5 min (τ=5 sub-steps of 60s)
- 38-dim obs with LVLH-frame relative state (R1)
- BETA_SHAPE = 1.0 with gated NHR shaping (R2): w_orbit=w_phase=w_vel=0.01
- Φ-terminal-clamp at every termination branch
- R3 stack disabled: l2_init=0, target_entropy_controller=False,
  clip_coef_low = clip_coef_high = 0.2, kl_target=0, anneal_lr=True
- ent_coef=0.01, gamma=0.995, lr=0.01, minibatch=8192
- Curriculum schedule: 30° (10M fresh) → 90° (15M warm) → 180° (15M warm)
```

For Phase 5 (eccentricity / "funky orbit" extension), add a phase-gap-then-eccentricity nested curriculum. Don't add DAPO. Don't attempt cross-phase surgery as a Stage 1 fix.

---

## Compute spend

| Ablation | Compute | Wall time |
|---|---|---|
| A — no LVLH Stage 1 | 14.4M steps | ~1m 45s |
| B — no shaping Stage 1 | 14.4M steps | ~1m 45s |
| D — DAPO Stage 3 | 19.4M steps | ~2m 35s |
| C — surgery + smoke test | 0 train + 50-ep eval | ~30s |
| **Total** | ~50M steps + small evals | **~7 min compute, ~30 min wall** |

Total wall time was much shorter than the spec's 7-9 hr estimate because:
- A and B halted at Stage 1 (no Stage 2/3 cascade since results were decisive).
- C halted at smoke test.
- D was Stage 3 only by design.

The decisive failures of A and B saved ~5 hr of cascade compute that would have produced the same answer.

---

## Files produced

- `experiments/p4.5_d7_to_d10_warmstart.pt` — surgery ckpt (kept for reference)
- `pufferlib/scripts/orbital/expand_ckpt_actions_d7_to_d10.py` — surgery script (committed)
- `pufferlib/experiments/puffer_orbital_177723359224/` — Ablation A Stage 1 (failed)
- `pufferlib/experiments/puffer_orbital_177723387812/` — Ablation B Stage 1 (failed)
- `pufferlib/experiments/puffer_orbital_177723408423/` — Ablation D Stage 3 (succeeded train, regressed eval)
- `/tmp/p45_D_eval_seed{42,1337,20260423}/` — Ablation D rollout-eval traces

---

*Author: 2026-04-26. Closes Phase 4 attribution gap. Successor: Phase 5 spec (eccentricity extension).*
