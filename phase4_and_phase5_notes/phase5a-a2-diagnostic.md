# Phase 5a — A2 Peak Checkpoint Diagnostic

> **Status:** 2026-04-28. ~10 minutes of work. Distinguishes between three candidate mechanisms (PPO instability, catastrophic forgetting, small-sample mirage) for the warm-start collapse pattern observed in probes A2 and E1. Output gates the Phase 5 main spec design.

---

## 0. What this answers

Probe A2 reached 88.9% peak at e=0.05 then collapsed to 0%. Probe E1 reached 100% peak at e=0.025 then collapsed to 0.2%. The interim findings doc proposes "tight checkpointing + best-ckpt selection" (Mitigation 1) as the operational fix.

Mitigation 1 is correct *as infrastructure* regardless. The diagnostic question is whether **additional** mitigations (KL-regularization, BC-kickstart, etc.) are needed, and the answer depends on which mechanism is producing the collapse.

Three candidate mechanisms (call them stories):

- **Story A — PPO instability.** Value head mis-calibrates for new reward landscape; entropy collapses; policy converges to deterministic-bad mode. Same R5 pattern from Phase 4.5. Implication: best-ckpt selection works alone.
- **Story B — Catastrophic forgetting.** Peak ckpt uses the e=0 strategy that coincidentally still works at e=0.05. PPO continues to update toward new gradient landscape, overwriting e=0 capability without converging on a stable e>0 policy. Implication: need anti-forgetting machinery (BC-kickstart against e=0 anchor) in addition to best-ckpt selection.
- **Story C — Small-sample mirage.** The 88.9% / 100% peaks were lucky eval batches during training. The peak ckpt's true held-out performance is much lower; "collapse" is just plateau plus noise. Implication: the recipe doesn't actually generalize to e>0; we need a different intervention entirely.

Stories A and B both predict that the peak ckpt *does* genuinely solve the e>0 task; they differ on whether it preserves earlier capability. Story C predicts the peak ckpt does not solve it.

---

## 1. The diagnostic protocol

Two evaluations on the A2 peak checkpoint, both deterministic (eval-time only, no training).

### 1.1 Locate the peak checkpoint

A2 was the (π/6, 0.05) warm-start from A1, 10M steps. From the interim findings:

```
pufferlib/experiments/puffer_orbital_177738633600/
```

That directory has all the saved checkpoints. The peak ckpt is the one corresponding to the 88.9% peak — *not* the final ckpt (which is 0%).

**Identifying the peak ckpt.** If wandb logs were saved with the run, the per-checkpoint train-time eval performance is recorded. Look at the wandb run for this experiment ID, find the checkpoint_id with highest eval success rate, pull that ckpt. If wandb isn't available, eval each saved checkpoint in the directory at e=0.05 and pick the best.

If checkpoint_interval was 200 epochs as in the interim findings doc, this checkpoint may not exist — the peak occurred between saved checkpoints. In that case, the diagnostic cannot be run as designed; see §3 (Failure mode).

### 1.2 Eval at training condition (e=0.05)

```bash
python3 scripts/orbital/eval_checkpoint.py \
  pufferlib/experiments/puffer_orbital_177738633600/model_puffer_orbital_<peak_epoch>.pt \
  --episodes 200 \
  --init-phase-gap-max 0.524 \
  --e-max-target 0.05 \
  --rollout-seeds 42,1337,20260423
```

Output: per-rollout-seed success rate at e=0.05.

**Statistical setup.** 200 episodes × 3 rollout seeds = 600 episodes. Per-seed std of ~5pp expected at this episode count. Mean across rollouts gives our point estimate.

### 1.3 Eval at Phase 4 capability condition (e=0)

Same protocol, e_max_target = 0:

```bash
python3 scripts/orbital/eval_checkpoint.py \
  pufferlib/experiments/puffer_orbital_177738633600/model_puffer_orbital_<peak_epoch>.pt \
  --episodes 200 \
  --init-phase-gap-max 0.524 \
  --e-max-target 0.0 \
  --rollout-seeds 42,1337,20260423
```

Output: per-rollout-seed success rate at e=0.

### 1.4 (Optional but recommended) Same evals on A1 ckpt

For comparison reference, run both evals on A1's final ckpt (the e=0 baseline that A2 warm-started from). This anchors what "Phase 4 capability" looks like at the same eval protocol.

```bash
python3 scripts/orbital/eval_checkpoint.py \
  pufferlib/experiments/puffer_orbital_177738621627/model_puffer_orbital_000077.pt \
  --episodes 200 \
  --init-phase-gap-max 0.524 \
  --e-max-target 0.0 \
  --rollout-seeds 42,1337,20260423

python3 scripts/orbital/eval_checkpoint.py \
  pufferlib/experiments/puffer_orbital_177738621627/model_puffer_orbital_000077.pt \
  --episodes 200 \
  --init-phase-gap-max 0.524 \
  --e-max-target 0.05 \
  --rollout-seeds 42,1337,20260423
```

A1 was the e=0 baseline that hit 80% at training time. The eval at e=0 should reproduce that. The eval at e=0.05 tells us whether the *unfine-tuned* Phase 4 policy already has some e>0 capability — useful baseline.

---

## 2. Decision matrix

Four cells of interest (A2 peak at e=0.05, A2 peak at e=0, A1 at e=0.05, A1 at e=0). The story diagnosis comes from comparing them.

### 2.1 Outcome → mechanism mapping

| A2 peak @ e=0.05 | A2 peak @ e=0 | A1 @ e=0.05 | Story | Phase 5 implication |
|---|---|---|---|---|
| ≥75% | ≥70% | irrelevant | **A — PPO instability** | Tight checkpointing alone is sufficient. Phase 5 main = Phase 4 recipe + tight checkpoint + eccentricity ramp. |
| ≥75% | <40% | irrelevant | **B — Catastrophic forgetting** | Tight checkpointing + BC-kickstart from A1 anchor. Phase 5 main needs anti-forgetting machinery. |
| <40% | irrelevant | ≥75% | **C-strong — Mirage** | The 88.9% peak was eval noise. The Phase 4 policy already handles e=0.05; A2 was discovering nothing new. e>0 generalization is from the inherited recipe, not from training. |
| <40% | irrelevant | <40% | **C-weak — Recipe doesn't transfer** | Phase 4 recipe doesn't generalize to e>0 even at small e. Need a substantively different approach (different reward shaping for eccentric targets, different action space, etc). |
| 40-75% | irrelevant | irrelevant | **Hybrid / unclear** | Run additional probes; see §2.3. |

### 2.2 The four stories in plain language

**A — PPO instability.** "The agent learned to handle eccentric targets, but PPO's continued training destroyed the policy after the peak. The peak ckpt is a real, capable agent."

**B — Catastrophic forgetting.** "The peak ckpt is using the Phase 4 strategy, which happens to also solve e=0.05. Continued training is reshaping the policy toward a new objective without converging on it; we're seeing a transient that came from the inherited capability, not learned capability."

**C-strong (mirage).** "The peak number was a small-sample fluke during training-time eval. The Phase 4 policy actually already handles e=0.05 just fine without any additional training. We've been chasing a non-finding."

**C-weak (recipe doesn't transfer).** "The Phase 4 recipe, even with the inherited capability, doesn't extend to e>0. We need a different approach to make e>0 work at all."

### 2.3 Hybrid outcomes

If the peak ckpt evals between 40-75% at e=0.05, the picture is unclear and one or two follow-up probes are warranted:

- **Eval at e=0.025.** If A2 peak hits 75%+ at e=0.025 but only 60% at e=0.05, the recipe is partially generalizing — it loses capability gradually with eccentricity. This is more like Story A with a soft ceiling. Mitigation: tight checkpointing + slower eccentricity ramp (smaller increments per stage).

- **Eval E1 peak ckpt at e=0.025.** E1 hit 100% peak; if it sustains at eval, that's confirmation that the recipe handles small e cleanly and the question is just where the soft ceiling sits.

These probes are also ~5 minutes each; cheap to run if the primary diagnostic comes back ambiguous.

---

## 3. Failure mode: peak ckpt not saved

If checkpoint_interval was 200 epochs and the peak occurred between saved checkpoints, we may not have the peak ckpt to evaluate. Two responses:

**Response 1 (preferred).** Implement tight checkpointing in the training loop (the §6.2 / Mitigation 1 infrastructure) and re-run probe A2 with checkpoint_interval = 10 epochs. ~6 minutes of training compute. Then run the diagnostic on the new peak ckpt.

**Response 2 (fallback).** Eval all saved A2 checkpoints (including the final 0% one) and find the highest-performing among them. May not match the train-time peak, but bounds the question.

Response 1 is preferred because it builds the infrastructure we need anyway and gives a clean diagnostic. Response 2 is a fallback if compute is unavailable.

---

## 4. What this does NOT test

- **Does the peak ckpt generalize beyond e=0.05?** No. We're testing the specific peak from the A2 probe. Eccentricity ceiling is a separate question for Phase 5 main.
- **Is tight checkpointing the right operational fix?** Implicitly yes (it's needed regardless), but the diagnostic is not testing whether tight checkpointing makes Phase 5 main work — it's testing whether *additional* mitigations are needed beyond it.
- **Is the apoapsis-bias question resolved?** No. Investigation D remains deferred until we have a checkpoint that solves at e>0 with reasonable success rate. The A2 peak ckpt (if it survives this diagnostic) becomes the natural candidate for the apoapsis-bias check. Worth running it as a fifth eval if time allows: log target true anomaly at successful rendezvous moments and check distribution.

---

## 5. Total compute and time

- Locate peak ckpt: <5 min if wandb available, ~3 min if needs eval scan
- 4 evals × ~1 min each at 600 episodes per eval: ~5 min
- Optional A1 baseline evals: +2 min
- (If failure mode triggered) tight-checkpoint A2 redo: ~6 min training + ~5 min eval

Total: 10-25 minutes wall, depending on whether the peak ckpt is saved.

---

## 6. Output

A short addition to PHASE5a_INTERIM_FINDINGS.md as §X "A2 Peak Diagnostic":

- Four eval numbers (A2 peak @ {e=0, e=0.05}, A1 @ {e=0, e=0.05})
- Story verdict (A / B / C-strong / C-weak / hybrid)
- One sentence on Phase 5 main implication

Then proceed to draft Phase 5 main spec with the verdict baked in.

---

## 7. What's at stake

If Story A: Phase 5 main is straightforward. Tight checkpointing + eccentricity ramp + multi-seed protocol. ~2-3 weeks of work to land the capability surface.

If Story B: Phase 5 main is harder. Need to implement BC-kickstart (KL-regularization against the canonical Phase 4 endpoint), test it on A2-equivalent conditions, then run the eccentricity ramp. ~3-4 weeks.

If Story C-strong: The "Phase 4 generalizes to small e without retraining" finding is itself the Phase 5 result for small e. Phase 5 main becomes "characterize the Phase 4 recipe's eccentricity ceiling without further training, and ramp from there." Different scope, different writeup, possibly a stronger negative-result-then-positive-finding story.

If Story C-weak: Phase 5 main needs structural rework. The Phase 4 recipe doesn't transfer. We'd be revisiting reward shaping or action space for eccentric targets specifically. Several weeks of investigation before the eccentricity ramp can run.

The diagnostic is 10-25 minutes and the answer changes Phase 5 main scope by a factor of 2-4×. Highest information-per-compute experiment in the project so far.

---

*Author: 2026-04-28. Pre-Phase-5-main diagnostic. Successor: Phase 5 main spec with verdict-conditional design.*
