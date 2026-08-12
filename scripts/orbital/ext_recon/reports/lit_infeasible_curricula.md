## Literature recon — training under partially infeasible task distributions (for the corrected wide-3D ladder)

Scope note: read-only, nothing written to the repo. Findings keyed to the post-fix decision set (sampler defect found; V-ladder retraining). Where the literature does not directly cover our exact measurement I say so and mark the reasoning as inference.

---

## 1. What a 34–57% infeasible mass does to PPO

### 1a. Infeasible episodes are not "noise" — they are silently *optimum-shifting*

The single most useful primitive is PLR's **Value Correction Hypothesis** ([Jiang et al. 2021, PLR](https://arxiv.org/abs/2010.03934)): levels *at* the agent's ability threshold have non-stationary returns and high value error; **"levels beyond the agent's current abilities tend to result in stationary value targets signaling failure and therefore low value errors."**

Implication for us, and it cuts against the "advantage noise" hypothesis: once the critic has seen enough infeasible draws, it *correctly* predicts their failure, so their GAE advantages go to ≈0 and they contribute almost no gradient. They do not inject variance. They do three other things:

1. **Sample tax.** At 34–57% infeasible, 1/3–1/2 of every rollout batch is gradient-dead. Effective batch shrinks by that factor; gradient SNR falls ~√2 at 50%. (Inference from the above, not a measured result in any paper.)
2. **They change which policy is optimal on the *pooled* distribution** (see §1c) — the mechanism that actually produces a 7% number.
3. **They occupy the network's capacity with a "predict failure" function.** [Lyle et al., *Understanding and Preventing Capacity Loss in RL*](https://arxiv.org/abs/2204.09560): networks trained on a sequence of target values lose the ability to update predictions, **"particularly damaging to performance in sparse-reward tasks"** — our exact regime.

**Quantitative anchor for "how much infeasible mass is survivable":** the only benchmark that deliberately does this is **TeachMyAgent** ([Romac et al. 2021](https://arxiv.org/abs/2103.09815)), whose *Mostly unfeasible task space* condition grows stump height until **~80% of tasks are unfeasible**. Results there:
- ALP-GMM / Covar-GMM / RIAC: significantly beat Random (p<0.05)
- **ADR: no better than Random. GoalGAN: no better than Random. SPDL: poor.**
- Companion condition, *Rugged difficulty landscape* (feasible tasks **scattered among** unfeasible ones): **ADR significantly *worse* than Random (p<0.05)**, because "ADR can get stuck by subspaces of very hard (or unfeasible) difficulty."

So: 80% infeasible is survivable *only* by learning-progress-driven samplers, and naive boundary-expansion (ADR) is actively harmful when feasibility is non-monotone in the knobs. Our post-fix e/di/da space is very plausibly rugged rather than monotone (feasibility is a joint Δv+time constraint, not axis-aligned) — this is the strongest caveat on recommendation R4 below.

There is **no paper giving a threshold like "X% infeasible breaks PPO."** The nearest thing to a decision rule in the literature is CURROT's per-context constraint (§1b), which is "0% infeasible mass by construction, at every step."

### 1b. Regret-based curricula: the mechanism, and how an analytic oracle replaces it for free

- **PAIRED / minimax regret** ([Dennis et al. 2020](https://arxiv.org/abs/2012.02096)): regret = V\*(θ) − V^π(θ). An unsolvable level has V\* = V^π ⇒ **regret 0** ⇒ the teacher is *disincentivized* from generating it. This is the entire mechanism; it is not a heuristic.
- **Robust PLR / PLR^⊥** ([Jiang et al. 2021, *Replay-Guided Adversarial Environment Design*](https://arxiv.org/abs/2110.02439)) is the directly transferable trick: **gradient updates are taken ONLY on curated/replayed levels; newly-sampled random levels are rolled out with a stop-gradient.** "Training on less data improves robustness." Numbers: CarRacing-F1 zero-shot return **534±7 (PLR^⊥) vs 408±12 (PLR) vs 341±22 (DR) vs 19±15 (PAIRED)**; MiniGrid solved rate 0.6±0.1 vs 0.4±0.0 (DR) vs 0.2±0.0 (PAIRED).
- Practical regret estimators: **positive value loss** and **MaxMC** = (1/T)Σ R_max − V(s₀). The paper explicitly notes L1 value loss **"does not generally correspond to regret"** — relevant because L1 value loss is inflated by aleatoric/irreducible variance, which is what an infeasible-but-not-yet-learned draw looks like early in training.

**Cheap approximation given our analytic oracle:** we do not need an antagonist agent or a value-loss proxy. The closed-form Δv+time screen *is* an upper bound on V\*(θ). Concretely:
- `screen(θ) = infeasible` ⇒ regret ≡ 0 ⇒ **exclude from the gradient** (PLR^⊥ pattern) or from the sampler (CURROT-oracle pattern).
- `regret_proxy(θ) ≈ V*_analytic(θ) − R_observed(θ)` gives a *free, unbiased, policy-independent* prioritization score — strictly better than positive-value-loss, because it is not contaminated by the critic's own error. This is the one place where our domain beats the literature's tooling.

### 1c. Does the literature confirm "collapse spreads to feasible neighbors"? — Yes, under three names

The measured result (W4's trained child scored **worse than its untrained parent on the parent's own distribution**) is a documented, named phenomenon. It is not exotic and does not require a novel explanation:

- **Negative transfer in continual RL.** [Reset & Distill (2403.05066)](https://arxiv.org/pdf/2403.05066) and [*Prevalence of Negative Transfer in Continual RL* (ICLR 2025)](https://proceedings.iclr.cc/paper_files/paper/2025/file/ba9e3d60610f3525717665966d86e0cd-Paper-Conference.pdf): a transferred model can perform **worse than learning the target from scratch**; their prescribed remedy is to **reset the value function** — "learning with a re-initialized value function rather than fine-tuning prevents negative effects from the old value function during transfer, especially to less similar transfer tasks." That is a direct hit on the "value-function poisoning" hypothesis: the literature's own fix targets the critic, not the actor.
- **Curriculum-Induced Covariate Shift (CICS)** ([Jiang, Dennis et al., NeurIPS 2022, *Grounding Aleatoric Uncertainty for UED*](https://arxiv.org/abs/2207.05219)): formalizes that shifting the *training* distribution over environment parameters yields policies that are **suboptimal on the ground-truth deployment distribution** — "value estimates and policy decisions optimized for intermediate curriculum stages don't transfer... even if individual tasks appear solvable during training." W4's distribution *is* a shifted training distribution relative to W3's; CICS is exactly the claim that performance on W3's distribution degrades.
- **Capacity/plasticity loss:** [Lyle et al. 2022](https://arxiv.org/abs/2204.09560); [Abbas et al. 2023, *Loss of Plasticity in Continual Deep RL*](https://arxiv.org/abs/2303.07507) (activation footprint becomes sparser → diminishing gradients); [Nikishin et al. 2022, primacy bias](https://proceedings.mlr.press/v162/nikishin22a/nikishin22a.pdf).

**What the literature does *not* have** is a paper measuring "pessimism generalizes across the context vector to feasible neighbors" as a distinct mechanism. The mechanism is consistent with everything above (a context-conditioned critic is a smooth function of θ; driving V(s₀)→floor on half the θ-space necessarily depresses it on the interpolating feasible θ), but treat it as *mechanism-consistent-with*, not *confirmed-by*. The measured child-worse-than-parent result is well covered by negative transfer + CICS without needing it.

### 1d. `cap_terminal_reward = 0` — the literature strongly *supports* this being a real, compounding factor

This is the cleanest result in the sweep. The governing rule ([Voelcker, *Reward Design and Termination*](https://cvoelcker.de/blog/2025/reward-functions/), which is a clear synthesis of the folklore):

> **"When using negative rewards, termination is good [for the agent]. When using positive rewards, it is bad... If you use both negative and positive rewards AND early termination, all bets are off."**

Our design is precisely the "all bets are off" case: success **+R_s**, physical failure **−R_f**, cap/timeout **0**. Since 0 > −R_f, **timeout strictly dominates every failure mode.** The indifference threshold is exact and derivable (my derivation, matching the design rules in the sources): attempting beats stalling only when

```
p_success  >  R_f / (R_s + R_f)          [cap valued at 0]
```

For symmetric ±10 this is **p > 0.50**. With 34–57% of draws infeasible, the *maximum achievable* pooled success is 43–66% — i.e. the pooled distribution sat **straddling the indifference point**, and for any policy not already near-perfect on the feasible subset, **run-out-the-clock was genuinely reward-optimal.** The agent was not failing to learn; it learned the correct policy for a mis-specified pooled MDP. This is a real second-order effect, not an artifact — and it explains why the collapse was to a *coherent* degenerate behavior rather than to noise.

Two further supporting cites:
- **[Pardo et al., ICML 2018, *Time Limits in RL*](https://arxiv.org/abs/1712.00378)**: if the time limit is *not* part of the task, you must **bootstrap from V(s_T) at the cap** (partial-episode bootstrapping) — treating the cap as a true terminal with V=0 biases the value function ("the agent forgets that more rewards would actually be available thereafter"). If the limit *is* part of the task, **remaining time must be in the observation** or you violate Markov via state aliasing. W4 changed the cap 3000→6000 with (presumably) neither treatment — the same state now has two different correct values across rungs.
- **[Intentionally-Underestimated Value at Terminal State (2308.12772)](https://arxiv.org/pdf/2308.12772)**: "considering the value after termination to be zero unintentionally overestimates the reached endpoints," forcing the agent to seek those endpoints even when they are task failures.

**Actionable:** the cap must be valued at ≤ the worst physical failure (make it a genuine failure terminal, e.g. −R_f), **or** bootstrapped at the cap, **or** time-remaining must enter the observation. Pick one and never change the cap value in the same rung step as a distribution widening.

---

## 2. The `feasible_init_only` pattern: precedents, and honest reporting

### Precedents (this is a well-established pattern, not a hack)

| Precedent | Filter | Reported how |
|---|---|---|
| **[Reverse Curriculum Generation, Florensa et al. 2017](https://arxiv.org/abs/1707.05300)** | Sample start states; estimate success from each; **discard any start whose success is outside [R_min, R_max]** (they used ~0.1/0.9) so it is not trained on next iteration | **"learning progress still evaluated based on the original start-state distribution"** — the canonical honest-reporting move |
| **[GoalGAN / GOID, Florensa et al. 2018](https://arxiv.org/abs/1705.06366)** | Train only on **G**oals **o**f **I**ntermediate **D**ifficulty: R_min ≤ success ≤ R_max (TeachMyAgent's reimpl. used **0.25 / 0.75**) | Reports coverage over the full goal space; note only **~20%** of generated goals were GOID during training |
| **[CURROT, Klink et al. 2022](https://proceedings.mlr.press/v162/klink22a.html) / [OT-benefit, 2309.14091](https://arxiv.org/abs/2309.14091)** | Per-context constraint: contexts with nonzero mass must have expected return ≥ δ, plus a Wasserstein step bound ε | In the sparse-goal-reaching env with walls making many targets infeasible, **an "oracle which trains only on the feasible subspace" is used as the upper baseline and is the only thing that beats CURROT.** ⇒ *An oracle-filtered training distribution is the gold standard when you have an oracle. We have one.* |
| **[Robust PLR (PLR^⊥)](https://arxiv.org/abs/2110.02439)** | Sample the full distribution, **take gradients only on curated levels** | Evaluate zero-shot on the full/held-out distribution |
| **Sokoban/Boxoban PCG** ([Procedural Generation of Initial States of Sokoban, IJCAI 2019](https://www.ijcai.org/proceedings/2019/0646.pdf)) | Levels **guaranteed solvable by construction/solver**; solvability classifiers used to filter generated levels before RL | Standard practice; solvability is assumed, difficulty is the reported axis |

**Filter the gradient, not the sampler (preferred).** The PLR^⊥ variant is strictly better bookkeeping for us: keep sampling the true wide distribution so your rollout statistics measure the feasibility fraction φ *for free* every iteration, and mask infeasible episodes out of the loss. You get the honest denominator without a second eval pass, and no sampler bias in logging. Cost: you pay the rollout compute on dead episodes (~1/3–1/2). If throughput matters more than free measurement, reject at reset and measure φ in a separate periodic eval.

### Honest reporting — report a triple, never a single number

1. **φ = feasibility fraction** of the declared envelope under the oracle at margin *m* (a property of the *task spec*, reported with the spec).
2. **Success | feasible** (the policy's capability).
3. **Unconditional success on the unfiltered envelope** — **measured, not computed as (1)×(2)**. Multiplying assumes the oracle is perfectly calibrated; measuring catches oracle error in both directions.

The repo's own precedent demands this: `project_orbital_rl_phase5_verification.md` records the e=0.5/0.7 headlines being confounded exactly by a pass-only vs headline discrepancy (97% vs 72%). Same failure, avoided by the same discipline.

### Warnings

- **Eval-time distribution shift = CICS.** [Jiang et al. 2022](https://arxiv.org/abs/2207.05219) is the citation: the curriculum distribution "provides misleading performance signals that don't predict actual deployment success"; their recommendation is unambiguous — **always evaluate on the ground-truth deployment distribution.** SAMPLR's fix (importance weighting / fictitious transitions) is overkill here; just evaluate unfiltered.
- **Conservative oracle ⇒ blind spot.** A two-impulse/Lambert Δv lower bound will screen out draws solvable only by multi-rev or three-burn solutions. Detect it: periodically run the policy on screened-*out* draws and count solves. Non-zero solves = the oracle is costing you capability, and the screen (not the policy) is the thing to fix.
- **Optimistic oracle ⇒ infeasible mass leaks back**, reinstating §1c/§1d. Add a Δv margin *m* and report it.
- **Deployment behavior on infeasible draws is now untrained.** If the real mission spec includes draws outside the envelope, the correct behavior is *detect-and-abort*, and a `feasible_init_only`-trained policy has never seen one. Either train the abort behavior explicitly or state the limitation. (Given the Draper framing memo — "lead with limitations + verification story" — this is a *feature* to state, not a hole to hide.)
- **Margin scheduling.** Start the rung at `required_Δv ≤ (1−m)·budget`, m ≈ 0.15–0.25, then anneal m → 0 as the rung matures. Note this makes *m itself the cleanest single knob for ADR-style expansion* (§3) — it is one-dimensional, monotone in difficulty, and directly interpretable.

---

## 3. ADR-lite: success-gated knob expansion as a replacement for the hand ladder

### The mechanism, verified from the source

[Solving Rubik's Cube with a Robot Hand (OpenAI, 1910.07113)](https://arxiv.org/abs/1910.07113), Algorithm 1 — confirmed from the paper text:

- Maintain a per-dimension interval [φ_L^i, φ_H^i].
- With probability p_b, **randomly select exactly ONE dimension i and clamp it to one boundary** (φ_L^i or φ_H^i); all other parameters are sampled normally.
- Append that episode's performance to that boundary's buffer.
- When the buffer reaches length *m*: average ≥ t_H ⇒ **expand** that boundary by Δ; ≤ t_L ⇒ **contract** by Δ. Clear buffer.
- Progress metric: **ADR entropy** `H(P_φ) = (1/d) Σ_i log(φ_H^i − φ_L^i)` nats/dimension — one scalar that replaces the entire hand ladder as the thing you watch.

**This is the literature's own answer to "how many knobs may move at once": exactly one, and only at its boundary.** TeachMyAgent's reimplementation hyperparameters, as a starting point: **t_L=0, t_H=180 (env-specific return scale), p_b=0.7, m=10, Δ=0.1.**

### Critical caveat before adopting it

TeachMyAgent measured **ADR ≈ Random on the mostly-unfeasible space and significantly *worse* than Random on the rugged landscape** ("ADR can get stuck by subspaces of very hard (or unfeasible) difficulty"). ADR has no notion of feasibility — it only knows "the policy fails here," which is indistinguishable from "this is impossible." **ADR-lite is therefore only safe in our setting when composed with the analytic screen**, which supplies exactly the expert knowledge ADR lacks. Screen-gated ADR ≈ CURROT's structure (per-context feasibility constraint + bounded step), which *was* the top method on TeachMyAgent's mostly-unfeasible benchmark.

### Implementation sketch (given every knob is already an env kwarg)

1. **Knob-state object in the Python wrapper on the master process**: `{name: [lo, hi, lo_buf, hi_buf]}` plus hard caps `φ_target` = the declared wide envelope (guarantees termination). Do not put this in C; it needs to be shared across vec workers, and it updates once per ~m episodes, so the Python-side cost is nil.
2. **At reset**: with prob `p_b` pick one dim + side, clamp; sample the rest from current intervals; **then apply the analytic screen and resample the whole draw if infeasible** (or tag it for gradient masking). Tag the episode with `(dim, side)`.
3. **At episode end**: push success∈{0,1} into `buf[dim][side]`.
4. **Every |buf| == m**: expand/contract by Δ, clear.
5. **Log `H(P_φ)` per iteration** as the curriculum progress metric, alongside φ (feasibility fraction) and success|feasible. Three scalars; the ladder becomes an emergent trace instead of a spec.
6. **Abort rule** (already specced per the coordinator): if `H(P_φ)` fails to increase over K windows *and* success|feasible < gate, stop — this is the ADR-gets-stuck signature that TeachMyAgent measured.

Cost estimate: a few dozen lines in the Python wrapper + one reset-time hook. No C changes. The main non-trivial piece is that PufferLib vec workers need the current intervals at reset — simplest is to broadcast the interval dict on each `reset()` batch rather than trying to share mutable state.

**Recommended first target for ADR-lite:** not the five task knobs at once — start with **the feasibility margin m alone** (one dimension, monotone, guaranteed-safe), then add `di_max` and `e_max` as dimensions 2 and 3. Keep `da`, altitude band, and episode cap on the hand ladder until the ADR loop has demonstrated a monotone `H(P_φ)` trace.

---

## 4. Compound rung steps and warm-start vs fresh (condensed — still ladder-relevant)

- **How many axes per step: one.** ADR (one dimension, at boundary). **ACCEL** ([Parker-Holder et al. 2022](https://arxiv.org/abs/2203.01302)) makes "a handful of changes" to already-curated levels under the explicit assumption that **"regret varies smoothly with the environment parameters, such that the regret of a level is close to the regret of others within a small edit distance"**, and contrasts this with random sampling, which "risks discovering fundamentally disconnected challenges." **Rudin et al.** ([Learning to Walk in Minutes](https://proceedings.mlr.press/v164/rudin22a.html)) promote/demote by **one terrain level** per episode based on distance walked, with graded knobs (step 5→20 cm, slope 0→25°).
- **Jumps are a named failure.** CURROT's OT paper diagnoses KL-based interpolation directly: KL "displaces density from contexts to contexts with large Euclidean distance... the observed ignorance of KL divergence w.r.t. the underlying geometry **leads to curricula with 'jumps' in task similarity**." W3→W4 was a five-dimensional jump of exactly this type. The fix in the literature is a **metric step-size bound** (CURROT's ε on W₂ between consecutive context distributions) — the practical analogue is *anneal the knob linearly over ≥N iterations rather than stepping it*.
- **Also from CURROT, subtle and relevant:** *expected*-performance constraints (SPDL-style, and implicitly any "mean success ≥ X" gate) can be satisfied by mixing very easy and very hard tasks while **ignoring intermediate difficulty**. Gate on the **per-context / per-quantile** success, not just the mean, or a 70% rung mean can hide a bimodal "solves the easy half, never the hard half" policy — which is what a 70% W3 probably was.
- **Warm-start vs fresh, under distribution widening with fixed reward:** warm-start the **actor**, re-initialize or shrink-and-perturb the **critic**. Sources: Reset & Distill (re-initialized value function prevents the old value function from harming transfer to dissimilar tasks); [Ash & Adams 2020](https://papers.neurips.cc/paper_files/paper/2020/file/288cd2567953f06e460a33951f55daaf-Paper.pdf) (shrink-and-perturb closed the warm-start generalization gap and *improved on from-scratch*); [DASH, NeurIPS 2024](https://arxiv.org/html/2410.23495v2) (warm-starting in stationary settings without plasticity loss); Nikishin (periodic reset of last layers). This is consistent with the repo's own R3/R5 findings that plasticity is a prerequisite for any post-warm-start change.
- **Rehearsal.** Keep a fixed fraction (10–20%) of parent-distribution draws in every child rung's sampler. This is the cheapest available guard against the exact regression measured (child worse than parent on parent's distribution) and costs one kwarg.

---

## 5. Karpathy lens — what W3→W4 violated, plainly

([A Recipe for Training Neural Networks](https://karpathy.github.io/2019/04/25/recipe/))

| Principle | Violation |
|---|---|
| **"Become one with the data"** (step 1, before any model code) | The realized `e`-vector mismatch ran **4.5–8× the `de` knob** and nobody looked. A histogram of *realized* Δe and *required* Δv per draw — 20 lines, zero GPU — would have caught the inert knob before the first W4 run. This is the single largest process finding. |
| **"No untested code" / "visualize just before the net"** | `de_max` was untested code with a silent no-op. The assertion must be on the **env's realized output**, not on the kwarg being plumbed. Plumbing tests pass on inert knobs (repo precedent: `project_orbital_rl_phase5_verification.md` — "plumbing correct, but…"). |
| **"Get dumb baselines" / input-independent baseline** | An analytic two-impulse open-loop and a coast-only null, scored **on each rung's own draws**, separate "task infeasible" from "learning broken" on day one. The repo already knows this works — Phase 5.5's coast-only null demolished the MEO/GEO "capability" claim. |
| **"Plug signals in one by one"** | W4 moved **five** knobs. Direct, unambiguous violation. |
| **"Overfit first, then regularize"** | The ladder's healthy pattern was 100% per rung. Accepting a **70%** parent (W3) and warm-starting from it built on a degraded base — and per CURROT's critique, a 70% mean may itself be a bimodal artifact rather than uniform 70% competence. |
| **"Fix random seed" / independent verification** | Multi-seed is mandatory here by the repo's own history: Phase 4 had seeds 42 and 20260423 succeed while seed 1337 gave **0%**. Any single-seed W-rung number is uninterpretable. |

---

## 6. Ranked recommendations for the corrected wide-3D ladder

Each tagged with source + the failure mode it prevents.

**R1 — `feasible_init_only` via the analytic screen at margin m, filtering the *gradient* not the sampler. Report the triple (φ, success|feasible, unconditional).**
*Source:* Robust PLR stop-gradient-on-uncurated ([2110.02439](https://arxiv.org/abs/2110.02439), 534±7 vs 408±12); CURROT's feasible-subspace oracle as the top baseline ([2309.14091](https://arxiv.org/abs/2309.14091)); Florensa 2017 (train filtered, **evaluate on the original distribution**).
*Prevents:* sample tax, degenerate-optimum shift (§1d), and the missing-denominator reporting failure the repo already committed once in Phase 5.
*Cost:* one boolean mask in the loss + one screen call at reset. Highest value-per-line item in this list.

**R2 — Fix the cap-terminal semantics before the next rung, and never move the cap in the same step as a widening.**
Choose one: (a) cap reward = −R_f (make it a real failure), (b) bootstrap V(s_T) at the cap (partial-episode bootstrapping), or (c) put normalized time-remaining in the observation. Verify by the derived threshold: with cap=0, stalling dominates whenever `p_success < R_f/(R_s+R_f)` — 50% for symmetric ±10.
*Source:* [Voelcker](https://cvoelcker.de/blog/2025/reward-functions/) ("all bets are off"); [Pardo et al. ICML 2018](https://arxiv.org/abs/1712.00378); [2308.12772](https://arxiv.org/pdf/2308.12772).
*Prevents:* run-out-the-clock becoming genuinely reward-optimal; state aliasing across rungs with different caps.

**R3 — Never step more than ONE knob per rung; if a knob must move far, anneal it linearly over ≥N iterations rather than stepping.**
*Source:* ADR (exactly one dimension, boundary-clamped); ACCEL ("regret varies smoothly... within a small edit distance"); CURROT's Wasserstein step bound ε and its diagnosis of KL-induced "jumps"; Rudin et al. one-level promotion.
*Prevents:* the W3→W4 class of collapse; makes any future failure attributable to a single cause (Karpathy).

**R4 — Gate rung entry at parent success ≥ 95–97%, measured *per-quantile* on the feasible subset, not on the pooled mean.**
*Source:* CURROT's finding that expected-performance constraints are satisfiable by mixing very-easy + very-hard while skipping intermediate difficulty; GOID's per-goal bounds (0.25/0.75); the ladder's own empirical record (100% rungs held; the 70% rung broke).
*Prevents:* warm-starting from a bimodal parent that looks 70%-competent but is 0% on the region the child needs.

**R5 — On any rung that has already been trained on poisoned data, re-initialize or shrink-and-perturb the critic while warm-starting the actor; carry 10–20% parent-distribution rehearsal in every child rung.**
*Source:* Reset & Distill (re-init value function prevents negative transfer); Ash & Adams shrink-and-perturb; DASH; Nikishin resets; Abbas/Lyle plasticity-and-capacity loss.
*Prevents:* the measured child-worse-than-untrained-parent regression; value-function poisoning persisting across rungs.

**R6 — ADR-lite, screen-gated, starting with the feasibility margin m as its only dimension; log `H(P_φ)` as the single curriculum-progress scalar.**
Params to start: p_b=0.7, m_buf=10, Δ=0.1, t_H = your rung gate, t_L ≈ 0.5·t_H; hard caps at the declared envelope.
*Source:* ADR Algorithm 1 ([1910.07113](https://arxiv.org/abs/1910.07113)); TeachMyAgent's reimplementation hyperparameters.
**Hard caveat (do not skip):** TeachMyAgent measured plain ADR at **no better than Random on an 80%-unfeasible space and significantly worse than Random on a rugged one**. Adopt ADR *only* composed with R1's screen; otherwise it will walk the boundary straight into the infeasible frontier and stall there. Expand to `di_max`/`e_max` as dimensions 2–3 only after `H(P_φ)` shows a monotone trace on m alone.

**R7 — Add the analytic screen as a *regret proxy*, not just a filter: score draws by `V*_analytic(θ) − R_observed(θ)` and oversample the top quantile.**
*Source:* PAIRED's regret objective (unsolvable ⇒ regret 0); PLR's Value Correction Hypothesis; the explicit note that L1 value loss "does not generally correspond to regret."
*Prevents:* wasting the feasible budget on trivially-easy draws once the screen removes the impossible ones. This is a strict upgrade over the literature's value-loss proxies because our V\* is exact and policy-independent. Lowest priority only because R1–R5 must land first.

**R8 — Standing instrumentation (pre-flight, mandatory, per the corrected ladder spec):** per-rung histograms of *realized* (not requested) Δe/Δi/Δa and required Δv; φ from the screen; analytic-open-loop and coast-only null scores on the rung's own draws; ≥3 seeds before any rung is declared solved.
*Source:* Karpathy steps 1–2; the repo's own Phase 4 seed-1337 0% and Phase 5.5 coast-only-null result.
*Prevents:* inert knobs, pointy-max single-seed headlines, and reward-leak "capability."

---

### Two things the literature does **not** support and I'd advise against

- **Reward reshaping to "compensate" for infeasible mass** (e.g., scaling terminal rewards by feasibility). No precedent; and the repo's R5 memo already measured all four reshape variants collapsing to <10% on a committed warm-start. Fix the distribution, not the reward.
- **Relying on learning-progress samplers (ALP-GMM) *instead of* the oracle.** They beat Random at 80% infeasible, but the same benchmark shows an oracle-filtered feasible subspace is the upper baseline. When you have a closed-form oracle, using a learned progress estimator to rediscover it is strictly worse and adds a hyperparameter surface.

---

**Sources:**
[Prioritized Level Replay (Jiang et al. 2021)](https://arxiv.org/abs/2010.03934) · [Replay-Guided Adversarial Environment Design / Robust PLR (Jiang et al. 2021)](https://arxiv.org/abs/2110.02439) · [Emergent Complexity and Zero-shot Transfer via UED / PAIRED (Dennis et al. 2020)](https://arxiv.org/abs/2012.02096) · [Grounding Aleatoric Uncertainty for UED / CICS + SAMPLR (Jiang et al. 2022)](https://arxiv.org/abs/2207.05219) · [Evolving Curricula with Regret-Based Environment Design / ACCEL (Parker-Holder et al. 2022)](https://arxiv.org/abs/2203.01302) · [TeachMyAgent (Romac et al. 2021)](https://arxiv.org/abs/2103.09815) · [Curriculum RL via Constrained Optimal Transport / CURROT (Klink et al. 2022)](https://proceedings.mlr.press/v162/klink22a.html) · [On the Benefit of Optimal Transport for Curriculum RL (2309.14091)](https://arxiv.org/abs/2309.14091) · [Self-Paced Deep RL (Klink et al. 2020)](https://arxiv.org/pdf/2004.11812) · [Teacher-Student Curriculum Learning (Matiisen et al. 2017)](https://arxiv.org/abs/1707.00183) · [Automatic Curriculum Learning for Deep RL: A Short Survey (Portelas et al. 2020)](https://www.ijcai.org/Proceedings/2020/0671.pdf) · [Solving Rubik's Cube with a Robot Hand / ADR (OpenAI 2019)](https://arxiv.org/abs/1910.07113) · [Reverse Curriculum Generation (Florensa et al. 2017)](https://arxiv.org/abs/1707.05300) · [Automatic Goal Generation / GoalGAN (Florensa et al. 2018)](https://arxiv.org/abs/1705.06366) · [Learning to Walk in Minutes (Rudin et al. 2021)](https://proceedings.mlr.press/v164/rudin22a.html) · [Time Limits in Reinforcement Learning (Pardo et al. 2018)](https://arxiv.org/abs/1712.00378) · [Reward Design and Termination (Voelcker 2025)](https://cvoelcker.de/blog/2025/reward-functions/) · [Intentionally-Underestimated Value Function at Terminal State (2308.12772)](https://arxiv.org/pdf/2308.12772) · [Revisiting Constant Negative Rewards for Goal-Reaching Tasks (2407.00324)](https://arxiv.org/abs/2407.00324) · [Understanding and Preventing Capacity Loss in RL (Lyle et al. 2022)](https://arxiv.org/abs/2204.09560) · [Loss of Plasticity in Continual Deep RL (Abbas et al. 2023)](https://arxiv.org/abs/2303.07507) · [Continual Backprop (Dohare et al.)](https://arxiv.org/abs/2108.06325) · [On Warm-Starting Neural Network Training (Ash & Adams 2020)](https://papers.neurips.cc/paper_files/paper/2020/file/288cd2567953f06e460a33951f55daaf-Paper.pdf) · [DASH (NeurIPS 2024)](https://arxiv.org/html/2410.23495v2) · [The Primacy Bias in Deep RL (Nikishin et al. 2022)](https://proceedings.mlr.press/v162/nikishin22a/nikishin22a.pdf) · [Reset & Distill (2403.05066)](https://arxiv.org/pdf/2403.05066) · [Prevalence of Negative Transfer in Continual RL (ICLR 2025)](https://proceedings.iclr.cc/paper_files/paper/2025/file/ba9e3d60610f3525717665966d86e0cd-Paper-Conference.pdf) · [Procedural Generation of Initial States of Sokoban (IJCAI 2019)](https://www.ijcai.org/proceedings/2019/0646.pdf) · [Dead-ends and Secure Exploration (Fatemi et al. 2019)](https://proceedings.mlr.press/v97/fatemi19a.html) · [A Recipe for Training Neural Networks (Karpathy 2019)](https://karpathy.github.io/2019/04/25/recipe/)