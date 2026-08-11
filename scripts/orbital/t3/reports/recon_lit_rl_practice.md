## T3 Recon — Literature Distillation for This Failure Signature

Report combines (i) fetched-source distillation and (ii) **new measurements taken this session against the real C env under corrected dynamics**, because three of the four failure modes turned out to be quantifiable rather than speculative.

---

## 0. New ground-truth measured this session (use these numbers, not the hypotheses)

**Scripts:** `/Users/pete/space_training/scripts/orbital/t3/probe_shaping_leak.py`, `/Users/pete/space_training/scripts/orbital/t3/feasibility_and_horizon.py`
**CSVs:** `/Users/pete/space_training/web_data/results/t3_shaping_leak_q1.csv`, `t3_shaping_leak_q2.csv`, `t3_phasing_feasibility.csv`, `t3_credit_horizon.csv`

**F-1. The shaping pays a stall income of ≈ +1.78 reward per episode for doing nothing.**
Do-nothing policies hit the safety cap, which by design skips the Φ-clamp, so `episode_return + 10 == total accumulated shaping`. Measured over 120 headline-condition episodes each:

| scripted policy | τ | decisions/ep | mean return | implied total shaping |
|---|---|---|---|---|
| coast (a=0) | 1 | 2000 | −8.221 | **+1.779** |
| warp-5min (a=9) | 5 | 400 | −8.243 | **+1.757** |
| warp-1hr (a=11) | 60 | 34 | −8.475 | **+1.525** |

Algebra: Σₜ β(γ^τΦ′−Φ) telescopes to (Φ_T−Φ₀) + (1−γ)Σₜ|Φₜ| — the second term is a pure **living reward proportional to |Φ|, i.e. proportional to |Δa|**. It is ≈ invariant to τ because it accrues per unit *sim time*, not per decision. Magnitude checks out: 0.005 × 2000 × mean|Φ| = 1.78 ⟹ mean|Φ| ≈ 0.178 ⟹ mean|Δa| ≈ 178 km ≈ the expected |difference| of two U(300,800) km altitudes. **The agent is paid to stay far from the target orbit and burn the clock.**

**F-2. The shaping is net-negative on the drift orbit — the only phase-change mechanism that exists under correct physics.** Paired seeds, 33-decision (= full-episode) budget, cumulative reward:

| policy | mean cumulative reward | vs. stalling |
|---|---|---|
| stall (warp-1hr only) | **+1.775** | — |
| drift up, Δa≈+175 km | +1.546 | **−0.23** |
| drift up, Δa≈+350 km | +2.054 | +0.28 |
| drift down, Δa≈−175 km | **−2.853** | **−4.63** (3/8 seeds terminate early, reentry) |

Two pathologies: (a) lowering the orbit — the direction that *catches up* in phase — is catastrophically punished and physically kills the episode from a 300 km floor; (b) among raise-drifts, the shaping gradient points toward **larger** Δa (more income), i.e. away from the Δa→0 closure the terminal requires. This is a textbook reward valley/cycle.

**F-3. The σ₂ gate is fully closed for the entire phasing leg.** `σ₂ = sigmoid((2 − φ_orbit)/0.2)`, `φ_orbit = |Δa|/10 km + ‖Δē‖`. Gate half-open at Δa = 20 km, effectively dead (σ₂ < 0.05) beyond Δa ≈ 26 km. The measured drift orbits require Δa = 110–350 km ⟹ **σ₂ ≈ e⁻⁴⁵ ≈ 0. Zero phase gradient for the ~20 hours that phasing takes.** Worse, the weights are inverted against task difficulty: W_PHASE·φ_phase ≤ 0.02 while W_ORBIT·φ_orbit ≈ 0.11–0.35 at drift altitudes — the hard subtask carries 1/5th to 1/17th the potential mass of the easy one, *and* it is gated off.

**F-4. `MAX_STEPS = 2000` counts 60 s SUB-steps, not agent decisions** (`env->step++` is inside the warp loop in `c_step`). Wall-clock budget is therefore **33.3 h regardless of warping**. Warp buys decisions, not time. Phasing obeys an exact invariant (Δa cancels): `Δv_total · t_drift = Δθ · v / (1.5 n)`.

| Δθ | min Δv (whole 33.3 h) | min time (whole 478 m/s tank) | Δa at min-Δv |
|---|---|---|---|
| 90° | 60.4 m/s | 4.21 h | 55 km |
| 180° | **120.8 m/s** | **8.42 h** | 110 km |

Worked 180° point: spend 200 m/s ⟹ Δa = 182 km, drift 20.1 h (**60% of the entire episode**), leaving 278 m/s and 13.2 h for the altitude transfer and terminal closure. **The task is feasible but the optimal policy requires a single ~20-hour uninterrupted commitment.**

**F-5. The learner's credit window covers ~10 decisions.** `gamma=0.995`, `gae_lambda=0.90` (PufferLib default) ⟹ effective GAE n-step window `1/(1−γλ) = 9.6 decisions`.

| temporal unit | decisions/ep | γ^H | GAE window as % of episode |
|---|---|---|---|
| coast/burn (τ=1) | 2000 | 4.4e−05 | 0.48% |
| warp-5min (τ=5) | 400 | 0.135 | 2.4% |
| warp-30min (τ=30) | 67 | 0.716 | 14.4% |
| warp-1hr (τ=60) | 33 | **0.846** | **28.7%** |

**Legacy 10-logit checkpoints have only warp-5min ⟹ γ^H = 0.135 and a 2.4% credit window. They are structurally incapable of seeing the terminal reward across a 20-hour commitment.** That is a sufficient explanation for failure mode (2) on its own.

**F-6. Semi-MDP discount mismatch, confirmed in code.** The env shapes with `γ^τ` (`orbital.h` c_step: `pow(0.995, env->last_tau)`), but the learner's advantage function is `compute_puff_advantage(values, rewards, terminals, ratio, advantages, gamma, gae_lambda, ...)` (`pufferlib/pufferl.py:716`) — **no τ argument; one decision = one γ**. So the shaping term is *not* of the form γΦ′−Φ in the MDP the optimizer actually solves. **Ng–Harada–Russell policy invariance does not hold here.** Combined with the deliberate omission of the Φ-clamp at the safety-cap terminal (violating Φ(absorbing)=0), the shaping is formally non-potential-based on two independent counts — which is exactly why F-1/F-2 are possible.

*(Non-finding, checked so it isn't mis-flagged later: `learning_rate = 0.01` is normal — PufferLib's default optimizer is `muon` at lr 0.015. Do not "fix" this.)*

---

## (a) Ranked top-10 recommendations

**1. Make warp-1hr the temporal unit and fix the semi-MDP discount. [37 Details #1/#5; SMDP options; F-4/F-5/F-6]** → *fixes modes 1, 2, 4.*
Either (i) pass τ to the advantage computation so bootstrapping uses `γ^τ` and the reward is the τ-step accumulation (true SMDP option value backup, `V(s)=E[R_τ + γ^τ V(s′)]`), or (ii) if you won't touch the trainer, **prune the action set so every action has the same τ** — e.g. a "burn-then-warp-1hr" macro-action set where every decision advances 60 sub-steps. Option (ii) is cheaper, removes the mismatch by construction, and lands you at γ^H = 0.85 with a credit window covering 29% of the episode. Do not train the legacy 10-action head at all. Sources: [iclr-blog-track.github.io/2022/03/25/ppo-implementation-details](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/), [arxiv.org/html/2306.13284](https://arxiv.org/html/2306.13284) (discount-factor mismatch in on-policy PG), [arxiv.org/pdf/2203.09365](https://arxiv.org/pdf/2203.09365) (semi-Markov RL, γ^τ backup).

**2. Replace Φ with an estimate of −V\*: Δv-to-go from your own Lambert/phasing solver. [Ng–Harada–Russell 1999] ** → *fixes mode 3, helps 1 and 4.*
Ng et al.'s own conclusion is that **the best potential is Φ = V\***, and the shaping speedup is proportional to how well Φ approximates it. Your reward is essentially `+10·(0.5+0.5·fuel_left)` on arrival, so `V*(s) ≈ 10 − c·Δv_to_go(s)`. You already have `T1_LAMBERT_BASELINE.md` and `scripts/orbital/nav/orbital_math.py`. Set `Φ(s) = −k·Δv_lambert_to_go(s)` (or the phasing-invariant `Δv = K/t_remaining` when the transfer is phase-dominated). This potential *decreases monotonically along the correct drift-orbit strategy*, which the current one provably does not (F-2). Source: [people.eecs.berkeley.edu/~pabbeel/cs287-fa09/readings/NgHaradaRussell-shaping-ICML1999.pdf](https://people.eecs.berkeley.edu/~pabbeel/cs287-fa09/readings/NgHaradaRussell-shaping-ICML1999.pdf).

**3. Kill the (1−γ)Σ|Φ| stall income. [Ng 1999 + F-1]** → *fixes mode 3.*
Use `γ_shape = 1` in the shaping term (`r_shape = β(Φ′ − Φ)`) so it telescopes exactly to `Φ_T − Φ₀` and pays *nothing* for stalling. This is the standard fix when the shaping discount cannot be made to match the learner's, and it is already flagged as a known issue in your own code comment at `orbital.h:822`. Alternatively bound Φ. Either way the measured +1.78/episode do-nothing subsidy — 18–35% of a success payout — must go to zero. **Verify by re-running `probe_shaping_leak.py`: the coast/warp-only implied shaping must come out ≈ 0.**

**4. Restore Φ(absorbing) = 0 at *every* terminal, including the safety cap. [Ng 1999]** → *fixes mode 3.*
The cap exemption was a patch for the Φ-clamp leak; once #3 removes the leak's cause, the exemption becomes the remaining PBRS violation. Non-uniform terminal handling means timeout and success are shaped inconsistently, which is precisely the "cycling" class of failure Ng et al. constructed potential-based shaping to rule out (the motivating example being Randløv & Alstrøm's bicycle agent that rode in circles to farm a progress bonus).

**5. Reverse curriculum over *start states*, not over reward. [Florensa et al. 2017]** → *fixes modes 1, 2.*
You have full reset control (`phase_gap_fixed`, `a_min/a_max_override`, `e_*_fixed`, `same_orbit_init`) — Florensa's Assumption 1 is satisfied for free, which is rare. Implement: seed from a near-goal state, generate nearby starts by short random-action rollouts (their `SampleNearby`, T_B = 50 steps, Gaussian action noise), keep **"good starts" with success rate in (R_min, R_max) = (0.1, 0.9)**, sample N_new = 200 fresh + N_old = 100 replayed old starts per iteration to prevent forgetting. Critically, **Florensa explicitly report that distance-based shaping guides the policy into a suboptimal local optimum for far-from-goal states** — the same pathology as F-2. This is the single highest-value structural change and it *replaces* rather than patches shaping. Source: [arxiv.org/abs/1707.05300](https://arxiv.org/abs/1707.05300) / [ar5iv.labs.arxiv.org/html/1707.05300](https://ar5iv.labs.arxiv.org/html/1707.05300).

**6. Jump-Start (roll-in from the scripted Lambert controller), curriculum on the handover step h. [Uchendu et al. 2022]** → *fixes modes 1, 2.*
Run the guide policy π_g (Lambert/two-impulse controller) for the first h decisions of every episode, then hand to the learner π_e for the remaining H−h; decrease h on performance milestones. Requires only that π_g beat random. JSRL's theory: a guide policy that merely *covers the states the optimal policy visits* converts exponential-in-horizon sample complexity to polynomial. **JSRL substantially outperformed BC-init-then-finetune in the low-data regime** (e.g. 72.6% vs 33.1% on antmaze-umaze-diverse at 1k demos; 91% vs ≈0% grasping with 20 demos). For you, π_g rolling in through the ~20 h drift leg drops the learner into the endgame — exactly the part the current shaping can actually help with (σ₂ open, Δa < 26 km). Source: [ar5iv.labs.arxiv.org/html/2204.02372](https://ar5iv.labs.arxiv.org/html/2204.02372).

**7. Karpathy's dumb-baseline battery *before* the next training run. [Karpathy 2019]** → *fixes mode 1 (and prevents the next silent failure).*
This project has already lost a full campaign to a silent physics bug; the recipe is written for exactly that. Concretely: fix the seed; run an **input-independent baseline** (zero the observations — training must get *worse*; if it doesn't, the policy is riding the shaping, not the state); **overfit one batch** (single fixed init via `phase_gap_fixed`/`e_*_fixed`/`a_*_override` — if PPO cannot reach ~100% on ONE deterministic scenario, nothing about the harder distribution is worth running); **verify loss at init** (entropy should be ln 16 = 2.77 for a Discrete(16) head, ln 10 = 2.30 for legacy); log the reward **decomposition** (terminal vs shaping vs leak) per episode, not just the return; visualize prediction dynamics (Φ-trace and Δa-trace over a fixed eval batch each checkpoint). "Complexify only one at a time" — the current env has ~25 interacting kwargs and a 16-action space accreted across 5 phases. Source: [karpathy.github.io/2019/04/25/recipe](https://karpathy.github.io/2019/04/25/recipe/).

**8. Raise `gae_lambda` and treat γ as a first-class tuned hyperparameter. [Andrychowicz et al. 2020; 37 Details #5]** → *fixes mode 4.*
λ = 0.90 gives a 9.6-decision credit window (F-5). Andrychowicz et al. call **the discount factor "one of the most important hyperparameters" and say it should be tuned per environment**; with an SMDP-corrected γ^τ (#1) the effective per-hour discount changes meaning entirely, so γ must be re-swept afterward, not before. Sweep λ ∈ {0.90, 0.95, 0.99} jointly with the temporal unit. Their other directly applicable defaults: observation normalization is "crucial"; separate policy/value networks; tanh over ReLU; multiple epochs over the data are "crucial for sample complexity" (PufferLib default `update_epochs = 1`); initialize the policy's last layer ~100× smaller. Sources: [ar5iv.labs.arxiv.org/html/2006.05990](https://ar5iv.labs.arxiv.org/html/2006.05990), [iclr-blog-track.github.io/2022/03/25/ppo-implementation-details](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/).

**9. ≥5 seeds on every claim; report the distribution, never the max. [Irpan 2018; Henderson et al. 2018]** → *governs modes 1, 2.*
Irpan: 10 seeds on *Pendulum* with identical hyperparameters gave a **30% outright failure rate**; VIME's 25th-percentile curve sits near zero reward from initialization alone; Henderson et al. found **five seeds insufficient for significance** and that multiplying rewards by a constant changes results materially. Your own history already contains this failure (memory: Stage 3's "64%" was a seed-42 pointy-max against a 42–64% true distribution; R4 seed 1337 flatlined at 0% where 42 and 20260423 succeeded). Any "gentle re-adapt plateaus at 33.5%" claim is a single-seed claim until proven otherwise. Source: [alexirpan.com/2018/02/14/rl-hard.html](https://www.alexirpan.com/2018/02/14/rl-hard.html).

**10. Shrink the task until something learns, then re-expand one axis at a time. [Irpan 2018; Karpathy 2019]** → *fixes mode 1.*
Irpan's list of deep-RL successes all share **aggressive problem simplification** (OpenAI's DOTA bot: 1v1, Shadow Fiend mirror, hardcoded items; the SSBM bot: one character, one stage). Your corrected-dynamics ladder should start at: fixed a_sat = a_target, e = 0 both, Δθ = 30°, warp-1hr-only action set, Φ from #2. F-4 says 30° needs only 20 m/s and 1.4 h — that scenario is *comfortably* inside both budgets and should be near-100% or the pipeline is broken. Only then relax, one axis per step: Δθ → 90 → 180; then e; then Δa band. Do not re-introduce debris (standing user instruction).

*Also evaluated, not in the top 10:* **HER** ([arxiv.org/abs/1707.01495](https://arxiv.org/abs/1707.01495)) is conceptually ideal here — the env is goal-conditioned (target elements + LVLH already in obs), and HER's headline claim is that sparse binary reward + goal relabeling beats shaped reward on exactly this class of task. It is deprioritized only because it requires an off-policy replay learner and PufferLib's stack is on-policy PPO; that is a rewrite, not a knob. Revisit if #1–#6 fail.

---

## (b) Anti-checklist — what the sources warn against that you are doing now

| # | Current practice | Warned against by | Why it bites here |
|---|---|---|---|
| A1 | Shaping term uses γ^τ while the learner discounts γ¹ per decision | Ng 1999 (form requirement); SMDP option backup | The shaping is **not** potential-based in the optimized MDP ⟹ **no policy-invariance guarantee at all**. Everything below follows from this. |
| A2 | Safety-cap terminal deliberately skips Φ(absorbing)=0 | Ng 1999 (explicit terminal condition) | Timeout and success are shaped inconsistently; the fix for one leak created a second violation. |
| A3 | Non-telescoping (1−γ)Σ\|Φ\| term pays **+1.78/episode to a frozen policy**, scaling with distance-from-goal | Ng 1999; Irpan ("RL overfits to your reward"; boat-race powerup farming) | Measured. This is the boat race: a positive-income loop that never completes the task. |
| A4 | σ₂ gate closes the phase potential for all \|Δa\| > 26 km, i.e. for 100% of the phasing leg | Ng 1999 (Φ should approximate V\*) | The potential is *silent* during the hardest, longest subtask and *actively negative* on entering it (F-2). Gated/staged potentials are only safe if the gate opens along the optimal trajectory; yours opens only at the very end. |
| A5 | Distance/orbit-error shaping used as the primary exploration mechanism for a far-from-goal problem | **Florensa et al. explicitly**: for distant states "the distance to the goal is actually not a useful metric… the distance reward actually guides the policy updates towards a suboptimal local optimum" | Under correct dynamics you must *increase* Δa to make progress. Any monotone-in-Δa potential opposes the solution. |
| A6 | Legacy 10-action checkpoints (warp-5min only) used as the warm-start base | 37 Details (γ, GAE); F-5 | γ^H = 0.135, credit window 2.4% of episode. Re-adapting this head is re-adapting a policy that cannot represent the solution's timescale. The 33.5% plateau is expected, not surprising. |
| A7 | Warm-start + reward-reshape as the default recovery move | Rajeswaran (BC init is destroyed by the first PG updates); your own R5 memory (all 4 reshape-on-warm-start variants collapsed to <10%) | Already empirically refuted twice in this project. Do not attempt a third time without the plasticity/curriculum prerequisites. |
| A8 | Single-seed headline numbers; "best checkpoint" reporting | Irpan; Henderson et al. | 30% seed-failure rate on *Pendulum*. Your own Stage-3 and R4-seed-1337 history. |
| A9 | 16-action space, ~25 env kwargs, 5 phases of accreted shaping, all live simultaneously | Karpathy ("complexify only one at a time"; "generalize a special case") | Unverified complexity that cannot be debugged. The dynamics bug survived 5 phases inside this. |
| A10 | Treating `MAX_STEPS=2000` as a generous "safety cap" | F-4 | It is a **hard 33.3 h feasibility constraint** consuming 60% of the episode for a single 180° phasing leg at a sane Δv. It is part of the problem definition, not a guard rail. |
| A11 | Assuming more compute (2× budget) fixes a flatline | Irpan (sample inefficiency ≠ the bug; "when your training algorithm is both sample inefficient and unstable it slows your rate of productive research") | 2× a budget that was only ever sufficient *because of the bug subsidy* proves nothing. Fix the objective, then re-budget. |

---

## (c) BC-from-scripted-expert: what the literature actually prescribes

You have (or can trivially produce) a classical controller — Lambert two-impulse plus a phasing solver. The literature is unusually consistent here, and the consistent message is: **do not do BC-init → PPO.**

**C-1. BC-init alone is destroyed by the first policy-gradient updates.**
Rajeswaran et al. (DAPG): BC gives you `max_θ Σ_{(s,a)∈ρ_D} ln π_θ(a|s)`, but "behavior cloning does not guarantee that the cloned policy will be effective, due to the distributional shift between the demonstrated states and the policy's own states," and — the key point — "different parts of the demonstration data are useful in different stages of learning." A BC-initialized policy that then takes on-policy PPO steps under a *sparse* reward has no gradient holding it near the demonstrated manifold, so the first few updates (driven by advantage noise and the entropy bonus) walk it off. This is the same mechanism your own R5/R3 memory records: reshaping a committed warm-start collapses to <10%.

**C-2. Instead: keep a demo term in the loss throughout, with a decaying weight (DAPG).**

```
g_aug = Σ_{(s,a)∈ρ_π} ∇ ln π(a|s) A^π(s,a)  +  Σ_{(s,a)∈ρ_D} ∇ ln π(a|s) · w(s,a)
w(s,a) = λ₀ · λ₁^k · max_{(s',a')∈ρ_π} A^π(s',a')
```
with **λ₀ = 0.1, λ₁ = 0.95** (k = iteration counter). The `max A^π` factor auto-scales the demo term to the current advantage magnitude, so it degrades gracefully as the policy improves. Reported gains: **~30× on relocation (5.77 h vs 98 h), ~8× on hammer, ~3.5× on door.** [ar5iv.labs.arxiv.org/html/1709.10087](https://ar5iv.labs.arxiv.org/html/1709.10087)

**C-3. Dataset size: small is fine — 25 to 100 trajectories.**
DAPG used **25 demos per task** and reports no sensitivity ablation (they note performance "is not very sensitive to the choice of" hyperparameters, with demo count fixed at 25). Nair et al. used **100** VR demos. JSRL got 91% grasping from **20** demonstrations. **Do not build a 10k-episode dataset.** For you: generate 50–200 Lambert/phasing rollouts stratified over (Δθ, Δa, e) — a few minutes of compute — and spend the saved effort on demo *coverage of the state distribution the optimal policy visits* (JSRL's coverage assumption), which matters far more than count. Include the endgame closure states, not just the burn points, or the demo term teaches only the two impulses and nothing about the 20-hour drift.

**C-4. Guard against being *capped* by a suboptimal scripted expert — use a Q-filter or an annealed weight.**
Nair et al.'s central finding: an unfiltered BC auxiliary loss "prevents the learned policy from improving significantly beyond the demonstration policy"; their **Q-filter** applies the BC loss only on demo samples where the critic judges the demo action better than the current policy's. DAPG's λ₁^k decay and Kickstarting's annealed λ_k are the on-policy analogues. Concretely for PPO: gate the demo log-likelihood term on `A^π(s, a_demo) > 0` using your own value head — a one-line PPO analogue of the Q-filter. Your Lambert controller is a genuinely good but not optimal expert (median learned Δv was 2.49× Hohmann pre-fix; Lambert two-impulse will beat that but is not fuel-optimal under phasing constraints), so this guard is load-bearing. [ar5iv.labs.arxiv.org/html/1709.10089](https://ar5iv.labs.arxiv.org/html/1709.10089)

**C-5. Kickstarting is the drop-in variant if you'd rather distill a *policy* than a dataset.**
`ℓ_kick = ℓ_RL + λ_k · H(π_T(a|x_t) ‖ π_S(a|x_t))` — an auxiliary cross-entropy/KL to the teacher added to the RL loss, with λ_k annealed to ≈0 over training (they anneal via population-based training rather than a hand schedule). Reported **6.92× speedup to score 30**, and the student **surpasses the teacher by +43.4%** — i.e. the annealing genuinely releases the ceiling. Explicitly contrasted with pretrain-then-finetune: kickstarting is "a continuous version… joint optimisation rather than sequential stages," giving a dense signal throughout instead of needing an arbitrary stopping criterion. If you wrap the Lambert controller as a callable π_T over the same Discrete-N head, this needs no demo dataset at all. [ar5iv.labs.arxiv.org/html/1803.03835](https://ar5iv.labs.arxiv.org/html/1803.03835)

**C-6. Recommended concrete recipe for this project (combining C-2/C-4/C-6 with #5/#6 above):**
1. Wrap Lambert + phasing solver as π_g emitting actions in the *pruned, uniform-τ* action set (rec. #1). Verify π_g's own success rate under the corrected env first — **if the scripted expert cannot clear ~90% at headline conditions, stop; the problem or the success box is wrong, not the RL.**
2. Collect 100 stratified demo trajectories; BC to convergence; record BC-only eval success (this is your `Φ ≈ V*` sanity number too).
3. Train PPO with the DAPG demo term (λ₀ = 0.1, λ₁ = 0.95) *and* JSRL roll-in (h decreasing on milestones), under the repaired potential (#2/#3/#4).
4. Ablate in this order and report each with ≥5 seeds: no-demo-term; no-roll-in; old-Φ; τ-mismatched. That ablation table is also the strongest possible artifact for the Draper narrative — "here is the exploration subsidy the bug was providing, and here is what legitimately replaces it."

**C-7. Expected honest ceiling.** Irpan's success criteria — unbounded cheap simulation ✔, aggressive problem simplification ✔ (if you follow #10), self-play ✘, **learnable and ungameable reward ✘ (currently gameable — F-1/F-2), rich reward signal ✘ (currently zero during the longest leg — F-3)**. Two of five are broken and both are fixable by #2/#3/#4. Fix them before spending another training budget.

---

## Sources

- Karpathy, *A Recipe for Training Neural Networks* — https://karpathy.github.io/2019/04/25/recipe/
- Irpan, *Deep Reinforcement Learning Doesn't Work Yet* — https://www.alexirpan.com/2018/02/14/rl-hard.html
- Huang et al., *The 37 Implementation Details of PPO* — https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/
- Ng, Harada & Russell 1999, *Policy Invariance Under Reward Transformations* — https://people.eecs.berkeley.edu/~pabbeel/cs287-fa09/readings/NgHaradaRussell-shaping-ICML1999.pdf
- Florensa et al. 2017, *Reverse Curriculum Generation for RL* — https://arxiv.org/abs/1707.05300 · https://ar5iv.labs.arxiv.org/html/1707.05300
- Rajeswaran et al. 2018, *DAPG / Learning Complex Dexterous Manipulation with Demonstrations* — https://arxiv.org/abs/1709.10087 · https://ar5iv.labs.arxiv.org/html/1709.10087
- Nair et al. 2018, *Overcoming Exploration in RL with Demonstrations* — https://ar5iv.labs.arxiv.org/html/1709.10089
- Uchendu et al. 2022, *Jump-Start Reinforcement Learning* — https://ar5iv.labs.arxiv.org/html/2204.02372
- Schmitt et al. 2018, *Kickstarting Deep RL* — https://arxiv.org/abs/1803.03835 · https://ar5iv.labs.arxiv.org/html/1803.03835
- Andrychowicz et al. 2020, *What Matters In On-Policy RL?* — https://ar5iv.labs.arxiv.org/html/2006.05990
- Andrychowicz et al. 2017, *Hindsight Experience Replay* — https://arxiv.org/abs/1707.01495 *(cited from prior knowledge; ar5iv fetch returned no body text this session — verify before quoting specifics)*
- SMDP / variable-duration discounting — https://arxiv.org/pdf/2203.09365 (semi-Markov offline RL, γ^τ backup) · https://arxiv.org/html/2306.13284 (discount-factor mismatch in on-policy PG) · https://arxiv.org/pdf/1711.07832 (situationally aware options)

## Artifacts written (nothing committed; no files modified outside the two permitted dirs)

- `/Users/pete/space_training/scripts/orbital/t3/probe_shaping_leak.py`
- `/Users/pete/space_training/scripts/orbital/t3/feasibility_and_horizon.py`
- `/Users/pete/space_training/web_data/results/t3_shaping_leak_q1.csv`
- `/Users/pete/space_training/web_data/results/t3_shaping_leak_q2.csv`
- `/Users/pete/space_training/web_data/results/t3_phasing_feasibility.csv`
- `/Users/pete/space_training/web_data/results/t3_credit_horizon.csv`