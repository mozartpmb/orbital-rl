# Autonomous Orbital Rendezvous — Project Review

**Peter M. Brown · August 2026**

A consolidated view of what the project is, what's genuinely strong about it, what a skeptical reviewer will attack, and what to build next. Written from your descriptions and your agent's technical answers, not from reading the code, so treat the critiques as things to verify rather than confirmed defects.

---

## 1. What it is

A 2D coplanar restricted two-body orbital rendezvous environment and a learned guidance policy.

| Component | Implementation |
|---|---|
| Dynamics | Analytic Kepler propagation; Newton-Raphson on Kepler's equation. No J2, drag, third-body, SRP, or attitude. |
| Maneuvering | Impulsive burns with Tsiolkovsky mass depletion. Isp 300 s, ~480 m/s Δv from a 15% propellant fraction. |
| Task | Rendezvous with an independently propagated moving target from arbitrary initial orbits. |
| Actions | Discrete(16): coast, prograde/retrograde 1–25 m/s, radial ±10 m/s, time-warp 5 min / 30 min / 1 hr. Deliverable checkpoint trained on Discrete(10). |
| Observations | 38-dim: 7 chaser elements, 2 target, 16 nearest-body proximity (unused), 5 LVLH relative, 4 ω sin/cos, 4 phase. |
| Policy | PPO + LSTM via PufferLib Ocean. |
| Termination | success / collision / escape / stranded / safety_cap (2000 steps). |
| Reward | Gated potential-based shaping (orbit / phase / velocity components) with terminal Φ-clamp, ±10 terminal, γ=0.995. |
| Performance | ~500k env steps/sec, 1,024 parallel instances. 96–98% multi-seed at 300–800 km, e ≤ 0.05. |
| Status | Private fork of PufferLib. Extending toward MEO/GEO via curriculum learning. |

---

## 2. What's genuinely strong

Listing these first because the critique section is long, and the critique is long *because* the project is substantial enough to be worth criticizing at this level.

**Writing the environment in C as a Python extension.** Most RL practitioners write environments in Python, eat a 100× performance penalty, and never question it. Building a vectorized C env with a Python binding is a real systems-engineering decision with a real payoff, and it's the single most credible signal on the project that you're an engineer rather than someone running notebooks.

**Choosing analytic Kepler propagation over numerical integration.** This is the correct call and it isn't the obvious one. Naively you'd reach for an RK4 integrator on the two-body ODE. Solving Kepler's equation instead gives you exact propagation, arbitrary step sizes for free (which is what makes your time-warp actions possible at all), and no energy drift over long episodes. That choice tells a reviewer you understood the structure of the problem before you started coding.

**LVLH relative state in the observation.** The right frame, and the one actual RPO work uses. Feeding raw inertial states and hoping the network learns the transform would have been the lazy path.

**Tsiolkovsky mass depletion.** Fuel is a real constraint that couples to the dynamics rather than a decorative counter. This makes the problem meaningfully harder and more honest than most RL orbital work, where propellant is either infinite or a flat per-burn penalty.

**A moving, independently propagated target.** Substantially harder than targeting a fixed orbit, and it's what makes the phasing problem interesting.

**Potential-based reward shaping with gating.** You clearly hit the reward-design wall and solved it the theoretically correct way. PBRS is policy-invariant under the right construction, and gating lower components until higher ones converge is a sensible way to impose a natural task ordering. Most people hack together a weighted sum of distances and wonder why the agent learns something insane.

**Multi-seed evaluation and a real termination taxonomy.** Reporting across seeds instead of a cherry-picked run is better discipline than a lot of published RL. Separating collision, escape, stranded, and cap means you can actually attribute failures rather than just counting them.

---

## 3. Issues, ranked by how fast they hurt you

### 🔴 3.1 The terminal criterion invalidates the headline number

Covered before but it belongs here as the top item. At 500 km altitude, v ≈ 7.617 km/s, and a 50 m/s relative velocity implies

```
Δa ≈ 2a·Δv/v = 2(6871)(0.050)/7.617 ≈ 90 km
```

Two vehicles 30 km apart with 90 km of semimajor-axis mismatch are on different orbits and are separating. You've built a conjunction detector, not a rendezvous.

**The important refinement: velocity is the disqualifying term, not distance.** 30 km separation is defensible on its own — that's a legitimate far-field rendezvous handover point, roughly where you'd transition to proximity operations. But at 30 km with relative velocity under ~1 m/s, the two vehicles are effectively co-orbiting and the result is real. **Tighten velocity hard, distance moderately.** Something like 5 km / 1 m/s is arguably a defensible far-field rendezvous; 1 km / 0.5 m/s is unambiguous.

There's a second-order concern worth checking: your shaping potential includes a velocity-match component and your success criterion is loose on velocity. A policy optimized against a loose criterion will satisfy the letter of it. Some of your 96–98% may be trajectories that graze the success box rather than actually close. Tightening the criterion and retraining will tell you which.

### 🔴 3.2 No baseline, and no articulated reason for RL to exist

The coplanar two-impulse rendezvous problem has a **closed-form solution**. A Lambert targeter solves it essentially optimally and near-deterministically. The first question any astrodynamicist asks is not "how good is your policy" but "why is there a policy?"

Without a baseline, the default assumption is that Lambert beats you on Δv and reliability, and your project reads as a worse solution to a solved problem.

You need two things:

**The number.** Compute two-impulse Lambert Δv across your test distribution and report the ratio. If the policy is within ~2× of optimal, that's an honest and respectable result — RL trading optimality for robustness. If it's 5×, you've learned something important about your reward or action set.

**The argument.** Have a real answer for why RL. The defensible ones, roughly in order of strength:

- Your action set is **discrete and quantized** (1–25 m/s), so the continuous Lambert solution isn't directly executable — this is a combinatorial sequencing problem, not a two-point boundary value problem.
- The **fuel budget makes some Lambert solutions infeasible**, and the policy has to find feasible-within-budget trajectories rather than optimal-unconstrained ones.
- The policy is **reactive at runtime** — no onboard targeting solve, no re-planning loop.
- Most importantly: **it's a testbed for the version where Lambert breaks.** Add measurement noise, partial observability, perturbations, keep-out constraints, and debris, and the closed-form solution stops applying while the learned policy framework carries over unchanged.

That last one is the honest answer and it's a good one. It also explains why your next steps are the right next steps.

### 🟠 3.3 Perfect state — there is no navigation in the navigation project

Full-state feedback with no sensor model means you've built guidance and targeting. For the Space and Strategic Navigation req specifically, this is the gap. It's item one in future work.

### 🟠 3.4 Seed variance is being under-reported

Your agent noted that **2 of 5 seeds collapse to 2–16% at the recipe edge.** If that's true in the regime you're quoting, then "96–98% multi-seed" needs a qualifier, because as written it implies all seeds land there.

The honest framing is stronger than the sanitized one: *"3 of 5 seeds converge to 96–98%; 2 collapse at the recipe boundary, which is the training-stability problem I'm currently working."* That's a candidate who understands his own system. The unqualified number is a candidate who'll get caught.

### 🟠 3.5 42% of your observation space carries no signal

16 of 38 dimensions are a nearest-body proximity block that the current recipe doesn't exercise (body[0] is Earth; the debris infrastructure is unused). That's a large fraction of the input carrying nothing, which costs sample efficiency and is exactly the sort of thing that gets flagged in a design review.

Either mask it out of the current recipe, or exercise it — a debris field with keep-out constraints is a genuinely interesting extension and it's already scaffolded.

### 🟡 3.6 Env/deliverable version skew

The env exposes Discrete(16); the deliverable checkpoint was trained on Discrete(10) and masks 10–15 via `legacy_action_space`. That's fine as an engineering reality but it means your reported result and your current environment aren't the same system. Either retrain on the full action space or document the skew explicitly — don't let someone discover it.

### 🟡 3.7 Discounting is time-inconsistent under time-warp

This is subtle and I think it's a real design question. With a **fixed per-step γ=0.995**, a 60-second coast and a 1-hour warp incur identical discount. That means the agent gets more physical time advanced per unit of discount by choosing warps, which creates a structural bias toward time-warp actions independent of whether they're good guidance.

In a semi-MDP with variable-duration actions, time-consistent discounting uses **γ^Δt** rather than a flat γ. You may have chosen the flat version deliberately — it's a reasonable exploration aid, and it may be what makes long-horizon phasing learnable at all. But you should be able to say which, and why. "I chose per-step discounting because it makes long phasing arcs reachable in the effective horizon, and I checked that the policy doesn't degenerate into warp-spamming" is a great answer. Not having considered it is a worse one.

Related: γ=0.995 gives an effective horizon around 200 steps against a 2000-step cap. Worth confirming that's intentional.

### 🟡 3.8 The escape termination is unreachable

Escape from 500 km LEO requires roughly 3.15 km/s (v_circ ≈ 7.6, v_esc ≈ 10.75). You have 480 m/s. **Specific orbital energy can never reach zero**, so that termination branch never fires.

Harmless, but it's dead code in a physics model, and a reviewer who spots it wonders what else is untested. Either remove it, or keep it as a guard and say so.

### 🟡 3.9 2D coplanar removes the expensive part of the problem

Worth being precise about, because "it's 2D" undersells what's missing. Coplanar doesn't just simplify — it eliminates plane changes, which are the *dominant* Δv cost in real rendezvous. An inclination change of even a few degrees in LEO costs hundreds of m/s, comparable to your entire budget. Going to 3D isn't an incremental fidelity bump; it changes which maneuvers are affordable and therefore what the optimal strategy looks like.

Say this yourself before someone says it to you. It reads as understanding rather than as a gap.

### 🟡 3.10 Perturbations matter more than "no J2, no drag" suggests

At **300 km**, drag is not a rounding error — decay is meaningful over hours, and your episodes span up to 2000 steps with hour-scale warps, so potentially days of simulated time. Over that horizon, real drag would materially change the phasing problem.

**J2** in-plane causes apsidal precession (ω rotates) and modifies the orbital period. For your e ≤ 0.05 cases over long horizons, that's a real phasing perturbation. J2's out-of-plane effect (nodal regression) doesn't apply in 2D, but the in-plane effect does.

Neither is a flaw — it's a scoped testbed. But "no J2 or drag" should come with "and here's specifically what that would change," which is a much stronger thing to be able to say.

---

## 4. Future work, in the order I'd do it

### Tier 1 — before anything else (~1 week)

These affect every number you'd report, so they gate everything downstream.

1. **Tighten the terminal criterion and retrain.** Test 5 km / 1 m/s and 1 km / 0.5 m/s. Report the honest degraded number. This is the credibility fix.
2. **Lambert baseline** across the test distribution. Policy Δv vs. optimal, as a ratio. Also: success rate if you constrained the budget to Lambert-optimal + margin.
3. **Failure mode taxonomy** from your existing 200-episode eval trajectories. Three named modes with prevalence, in the working regime rather than the collapse regime.
4. **Seed variance stated plainly** in whatever you report.

### Tier 2 — the navigation extension (~2 weeks)

This is what converts the project from guidance to GN&C, and it's the specific thing the Draper Space Nav role hires for.

5. **Measurement model.** Range and bearing to target with realistic noise — white noise, bias, scale-factor error. Noisy Δv accumulation on burns as an IMU proxy.
6. **Extended Kalman Filter** over relative state in LVLH. Propagate covariance through the Kepler steps, update on measurements.
7. **Validate the filter, don't just run it.** This is the part that separates a candidate from a hobbyist:
   - Position/velocity RMSE vs. truth across a Monte Carlo campaign
   - **NEES/NIS chi-square consistency** — proves the covariance isn't lying
   - Error plotted against ±3σ bounds
8. **Rerun rendezvous on estimated state** and report the degradation. "96–98% on truth, 8X% on filtered state under realistic noise" is the single best sentence you could put in front of a navigation hiring manager, and it gives you a real dispersion story for free.

### Tier 3 — deepening (opportunistic)

9. **Clohessy–Wiltshire + LQR baseline in MATLAB/Simulink.** A weekend. Gets you Simulink honestly, gives you the classical optimal-control comparison, and lets you say "I benchmarked the learned policy against a CW/LQR baseline" — which reads as literacy in the field rather than in one technique.
10. **3D.** The big one. Inclination, RAAN, plane-change cost.
11. **J2 and drag**, at least as a perturbation you can toggle to quantify what they'd change.
12. **Exercise the debris/keep-out block** you already built. Keep-out constraints are a genuinely interesting RL problem and would justify 16 dimensions currently doing nothing.
13. **A real Monte Carlo dispersion campaign** with an actual dispersion table — nav error, thrust magnitude and pointing error, mass properties, Isp variation, timing latency. Report 3σ miss distance and 95th-percentile propellant. That's what "Monte Carlo verification" means in a GNC shop, and once you have the EKF you'll have the machinery for it.
14. **Finite burns** instead of impulsive, which introduces a real thrust-duration and pointing problem.

---

## 5. How to present it

**Lead with the limitations.** Draper's onsite includes a 30–40 minute technical presentation, and the interview style is project-interrogation. The candidate who opens with "here's what this model does and doesn't capture, and here's why those choices were right for what I was trying to learn" wins the room in the first two minutes. The candidate who gets caught at minute twenty loses it.

**Structure the talk the way the req describes the job:** problem → requirements → design trades → implementation → verification → what I'd do differently. That arc is nearly verbatim from the Space Nav posting, and your project maps onto it cleanly.

**Have these five answers loaded:**

1. Why RL and not Lambert (§3.2)
2. What your terminal criterion actually means physically (§3.1)
3. Why analytic Kepler propagation rather than numerical integration
4. What the other 26 observation dimensions do, and that 16 are currently inert
5. Three named failure modes with prevalence

**Write the memo.** Three to four pages: architecture, dynamics model, action and observation design, reward construction, results with baselines, known limitations, future work. Draper's req literally lists "prepares technical data, procedures, reports" as a duty. A well-written memo is simultaneously your writing sample, your interview prep, and the thing you attach when someone asks to see the work.

**Fix the repo situation.** Private fork means "can I see it?" gets an awkward answer. Either extract a clean public repo with a real README and results plots, or don't link anything and lead with the memo.

---

## 6. The honest bottom line

This is a genuinely good project and it's the reason you're a credible candidate for these roles rather than an implausible one. The C environment, the analytic propagation choice, the LVLH framing, and the PBRS reward design are all things a Draper engineer would recognize as real work.

It has one structural weakness — **the headline metric is measured against a criterion that doesn't mean what it says** — and one strategic gap — **there's no navigation in it.** Both are fixable in about three weeks, and neither requires starting over.

The version of this project with a tightened criterion, a Lambert baseline, named failure modes, and a validated EKF is not a side project a hiring manager tolerates. It's the thing that gets you the interview.
