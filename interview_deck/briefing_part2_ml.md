# Defensive briefing, part 2: RL and ML methodology

For an ML-literate interviewer (Draper autonomy group). Each question has the picture, the two-minute answer, the numbers, the likely follow-ups, and the one honest gap. Appendix slides A12 to A17 of `deck_hooks.pdf` carry the same material in slide form.

---

## 6. "Walk me through the curriculum. How did you pick stage order and boundaries?"

**The picture.** Nobody learns to land a plane by starting in a crosswind at night. You learn straight-and-level, then turns, then landings, then crosswinds. A curriculum is the same for a policy: start with the easiest version of the task that still contains the core skill, let the network solve it, then add one difficulty at a time, each stage starting from the weights that solved the last one.

**The two-minute answer.**
"The core skill is phasing: catching up to something on a different part of the same orbit. So stage one is circular orbits, any phase gap, nothing else. Once that is at ceiling, add eccentricity. Then widen the altitude band. Then 3-D with inclined planes and out-of-plane burns. Then replace the true target state with the filter's estimate. Then switch J2 on. Finally mix all seven scenario cells into one training distribution with the fuel tank sampled per episode.

One new variable per stage, because fresh training at the hard endpoint flat-lined every time I tried it, six times. Each stage starts from the previous stage's weights, so the only thing left to learn is the new difficulty. When the action head had to grow for 3-D, the new rows were zero-initialised so the policy behaved identically at step zero.

A stage was declared done by a pre-registered gate: at least 60 percent held-out success across three training seeds. In practice every rung cleared 200 of 200 before I advanced, so the gate never had to make a hard call. Advancement was decided by the held-out number, not by eye.

Cost: 50 million environment steps per stage, about two hours on a laptop CPU at half a million steps a second. The whole chain is a day or two of compute; the expensive part was deciding what each stage should be, which came from looking at how the previous policy failed."

**Follow-up: how did you pick the boundaries, e.g. why e ≤ 0.05 and then 0.30?** "From the failure modes of the previous policy. Each cell in the final mixture answers a measured failure of an earlier one: the eccentric cells because the circular policy fell apart above e = 0.075; the long-range cell because the LEO policy could not phase across a 500 km altitude gap; the tight box because the wide-box policy never used fine burns."

**Follow-up: did the curriculum ever hurt?** "Yes. Fine-tuning for the tight box after the six wide skills destroyed three of them (99 to 29, 98 to 6, 99 to 40). That is the interference story. The fix was a low learning rate plus replaying each skill's own trajectories."

**Redirect.** "The part I would defend hardest is that nothing advanced on a feeling; every boundary has a 200-episode held-out number behind it."

---

## 7. "Why did reward never find the seventh skill?"

**The picture.** A maze where the only reward is at the exit, the maze takes a week to walk, and the correct first move is to stand still for a day. Random wandering never does that. That is the plane-change skill: coast for days at a chosen altitude so Earth's bulge drifts the planes together, then rendezvous. The payoff is one +1 at the end of a 175-hour episode, and the shaping hint cannot help because it rewards closing the gap now, not setting up conditions that close it later.

**The answer.** "Sparse terminal reward over a 175-hour episode where the winning move is to coast for days is an exploration problem, not a capacity problem. Zero of 200 under every reward variant I tried. So I wrote a scripted expert that knows the manoeuvre, trained the policy on its demonstrations, then ran four rounds of DAgger: let the policy fly, have the expert label what it should have done at the states the policy actually visited, retrain. 31.5, 45, 59, 70, 74 percent, each a 200-episode held-out number. The same 128-unit network holds the skill once shown it, which is why I say capacity was never the issue."

**Follow-up: why DAgger, not plain behaviour cloning?** "Plain cloning only sees the expert's states. The student drifts into states the expert never visited and has no idea what to do there. Relabelling the student's own visited states is the fix for that compounding error."

**Follow-up: is 74 good?** "The teacher gets 91, so there is headroom, and the endpoint chain is one training seed. The mechanism reproduced at a second seed step for step: 33.5 at kickstart, plus 14 on the first refresh, same as seed 42."

**Follow-up: couldn't you have shaped the reward toward the drift?** "I tried reward variants and all were 0/200. A shaping term that pays for being at the right altitude would have to know the manoeuvre already, at which point you are writing the expert anyway. Demonstrations were the honest way to inject that knowledge."

---

## 8. "Why an LSTM, and what does the memory carry?"

**The picture.** One photo of a car cannot tell you how fast it is going or whether it just braked. Two photos a second apart can. The policy's observation is one photo: the current state and the filter's current estimate. Two things that matter are only visible across photos: what the last burn actually did, and whether the filter's estimate is still moving or has settled. An LSTM is a small memory that carries that across decisions.

**The answer.** "The policy flies on an estimate, not truth, so the problem is partially observable. The LSTM carries what the last burn did, since the observation shows the state after and not the change, and where the filter is in its convergence. Architecture is a 38-to-128 linear encoder with GELU, a 128-unit LSTM, two linear heads, 139 thousand parameters. Hidden state persists within an episode and is zeroed at reset; training uses truncated backprop through time over 64 decisions.

I tested the obvious alternative: hand the policy the filter's covariance directly as an extra channel. It changed nothing, 97.0 versus 97.0 percent. The recurrent policy already infers what it needs about estimate quality from how the estimate moves."

**Follow-up: did you probe what the hidden state encodes?** "Not directly. The evidence is behavioural: it suppresses burns three-fold while the filter is blind and spends a median of zero Δv before acquisition, which requires knowing acquisition has not happened, and that is not in the instantaneous observation."

**Honest gap, volunteer it.** "I never ran the memoryless control. The LSTM was the library default and it worked; a feed-forward ablation is on the list and I would expect it to hurt most in the bearings-only setting."

---

## 9. "How would you take this to a maneuvering target, or to multiple agents?"

**Maneuvering target: the filter changes first.** Today's filter assumes a ballistic target. If the target burns, the filter's innovations grow and its estimate diverges, and a policy trained on that filter will be confidently wrong. So the order is: first a residual monitor (a chi-square gate on the innovation) to detect a manoeuvre and inflate the covariance; then an interacting-multiple-model filter that runs ballistic and manoeuvring hypotheses side by side and weights them by likelihood; only then retrain the policy on the new estimate statistics, the same recipe as the 51-to-97 step. The target's manoeuvre model also has to go into the simulator, and that choice (random impulses, evasive, cooperative) defines the task.

**Multiple agents: the shape, not the depth.** "I have not built multi-agent. The structure I would start from is centralised training with decentralised execution: a shared critic that sees every agent during training, individual policies that act on their own observations at run time. The simulator needs a communications model, what each agent can see of the others with delay and dropout, because the policy can only use what it would really have. My simulator is already vectorised over 1,024 independent episodes; making agents share an episode is the structural change. First reading would be the MAPPO paper (Yu et al. 2022) for the training structure and Bar-Shalom's IMM chapters for the filter."

**Redirect.** "The single-agent pieces that carry over unchanged are the verification loop and the policy-on-estimate training. Those are the parts I would want to keep."

---

## 10. PPO, if she goes there

- **What it is.** Collect a batch of experience with the current policy; estimate how much better each action was than expected (the advantage, by GAE with λ = 0.9); update the policy to raise the probability of better-than-expected actions, but clip the update so the new action probabilities stay within 20 percent of the old ones. The clip is what makes it stable.
- **As configured.** 1,024 parallel environments; γ = 0.995, a credit horizon of about 200 decisions; one epoch per batch; minibatch 8,192; entropy bonus 0.01; value-loss weight 2; learning rate 0.01 annealed, 0.001 for consolidation; Muon optimizer.
- **The reward clamp.** Rewards are clamped to [−1, 1] before advantages are computed. This bounds the gradient from rare terminal events. Side effect: the environment's fuel bonus on success (5 to 10 before the clamp) carries no gradient. Fuel awareness came from sampling the tank size, not from reward.
- **Potential-based shaping.** The dense term is Φ(next) − Φ(now) with γ_shape = 1. Summed over an episode it telescopes to Φ(end) − Φ(start), so it cannot change which policy is optimal; it only densifies the gradient. Without it, training peaked at 3.6 percent.
- **Entropy 0.01 mattered for a reason.** At lower entropy the long-coast actions died out before the first success was ever sampled, and without them nothing learns.
- **Never ablated, say so.** The 128 width, GELU, the LSTM itself. Library defaults that worked.

---

## 11. Fuel awareness, stated correctly

What happened: the six-skill generalist was trained with the tank sampled per episode between 353 and 656 m/s and shown its fuel fraction. Evaluated with a fixed lean tank it spends about 40 percent less Δv than with a rich one (medians 246 to 272 m/s versus 372 to 437), at a success cost of 1 to 11 points per cell. Against a quick linearised direct-burn reference the lean spend is 1.02 to 1.07×, which is the "near 1.0" number. Against the two-impulse Lambert optimum it was not measured and would be higher (the same policy at the sampled tank is about 2× box-credited Lambert). Two caveats to carry: lean medians are over successes only, and the lean tank is a hard ceiling, so part of the drop is enforced. Separately, the low-learning-rate consolidation stage cut wide-cell spend 14 to 21 percent without being asked. Nothing was ever trained below 353 m/s; at 245 m/s roughly half of episodes are infeasible.

**The sentence.** "The policy is budget-aware, not fuel-optimal. It spends what it has, and it degrades gracefully when it has less."

---

## 12. Other ML questions worth a prepared line

- **Why PPO and not SAC or DQN?** Discrete actions, on-policy stability with a cheap simulator, and a recurrent policy, which PPO handles cleanly with BPTT. Sample efficiency was never the constraint at half a million steps a second.
- **Why discrete actions?** A continuous Box(2) head collapsed (entropy went to zero) in early trials; fixed grades are simpler for PPO and match impulsive chemical burns. The cost is fuel: 1.14 to 1.18× the two-impulse optimum.
- **How do you know it is not overfitting?** Held-out evaluation seed never seen in training; every number is on fresh scenario draws; the wide-envelope policy generalises across altitude and eccentricity it was trained on but not to a different observation normaliser.
- **Overfitting you did find.** A variant (DAPO) that improved rolling training success by 5 points lost 7 on held-out. Dropped.
- **What was the biggest ML lesson?** Verification machinery found a dynamics bug that made every result look 10 points better. Measure against physics, not just against reward.
- **What would ten times the compute buy?** Repeats, not skills: multi-seed the generalist endpoint, the memoryless ablation, and a matched classical comparator. Not a bigger network.
