# Defensive briefing: the four areas to answer briefly and steer away from

Each section has: the picture to hold in your head, a 30-second answer, the two most likely follow-ups with short answers, and the redirect sentence that brings the conversation back to ground you own (policy, physics, verification). Numbers are the audited post-fix ones.

---

## 1. The EKF (the range/bearing filter, 2-D; the "recursive filter" in the deck)

**The picture.** You have two opinions about where the target is: a prediction ("I knew where it was a minute ago, and orbits are predictable, so it should be here now") and a measurement ("the sensor says it is over there"). Each opinion comes with a cloud of uncertainty. The filter's whole job is to blend the two opinions, weighting whichever is more certain. If the prediction is sharp and the sensor is noisy, trust the prediction; if the sensor is sharp, trust the sensor. The blended cloud is always tighter than either input. Then propagate the blended estimate forward one step, which stretches the cloud a bit, measure again, blend again. Repeat.

"Extended" just means the orbit dynamics and the sensor geometry are nonlinear, so at each step you linearise them around your current best guess (a Jacobian) to do the blending arithmetic. That is the only difference from a plain Kalman filter.

**30-second answer.** "It is a predict-correct loop. The state is the target's position and velocity. Predict with exact two-body propagation, so there is no model error, only a small process-noise floor. Correct with a range and a bearing, each with a known noise. The gain is the usual ratio of prediction uncertainty to measurement uncertainty, with the Jacobian of the measurement model because the geometry is nonlinear. I checked it was honest with NEES and NIS."

**Likely follow-up: what are NEES and NIS?** "Consistency checks. NEES asks: is my actual error about the size my covariance claims? NIS asks the same for the innovations, the surprise in each measurement. For a calibrated filter they sit in a chi-square band about 95% of the time. Mine were in the band 94 and 95 percent of steps." If pushed on why NEES was slightly low: "a coarse single-measurement initialisation; the shortfall was not concentrated in the transient."

**Likely follow-up: how did you set the process noise?** "Not by hand. Swept it and picked the value that put NEES in bounds. Because propagation is exact, Q is a covariance floor, not a model-error term."

**Redirect.** "The filter was not the bottleneck at realistic noise. What mattered was that the policy had to be trained on the estimate rather than on truth, which is the 51 to 97 result."

---

## 2. Batch initial orbit determination (angles-only acquisition)

**The picture.** Stand on a road at night and see a single light in the sky. You know its direction but not its distance. It could be a plane at 10 km or a satellite at 1,000 km. One direction cannot tell you. But if you watch for a while and the light's direction changes in a particular way, and you know how planes and satellites move, only one distance makes the motion you saw consistent with the physics. That is angles-only orbit determination: range is unobservable from one bearing but becomes observable from a sequence of bearings plus the dynamics.

The batch solver does this bluntly. Lay down a set of candidate ranges along the first line of sight, spaced so each is 15% farther than the last. For each candidate assume a few plausible velocities. Each candidate is a full orbit guess. Propagate each one forward and score it by how badly its predicted bearings miss the measured ones. Then polish the best few with a local least-squares refinement. Accept the winner only if it fits as well as sensor noise alone would allow, and if the runner-up at a materially different range fits clearly worse. If the data cannot yet tell the ranges apart, wait and collect more bearings.

**30-second answer.** "The sensor gives direction only. A sequence of directions while both orbits curve does constrain range, so I seed a grid of range hypotheses along the first line of sight, propagate each, score by bearing misfit, refine the best few by Gauss-Newton, and accept only when the fit is at the noise floor and the runner-up is clearly worse. That hands a full state to the recursive filter."

**Likely follow-up: how long does acquisition take?** "About 45 minutes to two hours in LEO, median around 105 minutes. The policy learned to coast through that window instead of burning blind."

**Likely follow-up: why not just start the filter from a guess?** "I did at first, and the filter understates range error badly before acquisition. The batch step was the fix: never-acquire episodes went from 44 in 200 to 0."

**Redirect.** "What I find interesting is what the policy did with it: it learned information discipline, not information seeking. It burns nothing before acquisition, then acts on a converged estimate."

---

## 3. The controlled comparator (Lambert fuel audit)

**The picture.** You want to grade the policy's fuel use, so you need the "right answer" for each episode. Lambert's problem is: given where I am now and where the target will be at some future time, what single orbit connects those two points in exactly that time? Solve it and you get the velocity you need at departure and the velocity you arrive with. The difference from your current velocity is burn one; the difference from the target's velocity at arrival is burn two. Two burns, total Δv. Do that for every choice of "how long to wait before departing" and "how long to fly," and the smallest total over that grid is the two-impulse optimum for that episode.

The universal-variable form has a term with sin(Δθ) in it, which goes to zero for a 180-degree transfer, exactly the hard phase gap. So the solver uses the Lancaster-Blanchard / Izzo formulation, which stays well-behaved there and also handles multi-revolution transfers.

**30-second answer.** "For each episode I solve Lambert's problem over a grid of departure waits and flight times, including multi-rev solutions, on the same clock the policy used. The minimum is the two-impulse optimum. The policy's median was 1.18 times that; 1.55 when the optimum only has to reach the box. The circular Hohmann surrogate was retired because it ignores phasing and the velocity match."

**Likely follow-up: why is the policy sometimes below 1?** "Because it only has to reach a 30 km / 50 m/s box, not an exact state, and because a multi-burn plan can beat two burns under a deadline."

**Likely follow-up: why is it 18% over?** "Coarse burn grades, 1 to 25 m/s, and it trades fuel for faster closure. Finer grades are the obvious next step."

**Redirect.** "The comparator earned its keep by catching the dynamics bug: the policy was beating half the optimum on 15% of episodes, which is impossible."

---

## 4. The C port

**The picture.** A trained network is just a list of numbers (weights) and a recipe (multiply, add, apply a nonlinearity, repeat). PyTorch is the convenient way to run that recipe; a flight computer cannot carry PyTorch. So you write the recipe out in plain C: a matrix-vector multiply for the encoder, GELU, the four LSTM gate equations, two small output layers. You export the weights to a header file. Then you prove the C version makes the same decisions as PyTorch on recorded inputs and in a closed loop.

The flight-software properties people care about: no memory allocation during a call (so no fragmentation or out-of-memory path), the same amount of work every call (so timing is predictable), no hidden state the caller does not own, and a guard that refuses a NaN input without corrupting the recurrent memory.

**30-second answer.** "The policy is small enough to write by hand in C: the encoder, GELU, an LSTM cell, two heads. Weights live in a read-only header. No heap, fixed work per call, the caller owns the 1 kB of recurrent state. I verified action-for-action agreement with PyTorch on 7,048 recorded observations and an identical action stream over 200 closed-loop episodes. About 50 microseconds per decision on a laptop against a 60-second decision period."

**Likely follow-up: is that real-time?** "It is a desktop benchmark, not a worst-case execution time. The margin is six orders of magnitude, so even a slow rad-hard processor has headroom, but I would not claim WCET without the target hardware."

**Likely follow-up: did you test it with sanitizers?** "Yes, AddressSanitizer and UndefinedBehaviorSanitizer clean on the full closed loop."

**Redirect.** "The port was a verification exercise more than an embedded one. The thing I would want next is the filter in C too, since that is where the real onboard compute lives."

---

## General steering lines

- When a question goes deep into one of these: give the 30-second answer, then "the part of that I spent my time on was..." and name the policy or verification result it connects to.
- When you do not know: "I did not go that deep on that piece; what I verified is X, and the thing I would check next is Y." Specific beats vague.
- The three results you can always return to: 200/200 on five seeds in 2-D; one policy at 96 to 99.5% on five 3-D skills; the dynamics bug caught by the comparator.
