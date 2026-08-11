"""RED-TEAM B — "the 38-dim layout is frozen, so what tells a warm-started
policy that it is blind?"

Claim under test: the ext-3d vNext reserved slots 30-37 do NOT exist in the 2D
layout the nav lineage trains on, so a nav-valid bit has nowhere to live.

Counter-hypothesis this probe tests: every nav/T3 config runs
`num_debris_min=num_debris_max=0`, and `fill_observations` hard-zeroes the
3 unused body blocks. If obs[21..32] are identically zero for the whole
training distribution, there are TWELVE free channels inside the frozen 38-dim
layout — and the canonical checkpoint's weights on them received exactly zero
gradient, so we must also measure what happens to its policy when one is lit.

Measures:
  1. per-slot min/max/|max| over N greedy episodes of the canonical policy
  2. argmax-flip rate and value-head shift when obs[21] is set to 1.0 (and to
     other magnitudes), i.e. the cost of using a dead slot as a nav-valid bit
  3. per-decision |delta| of the TARGET-derived slots, to size how large an
     acquisition jump is relative to the obs motion the LSTM is used to
"""
import csv
import math
import os
import sys

import numpy as np
import torch

ROOT = "/Users/pete/space_training"
sys.path.insert(0, f"{ROOT}/pufferlib")
sys.path.insert(0, f"{ROOT}/scripts/orbital/nav")

from pufferlib.ocean.orbital.orbital import Orbital      # noqa: E402
from pufferlib.models import Default, LSTMWrapper        # noqa: E402
import eval_relnav as ev                                 # noqa: E402

OUT = f"{ROOT}/web_data/results/ext_rtnav_obs_slots.csv"
CKPT = f"{ROOT}/models/t3/seed42_L2_headline.pt"

TGT_SLOTS = (7, 8, 11, 12, 13, 14, 16, 33, 34, 35, 36, 37)
BODY_SLOTS = tuple(range(17, 33))


def main():
    n_eps = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    torch.set_num_threads(1)
    env = Orbital(**ev.T3_ENV_KWARGS)
    pol = LSTMWrapper(env, Default(env))
    pol.load_state_dict(torch.load(CKPT, map_location="cpu", weights_only=True))
    pol.eval()
    print(f"obs dim {env.single_observation_space.shape}  "
          f"act {env.single_action_space}  ckpt {os.path.basename(CKPT)}")

    obs, _ = env.reset(seed=123)
    lo = np.full(38, np.inf)
    hi = np.full(38, -np.inf)
    deltas = {i: [] for i in TGT_SLOTS}
    prev = None
    eps = 0
    steps = 0
    flips = {m: 0 for m in (0.1, 0.5, 1.0, 2.0)}
    dv_shift = {m: [] for m in flips}
    n_cmp = 0

    # LSTMWrapper state handling matching eval_relnav
    state = ev.zero_state(pol)
    while eps < n_eps and steps < 40000:
        o = np.asarray(obs, dtype=np.float32).reshape(-1)
        lo = np.minimum(lo, o)
        hi = np.maximum(hi, o)
        if prev is not None:
            for i in TGT_SLOTS:
                deltas[i].append(abs(float(o[i]) - float(prev[i])))
        prev = o.copy()

        t = torch.from_numpy(o).float().unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            snap = {'lstm_h': state['lstm_h'].clone(),
                    'lstm_c': state['lstm_c'].clone()}
            logits, _ = pol.forward_eval(t, state)
            lg = logits.reshape(-1)
            a0 = int(lg.argmax())
            # counterfactual: light obs[21] against a SNAPSHOT of the same state
            for m in flips:
                t2 = t.clone()
                t2[0, 0, 21] = m
                s2 = {'lstm_h': snap['lstm_h'].clone(),
                      'lstm_c': snap['lstm_c'].clone()}
                l2, _ = pol.forward_eval(t2, s2)
                l2 = l2.reshape(-1)
                if int(l2.argmax()) != a0:
                    flips[m] += 1
                dv_shift[m].append(float(torch.max(torch.abs(l2 - lg))))
            n_cmp += 1
        obs, rew, term, trunc, info = env.step(np.array([a0], dtype=np.int32))
        steps += 1
        if term[0] or trunc[0]:
            eps += 1
            state = ev.zero_state(pol)
            prev = None

    print(f"\nrolled {eps} episodes / {steps} decisions "
          f"({steps/max(eps,1):.1f} decisions/episode)")

    print("\n--- slot range over the T3 training distribution ---")
    dead = []
    for i in range(38):
        span = hi[i] - lo[i]
        tag = ""
        if abs(lo[i]) < 1e-9 and abs(hi[i]) < 1e-9:
            tag = "  <-- IDENTICALLY ZERO"
            dead.append(i)
        if i in BODY_SLOTS or tag:
            print(f"  obs[{i:2d}] min {lo[i]:+10.5f} max {hi[i]:+10.5f}"
                  f" span {span:9.5f}{tag}")
    print(f"\nDEAD SLOTS (free channels inside the frozen 38-dim layout): {dead}")
    print(f"count = {len(dead)}")

    print("\n--- canonical policy sensitivity to lighting obs[21] ---")
    for m in sorted(flips):
        print(f"  obs[21] := {m:4.1f}: argmax flips {flips[m]:5d}/{n_cmp} "
              f"= {100*flips[m]/max(n_cmp,1):5.2f}%   "
              f"max|dlogit| p50 {np.median(dv_shift[m]):.4f} "
              f"p95 {np.percentile(dv_shift[m],95):.4f}")

    print("\n--- per-decision |delta| of TARGET-derived slots (LSTM's diet) ---")
    rows = []
    for i in TGT_SLOTS:
        d = np.array(deltas[i])
        rows.append(dict(slot=i, p50=float(np.median(d)),
                         p90=float(np.percentile(d, 90)),
                         p99=float(np.percentile(d, 99)), max=float(d.max())))
        print(f"  obs[{i:2d}]  p50 {np.median(d):.6f}  p90 "
              f"{np.percentile(d,90):.6f}  p99 {np.percentile(d,99):.6f}  "
              f"max {d.max():.6f}")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["kind", "key", "v1", "v2", "v3", "v4"])
        for i in range(38):
            w.writerow(["slot_range", i, lo[i], hi[i], hi[i] - lo[i],
                        int(i in dead)])
        for m in sorted(flips):
            w.writerow(["obs21_flip", m, flips[m], n_cmp,
                        float(np.median(dv_shift[m])),
                        float(np.percentile(dv_shift[m], 95))])
        for r in rows:
            w.writerow(["tgt_delta", r['slot'], r['p50'], r['p90'], r['p99'],
                        r['max']])
        w.writerow(["episode_len_decisions", "", steps / max(eps, 1), eps,
                    steps, ""])


if __name__ == "__main__":
    main()
