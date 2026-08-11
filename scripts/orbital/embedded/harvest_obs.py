#!/usr/bin/env python3
"""Harvest reference inference data from the torch policy on the real C env.

Runs the exact eval_checkpoint.py greedy control loop (PufferLib Default +
LSTMWrapper, forward_eval, argmax) at the T3 canonical configuration and records
every decision epoch: the observation the env produced, the logits torch
computed, the value, the greedy action, and the episode-boundary flags that mark
where the LSTM state is zeroed.

The resulting .npz is the reference dataset for test_parity.py (the C
implementation replays the same observation sequence and must reproduce the
logits) and the raw .bin feeds bench_policy --obs.

Usage:
    python3 scripts/orbital/embedded/harvest_obs.py --episodes 200 --seed 123 \
        --out /tmp/emb_harvest_s123.npz
"""

import argparse
import os
import sys

import numpy as np
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
PUFFER = os.path.join(REPO, "pufferlib")
sys.path.insert(0, PUFFER)

from pufferlib.ocean.orbital.orbital import Orbital          # noqa: E402
from pufferlib.models import Default, LSTMWrapper            # noqa: E402

# T3 canonical env configuration — T3_RECOVERY_CAMPAIGN.md §5/§6,
# reproduced from scripts/orbital/t3/t3_ladder.sh::heval (L2 rung).
T3_ENV_KWARGS = dict(
    num_envs=1,
    num_debris_min=0,
    num_debris_max=0,
    e_max_target=0.05,
    e_max_sat=0.05,
    same_orbit_init=0,
    init_phase_gap_max=3.14159,
    valid_init_only=1,
    shaping_mode=1,
    shape_w_lambda=1.0,
    shape_w_match=0.35,
    shape_dv_ref_ms=300.0,
    shape_gamma=1.0,
    phase_gap_mode=1,
    phase_obs_mode=1,
    episode_cap_steps=3000,
    cap_terminal_reward=0.0,
    max_valid_init_attempts=4096,
    gave_up_action="terminate",
    obs_alt_scale_m=1.6e6,
    phi_orbit_scale_k=0.001,
    lvlh_scale_m=6.371e6,
    rendezvous_radius_m=30000.0,
    rel_vel_tol_ms=50.0,
)

CAUSE_NAMES = ['none', 'success', 'collision', 'escape', 'safety_cap',
               'stranded', 'hyperbolic', 'gave_up']


def build_policy(env, ckpt):
    base = Default(env)
    policy = LSTMWrapper(env, base)
    sd = torch.load(ckpt, map_location="cpu", weights_only=True)
    policy.load_state_dict(sd)
    policy.eval()
    return policy


def harvest(ckpt, episodes, seed, out_npz, trace_csv=None, obs_bin=None,
            record_state=False):
    torch.set_num_threads(1)
    env = Orbital(**T3_ENV_KWARGS)
    policy = build_policy(env, ckpt)

    obs, _ = env.reset(seed=seed)
    state = {'lstm_h': torch.zeros(1, policy.hidden_size),
             'lstm_c': torch.zeros(1, policy.hidden_size)}

    rec_obs, rec_logits, rec_value, rec_action = [], [], [], []
    rec_ep, rec_step, rec_is_start = [], [], []
    rec_h, rec_c = [], []
    causes, ep_rewards, ep_lengths = [], [], []

    trace = open(trace_csv, "w") if trace_csv else None
    if trace:
        trace.write("episode,step,action\n")

    ep, ep_step = 0, 0
    while ep < episodes:
        o = np.asarray(obs, dtype=np.float32).reshape(-1)
        with torch.no_grad():
            logits, value = policy.forward_eval(
                torch.from_numpy(o).unsqueeze(0), state)
        lg = logits.squeeze(0).numpy().astype(np.float32)
        act = int(np.argmax(lg))

        rec_obs.append(o.copy())
        rec_logits.append(lg.copy())
        rec_value.append(float(value.reshape(-1)[0]))
        rec_action.append(act)
        rec_ep.append(ep)
        rec_step.append(ep_step)
        rec_is_start.append(1 if ep_step == 0 else 0)
        if record_state:
            rec_h.append(state['lstm_h'].detach().numpy().reshape(-1).astype(np.float32).copy())
            rec_c.append(state['lstm_c'].detach().numpy().reshape(-1).astype(np.float32).copy())
        if trace:
            trace.write(f"{ep},{ep_step},{act}\n")

        obs, rewards, terminals, truncations, infos = env.step(
            np.array([act], dtype=np.int32))
        ep_step += 1

        if terminals[0]:
            sim_steps, cause = env.last_episode_result(0)
            causes.append(int(cause))
            ep_rewards.append(float(rewards[0]))
            ep_lengths.append(ep_step)
            ep += 1
            ep_step = 0
            state['lstm_h'] = torch.zeros(1, policy.hidden_size)
            state['lstm_c'] = torch.zeros(1, policy.hidden_size)

    if trace:
        trace.close()
    env.close()

    O = np.stack(rec_obs).astype(np.float32)
    L = np.stack(rec_logits).astype(np.float32)
    data = dict(
        obs=O,
        logits=L,
        value=np.asarray(rec_value, dtype=np.float32),
        action=np.asarray(rec_action, dtype=np.int32),
        episode=np.asarray(rec_ep, dtype=np.int32),
        step=np.asarray(rec_step, dtype=np.int32),
        is_episode_start=np.asarray(rec_is_start, dtype=np.int8),
        terminal_cause=np.asarray(causes, dtype=np.int32),
        episode_reward=np.asarray(ep_rewards, dtype=np.float32),
        episode_length=np.asarray(ep_lengths, dtype=np.int32),
        seed=np.asarray([seed], dtype=np.int64),
        checkpoint=np.asarray([os.path.abspath(ckpt)]),
        torch_version=np.asarray([torch.__version__]),
    )
    if record_state:
        data['lstm_h'] = np.stack(rec_h).astype(np.float32)
        data['lstm_c'] = np.stack(rec_c).astype(np.float32)
    np.savez_compressed(out_npz, **data)
    if obs_bin:
        O.tofile(obs_bin)

    n_succ = int((np.asarray(causes) == 1).sum())
    counts = {CAUSE_NAMES[c]: int((np.asarray(causes) == c).sum())
              for c in sorted(set(causes))}
    acts = np.asarray(rec_action)
    hist = {int(a): int((acts == a).sum()) for a in sorted(set(acts.tolist()))}
    print(f"harvested {len(rec_obs)} decision epochs over {episodes} episodes "
          f"(seed {seed})")
    print(f"  physical success : {n_succ}/{episodes}  causes={counts}")
    print(f"  mean ep length   : {np.mean(ep_lengths):.2f} decisions")
    print(f"  action histogram : {hist}")
    print(f"  obs range        : [{O.min():.4f}, {O.max():.4f}]")
    print(f"  wrote {out_npz}" + (f" and {obs_bin}" if obs_bin else ""))
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(REPO, "models/t3/seed42_L2_headline.pt"))
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--out", default="/tmp/emb_harvest.npz")
    ap.add_argument("--trace", default=None, help="write episode,step,action CSV")
    ap.add_argument("--obs-bin", default=None, help="write raw float32 obs matrix")
    ap.add_argument("--record-state", action="store_true",
                    help="also record the torch LSTM (h, c) at every epoch")
    args = ap.parse_args()
    harvest(args.ckpt, args.episodes, args.seed, args.out, args.trace,
            args.obs_bin, args.record_state)


if __name__ == "__main__":
    main()
