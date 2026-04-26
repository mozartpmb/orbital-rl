"""Evaluate a trained orbital model and save trajectory logs.

Usage:
    python scripts/orbital/eval_checkpoint.py experiments/puffer_orbital_177610309074/model_puffer_orbital_000153.pt
    python scripts/orbital/eval_checkpoint.py experiments/puffer_orbital_177610323688/model_puffer_orbital_000306.pt --debris
"""

import argparse
import os
import sys
import numpy as np
import torch

# Add pufferlib to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from pufferlib.ocean.orbital.orbital import Orbital
from pufferlib.models import Default, LSTMWrapper


def evaluate(checkpoint_path, num_episodes=50, debris=False, out_dir=None, seed=42,
             e_max_target=0.0, init_phase_gap_max=0.524):
    if out_dir is None:
        tag = "debris" if debris else "no_debris"
        ckpt_name = os.path.splitext(os.path.basename(checkpoint_path))[0]
        out_dir = f"logs/orbital/eval_{tag}_{ckpt_name}"

    os.makedirs(out_dir, exist_ok=True)

    # Create environment (single env for clean trajectory collection)
    num_debris_min = 4 if debris else 0
    num_debris_max = 8 if debris else 0
    env = Orbital(
        num_envs=1,
        num_debris_min=num_debris_min,
        num_debris_max=num_debris_max,
        e_max_target=e_max_target,
        init_phase_gap_max=init_phase_gap_max,
        traj_log_dir=out_dir,
        traj_log_every=1,  # save every episode
    )

    # Load model — PufferLib wraps Default in LSTMWrapper
    device = torch.device("cpu")
    base_policy = Default(env)
    policy = LSTMWrapper(env, base_policy)

    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    policy.load_state_dict(state_dict)
    policy.eval()

    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"Debris: {debris} (min={num_debris_min}, max={num_debris_max})")
    print(f"Saving trajectories to: {out_dir}")

    obs, _ = env.reset(seed=seed)
    episodes_done = 0
    episode_rewards = []
    episode_lengths = []
    successes = 0
    step_count = 0

    # LSTM hidden state (dict form for LSTMWrapper.forward_eval)
    state = {
        'lstm_h': torch.zeros(1, policy.hidden_size),
        'lstm_c': torch.zeros(1, policy.hidden_size),
    }

    while episodes_done < num_episodes:
        obs_tensor = torch.from_numpy(obs).float().unsqueeze(0)
        with torch.no_grad():
            action_logits, _ = policy.forward_eval(obs_tensor, state)
            action = torch.argmax(action_logits, dim=-1).numpy()

        obs, rewards, terminals, truncations, infos = env.step(action)
        step_count += 1

        if terminals[0]:
            episodes_done += 1
            rew = float(rewards[0])
            episode_rewards.append(rew)
            episode_lengths.append(step_count)
            if rew > 0:
                successes += 1
            step_count = 0
            # Reset LSTM state for new episode
            state['lstm_h'] = torch.zeros(1, policy.hidden_size)
            state['lstm_c'] = torch.zeros(1, policy.hidden_size)

            if episodes_done % 10 == 0:
                print(f"  Episode {episodes_done}/{num_episodes}: "
                      f"reward={rew:.1f}, success_rate={successes/episodes_done:.1%}")

    env.close()

    # Summary
    print(f"\n{'='*50}")
    print(f"Evaluation Summary ({num_episodes} episodes)")
    print(f"{'='*50}")
    print(f"Success rate:    {successes}/{num_episodes} ({successes/num_episodes:.1%})")
    print(f"Mean reward:     {np.mean(episode_rewards):.2f}")
    print(f"Mean ep length:  {np.mean(episode_lengths):.0f} steps")

    # Check saved files
    npz_files = sorted([f for f in os.listdir(out_dir) if f.endswith('.npz')])
    print(f"Saved {len(npz_files)} trajectory files to {out_dir}")
    return out_dir


def main():
    parser = argparse.ArgumentParser(description='Evaluate orbital RL checkpoint')
    parser.add_argument('checkpoint', help='Path to .pt checkpoint file')
    parser.add_argument('--episodes', type=int, default=50, help='Number of eval episodes')
    parser.add_argument('--debris', action='store_true', help='Enable debris (4-8)')
    parser.add_argument('--out-dir', default=None, help='Output directory for trajectories')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--e-max-target', type=float, default=0.0,
                        help='Max target eccentricity (default 0.0 = circular target, matches orbital.ini)')
    parser.add_argument('--init-phase-gap-max', type=float, default=0.524,
                        help='Max initial phase gap in radians (default 0.524 = 30°)')
    args = parser.parse_args()

    evaluate(args.checkpoint, args.episodes, args.debris, args.out_dir, args.seed,
             args.e_max_target, args.init_phase_gap_max)


if __name__ == '__main__':
    main()
