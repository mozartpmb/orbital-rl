"""Greedy-eval every checkpoint in an experiment dir; print success@e_max.

Usage:
    python scripts/orbital/eval_all_ckpts.py experiments/puffer_orbital_<id> \
        --e-max-target 0.20 --episodes 50
"""

import argparse, os, sys, glob, numpy as np, torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'pufferlib'))
from pufferlib.ocean.orbital.orbital import Orbital
from pufferlib.models import Default, LSTMWrapper


def run_eval(ckpt_path, num_episodes, e_max_target, seed, env, policy):
    state_dict = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    policy.load_state_dict(state_dict)
    policy.eval()

    obs, _ = env.reset(seed=seed)
    state = {
        'lstm_h': torch.zeros(1, policy.hidden_size),
        'lstm_c': torch.zeros(1, policy.hidden_size),
    }
    successes = 0
    done = 0
    step = 0
    lens = []
    while done < num_episodes:
        obs_t = torch.from_numpy(obs).float().unsqueeze(0)
        with torch.no_grad():
            logits, _ = policy.forward_eval(obs_t, state)
        # Discrete: logits is a tensor (B, n_actions). Continuous: logits is a Normal dist.
        if isinstance(logits, torch.distributions.Normal):
            action = logits.mean.cpu().numpy().astype(np.float32).reshape(1, -1)
        else:
            action = np.array([int(torch.argmax(logits, dim=-1).item())])
        obs, rewards, terms, _, _ = env.step(action)
        step += 1
        if terms[0]:
            if float(rewards[0]) > 0:
                successes += 1
            lens.append(step)
            done += 1
            step = 0
            state = {
                'lstm_h': torch.zeros(1, policy.hidden_size),
                'lstm_c': torch.zeros(1, policy.hidden_size),
            }
    return successes / num_episodes, float(np.mean(lens))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('exp_dir')
    p.add_argument('--e-max-target', type=float, required=True)
    p.add_argument('--init-phase-gap-max', type=float, default=0.0,
                   help='Phase 3 rendezvous: max initial sat/target angular gap (rad)')
    p.add_argument('--episodes', type=int, default=50)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    ckpts = sorted(glob.glob(os.path.join(args.exp_dir, 'model_puffer_orbital_*.pt')))
    if not ckpts:
        print(f"No checkpoints in {args.exp_dir}")
        return

    env = Orbital(num_envs=1, num_debris_min=0, num_debris_max=0,
                  e_max_target=args.e_max_target,
                  init_phase_gap_max=args.init_phase_gap_max)
    policy = LSTMWrapper(env, Default(env))

    print(f"Evaluating {len(ckpts)} ckpts at e_max={args.e_max_target}, "
          f"phase_gap={args.init_phase_gap_max:.3f} rad, {args.episodes} eps each")
    print(f"{'ckpt':>10s}  {'success':>7s}  {'mean_len':>8s}")
    best = (None, -1.0, 0.0)
    for ck in ckpts:
        succ, mlen = run_eval(ck, args.episodes, args.e_max_target, args.seed, env, policy)
        epoch = os.path.basename(ck).split('_')[-1].split('.')[0]
        print(f"{epoch:>10s}  {succ:>7.1%}  {mlen:>8.0f}")
        if succ > best[1]:
            best = (ck, succ, mlen)
    env.close()

    print(f"\nBest: {os.path.basename(best[0])}  success={best[1]:.1%}  mean_len={best[2]:.0f}")


if __name__ == '__main__':
    main()
