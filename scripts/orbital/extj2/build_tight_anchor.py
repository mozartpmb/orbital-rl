#!/usr/bin/env python3
"""Build the T13b tight-cell anchor dataset.

Rolls the ROOT policy on TIGHT_5k1 and stores fixed-length observation windows
for the trainer's anchor loss.

THE ONE THING THAT MATTERS HERE IS THE WINDOW CONVENTION, and it is not the
obvious one. `pufferl.train()` forwards its minibatches with
`lstm_h=None, lstm_c=None`, and `evaluate()` advances buffer rows every
`bptt_horizon` steps REGARDLESS of episode boundaries. So the trainer already
evaluates every 64-step window from a ZERO hidden state, wherever that window
happens to sit inside an episode.

The anchor therefore stores windows cut on that same cadence — every `horizon`
steps of the env's own stream, not episode-aligned, no stored h0/c0, no burn-in.
Reaching for R2D2 machinery here would anchor a DIFFERENT function than the one
PPO's gradient actually shapes, and it would do it silently. This is a fact
about the trainer's code, not a claim about LSTMs in general.

Usage:
    python3 scripts/orbital/extj2/build_tight_anchor.py \
        --ckpt models/t3/t11t_tight_child.pt \
        --out  models/t3/t11b_anchor_tight.pt \
        --windows 1200 --horizon 64
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'pufferlib'))

import pufferlib                                                    # noqa: E402
from pufferlib.models import Default, LSTMWrapper                   # noqa: E402
from pufferlib.ocean.orbital_nav.orbital_nav import OrbitalNav      # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import t11_cells as T                                               # noqa: E402


def cell_kwargs(cell, num_envs):
    c = T.ALL_CELLS[cell]
    return T.nav_env_kwargs(
        num_envs=num_envs, nav_mode='bearings_only', cell_mixture_mode=0,
        episode_cap_steps=int(c['cap']), rendezvous_radius_m=c['box_r'],
        rel_vel_tol_ms=c['box_v'], a_min_override=c['a_min'],
        a_max_override=c['a_max'], e_max_target=c['e_max_target'],
        e_max_sat=c['e_max_sat'], de_max=c['de_max'], da_max_m=c['da_max'],
        di_max_rad=c['di_max'], di_min_rad=c['di_min'],
        di_phase_mode=int(c['di_phase']), j2_mode=int(c['j2']),
        nav_j2_mode=int(c['j2']),
        i_target_min_rad=c['i_t_min'], i_target_max_rad=c['i_t_max'],
        fuel_frac_min=c['fuel_min'], fuel_frac_max=c['fuel_max'])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True, help='the ROOT the run warm-starts from')
    p.add_argument('--out', required=True)
    p.add_argument('--cell', default='TIGHT_5k1')
    p.add_argument('--windows', type=int, default=1200,
                   help='number of horizon-length windows to keep')
    p.add_argument('--horizon', type=int, default=64,
                   help='MUST equal the trainer bptt_horizon')
    p.add_argument('--envs', type=int, default=64)
    p.add_argument('--seed', type=int, default=17)
    p.add_argument('--device', default='cpu')
    p.add_argument('--max-ticks', type=int, default=None,
                   help='nav_max_ticks for collection. MUST match the training '
                        'harness: the states are filter outputs, so a dataset '
                        'collected at a different tick cadence is a dataset '
                        'from a different observation distribution.')
    p.add_argument('--filter-impl', default='py', choices=('py', 'c'))
    args = p.parse_args()

    torch.manual_seed(args.seed)
    kw = cell_kwargs(args.cell, args.envs)
    if args.max_ticks is not None:
        kw['nav_max_ticks'] = args.max_ticks
    if args.filter_impl != 'py':
        kw['nav_filter_impl'] = args.filter_impl
    env = OrbitalNav(**kw)
    obs, _ = env.reset(seed=args.seed)

    # Same construction the evaluator uses: the checkpoints are state_dicts,
    # not pickled policies.
    policy = LSTMWrapper(env, Default(env))
    policy.load_state_dict(torch.load(args.ckpt, map_location=args.device,
                                      weights_only=True))
    policy.eval()
    torch.set_num_threads(1)

    n_env = args.envs
    horizon = args.horizon
    buf = np.zeros((n_env, horizon, env.single_observation_space.shape[0]),
                   dtype=np.float32)
    fill = 0
    windows = []
    lstm_h = lstm_c = None
    steps = 0

    # NOTE: each append is a BLOCK of n_env windows, so the loop counts
    # windows, not appends — the obvious `len(windows)` test over-collects
    # by a factor of n_env.
    while len(windows) * n_env < args.windows:
        o = torch.as_tensor(np.asarray(obs), dtype=torch.float32,
                            device=args.device)
        if lstm_h is None:
            lstm_h = torch.zeros(n_env, policy.hidden_size)
            lstm_c = torch.zeros(n_env, policy.hidden_size)
        state = dict(lstm_h=lstm_h, lstm_c=lstm_c)
        with torch.no_grad():
            logits, _ = policy.forward_eval(o, state)
            lstm_h, lstm_c = state.get('lstm_h'), state.get('lstm_c')
            if isinstance(logits, (list, tuple)):
                logits = logits[0]
            action, _, _ = pufferlib.pytorch.sample_logits(logits)

        buf[:, fill] = np.asarray(obs, dtype=np.float32)
        fill += 1
        if fill == horizon:
            # window boundary — exactly the trainer's cadence, episode
            # boundaries deliberately ignored (see module docstring)
            windows.append(buf.copy())
            fill = 0

        obs, _, _, _, _ = env.step(action.cpu().numpy().astype(np.int32))
        steps += 1

    env.close()
    out = np.concatenate(windows, axis=0)[:args.windows]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save({
        'obs': torch.from_numpy(out),
        'cell': args.cell,
        'ckpt': os.path.basename(args.ckpt),
        'horizon': horizon,
        'env_steps': steps * n_env,
        'note': 'windows cut every `horizon` env steps, NOT episode-aligned, '
                'to match pufferl.train()\'s zero-init BPTT convention',
    }, args.out)
    print(f'  wrote {args.out}')
    print(f'    windows {out.shape[0]}  shape {tuple(out.shape)}  '
          f'({out.shape[0] * horizon} states from {steps * n_env} env-steps)')
    print(f'    obs range [{out.min():.3f}, {out.max():.3f}]  '
          f'mean |obs| {np.abs(out).mean():.4f}')


if __name__ == '__main__':
    main()
