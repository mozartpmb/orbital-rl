"""Action / tau distribution of the T3 canonical policy — grounds the NAV-H
SPS projection (filter work per env.step() scales with tau, the sub-step count).

Read-only rollout of models/t3/seed42_L2_headline.pt at the headline config.
"""
import sys

import numpy as np
import torch

sys.path.insert(0, "/Users/pete/space_training/pufferlib")
from pufferlib.ocean.orbital.orbital import Orbital        # noqa: E402
from pufferlib.models import Default, LSTMWrapper          # noqa: E402

CKPT = "/Users/pete/space_training/models/t3/seed42_L2_headline.pt"
TAU = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 5, 30, 60, 1, 1, 1, 1, 180, 360, 1, 1])
T3 = dict(num_envs=64, num_debris_min=0, num_debris_max=0,
          e_max_target=0.05, e_max_sat=0.05, init_phase_gap_max=3.14159,
          valid_init_only=1, shaping_mode=1, shape_gamma=1.0,
          phase_gap_mode=1, phase_obs_mode=1, episode_cap_steps=3000,
          cap_terminal_reward=0.0)


def main(n_decisions=600):
    torch.set_num_threads(1)
    env = Orbital(**T3)
    pol = LSTMWrapper(env, Default(env))
    pol.load_state_dict(torch.load(CKPT, map_location='cpu', weights_only=True))
    pol.eval()
    obs, _ = env.reset(seed=123)
    N = env.num_agents
    st = {'lstm_h': torch.zeros(N, pol.hidden_size),
          'lstm_c': torch.zeros(N, pol.hidden_size)}
    counts = np.zeros(20, dtype=np.int64)
    eps = 0
    for _ in range(n_decisions):
        with torch.no_grad():
            logits, _ = pol.forward_eval(torch.from_numpy(obs).float(), st)
            a = torch.argmax(logits, dim=-1).flatten().numpy().astype(np.int32)
        np.add.at(counts, a, 1)
        obs, r, d, t, _ = env.step(a)
        eps += int(d.sum())
        if d.any():
            m = torch.from_numpy(d.astype(bool))
            st['lstm_h'][m] = 0.0
            st['lstm_c'][m] = 0.0
    env.close()
    tot = counts.sum()
    mean_tau = float((counts * TAU[:20]).sum() / tot)
    print(f"decisions {tot}, episodes finished {eps}")
    for i in np.argsort(-counts)[:8]:
        if counts[i]:
            print(f"  action {i:2d} (tau {TAU[i]:3d}): {counts[i]:7d} "
                  f"= {100*counts[i]/tot:5.1f}%")
    print(f"  MEAN TAU = {mean_tau:.2f} sub-steps/decision")
    print(f"  ticks/decision at 60 s nav cadence  = {mean_tau:.2f}")
    for cap in (4, 6, 8, 12, 20):
        eff = float((counts * np.minimum(TAU[:20], cap)).sum() / tot)
        print(f"  ticks/decision capped at {cap:3d}/decision = {eff:6.2f} "
              f"(worst nav gap {max(TAU[:20][counts>0])*60/cap/60:.0f} min)")


if __name__ == '__main__':
    main()
