"""N3D-C probe 4 — ticks-per-decision for the 3D (Discrete-30) lineage.

NAV-H section 2.4 sized the whole filter-in-the-loop budget on the canonical 2D
policy's mean tau = 32.50 sub-steps/decision. The 3D lineage has ten extra burn
rows (all tau=1) and a different reward, so its action mix — and therefore the
number of 60 s filter ticks a 3D-nav wrapper must run inside one env.step() — is
an empirical question. Greedy rollout of the shipped X3 checkpoint at its own
training config.

Output: web_data/results/n3d_tau30_action_mix.csv
"""
import csv
import os
import sys

import numpy as np
import torch

sys.path.insert(0, '/Users/pete/space_training/pufferlib')
from pufferlib.ocean.orbital.orbital import Orbital  # noqa: E402
from pufferlib.models import Default, LSTMWrapper  # noqa: E402

OUT = '/Users/pete/space_training/web_data/results'
CKPT = '/Users/pete/space_training/models/t3/seed42_X3_3d_di1deg.pt'
TAU = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 5, 30, 60, 1, 1, 1, 1, 180, 360,
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
DV = np.array([0., 5., 10., 25., 5., 10., 25., 10., 10., 0., 0., 0., 1., 1.,
               2., 2., 0., 0., 1., 1., 1., 1., 10., 10., 25., 25.,
               np.hypot(25., 25.), np.hypot(25., 25.),
               np.hypot(25., 25.), np.hypot(25., 25.)])

X3 = dict(num_debris_min=0, num_debris_max=0, dim3_mode=1,
          legacy_action_space=30, e_max_target=0.05, e_max_sat=0.05,
          same_orbit_init=0, di_max_rad=0.017453, a_min_override=6.871e6,
          a_max_override=7.171e6, valid_init_only=1, init_phase_gap_max=3.14159,
          shaping_mode=2, shape_gamma=1.0, shape_dv_ref_ms=700.0,
          phase_gap_mode=1, phase_obs_mode=1, episode_cap_steps=3000,
          cap_terminal_reward=0.0, gave_up_action='terminate',
          max_valid_init_attempts=4096)


def main(episodes=15, seed=123):
    env = Orbital(num_envs=1, **X3)
    p = LSTMWrapper(env, Default(env))
    p.load_state_dict(torch.load(CKPT, map_location='cpu', weights_only=True))
    p.eval()
    torch.manual_seed(seed); torch.set_num_threads(1)
    obs, _ = env.reset(seed=seed)
    st = {'lstm_h': torch.zeros(1, p.hidden_size),
          'lstm_c': torch.zeros(1, p.hidden_size)}
    acts, n_done, causes = [], 0, []
    while n_done < episodes:
        with torch.no_grad():
            lg, _ = p.forward_eval(torch.from_numpy(np.asarray(obs)).float().unsqueeze(0), st)
            a = int(torch.argmax(lg, -1).item())
        acts.append(a)
        obs, _, term, _, _ = env.step(np.array([a], dtype=np.int32))
        if term[0]:
            n_done += 1
            causes.append(env.last_episode_result(0)[1])
            st = {'lstm_h': torch.zeros(1, p.hidden_size),
                  'lstm_c': torch.zeros(1, p.hidden_size)}
    env.close()
    acts = np.array(acts)
    cnt = np.bincount(acts, minlength=30)
    mean_tau = float(TAU[acts].mean())
    rows = [dict(action=int(i), tau=int(TAU[i]), dv_ms=float(DV[i]),
                 count=int(cnt[i]), share=float(cnt[i] / len(acts)))
            for i in range(30) if cnt[i]]
    rows.append(dict(action=-1, tau=0, dv_ms=0.0, count=len(acts),
                     share=mean_tau))   # share column carries mean tau
    print(f"episodes {episodes}  decisions {len(acts)}  "
          f"success {sum(1 for c in causes if c == 1)}/{episodes}")
    print(f"MEAN TAU = {mean_tau:.2f} sub-steps/decision "
          f"(2D canonical: 32.50, NAV-H 2.3)")
    for r in sorted(rows[:-1], key=lambda r: -r['count'])[:12]:
        print(f"  action {r['action']:2d} tau {r['tau']:3d} "
              f"dv {r['dv_ms']:6.2f}  {r['share']:6.2%}")
    plane = sum(r['count'] for r in rows[:-1] if 20 <= r['action'] <= 29)
    print(f"  plane/combined rows 20-29: {plane/len(acts):.2%}")
    os.makedirs(OUT, exist_ok=True)
    p_ = os.path.join(OUT, 'n3d_tau30_action_mix.csv')
    with open(p_, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"wrote {p_}")


if __name__ == '__main__':
    main()
