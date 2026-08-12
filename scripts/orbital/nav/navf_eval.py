#!/usr/bin/env python3
"""NAV-F per-arm evaluation: success at both boxes + the detection metrics.

Every arm is evaluated under BEARINGS-ONLY with the real batch acquisition, at
both the TB5 (5 km / 1 m/s) and TB4 (5 km / 2 m/s) boxes, 200 episodes, held-out
seed 123. Because the C env is seeded identically per episode index, episode i
is the SAME scenario across arms — which is what makes the paired delta-v
comparison (NAV-F §3.3-4) valid.

Usage:
    navf_eval.py --ckpt PATH --label T-BO [--boxes tb5,tb4] [--eps 200]
                 [--sigma-channel] [--block-fine-below-m 10000]
"""

import argparse
import json
import math
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
WT = os.path.abspath(os.path.join(_HERE, '..', '..', '..'))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(WT, 'pufferlib'))

import eval_relnav as ER                                     # noqa: E402
import navf_metrics as NM                                    # noqa: E402
from pufferlib.ocean.orbital.orbital import Orbital          # noqa: E402
from verify_extnav import rollout, show                      # noqa: E402

BASE = dict(
    num_envs=1, num_debris_min=0, num_debris_max=0,
    e_max_target=0.05, e_max_sat=0.05, same_orbit_init=0,
    init_phase_gap_max=3.14159, valid_init_only=1,
    gave_up_action='terminate', max_valid_init_attempts=4096,
    obs_alt_scale_m=1.6e6, lvlh_scale_m=6.371e6,
    shaping_mode=1, shape_gamma=1.0, phase_gap_mode=1, phase_obs_mode=1,
    episode_cap_steps=3000, cap_terminal_reward=0.0,
    legacy_action_space=20,
)
BOXES = {
    'tb5': dict(rendezvous_radius_m=5000.0, rel_vel_tol_ms=1.0),
    'tb4': dict(rendezvous_radius_m=5000.0, rel_vel_tol_ms=2.0),
}


def box_kwargs(box):
    kw = dict(BASE)
    kw.update(BOXES[box])
    return kw


def run_box(ckpt, label, box, eps, seed, sigma_channel=0):
    kw = box_kwargs(box)
    ER.ENV_KWARGS = kw
    ER.PHASE_OBS_MODE = 1
    ER.CKPT = ckpt
    ER.SENSOR_DT = 60.0
    rec = []
    b = ER.run_bo(eps, noise_scale=1.0, seed=seed, label=f'{label}/{box}',
                  collect=rec, sigma_channel=sigma_channel)
    ER._bo_report(b)
    beh = ER.behaviour_report(b, quiet=True)
    met = NM.analyse(rec, f'{label}/{box}')

    cause = b['cause']
    out = {
        'label': label, 'box': box,
        'native_rate': float(b['success'].mean()),
        'causes': ', '.join(f'{ER.CAUSE_NAMES[c]}={int((cause == c).sum())}'
                            for c in range(8) if (cause == c).any()),
        'timeout_rate': float((cause == 4).mean()),     # pre-registered confound
        'never_acquire': int((b['blind_dec'] == b['dec']).sum()),
        'acq_frac': float(b['flat_acq'].mean()),
        'n_diverge': int(b['n_diverge']),
        'total_dv_med': float(np.median(b['total_dv'])),
        'ep_dv': b['total_dv'].tolist(),
        'ep_success': b['success'].tolist(),
    }
    out.update({f'beh_{k}': v for k, v in beh.items()})
    out.update({f'met_{k}': v for k, v in met.items()
                if not isinstance(v, dict)})
    if 'action_mix_close' in met:
        out['met_action_mix_close'] = met['action_mix_close']
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True)
    p.add_argument('--label', required=True)
    p.add_argument('--boxes', default='tb5,tb4')
    p.add_argument('--eps', type=int, default=200)
    p.add_argument('--seed', type=int, default=123)
    p.add_argument('--sigma-channel', action='store_true')
    p.add_argument('--json-out', default=None)
    a = p.parse_args()
    torch.set_num_threads(1)

    print(f'===== NAV-F arm {a.label} =====')
    print(f'  ckpt {a.ckpt}')
    res = {'label': a.label, 'ckpt': a.ckpt, 'boxes': {}}
    for box in a.boxes.split(','):
        box = box.strip()
        print(f'\n-- {box.upper()} '
              f'({BOXES[box]["rendezvous_radius_m"]/1e3:.0f} km / '
              f'{BOXES[box]["rel_vel_tol_ms"]:.0f} m/s), bearings-only, '
              f'real acquisition --')
        res['boxes'][box] = run_box(a.ckpt, a.label, box, a.eps, a.seed,
                                    int(a.sigma_channel))
        # truth control at the same box
        t = rollout(Orbital(**box_kwargs(box)), a.ckpt, a.eps, a.seed,
                    f'{a.label}/{box} truth', verbose=False)
        show(t)
        res['boxes'][box]['truth_rate'] = t['rate']
        res['boxes'][box]['truth_causes'] = t['cause_str']

    print('\nSUMMARY ' + json.dumps(
        {k: (v if k != 'boxes' else
             {b: {kk: vv for kk, vv in d.items() if kk not in ('ep_dv', 'ep_success')}
              for b, d in v.items()}) for k, v in res.items()}, sort_keys=True))
    if a.json_out:
        with open(a.json_out, 'w') as f:
            json.dump(res, f, indent=2, sort_keys=True)


if __name__ == '__main__':
    main()
