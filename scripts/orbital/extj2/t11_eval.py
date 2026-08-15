#!/usr/bin/env python3
"""T11 evaluator — per-cell scoring with the FUEL-EFFICIENCY AUDIT built in.

The user's question is not only "does it succeed" but "how efficiently" — so
the Dv audit is part of the evaluator from the start rather than a retrofit.
Per cell it reports:

    dv_used           Tsiolkovsky from the tank actually consumed
    dv_direct_ref     the cell's own linearised direct-burn cost at reset,
                      0.5*v_c*sqrt(da_rel^2+|de|^2) + v_c*|dh| — the same
                      estimate obs[28] carries
    dv_budget         the EPISODE's sampled budget, -VE*ln(1-f)
    efficiency        dv_used / dv_direct_ref   (< 1 means it beat the
                      direct-burn reference, which is what drift-and-wait does)
    budget_used_frac  dv_used / dv_budget       (scarcity response)

`--cell all` runs the live mixture; `--cell <name>` pins one cell with the
mixture off, which is how the per-cell floors and the trained rows are scored.

ALL CLAIMS ARE IN MEAN ELEMENTS under j2_mode=1, and every plane number in the
W1 cell is a NODE-DOMINANT gap, not a general plane error.
"""

import argparse
import hashlib
import json
import math
import os
import sys
import time

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
WT = os.path.abspath(os.path.join(_HERE, '..', '..', '..'))
sys.path.insert(0, os.path.join(WT, 'pufferlib'))
sys.path.insert(0, _HERE)

import pufferlib                                                     # noqa: E402
if not os.path.abspath(pufferlib.__file__).startswith(os.path.abspath(WT) + os.sep):
    raise SystemExit('REFUSING TO RUN: pufferlib is not the worktree build')
from pufferlib.ocean.orbital_nav.orbital_nav import OrbitalNav       # noqa: E402
from pufferlib.models import Default, LSTMWrapper                    # noqa: E402
from pufferlib.ocean.orbital import t11_cells as T                   # noqa: E402

MU = 3.986004418e14
VE = 300.0 * 9.80665
DRY = 850.0
CAUSES = ['none', 'success', 'collision', 'escape', 'safety_cap',
          'stranded', 'hyperbolic', 'gave_up']


def env_kwargs(cell, nav_mode, fuel_fixed=None):
    kw = dict(
        num_debris_min=0, num_debris_max=0, same_orbit_init=0,
        init_phase_gap_max=3.14159, valid_init_only=1,
        gave_up_action='terminate', max_valid_init_attempts=4096,
        obs_alt_scale_m=T.OBS_ALT_SCALE, lvlh_scale_m=T.LVLH_SCALE,
        shaping_mode=2, shape_w_lambda=1.0, shape_w_match=0.8166667,
        shape_dv_ref_ms=700.0, shape_gamma=1.0,
        phase_gap_mode=1, phase_obs_mode=1, cap_terminal_reward=0.0,
        dim3_mode=1, j2_mode=1, nav_j2_mode=1, lvlh_frame_mode=1,
        raan_target_sample=0, legacy_action_space=31,
        nav_sensor_dt=60.0, nav_noise_mult=1.0, nav_acq_min_sec=2700.0,
        nav_acq_gate=0.20, nav_acq_mode='crlb_online', nav_max_ticks=0,
    )
    if cell == 'all':
        kw.update(t11_mixture=1)
    else:
        c = dict(T.CELLS)[cell]
        kw.update(
            cell_mixture_mode=0, episode_cap_steps=int(c['cap']),
            rendezvous_radius_m=c['box_r'], rel_vel_tol_ms=c['box_v'],
            a_min_override=c['a_min'], a_max_override=c['a_max'],
            e_max_target=c['e_max_target'], e_max_sat=c['e_max_sat'],
            de_max=c['de_max'], da_max_m=c['da_max'],
            di_max_rad=c['di_max'], di_min_rad=c['di_min'],
            di_phase_mode=int(c['di_phase']), j2_mode=int(c['j2']),
            nav_j2_mode=int(c['j2']),
            i_target_min_rad=c['i_t_min'], i_target_max_rad=c['i_t_max'],
            fuel_frac_min=c['fuel_min'], fuel_frac_max=c['fuel_max'])
    if fuel_fixed is not None:
        kw['fuel_frac_min'] = fuel_fixed
        kw['fuel_frac_max'] = fuel_fixed + 1e-9
    return kw


def direct_ref(st):
    """The cell's own linearised direct-burn cost at reset (obs[28]'s estimate)."""
    a_s, a_t = st[0], st[15]
    hs, ht = st[5:8], st[20:23]
    es, et = st[30:33], st[33:36]
    v_c = math.sqrt(MU / a_t)
    de = float(np.linalg.norm(es - et))
    dh = float(np.linalg.norm(hs - ht))
    return 0.5 * v_c * math.hypot((a_s - a_t) / a_t, de) + v_c * dh


def run(args):
    kw = env_kwargs(args.cell, args.nav_mode, args.fuel_fixed)
    env = OrbitalNav(num_envs=1, nav_mode=args.nav_mode, **kw)
    env._acq_real = (args.nav_mode == 'bearings_only')
    policy = LSTMWrapper(env, Default(env))
    policy.load_state_dict(torch.load(args.ckpt, map_location='cpu',
                                      weights_only=True))
    policy.eval()
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)

    obs, _ = env.reset(seed=args.seed)
    st_hidden = {'lstm_h': torch.zeros(1, policy.hidden_size),
                 'lstm_c': torch.zeros(1, policy.hidden_size)}
    ec = env.episode_cells()
    cur = dict(cell=int(ec[0, 0]), cap=float(ec[0, 1]), fuel=float(ec[0, 2]),
               ref=direct_ref(env.get_state()[0]))
    eps, acts = [], []
    n_done, t0 = 0, time.time()
    while n_done < args.episodes:
        with torch.no_grad():
            logits, _ = policy.forward_eval(
                torch.from_numpy(np.asarray(obs)).float().unsqueeze(0), st_hidden)
            a = int(torch.argmax(logits, dim=-1).item())
        acts.append(a)
        f_before = float(env.get_state()[0][14])
        obs, _, term, _, _ = env.step(np.array([a], dtype=np.int32))
        if term[0]:
            steps, cause = env.last_episode_result(0)
            m_tot = DRY / max(1.0 - f_before, 1e-12)
            fuel0 = DRY * cur['fuel'] / (1.0 - cur['fuel'])
            cur.update(cause=int(cause), steps=int(steps),
                       dv_used=VE * math.log((DRY + fuel0) / max(m_tot, DRY + 1e-9)),
                       budget=-VE * math.log(1.0 - cur['fuel']))
            eps.append(cur)
            n_done += 1
            ec = env.episode_cells()
            cur = dict(cell=int(ec[0, 0]), cap=float(ec[0, 1]),
                       fuel=float(ec[0, 2]), ref=direct_ref(env.get_state()[0]))
            st_hidden = {'lstm_h': torch.zeros(1, policy.hidden_size),
                         'lstm_c': torch.zeros(1, policy.hidden_size)}

    causes = np.array([e['cause'] for e in eps])
    n_gave = int((causes == 7).sum())
    n_valid = len(causes) - n_gave
    succ = int((causes == 1).sum())
    win = [e for e in eps if e['cause'] == 1]

    def med(k, src=win):
        v = [e[k] for e in src]
        return float(np.median(v)) if v else float('nan')

    res = dict(
        label=args.label, ckpt=os.path.basename(args.ckpt), cell=args.cell,
        nav_mode=args.nav_mode, episodes=args.episodes, seed=args.seed,
        fuel_fixed=args.fuel_fixed,
        success=succ, n_valid=n_valid, gave_up=n_gave,
        rate=succ / max(n_valid, 1),
        causes={CAUSES[c]: int((causes == c).sum()) for c in range(8)
                if (causes == c).any()},
        md5=hashlib.md5(np.array(acts, dtype=np.int32).tobytes()).hexdigest(),
        dv_used_med=med('dv_used'), dv_ref_med=med('ref'),
        dv_budget_med=med('budget'),
        efficiency_med=float(np.median([e['dv_used'] / max(e['ref'], 1e-9)
                                        for e in win])) if win else float('nan'),
        budget_used_med=float(np.median([e['dv_used'] / max(e['budget'], 1e-9)
                                         for e in win])) if win else float('nan'),
        ep_len_med=med('steps'),
        state='MEAN ELEMENTS; W1 plane numbers are NODE-DOMINANT gaps',
        wall_s=time.time() - t0)

    print(f"  {args.label:26s} success {res['success']}/{res['n_valid']} = "
          f"{res['rate']:6.1%}   md5 {res['md5'][:12]}   {res['wall_s']:.0f}s")
    print(f"  {'':26s} causes: "
          + ', '.join(f'{k}={v}' for k, v in res['causes'].items()))
    print(f"  {'':26s} FUEL AUDIT (successes): dv used {res['dv_used_med']:.1f} m/s "
          f"vs direct ref {res['dv_ref_med']:.1f} "
          f"(efficiency {res['efficiency_med']:.3f}x), "
          f"budget {res['dv_budget_med']:.1f} "
          f"({res['budget_used_med']:.1%} of tank)")
    if args.cell == 'all':
        per = {}
        for e in eps:
            per.setdefault(e['cell'], []).append(e['cause'] == 1)
        line = '  '.join(f'{T.NAMES[k]}={np.mean(v):.0%}({len(v)})'
                         for k, v in sorted(per.items()))
        print(f"  {'':26s} per-cell: {line}")
        res['per_cell'] = {T.NAMES[k]: [float(np.mean(v)), len(v)]
                           for k, v in per.items()}
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        json.dump(res, open(args.out, 'w'), indent=2)
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True)
    p.add_argument('--cell', default='all',
                   choices=['all'] + list(T.NAMES))
    p.add_argument('--nav-mode', default='bearings_only',
                   choices=['truth', 'bearings_only', 'rb_ekf', 'recon'])
    p.add_argument('--episodes', type=int, default=200)
    p.add_argument('--seed', type=int, default=123)
    p.add_argument('--label', default='cell')
    p.add_argument('--fuel-fixed', type=float, default=None,
                   help='pin the budget instead of sampling — the lean/rich '
                        'pair is how the scarcity response is measured')
    p.add_argument('--out', default=None)
    args = p.parse_args()
    run(args)
    return 0


if __name__ == '__main__':
    sys.exit(main())
