"""EXT-ANGLES — baseline maneuver statistics of the TRUTH-trained policy vs separation.

The campaign hypothesis is that a policy trained on ESTIMATED state under
bearings-only sensing will learn observability-inducing maneuvers.  Any such
claim needs a matched-state control: what does the current truth-trained policy
already do at the same separations?  "Excess burns" is only meaningful against
that baseline, and the baseline has never been measured.

This probe measures, per decision epoch, the action taken and the true chaser-
target separation, then bins by separation.  The bins are chosen from the
observability map (`ext_angles_observability_profile.py`):

    rho < 10 km      drift-only bearings CRLB is 10-500x the separation itself
                     unless the arc exceeds ~4 orbits.  A burn buys 100x.
    10-50 km         same regime, weaker.
    50-500 km        drift-only reaches ~2% of rho within one orbit.
    0.5-5 Mm         drift-only already sub-1%.
    > 5 Mm           range is essentially free from the arc curvature alone.

The decisive number for the campaign is DWELL TIME BELOW 50 km WITHOUT A BURN:
if the policy crosses the observability-poor region faster than the drift-only
arc needs, then bearings-only nav there is genuinely starved and an
observability maneuver is required rather than merely helpful.

Env is only read, never modified.  Runs the shipped T3/T4 checkpoints.

Run:  python3 ext_angles_policy_actions.py [--episodes N] [--ckpt t3|tb5|wl4]
Writes web_data/results/ext_angles_policy_actions.csv
       web_data/results/ext_angles_policy_dwell.csv
"""

import argparse
import csv
import math
import os
import sys

import numpy as np
import torch

ROOT = "/Users/pete/space_training"
sys.path.insert(0, f"{ROOT}/scripts/orbital/nav")
sys.path.insert(0, f"{ROOT}/pufferlib")

from pufferlib.ocean.orbital.orbital import Orbital          # noqa: E402
from pufferlib.models import Default, LSTMWrapper            # noqa: E402
import orbital_math as om                                    # noqa: E402

RESULTS = f"{ROOT}/web_data/results"
OUT_ACT = f"{RESULTS}/ext_angles_policy_actions.csv"
OUT_DWELL = f"{RESULTS}/ext_angles_policy_dwell.csv"

# Discrete-20 tau table (sub-steps of 60 s per decision). Actions 0-8 are
# coast/burns at tau=1, 9-11 are the warps, 12-15 fine burns, 16-17 the long
# warps, 18-19 the radial 1 m/s burns added in T4.
TAU20 = (1, 1, 1, 1, 1, 1, 1, 1, 1, 5, 30, 60, 1, 1, 1, 1, 180, 360, 1, 1)
BURN_ACTIONS = set(range(1, 9)) | set(range(12, 16)) | {18, 19}
WARP_ACTIONS = {9, 10, 11, 16, 17}

CONFIGS = {
    't3': dict(
        ckpt=f"{ROOT}/models/t3/seed42_L2_headline.pt",
        kwargs=dict(num_envs=1, num_debris_min=0, num_debris_max=0,
                    e_max_target=0.05, e_max_sat=0.05,
                    init_phase_gap_max=3.14159, valid_init_only=1,
                    gave_up_action="terminate", max_valid_init_attempts=4096,
                    obs_alt_scale_m=1.6e6, lvlh_scale_m=6.371e6,
                    shaping_mode=1, shape_gamma=1.0, phase_gap_mode=1,
                    phase_obs_mode=1, episode_cap_steps=3000,
                    cap_terminal_reward=0.0)),
    'tb5': dict(
        ckpt=f"{ROOT}/models/t3/seed42_TB5_box5k1.pt",
        kwargs=dict(num_envs=1, num_debris_min=0, num_debris_max=0,
                    e_max_target=0.05, e_max_sat=0.05,
                    init_phase_gap_max=3.14159, valid_init_only=1,
                    gave_up_action="terminate", max_valid_init_attempts=4096,
                    obs_alt_scale_m=1.6e6, lvlh_scale_m=6.371e6,
                    shaping_mode=1, shape_gamma=1.0, phase_gap_mode=1,
                    phase_obs_mode=1, episode_cap_steps=3000,
                    cap_terminal_reward=0.0,
                    rendezvous_radius_m=5000.0, rel_vel_tol_ms=1.0,
                    legacy_action_space=20)),
}

BINS = [(0.0, 10e3, '<10km'), (10e3, 50e3, '10-50km'),
        (50e3, 500e3, '50-500km'), (500e3, 5e6, '0.5-5Mm'),
        (5e6, float('inf'), '>5Mm')]


def bin_of(rho):
    for lo, hi, lab in BINS:
        if lo <= rho < hi:
            return lab
    return BINS[-1][2]


def separation_from_obs(o, obs_alt_scale_m, lvlh_scale_m):
    """True separation, from the LVLH block of the observation (slots 33-34)."""
    return math.hypot(float(o[33]) * lvlh_scale_m, float(o[34]) * lvlh_scale_m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--episodes', type=int, default=40)
    ap.add_argument('--ckpt', default='t3', choices=sorted(CONFIGS))
    ap.add_argument('--seed', type=int, default=123)
    args = ap.parse_args()

    cfg = CONFIGS[args.ckpt]
    if not os.path.exists(cfg['ckpt']):
        alt = sorted(x for x in os.listdir(f"{ROOT}/models/t3")
                     if x.endswith('.pt'))
        print(f"checkpoint {cfg['ckpt']} missing; available: {alt}")
        return
    env = Orbital(**cfg['kwargs'])
    policy = LSTMWrapper(env, Default(env))
    policy.load_state_dict(torch.load(cfg['ckpt'], map_location='cpu',
                                      weights_only=True))
    policy.eval()

    lv = cfg['kwargs']['lvlh_scale_m']
    oa = cfg['kwargs']['obs_alt_scale_m']

    obs, _ = env.reset(seed=args.seed)
    state = {'lstm_h': torch.zeros(1, policy.hidden_size),
             'lstm_c': torch.zeros(1, policy.hidden_size)}

    rows = []          # per-decision
    dwell = []         # per-episode dwell statistics
    ep = 0
    cur = dict(n_dec=0, sim_s=0.0, per_bin={}, burns_in_bin={},
               last_burn_t={}, gap_max={}, rho_min=float('inf'))
    for lab in (b[2] for b in BINS):
        cur['per_bin'][lab] = 0.0
        cur['burns_in_bin'][lab] = 0
        cur['gap_max'][lab] = 0.0
    t_since_burn = 0.0

    max_dec = 400000
    for _ in range(max_dec):
        # forward_eval is the convention eval_relnav.py uses: (1, 1, obs_dim)
        # and a state dict that the call mutates in place.
        pol_obs = np.array(obs[0], dtype=np.float32, copy=True)
        with torch.no_grad():
            logits, _ = policy.forward_eval(
                torch.from_numpy(pol_obs).float().unsqueeze(0).unsqueeze(0),
                state)
            a = int(torch.argmax(logits, dim=-1).item())
        rho = separation_from_obs(obs[0], oa, lv)
        lab = bin_of(rho)
        tau = TAU20[a] if a < len(TAU20) else 1
        dt = tau * 60.0
        is_burn = a in BURN_ACTIONS

        rows.append(dict(ckpt=args.ckpt, ep=ep, dec=cur['n_dec'],
                         t_min=cur['sim_s'] / 60.0, rho_km=rho / 1e3,
                         rho_bin=lab, action=a, tau=tau,
                         is_burn=int(is_burn),
                         is_warp=int(a in WARP_ACTIONS)))
        cur['per_bin'][lab] += dt
        cur['rho_min'] = min(cur['rho_min'], rho)
        if is_burn:
            cur['burns_in_bin'][lab] += 1
            t_since_burn = 0.0
        else:
            t_since_burn += dt
            cur['gap_max'][lab] = max(cur['gap_max'][lab], t_since_burn)
        cur['n_dec'] += 1
        cur['sim_s'] += dt

        obs, rew, term, trunc, info = env.step(np.array([a], dtype=np.int32))
        done = bool(np.asarray(term).reshape(-1)[0]) or \
            bool(np.asarray(trunc).reshape(-1)[0])
        if done:
            d = dict(ckpt=args.ckpt, ep=ep, n_dec=cur['n_dec'],
                     dur_min=cur['sim_s'] / 60.0,
                     rho_min_km=cur['rho_min'] / 1e3)
            for lab in (b[2] for b in BINS):
                d[f'min_{lab}'] = cur['per_bin'][lab] / 60.0
                d[f'burns_{lab}'] = cur['burns_in_bin'][lab]
                d[f'maxgap_min_{lab}'] = cur['gap_max'][lab] / 60.0
            dwell.append(d)
            ep += 1
            if ep >= args.episodes:
                break
            state = {'lstm_h': torch.zeros(1, policy.hidden_size),
                     'lstm_c': torch.zeros(1, policy.hidden_size)}
            cur = dict(n_dec=0, sim_s=0.0, per_bin={}, burns_in_bin={},
                       last_burn_t={}, gap_max={}, rho_min=float('inf'))
            for lab in (b[2] for b in BINS):
                cur['per_bin'][lab] = 0.0
                cur['burns_in_bin'][lab] = 0
                cur['gap_max'][lab] = 0.0
            t_since_burn = 0.0

    with open(OUT_ACT, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(OUT_DWELL, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(dwell[0].keys()))
        w.writeheader()
        w.writerows(dwell)

    # ── summary ──
    print(f"checkpoint {args.ckpt}: {ep} episodes, {len(rows)} decisions\n")
    print(f"{'rho bin':<12} {'decisions':>10} {'%dec':>7} {'sim_min':>10} "
          f"{'%time':>7} {'burn%':>7} {'warp%':>7} {'coast%':>7}")
    tot_dec = len(rows)
    tot_t = sum(r['tau'] for r in rows) * 60.0
    for lo, hi, lab in BINS:
        sel = [r for r in rows if r['rho_bin'] == lab]
        if not sel:
            print(f"{lab:<12} {0:>10}")
            continue
        t = sum(r['tau'] for r in sel) * 60.0
        nb = sum(r['is_burn'] for r in sel)
        nw = sum(r['is_warp'] for r in sel)
        print(f"{lab:<12} {len(sel):>10} {100*len(sel)/tot_dec:7.1f} "
              f"{t/60:>10.0f} {100*t/tot_t:7.1f} {100*nb/len(sel):7.1f} "
              f"{100*nw/len(sel):7.1f} "
              f"{100*(len(sel)-nb-nw)/len(sel):7.1f}")

    print(f"\n{'rho bin':<12} {'dwell_min p50':>14} {'p90':>8} {'max':>8} "
          f"{'burns p50':>10} {'max coast-gap p50':>18} {'p90':>8}")
    for lo, hi, lab in BINS:
        dv = np.array([d[f'min_{lab}'] for d in dwell])
        bv = np.array([d[f'burns_{lab}'] for d in dwell])
        gv = np.array([d[f'maxgap_min_{lab}'] for d in dwell])
        print(f"{lab:<12} {np.median(dv):14.1f} {np.percentile(dv,90):8.1f} "
              f"{dv.max():8.1f} {np.median(bv):10.0f} "
              f"{np.median(gv):18.1f} {np.percentile(gv,90):8.1f}")

    rmin = np.array([d['rho_min_km'] for d in dwell])
    print(f"\nclosest approach km: p50 {np.median(rmin):.2f}  "
          f"p90 {np.percentile(rmin,90):.2f}  max {rmin.max():.2f}")
    print(f"episode duration min: p50 {np.median([d['dur_min'] for d in dwell]):.0f}")
    print(f"\nwrote {OUT_ACT} ({len(rows)} rows)\nwrote {OUT_DWELL} ({len(dwell)} rows)")


if __name__ == '__main__':
    main()
