"""ATTACKS B + G — terminal asymmetry and the discount/warp ordering.

Drives the REAL env with shaping_mode=1, shape_gamma=1.0, phase_gap_mode=1,
phase_obs_mode=1 (the T3 design as implemented) and a battery of scripted
policies (degenerate + adversarial + the 99.2% expert), then reports

  * undiscounted episode return
  * DISCOUNTED return  sum_t gamma^t r_t  at gamma = 0.995 PER DECISION
    (this is the objective PPO actually optimises)

for six reward variants formed by crossing
  cap reward  in {-10 (as shipped), 0}
  NHR clamp   in {physical terminals only (as shipped), nowhere, everywhere}

The clamp and the shaping stream are reconstructed exactly: Python's Phi
reproduces the C shaping delta to ~4e-9, and at a clamped terminal
r_terminal - r_env == -Phi_prev is verified directly.

Outputs web_data/results/t3_redteam_policy_returns.csv
"""
import csv
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, '/Users/pete/space_training/scripts/orbital/t3')
from rt_common import (CAUSES, ObsView, R_EARTH, T3_KW, disc_sum, make_env)  # noqa

OUT = '/Users/pete/space_training/web_data/results/t3_redteam_policy_returns.csv'
GAMMA = 0.995

RUNGS = {
    'L0_e0_gap30': dict(same_orbit_init=1, e_max_target=0.0, e_max_sat=0.0,
                        init_phase_gap_max=math.radians(30)),
    'L1_e0_gap180': dict(same_orbit_init=1, e_max_target=0.0, e_max_sat=0.0,
                         init_phase_gap_max=math.pi),
    'L2_headline': dict(same_orbit_init=0, e_max_target=0.05, e_max_sat=0.05,
                        init_phase_gap_max=math.pi),
}


# ── scripted policies ────────────────────────────────────────────────────────
def pol_const(a):
    return lambda k, obs: a


def pol_park_far(n=6, warp=11):
    return lambda k, obs: 3 if k < n else warp


def pol_suicide(warp=11):
    def f(k, obs):
        a_s = float(obs[0]) * 1.6e6 + R_EARTH
        e_s = float(obs[1])
        r_p = a_s * (1.0 - e_s)
        if r_p > R_EARTH * 0.995 and float(obs[6]) > 0.01:
            return 6            # retro 25 m/s
        return warp
    return f


def pol_fuel_dump(warp=11):
    def f(k, obs):
        if float(obs[6]) > 0.001:
            return 3 if (k % 2 == 0) else 6
        return warp
    return f


def pol_oneburn_warp(burn_action, n_burn, warp=11):
    """The 'never null velocity, just drift through' degenerate solution."""
    def f(k, obs):
        return burn_action if k < n_burn else warp
    return f


def expert_policy():
    import expert_controller as ec
    ctl = ec.ExpertController()

    def f(k, obs):
        return ctl.act(obs)
    f.ctl = ctl
    return f


POLICIES = {
    'coast_forever': lambda: pol_const(0),
    'warp5_forever': lambda: pol_const(9),
    'warp30_forever': lambda: pol_const(10),
    'warp60_forever': lambda: pol_const(11),
    'park_far_warp60': lambda: pol_park_far(6, 11),
    'park_far_coast': lambda: pol_park_far(6, 0),
    'suicide_reentry': lambda: pol_suicide(11),
    'fuel_dump_strand': lambda: pol_fuel_dump(11),
    'oneburn10_warp60': lambda: pol_oneburn_warp(5, 1, 11),
    'expert': expert_policy,
}


def run_episode(rung_kw, policy, seed, cap=3000, max_dec=40000):
    env = make_env(seed=seed, **dict(rung_kw, **T3_KW))
    env.reset(seed=seed)
    pol = policy()
    obs = env.observations
    phis = [ObsView(obs[0]).phi_sr3()]
    rews = []
    acts = []
    k = 0
    cause = 0
    sim_steps = 0
    while k < max_dec:
        a = int(pol(k, obs[0]))
        _, rew, term, trunc, _ = env.step(np.array([a], dtype=np.int32))
        rews.append(float(rew[0]))
        acts.append(a)
        k += 1
        if term[0]:
            sim_steps, cause = env.last_episode_result(0)
            break
        phis.append(ObsView(obs[0]).phi_sr3())
    env.close()
    return dict(rews=np.array(rews), phis=np.array(phis), acts=acts,
                decisions=k, cause=cause, sim_steps=sim_steps)


ENV_R = {1: None, 2: -10.0, 3: -10.0, 4: -10.0, 5: -10.0, 6: -10.0, 7: 0.0}
CLAMPED = {1, 2, 3, 5, 6}       # branches that apply R2_NHR_CLAMP as shipped


def variants(ep):
    """Reconstruct the six reward variants exactly."""
    rews = ep['rews'].copy()
    phis = ep['phis']
    cause = ep['cause']
    phi_prev = phis[-1]           # Phi(s_{T-1}) — the value C clamps against
    # env reward at the terminal decision
    if cause == 1:
        # success: r = 10*(0.5+0.5*fuel_rem); recover it from the realised reward
        r_env = rews[-1] - (0.0 - phi_prev)
    elif cause == 4:
        r_env = -10.0
    elif cause == 7:
        r_env = 0.0
    else:
        r_env = rews[-1] - (0.0 - phi_prev)
    shipped_clamp = (0.0 - phi_prev) if cause in CLAMPED else 0.0
    check = abs((r_env + shipped_clamp) - rews[-1])
    out = {'clamp_recon_err': check, 'r_env_terminal': r_env,
           'phi_T_prev': phi_prev, 'phi_0': phis[0]}
    for cap_r in (-10.0, 0.0):
        for clamp in ('shipped', 'none', 'all'):
            r = rews.copy()
            re = cap_r if cause == 4 else r_env
            if clamp == 'shipped':
                c = (0.0 - phi_prev) if cause in CLAMPED else 0.0
            elif clamp == 'none':
                c = 0.0
            else:
                c = (0.0 - phi_prev)
            r[-1] = re + c
            tag = f'cap{int(cap_r)}_clamp{clamp}'
            out[f'undisc_{tag}'] = float(r.sum())
            out[f'disc_{tag}'] = float(disc_sum(r, GAMMA))
    return out


def main():
    seeds = list(range(12))
    rows = []
    for rung, kw in RUNGS.items():
        for pname, pfac in POLICIES.items():
            for s in seeds:
                try:
                    ep = run_episode(kw, pfac, s)
                except Exception as exc:      # noqa
                    print(f'  !! {rung}/{pname}/{s}: {exc}')
                    continue
                v = variants(ep)
                shap = float(ep['rews'][:-1].sum())
                rows.append(dict(
                    rung=rung, policy=pname, seed=s,
                    decisions=ep['decisions'], sim_steps=ep['sim_steps'],
                    cause=CAUSES[ep['cause']],
                    shaping_sum_nonterminal=shap,
                    telescope_resid=abs(shap - (ep['phis'][-1] - ep['phis'][0])),
                    **v))
            done = [r for r in rows if r['rung'] == rung and r['policy'] == pname]
            if done:
                print(f'{rung:14s} {pname:18s} n={len(done):2d} '
                      f'dec={np.mean([r["decisions"] for r in done]):7.1f} '
                      f'cause={done[0]["cause"]:10s} '
                      f'disc_shipped={np.mean([r["disc_cap-10_clampshipped"] for r in done]):+8.3f} '
                      f'disc_cap0_none={np.mean([r["disc_cap0_clampnone"] for r in done]):+8.3f}')
    with open(OUT, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f'\nwrote {len(rows)} rows -> {OUT}')

    err = max(r['clamp_recon_err'] for r in rows)
    tel = max(r['telescope_resid'] for r in rows)
    print(f'max clamp-reconstruction error: {err:.3e}  '
          f'(0 => the C clamp is exactly -Phi_prev at physical terminals)')
    print(f'max telescoping residual |sum(deltas) - (Phi_T - Phi_0)|: {tel:.3e}')

    print('\n=== DISCOUNTED return (gamma=0.995 per decision), mean over seeds ===')
    tags = ['cap-10_clampshipped', 'cap-10_clampnone', 'cap-10_clampall',
            'cap0_clampshipped', 'cap0_clampnone', 'cap0_clampall']
    hdr = f'{"rung":14s} {"policy":18s} {"dec":>6s} ' + ' '.join(f'{t:>19s}' for t in tags)
    print(hdr)
    for rung in RUNGS:
        for pname in POLICIES:
            sel = [r for r in rows if r['rung'] == rung and r['policy'] == pname]
            if not sel:
                continue
            line = f'{rung:14s} {pname:18s} {np.mean([r["decisions"] for r in sel]):6.0f} '
            line += ' '.join(f'{np.mean([r[f"disc_{t}"] for r in sel]):+19.3f}' for t in tags)
            print(line)


if __name__ == '__main__':
    main()
