"""ATTACK A — is mean longitude lambda = M + omega continuous across an impulse?

The S-R3 audit asserts "lambda is continuous through a burn by construction".
That is FALSE in general: lambda = u - EOC where u = theta + omega is the (truly
continuous) true longitude and EOC = theta - M is the equation of centre, which
depends on (e, theta) and therefore jumps whenever a burn changes the
eccentricity vector. This probe measures the jump exactly.

Method (no control run needed, exact): within c_step the impulse is applied
instantaneously, then the state is propagated one 60 s sub-step at the NEW
semi-major axis. So

    lambda_after = lambda_burninstant + n(a_after) * DT
    jump         = wrap(lambda_after - n(a_after)*DT - lambda_before)

Everything is read back from the env's own observation vector.

Outputs web_data/results/t3_redteam_lambda_jump.csv
"""
import csv
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rt_common import (ACTION_DV, MU, ObsView, T3_KW, W_LAMBDA, disc_sum,  # noqa
                       make_env, wrap_pi)

OUT = '/Users/pete/space_training/web_data/results/t3_redteam_lambda_jump.csv'

BURN_ACTIONS = [1, 2, 3, 4, 5, 6, 7, 8, 12, 13, 14, 15]
CONFIGS = [
    ('e0_sameorbit', dict(same_orbit_init=1, e_max_target=0.0, e_max_sat=0.0,
                          init_phase_gap_max=math.pi)),
    ('e001', dict(same_orbit_init=0, e_target_fixed=0.01, e_sat_fixed=0.01,
                  init_phase_gap_max=math.pi)),
    ('e005_headline', dict(same_orbit_init=0, e_max_target=0.05, e_max_sat=0.05,
                           init_phase_gap_max=math.pi)),
    ('e010', dict(same_orbit_init=0, e_target_fixed=0.10, e_sat_fixed=0.10,
                  a_min_override=7.0e6, a_max_override=7.3e6,
                  init_phase_gap_max=math.pi)),
]
N_SEEDS = 24
PRE_STEPS = [0, 7, 14, 21, 28, 35, 42, 49, 56, 63, 70, 77]  # sample burn phase


def measure(cfg_kw, seed, pre, action):
    env = make_env(seed=seed, **dict(cfg_kw, **T3_KW))
    env.reset(seed=seed)
    a0 = np.array([0], dtype=np.int32)
    for _ in range(pre):
        env.step(a0)
        if env.terminals[0]:
            env.close()
            return None
    before = ObsView(env.observations[0])
    env.step(np.array([action], dtype=np.int32))
    if env.terminals[0]:
        env.close()
        return None
    after = ObsView(env.observations[0])
    env.close()
    n_after = math.sqrt(MU / after.a_s ** 3)
    jump = wrap_pi(after.lam_s - n_after * 60.0 - before.lam_s)
    # continuous reference: true longitude u = theta + omega must be exactly
    # invariant across the impulse (position does not move)
    u_before = before.theta_s + before.omega_s
    # propagate u is not linear, so only compare the equation-of-centre change
    eoc_before = wrap_pi(before.theta_s - before.M_s)
    eoc_after = wrap_pi(after.theta_s - after.M_s)
    return dict(
        seed=seed, pre=pre, action=action,
        dv_pro=ACTION_DV[action][0], dv_rad=ACTION_DV[action][1],
        dv_mag=math.hypot(*ACTION_DV[action]),
        e_before=before.e_s, e_after=after.e_s,
        theta_before_deg=math.degrees(before.theta_s),
        da_m=after.a_s - before.a_s,
        jump_rad=jump, jump_deg=math.degrees(jump),
        eoc_before_deg=math.degrees(eoc_before),
        eoc_after_deg=math.degrees(eoc_after),
        dlam_before=before.dlam, dlam_after=after.dlam,
        phi_before=before.phi_sr3(), phi_after=after.phi_sr3(),
        u_before_deg=math.degrees(wrap_pi(u_before)),
    )


def main():
    rows = []
    for cfg_name, cfg_kw in CONFIGS:
        for seed in range(N_SEEDS):
            for pre in PRE_STEPS:
                for act in BURN_ACTIONS:
                    r = measure(cfg_kw, seed, pre, act)
                    if r is None:
                        continue
                    r['config'] = cfg_name
                    rows.append(r)
    with open(OUT, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f'wrote {len(rows)} rows -> {OUT}')

    import collections
    print('\n=== |lambda jump| across an impulse (deg) ===')
    print(f'{"config":16s} {"act":>4s} {"dv":>6s} {"n":>5s} '
          f'{"med":>8s} {"p90":>8s} {"max":>8s} {"med|dPhi_jump|":>14s} '
          f'{"max rad/(m/s)":>14s}')
    agg = collections.defaultdict(list)
    for r in rows:
        agg[(r['config'], r['action'])].append(r)
    for k in sorted(agg):
        v = agg[k]
        j = np.abs([x['jump_deg'] for x in v])
        dv = v[0]['dv_mag']
        dphi = W_LAMBDA * np.abs([x['jump_rad'] for x in v]) / math.pi
        per = np.abs([x['jump_rad'] for x in v]) / max(dv, 1e-9)
        print(f'{k[0]:16s} {k[1]:4d} {dv:6.1f} {len(v):5d} '
              f'{np.median(j):8.4f} {np.percentile(j, 90):8.4f} {j.max():8.4f} '
              f'{np.median(dphi):14.6f} {per.max():14.3e}')

    # Headline attack number: max shaping obtainable per m/s of dv from jumps,
    # vs honest drift.
    print('\n=== farming arithmetic ===')
    allj = np.abs([r['jump_rad'] for r in rows])
    alldv = np.array([max(r['dv_mag'], 1e-9) for r in rows])
    eff = allj / alldv
    order = np.argsort(-eff)
    best = order[0]
    print(f'best single burn: {rows[best]["config"]} act={rows[best]["action"]} '
          f'dv={rows[best]["dv_mag"]} jump={rows[best]["jump_deg"]:.4f} deg '
          f'-> {W_LAMBDA*allj[best]/math.pi:.5f} shaping for '
          f'{alldv[best]:.1f} m/s')
    dv_budget = 478.0
    print(f'upper bound on total farmable shaping if EVERY m/s of the '
          f'{dv_budget:.0f} m/s budget were spent at the best observed '
          f'rate: {dv_budget*eff[best]/math.pi*W_LAMBDA:.4f} '
          f'(and this is an over-estimate: Phi telescopes, so a closed cycle '
          f'nets exactly 0)')


if __name__ == '__main__':
    main()
