"""ATTACK C — gradient adequacy and dead zones of the S-R3 potential.

C1  Per-decision shaping magnitude on the drift leg, per warp granularity,
    against the PPO entropy floor (ent_coef = 0.01 in orbital.ini).
C2  Distribution of Dv_match at reset per rung: what fraction of episodes
    START inside the min(1, .) saturation, where the orbit-match term has
    exactly zero gradient?
C3  Does the saturated region cover states the policy must traverse EARLY
    (opening a drift orbit on top of an a-transfer)?  Measured along the
    scripted expert's own trajectories.

Outputs web_data/results/t3_redteam_gradient.csv
"""
import csv
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, '/Users/pete/space_training/scripts/orbital/t3')
from rt_common import (DV_REF, MU, ObsView, R_EARTH, T3_KW, W_LAMBDA,  # noqa
                       W_MATCH, make_env)

OUT = '/Users/pete/space_training/web_data/results/t3_redteam_gradient.csv'
ENT_COEF = 0.01

RUNGS = {
    'L0_e0_gap30': dict(same_orbit_init=1, e_max_target=0.0, e_max_sat=0.0,
                        init_phase_gap_max=math.radians(30)),
    'L1_e0_gap180': dict(same_orbit_init=1, e_max_target=0.0, e_max_sat=0.0,
                         init_phase_gap_max=math.pi),
    'L2_headline': dict(same_orbit_init=0, e_max_target=0.05, e_max_sat=0.05,
                        init_phase_gap_max=math.pi),
    'L3_e015_wide': dict(same_orbit_init=0, e_max_target=0.15, e_max_sat=0.15,
                         init_phase_gap_max=math.pi, valid_init_only=1,
                         a_min_override=7.2e6, a_max_override=8.4e6),
}


def main():
    rows = []

    print('=== C1: per-decision shaping on the drift leg vs the entropy floor ===')
    print(f'{"action":10s} {"tau(min)":>9s} {"dlam/dec(deg)":>14s} '
          f'{"dPhi/dec":>10s} {"vs ent_coef 0.01":>17s}')
    a = 7.0e6
    n = math.sqrt(MU / a ** 3)
    for da_km in (18.6, 133.0, 340.0):
        print(f'  -- drift orbit |da| = {da_km:.1f} km --')
        rate = 1.5 * n * (da_km * 1e3) / a          # rad/s
        for name, tau in (('coast', 1), ('warp5', 5), ('warp30', 30), ('warp60', 60)):
            dl = rate * tau * 60.0
            dphi = W_LAMBDA * dl / math.pi
            print(f'{name:10s} {tau:9d} {math.degrees(dl):14.4f} {dphi:10.5f} '
                  f'{dphi / ENT_COEF:16.2f}x')
            rows.append(dict(probe='C1', da_km=da_km, action=name, tau=tau,
                             dlam_per_dec_deg=math.degrees(dl),
                             dphi_per_dec=dphi, ratio_to_ent_coef=dphi / ENT_COEF))

    print('\n=== C2: Dv_match at reset — saturation fraction (min(1,.) dead zone) ===')
    print(f'{"rung":16s} {"n":>5s} {"mean":>8s} {"median":>8s} {"p90":>8s} '
          f'{"max":>8s} {"frac >= 300 m/s":>16s}')
    N = 1500
    for rung, kw in RUNGS.items():
        env = make_env(seed=3, **dict(kw, **T3_KW))
        vals, phis, dlams = [], [], []
        for i in range(N):
            env.reset(seed=5000 + i)
            o = ObsView(env.observations[0])
            vals.append(o.dv_match())
            phis.append(o.phi_sr3())
            dlams.append(abs(o.dlam))
        env.close()
        v = np.array(vals)
        sat = float((v >= DV_REF).mean())
        print(f'{rung:16s} {N:5d} {v.mean():8.1f} {np.median(v):8.1f} '
              f'{np.percentile(v, 90):8.1f} {v.max():8.1f} {sat:16.3f}')
        rows.append(dict(probe='C2', rung=rung, n=N, mean=v.mean(),
                         median=float(np.median(v)),
                         p90=float(np.percentile(v, 90)), max=v.max(),
                         frac_saturated=sat,
                         mean_phi=float(np.mean(phis)),
                         mean_abs_dlam_deg=float(np.degrees(dlams).mean())))

    print('\n=== C3: saturation ALONG the expert trajectory (L2 headline) ===')
    import expert_controller as ec
    ec.MAX_STEPS = 3000          # runtime patch: match episode_cap_steps=3000
    for rung in ('L2_headline', 'L3_e015_wide'):
        kw = RUNGS[rung]
        fr_sat, fr_sat_first20, n_ep, succ = [], [], 0, 0
        for seed in range(8):
            env = make_env(seed=seed, **dict(kw, **T3_KW))
            env.reset(seed=seed)
            ctl = ec.ExpertController()
            sats = []
            for k in range(4000):
                o = ObsView(env.observations[0])
                sats.append(1.0 if o.dv_match() >= DV_REF else 0.0)
                act = ctl.act(env.observations[0])
                env.step(np.array([act], dtype=np.int32))
                if env.terminals[0]:
                    _, cause = env.last_episode_result(0)
                    succ += int(cause == 1)
                    break
            env.close()
            n_ep += 1
            sats = np.array(sats)
            fr_sat.append(sats.mean())
            fr_sat_first20.append(sats[:max(1, len(sats) // 5)].mean())
        print(f'{rung:16s} expert success {succ}/{n_ep}  '
              f'saturated on {np.mean(fr_sat):.3f} of decisions '
              f'({np.mean(fr_sat_first20):.3f} over the first 20% of the episode)')
        rows.append(dict(probe='C3', rung=rung, n=n_ep,
                         frac_saturated=float(np.mean(fr_sat)),
                         frac_saturated_first20=float(np.mean(fr_sat_first20)),
                         expert_success=succ / max(1, n_ep)))

    print('\n=== C4: sensitivity of Phi to a 1 km change in da (unsaturated) ===')
    at = 7.0e6
    dphi_dkm = W_MATCH * (0.5 * math.sqrt(MU / at) / at) * 1e3 / DV_REF
    print(f'  dPhi/d|da| = {dphi_dkm:.3e} per km  '
          f'-> opening a 133 km drift orbit costs {133 * dphi_dkm:.4f} potential, '
          f'refunded on close; drift leg pays up to {W_LAMBDA:.2f}. '
          f'ratio {W_LAMBDA / (133 * dphi_dkm):.1f} : 1 in favour')
    rows.append(dict(probe='C4', dphi_per_km=dphi_dkm,
                     entry_cost_133km=133 * dphi_dkm,
                     ratio=W_LAMBDA / (133 * dphi_dkm)))

    with open(OUT, 'w', newline='') as f:
        keys = sorted({k for r in rows for k in r})
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f'\nwrote {len(rows)} rows -> {OUT}')


if __name__ == '__main__':
    main()
