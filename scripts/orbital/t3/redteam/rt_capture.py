"""ATTACK D — capture window vs warp granularity, and L0 trivialisation.

Three questions, all measured against the real env:

 D1  Does a warp THROUGH the success box trigger success?  (check_termination
     runs inside the sub-step loop in c_step, so it should.)
 D2  At what residual |da| does a drift-through pass satisfy BOTH the 30 km
     position box and the 50 m/s relative-velocity box?  Is the velocity box
     ever the binding constraint in the co-orbital regime?
 D3  Is "one small correctly-signed burn, then warp-60 forever, never null
     anything" a winning policy?  For which phase gaps?  This is the
     degenerate solution that would trivialise the L0 rung -- and raising the
     cap from 2000 to 3000 sub-steps widens the regime where it works.

Outputs web_data/results/t3_redteam_capture.csv
"""
import csv
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rt_common import CAUSES, MU, ObsView, R_EARTH, T3_KW, make_env  # noqa

OUT = '/Users/pete/space_training/web_data/results/t3_redteam_capture.csv'


def cart(a, e, theta, omega):
    p = a * (1.0 - e * e)
    r = p / (1.0 + e * math.cos(theta))
    h = math.sqrt(MU * p)
    xp, yp = r * math.cos(theta), r * math.sin(theta)
    vxp = -(MU / h) * math.sin(theta)
    vyp = (MU / h) * (e + math.cos(theta))
    c, s = math.cos(omega), math.sin(omega)
    return (c * xp - s * yp, s * xp + c * yp,
            c * vxp - s * vyp, s * vxp + c * vyp)


def rel_state(o: ObsView):
    sx, sy, svx, svy = cart(o.a_s, o.e_s, o.theta_s, o.omega_s)
    # target theta from its mean anomaly
    from rt_common import true_to_mean  # noqa
    M = o.M_t
    E = M
    for _ in range(8):
        E -= (E - o.e_t * math.sin(E) - M) / (1.0 - o.e_t * math.cos(E))
    th = 2.0 * math.atan2(math.sqrt(1 + o.e_t) * math.sin(E / 2),
                          math.sqrt(1 - o.e_t) * math.cos(E / 2))
    tx, ty, tvx, tvy = cart(o.a_t, o.e_t, th, o.omega_t)
    return math.hypot(sx - tx, sy - ty), math.hypot(svx - tvx, svy - tvy)


def run(gap_deg, burn_action, n_burn, cap, e=0.0, seed=0, warp=11,
        track_box=False, same_orbit=1):
    kw = dict(T3_KW)
    kw['episode_cap_steps'] = cap
    env = make_env(seed=seed, same_orbit_init=same_orbit,
                   e_max_target=e, e_max_sat=e,
                   e_target_fixed=(e if e > 0 else -1.0),
                   e_sat_fixed=(e if e > 0 else -1.0),
                   phase_gap_fixed=math.radians(gap_deg), **kw)
    env.reset(seed=seed)
    obs = env.observations
    o0 = ObsView(obs[0])
    a0 = o0.a_s
    k = 0
    min_d, v_at_min = 1e30, None
    dwell = 0
    da_final = 0.0
    while k < 200000:
        act = burn_action if k < n_burn else warp
        env.step(np.array([act], dtype=np.int32))
        k += 1
        if env.terminals[0]:
            steps, cause = env.last_episode_result(0)
            break
        o = ObsView(obs[0])
        da_final = o.a_s - o.a_t
        if track_box:
            d, v = rel_state(o)
            if d < min_d:
                min_d, v_at_min = d, v
            if d < 30000.0:
                dwell += 1
    return dict(gap_deg=gap_deg, e=e, seed=seed, cap=cap,
                burn_action=burn_action, n_burn=n_burn,
                decisions=k, sim_steps=steps, cause=CAUSES[cause],
                success=int(cause == 1), a0_km=(a0 - R_EARTH) / 1e3,
                da_final_km=da_final / 1e3,
                min_d_km=(min_d / 1e3 if track_box else float('nan')),
                relv_at_min=(v_at_min if track_box else float('nan')),
                dwell_decisions=dwell)


def main():
    rows = []

    print('=== D1/D3: single retro burn then warp-60 forever (never null) ===')
    print('gap  burn dv | cap=2000 succ | cap=3000 succ | da_km  | note')
    # retro actions: 12/13 are +-1, 14/15 +-2, 1/4 +-5, 2/5 +-10, 3/6 +-25
    RETRO = {1: 13, 2: 15, 5: 4, 10: 5, 25: 6}
    for gap in [10, 20, 30, 45, 60, 75, 90, 120, 150, 179]:
        for dv, act in sorted(RETRO.items()):
            for cap in (2000, 3000):
                s = 0
                for seed in range(8):
                    r = run(gap, act, 1, cap, seed=seed, track_box=False)
                    r['probe'] = 'oneburn_warp60'
                    rows.append(r)
                    s += r['success']
                if cap == 2000:
                    s2000 = s
                else:
                    s3000 = s
            print(f'{gap:4d}  {dv:4.0f} m/s |     {s2000}/8      |     {s3000}/8      |'
                  f' {rows[-1]["da_final_km"]:+7.2f} |')

    print('\n=== D2: capture geometry — min distance and relative velocity ===')
    print('  burn dv   da_km    min_d_km   relv_at_min   dwell(dec)  success')
    for dv, act in sorted(RETRO.items()):
        r = run(30, act, 1, 3000, seed=3, track_box=True)
        r['probe'] = 'geometry'
        rows.append(r)
        print(f'  {dv:5.0f}  {r["da_final_km"]:+8.2f} {r["min_d_km"]:10.3f} '
              f'{r["relv_at_min"]:12.3f} {r["dwell_decisions"]:9d}   {r["cause"]}')
    # multi-quantum: 2x, 3x of 10 m/s to push |da| past 30 km
    for n in (2, 3, 4):
        r = run(30, 5, n, 3000, seed=3, track_box=True)
        r['probe'] = 'geometry'
        rows.append(r)
        print(f'  {10*n:5d}  {r["da_final_km"]:+8.2f} {r["min_d_km"]:10.3f} '
              f'{r["relv_at_min"]:12.3f} {r["dwell_decisions"]:9d}   {r["cause"]}')

    print('\n=== D1 explicit: does a warp-60 decision that STRADDLES the box '
          'still register success? ===')
    n_w60, n_w5, n_coast = 0, 0, 0
    for seed in range(24):
        for warp, name in ((11, 'warp60'), (9, 'warp5'), (0, 'coast')):
            r = run(30, 5, 1, 3000, seed=seed, warp=warp)
            r['probe'] = f'warpgran_{name}'
            rows.append(r)
            if r['success']:
                if warp == 11:
                    n_w60 += 1
                elif warp == 9:
                    n_w5 += 1
                else:
                    n_coast += 1
    print(f'  30 deg gap, one 10 m/s retro burn, 24 seeds: '
          f'warp60 {n_w60}/24   warp5 {n_w5}/24   coast {n_coast}/24')
    print('  (equal counts => the sub-step termination check makes warp '
          'granularity irrelevant to capture)')

    with open(OUT, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f'\nwrote {len(rows)} rows -> {OUT}')

    # analytic cross-check
    a = 7.0e6
    n = math.sqrt(MU / a ** 3)
    for cap in (2000, 3000):
        T = cap * 60.0
        g_max = 1.5 * n * (30000.0 / a) * T
        print(f'analytic: cap={cap} -> largest gap closable with |da|<30 km '
              f'(single burn, no closing burn) = {math.degrees(g_max):.1f} deg')


if __name__ == '__main__':
    main()
