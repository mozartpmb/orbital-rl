"""ATTACK G (time observability) + ATTACK H (long-way-round wrap valley).

G  With warps, elapsed sim time decouples from decision count and the cap is
   invisible in the observation.  At L1 (same_orbit_init=1, e=0) coasting
   changes NOTHING physical: dlam, a_s, a_t, e, the LVLH block are all exactly
   invariant.  The only channels that move are the two pure rotation-phase
   pairs obs[2,3] = sin/cos(theta_sat) and obs[15,16] (= sin/cos(lambda_tgt)
   under phase_obs_mode 1) -- i.e. a clock modulo the ~95 min orbit period,
   not an absolute clock.  Hand the *same* observation to the 99.2% scripted
   expert at different absolute times and the achievable value swings from
   success to forced timeout.

H  When the perigee floor forbids lowering, the only route is to RAISE and lap
   the long way round: |dlam| grows through pi before it shrinks, so Phi digs a
   transient valley of depth W_lambda*(pi-|dlam_0|)/pi.  Measured on identical
   inits, tracked only up to the FIRST dlam = 0 crossing (the arrival), so the
   two routes are compared over the same task.

Outputs web_data/results/t3_redteam_time_longway.csv
"""
import csv
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, '/Users/pete/space_training/scripts/orbital/t3')
from rt_common import (CAUSES, ObsView, R_EARTH, T3_KW, disc_sum, make_env)  # noqa

OUT = '/Users/pete/space_training/web_data/results/t3_redteam_time_longway.csv'
GAMMA = 0.995

# Everything except obs[2,3] and obs[15,16] (pure rotation-phase channels)
TASK_IDX = [0, 1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14] + list(range(17, 38))


def main():
    import expert_controller as ec
    ec.MAX_STEPS = 3000
    rows = []

    print('=== G: identical task observation, different deadline ===')
    print('L1 (e=0, same-orbit, 90 deg gap), cap=3000, phase_obs_mode=0 so the '
          'scripted expert can drive')
    print(f'{"coast delay":>12s} {"max|d obs| task ch":>19s} {"expert result":>14s} '
          f'{"sim_steps":>10s} {"disc return":>12s}')
    kw = dict(T3_KW)
    kw['episode_cap_steps'] = 3000
    kw['phase_obs_mode'] = 0
    base = None
    for delay in (0, 300, 600, 900, 1200, 1500, 1800):
        env = make_env(seed=1, same_orbit_init=1, e_max_target=0.0,
                       e_max_sat=0.0, phase_gap_fixed=math.radians(90), **kw)
        env.reset(seed=1)
        rews = []
        for _ in range(delay):
            env.step(np.array([0], dtype=np.int32))
            rews.append(float(env.rewards[0]))
        handover = env.observations[0].copy()
        if base is None:
            base = handover
            d = 0.0
        else:
            d = float(np.max(np.abs(handover[TASK_IDX] - base[TASK_IDX])))
        ctl = ec.ExpertController()
        ctl.steps = delay          # tell the expert the truth about the clock
        for _ in range(20000):
            act = ctl.act(env.observations[0])
            env.step(np.array([act], dtype=np.int32))
            rews.append(float(env.rewards[0]))
            if env.terminals[0]:
                steps, cause = env.last_episode_result(0)
                break
        env.close()
        dr = disc_sum(rews, GAMMA)
        print(f'{delay:12d} {d:19.2e} {CAUSES[cause]:>14s} {steps:10d} {dr:+12.3f}')
        rows.append(dict(probe='G', delay=delay, max_task_obs_delta=d,
                         cause=CAUSES[cause], sim_steps=steps,
                         success=int(cause == 1), disc=dr))
    ok = [r for r in rows if r['probe'] == 'G' and r['success']]
    bad = [r for r in rows if r['probe'] == 'G' and not r['success']]
    if ok and bad:
        print(f'  -> the SAME task observation is worth {np.mean([r["disc"] for r in ok]):+.2f} '
              f'or {np.mean([r["disc"] for r in bad]):+.2f} depending on an '
              f'unobservable clock: a {np.mean([r["disc"] for r in ok]) - np.mean([r["disc"] for r in bad]):.2f} '
              f'irreducible value error')

    print('\n=== H: long-way-round (raise & lap) vs direct (lower & catch) ===')
    print('   tracked to the FIRST dlam=0 crossing')
    print(f'{"gap":>5s} {"route":>8s} {"dec to arrive":>14s} {"Phi valley":>11s} '
          f'{"undisc shaping":>15s} {"disc shaping":>13s} {"predicted valley":>17s}')
    kw = dict(T3_KW)
    kw['episode_cap_steps'] = 3000
    for gap in (30.0, 90.0, 150.0):
        for route, a_act in (('direct', 6), ('longway', 3)):
            env = make_env(seed=2, same_orbit_init=1, e_max_target=0.0,
                           e_max_sat=0.0, phase_gap_fixed=math.radians(gap), **kw)
            env.reset(seed=2)
            phis = [ObsView(env.observations[0]).phi_sr3()]
            rews = []
            arrived = None
            for k in range(200000):
                act = a_act if k < 3 else 11
                env.step(np.array([act], dtype=np.int32))
                rews.append(float(env.rewards[0]))
                if env.terminals[0]:
                    break
                o = ObsView(env.observations[0])
                phis.append(o.phi_sr3())
                if k > 3 and abs(o.dlam) < math.radians(4.0):
                    arrived = k + 1
                    break
            env.close()
            phis = np.array(phis)
            valley = float(phis.min() - phis[0])
            sh = np.array(rews)
            pred = -(math.pi - math.radians(gap)) / math.pi if route == 'longway' else 0.0
            rows.append(dict(probe='H_route', gap_deg=gap, route=route,
                             decisions_to_arrive=arrived, phi_valley=valley,
                             undisc_shaping=float(sh.sum()),
                             disc_shaping=float(disc_sum(sh, GAMMA)),
                             predicted_valley=pred))
            print(f'{gap:5.0f} {route:>8s} {str(arrived):>14s} {valley:11.4f} '
                  f'{sh.sum():+15.4f} {disc_sum(sh, GAMMA):+13.4f} {pred:17.4f}')

    with open(OUT, 'w', newline='') as f:
        keys = sorted({k for r in rows for k in r})
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f'\nwrote {len(rows)} rows -> {OUT}')


if __name__ == '__main__':
    main()
