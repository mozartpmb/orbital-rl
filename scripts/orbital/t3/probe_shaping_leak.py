"""T3 recon probe: measure the shaping-reward accounting of the CURRENT
gated potential under CORRECTED dynamics, using the real C env.

Question 1 (discount leak / "living reward"):
  A frozen do-nothing policy that runs to the safety cap terminates with
  r_env = -10 and NO Phi-clamp. So:
       episode_return + 10  ==  total accumulated shaping.
  If that quantity is POSITIVE, the shaping pays the agent to stall, and the
  payment rate scales with |Phi| i.e. with distance from the target orbit.

Question 2 (does shaping oppose the drift-orbit strategy?):
  Compare cumulative shaping over a fixed decision budget for
    - stall-in-place  (warp only, da ~ 0)
    - drift orbit     (burn prograde to raise da ~ 175 km, then warp)
  Under the correct dynamics the drift orbit is the ONLY way to change phase.

Usage:  python3 scripts/orbital/t3/probe_shaping_leak.py
Run from /Users/pete/space_training/pufferlib
"""
import os
import sys
import csv
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'pufferlib'))
from pufferlib.ocean.orbital.orbital import Orbital

HEADLINE = dict(
    num_envs=1,
    num_debris_min=0, num_debris_max=0,
    e_max_target=0.05, e_max_sat=0.05,
    init_phase_gap_max=3.14159265,
    valid_init_only=1,
    same_orbit_init=0,
)

CAUSES = ['none', 'success', 'collision', 'escape', 'safety_cap',
          'stranded', 'hyperbolic', 'gave_up']

# scripted policies: name -> callable(decision_index) -> action int
def make_policy(kind):
    if kind == 'coast':          # tau=1, 2000 decisions
        return lambda k: 0
    if kind == 'warp5':          # tau=5, 400 decisions
        return lambda k: 9
    if kind == 'warp60':         # tau=60, ~33 decisions
        return lambda k: 11
    if kind == 'drift_up':       # 4x +25 m/s prograde (da ~ +175 km) then warp 1hr
        return lambda k: 3 if k < 4 else 11
    if kind == 'drift_down':     # 4x -25 m/s retrograde then warp 1hr
        return lambda k: 6 if k < 4 else 11
    if kind == 'drift_up_big':   # 8x +25 m/s (da ~ +350 km) then warp 1hr
        return lambda k: 3 if k < 8 else 11
    raise ValueError(kind)


def run(kind, n_ep=200, seed=17):
    env = Orbital(seed=seed, **HEADLINE)
    pol = make_policy(kind)
    env.reset(seed=seed)
    rows = []
    ep_ret = 0.0
    k = 0
    ep = 0
    guard = 0
    while ep < n_ep and guard < n_ep * 3000:
        guard += 1
        a = np.array([pol(k)], dtype=np.int32)
        _, rew, term, trunc, _ = env.step(a)
        ep_ret += float(rew[0])
        k += 1
        if term[0] or trunc[0]:
            cause = int(env.last_terminal_cause) if hasattr(env, 'last_terminal_cause') else -1
            rows.append((ep, k, ep_ret, cause))
            ep_ret = 0.0
            k = 0
            ep += 1
    env.close()
    return rows


def cumulative_shaping_trace(kind, n_dec, seed=17):
    """Per-decision reward trace for one episode (no terminal), same init seed
    across policies so the comparison is paired."""
    env = Orbital(seed=seed, **HEADLINE)
    pol = make_policy(kind)
    env.reset(seed=seed)
    trace = []
    for k in range(n_dec):
        a = np.array([pol(k)], dtype=np.int32)
        _, rew, term, trunc, _ = env.step(a)
        trace.append(float(rew[0]))
        if term[0] or trunc[0]:
            break
    env.close()
    return np.array(trace)


if __name__ == '__main__':
    out_dir = '/Users/pete/space_training/web_data/results'
    os.makedirs(out_dir, exist_ok=True)

    print('=== Q1: do-nothing episode returns (return + 10 == total shaping) ===')
    q1 = []
    for kind in ['coast', 'warp5', 'warp60']:
        rows = run(kind, n_ep=120)
        rets = np.array([r[2] for r in rows])
        lens = np.array([r[1] for r in rows])
        causes = [CAUSES[r[3]] if 0 <= r[3] < len(CAUSES) else '?' for r in rows]
        cap = sum(1 for c in causes if c == 'safety_cap')
        shap = rets + 10.0
        print(f'{kind:10s} n={len(rets):4d} mean_return={rets.mean():+8.3f} '
              f'median={np.median(rets):+8.3f} mean_decisions={lens.mean():7.1f} '
              f'safety_cap_frac={cap/len(rets):.2f}  implied_total_shaping={shap.mean():+7.3f}')
        for c in sorted(set(causes)):
            print(f'    cause {c:12s} {sum(1 for x in causes if x==c)}')
        q1.append(dict(policy=kind, n=len(rets), mean_return=rets.mean(),
                       median_return=float(np.median(rets)),
                       mean_decisions=float(lens.mean()),
                       safety_cap_frac=cap/len(rets),
                       implied_total_shaping=float(shap.mean())))

    print()
    print('=== Q2: paired cumulative reward, stall vs drift orbit (same seeds) ===')
    q2 = []
    for seed in [1, 2, 3, 4, 5, 6, 7, 8]:
        line = {'seed': seed}
        for kind in ['warp60', 'drift_up', 'drift_down', 'drift_up_big']:
            tr = cumulative_shaping_trace(kind, n_dec=33, seed=seed)
            line[kind] = float(tr.sum())
        q2.append(line)
        print('seed %d: ' % seed + '  '.join(f'{k}={line[k]:+8.4f}'
              for k in ['warp60', 'drift_up', 'drift_down', 'drift_up_big']))
    for k in ['warp60', 'drift_up', 'drift_down', 'drift_up_big']:
        m = np.mean([r[k] for r in q2])
        print(f'MEAN {k:14s} {m:+8.4f}')

    with open(os.path.join(out_dir, 't3_shaping_leak_q1.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(q1[0].keys()))
        w.writeheader()
        w.writerows(q1)
    with open(os.path.join(out_dir, 't3_shaping_leak_q2.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(q2[0].keys()))
        w.writeheader()
        w.writerows(q2)
    print('\nwrote t3_shaping_leak_q1.csv / t3_shaping_leak_q2.csv to', out_dir)
