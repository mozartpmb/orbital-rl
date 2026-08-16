#!/usr/bin/env python3
"""T11 — which normalizer family can represent the generalist's task space?

THE DESIGN CONFLICT. GEN_MATRIX's read recommends warm-rooting at A3b-j2 (a
NARROW-normalizer lineage, obs_alt_scale_m = 1.6e6) and "restating the E-ladder
in narrow normalizers". But eccentricity and altitude are not independent in
this env: `c_reset` rejection-samples until perigee a(1-e) >= EARTH_KEEPOUT
(6.571e6 m), so a high e is only reachable at a high a. If the narrow
normalizer cannot represent the altitudes that high e requires, the
recommendation is unimplementable and the generalist must live in the WIDE
family instead.

obs[0] = (a - R_EARTH) / obs_alt_scale_m, written RAW (no clamp), against a
declared Box(-2, 2). So a family can "represent" a band iff |obs[0]| stays
inside the space the trainer was told about.

    narrow (1.6e6):  |obs0| <= 1 => a <= 7.971e6 ;  <= 2 => a <= 9.571e6
    wide   (8.0e6):  |obs0| <= 1 => a <= 14.371e6

This probe measures, for each candidate band x e_max setting: the REALIZED
eccentricity distribution, the infeasible-init (gave_up) mass, and |obs0| p99
in BOTH families. Reset-only; no policy, no training.

Run:
    PYTHONPATH=<worktree>/pufferlib python3 scripts/orbital/ext_recon/t11_family_probe.py
"""

import math
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
WT = os.path.abspath(os.path.join(_HERE, '..', '..', '..'))
sys.path.insert(0, os.path.join(WT, 'pufferlib'))

import pufferlib                                                     # noqa: E402
if not os.path.abspath(pufferlib.__file__).startswith(os.path.abspath(WT) + os.sep):
    raise SystemExit('REFUSING TO RUN: pufferlib is not the worktree build')
from pufferlib.ocean.orbital.orbital import Orbital                  # noqa: E402
import pufferlib.ocean.orbital.binding as _binding                   # noqa: E402

R_EARTH = 6.371e6
KEEPOUT = 6.571e6
NARROW, WIDE = 1.6e6, 8.0e6

BASE = dict(
    num_debris_min=0, num_debris_max=0, same_orbit_init=0,
    init_phase_gap_max=3.14159, valid_init_only=1,
    gave_up_action='terminate', max_valid_init_attempts=4096,
    lvlh_scale_m=6.371e6,
    rendezvous_radius_m=30000.0, rel_vel_tol_ms=50.0,
    shaping_mode=2, shape_w_lambda=1.0, shape_w_match=0.8166667,
    shape_dv_ref_ms=700.0, shape_gamma=1.0,
    phase_gap_mode=1, phase_obs_mode=1, cap_terminal_reward=0.0,
    dim3_mode=1, di_max_rad=0.017453, legacy_action_space=30,
    episode_cap_steps=3000,
)


def measure(a_min, a_max, e_max, scale, n=4096, de_max=-1.0, da_max=-1.0):
    """Realized (e_target, e_sat, a), |obs0|, and infeasible mass over n resets."""
    kw = dict(BASE, a_min_override=a_min, a_max_override=a_max,
              e_max_target=e_max, obs_alt_scale_m=scale)
    if de_max >= 0:
        kw['de_max'] = de_max
        kw['da_max_m'] = da_max
    else:
        kw['e_max_sat'] = e_max
    env = Orbital(num_envs=256, **kw)
    et, es, aa, o0, gu = [], [], [], [], []
    for k in range(n // 256):
        obs, _ = env.reset(seed=9000 + k)
        st = env.get_state()
        aa.append(st[:, 15].copy())            # target a
        et.append(st[:, 16].copy())            # target e
        es.append(st[:, 1].copy())             # chaser e
        o0.append(np.abs(np.asarray(obs)[:, 0]).copy())
        for i in range(256):
            _, gv = _binding.vec_get_episode_init_info(env.c_envs, i)
            gu.append(int(gv))
    et = np.concatenate(et); es = np.concatenate(es)
    aa = np.concatenate(aa); o0 = np.concatenate(o0)
    return dict(
        e_t_p50=float(np.median(et)), e_t_p90=float(np.percentile(et, 90)),
        e_t_p99=float(np.percentile(et, 99)), e_t_max=float(et.max()),
        e_s_p90=float(np.percentile(es, 90)),
        a_max_real=float(aa.max()),
        obs0_p99=float(np.percentile(o0, 99)), obs0_max=float(o0.max()),
        gave_up=float(np.mean(gu)) if gu else float('nan'),
        peri_min_km=float((aa * (1 - et)).min() / 1e3),
    )


def main():
    print('=== T11 family probe: what e is reachable inside each normalizer? ===')
    print(f'    obs[0] = (a - R_EARTH)/scale, RAW against a declared Box(-2,2).')
    print(f'    perigee keepout {KEEPOUT/1e6:.3f}e6 m => e_max(a) = 1 - {KEEPOUT/1e6:.3f}/a\n')

    print('  Analytic ceiling: the largest e a band can hold at all')
    for name, amax in (('X3 band top      7.171e6', 7.171e6),
                       ('E1 band top      7.871e6', 7.871e6),
                       ('narrow |obs0|<=1 7.971e6', 7.971e6),
                       ('narrow |obs0|<=2 9.571e6', 9.571e6),
                       ('E2 band top      9.871e6', 9.871e6),
                       ('E3 band top     14.371e6', 14.371e6)):
        print(f'    {name}  e_max_at_apogee {1 - KEEPOUT/amax:6.4f}   '
              f'|obs0| narrow {(amax-R_EARTH)/NARROW:5.2f}  wide {(amax-R_EARTH)/WIDE:5.2f}')

    print('\n  Measured, 4096 resets per row. "setting" is the KNOB, "realized" is the RESULT.')
    print(f'  {"band (a_min-a_max)/1e6":26s} {"e_max set":>9s} {"e_t p50":>8s} '
          f'{"p90":>7s} {"p99":>7s} {"max":>7s} {"|obs0|p99 N":>12s} {"W":>6s} {"gave_up":>8s}')
    rows = []
    for label, a_min, a_max, de, da in (
            ('X3     6.671-7.171', 6.671e6, 7.171e6, -1.0, -1.0),
            ('E1     6.671-7.871', 6.671e6, 7.871e6, 0.05, 300e3),
            ('N1max  6.671-7.971', 6.671e6, 7.971e6, 0.05, 300e3),
            ('N1hi   7.200-7.971', 7.200e6, 7.971e6, 0.05, 300e3),
            ('N2max  6.671-9.571', 6.671e6, 9.571e6, 0.065, 450e3),
            ('E2     6.671-9.871', 6.671e6, 9.871e6, 0.065, 450e3),
            ('E3    6.671-14.371', 6.671e6, 14.371e6, 0.08, 600e3)):
        for e_max in (0.10, 0.30):
            r = measure(a_min, a_max, e_max, NARROW, de_max=de, da_max=da)
            rows.append((label, e_max, r))
            print(f'  {label:26s} {e_max:9.2f} {r["e_t_p50"]:8.4f} '
                  f'{r["e_t_p90"]:7.4f} {r["e_t_p99"]:7.4f} {r["e_t_max"]:7.4f} '
                  f'{r["obs0_p99"]:12.2f} {(r["a_max_real"]-R_EARTH)/WIDE:6.2f} '
                  f'{r["gave_up"]:8.4f}')

    print('\n  THE ANSWER. Highest realized e_t p90 reachable with narrow |obs0| p99 <= 1:')
    ok = [(lab, em, r) for lab, em, r in rows if r['obs0_p99'] <= 1.0]
    if ok:
        best = max(ok, key=lambda t: t[2]['e_t_p90'])
        print(f'    {best[0]}  e_max={best[1]}  ->  realized e_t p90 = '
              f'{best[2]["e_t_p90"]:.4f}, p99 {best[2]["e_t_p99"]:.4f}, '
              f'max {best[2]["e_t_max"]:.4f}  (|obs0| p99 {best[2]["obs0_p99"]:.2f})')
    print('    E-ladder reference (measured previously, WIDE normalizers):')
    print('      E1 realized e_t p90 0.081 | E2 0.166 | E3 0.257')
    return 0


if __name__ == '__main__':
    sys.exit(main())
