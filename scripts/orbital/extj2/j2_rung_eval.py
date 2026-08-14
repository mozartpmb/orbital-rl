#!/usr/bin/env python3
"""ext-j2 rung evaluator — one cell of the J2 campaign.

EVERY NUMBER THIS SCRIPT PRINTS IS IN MEAN ELEMENTS. Under `j2_mode=1` the
env's state IS the mean element set; no mean/osculating conversion is performed
anywhere, not at reset, not at a burn, not at the success test (j2_A_design
§1.4, J2_DESIGN_NOTES.md §3). A success here means "the mean elements are
inside the box", which at the 30 km / 50 m/s box is benign — the relative-state
error of the mean-element approximation is 83 m / 0.094 m/s per orbit at a 5 km
separation — but it is NOT an osculating-grade claim and must never be reported
as one.

Usage (the campaign calls it; it also runs standalone):
    python3 scripts/orbital/extj2/j2_rung_eval.py \
        --ckpt models/t3/seed42_X3_3d_di1deg.pt --label A1 \
        --j2-mode 1 --i-band 30,60 --lvlh-frame-mode 1 \
        --episodes 200 --seed 123 --out /tmp/a1.json
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

import pufferlib                                                     # noqa: E402
from pufferlib.ocean.orbital.orbital import Orbital                  # noqa: E402
from pufferlib.models import Default, LSTMWrapper                    # noqa: E402

_PL = os.path.abspath(pufferlib.__file__)
if not _PL.startswith(os.path.abspath(WT) + os.sep):
    raise SystemExit(
        f'REFUSING TO RUN: imported pufferlib is {_PL}, not under the checkout '
        f'that owns this script ({WT}). The `puffer` console script and bare '
        f'imports resolve elsewhere; run with PYTHONPATH={WT}/pufferlib.')
import pufferlib.ocean.orbital.binding as _b                          # noqa: E402
if not os.path.abspath(_b.__file__).startswith(os.path.abspath(WT) + os.sep):
    raise SystemExit(f'REFUSING TO RUN: binding .so is {_b.__file__}, not under {WT}.')

MU = 3.986004418e14
R_EARTH = 6.371e6
J2_COEF = 1.08262668e-3
J2_R_EQ = 6.378137e6
CAUSES = ['none', 'success', 'collision', 'escape', 'safety_cap',
          'stranded', 'hyperbolic', 'gave_up']

# The X3 rung, held fixed across every arm. Only the J2 knobs move.
BASE_KW = dict(
    num_debris_min=0, num_debris_max=0,
    e_max_target=0.05, e_max_sat=0.05, same_orbit_init=0,
    init_phase_gap_max=3.14159, valid_init_only=1,
    gave_up_action='terminate', max_valid_init_attempts=4096,
    obs_alt_scale_m=1.6e6, lvlh_scale_m=6.371e6,
    shaping_mode=2, shape_w_lambda=1.0, shape_w_match=0.8166667,
    shape_dv_ref_ms=700.0, shape_gamma=1.0,
    phase_gap_mode=1, phase_obs_mode=1,
    episode_cap_steps=3000, cap_terminal_reward=0.0,
    dim3_mode=1, di_max_rad=0.017453, legacy_action_space=30,
)


def raan_dot(a, e, inc):
    n = math.sqrt(MU / a ** 3)
    p = a * (1.0 - e * e)
    k = 1.5 * n * J2_COEF * (J2_R_EQ / p) ** 2
    return -k * math.cos(inc)


def elems(row):
    """(a, e, inc) for chaser and target out of get_state()."""
    def one(a, e, hx, hy, hz):
        return a, e, math.atan2(math.hypot(hx, hy), hz)
    return (one(row[0], row[1], row[5], row[6], row[7]),
            one(row[15], row[16], row[20], row[21], row[22]))


def plane_angle(row):
    hs = np.array(row[5:8]);  ht = np.array(row[20:23])
    return math.atan2(float(np.linalg.norm(np.cross(ht, hs))), float(np.dot(ht, hs)))


def build_env(args, episodes=1):
    kw = dict(BASE_KW)
    kw['j2_mode'] = args.j2_mode
    kw['lvlh_frame_mode'] = args.lvlh_frame_mode
    kw['raan_target_sample'] = args.raan_sample
    kw['rendezvous_radius_m'] = args.rendezvous_radius_m
    kw['rel_vel_tol_ms'] = args.rel_vel_tol_ms
    kw['shape_w_match'] = args.shape_w_match
    if args.i_band.lower() in ('off', 'none', ''):
        kw['i_target_min_rad'] = -1.0
        kw['i_target_max_rad'] = -1.0
        kw['i_target_rad'] = math.radians(args.i_fixed_deg)
    else:
        lo, hi = (float(x) for x in args.i_band.split(','))
        kw['i_target_min_rad'] = math.radians(lo)
        kw['i_target_max_rad'] = math.radians(hi)
    return Orbital(num_envs=episodes, **kw), kw


def run(args):
    env, kw = build_env(args)
    policy = LSTMWrapper(env, Default(env))
    policy.load_state_dict(torch.load(args.ckpt, map_location='cpu', weights_only=True))
    policy.eval()
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)

    obs, _ = env.reset(seed=args.seed)
    state = {'lstm_h': torch.zeros(1, policy.hidden_size),
             'lstm_c': torch.zeros(1, policy.hidden_size)}
    actions, causes, lengths, dvs = [], [], [], []
    it_deg, dom_dd, di0_deg, di1_deg = [], [], [], []

    st = env.get_state()[0]
    (sa, se, si), (ta, te, ti) = elems(st)
    it_deg.append(math.degrees(ti))
    dom_dd.append(abs(raan_dot(sa, se, si) - raan_dot(ta, te, ti)) * 86400 * 180 / math.pi)
    di0_deg.append(math.degrees(plane_angle(st)))

    n_done, k, t0 = 0, 0, time.time()
    fuel0 = float(st[14])
    while n_done < args.episodes:
        with torch.no_grad():
            logits, _ = policy.forward_eval(
                torch.from_numpy(np.asarray(obs)).float().unsqueeze(0), state)
            a = int(torch.argmax(logits, dim=-1).item())
        actions.append(a)
        obs, _, term, _, _ = env.step(np.array([a], dtype=np.int32))
        k += 1
        if term[0]:
            n_done += 1
            steps, cause = env.last_episode_result(0)
            causes.append(int(cause))
            lengths.append(steps)
            st = env.get_state()[0]
            # get_state() is post-reset, so terminal di_rel is read one step
            # early; record the RESET values for the next episode instead and
            # accept that di1 lags by one. Cheap and honest.
            di1_deg.append(di0_deg[-1])
            (sa, se, si), (ta, te, ti) = elems(st)
            it_deg.append(math.degrees(ti))
            dom_dd.append(abs(raan_dot(sa, se, si) - raan_dot(ta, te, ti))
                          * 86400 * 180 / math.pi)
            di0_deg.append(math.degrees(plane_angle(st)))
            state = {'lstm_h': torch.zeros(1, policy.hidden_size),
                     'lstm_c': torch.zeros(1, policy.hidden_size)}
            k = 0
            if args.verbose and n_done % 50 == 0:
                sr = sum(1 for c in causes if c == 1) / n_done
                print(f'    [{args.label}] {n_done}/{args.episodes} '
                      f'success {sr:.1%} ({time.time()-t0:.0f}s)', flush=True)
    env.close()

    causes = np.array(causes)
    n_gave = int((causes == 7).sum())
    n_valid = len(causes) - n_gave
    succ = int((causes == 1).sum())
    res = dict(
        label=args.label, ckpt=os.path.basename(args.ckpt),
        j2_mode=args.j2_mode, i_band=args.i_band,
        raan_sample=args.raan_sample, lvlh_frame_mode=args.lvlh_frame_mode,
        box=[args.rendezvous_radius_m, args.rel_vel_tol_ms],
        shape_w_match=args.shape_w_match,
        episodes=args.episodes, seed=args.seed,
        success=succ, n_valid=n_valid, gave_up=n_gave,
        rate=succ / max(n_valid, 1),
        causes={CAUSES[c]: int((causes == c).sum()) for c in range(8)
                if (causes == c).any()},
        md5=hashlib.md5(np.array(actions, dtype=np.int32).tobytes()).hexdigest(),
        n_decisions=len(actions),
        ep_len_median=float(np.median(lengths)),
        i_target_mean_deg=float(np.mean(it_deg)),
        dOmega_dot_median_deg_day=float(np.median(dom_dd)),
        di_rel_init_median_deg=float(np.median(di0_deg)),
        wall_s=time.time() - t0,
        state='MEAN ELEMENTS (no mean/osculating conversion anywhere)',
    )
    print(f"  {args.label:24s} success {res['success']}/{res['n_valid']} = "
          f"{res['rate']:6.1%}   gave_up {res['gave_up']}   "
          f"md5 {res['md5'][:12]}   {res['wall_s']:.0f}s")
    print(f"  {'':24s} causes: "
          + ', '.join(f'{k}={v}' for k, v in res['causes'].items()))
    print(f"  {'':24s} MEAN-ELEMENT claim. i_t mean {res['i_target_mean_deg']:.2f} deg, "
          f"|dOmega-dot| median {res['dOmega_dot_median_deg_day']:.4f} deg/day, "
          f"ep len median {res['ep_len_median']:.0f} substeps")
    if args.expect:
        want = int(args.expect.split('/')[0])
        ok = res['success'] == want
        print(f"  ANCHOR: expected {args.expect}, got "
              f"{res['success']}/{res['n_valid']} -> {'PASS' if ok else 'FAIL'}")
        res['anchor_pass'] = bool(ok)
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, 'w') as f:
            json.dump(res, f, indent=2)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--label', default='cell')
    ap.add_argument('--episodes', type=int, default=200)
    ap.add_argument('--seed', type=int, default=123)
    ap.add_argument('--j2-mode', type=int, default=0)
    ap.add_argument('--i-band', default='off',
                    help='"lo,hi" in degrees, or "off" for a fixed i_target')
    ap.add_argument('--i-fixed-deg', type=float, default=0.0)
    ap.add_argument('--raan-sample', type=int, default=0)
    ap.add_argument('--lvlh-frame-mode', type=int, default=0)
    ap.add_argument('--rendezvous-radius-m', type=float, default=30000.0)
    ap.add_argument('--rel-vel-tol-ms', type=float, default=50.0)
    ap.add_argument('--shape-w-match', type=float, default=0.8166667,
                    help='0.8166667 for the X3/rung-1 lineage, 0.35 for the TB3D ladder. '
                         "An arm must inherit its OWN parent's value.")
    ap.add_argument('--expect', default=None, help='e.g. 200/200 for anchor cells')
    ap.add_argument('--out', default=None)
    ap.add_argument('--verbose', action='store_true')
    args = ap.parse_args()
    r = run(args)
    sys.exit(0 if (args.expect is None or r.get('anchor_pass')) else 1)


if __name__ == '__main__':
    main()
