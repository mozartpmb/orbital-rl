#!/usr/bin/env python3
"""ext-j2wait evaluator — the drift-and-wait DECOMPOSITION.

THE EXPERIMENT, in one sentence: when the node-dominant plane gap exceeds what
the Delta-v budget can buy directly, does the policy dip its semi-major axis,
let differential nodal precession rotate the relative node, and come back —
instead of burning?

THE MEASUREMENT is a decomposition, not a success rate. Every decision either
carries an impulse (ACTION_DV nonzero and tau == 1) or does not, and the
relative-plane angle di_rel is a pure function of state, so the total plane
closure splits EXACTLY into

    closure_impulse    = sum over BURN decisions of  (di_rel_before - di_rel_after)
    closure_precession = sum over COAST/WARP decisions of (di_rel_before - di_rel_after)

with no residual by construction. The 60 s of precession that elapses inside a
burn decision is misattributed to the impulse; at 0.65 deg/day that is 4.5e-4
deg against a 25 m/s burn's 0.19 deg, i.e. 0.2%, and it is reported.

The headline the campaign is built to produce:
    "the policy buys X deg of plane alignment from precession at Y m/s,
     where direct burning the same X deg costs Z m/s"
with Z = 2*v_c*sin(X/2), the exact single-impulse chord (134 m/s/deg at LEO).

ALL CLAIMS ARE IN MEAN ELEMENTS (j2_A_design 1.4): under j2_mode=1 the env's
state IS the mean element set and no mean/osculating conversion happens
anywhere. And every plane number here is a NODE-DOMINANT plane error — drift
moves Omega and nothing else, so an inclination-component error is not
addressable by waiting at all (j2_plane_change E).

Usage:
    python3 scripts/orbital/extj2/j2wait_eval.py --ckpt <pt> --label A --episodes 200
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
    raise SystemExit(f'REFUSING TO RUN: pufferlib is {_PL}, not under {WT}')
import pufferlib.ocean.orbital.binding as _b                          # noqa: E402
if not os.path.abspath(_b.__file__).startswith(os.path.abspath(WT) + os.sep):
    raise SystemExit('REFUSING TO RUN: binding .so is not the worktree build')

MU = 3.986004418e14
R_EARTH = 6.371e6
DT = 60.0
ISP, G0 = 300.0, 9.80665
VE = ISP * G0
DRY = 850.0
FUEL_FRAC = 0.15
CAUSES = ['none', 'success', 'collision', 'escape', 'safety_cap',
          'stranded', 'hyperbolic', 'gave_up']

# ACTION_TAU / "is this row a burn" — mirrors orbital.h. Row 30 is the day-warp.
TAU = [1] * 9 + [5, 30, 60] + [1] * 4 + [180, 360] + [1] * 12 + [1440]
IS_BURN = [False] + [True] * 8 + [False, False, False] + [True] * 4 \
    + [False, False] + [True] * 12 + [False]
assert len(TAU) == 31 and len(IS_BURN) == 31

BASE_KW = dict(
    num_debris_min=0, num_debris_max=0,
    e_max_target=0.05, e_max_sat=0.05, same_orbit_init=0,
    init_phase_gap_max=3.14159, valid_init_only=1,
    gave_up_action='terminate', max_valid_init_attempts=4096,
    obs_alt_scale_m=1.6e6, lvlh_scale_m=6.371e6,
    shaping_mode=2, shape_w_lambda=1.0, shape_w_match=0.8166667,
    shape_dv_ref_ms=700.0, shape_gamma=1.0,
    phase_gap_mode=1, phase_obs_mode=1, cap_terminal_reward=0.0,
    dim3_mode=1, j2_mode=1, lvlh_frame_mode=1,
    i_target_min_rad=math.radians(30.0), i_target_max_rad=math.radians(60.0),
    raan_target_sample=0, legacy_action_space=31,
)


def plane_state(st):
    """(di_rel, node_frac, a_sat, a_tgt, fuel_frac) from one get_state() row."""
    hs, ht = st[5:8], st[20:23]
    di = math.atan2(float(np.linalg.norm(np.cross(ht, hs))), float(np.dot(ht, hs)))
    i_t = math.atan2(math.hypot(ht[0], ht[1]), ht[2])
    O_s = math.atan2(hs[0], -hs[1])
    O_t = math.atan2(ht[0], -ht[1])
    dO = math.atan2(math.sin(O_s - O_t), math.cos(O_s - O_t))
    node = abs(dO * math.sin(i_t))
    return di, (node / di if di > 1e-12 else 0.0), st[0], st[15], st[14]


def direct_dv(a, di_rad):
    """The exact single-impulse chord for a plane rotation of di at radius a."""
    return 2.0 * math.sqrt(MU / a) * math.sin(0.5 * di_rad)


def run(args):
    kw = dict(BASE_KW)
    kw['episode_cap_steps'] = args.cap
    kw['rendezvous_radius_m'] = args.rendezvous_radius_m
    kw['rel_vel_tol_ms'] = args.rel_vel_tol_ms
    kw['di_max_rad'] = math.radians(args.di_max_deg)
    kw['di_min_rad'] = math.radians(args.di_min_deg) if args.di_min_deg >= 0 else -1.0
    kw['di_phase_mode'] = args.di_phase_mode
    if args.no_daywarp:
        kw['legacy_action_space'] = 30

    env = Orbital(num_envs=1, **kw)
    policy = LSTMWrapper(env, Default(env))
    policy.load_state_dict(torch.load(args.ckpt, map_location='cpu', weights_only=True))
    policy.eval()
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)

    obs, _ = env.reset(seed=args.seed)
    state = {'lstm_h': torch.zeros(1, policy.hidden_size),
             'lstm_c': torch.zeros(1, policy.hidden_size)}
    fuel0 = DRY * FUEL_FRAC / (1.0 - FUEL_FRAC)

    ep = []          # one dict per finished episode
    acts_all = []
    cur = None

    def fresh():
        st = env.get_state()[0]
        di, nf, a_s, a_t, fu = plane_state(st)
        return dict(di0=di, node_frac0=nf, a_t=a_t, imp=0.0, prec=0.0,
                    dip_depth=0.0, dip_steps=0, decisions=0, daywarps=0,
                    warp_substeps=0, burns=0, prev_di=di, prev_step=0,
                    di_min=di, dv_used=0.0)

    cur = fresh()
    n_done = 0
    t0 = time.time()
    while n_done < args.episodes:
        with torch.no_grad():
            logits, _ = policy.forward_eval(
                torch.from_numpy(np.asarray(obs)).float().unsqueeze(0), state)
            a = int(torch.argmax(logits, dim=-1).item())
        acts_all.append(a)
        st_b = env.get_state()[0]
        di_b, _, a_s_b, a_t_b, fuel_b = plane_state(st_b)

        obs, _, term, _, _ = env.step(np.array([a], dtype=np.int32))

        st_a = env.get_state()[0]
        cur['decisions'] += 1
        if a == 30:
            cur['daywarps'] += 1
        if not IS_BURN[a]:
            cur['warp_substeps'] += TAU[a]
        else:
            cur['burns'] += 1

        if not term[0]:
            di_a, _, a_s_a, a_t_a, fuel_a = plane_state(st_a)
            d_close = di_b - di_a                     # positive = plane closed
            if IS_BURN[a]:
                cur['imp'] += d_close
            else:
                cur['prec'] += d_close
            # the dip: how far below the target's semi-major axis, for how long
            dip = a_t_b - a_s_b
            if dip > cur['dip_depth']:
                cur['dip_depth'] = dip
            if dip > 20e3:                            # 20 km = a deliberate dip
                cur['dip_steps'] += TAU[a]
            cur['di_min'] = min(cur['di_min'], di_a)
            # get_state()[14] is m_fuel/(m_dry+m_fuel); invert for the total
            # mass, then Tsiolkovsky gives the Dv spent so far.
            f = float(st_a[14])
            m_tot = DRY / max(1.0 - f, 1e-12)
            cur['dv_used'] = VE * math.log((DRY + fuel0) / max(m_tot, DRY + 1e-9))
        else:
            steps, cause = env.last_episode_result(0)
            cur['cause'] = int(cause)
            cur['substeps'] = int(steps)
            cur['hours'] = steps * DT / 3600.0
            cur['di_end'] = di_b            # last pre-terminal reading
            cur['closed'] = cur['di0'] - di_b
            cur['direct_ref'] = direct_dv(cur['a_t'], max(cur['di0'] - cur['di_min'], 0.0))
            cur['direct_ref_full'] = direct_dv(cur['a_t'], cur['di0'])
            ep.append(cur)
            n_done += 1
            cur = fresh()
            state = {'lstm_h': torch.zeros(1, policy.hidden_size),
                     'lstm_c': torch.zeros(1, policy.hidden_size)}
            if args.verbose and n_done % 25 == 0:
                sr = sum(1 for e in ep if e['cause'] == 1) / n_done
                print(f'    [{args.label}] {n_done}/{args.episodes} '
                      f'success {sr:.1%} ({time.time()-t0:.0f}s)', flush=True)
    env.close()

    causes = np.array([e['cause'] for e in ep])
    n_gave = int((causes == 7).sum())
    n_valid = len(causes) - n_gave
    succ = int((causes == 1).sum())

    def arr(k, sel=None):
        v = [e[k] for e in ep if (sel is None or sel(e))]
        return np.array(v, dtype=np.float64) if v else np.array([0.0])

    win = lambda e: e['cause'] == 1                                   # noqa: E731
    D = math.degrees

    def decomp(sel, n):
        """The exact split of plane closure into precession- vs impulse-bought."""
        if n == 0:
            return dict(prec=0.0, imp=0.0, share=0.0, dv=0.0, ref=0.0, ratio=0.0,
                        dip_km=0.0, dip_h=0.0, hrs=0.0, dec=0.0, dw=0.0)
        pr = D(arr('prec', sel).sum() / n)
        im = D(arr('imp', sel).sum() / n)
        dv = float(np.median(arr('dv_used', sel)))
        rf = float(np.median(arr('direct_ref_full', sel)))
        return dict(prec=pr, imp=im,
                    share=(pr / (pr + im)) if (pr + im) > 1e-9 else 0.0,
                    dv=dv, ref=rf, ratio=(rf / dv) if dv > 1e-9 else 0.0,
                    dip_km=float(np.median(arr('dip_depth', sel))) / 1e3,
                    dip_h=float(np.median(arr('dip_steps', sel))) * DT / 3600.0,
                    hrs=float(np.median(arr('hours', sel))),
                    dec=float(np.median(arr('decisions', sel))),
                    dw=float(np.median(arr('daywarps', sel))))

    d_win = decomp(win, succ)
    d_all = decomp(None, len(ep))
    prec_deg, imp_deg = d_win['prec'], d_win['imp']
    dv_med, ref_med = d_win['dv'], d_win['ref']

    res = dict(
        label=args.label, ckpt=os.path.basename(args.ckpt),
        episodes=args.episodes, seed=args.seed, cap=args.cap,
        box=[args.rendezvous_radius_m, args.rel_vel_tol_ms],
        plane_band_deg=[args.di_min_deg, args.di_max_deg],
        di_phase_mode=args.di_phase_mode, daywarp=not args.no_daywarp,
        success=succ, n_valid=n_valid, gave_up=n_gave,
        rate=succ / max(n_valid, 1),
        causes={CAUSES[c]: int((causes == c).sum()) for c in range(8)
                if (causes == c).any()},
        md5=hashlib.md5(np.array(acts_all, dtype=np.int32).tobytes()).hexdigest(),
        # ── THE DECOMPOSITION ──
        plane_closed_precession_deg=prec_deg,
        plane_closed_impulse_deg=imp_deg,
        precession_share=(prec_deg / (prec_deg + imp_deg))
        if (prec_deg + imp_deg) > 1e-9 else 0.0,
        dv_used_median_ms=dv_med,
        direct_burn_reference_ms=ref_med,
        dv_ratio_vs_direct=(ref_med / dv_med) if dv_med > 1e-9 else float('inf'),
        # ── the maneuver itself ──
        dip_depth_median_km=float(np.median(arr('dip_depth', win))) / 1e3,
        dip_duration_median_h=float(np.median(arr('dip_steps', win))) * DT / 3600.0,
        episode_hours_median=float(np.median(arr('hours', win))),
        decisions_median=float(np.median(arr('decisions', win))),
        daywarps_median=float(np.median(arr('daywarps', win))),
        di0_median_deg=D(float(np.median(arr('di0')))),
        node_frac_median=float(np.median(arr('node_frac0'))),
        # the same decomposition over EVERY episode, so a 0%-success floor row
        # still says whether the policy TRIED to drift or simply sat there
        all_episodes=dict(
            plane_closed_precession_deg=d_all['prec'],
            plane_closed_impulse_deg=d_all['imp'],
            precession_share=d_all['share'],
            dv_used_median_ms=d_all['dv'],
            direct_burn_reference_ms=d_all['ref'],
            dv_ratio_vs_direct=d_all['ratio'],
            dip_depth_median_km=d_all['dip_km'],
            dip_duration_median_h=d_all['dip_h'],
            episode_hours_median=d_all['hrs'],
            decisions_median=d_all['dec'],
            daywarps_median=d_all['dw'],
        ),
        state='MEAN ELEMENTS; plane errors are NODE-DOMINANT',
        wall_s=time.time() - t0,
    )

    print(f"  {args.label:26s} success {res['success']}/{res['n_valid']} = "
          f"{res['rate']:6.1%}   md5 {res['md5'][:12]}   {res['wall_s']:.0f}s")
    print(f"  {'':26s} causes: "
          + ', '.join(f'{k}={v}' for k, v in res['causes'].items()))
    print(f"  {'':26s} initial NODE-DOMINANT plane gap "
          f"{res['di0_median_deg']:.3f} deg (node fraction "
          f"{res['node_frac_median']:.3f})")
    print(f"  {'':26s} DECOMPOSITION over successes: precession bought "
          f"{res['plane_closed_precession_deg']:+.3f} deg, impulses bought "
          f"{res['plane_closed_impulse_deg']:+.3f} deg "
          f"({res['precession_share']:.1%} from precession)")
    print(f"  {'':26s} Dv used {res['dv_used_median_ms']:.1f} m/s vs direct-burn "
          f"reference {res['direct_burn_reference_ms']:.1f} m/s "
          f"({res['dv_ratio_vs_direct']:.2f}x cheaper)")
    print(f"  {'':26s} dip {res['dip_depth_median_km']:.1f} km deep for "
          f"{res['dip_duration_median_h']:.1f} h; episode "
          f"{res['episode_hours_median']:.1f} h, "
          f"{res['decisions_median']:.0f} decisions "
          f"({res['daywarps_median']:.0f} day-warps)")
    print(f"  {'':26s} ALL EPISODES (incl. failures): precession "
          f"{d_all['prec']:+.3f} deg, impulses {d_all['imp']:+.3f} deg "
          f"({d_all['share']:.1%} precession); Dv {d_all['dv']:.1f} m/s; "
          f"dip {d_all['dip_km']:.1f} km for {d_all['dip_h']:.1f} h; "
          f"{d_all['hrs']:.1f} h, {d_all['dec']:.0f} decisions "
          f"({d_all['dw']:.0f} day-warps)")
    print(f"  {'':26s} CLAIM: meets the box IN MEAN ELEMENTS; plane numbers are "
          f"NODE-DOMINANT gaps, not general plane errors.")
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
    ap.add_argument('--cap', type=int, default=22000)
    ap.add_argument('--di-min-deg', type=float, default=2.0)
    ap.add_argument('--di-max-deg', type=float, default=5.0)
    ap.add_argument('--di-phase-mode', type=int, default=1)
    ap.add_argument('--rendezvous-radius-m', type=float, default=30000.0)
    ap.add_argument('--rel-vel-tol-ms', type=float, default=50.0)
    ap.add_argument('--no-daywarp', action='store_true',
                    help='expose Discrete(30) instead of 31 — the ablation that '
                         'shows whether the day-warp is what makes it possible')
    ap.add_argument('--out', default=None)
    ap.add_argument('--verbose', action='store_true')
    args = ap.parse_args()
    run(args)
    return 0


if __name__ == '__main__':
    sys.exit(main())
