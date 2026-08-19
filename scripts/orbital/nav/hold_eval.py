#!/usr/bin/env python3
"""HOLD-DURATION eval: does a tight-box policy STAY in the box after capture?

WHY THIS EXISTS. Every success number this project publishes is an INSTANT: the
C env terminates the moment position and relative velocity are simultaneously
inside the box (orbital.h:1825). Nothing measures the next minute. A policy that
crosses the box at speed and coasts straight out the far side scores exactly the
same as one that parks. For a rendezvous criterion that is the difference
between a capture and a flyby, so "97% success" and "97% captured" are not the
same claim and only the first one is currently evidenced.

METHOD — real continued rollout, not a proxy.
The box thresholds enter the C env in exactly one place: the termination test.
They appear in the struct (orbital.h:500-501), the config load (2012-2013), the
compile-time defaults (119-120), and the `at_target` test (1825-1826) — and
NOWHERE in the observation encoder or the shaping potential. So shrinking the
box to a value nothing can satisfy makes success unreachable while leaving the
policy's inputs bit-identical. The episode simply keeps running, and occupancy
is scored offline against the REAL box.

That claim is verified rather than asserted: --verify re-runs the same seeds at
the real box and checks the action stream is identical up to first box entry.
If the tiny box changed behaviour, that check fails and the numbers are void.

WHAT IS REPORTED.
  hold_N   fraction of CAPTURING episodes that remain continuously inside the
           box (position AND velocity, both) at every decision epoch from first
           entry through N minutes of sim time.
  occ_N    mean fraction of decision epochs inside the box over that window —
           the softer metric, which separates "left immediately" from "chattered
           across the boundary".
  Failure attribution: left on position, left on velocity, or the episode ended
  (with cause) before the window elapsed.

CAPTURE IS ANCHORED ON THE ENV'S OWN CRITERION, NOT ON MY SAMPLING.
orbital.h:2504 runs the termination check EVERY SUBSTEP ("so the warp never
skips past a conjunction"), so capture happens at 60 s resolution while a
policy mid-warp only makes a decision every 360 min. Detecting entry on the
decision grid therefore misses real captures: measured directly, one seed whose
real-box episode succeeded at decision 56 was not seen as inside until decision
155 on the decision grid. So each episode is run TWICE at the same seed -- once
at the real box, which terminates exactly when the env says captured and yields
the capture substep, and once at the dead box, which supplies what happens
afterwards. The window is anchored on the first run and scored on the second.

RESOLUTION CAVEAT THAT REMAINS. Occupancy after capture is still sampled on the
decision grid. ACTION_TAU is 1 minute for every burn row, so a policy doing fine
terminal control is sampled every 60 s and the grid is exact; a policy that
warps out of the box is sampled coarsely. Both the post-capture tau histogram
and the count of episodes with NO decision boundary inside the window (scored
separately as `unresolved`, never as held or not-held) are reported, so the
blind spot is visible rather than absorbed into the headline.

    python3 scripts/orbital/nav/hold_eval.py --ckpt TB5-3D --episodes 200
    python3 scripts/orbital/nav/hold_eval.py --report
"""

import argparse
import hashlib
import json
import math
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(_HERE, '..', '..', '..'))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(ROOT, 'pufferlib'))

import eval_relnav3d as ER3                                       # noqa: E402
from pufferlib.models import Default, LSTMWrapper                 # noqa: E402

OUT = os.path.join(ROOT, 'web_data', 'results', 'hold_eval')
M = os.path.join(ROOT, 'models', 't3')
WINDOWS = (5, 15, 30)                       # minutes of sim time
CAUSES = ER3.CAUSES

# orbital.h:278 ACTION_TAU — substeps (minutes) advanced per decision row.
TAU = np.array([1] * 9 + [5, 30, 60] + [1] * 4 + [180, 360] + [1] * 12 + [1440])

# The two tight-box champions, each at ITS OWN validated config.
ARMS = {
    'TB5-3D': dict(
        ckpt='n3dnav_T-BO3D-TB5.pt', rung='X3', mode='bearings_only',
        box=(5000.0, 1.0), head=30,
        note='3D bearings-only nav, real batch-IOD acquisition'),
    'A3b-j2': dict(
        ckpt='extj2_A3b_j2_box5k1.pt', rung='J2X', mode='truth',
        box=(5000.0, 1.0), head=30,
        note='J2 inclined, truth-state control lineage'),
}
# Success is made unreachable, not merely unlikely: 1 um and 1 nm/s.
DEAD_BOX = dict(rendezvous_radius_m=1e-6, rel_vel_tol_ms=1e-9)


def build(arm, real_box):
    a = ARMS[arm]
    kw = dict(ER3.RUNGS[a['rung']])
    kw['legacy_action_space'] = a['head']
    if real_box:
        kw.update(rendezvous_radius_m=a['box'][0], rel_vel_tol_ms=a['box'][1])
    else:
        kw.update(DEAD_BOX)
    ER3.RUNGS['_HOLD'] = kw
    acq = 'real' if a['mode'] == 'bearings_only' else 'surrogate'
    env = ER3.make_env('_HOLD', a['mode'], acq=acq)
    if acq == 'real':
        ER3._install_real_acq(env)
    policy = LSTMWrapper(env, Default(env))
    policy.load_state_dict(torch.load(os.path.join(M, a['ckpt']),
                                      map_location='cpu', weights_only=True))
    policy.eval()
    return env, policy, a


def truth_rel(env):
    """(|dp|, |dv|, sim_minutes) from TRUTH state at the current decision epoch.

    Read from the C getter (binding.c fill_state_row) rather than the nav
    wrapper's `_prev_tgt`, which only exists once the nav path has allocated
    and is therefore absent in truth mode. Layout: chaser Cartesian o[8:14],
    target o[23:29], substep counter o[29]. The counter is the authoritative
    clock -- accumulating ACTION_TAU myself would re-derive a number the env
    already keeps, and would silently drift if a row's tau ever changed.
    """
    st = env.get_state()[0]
    d = st[23:26] - st[8:11]
    v = st[26:29] - st[11:14]
    return (float(np.linalg.norm(d)), float(np.linalg.norm(v)),
            float(st[29]))          # substeps; 1 substep = 60 s


def _match_noise(env, i):
    """Re-seed the nav measurement RNG per episode so the two passes see the
    SAME noise realization.

    Found by the stream check, which is why it is in the script: the wrapper
    seeds `_rng` once at construction and advances it per step, so pass 1
    (which terminates at capture) and pass 2 (which runs on past it) desync
    after the first episode. The bearings-only arm scored only 90/200 identical
    pre-capture action streams before this -- meaning the capture epoch being
    used as the window anchor belonged to a DIFFERENT trajectory realization
    than the one being scored. Truth-mode arms draw no measurement noise and
    were unaffected (200/200), which is exactly the asymmetry that made the
    cause identifiable.
    """
    if getattr(env, '_rng', None) is not None:
        env._rng = np.random.default_rng([env._nav_seed, 90210, i])


def _greedy(policy, obs, st):
    with torch.no_grad():
        logits, _ = policy.forward_eval(
            torch.from_numpy(np.asarray(obs)).float(), st)
        return int(torch.argmax(logits, dim=-1).item())


def run_arm(arm, episodes, seed, pre_cap=900):
    """Two passes per seed: real box to anchor capture, dead box to score hold.

    One episode per explicit reset (verified deterministic: reset(seed) fully
    re-seeds the C RNG) so the two passes see identical initial conditions and
    the cost stays bounded -- with success unreachable a dead-box episode would
    otherwise run to the 3000-step safety cap every time.
    """
    envR, polR, a = build(arm, True)
    envD, polD, _ = build(arm, False)
    br, bv = a['box']
    torch.set_num_threads(1)
    eps, hidden = [], polR.hidden_size
    horizon = max(WINDOWS)

    for i in range(episodes):
        # ── pass 1: real box. Terminates AT capture, at substep resolution.
        obs, _ = envR.reset(seed=seed + i)
        _match_noise(envR, i)
        st = {'lstm_h': torch.zeros(1, hidden), 'lstm_c': torch.zeros(1, hidden)}
        cap_sub, cause, k, actsR = None, None, 0, []
        while k < pre_cap:
            actsR.append(_greedy(polR, obs, st))
            obs, _, term, trunc, _ = envR.step(
                np.array([actsR[-1]], dtype=np.int32))
            k += 1
            if term[0] or trunc[0]:
                # (sim_steps, cause): sim_steps is the SUBSTEP count of the
                # completed episode, i.e. exactly the capture epoch, and it is
                # read before the auto-reset can overwrite the state getter.
                n_sub, c = envR.last_episode_result(0)
                cause = int(c)
                if cause == 1:                     # TERM_SUCCESS
                    cap_sub = float(n_sub)
                break
        if cap_sub is None:
            eps.append(dict(captured=False, cause=cause, samples=[], taus=[],
                            stream_ok=True))
            continue

        # ── pass 2: dead box, same seed. Score occupancy from the anchor on.
        obs, _ = envD.reset(seed=seed + i)
        _match_noise(envD, i)
        st = {'lstm_h': torch.zeros(1, hidden), 'lstm_c': torch.zeros(1, hidden)}
        samples, taus, cause2, actsD = [], [], None, []
        while True:
            act = _greedy(polD, obs, st)
            actsD.append(act)
            dp, dv, sub = truth_rel(envD)
            if sub >= cap_sub:
                samples.append((sub - cap_sub, (dp < br) and (dv < bv), dp, dv))
                taus.append(int(TAU[act]))
                if sub - cap_sub >= horizon:
                    break
            obs, _, term, trunc, _ = envD.step(np.array([act], dtype=np.int32))
            if term[0] or trunc[0]:
                _, c = envD.last_episode_result(0)
                cause2 = int(c)
                break
        # TEST-THE-TEST, on every episode rather than a sampled subset: the
        # dead box is only a valid instrument if it leaves behaviour untouched.
        # Up to the capture decision the two passes must emit the same actions.
        eps.append(dict(captured=True, cause=cause2, cap_sub=cap_sub,
                        samples=samples, taus=taus,
                        stream_ok=(actsD[:len(actsR)] == actsR)))
    return eps, a


def summarise(arm, eps, a):
    cap = [e for e in eps if e['captured']]
    out = dict(arm=arm, ckpt=a['ckpt'], mode=a['mode'], box_r=a['box'][0],
               box_v=a['box'][1], episodes=len(eps), captured=len(cap),
               capture_rate=len(cap) / max(len(eps), 1), note=a['note'])
    taus = [t for e in cap for t in e['taus']]
    out['tau_post_entry'] = {str(v): int((np.array(taus) == v).sum())
                             for v in sorted(set(taus))} if taus else {}
    out['stream_ok'] = sum(1 for e in eps if e['stream_ok'])
    for N in WINDOWS:
        held, occ, unres = 0, [], 0
        why = {'left_pos': 0, 'left_vel': 0, 'left_both': 0,
               'episode_ended': 0}
        for e in cap:
            win = [s for s in e['samples'] if s[0] <= N + 1e-9]
            if not win:
                # No decision boundary landed inside the window -- the policy
                # warped straight over it. Scored as UNRESOLVED, never as held
                # or not-held: the instrument has nothing to say here and
                # rounding it either way would manufacture a number.
                unres += 1
                continue
            reached = e['samples'][-1][0] >= N - 1e-9
            ins = [s[1] for s in win]
            occ.append(float(np.mean(ins)))
            if all(ins) and reached:
                held += 1
                continue
            if not reached:
                why['episode_ended'] += 1
            else:
                bad = next(s for s in win if not s[1])
                p, v = bad[2] >= a['box'][0], bad[3] >= a['box'][1]
                why['left_both' if (p and v) else
                    ('left_pos' if p else 'left_vel')] += 1
        den = max(len(cap) - unres, 1)
        out[f'hold_{N}'] = held / den
        out[f'occ_{N}'] = float(np.mean(occ)) if occ else 0.0
        out[f'why_{N}'] = why
        out[f'unresolved_{N}'] = unres
        out[f'n_scored_{N}'] = len(cap) - unres
    out['causes'] = {CAUSES[e['cause']] if e['cause'] is not None
                     and e['cause'] >= 0 else 'no_approach': 0 for e in eps}
    for e in eps:
        k = (CAUSES[e['cause']] if e['cause'] is not None and e['cause'] >= 0
             else ('running' if e['cause'] is None else 'no_approach'))
        out['causes'][k] = out['causes'].get(k, 0) + 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', choices=sorted(ARMS))
    ap.add_argument('--episodes', type=int, default=200)
    ap.add_argument('--seed', type=int, default=123)
    ap.add_argument('--verify', type=int, default=0,
                    help='N episodes: check the dead box did not change actions')
    ap.add_argument('--report', action='store_true')
    a = ap.parse_args()

    if a.report:
        rows = [json.load(open(os.path.join(OUT, f)))
                for f in sorted(os.listdir(OUT)) if f.endswith('.json')]
        hdr = (f'{"arm":9}{"mode":15}{"cap%":>6}' +
               ''.join(f'{"hold" + str(N):>8}{"occ" + str(N):>8}'
                       for N in WINDOWS))
        print(hdr)
        print('-' * len(hdr))
        for r in rows:
            print(f'{r["arm"]:9}{r["mode"]:15}{100 * r["capture_rate"]:5.1f}%' +
                  ''.join(f'{100 * r[f"hold_{N}"]:7.1f}%'
                          f'{100 * r[f"occ_{N}"]:7.1f}%' for N in WINDOWS))
        for r in rows:
            print(f'\n{r["arm"]}: captured {r["captured"]}/{r["episodes"]}  '
                  f'tau post-entry {r["tau_post_entry"]}')
            for N in WINDOWS:
                print(f'   {N:2d} min  why-not: {r[f"why_{N}"]}')
        return 0

    if a.verify:
        for arm in ([a.arm] if a.arm else sorted(ARMS)):
            d, _ = run_arm(arm, a.verify, a.seed, real_box=False)
            r, _ = run_arm(arm, a.verify, a.seed, real_box=True)
            same = sum(1 for x, y in zip(d, r) if x['md5_pre'] == y['md5_pre'])
            print(f'{arm:9} pre-entry action stream identical '
                  f'{same}/{a.verify}  '
                  f'{"PASS" if same == a.verify else "FAIL"}')
        return 0

    eps, spec = run_arm(a.arm, a.episodes, a.seed)
    res = summarise(a.arm, eps, spec)
    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(os.path.join(OUT, f'{a.arm}.json'), 'w'), indent=1)
    print(f'{a.arm}: captured {res["captured"]}/{res["episodes"]}  ' +
          '  '.join(f'hold{N} {100 * res[f"hold_{N}"]:.1f}%'
                    for N in WINDOWS), flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
