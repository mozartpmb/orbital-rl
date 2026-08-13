#!/usr/bin/env python3
"""ext-j2 verification ladder — the Python half of the J2 validation.

The C half (`j2_gates.c`) tests the propagator arithmetic directly against the
shipped header. This half tests the SHIPPED BUILD end-to-end: the kwarg
plumbing, the obs slots, the preconditions, the closed-loop dynamics, and the
regression anchors that must not move because `j2_mode` defaults to 0.

Stages
  a1  build identity + kwarg plumbing + preconditions + obs[29-32] semantics
  a2  closed-loop dynamics under j2_mode=1 through the real c_step
      (a/e/i invariant, Omega/omega/M drifting at the closed-form rates,
       and warp actions agreeing with an equal number of coast sub-steps)
  a3  regression anchors on the WORKTREE build at default flags:
        legacy       26/200 (success=26, collision=1, safety_cap=142, stranded=31)
        T3 canonical 100/100 on models/t3/seed42_L2_headline.pt
        X3           100/100 on models/t3/seed42_X3_3d_di1deg.pt

BUILD TRAP (n3d_REDTEAM MAJOR-6 / T5 notes): the `puffer` console script and a
bare `import pufferlib` both resolve to the MAIN checkout. Every stage here
asserts that the imported pufferlib lives under this worktree before it runs,
so a stale-build pass is impossible.

Run:
    cd /Users/pete/space_training-j2
    PYTHONPATH=/Users/pete/space_training-j2/pufferlib \
      python3 scripts/orbital/extj2/verify_extj2.py --stage all
"""

import argparse
import hashlib
import math
import os
import sys
import time

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
WT = os.path.abspath(os.path.join(_HERE, '..', '..', '..'))          # worktree root
sys.path.insert(0, os.path.join(WT, 'pufferlib'))

import pufferlib                                                     # noqa: E402
from pufferlib.ocean.orbital.orbital import Orbital                  # noqa: E402
from pufferlib.models import Default, LSTMWrapper                    # noqa: E402

# ── the build assertion, run at import so nothing can slip past it ──────────
_PL = os.path.abspath(pufferlib.__file__)
if not _PL.startswith(os.path.abspath(WT) + os.sep):
    raise SystemExit(
        f'REFUSING TO RUN: imported pufferlib is {_PL}, which is NOT under the '
        f'worktree {WT}. Re-run with PYTHONPATH={WT}/pufferlib. (The `puffer` '
        f'console script and bare imports resolve to the MAIN checkout.)')

import pufferlib.ocean.orbital.binding as _binding                   # noqa: E402
_SO = os.path.abspath(_binding.__file__)
if not _SO.startswith(os.path.abspath(WT) + os.sep):
    raise SystemExit(f'REFUSING TO RUN: binding .so is {_SO}, not under {WT}.')

MU = 3.986004418e14
R_EARTH = 6.371e6
J2_COEF = 1.08262668e-3
J2_R_EQ = 6.378137e6
DT = 60.0

CAUSES = ['none', 'success', 'collision', 'escape', 'safety_cap',
          'stranded', 'hyperbolic', 'gave_up']

MAIN = '/Users/pete/space_training'
T3_CKPT = f'{WT}/models/t3/seed42_L2_headline.pt'
X3_CKPT = f'{WT}/models/t3/seed42_X3_3d_di1deg.pt'
LEGACY_CKPT = (f'{MAIN}/pufferlib/experiments/puffer_orbital_177765503091/'
               f'model_puffer_orbital_000325.pt')

# T3 headline eval conditions (t3_ladder.sh heval).
T3_KW = dict(
    num_debris_min=0, num_debris_max=0,
    e_max_target=0.05, e_max_sat=0.05, same_orbit_init=0,
    init_phase_gap_max=3.14159, valid_init_only=1,
    gave_up_action='terminate', max_valid_init_attempts=4096,
    obs_alt_scale_m=1.6e6, lvlh_scale_m=6.371e6,
    shaping_mode=1, shape_gamma=1.0, phase_gap_mode=1, phase_obs_mode=1,
    episode_cap_steps=3000, cap_terminal_reward=0.0,
)

# X3 eval conditions (scripts/orbital/ext3d/t5_x3_seeds.sh, verbatim).
X3_KW = dict(T3_KW, dim3_mode=1, di_max_rad=0.017453, legacy_action_space=30,
             shaping_mode=2, shape_w_lambda=1.0, shape_w_match=0.8166667,
             shape_dv_ref_ms=700.0)

# Legacy anchor conditions (t1_readapt_v2.sh scan_one, seed 42).
LEGACY_KW = dict(
    num_debris_min=0, num_debris_max=0,
    e_max_target=0.05, e_max_sat=0.05, same_orbit_init=0,
    init_phase_gap_max=3.14159, valid_init_only=1,
    gave_up_action='terminate', max_valid_init_attempts=4096,
    legacy_action_space=10,
)

_RESULTS = []


def record(name, ok, detail=''):
    _RESULTS.append((name, bool(ok), detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    if detail:
        print(f"         {detail}")
    return bool(ok)


def secular_rates(a, e, i):
    """Closed form, written longhand — the comparand, not shared with the C."""
    n = math.sqrt(MU / a ** 3)
    p = a * (1.0 - e * e)
    k = 1.5 * n * J2_COEF * (J2_R_EQ / p) ** 2
    return (n,
            -k * math.cos(i),
            0.5 * k * (4.0 - 5.0 * math.sin(i) ** 2),
            n + 0.5 * k * math.sqrt(1.0 - e * e) * (2.0 - 3.0 * math.sin(i) ** 2))


def elements_from_state(row):
    """(a, e, M, theta, omega, inc, raan) for the chaser out of get_state()."""
    a, e, M, th, w = row[0], row[1], row[2], row[3], row[4]
    hx, hy, hz = row[5], row[6], row[7]
    inc = math.atan2(math.hypot(hx, hy), hz)
    raan = math.atan2(hx, -hy) if math.hypot(hx, hy) > 0 else 0.0
    return a, e, M, th, w, inc, raan


# ── a1: plumbing, preconditions, obs semantics ──────────────────────────────
def stage_a1(args):
    print('== A1  build identity, kwarg plumbing, preconditions, obs[29-32] ===')
    record('imported pufferlib is the WORKTREE build', True,
           f'{_PL}\n         {_SO}')

    # default j2_mode is 0 and is accepted without being passed
    e = Orbital(num_envs=1, **X3_KW)
    o, _ = e.reset(seed=3)
    off = np.array(o[0][29:33], dtype=np.float64)
    e.close()
    record('j2_mode defaults to 0; obs[29-32] all zero', np.all(off == 0.0),
           f'obs[29:33] = {off}')

    # j2_mode=1 writes cos i into 29/30 and leaves 31/32 reserved
    i_t = 0.9
    e = Orbital(num_envs=1, j2_mode=1, i_target_rad=i_t, **X3_KW)
    o, _ = e.reset(seed=3)
    st = e.get_state()[0]
    _, _, _, _, _, inc_s, _ = elements_from_state(st)
    on = np.array(o[0][29:33], dtype=np.float64)
    e.close()
    ok = (abs(on[1] - math.cos(i_t)) < 1e-6
          and abs(on[0] - math.cos(inc_s)) < 1e-6
          and on[2] == 0.0 and on[3] == 0.0)
    record('j2_mode=1: obs[29]=cos i_sat, obs[30]=cos i_tgt, obs[31,32]=0', ok,
           f'obs[29]={on[0]:.9f} (cos i_sat = {math.cos(inc_s):.9f}), '
           f'obs[30]={on[1]:.9f} (cos i_tgt = {math.cos(i_t):.9f}), '
           f'obs[31,32]={on[2]}, {on[3]}')

    # preconditions
    hits = {}
    for label, kw in (('dim3_mode=0', dict(j2_mode=1)),
                      ('num_debris>0', dict(j2_mode=1, dim3_mode=1,
                                            num_debris_min=4, num_debris_max=8))):
        try:
            Orbital(num_envs=1, **kw).close()
            hits[label] = None
        except Exception as ex:                                     # noqa: BLE001
            hits[label] = f'{type(ex).__name__}: {ex}'
    record('precondition j2_mode=1 requires dim3_mode=1 (hard error)',
           hits['dim3_mode=0'] is not None, hits['dim3_mode=0'])
    record('precondition j2_mode=1 requires num_debris=0 (hard error)',
           hits['num_debris>0'] is not None, hits['num_debris>0'])

    # i_target_rad = 0 must WARN, not raise (anchors J-A3/J-A5 live there)
    try:
        Orbital(num_envs=1, j2_mode=1, i_target_rad=0.0, **X3_KW).close()
        ok = True
        det = 'constructed; C emits a stderr warning (plane channel inert)'
    except Exception as ex:                                         # noqa: BLE001
        ok, det = False, f'{type(ex).__name__}: {ex}'
    record('i_target_rad=0 under j2_mode=1 warns but does NOT raise', ok, det)

    # obs stream at j2_mode=0 is byte-identical to the pre-change lineage shape
    n, steps = 32, 120
    rng = np.random.default_rng(0)
    acts = [rng.integers(0, 30, n).astype(np.int32) for _ in range(steps)]
    env = Orbital(num_envs=n, seed=7, **X3_KW)
    o, _ = env.reset(seed=42)
    acc = [np.array(o, copy=True)]
    for a in acts:
        o, _, _, _, _ = env.step(a)
        acc.append(np.array(o, copy=True))
    env.close()
    stream = np.stack(acc)
    md5 = hashlib.md5(stream.astype(np.float32).tobytes()).hexdigest()
    cols = stream[:, :, 29:33]
    record('j2_mode=0: obs[29-32] stay identically 0 over 120 steps x 32 envs',
           np.all(cols == 0.0),
           f'max|obs[29:33]| = {np.abs(cols).max():.3e}   '
           f'obs-stream md5 = {md5}')
    return dict(md5=md5)


# ── a2: closed-loop dynamics through the real c_step ────────────────────────
def stage_a2(args):
    print('== A2  closed-loop dynamics under j2_mode=1 (real c_step) =========')
    kw = dict(X3_KW, j2_mode=1, i_target_rad=math.radians(51.6),
              di_max_rad=-1.0, episode_cap_steps=12000)
    env = Orbital(num_envs=1, **kw)
    env.reset(seed=11)
    s0 = env.get_state()[0].copy()
    a0, e0, M0, _, w0, i0, O0 = elements_from_state(s0)

    N = 3000                       # coast only: action 0 is a 1-sub-step coast
    for _ in range(N):
        _, _, term, _, _ = env.step(np.array([0], dtype=np.int32))
        if term[0]:
            raise SystemExit('episode terminated during the coast probe; '
                             'raise episode_cap_steps')
    s1 = env.get_state()[0].copy()
    env.close()
    a1, e1, M1, _, w1, i1, O1 = elements_from_state(s1)

    n, rO, ro, rM = secular_rates(a0, e0, i0)
    T = N * DT

    def unwrap_rate(x1, x0, ref):
        """Recover the rate given we know its approximate value."""
        k = round((ref * T - (x1 - x0)) / (2 * math.pi))
        return ((x1 - x0) + 2 * math.pi * k) / T

    mO = unwrap_rate(O1, O0, rO)
    mw = unwrap_rate(w1, w0, ro)
    mM = unwrap_rate(M1, M0, rM)

    inv_ok = (a1 == a0) and (e1 == e0) and abs(i1 - i0) < 1e-12
    record('a, e, i unchanged over 3000 coast sub-steps in the real env', inv_ok,
           f'da = {abs(a1-a0):.3e} m, de = {abs(e1-e0):.3e}, '
           f'di = {math.degrees(abs(i1-i0)):.3e} deg')

    errs = [abs(mO - rO) / abs(rO), abs(mw - ro) / abs(ro), abs(mM - rM) / abs(rM)]
    record('Omega/omega/M drift at the closed-form secular rates', max(errs) < 1e-8,
           f'Om {math.degrees(mO)*86400:+.6f} vs {math.degrees(rO)*86400:+.6f} deg/day, '
           f'om {math.degrees(mw)*86400:+.6f} vs {math.degrees(ro)*86400:+.6f}, '
           f'worst rel err {max(errs):.2e}')

    # warp action 17 (tau = 360) must equal 360 coast actions, exactly as the
    # env computes it (both paths sub-step, so this should be BITWISE equal).
    def run(actions):
        e2 = Orbital(num_envs=1, **kw)
        e2.reset(seed=11)
        for a in actions:
            e2.step(np.array([a], dtype=np.int32))
        st = e2.get_state()[0].copy()
        e2.close()
        return st

    A = run([17])                      # one warp-6h decision
    B = run([0] * 360)                 # 360 coast decisions
    d = np.abs(A - B)
    record('warp action 17 (tau=360) == 360 coast sub-steps, bitwise',
           np.all(d == 0.0),
           f'max |diff| over the 36-element state vector = {d.max():.3e}')

    # equatorial target: raan must stay exactly 0.0 through the real c_step
    kw0 = dict(X3_KW, j2_mode=1, i_target_rad=0.0, di_max_rad=-1.0,
               episode_cap_steps=12000)
    e3 = Orbital(num_envs=1, **kw0)
    e3.reset(seed=11)
    worst = 0.0
    for _ in range(600):
        e3.step(np.array([9], dtype=np.int32))     # warp-5min
        st = e3.get_state()[0]
        hx, hy = st[20], st[21]                     # target h-hat x, y
        worst = max(worst, abs(hx), abs(hy))
    e3.close()
    record('equatorial target stays exactly equatorial (h_t = z-hat) under J2',
           worst == 0.0, f'max |h_t.x|, |h_t.y| over 3000 sub-steps = {worst:.3e}')
    return {}


# ── a3: regression anchors ──────────────────────────────────────────────────
def load_policy(env, ckpt):
    p = LSTMWrapper(env, Default(env))
    p.load_state_dict(torch.load(ckpt, map_location='cpu', weights_only=True))
    p.eval()
    return p


def rollout(env, ckpt, episodes, seed, label=''):
    """eval_checkpoint.py protocol: num_envs=1, greedy argmax, LSTM zeroed per
    episode, success = terminal cause 1, gave-up inits out of the denominator."""
    policy = load_policy(env, ckpt)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    obs, _ = env.reset(seed=seed)
    state = {'lstm_h': torch.zeros(1, policy.hidden_size),
             'lstm_c': torch.zeros(1, policy.hidden_size)}
    actions, causes = [], []
    n_done, t0 = 0, time.time()
    while n_done < episodes:
        with torch.no_grad():
            logits, _ = policy.forward_eval(
                torch.from_numpy(np.asarray(obs)).float().unsqueeze(0), state)
            a = int(torch.argmax(logits, dim=-1).item())
        actions.append(a)
        obs, _, term, _, _ = env.step(np.array([a], dtype=np.int32))
        if term[0]:
            n_done += 1
            _, cause = env.last_episode_result(0)
            causes.append(int(cause))
            state = {'lstm_h': torch.zeros(1, policy.hidden_size),
                     'lstm_c': torch.zeros(1, policy.hidden_size)}
    env.close()
    causes = np.array(causes)
    n_gave = int((causes == 7).sum())
    succ = int((causes == 1).sum())
    return dict(
        label=label, success=succ, n_valid=len(causes) - n_gave, gave_up=n_gave,
        cause_str=', '.join(f'{CAUSES[c]}={int((causes == c).sum())}'
                            for c in range(8) if (causes == c).any()),
        md5=hashlib.md5(np.array(actions, dtype=np.int32).tobytes()).hexdigest(),
        n_actions=len(actions), wall=time.time() - t0)


def stage_a3(args):
    print('== A3  regression anchors on the WORKTREE build, default flags =====')
    out = {}

    cells = [
        ('legacy', LEGACY_CKPT, LEGACY_KW, 200, 42, 26,
         'success=26, collision=1, safety_cap=142, stranded=31'),
        ('T3 canonical', T3_CKPT, T3_KW, args.eps, 123, args.eps, f'success={args.eps}'),
        ('X3 3D', X3_CKPT, X3_KW, args.eps, 123, args.eps, f'success={args.eps}'),
    ]
    for label, ckpt, kw, eps, seed, want, want_causes in cells:
        if not os.path.exists(ckpt):
            record(f'{label} anchor', False, f'checkpoint missing: {ckpt}')
            continue
        r = rollout(Orbital(num_envs=1, **kw), ckpt, eps, seed, label)
        out[label] = r
        print(f"    {label:14s} {r['success']}/{r['n_valid']}   "
              f"decisions {r['n_actions']}   md5 {r['md5'][:12]}   {r['wall']:.0f}s")
        print(f"    {'':14s} causes: {r['cause_str']}")
        record(f'{label} anchor unchanged under j2_mode=0 default',
               r['success'] == want and r['cause_str'] == want_causes,
               f"expected {want}/{r['n_valid']} [{want_causes}], "
               f"got {r['success']}/{r['n_valid']} [{r['cause_str']}]")
    return out


STAGES = {'a1': stage_a1, 'a2': stage_a2, 'a3': stage_a3}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', default='all')
    ap.add_argument('--eps', type=int, default=100,
                    help='episodes for the T3/X3 anchors (legacy is fixed at 200)')
    args = ap.parse_args()
    names = list(STAGES) if args.stage == 'all' else args.stage.split(',')
    for nme in names:
        STAGES[nme](args)
        print()
    print('== SUMMARY ========================================================')
    for name, ok, _ in _RESULTS:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    n_fail = sum(1 for _, ok, _ in _RESULTS if not ok)
    print(f'\n  {len(_RESULTS) - n_fail}/{len(_RESULTS)} checks pass')
    sys.exit(1 if n_fail else 0)


if __name__ == '__main__':
    main()
