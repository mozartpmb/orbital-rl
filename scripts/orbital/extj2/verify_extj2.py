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


def stage_a4(args):
    """ext-j2 rung: the anchors must survive BOTH new env changes."""
    print('== A4  rung changes vs the anchors ================================')
    print('  Two changes land in this branch that touch code the anchors run:')
    print('   (1) the de_max inertial-varpi correction is now node-relative to')
    print('       the TARGET ((raan_s - raan_t), not raan_s). raan_t is exactly')
    print('       0.0 in every shipped lineage and x - 0.0 == x, so bit-exact.')
    print('   (2) lvlh_frame_mode=1 builds obs[33-36] in the true target orbital')
    print('       frame. At i = Omega = 0 that must reduce to the legacy block.')
    print('  Both are re-run through the CHECKPOINT anchors, not just at reset.')
    out = {}
    for label, ckpt, kw, eps, seed, want, want_causes, md5_expect in [
        ('legacy lvlh1', LEGACY_CKPT, dict(LEGACY_KW, lvlh_frame_mode=1), 200, 42, 26,
         'success=26, collision=1, safety_cap=142, stranded=31', 'f8a2388f0992'),
        ('T3 lvlh1', T3_CKPT, dict(T3_KW, lvlh_frame_mode=1), args.eps, 123, args.eps,
         f'success={args.eps}', '68b267bed369'),
        ('X3 lvlh1', X3_CKPT, dict(X3_KW, lvlh_frame_mode=1), args.eps, 123, args.eps,
         f'success={args.eps}', '003105f29898'),
    ]:
        if not os.path.exists(ckpt):
            record(f'{label} anchor', False, f'checkpoint missing: {ckpt}')
            continue
        r = rollout(Orbital(num_envs=1, **kw), ckpt, eps, seed, label)
        out[label] = r
        print(f"    {label:14s} {r['success']}/{r['n_valid']}   md5 {r['md5'][:12]}   "
              f"{r['wall']:.0f}s   causes: {r['cause_str']}")
        record(f'{label}: lvlh_frame_mode=1 is md5-IDENTICAL to the legacy frame',
               r['success'] == want and r['cause_str'] == want_causes
               and r['md5'][:12] == md5_expect,
               f"expected {want}/{r['n_valid']} md5 {md5_expect}, got "
               f"{r['success']}/{r['n_valid']} md5 {r['md5'][:12]}")

    # the sampler must be inert when off, through a full closed loop
    a = rollout(Orbital(num_envs=1, **X3_KW), X3_CKPT, 50, 123, 'X3 sampler-off')
    b = rollout(Orbital(num_envs=1, i_target_min_rad=-1.0, i_target_max_rad=-1.0,
                        raan_target_sample=0, **X3_KW), X3_CKPT, 50, 123, 'X3 explicit-off')
    record('sampler kwargs present-but-off are a no-op (md5 identical)',
           a['md5'] == b['md5'],
           f"md5 {a['md5'][:12]} vs {b['md5'][:12]}")
    return out


def stage_a5(args):
    """ext-j2wait: the day-warp row, the new cap, and the stall gate."""
    print('== A5  ext-j2wait: day-warp row, 22000 cap, do-nothing gate ========')
    import math as _m
    WAIT_KW = dict(X3_KW, j2_mode=1, lvlh_frame_mode=1,
                   i_target_min_rad=_m.radians(30), i_target_max_rad=_m.radians(60),
                   legacy_action_space=31, episode_cap_steps=22000,
                   di_max_rad=_m.radians(5.0), di_min_rad=_m.radians(2.0),
                   di_phase_mode=1)

    # (1) exposure: default stays Discrete(16); 31 must be opt-in; 32 rejected
    e = Orbital(num_envs=1, **dict(X3_KW, legacy_action_space=None))
    n_def = e.single_action_space.n; e.close()
    e = Orbital(num_envs=1, **WAIT_KW)
    n_opt = e.single_action_space.n; e.close()
    bad = None
    try:
        Orbital(num_envs=1, **dict(WAIT_KW, legacy_action_space=32)).close()
    except Exception as ex:                                        # noqa: BLE001
        bad = f'{type(ex).__name__}'
    record('A5a exposed space: default 16, opt-in 31, 32 rejected',
           n_def == 16 and n_opt == 31 and bad is not None,
           f'default {n_def}, opt-in {n_opt}, 32 -> {bad}')

    # (2) the day-warp row IS 1440 coast sub-steps, bitwise, through c_step
    def roll(actions, cap=22000):
        env = Orbital(num_envs=1, **dict(WAIT_KW, episode_cap_steps=cap))
        env.reset(seed=77)
        for a in actions:
            env.step(np.array([a], dtype=np.int32))
        st = env.get_state()[0].copy()
        env.close()
        return st
    A = roll([30])
    B = roll([0] * 1440)
    d = np.abs(A - B)
    record('A5b action 30 (tau=1440) == 1440 coast sub-steps, BITWISE',
           np.all(d == 0.0),
           f'max |diff| over the 36-element state vector = {d.max():.3e}')
    # and that it is inert when not exposed: a Discrete-30 env cannot emit it
    A2 = roll([17] * 4)
    B2 = roll([0] * 1440)
    record('A5c four 6 h warps == one day of coasting, BITWISE (cross-check)',
           np.all(np.abs(A2 - B2) == 0.0),
           f'max |diff| = {np.abs(A2 - B2).max():.3e}')

    # (3) THE STALL GATE. cap_terminal_reward=0 + shape_gamma=1 means a capped
    #     episode's whole return is the shaping it accumulated. Under
    #     shaping_mode 2 the per-step reward IS that shaping delta, so a
    #     do-nothing episode's running reward sum IS Phi(t) - Phi(0). Measure
    #     the max FARMABLE gain over a FULL 22000-step cap driven entirely by
    #     day-warps — the J-A5 gate re-run at the new horizon.
    NE = 64
    env = Orbital(num_envs=NE, **WAIT_KW)
    env.reset(seed=99)
    a = np.full(NE, 30, dtype=np.int32)
    run = np.zeros(NE); peak = np.zeros(NE); tot = np.zeros(NE)
    n_term = 0
    for _ in range(22000 // 1440 + 2):
        _, rew, term, _, _ = env.step(a)
        run += np.asarray(rew, dtype=np.float64)
        peak = np.maximum(peak, run)
        if term.any():
            n_term += int(term.sum())
            tot = np.where(term, run, tot)
            run = np.where(term, 0.0, run)
            peak = np.where(term, 0.0, peak)
    env.close()
    worst_gain = float(peak.max())
    term_disc = 10.0 * 0.995 ** (22000 / 1440)     # +10 discounted over a full cap
    record('A5d do-nothing gain is non-competitive with the success terminal',
           worst_gain < 0.25 * term_disc,
           f'max running Phi gain over a FULL 22000-step cap, day-warps only, '
           f'{NE} envs at the campaign plane distribution = {worst_gain:+.4f}; '
           f'discounted success terminal = {term_disc:.3f}; ratio '
           f'{worst_gain / term_disc:.1%} (gate 25%). Terminals seen: {n_term}. '
           f'NOTE this is 95x the 6000-step-cap value (0.006) BY DESIGN: over '
           f'15.3 d precession really does close plane error for free, which is '
           f'the phenomenon under test, not a leak. shape_gamma=1 telescopes so '
           f'it cannot be farmed by looping.')

    # (3b) the plane sampler: band honoured, and the error is node-DOMINANT
    env = Orbital(num_envs=256, **WAIT_KW)
    dmin = 9e9; dmax = -9e9; nf = []
    for k in range(24):
        env.reset(seed=1000 + k)
        st = env.get_state()
        hs, ht = st[:, 5:8], st[:, 20:23]
        dot = np.clip(np.einsum('ni,ni->n', hs, ht), -1, 1)
        di_rel = np.arccos(dot)
        i_s = np.arctan2(np.hypot(hs[:, 0], hs[:, 1]), hs[:, 2])
        i_t = np.arctan2(np.hypot(ht[:, 0], ht[:, 1]), ht[:, 2])
        O_s = np.arctan2(hs[:, 0], -hs[:, 1])
        O_t = np.arctan2(ht[:, 0], -ht[:, 1])
        dO = np.arctan2(np.sin(O_s - O_t), np.cos(O_s - O_t))
        node = np.abs(dO * np.sin(i_t))          # drift-CORRECTABLE component
        frac = node / np.maximum(di_rel, 1e-12)
        dmin = min(dmin, np.degrees(di_rel).min())
        dmax = max(dmax, np.degrees(di_rel).max())
        nf.append(frac)
    env.close()
    nf = np.concatenate(nf)
    record('A5e plane error lands in U(2,5) deg and is NODE-DOMINANT',
           dmin >= 1.95 and dmax <= 5.05 and np.percentile(nf, 5) >= 0.80,
           f'realized di_rel {dmin:.3f}-{dmax:.3f} deg (knob 2.000-5.000); '
           f'node fraction p05 {np.percentile(nf, 5):.3f} p50 {np.median(nf):.3f} '
           f'(uniform-phase would be 2/pi = 0.637 on average, and ~0 sometimes)')

    # (3c) the new sampler knobs are inert when off — RNG STREAM identity
    def stream(kw):
        e2 = Orbital(num_envs=1, **kw); out = []
        for k in range(400):
            e2.reset(seed=5000 + k)
            out.append(e2.get_state()[0].copy())
        e2.close(); return np.stack(out)
    OFF = dict(X3_KW, j2_mode=1, lvlh_frame_mode=1,
               i_target_min_rad=_m.radians(30), i_target_max_rad=_m.radians(60),
               legacy_action_space=31, episode_cap_steps=22000)
    A = stream(OFF)
    B = stream(dict(OFF, di_min_rad=-1.0, di_phase_mode=0))
    record('A5f di_min_rad / di_phase_mode present-but-off are a no-op',
           np.array_equal(A, B), f'400 resets, max |diff| = {np.abs(A - B).max():.3e}')

    # (3d) EXPLORABILITY of the appended row. A zero-init row is unreachable
    #      under a saturated warm start (median argmax softmax 0.986): measured
    #      P(row 30) = 3.9e-9, i.e. 0 expected samples in 100M decisions — the
    #      campaign would burn its whole budget with the mechanism under test
    #      never available. Seeding row 30 from row 17 fixes it. This gate is
    #      what stops that failing silently.
    warm = f'{WT}/models/t3/j2wait_warm_A2.pt'
    if os.path.exists(warm):
        env = Orbital(num_envs=64, **WAIT_KW)
        pol = LSTMWrapper(env, Default(env))
        pol.load_state_dict(torch.load(warm, map_location='cpu', weights_only=True))
        pol.eval()
        o, _ = env.reset(seed=5)
        st = {'lstm_h': torch.zeros(64, pol.hidden_size),
              'lstm_c': torch.zeros(64, pol.hidden_size)}
        p30 = []
        for _ in range(40):
            with torch.no_grad():
                lg, _ = pol.forward_eval(torch.from_numpy(np.asarray(o)).float(), st)
                pr = torch.softmax(lg, dim=-1)
            p30.append(pr[..., 30].numpy().ravel())
            o, _, _, _, _ = env.step(
                torch.argmax(lg, dim=-1).numpy().astype(np.int32).ravel())
        env.close()
        p30 = np.concatenate(p30)
        exp100m = float(p30.mean()) * 1e8
        record('A5g the day-warp row is EXPLORABLE under the warm start',
               exp100m > 1e5,
               f'P(row 30) mean {p30.mean():.3e}, median {np.median(p30):.3e}, '
               f'max {p30.max():.3e} -> {exp100m:,.0f} expected samples per 100M '
               f'decisions (a ZERO-init row measures 3.9e-9 -> 0; gate 1e5)')
    else:
        record('A5g the day-warp row is EXPLORABLE under the warm start', False,
               f'warm start missing: {warm} (build it with expand_ckpt_30_to_31.py)')

    # (4) the credit-horizon arithmetic this design depends on
    g = 0.995
    for tau, name in ((1440, 'day-warp  (row 30)'), (360, '6 h warp  (row 17)'),
                      (60, '1 h warp  (row 11)')):
        nd = 22000 / tau
        print(f'    a full 22000-substep cap via {name}: {nd:6.1f} decisions, '
              f'gamma^n = {g ** nd:6.4f}, terminal worth {10 * g ** nd:6.3f}')
    print('    Phi range is 1.8167 (W_lambda 1.0 + W_match 0.8167), so a stalled '
          'episode\n    can bank at most that much; success must beat it.')
    return {}


STAGES = {'a1': stage_a1, 'a2': stage_a2, 'a3': stage_a3, 'a4': stage_a4,
          'a5': stage_a5}


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
