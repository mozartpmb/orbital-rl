#!/usr/bin/env python3
"""ext-nav verification ladder V1-V4.

Run from the worktree's pufferlib directory:

    cd /Users/pete/space_training-extnav/pufferlib
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
      python3 ../scripts/orbital/nav/verify_extnav.py --stage all

Stages
  v1  anchors: puffer_orbital untouched (legacy 26/200 + T3 canonical 200/200,
      by direct closed-loop replay), OrbitalNav(nav_mode='truth') closed loop
      200 eps == 200/200 with a bit-identical action stream (md5).
  v2  recon gate: encode/decode layer with zero estimation error vs truth
      (dim-wise residual, policy argmax mismatches, causes), plus rb_ekf at
      zero measurement noise vs recon.
  v3  rb_ekf at 1x noise, canonical ckpt zero-shot, 200 eps, + NEES.
  v4  bearings-only: surrogate calibration table, post-acquisition NEES,
      canonical ckpt zero-shot, separation-binned error table.

Every closed-loop cell reproduces `eval_checkpoint.py`'s protocol exactly:
num_envs=1, greedy argmax, held-out seed 123, LSTM zeroed per episode, success
= terminal cause == 1, gave-up inits excluded from the denominator.
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
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(WT, 'pufferlib'))

from pufferlib.ocean.orbital.orbital import Orbital                  # noqa: E402
from pufferlib.ocean.orbital_nav.orbital_nav import OrbitalNav       # noqa: E402
from pufferlib.ocean.orbital_nav import nav_math as nm               # noqa: E402
from pufferlib.ocean.orbital_nav import nav_surrogate as ns          # noqa: E402
from pufferlib.models import Default, LSTMWrapper                    # noqa: E402

# Checkpoints live in the MAIN checkout (read-only); `models/` is tracked and
# therefore present in this worktree too, `experiments/` is not.
MAIN = '/Users/pete/space_training'
T3_CKPT = f'{WT}/models/t3/seed42_L2_headline.pt'
LEGACY_CKPT = (f'{MAIN}/pufferlib/experiments/puffer_orbital_177765503091/'
               f'model_puffer_orbital_000325.pt')

CAUSES = ['none', 'success', 'collision', 'escape', 'safety_cap',
          'stranded', 'hyperbolic', 'gave_up']

# T3 headline eval conditions (t3_ladder.sh heval + T3_ENV).
T3_KW = dict(
    num_debris_min=0, num_debris_max=0,
    e_max_target=0.05, e_max_sat=0.05, same_orbit_init=0,
    init_phase_gap_max=3.14159, valid_init_only=1,
    gave_up_action='terminate', max_valid_init_attempts=4096,
    obs_alt_scale_m=1.6e6, lvlh_scale_m=6.371e6,
    shaping_mode=1, shape_gamma=1.0, phase_gap_mode=1, phase_obs_mode=1,
    episode_cap_steps=3000, cap_terminal_reward=0.0,
)

# Legacy anchor conditions (t1_readapt_v2.sh scan_one, seed 42).
LEGACY_KW = dict(
    num_debris_min=0, num_debris_max=0,
    e_max_target=0.05, e_max_sat=0.05, same_orbit_init=0,
    init_phase_gap_max=3.14159, valid_init_only=1,
    gave_up_action='terminate', max_valid_init_attempts=4096,
    legacy_action_space=10,
)


def load_policy(env, ckpt):
    p = LSTMWrapper(env, Default(env))
    p.load_state_dict(torch.load(ckpt, map_location='cpu', weights_only=True))
    p.eval()
    return p


def rollout(env, ckpt, episodes, seed, label='', probe=None, verbose=True):
    """eval_checkpoint.py protocol. Returns a result dict."""
    policy = load_policy(env, ckpt)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    obs, _ = env.reset(seed=seed)
    state = {'lstm_h': torch.zeros(1, policy.hidden_size),
             'lstm_c': torch.zeros(1, policy.hidden_size)}
    actions, causes, lengths = [], [], []
    probe_out = []
    n_done, k, t0 = 0, 0, time.time()
    while n_done < episodes:
        with torch.no_grad():
            logits, _ = policy.forward_eval(
                torch.from_numpy(np.asarray(obs)).float().unsqueeze(0), state)
            a = int(torch.argmax(logits, dim=-1).item())
        actions.append(a)
        if probe is not None:
            probe_out.append(probe(env, obs, a, state, policy))
        obs, rew, term, trunc, _ = env.step(np.array([a], dtype=np.int32))
        k += 1
        if term[0]:
            n_done += 1
            _, cause = env.last_episode_result(0)
            causes.append(int(cause))
            lengths.append(k)
            k = 0
            state = {'lstm_h': torch.zeros(1, policy.hidden_size),
                     'lstm_c': torch.zeros(1, policy.hidden_size)}
            if verbose and n_done % 50 == 0:
                sr = sum(1 for c in causes if c == 1) / n_done
                print(f'    [{label}] {n_done}/{episodes} success {sr:.1%} '
                      f'({time.time()-t0:.0f}s)', flush=True)
    env.close()
    causes = np.array(causes)
    n_gave = int((causes == 7).sum())
    n_valid = len(causes) - n_gave
    succ = int((causes == 1).sum())
    return dict(
        label=label, success=succ, n_valid=n_valid, gave_up=n_gave,
        rate=succ / max(n_valid, 1), causes=causes,
        cause_str=', '.join(f'{CAUSES[c]}={int((causes == c).sum())}'
                            for c in range(8) if (causes == c).any()),
        md5=hashlib.md5(np.array(actions, dtype=np.int32).tobytes()).hexdigest(),
        n_actions=len(actions), mean_len=float(np.mean(lengths)),
        probe=probe_out, wall=time.time() - t0)


def show(r):
    print(f"  {r['label']:26s} success {r['success']}/{r['n_valid']} = "
          f"{r['rate']:6.1%}   gave_up {r['gave_up']}   "
          f"decisions {r['n_actions']} (mean {r['mean_len']:.1f}/ep)   "
          f"md5 {r['md5'][:12]}   {r['wall']:.0f}s")
    print(f"  {'':26s} causes: {r['cause_str']}")


# ── V1 ───────────────────────────────────────────────────────────────────────
def stage_v1(args):
    print('== V1  puffer_orbital untouched + truth-mode passthrough ==========')
    print(f'  T3 ckpt     {T3_CKPT}')
    print(f'  legacy ckpt {LEGACY_CKPT}')

    print('\n-- legacy anchor (Phase-5e ckpt under corrected dynamics, seed 42, '
          'expect 26/200) --')
    r = rollout(Orbital(num_envs=1, **LEGACY_KW), LEGACY_CKPT, 200, 42,
                'legacy Orbital', verbose=False)
    show(r)

    print('\n-- T3 canonical anchor (seed 123, expect 200/200) --')
    a = rollout(Orbital(num_envs=1, **T3_KW), T3_CKPT, args.eps, 123,
                'T3 Orbital', verbose=False)
    show(a)

    print("\n-- OrbitalNav(nav_mode='truth') closed loop, same ckpt/seed --")
    b = rollout(OrbitalNav(num_envs=1, nav_mode='truth', **T3_KW), T3_CKPT,
                args.eps, 123, 'T3 OrbitalNav truth', verbose=False)
    show(b)
    same = (a['md5'] == b['md5'] and a['n_actions'] == b['n_actions']
            and np.array_equal(a['causes'], b['causes']))
    print(f"\n  ACTION STREAM IDENTICAL: {same}   "
          f"({a['n_actions']} vs {b['n_actions']} decisions, "
          f"md5 {a['md5'][:16]} vs {b['md5'][:16]})")
    return dict(legacy=r, orbital=a, nav_truth=b, identical=same)


# ── V2 ───────────────────────────────────────────────────────────────────────
def stage_v2(args):
    print('== V2  recon gate ================================================')
    n, steps = 64, 400
    rng = np.random.default_rng(0)
    acts = [rng.integers(0, 16, n).astype(np.int32) for _ in range(steps)]

    def roll(env):
        o, _ = env.reset(seed=42)
        out = [np.array(o, copy=True)]
        for a in acts:
            o, _, _, _, _ = env.step(a)
            out.append(np.array(o, copy=True))
        env.close()
        return np.stack(out)

    A = roll(Orbital(num_envs=n, seed=7, **T3_KW))
    C = roll(OrbitalNav(num_envs=n, seed=7, nav_mode='recon', **T3_KW))
    d = np.abs(A.astype(np.float64) - C.astype(np.float64))
    rows = d.shape[0] * d.shape[1]
    print(f'  open-loop obs residual over {rows} row-steps (random actions):')
    print(f'    bit-identical rows      {int((d.max(axis=2) == 0).sum())}/{rows} '
          f'= {(d.max(axis=2) == 0).mean():.2%}')
    print(f'    max |recon - truth|     {d.max():.3e}  '
          f'(dim {int(np.unravel_index(d.argmax(), d.shape)[2])})')
    print(f'    / eps_f32(1.0)          {d.max() / float(np.spacing(np.float32(1.0))):.2f}')
    mx = d.max(axis=(0, 1))
    print('    non-zero dims: '
          + ', '.join(f'[{i}]={mx[i]:.2e}' for i in range(38) if mx[i] > 0))

    # closed loop: recon and rb_ekf @ 0 noise must reproduce truth's causes
    print('\n  closed loop, canonical ckpt, seed 123:')
    out = {}
    for lbl, env in (
            ('truth', OrbitalNav(num_envs=1, nav_mode='truth', **T3_KW)),
            ('recon', OrbitalNav(num_envs=1, nav_mode='recon', **T3_KW)),
            ('rb_ekf noise=0', OrbitalNav(num_envs=1, nav_mode='rb_ekf',
                                          nav_noise_mult=0.0, **T3_KW))):
        r = rollout(env, T3_CKPT, args.v2_eps, 123, lbl, verbose=False)
        out[lbl] = r
        show(r)
    ok = all(np.array_equal(out['truth']['causes'], out[k]['causes'])
             for k in out)
    print(f"\n  CAUSES IDENTICAL TO TRUTH (all three): {ok}")
    print(f"  action-stream md5: "
          + '  '.join(f"{k}={v['md5'][:10]}" for k, v in out.items()))
    return dict(residual_max=float(d.max()), cells=out, causes_identical=ok)


# ── shared: closed loop with filter telemetry ────────────────────────────────
def _telemetry_probe(store):
    def probe(env, obs, a, state, policy):
        est = env._mean()
        tgt = env._prev_tgt
        sat = env._prev_sat
        d = est[:, :2] - tgt[:, :2]
        pe = float(np.hypot(d[0, 0], d[0, 1]))
        ve = float(np.hypot(est[0, 2] - tgt[0, 2], est[0, 3] - tgt[0, 3]))
        dl = tgt[0, :2] - sat[0, :2]
        rho = float(max(math.hypot(dl[0], dl[1]), 1.0))
        idx = np.arange(1)
        x, P = env._filt.mean_cov(idx)
        with np.errstate(all='ignore'):
            v = nm.nees(x, P, tgt)
            u = dl / rho
            slr = math.sqrt(max(float(u @ P[0, :2, :2] @ u), 0.0)) / rho
        acq = (bool(env._acq.acquired[0])
               if getattr(env, '_acq', None) is not None else True)
        store.append((pe, ve, rho, float(v[0]), slr, acq))
        return None
    return probe


def _telemetry_report(store, title, nees_lo=nm.NEES_LO, nees_hi=nm.NEES_HI):
    if not store:
        print('   (no telemetry)')
        return {}
    A = np.array([(p, v, r, n, s, float(q)) for p, v, r, n, s, q in store])
    pe, ve, rho, nees, slr, acq = (A[:, i] for i in range(6))
    fin = np.isfinite(nees)
    ib = float(((nees[fin] >= nees_lo) & (nees[fin] <= nees_hi)).mean())
    below = float((nees[fin] < nees_lo).mean())
    above = float((nees[fin] > nees_hi).mean())
    print(f'  {title}')
    print(f'    steps {len(A)}   acquired fraction {acq.mean():.3f}')
    print(f'    pos err  median {np.median(pe):10.1f} m   p95 {np.percentile(pe, 95):10.1f} m')
    print(f'    vel err  median {np.median(ve):10.4f} m/s p95 {np.percentile(ve, 95):10.4f} m/s')
    print(f'    NEES     median {np.nanmedian(nees[fin]):8.3f}   in-bounds {ib:.3f} '
          f'(below {below:.3f} / above {above:.3f})   bounds [{nees_lo:.3f}, {nees_hi:.3f}]')
    print(f'    sigma_LOS/rho median {np.median(slr[np.isfinite(slr)]):.4f}')
    edges = (0.0, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9)
    print(f'    {"range bin":>18s} {"n":>7s} {"med rho":>12s} {"med |dp|":>12s} '
          f'{"sb*rho":>12s} {"err/(sb*rho)":>13s} {"med |dv|":>12s}')
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (rho >= lo) & (rho < hi)
        if m.sum() < 20:
            continue
        mr, mp = float(np.median(rho[m])), float(np.median(pe[m]))
        tr = mr * nm.SIGMA_BETA_RAD
        rows.append((lo, hi, int(m.sum()), mr, mp, tr, mp / max(tr, 1e-12),
                     float(np.median(ve[m]))))
        print(f'    {lo:8.0e}-{hi:<8.0e} {int(m.sum()):7d} {mr/1e3:10.1f} km '
              f'{mp:10.1f} m {tr:10.1f} m {mp/max(tr,1e-12):13.3f} '
              f'{float(np.median(ve[m])):10.4f} m/s')
    return dict(nees_in_bounds=ib, nees_below=below, nees_above=above,
                pos_med=float(np.median(pe)), vel_med=float(np.median(ve)),
                acq_frac=float(acq.mean()), bins=rows)


# ── V3 ───────────────────────────────────────────────────────────────────────
def stage_v3(args):
    print('== V3  rb_ekf @ 1x noise, canonical ckpt zero-shot ================')
    store = []
    env = OrbitalNav(num_envs=1, nav_mode='rb_ekf', nav_noise_mult=1.0, **T3_KW)
    r = rollout(env, T3_CKPT, args.eps, 123, 'rb_ekf 1x',
                probe=_telemetry_probe(store), verbose=True)
    show(r)
    t = _telemetry_report(store, 'filter telemetry (decision epochs):')
    return dict(run=r, tel=t)


# ── V4 ───────────────────────────────────────────────────────────────────────
def stage_v4(args):
    print('== V4  bearings-only ==============================================')
    print('\n-- surrogate calibration: design map vs each scenario\'s own CRLB --')
    ns.map_vs_scenario_crlb()

    print('\n-- surrogate draw vs the measured BO-BLS-MPC error, per geometry --')
    _surrogate_qq(args)

    print('\n-- closed loop, canonical ckpt zero-shot --')
    store = []
    env = OrbitalNav(num_envs=1, nav_mode='bearings_only', **T3_KW)
    env._nav_alloc()
    env._acq.log_draws = []
    r = rollout(env, T3_CKPT, args.eps, 123, 'bearings_only 1x',
                probe=_telemetry_probe(store), verbose=True)
    D = np.array(env._acq.log_draws) if env._acq.log_draws else np.zeros((0, 5))
    if len(D):
        ratio = D[:, 2] / np.maximum(D[:, 1], 1e-9)
        print(f'\n  handoff draws: n={len(D)}   realised |dpos| / predicted sigma_pos:')
        print(f'    p10 {np.percentile(ratio,10):.2f}  p50 {np.percentile(ratio,50):.2f}  '
              f'p90 {np.percentile(ratio,90):.2f}   (Rayleigh reference 0.46/1.18/2.15)')
        print(f'    predicted sigma_pos median {np.median(D[:,1]):.1f} m   '
              f'sigma_LOS median {np.median(D[:,0]):.1f} m   '
              f'rho median {np.median(D[:,3])/1e3:.1f} km   '
              f'n_obs median {np.median(D[:,4]):.0f}')
    show(r)
    t = _telemetry_report(store, 'filter telemetry (decision epochs, all):')
    A = np.array([(p, v, r_, n, s, float(q)) for p, v, r_, n, s, q in store])
    post = A[A[:, 5] > 0.5]
    if len(post):
        fin = np.isfinite(post[:, 3])
        ib = float(((post[fin, 3] >= nm.NEES_LO) & (post[fin, 3] <= nm.NEES_HI)).mean())
        print(f'\n    POST-ACQUISITION NEES in-bounds {ib:.3f} over {int(fin.sum())} '
              f'epochs (gate >= 0.85)')
        print(f'    post-acq pos err median {np.median(post[:, 0]):.1f} m  '
              f'p95 {np.percentile(post[:, 0], 95):.1f} m')
    return dict(run=r, tel=t)


def _surrogate_qq(args, n_draw=4000):
    """Replay each NAV-G geometry's acquisition through the surrogate draw and
    compare the realised error distribution against the measured settled error.

    The surrogate hands off at the ACQUISITION arc; `pos_rmse_settled_m` is the
    measured error after the recursive stage has run the full arc. The handoff
    is therefore expected to be the looser of the two — the ratio column is the
    factor the live modified-polar filter has to buy back, and V4's closed-loop
    post-acquisition error is where that is actually tested.
    """
    t = ns.BLSRatioTable()
    rng = np.random.default_rng(1234)
    # (scenario, epoch separation m, acquisition latency min, dv in arc m/s)
    geoms = [('G6_leo_5km_1ms_burns', 5e3, 44, 1.0),
             ('G5_leo_10km_burns', 10e3, 44, 5.0),
             ('G3_leo_180deg', 13000e3, 44, 0.0),
             ('G2_leo_300km', 300e3, 71, 0.0),
             ('G4_wide_e30_90deg', 9771e3, 71, 0.0),
             ('G1_leo_10km_drift', 10e3, 114, 0.0)]
    print(f"  {'geometry':24s} {'rho/r':>8s} {'brn':>4s} {'CRLB':>9s} "
          f"{'ratio':>6s} {'sig_LOS':>10s} {'sig_trans':>10s} {'sig_tot':>10s} "
          f"{'drawRMS':>10s} {'measured':>10s} {'pred/meas':>10s}")
    out = []
    for name, sep0, lat, dv in geoms:
        m = t.rows[name]
        crlb = m['crlb']
        ratio = float(t.ratio(m['rho_over_r'], dv > 0))
        sig = ratio * crlb
        # exactly the construction in AcquisitionSurrogate.draw()
        sig_tr = m['rho_km'] * 1e3 * nm.SIGMA_BETA_RAD / math.sqrt(lat)
        z = rng.standard_normal((n_draw, 2))
        e = np.hypot(z[:, 0] * sig, z[:, 1] * sig_tr)
        tot = math.hypot(sig, sig_tr)
        drms = float(np.sqrt(np.mean(e ** 2)))
        out.append((name, tot, drms, m['pos']))
        print(f"  {name:24s} {m['rho_over_r']:8.4f} "
              f"{'yes' if dv else 'no':>4s} {crlb:7.1f} m {ratio:6.2f} "
              f"{sig:8.1f} m {sig_tr:8.1f} m {tot:8.1f} m {drms:8.1f} m "
              f"{m['pos']:8.1f} m {tot/m['pos']:10.2f}")
    r = np.array([o[1] / o[3] for o in out])
    print(f"  predicted total sigma / measured settled RMSE: "
          f"median {np.median(r):.2f}  min {r.min():.2f}  max {r.max():.2f}")
    print("  (1.0 = the surrogate's handoff covariance reproduces the shipped "
          "BO-BLS-MPC settled error)")
    return out


# ── V6 ───────────────────────────────────────────────────────────────────────
def stage_v6(args):
    """Does the T4 sensor-cadence finding reproduce inside the wrapper?

    The wrapper runs navigation at a fixed 60 s cadence, sub-propagating both
    truth states through warps, DECOUPLED from the guidance cadence. Welding
    them ('perdec': one measurement per decision, the filter propagated by the
    whole tau*60 s) is what T3/T4 measured at 50.5% closed loop against 100%.
    That mode is reachable here only via nav_sensor_dt=0 and is never used for
    training; this cell exists so the finding is reproducible in the wrapper
    rather than only in the serial harness.
    """
    print('== V6  sensor cadence is decoupled from the decision cadence ======')
    for lbl, dt in (('nav60 (shipped)', 60.0), ('perdec (welded, OFF)', 0.0)):
        env = OrbitalNav(num_envs=1, nav_mode='rb_ekf', nav_sensor_dt=dt, **T3_KW)
        show(rollout(env, T3_CKPT, args.v6_eps, 123, lbl, verbose=False))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--stage', default='all',
                   choices=['v1', 'v2', 'v3', 'v4', 'v6', 'all'])
    p.add_argument('--v6-eps', type=int, default=100)
    p.add_argument('--eps', type=int, default=200)
    p.add_argument('--v2-eps', type=int, default=60)
    args = p.parse_args()
    torch.set_num_threads(1)
    fns = dict(v1=stage_v1, v2=stage_v2, v3=stage_v3, v4=stage_v4, v6=stage_v6)
    order = (['v1', 'v2', 'v3', 'v4', 'v6'] if args.stage == 'all'
             else [args.stage])
    for s in order:
        fns[s](args)
        print()


if __name__ == '__main__':
    main()
