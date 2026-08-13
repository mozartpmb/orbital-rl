#!/usr/bin/env python3
"""Per-arm evaluation for the T5 NB (bearings-only) training campaign.

Three blocks per arm, all on the held-out seed 123 at the T3 headline config:

  (a) NATIVE   `eval_relnav` bearings path with the REAL batch angles-only
               acquisition (`bls_acquire_adaptive`) — success, never-acquire,
               close-range covariance gate. Number to beat: the canonical
               truth-trained checkpoint's 69.5% zero-shot.
  (b) TRUTH    the same checkpoint on the plain truth observation. Answers the
               question a reviewer actually asks: did learning to fly on a
               filter cost anything when the filter is perfect? Gate <= 2 pp.
  (c) BEHAVIOUR the blind-window mechanism. The canonical policy, fed a
               ~9,000 km-wrong prior, dumps its whole 478 m/s budget in ~16 min
               of sim time and strands itself before any angles-only solver
               could converge. This block measures whether training fixed that:
               delta-v spent before acquisition, fuel left at acquisition,
               stranded-while-blind rate, and burn / delta-v rate inside the
               blind window versus outside it.

Usage (from the worktree's pufferlib dir):
    python3 ../scripts/orbital/nav/nb_eval.py --ckpt PATH --label nb1_warm \
        [--eps 200] [--skip-native]
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
WT = os.path.abspath(os.path.join(_HERE, '..', '..', '..'))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(WT, 'pufferlib'))

import eval_relnav as ER                                        # noqa: E402
from pufferlib.ocean.orbital.orbital import Orbital             # noqa: E402

sys.path.insert(0, _HERE)
from verify_extnav import T3_KW, rollout, show                  # noqa: E402

CANONICAL_NATIVE = 0.695     # 139/200, measured in V4 / eval_relnav
CANONICAL_TRUTH = 1.000      # 200/200


def run_arm(ckpt, label, eps=200, seed=123, skip_native=False):
    out = {'label': label, 'ckpt': ckpt}

    # ── (b) truth tax ────────────────────────────────────────────────────────
    r = rollout(Orbital(num_envs=1, **T3_KW), ckpt, eps, seed,
                f'{label} truth', verbose=False)
    show(r)
    out['truth_rate'] = r['rate']
    out['truth_causes'] = r['cause_str']
    out['truth_tax_pp'] = 100.0 * (CANONICAL_TRUTH - r['rate'])

    if skip_native:
        return out

    # ── (a) native, REAL batch acquisition ───────────────────────────────────
    ER.ENV_KWARGS = ER.T3_ENV_KWARGS
    ER.PHASE_OBS_MODE = 1
    ER.CKPT = ckpt
    ER.SENSOR_DT = 60.0
    torch.set_num_threads(1)
    b = ER.run_bo(eps, noise_scale=1.0, seed=seed, label=f'{label} bo')
    ER._bo_report(b)
    out['native_rate'] = float(b['success'].mean())
    out['native_causes'] = ', '.join(
        f'{ER.CAUSE_NAMES[c]}={int((b["cause"] == c).sum())}'
        for c in range(8) if (b['cause'] == c).any())
    out['never_acquire'] = int((b['blind_dec'] == b['dec']).sum())
    out['acq_frac'] = float(b['flat_acq'].mean())
    out['blind_dec_frac'] = float(b['blind_dec'].sum() / max(b['dec'].sum(), 1))
    out['n_diverge'] = int(b['n_diverge'])
    out['acq_fail'] = int(b['n_acq_fail'])
    if b['acq_epoch_err'].size:
        out['acq_epoch_err_med_km'] = float(np.median(b['acq_epoch_err']) / 1e3)
        out['acq_epoch_err_max_km'] = float(b['acq_epoch_err'].max() / 1e3)
        # BLOCKER-2: seconds, not decisions and not ticks.
        out['acq_latency_med_s'] = float(np.median(b['acq_latency_s']))
    close = b['minrho'] < 2e5
    out['close200_succ'] = (f"{int(b['success'][close].sum())}/{int(close.sum())}"
                            if close.any() else 'n/a')
    rho, slr = b['flat_rho'], b['flat_slr']
    m = (rho >= 1e5) & (rho < 1e6)
    out['sigma_los_over_rho_1e5_1e6'] = (float(np.median(slr[m])) if m.sum() > 10
                                         else float('nan'))
    out.update({f'beh_{k}': v for k, v in ER.behaviour_report(b, quiet=True).items()})
    out['native_delta_pp'] = 100.0 * (out['native_rate'] - CANONICAL_NATIVE)
    out['PASS'] = bool(out['native_rate'] >= 0.85 and out['truth_tax_pp'] <= 2.0)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True)
    p.add_argument('--label', required=True)
    p.add_argument('--eps', type=int, default=200)
    p.add_argument('--seed', type=int, default=123)
    p.add_argument('--skip-native', action='store_true')
    p.add_argument('--json-out', default=None)
    a = p.parse_args()
    torch.set_num_threads(1)
    print(f'===== NB arm {a.label} =====')
    print(f'  ckpt {a.ckpt}')
    res = run_arm(a.ckpt, a.label, a.eps, a.seed, a.skip_native)
    print('\nSUMMARY ' + json.dumps(res, sort_keys=True))
    if a.json_out:
        with open(a.json_out, 'w') as f:
            json.dump(res, f, indent=2, sort_keys=True)


if __name__ == '__main__':
    main()
