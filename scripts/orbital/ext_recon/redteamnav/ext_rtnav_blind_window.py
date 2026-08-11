"""RED-TEAM B/C — what the blind window actually costs, measured on the T3
canonical checkpoint, zero-shot.

NAV-G: "hold a coast/warp until the acquisition gate passes ... feeding the
policy an unacquired mixture mean risks an early wrong burn." Neither arm was
ever measured. This probe measures both, plus the acquisition DISCONTINUITY
that arm (C) worries about.

Conditions (all on models/t3/seed42_L2_headline.pt, greedy, held-out seed 123,
T3 env kwargs, truth after the window so the ONLY variable is the window):

  truth            control. no window.
  blind{N}         first N decisions: target-derived slots rebuilt from the
                   ANALYTIC RANGE PRIOR's log-mean along the true LOS (exactly
                   `rho0 = sqrt(lo*hi)` + circular velocity, i.e. NAV-G's own
                   pre-acquisition state). Policy acts on it freely.
  hold{N}          same blind obs, but the env OVERRIDES the action to warp-1h
                   for those N decisions (the "hold until acquired" arm).
  holdc{N}         same, override to COAST (tau=1) — the naive reading of
                   "hold a coast".
  jump{N}          truth obs everywhere EXCEPT one blind decision at t=N-1,
                   i.e. an isolated discontinuity with no sustained blindness.
                   Isolates arm (C): does the LSTM survive a single jump?

Reported: physical-success rate + cause histogram + Clopper-Pearson interval.
"""
import csv
import math
import os
import sys
from collections import Counter

import numpy as np
import torch

ROOT = "/Users/pete/space_training"
sys.path.insert(0, f"{ROOT}/pufferlib")
sys.path.insert(0, f"{ROOT}/scripts/orbital/nav")
sys.path.insert(0, f"{ROOT}/scripts/orbital/ext_recon")

from pufferlib.ocean.orbital.orbital import Orbital      # noqa: E402
from pufferlib.models import Default, LSTMWrapper        # noqa: E402
import orbital_math as om                                # noqa: E402
import eval_relnav as ev                                 # noqa: E402
from ext_bo_filter import range_prior_intervals, prior_span, circular_state  # noqa: E402

OUT = f"{ROOT}/web_data/results/ext_rtnav_blind_window.csv"
CKPT = f"{ROOT}/models/t3/seed42_L2_headline.pt"
CAUSES = ['none', 'success', 'collision', 'escape', 'safety_cap', 'stranded',
          'hyperbolic', 'gave_up']
R_MIN, R_MAX = om.R_EARTH + 300e3, om.R_EARTH + 800e3
WARP_1H, COAST = 11, 0
TGT = (7, 8, 11, 12, 13, 14, 16, 33, 34, 35, 36, 37)


def cp_interval(k, n):
    """Clopper-Pearson 95% two-sided."""
    from math import inf
    try:
        from scipy.stats import beta
        lo = 0.0 if k == 0 else beta.ppf(0.025, k, n - k + 1)
        hi = 1.0 if k == n else beta.ppf(0.975, k + 1, n - k)
        return lo, hi
    except Exception:
        p = k / n
        s = math.sqrt(max(p * (1 - p), 1e-12) / n)
        return max(0.0, p - 1.96 * s), min(1.0, p + 1.96 * s)


def unacquired_target(sat_el, tgt_el):
    """NAV-G's pre-acquisition state: log-mean of the analytic range prior along
    the true LOS, with a prograde circular velocity guess."""
    sx, sy, _, _ = om.orbit_to_cartesian(sat_el)
    tx, ty, _, _ = om.orbit_to_cartesian(tgt_el)
    beta = math.atan2(ty - sy, tx - sx)
    iv = range_prior_intervals((sx, sy), beta, R_MIN, R_MAX)
    lo, hi = prior_span(iv)
    rho0 = math.sqrt(max(lo, 1.0) * hi)
    px, py = sx + rho0 * math.cos(beta), sy + rho0 * math.sin(beta)
    vx, vy = circular_state(px, py)
    el = om.cartesian_to_elements(px, py, vx, vy)
    if not (0.0 <= el.get('e', 1.0) < 0.999) or el['a'] <= 0.0:
        return None, None
    el['M'] = om.mean_from_true(el['theta'], el['e'])
    return el, (px, py, vx, vy)


def run(kind, n_win, n_eps, env, pol, seed=123):
    ev.PHASE_OBS_MODE = 1
    ev.ENV_KWARGS = ev.T3_ENV_KWARGS
    obs, _ = env.reset(seed=seed)
    state = ev.zero_state(pol)
    causes = Counter()
    eps = 0
    k = 0                 # decisions since episode start
    overrides = 0
    blind_rho_err = []
    slot_err = []
    while eps < n_eps:
        o = np.asarray(obs, dtype=np.float32).reshape(-1)
        pol_obs = o
        close = None
        if kind.startswith('blindclose'):
            sat_el, tgt_el = ev.recover_states_t3(o, 1.6e6)
            sx, sy, _, _ = om.orbit_to_cartesian(sat_el)
            tx, ty, _, _ = om.orbit_to_cartesian(tgt_el)
            close = math.hypot(tx - sx, ty - sy) < n_win * 1e3
        in_win = (kind != 'truth' and (
            (kind.startswith('jump') and k == n_win - 1) or
            (kind.startswith('blindclose') and close) or
            (kind == 'blindall') or
            (kind.startswith('blind') and not kind.startswith('blindall')
             and not kind.startswith('blindclose') and k < n_win) or
            (kind.startswith('hold') and k < n_win)))
        if in_win:
            sat_el, tgt_el = ev.recover_states_t3(o, 1.6e6)
            est_el, est_c = unacquired_target(sat_el, tgt_el)
            if est_el is not None:
                pol_obs = ev.build_obs_t3(o, sat_el, est_el, 1.6e6, 6.371e6,
                                          tgt_cart=est_c)
                sx, sy, _, _ = om.orbit_to_cartesian(sat_el)
                tx, ty, _, _ = om.orbit_to_cartesian(tgt_el)
                blind_rho_err.append(
                    abs(math.hypot(est_c[0] - sx, est_c[1] - sy)
                        - math.hypot(tx - sx, ty - sy)))
                slot_err.append(np.abs(
                    np.asarray(pol_obs, dtype=np.float64)
                    - o.astype(np.float64))[list(TGT)])
        with torch.no_grad():
            lg, _ = pol.forward_eval(
                torch.from_numpy(np.ascontiguousarray(pol_obs)).float()
                .unsqueeze(0).unsqueeze(0), state)
            a = int(lg.argmax())
        if kind.startswith('hold') and k < n_win:
            a = COAST if kind.startswith('holdc') else WARP_1H
            overrides += 1
        obs, rew, term, trunc, _ = env.step(np.array([a], dtype=np.int32))
        k += 1
        if term[0]:
            _, cause = env.last_episode_result(0)
            causes[int(cause)] += 1
            eps += 1
            k = 0
            state = ev.zero_state(pol)
    return causes, overrides, blind_rho_err, slot_err


def main():
    n_eps = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    torch.set_num_threads(1)
    env = Orbital(**ev.T3_ENV_KWARGS)
    pol = LSTMWrapper(env, Default(env))
    pol.load_state_dict(torch.load(CKPT, map_location="cpu", weights_only=True))
    pol.eval()

    conds = [('truth', 0)]
    for n in (1, 2, 4, 8):
        conds.append((f'blind{n}', n))
    for n in (1, 2, 4, 8):
        conds.append((f'hold{n}', n))
    for n in (2, 8):
        conds.append((f'holdc{n}', n))
    for n in (1, 4):
        conds.append((f'jump{n}', n))
    for n in (16, 32):
        conds.append((f'blind{n}', n))
    conds.append(('blindall', 0))
    for n in (50, 200, 1000):
        conds.append((f'blindclose{n}', n))

    rows = []
    for kind, n in conds:
        causes, ov, rho_err, slot_err = run(kind, n, n_eps, env, pol)
        succ = causes[1]
        lo, hi = cp_interval(succ, n_eps)
        hist = ", ".join(f"{CAUSES[c]}={v}" for c, v in sorted(causes.items()))
        re_med = float(np.median(rho_err)) / 1e3 if rho_err else float('nan')
        se = np.asarray(slot_err) if slot_err else np.zeros((1, len(TGT)))
        se_med = float(np.median(se.max(axis=1)))
        se_a = float(np.median(se[:, 0]))   # obs[7] = a_target
        print(f"{kind:<10} N={n:<2} success {succ:>4}/{n_eps} "
              f"= {100*succ/n_eps:5.1f}%  [{100*lo:5.1f},{100*hi:5.1f}]  "
              f"| rho-err p50 {re_med:7.0f} km | max|dobs| p50 {se_med:6.3f} "
              f"| d obs[7] p50 {se_a:6.4f} | {hist}")
        rows.append(dict(cond=kind, n_win=n, n_eps=n_eps, success=succ,
                         rate=succ / n_eps, cp_lo=lo, cp_hi=hi,
                         overrides=ov, blind_rho_err_p50_km=re_med,
                         causes=hist))

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
