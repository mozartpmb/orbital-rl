"""RED-TEAM E — reward-on-truth while the policy flies on estimates.

Two measurable questions:
  1. How much of the per-decision shaping delta is UNPREDICTABLE from what the
     policy sees?  Phi is computed env-side from the TRUE target elements; the
     policy sees estimated ones. Under partial observability, potential-based
     shaping is no longer policy-invariant and the unexplainable part becomes
     advantage noise the value head cannot cancel.
  2. What is the bound on the exploit? Phi is telescoping at shape_gamma=1.0,
     so the episode's total shaping return is exactly Phi_T - Phi_0 - a function
     of the truth endpoints only. Measure its range against the terminal reward.

Phi replica (orbital.h:804-819, shaping_mode=1):
    Phi = -[ W_lam*|wrap(lam_s - lam_t)|/pi + W_m*min(1, dv_match/DV_REF) ]
    lam = M + omega ; dv_match = 0.5*v_t*hypot(da/a_t, |e_vec_s - e_vec_t|)
"""
import csv
import math
import os
import sys

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

OUT = f"{ROOT}/web_data/results/ext_rtnav_phi_leak.csv"
CKPT = f"{ROOT}/models/t3/seed42_L2_headline.pt"
W_LAM, W_M, DV_REF, BETA_SHAPE = 1.0, 0.35, 300.0, 1.0
R_MIN, R_MAX = om.R_EARTH + 300e3, om.R_EARTH + 800e3


def phi(sat, tgt):
    dlam = ev.wrap_pi((om.mean_from_true(sat['theta'], sat['e']) + sat['omega'])
                      - (om.mean_from_true(tgt['theta'], tgt['e']) + tgt['omega']))
    esx, esy = sat['e'] * math.cos(sat['omega']), sat['e'] * math.sin(sat['omega'])
    etx, ety = tgt['e'] * math.cos(tgt['omega']), tgt['e'] * math.sin(tgt['omega'])
    de = math.hypot(esx - etx, esy - ety)
    da = (sat['a'] - tgt['a']) / tgt['a']
    v_t = math.sqrt(om.MU / tgt['a'])
    m = min(1.0, 0.5 * v_t * math.hypot(da, de) / DV_REF)
    return -(W_LAM * abs(dlam) / math.pi + W_M * m)


def unacq(sat, tgt):
    sx, sy, _, _ = om.orbit_to_cartesian(sat)
    tx, ty, _, _ = om.orbit_to_cartesian(tgt)
    b = math.atan2(ty - sy, tx - sx)
    lo, hi = prior_span(range_prior_intervals((sx, sy), b, R_MIN, R_MAX))
    rho = math.sqrt(max(lo, 1.0) * hi)
    px, py = sx + rho * math.cos(b), sy + rho * math.sin(b)
    vx, vy = circular_state(px, py)
    el = om.cartesian_to_elements(px, py, vx, vy)
    if not (0.0 <= el.get('e', 1.0) < 0.999) or el['a'] <= 0.0:
        return None
    el['M'] = om.mean_from_true(el['theta'], el['e'])
    return el


def perturb(tgt, sig_a, sig_e, sig_lam, rng):
    """Target with a filter-like error: sigma on a, on the e-vector, on lambda."""
    out = dict(tgt)
    out['a'] = tgt['a'] + rng.normal(0, sig_a)
    ex = tgt['e'] * math.cos(tgt['omega']) + rng.normal(0, sig_e)
    ey = tgt['e'] * math.sin(tgt['omega']) + rng.normal(0, sig_e)
    out['e'] = min(0.9, math.hypot(ex, ey))
    out['omega'] = math.atan2(ey, ex)
    lam = om.mean_from_true(tgt['theta'], tgt['e']) + tgt['omega'] \
        + rng.normal(0, sig_lam)
    out['M'] = (lam - out['omega']) % (2 * math.pi)
    out['theta'] = om.eccentric_to_true(om.solve_kepler(out['M'], out['e']),
                                        out['e'])
    return out


def main():
    n_eps = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    torch.set_num_threads(1)
    env = Orbital(**ev.T3_ENV_KWARGS)
    pol = LSTMWrapper(env, Default(env))
    pol.load_state_dict(torch.load(CKPT, map_location="cpu", weights_only=True))
    pol.eval()
    ev.PHASE_OBS_MODE = 1
    rng = np.random.default_rng(11)

    obs, _ = env.reset(seed=123)
    state = ev.zero_state(pol)
    eps, prev = 0, None
    d_true, d_est = {}, {}
    ep_phi0, ep_phiT = [], []
    # filter-error scales: settled pos RMSE 8 m (T4 nominal), 100x, and the
    # unacquired analytic prior
    LEVELS = [('ekf1x', 8.0, 1e-6, 8.0 / 6.9e6),
              ('ekf100x', 800.0, 1e-4, 800.0 / 6.9e6),
              ('bo_settled_3km', 3000.0, 4e-4, 3000.0 / 6.9e6),
              ('unacquired', None, None, None)]
    for L, _, _, _ in LEVELS:
        d_est[L] = []
    d_true = []
    while eps < n_eps:
        o = np.asarray(obs, dtype=np.float32).reshape(-1)
        sat, tgt = ev.recover_states_t3(o, 1.6e6)
        p_t = phi(sat, tgt)
        cur = {'truth': p_t}
        for L, sa, se, sl in LEVELS:
            est = unacq(sat, tgt) if sa is None else perturb(tgt, sa, se, sl, rng)
            cur[L] = phi(sat, est) if est is not None else p_t
        if prev is not None:
            d_true.append(BETA_SHAPE * (cur['truth'] - prev['truth']))
            for L, _, _, _ in LEVELS:
                d_est[L].append(BETA_SHAPE * (cur[L] - prev[L]))
        else:
            ep_phi0.append(p_t)
        prev = cur
        with torch.no_grad():
            lg, _ = pol.forward_eval(torch.from_numpy(o).float()
                                     .unsqueeze(0).unsqueeze(0), state)
            a = int(lg.argmax())
        obs, rew, term, trunc, _ = env.step(np.array([a], dtype=np.int32))
        if term[0]:
            ep_phiT.append(prev['truth'])
            eps += 1
            prev = None
            state = ev.zero_state(pol)

    dt = np.array(d_true)
    print(f"episodes {eps}, decisions {len(dt)}")
    print(f"\nper-decision shaping delta on TRUTH: "
          f"rms {dt.std():.5f}  p95|.| {np.percentile(np.abs(dt),95):.5f}  "
          f"max|.| {np.abs(dt).max():.5f}")
    print("\nhow much of that delta survives when Phi is evaluated on the "
          "ESTIMATE instead (i.e. the part a value head fed with the estimate "
          "could in principle predict):")
    rows = []
    for L, _, _, _ in LEVELS:
        de = np.array(d_est[L])
        res = dt - de
        r2 = 1.0 - (res.var() / max(dt.var(), 1e-18))
        print(f"  {L:<16} corr {np.corrcoef(dt, de)[0,1]:+.4f}  "
              f"explained-var {r2:+.4f}  unexplained rms {res.std():.5f} "
              f"({100*res.std()/max(dt.std(),1e-12):6.1f}% of the truth delta)")
        rows.append(dict(level=L, corr=float(np.corrcoef(dt, de)[0, 1]),
                         explained_var=float(r2), unexpl_rms=float(res.std()),
                         truth_rms=float(dt.std())))

    tot = np.array(ep_phiT) - np.array(ep_phi0[:len(ep_phiT)])
    print(f"\ntelescoped episode shaping return (Phi_T - Phi_0), "
          f"shape_gamma=1.0: p50 {np.median(tot):+.4f}  "
          f"min {tot.min():+.4f}  max {tot.max():+.4f}")
    print("terminal reward for success = 10*(0.5+0.5*fuel) in [5, 10]  "
          "→ shaping is at most "
          f"{100*abs(tot).max()/5.0:.1f}% of the smallest success payout")
    print("\nPhi is bounded in [-1.35, 0] by construction (orbital.h:817), so an "
          "estimate-based Phi could be gamed for at most +1.35 total.")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
