#!/usr/bin/env python3
"""NAV-F dual-control detection metrics.

The question: at the tight terminal box, does training on bearings-only
estimates produce *information-seeking* burns? NAV-F §3.1 measured that the TB5
policy already burns on ~29% of decisions inside 10 km for guidance reasons, so
**presence-of-burn is not a valid metric**. Only placement, timing, direction
and counterfactual information are.

Metrics implemented, in the report's priority order:

  1. COUNTERFACTUAL INFORMATION GAIN (primary). At every decision epoch, replay
     the geometry forward over a fixed horizon under (a) a standardised 1 m/s
     probe burn and (b) coast, accumulating the exact bearings-only Fisher
     information, and take
         G_avail = (sigma_coast / sigma_burn)^2
     — how much information a burn WOULD buy in this state. Then regress the
     binary "did the policy burn here" on log10(G_avail). A positive,
     significant slope for T-BO with a null for T-truth is the clean detection.

     The regressor must be STATE-dependent, not action-dependent: replaying only
     the action actually taken makes the regressor identically zero on every
     coast and the regression degenerates. G_taken (the information the executed
     action actually bought) is reported alongside as a secondary statistic.

  2. Information-weighted placement in matched (rho, delta-a, t-to-go,
     delta-v-left) bins, tested against T-truth.
  3. Burn direction relative to the instantaneous LOS. NAV-F §2.5: radial
     (90/270 deg) is the WORST direction at the multi-hour arcs the policy
     flies, and the along-track fine burns 12-15 are 1.3-4.7x better.
  4. Paired-episode excess delta-v (same seed => same scenario per index).
  5. Delta tr(P) over K updates after a burn vs matched no-burn windows.
  6. Success + timeout rate (the pre-registered `cap_terminal_reward = 0`
     stalling confound).

The FIM is accumulated exactly as `ext_bo_filter.crlb_range_sigma` does, which
is the same estimator NAV-F's own tables were built with. The target's state
transition matrix chain is IDENTICAL between the two branches (a chaser burn
does not move the target), so it is computed once and shared — the two branches
differ only in the measurement rows.
"""

import argparse
import math
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
WT = os.path.abspath(os.path.join(_HERE, '..', '..', '..'))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(WT, 'pufferlib'))

from pufferlib.ocean.orbital_nav import nav_math as nm       # noqa: E402

SIGMA_BETA = nm.SIGMA_BETA_RAD
# Arc geometry for the counterfactual. NAV-F §2.5 measured d(info)/d(epoch)
# sharply peaked at ~20% into the arc, so the probe burn sits at pre/(pre+post)
# = 0.25 of a 4 h window at the 60 s nav cadence.
PRE_TICKS = 60             # 1 h before the decision epoch
POST_TICKS = 180           # 3 h after
PROBE_DV = 1.0             # m/s — NAV-F's canonical observability quantum
FINE_BURNS = (12, 13, 14, 15, 18, 19)


def is_burn(a):
    a = np.asarray(a)
    return (a != 0) & (nm.ACTION_TAU[a] == 1)


def apply_impulse(S, dv_pro, dv_rad):
    """Batched port of orbital.h:apply_impulse (frame only, no fuel model).

    prograde = velocity direction, radial = position direction — oblique at
    e > 0, which is the env's own convention and must be reproduced exactly.
    """
    x, y, vx, vy = S[:, 0], S[:, 1], S[:, 2], S[:, 3]
    vm = np.maximum(np.hypot(vx, vy), 1e-9)
    rm = np.maximum(np.hypot(x, y), 1e-9)
    dvx = dv_pro * vx / vm + dv_rad * x / rm
    dvy = dv_pro * vy / vm + dv_rad * y / rm
    out = S.copy()
    out[:, 2] += dvx
    out[:, 3] += dvy
    return out


def sigma_range_batch(S0, G0, dv=(0.0, 0.0), pre=PRE_TICKS, post=POST_TICKS,
                      dt=60.0, shared=None):
    """1-sigma LOS range uncertainty at the DECISION EPOCH, batched.

    The arc runs from `pre` ticks BEFORE the decision epoch to `post` ticks
    after, and the probe impulse is applied AT the epoch — i.e. at
    pre/(pre+post) of the way through. That placement is not a detail:
    NAV-F §2.5 measured the information gain of a 1 m/s burn at 1.06 when the
    burn sits at the START of the arc, rising to 1758 at 20% in, because a burn
    at the arc start merely redefines the initial state and carries no
    pre/post contrast. Accumulating over [epoch, epoch+H] therefore measures
    ~1.0 for every state by construction and the whole metric reads null.

    Both truth states are back-propagated exactly (two-body flow is reversible),
    the Fisher information is accumulated in arc-start coordinates, and the
    covariance is mapped to the decision epoch with the accumulated STM so the
    reported sigma is the uncertainty in the quantity the policy acts on.

    `shared` caches the target trajectory and its STM chain: a chaser impulse
    does not move the target, so both branches reuse them.
    """
    n = S0.shape[0]
    total = pre + post
    # back-propagate to the arc start
    Sa, _ = nm.propagate_cartesian(S0, -pre * dt)
    if shared is None:
        Ga, _ = nm.propagate_cartesian(G0, -pre * dt)
        Gs, Phis = [Ga], [np.tile(np.eye(4), (n, 1, 1))]
        g, Phi = Ga, np.tile(np.eye(4), (n, 1, 1))
        for _ in range(total):
            F, _, g = nm.stm_fd(g, dt)
            Phi = F @ Phi
            Gs.append(g)
            Phis.append(Phi)
        shared = (Gs, Phis)
    Gs, Phis = shared

    FIM = np.zeros((n, 4, 4))
    s = Sa
    for k in range(total + 1):
        if k > 0:
            s, _ = nm.propagate_cartesian(s, dt)
        if k == pre and (np.any(dv[0]) or np.any(dv[1])):
            s = apply_impulse(s, dv[0], dv[1])
        dx = Gs[k][:, 0] - s[:, 0]
        dy = Gs[k][:, 1] - s[:, 1]
        r2 = np.maximum(dx * dx + dy * dy, 1.0)
        H = np.zeros((n, 1, 4))
        H[:, 0, 0] = -dy / r2
        H[:, 0, 1] = dx / r2
        H = H @ Phis[k]
        FIM += (np.swapaxes(H, 1, 2) @ H) / SIGMA_BETA ** 2

    w, V = np.linalg.eigh(0.5 * (FIM + np.swapaxes(FIM, 1, 2)))
    wmax = np.maximum(w.max(axis=1, keepdims=True), 1e-300)
    w = np.maximum(w, 1e-14 * wmax)
    C = (V * (1.0 / w)[:, None, :]) @ np.swapaxes(V, 1, 2)
    # map arc-start covariance to the decision epoch
    Pe = Phis[pre]
    C = Pe @ C @ np.swapaxes(Pe, 1, 2)
    d0 = G0[:, :2] - S0[:, :2]
    rho0 = np.maximum(np.hypot(d0[:, 0], d0[:, 1]), 1.0)
    u0 = d0 / rho0[:, None]
    v = np.einsum('ni,nij,nj->n', u0, C[:, :2, :2], u0)
    return np.sqrt(np.maximum(v, 0.0)), shared


def counterfactual(records, chunk=2048):
    """G_avail and G_taken for every recorded decision."""
    S = np.array([r['sat'] for r in records])
    G = np.array([r['tgt'] for r in records])
    A = np.array([r['action'] for r in records])
    n = len(records)
    g_av = np.zeros(n)
    g_tk = np.zeros(n)
    for i in range(0, n, chunk):
        sl = slice(i, min(i + chunk, n))
        s0, g0, a0 = S[sl], G[sl], A[sl]
        sig_c, shared = sigma_range_batch(s0, g0, (0.0, 0.0))
        sig_p, _ = sigma_range_batch(s0, g0, (PROBE_DV, 0.0), shared=shared)
        g_av[sl] = (sig_c / np.maximum(sig_p, 1e-12)) ** 2
        # the delta-v the executed action actually applied, in the local frame
        dvp = np.array([nm.ACTION_DV_MAG[a] if a in (1, 2, 3, 12, 14) else
                        (-nm.ACTION_DV_MAG[a] if a in (4, 5, 6, 13, 15) else 0.0)
                        for a in a0])
        dvr = np.array([nm.ACTION_DV_MAG[a] if a in (7, 18) else
                        (-nm.ACTION_DV_MAG[a] if a in (8, 19) else 0.0)
                        for a in a0])
        act = (np.abs(dvp) + np.abs(dvr)) > 0
        if act.any():
            sig_t, _ = sigma_range_batch(s0, g0, (dvp, dvr), shared=shared)
            g_tk[sl] = np.where(act, (sig_c / np.maximum(sig_t, 1e-12)) ** 2, 1.0)
        else:
            g_tk[sl] = 1.0
    return g_av, g_tk


def burn_direction(records):
    """Angle of the applied delta-v from the instantaneous LOS, degrees [0,180].

    NAV-F §2.5: the optimal direction is perpendicular to the LOS; radial is the
    WORST at multi-hour arcs. 0/180 = along the LOS, 90 = perpendicular.
    """
    ang, acts = [], []
    for r in records:
        a = r['action']
        if not is_burn(a):
            continue
        S = r['sat']
        vm = max(math.hypot(S[2], S[3]), 1e-9)
        rm = max(math.hypot(S[0], S[1]), 1e-9)
        m = nm.ACTION_DV_MAG[a]
        if a in (1, 2, 3, 12, 14):
            dv = (m * S[2] / vm, m * S[3] / vm)
        elif a in (4, 5, 6, 13, 15):
            dv = (-m * S[2] / vm, -m * S[3] / vm)
        elif a in (7, 18):
            dv = (m * S[0] / rm, m * S[1] / rm)
        elif a in (8, 19):
            dv = (-m * S[0] / rm, -m * S[1] / rm)
        else:
            continue
        d = r['tgt'][:2] - S[:2]
        n_ = max(math.hypot(d[0], d[1]), 1e-9)
        dn = max(math.hypot(dv[0], dv[1]), 1e-12)
        c = (dv[0] * d[0] + dv[1] * d[1]) / (dn * n_)
        ang.append(math.degrees(math.acos(max(-1.0, min(1.0, c)))))
        acts.append(a)
    return np.array(ang), np.array(acts)


def logistic_fit(X, y, iters=200, l2=1e-4):
    """Newton-IRLS logistic regression. Returns (beta, se, z)."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    X = np.column_stack([np.ones(len(y)), X])
    b = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-np.clip(X @ b, -30, 30)))
        W = np.maximum(p * (1 - p), 1e-9)
        Hm = X.T @ (X * W[:, None]) + l2 * np.eye(X.shape[1])
        gr = X.T @ (y - p) - l2 * b
        try:
            step = np.linalg.solve(Hm, gr)
        except np.linalg.LinAlgError:
            break
        b = b + step
        if np.abs(step).max() < 1e-10:
            break
    p = 1.0 / (1.0 + np.exp(-np.clip(X @ b, -30, 30)))
    W = np.maximum(p * (1 - p), 1e-9)
    try:
        cov = np.linalg.inv(X.T @ (X * W[:, None]) + l2 * np.eye(X.shape[1]))
        se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    except np.linalg.LinAlgError:
        se = np.full_like(b, np.nan)
    return b, se, b / np.maximum(se, 1e-12)


def analyse(records, label, close_m=10e3, verbose=True):
    """Full metric block for one arm."""
    out = {'label': label, 'n_dec': len(records)}
    if not records:
        return out
    g_av, g_tk = counterfactual(records)
    A = np.array([r['action'] for r in records])
    rho = np.array([r['rho'] for r in records])
    da = np.array([abs(r['da']) for r in records])
    dvl = np.array([r['dv_left'] for r in records])
    stp = np.array([r['step'] for r in records])
    acq = np.array([r['acquired'] for r in records])
    b = is_burn(A).astype(float)
    lg = np.log10(np.maximum(g_av, 1e-12))

    # steady state only: acquisition-phase burns are a separate population
    ss = acq == 1
    close = (rho < close_m) & ss
    out['n_close_ss'] = int(close.sum())
    out['burn_rate_close'] = float(b[close].mean()) if close.any() else float('nan')
    out['logG_close_med'] = float(np.median(lg[close])) if close.any() else float('nan')

    # ── 1. primary regression, steady state, inside the close regime ────────
    for tag, m in (('close', close), ('all_ss', ss)):
        if m.sum() < 50 or len(np.unique(b[m])) < 2:
            out[f'beta_{tag}'] = float('nan')
            out[f'z_{tag}'] = float('nan')
            continue
        # univariate on log10 G_avail
        bb, se, z = logistic_fit(lg[m][:, None], b[m])
        out[f'beta_{tag}'] = float(bb[1])
        out[f'z_{tag}'] = float(z[1])
        # multivariate with the §3.3-2 matching covariates
        Z = np.column_stack([lg[m], np.log10(np.maximum(rho[m], 1.0)),
                             np.log10(np.maximum(da[m], 1.0)),
                             dvl[m] / 478.0, stp[m] / 100.0])
        bb2, se2, z2 = logistic_fit(Z, b[m])
        out[f'beta_{tag}_adj'] = float(bb2[1])
        out[f'z_{tag}_adj'] = float(z2[1])

    # ── secondary: information the taken burns actually bought ──────────────
    bm = (b > 0) & ss
    out['G_taken_med_burns'] = (float(np.median(g_tk[bm])) if bm.any()
                                else float('nan'))
    out['G_avail_med_burns'] = (float(np.median(g_av[bm])) if bm.any()
                                else float('nan'))
    out['G_avail_med_coasts'] = (float(np.median(g_av[(b == 0) & ss]))
                                 if ((b == 0) & ss).any() else float('nan'))

    # ── 3. direction ────────────────────────────────────────────────────────
    ang, acts = burn_direction([r for r, m in zip(records, ss) if m])
    if ang.size:
        out['dir_med_deg'] = float(np.median(ang))
        out['dir_frac_radialish'] = float(np.mean((ang > 45) & (ang < 135)))
        out['frac_fine_burns'] = float(np.mean(np.isin(acts, FINE_BURNS)))
        out['n_burns_ss'] = int(ang.size)
    # action mix inside the close regime
    if close.any():
        mix = {}
        for a in np.unique(A[close]):
            mix[int(a)] = float(np.mean(A[close] == a))
        out['action_mix_close'] = dict(sorted(mix.items(),
                                              key=lambda kv: -kv[1])[:6])

    # ── 5. filter response to burns ─────────────────────────────────────────
    trP = np.array([r['trP'] for r in records])
    ep = np.array([r['ep'] for r in records])
    K = 3
    dpost_b, dpost_c = [], []
    for i in range(len(records) - K):
        if ep[i + K] != ep[i] or not ss[i]:
            continue
        d = math.log10(max(trP[i + K], 1e-30)) - math.log10(max(trP[i], 1e-30))
        (dpost_b if b[i] > 0 else dpost_c).append(d)
    if dpost_b and dpost_c:
        out['dlog_trP_after_burn'] = float(np.median(dpost_b))
        out['dlog_trP_after_coast'] = float(np.median(dpost_c))

    if verbose:
        _print_block(out)
    return out


def _print_block(o):
    print(f"  [{o['label']}] decisions {o['n_dec']}   steady-state inside 10 km "
          f"{o.get('n_close_ss', 0)}")
    print(f"    burn rate inside 10 km            {o.get('burn_rate_close', float('nan')):.4f}")
    print(f"    log10 G_avail inside 10 km (med)  {o.get('logG_close_med', float('nan')):.3f}")
    print(f"    PRIMARY  logit(burn) ~ log10 G_avail   inside 10 km: "
          f"beta {o.get('beta_close', float('nan')):+.4f}  z {o.get('z_close', float('nan')):+.2f}"
          f"   | adjusted beta {o.get('beta_close_adj', float('nan')):+.4f}"
          f"  z {o.get('z_close_adj', float('nan')):+.2f}")
    print(f"             all steady state:            "
          f"beta {o.get('beta_all_ss', float('nan')):+.4f}  z {o.get('z_all_ss', float('nan')):+.2f}"
          f"   | adjusted beta {o.get('beta_all_ss_adj', float('nan')):+.4f}"
          f"  z {o.get('z_all_ss_adj', float('nan')):+.2f}")
    print(f"    G_avail median   at burns {o.get('G_avail_med_burns', float('nan')):.3g}"
          f"   at coasts {o.get('G_avail_med_coasts', float('nan')):.3g}")
    print(f"    G_taken median   at burns {o.get('G_taken_med_burns', float('nan')):.3g}")
    if 'dir_med_deg' in o:
        print(f"    burn direction from LOS: median {o['dir_med_deg']:.1f} deg   "
              f"frac in 45-135 (perp-ish, GOOD) {o['dir_frac_radialish']:.3f}   "
              f"frac fine burns {o['frac_fine_burns']:.3f}   n {o['n_burns_ss']}")
    if 'action_mix_close' in o:
        print(f"    action mix inside 10 km: {o['action_mix_close']}")
    if 'dlog_trP_after_burn' in o:
        print(f"    dlog10 tr(P) over 3 decisions: after burn "
              f"{o['dlog_trP_after_burn']:+.3f}   after coast "
              f"{o['dlog_trP_after_coast']:+.3f}")
