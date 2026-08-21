#!/usr/bin/env python3
"""Task 1: the CRLB-online acquisition surrogate vs the REAL batch IOD, at
W1 drift-and-wait arcs.

APPLES TO APPLES BY CONSTRUCTION (self-red-team (i)). `RealAcq3D` is a
declared drop-in for `AcquisitionSurrogate` with the same call surface, so both
arms run inside the SAME `OrbitalNav`, on the SAME episodes, from the SAME
seed, and both are asked the same question through the same gate:

    sigma_LOS / rho <= acq_gate (0.20)   AND   elapsed >= acq_min_sec (2700 s)

The real solver adds the two gates a batch IOD must also pass and the surrogate
has no analogue for — a chi-square on the batch residual (cost <= dof +
3*sqrt(2*dof), dof = 2w - 6) and an ambiguity margin (>= 16.0 against the best
solution in a different range basin). Those are not extra strictness bolted on
for this comparison; they are what "the batch solver would have converged"
actually means, and the surrogate's whole claim is to predict their outcome.

WHAT SEPARATES THE TWO FAILURE MODES (self-red-team (ii)). A geometry where the
real solver fails can mean either
    (a) the surrogate is wrong  -> surrogate says acquired, real says no; or
    (b) acquisition is genuinely impossible here -> BOTH say no.
Only (a) is a training-signal bug. (b) means W1xnav needs an observability-aware
policy, not a better signal, and the script reports the two separately rather
than collapsing them into one "disagreement" number.

A NOTE ON THE WINDOW, which matters more than it looks. `RealAcq3D.w0` is
derived from the sim-time floor and the cadence, never hardcoded:
w0 = round(min_sec / dt_tick). At the 60 s baseline that is 45 observations.
At the DAY-WARP cadence (nav_max_ticks=120 -> dt = 720 s) it is FOUR, against
six unknowns — dof = 2*4 - 6 = 2. Long arcs at warp cadence are not
information-rich in the way arc LENGTH suggests; they are four-to-a-hundred
sparse looks.
"""
import argparse
import os
import sys

import numpy as np

WT = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
sys.path.insert(0, os.path.join(WT, 'pufferlib'))
sys.path.insert(0, os.path.join(WT, 'scripts', 'orbital', 'extj2'))
sys.path.insert(0, os.path.join(WT, 'scripts', 'orbital', 'nav'))

from pufferlib.ocean.orbital_nav.orbital_nav import OrbitalNav  # noqa: E402
import t11_cells as T                                           # noqa: E402

DAY_WARP, SIX_H = 30, 17


def w1_env(n, seed, max_ticks=120):
    c = dict(T.CELLS)['W1_driftwait']
    kw = T.nav_env_kwargs(
        num_envs=n, nav_mode='bearings_only', cell_mixture_mode=0,
        j2_mode=int(c['j2']), nav_j2_mode=int(c['j2']),
        episode_cap_steps=int(c['cap']), rendezvous_radius_m=c['box_r'],
        rel_vel_tol_ms=c['box_v'], a_min_override=c['a_min'],
        a_max_override=c['a_max'], e_max_target=c['e_max_target'],
        e_max_sat=c['e_max_sat'], de_max=c['de_max'], da_max_m=c['da_max'],
        di_max_rad=c['di_max'], di_min_rad=c['di_min'],
        di_phase_mode=int(c['di_phase']),
        i_target_min_rad=c['i_t_min'], i_target_max_rad=c['i_t_max'],
        fuel_frac_min=c['fuel_min'], fuel_frac_max=c['fuel_max'])
    kw['nav_max_ticks'] = max_ticks
    e = OrbitalNav(**kw)
    e.reset(seed=seed)
    return e


def swap_real(e, dt_tick):
    """Replace the surrogate with the real batch IOD, same gate constants."""
    from eval_relnav3d import RealAcq3D
    e._acq = RealAcq3D(
        e.num_agents, sigma_beta=e._s_beta, dt_tick=dt_tick,
        gate=e._acq_gate, min_sec=e._acq_min_sec,
        cov_inflate=e._cov_inflate, r_min=e._r_min, r_max=e._r_max)
    return e


def run_arm(n, seed, hours, real, max_ticks=120):
    """Coast to the floor, then fly `hours` of warp-cadence arc. Return the
    acquired mask and the realised tick spacing."""
    e = w1_env(n, seed, max_ticks)
    row, per = (DAY_WARP, 24.0) if hours >= 24.0 else (SIX_H, 6.0)
    # dt_tick MUST be the spacing of the row actually flown: RealAcq3D derives
    # its window w0 = min_sec/dt from it, so passing the day-warp's 720 s while
    # flying the 6 h row (180 s) would hand the solver a 4-observation window
    # where it should have had 15, and the failure would look like geometry.
    tau_min = 1440 if row == DAY_WARP else 360
    n_t = tau_min if max_ticks <= 0 else min(tau_min, max_ticks)
    dt_tick = tau_min * 60.0 / n_t
    if real:
        swap_real(e, dt_tick)
    a_coast = np.zeros(e.num_agents, dtype=np.int32)
    for _ in range(45):                      # clear the 2700 s sim-time floor
        e.step(a_coast)
    for _ in range(max(1, int(round(hours / per)))):
        e.step(np.full(e.num_agents, row, dtype=np.int32))
    acq = np.asarray(e._acq.acquired).copy()
    e.close()
    return acq, dt_tick


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=24)
    ap.add_argument('--seeds', type=int, nargs='+', default=[11, 4242])
    ap.add_argument('--hours', type=float, nargs='+', default=[6, 12, 24, 48])
    ap.add_argument('--max-ticks', type=int, default=120)
    a = ap.parse_args()

    print('== surrogate vs REAL batch IOD at W1 arcs '
          f'(n={a.n} x {len(a.seeds)} seeds, gate 0.20, floor 2700 s) ==')
    print(f'  {"arc":>6s} {"surrogate":>10s} {"real IOD":>9s} {"optimism":>9s}  '
          f'{"both-no":>8s} {"disagree":>8s}   reading')
    for hours in a.hours:
        S, R = [], []
        for sd in a.seeds:
            s_acq, dt = run_arm(a.n, sd, hours, real=False,
                                max_ticks=a.max_ticks)
            r_acq, _ = run_arm(a.n, sd, hours, real=True,
                               max_ticks=a.max_ticks)
            S.append(s_acq); R.append(r_acq)
        s = np.concatenate(S); r = np.concatenate(R)
        opt = s.mean() - r.mean()
        both_no = float(np.mean(~s & ~r))
        disagree = float(np.mean(s & ~r))
        if disagree <= 0.02:
            note = 'surrogate calibrated here'
        elif both_no > 0.5:
            note = 'ACQUISITION LARGELY IMPOSSIBLE (not a surrogate bug)'
        else:
            note = 'SURROGATE OPTIMISTIC — fictional signal risk'
        print(f'  {hours:5.1f}h {s.mean():9.1%} {r.mean():8.1%} '
              f'{opt:+8.1%}  {both_no:7.1%} {disagree:7.1%}   {note}')
    for tau_min, lbl in ((360, '6 h row'), (1440, 'day-warp')):
        n_t = tau_min if a.max_ticks <= 0 else min(tau_min, a.max_ticks)
        dt = tau_min * 60.0 / n_t
        print(f'  {lbl:9s} cadence dt = {dt:6.0f} s -> real-IOD window '
              f'w0 = {max(2, int(round(2700/dt))):3d} obs against 6 unknowns')


if __name__ == '__main__':
    main()
