"""T3 recon: closed-form feasibility band for phase-change-by-drift-orbit under
CORRECTED dynamics, plus the RL credit-assignment horizon implied by the
current (gamma, gae_lambda, action-tau) triple.

Physics: to change relative phase by dtheta you raise/lower semi-major axis by
da, coast, then undo it. For small da at a circular LEO orbit:
    dv_one_burn  = v * da / (2a)          (vis-viva, small da)
    dv_total     = v * da / a             (raise + lower)
    drift_rate   = 1.5 * n * da / a       (rad/s)
    t_drift      = dtheta / drift_rate
  =>  dv_total * t_drift = dtheta * v / (1.5 * n)      <-- INVARIANT, da drops out

So the phasing problem is a pure hyperbola in (delta-v, time). The episode is
feasible only if that hyperbola passes under BOTH the fuel ceiling and the
wall-clock ceiling.

MAX_STEPS = 2000 is counted in 60 s SUB-steps (c_step increments env->step once
per sub-step inside the warp loop), so the wall-clock budget is 2000*60 s =
33.3 h REGARDLESS of how much the agent warps. Warping buys decisions, not time.
"""
import csv, os
import numpy as np

MU = 3.986004418e14
R_EARTH = 6.371e6
DT = 60.0
MAX_STEPS = 2000
DV_BUDGET = 478.0          # ~15% fuel fraction, Isp 300 s
GAMMA = 0.995
GAE_LAMBDA = 0.90

T_BUDGET = MAX_STEPS * DT  # 120000 s = 33.33 h


def phasing_invariant(dtheta, alt_km=550.0):
    a = R_EARTH + alt_km * 1e3
    v = np.sqrt(MU / a)
    n = np.sqrt(MU / a**3)
    return dtheta * v / (1.5 * n), a, v, n


if __name__ == '__main__':
    out = '/Users/pete/space_training/web_data/results'
    os.makedirs(out, exist_ok=True)

    rows = []
    print(f'wall-clock budget = {T_BUDGET:.0f} s = {T_BUDGET/3600:.2f} h '
          f'(MAX_STEPS={MAX_STEPS} counted in {DT:.0f}s SUB-steps)')
    print(f'delta-v budget    = {DV_BUDGET:.0f} m/s\n')
    print(f'{"dtheta_deg":>10} {"K=dv*t":>12} {"dv_min@Tcap":>12} {"t_min@dvcap_h":>14} '
          f'{"da_at_dvmin_km":>15} {"feasible":>9}')
    for dtheta_deg in [15, 30, 45, 60, 90, 120, 150, 180]:
        dtheta = np.radians(dtheta_deg)
        K, a, v, n = phasing_invariant(dtheta)
        dv_min = K / T_BUDGET                 # cheapest phasing, uses whole episode
        t_min = K / DV_BUDGET                 # fastest phasing, uses whole tank
        da_at_dvmin = dv_min * a / v
        feasible = (dv_min < DV_BUDGET) and (t_min < T_BUDGET)
        print(f'{dtheta_deg:10d} {K:12.3e} {dv_min:12.1f} {t_min/3600:14.2f} '
              f'{da_at_dvmin/1e3:15.1f} {str(feasible):>9}')
        rows.append(dict(dtheta_deg=dtheta_deg, K_dv_times_t=K,
                         dv_min_ms=dv_min, t_min_h=t_min/3600,
                         da_at_dv_min_km=da_at_dvmin/1e3,
                         frac_of_dv_budget=dv_min/DV_BUDGET,
                         frac_of_time_budget=t_min/T_BUDGET,
                         feasible=int(feasible)))
    with open(os.path.join(out, 't3_phasing_feasibility.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    print('\n--- worked point: dtheta=180 deg, spend 200 m/s on phasing ---')
    K, a, v, n = phasing_invariant(np.pi)
    dv = 200.0
    t = K / dv
    print(f'  drift time   = {t:.0f} s = {t/3600:.2f} h  ({100*t/T_BUDGET:.0f}% of episode)')
    print(f'  da           = {dv*a/v/1e3:.0f} km')
    print(f'  leftover     = {DV_BUDGET-dv:.0f} m/s and {(T_BUDGET-t)/3600:.2f} h '
          f'for the altitude transfer + terminal closure')

    print('\n=== credit-assignment horizon (learner side) ===')
    print(f'gamma={GAMMA}, gae_lambda={GAE_LAMBDA}')
    print(f'GAE effective n-step window = 1/(1-gamma*lambda) = '
          f'{1/(1-GAMMA*GAE_LAMBDA):.1f} DECISIONS')
    hrows = []
    for name, tau in [('coast/burn', 1), ('warp5min', 5), ('warp30min', 30), ('warp1hr', 60)]:
        ndec = MAX_STEPS / tau
        disc = GAMMA ** ndec
        hrows.append(dict(action=name, tau_substeps=tau, decisions_per_episode=ndec,
                          gamma_pow_H=disc,
                          gae_window_decisions=1/(1-GAMMA*GAE_LAMBDA),
                          gae_window_as_frac_of_episode=(1/(1-GAMMA*GAE_LAMBDA))/ndec))
        print(f'  {name:11s} tau={tau:3d}  decisions/episode={ndec:7.0f}  '
              f'gamma^H={disc:9.3e}  GAE window covers {100*(1/(1-GAMMA*GAE_LAMBDA))/ndec:6.2f}% of episode')
    with open(os.path.join(out, 't3_credit_horizon.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(hrows[0].keys())); w.writeheader(); w.writerows(hrows)
    print('\nwrote t3_phasing_feasibility.csv / t3_credit_horizon.csv to', out)
