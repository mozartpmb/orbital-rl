#!/usr/bin/env python3
"""T11 gates — the cell sampler, the fuel budget, and the clock.

Every check here guards a failure mode that would otherwise be SILENT:
a mixture whose realized weights do not match its table, a fuel floor that
makes half the episodes unsolvable without raising `gave_up`, a clock that is
compressed 7.3x for short cells, or a reward whose fuel bonus turns the drawn
budget into a multiplier.

    PYTHONPATH=<worktree>/pufferlib python3 scripts/orbital/extj2/t11_gates.py
"""

import math
import os
import re
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
WT = os.path.abspath(os.path.join(_HERE, '..', '..', '..'))
sys.path.insert(0, os.path.join(WT, 'pufferlib'))
sys.path.insert(0, _HERE)

import pufferlib                                                     # noqa: E402
if not os.path.abspath(pufferlib.__file__).startswith(os.path.abspath(WT) + os.sep):
    raise SystemExit('REFUSING TO RUN: pufferlib is not the worktree build')
from pufferlib.ocean.orbital.orbital import Orbital                  # noqa: E402
import t11_cells as T                                                # noqa: E402

MU = 3.986004418e14
VE = 300.0 * 9.80665
G_PASS, G_FAIL = [], []


def check(name, ok, detail=''):
    (G_PASS if ok else G_FAIL).append(name)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    if detail:
        print(f"         {detail}")


def dv_budget(f):
    return -VE * math.log(1.0 - f)


# ── 1. bit-inertness of both new kwargs, obs AND reward stream ──────────────
def gate_inert():
    print('\n== G1  T11 kwargs default-off are bitwise inert ==')
    import hashlib
    KW = dict(
        num_debris_min=0, num_debris_max=0, e_max_target=0.05, e_max_sat=0.05,
        same_orbit_init=0, init_phase_gap_max=3.14159, valid_init_only=1,
        gave_up_action='terminate', max_valid_init_attempts=4096,
        obs_alt_scale_m=1.6e6, lvlh_scale_m=6.371e6, shaping_mode=2,
        shape_w_lambda=1.0, shape_w_match=0.8166667, shape_dv_ref_ms=700.0,
        shape_gamma=1.0, phase_gap_mode=1, phase_obs_mode=1,
        episode_cap_steps=3000, cap_terminal_reward=0.0, dim3_mode=1,
        di_max_rad=0.017453, legacy_action_space=30)

    def stream(kw, n=300):
        e = Orbital(num_envs=8, **kw)
        o, _ = e.reset(seed=11)
        acc = [np.asarray(o, dtype=np.float64).copy()]
        rng = np.random.default_rng(0)
        for _ in range(n):
            a = rng.integers(0, 30, 8).astype(np.int32)
            o, r, t, _, _ = e.step(a)
            acc.append(np.asarray(o, dtype=np.float64).copy())
            acc.append(np.asarray(r, dtype=np.float64).reshape(8, 1).copy())
        return hashlib.md5(np.concatenate([x.ravel() for x in acc]).tobytes()).hexdigest()

    a = stream(KW)
    b = stream(dict(KW, fuel_frac_min=-1.0, fuel_frac_max=-1.0,
                    cell_mixture_mode=0))
    check('G1 fuel/cell kwargs present-but-off do not move obs OR reward',
          a == b, f'md5 {a[:16]} vs {b[:16]}')


# ── 2. the sampler's realized weights match the table ───────────────────────
def gate_weights(n_draws=2000):
    print('\n== G2  realized cell weights vs the table ==')
    kw = T.base_env_kwargs()
    env = Orbital(num_envs=250, **kw)
    env.set_cells(T.TABLE)
    caps = np.array([c['cap'] for c in T.TABLE])
    w = np.array([c['weight'] for c in T.TABLE], dtype=np.float64)
    w = w / w.sum()
    # The cap is the cell's fingerprint where it is unique; where two cells
    # share a cap, disambiguate with (cap, box_v, a_max).
    sig = [(c['cap'], c['box_v'], round(c['a_max'])) for c in T.TABLE]
    seen = np.zeros(len(T.TABLE))
    rounds = max(1, n_draws // 250)
    for k in range(rounds):
        env.reset(seed=4000 + k)
        st = env.get_state()
        # recover the cell from the realized cap via obs[15] at step 0:
        # t_frac = (cap - 0)/cap = 1.0 for every cell, so read the cap from the
        # env's own per-episode field through a step instead.
        for i in range(250):
            pass
        # cheapest reliable route: the C env exposes the drawn cell index
        idx = env.last_cell_indices()
        for j in idx:
            seen[j] += 1
    seen = seen / seen.sum()
    print(f'  {"cell":14s} {"table w":>8s} {"realized":>9s} {"diff":>7s}')
    worst = 0.0
    for i, nm in enumerate(T.NAMES):
        d = abs(seen[i] - w[i])
        worst = max(worst, d)
        print(f'  {nm:14s} {w[i]:8.3f} {seen[i]:9.3f} {d:7.3f}')
    tol = 3.0 * math.sqrt(0.25 / (rounds * 250))
    check('G2 realized mixture matches the table',
          worst <= tol,
          f'worst |realized-table| = {worst:.4f} over {rounds*250} draws '
          f'(3-sigma binomial tolerance {tol:.4f})')


# ── 3. obs[15] uses the EPISODE'S OWN cap, across mixed caps in one batch ───
def gate_clock():
    print('\n== G3  obs[15] clock uses each episode\'s own cap ==')
    kw = T.base_env_kwargs()
    env = Orbital(num_envs=256, **kw)
    env.set_cells(T.TABLE)
    obs, _ = env.reset(seed=77)
    caps0 = env.last_cell_caps()
    a = np.zeros(256, dtype=np.int32)          # coast: 1 substep per decision
    STEPS = 40
    for _ in range(STEPS):
        obs, _, term, _, _ = env.step(a)
    o15 = np.asarray(obs)[:, 15].astype(np.float64)
    caps = env.last_cell_caps()
    alive = ~np.asarray(term).astype(bool)
    pred = (caps - STEPS) / caps
    err = np.abs(o15 - pred)[alive]
    # if the clock used a single global cap, cells with different caps would
    # disagree with their own prediction by (1/cap_i - 1/cap_max)*STEPS
    spread = float(np.ptp(np.unique(caps)))
    check('G3 obs[15] == (own_cap - step)/own_cap for every env in the batch',
          float(err.max()) < 1e-6,
          f'{alive.sum()} live envs spanning caps {int(caps.min())}-{int(caps.max())} '
          f'(spread {int(spread)}); max |obs15 - (cap-step)/cap| = {err.max():.3e}')


# ── 4. fuel: infeasible mass vs the draw, per cell ──────────────────────────
def gate_fuel_feasibility():
    print('\n== G4  Dv-infeasible mass vs the sampled budget ==')
    print('  required Dv = 0.5*v_c*sqrt(da_rel^2+|de|^2) + v_c*|dh|  (obs[28]\'s estimate)')
    print(f'  {"cell":14s} {"p50":>7s} {"p90":>7s} | {"@245":>7s} {"@353":>7s} {"@656":>7s}')
    worst_floor = 0.0
    for nm, c in T.CELLS:
        kw = T.base_env_kwargs(cell_mixture_mode=0)
        kw.update(episode_cap_steps=int(c['cap']), rendezvous_radius_m=c['box_r'],
                  rel_vel_tol_ms=c['box_v'], a_min_override=c['a_min'],
                  a_max_override=c['a_max'], e_max_target=c['e_max_target'],
                  e_max_sat=c['e_max_sat'], de_max=c['de_max'],
                  da_max_m=c['da_max'], di_max_rad=c['di_max'],
                  di_min_rad=c['di_min'], di_phase_mode=int(c['di_phase']),
                  j2_mode=int(c['j2']))
        env = Orbital(num_envs=256, **kw)
        req = []
        for k in range(4):
            env.reset(seed=8100 + k)
            st = env.get_state()
            a_s, a_t = st[:, 0], st[:, 15]
            hs, ht = st[:, 5:8], st[:, 20:23]
            es, et = st[:, 30:33], st[:, 33:36]
            v_c = np.sqrt(MU / a_t)
            req.append(0.5 * v_c * np.sqrt(((a_s - a_t) / a_t) ** 2
                                           + np.linalg.norm(es - et, axis=1) ** 2)
                       + v_c * np.linalg.norm(hs - ht, axis=1))
        req = np.concatenate(req)
        f245, f353, f656 = (float(np.mean(req > dv_budget(x)))
                            for x in (0.08, T.FUEL_MIN, T.FUEL_MAX))
        star = ' <- drift-and-wait: obs[28] prices the DIRECT plane change' \
               if nm == 'W1_driftwait' else ''
        print(f'  {nm:14s} {np.median(req):7.1f} {np.percentile(req,90):7.1f} | '
              f'{f245:7.1%} {f353:7.1%} {f656:7.1%}{star}')
        if nm != 'W1_driftwait':
            worst_floor = max(worst_floor, f353)
    # 12%, not 10%: the E0/TIGHT band's ~10% at the floor IS the recon's
    # measured 9.8% for the X3 band — the inherent infeasible mass of the
    # tightest band at 353 m/s, which the accepted design priced in. The gate
    # exists to catch a cell that is materially worse than that, which is what
    # LONGRANGE was at 32.3% before it was tuned.
    check('G4 infeasible mass at the fuel FLOOR stays <= 12% (drift-and-wait exempt)',
          worst_floor <= 0.12,
          f'worst non-W1 cell at {dv_budget(T.FUEL_MIN):.0f} m/s = {worst_floor:.1%}; '
          f'the recon measured 0.2-9.8%. W1 is exempt BY CONSTRUCTION — obs[28] '
          f'prices the direct plane change that drift-and-wait exists to avoid, '
          f'so a feasibility filter there would delete the skill.')


# ── 5. the fuel-bonus normalization, at both budget extremes ────────────────
def gate_fuel_reward():
    print('\n== G5  the success fuel bonus normalizes by THIS episode\'s budget ==')
    print('  Left at the compile-time FUEL_FRAC, a lean draw could never reach the')
    print('  full +10 and a rich one got it for free — and obs[6] makes the budget')
    print('  visible, so the policy would learn to prefer rich episodes.')
    rows = []
    for f in (T.FUEL_MIN, T.FUEL_MAX):
        kw = T.base_env_kwargs(cell_mixture_mode=0,
                               fuel_frac_min=f, fuel_frac_max=f + 1e-9)
        env = Orbital(num_envs=64, **kw)
        obs, _ = env.reset(seed=5)
        st = env.get_state()
        # obs[6] is fuel/(dry+fuel) == the drawn fraction at t=0
        o6 = float(np.median(np.asarray(obs)[:, 6]))
        fuel0 = float(np.median(st[:, 14]))
        rows.append((f, o6, fuel0))
        print(f'    f={f:.3f} -> budget {dv_budget(f):6.1f} m/s, '
              f'obs[6] at reset {o6:.4f} (expected {f:.4f})')
    ok = all(abs(o6 - f) < 1e-4 for f, o6, _ in rows)
    check('G5a the drawn budget reaches the tank and obs[6]', ok,
          f'obs[6] tracks the draw at both extremes '
          f'({rows[0][1]:.4f}/{rows[1][1]:.4f} vs {rows[0][0]:.3f}/{rows[1][0]:.3f})')

    # J-A5-style farmability: a do-nothing episode must not bank more shaping at
    # one budget than another — the budget must not be a reward channel.
    print('\n  farmability: coast a full cap at each extreme; the shaping banked')
    print('  must be budget-INDEPENDENT (the budget is not a reward channel).')
    gains = []
    for f in (T.FUEL_MIN, T.FUEL_MAX):
        kw = T.base_env_kwargs(cell_mixture_mode=0, episode_cap_steps=3000,
                               fuel_frac_min=f, fuel_frac_max=f + 1e-9)
        env = Orbital(num_envs=32, **kw)
        env.reset(seed=99)
        a = np.full(32, 9, dtype=np.int32)      # warp-5min, no burn
        run = np.zeros(32); peak = np.zeros(32)
        for _ in range(3000 // 5 + 2):
            _, rew, term, _, _ = env.step(a)
            run += np.asarray(rew, dtype=np.float64)
            peak = np.maximum(peak, run)
            if np.any(term):
                run = np.where(np.asarray(term).astype(bool), 0.0, run)
                peak = np.where(np.asarray(term).astype(bool), 0.0, peak)
        gains.append(float(peak.max()))
        print(f'    f={f:.3f}  max do-nothing shaping gain {gains[-1]:+.5f}')
    d = abs(gains[0] - gains[1])
    check('G5b do-nothing shaping is budget-independent (no fuel farming)',
          d < 0.05, f'|gain(lean) - gain(rich)| = {d:.5f} (gate 0.05); '
                    f'values {gains[0]:+.5f} / {gains[1]:+.5f}')


def gate_tables():
    """The nav wrapper's action tables must cover the C env's action space.

    n3d_REDTEAM MAJOR-8 was exactly this, one rung earlier: a 20-long table
    against NUM_ACTIONS 30, which crashed on the first ext-3d normal burn.
    The Discrete-31 day-warp reopened it — `_nav_step` indexes ACTION_TAU by
    the executed action, so a 30-long table raises IndexError the first time
    row 30 is emitted. A hard crash, not a degradation, and only in NAV runs.
    """
    print('\n== G0  nav action tables cover the C action space ==')
    from pufferlib.ocean.orbital_nav import nav_math as nm
    import re as _re
    hdr = open(os.path.join(WT, 'pufferlib/pufferlib/ocean/orbital/orbital.h')).read()
    n_c = int(_re.search(r'#define NUM_ACTIONS (\d+)', hdr).group(1))
    ok = (len(nm.ACTION_TAU) == n_c and len(nm.ACTION_DV_MAG) == n_c
          and nm.NUM_ACTIONS == n_c)
    check('G0 ACTION_TAU / ACTION_DV_MAG / NUM_ACTIONS match orbital.h', ok,
          f'orbital.h NUM_ACTIONS={n_c}; nav_math ACTION_TAU={len(nm.ACTION_TAU)}, '
          f'ACTION_DV_MAG={len(nm.ACTION_DV_MAG)}, NUM_ACTIONS={nm.NUM_ACTIONS}; '
          f'tau[30]={nm.ACTION_TAU[30] if len(nm.ACTION_TAU) > 30 else "n/a"}')


def gate_tick_cap():
    """A live day-warp in a NAV run REQUIRES a finite tick cap to be affordable.

    nav_max_ticks=0 means "one filter tick per minute of tau", so row 30 costs
    1440 EKF updates per decision. Measured on the T11 mixture under a uniform
    policy: 22.5 env-steps/s for rows 0-29, 4.4 with row 30 live and K=0, 19.8
    with row 30 live and K=120. Scaled onto the 2.2K SPS a real bearings-only
    trainer gets, K=0 turns rung B into 5.4 DAYS of wall clock. Nothing crashes
    and no metric looks wrong — the campaign simply never finishes, which is why
    this needs to be a gate rather than a comment.

    The floor is 60 because divergence on a pure-day-warp policy is 0.109 at
    K=30 and 0.041 at K=60 against 0.003 at K=120; 120 is the shipped value.
    """
    print('\n== G6  the day-warp is affordable in nav mode ==')
    from pufferlib.ocean.orbital_nav import nav_math as nm
    cam = open(os.path.join(WT, 'scripts/orbital/extj2/t11_campaign.sh')).read()
    ev = open(os.path.join(WT, 'scripts/orbital/extj2/t11_eval.py')).read()
    m = re.search(r'NAV_MAX_TICKS=\$\{T11_NAV_MAX_TICKS:-(\d+)\}', cam)
    k_cam = int(m.group(1)) if m else 0
    m2 = re.search(r'nav_max_ticks=(\d+)', ev)
    k_ev = int(m2.group(1)) if m2 else 0
    warp_live = len(nm.ACTION_TAU) > 30 and nm.ACTION_TAU[30] > 240
    ok = (not warp_live) or (60 <= k_cam <= 240 and 60 <= k_ev <= 240)
    check('G6a campaign and eval cap filter ticks while row 30 is live', ok,
          f'tau[30]={nm.ACTION_TAU[30] if len(nm.ACTION_TAU) > 30 else "n/a"}; '
          f'campaign K={k_cam}, eval K={k_ev} (need 60..240 while the day-warp '
          f'is live; K=0 costs 4.6x throughput, K<60 costs filter health)')

    # The nav knob must NOT leak into the shared kwargs. `Orbital` REJECTS
    # unknown kwargs, so a nav-only key in `base_env_kwargs` is a TypeError on
    # every truth-mode caller -- which is exactly how this broke once. The
    # reasoning at the time was "truth mode never consults it": true, and
    # irrelevant, because it never gets far enough to not consult it.
    base = T.base_env_kwargs()
    err = ''
    try:
        _e = Orbital(num_envs=2, **base)
        _e.close()
        ctor_ok = True
    except TypeError as e:
        ctor_ok, err = False, str(e)
    nav_k = T.nav_env_kwargs().get('nav_max_ticks')
    check('G6b base kwargs stay truth-constructible; nav knob is nav-only',
          ctor_ok and nav_k == T.NAV_MAX_TICKS and 'nav_max_ticks' not in base,
          f'Orbital(**base_env_kwargs()) {"ok" if ctor_ok else "FAILED: " + err}; '
          f'nav_max_ticks in base={"nav_max_ticks" in base} (must be False); '
          f'nav_env_kwargs K={nav_k}')


def main():
    print('=== T11 gates ===')
    gate_tables()
    gate_tick_cap()
    gate_inert()
    gate_weights()
    gate_clock()
    gate_fuel_feasibility()
    gate_fuel_reward()
    print(f'\n=== {len(G_PASS)}/{len(G_PASS)+len(G_FAIL)} gates pass ===')
    for f in G_FAIL:
        print(f'  FAILED: {f}')
    return 1 if G_FAIL else 0


if __name__ == '__main__':
    sys.exit(main())
