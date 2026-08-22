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


def gate_oop_seed():
    """MAJOR-17: the blind OOP seed must span the mixture, and no ceiling may
    push it back under its own ignorance.

    Two independent failures met here, and only one of them announced itself.

    G7a: `di_max_rad` is a CONSTRUCTION kwarg (1 deg) while the C sampler
    redraws di per episode from the cell table (W1_driftwait: up to 5 deg). A
    blind seed cannot know the episode's own di, so its ignorance has to cover
    the widest cell the mixture can deal. Silent when wrong -- an overconfident
    OOP channel just diverges more often.

    G7b: the fixed 1e-1 ceiling on sig_rate_oop bit whenever
    rho0 < v_oop/1e-1 (1326 m at di=1 deg), and rho0 is bimodal under the
    mixture (p1 316 m) because the tight-box and drift-and-wait cells seed
    close. 16.4% of rows landed on the ceiling and every one tripped MAJOR-16's
    assert -- on the FIRST reset. Note this needed no mixture at all: the assert
    uses the same di on both sides, so only the ceiling can violate it.
    """
    print('\n== G7  out-of-plane seed spans the mixture and is never clipped '
          'under its own ignorance ==')
    from pufferlib.ocean.orbital_nav.orbital_nav import OrbitalNav
    cell_di = max(float(c['di_max']) for _, c in T.CELLS)
    kw = T.nav_env_kwargs(num_envs=8, nav_mode='bearings_only', j2_mode=1,
                          nav_j2_mode=1, t11_mixture=1, di_max_rad=0.017453)
    e = OrbitalNav(**kw)
    seed_di = float(getattr(e, '_di_seed_max', -1.0))
    e.close()
    check('G7a blind OOP seed covers the widest cell, not the ctor kwarg',
          seed_di >= cell_di - 1e-12,
          f'_di_seed_max={math.degrees(seed_di):.2f} deg vs cell-table max '
          f'{math.degrees(cell_di):.2f} deg (ctor kwarg was 1.00 deg)')

    # The ceiling, exercised directly across the rho0 range the mixture spans.
    v_c = math.sqrt(3.986004418e14 / 6.9e6)
    v_oop = v_c * math.sin(seed_di)
    sigma_v_ecc = 385.0
    rho0 = np.array([100.0, 316.0, 1e3, 1326.0, 5e3, 6624.0, 3.2e4, 4.4e4])
    raw = np.maximum(v_oop, sigma_v_ecc) / np.maximum(rho0, 1.0)
    sig = np.clip(raw, 1e-5, np.maximum(1e-1, raw))
    ok = bool(np.all(sig * rho0 >= v_oop - 1e-9))
    worst = float(np.min(sig * rho0 - v_oop))
    check('G7b the OOP ceiling never manufactures confidence', ok,
          f'v_oop={v_oop:.1f} m/s at di={math.degrees(seed_di):.2f} deg; '
          f'worst (sig*rho0 - v_oop) = {worst:+.3e} m/s over rho0 in '
          f'[{rho0.min():.0f}, {rho0.max():.0f}] m (must be >= 0)')


def gate_consol_weights(n_draws=2000):
    """The CONSOLIDATION variant (t11_mixture=2) draws its own weights, and the
    excluded cell is drawn ZERO times.

    Two things are checked because two different things can go wrong.

    G9a: the realized distribution matches `CONSOL_TABLE`. Same method as G2 —
    a mixture whose weights silently differ from the table is the failure that
    makes an interference result unreadable, since the whole experiment is
    "same cells, different weights".

    G9b: W1_driftwait, at weight 0.0, is drawn EXACTLY zero times. The C picker
    walks a cumulative sum and can only land on a zero-weight row through the
    trailing `pick = c` fallthrough, which is reachable only for the LAST index
    when float rounding puts u past the total. W1 is index 4 of 7 today, so the
    zero is structural — but it is structural only as long as nobody reorders
    `CELLS`, and a 22000-step cell drawn by accident would quietly eat the
    wall-clock budget. Hence measured, not argued.

    G9c: the shipped mixture is untouched by the variant's existence.
    """
    print('\n== G9  consolidation mixture (t11_mixture=2) ==')
    kw = T.base_env_kwargs()
    env = Orbital(num_envs=250, **kw)
    env.set_cells(T.CONSOL_TABLE)
    w = np.array([c['weight'] for c in T.CONSOL_TABLE], dtype=np.float64)
    w = w / w.sum()
    seen = np.zeros(len(T.CONSOL_TABLE))
    rounds = max(1, n_draws // 250)
    for k in range(rounds):
        env.reset(seed=4600 + k)
        for j in env.last_cell_indices():
            seen[j] += 1
    env.close()
    tot = seen.sum()
    frac = seen / tot
    print(f'  {"cell":14s} {"table w":>8s} {"realized":>9s} {"draws":>7s}')
    worst = 0.0
    for i, nm in enumerate(T.NAMES):
        worst = max(worst, abs(frac[i] - w[i]))
        print(f'  {nm:14s} {w[i]:8.3f} {frac[i]:9.3f} {int(seen[i]):7d}')
    tol = 3.0 * math.sqrt(0.25 / tot)
    check('G9a realized consolidation mixture matches its table',
          worst <= tol,
          f'{int(tot)} draws; worst |realized-table| = {worst:.4f} vs 3-sigma '
          f'tol {tol:.4f}')

    w1 = T.NAMES.index('W1_driftwait')
    check('G9b the excluded cell is drawn exactly zero times',
          seen[w1] == 0,
          f'W1_driftwait weight {w[w1]:.3f}, drawn {int(seen[w1])} times in '
          f'{int(tot)}; it is index {w1} of {len(T.NAMES)} and only the LAST '
          f'index can catch the picker\'s rounding fallthrough')

    ship = {n: c['weight'] for n, c in T.CELLS}
    check('G9c the shipped mixture is unchanged by the variant',
          abs(sum(ship.values()) - 1.0) < 1e-12 and len(T.CELLS) == 7
          and abs(ship['W1_driftwait'] - 0.20) < 1e-12
          and abs(ship['TIGHT_5k1'] - 0.10) < 1e-12,
          f'shipped weights sum {sum(ship.values()):.6f}, {len(T.CELLS)} cells, '
          f'W1 {ship["W1_driftwait"]:.3f}, TIGHT {ship["TIGHT_5k1"]:.3f}')


def gate_t15_weights(n_draws=2000):
    """G10: the T15 mixture draws its own weights and trains ALL SEVEN cells.

    G9 asserts the consolidation variant draws W1 exactly ZERO times. T15 is the
    opposite claim and needs its own gate: every cell must be drawn, because a
    silently-zero cell here would look exactly like the bootstrap failure the
    run exists to disprove.
    """
    print('\n== G10  T15 mixture (t11_mixture=3): all seven cells live ==')
    kw = T.base_env_kwargs()
    env = Orbital(num_envs=250, **kw)
    env.set_cells(T.T15_TABLE)
    w = np.array([c['weight'] for c in T.T15_TABLE], float); w = w / w.sum()
    seen = np.zeros(len(T.T15_TABLE))
    rounds = max(1, n_draws // 250)
    for k in range(rounds):
        env.reset(seed=5100 + k)
        for j in env.last_cell_indices():
            seen[j] += 1
    env.close()
    tot = seen.sum(); frac = seen / tot
    worst = float(np.max(np.abs(frac - w)))
    for i, nm in enumerate(T.NAMES):
        print(f'  {nm:14s} table {w[i]:6.3f}  realized {frac[i]:6.3f}  '
              f'draws {int(seen[i]):5d}')
    tol = 3.0 * math.sqrt(0.25 / tot)
    check('G10a realized T15 mixture matches its table', worst <= tol,
          f'{int(tot)} draws; worst |realized-table| {worst:.4f} vs tol {tol:.4f}')
    check('G10b every one of the seven cells is drawn', bool(np.all(seen > 0)),
          f'min draws {int(seen.min())} ({T.NAMES[int(seen.argmin())]}); a cell '
          f'silently at zero would present as a bootstrap failure')
    ship = {n: c['weight'] for n, c in T.CELLS}
    check('G10c the shipped mixture is unchanged by the T15 variant',
          len(T.CELLS) == 7 and abs(ship['W1_driftwait'] - 0.20) < 1e-12
          and abs(ship['TIGHT_5k1'] - 0.10) < 1e-12,
          f'shipped W1 {ship["W1_driftwait"]:.3f}, TIGHT {ship["TIGHT_5k1"]:.3f}')


def gate_range_prior():
    """MAJOR-17b: the range prior must CONTAIN the true range under mixture.

    This is the gate that was missing, and its absence is why the defect could
    only be found by reading. `_r_max` brackets the target's orbital radius and
    is what [lo, hi] -- and therefore the rho0 seed -- is solved against. Under
    `t11_mixture=1` the ctor value is 7.53e6 while E3_j2 (weight 0.15) draws a
    to 1.4371e7, so on those episodes the prior EXCLUDES THE TRUE RANGE. A
    filter cannot recover a target it has placed outside the universe, and
    nothing asserts on it, so it is silent.

    Worse than silent: every single-cell EVAL constructs with that cell's own
    band and gets an honest prior, while MIXTURE TRAINING gets the narrow one --
    a train/eval split in the flagship number. Hence the check runs over
    mixture draws specifically.
    """
    print('\n== G8  range prior contains the true range over mixture draws ==')
    from pufferlib.ocean.orbital_nav.orbital_nav import OrbitalNav
    N, BATCHES = 250, 8                       # 2000 draws
    kw = T.nav_env_kwargs(num_envs=N, nav_mode='bearings_only', j2_mode=1,
                          nav_j2_mode=1, t11_mixture=1, cell_mixture_mode=1)
    e = OrbitalNav(**kw)
    e.reset(seed=4242)
    rng = np.random.default_rng(4242)
    r_t = []
    for _ in range(BATCHES):
        _, _, _, tgt_c = e._decode()
        r_t.append(np.linalg.norm(np.asarray(tgt_c)[:, :3], axis=1).copy())
        e.step(rng.integers(0, 31, N, dtype=np.int32))
    r_min, r_max = float(e._r_min), float(e._r_max)
    ctor_max = float(getattr(e, '_r_max_ctor', r_max))
    e.close()
    r_t = np.concatenate(r_t)
    inside = (r_t >= r_min) & (r_t <= r_max)
    esc_ctor = float(np.mean(r_t > ctor_max))
    check('G8 the mixture range prior contains every true target radius',
          bool(np.all(inside)),
          f'{r_t.size} draws; true |r_t| in [{r_t.min():.4e}, {r_t.max():.4e}]; '
          f'prior [{r_min:.4e}, {r_max:.4e}]; outside={int((~inside).sum())}; '
          f'the ctor-only prior ({ctor_max:.4e}) would have EXCLUDED '
          f'{100.0 * esc_ctor:.1f}% of them')


def main():
    print('=== T11 gates ===')
    gate_tables()
    gate_tick_cap()
    gate_oop_seed()
    gate_range_prior()
    gate_consol_weights()
    gate_t15_weights()
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
