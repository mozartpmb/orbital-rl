"""T3 — scripted classical-GNC expert for the orbital rendezvous env.

A NON-LEARNED controller that solves the rendezvous task under the corrected
dynamics using ONLY the environment's 16 discrete actions and ONLY the 38-dim
observation vector (no ground-truth accessor, no privileged state).

────────────────────────────────────────────────────────────────────────────
CONTROLLER
────────────────────────────────────────────────────────────────────────────
Two nested loops, both classical:

1. SHAPE loop — matched-orbit targeting (inner, every decision)

   error   X = ( n·(a − a_g)/2 ,  v·(ē − ē_g)/2 )   [m/s, 3 components]
   value   V(X) = max(|A|, |E|)

   V is *exactly* the minimum two-impulse Δv needed to null X with a tangential
   burn pair at antipodal apsides (a burn δv at position angle φ moves X by
   δv·(1, cos φ, sin φ); an antipodal pair gives |δv₁|+|δv₂| = max(|A|,|E|)).
   The control law is a one-step lookahead on that value:

       pick the quantum maximising  ΔV − |Δv|·(1−slack)

   With no tuned gate this reproduces the Hohmann split exactly — burn 1 stops
   at the point where the induced |E| would exceed the remaining |A| — and it
   waits for the correct apsis when the error is a pure eccentricity-vector
   rotation.  Post-burn perigee is checked against a 200 km-altitude floor.

2. PHASE loop — along-track tracking (outer, every decision)

   Δλ = λ_sat − λ_tgt (mean longitudes) drifts at n(a) − n(a_t).  A first-order
   receding-horizon law commands the drift-orbit offset

       da_cmd = −Δ · a_t / (1.5 · n_t · T_ctrl),   T_ctrl = K · t_remaining

   where Δ is the signed phase still to close.  Constant Δ/T is the classical
   phasing orbit; because da_cmd shrinks with Δ the approach is a smooth spiral
   with no discrete "start the closing transfer NOW" trigger to miss.  (An
   earlier build used exactly such a trigger; the closure duration jumps by a
   half-period whenever the required apsis flips, so the trigger skipped the
   crossing and cost a whole relative revolution.  Measured: 91.5% → the smooth
   law removes that failure mode entirely.)  |da_cmd| is floored at 3 km so the
   drift always carries Δλ through zero — a 3 km radial offset is well inside
   the 30 km box, so the sweep itself scores.

   PLAN picks the drift *direction* (short way vs long way round) by Δv cost,
   respecting the perigee floor and the fuel budget.

3. HOLD — a forward scan of the exact model predicts the closest approach; if
   it clears 26 km / 42 m/s the controller stops burning and warps into it.

Usage:
    python3 scripts/orbital/t3/expert_controller.py --episodes 200
    python3 scripts/orbital/t3/expert_controller.py --suite --csv out.csv
"""
import argparse
import csv
import math
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..', '..'))
sys.path.insert(0, os.path.join(_ROOT, 'pufferlib'))
sys.path.insert(0, _HERE)

import t3_expert_model as om  # noqa: E402  (private copy of the physics replica)

MU, R_EARTH, DT = om.MU, om.R_EARTH, om.DT
TWO_PI = om.TWO_PI
MAX_STEPS = om.MAX_STEPS

# ── tuning constants ─────────────────────────────────────────────────────────
PERIGEE_FLOOR = R_EARTH + 200e3   # operational floor (drag); the env itself only
                                  # terminates at R_EARTH
BURN_SLACK = 0.12        # accept a burn whose ΔV ≥ |Δv|·(1−slack)
DV_FLOOR = 6.0           # m/s of tank never spent (stranded guard)
DA_FLOOR = 3.0e3         # m: smallest commanded drift offset (keeps Δλ moving)
DA_MAX = 340.0e3         # m: largest drift offset
TRACK_K = float(os.environ.get('T3_K', 0.65))   # T_ctrl = K · t_remaining
T_CTRL_MIN = 5400.0      # s
DV_PLAN_FRAC = 0.75      # law may commit this fraction of remaining Δv
SAFE_R = 26.0e3          # aim inside the 30 km box with margin
SAFE_V = 42.0            # ... and inside the 50 m/s box
NEAR_LATCH = math.radians(25.0)
A_DEADBAND = float(os.environ.get('T3_DEADBAND', 6.0e3))     # m
A_DEADBAND_REL = float(os.environ.get('T3_DEADBAND_REL', 0.25))

BURN_ACTIONS = (1, 2, 3, 4, 5, 6, 7, 8, 12, 13, 14, 15)
WARPS = ((11, 60), (10, 30), (9, 5))

# ── actuation-matched variant (2026-08-20) ──────────────────────────────────
# The published tight-box expert rows were measured on Discrete-16, whose
# finest radial authority is rows 7/8 at +-10 m/s. The tight-box POLICY
# lineages fly Discrete-20, which adds rows 18/19 at +-1 m/s (orbital.h
# ACTION_DV; mirrored in t3_expert_model.ACTION_DV). Comparing them at 5 km /
# 1 m/s therefore compared two different actuators, not two guidance laws.
#
# `_burn` selects by searching BURN_ACTIONS for the best gain, so extending
# this tuple is what actually grants the finer authority -- the RAD_OUT/RAD_IN
# constants at t3_expert_model.py:53 are dead code, referenced nowhere.
#
# WARPS is deliberately NOT extended with the D20 rows 16/17 (180/360 min).
# Adding them would change the cruise schedule as well as the actuator, and
# arm B exists to isolate the actuator alone.
BURN_ACTIONS_D16 = BURN_ACTIONS
BURN_ACTIONS_D20 = BURN_ACTIONS + (18, 19)

# Controller constants at their 30 km design point, as FRACTIONS of the box.
# Both are absolute metres in the published controller and neither scales, so
# at a 5 km box they are 120% and 60% of the whole tolerance. DA_FLOOR is the
# more suspicious of the two: a parked semi-major-axis offset da implies a
# relative speed |dv| ~ 0.5 * v * da / a, so at a = 7e6 m (v = 7546 m/s) the
# 3 km floor alone implies 1.62 m/s -- already over a 1 m/s tolerance before
# any guidance error. Scaling them is arm C.
A_DEADBAND_FRAC = 6.0e3 / 30.0e3        # 0.20 of box_r
DA_FLOOR_FRAC = 3.0e3 / 30.0e3          # 0.10 of box_r


def _n(a):
    return math.sqrt(MU / (a * a * a))


def _evec(el):
    return el['e'] * math.cos(el['omega']), el['e'] * math.sin(el['omega'])


def _lam(el):
    return el['omega'] + el['M']


def hohmann_dv(r1, r2):
    v1, v2 = math.sqrt(MU / r1), math.sqrt(MU / r2)
    at = 0.5 * (r1 + r2)
    return (abs(v1 * (math.sqrt(r2 / at) - 1.0)) +
            abs(v2 * (1.0 - math.sqrt(r1 / at))))


# ═════════════════════════════════════════════════════════════════════════════
class ExpertController:
    def __init__(self, verbose=False, trace=False, cap_steps=None):
        self.verbose = verbose
        self.trace = trace
        # T3: the episode cap is a runtime env kwarg now; the controller's
        # horizon must match it (red-team #11: a clock-blind or wrong-clock
        # controller times out on states a clock-aware one solves).
        self.cap_steps = MAX_STEPS if cap_steps is None else int(cap_steps)
        self.reset()

    # ── episode lifecycle ───────────────────────────────────────────────────
    def reset(self):
        self.steps = 0            # mirrors env->step
        self.decisions = 0
        self.mode = 'PLAN'
        self.dirn = -1            # +1 → drive Δλ up (a<a_t); −1 → down (a>a_t)
        self.near = False         # latched once |Δλ| < NEAR_LATCH
        self.dv_cmd = 0.0
        # 30, not 16: rows 18/19 (fine radial) index past the end of a
        # 16-slot histogram and raise IndexError at the first fine burn.
        self.acts = [0] * 30
        self.scan_cd = 0
        self.hold_t = 0.0
        self.plan_dv = 0.0
        self.n_plans = 0
        self.min_d = 1e30
        self.a_hold = None

    # ── main entry ──────────────────────────────────────────────────────────
    def act(self, obs):
        sat, tgt, fuel = om.decode_obs(obs)
        dv_left = max(0.0, om.dv_remaining(fuel) - DV_FLOOR)
        t_rem = (self.cap_steps - self.steps) * DT
        a = self._decide(sat, tgt, dv_left, t_rem)
        dvp, dvr = om.ACTION_DV[a]
        self.dv_cmd += math.hypot(dvp, dvr)
        self.steps += om.ACTION_TAU[a]
        self.decisions += 1
        self.acts[a] += 1
        return a

    # ── decision logic ──────────────────────────────────────────────────────
    def _decide(self, sat, tgt, dv_left, t_rem):
        if t_rem <= DT:
            return 0

        if self.mode == 'PLAN':
            self._plan(sat, tgt, dv_left, t_rem)
            self.mode = 'TRACK'

        d_now, _ = om.rel_state(sat, tgt)
        self.min_d = min(self.min_d, d_now)

        if self.mode == 'HOLD':
            self.hold_t -= DT
            if self.hold_t <= 0.0 or self.scan_cd <= 0:
                hit, t_hit = self._scan(sat, tgt, t_rem)
                self.scan_cd = 6
                if hit:
                    self.hold_t = t_hit
                else:
                    self.mode = 'TRACK'
                    self._tr('HOLD->TRACK', sat, tgt)
            self.scan_cd -= 1
            if self.mode == 'HOLD':
                return self._warp(max(0.0, self.hold_t - DT))

        # ── TRACK ───────────────────────────────────────────────────────────
        a_g = self._track_goal(sat, tgt, dv_left, t_rem)
        eg = _evec(tgt)
        V = self._V(sat, a_g, eg)
        act = self._burn(sat, a_g, eg, dv_left)
        if act is not None:
            return act

        # coasting: is a box entry already predicted?
        dl = abs(om.wrap_pi(_lam(sat) - _lam(tgt)))
        if self.scan_cd <= 0 and dl < 0.09 and V < 12.0:
            hit, t_hit = self._scan(sat, tgt, t_rem)
            self.scan_cd = 4
            if hit:
                self.mode = 'HOLD'
                self.hold_t = t_hit
                self._tr('TRACK->HOLD', sat, tgt)
                return self._warp(max(0.0, t_hit - DT))
        self.scan_cd -= 1

        budget = 1800.0
        if V > 0.6:
            budget = min(budget, self._t_to_window(sat, a_g, eg))
        if dl < 0.25:
            budget = min(budget, 300.0)     # fine control near alignment
        return self._warp(budget)

    # ── phase-tracking law ──────────────────────────────────────────────────
    def _track_goal(self, sat, tgt, dv_left, t_rem):
        a_s, a_t = sat['a'], tgt['a']
        n_t = _n(a_t)
        dl = om.wrap_pi(_lam(sat) - _lam(tgt))
        if abs(dl) < NEAR_LATCH:
            self.near = True
        if self.near:
            delta = -dl                                  # short way, either sign
        elif self.dirn > 0:
            delta = ((-dl) % TWO_PI)                     # drive Δλ up
        else:
            delta = -(dl % TWO_PI)                       # drive Δλ down

        T_ctrl = max(T_CTRL_MIN, TRACK_K * t_rem)
        a_lim = PERIGEE_FLOOR / max(1e-6, 1.0 - tgt['e'])

        # Direction feasibility. Closing Δλ the short way may require a drift
        # orbit BELOW the target, which the perigee floor can forbid outright
        # when the target's own perigee is barely above 200 km altitude. In that
        # case the only option is the long way round (Δ ∓ 2π). Score both.
        best = None
        cands = [delta]
        if abs(delta) > 0.01:
            cands.append(delta - math.copysign(TWO_PI, delta))
        for dc in cands:
            da = -dc * a_t / (1.5 * n_t * T_ctrl)
            if abs(da) < DA_FLOOR:
                da = math.copysign(DA_FLOOR, da if da != 0.0 else -1.0)
            da = math.copysign(min(abs(da), DA_MAX), da)
            a_c = max(a_t + da, a_lim)
            da_eff = a_c - a_t
            ok = (da_eff * da > 0.0) and abs(da_eff) >= 0.9 * DA_FLOOR
            rate = 1.5 * n_t * abs(da_eff) / a_t
            t_need = abs(dc) / rate if rate > 1e-14 else 1e30
            cost = n_t * (abs(a_c - a_s) + abs(da_eff)) / 2.0
            score = cost + (0.0 if (ok and t_need < 0.95 * t_rem) else 1e5) + 1e-3 * t_need
            if best is None or score < best[0]:
                best = (score, a_c)
        a_g = best[1]
        # Δv affordability: shrink the drift offset until the round trip fits
        for _ in range(40):
            cost = n_t * (abs(a_g - a_s) + abs(a_g - a_t)) / 2.0
            if cost <= DV_PLAN_FRAC * dv_left or abs(a_g - a_t) <= DA_FLOOR:
                break
            a_g = max(a_t + 0.85 * (a_g - a_t), a_lim)
        # Deadband: hold the commanded drift orbit until the law has moved a
        # meaningful distance.  Without it the burn law chases a continuously
        # sliding set-point and pays spiral-transfer overhead instead of
        # executing discrete Hohmann pairs.
        if self.a_hold is None or \
                abs(a_g - self.a_hold) > max(A_DEADBAND, A_DEADBAND_REL * abs(self.a_hold - a_t)):
            self.a_hold = a_g
        return self.a_hold

    # ── planner: pick the drift direction ───────────────────────────────────
    def _plan(self, sat, tgt, dv_left, t_rem):
        a_s, a_t = sat['a'], tgt['a']
        n_t = _n(a_t)
        es, et = _evec(sat), _evec(tgt)
        dv_e = math.sqrt(MU / a_t) * math.hypot(es[0] - et[0], es[1] - et[1]) / 2.0
        dl = om.wrap_pi(_lam(sat) - _lam(tgt))
        T_ctrl = max(T_CTRL_MIN, TRACK_K * t_rem)
        a_lim = PERIGEE_FLOOR / max(1e-6, 1.0 - tgt['e'])

        best = None
        for dirn in (+1, -1):
            psi = ((-dl) % TWO_PI) if dirn > 0 else (dl % TWO_PI)
            da = -math.copysign(psi, dirn) * a_t / (1.5 * n_t * T_ctrl)
            da = math.copysign(min(abs(da), DA_MAX), da)
            a_p = max(a_t + da, a_lim)
            rate = 1.5 * n_t * abs(a_p - a_t) / a_t
            eta = psi / rate if rate > 1e-12 else 1e30
            cost = max(dv_e, n_t * abs(a_p - a_s) / 2.0) + n_t * abs(a_p - a_t) / 2.0
            score = cost + (400.0 if eta > 0.85 * t_rem else 0.0) + 3e-4 * eta
            if best is None or score < best[0]:
                best = (score, dirn, cost, eta)
        self.dirn = best[1]
        self.n_plans += 1
        if self.n_plans == 1:
            self.plan_dv = best[2]
        if self.verbose:
            print(f"    [plan] a_s={(a_s-R_EARTH)/1e3:.1f} a_t={(a_t-R_EARTH)/1e3:.1f} km "
                  f"dlam={math.degrees(dl):+.1f}deg dirn={best[1]:+d} "
                  f"cost~{best[2]:.0f} m/s eta~{best[3]/3600:.1f} h")

    # ── metric & burn law ───────────────────────────────────────────────────
    @staticmethod
    def _X(el, a_g, eg, n_ref, v_ref):
        ex, ey = _evec(el)
        return (n_ref * (el['a'] - a_g) / 2.0,
                v_ref * (ex - eg[0]) / 2.0,
                v_ref * (ey - eg[1]) / 2.0)

    def _V(self, el, a_g, eg):
        n_ref, v_ref = _n(a_g), math.sqrt(MU / a_g)
        A, Ex, Ey = self._X(el, a_g, eg, n_ref, v_ref)
        return max(abs(A), math.hypot(Ex, Ey))

    def _burn(self, sat, a_g, eg, dv_left):
        n_ref, v_ref = _n(a_g), math.sqrt(MU / a_g)
        A, Ex, Ey = self._X(sat, a_g, eg, n_ref, v_ref)
        V0 = max(abs(A), math.hypot(Ex, Ey))
        if V0 < 0.4:
            return None
        best_act, best_gain = None, 0.0
        for act in BURN_ACTIONS:
            dvp, dvr = om.ACTION_DV[act]
            dvm = math.hypot(dvp, dvr)
            if dvm > dv_left:
                continue
            el2 = om.apply_impulse(sat, dvp, dvr)
            if el2['a'] <= 0.0:
                continue
            if el2['a'] * (1.0 - el2['e']) < PERIGEE_FLOOR:
                continue
            A2, Ex2, Ey2 = self._X(el2, a_g, eg, n_ref, v_ref)
            V2 = max(abs(A2), math.hypot(Ex2, Ey2))
            gain = (V0 - V2) - dvm * (1.0 - BURN_SLACK)
            if gain > best_gain:
                best_gain, best_act = gain, act
        return best_act

    def _t_to_window(self, sat, a_g, eg):
        """Seconds until the next favourable burn phase (apsis of the error ē)."""
        n_ref, v_ref = _n(a_g), math.sqrt(MU / a_g)
        _, Ex, Ey = self._X(sat, a_g, eg, n_ref, v_ref)
        if math.hypot(Ex, Ey) < 1e-9:
            return 0.0
        phi_e = math.atan2(Ey, Ex)
        u = sat['omega'] + sat['theta']
        n_s = _n(sat['a'])
        best = min(((phi_e - u) % TWO_PI), ((phi_e + math.pi - u) % TWO_PI))
        return max(0.0, best / n_s - 3.0 * DT)

    # ── forward scan of the exact model ─────────────────────────────────────
    def _scan(self, sat, tgt, t_rem):
        horizon = min(t_rem - DT, 20 * 3600.0)
        if horizon <= 0:
            return False, 0.0
        drift = abs(1.5 * _n(tgt['a']) * (sat['a'] - tgt['a']))
        stride = 1 if drift < 1e-6 else max(1, min(10, int(10.0e3 / max(drift, 1.0) / DT)))
        best_d, best_t = 1e30, 0.0
        s, t, k = sat, tgt, 0
        while k * DT <= horizon:
            d, dv = om.rel_state(s, t)
            if d < best_d:
                best_d, best_t = d, k * DT
            if d < SAFE_R and dv < SAFE_V:
                return True, k * DT
            s = om.propagate(s, stride * DT)
            t = om.propagate(t, stride * DT)
            k += stride
        if stride > 1:
            t0 = max(0.0, best_t - stride * DT)
            s, t = om.propagate(sat, t0), om.propagate(tgt, t0)
            k = int(round(t0 / DT))
            end = min(horizon, best_t + stride * DT)
            while k * DT <= end:
                d, dv = om.rel_state(s, t)
                if d < SAFE_R and dv < SAFE_V:
                    return True, k * DT
                s, t = om.propagate(s, DT), om.propagate(t, DT)
                k += 1
        return False, best_t

    # ── warp selection ──────────────────────────────────────────────────────
    @staticmethod
    def _warp(budget_s):
        for act, tau in WARPS:
            if tau * DT <= budget_s:
                return act
        return 0

    def _tr(self, msg, sat, tgt):
        if self.trace:
            d, dv = om.rel_state(sat, tgt)
            print(f"      step={self.steps:4d} {msg:<14s} d={d/1e3:9.1f}km |vr|={dv:7.1f} "
                  f"da={(sat['a']-tgt['a'])/1e3:+7.1f}km "
                  f"dlam={math.degrees(om.wrap_pi(_lam(sat)-_lam(tgt))):+7.2f}deg")


# ═════════════════════════════════════════════════════════════════════════════
CAUSES = ['none', 'success', 'collision', 'escape', 'safety_cap',
          'stranded', 'hyperbolic', 'gave_up']


def _meta(sat, tgt):
    dlam = om.wrap_pi(_lam(sat) - _lam(tgt))
    dM = om.wrap_pi(sat['M'] - tgt['M'])
    es, et = _evec(sat), _evec(tgt)
    dv_e = math.sqrt(MU / tgt['a']) * math.hypot(es[0] - et[0], es[1] - et[1]) / 2.0
    dv_h = hohmann_dv(sat['a'], tgt['a'])
    return dict(gap_deg=round(math.degrees(dlam), 2),
                gapM_deg=round(math.degrees(dM), 2),
                e_sat=round(sat['e'], 4), e_tgt=round(tgt['e'], 4),
                alt_sat_km=round((sat['a'] - R_EARTH) / 1e3, 1),
                alt_tgt_km=round((tgt['a'] - R_EARTH) / 1e3, 1),
                da_km=round((sat['a'] - tgt['a']) / 1e3, 1),
                hohmann_dv=round(dv_h, 2), evec_dv=round(dv_e, 2),
                shape_lb_dv=round(max(dv_h, dv_e), 2))


def run(episodes=200, seed=42, same_orbit=0, verbose=False, out_csv=None,
        e_max=0.05, gap_max=math.pi, debris=False, tag='headline', trace=False,
        rows=None, box_r=30000.0, box_v=50.0, t3_mode=0,
        action_space=16, fine_radial=False, scale_consts=False):
    from pufferlib.ocean.orbital.orbital import Orbital

    # ── actuation / constant arming, before the controller is constructed ──
    global BURN_ACTIONS, DA_FLOOR, A_DEADBAND
    BURN_ACTIONS = BURN_ACTIONS_D20 if fine_radial else BURN_ACTIONS_D16
    if scale_consts:
        A_DEADBAND = min(6.0e3, A_DEADBAND_FRAC * box_r)
        DA_FLOOR = min(3.0e3, DA_FLOOR_FRAC * box_r)
    else:
        A_DEADBAND = float(os.environ.get('T3_DEADBAND', 6.0e3))
        DA_FLOOR = float(os.environ.get('T3_DA_FLOOR', 3.0e3))

    # T3 regression config (red-team #3): the exact env the recovery arms
    # train in. decode_obs switches via om.DEFAULT_PHASE_OBS_MODE.
    t3_kwargs = dict(
        shaping_mode=1, shape_gamma=1.0, phase_gap_mode=1, phase_obs_mode=1,
        episode_cap_steps=3000, cap_terminal_reward=0.0,
    ) if t3_mode else {}
    om.DEFAULT_PHASE_OBS_MODE = 1 if t3_mode else 0

    env = Orbital(
        num_envs=1,
        num_debris_min=4 if debris else 0,
        num_debris_max=8 if debris else 0,
        e_max_target=e_max,
        e_max_sat=e_max,
        init_phase_gap_max=gap_max,
        same_orbit_init=same_orbit,
        valid_init_only=1,
        gave_up_action="terminate",
        rendezvous_radius_m=box_r,
        rel_vel_tol_ms=box_v,
        legacy_action_space=action_space,
        **t3_kwargs,
    )
    # the controller aims for the inside of whatever box the env enforces
    global SAFE_R, SAFE_V
    SAFE_R, SAFE_V = 0.87 * box_r, 0.84 * box_v
    obs, _ = env.reset(seed=seed)
    ctl = ExpertController(verbose=verbose, trace=trace,
                           cap_steps=3000 if t3_mode else None)
    if rows is None:
        rows = []
    out = []

    ep = 0
    t0 = time.time()
    o = obs[0]
    sat, tgt, fuel = om.decode_obs(o)
    ep_meta = _meta(sat, tgt)

    # Per-episode terminal-geometry accumulators. `best_vrel_inbox` is the
    # quantity the actuator hypothesis is actually about: the closest the
    # controller ever gets to satisfying the VELOCITY half of the success test
    # while already satisfying the position half. `n_inbox_tau1` supports the
    # cadence red-team -- if the expert were coasting through the box on long
    # warps it would be disadvantaged relative to a tau=1 policy, and this
    # counts how often that happens.
    def _fresh():
        return dict(best_vrel_inbox=float('inf'), best_vrel=float('inf'),
                    best_d=float('inf'), n_inbox=0, n_inbox_tau1=0)
    acc = _fresh()

    while ep < episodes:
        act = ctl.act(o)
        st = env.get_state()[0]
        d_now = float(np.linalg.norm(st[23:26] - st[8:11]))
        v_now = float(np.linalg.norm(st[26:29] - st[11:14]))
        acc['best_d'] = min(acc['best_d'], d_now)
        acc['best_vrel'] = min(acc['best_vrel'], v_now)
        if d_now < box_r:
            acc['n_inbox'] += 1
            acc['best_vrel_inbox'] = min(acc['best_vrel_inbox'], v_now)
            if om.ACTION_TAU[act] == 1:
                acc['n_inbox_tau1'] += 1
        fuel_before = float(o[6])
        obs, rew, term, trunc, _ = env.step(np.array([act], dtype=np.int32))
        o = obs[0]
        if term[0]:
            sim_steps, cause = env.last_episode_result(0)
            dvp, dvr = om.ACTION_DV[act]
            dv_used = om.dv_spent(fuel_before) + math.hypot(dvp, dvr)
            nb = sum(ctl.acts[i] for i in BURN_ACTIONS)
            nw = sum(ctl.acts[i] for i in (9, 10, 11))
            row = dict(ep_meta)
            row.update(condition=tag, episode=ep, seed=seed,
                       success=int(cause == 1), cause=CAUSES[cause],
                       steps=int(sim_steps), decisions=ctl.decisions,
                       best_vrel_inbox=(round(acc['best_vrel_inbox'], 4)
                                        if acc['best_vrel_inbox'] < 1e29 else ''),
                       best_vrel=round(acc['best_vrel'], 4),
                       best_d_km=round(acc['best_d'] / 1e3, 3),
                       n_inbox=acc['n_inbox'], n_inbox_tau1=acc['n_inbox_tau1'],
                       n_fine_rad=ctl.acts[18] + ctl.acts[19],
                       dv_used=round(dv_used, 2), plan_dv=round(ctl.plan_dv, 2),
                       n_plans=ctl.n_plans, n_burns=nb, n_warps=nw,
                       n_coast=ctl.acts[0], min_d_km=round(ctl.min_d / 1e3, 2))
            rows.append(row)
            out.append(row)
            if verbose:
                print(f"  ep {ep:3d} {CAUSES[cause]:>10s} gap={row['gap_deg']:+7.1f} "
                      f"e_t={row['e_tgt']:.3f} steps={sim_steps:4d} dv={dv_used:6.1f} "
                      f"dec={ctl.decisions} min_d={row['min_d_km']:.1f}km")
            ep += 1
            ctl.reset()
            acc = _fresh()
            sat, tgt, fuel = om.decode_obs(o)
            ep_meta = _meta(sat, tgt)
            if not verbose and ep % 25 == 0:
                sr = sum(r['success'] for r in out) / len(out)
                print(f"  [{tag}] {ep}/{episodes}  running success {sr:.1%} "
                      f"({time.time()-t0:.0f}s)", flush=True)

    env.close()
    if out_csv:
        d = os.path.dirname(out_csv)
        if d:
            os.makedirs(d, exist_ok=True)
        with open(out_csv, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    return out


def summarize(rows, title):
    n = len(rows)
    ok = sum(r['success'] for r in rows)
    print(f"\n=== {title}: {ok}/{n} = {ok/n:.1%} ===")
    causes = {}
    for r in rows:
        causes[r['cause']] = causes.get(r['cause'], 0) + 1
    print("  causes: " + ", ".join(f"{k}={v}" for k, v in
                                   sorted(causes.items(), key=lambda x: -x[1])))
    print("  by |initial Delta-lambda|:")
    for lo, hi in [(0, 45), (45, 90), (90, 135), (135, 181)]:
        sel = [r for r in rows if lo <= abs(r['gap_deg']) < hi]
        if sel:
            print(f"    {lo:3d}-{hi:3d} deg  n={len(sel):3d}  "
                  f"{sum(r['success'] for r in sel)/len(sel):6.1%}")
    print("  by e_target:")
    for lo, hi in [(0, .0125), (.0125, .025), (.025, .0375), (.0375, .051)]:
        sel = [r for r in rows if lo <= r['e_tgt'] < hi]
        if sel:
            print(f"    {lo:.4f}-{hi:.4f}  n={len(sel):3d}  "
                  f"{sum(r['success'] for r in sel)/len(sel):6.1%}")
    succ = [r for r in rows if r['success']]
    if succ:
        def med(k):
            v = sorted(r[k] for r in succ)
            return v[len(v) // 2]
        print(f"  median (successes): steps={med('steps')} decisions={med('decisions')} "
              f"dv={med('dv_used'):.1f} m/s  plan_dv={med('plan_dv'):.1f}  "
              f"shape_lb={med('shape_lb_dv'):.1f}  hohmann={med('hohmann_dv'):.1f}")
    tot = sum(r['decisions'] for r in rows)
    print(f"  action mix: burns {sum(r['n_burns'] for r in rows)/tot:.1%}, "
          f"warps {sum(r['n_warps'] for r in rows)/tot:.1%}, "
          f"coast {sum(r['n_coast'] for r in rows)/tot:.1%}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--episodes', type=int, default=200)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--same-orbit', action='store_true')
    p.add_argument('--e-max', type=float, default=0.05)
    p.add_argument('--gap-max', type=float, default=math.pi)
    p.add_argument('--verbose', action='store_true')
    p.add_argument('--trace', action='store_true')
    p.add_argument('--csv', type=str, default=None)
    p.add_argument('--tag', type=str, default='headline')
    p.add_argument('--t3', action='store_true',
                   help='run against the T3 recovery env config '
                        '(shaping_mode 1, phase_obs_mode 1, cap 3000, cap reward 0)')
    p.add_argument('--box-r', type=float, default=30000.0,
                   help='rendezvous position tolerance (m)')
    p.add_argument('--box-v', type=float, default=50.0,
                   help='relative-velocity tolerance (m/s)')
    p.add_argument('--action-space', type=int, default=16,
                   choices=[16, 20, 30], help='env legacy_action_space')
    p.add_argument('--fine-radial', action='store_true',
                   help='arm B: add rows 18/19 (+-1 m/s radial) to BURN_ACTIONS '
                        '-- requires --action-space 20 or more')
    p.add_argument('--scale-consts', action='store_true',
                   help='arm C: scale A_DEADBAND and DA_FLOOR to the box at '
                        'their 30 km design fractions (0.20 and 0.10 of box_r)')
    p.add_argument('--suite', action='store_true',
                   help='200 headline episodes + 100 same_orbit_init episodes')
    a = p.parse_args()

    if a.suite:
        allrows = []
        r1 = run(episodes=200, seed=a.seed, same_orbit=0, tag='headline',
                 verbose=a.verbose, trace=a.trace, rows=allrows, t3_mode=int(a.t3))
        summarize(r1, 'HEADLINE (LEO 300-800 km, e<=0.05 both, gap +-180 deg)')
        r2 = run(episodes=100, seed=a.seed, same_orbit=1, tag='same_orbit',
                 verbose=a.verbose, trace=a.trace, rows=allrows, out_csv=a.csv,
                 t3_mode=int(a.t3))
        summarize(r2, 'SAME_ORBIT_INIT (identical a,e,omega; only theta differs)')
        return

    if a.fine_radial and a.action_space < 20:
        p.error('--fine-radial needs --action-space 20 (rows 18/19 do not '
                'exist in a 16-action env; the ctor would reject the emit)')
    rows = run(episodes=a.episodes, seed=a.seed, same_orbit=int(a.same_orbit),
               verbose=a.verbose, out_csv=a.csv, e_max=a.e_max,
               gap_max=a.gap_max, tag=a.tag, trace=a.trace, t3_mode=int(a.t3),
               box_r=a.box_r, box_v=a.box_v, action_space=a.action_space,
               fine_radial=a.fine_radial, scale_consts=a.scale_consts)
    summarize(rows, a.tag)


if __name__ == '__main__':
    main()
