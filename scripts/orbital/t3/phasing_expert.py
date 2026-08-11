#!/usr/bin/env python3
"""
T3: closed-loop analytic phasing expert, run against the REAL C env.

Purpose: convert the feasibility tables from "analytically feasible" to
"demonstrated feasible under the shipped corrected-dynamics env". This is a
hand-written controller, not a learned policy — it exists to establish that a
solution EXISTS inside the env's action set, horizon and dv budget, so that a
training failure can be attributed to learning rather than to physics.

Strategy (the drift-orbit maneuver the shaping potential currently punishes):
  1. TRIM   : if |da| already large enough to phase within the horizon, use it.
              else open a drift orbit by burning tangentially to da_cmd.
  2. DRIFT  : warp/coast while the relative phase closes; stop when the
              remaining gap equals what the closing burn sequence will consume.
  3. CLOSE  : burn tangentially back toward da = 0, but deliberately leave a
              residual |da| < 88 km (the 50 m/s rel-vel box) so the along-track
              closure continues into the 30 km position box.
  4. TERMINAL: fine burns (1/2/5 m/s) to hold |v_rel| under tolerance.

Run: python3 scripts/orbital/t3/phasing_expert.py
"""
import csv
import math
import os
import sys

import numpy as np

sys.path.insert(0, "/Users/pete/space_training/pufferlib")
sys.path.insert(0, "/Users/pete/space_training/scripts/orbital/nav")

from pufferlib.ocean.orbital.orbital import Orbital          # noqa: E402
import orbital_math as om                                     # noqa: E402

MU, R_EARTH, DT = 3.986004418e14, 6.371e6, 60.0
MAX_STEPS = 2000
DV_BUDGET = 300.0 * 9.80665 * math.log(1 / 0.85)
OUT = "/Users/pete/space_training/web_data/results"

# action index -> (dv_prograde, dv_radial), tau
A_DV = {0: 0.0, 1: 5.0, 2: 10.0, 3: 25.0, 4: -5.0, 5: -10.0, 6: -25.0,
        12: 1.0, 13: -1.0, 14: 2.0, 15: -2.0}
A_TAU = {0: 1, 9: 5, 10: 30, 11: 60}
BURNS = sorted(A_DV.items(), key=lambda kv: -abs(kv[1]))      # 25,25,10,...


def wrap(x):
    return (x + math.pi) % (2 * math.pi) - math.pi


def n_of(a):
    return math.sqrt(MU / a**3)


def pick_burn(dv_needed):
    """Largest available burn quantum not overshooting dv_needed (signed)."""
    best, bestmag = 0, 0.0
    for idx, dv in A_DV.items():
        if dv == 0:
            continue
        if dv * dv_needed > 0 and abs(dv) <= abs(dv_needed) + 1e-9 \
                and abs(dv) > bestmag:
            best, bestmag = idx, abs(dv)
    if best == 0:                       # dv_needed smaller than finest quantum
        return 0
    return best


def pick_warp(steps_left, allow):
    for idx in (11, 10, 9):
        if idx in allow and A_TAU[idx] <= steps_left:
            return idx
    return 0


def run_episode(env, obs, warp_allow=(9, 10, 11), da_cmd_km=200.0,
                res_da_km=60.0, verbose=False):
    """Drive one episode. Returns dict of outcome metrics."""
    sat, tgt = om.recover_states(obs[0])
    a_t = tgt["a"]
    vc = math.sqrt(MU / a_t)
    dv_used = 0.0
    steps = 0
    decisions = 0
    phase = "TRIM"
    res_da = res_da_km * 1e3
    spin = 0            # guard: consecutive phase switches with no env.step

    while True:
        if spin > 6:    # phase logic is oscillating — coast one step to break it
            obs, rew, term, trunc, info = env.step(np.array([0], dtype=np.int32))
            decisions += 1
            steps += 1
            spin = 0
            if term[0] or trunc[0] or steps >= MAX_STEPS - 1:
                sat, tgt = om.recover_states(obs[0])
                return dict(outcome="spin_abort", steps=steps,
                            decisions=decisions, dv_used=dv_used,
                            dist_km=float("nan"), relv=float("nan"))
            continue
        sat, tgt = om.recover_states(obs[0])
        a_s, a_t = sat["a"], tgt["a"]
        da = a_s - a_t
        # relative mean-anomaly gap the chaser must still close
        dth = wrap((sat["M"] + sat["omega"]) - (tgt["M"] + tgt["omega"]))
        w = n_of(a_s) - n_of(a_t)                 # rad/s, chaser gains if >0
        steps_left = MAX_STEPS - steps

        # geometric distance / rel-vel (for the terminal phase)
        sc = om.orbit_to_cartesian(sat)
        tc = om.orbit_to_cartesian(tgt)
        dist = math.hypot(sc[0] - tc[0], sc[1] - tc[1])
        relv = math.hypot(sc[2] - tc[2], sc[3] - tc[3])

        # ---- decide ------------------------------------------------------
        act = 0
        if phase == "TRIM":
            # required drift direction: close dth (chaser behind -> speed up)
            # to drive dth -> 0 we need w = n(a_s) - n(a_t) opposite in sign to
            # dth. n decreases with a, so dth > 0 (chaser ahead) => RAISE a.
            want_sign = 1.0 if dth > 0 else -1.0
            da_want = want_sign * da_cmd_km * 1e3
            # if the natural da already drifts the right way fast enough, keep it
            t_need = abs(dth) / abs(w) if abs(w) > 1e-12 else math.inf
            natural_ok = (w * (-dth) > 0) and (t_need / DT < 0.75 * steps_left)
            if natural_ok:
                phase = "DRIFT"; spin += 1
                continue
            err = da_want - da
            dv_need = vc * err / (2.0 * a_t)
            if abs(dv_need) < 1.0:
                phase = "DRIFT"; spin += 1
                continue
            act = pick_burn(dv_need) or 0
            if act == 0:
                phase = "DRIFT"; spin += 1
                continue

        elif phase == "DRIFT":
            if abs(w) < 1e-12:
                phase = "TRIM"; spin += 1
                continue
            if w * (-dth) <= 0:                    # drifting the wrong way
                phase = "TRIM"; spin += 1
                continue
            t_to_close = abs(dth) / abs(w)
            # dv (and therefore steps) needed to shed down to the residual
            dv_close = abs(vc * (abs(da) - res_da) / (2.0 * a_t))
            steps_close = max(1.0, math.ceil(dv_close / 25.0))
            # phase consumed while closing
            lead = abs(w) * steps_close * DT * 1.0
            if t_to_close <= lead + abs(w) * DT:
                phase = "CLOSE"; spin += 1
                continue
            remain = int((t_to_close - lead) / DT)
            act = pick_warp(min(remain, steps_left - 2), warp_allow)

        elif phase == "CLOSE":
            target_da = math.copysign(res_da, da) if abs(da) > res_da else da
            err = target_da - da
            dv_need = vc * err / (2.0 * a_t)
            if abs(dv_need) >= 1.0:
                act = pick_burn(dv_need) or 0
            else:
                phase = "TERMINAL"; spin += 1
                continue

        else:  # TERMINAL — coast into the box, shed rel-vel if over tolerance
            if relv > 45.0:
                dv_need = -math.copysign(min(relv - 40.0, 25.0), da)
                act = pick_burn(dv_need) or 0
            else:
                act = 0

        # ---- act ---------------------------------------------------------
        obs, rew, term, trunc, info = env.step(np.array([act], dtype=np.int32))
        spin = 0
        decisions += 1
        steps += A_TAU.get(act, 1)
        dv_used += abs(A_DV.get(act, 0.0))
        if verbose and decisions % 20 == 0:
            print(f"  d={decisions:4d} s={steps:4d} {phase:8s} a={act:2d} "
                  f"da={da/1e3:8.1f}km dth={math.degrees(dth):7.2f} "
                  f"dist={dist/1e3:9.1f}km relv={relv:7.2f}")
        if dv_used > DV_BUDGET:
            return dict(outcome="dv_exhausted", steps=steps, decisions=decisions,
                        dv_used=dv_used, dist_km=dist / 1e3, relv=relv)
        if term[0] or trunc[0]:
            success = float(rew[0]) > 0.0 and dist < 30_000.0 and relv < 50.0
            # env resets internally; classify from the last pre-reset geometry
            return dict(outcome="success" if dist < 30_000 and relv < 50
                        else "terminal_other",
                        reward=float(rew[0]), steps=steps, decisions=decisions,
                        dv_used=dv_used, dist_km=dist / 1e3, relv=relv)
        if steps >= MAX_STEPS - 1:
            return dict(outcome="timeout", steps=steps, decisions=decisions,
                        dv_used=dv_used, dist_km=dist / 1e3, relv=relv)


def main():
    scen = []
    # (label, same_orbit, phase_gap_deg, warp set)
    for gap in (30.0, 90.0, 180.0):
        for wname, wallow in (("legacy w5", (9,)), ("D16 w5/30/60", (9, 10, 11))):
            scen.append((f"same-orbit gap={gap:.0f} [{wname}]", 1, gap, wallow))
    for gap in (30.0, 90.0, 180.0):
        scen.append((f"headline gap={gap:.0f} [D16 w5/30/60]", 0, gap, (9, 10, 11)))

    rows = []
    for label, same, gap, wallow in scen:
        succ = 0
        n = 8
        recs = []
        for seed in range(n):
            kw = dict(num_envs=1, e_max_target=0.0, e_max_sat=0.0,
                      same_orbit_init=same, valid_init_only=1,
                      phase_gap_fixed=math.radians(gap), seed=1000 + seed)
            env = Orbital(**kw)
            obs, _ = env.reset(seed=1000 + seed)
            try:
                r = run_episode(env, obs, warp_allow=wallow)
            except Exception as e:
                r = dict(outcome=f"err:{e}", steps=0, decisions=0, dv_used=0,
                         dist_km=float("nan"), relv=float("nan"))
            recs.append(r)
            succ += (r["outcome"] == "success")
        med = lambda k: float(np.median([x[k] for x in recs]))
        rows.append(dict(scenario=label, n=n, success=succ, success_rate=succ / n,
                         med_steps=med("steps"), med_decisions=med("decisions"),
                         med_dv_ms=med("dv_used"),
                         med_final_dist_km=med("dist_km"),
                         med_final_relv_ms=med("relv"),
                         outcomes=";".join(sorted({x["outcome"] for x in recs}))))
        print(f"{label:42s} succ {succ}/{n}  steps~{med('steps'):.0f} "
              f"dec~{med('decisions'):.0f} dv~{med('dv_used'):.0f} "
              f"dist~{med('dist_km'):.0f}km relv~{med('relv'):.1f}")

    os.makedirs(OUT, exist_ok=True)
    p = f"{OUT}/t3_phasing_expert.csv"
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("wrote", p)


if __name__ == "__main__":
    main()
