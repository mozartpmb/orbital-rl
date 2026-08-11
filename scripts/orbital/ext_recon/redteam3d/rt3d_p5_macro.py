"""RT3D-P5 — attack the coast-to-next-relative-node macro (action 26) and the ladder.

C1  Is the macro NEEDED?  Simulate the plane crank with the ONLY granularity the
    existing action set offers (tau in {1, 5, 30, 60, 180, 360}) and measure the
    realised Delta-v against the ideal 2 v sin(di/2).  If tau<=5 costs a couple of
    percent, the macro is deferrable and its implementation risk is avoidable.

C2  Discount economics.  gamma is FLAT PER DECISION (0.995).  A variable-tau
    action changes sim-seconds-per-decision.  Tabulate seconds/decision and
    terminal visibility gamma^N for each granularity at each rung, and check
    whether the macro creates a NEW extreme beyond the existing warp ladder.

C3  The Delta-i -> 0 clamp.  tau jumps from ~P/2 to 1 at the clamp threshold.
    Quantify the size of the discontinuity in sim-seconds-per-decision.

G1  Joint Dv feasibility of X3 (LEO 300-800, e<=0.05, di 1.0 deg): in-plane
    two-impulse + e-match + drift-phasing surcharge + plane leg vs the 478 m/s
    budget, independent of 3d_B's script.
"""
import csv
import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rt3d_common import MU, R_EARTH, BUDGET_DV, pct

OUT_CRANK = "/Users/pete/space_training/web_data/results/ext_rt3d_crank_granularity.csv"
OUT_ECON = "/Users/pete/space_training/web_data/results/ext_rt3d_macro_economics.csv"
OUT_FEAS = "/Users/pete/space_training/web_data/results/ext_rt3d_ladder_feasibility.csv"

DT = 60.0
GAMMA = 0.995


def crank(di, v, quanta, tau, n_trials=400, seed=5):
    """Greedy crank of the relative-inclination VECTOR under timing quantization.

    State = (x, y) inclination-vector error; the node is at atan2(y, x) == 0.
    A normal burn of q at argument-of-latitude offset du removes q/v * cos(du)
    along the current node axis and injects q/v * sin(du) orthogonal.
    The agent may only act every `tau` sub-steps; per-sub-step along-track motion
    is 360 deg / (P/DT).
    """
    rng = random.Random(seed)
    P = 2 * math.pi * math.sqrt((MU / v ** 2) ** 3 / MU)   # r = MU/v^2 for circular
    du_step = 2 * math.pi * DT / P
    tot = []
    for _ in range(n_trials):
        x, y = di, 0.0
        u = rng.random() * 2 * math.pi     # where we happen to be
        spent = 0.0
        for _ in range(4000):
            err = math.hypot(x, y)
            if err < 0.02 * di or err * v < 0.5:
                break
            # node direction of the CURRENT error, in argument-of-latitude terms
            node_u = math.atan2(y, x)
            # phase offset from the node if we burn right now
            duo = (node_u - u + math.pi) % (2 * math.pi) - math.pi
            # burn now if we are within half a decision interval of the node,
            # or if waiting costs more than a full revolution
            half = max(du_step * tau / 2.0, du_step / 2.0)
            if abs(duo) <= half or abs(abs(duo) - math.pi) <= half:
                sgn = 1.0 if abs(duo) <= math.pi / 2 else -1.0
                q = min(max(quanta), err * v)
                q = max([qq for qq in quanta if qq <= q] or [min(quanta)])
                dtheta = q / v
                # effective removal along the node axis, orthogonal injection
                eff = math.cos(duo) * sgn
                orth = math.sin(duo) * sgn
                ex, ey = x - dtheta * eff * math.cos(node_u), y - dtheta * eff * math.sin(node_u)
                ex -= dtheta * orth * -math.sin(node_u)
                ey -= dtheta * orth * math.cos(node_u)
                x, y = ex, ey
                spent += q
                u = (u + du_step) % (2 * math.pi)
            else:
                u = (u + du_step * tau) % (2 * math.pi)
        tot.append(spent)
    ideal = 2 * v * math.sin(di / 2)
    return pct(tot, .5) / ideal, pct(tot, .9) / ideal, ideal


def run_crank():
    rows = []
    for alt_km in (550, 8000, 20200):
        r = R_EARTH + alt_km * 1e3
        v = math.sqrt(MU / r)
        P = 2 * math.pi * math.sqrt(r ** 3 / MU)
        for di_deg in (0.25, 0.75, 1.0, 2.0):
            di = math.radians(di_deg)
            for tau, label in ((1, "coast tau=1"), (5, "warp-5min"), (30, "warp-30min"),
                               (60, "warp-1hr"), (180, "warp-3hr"), (360, "warp-6hr")):
                m50, m90, ideal = crank(di, v, (1.0, 10.0, 25.0), tau)
                rows.append(dict(alt_km=alt_km, di_deg=di_deg, tau=tau, label=label,
                                 period_min=round(P / 60, 1),
                                 substeps_per_orbit=round(P / DT, 1),
                                 deg_per_decision=round(360.0 * tau * DT / P, 2),
                                 ideal_dv_ms=round(ideal, 1),
                                 realised_over_ideal_p50=round(m50, 4),
                                 realised_over_ideal_p90=round(m90, 4),
                                 excess_p90_ms=round((m90 - 1) * ideal, 1)))
            print(f"alt={alt_km:6d} di={di_deg:4.2f} (ideal {ideal:6.1f} m/s): " +
                  "  ".join(f"tau{r_['tau']}={r_['realised_over_ideal_p90']:5.2f}x"
                            for r_ in rows[-6:]))
    with open(OUT_CRANK, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print("wrote", OUT_CRANK, "\n")


def run_econ():
    rows = []
    for name, alt_km, cap in (("X3_LEO", 550, 3000), ("X4_wide", 8000, 6000),
                              ("X5_MEO", 20200, 12000)):
        r = R_EARTH + alt_km * 1e3
        P = 2 * math.pi * math.sqrt(r ** 3 / MU)
        horizon_s = cap * DT
        for tau, label in ((1, "coast"), (5, "warp-5min"), (30, "warp-30min"),
                           (60, "warp-1hr"), (180, "warp-3hr"), (360, "warp-6hr"),
                           (int(round(P / 2 / DT)), "MACRO coast-to-node (P/2)"),
                           (int(round(P / DT)), "MACRO clamp (one period)")):
            tau = max(1, tau)
            ndec = horizon_s / (tau * DT)
            rows.append(dict(rung=name, alt_km=alt_km, cap_steps=cap, tau=tau,
                             label=label, sim_s_per_decision=round(tau * DT, 1),
                             decisions_to_cap=round(ndec, 1),
                             terminal_visibility_gamma_pow=round(GAMMA ** ndec, 6)))
        best_existing = max(x["terminal_visibility_gamma_pow"] for x in rows
                            if x["rung"] == name and "MACRO" not in x["label"])
        for x in rows:
            if x["rung"] == name and "MACRO" in x["label"]:
                x["vs_best_existing_warp"] = round(
                    x["terminal_visibility_gamma_pow"] / best_existing, 3)
        print(f"{name}: period {P/60:6.1f} min; macro tau = {int(round(P/2/DT))}-"
              f"{int(round(P/DT))} substeps vs warp set max 360. "
              f"gamma^N: best warp {best_existing:.4f}, macro "
              + ", ".join(f"{x['label'].split()[0]}={x['terminal_visibility_gamma_pow']:.4f}"
                          for x in rows if x['rung'] == name and 'MACRO' in x['label']))
    # C3 the clamp discontinuity
    for name, alt_km in (("X3_LEO", 550), ("X5_MEO", 20200)):
        r = R_EARTH + alt_km * 1e3
        P = 2 * math.pi * math.sqrt(r ** 3 / MU)
        print(f"  {name} clamp cliff: tau goes {int(round(P/2/DT))} -> 1 at the "
              f"threshold = {int(round(P/2/DT))}x jump in sim-s/decision "
              f"({P/2:.0f} s -> 60 s)")
    with open(OUT_ECON, "w", newline="") as f:
        keys = ["rung", "alt_km", "cap_steps", "tau", "label", "sim_s_per_decision",
                "decisions_to_cap", "terminal_visibility_gamma_pow",
                "vs_best_existing_warp"]
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)
    print("wrote", OUT_ECON, "\n")


def run_feasibility(n=20000):
    """Independent joint Dv screen. in-plane = Hohmann(a_s->a_t) + e-match
    + drift-phasing surcharge; plane = 2 v sin(di/2) (best of two nodes)."""
    import importlib
    p3 = importlib.import_module("rt3d_p3_phi")
    rows = []
    for cfg in p3.RUNGS:
        name, amin, amax, emt, ems, demax, damax, same, di_deg, dvref = cfg
        rng = random.Random(2718)
        tot, ok_seq, ok_comb = [], 0, 0
        for _ in range(n):
            a_s, e_s, w_s, a_t, e_t, w_t = p3.sample_init(rng, cfg)
            _, _, _, _, wh_t, wh_s, delta = p3.plane_pair(rng, math.radians(di_deg))
            v_s, v_t = math.sqrt(MU / a_s), math.sqrt(MU / a_t)
            # Hohmann between the two semi-major axes
            at = 0.5 * (a_s + a_t)
            dv_h = abs(math.sqrt(MU * (2 / a_s - 1 / at)) - v_s) + \
                   abs(v_t - math.sqrt(MU * (2 / a_t - 1 / at)))
            # e-vector match
            esx, esy = e_s * math.cos(w_s), e_s * math.sin(w_s)
            etx, ety = e_t * math.cos(w_t), e_t * math.sin(w_t)
            dv_e = 0.5 * v_t * math.hypot(esx - etx, esy - ety)
            # drift-phasing surcharge: the T3 screen's model, uniform gap +-180
            gap = (rng.random() * 2 - 1) * math.pi
            # drift orbit da that closes the gap inside the cap
            cap_s = (3000 if amax < 1.5e7 else (6000 if amax < 2e7 else 12000)) * DT
            P_t = 2 * math.pi * math.sqrt(a_t ** 3 / MU)
            nrev = max(1.0, 0.5 * cap_s / P_t)
            dadrift = abs(gap) / (2 * math.pi * nrev) * (2.0 / 3.0) * a_t
            dv_drift = 2 * (0.5 * v_t * dadrift / a_t)
            dv_in = dv_h + dv_e + dv_drift
            dv_pl = 2 * v_t * math.sin(delta / 2)
            seq = dv_in + dv_pl
            comb = math.hypot(dv_in, dv_pl) if dv_in > 0 else dv_pl
            tot.append(seq)
            ok_seq += seq <= BUDGET_DV
            ok_comb += comb <= BUDGET_DV
        rows.append(dict(rung=name, di_max_deg=di_deg,
                         dv_tot_p50=round(pct(tot, .5), 1),
                         dv_tot_p90=round(pct(tot, .9), 1),
                         dv_tot_p99=round(pct(tot, .99), 1),
                         screen_sequential=round(ok_seq / n, 4),
                         screen_combined=round(ok_comb / n, 4)))
        print(f"{name:24s} dv p50={pct(tot,.5):6.1f} p90={pct(tot,.9):6.1f} "
              f"p99={pct(tot,.99):6.1f}  feasible(seq)={ok_seq/n:6.2%} "
              f"(comb)={ok_comb/n:6.2%}")
    with open(OUT_FEAS, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print("wrote", OUT_FEAS)


if __name__ == "__main__":
    run_crank()
    run_econ()
    run_feasibility()
