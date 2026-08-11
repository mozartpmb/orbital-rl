#!/usr/bin/env python3
"""ext-3d wide lineage: per-episode feature extraction from eval trajectory npz.

Reads /tmp/t5_w3d_eval_W{3,4} (200 episodes each), emits one row per episode with
init geometry, realized spend, action mix, endgame residuals and an analytic
Delta-v-need proxy, so failures can be attributed to budget vs policy.

Nothing here touches env code. Output: /tmp/w3d_episodes.csv
"""
import csv
import math
import os
import sys

import numpy as np

MU = 3.986004418e14
R_EARTH = 6.371e6
DT = 60.0
ISP, G0 = 300.0, 9.80665
BUDGET = ISP * G0 * math.log(1.0 / (1.0 - 0.15))   # 478.13 m/s

# ACTION_TAU from orbital.h (Discrete-30)
TAU = ([1] * 9) + [5, 30, 60] + [1, 1, 1, 1] + [180, 360] + [1, 1] + \
      [1] * 6 + [1] * 4
assert len(TAU) == 30
WARP_ACTIONS = {9, 10, 11, 16, 17}
NORMAL_ACTIONS = set(range(20, 26))
COMBINED_ACTIONS = set(range(26, 30))
INPLANE_BURNS = {1, 2, 3, 4, 5, 6, 7, 8, 12, 13, 14, 15, 18, 19}

TERM = {0: "none", 1: "success", 2: "collision", 3: "escape",
        4: "safety_cap", 5: "stranded", 6: "hyperbolic", 7: "gave_up"}

RUNGS = {
    "W3": dict(dir="/tmp/t5_w3d_eval_W3", e_max=0.15, de_max=0.06,
               da_max_km=400.0, a_lo=6.671e6, a_hi=8.371e6, di_deg=1.0,
               cap=3000),
    "W4": dict(dir="/tmp/t5_w3d_eval_W4", e_max=0.30, de_max=0.08,
               da_max_km=600.0, a_lo=6.671e6, a_hi=14.371e6, di_deg=1.0,
               cap=6000),
}


def hhat(inc, raan):
    si, ci = math.sin(inc), math.cos(inc)
    return (si * math.sin(raan), -si * math.cos(raan), ci)


def plane_angle(w1, w2):
    cx = w1[1] * w2[2] - w1[2] * w2[1]
    cy = w1[2] * w2[0] - w1[0] * w2[2]
    cz = w1[0] * w2[1] - w1[1] * w2[0]
    dot = sum(a * b for a, b in zip(w1, w2))
    return math.atan2(math.sqrt(cx * cx + cy * cy + cz * cz), dot)


def evec(e, inc, raan, argp):
    co, so = math.cos(raan), math.sin(raan)
    cw, sw = math.cos(argp), math.sin(argp)
    ci, si = math.cos(inc), math.sin(inc)
    return (e * (co * cw - so * sw * ci),
            e * (so * cw + co * sw * ci),
            e * (sw * si))


def hoh(a1, a2):
    """Two-impulse Hohmann: (total dv, half-period transfer time, largest leg)."""
    at = 0.5 * (a1 + a2)
    dv1 = abs(math.sqrt(MU / a1) * (math.sqrt(2 * a2 / (a1 + a2)) - 1.0))
    dv2 = abs(math.sqrt(MU / a2) * (1.0 - math.sqrt(2 * a1 / (a1 + a2))))
    return dv1 + dv2, math.pi * math.sqrt(at ** 3 / MU), max(dv1, dv2)


def v_at(a, e, nu):
    r = a * (1 - e * e) / (1 + e * math.cos(nu))
    return math.sqrt(MU * (2.0 / r - 1.0 / a))


def true_to_mean(nu, e):
    E = 2.0 * math.atan2(math.sqrt(max(1.0 - e, 0.0)) * math.sin(nu / 2.0),
                         math.sqrt(1.0 + e) * math.cos(nu / 2.0))
    return E - e * math.sin(E)


def decisions(actions):
    """Walk the per-sub-step action log, grouping warps by their tau."""
    out = []
    i, n = 0, len(actions)
    while i < n:
        a = int(actions[i])
        if a < 0 or a >= 30:
            i += 1
            continue
        out.append(a)
        i += TAU[a]
    return out


def analyze(path, rung):
    d = np.load(path, allow_pickle=True)
    ns = int(d["episode_steps"][0])
    cause = int(d["terminal_cause"][0])
    sl = slice(0, ns + 1)

    sa, se, som = float(d["sat_a"][0]), float(d["sat_e"][0]), float(d["sat_omega"][0])
    si, sr = float(d["sat_inc"][0]), float(d["sat_raan"][0])
    ta, te, tom = float(d["target_a"][0]), float(d["target_e"][0]), float(d["target_omega"][0])
    ti, tr = float(d["target_inc"][0]), float(d["target_raan"][0])

    di = plane_angle(hhat(si, sr), hhat(ti, tr))
    ev_s, ev_t = evec(se, si, sr, som), evec(te, ti, tr, tom)
    de0 = math.sqrt(sum((a - b) ** 2 for a, b in zip(ev_s, ev_t)))
    da0 = abs(sa - ta)

    # initial mean-longitude gap (target plane is the i=O=0 gauge here)
    lam_s = true_to_mean(float(d["sat_theta"][0]), se) + som
    # target theta from its cartesian state
    tx, ty = float(d["target_x"][0]), float(d["target_y"][0])
    nu_t = math.atan2(ty, tx) - tom
    lam_t = true_to_mean(nu_t, te) + tom
    dlam = (lam_s - lam_t + math.pi) % (2 * math.pi) - math.pi

    dv = np.asarray(d["delta_v"][sl], dtype=float)
    dv_spent = float(dv.sum())
    fuel = np.asarray(d["fuel"][sl], dtype=float)
    fuel_end = float(fuel[-1])
    exhausted = fuel_end <= 1e-9

    acts = decisions(np.asarray(d["action"][sl]))
    nd = max(len(acts), 1)
    n_norm = sum(1 for a in acts if a in NORMAL_ACTIONS)
    n_comb = sum(1 for a in acts if a in COMBINED_ACTIONS)
    n_warp = sum(1 for a in acts if a in WARP_ACTIONS)
    n_ip = sum(1 for a in acts if a in INPLANE_BURNS)
    n_coast = sum(1 for a in acts if a == 0)

    # plane delta-v actually applied out of plane (|normal| component)
    dv_normal = 25.0 * (sum(1 for a in acts if a in (24, 25)) +
                        sum(1 for a in acts if a in (26, 27, 28, 29))) \
        + 10.0 * sum(1 for a in acts if a in (22, 23)) \
        + 1.0 * sum(1 for a in acts if a in (20, 21))

    # endgame residuals
    di_end = plane_angle(hhat(float(d["sat_inc"][ns]), float(d["sat_raan"][ns])),
                         hhat(float(d["target_inc"][ns]), float(d["target_raan"][ns])))
    ev_se = evec(float(d["sat_e"][ns]), float(d["sat_inc"][ns]),
                 float(d["sat_raan"][ns]), float(d["sat_omega"][ns]))
    ev_te = evec(float(d["target_e"][ns]), float(d["target_inc"][ns]),
                 float(d["target_raan"][ns]), float(d["target_omega"][ns]))
    de_end = math.sqrt(sum((a - b) ** 2 for a, b in zip(ev_se, ev_te)))
    da_end = abs(float(d["sat_a"][ns]) - float(d["target_a"][ns]))

    # closest approach over the episode
    rx = np.asarray(d["sat_x"][sl]) - np.asarray(d["target_x"][sl])
    ry = np.asarray(d["sat_y"][sl]) - np.asarray(d["target_y"][sl])
    rz = np.asarray(d["sat_z"][sl]) - np.asarray(d["target_z"][sl])
    rng = np.sqrt(rx * rx + ry * ry + rz * rz)
    vx = np.asarray(d["sat_vx"][sl]) - np.asarray(d["target_vx"][sl])
    vy = np.asarray(d["sat_vy"][sl]) - np.asarray(d["target_vy"][sl])
    vz = np.asarray(d["sat_vz"][sl]) - np.asarray(d["target_vz"][sl])
    rv = np.sqrt(vx * vx + vy * vy + vz * vz)
    k = int(np.argmin(rng))
    # best "joint" step: closest step among those with rel_vel under tolerance
    ok = rv < 50.0
    r_min_velok = float(rng[ok].min()) if ok.any() else float("nan")

    # ── analytic delta-v need (same cost model as ext_3d_joint_feasibility) ──
    dv_a, t_h, dv_big_a = hoh(sa, ta)
    vc = math.sqrt(MU / (0.5 * (sa + ta)))
    dv_e = vc * de0 / 2.0
    u_node = math.atan2(hhat(si, sr)[0] * 0 + 1e-30, 1.0)  # placeholder, set below
    # relative node: the line where the two planes intersect = h_s x h_t
    hs, ht = hhat(si, sr), hhat(ti, tr)
    nx = hs[1] * ht[2] - hs[2] * ht[1]
    ny = hs[2] * ht[0] - hs[0] * ht[2]
    nz = hs[0] * ht[1] - hs[1] * ht[0]
    nn = math.sqrt(nx * nx + ny * ny + nz * nz)
    if nn > 1e-14:
        u_node = math.atan2(ny * 1.0, nx * 1.0)  # in the target plane (i_t=0)
    else:
        u_node = 0.0
    cand = []
    for du in (0.0, math.pi):
        cand.append(v_at(sa, se, u_node + du - som))
        cand.append(v_at(ta, te, u_node + du - tom))
    v_node = min(cand)
    dv_pl = 2.0 * v_node * math.sin(0.5 * di)
    dv_big = max(dv_big_a, dv_e)
    need_seq = dv_a + dv_e + dv_pl
    need_comb = (dv_a + dv_e - dv_big) + math.hypot(dv_big, dv_pl)

    cfg = RUNGS[rung]
    return dict(
        rung=rung, ep=int(d["episode_id"][0]), cause=cause,
        cause_name=TERM.get(cause, "?"), success=int(cause == 1),
        steps=ns, ep_reward=float(d["episode_reward"][0]),
        init_attempts=int(d["last_init_attempts"][0]),
        di_rel_deg=math.degrees(di), da0_km=da0 / 1e3, de0=de0,
        e_sat0=se, e_tgt0=te, e_max_pair=max(se, te),
        a_sat_km=(sa - R_EARTH) / 1e3, a_tgt_km=(ta - R_EARTH) / 1e3,
        band_pos=(sa - cfg["a_lo"]) / (cfg["a_hi"] - cfg["a_lo"]),
        dlam0_deg=math.degrees(dlam),
        rp_sat_km=(sa * (1 - se) - R_EARTH) / 1e3,
        rp_tgt_km=(ta * (1 - te) - R_EARTH) / 1e3,
        dv_spent=dv_spent, dv_frac_budget=dv_spent / BUDGET,
        fuel_end=fuel_end, exhausted=int(exhausted),
        n_decisions=nd, n_normal=n_norm, n_combined=n_comb,
        n_plane=n_norm + n_comb, frac_plane=(n_norm + n_comb) / nd,
        n_warp=n_warp, frac_warp=n_warp / nd, n_inplane=n_ip, n_coast=n_coast,
        dv_normal_cmd=dv_normal,
        di_end_deg=math.degrees(di_end), de_end=de_end, da_end_km=da_end / 1e3,
        r_min_km=float(rng[k]) / 1e3, relv_at_rmin=float(rv[k]),
        r_min_velok_km=r_min_velok / 1e3 if r_min_velok == r_min_velok else float("nan"),
        need_seq=need_seq, need_comb=need_comb,
        margin_seq=BUDGET - need_seq, margin_comb=BUDGET - need_comb,
        feas_seq=int(need_seq <= BUDGET), feas_comb=int(need_comb <= BUDGET),
        dv_pl=dv_pl, dv_hoh=dv_a, dv_e=dv_e, t_hoh_steps=t_h / DT,
    )


def main():
    rows = []
    for rung, cfg in RUNGS.items():
        files = sorted(os.listdir(cfg["dir"]))
        for fn in files:
            if not fn.endswith(".npz"):
                continue
            rows.append(analyze(os.path.join(cfg["dir"], fn), rung))
    out = "/tmp/w3d_episodes.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out}  n={len(rows)}")
    for rung in RUNGS:
        r = [x for x in rows if x["rung"] == rung]
        print(f"  {rung}: n={len(r)} success={sum(x['success'] for x in r)}")


if __name__ == "__main__":
    main()
