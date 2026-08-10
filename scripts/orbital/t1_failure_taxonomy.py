#!/usr/bin/env python3
"""T1 failure taxonomy: decompose safety-cap timeouts into named physical modes.

Reads eval trajectory .npz dirs, extracts per-episode geometry for every
non-success terminal, and clusters cap failures into named modes.

Usage: python3 t1_failure_taxonomy.py DIR [DIR ...] --out CSV
"""
import argparse
import glob
import json
import os

import numpy as np

MU = 3.986004418e14
R_EARTH = 6.371e6
SUCC_DIST = 30_000.0   # m
SUCC_VEL = 50.0        # m/s
TERM = {0: "none", 1: "success", 2: "collision", 3: "escape", 4: "safety_cap",
        5: "stranded", 6: "hyperbolic"}


def wrap_pi(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


def load_ep(path):
    d = np.load(path)
    n = int(d["episode_steps"][0])
    L = d["sat_x"].shape[0]
    nrec = min(L, n + 1)
    g = lambda k: np.asarray(d[k][:nrec], dtype=np.float64)
    ep = dict(
        path=path,
        cause=int(d["terminal_cause"][0]),
        steps=n,
        nrec=nrec,
        reward=float(d["episode_reward"][0]),
        t=g("sim_time"),
        sx=g("sat_x"), sy=g("sat_y"), svx=g("sat_vx"), svy=g("sat_vy"),
        sa=g("sat_a"), se=g("sat_e"),
        tx=g("target_x"), ty=g("target_y"), tvx=g("target_vx"), tvy=g("target_vy"),
        ta=g("target_a"), te=g("target_e"),
        fuel=g("fuel"), action=g("action"), dv=g("delta_v"),
    )
    return ep


def metrics(ep):
    dx = ep["sx"] - ep["tx"]
    dy = ep["sy"] - ep["ty"]
    dist = np.hypot(dx, dy)
    rvx = ep["svx"] - ep["tvx"]
    rvy = ep["svy"] - ep["tvy"]
    relv = np.hypot(rvx, rvy)

    ang_s = np.arctan2(ep["sy"], ep["sx"])
    ang_t = np.arctan2(ep["ty"], ep["tx"])
    phase = np.degrees(wrap_pi(ang_s - ang_t))  # signed sat-ahead-of-target

    da = ep["sa"] - ep["ta"]
    r_sat = np.hypot(ep["sx"], ep["sy"])

    i_cl = int(np.argmin(dist))
    # joint normalized distance to the success box; <=1 anywhere == success
    joint = np.maximum(dist / SUCC_DIST, relv / SUCC_VEL)
    i_j = int(np.argmin(joint))

    in30 = dist < SUCC_DIST
    in60 = dist < 60_000.0
    in100 = dist < 100_000.0
    slow = relv < SUCC_VEL

    burn = ep["dv"] > 0
    nb = int(burn.sum())
    t_first = float(ep["t"][burn][0]) if nb else np.nan
    t_last = float(ep["t"][burn][-1]) if nb else np.nan

    tail = slice(max(0, ep["nrec"] - 300), ep["nrec"])
    n_burns_last300 = int(burn[tail].sum())
    lo, hi = int(0.25 * ep["nrec"]), int(0.7 * ep["nrec"])
    plateau_da_km = float(np.median(da[lo:hi]) / 1e3) if hi > lo else float(da[0] / 1e3)
    # phase-gap zero crossings (rendezvous alignment windows), ignoring ±180 wrap
    sgn = np.sign(phase)
    n_align = 0
    align_dist = []
    for i in range(1, ep["nrec"]):
        if sgn[i] != sgn[i - 1] and abs(phase[i] - phase[i - 1]) < 180.0:
            n_align += 1
            align_dist.append(dist[i] / 1e3)
    # phase drift rate over the last 500 records (deg per 1000 records)
    k = min(500, ep["nrec"] - 1)
    dph = wrap_pi(np.radians(phase[-1] - phase[-1 - k]))
    drift = np.degrees(dph) / k * 1000.0 if k > 0 else np.nan

    m = dict(
        n_rec=ep["nrec"],
        steps=ep["steps"],
        cause=TERM.get(ep["cause"], str(ep["cause"])),
        reward=ep["reward"],
        init_phase_gap_deg=float(phase[0]),
        init_abs_phase_gap_deg=float(abs(phase[0])),
        init_da_km=float(da[0] / 1e3),
        init_alt_sat_km=float((ep["sa"][0] - R_EARTH) / 1e3),
        init_alt_tgt_km=float((ep["ta"][0] - R_EARTH) / 1e3),
        init_e_sat=float(ep["se"][0]),
        init_e_tgt=float(ep["te"][0]),
        closest_km=float(dist[i_cl] / 1e3),
        t_closest_s=float(ep["t"][i_cl]),
        frac_t_closest=float(i_cl / max(1, ep["nrec"] - 1)),
        relv_at_closest=float(relv[i_cl]),
        joint_min=float(joint[i_j]),
        joint_min_dist_km=float(dist[i_j] / 1e3),
        joint_min_relv=float(relv[i_j]),
        t_joint_min_s=float(ep["t"][i_j]),
        ever_100km=bool(in100.any()),
        ever_60km=bool(in60.any()),
        ever_30km=bool(in30.any()),
        n_rec_in_30km=int(in30.sum()),
        n_rec_in_100km=int(in100.sum()),
        min_relv_inside_30km=float(relv[in30].min()) if in30.any() else np.nan,
        min_dist_km_when_slow=float(dist[slow].min() / 1e3) if slow.any() else np.nan,
        frac_rec_slow=float(slow.mean()),
        min_relv=float(relv.min()),
        fuel_start=float(ep["fuel"][0]),
        fuel_end=float(ep["fuel"][-1]),
        fuel_used_frac=float(ep["fuel"][0] - ep["fuel"][-1]),
        fuel_used_pct_of_budget=float((ep["fuel"][0] - ep["fuel"][-1]) / ep["fuel"][0] * 100.0),
        total_dv=float(ep["dv"].sum()),
        n_burns=nb,
        n_burns_last300=n_burns_last300,
        plateau_da_km=plateau_da_km,
        n_phase_alignments=n_align,
        min_align_dist_km=float(min(align_dist)) if align_dist else np.nan,
        t_first_burn_s=t_first,
        t_last_burn_s=t_last,
        frac_burns_last_quarter=float(
            (ep["t"][burn] > 0.75 * ep["t"][-1]).mean()) if nb else np.nan,
        quiet_tail_s=float(ep["t"][-1] - t_last) if nb else np.nan,
        final_dist_km=float(dist[-1] / 1e3),
        final_relv=float(relv[-1]),
        final_phase_gap_deg=float(phase[-1]),
        final_abs_phase_gap_deg=float(abs(phase[-1])),
        final_da_km=float(da[-1] / 1e3),
        final_de=float(ep["se"][-1] - ep["te"][-1]),
        tail_med_abs_da_km=float(np.median(np.abs(da[tail])) / 1e3),
        tail_med_relv=float(np.median(relv[tail])),
        tail_med_dist_km=float(np.median(dist[tail]) / 1e3),
        phase_drift_deg_per_1000rec=float(drift),
        t_end_s=float(ep["t"][-1]),
        alt_sat_end_km=float((ep["sa"][-1] - R_EARTH) / 1e3),
        perigee_end_km=float((ep["sa"][-1] * (1 - ep["se"][-1]) - R_EARTH) / 1e3),
        r_sat_end_km=float((r_sat[-1] - R_EARTH) / 1e3),
    )
    return m


def assign_mode(m):
    """Named failure modes. Rules are mechanistic, evaluated in order.

    M1 DRIFT_ORBIT_LOCK    — never initiates the transfer: <=3 burns, <=30 m/s,
                             semi-major axis essentially unchanged for 2000 steps.
    M3 CLOSURE_TRUNCATED   — terminal closure burst active at the cap (>=10 burns
                             in the last 300 sim steps), still converging.
    M2 WINDOW_BEYOND_HORIZON — canonical drift orbit established, but the phase
                             gap never reaches alignment inside the 2000-step cap.
    M4 FLOOR_CLIP          — non-timeout: drift orbit opened downward from a
                             low-LEO start, perigee driven below the surface.
    """
    if m["cause"] == "collision":
        return "M4_FLOOR_CLIP"
    if m["cause"] != "safety_cap":
        return "success" if m["cause"] == "success" else m["cause"]
    if m["n_burns"] <= 3 and m["total_dv"] <= 30.0:
        return "M1_DRIFT_ORBIT_LOCK"
    if m["n_burns_last300"] >= 10:
        return "M3_CLOSURE_TRUNCATED"
    return "M2_WINDOW_BEYOND_HORIZON"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--out", default=None)
    ap.add_argument("--dump-json", default=None)
    ap.add_argument("--failures-only", action="store_true")
    args = ap.parse_args()

    rows = []
    for d in args.dirs:
        for f in sorted(glob.glob(os.path.join(d, "*.npz"))):
            ep = load_ep(f)
            m = metrics(ep)
            m["dir"] = os.path.basename(d.rstrip("/"))
            m["file"] = os.path.basename(f)
            m["assigned_mode"] = assign_mode(m)
            rows.append(m)

    if args.dump_json:
        with open(args.dump_json, "w") as fh:
            json.dump(rows, fh)
    print(f"loaded {len(rows)} episodes")
    import collections
    print(collections.Counter(r["cause"] for r in rows))
    print(collections.Counter(r["assigned_mode"] for r in rows
                              if r["cause"] != "success"))
    if args.out:
        import csv
        order = ["dir", "file", "assigned_mode", "cause",
                 "init_phase_gap_deg", "init_abs_phase_gap_deg", "init_da_km",
                 "init_alt_sat_km", "init_alt_tgt_km", "init_e_sat", "init_e_tgt",
                 "closest_km", "t_closest_s", "frac_t_closest", "relv_at_closest",
                 "joint_min", "joint_min_dist_km", "joint_min_relv", "t_joint_min_s",
                 "ever_100km", "ever_60km", "ever_30km", "min_relv_inside_30km",
                 "min_dist_km_when_slow", "min_relv",
                 "fuel_used_frac", "fuel_used_pct_of_budget", "total_dv",
                 "n_burns", "n_burns_last300", "t_first_burn_s", "t_last_burn_s",
                 "plateau_da_km", "n_phase_alignments", "min_align_dist_km",
                 "final_dist_km", "final_relv", "final_abs_phase_gap_deg",
                 "final_da_km", "phase_drift_deg_per_1000rec",
                 "n_rec", "t_end_s", "reward"]
        keys = order + [k for k in rows[0].keys() if k not in order]
        sel = [r for r in rows if r["cause"] != "success"] if args.failures_only else rows
        sel.sort(key=lambda r: (r["assigned_mode"], r["dir"], r["file"]))
        with open(args.out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            for r in sel:
                w.writerow(r)
        print("wrote", args.out, len(sel), "rows")


if __name__ == "__main__":
    main()
