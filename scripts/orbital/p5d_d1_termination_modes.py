"""D1 — Termination-mode classification (corrected threshold).

Env's collision is at r < R_EARTH (6.371e6). Earlier version used the
keepout radius (6.571e6 = R_EARTH + 200km) which mislabeled near-surface
non-collision terminations.
"""
import glob, math, sys, numpy as np

MU = 3.986004418e14
R_EARTH = 6.371e6
MAX_STEPS = 2000


def classify(d):
    er = float(d["episode_reward"][0])
    mask = (d["sat_x"] != 0) | (d["sat_y"] != 0)
    if mask.sum() < 1:
        return "empty", 0, float("nan"), float("nan"), float("nan")
    active = np.where(mask)[0]
    last = int(active[-1])
    n = len(active)
    sx = d["sat_x"][active].astype(np.float64)
    sy = d["sat_y"][active].astype(np.float64)
    r_all = np.sqrt(sx*sx + sy*sy)
    r_min = float(r_all.min())
    svx, svy = float(d["sat_vx"][last]), float(d["sat_vy"][last])
    r_last = float(r_all[-1])
    v2 = svx*svx + svy*svy
    E = 0.5 * v2 - MU / max(r_last, 1.0)
    fuel = float(d["fuel"][last])

    sa = float(d["sat_a"][last]); se = float(d["sat_e"][last])
    rp = sa * (1 - se) if sa > 0 else -1.0
    if er > 0:
        mode = "success"
    elif r_min < R_EARTH or (rp > 0 and rp < R_EARTH):
        # Either logged position was inside Earth, OR current orbit's perigee
        # is inside Earth (collision imminent on next perigee pass).
        mode = "collision"
    elif E >= 0:
        mode = "escape"
    elif fuel <= 1e-6:
        mode = "stranded"
    elif n >= MAX_STEPS - 1:
        mode = "safety_cap"
    else:
        mode = "other"
    return mode, n, r_min, E, fuel


def phi_orbit(d, step):
    SUCCESS_TOL_A = 10000.0
    sat_a = float(d["sat_a"][step]); sat_e = float(d["sat_e"][step])
    sat_om = float(d["sat_omega"][step])
    tgt_a = float(d["target_a"][step]); tgt_e = float(d["target_e"][step])
    tgt_om = float(d["target_omega"][step])
    da = abs(sat_a - tgt_a) / SUCCESS_TOL_A
    e_sx = sat_e * math.cos(sat_om); e_sy = sat_e * math.sin(sat_om)
    e_tx = tgt_e * math.cos(tgt_om); e_ty = tgt_e * math.sin(tgt_om)
    de = math.sqrt((e_sx - e_tx)**2 + (e_sy - e_ty)**2)
    return da + de


def run(path):
    files = sorted(glob.glob(f"{path}/ep_*.npz"))
    recs = []
    for f in files:
        d = np.load(f)
        mode, n, r_min, E, fuel = classify(d)
        mask = (d["sat_x"] != 0) | (d["sat_y"] != 0)
        last = int(np.where(mask)[0][-1]) if mask.sum() else 0
        po_T = phi_orbit(d, last) if mask.sum() else float("nan")
        recs.append({"mode": mode, "n": n, "r_min_km": (r_min - R_EARTH)/1000 if not math.isnan(r_min) else float("nan"),
                     "po_T": po_T, "fuel": fuel})

    print(f"=== {path} (N={len(recs)}) ===")
    print(f"{'mode':>11} {'count':>6} {'pct':>6} {'med_len':>8} {'med_Φ_T':>8} {'med_fuel':>10} {'med_r_min_km':>14}")
    for m in ["success", "collision", "escape", "stranded", "safety_cap", "other"]:
        sub = [r for r in recs if r["mode"] == m]
        if not sub: continue
        n = len(sub)
        ml = int(np.median([r["n"] for r in sub]))
        mp = np.median([r["po_T"] for r in sub])
        mf = np.median([r["fuel"] for r in sub])
        mr = np.median([r["r_min_km"] for r in sub])
        print(f"{m:>11} {n:>6} {100*n/len(recs):>5.1f}% {ml:>8} {mp:>8.2f} {mf:>10.4f} {mr:>14.0f}")


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "/Users/pete/space_training/logs/orbital/p5c_s40_at_e020")
