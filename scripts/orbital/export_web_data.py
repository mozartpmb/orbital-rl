"""Convert .npz trajectory logs into web frontend JSON format.

Per phase5-closure-spec §3.B2. Reads .npz files (PufferLib eval output),
computes per-episode metadata, derives target_theta from cartesian, writes
one JSON per episode.

Usage:
    python3 scripts/orbital/export_web_data.py \\
        --src-dir /tmp/p5e_s42_e020_200 \\
        --out-dir web_data/runs/phase_5e_seed42_e020 \\
        --phase phase_5e_stage_4_0 \\
        --checkpoint pufferlib/experiments/puffer_orbital_177765503091/model_puffer_orbital_000325.pt \\
        --e-max-target 0.20 --e-max-sat 0.20 \\
        --same-orbit-init 0 --valid-init-only 1 \\
        [--max-eps 50] [--downsample 1]
"""
import argparse, glob, json, math, os, sys
import numpy as np

MU = 3.986004418e14
R_EARTH = 6.371e6
EARTH_KEEPOUT = R_EARTH + 200e3
MAX_STEPS = 2000


def derive_target_theta(target_x, target_y, target_omega):
    """Recover target true anomaly from inertial position + ω.
    theta_inertial = atan2(y, x); theta_perifocal = theta_inertial - omega."""
    th_inert = np.arctan2(target_y, target_x)
    th_perif = (th_inert - target_omega) % (2 * np.pi)
    return th_perif


def hohmann_dv(r1, r2):
    """Closed-form Hohmann between circular endpoints."""
    a_t = 0.5 * (r1 + r2)
    v1c = math.sqrt(MU / r1); v2c = math.sqrt(MU / r2)
    v1t = math.sqrt(MU * (2.0/r1 - 1.0/a_t))
    v2t = math.sqrt(MU * (2.0/r2 - 1.0/a_t))
    return abs(v1t - v1c) + abs(v2c - v2t)


def classify_termination(d, last):
    """Infer termination mode from final state + orbit elements."""
    er = float(d["episode_reward"][0])
    if er > 0: return "success"
    sx, sy = float(d["sat_x"][last]), float(d["sat_y"][last])
    svx, svy = float(d["sat_vx"][last]), float(d["sat_vy"][last])
    r_last = math.sqrt(sx*sx + sy*sy)
    v2 = svx*svx + svy*svy
    E = 0.5 * v2 - MU / max(r_last, 1.0)
    fuel = float(d["fuel"][last])
    sa = float(d["sat_a"][last]); se = float(d["sat_e"][last])
    rp = sa * (1 - se) if sa > 0 else -1.0
    # r_min over active steps
    mask = (d["sat_x"] != 0) | (d["sat_y"] != 0)
    active = np.where(mask)[0]
    r_all = np.sqrt(d["sat_x"][active].astype(np.float64)**2 +
                     d["sat_y"][active].astype(np.float64)**2)
    r_min = float(r_all.min())
    n = len(active)

    if r_min < R_EARTH or (rp > 0 and rp < R_EARTH):
        return "collision"
    if E >= 0: return "escape"
    if fuel <= 1e-6: return "stranded"
    if n >= MAX_STEPS - 1: return "safety_cap"
    return "other"


def export_episode(npz_path, env_config, phase_label, ckpt_label, downsample=1):
    d = np.load(npz_path)
    mask = (d["sat_x"] != 0) | (d["sat_y"] != 0)
    if mask.sum() < 2: return None
    active = np.where(mask)[0]
    last = int(active[-1])
    n_active = len(active)

    # Per-episode metadata
    er = float(d["episode_reward"][0])
    success = bool(er > 0)
    term_mode = classify_termination(d, last)
    total_dv = float(np.abs(d["delta_v"][active]).sum())
    sat_a0 = float(d["sat_a"][0]); tgt_a0 = float(d["target_a"][0])
    hohm_est = hohmann_dv(sat_a0, tgt_a0)
    fuel_eff = (hohm_est / total_dv) if total_dv > 1e-6 else None
    sat_th0 = float(d["sat_theta"][0])
    tgt_x0, tgt_y0 = float(d["target_x"][0]), float(d["target_y"][0])
    tgt_om0 = float(d["target_omega"][0])
    tgt_th0 = float((np.arctan2(tgt_y0, tgt_x0) - tgt_om0) % (2*np.pi))
    phase_gap = math.atan2(math.sin(sat_th0 - tgt_th0), math.cos(sat_th0 - tgt_th0))
    init_e_sat = float(d["sat_e"][0]); init_e_tgt = float(d["target_e"][0])
    init_om_sat = float(d["sat_omega"][0])

    # Phase 5 env-fix F2: realized-init outcome metadata. Distinct from env_config:
    # `env_config` records intent (the kwargs requested at eval time); `realized_init`
    # records what c_reset actually produced. The two diverge when the rejection
    # sampler exhausts its attempt cap and accepts a doomed init.
    realized_sat_perigee_m = sat_a0 * (1.0 - init_e_sat)
    realized_target_perigee_m = tgt_a0 * (1.0 - init_e_tgt)
    if "last_init_attempts" in d.files:
        last_init_attempts = int(d["last_init_attempts"][0])
    else:
        last_init_attempts = -1   # Pre-F2 .npz files; outcome not recorded
    if "last_init_gave_up" in d.files:
        last_init_gave_up = bool(int(d["last_init_gave_up"][0]))
    else:
        last_init_gave_up = None  # Pre-F2 .npz files; outcome not recorded

    # Per-step
    target_theta = derive_target_theta(d["target_x"][active], d["target_y"][active],
                                         d["target_omega"][active])
    steps = []
    step_idx = active[::downsample]
    for i in step_idx:
        ai = int(np.where(active == i)[0][0])  # index within active
        # Min-debris distance: use min_conj_dist if non-zero, else None (no debris)
        mcd = float(d["min_conj_dist"][i])
        if not math.isfinite(mcd) or mcd == 0.0: mcd = None
        steps.append({
            "t": float(d["sim_time"][i]),
            "x": float(d["sat_x"][i]),
            "y": float(d["sat_y"][i]),
            "vx": float(d["sat_vx"][i]),
            "vy": float(d["sat_vy"][i]),
            "a": float(d["sat_a"][i]),
            "e": float(d["sat_e"][i]),
            "theta": float(d["sat_theta"][i]),
            "omega": float(d["sat_omega"][i]),
            "fuel": float(d["fuel"][i]),
            "action": int(d["action"][i]),
            "dv": float(d["delta_v"][i]),
            "reward": float(d["reward"][i]),
            "target_x": float(d["target_x"][i]),
            "target_y": float(d["target_y"][i]),
            "target_theta": float(target_theta[ai]),
            "min_debris_dist": mcd,
        })

    # Bodies (only Earth + any non-zero debris recorded at first step)
    bodies = [{"type": "earth", "x": 0.0, "y": 0.0,
               "hard_r": R_EARTH, "keepout_r": EARTH_KEEPOUT}]
    n_bodies = int(d["num_bodies"][0]) if "num_bodies" in d.files else 1
    for bi in range(1, min(n_bodies, 16)):
        # Debris are static-orbit logged per step; web frontend may animate them.
        # For simplicity export initial positions only.
        bx = float(d[f"body_x_{bi}"][0]); by = float(d[f"body_y_{bi}"][0])
        if bx == 0.0 and by == 0.0: continue
        bodies.append({"type": "debris", "x": bx, "y": by,
                       "hard_r": float(d[f"body_hard_r_{bi}"][0]),
                       "keepout_r": float(d[f"body_keepout_r_{bi}"][0])})

    return {
        "episode_id": int(d["episode_id"][0]),
        "metadata": {
            "checkpoint": ckpt_label,
            "phase": phase_label,
            "success": success,
            "termination_mode": term_mode,
            "total_dv_ms": total_dv,
            "hohmann_dv_estimate_ms": hohm_est,
            "fuel_efficiency": fuel_eff,
            "num_steps": int(n_active),
            "init_phase_gap_rad": phase_gap,
            "init_phase_gap_deg": math.degrees(phase_gap),
            "init_e_target": init_e_tgt,
            "init_e_sat": init_e_sat,
            "env_config": env_config,
            "realized_init": {
                "last_init_attempts": last_init_attempts,
                "last_init_gave_up": last_init_gave_up,
                "realized_sat_perigee_m": realized_sat_perigee_m,
                "realized_target_perigee_m": realized_target_perigee_m,
            },
        },
        "initial": {
            "sat_a_m": sat_a0,
            "sat_e": init_e_sat,
            "sat_omega_rad": init_om_sat,
            "sat_theta_rad": sat_th0,
            "target_a_m": tgt_a0,
            "target_e": init_e_tgt,
            "target_omega_rad": tgt_om0,
            "target_theta_rad": tgt_th0,
        },
        "target": {
            "a_m": tgt_a0,
            "e": init_e_tgt,
            "omega_rad": tgt_om0,
        },
        "bodies": bodies,
        "steps": steps,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--phase", required=True, help="Logical phase label, e.g. phase_5e_stage_4_0")
    p.add_argument("--checkpoint", required=True, help="Ckpt path label")
    p.add_argument("--e-max-target", type=float, required=True)
    p.add_argument("--e-max-sat", type=float, required=True)
    p.add_argument("--same-orbit-init", type=int, default=0)
    p.add_argument("--valid-init-only", type=int, default=0)
    p.add_argument("--max-eps", type=int, default=None,
                    help="Limit number of episodes exported")
    p.add_argument("--downsample", type=int, default=1,
                    help="Keep every Nth step (1=all)")
    args = p.parse_args()

    env_config = {
        "e_max_target": args.e_max_target,
        "e_max_sat": args.e_max_sat,
        "same_orbit_init": args.same_orbit_init,
        "valid_init_only": args.valid_init_only,
    }

    os.makedirs(args.out_dir, exist_ok=True)
    files = sorted(glob.glob(f"{args.src_dir}/ep_*.npz"))
    if args.max_eps: files = files[:args.max_eps]

    n_ok = n_skip = 0
    for f in files:
        ep_data = export_episode(f, env_config, args.phase, args.checkpoint, args.downsample)
        if ep_data is None:
            n_skip += 1; continue
        out_path = os.path.join(args.out_dir, os.path.basename(f).replace(".npz", ".json"))
        with open(out_path, "w") as fh:
            json.dump(ep_data, fh, separators=(",", ":"))
        n_ok += 1

    print(f"Exported {n_ok}/{len(files)} episodes to {args.out_dir} (skipped {n_skip})")


if __name__ == "__main__":
    main()
