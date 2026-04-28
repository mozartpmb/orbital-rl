"""Audit 2.1 + 3: port compute_phi to Python, recompute Φ components per step
from trajectory npz, analyze gate activation rates and Goodhart ratio.

Mirrors orbital.h:514-562 compute_phi(env)."""
import argparse, glob, os, numpy as np

# Constants from orbital.h
MU = 3.986004418e14
SUCCESS_TOL_A = 10000.0
REL_VEL_TOL = 50.0
W_ORBIT = 0.01
W_PHASE = 0.01
W_VEL = 0.01
EPS_ORBIT = 2.0
EPS_PHASE = 0.3
TAU_ORBIT = 0.2
TAU_PHASE = 0.03
BETA_SHAPE = 1.0
GAMMA = 0.995


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def compute_phi_components(d, step):
    """Recompute (Φ_orbit, σ₂, Φ_phase, σ₃, Φ_vel, Φ_total) at given step."""
    sat_a = float(d["sat_a"][step])
    sat_e = float(d["sat_e"][step])
    sat_omega = float(d["sat_omega"][step])
    sat_theta = float(d["sat_theta"][step])
    tgt_a = float(d["target_a"][step])
    tgt_e = float(d["target_e"][step])
    tgt_omega = float(d["target_omega"][step])
    # Target true anomaly: not directly logged. Recompute from cartesian via atan2 - omega.
    tx, ty = float(d["target_x"][step]), float(d["target_y"][step])
    inertial_angle_t = np.arctan2(ty, tx)
    tgt_theta = (inertial_angle_t - tgt_omega) % (2 * np.pi)

    # Φ_orbit = |Δa|/SUCCESS_TOL_A + ||Δē||
    da = abs(sat_a - tgt_a)
    e_sx = sat_e * np.cos(sat_omega)
    e_sy = sat_e * np.sin(sat_omega)
    e_tx = tgt_e * np.cos(tgt_omega)
    e_ty = tgt_e * np.sin(tgt_omega)
    de = np.sqrt((e_sx - e_tx) ** 2 + (e_sy - e_ty) ** 2)
    phi_orbit = da / SUCCESS_TOL_A + de

    # Φ_phase = 1 - cos(Δθ)
    dtheta = sat_theta - tgt_theta
    phi_phase = 1.0 - np.cos(dtheta)

    # Φ_vel = ||v_rel_lvlh|| / REL_VEL_TOL
    # Compute LVLH-frame relative velocity from cartesian. Same formula as orbital.h.
    sx, sy = float(d["sat_x"][step]), float(d["sat_y"][step])
    svx, svy = float(d["sat_vx"][step]), float(d["sat_vy"][step])
    tvx, tvy = float(d["target_vx"][step]), float(d["target_vy"][step])
    theta_t_inertial = tgt_theta + tgt_omega
    ct, st = np.cos(theta_t_inertial), np.sin(theta_t_inertial)
    dxi, dyi = sx - tx, sy - ty
    dvxi, dvyi = svx - tvx, svy - tvy
    dx_l = ct * dxi + st * dyi
    dy_l = -st * dxi + ct * dyi
    dvx_l = ct * dvxi + st * dvyi
    dvy_l = -st * dvxi + ct * dvyi
    n_tgt = np.sqrt(MU / (tgt_a ** 3))
    dvx_l += n_tgt * dy_l
    dvy_l -= n_tgt * dx_l
    rel_vel = np.sqrt(dvx_l ** 2 + dvy_l ** 2)
    phi_vel = rel_vel / REL_VEL_TOL

    # Gates
    sigma_1 = 1.0
    sigma_2 = sigmoid((EPS_ORBIT - phi_orbit) / TAU_ORBIT)
    sigma_3 = sigma_2 * sigmoid((EPS_PHASE - phi_phase) / TAU_PHASE)

    phi_total = -(W_ORBIT * phi_orbit * sigma_1 + W_PHASE * phi_phase * sigma_2 + W_VEL * phi_vel * sigma_3)
    return dict(phi_orbit=phi_orbit, sigma_2=sigma_2, phi_phase=phi_phase, sigma_3=sigma_3,
                phi_vel=phi_vel, phi_total=phi_total)


def analyze_dir(log_dir, label):
    files = sorted(glob.glob(os.path.join(log_dir, "ep_*.npz")))
    print(f"\n=== {label}: {len(files)} eps in {log_dir} ===")

    s2_acts = []
    s3_acts = []
    phi_orbit_means = []
    phi_total_means = []
    shaping_sums = []
    terminal_rewards = []
    success = 0

    for f in files:
        d = np.load(f)
        n = d["sat_x"].shape[0]
        if n < 2:
            continue
        # Recompute Φ at each step
        phis = []
        s2s = []
        s3s = []
        po_arr = []
        for step in range(n):
            c = compute_phi_components(d, step)
            phis.append(c["phi_total"])
            s2s.append(c["sigma_2"])
            s3s.append(c["sigma_3"])
            po_arr.append(c["phi_orbit"])
        phis = np.array(phis)
        # Per-step shaping reward = β · (γ^τ · Φ_curr − Φ_prev), τ=1 for non-warp steps
        # Approximation: τ=1 throughout (warp steps are minority). Refine later if needed.
        shaping = BETA_SHAPE * (GAMMA * phis[1:] - phis[:-1])
        # NHR clamp at terminal: rewards += β·(0 − phi_prev)
        # Treat last step's reward addition specially
        # Sum of per-step shaping rewards in the npz `reward` array (terminal step's
        # reward is NOT in this array — npz logging rejects step >= MAX_STEPS).
        shaping_sum = float(np.sum(d["reward"]))
        # Terminal step reward is captured in episode_reward field (eval_checkpoint.py
        # stores rewards[0] at terminal step → episode_reward).
        terminal_r = float(d["episode_reward"][0])

        s2_acts.append(np.mean(np.array(s2s) > 0.5))
        s3_acts.append(np.mean(np.array(s3s) > 0.5))
        phi_orbit_means.append(np.mean(po_arr))
        phi_total_means.append(np.mean(phis))
        shaping_sums.append(shaping_sum)
        terminal_rewards.append(terminal_r)
        if float(d["episode_reward"][0]) > 0:
            success += 1

    s2_acts = np.array(s2_acts)
    s3_acts = np.array(s3_acts)
    phi_total_means = np.array(phi_total_means)
    shaping_sums = np.array(shaping_sums)
    terminal_rewards = np.array(terminal_rewards)

    print(f"  Success rate: {success}/{len(files)}  ({100*success/len(files):.1f}%)")
    print(f"  σ₂ activation rate (% steps σ₂>0.5):  mean={np.mean(s2_acts):.3f}  median={np.median(s2_acts):.3f}")
    print(f"  σ₃ activation rate (% steps σ₃>0.5):  mean={np.mean(s3_acts):.3f}  median={np.median(s3_acts):.3f}")
    print(f"  Mean Φ_orbit per ep:  mean={np.mean(phi_orbit_means):.3f}  median={np.median(phi_orbit_means):.3f}")
    print(f"  Mean Φ_total  per ep:  mean={np.mean(phi_total_means):.4f}  median={np.median(phi_total_means):.4f}")
    print(f"  Cumulative shaping reward per ep: mean={np.mean(shaping_sums):.3f}  std={np.std(shaping_sums):.3f}")
    print(f"  Terminal-step reward            : mean={np.mean(terminal_rewards):.3f}  median={np.median(terminal_rewards):.3f}")
    # Audit 3: Goodhart ratio
    abs_shaping = np.abs(shaping_sums)
    abs_terminal = np.abs(terminal_rewards)
    ratios = abs_shaping / np.maximum(abs_terminal, 1e-3)
    print(f"  Goodhart ratio |Σshaping|/|terminal|: mean={np.mean(ratios):.3f}  median={np.median(ratios):.3f}  >0.1: {100*np.mean(ratios>0.1):.1f}% of eps")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+", help="log dirs in order, label inferred from path")
    args = ap.parse_args()
    for d in args.dirs:
        label = os.path.basename(d.rstrip("/"))
        analyze_dir(d, label)
