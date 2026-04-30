"""Probe 1: detailed math check on the shaping accounting.
For one success ep + one failure ep, dump per-step Φ values, compute
γ-discounted shaping deltas, and reconcile with the A1 cumulative."""
import glob, math, os, numpy as np

MU = 3.986004418e14
SUCCESS_TOL_A = 10000.0
REL_VEL_TOL = 50.0
W_ORBIT, W_PHASE, W_VEL = 0.01, 0.01, 0.01
EPS_ORBIT, EPS_PHASE = 2.0, 0.3
TAU_ORBIT, TAU_PHASE = 0.2, 0.03
GAMMA = 0.995


def sigmoid(x): return 1.0 / (1.0 + np.exp(-x))


def phi_components(d, step):
    sat_a = float(d["sat_a"][step]); sat_e = float(d["sat_e"][step])
    sat_omega = float(d["sat_omega"][step]); sat_theta = float(d["sat_theta"][step])
    tgt_a = float(d["target_a"][step]); tgt_e = float(d["target_e"][step])
    tgt_omega = float(d["target_omega"][step])
    da = abs(sat_a - tgt_a) / SUCCESS_TOL_A
    e_sx = sat_e * math.cos(sat_omega); e_sy = sat_e * math.sin(sat_omega)
    e_tx = tgt_e * math.cos(tgt_omega); e_ty = tgt_e * math.sin(tgt_omega)
    de = math.sqrt((e_sx - e_tx) ** 2 + (e_sy - e_ty) ** 2)
    phi_orbit = da + de
    tx, ty = float(d["target_x"][step]), float(d["target_y"][step])
    tgt_inertial = math.atan2(ty, tx)
    tgt_theta_recomp = (tgt_inertial - tgt_omega) % (2 * math.pi)
    dtheta = sat_theta - tgt_theta_recomp
    phi_phase = 1.0 - math.cos(dtheta)
    sx, sy = float(d["sat_x"][step]), float(d["sat_y"][step])
    svx, svy = float(d["sat_vx"][step]), float(d["sat_vy"][step])
    tvx, tvy = float(d["target_vx"][step]), float(d["target_vy"][step])
    tt = tgt_theta_recomp + tgt_omega
    ct, st = math.cos(tt), math.sin(tt)
    dxi, dyi = sx - tx, sy - ty
    dvxi, dvyi = svx - tvx, svy - tvy
    dx_l = ct*dxi + st*dyi; dy_l = -st*dxi + ct*dyi
    dvx_l = ct*dvxi + st*dvyi; dvy_l = -st*dvxi + ct*dvyi
    n_tgt = math.sqrt(MU / (tgt_a**3))
    dvx_l += n_tgt * dy_l; dvy_l -= n_tgt * dx_l
    rel_vel = math.sqrt(dvx_l**2 + dvy_l**2)
    phi_vel = rel_vel / REL_VEL_TOL
    sigma_2 = sigmoid((EPS_ORBIT - phi_orbit) / TAU_ORBIT)
    sigma_3 = sigma_2 * sigmoid((EPS_PHASE - phi_phase) / TAU_PHASE)
    return phi_orbit, phi_phase, phi_vel, sigma_2, sigma_3


def detailed_episode(d, label):
    print(f"\n=== {label} ===")
    n = d["sat_x"].shape[0]
    er = float(d["episode_reward"][0])
    mask = (d["sat_x"] != 0) | (d["sat_y"] != 0)
    active = np.where(mask)[0]
    print(f"  total steps: {n}, active (non-warp-substep): {len(active)}, terminal reward: {er:.2f}")
    if len(active) < 2: return
    po_arr, ph_arr, pv_arr, s2_arr, s3_arr = [], [], [], [], []
    for s in active:
        po, ph, pv, s2, s3 = phi_components(d, int(s))
        po_arr.append(po); ph_arr.append(ph); pv_arr.append(pv); s2_arr.append(s2); s3_arr.append(s3)
    po, ph, pv = np.array(po_arr), np.array(ph_arr), np.array(pv_arr)
    s2, s3 = np.array(s2_arr), np.array(s3_arr)
    # Per-step shaping per Phase 4 form
    # r = β · (W_O·(γ·Φ_O' − Φ_O) + W_P·(γ·σ₂'·Φ_P' − σ₂·Φ_P) + W_V·(γ·σ₃'·Φ_V' − σ₃·Φ_V))
    r_o_per = W_ORBIT * (GAMMA * po[1:] - po[:-1])
    r_p_per = W_PHASE * (GAMMA * s2[1:]*ph[1:] - s2[:-1]*ph[:-1])
    r_v_per = W_VEL * (GAMMA * s3[1:]*pv[1:] - s3[:-1]*pv[:-1])
    print(f"\n  Φ_orbit  init={po[0]:.2f} final={po[-1]:.2f} mean={po.mean():.2f} max={po.max():.2f}")
    print(f"  Φ_phase  init={ph[0]:.3f} final={ph[-1]:.3f} mean={ph.mean():.3f}")
    print(f"  Φ_vel    init={pv[0]:.2f} final={pv[-1]:.2f} mean={pv.mean():.2f}")
    print(f"  σ₂       active rate (>0.5): {100*np.mean(s2 > 0.5):.1f}%")
    print(f"  σ₃       active rate (>0.5): {100*np.mean(s3 > 0.5):.1f}%")
    print(f"  cum r_orbit:  {r_o_per.sum():.4f}")
    print(f"  cum r_phase:  {r_p_per.sum():.4f}  (σ₂-gated)")
    print(f"  cum r_vel:    {r_v_per.sum():.4f}  (σ₃-gated)")
    print(f"  cum r_total:  {(r_o_per.sum() + r_p_per.sum() + r_v_per.sum()):.4f}")
    # Telescoping decomposition of cum r_orbit
    # Σ (γ·Φ_t+1 − Φ_t) = γ·(Φ_1+...+Φ_T) − (Φ_0+...+Φ_{T-1})
    # = (γ-1)·(Φ_1+...+Φ_{T-1}) + γ·Φ_T − Φ_0
    drift = (GAMMA - 1) * po[1:-1].sum()  # discount drag on internal states
    boundary = GAMMA * po[-1] - po[0]
    print(f"\n  Telescoping cum r_orbit = W·[ (γ-1)·Σ_internal + γ·Φ_T − Φ_0 ]")
    print(f"    drag term  W·(γ-1)·Σ_internal_Φ:  {W_ORBIT * drift:.4f}")
    print(f"    boundary term W·(γ·Φ_T − Φ_0):   {W_ORBIT * boundary:.4f}")
    print(f"    sum:                              {W_ORBIT * (drift + boundary):.4f}")
    print(f"    actual:                           {r_o_per.sum():.4f}")


# Find one success and one failure from the eval logs
files = sorted(glob.glob("/Users/pete/space_training/logs/orbital/p5c_s40_at_e020/ep_*.npz"))
success_eps = []; failure_eps = []
for f in files:
    d = np.load(f)
    if float(d["episode_reward"][0]) > 0: success_eps.append(f)
    else: failure_eps.append(f)
print(f"Found {len(success_eps)} successes, {len(failure_eps)} failures")

# Detailed analysis on representative ones
detailed_episode(np.load(success_eps[0]), f"SUCCESS ep ({os.path.basename(success_eps[0])})")
detailed_episode(np.load(success_eps[len(success_eps)//2]), f"SUCCESS ep (median)")
detailed_episode(np.load(failure_eps[0]), f"FAILURE ep ({os.path.basename(failure_eps[0])})")
detailed_episode(np.load(failure_eps[len(failure_eps)//2]), f"FAILURE ep (median)")
