"""Phase 5c A1: per-step shaping decomposition.
For each trajectory, compute the per-step contribution of each Φ component
to the shaping reward. Identify whether γ-discount bias on growing |Φ|
explains the cumulative-positive shaping (Wiewiora 2003 effect)."""
import argparse, glob, math, os, numpy as np

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


def phi_components(d, step):
    """Returns (Φ_orbit, σ₂·Φ_phase, σ₃·Φ_vel) — gated form per orbital.h:514-562."""
    sat_a = float(d["sat_a"][step])
    sat_e = float(d["sat_e"][step])
    sat_omega = float(d["sat_omega"][step])
    sat_theta = float(d["sat_theta"][step])
    tgt_a = float(d["target_a"][step])
    tgt_e = float(d["target_e"][step])
    tgt_omega = float(d["target_omega"][step])
    # Φ_orbit
    da = abs(sat_a - tgt_a) / SUCCESS_TOL_A
    e_sx = sat_e * np.cos(sat_omega); e_sy = sat_e * np.sin(sat_omega)
    e_tx = tgt_e * np.cos(tgt_omega); e_ty = tgt_e * np.sin(tgt_omega)
    de = np.sqrt((e_sx - e_tx) ** 2 + (e_sy - e_ty) ** 2)
    phi_orbit = da + de
    # Target true anomaly (recomputed)
    tx, ty = float(d["target_x"][step]), float(d["target_y"][step])
    tgt_inertial = np.arctan2(ty, tx)
    tgt_theta = (tgt_inertial - tgt_omega) % (2 * np.pi)
    # Φ_phase
    dtheta = sat_theta - tgt_theta
    phi_phase = 1.0 - np.cos(dtheta)
    # Φ_vel via LVLH-relative
    sx, sy = float(d["sat_x"][step]), float(d["sat_y"][step])
    svx, svy = float(d["sat_vx"][step]), float(d["sat_vy"][step])
    tvx, tvy = float(d["target_vx"][step]), float(d["target_vy"][step])
    tt = tgt_theta + tgt_omega
    ct, st = np.cos(tt), np.sin(tt)
    dxi = sx - tx; dyi = sy - ty
    dvxi = svx - tvx; dvyi = svy - tvy
    dx_l = ct*dxi + st*dyi; dy_l = -st*dxi + ct*dyi
    dvx_l = ct*dvxi + st*dvyi; dvy_l = -st*dvxi + ct*dvyi
    n_tgt = np.sqrt(MU / (tgt_a**3))
    dvx_l += n_tgt * dy_l; dvy_l -= n_tgt * dx_l
    rel_vel = np.sqrt(dvx_l**2 + dvy_l**2)
    phi_vel = rel_vel / REL_VEL_TOL
    # Gates
    sigma_2 = sigmoid((EPS_ORBIT - phi_orbit) / TAU_ORBIT)
    sigma_3 = sigma_2 * sigmoid((EPS_PHASE - phi_phase) / TAU_PHASE)
    return phi_orbit, sigma_2 * phi_phase, sigma_3 * phi_vel


def analyze_episode(d):
    """Return per-component cumulative shaping breakdown for one episode."""
    n = d["sat_x"].shape[0]
    mask = (d["sat_x"] != 0) | (d["sat_y"] != 0)
    if mask.sum() < 2: return None
    active = np.where(mask)[0]
    # Per step Φ components
    po, ph, pv = [], [], []
    for s in active:
        a, b, c = phi_components(d, int(s))
        po.append(a); ph.append(b); pv.append(c)
    po, ph, pv = np.array(po), np.array(ph), np.array(pv)
    # Per-step shaping contributions (Phase 4 form: r = β · Σ_k W_k · (γ^τ · Φ_k(s') − Φ_k(s)))
    # Approximating τ=1 for non-warp; warps make this slightly off but fine for first-pass.
    r_orbit = BETA_SHAPE * W_ORBIT * (GAMMA * po[1:] - po[:-1])
    r_phase = BETA_SHAPE * W_PHASE * (GAMMA * ph[1:] - ph[:-1])
    r_vel = BETA_SHAPE * W_VEL * (GAMMA * pv[1:] - pv[:-1])
    # γ=1 counterfactual (Ng-Harada-Russell)
    r_orbit_g1 = BETA_SHAPE * W_ORBIT * (po[1:] - po[:-1])
    r_phase_g1 = BETA_SHAPE * W_PHASE * (ph[1:] - ph[:-1])
    r_vel_g1 = BETA_SHAPE * W_VEL * (pv[1:] - pv[:-1])
    return dict(
        po=po, ph=ph, pv=pv,
        cum_r_orbit=float(r_orbit.sum()), cum_r_phase=float(r_phase.sum()), cum_r_vel=float(r_vel.sum()),
        cum_r_total=float(r_orbit.sum() + r_phase.sum() + r_vel.sum()),
        cum_r_orbit_g1=float(r_orbit_g1.sum()), cum_r_phase_g1=float(r_phase_g1.sum()), cum_r_vel_g1=float(r_vel_g1.sum()),
        cum_r_total_g1=float(r_orbit_g1.sum() + r_phase_g1.sum() + r_vel_g1.sum()),
        # γ-bias estimate: cumulative shaping under γ=0.995 minus γ=1
        gamma_bias=float((r_orbit.sum() + r_phase.sum() + r_vel.sum()) - (r_orbit_g1.sum() + r_phase_g1.sum() + r_vel_g1.sum())),
        po_init=float(po[0]), po_final=float(po[-1]), po_max=float(po.max()),
    )


ap = argparse.ArgumentParser()
ap.add_argument("log_dir")
args = ap.parse_args()
files = sorted(glob.glob(os.path.join(args.log_dir, "ep_*.npz")))
print(f"=== {os.path.basename(args.log_dir)}: {len(files)} eps ===\n")

succ_results, fail_results = [], []
for f in files:
    d = np.load(f)
    er = float(d["episode_reward"][0])
    res = analyze_episode(d)
    if res is None: continue
    if er > 0: succ_results.append(res)
    else: fail_results.append(res)

def stats(label, group, key, fmt="{:.4f}"):
    arr = np.array([r[key] for r in group])
    print(f"  {label:>30}: mean={fmt.format(arr.mean())} median={fmt.format(np.median(arr))} std={fmt.format(arr.std())}")

print(f"--- A1: Per-component cumulative shaping (γ=0.995 form) ---")
print(f"\nFailures (n={len(fail_results)}):")
stats("cum r_orbit (γ=0.995)", fail_results, "cum_r_orbit")
stats("cum r_phase (γ=0.995)", fail_results, "cum_r_phase")
stats("cum r_vel (γ=0.995)", fail_results, "cum_r_vel")
stats("cum r_total (γ=0.995)", fail_results, "cum_r_total")
stats("cum r_orbit (γ=1.0)", fail_results, "cum_r_orbit_g1")
stats("cum r_total (γ=1.0)", fail_results, "cum_r_total_g1")
stats("γ-bias (γ=0.995 − γ=1.0)", fail_results, "gamma_bias")
stats("Φ_orbit init", fail_results, "po_init", "{:.2f}")
stats("Φ_orbit final", fail_results, "po_final", "{:.2f}")
stats("Φ_orbit max", fail_results, "po_max", "{:.2f}")

if succ_results:
    print(f"\nSuccesses (n={len(succ_results)}):")
    stats("cum r_orbit (γ=0.995)", succ_results, "cum_r_orbit")
    stats("cum r_total (γ=0.995)", succ_results, "cum_r_total")
    stats("cum r_orbit (γ=1.0)", succ_results, "cum_r_orbit_g1")
    stats("cum r_total (γ=1.0)", succ_results, "cum_r_total_g1")
    stats("γ-bias", succ_results, "gamma_bias")
    stats("Φ_orbit init", succ_results, "po_init", "{:.2f}")
    stats("Φ_orbit final", succ_results, "po_final", "{:.2f}")

print("\n--- A2 / smoking-gun test: cumulative shaping success vs failure ---")
print("If failures > successes, shaping signal is REVERSED.")
if succ_results:
    sf_total = np.median([r["cum_r_total"] for r in succ_results])
    ff_total = np.median([r["cum_r_total"] for r in fail_results])
    print(f"  Success median cum r_total: {sf_total:.4f}")
    print(f"  Failure median cum r_total: {ff_total:.4f}")
    print(f"  Direction: {'SUCCESS > FAILURE (shaping correct)' if sf_total > ff_total else 'FAILURE > SUCCESS (shaping reversed!)'}")
    sf_g1 = np.median([r["cum_r_total_g1"] for r in succ_results])
    ff_g1 = np.median([r["cum_r_total_g1"] for r in fail_results])
    print(f"\n  Under γ=1 (NHR formulation):")
    print(f"    Success median: {sf_g1:.4f}")
    print(f"    Failure median: {ff_g1:.4f}")
    print(f"    Direction: {'CORRECT' if sf_g1 > ff_g1 else 'REVERSED'}")
