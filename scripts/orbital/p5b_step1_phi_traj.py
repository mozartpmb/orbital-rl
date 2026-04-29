"""Phase 5b Step 1 follow-up #1: Φ_orbit trajectory comparison
on success vs failure episodes."""
import argparse, glob, math, os, numpy as np

MU = 3.986004418e14
SUCCESS_TOL_A = 10000.0


def phi_orbit_series(d):
    """Compute Φ_orbit per step from the trajectory npz (matches orbital.h:530)."""
    sat_a = d["sat_a"]
    sat_e = d["sat_e"]
    sat_omega = d["sat_omega"]
    tgt_a = d["target_a"]
    tgt_e = d["target_e"]
    tgt_omega = d["target_omega"]
    # mask out zero-init rows (from warp sub-step logging quirks)
    mask = sat_a > 0
    da = np.abs(sat_a - tgt_a) / SUCCESS_TOL_A
    e_sx = sat_e * np.cos(sat_omega)
    e_sy = sat_e * np.sin(sat_omega)
    e_tx = tgt_e * np.cos(tgt_omega)
    e_ty = tgt_e * np.sin(tgt_omega)
    de = np.sqrt((e_sx - e_tx) ** 2 + (e_sy - e_ty) ** 2)
    phi = da + de
    return phi, mask


ap = argparse.ArgumentParser()
ap.add_argument("log_dir")
args = ap.parse_args()
files = sorted(glob.glob(os.path.join(args.log_dir, "ep_*.npz")))

succ_phi_summary = []  # (max_phi, end_phi, mean_phi)
fail_phi_summary = []
succ_ep_phi_excursion = []  # max(phi) - phi[0]
fail_ep_phi_excursion = []

for f in files:
    d = np.load(f)
    er = float(d["episode_reward"][0])
    phi, mask = phi_orbit_series(d)
    if mask.sum() < 2:
        continue
    phi_m = phi[mask]
    excursion = float(phi_m.max() - phi_m[0])
    summary = (float(phi_m.max()), float(phi_m[-1]), float(phi_m.mean()))
    if er > 0:
        succ_phi_summary.append(summary)
        succ_ep_phi_excursion.append(excursion)
    else:
        fail_phi_summary.append(summary)
        fail_ep_phi_excursion.append(excursion)


def stats(name, arr):
    a = np.array(arr)
    print(f"  {name:>26}: min={a.min():7.3f}  med={np.median(a):7.3f}  mean={a.mean():7.3f}  max={a.max():7.3f}  p90={np.percentile(a,90):7.3f}")


print(f"=== {os.path.basename(args.log_dir)}: {len(succ_phi_summary)} success / {len(fail_phi_summary)} fail ===\n")
print("--- Success episodes ---")
stats("max Φ_orbit per ep", [s[0] for s in succ_phi_summary])
stats("end Φ_orbit per ep", [s[1] for s in succ_phi_summary])
stats("Φ_orbit excursion (max-init)", succ_ep_phi_excursion)
print("\n--- Failure episodes ---")
stats("max Φ_orbit per ep", [s[0] for s in fail_phi_summary])
stats("end Φ_orbit per ep", [s[1] for s in fail_phi_summary])
stats("Φ_orbit excursion (max-init)", fail_ep_phi_excursion)

# Distribution of excursion (how many fails have very small / very large excursion)
fail_exc = np.array(fail_ep_phi_excursion)
succ_exc = np.array(succ_ep_phi_excursion)
print(f"\n--- Excursion distribution ---")
print(f"  Failures with excursion < 0.1 (chaser stays put): {int(np.sum(fail_exc < 0.1))}/{len(fail_exc)}")
print(f"  Failures with excursion < 0.5 (small attempt):    {int(np.sum(fail_exc < 0.5))}/{len(fail_exc)}")
print(f"  Failures with excursion ≥ 1.0 (genuine attempt):  {int(np.sum(fail_exc >= 1.0))}/{len(fail_exc)}")
print(f"  Successes with excursion < 0.1 (no transfer):     {int(np.sum(succ_exc < 0.1))}/{len(succ_exc)}")
print(f"  Successes with excursion ≥ 1.0 (transfer used):   {int(np.sum(succ_exc >= 1.0))}/{len(succ_exc)}")
