"""Phase 5c Block I: post-hoc analytics on existing eval traces.
Combines A5 (Φ-success correlation), A8 (action saturation), A10 (effective
horizon), A11 (failure clustering by phase × e)."""
import argparse, glob, math, os, numpy as np

# Constants from orbital.h
MU = 3.986004418e14
SUCCESS_TOL_A = 10000.0
REL_VEL_TOL = 50.0


def phi_orbit(d, step):
    sat_a = float(d["sat_a"][step]); sat_e = float(d["sat_e"][step])
    sat_omega = float(d["sat_omega"][step])
    tgt_a = float(d["target_a"][step]); tgt_e = float(d["target_e"][step])
    tgt_omega = float(d["target_omega"][step])
    da = abs(sat_a - tgt_a) / SUCCESS_TOL_A
    e_sx = sat_e * math.cos(sat_omega); e_sy = sat_e * math.sin(sat_omega)
    e_tx = tgt_e * math.cos(tgt_omega); e_ty = tgt_e * math.sin(tgt_omega)
    de = math.sqrt((e_sx - e_tx) ** 2 + (e_sy - e_ty) ** 2)
    return da + de


ap = argparse.ArgumentParser()
ap.add_argument("log_dir")
args = ap.parse_args()
files = sorted(glob.glob(os.path.join(args.log_dir, "ep_*.npz")))
print(f"=== {os.path.basename(args.log_dir)}: {len(files)} eps ===\n")

success = []  # list of dicts per success
failure = []
for f in files:
    d = np.load(f)
    er = float(d["episode_reward"][0])
    is_success = er > 0
    n = d["sat_x"].shape[0]
    # ep length (active steps where sat is at non-(0,0))
    active_mask = (d["sat_x"] != 0) | (d["sat_y"] != 0)
    actual_n = int(active_mask.sum())
    # Last step's Φ_orbit
    if actual_n > 0:
        last_active = int(np.where(active_mask)[0].max())
        terminal_phi = phi_orbit(d, last_active)
    else:
        terminal_phi = float("nan")
    # Initial state
    sx0, sy0 = float(d["sat_x"][0]), float(d["sat_y"][0])
    tx0, ty0 = float(d["target_x"][0]), float(d["target_y"][0])
    init_gap = abs(math.degrees(math.atan2(ty0, tx0) - math.atan2(sy0, sx0)))
    if init_gap > 180: init_gap = 360 - init_gap
    init_e = float(d["target_e"][0])
    # Action stats
    actions = d["action"].astype(int)
    delta_v = d["delta_v"]
    burns = (actions >= 1) & (actions <= 8)
    burn_count = int(burns.sum())
    big_burns = ((actions == 2) | (actions == 4)).sum()  # ±25 m/s prograde/retro
    sat_burn = int(big_burns)
    rec = dict(success=is_success, terminal_phi=terminal_phi, ep_len=actual_n,
               init_gap=init_gap, init_e=init_e, burn_count=burn_count,
               saturated_burns=sat_burn, total_dv=float(delta_v.sum()))
    if is_success: success.append(rec)
    else: failure.append(rec)

ns, nf = len(success), len(failure)
print(f"successes: {ns}, failures: {nf}, success rate: {100*ns/(ns+nf):.1f}%\n")

# === A5: success rate by terminal Φ_orbit bin ===
print("--- A5: Success rate by terminal Φ_orbit bin ---")
all_phi = [r["terminal_phi"] for r in success + failure if not math.isnan(r["terminal_phi"])]
phi_bins = [0, 1, 2, 5, 10, 20, 50]
for i in range(len(phi_bins) - 1):
    lo, hi = phi_bins[i], phi_bins[i+1]
    in_bin = [r for r in success + failure if lo <= r["terminal_phi"] < hi]
    if not in_bin: continue
    succ = sum(1 for r in in_bin if r["success"])
    print(f"  Φ ∈ [{lo:>2}, {hi:>2}): {succ}/{len(in_bin)} = {100*succ/len(in_bin):5.1f}%")

# === A8: action saturation in successful episodes ===
print("\n--- A8: Action saturation in successes ---")
if success:
    sat_rates = [r["saturated_burns"] / max(r["burn_count"], 1) for r in success]
    print(f"  Median ±25 m/s burn share among burns: {100*np.median(sat_rates):5.1f}%")
    print(f"  Mean: {100*np.mean(sat_rates):5.1f}%")
    print(f"  Median total Δv per success: {np.median([r['total_dv'] for r in success]):.0f} m/s")

# === A10: effective horizon analysis ===
print("\n--- A10: Successful trajectory length distribution ---")
if success:
    ep_lens = [r["ep_len"] for r in success]
    print(f"  Successful ep length: min={min(ep_lens)} med={int(np.median(ep_lens))} mean={int(np.mean(ep_lens))} max={max(ep_lens)}")
    print(f"  γ=0.995 effective horizon: 200 steps")
    print(f"  Successes longer than effective horizon (>200): {sum(1 for L in ep_lens if L > 200)}/{len(ep_lens)}")
    print(f"  Successes longer than 800 steps: {sum(1 for L in ep_lens if L > 800)}/{len(ep_lens)}")

# === A11: failure clustering by (phase × e) ===
print("\n--- A11: Failure clustering by (phase × e) ---")
phase_bins = [(0, 60), (60, 120), (120, 180)]
e_bins = [(0.0, 0.05), (0.05, 0.10), (0.10, 0.20)]
print(f"  {'phase':>10} | " + "  ".join(f"{eb[0]:.2f}-{eb[1]:.2f}" for eb in e_bins))
for pb in phase_bins:
    row = [f"{pb[0]}-{pb[1]}°"]
    for eb in e_bins:
        in_bin = [r for r in success + failure if pb[0] <= r["init_gap"] < pb[1] and eb[0] <= r["init_e"] < eb[1]]
        if not in_bin: row.append("  --  ")
        else:
            succ = sum(1 for r in in_bin if r["success"])
            row.append(f"{succ}/{len(in_bin)}={100*succ/len(in_bin):4.1f}%")
    print(f"  {row[0]:>10} | " + "  ".join(row[1:]))
