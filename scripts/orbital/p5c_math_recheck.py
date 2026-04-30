"""Refined A1: bin episodes by length AND outcome.
Tests whether the 'direction reversal' was a length-mixing artifact."""
import glob, math, os, numpy as np

MU = 3.986004418e14
SUCCESS_TOL_A = 10000.0
W_ORBIT, GAMMA = 0.01, 0.995


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


files = sorted(glob.glob("/Users/pete/space_training/logs/orbital/p5c_s40_at_e020/ep_*.npz"))
records = []
for f in files:
    d = np.load(f)
    er = float(d["episode_reward"][0])
    mask = (d["sat_x"] != 0) | (d["sat_y"] != 0)
    if mask.sum() < 2: continue
    active = np.where(mask)[0]
    po = np.array([phi_orbit(d, int(s)) for s in active])
    cum_r = (W_ORBIT * (GAMMA * po[1:] - po[:-1])).sum()
    drift = W_ORBIT * (GAMMA - 1) * po[1:-1].sum()
    boundary = W_ORBIT * (GAMMA * po[-1] - po[0])
    records.append({
        "success": er > 0, "n": len(active),
        "po_init": float(po[0]), "po_final": float(po[-1]), "po_mean": float(po.mean()),
        "cum_r_orbit": float(cum_r), "drift": float(drift), "boundary": float(boundary),
    })

# Bin by length
length_bins = [(0, 100), (100, 300), (300, 700), (700, 2000)]
print(f"{'len_bin':>10} | {'n_succ':>6} {'cum_r succ':>11} {'po_T succ':>10} | {'n_fail':>6} {'cum_r fail':>11} {'po_T fail':>10}")
for lb in length_bins:
    succ_in = [r for r in records if r["success"] and lb[0] <= r["n"] < lb[1]]
    fail_in = [r for r in records if (not r["success"]) and lb[0] <= r["n"] < lb[1]]
    ns, nf = len(succ_in), len(fail_in)
    cs = np.median([r["cum_r_orbit"] for r in succ_in]) if ns else float("nan")
    cf = np.median([r["cum_r_orbit"] for r in fail_in]) if nf else float("nan")
    poTs = np.median([r["po_final"] for r in succ_in]) if ns else float("nan")
    poTf = np.median([r["po_final"] for r in fail_in]) if nf else float("nan")
    print(f"  [{lb[0]:>3},{lb[1]:>4}) | {ns:>6} {cs:>11.4f} {poTs:>10.2f} | {nf:>6} {cf:>11.4f} {poTf:>10.2f}")

print(f"\nTotal: {len([r for r in records if r['success']])} successes / {len([r for r in records if not r['success']])} failures")

# Failures by terminal Φ
print(f"\n=== Failures binned by terminal Φ_orbit ===")
fail_records = [r for r in records if not r["success"]]
phi_bins = [(0, 2), (2, 5), (5, 10), (10, 20), (20, 50)]
for pb in phi_bins:
    in_bin = [r for r in fail_records if pb[0] <= r["po_final"] < pb[1]]
    if not in_bin: continue
    n = len(in_bin)
    med_n = int(np.median([r["n"] for r in in_bin]))
    med_cum = np.median([r["cum_r_orbit"] for r in in_bin])
    print(f"  Φ_T ∈ [{pb[0]:>2}, {pb[1]:>2}): n={n:>3}, median len={med_n:>4}, median cum_r_orbit={med_cum:>7.4f}")
