"""Generalist (T11/T15e) result figures from the tabulated post-fix results.
Sources: web_data/results/t15e_dagger/fin_*.json ; T11_GENERALIST.md tables."""
import os, sys, json, glob, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
out = sys.argv[1]; os.makedirs(out, exist_ok=True)
REPO = "/Users/pete/space_training"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 150, "savefig.bbox": "tight"})
C_CH, C_TG, C_BURN = "#d7301f", "#2c7fb8", "#f0a202"

# ---- 1. seven-skill bar: T15e final (single policy), 200 eps/cell, seed 123 ----
cells = ["E0_j2", "E1_j2", "E2_j2", "E3_j2", "LONGRANGE", "TIGHT_5k1", "W1_driftwait"]
labels = ["LEO\ne≤0.05", "e≤0.10", "e≤0.20\nto 3,500 km", "e≤0.30\nto 8,000 km", "long range\n200 h", "tight box\n5 km, 1 m/s", "plane change\nvia J2 drift"]
rate = {}
for c in cells:
    rate[c] = json.load(open(f"{REPO}/web_data/results/t15e_dagger/fin_{c}.json"))["rate"] * 100
tight_floor = json.load(open(f"{REPO}/web_data/results/t15e_dagger/fin_TIGHT_5k1_T.json"))["rate"] * 100
fig, ax = plt.subplots(figsize=(6.3, 3.0))
x = np.arange(len(cells))
vals = [rate[c] for c in cells]
bars = ax.bar(x, vals, color=[C_TG]*5 + [C_BURN, C_CH], width=0.62)
for i, v in enumerate(vals):
    ax.text(i, v + 1.2, f"{v:.1f}%", ha="center", fontsize=9)
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
ax.set_ylim(0, 108); ax.set_ylabel("rendezvous success (%)")
ax.set_title("One policy, seven skills: held-out success, 200 episodes per cell", fontsize=9)
def wilson(p, n=200, z=1.96):
    p = p/100; c = (p + z*z/(2*n))/(1 + z*z/n); h = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n))/(1 + z*z/n)
    return (p - (c - h))*100, ((c + h) - p)*100
err = np.array([wilson(v) for v in vals]).T
ax.errorbar(x, vals, yerr=err, fmt="none", ecolor="k", capsize=3, lw=0.8)
ax.text(0.99, 0.04, "bars: 95% Wilson interval", transform=ax.transAxes, ha="right", fontsize=7.5, color="#555")
fig.savefig(os.path.join(out, "generalist_cells.pdf")); plt.close(fig)

# ---- 2. DAgger iteration curve on the hardest skill (W1) ----
it = ["kickstart", "iter 1", "iter 2", "iter 3", "iter 4"]
w1 = [31.5, 45.0, 59.0, 70.0, 74.0]
tight = [85.5, 86.0, 84.5, 87.0, 86.0]
wides_lo = [94.5, 94.0, 95.0, 96.0, 96.0]; wides_hi = [100, 99.0, 99.5, 99.5, 99.5]
fig, ax = plt.subplots(figsize=(6.3, 3.1))
ax.plot(it, w1, "o-", color=C_CH, lw=2, label="plane-change-via-J2 (hardest skill)")
ax.plot(it, tight, "s-", color=C_BURN, lw=1.4, label="tight box 5 km / 1 m/s")
ax.fill_between(range(5), wides_lo, wides_hi, color=C_TG, alpha=0.25, label="five wide-box skills (range)")
for i, v in enumerate(w1):
    ax.text(i, v + 2.5, f"{v:.1f}", ha="center", fontsize=9, color=C_CH)
ax.set_ylim(0, 105); ax.set_ylabel("held-out success (%)"); ax.set_xlabel("expert-relabelled refresh (DAgger) iteration")
ax.set_title("Learning on checkpoints: the seventh skill climbs, the other six hold", fontsize=9)
ax.legend(fontsize=8, frameon=False, loc="lower right")
fig.savefig(os.path.join(out, "dagger_curve.pdf")); plt.close(fig)

# ---- 3. interference seesaw: why naive fine-tuning fails ----
rows = ["E0_j2", "E1_j2", "E2_j2", "E3_j2", "LONGRANGE", "TIGHT_5k1"]
short = ["e≤.05", "e≤.10", "e≤.20", "e≤.30", "long range", "tight box"]
data = {
    "generalist\n(6 wide skills)": [97.0, 100.0, 99.5, 97.5, 99.0, 0.0],
    "+ tight-box\nfine-tune": [97.0, 96.5, 29.0, 6.5, 40.0, 92.5],
    "+ 50M\nre-mix": [93.5, 95.5, 98.5, 84.5, 99.5, 0.0],
    "consolidate\n@ low LR": [95.0, 98.0, 99.5, 94.0, 99.5, 37.5],
    "+ anchor\n(T13b)": [99.0, 99.5, 98.5, 94.5, 100.0, 90.5],
}
M = np.array(list(data.values())).T
fig, ax = plt.subplots(figsize=(6.3, 3.1))
im = ax.imshow(M, cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")
ax.set_xticks(range(M.shape[1])); ax.set_xticklabels(list(data.keys()), fontsize=8.5)
ax.set_yticks(range(M.shape[0])); ax.set_yticklabels(short, fontsize=9)
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        ax.text(j, i, f"{M[i,j]:.1f}", ha="center", va="center", fontsize=9, color="k" if 25 < M[i,j] < 85 else "w")
ax.set_title("Catastrophic interference, measured: each recipe step, held-out success (%) per skill", fontsize=9)
for s in ax.spines.values(): s.set_visible(False)
fig.savefig(os.path.join(out, "seesaw.pdf")); plt.close(fig)

# ---- 4. EKF noise sweep, redrawn clean (numbers from plots/relnav_t3/success_vs_noise.png) ----
lab = ["truth\nstate", "recon\n(no noise)", "EKF\n1×", "EKF\n3×", "EKF\n10×", "EKF\n30×", "EKF\n100×", "EKF\n300×"]
v = [100, 100, 100, 100, 99.5, 96.0, 81.5, 61.0]
fig, ax = plt.subplots(figsize=(6.3, 2.8))
ax.bar(range(8), v, color=["#777", "#aaa"] + [C_TG]*6, width=0.65)
for i, y in enumerate(v): ax.text(i, y + 1, f"{y:g}", ha="center", fontsize=9)
ax.set_xticks(range(8)); ax.set_xticklabels(lab, fontsize=8.5); ax.set_ylim(0, 110); ax.set_ylabel("success (%)")
ax.set_title("Policy flown on the filter's estimate: sensor-noise multiplier sweep (200 eps each)", fontsize=9)
fig.savefig(os.path.join(out, "noise_sweep.pdf")); plt.close(fig)

# ---- 5. C-port timing margin ----
fig, ax = plt.subplots(figsize=(6.3, 2.3))
items = [("p50 inference", 50.5e-6), ("p99 inference", 62.0e-6), ("decision period", 60.0)]
ax.barh([i[0] for i in items], [i[1] for i in items], color=[C_TG, C_TG, "#999"])
ax.set_xscale("log"); ax.set_xlabel("seconds (log)")
for i, (n, t) in enumerate(items):
    ax.text(t * 1.3, i, f"{t*1e6:.0f} µs" if t < 1 else f"{t:.0f} s", va="center", fontsize=9)
ax.set_xlim(1e-5, 1e3); ax.set_title("Embedded C port: ~10⁶× real-time margin, no heap, bit-exact vs PyTorch", fontsize=9)
fig.savefig(os.path.join(out, "cport_timing.pdf")); plt.close(fig)
print("done", rate, tight_floor)
