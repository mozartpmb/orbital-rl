"""Fuel-awareness figure: T11 generalist evaluated with a fixed lean (353 m/s) vs rich (656 m/s) tank.
Source: web_data/results/t11/s4_{lean,rich}_*.json (200 eps/cell, seed 123, bearings-only)."""
import json, sys, os, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
out = sys.argv[1]; REPO = "/Users/pete/space_training"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150, "savefig.bbox": "tight"})
cells = ["E0_j2", "E1_j2", "E2_j2", "E3_j2", "LONGRANGE"]
lab = ["LEO e≤.05", "e≤.10", "e≤.20", "e≤.30", "long range"]
L = [json.load(open(f"{REPO}/web_data/results/t11/s4_lean_{c}.json")) for c in cells]
R = [json.load(open(f"{REPO}/web_data/results/t11/s4_rich_{c}.json")) for c in cells]
x = np.arange(5); w = 0.38
fig, (a, b) = plt.subplots(1, 2, figsize=(6.3, 2.6), gridspec_kw={"width_ratios": [1.1, 1]})
a.bar(x - w/2, [d["dv_used_med"] for d in L], w, color="#9ec5f4", label="lean tank, 353 m/s")
a.bar(x + w/2, [d["dv_used_med"] for d in R], w, color="#1c5cab", label="rich tank, 656 m/s")
a.axhline(353, color="#9ec5f4", ls="--", lw=0.8); a.axhline(656, color="#1c5cab", ls="--", lw=0.8)
a.set_xticks(x); a.set_xticklabels(lab, fontsize=8); a.set_ylabel("median Δv spent (m/s)"); a.set_ylim(0, 700)
a.legend(fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2); a.set_title("same weights, different tank", fontsize=9)
b.bar(x - w/2, [100*d["rate"] for d in L], w, color="#9ec5f4"); b.bar(x + w/2, [100*d["rate"] for d in R], w, color="#1c5cab")
for i, (l, r) in enumerate(zip(L, R)):
    b.text(i - w/2, 100*l["rate"] + 1, f"{100*l['rate']:.0f}", ha="center", fontsize=7)
    b.text(i + w/2, 100*r["rate"] + 1, f"{100*r['rate']:.0f}", ha="center", fontsize=7)
b.set_xticks(x); b.set_xticklabels(lab, fontsize=8); b.set_ylabel("success (%)"); b.set_ylim(0, 110)
b.set_title("price of the smaller tank", fontsize=9)
fig.savefig(os.path.join(out, "fuel_aware.pdf")); plt.close(fig)
print([(c, round(l["dv_used_med"]), round(r["dv_used_med"]), l["rate"], r["rate"]) for c, l, r in zip(cells, L, R)])
