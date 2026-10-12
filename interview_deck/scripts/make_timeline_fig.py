"""Decision timeline of one headline episode: what the policy actually decided, and when.
Usage: python make_timeline_fig.py <episode.npz> <out_dir>"""
import sys, os, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
f, out = sys.argv[1], sys.argv[2]
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150, "savefig.bbox": "tight"})
C_CH, C_TG, C_BURN, C_COAST = "#d7301f", "#2c7fb8", "#f0a202", "#b0b0b0"
d = np.load(f)
act = d["action"].astype(int); t = d["sim_time"] / 3600; dv = d["delta_v"]
tau = {9: 5, 10: 30, 11: 60}
names = {0: "coast 1 min", 9: "coast 5 min", 10: "coast 30 min", 11: "coast 1 h"}
# collapse sub-steps into decisions
dec = []; i = 0; burns = set(np.nonzero(dv > 0)[0].tolist())
while i < len(act):
    if i in burns:
        dec.append((t[i], 1 / 60.0, act[i], dv[i])); i += 1; continue
    a = act[i]; n = tau.get(a, 1)
    nxt = [b for b in burns if b > i]
    if nxt: n = min(n, min(nxt) - i)
    dec.append((t[i], n / 60.0, 0 if a in tau or a == 0 else 0, 0.0)); i += n
fig, ax = plt.subplots(figsize=(6.3, 2.0))
y = 0
for t0, dur, a, v in dec:
    if v <= 0:
        ax.barh(y, dur, left=t0, height=0.5, color=C_COAST, ec="white", lw=0.3)
    else:
        pro = (1 <= a <= 3) or a in (12, 14)
        col = C_CH if pro else C_TG
        ax.barh(y, 0.05, left=t0 - 0.025, height=0.9, color=col)
        ax.text(t0, 0.55, f"{'+' if pro else '-'}{v:.0f}", ha="center", fontsize=7, color=col)
ax.set_xlim(0, t[-1] + 0.1); ax.set_ylim(-0.6, 0.9); ax.set_yticks([])
ax.set_xlabel("time (h)")
nburn = sum(1 for _, _, _, v in dec if v > 0)
ax.set_title(f"One held-out episode: {len(dec)} decisions over {t[-1]:.1f} h, {nburn} burns ({dv.sum():.0f} m/s), the rest coasts", fontsize=9)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=C_COAST, label="coast (5 min to 1 h)"), Patch(color=C_CH, label="prograde burn (m/s)"), Patch(color=C_TG, label="retrograde burn (m/s)")],
          loc="upper center", bbox_to_anchor=(0.5, -0.35), ncol=3, fontsize=8, frameon=False)
fig.savefig(os.path.join(out, "decision_timeline.pdf")); plt.close(fig)
print(len(dec), nburn, dv.sum())
