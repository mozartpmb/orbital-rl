"""Plot LEO probe — per-eccentricity capability at trained altitude band."""
import csv, os
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

agg = defaultdict(list)
with open("web_data/results/phase5_leo_probe.csv") as f:
    next(f)
    for row in csv.reader(f):
        agg[(float(row[1]), int(row[2]))].append(int(row[3])/int(row[4])*100)

es = sorted(set(k[0] for k in agg))
phases = sorted(set(k[1] for k in agg))

fig, ax = plt.subplots(figsize=(8.5, 5))
colors = plt.cm.viridis(np.linspace(0, 0.9, len(phases)))
for ph, color in zip(phases, colors):
    means = [np.mean(agg[(e, ph)]) for e in es]
    stds = [np.std(agg[(e, ph)]) for e in es]
    ax.errorbar(es, means, yerr=stds, fmt='o-', color=color, capsize=4,
                 markersize=7, label=f'phase {ph}°', linewidth=1.6)
ax.axvline(0.05, ls='--', color='#888', alpha=0.6)
ax.text(0.052, 50, 'training\ne_max_sat = 0.05', fontsize=9, color='#666')
ax.set_xlabel("Fixed sat & target eccentricity", fontsize=12)
ax.set_ylabel("Success rate (%)", fontsize=12)
ax.set_title("Phase 5 LEO probe — per-eccentricity capability at trained altitude band\n"
             "(300-800 km altitude, fully random sat init, 5 seeds × 100 eps each)", fontsize=11)
ax.set_xlim(-0.005, 0.105); ax.set_ylim(-2, 105)
ax.grid(alpha=0.3); ax.legend(loc='lower left')
fig.tight_layout()
fig.savefig("plots/phase5_leo_probe.png", dpi=120)
plt.close(fig)
print("Wrote plots/phase5_leo_probe.png")
