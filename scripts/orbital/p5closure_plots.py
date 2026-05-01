"""Static PNGs for blog post / web frontend hero images."""
import json, os, math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

os.makedirs("plots", exist_ok=True)

# ─── Plot 1: capability surface across e_max ──────────────────────────────

import csv
results = {}
with open("web_data/results/multiseed_escan.csv") as f:
    next(f)
    for row in csv.reader(f):
        e = float(row[1]); pct = int(row[2])/int(row[3])*100
        results.setdefault(e, []).append(pct)

es = sorted(results.keys())
means = [np.mean(results[e]) for e in es]
stds = [np.std(results[e]) for e in es]
mins = [min(results[e]) for e in es]
maxs = [max(results[e]) for e in es]

fig, ax = plt.subplots(figsize=(8, 5))
ax.errorbar(es, means, yerr=stds, fmt='o-', color='#1f77b4', capsize=5, linewidth=2,
             markersize=8, label='Multi-seed mean ± std')
ax.fill_between(es, mins, maxs, alpha=0.15, color='#1f77b4', label='min/max range')
ax.axhline(70, ls='--', color='#888', alpha=0.5, label='70% target')
ax.axhline(50, ls=':', color='#aaa', alpha=0.5, label='50% stretch')
ax.set_xlabel('Maximum eccentricity (e_max)', fontsize=12)
ax.set_ylabel('Success rate (%)', fontsize=12)
ax.set_title('Phase 5e — Capability surface across eccentricity\n(5 seeds × 200 eps each, valid_init_only=1)', fontsize=13)
ax.set_ylim(0, 100); ax.set_xlim(0, 0.75)
ax.grid(alpha=0.3)
ax.legend(loc='lower left')
fig.tight_layout()
fig.savefig("plots/p5e_capability_surface.png", dpi=120)
plt.close(fig)
print("Wrote plots/p5e_capability_surface.png")

# ─── Plot 2: training progression (seed 42) ─────────────────────────────

prog = []
with open("web_data/results/seed42_training_progression.csv") as f:
    next(f)
    for row in csv.reader(f):
        prog.append((int(row[0]), float(row[1])))
ep, perf = zip(*prog)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(ep, perf, 'o-', color='#d62728', linewidth=2, markersize=8)
ax.axhline(90, ls='--', color='#aaa', alpha=0.6)
ax.set_xlabel('Stage 4.0 epoch (≈ 144K steps each)', fontsize=12)
ax.set_ylabel('Held-out success rate at e=0.20 (%)', fontsize=12)
ax.set_title('Phase 5e seed 42 — Training progression\n(Stage 1.0 warm-start initially, Stage 4.0 retrains random sat init)', fontsize=12)
ax.set_ylim(0, 100)
ax.grid(alpha=0.3)
# Annotate the dip (warm-start collapse → recovery)
dip_idx = np.argmin(perf)
ax.annotate(f"warm-start adjust\n({perf[dip_idx]:.0f}% at ep{ep[dip_idx]})",
             xy=(ep[dip_idx], perf[dip_idx]),
             xytext=(ep[dip_idx]+40, perf[dip_idx]-15),
             arrowprops=dict(arrowstyle='->', color='gray', alpha=0.6),
             fontsize=10, color='gray')
fig.tight_layout()
fig.savefig("plots/p5e_training_progression.png", dpi=120)
plt.close(fig)
print("Wrote plots/p5e_training_progression.png")

# ─── Plot 3: action distribution (seed 42, e=0.20) ──────────────────────

bonus = json.load(open("web_data/results/bonus_stats.json"))
s = bonus["phase5e_seed42"]["e_0.20"]
actions = s["action_distribution"]
total = s["action_total"]
order = ["coast", "warp_5min", "prograde_5", "prograde_10", "prograde_25",
         "retrograde_5", "retrograde_10", "retrograde_25", "radial_out", "radial_in"]
counts = [actions.get(a, 0) for a in order]
pcts = [100*c/total for c in counts]

fig, ax = plt.subplots(figsize=(10, 4))
colors = ['#888', '#1f77b4', '#2ca02c', '#2ca02c', '#2ca02c',
          '#d62728', '#d62728', '#d62728', '#9467bd', '#9467bd']
ax.barh(order, pcts, color=colors)
for i, (a, p) in enumerate(zip(order, pcts)):
    ax.text(p + 1, i, f'{p:.1f}%', va='center', fontsize=9)
ax.set_xlabel('Action frequency (% of all action selections)', fontsize=12)
ax.set_xlim(0, 100)
ax.set_title('Phase 5e — Action distribution at e=0.20\n(seed 42, 200 episodes)\nThe agent learned to wait — 94% of actions are time-warps.', fontsize=12)
ax.invert_yaxis()
fig.tight_layout()
fig.savefig("plots/p5e_action_distribution.png", dpi=120)
plt.close(fig)
print("Wrote plots/p5e_action_distribution.png")

# ─── Plot 4: fuel efficiency histogram (e=0.20 successes) ──────────────

# Re-extract dv/Hohmann ratio from highlights data + raw eval
import glob
ratios = []
for f in glob.glob("/tmp/p5closure_escan_puffer_orbital_177765503091_e0.20/ep_*.npz"):
    d = np.load(f)
    if float(d["episode_reward"][0]) <= 0: continue
    mask = (d["sat_x"] != 0) | (d["sat_y"] != 0)
    active = np.where(mask)[0]
    total_dv = float(np.abs(d["delta_v"][active]).sum())
    sat_a0 = float(d["sat_a"][0]); tgt_a0 = float(d["target_a"][0])
    MU = 3.986004418e14
    a_t = 0.5*(sat_a0 + tgt_a0)
    v1c = math.sqrt(MU/sat_a0); v2c = math.sqrt(MU/tgt_a0)
    v1t = math.sqrt(MU*(2/sat_a0 - 1/a_t)); v2t = math.sqrt(MU*(2/tgt_a0 - 1/a_t))
    h = abs(v1t-v1c) + abs(v2c-v2t)
    if h > 1e-3: ratios.append(total_dv / h)

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.hist(ratios, bins=30, color='#2ca02c', edgecolor='black', alpha=0.75)
ax.axvline(1.0, ls='--', color='#000', alpha=0.6, label='Hohmann optimal (circular)')
ax.axvline(np.median(ratios), ls='-', color='#d62728', alpha=0.8,
           label=f'Median: {np.median(ratios):.2f}×')
ax.set_xlabel('Δv used / Hohmann transfer Δv', fontsize=12)
ax.set_ylabel('Count of successful episodes', fontsize=12)
ax.set_title(f'Phase 5e — Fuel efficiency at e=0.20\n(seed 42, {len(ratios)} successful episodes)', fontsize=12)
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("plots/p5e_fuel_efficiency.png", dpi=120)
plt.close(fig)
print("Wrote plots/p5e_fuel_efficiency.png")

print("\nAll plots written.")
