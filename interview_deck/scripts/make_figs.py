"""Deck figures from post-fix data. Usage: python make_figs.py <traj_dir> <out_dir>
traj_dir = output of eval_checkpoint.py on models/t3/seed42_L2_headline.pt (R1 flags)."""
import sys, glob, os, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import pandas as pd

traj_dir, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
REPO = "/Users/pete/space_training"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 150, "savefig.bbox": "tight", "font.family": "sans-serif"})
C_EARTH, C_CH, C_TG, C_BURN = "#9ecae1", "#d7301f", "#2c7fb8", "#f0a202"
R_E = 6371.0

files = sorted(glob.glob(os.path.join(traj_dir, "ep_*.npz")))
def load(f):
    d = np.load(f); return {k: d[k] for k in d.files}

# ---------- 1. single rendezvous episode: chaser + target + closing ----------
def pick_episode():
    best = None
    for f in files:
        d = load(f)
        gap = np.degrees(np.arctan2(d["target_y"][0], d["target_x"][0]) - np.arctan2(d["sat_y"][0], d["sat_x"][0]))
        gap = (gap + 180) % 360 - 180
        n = len(d["sim_time"])
        if 90 < abs(gap) < 180 and 300 < n < 1500 and (best is None or n < best[0]):
            best = (n, f, gap)
    return best[1], best[2]

f, gap = pick_episode(); d = load(f)
t_h = d["sim_time"] / 3600
burn = d["delta_v"] > 0
rel = np.hypot(d["sat_x"] - d["target_x"], d["sat_y"] - d["target_y"]) / 1e3
relv = np.hypot(d["sat_vx"] - d["target_vx"], d["sat_vy"] - d["target_vy"])
# target-relative phasing view: phase gap (deg, along-track) vs radial offset (km)
th_c = np.arctan2(d["sat_y"], d["sat_x"]); th_t = np.arctan2(d["target_y"], d["target_x"])
dphase = np.degrees((th_c - th_t + np.pi) % (2*np.pi) - np.pi)
dr = (np.hypot(d["sat_x"], d["sat_y"]) - np.hypot(d["target_x"], d["target_y"])) / 1e3
fig, (a1, a0) = plt.subplots(1, 2, figsize=(6.3, 3.1), gridspec_kw={"width_ratios": [1, 1.1]})
fig.subplots_adjust(bottom=0.24, wspace=0.35)
fig2, a2 = plt.subplots(figsize=(6.3, 2.9))
a0.plot(dphase, dr, color=C_CH, lw=1.2)
a0.scatter(dphase[burn], dr[burn], marker="^", s=55, color=C_BURN, ec="k", lw=0.5, zorder=5)
a0.plot(dphase[0], dr[0], "o", color=C_CH, ms=8, mec="k"); a0.plot(dphase[-1], dr[-1], "k*", ms=13)
a0.axhline(0, color=C_TG, lw=1, alpha=0.7); a0.axvline(0, color=C_TG, lw=1, alpha=0.7)
a0.set_xlabel("phase relative to target (deg, + = ahead)"); a0.set_ylabel("radial offset from target (km)")
a0.set_title("target-relative: below and behind, catch up, raise", fontsize=9)
a0.annotate("target", (0, 0), xytext=(6, 6), textcoords="offset points", color=C_TG, fontsize=9)
a1.add_patch(Circle((0, 0), R_E, color=C_EARTH, zorder=0))
a1.plot(d["target_x"] / 1e3, d["target_y"] / 1e3, color=C_TG, lw=1.0, alpha=0.6, label="target")
a1.plot(d["sat_x"] / 1e3, d["sat_y"] / 1e3, color=C_CH, lw=1.4, label="chaser (learned policy)")
a1.scatter(d["sat_x"][burn] / 1e3, d["sat_y"][burn] / 1e3, marker="^", s=55, color=C_BURN, ec="k", lw=0.5, zorder=5, label="burn")
a1.plot(d["sat_x"][0] / 1e3, d["sat_y"][0] / 1e3, "o", color=C_CH, ms=8, mec="k", label="chaser start")
a1.plot(d["target_x"][0] / 1e3, d["target_y"][0] / 1e3, "o", color=C_TG, ms=8, mec="k", label="target start")
a1.plot(d["sat_x"][-1] / 1e3, d["sat_y"][-1] / 1e3, "*", color="k", ms=13, label="rendezvous")
a1.set_aspect("equal"); a1.set_xlabel("x (km)"); a1.set_ylabel("y (km)")
fig.legend(*a1.get_legend_handles_labels(), fontsize=7, loc="lower center", ncol=6, frameon=False, bbox_to_anchor=(0.5, -0.04))
a1.set_title(f"gap {abs(gap):.0f}°, Δv {d['delta_v'].sum():.0f} m/s, {t_h[-1]:.1f} h", fontsize=9)
a2.plot(t_h, rel, color=C_CH, lw=1.6, label="range (km)")
a2.plot(t_h, relv, color=C_TG, lw=1.2, alpha=0.9, label="relative speed (m/s)")
a2.set_yscale("log"); a2.set_xlabel("time (h)"); a2.set_ylabel("range (km)  /  relative speed (m/s)")
a2.axhline(30, color=C_CH, ls="--", lw=0.8); a2.text(0.1, 36, "30 km", fontsize=8, color=C_CH)
a2.axhline(50, color=C_TG, ls=":", lw=0.8); a2.text(0.1, 60, "50 m/s", fontsize=8, color=C_TG)
for tb, dv in zip(t_h[burn], d["delta_v"][burn]):
    a2.axvline(tb, color=C_BURN, lw=0.8, alpha=0.8)
a2.legend(fontsize=8, frameon=False, loc="upper right")
a2.set_title("closing into the 30 km / 50 m/s box; bars are burns", fontsize=9)
fig.savefig(os.path.join(out, "traj_rendezvous.pdf")); plt.close(fig)
fig2.savefig(os.path.join(out, "traj_closing.pdf")); plt.close(fig2)
print("episode", f, "gap", gap)

# ---------- 2. 30-episode overlay ----------
fig, ax = plt.subplots(figsize=(3.0, 3.0))
ax.add_patch(Circle((0, 0), R_E, color=C_EARTH, zorder=0))
for f2 in files[:30]:
    e = load(f2); ax.plot(e["sat_x"] / 1e3, e["sat_y"] / 1e3, color=C_CH, lw=0.5, alpha=0.5)
    ax.plot(e["sat_x"][-1] / 1e3, e["sat_y"][-1] / 1e3, "k*", ms=5)
ax.set_aspect("equal"); ax.set_xlabel("x (km)"); ax.set_ylabel("y (km)")
ax.set_title("30 held-out episodes, every one a rendezvous", fontsize=9)
fig.savefig(os.path.join(out, "traj_overlay.pdf")); plt.close(fig)

# ---------- 3. action histogram over 200 episodes ----------
acts = np.concatenate([load(f2)["action"] for f2 in files]).astype(int)
names = ["coast 1 min","pro +5","pro +10","pro +25","retro −5","retro −10","retro −25","radial out 10","radial in 10",
         "warp 5 min","warp 30 min","warp 1 h","pro +1","retro −1","pro +2","retro −2"]
tau = np.array([1]*9 + [5, 30, 60] + [1]*4)
cnt = np.bincount(acts, minlength=16)[:16] / tau   # substeps -> decisions
fig, ax = plt.subplots(figsize=(6.3, 2.9))
cols = ["#999"] + [C_CH]*3 + [C_TG]*3 + [C_BURN]*2 + ["#555"]*3 + [C_CH, C_TG, C_CH, C_TG]
ax.bar(range(len(cnt)), 100*cnt/cnt.sum(), color=cols[:len(cnt)])
ax.set_xticks(range(len(cnt))); ax.set_xticklabels(names[:len(cnt)], rotation=55, ha="right", fontsize=8)
ax.set_ylabel("% of decisions"); ax.set_title("Headline policy, 200 held-out episodes: %d decisions, %.1f per episode" % (cnt.sum(), cnt.sum()/len(files)), fontsize=9)
fig.savefig(os.path.join(out, "action_hist.pdf")); plt.close(fig)
np.save(os.path.join(out, "action_counts.npy"), cnt)

# ---------- 4. Δv ratio histograms from fuel_audit.csv ----------
fa = pd.read_csv(os.path.join(REPO, "web_data/results/fuel_audit.csv"))
t3 = fa[(fa.lineage == "T3-canonical") & (fa.success == 1)]
fig, ax = plt.subplots(figsize=(6.3, 3.0))
bins = np.linspace(0, 4, 41)
ax.hist(t3.dv_total / t3.lambert_dv_boxcredited, bins, alpha=0.6, color=C_TG, label=f"vs Lambert, box-credited (median {np.median(t3.dv_total/t3.lambert_dv_boxcredited):.2f}×)")
ax.hist(t3.dv_total / t3.lambert_dv_timematched, bins, alpha=0.6, color=C_CH, label=f"vs Lambert, time-matched (median {np.median(t3.dv_total/t3.lambert_dv_timematched):.2f}×)")
ax.axvline(1, color="k", lw=0.8, ls="--"); ax.set_xlabel("learned Δv / two-impulse Lambert optimum"); ax.set_ylabel("episodes")
ax.legend(fontsize=8, frameon=False); ax.set_title("Fuel audit, 200 held-out headline episodes", fontsize=9)
fig.savefig(os.path.join(out, "dv_ratio.pdf")); plt.close(fig)
print("medians", np.median(t3.dv_total/t3.hohmann_dv), np.median(t3.dv_total/t3.lambert_dv_boxcredited), np.median(t3.dv_total/t3.lambert_dv_timematched))
print("lineages", fa.groupby("lineage").apply(lambda g: (len(g), g.success.mean(), np.median(g[g.success==1].ratio_boxcredited))).to_dict())

# ---------- 5. Lambert porkchop for the chosen episode ----------
sys.path.insert(0, os.path.join(REPO, "scripts/orbital"))
import t1_lambert_baseline as L
rc0 = np.array([d["sat_x"][0], d["sat_y"][0]], float); vc0 = np.array([d["sat_vx"][0], d["sat_vy"][0]], float)
rt0 = np.array([d["target_x"][0], d["target_y"][0]], float); vt0 = np.array([d["target_vx"][0], d["target_vy"][0]], float)
P = 2*np.pi*np.sqrt(float(d["target_a"][0])**3 / L.MU)
T_ep = float(d["sim_time"][-1])          # time-matched comparator: wait + flight <= policy's own episode time
waits = np.linspace(0, T_ep, 240); tofs = np.geomspace(600, T_ep, 240)
W, F = waits[:, None], tofs[None, :]
rc, vc = L.propagate_rv(rc0, vc0, W); r2, v2t = L.propagate_rv(rt0, vt0, W + F)
rc = np.broadcast_to(rc, r2.shape); vc = np.broadcast_to(vc, r2.shape)
best = np.full((len(waits), len(tofs)), np.inf)
for revs in range(17):
    for br in ((0,) if revs == 0 else (0, 1)):
        best = np.minimum(best, L._dv_on_grid(rc, vc, r2, v2t, F, revs, br))
best[(W + F) > T_ep] = np.inf
fig, ax = plt.subplots(figsize=(6.3, 3.6))
m = ax.pcolormesh(waits/3600, tofs/3600, np.log10(np.clip(best.T, 50, 3000)), shading="auto", cmap="viridis_r", vmin=np.log10(50), vmax=np.log10(3000))
i, j = np.unravel_index(np.argmin(best), best.shape)
ax.plot(waits[i]/3600, tofs[j]/3600, "*", color="w", mec="k", ms=16, label=f"Lambert optimum {best[i,j]:.0f} m/s")
ax.set_yscale("log"); ax.set_xlabel("coast before first burn (h)"); ax.set_ylabel("time of flight (h)")
cb = fig.colorbar(m, ax=ax, ticks=np.log10([50, 100, 200, 500, 1000, 3000])); cb.ax.set_yticklabels(["50", "100", "200", "500", "1000", "3000"]); cb.set_label("two-impulse Δv (m/s), best of ≤16 revs")
ax.legend(fontsize=8, loc="upper right"); ax.set_title(f"Lambert, same episode and time budget ({T_ep/3600:.1f} h): policy spent {d['delta_v'].sum():.0f} m/s", fontsize=9)
fig.savefig(os.path.join(out, "lambert_porkchop.pdf")); plt.close(fig)
print("lambert min", best[i,j])
