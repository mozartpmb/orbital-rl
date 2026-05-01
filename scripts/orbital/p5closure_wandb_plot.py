"""Plot wandb training curves for canonical seed 42 retrain."""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

s10 = json.load(open("web_data/results/wandb_curves/stage_1_0_seed42.json"))["history"]
s40 = json.load(open("web_data/results/wandb_curves/stage_4_0_seed42.json"))["history"]

def extract(history, key):
    pairs = [(r["agent_steps"], r[key]) for r in history if key in r and "agent_steps" in r]
    if not pairs: return [], []
    pairs.sort(key=lambda x: x[0])
    return zip(*pairs)

fig, axs = plt.subplots(2, 2, figsize=(12, 8), sharex=False)

# Top-left: success rate (perf)
for h, label, color in [(s10, "Stage 1.0 (e=0.05, same_orbit_init=1)", "#1f77b4"),
                          (s40, "Stage 4.0 (e=0.05, random sat init)", "#d62728")]:
    x, y = extract(h, "environment/perf")
    if x: axs[0,0].plot(np.array(x)/1e6, [v*100 for v in y], color=color, alpha=0.7, label=label)
axs[0,0].set_xlabel("Agent steps (M)"); axs[0,0].set_ylabel("Success rate (%)")
axs[0,0].set_title("Training-time success rate"); axs[0,0].legend(loc='lower right'); axs[0,0].grid(alpha=0.3)

# Top-right: episode return
for h, label, color in [(s10, "Stage 1.0", "#1f77b4"), (s40, "Stage 4.0", "#d62728")]:
    x, y = extract(h, "environment/episode_return")
    if x: axs[0,1].plot(np.array(x)/1e6, list(y), color=color, alpha=0.7, label=label)
axs[0,1].set_xlabel("Agent steps (M)"); axs[0,1].set_ylabel("Episode return")
axs[0,1].set_title("Episode return (terminal +10/-10 + shaping)"); axs[0,1].legend(loc='lower right'); axs[0,1].grid(alpha=0.3)

# Bottom-left: entropy
for h, label, color in [(s10, "Stage 1.0", "#1f77b4"), (s40, "Stage 4.0", "#d62728")]:
    x, y = extract(h, "losses/entropy")
    if x: axs[1,0].plot(np.array(x)/1e6, list(y), color=color, alpha=0.7, label=label)
axs[1,0].set_xlabel("Agent steps (M)"); axs[1,0].set_ylabel("Entropy")
axs[1,0].set_title("Policy entropy"); axs[1,0].legend(loc='upper right'); axs[1,0].grid(alpha=0.3)

# Bottom-right: explained variance
for h, label, color in [(s10, "Stage 1.0", "#1f77b4"), (s40, "Stage 4.0", "#d62728")]:
    x, y = extract(h, "losses/explained_variance")
    if x: axs[1,1].plot(np.array(x)/1e6, list(y), color=color, alpha=0.7, label=label)
axs[1,1].set_xlabel("Agent steps (M)"); axs[1,1].set_ylabel("Explained variance")
axs[1,1].set_title("Value-function fit quality"); axs[1,1].legend(loc='lower right'); axs[1,1].grid(alpha=0.3)

fig.suptitle("Phase 5e — Canonical seed 42 training curves (wandb)", fontsize=14, y=1.00)
fig.tight_layout()
fig.savefig("plots/p5e_wandb_curves.png", dpi=120)
plt.close(fig)
print("Wrote plots/p5e_wandb_curves.png")
