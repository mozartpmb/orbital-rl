"""Aggregate per-seed surface CSV into mean ± std per cell + plot heatmaps."""
import csv, json, os
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            r["e_fixed"] = float(r["e_fixed"])
            r["phase_gap_deg"] = int(r["phase_gap_deg"])
            r["success_n"] = int(r["success_n"])
            r["total"] = int(r["total"])
            r["pct"] = 100 * r["success_n"] / max(r["total"], 1)
            rows.append(r)
    return rows


def main():
    rows = load("web_data/results/phase5_capability_surface.csv")
    # Aggregate: (e, phase, relation) → list of pct across seeds
    agg = defaultdict(list)
    for r in rows:
        key = (r["e_fixed"], r["phase_gap_deg"], r["relation"])
        agg[key].append(r["pct"])

    # Write aggregate CSV
    out_csv = "web_data/results/phase5_capability_surface_agg.csv"
    summary = []
    with open(out_csv, "w") as f:
        w = csv.writer(f)
        w.writerow(["e_fixed", "phase_gap_deg", "relation", "n_seeds",
                     "mean_pct", "std_pct", "min_pct", "max_pct"])
        for key in sorted(agg.keys()):
            arr = np.array(agg[key])
            row = (*key, len(arr), arr.mean(), arr.std(), arr.min(), arr.max())
            w.writerow([f"{v:.4g}" if isinstance(v, float) else v for v in row])
            summary.append({"e": key[0], "phase": key[1], "relation": key[2],
                            "n_seeds": len(arr), "mean": float(arr.mean()),
                            "std": float(arr.std())})
    print(f"Wrote {out_csv}")

    # JSON summary
    out_json = "web_data/results/phase5_capability_surface.json"
    with open(out_json, "w") as f:
        json.dump({"cells": summary,
                    "axes": {"e_fixed": sorted(set(r["e_fixed"] for r in rows)),
                             "phase_gap_deg": sorted(set(r["phase_gap_deg"] for r in rows)),
                             "relation": sorted(set(r["relation"] for r in rows))},
                    "n_seeds_per_cell": len(set(r["seed"] for r in rows))}, f, indent=2)
    print(f"Wrote {out_json}")

    # Heatmap: per relation, e × phase
    relations = sorted(set(r["relation"] for r in rows))
    phases = sorted(set(r["phase_gap_deg"] for r in rows))
    es = sorted(set(r["e_fixed"] for r in rows))

    fig, axs = plt.subplots(1, len(relations), figsize=(11, 4.5))
    for i, rel in enumerate(relations):
        # Build matrix [phase × e]
        M = np.zeros((len(phases), len(es)))
        S = np.zeros_like(M)
        for pi, p in enumerate(phases):
            for ei, e in enumerate(es):
                arr = np.array(agg[(e, p, rel)])
                M[pi, ei] = arr.mean()
                S[pi, ei] = arr.std()
        ax = axs[i] if len(relations) > 1 else axs
        im = ax.imshow(M, aspect="auto", cmap="RdYlGn", vmin=0, vmax=100,
                        origin="lower")
        ax.set_xticks(range(len(es))); ax.set_xticklabels([f"{e:g}" for e in es])
        ax.set_yticks(range(len(phases))); ax.set_yticklabels([f"{p}°" for p in phases])
        ax.set_xlabel("Fixed eccentricity (sat & target)")
        ax.set_ylabel("Initial phase gap")
        title_relation = "fully random" if rel == "fully_random" else "same orbit init"
        ax.set_title(f"{title_relation}", fontsize=11)
        # Annotate cells
        for pi in range(len(phases)):
            for ei in range(len(es)):
                ax.text(ei, pi, f"{M[pi,ei]:.0f}", ha="center", va="center",
                         fontsize=8, color="black" if 30 < M[pi,ei] < 80 else "white")
    fig.colorbar(im, ax=axs, label="Success rate (%)", fraction=0.04, pad=0.02)
    fig.suptitle("Phase 5 capability surface — per-cell success rate (5-seed mean)\n"
                  "Altitude band auto-scaled per e to keep perigee >= EARTH_KEEPOUT", fontsize=12)
    fig.savefig("plots/phase5_capability_surface_fixed.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("Wrote plots/phase5_capability_surface_fixed.png")

    # Print headline numbers
    print("\n=== Headline per-condition multi-seed mean ± std ===")
    print(f"{'e':>5} {'phase':>6} {'relation':>16} {'mean':>7} {'std':>5}")
    for key in sorted(agg.keys()):
        arr = np.array(agg[key])
        print(f"{key[0]:>5.2f} {key[1]:>4}° {key[2]:>16} {arr.mean():>6.1f}% {arr.std():>4.1f}")


if __name__ == "__main__":
    main()
