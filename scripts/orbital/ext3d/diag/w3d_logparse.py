#!/usr/bin/env python3
"""Parse the PufferLib TUI training logs into a rolling-metric CSV.

The dashboard repaints the whole panel each epoch, so every frame carries a
consistent (Steps, Epoch, perf, episode_length, ...) tuple. Strip ANSI, then
pull the labelled values frame by frame.

Usage: w3d_logparse.py /tmp/t5_w3d_W3_s42_train.log [...]
Writes /tmp/w3d_trainlog.csv
"""
import csv
import re
import sys

ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
NUM = r"([-+]?[0-9]*\.?[0-9]+)([KMGB]?)"
MULT = {"": 1.0, "K": 1e3, "M": 1e6, "G": 1e9, "B": 1e9}


def num(tok):
    m = re.match(NUM + r"$", tok)
    if not m:
        return None
    return float(m.group(1)) * MULT.get(m.group(2), 1.0)


def field(line, label):
    """Extract 'label  <value>' from a de-ansi'd dashboard row."""
    i = line.find(label)
    if i < 0:
        return None
    rest = line[i + len(label):]
    toks = rest.replace("│", " ").split()
    for t in toks[:3]:
        v = num(t)
        if v is not None:
            return v
    return None


def parse(path):
    rows, cur = [], {}
    with open(path, errors="ignore") as f:
        for raw in f:
            line = ANSI.sub("", raw).replace("\r", "")
            if "Steps" in line and "Env" not in line:
                v = field(line, "Steps")
                if v is not None:
                    cur["steps"] = v
            if "Epoch" in line:
                v = field(line, "Epoch")
                if v is not None:
                    cur["epoch"] = v
            for lab, key in (("perf", "perf"), ("episode_return", "episode_return"),
                             ("episode_length", "episode_length"),
                             ("fuel_used", "fuel_used"), ("entropy", "entropy"),
                             ("policy_loss", "policy_loss"),
                             ("explained_var", "explained_var"),
                             ("realized_e_target_mean", "e_tgt"),
                             ("realized_e_sat_mean", "e_sat"),
                             ("init_attempts_mean", "init_attempts"),
                             ("n ", "n_eps")):
                if lab in line:
                    v = field(line, lab)
                    if v is not None:
                        cur[key] = v
            # a frame ends at the bottom border
            if "╰" in line and "perf" in cur and "steps" in cur:
                rows.append(dict(cur))
                cur = {k: v for k, v in cur.items() if k in ("steps", "epoch")}
    # dedupe by epoch, keep last
    seen = {}
    for r in rows:
        seen[r.get("epoch", -1)] = r
    return [seen[k] for k in sorted(seen)]


def main():
    allr = []
    for p in sys.argv[1:]:
        tag = p.split("/")[-1].replace("_train.log", "")
        for r in parse(p):
            r["run"] = tag
            allr.append(r)
        rr = [r for r in allr if r["run"] == tag]
        print(f"\n=== {tag}  frames={len(rr)} ===")
        if not rr:
            continue
        keys = ["steps", "perf", "episode_length", "episode_return", "fuel_used",
                "entropy", "e_tgt", "e_sat"]
        print("  " + "  ".join(f"{k:>14}" for k in keys))
        step = max(1, len(rr) // 24)
        for r in rr[::step] + [rr[-1]]:
            print("  " + "  ".join(
                (f"{r.get(k, float('nan')):14.4g}" if isinstance(r.get(k), float) else f"{'':>14}")
                for k in keys))
        pk = max(rr, key=lambda r: r.get("perf", -1))
        print(f"  peak perf {pk.get('perf'):.3f} @ steps {pk.get('steps'):.0f} "
              f"(epoch {pk.get('epoch')})   final perf {rr[-1].get('perf'):.3f}")
    if allr:
        keys = sorted({k for r in allr for k in r})
        with open("/tmp/w3d_trainlog.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader(); w.writerows(allr)
        print("\nwrote /tmp/w3d_trainlog.csv")


if __name__ == "__main__":
    main()
