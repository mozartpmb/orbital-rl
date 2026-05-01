"""D2 — Fine-grained length histogram, success vs failure."""
import glob, numpy as np

files = sorted(glob.glob("/Users/pete/space_training/logs/orbital/p5c_s40_at_e020/ep_*.npz"))
succ_lens, fail_lens = [], []
for f in files:
    d = np.load(f)
    mask = (d["sat_x"] != 0) | (d["sat_y"] != 0)
    n = int(mask.sum())
    if n < 1: continue
    if float(d["episode_reward"][0]) > 0:
        succ_lens.append(n)
    else:
        fail_lens.append(n)

print(f"Successes: {len(succ_lens)}, Failures: {len(fail_lens)}")
print(f"\n=== Failure length histogram (bin=100) ===")
bins = list(range(0, 2100, 100))
hist, _ = np.histogram(fail_lens, bins=bins)
for i, c in enumerate(hist):
    if c == 0: continue
    bar = "█" * c
    print(f"  [{bins[i]:>4}, {bins[i+1]:>4}): {c:>3} {bar}")

print(f"\n=== Success length histogram (bin=100) ===")
hist, _ = np.histogram(succ_lens, bins=bins)
for i, c in enumerate(hist):
    if c == 0: continue
    bar = "█" * c
    print(f"  [{bins[i]:>4}, {bins[i+1]:>4}): {c:>3} {bar}")

print(f"\n=== Failure length histogram (bin=20, first 200 only) ===")
bins = list(range(0, 220, 20))
sub = [n for n in fail_lens if n < 200]
hist, _ = np.histogram(sub, bins=bins)
for i, c in enumerate(hist):
    if c == 0: continue
    bar = "█" * c
    print(f"  [{bins[i]:>4}, {bins[i+1]:>4}): {c:>3} {bar}")

print(f"\nFailure length percentiles: p10={np.percentile(fail_lens, 10):.0f} p25={np.percentile(fail_lens, 25):.0f} p50={np.percentile(fail_lens, 50):.0f} p75={np.percentile(fail_lens, 75):.0f} p90={np.percentile(fail_lens, 90):.0f}")
print(f"Failures n<50: {sum(1 for n in fail_lens if n<50)} ({100*sum(1 for n in fail_lens if n<50)/len(fail_lens):.1f}%)")
print(f"Failures n<100: {sum(1 for n in fail_lens if n<100)} ({100*sum(1 for n in fail_lens if n<100)/len(fail_lens):.1f}%)")
print(f"Failures n>=1900: {sum(1 for n in fail_lens if n>=1900)} ({100*sum(1 for n in fail_lens if n>=1900)/len(fail_lens):.1f}%)")
