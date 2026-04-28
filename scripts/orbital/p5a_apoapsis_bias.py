"""Phase 5a Investigation D: apoapsis-bias analysis.

Loads the eval npz files from a Phase 4 ckpt run on eccentric targets, finds
the rendezvous step in each successful episode, and histograms the target
true anomaly at that moment. If the distribution concentrates near θ = π,
the agent has learned to preferentially rendezvous at apoapsis.

Usage:
  python p5a_apoapsis_bias.py LOG_DIR [--out PNG_PATH]
"""
import argparse
import glob
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("log_dir", help="dir of ep_*.npz files from eval run")
    ap.add_argument("--out", default="plots/p5a_D_apoapsis_hist.png")
    ap.add_argument("--show-failures", action="store_true",
                    help="also print stats on failed episodes (closest approach)")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.log_dir, "ep_*.npz")))
    print(f"Found {len(files)} episodes in {args.log_dir}")

    success_anomalies = []
    fail_min_dists = []
    n_success = 0
    n_fail = 0

    for f in files:
        d = np.load(f)
        ep_reward = float(d["episode_reward"][0])
        rewards = d["reward"]
        # Compute target true anomaly from cartesian: θ = atan2(y, x) - ω (perifocal frame)
        tx = d["target_x"]
        ty = d["target_y"]
        tom = d["target_omega"]

        if ep_reward > 0:
            # Success — last step with reward > 1.0 is the terminal success step
            # (shaping rewards are << 1).
            success_steps = np.where(rewards > 1.0)[0]
            if len(success_steps) == 0:
                success_step = len(rewards) - 1
            else:
                success_step = int(success_steps[-1])
            inertial_angle = np.arctan2(ty[success_step], tx[success_step])
            theta = float(inertial_angle - tom[success_step]) % (2 * np.pi)
            success_anomalies.append(theta)
            n_success += 1
        else:
            n_fail += 1
            if args.show_failures:
                # Use min_conj_dist as proxy for closest approach
                if "min_conj_dist" in d:
                    fail_min_dists.append(float(d["min_conj_dist"].min()))

    print(f"\nSuccess rate: {n_success}/{n_success + n_fail} ({100*n_success/(n_success+n_fail):.1f}%)")

    if not success_anomalies:
        print("No successful episodes — apoapsis-bias check is untestable with this ckpt at this difficulty.")
        return

    arr = np.array(success_anomalies)
    print(f"\nRendezvous true anomaly stats over {len(arr)} successes:")
    print(f"  mean={np.mean(arr):.3f} rad ({np.degrees(np.mean(arr)):.1f}°)")
    print(f"  std ={np.std(arr):.3f} rad ({np.degrees(np.std(arr)):.1f}°)")
    print(f"  median={np.median(arr):.3f} rad ({np.degrees(np.median(arr)):.1f}°)")

    # Bias windows around apoapsis (π) and periapsis (0 / 2π)
    near_apo = np.sum(np.abs(arr - np.pi) < np.pi / 4)
    near_peri = np.sum(np.minimum(arr, 2*np.pi - arr) < np.pi / 4)
    print(f"\n  near apoapsis (π ± π/4):  {near_apo}/{len(arr)} ({100*near_apo/len(arr):.1f}%)")
    print(f"  near periapsis (0 ± π/4): {near_peri}/{len(arr)} ({100*near_peri/len(arr):.1f}%)")
    uniform_expected = len(arr) * (np.pi / 2) / (2 * np.pi)  # ±π/4 = π/2 width over 2π circle
    print(f"  uniform-prior expectation: ~{uniform_expected:.1f} per window")

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(arr, bins=24, range=(0, 2 * np.pi), edgecolor="black")
        ax.axvline(np.pi, color="red", linestyle="--", label="apoapsis (π)")
        ax.axvline(0, color="orange", linestyle="--", label="periapsis (0)")
        ax.axvline(2 * np.pi, color="orange", linestyle="--")
        ax.set_xlabel("Target true anomaly at rendezvous (rad)")
        ax.set_ylabel("Successful episodes")
        ax.set_title(f"Apoapsis-bias check ({len(arr)} successes / {n_success + n_fail} eps)")
        ax.legend()
        plt.tight_layout()
        plt.savefig(args.out, dpi=120)
        print(f"\nSaved histogram → {args.out}")
    except Exception as e:
        print(f"Plot skipped: {e}")


if __name__ == "__main__":
    main()
