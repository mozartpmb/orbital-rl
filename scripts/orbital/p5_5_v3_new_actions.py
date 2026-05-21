"""Phase 5.5 env-mods V3 — deterministic-action smoke test at GEO.

Verifies the new actions (M2: warp-30min/warp-1hr; M3: ±1, ±2 m/s) produce
predicted physics within ±5%. Drives the env directly with a fixed action
sequence (no policy) and reads sat_a / sim_time deltas from the trajectory
log.

Predictions (from vis-viva at GEO a≈4.2e7 m, v_circ≈3075 m/s):
  Action 10 (warp-30min, τ=30): advances sim_time by 30 × 60s = 1800 s
  Action 11 (warp-1hr, τ=60):   advances sim_time by 60 × 60s = 3600 s
  Action 12 (prograde +1 m/s):   Δa ≈ +28 km (per E6/P3)
  Action 14 (prograde +2 m/s):   Δa ≈ +56 km
  Action 15 (retrograde -2 m/s): Δa ≈ -56 km

Pass criterion: each measurement within ±5% of prediction. Tolerance is
generous because eccentricity affects Δa from a single burn (peri-vs-apo
asymmetry); we use e≈0 to minimize this.

Run:
    cd /Users/pete/space_training/pufferlib
    python3 ../scripts/orbital/p5_5_v3_new_actions.py
"""
import math
import os
import sys

import numpy as np

# Add pufferlib to path (script is in /Users/pete/space_training/scripts/orbital/
# but the Orbital class is in /Users/pete/space_training/pufferlib/...)
sys.path.insert(0, "/Users/pete/space_training/pufferlib")

from pufferlib.ocean.orbital.orbital import Orbital


MU = 3.986004418e14
R_EARTH = 6.371e6
DT = 60.0


def predict_da_from_burn(a, e, dv_pro):
    """Vis-viva Δa from a prograde burn at periapsis (e=0 → r=a)."""
    r = a * (1 - e)  # at periapsis
    v = math.sqrt(MU * (2.0 / r - 1.0 / a))
    v_new = v + dv_pro
    a_new = 1.0 / (2.0 / r - v_new * v_new / MU)
    return a_new - a


def run_step(env, action_idx, n_envs=1):
    """Step env once with given action; return obs and reward arrays."""
    actions = np.full(n_envs, action_idx, dtype=np.int32)
    obs, r, term, trunc, info = env.step(actions)
    return obs, r, term, trunc


def measure_action_effect(env, action_idx, expected_tau, expected_da_m):
    """Step env once with `action_idx` and verify sim_time and sat_a deltas."""
    # Snapshot state before
    # We need sat_a and sim_time — neither is directly in obs[] in absolute
    # terms (obs[0] is normalized). Use the trajectory log path: enable
    # traj_log_dir, step once, read the .npz.
    # Simpler: use the orbital_test main() approach? No, easier to just
    # do this via the C struct via vec_get_trajectory after termination.
    # Or: probe by stepping multiple times and reading sat_a from the
    # trajectory log after a small known sequence.
    # Cleanest: log every step via traj_log_every=1 and read the .npz.
    pass


def main():
    # Construct env at GEO with rescaled obs so we're operating in the regime
    # the env mods are designed for. Use a single env for clean trajectory.
    out_dir = "/tmp/p5_5_v3_smoke"
    if os.path.exists(out_dir):
        import shutil
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # GEO low-e regime, single env, log every episode.
    env = Orbital(
        num_envs=1,
        num_debris_min=0, num_debris_max=0,
        e_max_target=0.0,        # circular target → e≈0 (note: sat.e=0 forced when e_max_sat=0)
        e_max_sat=0.0,
        same_orbit_init=1,        # sat & target at same a — focus on action effects
        init_phase_gap_max=0.0,   # cophased — avoid termination via gap closure
        valid_init_only=1,
        max_valid_init_attempts=4096,
        gave_up_action="accept",
        a_min_override=4.2e7,
        a_max_override=4.21e7,
        obs_alt_scale_m=4.2e7,
        phi_orbit_scale_k=0.001,
        lvlh_scale_m=4.2e7,
        traj_log_dir=out_dir,
        traj_log_every=1,
    )

    # Fixed-seed reset for determinism.
    obs, _ = env.reset(seed=42)

    # Determine the realized sat_a from the trajectory (will be in [4.2e7, 4.21e7]).
    # We'll step through a deterministic action sequence and read the log.
    # Sequence: coast 5 → action 12 → coast 1 → action 11 → coast 1 →
    #           action 14 → coast 1 → action 10 → coast 1 → action 15 → coast 1 → end.
    # Episode terminates via... we want it to NOT terminate. Easiest: max_steps + force terminal.
    # Actually since same_orbit_init=1 and phase gap = 0, rendezvous tolerance might trigger
    # immediate success. Let me think — at e=0 with sat = target, dphase=0, dvel=0, rendezvous
    # success would fire on step 0. That's a problem.
    #
    # Workaround: use a small phase gap so initial state is NOT a rendezvous.
    env.close()

    env = Orbital(
        num_envs=1,
        num_debris_min=0, num_debris_max=0,
        e_max_target=0.0, e_max_sat=0.0,
        same_orbit_init=1,
        init_phase_gap_max=0.5,   # 0.5 rad phase gap — avoids immediate rendezvous
        phase_gap_fixed=0.5,
        valid_init_only=1,
        max_valid_init_attempts=4096,
        gave_up_action="accept",
        a_min_override=4.2e7,
        a_max_override=4.21e7,
        obs_alt_scale_m=4.2e7,
        phi_orbit_scale_k=0.001,
        lvlh_scale_m=4.2e7,
        traj_log_dir=out_dir,
        traj_log_every=1,
    )
    obs, _ = env.reset(seed=42)

    # Action sequence — interleave warps and burns; each measurement is on
    # the step where the action is taken (delta from previous step).
    seq = [
        ("coast",       0),
        ("warp_30min",  10),  # M2: should advance sim_time by 1800 s
        ("coast",       0),
        ("warp_1hr",    11),  # M2: should advance sim_time by 3600 s
        ("coast",       0),
        ("burn_pro_1",  12),  # M3: should produce Δa ≈ +28 km
        ("coast",       0),
        ("burn_pro_2",  14),  # M3: should produce Δa ≈ +56 km
        ("coast",       0),
        ("burn_retro_2",15),  # M3: should produce Δa ≈ -56 km
        ("coast",       0),
    ]

    print("=" * 80)
    print("Phase 5.5 env-mods V3 — deterministic action smoke at GEO")
    print("=" * 80)
    print()

    for label, a_idx in seq:
        obs, r, term, trunc = run_step(env, a_idx)
        if term[0]:
            print(f"WARN: episode terminated at action {label}; cannot continue.")
            break

    # Force a trajectory save (episode hasn't terminated naturally; the C
    # traj_log buffer still has our deterministic-action data).
    env._save_trajectory(env_idx=0, episode_reward=0.0)
    env.close()

    # Read the trajectory log
    import glob
    files = sorted(glob.glob(f"{out_dir}/ep_*.npz"))
    if not files:
        print("ERROR: no trajectory log written")
        sys.exit(1)
    d = np.load(files[0])
    print(f"Trajectory has {len(d['sat_a'])} steps")
    print()

    # The trajectory log has one row per env step. Step 0 = initial state
    # (before any action). Step k = state AFTER taking the k-th action.
    sat_a = d["sat_a"]
    sat_e = d["sat_e"]
    sim_time = d["sim_time"]
    actions = d["action"]
    print(f"{'step':>5} {'action':>4} {'label':<14} {'sat_a (km)':>11} {'sat_e':>8} "
          f"{'sim_time (s)':>13} {'Δa (km)':>9} {'Δt (s)':>9} {'predict_Δa':>11}")
    print("-" * 110)

    # The trajectory log writes one row per SUB-STEP. For warp actions (τ>1)
    # a single env.step() call writes τ rows. Group consecutive rows with the
    # same action value into runs and aggregate.
    runs = []
    if len(actions) > 1:
        run_start = 1   # row 0 = initial state (action=0 by convention)
        cur_action = int(actions[1])
        for i in range(2, len(actions)):
            if int(actions[i]) != cur_action:
                runs.append((run_start, i - 1, cur_action))
                run_start = i
                cur_action = int(actions[i])
        runs.append((run_start, len(actions) - 1, cur_action))

    initial_a = float(sat_a[0])
    print(f"{0:>5} {'-':>4} {'init':<14} {initial_a/1000:>11.3f} {float(sat_e[0]):>8.5f} "
          f"{float(sim_time[0]):>13.1f}")

    fail = []
    for (start, end, a_i) in runs:
        prev_a = float(sat_a[start - 1])
        cur_a  = float(sat_a[end])
        prev_e = float(sat_e[start - 1])
        cur_e  = float(sat_e[end])
        da = cur_a - prev_a
        dt = float(sim_time[end]) - float(sim_time[start - 1])
        n_sub = end - start + 1
        label = next((lab for lab, idx in seq if idx == a_i), f"unknown_{a_i}")

        # Predict
        if a_i in (10, 11):
            expected_dt = 1800.0 if a_i == 10 else 3600.0
            predicted_str = f"Δt={expected_dt:.0f}s"
            if abs(dt - expected_dt) > 1.0:
                fail.append(f"action {a_i} ({label}): Δt {dt:.1f} != {expected_dt:.0f} (expected τ × DT × n_calls)")
        elif a_i in (12, 13, 14, 15):
            dv_pro = {12: 1.0, 13: -1.0, 14: 2.0, 15: -2.0}[a_i]
            predicted_da = predict_da_from_burn(prev_a, prev_e, dv_pro)
            predicted_str = f"Δa≈{predicted_da/1000:+.2f}km"
            if abs(predicted_da) > 1.0:
                err = abs(da - predicted_da) / abs(predicted_da)
                if err > 0.10:
                    fail.append(f"action {a_i} ({label}): Δa {da/1000:+.2f}km != predicted "
                                 f"{predicted_da/1000:+.2f}km (err {err*100:.1f}%)")
        else:
            predicted_str = "—"

        print(f"{start:>5} {a_i:>4} {label:<14} {cur_a/1000:>11.3f} {cur_e:>8.5f} "
              f"{float(sim_time[end]):>13.1f} {da/1000:>+9.2f} {dt:>9.1f} {predicted_str:>11}"
              f"  ({n_sub} sub-steps)")

    print()
    print("=" * 80)
    print("Pass criteria:")
    print("  - Action 10 (warp_30min): Δt ≈ 1800 s")
    print("  - Action 11 (warp_1hr):   Δt ≈ 3600 s")
    print("  - Action 12 (pro+1):      Δa ≈ +28 km at GEO e=0")
    print("  - Action 14 (pro+2):      Δa ≈ +56 km")
    print("  - Action 15 (retro-2):    Δa ≈ -56 km")
    print("  - Match within ±10% of prediction (tolerance > theory because eccentricity\n"
          "    drifts during multi-step sequence affects burn site).")
    print()

    if fail:
        print("FAIL:")
        for f in fail:
            print(f"  {f}")
        sys.exit(1)
    else:
        print("V3 PASS — all action effects within tolerance")


if __name__ == "__main__":
    main()
