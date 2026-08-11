"""Validate the Python replica (orb_model) against the C env, obs-only.

Checks:
  P1  coast/warp propagation matches the env to <1 m for 200 steps
  P2  every burn action's predicted post-burn elements match the env
  P3  phase-drift sign: chaser below target drifts AHEAD (Delta-lambda grows)
  P4  fuel bookkeeping (obs[6]) matches Tsiolkovsky
"""
import os
import sys
import math
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'pufferlib'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pufferlib.ocean.orbital.orbital import Orbital  # noqa: E402
import t3_expert_model as om                         # noqa: E402


def make_env(**kw):
    base = dict(num_envs=1, num_debris_min=0, num_debris_max=0,
                e_max_target=0.05, e_max_sat=0.05,
                init_phase_gap_max=math.pi, valid_init_only=1,
                gave_up_action="terminate")
    base.update(kw)
    return Orbital(**base)


def pos_err(el_pred, el_true):
    xp = om.orbit_to_cartesian(el_pred)
    xt = om.orbit_to_cartesian(el_true)
    return math.hypot(xp[0] - xt[0], xp[1] - xt[1]), math.hypot(xp[2] - xt[2], xp[3] - xt[3])


def main():
    env = make_env()
    obs, _ = env.reset(seed=7)
    obs = obs[0]

    # ── P1: propagation ──────────────────────────────────────────────────────
    sat, tgt, fuel = om.decode_obs(obs)
    worst_s = worst_t = 0.0
    acts = [0, 9, 10, 11] * 30
    tot_steps = 0
    for a in acts:
        tau = om.ACTION_TAU[a]
        sat = om.propagate(sat, tau * om.DT)
        tgt = om.propagate(tgt, tau * om.DT)
        obs, rew, term, trunc, _ = env.step(np.array([a], dtype=np.int32))
        obs = obs[0]
        tot_steps += tau
        if term[0]:
            break
        s2, t2, _ = om.decode_obs(obs)
        worst_s = max(worst_s, pos_err(sat, s2)[0])
        worst_t = max(worst_t, pos_err(tgt, t2)[0])
        sat, tgt = s2, t2   # resync (float32 obs floor)
    print(f"P1 coast/warp: {tot_steps} sim steps, max chaser pos err {worst_s:.3f} m, "
          f"target {worst_t:.3f} m")

    # ── P2: burns ────────────────────────────────────────────────────────────
    env2 = make_env()
    obs, _ = env2.reset(seed=11)
    obs = obs[0]
    worst = 0.0
    worst_v = 0.0
    rng = np.random.default_rng(0)
    burn_acts = [1, 2, 3, 4, 5, 6, 7, 8, 12, 13, 14, 15]
    for i in range(120):
        a = int(burn_acts[i % len(burn_acts)])
        sat, tgt, fuel = om.decode_obs(obs)
        dvp, dvr = om.ACTION_DV[a]
        pred = om.apply_impulse(sat, dvp, dvr)
        pred = om.propagate(pred, om.DT)
        fuel_pred = om.fuel_after(fuel, math.hypot(dvp, dvr))
        obs, rew, term, trunc, _ = env2.step(np.array([a], dtype=np.int32))
        obs = obs[0]
        if term[0]:
            obs, _ = env2.reset(seed=11 + i)
            obs = obs[0]
            continue
        s2, _, f2 = om.decode_obs(obs)
        pe, ve = pos_err(pred, s2)
        worst = max(worst, pe)
        worst_v = max(worst_v, ve)
        if abs(f2 - fuel_pred) > 2e-6:
            print(f"  fuel mismatch a={a}: {f2:.8f} vs {fuel_pred:.8f}")
        # random coast to vary the true anomaly
        for _ in range(int(rng.integers(1, 20))):
            obs, rew, term, trunc, _ = env2.step(np.array([0], dtype=np.int32))
            obs = obs[0]
            if term[0]:
                break
        if term[0]:
            obs, _ = env2.reset(seed=911 + i)
            obs = obs[0]
    print(f"P2 burns: max post-burn+propagate pos err {worst:.3f} m, vel err {worst_v:.5f} m/s")

    # ── P3: phase-drift sign ────────────────────────────────────────────────
    env3 = make_env(same_orbit_init=0, e_max_target=0.0, e_max_sat=0.0,
                    init_phase_gap_max=0.2)
    obs, _ = env3.reset(seed=3)
    obs = obs[0]
    sat, tgt, _ = om.decode_obs(obs)
    lam0 = om.wrap_pi((sat['omega'] + sat['M']) - (tgt['omega'] + tgt['M']))
    a_s, a_t = sat['a'], tgt['a']
    for _ in range(100):
        obs, _, term, _, _ = env3.step(np.array([11], dtype=np.int32))  # 1 hr warp
        obs = obs[0]
        if term[0]:
            break
    sat, tgt, _ = om.decode_obs(obs)
    lam1 = om.wrap_pi((sat['omega'] + sat['M']) - (tgt['omega'] + tgt['M']))
    print(f"P3 drift: a_sat-a_tgt = {(a_s-a_t)/1e3:+.1f} km, "
          f"d(Delta-lambda) = {math.degrees(om.wrap_pi(lam1-lam0)):+.2f} deg "
          f"(expect sign opposite to a_sat-a_tgt)")

    # ── P4: dv budget ────────────────────────────────────────────────────────
    print(f"P4 total dv budget from full tank: {om.dv_remaining(0.15):.2f} m/s")
    env.close(); env2.close(); env3.close()


if __name__ == "__main__":
    main()
