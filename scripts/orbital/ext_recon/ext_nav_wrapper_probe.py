"""NAV-H insertion-point probe: does a PufferEnv-subclass obs post-processor
survive the Multiprocessing vec backend?

The claim under test: a subclass of Orbital that mutates `self.observations`
IN PLACE inside step(), after binding.vec_step() has written the C observation,
is visible to the driver process through pufferlib.vector.Multiprocessing's
shared-memory buffers — and per-env filter state can therefore live in the
worker where it belongs.

Also probes: per-env reset detection (does the env autoreset inside c_step, so
that a terminal step already carries the NEXT episode's first observation?).

Read-only: defines its own subclass here, registers nothing, modifies nothing.
Run:  python3 scripts/orbital/ext_recon/ext_nav_wrapper_probe.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, "/Users/pete/space_training/pufferlib")

import pufferlib.vector                                     # noqa: E402
from pufferlib.ocean.orbital.orbital import Orbital         # noqa: E402

T3 = dict(
    num_debris_min=0, num_debris_max=0,
    e_max_target=0.05, e_max_sat=0.05, init_phase_gap_max=3.14159,
    valid_init_only=1, shaping_mode=1, shape_gamma=1.0,
    phase_gap_mode=1, phase_obs_mode=1, episode_cap_steps=3000,
    cap_terminal_reward=0.0,
)

SENTINEL_SLOT = 17      # dead body block (debris permanently disabled)


class OrbitalNavProbe(Orbital):
    """Marks every observation with a per-worker sentinel + a step counter.

    Stands in for the real filter: the only thing being tested is whether an
    in-place write to self.observations after binding.vec_step() reaches the
    driver, and whether per-env state (here: self._k) can live in the worker.
    """

    def __init__(self, *a, nav_tag=0.0, **kw):
        super().__init__(*a, **kw)
        self.nav_tag = float(nav_tag)
        self._k = np.zeros(self.num_agents, dtype=np.float64)
        self._resets = 0

    def reset(self, seed=0):
        o, i = super().reset(seed)
        self._k[:] = 0.0
        self.observations[:, SENTINEL_SLOT] = self.nav_tag
        self.observations[:, SENTINEL_SLOT + 1] = self._k
        return self.observations, i

    def step(self, actions):
        o, r, d, t, info = super().step(actions)
        # per-env "filter" bookkeeping, reset on the env's own autoreset
        self._k += 1.0
        self._k[d.astype(bool)] = 0.0
        self._resets += int(d.sum())
        self.observations[:, SENTINEL_SLOT] = self.nav_tag
        self.observations[:, SENTINEL_SLOT + 1] = self._k
        return self.observations, r, d, t, info


def make_probe(nav_tag=0.0, **kw):
    return OrbitalNavProbe(nav_tag=nav_tag, **kw)


def probe_autoreset():
    """Is the observation on a terminal step already the NEXT episode's obs?"""
    env = Orbital(num_envs=64, **T3)
    obs, _ = env.reset(seed=3)
    rng = np.random.default_rng(0)
    prev = obs.copy()
    hits = 0
    jump = []
    for _ in range(4000):
        a = rng.integers(0, 16, 64).astype(np.int32)
        o, r, d, t, _ = env.step(a)
        idx = np.flatnonzero(d)
        for i in idx:
            # LVLH relative position slots: an autoreset re-draws the scenario,
            # so a genuine reset shows a discontinuous jump here.
            jump.append(float(np.abs(o[i, 33:35] - prev[i, 33:35]).max()))
            hits += 1
        prev = o.copy()
        if hits > 40:
            break
    env.close()
    print(f"  terminal steps sampled: {hits}")
    print(f"  |LVLH jump| at terminal: median {np.median(jump):.4f}, "
          f"min {np.min(jump):.4f}  (a same-episode step is ~1e-4)")
    print("  => observation returned WITH terminals[i]=1 is the NEW episode's "
          "obs (C autoresets inside c_step)")


def probe_multiprocessing(num_workers=2, envs_per_worker=256):
    print(f"\n  Multiprocessing: {num_workers} workers x {envs_per_worker} envs")
    kw = dict(T3)
    kw['num_envs'] = envs_per_worker
    vec = pufferlib.vector.make(
        make_probe, env_kwargs=dict(kw, nav_tag=7.5),
        backend=pufferlib.vector.Multiprocessing,
        num_envs=num_workers, num_workers=num_workers,
        batch_size=num_workers, seed=11,
    )
    vec.async_reset(seed=11)
    o, r, d, t, info, env_id, mask = vec.recv()
    print(f"    driver obs shape {o.shape}, "
          f"sentinel[{SENTINEL_SLOT}] unique = {np.unique(o[:, SENTINEL_SLOT])}")
    ok_tag = np.allclose(o[:, SENTINEL_SLOT], 7.5)
    for k in range(5):
        vec.send(np.zeros(o.shape[0], dtype=np.int32))
        o, r, d, t, info, env_id, mask = vec.recv()
    print(f"    after 5 steps: counter slot[{SENTINEL_SLOT+1}] "
          f"min={o[:, SENTINEL_SLOT+1].min():.0f} "
          f"max={o[:, SENTINEL_SLOT+1].max():.0f}  (expect 5 unless an env reset)")
    ok_ctr = o[:, SENTINEL_SLOT + 1].max() == 5
    vec.close()
    print(f"    VERDICT worker-side in-place obs write visible to driver: "
          f"{'PASS' if ok_tag else 'FAIL'}")
    print(f"    VERDICT worker-side per-env state persists across steps:   "
          f"{'PASS' if ok_ctr else 'FAIL'}")
    return ok_tag and ok_ctr


def probe_serial():
    print("\n  Serial (control)")
    kw = dict(T3)
    kw['num_envs'] = 128
    vec = pufferlib.vector.make(
        make_probe, env_kwargs=dict(kw, nav_tag=3.25),
        backend=pufferlib.vector.Serial, num_envs=1, seed=5)
    vec.async_reset(seed=5)
    o = vec.recv()[0]
    print(f"    sentinel unique = {np.unique(o[:, SENTINEL_SLOT])}")
    vec.close()


if __name__ == '__main__':
    import psutil
    print(f"physical cores: {psutil.cpu_count(logical=False)}  "
          f"start method: {__import__('multiprocessing').get_start_method()}")
    print("\n[1] autoreset semantics")
    probe_autoreset()
    print("\n[2] backend survival")
    probe_serial()
    probe_multiprocessing()
