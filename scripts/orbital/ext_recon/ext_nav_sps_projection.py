"""NAV-H SPS projection from measured constants. Writes ext_nav_sps_projection.csv.

Measured on this machine (Apple silicon, 14 physical cores, numpy 1.26.4, OMP=1):
  env, B=1024, per-decision wall  = 0.070 ms + 0.348 ms x tau
      (fit: tau=1 -> 0.419 ms, tau=5 -> 1.830 ms, tau=60 -> 20.92 ms)
  batched EKF sensor tick, B=1024 = 1.353 ms   (4-iter Newton + FD STM + matmul Joseph)
  obs decode + re-encode, B=1024  = 0.263 ms   (once per decision)
  T3 canonical policy mean tau    = 32.50 sub-steps/decision (51% warp-1h)
  trainer profile at the L1 operating point: Env 33%, Forward 19%, Learn 27%,
      wall SPS 139.4K agent-decisions/s (2 workers x 1024 envs)
"""
import csv

ENV_FIX_MS = 0.070
ENV_SUBSTEP_MS = 0.348
TICK_MS = 1.353
DECODE_MS = 0.263
MEAN_TAU = 32.50
SPS0 = 139_400.0
ENV_FRAC = 0.33
OUT = "/Users/pete/space_training/web_data/results/ext_nav_sps_projection.csv"

# mean ticks/decision under each nav-cadence policy, measured from the canonical
# policy's action histogram (ext_nav_tau_hist.py)
CADENCES = [
    ("perdec (1 tick/decision)", 1.00, "50.5% closed-loop — REJECTED by T3 eval"),
    ("cap 4 ticks/decision", 2.68, "worst gap 15 min at tau=60; unvalidated"),
    ("cap 6 ticks/decision", 3.80, "worst gap 10 min at tau=60; unvalidated"),
    ("nav300 (300 s cadence)", 6.50, "100.0% @1x, 99.5% @10x — VALIDATED"),
    ("cap 12 ticks/decision", 7.15, "== nav300 at tau=60, finer at tau<60"),
    ("nav60 (60 s cadence)", 32.50, "100.0% @1x, 96.0% @30x — VALIDATED, headline"),
]


def main():
    env_ms = ENV_FIX_MS + ENV_SUBSTEP_MS * MEAN_TAU
    rows = []
    for name, ticks, note in CADENCES:
        nav_ms = DECODE_MS + ticks * TICK_MS
        rho = nav_ms / env_ms
        f2 = ENV_FRAC * (1 + rho) + (1 - ENV_FRAC)            # 2 workers, as shipped
        f4 = ENV_FRAC * (1 + rho * 1.07) / 2 + (1 - ENV_FRAC)  # 4 workers x 512 envs
        rows.append(dict(
            cadence=name, ticks_per_decision=f"{ticks:.2f}",
            env_ms_per_decision=f"{env_ms:.2f}", nav_ms_per_decision=f"{nav_ms:.2f}",
            nav_over_env=f"{rho:.2f}",
            sps_2workers=f"{SPS0/f2:.0f}", slowdown_2w=f"{f2:.2f}",
            sps_4workers_512=f"{SPS0/f4:.0f}", slowdown_4w=f"{f4:.2f}",
            wall_50M_min_2w=f"{50e6/(SPS0/f2)/60:.1f}",
            wall_50M_min_4w=f"{50e6/(SPS0/f4)/60:.1f}",
            note=note))
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    hdr = f"{'cadence':26s} {'tick/dec':>8s} {'nav/env':>8s} {'SPS 2w':>9s} " \
          f"{'SPS 4w':>9s} {'50M min 2w':>11s} {'50M min 4w':>11s}"
    print(f"env per decision (B=1024, mean tau {MEAN_TAU}): {env_ms:.2f} ms")
    print(hdr)
    for r in rows:
        print(f"{r['cadence']:26s} {r['ticks_per_decision']:>8s} "
              f"{r['nav_over_env']:>8s} {r['sps_2workers']:>9s} "
              f"{r['sps_4workers_512']:>9s} {r['wall_50M_min_2w']:>11s} "
              f"{r['wall_50M_min_4w']:>11s}   {r['note']}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
