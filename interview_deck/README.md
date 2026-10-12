# Interview deck — Learning Orbital Rendezvous

Beamer deck for a 30-minute interview (about 10 minutes of talk, the rest Q&A).
12 core slides, appendix A1–A9 for deep dives. Speaker notes are in `\note{}` on every core slide.

Two versions:

- `deck.tex` — full version, 12 core slides + A1–A9.
- `deck_short.tex` — 7 core slides + 8 appendix slides; steers toward dynamics, policy, results, verification. Navigation, Lambert, and the C port each appear as one line and one appendix slide.
- `defensive_briefing.md` — plain-language 30-second answers, likely follow-ups, and redirects for the EKF, batch IOD, Lambert comparator, and C port.

## Build

```bash
brew install tectonic          # once
cd interview_deck
tectonic deck.tex              # -> deck.pdf
```

Notes-only handout: add `\setbeameroption{show only notes}` after `\usepackage{pgfplots}` and rebuild.

## Regenerate figures

All results figures are generated from post-fix data (2026-08-10 `true_to_mean` fix onward).

```bash
# 1. headline 2-D trajectories (eval seed 123 by default; the deck used a fresh run)
cd pufferlib && python3 scripts/orbital/eval_checkpoint.py ../models/t3/seed42_L2_headline.pt \
  --episodes 200 --seed 123 --out-dir /tmp/t3_headline \
  --e-max-target 0.05 --e-max-sat 0.05 --same-orbit-init 0 --init-phase-gap-max 3.14159 \
  --shaping-mode 1 --shape-gamma 1.0 --phase-gap-mode 1 --phase-obs-mode 1 \
  --episode-cap-steps 3000 --cap-terminal-reward 0.0 --valid-init-only 1 \
  --rendezvous-radius-m 30000 --rel-vel-tol-ms 50
# 2. 2-D figures: trajectory, overlay, action histogram, dv-ratio, Lambert porkchop
python3 interview_deck/scripts/make_figs.py /tmp/t3_headline interview_deck/figs
# 3. generalist / DAgger / seesaw / noise sweep / C-port figures (from tabulated results)
python3 interview_deck/scripts/make_gen_figs.py interview_deck/figs
# 4. 3-D generalist rollout figures (T15e checkpoint, cell E1_j2, bearings-only)
python3 interview_deck/scripts/make_3d_figs.py
```

`figs/nees.png`, `figs/nis.png`, `figs/err_3sigma_ep1.png` are copies of `plots/relnav_t3/` (post-fix, canonical checkpoint).

## Headline numbers used (edit the `\newcommand` block at the top of `deck.tex`)

| macro | value | source |
|---|---|---|
| 2-D headline | 200/200 on 5/5 seeds | `T3_RECOVERY_CAMPAIGN.md` §7 |
| Lambert time-matched ratio | 1.18× median (200 eps) | `web_data/results/fuel_audit.csv` |
| generalist cells | 97 / 99 / 99.5 / 96 / 99 / 86 / 74 % | `web_data/results/t15e_dagger/fin_*.json` |
| NEES / NIS in-bounds | 94.4 / 95.1 % (validate) | `plots/relnav_t3` |
| C port p99 | 62 µs (M3 Max, not WCET) | `scripts/orbital/embedded/results/bench.txt` |
| step rate | ~740 k steps/s env-only | `PROGRESS.md:184` |
