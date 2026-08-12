#!/usr/bin/env python3
"""ext-3d wide lineage: joint (Delta-v + horizon) feasibility screen at the EXACT
W1..W4 configs, with a sampler that replicates orbital.h c_reset.

Extends scripts/orbital/ext_recon/ext_3d_joint_feasibility.py in three ways:
  1. the init sampler is the env's, not an idealisation: a_target drawn inside
     the da_max window INTERSECTED with the band, the e-vector disc, the exact
     Rodrigues plane rotation, and the valid_init_only perigee rejection loop;
  2. two e-vector semantics are screened side by side --
       intended : |d e_vec| <= de_max, the knob's advertised meaning
       realized : the env applies the de_max disc to the NODE-RELATIVE 2-vector
                  (e cos w, e sin w) and then rotates the plane, which sets
                  RAAN ~ U(0,2pi) and therefore randomises the INERTIAL periapsis
                  longitude varpi = RAAN + w.  The realised inertial |d e_vec| is
                  then governed by e_max, not by de_max.
  3. di_max is solved for an ABSOLUTE >=97% combined-cost feasibility target.

Cost model, budget and phasing/horizon logic are unchanged from the red-team
screen so the numbers are comparable.

Writes /tmp/w3d_screen_sweep.csv and /tmp/w3d_screen_dimax.csv
"""
import csv
import math
import sys

import numpy as np

MU, R_EARTH, DT = 3.986004418e14, 6.371e6, 60.0
ISP, G0 = 300.0, 9.80665
EARTH_KEEPOUT = R_EARTH + 200e3
BUDGET = ISP * G0 * math.log(1.0 / (1.0 - 0.15))     # 478.13 m/s


def hoh(a1, a2):
    at = 0.5 * (a1 + a2)
    dv1 = abs(math.sqrt(MU / a1) * (math.sqrt(2 * a2 / (a1 + a2)) - 1.0))
    dv2 = abs(math.sqrt(MU / a2) * (1.0 - math.sqrt(2 * a1 / (a1 + a2))))
    return dv1 + dv2, math.pi * math.sqrt(at ** 3 / MU), max(dv1, dv2)


def n_of(a):
    return math.sqrt(MU / a ** 3)


def v_at(a, e, nu):
    r = a * (1 - e * e) / (1 + e * math.cos(nu))
    return math.sqrt(MU * (2.0 / r - 1.0 / a))


def draw_init(rng, cfg):
    """Replicate orbital.h c_reset for the (a, e, w, plane) block."""
    a_lo, a_hi = cfg["a_lo"], cfg["a_hi"]
    for _ in range(4096):                      # max_valid_init_attempts
        a_s = rng.uniform(a_lo, a_hi)
        if cfg["same_orbit"]:
            a_t = a_s
        else:
            lo, hi = a_lo, a_hi
            if cfg["da_max_m"] > 0:
                da = max(cfg["da_max_m"], 200e3)
                l2, h2 = max(a_lo, a_s - da), min(a_hi, a_s + da)
                if h2 - l2 >= 150e3:
                    lo, hi = l2, h2
            while True:
                a_t = rng.uniform(lo, hi)
                if abs(a_t - a_s) >= 50e3:
                    break
        e_t = rng.uniform(0.0, cfg["e_max_t"]) if cfg["e_max_t"] > 0 else 0.0
        w_t = rng.uniform(0, 2 * math.pi) if e_t > 0 else 0.0
        if cfg["same_orbit"]:
            e_s, w_s = e_t, w_t
        elif cfg["de_max"] >= 0.0:
            r_de = cfg["de_max"] * math.sqrt(rng.uniform(0, 1))
            ph = rng.uniform(0, 2 * math.pi)
            esx = e_t * math.cos(w_t) + r_de * math.cos(ph)
            esy = e_t * math.sin(w_t) + r_de * math.sin(ph)
            e_s = math.hypot(esx, esy)
            w_s = math.atan2(esy, esx) if e_s > 1e-9 else 0.0
        elif cfg["e_max_s"] > 0:
            e_s = rng.uniform(0.0, cfg["e_max_s"])
            w_s = rng.uniform(0, 2 * math.pi)
        else:
            e_s, w_s = 0.0, 0.0
        # plane: delta = di_max*sqrt(U), node phase uniform => RAAN ~ U(0,2pi)
        if cfg["di_max"] > 0 and not cfg["same_orbit"]:
            delta = cfg["di_max"] * math.sqrt(rng.uniform(0, 1))
            raan = rng.uniform(0, 2 * math.pi)
        else:
            delta, raan = 0.0, 0.0
        if cfg["valid_init_only"]:
            if a_s * (1 - e_s) < EARTH_KEEPOUT or a_t * (1 - e_t) < EARTH_KEEPOUT:
                continue
        return a_s, a_t, e_s, e_t, w_s, w_t, delta, raan
    return a_s, a_t, e_s, e_t, w_s, w_t, delta, raan


def evec3(e, inc, raan, argp):
    co, so = math.cos(raan), math.sin(raan)
    cw, sw = math.cos(argp), math.sin(argp)
    ci, si = math.cos(inc), math.sin(inc)
    return (e * (co * cw - so * sw * ci),
            e * (so * cw + co * sw * ci),
            e * (sw * si))


def screen(cfg, di_max_deg, n=20000, seed=4242, evec_mode="realized"):
    rng = np.random.default_rng(seed)
    c = dict(cfg)
    c["di_max"] = math.radians(di_max_deg)
    horizon = c["horizon"]
    ok_seq = ok_comb = ok_t = both_s = both_c = 0
    dvs, dvc, dpl, des, ths, sts = [], [], [], [], [], []
    for _ in range(n):
        a_s, a_t, e_s, e_t, w_s, w_t, delta, raan = draw_init(rng, c)
        inc = delta                                  # |plane angle| = delta
        if evec_mode == "realized":
            ev_s = evec3(e_s, inc, raan, w_s)        # varpi = raan + w_s
            ev_t = evec3(e_t, 0.0, 0.0, w_t)
            de = math.sqrt(sum((p - q) ** 2 for p, q in zip(ev_s, ev_t)))
            varpi_s = raan + w_s
        else:                                        # "intended": knob honoured
            de = math.hypot(e_s * math.cos(w_s) - e_t * math.cos(w_t),
                            e_s * math.sin(w_s) - e_t * math.sin(w_t))
            varpi_s = w_s
        vc = math.sqrt(MU / (0.5 * (a_s + a_t)))
        dv_e = vc * de / 2.0
        dv_a, th, dv_big_a = hoh(a_s, a_t)

        # in-plane phasing: free drift, else phasing-orbit rescue (screen model)
        d0 = rng.uniform(-math.pi, math.pi)
        w = n_of(a_s) - n_of(a_t)
        surcharge = 0.0
        if abs(w) < 1e-14:
            tot_s = math.inf
        else:
            num = (d0 + n_of(a_t) * th - math.pi) % (2 * math.pi)
            td = num / w if w >= 0 else (num - 2 * math.pi) / w
            tot_s = (td + th) / DT
        if tot_s > horizon:
            best, best_t = math.inf, math.inf
            span = max(600.0, c["da_max_m"] / 1e3 if c["da_max_m"] > 0 else 600.0)
            for off in np.arange(-span, span + 20.0, span / 24.0) * 1e3:
                ap = a_t + off
                if ap < R_EARTH + 250e3 or abs(ap - a_t) < 25e3:
                    continue
                dvA, thA, _ = hoh(a_s, ap)
                dvB, thB, _ = hoh(ap, a_t)
                wp = n_of(ap) - n_of(a_t)
                if abs(wp) < 1e-13:
                    continue
                numB = (d0 + n_of(a_t) * (thA + thB) - 2 * math.pi) % (2 * math.pi)
                tdr = numB / wp if wp >= 0 else (numB - 2 * math.pi) / wp
                t2 = (thA + tdr + thB) / DT
                if t2 <= horizon and (dvA + dvB - dv_a) < best:
                    best, best_t = dvA + dvB - dv_a, t2
            if math.isfinite(best):
                surcharge, tot_s = best, best_t
            else:
                tot_s = math.inf
        dv_in = dv_a + dv_e + surcharge
        dv_big = max(dv_big_a, dv_e)

        if c["di_max"] > 0:
            u_node = raan                       # node line in the target plane
            cand = []
            for du in (0.0, math.pi):
                cand.append(v_at(a_s, e_s, u_node + du - varpi_s))
                cand.append(v_at(a_t, e_t, u_node + du - w_t))
            v_node = min(cand)
            dv_pl = 2.0 * v_node * math.sin(0.5 * inc)
            P = 2 * math.pi * math.sqrt(max(a_s, a_t) ** 3 / MU)
            if tot_s * DT < 0.5 * P:
                tot_s += rng.uniform(0, 0.5 * P) / DT
        else:
            dv_pl = 0.0

        s = dv_in + dv_pl
        k = (dv_in - dv_big) + math.hypot(dv_big, dv_pl)
        dvs.append(s); dvc.append(k); dpl.append(dv_pl); des.append(de)
        ths.append(math.degrees(inc)); sts.append(tot_s)
        a1, a2, t_ok = s <= BUDGET, k <= BUDGET, tot_s <= horizon
        ok_seq += a1; ok_comb += a2; ok_t += t_ok
        both_s += (a1 and t_ok); both_c += (a2 and t_ok)

    dvs, dvc, des = np.array(dvs), np.array(dvc), np.array(des)
    st = np.array(sts); fin = st[np.isfinite(st)]
    return dict(rung=c["name"], evec_mode=evec_mode, di_max_deg=di_max_deg, n=n,
                frac_dv_ok_seq=ok_seq / n, frac_dv_ok_comb=ok_comb / n,
                frac_time_ok=ok_t / n,
                frac_feasible_seq=both_s / n, frac_feasible_comb=both_c / n,
                de_p50=float(np.percentile(des, 50)),
                de_p90=float(np.percentile(des, 90)),
                dv_e_p50=float(np.percentile(des, 50)) * 0,   # filled below
                dv_seq_p50=float(np.percentile(dvs, 50)),
                dv_seq_p90=float(np.percentile(dvs, 90)),
                dv_comb_p50=float(np.percentile(dvc, 50)),
                dv_comb_p90=float(np.percentile(dvc, 90)),
                dv_plane_p90=float(np.percentile(np.array(dpl), 90)),
                theta_p50=float(np.percentile(np.array(ths), 50)),
                steps_p90=float(np.percentile(fin, 90)) if len(fin) else math.nan)


RUNGS = [
    dict(name="W1 500-800 e=0 same-orbit", a_lo=6.871e6, a_hi=7.171e6,
         e_max_t=0.0, e_max_s=0.0, de_max=-1.0, da_max_m=-1.0, horizon=3000,
         same_orbit=True, valid_init_only=1),
    dict(name="W2 300-800 e<=0.05", a_lo=6.671e6, a_hi=7.171e6,
         e_max_t=0.05, e_max_s=0.05, de_max=-1.0, da_max_m=-1.0, horizon=3000,
         same_orbit=False, valid_init_only=1),
    dict(name="X3 300-800 e<=0.05 (LEO 3D, SOLVED)", a_lo=6.671e6, a_hi=7.171e6,
         e_max_t=0.05, e_max_s=0.05, de_max=-1.0, da_max_m=-1.0, horizon=3000,
         same_orbit=False, valid_init_only=1),
    dict(name="W3 300-2000 e<=0.15 de.06 da400", a_lo=6.671e6, a_hi=8.371e6,
         e_max_t=0.15, e_max_s=0.0, de_max=0.06, da_max_m=400e3, horizon=3000,
         same_orbit=False, valid_init_only=1),
    dict(name="W4 300-8000 e<=0.30 de.08 da600", a_lo=6.671e6, a_hi=14.371e6,
         e_max_t=0.30, e_max_s=0.0, de_max=0.08, da_max_m=600e3, horizon=6000,
         same_orbit=False, valid_init_only=1),
]
DI_SWEEP = [0.0, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.75, 0.9, 1.0, 1.25, 1.5, 2.0]


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
    rows = []
    for cfg in RUNGS:
        for mode in ("realized", "intended"):
            if cfg["de_max"] < 0 and mode == "intended":
                continue        # no de knob -> the two modes coincide
            for di in DI_SWEEP:
                r = screen(cfg, di, n=n, evec_mode=mode)
                rows.append(r)
                print(f"{r['rung'][:34]:34} {mode:9} di={di:4.2f}  "
                      f"FEAS seq {100*r['frac_feasible_seq']:5.1f}%  "
                      f"comb {100*r['frac_feasible_comb']:5.1f}%  "
                      f"dv_comb p50 {r['dv_comb_p50']:5.0f} p90 {r['dv_comb_p90']:5.0f}  "
                      f"|de| p50 {r['de_p50']:.3f}  dvpl_p90 {r['dv_plane_p90']:5.1f}")
    with open("/tmp/w3d_screen_sweep.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    # ── solve di_max at absolute >=97% combined feasibility ──
    print("\n=== di_max for ABSOLUTE >=97% combined-cost joint feasibility ===")
    solved = []
    for cfg in RUNGS:
        for mode in ("realized", "intended"):
            if cfg["de_max"] < 0 and mode == "intended":
                continue
            band = [r for r in rows if r["rung"] == cfg["name"]
                    and r["evec_mode"] == mode]
            base = band[0]["frac_feasible_comb"]
            best = None
            for r in sorted(band, key=lambda x: x["di_max_deg"]):
                if r["frac_feasible_comb"] >= 0.97:
                    best = r["di_max_deg"]
            # also the -1pp-relative criterion the red team used
            rel = 0.0
            for r in sorted(band, key=lambda x: x["di_max_deg"]):
                if r["frac_feasible_comb"] >= base - 0.01:
                    rel = r["di_max_deg"]
            solved.append(dict(rung=cfg["name"], evec_mode=mode,
                               feas_di0_comb=base,
                               di_max_97abs=(best if best is not None else -1),
                               di_max_lose1pp_comb=rel))
            print(f"  {cfg['name'][:34]:34} {mode:9}  di=0 comb {100*base:5.1f}%  "
                  f"di_max@97%abs = "
                  f"{('NONE (ceiling below 97%% at di=0)' if best is None else f'{best:.2f} deg')}"
                  f"   di_max@-1pp = {rel:.2f} deg")
    with open("/tmp/w3d_screen_dimax.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(solved[0].keys()))
        w.writeheader(); w.writerows(solved)
    print("\nwrote /tmp/w3d_screen_sweep.csv /tmp/w3d_screen_dimax.csv")


if __name__ == "__main__":
    main()
