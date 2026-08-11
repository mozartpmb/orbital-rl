#!/usr/bin/env python3
"""
T3 recon: analytic feasibility envelope for 2D coplanar rendezvous under
CORRECTED dynamics (post-f55d9cb, true_to_mean no longer inverted).

Everything here is closed-form orbital mechanics + Monte Carlo over the
headline init distribution. No training, no checkpoints.

Outputs (CSV) -> /Users/pete/space_training/web_data/results/
    t3_phasing_table.csv        drift-orbit phasing cost/time/decisions
    t3_discount_table.csv       10*gamma^N visibility of the terminal reward
    t3_headline_feasibility.csv MC over LEO 300-800 km headline init dist
    t3_ecc_ladder.csv           altitude band -> e_max, period, horizon needs
    t3_velocity_quanta.csv      action-set rel-velocity nulling resolution

Run:  python3 scripts/orbital/t3/feasibility_tables.py
"""

import csv
import math
import os
import sys

import numpy as np

# ── Constants mirrored from pufferlib/ocean/orbital/orbital.h ───────────────
MU = 3.986004418e14
R_EARTH = 6.371e6
ALT_MIN = 200e3               # EARTH_KEEPOUT = R_EARTH + ALT_MIN
DT = 60.0
MAX_STEPS = 2000              # *** sim-step cap, NOT agent-decision cap ***
FUEL_FRAC = 0.15
ISP, G0 = 300.0, 9.80665
VE = ISP * G0
DV_BUDGET = VE * math.log(1.0 / (1.0 - FUEL_FRAC))   # 478.1 m/s

RENDEZVOUS_RADIUS = 30_000.0
REL_VEL_TOL = 50.0
TIGHT_RADIUS = 5_000.0
TIGHT_VEL = 1.0

GAMMA_TRAIN = 0.995           # orbital.ini [train] gamma
TERMINAL_R = 10.0             # success reward at full fuel (10*(0.5+0.5*f))

# Action-set warp inventories (tau values > 1), largest first.
WARPSETS = {
    "coast-only (tau=1)":        [],
    "legacy D10 (warp-5min)":    [5],
    "D16 (warp 5/30min)":        [30, 5],
    "D16 (warp 5/30/60min)":     [60, 30, 5],
}

OUT = "/Users/pete/space_training/web_data/results"


# ── Core orbital relations ─────────────────────────────────────────────────
def n_of(a):
    """Mean motion (rad/s)."""
    return math.sqrt(MU / a**3)


def period(a):
    return 2.0 * math.pi / n_of(a)


def v_circ(a):
    return math.sqrt(MU / a)


def drift_rate(a, da, exact=True):
    """Relative along-track angular rate (rad/s) for a chaser offset by da.

    Linearized: dn/da = -1.5 n/a  =>  |omega| = 1.5 * (|da|/a) * n(a).
    Exact:      |omega| = |n(a) - n(a+da)|  (linearization over-predicts by
                ~1% at da=50 km and ~6% at da=340 km for a=6771 km).
    """
    if exact:
        return abs(n_of(a) - n_of(a + abs(da)))
    return 1.5 * (abs(da) / a) * n_of(a)


def dv_raise(a, da):
    """Two-impulse (Hohmann) dv to change semi-major axis by da, small-da limit.
    Exact-ish: dv = v_c * |da| / (2a).  Round trip (open + close) = 2x this."""
    return v_circ(a) * abs(da) / (2.0 * a)


def dv_hohmann(a1, a2):
    """Exact two-impulse Hohmann total dv between circular a1 and a2."""
    at = 0.5 * (a1 + a2)
    dv1 = abs(math.sqrt(MU / a1) * (math.sqrt(2 * a2 / (a1 + a2)) - 1.0))
    dv2 = abs(math.sqrt(MU / a2) * (1.0 - math.sqrt(2 * a1 / (a1 + a2))))
    return dv1 + dv2, math.pi * math.sqrt(at**3 / MU)


def warp_decisions(sim_steps, taus):
    """Agent decisions to execute `sim_steps` of pure coast using a greedy
    largest-warp-that-does-not-overshoot mix, falling back to tau=1 coast.
    NOTE: sim_steps consumed is identical for every warp set — warps buy
    DECISIONS, not TIME."""
    s = int(sim_steps)
    d = 0
    for t in sorted(taus, reverse=True):
        d += s // t
        s %= t
    return d + s


def burn_decisions(dv, quanta=25.0):
    """Decisions to deliver a dv magnitude using the largest burn quantum."""
    return int(math.ceil(dv / quanta)) if dv > 0 else 0


# ══════════════════════════════════════════════════════════════════════════
# 1. Phasing time / steps / decisions / dv table
# ══════════════════════════════════════════════════════════════════════════
def table_phasing(a=6.771e6):
    rows = []
    T = period(a)
    for gap_deg in (30.0, 90.0, 180.0):
        gap = math.radians(gap_deg)
        for da_km in (50.0, 100.0, 200.0, 340.0):
            da = da_km * 1e3
            w = drift_rate(a, da)          # exact |n(a) - n(a+da)|
            w_lin = drift_rate(a, da, exact=False)
            t_drift = gap / w
            s_drift = t_drift / DT
            n_orb = t_drift / T
            dv_1way = dv_raise(a, da)
            dv_rt = 2.0 * dv_1way
            # burn decisions: open + close the drift orbit, 25 m/s quantum
            burns = 2 * burn_decisions(dv_1way, 25.0)
            # transfer legs: half a drift-orbit period each way to circularize
            s_legs = 2 * (0.5 * period(a + da) / DT)
            s_total = s_drift + s_legs
            for name, taus in WARPSETS.items():
                dec = warp_decisions(round(s_drift), taus) + burns \
                    + warp_decisions(round(s_legs), taus)
                rows.append(dict(
                    gap_deg=gap_deg, da_km=da_km,
                    drift_rate_deg_per_hr=math.degrees(w) * 3600.0,
                    drift_rate_deg_per_hr_linearized=math.degrees(w_lin) * 3600.0,
                    drift_time_hr=t_drift / 3600.0,
                    drift_orbits=n_orb,
                    sim_steps_drift=round(s_drift),
                    sim_steps_total=round(s_total),
                    warpset=name,
                    agent_decisions=dec,
                    dv_open_close_ms=dv_rt,
                    dv_frac_budget=dv_rt / DV_BUDGET,
                    alongtrack_km_per_simstep=a * w * DT / 1e3,
                    FLAG_over_simstep_cap=int(s_total > MAX_STEPS),
                    FLAG_over_dv_budget=int(dv_rt > DV_BUDGET),
                    FLAG_decisions_over_2000=int(dec > 2000),
                ))
    return rows


# ══════════════════════════════════════════════════════════════════════════
# 2. Discount visibility
# ══════════════════════════════════════════════════════════════════════════
def table_discount(phasing_rows):
    rows = []
    for g in (0.995, 0.997, 0.999):
        for r in phasing_rows:
            N = r["agent_decisions"]
            S = r["sim_steps_total"]
            rows.append(dict(
                gamma=g, gap_deg=r["gap_deg"], da_km=r["da_km"],
                warpset=r["warpset"],
                N_decisions=N, S_simsteps=S,
                terminal_value_per_decision=TERMINAL_R * g**N,
                frac_per_decision=g**N,
                terminal_value_semiMDP=TERMINAL_R * g**S,
                frac_semiMDP=g**S,
                visible_10pct=int(g**N >= 0.10),
                visible_1pct=int(g**N >= 0.01),
            ))
    return rows


def discount_horizons():
    rows = []
    for g in (0.995, 0.997, 0.999):
        rows.append(dict(
            gamma=g,
            N_at_50pct=math.log(0.50) / math.log(g),
            N_at_10pct=math.log(0.10) / math.log(g),
            N_at_1pct=math.log(0.01) / math.log(g),
            effective_horizon_1_over_1mg=1.0 / (1.0 - g),
        ))
    return rows


# ══════════════════════════════════════════════════════════════════════════
# 3. Headline-distribution feasibility (Monte Carlo, zero-extra-dv strategy)
# ══════════════════════════════════════════════════════════════════════════
def mc_headline(alt_lo_km=300.0, alt_hi_km=800.0, n=200_000, seed=42,
                horizon_steps=MAX_STEPS, label="LEO 300-800"):
    """Optimal-window strategy: chaser LOITERS on its initial orbit (whose
    natural da w.r.t. the target IS the phasing drift, free of charge), then
    executes a single Hohmann transfer that arrives phase-matched.

    Departure condition (circular, coplanar):
        theta_s0 + n_s*td + pi  ==  theta_t0 + n_t*(td + th)   (mod 2pi)
    =>  td = wrap( (theta_t0-theta_s0) + n_t*th - pi ) / (n_s - n_t)
    dv is exactly the unavoidable Hohmann cost -> zero phasing surcharge.
    """
    rng = np.random.default_rng(seed)
    a_s = R_EARTH + rng.uniform(alt_lo_km, alt_hi_km, n) * 1e3
    a_t = R_EARTH + rng.uniform(alt_lo_km, alt_hi_km, n) * 1e3
    dth0 = rng.uniform(-math.pi, math.pi, n)          # theta_t0 - theta_s0

    ns, nt = np.sqrt(MU / a_s**3), np.sqrt(MU / a_t**3)
    at = 0.5 * (a_s + a_t)
    th = math.pi * np.sqrt(at**3 / MU)                # transfer duration
    dv = np.abs(np.sqrt(MU / a_s) * (np.sqrt(2 * a_t / (a_s + a_t)) - 1.0)) + \
         np.abs(np.sqrt(MU / a_t) * (1.0 - np.sqrt(2 * a_s / (a_s + a_t))))

    w = ns - nt                                        # relative drift rate
    num = np.mod(dth0 + nt * th - math.pi, 2 * math.pi)
    # drift must run in the direction sign(w); take smallest non-negative td
    td = np.where(w >= 0, num / np.maximum(w, 1e-30),
                  (num - 2 * math.pi) / np.minimum(w, -1e-30))
    td = np.where(np.isfinite(td) & (td >= 0), td, np.inf)

    total_s = (td + th) / DT
    ok_time = total_s <= horizon_steps
    ok_dv = dv <= DV_BUDGET
    ok = ok_time & ok_dv

    out = dict(band=label, horizon_steps=horizon_steps, n=n,
               dv_mean=float(dv.mean()), dv_p95=float(np.percentile(dv, 95)),
               dv_frac_budget_p95=float(np.percentile(dv, 95) / DV_BUDGET),
               feasible_frac=float(ok.mean()),
               feasible_frac_time_only=float(ok_time.mean()),
               feasible_frac_dv_only=float(ok_dv.mean()))
    fin = total_s[np.isfinite(total_s)]
    for p in (10, 25, 50, 75, 90, 95, 99):
        out[f"total_simsteps_p{p}"] = float(np.percentile(fin, p))
    # decisions for the median mission under each warp set
    for name, taus in WARPSETS.items():
        med = float(np.percentile(fin, 50))
        p90 = float(np.percentile(fin, 90))
        out[f"dec_p50[{name}]"] = warp_decisions(round(med), taus)
        out[f"dec_p90[{name}]"] = warp_decisions(round(p90), taus)
    # horizon needed to make 90 / 99 % of the distribution feasible
    out["simsteps_for_90pct"] = float(np.percentile(fin, 90))
    out["simsteps_for_99pct"] = float(np.percentile(fin, 99))
    return out, total_s, dv


def mc_dv_surcharge(alt_lo_km=300.0, alt_hi_km=800.0, n=20_000, seed=7,
                    horizon_steps=MAX_STEPS):
    """For the tail that does NOT fit the horizon by loitering, how much extra
    dv buys a phasing orbit that does fit?  Scan a phasing radius a_p."""
    rng = np.random.default_rng(seed)
    a_s = R_EARTH + rng.uniform(alt_lo_km, alt_hi_km, n) * 1e3
    a_t = R_EARTH + rng.uniform(alt_lo_km, alt_hi_km, n) * 1e3
    dth0 = rng.uniform(-math.pi, math.pi, n)
    surcharge = np.full(n, np.nan)
    # candidate phasing offsets relative to the *target* orbit
    offsets = np.concatenate([np.arange(-400, -20, 20.0), np.arange(20, 420, 20.0)]) * 1e3
    for i in range(n):
        a1, a2, d0 = a_s[i], a_t[i], dth0[i]
        n1, n2 = math.sqrt(MU / a1**3), math.sqrt(MU / a2**3)
        th = math.pi * math.sqrt((0.5 * (a1 + a2))**3 / MU)
        w = n1 - n2
        num = (d0 + n2 * th - math.pi) % (2 * math.pi)
        td = num / w if w >= 0 else (num - 2 * math.pi) / w
        if (td + th) / DT <= horizon_steps:
            surcharge[i] = 0.0
            continue
        best = np.inf
        for off in offsets:
            ap = a2 + off
            if ap < R_EARTH + 250e3:
                continue
            dvA, thA = dv_hohmann(a1, ap)
            dvB, thB = dv_hohmann(ap, a2)
            npp = math.sqrt(MU / ap**3)
            wp = npp - n2
            if abs(wp) < 1e-12:
                continue
            # phase after leg A, then drift on ap, then leg B arriving matched
            numB = (d0 + n2 * (thA + thB) - 2 * math.pi) % (2 * math.pi)
            tdrift = numB / wp if wp >= 0 else (numB - 2 * math.pi) / wp
            tot = (thA + tdrift + thB) / DT
            base, _ = dv_hohmann(a1, a2)
            if tot <= horizon_steps and (dvA + dvB) <= DV_BUDGET:
                best = min(best, dvA + dvB - base)
        surcharge[i] = best
    fin = surcharge[np.isfinite(surcharge)]
    return dict(
        n=n,
        frac_zero_surcharge=float((surcharge == 0).mean()),
        frac_rescued_by_phasing_orbit=float(((surcharge > 0) & np.isfinite(surcharge)).mean()),
        frac_infeasible_any=float((~np.isfinite(surcharge)).mean()),
        surcharge_mean_ms=float(fin[fin > 0].mean()) if (fin > 0).any() else 0.0,
        surcharge_p90_ms=float(np.percentile(fin[fin > 0], 90)) if (fin > 0).any() else 0.0,
    )


# ══════════════════════════════════════════════════════════════════════════
# 4. Eccentricity / altitude ladder
# ══════════════════════════════════════════════════════════════════════════
def table_ecc_ladder():
    """Geometric e ceiling for an altitude band, and the horizon/warp/gamma
    consequences of the resulting orbital period."""
    bands = [
        ("LEO-narrow",   300.0,   800.0),
        ("LEO-wide",     300.0,  2000.0),
        ("LEO/MEO",      300.0,  8000.0),
        ("MEO",          300.0, 20200.0),
        ("MEO/GEO",      300.0, 35786.0),
    ]
    rows = []
    for name, h1, h2 in bands:
        a_lo, a_hi = R_EARTH + h1 * 1e3, R_EARTH + h2 * 1e3
        r_peri_min = R_EARTH + ALT_MIN
        # e ceiling if a is sampled at the top of the band and perigee must
        # clear the 200 km keep-out
        e_max_top = 1.0 - r_peri_min / a_hi
        # e ceiling that holds for EVERY a in the band (worst case a = a_lo)
        e_max_all = 1.0 - r_peri_min / a_lo
        # apogee-in-band ceiling (both r_p and r_a inside [a_lo, a_hi])
        e_max_inband = (a_hi - a_lo) / (a_hi + a_lo)
        T_lo, T_hi = period(a_lo), period(a_hi)
        # phasing at the top of the band: f = da/a affordable by the dv budget,
        # reserving 40% of budget for the orbit transfer + terminal trim
        vc = v_circ(a_hi)
        f_afford = 0.60 * DV_BUDGET / vc          # dv_roundtrip = vc * f
        # orbits to close 180 deg:  N_orb = dtheta / (3*pi*f)
        n_orb_180 = math.pi / (3.0 * math.pi * f_afford)
        t_180 = n_orb_180 * T_hi
        steps_180 = t_180 / DT
        rows.append(dict(
            band=name, alt_lo_km=h1, alt_hi_km=h2,
            a_lo_km=a_lo / 1e3, a_hi_km=a_hi / 1e3,
            e_max_perigee_at_a_hi=e_max_top,
            e_max_valid_for_whole_band=e_max_all,
            e_max_both_apsides_in_band=e_max_inband,
            period_lo_min=T_lo / 60.0, period_hi_min=T_hi / 60.0,
            simsteps_per_orbit_hi=T_hi / DT,
            orbits_in_2000_steps=MAX_STEPS * DT / T_hi,
            v_circ_hi_ms=vc,
            f_affordable_da_over_a=f_afford,
            da_affordable_km=f_afford * a_hi / 1e3,
            orbits_to_phase_180=n_orb_180,
            simsteps_to_phase_180=steps_180,
            required_MAX_STEPS=math.ceil(steps_180 * 1.6 / 500.0) * 500,
            dec_180_warp5=warp_decisions(round(steps_180), [5]),
            dec_180_warp60=warp_decisions(round(steps_180), [60, 30, 5]),
            FLAG_needs_bigger_warp=int(warp_decisions(round(steps_180), [60, 30, 5]) > 460),
        ))
    return rows


# ══════════════════════════════════════════════════════════════════════════
# 5. Success-box quanta
# ══════════════════════════════════════════════════════════════════════════
def table_quanta(a=6.771e6):
    """Finest single-burn rel-velocity nulling. With quantum q on an axis, a
    greedy null leaves |residual| <= q/2 on that axis."""
    sets = {
        "legacy D10": dict(tangential=[5, 10, 25], radial=[10]),
        "D16":        dict(tangential=[1, 2, 5, 10, 25], radial=[10]),
        "D16 + 1 m/s radial (proposed)": dict(tangential=[1, 2, 5, 10, 25], radial=[1, 10]),
    }
    rows = []
    for name, s in sets.items():
        qt, qr = min(s["tangential"]), min(s["radial"])
        res_t, res_r = qt / 2.0, qr / 2.0
        res = math.hypot(res_t, res_r)
        # semi-major-axis resolution implied by the finest tangential burn
        da_q = 2.0 * a * qt / v_circ(a)
        # residual da allowed by the 50 m/s box: dv_tan = vc*da/(2a)
        da_allow_50 = 2.0 * a * REL_VEL_TOL / v_circ(a)
        da_allow_1 = 2.0 * a * TIGHT_VEL / v_circ(a)
        rows.append(dict(
            action_set=name,
            min_tangential_ms=qt, min_radial_ms=qr,
            best_residual_tangential_ms=res_t,
            best_residual_radial_ms=res_r,
            best_residual_relvel_ms=res,
            da_quantum_km=da_q / 1e3,
            meets_50ms_box=int(res < REL_VEL_TOL),
            meets_1ms_tight_box=int(res < TIGHT_VEL),
            box50_margin_x=REL_VEL_TOL / res,
            box1_margin_x=TIGHT_VEL / res,
            da_allowed_by_50ms_box_km=da_allow_50 / 1e3,
            da_allowed_by_1ms_box_km=da_allow_1 / 1e3,
            alongtrack_closure_km_per_step_at_50ms=a * drift_rate(a, da_allow_50) * DT / 1e3,
        ))
    return rows


# ══════════════════════════════════════════════════════════════════════════
# 6. Shaping-gate geometry:  where is sigma2 (the phase-reward gate) open?
# ══════════════════════════════════════════════════════════════════════════
W_ORBIT, W_PHASE, W_VEL = 0.01, 0.01, 0.01
BETA_SHAPE = 1.0
EPS_ORBIT, TAU_ORBIT = 2.0, 0.2
EPS_PHASE, TAU_PHASE = 0.3, 0.03
SUCCESS_TOL_A = 10_000.0


def sigma2_of_da(da, tol=SUCCESS_TOL_A):
    phi_orbit = abs(da) / tol
    return 1.0 / (1.0 + math.exp(-(EPS_ORBIT - phi_orbit) / TAU_ORBIT))


def table_gate(a=6.771e6):
    rows = []
    for da_km in (0, 5, 10, 15, 20, 25, 29, 30, 40, 50, 88, 100, 200, 340, 500):
        da = da_km * 1e3
        s2 = sigma2_of_da(da)
        phi = -(W_ORBIT * da / SUCCESS_TOL_A)          # e=0, phase term x sigma2
        rows.append(dict(
            da_km=da_km, phi_orbit=da / SUCCESS_TOL_A, sigma2=s2,
            phase_reward_scale=s2,
            Phi_orbit_only=phi,
            rel_vel_implied_ms=v_circ(a) * da / (2.0 * a),
            drift_deg_per_hr=math.degrees(drift_rate(a, da)) * 3600.0 if da else 0.0,
            note=("gate OPEN" if s2 > 0.5 else
                  "gate closed" if s2 > 1e-3 else "gate DEAD (<1e-3)"),
        ))
    return rows


def table_shaping_penalty(a=6.771e6):
    """Two shaping pathologies of the current Phi under CORRECT physics.

    (A) drift-orbit entry penalty: raising |da| to `da` costs
        Phi = -W_ORBIT*da/TOL immediately; the refund arrives N decisions
        later discounted by gamma^N, so the net cost is
        -W_ORBIT*(da/TOL)*(1 - gamma^N).
    (B) warp/coast reward leak: with Phi<0 constant, each decision pays
        beta*(gamma^tau - 1)*Phi = +beta*|Phi|*(1-gamma^tau) > 0.
        Summed over an episode this is the do-nothing floor bonus.
    """
    rows = []
    for da_km in (100, 200, 340):
        da = da_km * 1e3
        phi_mag = W_ORBIT * da / SUCCESS_TOL_A
        gap = math.pi
        s_drift = gap / drift_rate(a, da) / DT
        for g in (0.995, 0.997, 0.999):
            for name, taus in WARPSETS.items():
                N = warp_decisions(round(s_drift), taus) \
                    + 2 * burn_decisions(dv_raise(a, da), 25.0)
                entry_net = -phi_mag * (1.0 - g**N)
                # leak: sum over the episode of beta*|Phi|*(1-g^tau)
                tau_eff = max(1, round(s_drift / max(N, 1)))
                n_dec_leak = round(s_drift / tau_eff)
                leak = BETA_SHAPE * phi_mag * (1.0 - g**tau_eff) * n_dec_leak
                rows.append(dict(
                    da_km=da_km, gamma=g, warpset=name,
                    N_decisions=N, gamma_pow_N=g**N,
                    Phi_magnitude=phi_mag,
                    drift_entry_net_cost=entry_net,
                    drift_entry_cost_pct_of_terminal=100 * abs(entry_net) / TERMINAL_R,
                    coast_leak_over_drift=leak,
                    leak_pct_of_terminal=100 * leak / TERMINAL_R,
                ))
    return rows


def table_donothing_floor(a=6.771e6):
    """Full-episode shaping leak for a frozen do-nothing policy that times out
    at MAX_STEPS, for a few |Phi| scales. Compare to the -10 timeout."""
    rows = []
    for da_km in (50, 167, 500, 1000, 5000):
        phi_mag = W_ORBIT * (da_km * 1e3) / SUCCESS_TOL_A
        for g in (0.995, 0.997, 0.999):
            for name, taus in WARPSETS.items():
                tau = max(taus) if taus else 1
                n_dec = MAX_STEPS // tau
                leak = BETA_SHAPE * phi_mag * (1.0 - g**tau) * n_dec
                rows.append(dict(
                    da_km=da_km, gamma=g, warpset=name, tau_used=tau,
                    decisions_to_cap=n_dec, Phi_magnitude=phi_mag,
                    undiscounted_leak=leak,
                    timeout_return=-10.0 + leak,
                    FLAG_leak_flips_timeout_positive=int(leak > 10.0),
                ))
    return rows


# ══════════════════════════════════════════════════════════════════════════
# Oracle cross-check of the drift-rate formula
# ══════════════════════════════════════════════════════════════════════════
def oracle_check():
    sys.path.insert(0, "/Users/pete/space_training/scripts/orbital/nav")
    try:
        import orbital_math as om
    except Exception as e:                                  # pragma: no cover
        return {"status": f"oracle import failed: {e}"}
    a = 6.771e6
    out = []
    for da_km in (50.0, 200.0, 340.0):
        da = da_km * 1e3
        el_t = dict(a=a, e=0.0, theta=0.0, omega=0.0)
        el_s = dict(a=a + da, e=0.0, theta=0.0, omega=0.0)
        st = np.array(om.orbit_to_cartesian(el_t))
        ss = np.array(om.orbit_to_cartesian(el_s))
        T = 3600.0 * 6
        st2 = om.propagate_cartesian(st, T)
        ss2 = om.propagate_cartesian(ss, T)
        ang = lambda s: math.atan2(s[1], s[0])
        d = (ang(ss2) - ang(st2)) - (ang(ss) - ang(st))
        d = (d + math.pi) % (2 * math.pi) - math.pi
        meas = abs(d) / T
        pred = drift_rate(a, da)
        out.append(dict(da_km=da_km, measured_rad_s=meas, predicted_rad_s=pred,
                        rel_err=abs(meas - pred) / pred))
    return out


# ══════════════════════════════════════════════════════════════════════════
def write_csv(path, rows):
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote {path}  ({len(rows)} rows)")


def md(rows, cols, fmt=None, title=""):
    fmt = fmt or {}
    print(f"\n### {title}")
    print("| " + " | ".join(cols) + " |")
    print("|" + "|".join(["---"] * len(cols)) + "|")
    for r in rows:
        cells = []
        for c in cols:
            v = r[c]
            cells.append(fmt.get(c, "{}").format(v) if not isinstance(v, float)
                         else fmt.get(c, "{:.3g}").format(v))
        print("| " + " | ".join(cells) + " |")


def main():
    print(f"dv budget (Tsiolkovsky, 15% fuel, Isp 300s) = {DV_BUDGET:.1f} m/s")
    print(f"MAX_STEPS={MAX_STEPS} sim steps = {MAX_STEPS*DT/3600:.2f} h "
          f"= {MAX_STEPS*DT/period(6.771e6):.1f} LEO orbits (a=6771 km)")
    print(f"LEO a=6771 km: T={period(6.771e6)/60:.2f} min, "
          f"v_c={v_circ(6.771e6):.1f} m/s, n={n_of(6.771e6):.6e} rad/s")

    ph = table_phasing()
    write_csv(f"{OUT}/t3_phasing_table.csv", ph)

    disc = table_discount(ph)
    write_csv(f"{OUT}/t3_discount_table.csv", disc)
    dh = discount_horizons()
    write_csv(f"{OUT}/t3_discount_horizons.csv", dh)

    hl_rows = []
    for hz in (2000, 4000, 6000):
        r, _, _ = mc_headline(horizon_steps=hz)
        hl_rows.append(r)
    for band, lo, hi in (("LEO 300-2000", 300.0, 2000.0),
                         ("LEO 300-8000", 300.0, 8000.0)):
        r, _, _ = mc_headline(alt_lo_km=lo, alt_hi_km=hi, horizon_steps=2000,
                              label=band)
        hl_rows.append(r)
        r, _, _ = mc_headline(alt_lo_km=lo, alt_hi_km=hi, horizon_steps=8000,
                              label=band)
        hl_rows.append(r)
    write_csv(f"{OUT}/t3_headline_feasibility.csv", hl_rows)

    sur = mc_dv_surcharge()
    write_csv(f"{OUT}/t3_dv_surcharge.csv", [sur])
    print("\ndv surcharge for the horizon-infeasible tail:", sur)

    ec = table_ecc_ladder()
    write_csv(f"{OUT}/t3_ecc_ladder.csv", ec)

    q = table_quanta()
    write_csv(f"{OUT}/t3_velocity_quanta.csv", q)

    gate = table_gate()
    write_csv(f"{OUT}/t3_shaping_gate.csv", gate)
    shp = table_shaping_penalty()
    write_csv(f"{OUT}/t3_shaping_penalty.csv", shp)
    dnf = table_donothing_floor()
    write_csv(f"{OUT}/t3_donothing_floor.csv", dnf)

    oc = oracle_check()
    print("\noracle drift-rate cross-check:", oc)
    if isinstance(oc, list):
        write_csv(f"{OUT}/t3_oracle_driftcheck.csv", oc)

    # ── printed markdown ───────────────────────────────────────────────────
    md([r for r in ph if r["warpset"] == "coast-only (tau=1)"],
       ["gap_deg", "da_km", "drift_rate_deg_per_hr", "drift_time_hr",
        "sim_steps_total", "dv_open_close_ms", "dv_frac_budget",
        "FLAG_over_simstep_cap", "FLAG_over_dv_budget"],
       title="Phasing: time / sim-steps / dv (warp-independent)")

    for ws in WARPSETS:
        md([r for r in ph if r["warpset"] == ws],
           ["gap_deg", "da_km", "sim_steps_total", "agent_decisions"],
           title=f"Decisions — {ws}")

    md(dh, ["gamma", "N_at_50pct", "N_at_10pct", "N_at_1pct",
            "effective_horizon_1_over_1mg"],
       title="Discount horizons (decisions until 10*gamma^N drops below X)")

    md(hl_rows, ["band", "horizon_steps", "feasible_frac", "total_simsteps_p50",
                 "total_simsteps_p90", "total_simsteps_p99", "dv_mean",
                 "dv_frac_budget_p95"],
       title="Headline MC: loiter-then-Hohmann (zero phasing surcharge)")

    md(ec, ["band", "e_max_perigee_at_a_hi", "e_max_both_apsides_in_band",
            "period_hi_min", "simsteps_per_orbit_hi", "orbits_in_2000_steps",
            "da_affordable_km", "simsteps_to_phase_180", "required_MAX_STEPS",
            "dec_180_warp5", "dec_180_warp60"],
       title="Eccentricity / altitude ladder")

    md(gate, ["da_km", "phi_orbit", "sigma2", "rel_vel_implied_ms",
              "drift_deg_per_hr", "note"],
       title="sigma2 phase-reward gate vs semi-major-axis offset")

    md([r for r in shp if r["gamma"] == 0.995 and r["da_km"] == 200],
       ["warpset", "N_decisions", "gamma_pow_N", "drift_entry_net_cost",
        "drift_entry_cost_pct_of_terminal", "coast_leak_over_drift"],
       title="Drift-orbit shaping penalty, 180deg / da=200 km / gamma=0.995")

    md([r for r in dnf if r["gamma"] == 0.995
        and r["warpset"] == "D16 (warp 5/30/60min)"],
       ["da_km", "Phi_magnitude", "decisions_to_cap", "undiscounted_leak",
        "timeout_return", "FLAG_leak_flips_timeout_positive"],
       title="Do-nothing shaping leak at cap (gamma=0.995, warp-1hr)")

    md(q, ["action_set", "min_tangential_ms", "min_radial_ms",
           "best_residual_relvel_ms", "meets_50ms_box", "meets_1ms_tight_box",
           "box50_margin_x", "da_allowed_by_50ms_box_km",
           "da_allowed_by_1ms_box_km"],
       title="Success-box velocity quanta")


if __name__ == "__main__":
    main()
