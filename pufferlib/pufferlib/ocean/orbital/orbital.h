/*
 * orbital.h — Fuel-optimal orbital maneuver RL environment
 *
 * A satellite must transfer from an initial circular orbit to a target circular
 * orbit while avoiding debris, using minimal fuel. 2D coplanar orbits only (v1).
 *
 * Physics: two-body Keplerian dynamics, impulsive maneuvers, Tsiolkovsky fuel.
 * See CLAUDE.md for full design documentation.
 */

#pragma once

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

/* ── Physical constants ─────────────────────────────────────────────────── */
#define MU          3.986004418e14  /* Earth gravitational parameter (m³/s²) */
#define R_EARTH     6.371e6         /* Earth radius (m)                       */
#define ALT_MIN     200e3           /* Min survivable altitude above surface  */
#define ALT_MAX     1600e3          /* Scenario altitude ceiling              */
#define DT          60.0            /* Simulation timestep (seconds)          */
#define MAX_BODIES  16              /* Earth + debris (max)                   */
#define MAX_STEPS   2000            /* Safety cap (~33 hrs orbital time)      */
#define FUEL_FRAC   0.15            /* Fuel = 15% of initial total mass       */
#define ISP         300.0           /* Specific impulse (seconds)             */
#define G0          9.80665         /* Standard gravity (m/s²)                */
#define VE          (ISP * G0)      /* Exhaust velocity ≈ 2942 m/s           */
#define OBS_DIM     38              /* Observation vector length (33 + 5 LVLH)*/
#define N_OBS_BODY  4               /* # bodies reported in observation       */
#define SUCCESS_TOL_A   10000.0     /* Semi-major axis match tolerance (m)    */
#define SUCCESS_TOL_E   0.01        /* Eccentricity match tolerance           */
#define DEBRIS_KEEPOUT  5000.0      /* Debris keep-out radius (m)            */
#define DEBRIS_HARD_R   1.0         /* Debris hard collision radius (m)       */
#define EARTH_KEEPOUT   (R_EARTH + ALT_MIN)  /* ~6571 km                      */
#define RENDEZVOUS_RADIUS   30000.0 /* 30 km — rendezvous position tolerance (default; see rendezvous_radius_m kwarg) */
#define REL_VEL_TOL         50.0    /* 50 m/s — relative velocity tolerance (default; see rel_vel_tol_ms kwarg) */

/* Terminal-cause codes. Set at every episode end so success classification can
 * key on which branch fired instead of the terminal reward's sign — the sign
 * is corrupted by the Φ-clamp at wide altitude bands (Φ-clamp leak,
 * PHASE5_PRE_CLOSURE_MECHANISM_FINDINGS.md). */
#define TERM_NONE       0
#define TERM_SUCCESS    1
#define TERM_COLLISION  2
#define TERM_ESCAPE     3
#define TERM_SAFETY_CAP 4
#define TERM_STRANDED   5
#define TERM_HYPERBOLIC 6
#define TERM_GAVE_UP    7

/* ── Phase 4 R2: gated multi-stage potential shaping ────────────────────── */
#define BETA_SHAPE   1.0    /* Gated Φ weight; re-enabled for R3a to restore per-step signal */
#define W_ORBIT      0.01   /* Weight for Φ_orbit = |Δa|/TOL_A + ||Δē||       */
#define W_PHASE      0.01   /* Weight for Φ_phase = 1 - cos(Δθ)              */
#define W_VEL        0.01   /* Weight for Φ_vel   = ||v_rel_lvlh||/REL_VEL_TOL*/
#define EPS_ORBIT    2.0    /* Gate threshold: Φ_orbit below this opens σ₂  */
#define EPS_PHASE    0.3    /* Gate threshold: Φ_phase below this opens σ₃  */
#define TAU_ORBIT    (0.1 * EPS_ORBIT)
#define TAU_PHASE    (0.1 * EPS_PHASE)

/* ── Action Δv lookup table ──────────────────────────────────────────────
 * 7 actions (v2): coast + prograde/retrograde at 10/25 m/s + radial ±10 m/s.
 * Each row: { dv_prograde, dv_radial, dv_normal }. dv_normal kept at 0 for
 * forward compatibility with a future 3D upgrade.
 */
#define NUM_ACTIONS 16
#define WARP_ACTION 9    /* legacy: smallest warp action; still referenced */
#define WARP_TAU    5    /* legacy: 5 × 60s = 5 min per warp; supplanted by ACTION_TAU */
static const double ACTION_DV[NUM_ACTIONS][3] = {
    /* 0-9: Phase 4-5b legacy. DO NOT renumber; existing ckpts depend on these indices. */
    {   0.0,   0.0,  0.0 },  /* 0: coast                 */
    {   5.0,   0.0,  0.0 },  /* 1: prograde fine    (new) */
    {  10.0,   0.0,  0.0 },  /* 2: prograde small        */
    {  25.0,   0.0,  0.0 },  /* 3: prograde medium       */
    {  -5.0,   0.0,  0.0 },  /* 4: retrograde fine  (new) */
    { -10.0,   0.0,  0.0 },  /* 5: retrograde small      */
    { -25.0,   0.0,  0.0 },  /* 6: retrograde medium     */
    {   0.0,  10.0,  0.0 },  /* 7: radial out            */
    {   0.0, -10.0,  0.0 },  /* 8: radial in             */
    {   0.0,   0.0,  0.0 },  /* 9: warp 5min (τ=5)       */
    /* M2 (phase5-5-env-mods): longer warps for high-altitude training */
    {   0.0,   0.0,  0.0 },  /* 10: warp 30min (τ=30)    */
    {   0.0,   0.0,  0.0 },  /* 11: warp 1hr   (τ=60)    */
    /* M3: sub-5 m/s prograde/retrograde for MEO/GEO fine control */
    {   1.0,   0.0,  0.0 },  /* 12: prograde 1           */
    {  -1.0,   0.0,  0.0 },  /* 13: retrograde 1         */
    {   2.0,   0.0,  0.0 },  /* 14: prograde 2           */
    {  -2.0,   0.0,  0.0 },  /* 15: retrograde 2         */
};

/* M2 (phase5-5-env-mods): per-action sub-step count. τ=1 → single-step burn or
 * coast; τ>1 → warp action (no burn). Replaces the single WARP_TAU constant for
 * runtime dispatch in c_step. */
static const int ACTION_TAU[NUM_ACTIONS] = {
    1, 1, 1, 1, 1, 1, 1, 1, 1,   /* 0-8: single-step actions */
    5,                            /* 9:  warp 5min  */
    30, 60,                       /* 10-11: M2 longer warps */
    1, 1, 1, 1,                   /* 12-15: M3 sub-5 m/s burns */
};

/* ── PufferLib Log struct ────────────────────────────────────────────────
 * All floats. 'n' MUST be last — env_binding.h iterates by sizeof(Log)/sizeof(float).
 */
typedef struct {
    float perf;             /* success rate [0,1] over logged episodes */
    float episode_return;   /* sum of rewards per episode              */
    float episode_length;   /* steps per episode                       */
    float fuel_used;        /* fuel fraction consumed per episode      */
    float g_shape_abs;      /* Σ|shaping deltas| per episode (Goodhart)*/
    /* Phase 5 env-fix F4 + Phase 5.5 logging expansion: realized-init metrics.
     * vec_log auto-divides every field by n (env_binding.h:590), so accumulating
     * per-episode sums here yields per-epoch means automatically. */
    float init_attempts_mean;   /* mean rejection-sampling attempts per c_reset */
    float init_gave_up_rate;    /* fraction of resets that exhausted the cap */
    float realized_e_target_mean;
    float realized_e_sat_mean;
    float realized_a_target_mean_m;
    float realized_a_sat_mean_m;
    float n;                /* REQUIRED: episode count (last field)    */
} Log;

/* ── Orbital elements (2D coplanar) ─────────────────────────────────────── */
typedef struct {
    double a;       /* semi-major axis (m)      */
    double e;       /* eccentricity [0, 1)      */
    double M;       /* mean anomaly (rad)       */
    double theta;   /* true anomaly (rad)       */
    double omega;   /* argument of periapsis (rad) — orientation of the ellipse */
} Orbit;

/* ── Satellite ───────────────────────────────────────────────────────────── */
typedef struct {
    Orbit  orbit;
    double dry_mass;   /* kg, constant                                  */
    double fuel_mass;  /* kg, decreases with burns; 0 = stranded        */
} Satellite;

/* ── Generic physical body (Earth, debris, future planets) ──────────────── */
typedef struct {
    Orbit  orbit;
    double hard_radius;     /* physical collision radius (m)            */
    double keepout_radius;  /* exclusion zone radius (m)                */
    int    is_static;       /* 1 = fixed at origin (Earth), 0 = orbits  */
} Body;

/* ── Per-step trajectory record ──────────────────────────────────────────
 * Captured every step when log_enabled=1. Dumped to Python at episode end.
 * All positions in meters (Cartesian, Earth at origin).
 */
typedef struct {
    int   episode_id;
    int   step;
    float sim_time;              /* seconds since episode start            */
    float sat_x,  sat_y;        /* satellite Cartesian position (m)       */
    float sat_vx, sat_vy;       /* satellite velocity (m/s)               */
    float sat_a,  sat_e;        /* orbital elements                       */
    float sat_theta;            /* true anomaly (rad)                     */
    float sat_omega;            /* argument of periapsis (rad)            */
    float target_omega;         /* target argument of periapsis (rad)     */
    float fuel;                 /* remaining fuel fraction [0,1]          */
    int   action;               /* action taken (0-6)                     */
    float reward;               /* reward this step                       */
    float delta_v;              /* burn magnitude (m/s); 0 if coasting    */
    float min_conj_dist;        /* closest approach to any body (m)       */
    /* Target orbit — fixed per episode, repeated each step for easy parsing */
    float target_a;
    float target_e;
    float target_x, target_y;    /* target body's Cartesian position (m)   */
    float target_vx, target_vy;  /* target body's Cartesian velocity (m/s) */
    /* All body positions */
    int   num_bodies;
    float body_x[MAX_BODIES];
    float body_y[MAX_BODIES];
    float body_hard_r[MAX_BODIES];
    float body_keepout_r[MAX_BODIES];
} TrajectoryRecord;

/* ── Main environment struct ─────────────────────────────────────────────
 * 'log' MUST be first field. Buffer pointers set by PufferLib binding.
 */
typedef struct {
    Log            log;                     /* REQUIRED first field         */
    float*         observations;            /* float* for Box(float32, 29)  */
    int*           actions;                 /* int*   for Discrete(7)       */
    float*         rewards;
    unsigned char* terminals;

    Satellite      sat;
    Orbit          target;
    Body           bodies[MAX_BODIES];
    int            num_bodies;
    int            step;
    double         total_dv_used;           /* Δv used this episode (m/s)   */
    int            episode_id;

    /* Debris configuration (set from Python) */
    int              num_debris_min;        /* min debris per episode (0 = no debris) */
    int              num_debris_max;        /* max debris per episode                 */

    /* Target eccentricity curriculum (set from Python; 0.0 → circular only) */
    double           e_max_target;          /* upper bound for target.e ∈ [0, e_max]  */

    /* Phase 5b: Satellite eccentricity curriculum (analogous to e_max_target).
     * 0.0 → chaser starts circular (Phase 4 behavior). >0 → chaser samples
     * (e, ω) from same dist as target, optionally constrained to match target. */
    double           e_max_sat;             /* upper bound for sat.e ∈ [0, e_max_sat] */
    int              same_orbit_init;       /* 1 → sat.{a,e,ω} = target.{a,e,ω}, only θ differs (Stage 1) */

    /* Phase 5c B3: mixed-distribution training. With probability e_mix_easy_frac,
     * sample target.e (and sat.e if random) from [0, e_mix_easy_max] instead of
     * the full curriculum bound. Default 0.0 → off, full bound only. */
    double           e_mix_easy_frac;       /* probability of easy draw, default 0.0 */
    double           e_mix_easy_max;        /* easy upper bound, default 0.05 */

    /* Phase 5d I4: soft collision-prevention penalty. After each burn, if
     * post-burn perigee r_p = a(1-e) < EARTH_KEEPOUT, subtract this weight
     * from the per-step reward. Default 0.0 → off. */
    double           collision_penalty_w;

    /* Phase 5d I2: hard action masking. When 1, obs is extended by 10 floats
     * encoding which actions are valid (1.0) or invalid (0.0). Burns whose
     * post-burn perigee would be below EARTH_KEEPOUT are masked. Coast (0)
     * and warp (9) always valid. Default 0 → no mask, obs stays 38-dim. */
    int              enable_action_mask;

    /* Phase 5d D-late: when 1, c_reset rejection-samples sat and target init
     * until both have perigee >= EARTH_KEEPOUT (physically survivable orbits).
     * Default 0 preserves prior behavior (which sampled doomed inits when
     * e_max × altitude band produced sub-surface perigees). */
    int              valid_init_only;

    /* Phase 5 verification I1: when 1, c_reset prints a per-reset debug line
     * to stderr showing the valid_init flag, attempts taken, accepted sat &
     * target perigees, and whether the 256-attempt cap was hit. Default 0. */
    int              log_validation_debug;

    /* Phase 5 env-fix F1/F3: rejection-sampling cap configurable; outcome tracked. */
    int              max_valid_init_attempts;   /* cap on resample attempts; default 4096 */
    int              last_init_attempts;         /* attempts taken in last c_reset */
    int              last_init_gave_up;          /* 1 if cap exhausted with invalid init */
    int              gave_up_action;             /* 0 = accept doomed init (legacy); 1 = terminate next step */
    int              gave_up_terminate_pending;  /* internal flag: c_reset → c_step handoff */

    /* Phase 5 env-fix F4 + Phase 5.5: snapshot of realized init state for per-episode
     * logging (add_log reads at episode end; sat.{a,e} drift during episode via burns). */
    double           last_init_sat_a_m;
    double           last_init_sat_e;
    double           last_init_target_a_m;
    double           last_init_target_e;

    /* Phase 5 wrap-up W1: fixed-value sampling for per-condition surface eval.
     * Each defaults to negative (off → use uniform-up-to-bound). When >= 0,
     * use the exact value. Lets surface eval measure e=0.50 capability
     * specifically rather than aggregate over e ∈ [0, 0.50] uniform. */
    double           e_target_fixed;        /* < 0 → uniform; ≥ 0 → exact target.e */
    double           e_sat_fixed;           /* < 0 → uniform; ≥ 0 → exact sat.e */
    double           phase_gap_fixed;       /* < 0 → uniform [-init_phase_gap_max, +]; ≥ 0 → +exact */
    double           omega_offset_fixed;    /* < -10 → uniform; else target.ω = sat.ω + offset */
    /* W1 follow-up: altitude band override (in meters, NOT altitude floor —
     * raw a values). Defaults < 0 preserve hardcoded LEO 300-800km band.
     * Necessary because fixed-high-e cells require higher a so that
     * perigee = a(1-e) >= EARTH_KEEPOUT remains satisfiable. */
    double           a_min_override;        /* if >= R_EARTH, sample a from [a_min_override, a_max_override] */
    double           a_max_override;

    /* Phase 5.5 altitude expansion: configurable observation scaling so the env
     * can be trained beyond LEO. Default 1.6e6 (= ALT_MAX) preserves Phase 5b/5e
     * checkpoint compatibility. Set to ~4.2e7 for GEO-inclusive training. */
    double           obs_alt_scale_m;        /* altitude normalization scale (meters above R_EARTH) */
    double           phi_orbit_scale_k;      /* Φ_orbit scale gain; effective tol = max(SUCCESS_TOL_A, K * obs_alt_scale_m) */

    /* M1 (phase5-5-env-mods): LVLH spatial obs normalizer for obs[33-34]. Default
     * R_EARTH preserves Phase 5b/5e behavior byte-identically; set ~4.2e7 for GEO.
     * Per pre-experiments E4, at GEO the same-orbit Δr can reach ~84 Mm, divided
     * by R_EARTH gives obs values ~13 — far past Box(-2, 2) bound. */
    double           lvlh_scale_m;

    /* Runtime success-box tolerances (T1: terminal-criterion tightening).
     * Defaults RENDEZVOUS_RADIUS / REL_VEL_TOL preserve the historical 30 km /
     * 50 m/s box. These affect ONLY the termination check; the shaping
     * normalizers (Φ_vel / REL_VEL_TOL, Φ_orbit tol) deliberately keep the
     * historical constants so shaping magnitude stays comparable across
     * success-box sweeps. */
    double           rendezvous_radius_m;
    double           rel_vel_tol_ms;

    /* Rendezvous phase-gap curriculum (set from Python; 0.0 → target co-phased). */
    double           init_phase_gap_max;    /* max initial |θ_sat − θ_target| (rad)  */

    /* Dense shaping cache: prev-step sat↔target distance and phase offset. */
    double           dist_prev;             /* previous-step sat↔target distance (m) */
    double           dphase_prev;           /* previous-step wrapped (θ_sat − θ_target) ∈ [-π, π] */

    /* Phase 4 R2: potential shaping cache */
    double           phi_prev;              /* Φ(s) from previous decision           */
    int              last_tau;              /* sub-steps (1 except under warp)        */
    double           g_shape_accum;         /* |accumulated shaping| per episode      */
    double           last_g_shape;          /* snapshot at terminal, survives c_reset */

    /* Trajectory logging */
    TrajectoryRecord traj_log[MAX_STEPS];
    int              log_enabled;           /* 0 = off (fast), 1 = on       */
    int              last_episode_steps;    /* step count at last terminal (for Python export) */
    int              last_traj_records;     /* valid traj_log rows at last terminal (= steps+1
                                             * incl. the terminal record, capped at MAX_STEPS) */
    int              last_terminal_cause;   /* TERM_* code of the last episode's ending */
} Orbital;


/* ═══════════════════════════════════════════════════════════════════════════
 * PHYSICS FUNCTIONS
 * ═══════════════════════════════════════════════════════════════════════════ */

/* Solve Kepler's equation M = E - e*sin(E) via Newton-Raphson (5 iterations).
 * Returns eccentric anomaly E. */
static inline double solve_kepler(double M, double e) {
    /* Normalize M to [0, 2π) */
    M = fmod(M, 2.0 * M_PI);
    if (M < 0.0) M += 2.0 * M_PI;

    double E = (e < 0.8) ? M : M_PI;  /* initial guess */
    for (int i = 0; i < 5; i++) {
        double dE = (M - E + e * sin(E)) / (1.0 - e * cos(E));
        E += dE;
        if (fabs(dE) < 1e-12) break;
    }
    return E;
}

/* Convert eccentric anomaly E → true anomaly θ. */
static inline double eccentric_to_true(double E, double e) {
    double x = sqrt(1.0 - e) * cos(E / 2.0);
    double y = sqrt(1.0 + e) * sin(E / 2.0);
    return 2.0 * atan2(y, x);
}

/* Convert true anomaly θ → mean anomaly M (inverse Kepler, exact). */
static inline double true_to_mean(double theta, double e) {
    /* θ → E via half-angle */
    double x = sqrt(1.0 - e) * cos(theta / 2.0);
    double y = sqrt(1.0 + e) * sin(theta / 2.0);
    double E = 2.0 * atan2(y, x);
    return E - e * sin(E);
}

/* Advance orbit by dt seconds (Kepler propagation). Updates M and theta. */
static inline void propagate_orbit(Orbit* o, double dt) {
    double n = sqrt(MU / (o->a * o->a * o->a));  /* mean motion (rad/s) */
    o->M += n * dt;
    o->M = fmod(o->M, 2.0 * M_PI);
    if (o->M < 0.0) o->M += 2.0 * M_PI;
    double E = solve_kepler(o->M, o->e);
    o->theta = eccentric_to_true(E, o->e);
}

/* Convert orbital elements (a, e, θ, ω) → Cartesian (x, y, vx, vy).
 * Computes perifocal coordinates then rotates by ω into the inertial frame. */
static inline void orbit_to_cartesian(const Orbit* o,
                                       double* x, double* y,
                                       double* vx, double* vy) {
    double p = o->a * (1.0 - o->e * o->e);     /* semi-latus rectum */
    double r = p / (1.0 + o->e * cos(o->theta));
    double h = sqrt(MU * p);                    /* specific angular momentum */

    /* Perifocal frame (periapsis at +x) */
    double xp  =  r * cos(o->theta);
    double yp  =  r * sin(o->theta);
    double vxp = -(MU / h) * sin(o->theta);
    double vyp =  (MU / h) * (o->e + cos(o->theta));

    /* Rotate by ω into inertial frame */
    double co = cos(o->omega), so = sin(o->omega);
    *x  = co * xp - so * yp;
    *y  = so * xp + co * yp;
    *vx = co * vxp - so * vyp;
    *vy = so * vxp + co * vyp;
}

/* Convert Cartesian (x, y, vx, vy) → orbital elements (a, e, θ, M, ω).
 * Used after applying an impulse burn. */
static inline void cartesian_to_elements(double x, double y,
                                          double vx, double vy,
                                          Orbit* o) {
    double r  = sqrt(x*x + y*y);
    double v2 = vx*vx + vy*vy;
    double vr = (x*vx + y*vy) / r;  /* radial velocity */

    /* Vis-viva: a = 1 / (2/r - v²/μ) */
    o->a = 1.0 / (2.0/r - v2/MU);

    /* Eccentricity vector (inertial frame) */
    double ex = ((v2 - MU/r)*x - vr*r*vx) / MU;
    double ey = ((v2 - MU/r)*y - vr*r*vy) / MU;
    o->e = sqrt(ex*ex + ey*ey);

    /* Argument of periapsis: direction of eccentricity vector */
    if (o->e < 1e-10) {
        o->omega = 0.0;
    } else {
        o->omega = atan2(ey, ex);
    }

    /* True anomaly: angle from periapsis to position */
    if (o->e < 1e-10) {
        o->theta = atan2(y, x);
    } else {
        double cos_theta = (ex*x + ey*y) / (o->e * r);
        cos_theta = fmax(-1.0, fmin(1.0, cos_theta));
        o->theta = acos(cos_theta);
        if (vr < 0.0) o->theta = 2.0 * M_PI - o->theta;
    }

    o->M = true_to_mean(o->theta, o->e);
    if (o->M < 0.0) o->M += 2.0 * M_PI;
}

/* Apply an impulsive Δv in the satellite's local orbital frame.
 * dv_pro = prograde (along velocity), dv_rad = radial out, dv_nor = normal (out-of-plane, 2D→ignored).
 * Updates satellite orbital elements and consumes fuel (Tsiolkovsky).
 * Returns actual |Δv| applied (may be less than requested if fuel limited). */
static inline double apply_impulse(Orbital* env,
                                    double dv_pro, double dv_rad,
                                    double dv_nor) {
    (void)dv_nor;  /* 2D — normal burns have no effect in-plane */

    Satellite* sat = &env->sat;
    double x, y, vx, vy;
    orbit_to_cartesian(&sat->orbit, &x, &y, &vx, &vy);

    /* Local frame unit vectors */
    double v_mag = sqrt(vx*vx + vy*vy);
    double pro_x = vx / v_mag;   /* prograde = velocity direction */
    double pro_y = vy / v_mag;
    double rad_x =  x / sqrt(x*x + y*y);  /* radial = position direction */
    double rad_y =  y / sqrt(x*x + y*y);

    /* Requested Δv in inertial frame */
    double dvx = dv_pro * pro_x + dv_rad * rad_x;
    double dvy = dv_pro * pro_y + dv_rad * rad_y;
    double dv_mag = sqrt(dvx*dvx + dvy*dvy);

    if (dv_mag < 1e-10) return 0.0;

    /* Fuel consumption via Tsiolkovsky: Δm = m_total * (1 - exp(-|Δv|/Ve)) */
    double m_total = sat->dry_mass + sat->fuel_mass;
    double fuel_needed = m_total * (1.0 - exp(-dv_mag / VE));

    /* Clamp if insufficient fuel */
    if (fuel_needed > sat->fuel_mass) {
        double actual_dv = -VE * log(1.0 - sat->fuel_mass / m_total);
        if (actual_dv < 1e-6) {
            sat->fuel_mass = 0.0;
            return 0.0;
        }
        double scale = actual_dv / dv_mag;
        dvx *= scale;
        dvy *= scale;
        dv_mag = actual_dv;
        sat->fuel_mass = 0.0;
    } else {
        sat->fuel_mass -= fuel_needed;
        if (sat->fuel_mass < 0.0) sat->fuel_mass = 0.0;
    }

    /* Apply velocity change */
    vx += dvx;
    vy += dvy;

    /* Convert back to orbital elements */
    cartesian_to_elements(x, y, vx, vy, &sat->orbit);
    env->total_dv_used += dv_mag;

    return dv_mag;
}

/* ── Phase 5d I2: action-mask preview ────────────────────────────────────
 * For a candidate burn (dv_pro, dv_rad), compute the post-burn perigee
 * without mutating the satellite. Returns r_p = a(1-e), or -1.0 if the burn
 * would put the satellite on a hyperbolic trajectory.
 */
static inline double preview_perigee(const Satellite* sat,
                                      double dv_pro, double dv_rad) {
    double x, y, vx, vy;
    orbit_to_cartesian(&sat->orbit, &x, &y, &vx, &vy);
    double v_mag = sqrt(vx*vx + vy*vy);
    double r_mag = sqrt(x*x + y*y);
    if (v_mag < 1e-9 || r_mag < 1e-9) return -1.0;
    double pro_x = vx / v_mag, pro_y = vy / v_mag;
    double rad_x =  x / r_mag, rad_y =  y / r_mag;
    double dvx = dv_pro * pro_x + dv_rad * rad_x;
    double dvy = dv_pro * pro_y + dv_rad * rad_y;
    double new_vx = vx + dvx, new_vy = vy + dvy;
    Orbit po;
    cartesian_to_elements(x, y, new_vx, new_vy, &po);
    if (po.a <= 0.0) return -1.0;  /* hyperbolic — invalid */
    return po.a * (1.0 - po.e);
}

/* ── Observation packing ─────────────────────────────────────────────────── */

/* Get Cartesian position of a body (static bodies are at origin). */
static inline void body_position(const Body* b, double* bx, double* by) {
    if (b->is_static) {
        *bx = 0.0;
        *by = 0.0;
    } else {
        double bvx, bvy;
        orbit_to_cartesian(&b->orbit, bx, by, &bvx, &bvy);
    }
}

static inline void fill_observations(Orbital* env) {
    float* obs = env->observations;
    const Satellite* sat = &env->sat;
    /* Phase 5.5: observation altitude scale is configurable via obs_alt_scale_m
     * (default 1.6e6 = ALT_MAX, preserving LEO-trained ckpt compatibility). */
    const double scale_a    = env->obs_alt_scale_m;             /* for altitudes → [0,1]  */
    const double scale_dist = R_EARTH + env->obs_alt_scale_m;    /* for distances → [0,~2] */

    /* Satellite Cartesian state for velocity decomposition */
    double sx, sy, svx, svy;
    orbit_to_cartesian(&sat->orbit, &sx, &sy, &svx, &svy);

    /* Radial and tangential velocity components */
    double r = sqrt(sx*sx + sy*sy);
    double vr = (sx*svx + sy*svy) / r;
    double vt = (sx*svy - sy*svx) / r;  /* cross product / r = tangential speed */
    double v_circ = sqrt(MU / r);         /* circular velocity at current r */
    double fuel_frac = sat->fuel_mass / (sat->dry_mass + sat->fuel_mass);

    /* [0-6] Satellite state */
    obs[0] = (float)((sat->orbit.a - R_EARTH) / scale_a);
    obs[1] = (float)(sat->orbit.e);
    obs[2] = (float)sin(sat->orbit.theta);              /* [-1, 1] — no wrap discontinuity */
    obs[3] = (float)cos(sat->orbit.theta);              /* [-1, 1]                         */
    obs[4] = (float)(vr / v_circ);                       /* ~[-1, 1] */
    obs[5] = (float)(vt / v_circ);                       /* ~[0, 2] for LEO */
    obs[6] = (float)(fuel_frac);

    /* [7-8] Target orbit */
    obs[7] = (float)((env->target.a - R_EARTH) / scale_a);
    obs[8] = (float)(env->target.e);

    /* [9-12] Argument-of-periapsis encoding (sin/cos sat ω, sin/cos target ω) */
    obs[9]  = (float)sin(sat->orbit.omega);
    obs[10] = (float)cos(sat->orbit.omega);
    obs[11] = (float)sin(env->target.omega);
    obs[12] = (float)cos(env->target.omega);

    /* [13-16] Rendezvous phasing: sin/cos of orbit-angle phase gap (sat − target),
     * and sin/cos of target true anomaly. For circular orbits θ = M, so these
     * give the agent direct access to the angular offset it must close. */
    double dtheta_phase = sat->orbit.theta - env->target.theta;
    obs[13] = (float)sin(dtheta_phase);
    obs[14] = (float)cos(dtheta_phase);
    obs[15] = (float)sin(env->target.theta);
    obs[16] = (float)cos(env->target.theta);

    /* [17-32] Closest N_OBS_BODY bodies: Δr, Δθ, closing_rate, keepout_r
     * First compute distances to all bodies, then pick the N closest. */
    double dist[MAX_BODIES];
    for (int i = 0; i < env->num_bodies; i++) {
        double bx, by;
        body_position(&env->bodies[i], &bx, &by);
        double dx = sx - bx, dy = sy - by;
        dist[i] = sqrt(dx*dx + dy*dy);
    }

    /* Simple selection sort to get N_OBS_BODY closest indices */
    int idx[N_OBS_BODY];
    double used_dist[MAX_BODIES];
    memcpy(used_dist, dist, sizeof(double) * env->num_bodies);

    for (int k = 0; k < N_OBS_BODY; k++) {
        if (k >= env->num_bodies) {
            idx[k] = -1;
            continue;
        }
        int best = 0;
        for (int i = 1; i < env->num_bodies; i++) {
            if (used_dist[i] < used_dist[best]) best = i;
        }
        idx[k] = best;
        used_dist[best] = 1e30;  /* mark as used */
    }

    for (int k = 0; k < N_OBS_BODY; k++) {
        int base = 17 + k * 4;
        if (idx[k] < 0) {
            obs[base]   = 0.0f;
            obs[base+1] = 0.0f;
            obs[base+2] = 0.0f;
            obs[base+3] = 0.0f;
            continue;
        }
        int i = idx[k];
        const Body* b = &env->bodies[i];

        double bx, by, bvx = 0.0, bvy = 0.0;
        if (b->is_static) {
            bx = 0.0; by = 0.0;
        } else {
            orbit_to_cartesian(&b->orbit, &bx, &by, &bvx, &bvy);
        }

        double dx = sx - bx, dy = sy - by;
        double dr = dist[i];

        /* Relative bearing: angle difference */
        double sat_angle  = atan2(sy, sx);
        double body_angle = atan2(by, bx);
        double dtheta = sat_angle - body_angle;
        /* Normalize to [-π, π] */
        while (dtheta >  M_PI) dtheta -= 2.0 * M_PI;
        while (dtheta < -M_PI) dtheta += 2.0 * M_PI;

        /* Closing rate: d/dt(|r_sat - r_body|) = (r̂ · (v_sat - v_body)) */
        double closing = ((dx*(svx-bvx) + dy*(svy-bvy)) / dr);

        obs[base]   = (float)(dr / scale_dist);
        obs[base+1] = (float)(dtheta / M_PI);
        obs[base+2] = (float)(closing / v_circ);
        obs[base+3] = (float)(b->keepout_radius / scale_dist);
    }

    /* [33-37] LVLH-frame relative state — primary observation for rendezvous.
     * Rotating frame co-moving with target: x-axis = radial (outward from
     * Earth through target), y-axis = along-track (in direction of target
     * motion). Makes the rendezvous problem state-invariant to target's
     * inertial angular position, which is the key observation-aliasing fix
     * from Phase 4 spec §3.1. */
    {
        double tx, ty, tvx, tvy;
        orbit_to_cartesian(&env->target, &tx, &ty, &tvx, &tvy);
        /* Target inertial angle: ω + θ (perifocal→inertial) */
        double theta_t = env->target.theta + env->target.omega;
        double ct = cos(theta_t), st = sin(theta_t);

        double dxi  = sx  - tx;
        double dyi  = sy  - ty;
        double dvxi = svx - tvx;
        double dvyi = svy - tvy;

        /* Rotate inertial offset into target's LVLH frame */
        double dx_l  =  ct * dxi  + st * dyi;
        double dy_l  = -st * dxi  + ct * dyi;
        double dvx_l =  ct * dvxi + st * dvyi;
        double dvy_l = -st * dvxi + ct * dvyi;

        /* Subtract frame-rotation term n × r (so velocity is relative to
         * rotating LVLH frame, not inertial frame rotated). */
        double n_tgt = sqrt(MU / (env->target.a * env->target.a * env->target.a));
        dvx_l += n_tgt * dy_l;
        dvy_l -= n_tgt * dx_l;

        double v_circ_t = sqrt(MU / env->target.a);
        /* M1 (phase5-5-env-mods): spatial LVLH normalizer is configurable via
         * env->lvlh_scale_m (default R_EARTH; set ~4.2e7 for GEO training). */
        obs[33] = (float)(dx_l  / env->lvlh_scale_m);
        obs[34] = (float)(dy_l  / env->lvlh_scale_m);
        obs[35] = (float)(dvx_l / v_circ_t);
        obs[36] = (float)(dvy_l / v_circ_t);
        obs[37] = (float)(n_tgt / 1e-3);   /* ~LEO mean motion scale */
    }

    /* [38-47] Phase 5d I2: action-validity mask. Only written when
     * enable_action_mask=1 — the gymnasium obs space is 48-dim in that
     * mode and 38-dim otherwise, so writing past obs[37] in the disabled
     * path would corrupt the next env's obs slice in the vector buffer.
     * Coast (0) and warp (9) are always valid; burns 1..8 are masked if
     * their post-burn perigee would be below EARTH_KEEPOUT. */
    if (env->enable_action_mask) {
        obs[38] = 1.0f;  /* action 0: coast */
        for (int a = 1; a <= 8; a++) {
            double r_p = preview_perigee(&env->sat,
                                          ACTION_DV[a][0],
                                          ACTION_DV[a][1]);
            obs[38 + a] = (r_p > 0.0 && r_p >= EARTH_KEEPOUT) ? 1.0f : 0.0f;
        }
        obs[47] = 1.0f;  /* action 9: warp */
    }
}

/* ── Phase 4 R2: potential Φ(s) and gated shaping ───────────────────────── */

/* Compute gated multi-stage potential Φ(s) per spec §3.2. Returns Φ ≤ 0.
 *   Φ_orbit = |Δa|/SUCCESS_TOL_A + ||Δē||   — orbit-shape closure
 *   Φ_phase = 1 − cos(Δθ)                    — phase alignment (smooth across ±π)
 *   Φ_vel   = ||v_rel_lvlh|| / REL_VEL_TOL   — rel-velocity null-out
 * Gates σ₂, σ₃ open as earlier-stage potentials drop below calibrated ε.
 * Φ(s) = −(w₁·Φ_orbit·σ₁ + w₂·Φ_phase·σ₂ + w₃·Φ_vel·σ₃). */
static inline double compute_phi(const Orbital* env) {
    /* Φ_orbit: orbit shape match.
     * Phase 5.5 altitude expansion: effective orbit-match tolerance scales with
     * the env's altitude domain. tol_eff = max(SUCCESS_TOL_A, K * obs_alt_scale_m).
     * Default K=0.001 at LEO (obs_alt_scale_m=1.6e6): max(10km, 1.6km) = 10km →
     * backward compat with Phase 5b/5e checkpoints. At GEO (obs_alt_scale_m=4.2e7):
     * max(10km, 42km) = 42km → Φ_orbit stays O(1-10) instead of O(1000+). */
    double da = fabs(env->sat.orbit.a - env->target.a);
    double e_sx = env->sat.orbit.e * cos(env->sat.orbit.omega);
    double e_sy = env->sat.orbit.e * sin(env->sat.orbit.omega);
    double e_tx = env->target.e    * cos(env->target.omega);
    double e_ty = env->target.e    * sin(env->target.omega);
    double de = sqrt((e_sx - e_tx)*(e_sx - e_tx) + (e_sy - e_ty)*(e_sy - e_ty));
    double phi_tol_eff = fmax(SUCCESS_TOL_A, env->phi_orbit_scale_k * env->obs_alt_scale_m);
    double phi_orbit = da / phi_tol_eff + de;

    /* Φ_phase: true-anomaly alignment along target orbit (1 - cos wraps smoothly) */
    double dtheta = env->sat.orbit.theta - env->target.theta;
    double phi_phase = 1.0 - cos(dtheta);

    /* Φ_vel: LVLH-frame relative velocity magnitude */
    double sx, sy, svx, svy;
    orbit_to_cartesian(&env->sat.orbit, &sx, &sy, &svx, &svy);
    double tx, ty, tvx, tvy;
    orbit_to_cartesian(&env->target, &tx, &ty, &tvx, &tvy);
    double theta_t = env->target.theta + env->target.omega;
    double ct = cos(theta_t), st = sin(theta_t);
    double dvxi = svx - tvx, dvyi = svy - tvy;
    double dvx_l =  ct * dvxi + st * dvyi;
    double dvy_l = -st * dvxi + ct * dvyi;
    double dxi = sx - tx, dyi = sy - ty;
    double dx_l =  ct * dxi + st * dyi;
    double dy_l = -st * dxi + ct * dyi;
    double n_tgt = sqrt(MU / (env->target.a * env->target.a * env->target.a));
    dvx_l += n_tgt * dy_l;
    dvy_l -= n_tgt * dx_l;
    double phi_vel = sqrt(dvx_l*dvx_l + dvy_l*dvy_l) / REL_VEL_TOL;

    /* Gates: σ₁=1 always, σ₂ opens when Φ_orbit below ε_orbit,
     *        σ₃ = σ₂ · σ(ε_phase - Φ_phase)/τ_phase */
    double sigma2 = 1.0 / (1.0 + exp(-(EPS_ORBIT - phi_orbit) / TAU_ORBIT));
    double sigma3 = sigma2 / (1.0 + exp(-(EPS_PHASE - phi_phase) / TAU_PHASE));

    return -(W_ORBIT * phi_orbit * 1.0
           + W_PHASE * phi_phase * sigma2
           + W_VEL   * phi_vel   * sigma3);
}

/* ── Termination check ───────────────────────────────────────────────────── */

/* Returns 1 if episode is over, sets env->rewards[0]. Sets env->terminals[0].
 * Does NOT call c_reset — caller does that. */
static inline int check_termination(Orbital* env) {
    const Satellite* sat = &env->sat;

    /* Phase 4 R2: NHR boundary — clamp Φ to 0 at every terminal via
     *   reward += β · (0 - Φ_prev)
     * Use macro to apply uniformly at each terminal branch. */
    #define R2_NHR_CLAMP()                                               \
        do {                                                             \
            double _delta = BETA_SHAPE * (0.0 - env->phi_prev);          \
            env->rewards[0]   += (float)_delta;                          \
            env->g_shape_accum += fabs(_delta);                          \
            env->phi_prev       = 0.0;                                   \
        } while (0)

    /* 0. Hyperbolic orbit (a ≤ 0) — escape, skip cartesian conversion */
    if (sat->orbit.a <= 0.0) {
        env->rewards[0]   = -10.0f;
        env->terminals[0] = 1;
        env->last_terminal_cause = TERM_HYPERBOLIC;
        R2_NHR_CLAMP();
        return 1;
    }

    double sx, sy, svx, svy;
    orbit_to_cartesian(&sat->orbit, &sx, &sy, &svx, &svy);
    double r = sqrt(sx*sx + sy*sy);

    /* 1. Collision with any body */
    for (int i = 0; i < env->num_bodies; i++) {
        double bx, by;
        body_position(&env->bodies[i], &bx, &by);
        double dx = sx - bx, dy = sy - by;
        double d = sqrt(dx*dx + dy*dy);
        if (d < env->bodies[i].hard_radius) {
            env->rewards[0]   = -10.0f;
            env->terminals[0] = 1;
            env->last_terminal_cause = TERM_COLLISION;
            R2_NHR_CLAMP();
            return 1;
        }
    }

    /* 2. Escape trajectory: specific orbital energy E = ½v² - μ/r ≥ 0 */
    double v2 = svx*svx + svy*svy;
    double E_orb = 0.5 * v2 - MU / r;
    if (E_orb >= 0.0) {
        env->rewards[0]   = -10.0f;
        env->terminals[0] = 1;
        env->last_terminal_cause = TERM_ESCAPE;
        R2_NHR_CLAMP();
        return 1;
    }

    /* 3. Safety cap. NO Φ-clamp here — this terminal is clock-caused, not
     * state-caused. Clamping potential to zero on a timeout pays out
     * β·|Φ_prev|, which grows with distance-from-target and flipped the
     * terminal reward positive at wide altitude bands (the Φ-clamp leak:
     * do-nothing GEO timeouts scored as "successes"). State-caused terminals
     * keep the clamp (PBRS boundary condition at absorbing states).
     * KNOWN ISSUE for future wide-band TRAINING: the per-step shaping term
     * β(γ^τΦ' − Φ) still pays ~β(γ−1)|Φ| per step to a frozen policy far
     * from target; bound Φ (phi_orbit_scale_k) or use γ_shape=1 before
     * training with a_max − a_min ≳ 10,000 km. */
    if (env->step >= MAX_STEPS) {
        env->rewards[0]   = -10.0f;
        env->terminals[0] = 1;
        env->last_terminal_cause = TERM_SAFETY_CAP;
        env->phi_prev     = 0.0;   /* c_reset re-derives; keep state consistent */
        return 1;
    }

    /* 4. Stranded / 5. Success — Phase 3 rendezvous check.
     * Success requires BOTH position and relative-velocity match with the
     * propagated target body, not just orbit-shape (a, ē) matching. */
    double tx, ty, tvx, tvy;
    orbit_to_cartesian(&env->target, &tx, &ty, &tvx, &tvy);
    double dx = sx - tx, dy = sy - ty;
    double dist_to_target = sqrt(dx*dx + dy*dy);
    double rvx = svx - tvx, rvy = svy - tvy;
    double rel_vel = sqrt(rvx*rvx + rvy*rvy);
    int at_target = (dist_to_target < env->rendezvous_radius_m) &&
                    (rel_vel < env->rel_vel_tol_ms);
    if (sat->fuel_mass <= 0.0 && !at_target) {
        env->rewards[0]   = -10.0f;
        env->terminals[0] = 1;
        env->last_terminal_cause = TERM_STRANDED;
        R2_NHR_CLAMP();
        return 1;
    }

    /* 5. Success — keep Stage 2's single fuel bonus so the warm-started value
     * head sees a familiar terminal distribution. Reward-reshaping (dual-bonus)
     * is deferred; Phase 4 R5 isolates the shaping-alone effect. NHR clamp at
     * terminal preserves potential-based shaping optimality. */
    if (at_target) {
        double initial_fuel = sat->dry_mass * FUEL_FRAC / (1.0 - FUEL_FRAC);
        double fuel_remaining = sat->fuel_mass / initial_fuel;
        if (fuel_remaining < 0.0) fuel_remaining = 0.0;
        if (fuel_remaining > 1.0) fuel_remaining = 1.0;
        env->rewards[0]   = 10.0f * (float)(0.5 + 0.5 * fuel_remaining);
        env->terminals[0] = 1;
        env->last_terminal_cause = TERM_SUCCESS;
        R2_NHR_CLAMP();
        return 1;
    }

    #undef R2_NHR_CLAMP
    return 0;
}

/* ── Trajectory logging helper ───────────────────────────────────────────── */

static inline void write_traj_record(Orbital* env, float reward, float dv) {
    if (!env->log_enabled || env->step >= MAX_STEPS) return;

    TrajectoryRecord* rec = &env->traj_log[env->step];
    const Satellite* sat = &env->sat;

    double sx, sy, svx, svy;
    orbit_to_cartesian(&sat->orbit, &sx, &sy, &svx, &svy);

    rec->episode_id  = env->episode_id;
    rec->step        = env->step;
    rec->sim_time    = (float)(env->step * DT);
    rec->sat_x       = (float)sx;
    rec->sat_y       = (float)sy;
    rec->sat_vx      = (float)svx;
    rec->sat_vy      = (float)svy;
    rec->sat_a       = (float)sat->orbit.a;
    rec->sat_e       = (float)sat->orbit.e;
    rec->sat_theta   = (float)sat->orbit.theta;
    rec->sat_omega   = (float)sat->orbit.omega;
    rec->target_omega= (float)env->target.omega;
    rec->fuel        = (float)(sat->fuel_mass / (sat->dry_mass + sat->fuel_mass));
    rec->action      = env->actions[0];
    rec->reward      = reward;
    rec->delta_v     = dv;

    /* Target body snapshot — actual propagated position and velocity */
    rec->target_a  = (float)env->target.a;
    rec->target_e  = (float)env->target.e;
    double tx, ty, tvx, tvy;
    orbit_to_cartesian(&env->target, &tx, &ty, &tvx, &tvy);
    rec->target_x  = (float)tx;
    rec->target_y  = (float)ty;
    rec->target_vx = (float)tvx;
    rec->target_vy = (float)tvy;

    /* Min conjunction distance */
    double min_dist = 1e30;
    for (int i = 0; i < env->num_bodies; i++) {
        double bx, by;
        body_position(&env->bodies[i], &bx, &by);
        double dx = sx - bx, dy = sy - by;
        double d = sqrt(dx*dx + dy*dy);
        if (d < min_dist) min_dist = d;
    }
    rec->min_conj_dist = (float)min_dist;

    /* All body positions */
    rec->num_bodies = env->num_bodies;
    for (int i = 0; i < env->num_bodies; i++) {
        double bx, by;
        body_position(&env->bodies[i], &bx, &by);
        rec->body_x[i]        = (float)bx;
        rec->body_y[i]        = (float)by;
        rec->body_hard_r[i]   = (float)env->bodies[i].hard_radius;
        rec->body_keepout_r[i]= (float)env->bodies[i].keepout_radius;
    }
}

/* ── Log aggregation ─────────────────────────────────────────────────────── */

static inline void add_log(Orbital* env, int success) {
    /* Fraction of the fuel BUDGET consumed, 0 (no burns) .. 1 (tank empty).
     * The old formula (FUEL_FRAC - fuel_frac*FUEL_FRAC, with fuel_frac =
     * fuel/total mass) had FUEL_FRAC applied twice: its range was only
     * [0.1275, 0.15] and zero burns logged 0.1275. */
    double initial_fuel = env->sat.dry_mass * FUEL_FRAC / (1.0 - FUEL_FRAC);
    double budget_used  = 1.0 - env->sat.fuel_mass / initial_fuel;
    if (budget_used < 0.0) budget_used = 0.0;
    env->log.perf           += success ? 1.0f : 0.0f;
    env->log.episode_return += env->rewards[0];
    env->log.episode_length += (float)env->step;
    env->log.fuel_used      += (float)budget_used;
    env->log.g_shape_abs    += (float)env->g_shape_accum;
    /* Phase 5 env-fix F4 + Phase 5.5: realized-init metrics. vec_log divides
     * sums by n automatically, so per-episode contributions yield epoch means. */
    env->log.init_attempts_mean      += (float)env->last_init_attempts;
    env->log.init_gave_up_rate       += env->last_init_gave_up ? 1.0f : 0.0f;
    env->log.realized_e_target_mean  += (float)env->last_init_target_e;
    env->log.realized_e_sat_mean     += (float)env->last_init_sat_e;
    env->log.realized_a_target_mean_m += (float)env->last_init_target_a_m;
    env->log.realized_a_sat_mean_m   += (float)env->last_init_sat_a_m;
    env->log.n++;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * REQUIRED PUFFERLIB FUNCTIONS: c_reset, c_step, c_render, c_close
 * ═══════════════════════════════════════════════════════════════════════════ */

static inline void c_reset(Orbital* env) {
    env->step         = 0;
    env->total_dv_used = 0.0;
    env->episode_id++;

    /* Phase 5d: rejection-sampling wrapper. Resample initial conditions until
     * both sat and target orbits have perigee >= EARTH_KEEPOUT. Without this,
     * a*(1-e) < R_EARTH inits collide unavoidably (~64% of e_max=0.20 samples).
     * Bounded loop (256 attempts) so we never hang on degenerate kwargs. */
    int valid_init_attempts = 0;
    valid_init_resample:
    valid_init_attempts++;

    /* Randomize initial orbit: 300–800 km altitude (default), or override range. */
    double a_init;
    if (env->a_min_override >= R_EARTH && env->a_max_override > env->a_min_override) {
        a_init = env->a_min_override + (rand() / (double)RAND_MAX) * (env->a_max_override - env->a_min_override);
    } else {
        double alt_init = 300e3 + (rand() / (double)RAND_MAX) * 500e3;
        a_init = R_EARTH + alt_init;
    }

    /* Target orbit altitude — different band unless same_orbit_init. */
    double a_target;
    if (env->same_orbit_init) {
        /* Stage 1: sat and target share orbit shape; force same a. */
        a_target = a_init;
    } else {
        double a_min_t = (env->a_min_override >= R_EARTH) ? env->a_min_override : (R_EARTH + 300e3);
        double a_max_t = (env->a_max_override > env->a_min_override && env->a_min_override >= R_EARTH)
                          ? env->a_max_override : (R_EARTH + 800e3);
        do {
            a_target = a_min_t + (rand() / (double)RAND_MAX) * (a_max_t - a_min_t);
        } while (fabs(a_target - a_init) < 50e3);  /* ensure meaningful transfer */
    }

    /* Target orbit — eccentricity sampled from curriculum bound, orientation
     * uniform on [0, 2π). When e_max_target == 0, target is circular.
     * B3 mixed-distribution: with probability e_mix_easy_frac, draw from
     * [0, e_mix_easy_max] instead of [0, e_max_target]. */
    double e_tgt = 0.0;
    int draw_easy = 0;
    if (env->e_target_fixed >= 0.0) {
        /* W1: exact value override; bypass mix-easy logic */
        e_tgt = env->e_target_fixed;
    } else if (env->e_max_target > 0.0) {
        if (env->e_mix_easy_frac > 0.0 &&
            (rand() / (double)RAND_MAX) < env->e_mix_easy_frac &&
            env->e_max_target > env->e_mix_easy_max) {
            e_tgt = (rand() / (double)RAND_MAX) * env->e_mix_easy_max;
            draw_easy = 1;
        } else {
            e_tgt = (rand() / (double)RAND_MAX) * env->e_max_target;
        }
    }
    double omega_tgt = (e_tgt > 0.0)
                         ? (rand() / (double)RAND_MAX) * 2.0 * M_PI
                         : 0.0;
    env->target.a     = a_target;
    env->target.e     = e_tgt;
    env->target.omega = omega_tgt;

    /* Phase 5b: Satellite initial orbit. Three modes:
     *   1. same_orbit_init: sat.{a,e,ω} = target.{a,e,ω} — only θ differs.
     *   2. e_max_sat > 0:    sat.e ~ U(0, e_max_sat), sat.ω ~ U(0, 2π).
     *   3. else:              sat.e = 0, sat.ω = 0 (Phase 4 behavior). */
    env->sat.orbit.a     = a_init;
    if (env->e_sat_fixed >= 0.0) {
        /* W1: exact value override. Honor same_orbit_init for omega so the
         * relation flag composes with fixed-e cells. */
        env->sat.orbit.e = env->e_sat_fixed;
        if (env->same_orbit_init) {
            env->sat.orbit.omega = omega_tgt;
        } else {
            env->sat.orbit.omega = (env->sat.orbit.e > 0.0)
                                     ? (rand() / (double)RAND_MAX) * 2.0 * M_PI
                                     : 0.0;
        }
    } else if (env->same_orbit_init) {
        env->sat.orbit.e     = e_tgt;
        env->sat.orbit.omega = omega_tgt;
    } else if (env->e_max_sat > 0.0) {
        /* Mirror B3 mixed-distribution for sat: same easy-draw decision. */
        if (draw_easy && env->e_max_sat > env->e_mix_easy_max) {
            env->sat.orbit.e = (rand() / (double)RAND_MAX) * env->e_mix_easy_max;
        } else {
            env->sat.orbit.e = (rand() / (double)RAND_MAX) * env->e_max_sat;
        }
        env->sat.orbit.omega = (rand() / (double)RAND_MAX) * 2.0 * M_PI;
    } else {
        env->sat.orbit.e     = 0.0;
        env->sat.orbit.omega = 0.0;
    }
    /* W1: omega_offset_fixed > -10 → set target.omega = sat.omega + offset */
    if (env->omega_offset_fixed > -10.0) {
        env->target.omega = env->sat.orbit.omega + env->omega_offset_fixed;
        omega_tgt = env->target.omega;
    }
    env->sat.orbit.M     = (rand() / (double)RAND_MAX) * 2.0 * M_PI;
    {
        /* θ from M via Kepler solve (works for any e ≥ 0; for e=0 it returns θ = M). */
        double E_init = solve_kepler(env->sat.orbit.M, env->sat.orbit.e);
        env->sat.orbit.theta = eccentric_to_true(E_init, env->sat.orbit.e);
    }
    /* Phase 5d: validate sat & target perigees. If either is sub-keepout,
     * resample. After max_valid_init_attempts (default 4096, Phase 5 env-fix
     * F1; was hardcoded 256), give up and accept whatever was drawn (avoids
     * infinite loop under pathological kwargs). */
    if (env->valid_init_only && valid_init_attempts < env->max_valid_init_attempts) {
        double sat_rp = env->sat.orbit.a * (1.0 - env->sat.orbit.e);
        double tgt_rp = env->target.a * (1.0 - env->target.e);
        if (sat_rp < EARTH_KEEPOUT || tgt_rp < EARTH_KEEPOUT) {
            goto valid_init_resample;
        }
    }

    /* Phase 5 env-fix F1/F3: record reset outcome for downstream logging
     * (trajectory exports, wandb metrics) and arm gave-up termination if
     * gave_up_action == 1 ("terminate"). */
    {
        double sat_rp_final = env->sat.orbit.a * (1.0 - env->sat.orbit.e);
        double tgt_rp_final = env->target.a * (1.0 - env->target.e);
        env->last_init_attempts = valid_init_attempts;
        env->last_init_gave_up  = (env->valid_init_only
                                   && valid_init_attempts >= env->max_valid_init_attempts
                                   && (sat_rp_final < EARTH_KEEPOUT || tgt_rp_final < EARTH_KEEPOUT));
        env->gave_up_terminate_pending = (env->last_init_gave_up && env->gave_up_action == 1);

        /* Snapshot realized init state for per-episode wandb logging. sat.{a,e}
         * will drift during episode via burns; these fields freeze the init. */
        env->last_init_sat_a_m    = env->sat.orbit.a;
        env->last_init_sat_e      = env->sat.orbit.e;
        env->last_init_target_a_m = env->target.a;
        env->last_init_target_e   = env->target.e;

        /* Phase 5 verification I1: emit per-reset debug line. */
        if (env->log_validation_debug) {
            fprintf(stderr,
                    "RESET ep=%d valid_init=%d attempts=%d sat_rp=%.4e tgt_rp=%.4e "
                    "sat_a=%.4e sat_e=%.4f tgt_a=%.4e tgt_e=%.4f gave_up=%d\n",
                    env->episode_id, env->valid_init_only, valid_init_attempts,
                    sat_rp_final, tgt_rp_final,
                    env->sat.orbit.a, env->sat.orbit.e,
                    env->target.a, env->target.e,
                    env->last_init_gave_up);
        }
    }

    env->sat.dry_mass    = 850.0;   /* kg */
    env->sat.fuel_mass   = env->sat.dry_mass * FUEL_FRAC / (1.0 - FUEL_FRAC);

    /* fuel_mass = dry_mass * (fuel_fraction / (1 - fuel_fraction))
     * so that fuel_mass / (dry_mass + fuel_mass) = FUEL_FRAC */

    /* Target mean anomaly sampled so that initial orbit-angle offset between
     * sat and target is uniform in [−init_phase_gap_max, +init_phase_gap_max].
     * For circular orbits θ = M, so this directly controls the Cartesian
     * angular gap as seen from Earth's centre. */
    double phase_gap = 0.0;
    if (env->phase_gap_fixed >= 0.0) {
        /* W1: exact phase gap (positive sign) for surface eval */
        phase_gap = env->phase_gap_fixed;
    } else if (env->init_phase_gap_max > 0.0) {
        phase_gap = (2.0 * (rand() / (double)RAND_MAX) - 1.0) * env->init_phase_gap_max;
    }
    double tgt_M = env->sat.orbit.M + phase_gap;
    tgt_M = fmod(tgt_M, 2.0 * M_PI);
    if (tgt_M < 0.0) tgt_M += 2.0 * M_PI;
    env->target.M = tgt_M;
    double tgt_E = solve_kepler(env->target.M, env->target.e);
    env->target.theta = eccentric_to_true(tgt_E, env->target.e);

    /* Earth: bodies[0], static, at origin */
    env->bodies[0].orbit.a          = 0.0;
    env->bodies[0].orbit.e          = 0.0;
    env->bodies[0].orbit.M          = 0.0;
    env->bodies[0].orbit.theta      = 0.0;
    env->bodies[0].orbit.omega      = 0.0;
    env->bodies[0].hard_radius      = R_EARTH;
    env->bodies[0].keepout_radius   = EARTH_KEEPOUT;
    env->bodies[0].is_static        = 1;

    /* Debris: configurable count (0 = no debris, for curriculum learning) */
    int dmin = env->num_debris_min;
    int dmax = env->num_debris_max;
    if (dmax < dmin) dmax = dmin;
    int num_debris = (dmin == dmax) ? dmin : dmin + rand() % (dmax - dmin + 1);
    env->num_bodies = 1 + num_debris;

    for (int i = 0; i < num_debris; i++) {
        double d_alt = 300e3 + (rand() / (double)RAND_MAX) * 500e3;
        double d_a   = R_EARTH + d_alt;
        double d_e   = (rand() / (double)RAND_MAX) * 0.05;  /* low eccentricity */
        double d_M   = (rand() / (double)RAND_MAX) * 2.0 * M_PI;
        double d_E   = solve_kepler(d_M, d_e);
        double d_theta = eccentric_to_true(d_E, d_e);

        env->bodies[1 + i].orbit.a          = d_a;
        env->bodies[1 + i].orbit.e          = d_e;
        env->bodies[1 + i].orbit.M          = d_M;
        env->bodies[1 + i].orbit.theta      = d_theta;
        env->bodies[1 + i].orbit.omega      = (rand() / (double)RAND_MAX) * 2.0 * M_PI;
        env->bodies[1 + i].hard_radius      = DEBRIS_HARD_R;
        env->bodies[1 + i].keepout_radius   = DEBRIS_KEEPOUT;
        env->bodies[1 + i].is_static        = 0;
    }

    /* Seed dense-shaping cache: distance and phase offset at t=0. */
    {
        double sx, sy, svx, svy;
        orbit_to_cartesian(&env->sat.orbit, &sx, &sy, &svx, &svy);
        double tx, ty, tvx, tvy;
        orbit_to_cartesian(&env->target,     &tx, &ty, &tvx, &tvy);
        double dx = sx - tx, dy = sy - ty;
        env->dist_prev = sqrt(dx*dx + dy*dy);

        double dp = env->sat.orbit.theta - env->target.theta;
        dp = dp - 2.0*M_PI * floor((dp + M_PI) / (2.0*M_PI));
        env->dphase_prev = dp;
    }

    /* Phase 4 R2: seed Φ shaping cache at t=0. last_tau=1 (non-warp). */
    env->phi_prev      = compute_phi(env);
    env->last_tau      = 1;
    env->g_shape_accum = 0.0;

    fill_observations(env);
}

static inline void c_step(Orbital* env) {
    env->rewards[0]   = 0.0f;
    env->terminals[0] = 0;

    /* Phase 5 env-fix F3: gave-up termination. If c_reset's rejection sampler
     * exhausted its attempt cap with a sub-keepout init AND gave_up_action=1
     * ("terminate"), emit a single terminal step with reward 0 and skip physics.
     * The episode contributes nothing to learning (no -10 collision penalty
     * either — the doomed init was an env-sampling artifact, not policy fault).
     * This cleanly excludes doomed inits from eval success-rate denominators. */
    if (env->gave_up_terminate_pending) {
        env->terminals[0] = 1;
        env->last_terminal_cause = TERM_GAVE_UP;
        write_traj_record(env, env->rewards[0], 0.0f);
        fill_observations(env);
        env->last_episode_steps = 1;
        env->last_traj_records  = 1;
        env->last_g_shape       = 0.0;
        add_log(env, 0);
        env->gave_up_terminate_pending = 0;
        c_reset(env);
        return;
    }

    /* Write initial state at the start of each episode's first step.
     * Done here (not in c_reset) so that the autoreset at the end of a
     * terminal c_step doesn't clobber the completed episode's traj[0]
     * before Python can read it. */
    if (env->step == 0) {
        write_traj_record(env, 0.0f, 0.0f);
    }

    int action = env->actions[0];

    /* R4 time-warp: warp actions (9/10/11) advance τ sub-steps of DT with no
     * burn. Collision/termination checks run every sub-step so the warp
     * never skips past a conjunction. M2 (phase5-5-env-mods): per-action τ
     * lookup so we can have multiple warp durations without hardcoding. */
    int tau = ACTION_TAU[action];

    /* Apply burn if not coasting/warping and has fuel. M2/M3: any τ=1 non-coast
     * action is a burn (excludes all warps generically, including new 10/11). */
    float dv_applied = 0.0f;
    if (action != 0 && tau == 1 && env->sat.fuel_mass > 0.0) {
        double dv = apply_impulse(env,
                                   ACTION_DV[action][0],
                                   ACTION_DV[action][1],
                                   ACTION_DV[action][2]);
        dv_applied = (float)dv;

        /* Phase 5d I4: soft collision-prevention penalty. If the burn placed
         * the satellite on a reentry trajectory (perigee < EARTH_KEEPOUT),
         * subtract collision_penalty_w from the step reward. Gated on burn
         * (dv > 0) so coast/warp don't repeatedly fire on already-low orbits. */
        if (env->collision_penalty_w > 0.0 && dv > 0.0 &&
            env->sat.orbit.a > 0.0) {
            double r_p = env->sat.orbit.a * (1.0 - env->sat.orbit.e);
            if (r_p < EARTH_KEEPOUT) {
                env->rewards[0] -= (float)env->collision_penalty_w;
            }
        }
    }

    /* Early escape check — a ≤ 0 means hyperbolic; skip propagation.
     * Apply NHR clamp manually since check_termination's hyperbolic branch
     * can't safely reach cartesian conversion on un-propagated state. */
    if (env->sat.orbit.a <= 0.0) {
        env->rewards[0]    = -10.0f;
        double _clamp       = BETA_SHAPE * (0.0 - env->phi_prev);
        env->rewards[0]    += (float)_clamp;
        env->g_shape_accum += fabs(_clamp);
        env->phi_prev       = 0.0;
        env->terminals[0]   = 1;
        env->last_terminal_cause = TERM_HYPERBOLIC;
        write_traj_record(env, env->rewards[0], dv_applied);
        fill_observations(env);
        env->last_episode_steps = env->step;
        env->last_traj_records  = (env->step + 1 > MAX_STEPS) ? MAX_STEPS : env->step + 1;
        env->last_g_shape       = env->g_shape_accum;
        add_log(env, 0);
        c_reset(env);
        return;
    }

    /* Sub-step loop. τ=1 for non-warp actions (unchanged byte-for-byte
     * semantics vs pre-R4). τ=WARP_TAU (=5) for action=WARP_ACTION, with
     * fill_obs + check_termination executed after every sub-step so that
     * collisions/escapes inside the warp window terminate normally. */
    int actual_tau = 0;
    int warped_terminated = 0;
    for (int k = 0; k < tau; k++) {
        /* Propagate all orbiting bodies (not Earth) */
        for (int i = 1; i < env->num_bodies; i++) {
            propagate_orbit(&env->bodies[i].orbit, DT);
        }
        /* Propagate the target body — Phase 3 rendezvous requires a moving target. */
        propagate_orbit(&env->target, DT);
        /* Propagate satellite */
        propagate_orbit(&env->sat.orbit, DT);

        env->step++;
        actual_tau++;

        /* Update observations (state of s_{t+1}) */
        fill_observations(env);

        /* Per-sub-step trajectory snapshot so warp doesn't leave zero-filled
         * gaps in the log. dv attributed to the first sub-step only (warp
         * itself doesn't burn, so dv_applied is 0 for action=WARP_ACTION).
         * Reward is overwritten below by the terminal/shaping write. */
        write_traj_record(env, 0.0f, (k == 0) ? dv_applied : 0.0f);

        if (check_termination(env)) {
            warped_terminated = 1;
            break;
        }
    }

    env->last_tau = actual_tau;

    /* Check termination BEFORE applying per-step shaping.
     * At terminal: check_termination sets rewards[0] = r_env and applies
     * R2_NHR_CLAMP which adds β·(0 − env->phi_prev). phi_prev still holds
     * Φ(s_t) from the previous step — the correct NHR terminal form:
     *   r' = r_env + 0 − Φ(s_t)
     * Previously, the per-step shaping below ran first, updating phi_prev
     * to Φ(s_{t+1}); then check_termination overwrote rewards[0], erasing
     * the shaping delta and making the NHR clamp use the wrong value. */
    if (warped_terminated) {
        write_traj_record(env, env->rewards[0], dv_applied);
        /* Classify on the terminal branch, not the reward sign — the Φ-clamp
         * can push failure-terminal rewards positive at wide altitude bands
         * (Φ-clamp leak). */
        int success = (env->last_terminal_cause == TERM_SUCCESS);
        env->last_episode_steps = env->step;
        env->last_traj_records  = (env->step + 1 > MAX_STEPS) ? MAX_STEPS : env->step + 1;
        env->last_g_shape       = env->g_shape_accum;
        add_log(env, success);
        c_reset(env);
        return;
    }

    /* Non-terminal: gated multi-stage potential shaping (§3.2).
     *   r_shape = β · (γ^τ · Φ(s_{t+1}) − Φ(s_t))   — NHR form.
     * γ must match orbital.ini's gamma (0.995). τ=1 except under time-warp. */
    {
        double phi_curr    = compute_phi(env);
        double gamma_tau   = pow(0.995, (double)env->last_tau);
        double delta       = BETA_SHAPE * (gamma_tau * phi_curr - env->phi_prev);
        env->rewards[0]   += (float)delta;
        env->g_shape_accum += fabs(delta);
        env->phi_prev      = phi_curr;
    }

    write_traj_record(env, env->rewards[0], dv_applied);
}

static inline void c_render(Orbital* env) {
    /* ASCII polar grid visualization for Phase 1 debugging.
     * Prints a 21×41 character polar diagram to terminal. */
    const int ROWS = 21, COLS = 41;
    char grid[21][42];
    memset(grid, '.', sizeof(grid));
    for (int i = 0; i < ROWS; i++) grid[i][COLS] = '\0';

    /* Scale: ALT_MAX above Earth surface → half the grid */
    double scale = (double)(COLS / 2) / (R_EARTH + ALT_MAX);

    /* Helper: map (x,y) in meters to grid cell */
    #define GRID_COL(x) ((int)((x) * scale) + COLS/2)
    #define GRID_ROW(y) (ROWS/2 - (int)((y) * scale))
    #define GRID_SET(x, y, c) do { \
        int gc = GRID_COL(x), gr = GRID_ROW(y); \
        if (gc >= 0 && gc < COLS && gr >= 0 && gr < ROWS) grid[gr][gc] = (c); \
    } while(0)

    /* Draw target orbit ring (sampled points) */
    for (int k = 0; k < 64; k++) {
        double ang = k * 2.0 * M_PI / 64.0;
        double tx = env->target.a * cos(ang);
        double ty = env->target.a * sin(ang);
        GRID_SET(tx, ty, 'o');
    }

    /* Draw Earth */
    GRID_SET(0.0, 0.0, 'E');

    /* Draw debris */
    for (int i = 1; i < env->num_bodies; i++) {
        double bx, by;
        body_position(&env->bodies[i], &bx, &by);
        GRID_SET(bx, by, '*');
    }

    /* Draw satellite */
    double sx, sy, svx, svy;
    orbit_to_cartesian(&env->sat.orbit, &sx, &sy, &svx, &svy);
    GRID_SET(sx, sy, 'S');

    /* Print grid */
    printf("\033[H");  /* move cursor to top-left (ANSI) */
    printf("Step %4d | a=%.0f km | e=%.4f | fuel=%.2f%% | dv=%.1f m/s\n",
           env->step,
           (env->sat.orbit.a - R_EARTH) / 1000.0,
           env->sat.orbit.e,
           100.0 * env->sat.fuel_mass / (env->sat.dry_mass + env->sat.fuel_mass),
           env->total_dv_used);
    printf("target: a=%.0f km  |  S=sat  E=earth  *=debris  o=target orbit\n",
           (env->target.a - R_EARTH) / 1000.0);
    for (int i = 0; i < ROWS; i++) {
        printf("%s\n", grid[i]);
    }

    #undef GRID_COL
    #undef GRID_ROW
    #undef GRID_SET
}

static inline void c_close(Orbital* env) {
    (void)env;  /* nothing to free — buffers owned by PufferLib/caller */
}
