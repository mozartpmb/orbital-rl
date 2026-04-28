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
#define RENDEZVOUS_RADIUS   30000.0 /* 30 km — rendezvous position tolerance  */
#define REL_VEL_TOL         50.0    /* 50 m/s — relative velocity tolerance   */

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
#define NUM_ACTIONS 10
#define WARP_ACTION 9
#define WARP_TAU    5    /* 5 × 60s = 5 min of sim time per warp step */
static const double ACTION_DV[NUM_ACTIONS][3] = {
    {   0.0,   0.0,  0.0 },  /* 0: coast                 */
    {   5.0,   0.0,  0.0 },  /* 1: prograde fine    (new) */
    {  10.0,   0.0,  0.0 },  /* 2: prograde small        */
    {  25.0,   0.0,  0.0 },  /* 3: prograde medium       */
    {  -5.0,   0.0,  0.0 },  /* 4: retrograde fine  (new) */
    { -10.0,   0.0,  0.0 },  /* 5: retrograde small      */
    { -25.0,   0.0,  0.0 },  /* 6: retrograde medium     */
    {   0.0,  10.0,  0.0 },  /* 7: radial out            */
    {   0.0, -10.0,  0.0 },  /* 8: radial in             */
    {   0.0,   0.0,  0.0 },  /* 9: warp 5min (τ=5 sub-steps, no burn) */
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
    const double scale_a    = ALT_MAX;            /* for altitudes → [0,1]  */
    const double scale_dist = R_EARTH + ALT_MAX;  /* for distances → [0,~2] */

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
        obs[33] = (float)(dx_l  / R_EARTH);
        obs[34] = (float)(dy_l  / R_EARTH);
        obs[35] = (float)(dvx_l / v_circ_t);
        obs[36] = (float)(dvy_l / v_circ_t);
        obs[37] = (float)(n_tgt / 1e-3);   /* ~LEO mean motion scale */
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
    /* Φ_orbit: orbit shape match */
    double da = fabs(env->sat.orbit.a - env->target.a);
    double e_sx = env->sat.orbit.e * cos(env->sat.orbit.omega);
    double e_sy = env->sat.orbit.e * sin(env->sat.orbit.omega);
    double e_tx = env->target.e    * cos(env->target.omega);
    double e_ty = env->target.e    * sin(env->target.omega);
    double de = sqrt((e_sx - e_tx)*(e_sx - e_tx) + (e_sy - e_ty)*(e_sy - e_ty));
    double phi_orbit = da / SUCCESS_TOL_A + de;

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
        R2_NHR_CLAMP();
        return 1;
    }

    /* 3. Safety cap */
    if (env->step >= MAX_STEPS) {
        env->rewards[0]   = -10.0f;
        env->terminals[0] = 1;
        R2_NHR_CLAMP();
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
    int at_target = (dist_to_target < RENDEZVOUS_RADIUS) && (rel_vel < REL_VEL_TOL);
    if (sat->fuel_mass <= 0.0 && !at_target) {
        env->rewards[0]   = -10.0f;
        env->terminals[0] = 1;
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
    double fuel_frac = env->sat.fuel_mass /
                       (env->sat.dry_mass + env->sat.fuel_mass);
    env->log.perf           += success ? 1.0f : 0.0f;
    env->log.episode_return += env->rewards[0];
    env->log.episode_length += (float)env->step;
    env->log.fuel_used      += (float)(FUEL_FRAC - fuel_frac * FUEL_FRAC);
    env->log.g_shape_abs    += (float)env->g_shape_accum;
    env->log.n++;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * REQUIRED PUFFERLIB FUNCTIONS: c_reset, c_step, c_render, c_close
 * ═══════════════════════════════════════════════════════════════════════════ */

static inline void c_reset(Orbital* env) {
    env->step         = 0;
    env->total_dv_used = 0.0;
    env->episode_id++;

    /* Randomize initial orbit: 300–800 km altitude */
    double alt_init = 300e3 + (rand() / (double)RAND_MAX) * 500e3;
    double a_init   = R_EARTH + alt_init;

    /* Target orbit altitude — different band unless same_orbit_init. */
    double a_target;
    if (env->same_orbit_init) {
        /* Stage 1: sat and target share orbit shape; force same a. */
        a_target = a_init;
    } else {
        double alt_target;
        do {
            alt_target = 300e3 + (rand() / (double)RAND_MAX) * 500e3;
        } while (fabs(alt_target - alt_init) < 50e3);  /* ensure meaningful transfer */
        a_target = R_EARTH + alt_target;
    }

    /* Target orbit — eccentricity sampled from curriculum bound, orientation
     * uniform on [0, 2π). When e_max_target == 0, target is circular. */
    double e_tgt = 0.0;
    if (env->e_max_target > 0.0) {
        e_tgt = (rand() / (double)RAND_MAX) * env->e_max_target;
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
    if (env->same_orbit_init) {
        env->sat.orbit.e     = e_tgt;
        env->sat.orbit.omega = omega_tgt;
    } else if (env->e_max_sat > 0.0) {
        env->sat.orbit.e     = (rand() / (double)RAND_MAX) * env->e_max_sat;
        env->sat.orbit.omega = (rand() / (double)RAND_MAX) * 2.0 * M_PI;
    } else {
        env->sat.orbit.e     = 0.0;
        env->sat.orbit.omega = 0.0;
    }
    env->sat.orbit.M     = (rand() / (double)RAND_MAX) * 2.0 * M_PI;
    {
        /* θ from M via Kepler solve (works for any e ≥ 0; for e=0 it returns θ = M). */
        double E_init = solve_kepler(env->sat.orbit.M, env->sat.orbit.e);
        env->sat.orbit.theta = eccentric_to_true(E_init, env->sat.orbit.e);
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
    if (env->init_phase_gap_max > 0.0) {
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

    /* Write initial state at the start of each episode's first step.
     * Done here (not in c_reset) so that the autoreset at the end of a
     * terminal c_step doesn't clobber the completed episode's traj[0]
     * before Python can read it. */
    if (env->step == 0) {
        write_traj_record(env, 0.0f, 0.0f);
    }

    int action = env->actions[0];

    /* R4 time-warp: action 9 = advance WARP_TAU sub-steps of DT with no
     * burn. Collision/termination checks run every sub-step so the warp
     * never skips past a conjunction. Non-warp actions use τ=1. */
    int tau = (action == WARP_ACTION) ? WARP_TAU : 1;

    /* Apply burn if not coasting/warping and has fuel */
    float dv_applied = 0.0f;
    if (action != 0 && action != WARP_ACTION && env->sat.fuel_mass > 0.0) {
        double dv = apply_impulse(env,
                                   ACTION_DV[action][0],
                                   ACTION_DV[action][1],
                                   ACTION_DV[action][2]);
        dv_applied = (float)dv;
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
        write_traj_record(env, env->rewards[0], dv_applied);
        fill_observations(env);
        env->last_episode_steps = env->step;
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
        int success = (env->rewards[0] > 0.0f);
        env->last_episode_steps = env->step;
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
