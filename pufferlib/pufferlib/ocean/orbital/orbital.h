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
#include <stdint.h>
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
#define MAX_STEPS   22000           /* Trajectory-buffer size + max runtime episode cap.
                                     * The actual per-episode cap is env->episode_cap_steps
                                     * (default 2000 = legacy 33.3 h; T3 recovery 3000;
                                     * wide-envelope 6000 = 100 h; MEO 12000 = 200 h;
                                     * ext-j2wait 22000 = 366 h = 15.3 d, the
                                     * drift-and-wait horizon (j2_plane_change:
                                     * the >=3x fuel band costs a FIXED 12.8 d
                                     * regardless of the plane angle bought) —
                                     * periods scale as a^1.5, recon feasibility §3.5).
                                     * Counts 60 s SIM SUB-STEPS, not agent decisions —
                                     * warps buy decisions, never wall clock. Buffer memory
                                     * is virtual unless log_enabled (gated on traj_log_dir). */
#define FUEL_FRAC   0.15            /* Fuel = 15% of initial total mass (DEFAULT;
                                     * overridden per episode when fuel sampling
                                     * is on — see episode_fuel_frac)          */

/* ── T11: the per-episode CELL MIXTURE ────────────────────────────────────────
 * A generalist rung trains one policy over heterogeneous cells (e-band,
 * altitude band, box, J2/inclination config, episode cap, fuel budget). A
 * vec-env takes ONE kwarg set, so the mixture cannot be expressed by kwargs:
 * the draw has to happen inside c_reset, per episode.
 *
 * WHY THE CAP MUST BE PART OF THE CELL, not a global maximum. obs[15] is
 * (episode_cap_steps - step)/episode_cap_steps — the episode clock, which T3
 * measured as load-bearing (even the 99.2% scripted expert fails clock-blind).
 * Running every cell at one global cap of 22000 would put a 3000-substep cell
 * at t_frac in [0.86, 1.0]: the clock compressed 7.3x and near-constant. So
 * `episode_cap_steps` is assigned FROM THE CELL at reset, which makes obs[15]
 * correct per episode for free — no separate clock plumbing.
 *
 * STALL MATH, per cell. cap_terminal_reward = 0 and shape_gamma = 1 mean a
 * capped episode banks only its shaping, bounded by Phi's range
 * W_lambda + W_match = 1.8167. A success pays 10*gamma^n. Stalling therefore
 * wins iff 10*gamma^n < 1.8167, i.e. n > 340 decisions at gamma = 0.995.
 * Measured decisions/episode: X3 ~18, TB5 ~55, E-cells ~40-90, drift-and-wait
 * 46 WITH the day-warp and 544 without. Every cell in the shipped table clears
 * 340 by 4-20x PROVIDED row 30 exists and is used; the one configuration that
 * violates it is drift-and-wait on a 30-row head, which this rung does not run.
 *
 * The table is set through `vec_set_cells` (a float64 (n, CELL_FIELDS) array)
 * rather than through kwargs, because `unpack()` takes one scalar at a time and
 * a 7-cell table is ~126 numbers. cell_mixture_mode = 0 is the default and
 * consumes ZERO rand() draws, so every existing lineage is bit-exact. */
/* THE PER-ENV SEEDING TRAP. `env_binding.h:502` reseeds every env with
 * `srand(i + seed*num_envs)` — SEQUENTIAL integers — and glibc's rand() is an
 * additive-feedback generator whose FIRST outputs are strongly correlated for
 * nearby seeds. Every pre-existing sampler is safe only because it sits deep in
 * the stream. The cell and fuel draws are the 1st and 2nd rand() of the
 * episode, i.e. the worst possible position: measured, all 8 envs drew the same
 * cell and their fuel fractions marched linearly (0.133, 0.144, 0.156, ...).
 * So the draws are taken from a splitmix64 seeded once from rand() — one rand()
 * consumed, decorrelated by construction, and still fully reproducible. */
static inline uint64_t t11_mix64(uint64_t* x) {
    uint64_t z = (*x += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}
static inline double t11_u01(uint64_t* x) {
    return (double)(t11_mix64(x) >> 11) * (1.0 / 9007199254740992.0);
}

#define MAX_CELLS    16
#define CELL_FIELDS  18
/* field order — keep in sync with orbital.py::CELL_FIELDS */
#define CF_WEIGHT      0
#define CF_CAP         1
#define CF_BOX_R       2
#define CF_BOX_V       3
#define CF_A_MIN       4
#define CF_A_MAX       5
#define CF_E_MAX_T     6
#define CF_E_MAX_S     7
#define CF_DE_MAX      8
#define CF_DA_MAX      9
#define CF_DI_MAX     10
#define CF_DI_MIN     11
#define CF_DI_PHASE   12
#define CF_J2         13
#define CF_IT_MIN     14
#define CF_IT_MAX     15
#define CF_FUEL_MIN   16
#define CF_FUEL_MAX   17
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

/* ── ext-j2: secular mean-element J2 (j2_A_design.md §1.1) ────────────────────
 * J2_R_EQ is the EQUATORIAL radius and is deliberately NOT R_EARTH. R_EARTH
 * (6371 km) is the MEAN radius the env uses for altitudes and the keepout;
 * J2's reference radius is 6378.137 km. Substituting the env constant would
 * bias every rate by (6371/6378.137)² = −0.22%. The 7 km inconsistency in the
 * altitude bookkeeping is pre-existing and is NOT laundered into the dynamics.
 * J2_COEF is the WGS-84 / EGM96 value (the design memo quoted the 6-digit
 * 1.08263e-3; the extra digits are a 3.1e−6 relative change, below every
 * measurement floor in the oracle table). */
#define J2_COEF     1.08262668e-3   /* Earth J2 zonal harmonic (WGS-84)        */
#define J2_R_EQ     6.378137e6      /* equatorial radius (m) — NOT R_EARTH     */

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
#define NUM_ACTIONS 31
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
    /* T3 follow-ups (2026-08-11), Discrete-20. The exposed action space
     * DEFAULTS to Discrete(16) (orbital.py legacy_action_space) so every
     * pre-existing checkpoint and command is untouched; new lineages opt in
     * with legacy_action_space=20.
     * 16-17: MEO/GEO warps — periods scale a^1.5; a 12,000-step (200 h)
     *        MEO episode needs mean τ ≳ 27 to stay inside the γ=0.995
     *        credit horizon (recon feasibility §3.5).
     * 18-19: radial ±1 m/s — the 10 m/s radial quantum was the binding
     *        floor for the tight success box (best |v_rel| 5.02 m/s with
     *        16 actions → 0.71 m/s with fine radial; red-team/recon §4). */
    {   0.0,   0.0,  0.0 },  /* 16: warp 3hr  (τ=180)    */
    {   0.0,   0.0,  0.0 },  /* 17: warp 6hr  (τ=360)    */
    {   0.0,   1.0,  0.0 },  /* 18: radial out 1         */
    {   0.0,  -1.0,  0.0 },  /* 19: radial in 1          */
    /* ext-3d (2026-08-11), Discrete-30. Appended only — indices 0-19 are
     * frozen. Rows 20-25 are the pure out-of-plane axis (normal = +ĥ, orbit
     * north); rows 26-29 are COMBINED tangential+normal impulses.
     *
     * Why the combined rows exist (3d_REDTEAM MAJOR-3, option ii): the plane
     * screen that justified di_max ≈ 1° assumed ONE impulse in a combined
     * direction, but every legacy row is single-axis, so two impulses one
     * sub-step apart cost D + P in Tsiolkovsky rather than hypot(D, P). The
     * measured recovery from adding {±25, 0, ±25} is +15pp at X3 and +20pp at
     * X4. NOTE this also flips the L1-combiner rationale (3d_REDTEAM n1): with
     * combined actions the 1.41× L1 bonus IS realizable, by design.
     *
     * ±1 m/s normal is required for the tight box: 1 m/s ≡ 0.0075° of plane
     * rotation at LEO, exactly the 5 km / 1 m/s box's plane tolerance
     * (3d_C §4.8). All ten rows are τ = 1. */
    {   0.0,   0.0,   1.0 },  /* 20: normal +1            */
    {   0.0,   0.0,  -1.0 },  /* 21: normal −1            */
    {   0.0,   0.0,  10.0 },  /* 22: normal +10           */
    {   0.0,   0.0, -10.0 },  /* 23: normal −10           */
    {   0.0,   0.0,  25.0 },  /* 24: normal +25           */
    {   0.0,   0.0, -25.0 },  /* 25: normal −25           */
    {  25.0,   0.0,  25.0 },  /* 26: combined +25 / +25   */
    {  25.0,   0.0, -25.0 },  /* 27: combined +25 / −25   */
    { -25.0,   0.0,  25.0 },  /* 28: combined −25 / +25   */
    { -25.0,   0.0, -25.0 },  /* 29: combined −25 / −25   */
    /* ext-j2wait (2026-08-15), Discrete-31. APPENDED — rows 0-29 keep their
     * indices and their meanings bit-for-bit, so a Discrete-30 checkpoint's
     * decoder rows transfer unchanged and only a single zero-init row is
     * added (scripts/orbital/expand_ckpt_actions_7_to_9.py's pattern, minus
     * the remapping that one needed).
     *
     * WHY A NEW ROW AND NOT A REBIND OF AN EXISTING WARP. The prior was to
     * rebind row 17 (tau 360 -> 1440) under a flag, on the theory that the
     * 6 h warp is rarely used. Measured over 100 held-out episodes of the
     * J2 lineage this campaign warm-starts from, it is not:
     *     A2 loose box   row 17 = 11.1% of decisions; rows 16+17 carry
     *                    87.4% of all SUBSTEPS
     *     A3b tight box  row 17 =  2.3% of decisions; rows 16+17 carry
     *                    36.7% of all substeps
     * Row 17 is the warm start's primary time-advance tool. Rebinding it
     * would quadruple the effect of 11% of the policy's decisions while
     * leaving its logit untouched — a silent semantic swap under a trained
     * head, which is this project's defect class #1. Appending costs one
     * zero-init decoder row and changes nothing else. */
    {   0.0,   0.0,   0.0 },  /* 30: warp 1 day (tau=1440) */
};

/* ── ext-3d: two things deliberately NOT implemented ─────────────────────────
 *
 * (1) The "coast to next relative-node crossing" macro-action (3d_C §4.5) is
 *     DEFERRED, and no kwarg is reserved for it. The barrier it was meant to
 *     remove is not there: measured realized Δv / ideal 2v sin(Δi/2) at LEO is
 *     1.00× at τ = 5 (warp-5min, action 9, already in the table) and only
 *     degrades at τ ≥ 30 — 2-20 m/s on a 478 m/s budget. Against that, the
 *     macro's one-period clamp is τ = 718 sub-steps at MEO versus the warp
 *     set's 360, i.e. a new discount extreme that would confound "3D plane
 *     skill" with "a longer warp"; and its Δi → 0 clamp to τ = 1 is a 48×
 *     (LEO) / 359× (MEO) discontinuity in sim-seconds-per-decision sited
 *     exactly in the endgame. If longer warps are wanted at MEO, add a plain
 *     warp-12hr so the ablation stays separable (3d_REDTEAM m1).
 *
 * (2) J2 was deferred here; ext-j2 (2026-08-13) LANDED it as `j2_mode`, default
 *     0 = bit-exact legacy. See `propagate_orbit_j2` below and
 *     `J2_DESIGN_NOTES.md`. The verdict quoted in this note — "fidelity
 *     upgrade, not task-enrichment", priced as an alternative way to BUY a
 *     plane change (49.5 h + 222 m/s per 1° of ΔΩ vs 104 m/s direct) — was
 *     re-measured and holds only at MEO/wide. At LEO J2 is a DISTURBANCE
 *     coupled to the leg the policy already flies: ΔΩ per radian of phase
 *     closed is −3.5·J2·(R_EQ/p)²·cos i, independent of the drift δa, i.e.
 *     0.286° of relative inclination injected by one 180° phasing drift at
 *     LEO-500 / i = 51.6° — 76% of the free-plane zone at the 30 km / 50 m/s
 *     box (j2_A_design §0, §2.3). The shaping re-audit the note asks for was
 *     done: drift-leg monotonicity is preserved 36/36 and the worst adverse
 *     step is unchanged (§2.1); only "do-nothing ΔΦ ≡ 0" is lost, becoming
 *     |ΔΦ| ≤ 0.006 over a full cap (§2.2). */

/* M2 (phase5-5-env-mods): per-action sub-step count. τ=1 → single-step burn or
 * coast; τ>1 → warp action (no burn). Replaces the single WARP_TAU constant for
 * runtime dispatch in c_step. */
static const int ACTION_TAU[NUM_ACTIONS] = {
    1, 1, 1, 1, 1, 1, 1, 1, 1,   /* 0-8: single-step actions */
    5,                            /* 9:  warp 5min  */
    30, 60,                       /* 10-11: M2 longer warps */
    1, 1, 1, 1,                   /* 12-15: M3 sub-5 m/s burns */
    180, 360,                     /* 16-17: T3 MEO/GEO warps */
    1, 1,                         /* 18-19: T3 fine radial burns */
    1, 1, 1, 1, 1, 1,             /* 20-25: ext-3d normal burns */
    1, 1, 1, 1,                   /* 26-29: ext-3d combined burns */
    1440,                         /* 30: ext-j2wait day-warp (24 h/decision) */
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

/* ── Orbital elements (classical; 3D under dim3_mode=1) ─────────────────────
 * ext-3d: gains inc/raan. Zero-init (`Orbital env = {0}`, and c_reset writing
 * 0.0 whenever dim3_mode==0) ⇒ every Orbit is EXACTLY equatorial and every
 * value gate below takes the verbatim legacy statements ⇒ the 2D lineage is
 * bit-identical by construction (3d_A §5.1 closure argument).
 *
 * CONSUMER RULE (3d_A §3): no obs channel, shaping term, sampling constraint
 * or termination test may read `omega` or `raan` ALONE — at small inclination
 * each carries ~3.9° of noise per metre of state error while their sum ϖ and
 * λ = M + ϖ are flat across seven decades of inclination (3d_A P6). Use
 * orb_varpi / orb_lambda / orb_hhat / orb_evec. The pre-existing obs[9-12,16]
 * raw-ω channels are grandfathered: 3d_REDTEAM MAJOR-2 option (a) keeps slots
 * 0-16 bit-semantics so every validated decoder/expert/injector still works. */
typedef struct {
    double a;       /* semi-major axis (m)      */
    double e;       /* eccentricity [0, 1)      */
    double M;       /* mean anomaly (rad)       */
    double theta;   /* true anomaly (rad)       */
    double omega;   /* argument of periapsis (rad), FROM THE ASCENDING NODE.
                     * Identical to the legacy meaning whenever raan == 0. */
    double inc;     /* NEW (ext-3d): inclination (rad); 0 = equatorial prograde */
    double raan;    /* NEW (ext-3d): RAAN Ω (rad); 0 when equatorial            */
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
    /* ── ext-3d columns. APPENDED at the end of the exported row (binding.c
     * fill_traj_row) so every pre-existing column index stays stable. ────── */
    float sat_z,  sat_vz;
    float sat_inc, sat_raan;
    float target_z, target_vz;
    float target_inc, target_raan;
    /* Post-impulse, PRE-propagation chaser state for the sub-step that carried
     * a burn (zeros on coast/warp rows). Without it the 3D invariant battery's
     * I7/I8/I9/I11/I15 can only be evaluated against a reconstruction of the
     * burn — i.e. against themselves. With it they test the env. */
    float burn_post_x, burn_post_y, burn_post_z;
    float burn_post_vx, burn_post_vy, burn_post_vz;
    /* All body positions (bodies are equatorial by construction: body_z ≡ 0) */
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

    /* T3 corrected-dynamics recovery (2026-08-11): runtime-flagged reward and
     * coordinate fixes. Defaults preserve legacy behavior bit-exactly; see
     * T3_RECOVERY_CAMPAIGN.md §5 and scripts/orbital/t3/reports/. */
    int              shaping_mode;       /* 0 = legacy gated Φ; 1 = S-R3 phase-time potential   */
    double           shape_w_lambda;     /* mode 1: weight on |Δλ|/π              (default 1.0) */
    double           shape_w_match;      /* mode 1: weight on orbit-match term    (default 0.35)*/
    double           shape_dv_ref_ms;    /* mode 1: Δv_match normalizer           (default 300) */
    double           shape_gamma;        /* shaping discount base; >= 1.0 → γ_shape = 1 exactly
                                          * (kills the (1−γ^τ)·|Φ| stall income; recon F-1)     */
    int              phase_gap_mode;     /* 0 = legacy per-perifocal M offset (inert at e>0,
                                          * recon ANOM-4); 1 = physical mean-longitude gap      */
    int              phase_obs_mode;     /* 0 = legacy true-anomaly obs[13-16] (teleports on
                                          * burns, sign wrong 39% at e>0); 1 = mean-longitude
                                          * Δλ in [13,14] + clock obs[15] + apsidal obs[16]     */
    int              episode_cap_steps;  /* runtime safety cap in sim sub-steps; <= MAX_STEPS   */
    double           cap_terminal_reward;/* reward at TERM_SAFETY_CAP; −10 legacy, 0 for T3
                                          * (red-team #1: −10 under per-decision γ makes warps
                                          * a −7.8 bet until success rate ≈ 47%)                */

    /* T3 wide-eccentricity ladder (L3+; recon feasibility §3.4). With ω sampled
     * independently, matching e-VECTORS costs Δv_e ≈ v·|Δē|/2 — at e_max=0.15
     * that alone averages 396 m/s of a 478 m/s budget. Bounding |Δē| (not e)
     * keeps wide-e affordable: MEO/e=0.5 goes 16.6% → 93.0% feasible. */
    double           de_max;             /* < 0 off. Else sat ē = target ē + disc(de_max):
                                          * bounds the e-vector mismatch while both orbits
                                          * can be strongly eccentric. Overrides e_max_sat. */
    double           da_max_m;           /* < 0 off. Else |a_target − a_sat| ≤ da_max_m
                                          * (window ∩ altitude band). Values < 200 km are
                                          * raised to 200 km (transfer-floor guard).       */

    /* ── ext-3d (2026-08-11). All defaults preserve the 2D lineage bit-exactly.
     * Binding spec: scripts/orbital/ext_recon/reports/3d_{A,B,C,E}*.md as
     * amended by 3d_REDTEAM.md (the amendments win on every conflict). ───── */
    int              dim3_mode;          /* master gate. 0 ⇒ every 3D value gate takes
                                          * the legacy path and inc = raan = 0.0.     */
    double           di_max_rad;         /* < 0 off. Else ĥ_s = R(δ, n̂)·ĥ_t with
                                          * δ = di_max_rad·√U and n̂ uniform IN THE
                                          * TARGET PLANE — the rotation sampler
                                          * (3d_REDTEAM BLOCKER-1). The ī-disc form in
                                          * 3d_A §4 is DELETED: it under-measures by
                                          * cos i_t and realizes 21× the knob at
                                          * i_t = 98°.                                 */
    double           i_target_rad;       /* absolute target inclination (rad). Pure GAUGE
                                          * under two-body — default 0 (3d_C §4.4c).
                                          * Non-zero only as a test hook for the
                                          * sampler / frame gates.                     */
    double           raan_target_rad;    /* absolute target RAAN (rad); gauge, default 0 */
    double           obs_di_scale_rad;   /* obs[21,22] normalizer; <= 0 → max(di_max_rad, 0.25°) */
    double           obs_de_scale;       /* obs[23] normalizer;    <= 0 → max(de_max, 0.05)      */
    int              shape_match_squash; /* Φ match squash: 0 = min(1, x) (legacy, and the
                                          * A2 bit-exact anchor), 1 = x/(1+x) (no dead zone) */

    /* ── ext-j2 (2026-08-13). Binding spec: j2_A_design.md. Default 0 is the
     * verbatim legacy propagator, reached by branching, not by a zero term. */
    /* ── ext-j2 rung: inclined-target sampler ────────────────────────────────
     * Under two-body the target plane is pure gauge, so the env has always
     * pinned it (i_t = Ω_t = 0). Under J2 the EQUATOR IS PHYSICAL and that
     * default makes the whole upgrade inert: at an equatorial target
     * Δi_rel = i_s regardless of Ω_s, so differential Ω̇ cannot move the plane
     * term at all (measured Δ(dv_pl) = 0.000000000 m/s over a full cap —
     * J-G6/J-A5). These two knobs turn the target plane into a sampled
     * quantity. Defaults are off, and the OFF path consumes ZERO rand() draws,
     * so every pre-existing anchor stays bit-exact by RNG-stream identity.
     *
     * The two are NOT the same kind of knob, and conflating them is the trap:
     *   i_t  is TASK VARIATION. The channel strength is
     *        1.75·J2·(R_EQ/p)²·sin 2i per radian of λ closed, so it peaks at
     *        45° and is identically ZERO at 0° and 90°.
     *   Ω_t  is GAUGE. J2's potential is axisymmetric about ẑ, so rotating the
     *        whole scene about ẑ maps solutions to solutions exactly. Sampling
     *        Ω_t adds no task content; it is a LEAK DETECTOR (SO(2)-about-ẑ,
     *        the reduced form of the ext-3d SO(3) frame gate — j2_A_design
     *        §4.1). Differential Ω̇ does NOT depend on Ω. */
    double           i_target_min_rad;   /* < 0 off. Else i_t ~ U(min, max) per episode,
                                          * overriding i_target_rad. Recommended rung band
                                          * U(30°, 60°): sin 2i ∈ [0.866, 1.0], i.e. the J2
                                          * plane channel is within 13% of its maximum
                                          * everywhere in the band, and 63.43° (the critical
                                          * inclination, where ω̇ = 0) stays OUTSIDE so a
                                          * second unrelated degeneracy is not mixed in. */
    double           i_target_max_rad;   /* < 0 off (see above)                            */
    /* ext-j2 rung: obs[33-36] frame. 0 = LEGACY (bit-exact, and the only mode
     * any shipped checkpoint was trained under): the "LVLH" block is built
     * from the INERTIAL x,y offset rotated by the in-plane angle omega+theta.
     * That is exactly the LVLH frame when i_t = Omega_t = 0 — which every
     * shipped lineage pinned — and is neither LVLH nor rotation-invariant
     * about ẑ once i_t != 0: the equatorial projection of r̂_t is
     * (cos u, sin u · cos i) at Omega_t = 0, not (cos u, sin u), and Δz is
     * dropped outright. 1 = the true target orbital frame (R̂ = r̂_t 3D,
     * Ĉ = ĥ_t, T̂ = Ĉ × R̂), which reduces to the legacy construction at
     * i = Omega = 0 up to float round-off but NOT bitwise, hence the gate. */
    int              lvlh_frame_mode;

    /* ── ext-j2wait: the relative-plane error's SIZE and ORIENTATION ─────────
     * The shipped sampler draws δ = di_max_rad·√U (area-uniform on [0, di_max])
     * with the rotation axis n̂ UNIFORM in the target plane. Both defaults are
     * wrong for a drift-and-wait experiment:
     *
     *  - size:  the interesting band is BEYOND the direct-burn budget
     *           (134 m/s per degree at LEO ⇒ 478 m/s buys 3.57°), and an
     *           area-uniform draw to 5° still puts 16% of its mass below 2°.
     *  - orientation: drift moves Ω and NOTHING ELSE, so only the node
     *           component of the error is drift-correctable — E|cos φ| = 2/π
     *           = 63.7% under the uniform-phase draw, and a pure inclination
     *           error is untouchable no matter how long you wait
     *           (j2_plane_change §E). Sampling uniformly would mix
     *           "the policy failed to drift" with "there was nothing to drift
     *           for", which is the confound that makes the experiment
     *           unreadable.
     *
     * The basis the shipped sampler already builds makes the orientation knob
     * one line: û₁ = ẑ × ĥ_t IS the line of nodes, so rotating ĥ_t about û₁
     * (φ = 0) tilts the plane about the node line — a pure INCLINATION error —
     * while rotating about û₂ = ĥ_t × û₁ (φ = ±90°) moves the node — a pure
     * NODE error. Node-dominant is therefore φ near ±90°. */
    double           di_min_rad;         /* < 0 off (legacy √U draw). Else δ ~ U(di_min_rad,
                                          * di_max_rad) UNIFORM IN ANGLE, so the band is the
                                          * band and not an area-weighted version of it.  */
    int              di_phase_mode;      /* 0 = φ uniform (legacy, bit-exact).
                                          * 1 = NODE-DOMINANT: φ = ±(π/2 + U(−30°, +30°)),
                                          * i.e. the node component is ≥ cos30° = 86.6% of
                                          * the error, so drift is actually available.     */

    int              raan_target_sample; /* 0 = Ω_t = raan_target_rad exactly (legacy);
                                          * 1 = Ω_t = raan_target_rad + U(0, 2π). Gauge
                                          * under J2, so this is an invariance test, not
                                          * task enrichment.                                */

    int              j2_mode;            /* 0 = off (bit-exact anchor); 1 = secular
                                          * mean-element J2 on Ω, ω, M. Requires
                                          * dim3_mode = 1 (asserted in my_init):
                                          * under dim3_mode = 0 every orbit is
                                          * exactly equatorial and J2 would silently
                                          * become a ϖ̇ = +k precession on the 2D
                                          * lineage's ω, which no 2D anchor covers. */

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

    /* ext-3d: post-impulse / pre-propagation chaser state of the current
     * decision's burn sub-step, for the trajectory log (see TrajectoryRecord). */
    double           last_burn_post[6];
    int              last_burn_valid;

    /* Trajectory logging */
    TrajectoryRecord traj_log[MAX_STEPS];
    int              log_enabled;           /* 0 = off (fast), 1 = on       */
    int              last_episode_steps;    /* step count at last terminal (for Python export) */
    int              last_traj_records;     /* valid traj_log rows at last terminal (= steps+1
                                             * incl. the terminal record, capped at MAX_STEPS) */
    int              last_terminal_cause;   /* TERM_* code of the last episode's ending */

    /* ── T11: per-episode fuel budget ────────────────────────────────────────
     * fuel_frac_min/max < 0 (default) => the compile-time FUEL_FRAC, bit-exact.
     * Else f ~ U(min, max) per episode. dv_budget = -VE*ln(1-f), so
     * f = 0.113 -> 353 m/s and f = 0.20 -> 656 m/s.
     *
     * THE NORMALIZATION IS THE POINT. The success reward is
     * 10*(0.5 + 0.5*fuel_remaining) and `fuel_remaining` divides by the
     * INITIAL fuel. Left at the compile-time constant while the budget is
     * sampled, a 0.08 draw could never exceed 0.087/0.176 = 0.49 (success
     * capped at 7.5) while a 0.20 draw reached 1.0 after spending 29% of its
     * tank (a free 10) — and the policy SEES its fuel in obs[6], so it would
     * learn to prefer rich episodes instead of learning efficiency. All three
     * sites therefore divide by THIS EPISODE's budget. */
    double           fuel_frac_min;
    double           fuel_frac_max;
    double           episode_fuel_frac;     /* the draw in force this episode  */

    /* ── T11: the cell mixture (see the header block above) ──────────────── */
    int              cell_mixture_mode;     /* 0 = off, bit-inert, zero rand() */
    int              num_cells;
    int              last_cell;             /* index drawn this episode, for logs */
    double           cells[MAX_CELLS][CELL_FIELDS];
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

/* Convert true anomaly θ → mean anomaly M (inverse Kepler, exact).
 *
 * θ → E half-angle relation: tan(E/2) = sqrt((1−e)/(1+e)) · tan(θ/2), i.e.
 * E = 2·atan2(sqrt(1−e)·sin(θ/2), sqrt(1+e)·cos(θ/2)).
 *
 * BUG FIXED 2026-08-10: the factors were swapped (sqrt(1−e) on cos,
 * sqrt(1+e) on sin), which is the FORWARD map E→θ applied a second time
 * instead of its inverse. Since this function is reached only from
 * cartesian_to_elements() — i.e. immediately after a burn — the chaser's
 * mean anomaly was set with an error ≈ 2e·sin(θ), which materialized as an
 * along-track position glitch of up to 2e·a on the next propagation step
 * (~700 km measured at e=0.05, θ=90°, vs 0.1 m on coast steps). Burns at
 * apsides (sinθ≈0) were unaffected to first order, which is why the
 * Hohmann-style trained policy never exposed it. */
static inline double true_to_mean(double theta, double e) {
    /* θ → E via half-angle (inverse of eccentric_to_true) */
    double x = sqrt(1.0 + e) * cos(theta / 2.0);
    double y = sqrt(1.0 - e) * sin(theta / 2.0);
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

/* Wrap an angle into [0, 2π). Used only on the J2 path — the legacy propagator
 * open-codes the same two lines on M and must stay byte-identical. */
static inline double wrap_2pi(double x) {
    x = fmod(x, 2.0 * M_PI);
    if (x < 0.0) x += 2.0 * M_PI;
    return x;
}

/* ── ext-j2: propagate under SECULAR mean-element J2 ──────────────────────────
 * Binding spec: scripts/orbital/ext_recon/reports/j2_A_design.md §1.1-§1.3.
 *
 *   n  = sqrt(MU/a³)      p = a(1−e²)      k = 1.5·n·J2·(R_EQ/p)²
 *   Ω̇  = −k·cos i
 *   ω̇  = +0.5·k·(4 − 5 sin²i)
 *   Ṁ  =  n + 0.5·k·√(1−e²)·(2 − 3 sin²i)
 *   ȧ  = ė = i̇ = 0        (secular J2 has NO secular rate on a, e or i)
 *
 * WHY THIS FORM AND NOTHING MORE (§1.4). All short-period, m-daily and
 * long-period osculating terms are omitted, as are J2², J3+, drag, third-body
 * and SRP. Keeping only the secular rates is what buys the property this
 * project actually needs: the rates depend only on (a, e, i), which secular J2
 * leaves invariant, so they are CONSTANT for the life of an element set and the
 * map is closed-form at any dt. One warp of τ·DT therefore equals τ steps of DT
 * to float64 noise (measured ≤ 2.3e−8 m at every τ in the action table — the
 * same fmod-accumulation class the two-body path already shows), and the warp
 * actions stay exact instead of becoming an integration-error budget.
 *
 * TRUTH STATEMENT (§1.4), stated rather than corrected: under j2_mode = 1 the
 * env's state IS the mean element set. No mean↔osculating conversion happens
 * anywhere — not at reset, not at a burn, not at the success test. A burn is
 * applied to the mean state through the osculating Gauss response
 * (orbit_to_cartesian → +Δv → cartesian_to_elements), an O(J2) inconsistency.
 * Re-flown through a full-J2 Cowell integrator that reads these as osculating,
 * the RELATIVE state error is 83 m / 0.094 m/s per orbit at a 5 km separation
 * (1.66% of separation per orbit; the 30 km absolute divergence is common-mode
 * along-track and cancels) — benign where the success classifier looks, but at
 * the 5 km / 1 m/s box the velocity term is 9.4% of tolerance per orbit, so a
 * tight-box J2 rung must not also claim osculating-grade terminal fidelity.
 *
 * THE EQUATORIAL SPECIAL CASE (§1.3) is load-bearing, not tidiness. At i = 0, Ω
 * is a gauge coordinate but Ω̇ = −k is MAXIMAL; the physical rate is
 * ϖ̇ = Ω̇ + ω̇ = +k. Propagating Ω normally drives target.raan off 0.0 after one
 * step, which silently disengages gauge_from_orbit's `identity` fast path and
 * switches λ from the bit-exact M+ω form to the cartesian round trip mid-
 * episode. Conversely, "skip Ω̇ because Ω is gauge" while keeping ω̇ = 2k gives
 * ϖ̇ = 2k — wrong by exactly 2×. So: fold both rates into ω and leave raan at
 * 0.0. Cross-checked against a Cowell integrator that resolves the degeneracy
 * independently (1.5713e−6 vs 1.5586e−6 rad/s, 0.8%).
 *
 * j2_mode = 0 delegates to the verbatim legacy path by BRANCHING, never by
 * adding a zero — that is what makes the 2D/3D anchors bit-exact. */
static inline void propagate_orbit_j2(Orbit* o, double dt, int j2_mode) {
    if (!j2_mode) { propagate_orbit(o, dt); return; }

    double n   = sqrt(MU / (o->a * o->a * o->a));
    double p   = o->a * (1.0 - o->e * o->e);
    double rp  = J2_R_EQ / p;
    double k   = 1.5 * n * J2_COEF * rp * rp;
    double si  = sin(o->inc);
    double si2 = si * si;
    double Om  = -k * cos(o->inc);
    double om  =  0.5 * k * (4.0 - 5.0 * si2);
    double Md  =  n + 0.5 * k * sqrt(1.0 - o->e * o->e) * (2.0 - 3.0 * si2);

    if (o->inc == 0.0) {
        /* §1.3: Ω is gauge, ϖ̇ = Ω̇ + ω̇. raan MUST stay exactly 0.0 so the
         * identity gauge never disengages. */
        o->omega = wrap_2pi(o->omega + (om + Om) * dt);
    } else {
        o->raan  = wrap_2pi(o->raan  + Om * dt);
        o->omega = wrap_2pi(o->omega + om * dt);
    }

    /* Same ordering as the legacy path from here down. */
    o->M += Md * dt;
    o->M = fmod(o->M, 2.0 * M_PI);
    if (o->M < 0.0) o->M += 2.0 * M_PI;
    double E = solve_kepler(o->M, o->e);
    o->theta = eccentric_to_true(E, o->e);
}

/* ── ext-3d element combinations (the ONLY forms consumers may read) ────── */

/* Longitude of periapsis ϖ = ω + Ω. Well conditioned where ω and Ω are not. */
static inline double orb_varpi(const Orbit* o) { return o->omega + o->raan; }

/* Mean longitude λ = M + ω + Ω. Linear in time on a coast, burn-continuous. */
static inline double orb_lambda(const Orbit* o) { return o->M + o->omega + o->raan; }

/* Unit angular-momentum vector, 3-1-3 convention:
 *   ĥ = (sin i sin Ω, −sin i cos Ω, cos i)
 * ĥ is EXACTLY invariant under both legacy action axes (prograde: Δh ∥ h;
 * radial: Δh = 0), so the plane channels move only under a normal burn. */
static inline void orb_hhat(const Orbit* o, double* wx, double* wy, double* wz) {
    double si = sin(o->inc), ci = cos(o->inc);
    *wx =  si * sin(o->raan);
    *wy = -si * cos(o->raan);
    *wz =  ci;
}

/* Inertial eccentricity 3-vector built from ELEMENTS.
 * 3d_REDTEAM BLOCKER-2 variant V3: the Cartesian route ē = (v×h)/μ − r̂ is a
 * different FP path and breaks the A2 bit-exact anchor on 87.7% of draws; the
 * element route reduces to the legacy 2-vector (e cosω, e sinω) bit-exactly at
 * i = Ω = 0 (cosΩ=1, sinΩ=0, cos i=1, sin i=0 ⇒ x − 0.0 == x, 0.0 + x == x). */
static inline void orb_evec(const Orbit* o, double* ex, double* ey, double* ez) {
    double cO = cos(o->raan), sO = sin(o->raan);
    double cw = cos(o->omega), sw = sin(o->omega);
    double ci = cos(o->inc),   si = sin(o->inc);
    *ex = o->e * (cO*cw - sO*sw*ci);
    *ey = o->e * (sO*cw + cO*sw*ci);
    *ez = o->e * (sw*si);
}

/* Convert orbital elements (a, e, θ, ω, i, Ω) → Cartesian (x, y, z, vx, vy, vz).
 * Perifocal coordinates rotated by the 3-1-3 sequence R₃(−Ω)·R₁(−i)·R₃(−ω). */
static inline void orbit_to_cartesian(const Orbit* o,
                                       double* x, double* y, double* z,
                                       double* vx, double* vy, double* vz) {
    double p = o->a * (1.0 - o->e * o->e);     /* semi-latus rectum */
    double r = p / (1.0 + o->e * cos(o->theta));
    double h = sqrt(MU * p);                    /* specific angular momentum */

    /* Perifocal frame (periapsis at +x) */
    double xp  =  r * cos(o->theta);
    double yp  =  r * sin(o->theta);
    double vxp = -(MU / h) * sin(o->theta);
    double vyp =  (MU / h) * (o->e + cos(o->theta));

    if (o->inc == 0.0 && o->raan == 0.0) {
        /* ── 2D FAST PATH, value-gated. The four statements below are VERBATIM
         * from the pre-ext-3d build. The generic 3-1-3 block collapses to them
         * algebraically and was measured bit-exact at four optimisation
         * settings (3d_A §2.1, P1: 0/200000) — but a value gate makes the
         * anchor STRUCTURAL rather than empirical-on-one-toolchain, and it is
         * free speed for the 2D lineage. */
        double co = cos(o->omega), so = sin(o->omega);
        *x  = co * xp - so * yp;
        *y  = so * xp + co * yp;
        *vx = co * vxp - so * vyp;
        *vy = so * vxp + co * vyp;
        *z  = 0.0;
        *vz = 0.0;
        return;
    }

    double cO = cos(o->raan), sO = sin(o->raan);
    double cw = cos(o->omega), sw = sin(o->omega);
    double ci = cos(o->inc),   si = sin(o->inc);
    double R11 =  cO*cw - sO*sw*ci, R12 = -cO*sw - sO*cw*ci;
    double R21 =  sO*cw + cO*sw*ci, R22 = -sO*sw + cO*cw*ci;
    double R31 =  sw*si,            R32 =  cw*si;
    *x  = R11*xp  + R12*yp;
    *y  = R21*xp  + R22*yp;
    *z  = R31*xp  + R32*yp;
    *vx = R11*vxp + R12*vyp;
    *vy = R21*vxp + R22*vyp;
    *vz = R31*vxp + R32*vyp;
}

/* Convert Cartesian (x, y, z, vx, vy, vz) → orbital elements (a, e, θ, M, ω, i, Ω).
 * Used after applying an impulse burn.
 *
 * The `hxy == 0.0` test is EXACT, not a tolerance: with z = vz = 0 exactly,
 * hx = y·0.0 − 0.0·vy and hy = 0.0·vx − x·0.0 are each a difference of two
 * exact zeros, so the equatorial branch — which holds the verbatim legacy
 * statements — fires bit-deterministically. That is what closes the 2D
 * invariant (3d_A §5.1 step 5; measured 0 non-zero events in 200k draws,
 * 3d_REDTEAM n4). */
static inline void cartesian_to_elements(double x, double y, double z,
                                          double vx, double vy, double vz,
                                          Orbit* o) {
    double hx = y*vz - z*vy;
    double hy = z*vx - x*vz;
    double hz = x*vy - y*vx;
    double hxy = sqrt(hx*hx + hy*hy);

    if (hxy == 0.0) {
        /* ── EQUATORIAL (and the entire 2D lineage). Statements VERBATIM from
         * the pre-ext-3d build; two spellings in particular must not be
         * "improved" or the anchor drops from bit-exact to float-noise:
         *   · the e-vector as ((v²−μ/r)x − v_r·r·v_x)/μ — rewriting it as the
         *     algebraically identical (r̄·v̄)v̄ form costs 11175/200000
         *     mismatches (3d_A §5.1, P1b);
         *   · θ via acos + the v_r < 0 sign flip. atan2(r̄·q̂, r̄·ê) is four
         *     orders more accurate near apsides, so it is used in the INCLINED
         *     branch, where no anchor constrains it (3d_A P5).
         * Prograde-only by construction (trap T5): hz > 0 for every orbit this
         * env can reach — the sampler never sets i > 0 without dim3_mode, and
         * no affordable Δv budget reaches i = π. */
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

        o->inc  = 0.0;
        o->raan = 0.0;

        o->M = true_to_mean(o->theta, o->e);
        if (o->M < 0.0) o->M += 2.0 * M_PI;
        return;
    }

    /* ── INCLINED ─────────────────────────────────────────────────────────── */
    double r  = sqrt(x*x + y*y + z*z);
    double v2 = vx*vx + vy*vy + vz*vz;
    double rv = x*vx + y*vy + z*vz;
    double vr = rv / r;

    o->a = 1.0 / (2.0/r - v2/MU);

    double hmag = sqrt(hx*hx + hy*hy + hz*hz);
    /* atan2, NEVER acos(hz/hmag): acos loses half the mantissa near i ≈ 0 —
     * 1.0e-9 vs 4.1e-25 at i = 1e-9, a 2.4e15× degradation, which alone would
     * exceed the plane-invariance thresholds (3d_E F1, 3d_REDTEAM n5). */
    o->inc = atan2(hxy, hz);

    /* n̄ = ẑ × h̄ = (−hy, hx, 0) ⇒ Ω = atan2(hx, −hy). Never form n̄ and take
     * acos(n_x/|n|) (trap T2). hxy != 0 here, so no signed-zero branch. */
    double raan = atan2(hx, -hy);
    if (raan < 0.0) raan += 2.0 * M_PI;
    o->raan = raan;

    double ex = ((v2 - MU/r)*x - vr*r*vx) / MU;
    double ey = ((v2 - MU/r)*y - vr*r*vy) / MU;
    double ez = ((v2 - MU/r)*z - vr*r*vz) / MU;
    o->e = sqrt(ex*ex + ey*ey + ez*ez);

    double nx = -hy/hxy, ny = hx/hxy;                    /* n̂, n_z = 0 */
    double wx = hx/hmag, wy = hy/hmag, wz = hz/hmag;     /* ĥ          */
    double mx = -wz*ny,  my = wz*nx,   mz = wx*ny - wy*nx;  /* m̂ = ĥ × n̂ */

    if (o->e < 1e-10) {
        /* Circular-inclined: ω := 0 and θ := argument of latitude (trap T6). */
        o->omega = 0.0;
        double th = atan2(x*mx + y*my + z*mz, x*nx + y*ny);
        if (th < 0.0) th += 2.0 * M_PI;
        o->theta = th;
    } else {
        double w = atan2(ex*mx + ey*my + ez*mz, ex*nx + ey*ny);
        if (w < 0.0) w += 2.0 * M_PI;
        o->omega = w;
        double eux = ex/o->e, euy = ey/o->e, euz = ez/o->e;
        double qx = wy*euz - wz*euy;                     /* q̂ = ĥ × ê */
        double qy = wz*eux - wx*euz;
        double qz = wx*euy - wy*eux;
        double th = atan2(x*qx + y*qy + z*qz, x*eux + y*euy + z*euz);
        if (th < 0.0) th += 2.0 * M_PI;
        o->theta = th;
    }

    o->M = true_to_mean(o->theta, o->e);
    if (o->M < 0.0) o->M += 2.0 * M_PI;
}

/* ── Target-plane gauge (3d_REDTEAM MAJOR-1 fix (b)) ──────────────────────
 * Ω is measured from the inertial x̂, so when the two planes differ a global
 * SO(3) rotation moves Ω_s and Ω_t by DIFFERENT amounts and λ_s − λ_t is not a
 * function of the physical relative state — measured up to 59.8° of drift at
 * Δi = 1°, i.e. 0.33 Φ units, 8× the worst adverse step. The fix: express both
 * bodies in the target's own orbit frame (n̂_t, m̂_t, ĥ_t) first. The residual
 * gauge freedom is a rotation about ĥ_t, which shifts BOTH bodies' Ω by the
 * same amount and therefore cancels in the difference (measured invariant to
 * 2e-11°). At the design gauge i_t = Ω_t = 0 the frame is the identity and
 * this is the plain equinoctial λ = M + ω + Ω — bit-exactly. */
typedef struct { double e1[3], e2[3], e3[3]; int identity; } PlaneGauge;

static inline void gauge_from_orbit(const Orbit* t, PlaneGauge* g) {
    if (t->inc == 0.0 && t->raan == 0.0) {
        g->identity = 1;
        return;
    }
    g->identity = 0;
    double wx, wy, wz;
    orb_hhat(t, &wx, &wy, &wz);
    double nx = -wy, ny = wx;
    double nn = sqrt(nx*nx + ny*ny);
    if (nn > 1e-14) { g->e1[0] = nx/nn; g->e1[1] = ny/nn; g->e1[2] = 0.0; }
    else            { g->e1[0] = 1.0;   g->e1[1] = 0.0;   g->e1[2] = 0.0; }
    g->e3[0] = wx; g->e3[1] = wy; g->e3[2] = wz;
    g->e2[0] = g->e3[1]*g->e1[2] - g->e3[2]*g->e1[1];
    g->e2[1] = g->e3[2]*g->e1[0] - g->e3[0]*g->e1[2];
    g->e2[2] = g->e3[0]*g->e1[1] - g->e3[1]*g->e1[0];
}

/* ϖ = ω + Ω expressed in the target-plane gauge. Position-independent, so the
 * c_reset phase-gap patch that adds (ϖ_s − ϖ_t) to the target's M is exact. */
static inline double orb_varpi_gauge(const Orbit* o, const PlaneGauge* g) {
    if (g->identity) return o->omega + o->raan;
    double x, y, z, vx, vy, vz;
    orbit_to_cartesian(o, &x, &y, &z, &vx, &vy, &vz);
    double X  =  x*g->e1[0] +  y*g->e1[1] +  z*g->e1[2];
    double Y  =  x*g->e2[0] +  y*g->e2[1] +  z*g->e2[2];
    double Z  =  x*g->e3[0] +  y*g->e3[1] +  z*g->e3[2];
    double VX = vx*g->e1[0] + vy*g->e1[1] + vz*g->e1[2];
    double VY = vx*g->e2[0] + vy*g->e2[1] + vz*g->e2[2];
    double VZ = vx*g->e3[0] + vy*g->e3[1] + vz*g->e3[2];
    Orbit t;
    cartesian_to_elements(X, Y, Z, VX, VY, VZ, &t);
    return t.omega + t.raan;
}

/* λ in the target-plane gauge. M is a time coordinate and therefore frame
 * invariant, so it is taken from the element set directly rather than through
 * the round trip. At identity gauge this is M + (ω + 0.0) == M + ω bitwise. */
static inline double orb_lambda_gauge(const Orbit* o, const PlaneGauge* g) {
    return o->M + orb_varpi_gauge(o, g);
}

/* Apply an impulsive Δv in the satellite's local orbital frame.
 * dv_pro = prograde (along velocity), dv_rad = radial out, dv_nor = normal (out-of-plane, 2D→ignored).
 * Updates satellite orbital elements and consumes fuel (Tsiolkovsky).
 * Returns actual |Δv| applied (may be less than requested if fuel limited). */
static inline double apply_impulse(Orbital* env,
                                    double dv_pro, double dv_rad,
                                    double dv_nor) {
    Satellite* sat = &env->sat;
    double x, y, z, vx, vy, vz;
    orbit_to_cartesian(&sat->orbit, &x, &y, &z, &vx, &vy, &vz);

    /* Local frame unit vectors. p̂ and r̂ are NOT orthogonal at e > 0 — that is
     * pre-existing and deliberate (trap T15). n̂ = ĥ IS exactly orthogonal to
     * both at any e, so adding the normal axis perturbs no existing action
     * semantics (3d_E F4). |Δv| is the norm of the ASSEMBLED vector, never a
     * sum or quadrature of components, or the fuel ledger is wrong. */
    double v_mag = sqrt(vx*vx + vy*vy + vz*vz);
    double r_mag = sqrt(x*x + y*y + z*z);
    double pro_x = vx / v_mag;   /* prograde = velocity direction */
    double pro_y = vy / v_mag;
    double pro_z = vz / v_mag;
    double rad_x =  x / r_mag;   /* radial = position direction */
    double rad_y =  y / r_mag;
    double rad_z =  z / r_mag;
    double hx = y*vz - z*vy, hy = z*vx - x*vz, hz = x*vy - y*vx;
    double h_mag = sqrt(hx*hx + hy*hy + hz*hz);
    double nor_x = hx / h_mag;   /* +normal = +ĥ = orbit north (trap T14) */
    double nor_y = hy / h_mag;
    double nor_z = hz / h_mag;

    /* Requested Δv in inertial frame. In-plane terms summed FIRST, normal last,
     * so that with dv_nor == 0.0 and an equatorial orbit every component is
     * bitwise the legacy value (a + ±0.0 == a; z = vz = 0 ⇒ hx = hy = 0 ⇒
     * nor = (0,0,±1) ⇒ dvz = ±0.0 ⇒ vz stays exactly 0.0). */
    double dvx = dv_pro * pro_x + dv_rad * rad_x + dv_nor * nor_x;
    double dvy = dv_pro * pro_y + dv_rad * rad_y + dv_nor * nor_y;
    double dvz = dv_pro * pro_z + dv_rad * rad_z + dv_nor * nor_z;
    double dv_mag = sqrt(dvx*dvx + dvy*dvy + dvz*dvz);

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
        dvz *= scale;
        dv_mag = actual_dv;
        sat->fuel_mass = 0.0;
    } else {
        sat->fuel_mass -= fuel_needed;
        if (sat->fuel_mass < 0.0) sat->fuel_mass = 0.0;
    }

    /* Apply velocity change */
    vx += dvx;
    vy += dvy;
    vz += dvz;

    /* Convert back to orbital elements */
    cartesian_to_elements(x, y, z, vx, vy, vz, &sat->orbit);
    env->total_dv_used += dv_mag;

    /* ext-3d: snapshot the post-impulse / pre-propagation state for the
     * trajectory log (the 3D invariant battery needs the true rv_post). */
    env->last_burn_post[0] = x;  env->last_burn_post[1] = y;  env->last_burn_post[2] = z;
    env->last_burn_post[3] = vx; env->last_burn_post[4] = vy; env->last_burn_post[5] = vz;
    env->last_burn_valid   = 1;

    return dv_mag;
}

/* ── Phase 5d I2: action-mask preview ────────────────────────────────────
 * For a candidate burn (dv_pro, dv_rad), compute the post-burn perigee
 * without mutating the satellite. Returns r_p = a(1-e), or -1.0 if the burn
 * would put the satellite on a hyperbolic trajectory.
 */
static inline double preview_perigee(const Satellite* sat,
                                      double dv_pro, double dv_rad) {
    double x, y, z, vx, vy, vz;
    orbit_to_cartesian(&sat->orbit, &x, &y, &z, &vx, &vy, &vz);
    double v_mag = sqrt(vx*vx + vy*vy + vz*vz);
    double r_mag = sqrt(x*x + y*y + z*z);
    if (v_mag < 1e-9 || r_mag < 1e-9) return -1.0;
    double pro_x = vx / v_mag, pro_y = vy / v_mag, pro_z = vz / v_mag;
    double rad_x =  x / r_mag, rad_y =  y / r_mag, rad_z =  z / r_mag;
    double dvx = dv_pro * pro_x + dv_rad * rad_x;
    double dvy = dv_pro * pro_y + dv_rad * rad_y;
    double dvz = dv_pro * pro_z + dv_rad * rad_z;
    double new_vx = vx + dvx, new_vy = vy + dvy, new_vz = vz + dvz;
    Orbit po;
    cartesian_to_elements(x, y, z, new_vx, new_vy, new_vz, &po);
    if (po.a <= 0.0) return -1.0;  /* hyperbolic — invalid */
    return po.a * (1.0 - po.e);
}

/* ── Observation packing ─────────────────────────────────────────────────── */

/* Get Cartesian position of a body (static bodies are at origin).
 * Bodies (Earth + debris) are equatorial by construction — c_reset writes
 * inc = raan = 0.0 for every one — so bz is identically 0. It is still
 * returned and used in every distance test, because omitting z anywhere turns
 * the success sphere into a cylinder (trap T17). */
static inline void body_position(const Body* b, double* bx, double* by, double* bz) {
    if (b->is_static) {
        *bx = 0.0;
        *by = 0.0;
        *bz = 0.0;
    } else {
        double bvx, bvy, bvz;
        orbit_to_cartesian(&b->orbit, bx, by, bz, &bvx, &bvy, &bvz);
    }
}

/* Clamp an observation channel to the declared Box(-2, 2). Applied to every
 * ext-3d slot (3d_REDTEAM m3: the Δv-ledger channels were measured at 3.24 at
 * the widest rung, i.e. outside the space the trainer is told about). */
static inline float obs_clamp2(double v) {
    if (!(v > -2.0)) return (v > 0.0) ? 2.0f : -2.0f;  /* also traps NaN */
    if (v > 2.0) return 2.0f;
    return (float)v;
}

static inline void fill_observations(Orbital* env) {
    float* obs = env->observations;
    const Satellite* sat = &env->sat;
    /* Phase 5.5: observation altitude scale is configurable via obs_alt_scale_m
     * (default 1.6e6 = ALT_MAX, preserving LEO-trained ckpt compatibility). */
    const double scale_a    = env->obs_alt_scale_m;             /* for altitudes → [0,1]  */
    const double scale_dist = R_EARTH + env->obs_alt_scale_m;    /* for distances → [0,~2] */

    /* Satellite Cartesian state for velocity decomposition.
     * NOTE (3d_REDTEAM MAJOR-2 option (a)): slots 0-16 and 33-37 keep their
     * exact pre-ext-3d statements — including the in-plane (x, y) projections
     * below — so every validated downstream decoder (t3/redteam/rt_common.py
     * ObsView, nav/orbital_math.py decode_obs, nav/eval_relnav.py build_obs_t3)
     * and the 100/100 scripted expert keep working unmodified. The out-of-plane
     * state is carried by the repurposed dead body slots 21-32 instead. */
    double sx, sy, sz, svx, svy, svz;
    orbit_to_cartesian(&sat->orbit, &sx, &sy, &sz, &svx, &svy, &svz);

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
    if (env->phase_obs_mode == 1) {
        /* T3: mean-longitude phase channels. The legacy true-anomaly gap is a
         * per-body coordinate: its sign disagrees with the physical gap on 39%
         * of e>0 steps and a 1 m/s burn at e≈0 teleports it ~86° (recon
         * ANOM-1/2). λ = M + ω is sign-correct and burn-continuous.
         *
         * obs[15,16] (red-team #11): the legacy sin/cos(θ_t) — and sin/cos(λ_t)
         * — are rotation-only phases carrying zero task information (env is
         * rotation-invariant). Reused for the two things the policy actually
         * lacks: the episode clock (bit-identical observations were measured
         * 600 vs 1800 sub-steps from the cap — even the 99.2% expert fails
         * clock-blind) and the apsidal alignment cos(ω_s − ω_t) (pairs with
         * obs[9-12]; the driver of Δē-matching burn timing). */
        /* ext-3d: under dim3_mode=1 λ is the 3D mean longitude in the
         * target-plane gauge. Identical value at i = Ω = 0 by construction
         * (gauge = identity, ω + 0.0 == ω), so the 2D lineage is unchanged. */
        double lam_s, lam_t;
        if (env->dim3_mode) {
            PlaneGauge g;
            gauge_from_orbit(&env->target, &g);
            lam_s = orb_lambda_gauge(&sat->orbit, &g);
            lam_t = orb_lambda_gauge(&env->target, &g);
        } else {
            lam_s = sat->orbit.M + sat->orbit.omega;
            lam_t = env->target.M + env->target.omega;
        }
        double dlam_phase = lam_s - lam_t;
        double t_frac = (double)(env->episode_cap_steps - env->step)
                        / (double)env->episode_cap_steps;
        if (t_frac < 0.0) t_frac = 0.0;
        obs[13] = (float)sin(dlam_phase);
        obs[14] = (float)cos(dlam_phase);
        obs[15] = (float)t_frac;
        obs[16] = (float)cos(sat->orbit.omega - env->target.omega);
    } else {
        double dtheta_phase = sat->orbit.theta - env->target.theta;
        obs[13] = (float)sin(dtheta_phase);
        obs[14] = (float)cos(dtheta_phase);
        obs[15] = (float)sin(env->target.theta);
        obs[16] = (float)cos(env->target.theta);
    }

    /* [17-32] Closest N_OBS_BODY bodies: Δr, Δθ, closing_rate, keepout_r
     * First compute distances to all bodies, then pick the N closest. */
    double dist[MAX_BODIES];
    for (int i = 0; i < env->num_bodies; i++) {
        double bx, by, bz;
        body_position(&env->bodies[i], &bx, &by, &bz);
        double dx = sx - bx, dy = sy - by, dz = sz - bz;
        dist[i] = sqrt(dx*dx + dy*dy + dz*dz);
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

        double bx, by, bz, bvx = 0.0, bvy = 0.0, bvz = 0.0;
        if (b->is_static) {
            bx = 0.0; by = 0.0; bz = 0.0;
        } else {
            orbit_to_cartesian(&b->orbit, &bx, &by, &bz, &bvx, &bvy, &bvz);
        }

        double dx = sx - bx, dy = sy - by, dz = sz - bz;
        double dr = dist[i];

        /* Relative bearing: angle difference */
        double sat_angle  = atan2(sy, sx);
        double body_angle = atan2(by, bx);
        double dtheta = sat_angle - body_angle;
        /* Normalize to [-π, π] */
        while (dtheta >  M_PI) dtheta -= 2.0 * M_PI;
        while (dtheta < -M_PI) dtheta += 2.0 * M_PI;

        /* Closing rate: d/dt(|r_sat - r_body|) = (r̂ · (v_sat - v_body)) */
        double closing = ((dx*(svx-bvx) + dy*(svy-bvy) + dz*(svz-bvz)) / dr);

        obs[base]   = (float)(dr / scale_dist);
        obs[base+1] = (float)(dtheta / M_PI);
        obs[base+2] = (float)(closing / v_circ);
        obs[base+3] = (float)(b->keepout_radius / scale_dist);
    }

    /* ── [21-32] ext-3d block. OBS REPURPOSING, not relayout (3d_REDTEAM
     * MAJOR-2, option (a)): slots 21-32 were measured identically zero at the
     * project's standing no-debris configuration, so the 3D state goes there
     * and slots 0-16 / 33-37 keep their semantics — which preserves every
     * validated decoder, the scripted expert, the EKF injector, and the 2D→3D
     * warm-start. PRECONDITION: dim3_mode = 1 requires num_debris = 0 (the
     * project standard since the no-debris decision); with debris enabled these
     * writes overwrite bodies 1-3 of the conjunction block. */
    if (env->dim3_mode) {
        double hsx, hsy, hsz, htx, hty, htz;
        orb_hhat(&sat->orbit,  &hsx, &hsy, &hsz);
        orb_hhat(&env->target, &htx, &hty, &htz);

        /* Relative-inclination vector δı⃗ = Δi_rel·n̂, n̂ = (ĥ_t × ĥ_s)/|·|.
         * Continuous through Δi = 0 (the vector → 0⃗; only its derivative has a
         * corner, bounded by the norm's Lipschitz constant) — unlike i and Ω
         * separately, which are singular there. */
        double cx = hty*hsz - htz*hsy;
        double cy = htz*hsx - htx*hsz;
        double cz = htx*hsy - hty*hsx;
        double cn = sqrt(cx*cx + cy*cy + cz*cz);
        double hdot = htx*hsx + hty*hsy + htz*hsz;
        double di_rel = atan2(cn, hdot);
        double dix = 0.0, diy = 0.0, diz = 0.0;
        if (cn > 1e-300) {
            dix = di_rel * cx / cn; diy = di_rel * cy / cn; diz = di_rel * cz / cn;
        }

        /* Chaser RTN: R̂ = r̂_s, N̂ = ĥ_s, T̂ = N̂ × R̂. Projecting δı⃗ here (not
         * into the target's frame) makes atan2(δı⃗·T̂, δı⃗·R̂) exactly "how far
         * the chaser must coast to reach the relative node" — the burn-timing
         * signal, with no explicit node detector (3d_C §1). */
        double rmag3 = sqrt(sx*sx + sy*sy + sz*sz);
        double Rx = sx/rmag3, Ry = sy/rmag3, Rz = sz/rmag3;
        double Tx = hsy*Rz - hsz*Ry, Ty = hsz*Rx - hsx*Rz, Tz = hsx*Ry - hsy*Rx;

        double di_scale = (env->obs_di_scale_rad > 0.0)
                          ? env->obs_di_scale_rad
                          : fmax((env->di_max_rad > 0.0 ? env->di_max_rad : 0.0),
                                 0.25 * M_PI / 180.0);
        double de_scale = (env->obs_de_scale > 0.0)
                          ? env->obs_de_scale
                          : fmax((env->de_max > 0.0 ? env->de_max : 0.0), 0.05);

        obs[21] = obs_clamp2((dix*Rx + diy*Ry + diz*Rz) / di_scale);
        obs[22] = obs_clamp2((dix*Tx + diy*Ty + diz*Tz) / di_scale);

        double esx, esy, esz, etx, ety, etz;
        orb_evec(&sat->orbit,  &esx, &esy, &esz);
        orb_evec(&env->target, &etx, &ety, &etz);
        double dex = esx - etx, dey = esy - ety, dez = esz - etz;
        obs[23] = obs_clamp2((dex*htx + dey*hty + dez*htz) / de_scale);

        /* Cross-track LVLH pair, completing obs[33-36]'s in-plane block.
         * ω⃗_LVLH ∥ N̂_t, so (ω⃗ × ρ⃗)·N̂_t ≡ 0 and the frame-rotation correction
         * drops out of the N component exactly. */
        double tx3, ty3, tz3, tvx3, tvy3, tvz3;
        orbit_to_cartesian(&env->target, &tx3, &ty3, &tz3, &tvx3, &tvy3, &tvz3);
        double rho_N  = (sx - tx3)*htx + (sy - ty3)*hty + (sz - tz3)*htz;
        double rhod_N = (svx - tvx3)*htx + (svy - tvy3)*hty + (svz - tvz3)*htz;
        double v_c_t  = sqrt(MU / env->target.a);
        obs[24] = obs_clamp2(rho_N / env->lvlh_scale_m);
        obs[25] = obs_clamp2(rhod_N / v_c_t);

        /* Δv ledger, in the same units Φ(mode 2) uses: one obs unit per dv_ref
         * of remaining Δv. obs[28] is the feasibility margin — negative means
         * the tank cannot pay for the geometry. */
        double dv_ref = (env->shape_dv_ref_ms > 0.0) ? env->shape_dv_ref_ms : 300.0;
        double dhx = hsx - htx, dhy = hsy - hty, dhz = hsz - htz;
        double dv_pl = v_c_t * sqrt(dhx*dhx + dhy*dhy + dhz*dhz);
        double da_rel3 = (sat->orbit.a - env->target.a) / env->target.a;
        double de3 = sqrt(dex*dex + dey*dey + dez*dez);
        double dv_in = 0.5 * v_c_t * sqrt(da_rel3*da_rel3 + de3*de3);
        double dv_rem = VE * log((sat->dry_mass + sat->fuel_mass) / sat->dry_mass);
        obs[26] = obs_clamp2(dv_pl  / dv_ref);
        obs[27] = obs_clamp2(dv_rem / dv_ref);
        obs[28] = obs_clamp2((dv_rem - dv_pl - dv_in) / dv_ref);
        /* ── ext-j2 obs (j2_A_design §4.3, correcting 3d_C §2's stale "obs
         * 28-29 reserved" — obs[28] is the Δv feasibility margin, so the free
         * reserved block is 29-32).
         *
         * J2 is AXISYMMETRIC, so Ω remains a gauge coordinate and the ONLY new
         * physical information the equator adds is cos i. Two slots, not four:
         * the absolute Ω of either body is still unobservable-by-design, and
         * feeding it would reintroduce the rotation-variance the target-plane
         * gauge exists to remove.
         *
         * Written only under j2_mode = 1 so the X3 / T3 / legacy checkpoint
         * anchors stay bit-exact at the default (these columns are random-init
         * and zero-gradient in every shipped checkpoint — n3d_REDTEAM
         * MAJOR-14/NON-ISSUE-9 — so a nonzero write at j2_mode = 0 would be a
         * silent perturbation of a trained encoder).
         *
         * obs[31], obs[32] STAY RESERVED. The design flags Ω̇_s/|Ω̇|_ref as the
         * exact quantity the go-around decision keys on, but recommends
         * deferring it to a second arm so the attribution of any J2 result
         * stays clean; obs[32] has no assignment at all. */
        if (env->j2_mode) {
            obs[29] = obs_clamp2(cos(sat->orbit.inc));
            obs[30] = obs_clamp2(cos(env->target.inc));
        } else {
            obs[29] = 0.0f;
            obs[30] = 0.0f;
        }
        obs[31] = 0.0f;  /* reserved (J2 rate channel, deferred) */
        obs[32] = 0.0f;  /* reserved */
    }

    /* [33-37] LVLH-frame relative state — primary observation for rendezvous.
     * Rotating frame co-moving with target: x-axis = radial (outward from
     * Earth through target), y-axis = along-track (in direction of target
     * motion). Makes the rendezvous problem state-invariant to target's
     * inertial angular position, which is the key observation-aliasing fix
     * from Phase 4 spec §3.1. */
    {
        double tx, ty, tz, tvx, tvy, tvz;
        orbit_to_cartesian(&env->target, &tx, &ty, &tz, &tvx, &tvy, &tvz);
        /* tz/tvz are used only by lvlh_frame_mode = 1 below; the legacy
         * in-plane block deliberately ignores them (cross-track is obs[24,25]). */
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

        /* ext-j2 rung, lvlh_frame_mode = 1: the TRUE target orbital frame.
         * The block above projects onto the EQUATORIAL plane and rotates by
         * the in-plane angle u = omega + theta, which coincides with LVLH only
         * at i_t = Omega_t = 0. At i_t != 0 it is wrong twice over — the
         * along-track axis picks up a cos i_t squash, and the whole thing
         * rotates with Omega_t instead of with the scene, so it is not even
         * invariant under the SO(2)-about-ẑ symmetry that J2 still has. The
         * inclined-target sampler makes both live, so the corrected frame is
         * available here; default 0 keeps every trained checkpoint's primary
         * rendezvous channel bit-identical. */
        if (env->lvlh_frame_mode == 1) {
            double rtx, rty, rtz, htx2, hty2, htz2;
            double rn = sqrt(tx*tx + ty*ty + tz*tz);
            rtx = tx/rn; rty = ty/rn; rtz = tz/rn;
            orb_hhat(&env->target, &htx2, &hty2, &htz2);
            double ttx = hty2*rtz - htz2*rty;
            double tty = htz2*rtx - htx2*rtz;
            double ttz = htx2*rty - hty2*rtx;
            double dzi  = sz  - tz;
            double dvzi = svz - tvz;
            dx_l  = dxi*rtx  + dyi*rty  + dzi*rtz;
            dy_l  = dxi*ttx  + dyi*tty  + dzi*ttz;
            dvx_l = dvxi*rtx + dvyi*rty + dvzi*rtz;
            dvy_l = dvxi*ttx + dvyi*tty + dvzi*ttz;
        }

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
/* Wrap an angle to [−π, π). */
static inline double wrap_pi(double x) {
    x = fmod(x + M_PI, 2.0 * M_PI);
    if (x < 0.0) x += 2.0 * M_PI;
    return x - M_PI;
}

static inline double compute_phi(const Orbital* env) {
    /* T3 shaping_mode 1 — S-R3 "phase-time" potential (T3_RECOVERY_CAMPAIGN.md
     * §5; measured design comparison in recon_shaping_audit.md §1.2/§3).
     *   Φ = −[ W_λ·|Δλ|/π + W_m·min(1, Δv_match / DV_REF) ],  Φ ∈ [−(W_λ+W_m), 0]
     * λ = M + ω is MEAN longitude: linear in time on a coast (no equation-of-
     * centre ripple), continuous across burns up to O(2Δe), and — unlike the
     * legacy true-anomaly gap — measures the physically meaningful along-track
     * separation (recon ANOM-1/2). Δv_match is the linearised two-impulse
     * orbit-match cost (Edelbaum/Gauss); hypot avoids double-counting because
     * the burn pair that removes a phasing orbit's Δa also removes the e it
     * created. No gates: the drift leg earns phase progress densely instead of
     * being penalized (legacy σ₂ was dead for 100% of viable drift orbits). */
    if (env->shaping_mode == 1) {
        double dlam = wrap_pi((env->sat.orbit.M + env->sat.orbit.omega)
                            - (env->target.M   + env->target.omega));
        double e_sx = env->sat.orbit.e * cos(env->sat.orbit.omega);
        double e_sy = env->sat.orbit.e * sin(env->sat.orbit.omega);
        double e_tx = env->target.e    * cos(env->target.omega);
        double e_ty = env->target.e    * sin(env->target.omega);
        double de   = sqrt((e_sx - e_tx)*(e_sx - e_tx) + (e_sy - e_ty)*(e_sy - e_ty));
        double da_rel    = (env->sat.orbit.a - env->target.a) / env->target.a;
        double v_t       = sqrt(MU / env->target.a);
        double dv_match  = 0.5 * v_t * sqrt(da_rel*da_rel + de*de);
        double match     = dv_match / env->shape_dv_ref_ms;
        if (match > 1.0) match = 1.0;
        return -(env->shape_w_lambda * fabs(dlam) / M_PI
               + env->shape_w_match  * match);
    }

    /* ── ext-3d shaping_mode 2 — the 3D lift of mode 1 ────────────────────
     *   Δv_in = 0.5·v_t·sqrt(δa_rel² + ‖Δē₃‖²)        (in-plane, unchanged)
     *   Δv_pl = 1.0·v_t·‖ĥ_s − ĥ_t‖ = 2·v_t·sin(Δi/2) (exact single-impulse chord)
     *   Δv₃   = Δv_in + Δv_pl                          (L1, NOT hypot-3)
     *   Φ     = −[W_λ·|Δλ|/π + W_m·squash(Δv₃/dv_ref)]
     *
     * Coefficient 1.0 on the plane term, not 0.5 (3d_C §3): the tangential axis
     * has a Gauss lever of 2 on both δa and δē — two impulses at opposite u
     * trade one against the other, which is what the shared 0.5 encodes — while
     * the normal axis has a lever of 1 and no such trade. The rule is
     * "coefficient = 1/(best lever)", so Φ reads in true m/s of Δv-to-go and one
     * m/s of correctly-aimed burn buys exactly one unit, on either axis.
     *
     * L1 rather than hypot-3 because ∂hypot/∂x = x/‖·‖ attenuates the in-plane
     * gradient 4.8× whenever the plane error is large — a soft gate, i.e. the
     * project's defect class #1 in continuous clothing. L1 keeps ∂Φ/∂Δv equal
     * on both axes. It over-counts a combined burn by up to 1.41×, which with
     * actions 26-29 in the table is now a realizable bonus for the physically
     * optimal maneuver, and is unfarmable under shape_gamma = 1 (the episode
     * total telescopes to Φ_T − Φ₀ regardless of path).
     *
     * BIT-EXACT REDUCTION TO MODE 1 AT Δi = 0 (anchor A2, verified 0/200000):
     * identity gauge ⇒ λ is M + ω; element-derived ē collapses to (e cosω,
     * e sinω) with a +0.0 third component that squares to +0.0; ĥ_s − ĥ_t = 0⃗
     * ⇒ Δv_pl is exactly +0.0 ⇒ Δv_in + 0.0 == Δv_in. Hence `sqrt(x*x + y*y)`
     * and NOT `hypot` — hypot() is a different FP path and mismatched 87.7% of
     * draws in the red-team's probe (3d_REDTEAM BLOCKER-2). */
    if (env->shaping_mode == 2) {
        PlaneGauge g;
        gauge_from_orbit(&env->target, &g);
        double dlam = wrap_pi(orb_lambda_gauge(&env->sat.orbit, &g)
                            - orb_lambda_gauge(&env->target,    &g));

        double e_sx, e_sy, e_sz, e_tx, e_ty, e_tz;
        orb_evec(&env->sat.orbit, &e_sx, &e_sy, &e_sz);
        orb_evec(&env->target,    &e_tx, &e_ty, &e_tz);
        double dex = e_sx - e_tx, dey = e_sy - e_ty, dez = e_sz - e_tz;
        double de  = sqrt(dex*dex + dey*dey + dez*dez);

        double da_rel = (env->sat.orbit.a - env->target.a) / env->target.a;
        double v_t    = sqrt(MU / env->target.a);
        double dv_in  = 0.5 * v_t * sqrt(da_rel*da_rel + de*de);

        double hsx, hsy, hsz, htx, hty, htz;
        orb_hhat(&env->sat.orbit, &hsx, &hsy, &hsz);
        orb_hhat(&env->target,    &htx, &hty, &htz);
        double dhx = hsx - htx, dhy = hsy - hty, dhz = hsz - htz;
        double dv_pl = 1.0 * v_t * sqrt(dhx*dhx + dhy*dhy + dhz*dhz);

        double match = (dv_in + dv_pl) / env->shape_dv_ref_ms;
        if (env->shape_match_squash == 1) {
            match = match / (1.0 + match);   /* bounded, strictly monotone, no dead zone */
        } else if (match > 1.0) {
            match = 1.0;
        }
        return -(env->shape_w_lambda * fabs(dlam) / M_PI
               + env->shape_w_match  * match);
    }

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
    double sx, sy, sz, svx, svy, svz;
    orbit_to_cartesian(&env->sat.orbit, &sx, &sy, &sz, &svx, &svy, &svz);
    double tx, ty, tz, tvx, tvy, tvz;
    orbit_to_cartesian(&env->target, &tx, &ty, &tz, &tvx, &tvy, &tvz);
    (void)sz; (void)svz; (void)tz; (void)tvz;   /* legacy Φ is in-plane by definition */
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
            /* T3 red-team #5: under shaping_mode 1 the clamp is OFF at  \
             * every terminal ("clamp-nowhere"). Paying +|Φ_prev| at a   \
             * terminal rewards dying FAR from the goal (+0.54 mean /    \
             * +1.21 max premium measured for collision/stranded vs a    \
             * near-goal death). With the bounded S-R3 potential the     \
             * telescoped total is Φ_T − Φ₀ for every episode — progress \
             * credit with no perverse refunds. Legacy mode unchanged.   \
             * c_reset re-derives phi_prev.                              \
             * ext-3d: "clamp-nowhere" carries forward to mode 2 verbatim \
             * (3d_C §4.2) — under the plane term a normal burn that      \
             * INCREASES Δi and then dies would otherwise be refunded.    \
             * It is also required for anchor A2: with the clamp live in  \
             * mode 2 but dead in mode 1, 0.68% of one-step rewards       \
             * diverged by up to 1.49 at the terminals (measured).        */ \
            if (env->shaping_mode >= 1) break;                           \
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

    /* ext-3d trap T17: EVERY distance below must pick up z. Omitting it
     * anywhere turns the rendezvous sphere into a cylinder and awards success
     * to trajectories kilometres out of plane. All of these reduce bitwise at
     * z = vz = 0 (x*x + y*y + 0.0 == x*x + y*y). */
    double sx, sy, sz, svx, svy, svz;
    orbit_to_cartesian(&sat->orbit, &sx, &sy, &sz, &svx, &svy, &svz);
    double r = sqrt(sx*sx + sy*sy + sz*sz);

    /* 1. Collision with any body */
    for (int i = 0; i < env->num_bodies; i++) {
        double bx, by, bz;
        body_position(&env->bodies[i], &bx, &by, &bz);
        double dx = sx - bx, dy = sy - by, dz = sz - bz;
        double d = sqrt(dx*dx + dy*dy + dz*dz);
        if (d < env->bodies[i].hard_radius) {
            env->rewards[0]   = -10.0f;
            env->terminals[0] = 1;
            env->last_terminal_cause = TERM_COLLISION;
            R2_NHR_CLAMP();
            return 1;
        }
    }

    /* 2. Escape trajectory: specific orbital energy E = ½v² - μ/r ≥ 0 */
    double v2 = svx*svx + svy*svy + svz*svz;
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
    if (env->step >= env->episode_cap_steps) {
        /* T3 red-team #1 (BLOCKER): under flat per-decision γ a −10 cap
         * terminal prices warp-heavy play as a −7.8 bet while coast-to-cap
         * discount-hides to −0.0 (3000 decisions, γ^3000 ≈ 0) — PPO suppresses
         * warps before the first success is ever sampled, which alone
         * reproduces the post-fix flatline. cap_terminal_reward = 0.0 for T3
         * runs drops the warp break-even from 46.5% success to 0%. Default
         * −10.0 (set by the Python binding) = legacy. */
        env->rewards[0]   = (float)env->cap_terminal_reward;
        env->terminals[0] = 1;
        env->last_terminal_cause = TERM_SAFETY_CAP;
        env->phi_prev     = 0.0;   /* c_reset re-derives; keep state consistent */
        return 1;
    }

    /* 4. Stranded / 5. Success — Phase 3 rendezvous check.
     * Success requires BOTH position and relative-velocity match with the
     * propagated target body, not just orbit-shape (a, ē) matching. */
    double tx, ty, tz, tvx, tvy, tvz;
    orbit_to_cartesian(&env->target, &tx, &ty, &tz, &tvx, &tvy, &tvz);
    double dx = sx - tx, dy = sy - ty, dz = sz - tz;
    double dist_to_target = sqrt(dx*dx + dy*dy + dz*dz);
    double rvx = svx - tvx, rvy = svy - tvy, rvz = svz - tvz;
    double rel_vel = sqrt(rvx*rvx + rvy*rvy + rvz*rvz);
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
        /* T11: THIS EPISODE's budget, not the compile-time constant — under
         * per-episode sampling the constant turns the drawn budget into a
         * reward multiplier (a lean draw could not reach 10, a rich one got it
         * for free), and obs[6] makes that visible to the policy. */
        double _ff = (env->episode_fuel_frac > 0.0)
                     ? env->episode_fuel_frac : FUEL_FRAC;
        double initial_fuel = sat->dry_mass * _ff / (1.0 - _ff);
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

    double sx, sy, sz, svx, svy, svz;
    orbit_to_cartesian(&sat->orbit, &sx, &sy, &sz, &svx, &svy, &svz);

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
    double tx, ty, tz, tvx, tvy, tvz;
    orbit_to_cartesian(&env->target, &tx, &ty, &tz, &tvx, &tvy, &tvz);
    rec->target_x  = (float)tx;
    rec->target_y  = (float)ty;
    rec->target_vx = (float)tvx;
    rec->target_vy = (float)tvy;

    /* ext-3d appended columns */
    rec->sat_z       = (float)sz;
    rec->sat_vz      = (float)svz;
    rec->sat_inc     = (float)sat->orbit.inc;
    rec->sat_raan    = (float)sat->orbit.raan;
    rec->target_z    = (float)tz;
    rec->target_vz   = (float)tvz;
    rec->target_inc  = (float)env->target.inc;
    rec->target_raan = (float)env->target.raan;
    if (env->last_burn_valid) {
        rec->burn_post_x  = (float)env->last_burn_post[0];
        rec->burn_post_y  = (float)env->last_burn_post[1];
        rec->burn_post_z  = (float)env->last_burn_post[2];
        rec->burn_post_vx = (float)env->last_burn_post[3];
        rec->burn_post_vy = (float)env->last_burn_post[4];
        rec->burn_post_vz = (float)env->last_burn_post[5];
    } else {
        rec->burn_post_x = rec->burn_post_y = rec->burn_post_z = 0.0f;
        rec->burn_post_vx = rec->burn_post_vy = rec->burn_post_vz = 0.0f;
    }

    /* Min conjunction distance */
    double min_dist = 1e30;
    for (int i = 0; i < env->num_bodies; i++) {
        double bx, by, bz;
        body_position(&env->bodies[i], &bx, &by, &bz);
        double dx = sx - bx, dy = sy - by, dz = sz - bz;
        double d = sqrt(dx*dx + dy*dy + dz*dz);
        if (d < min_dist) min_dist = d;
    }
    rec->min_conj_dist = (float)min_dist;

    /* All body positions */
    rec->num_bodies = env->num_bodies;
    for (int i = 0; i < env->num_bodies; i++) {
        double bx, by, bz;
        body_position(&env->bodies[i], &bx, &by, &bz);
        (void)bz;   /* bodies are equatorial by construction: body_z ≡ 0 */
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
    double _ff_log = (env->episode_fuel_frac > 0.0)
                     ? env->episode_fuel_frac : FUEL_FRAC;
    double initial_fuel = env->sat.dry_mass * _ff_log / (1.0 - _ff_log);
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

    /* ── T11: draw this episode's CELL, before anything reads the config ──────
     * Runs first so that every field below — including the cap clamp and
     * obs[15]'s divisor — sees the cell's own values. OFF consumes zero rand()
     * draws, so the existing lineages keep their RNG stream bit-for-bit. */
    /* ONE rand() seeds the episode's mixer; both draws come from splitmix64.
     * See the t11_mix64 comment for why the raw stream cannot be used here. */
    uint64_t _t11_rs = 0;
    int _t11_seeded = 0;
    if ((env->cell_mixture_mode && env->num_cells > 0)
        || (env->fuel_frac_min >= 0.0 && env->fuel_frac_max > env->fuel_frac_min)) {
        _t11_rs = ((uint64_t)rand() * 0x9E3779B97F4A7C15ULL)
                  ^ ((uint64_t)(env->episode_id + 1) * 0xD1B54A32D192ED03ULL);
        _t11_seeded = 1;
    }
    (void)_t11_seeded;

    if (env->cell_mixture_mode && env->num_cells > 0) {
        double tot = 0.0;
        for (int c = 0; c < env->num_cells; c++) tot += env->cells[c][CF_WEIGHT];
        int pick = 0;
        if (tot > 0.0) {
            double u = t11_u01(&_t11_rs) * tot, acc = 0.0;
            for (int c = 0; c < env->num_cells; c++) {
                acc += env->cells[c][CF_WEIGHT];
                if (u <= acc) { pick = c; break; }
                pick = c;
            }
        }
        const double* cl = env->cells[pick];
        env->last_cell            = pick;
        env->episode_cap_steps    = (int)cl[CF_CAP];
        env->rendezvous_radius_m  = cl[CF_BOX_R];
        env->rel_vel_tol_ms       = cl[CF_BOX_V];
        env->a_min_override       = cl[CF_A_MIN];
        env->a_max_override       = cl[CF_A_MAX];
        env->e_max_target         = cl[CF_E_MAX_T];
        env->e_max_sat            = cl[CF_E_MAX_S];
        env->de_max               = cl[CF_DE_MAX];
        env->da_max_m             = cl[CF_DA_MAX];
        env->di_max_rad           = cl[CF_DI_MAX];
        env->di_min_rad           = cl[CF_DI_MIN];
        env->di_phase_mode        = (int)cl[CF_DI_PHASE];
        env->j2_mode              = (int)cl[CF_J2];
        env->i_target_min_rad     = cl[CF_IT_MIN];
        env->i_target_max_rad     = cl[CF_IT_MAX];
        env->fuel_frac_min        = cl[CF_FUEL_MIN];
        env->fuel_frac_max        = cl[CF_FUEL_MAX];
    }

    /* ── T11: this episode's fuel budget. Gated so OFF costs no rand() draw. */
    if (env->fuel_frac_min >= 0.0 && env->fuel_frac_max > env->fuel_frac_min) {
        env->episode_fuel_frac = env->fuel_frac_min
            + t11_u01(&_t11_rs)
              * (env->fuel_frac_max - env->fuel_frac_min);
    } else {
        env->episode_fuel_frac = FUEL_FRAC;
    }

    /* T3 kwarg sanitation: zero-initialized envs (orbital.c standalone tests,
     * any caller that predates the T3 kwargs) fall back to legacy values.
     * The Python binding always sets these explicitly. */
    if (env->episode_cap_steps <= 0 || env->episode_cap_steps > MAX_STEPS)
        env->episode_cap_steps = 2000;          /* legacy 33.3 h cap */
    if (env->shape_gamma <= 0.0)   env->shape_gamma   = 0.995;
    if (env->shape_dv_ref_ms <= 0.0) env->shape_dv_ref_ms = 300.0;

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
        /* T3 L3+: bound the transfer size independently of the altitude band —
         * wide bands otherwise make Δv_a unaffordable (300–8000 km band:
         * mean 783 m/s vs the 478 m/s budget). Window ∩ band; degenerate
         * windows fall back to the full band. */
        if (env->da_max_m > 0.0) {
            double da = (env->da_max_m < 200e3) ? 200e3 : env->da_max_m;
            double lo = fmax(a_min_t, a_init - da);
            double hi = fmin(a_max_t, a_init + da);
            if (hi - lo >= 150e3) { a_min_t = lo; a_max_t = hi; }
        }
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
    /* T3 L3+ (de_max): bound the chaser–target e-VECTOR mismatch. Replaces the
     * independent sat-e draw above: ē_sat = ē_target + area-uniform disc of
     * radius de_max. Both orbits may be strongly eccentric; the matching
     * maneuver stays affordable (Δv_e ≈ v·|Δē|/2 ≤ v·de_max/2). */
    if (env->de_max >= 0.0 && !env->same_orbit_init && env->e_sat_fixed < 0.0) {
        double r_de  = env->de_max * sqrt(rand() / (double)RAND_MAX);
        double ph_de = (rand() / (double)RAND_MAX) * 2.0 * M_PI;
        double e_sx  = e_tgt * cos(omega_tgt) + r_de * cos(ph_de);
        double e_sy  = e_tgt * sin(omega_tgt) + r_de * sin(ph_de);
        env->sat.orbit.e     = sqrt(e_sx * e_sx + e_sy * e_sy);
        env->sat.orbit.omega = (env->sat.orbit.e > 1e-9) ? atan2(e_sy, e_sx) : 0.0;
    }
    /* W1: omega_offset_fixed > -10 → set target.omega = sat.omega + offset */
    if (env->omega_offset_fixed > -10.0) {
        env->target.omega = env->sat.orbit.omega + env->omega_offset_fixed;
        omega_tgt = env->target.omega;
    }

    /* ── ext-3d: orbit planes ────────────────────────────────────────────────
     * dim3_mode = 0 ⇒ inc = raan = 0.0 EXACTLY for the chaser, the target and
     * every body. Combined with propagate_orbit never touching them and both
     * conversion routines value-gating on (inc==0 && raan==0), the invariant
     * "every Orbit is exactly equatorial" is closed under reset, propagation
     * and burns — which is what makes the 2D anchor bit-exact rather than
     * merely float-close (3d_A §5.1).
     *
     * The target plane is pure GAUGE under two-body dynamics (the env is
     * SO(3)-invariant), so it defaults to i_t = Ω_t = 0 and all relative
     * inclination lives in the chaser. i_target_rad / raan_target_rad exist as
     * test hooks for the sampler and frame gates.
     *
     * SAMPLER — rotation form only (3d_REDTEAM BLOCKER-1). ĥ_s = R(δ, n̂)·ĥ_t
     * with δ = di_max_rad·√U (area-uniform disc) and n̂ uniform IN THE TARGET
     * PLANE, then (inc, raan) recovered by atan2. Measured max/knob = 1.0000
     * with 0.0% over at every i_t. The alternative in 3d_A §4 — adding a disc
     * to ī = (sin i cosΩ, sin i sinΩ) and taking inc = asin|ī| — is DELETED: ī
     * is a projection, so it under-measures by cos i_t (1.64× the knob at
     * i_t = 51.6°, 21.4× at 98°), and asin cannot represent i > 90° at all. */
    {
        double i_t = 0.0, O_t = 0.0;
        if (env->dim3_mode) {
            i_t = env->i_target_rad;
            O_t = env->raan_target_rad;
            /* ── ext-j2 rung: sample the target plane. BOTH branches are gated
             * so that the OFF path consumes zero rand() draws — the sampler
             * must not shift the RNG stream, or every bit-exact anchor moves
             * for a reason that has nothing to do with physics. */
            if (env->i_target_min_rad >= 0.0
                && env->i_target_max_rad > env->i_target_min_rad) {
                i_t = env->i_target_min_rad
                    + (rand() / (double)RAND_MAX)
                      * (env->i_target_max_rad - env->i_target_min_rad);
            }
            if (env->raan_target_sample) {
                O_t = wrap_2pi(O_t + (rand() / (double)RAND_MAX) * 2.0 * M_PI);
            }
        }
        env->target.inc    = i_t;
        env->target.raan   = O_t;
        env->sat.orbit.inc = i_t;   /* di_max off ⇒ chaser plane = target plane */
        env->sat.orbit.raan= O_t;

        if (env->dim3_mode && env->di_max_rad >= 0.0 && !env->same_orbit_init) {
            /* ext-j2wait: both branches gated so the OFF path consumes exactly
             * the same two rand() draws in the same order as the legacy
             * sampler — the knobs change the VALUES drawn, never the stream. */
            double delta, ph;
            if (env->di_min_rad >= 0.0 && env->di_max_rad > env->di_min_rad) {
                delta = env->di_min_rad + (rand() / (double)RAND_MAX)
                                          * (env->di_max_rad - env->di_min_rad);
            } else {
                delta = env->di_max_rad * sqrt(rand() / (double)RAND_MAX);
            }
            if (env->di_phase_mode == 1) {
                /* node-dominant: ±90° ± 30°, sign carried by the same draw */
                double u = rand() / (double)RAND_MAX;
                double sgn = (u < 0.5) ? 1.0 : -1.0;
                double t   = (u < 0.5) ? (u * 2.0) : ((u - 0.5) * 2.0);
                ph = sgn * (0.5 * M_PI + (t - 0.5) * (M_PI / 3.0));
            } else {
                ph = (rand() / (double)RAND_MAX) * 2.0 * M_PI;
            }

            double wx, wy, wz;
            orb_hhat(&env->target, &wx, &wy, &wz);
            /* Orthonormal basis of the target PLANE: û₁ = ẑ × ĥ_t (or x̂ when
             * ĥ_t ∥ ẑ), û₂ = ĥ_t × û₁. */
            double u1x = -wy, u1y = wx, u1z = 0.0;
            double u1n = sqrt(u1x*u1x + u1y*u1y);
            if (u1n > 1e-14) { u1x /= u1n; u1y /= u1n; }
            else             { u1x = 1.0; u1y = 0.0; }
            double u2x = wy*u1z - wz*u1y;
            double u2y = wz*u1x - wx*u1z;
            double u2z = wx*u1y - wy*u1x;
            double cph = cos(ph), sph = sin(ph);
            double nx = u1x*cph + u2x*sph;
            double ny = u1y*cph + u2y*sph;
            double nz = u1z*cph + u2z*sph;

            /* Rodrigues: ĥ_s = ĥ_t cosδ + (n̂ × ĥ_t) sinδ + n̂ (n̂·ĥ_t)(1−cosδ).
             * n̂ ⟂ ĥ_t by construction, so the last term is ~0; it is kept for
             * exactness against round-off in the basis. */
            double cd = cos(delta), sd = sin(delta);
            double dot = nx*wx + ny*wy + nz*wz;
            double crx = ny*wz - nz*wy;
            double cry = nz*wx - nx*wz;
            double crz = nx*wy - ny*wx;
            double hsx = wx*cd + crx*sd + nx*dot*(1.0 - cd);
            double hsy = wy*cd + cry*sd + ny*dot*(1.0 - cd);
            double hsz = wz*cd + crz*sd + nz*dot*(1.0 - cd);
            double hn  = sqrt(hsx*hsx + hsy*hsy + hsz*hsz);
            if (hn > 0.0) { hsx /= hn; hsy /= hn; hsz /= hn; }

            double hxy = sqrt(hsx*hsx + hsy*hsy);
            if (hxy > 1e-14) {
                env->sat.orbit.inc  = atan2(hxy, hsz);
                double ra = atan2(hsx, -hsy);
                if (ra < 0.0) ra += 2.0 * M_PI;
                env->sat.orbit.raan = ra;
            } else {
                /* Signed-zero guard (3d_REDTEAM m5): an exactly equatorial
                 * ĥ = (0.0, −0.0, 1.0) pushed through the rotation becomes
                 * (0.0, +0.0, 1.0), and atan2(0.0, −0.0) = π while
                 * atan2(0.0, +0.0) = 0 — a π jump in Ω out of pure float sign. */
                env->sat.orbit.inc  = (hsz >= 0.0) ? 0.0 : M_PI;
                env->sat.orbit.raan = 0.0;
            }

            /* T5 W3/W4 diagnosis ROOT-CAUSE FIX: the de_max disc above is
             * drawn in the NODE-RELATIVE 2-vector (e cos omega, e sin omega),
             * but this rotation just assigned a uniform-random RAAN, which
             * rotates the periapsis in INERTIAL space (varpi = raan + omega)
             * and destroys the disc bound: realized inertial |dE| ran to
             * 4.5x (W3) / 8.0x (W4) the knob, making the e-match leg alone
             * exceed the whole 478 m/s budget in 28%/54% of episodes (the
             * measured W3 70% / W4 7% collapse; di_max=0 control: 199/200).
             * Preserve the drawn INERTIAL periapsis longitude by re-expressing
             * omega relative to the new node. Guarded on the de-disc branch,
             * so e_max_sat-sampled lineages (W1/W2/X3) are bit-exact. */
            if (env->de_max >= 0.0 && env->e_sat_fixed < 0.0) {
                /* ext-j2 rung FIX: subtract the node offset RELATIVE TO THE
                 * TARGET, not the chaser's absolute RAAN. The de_max disc is
                 * drawn in the TARGET's node-relative 2-vector, so what must be
                 * preserved is varpi_s - varpi_t, i.e. the correction is
                 * (raan_s - raan_t). Subtracting raan_s alone is only right
                 * when raan_t == 0, which every shipped lineage happened to
                 * satisfy because the target plane was PINNED. With a sampled
                 * target RAAN it over-rotates the chaser's e-vector by a
                 * uniform-random Omega_t and re-opens the exact W3/W4 failure
                 * this block was written to close: measured realized |de|
                 * 0.111 against a 0.020 knob, 55.6% of draws over.
                 * BIT-EXACT for every shipped lineage: raan_t is exactly 0.0
                 * there and x - 0.0 == x for all finite x. */
                env->sat.orbit.omega -= (env->sat.orbit.raan - env->target.raan);
                while (env->sat.orbit.omega < 0.0)
                    env->sat.orbit.omega += 2.0 * M_PI;
                env->sat.orbit.omega = fmod(env->sat.orbit.omega, 2.0 * M_PI);
            }
        }
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
    {
        double _ff0 = (env->episode_fuel_frac > 0.0)
                      ? env->episode_fuel_frac : FUEL_FRAC;
        env->sat.fuel_mass = env->sat.dry_mass * _ff0 / (1.0 - _ff0);
    }

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
    if (env->phase_gap_mode == 1) {
        /* T3: make the knob control the PHYSICAL mean-longitude gap
         * λ_t − λ_s = (M_t + ω_t) − (M_s + ω_s) = phase_gap exactly.
         * Legacy mode 0 offsets M per-perifocal-frame; with independent ω the
         * realized physical gap is uniform ±180° regardless of the knob
         * (recon ANOM-4 — every e>0 phase curriculum stage was unstaged).
         *
         * ext-3d: in 3D the knob goes inert again unless ϖ = ω + Ω replaces ω
         * (measured realized-gap error p50 = 90.4°, max 180° unpatched — the
         * ANOM-4 re-run). ϖ is taken in the target-plane gauge so the patch is
         * exact at any i_t; at the design gauge i_t = Ω_t = 0 that is literally
         * (ω_s + Ω_s) − (ω_t + Ω_t). ϖ_gauge does not depend on M, so writing
         * tgt_M afterwards cannot disturb it. */
        if (env->dim3_mode) {
            PlaneGauge g;
            gauge_from_orbit(&env->target, &g);
            tgt_M += orb_varpi_gauge(&env->sat.orbit, &g)
                   - orb_varpi_gauge(&env->target,    &g);
        } else {
            tgt_M += env->sat.orbit.omega - env->target.omega;
        }
    }
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
    env->bodies[0].orbit.inc        = 0.0;   /* ext-3d: bodies are equatorial */
    env->bodies[0].orbit.raan       = 0.0;
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
        env->bodies[1 + i].orbit.inc        = 0.0;   /* ext-3d: debris stay equatorial */
        env->bodies[1 + i].orbit.raan       = 0.0;
        env->bodies[1 + i].hard_radius      = DEBRIS_HARD_R;
        env->bodies[1 + i].keepout_radius   = DEBRIS_KEEPOUT;
        env->bodies[1 + i].is_static        = 0;
    }

    /* Seed dense-shaping cache: distance and phase offset at t=0. */
    {
        double sx, sy, sz, svx, svy, svz;
        orbit_to_cartesian(&env->sat.orbit, &sx, &sy, &sz, &svx, &svy, &svz);
        double tx, ty, tz, tvx, tvy, tvz;
        orbit_to_cartesian(&env->target,     &tx, &ty, &tz, &tvx, &tvy, &tvz);
        double dx = sx - tx, dy = sy - ty, dz = sz - tz;
        env->dist_prev = sqrt(dx*dx + dy*dy + dz*dz);

        double dp = env->sat.orbit.theta - env->target.theta;
        dp = dp - 2.0*M_PI * floor((dp + M_PI) / (2.0*M_PI));
        env->dphase_prev = dp;
    }

    /* Phase 4 R2: seed Φ shaping cache at t=0. last_tau=1 (non-warp). */
    env->phi_prev      = compute_phi(env);
    env->last_tau      = 1;
    env->g_shape_accum = 0.0;
    env->last_burn_valid = 0;   /* ext-3d: no burn has happened this episode */

    fill_observations(env);
}

static inline void c_step(Orbital* env) {
    env->rewards[0]   = 0.0f;
    env->terminals[0] = 0;
    env->last_burn_valid = 0;   /* ext-3d: cleared before this decision's burn */

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
        /* ext-j2: the ONLY three propagation call sites. j2_mode = 0 dispatches
         * to the verbatim legacy propagate_orbit inside propagate_orbit_j2, so
         * the default path is unchanged instruction-for-instruction. Warps are
         * this same loop run τ times; the J2 rates are constants of the element
         * set, so τ×DT here ≡ one call of τ·DT (anchor J-A4). */
        /* Propagate all orbiting bodies (not Earth) */
        for (int i = 1; i < env->num_bodies; i++) {
            propagate_orbit_j2(&env->bodies[i].orbit, DT, env->j2_mode);
        }
        /* Propagate the target body — Phase 3 rendezvous requires a moving target. */
        propagate_orbit_j2(&env->target, DT, env->j2_mode);
        /* Propagate satellite */
        propagate_orbit_j2(&env->sat.orbit, DT, env->j2_mode);

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
        /* T3: shape_gamma >= 1.0 → γ_shape = 1 exactly (pure telescoping;
         * kills the (1−γ^τ)·|Φ| do-nothing income — measured +1.78/episode
         * under the legacy γ^τ form, recon F-1 — and restores PBRS validity
         * vs the trainer's flat per-decision γ). Default 0.995 = legacy. */
        double gamma_tau   = (env->shape_gamma >= 1.0)
                             ? 1.0
                             : pow(env->shape_gamma, (double)env->last_tau);
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
        double bx, by, bz;
        body_position(&env->bodies[i], &bx, &by, &bz);
        GRID_SET(bx, by, '*');
    }

    /* Draw satellite (z-projection is an acceptable 2D view) */
    double sx, sy, sz, svx, svy, svz;
    orbit_to_cartesian(&env->sat.orbit, &sx, &sy, &sz, &svx, &svy, &svz);
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
