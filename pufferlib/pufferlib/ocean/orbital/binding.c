#include <Python.h>
#include <numpy/arrayobject.h>
#include "orbital.h"

/* Forward declarations so MY_METHODS can reference them before the function bodies. */
static PyObject* vec_get_trajectory(PyObject* self, PyObject* args);
static PyObject* vec_get_episode_init_info(PyObject* self, PyObject* args);
static PyObject* vec_get_episode_result(PyObject* self, PyObject* args);
static PyObject* vec_get_state(PyObject* self, PyObject* args);

/* Hook into env_binding.h method table — no trailing comma, the sentinel follows */
#define MY_METHODS \
    {"vec_get_trajectory", (PyCFunction)vec_get_trajectory, METH_VARARGS, \
     "Copy traj_log to numpy array: (vec_handle, env_idx, out_float32_array) -> records"}, \
    {"vec_get_episode_init_info", (PyCFunction)vec_get_episode_init_info, METH_VARARGS, \
     "Get last-reset info: (vec_handle, env_idx) -> (attempts:int, gave_up:int)"}, \
    {"vec_get_episode_result", (PyCFunction)vec_get_episode_result, METH_VARARGS, \
     "Get last-episode outcome: (vec_handle, env_idx) -> (sim_steps:int, terminal_cause:int)"}, \
    {"vec_get_state", (PyCFunction)vec_get_state, METH_VARARGS, \
     "Read chaser+target state for ALL envs: (vec_handle, out_float64_(N,36)) -> N"}

#define Env Orbital
#include "../env_binding.h"

/* ─────────────────────────────────────────────────────────────────────────────
 * Custom method implementations (after env_binding.h so VecEnv is defined)
 * ───────────────────────────────────────────────────────────────────────────*/

/* Number of float fields per TrajectoryRecord:
 * sim_time(1) sat_x(1) sat_y(1) sat_vx(1) sat_vy(1)    = 5
 * sat_a(1) sat_e(1) sat_theta(1) sat_omega(1)           = 4
 * fuel(1) action(1) reward(1) delta_v(1) min_conj_dist(1) = 5
 * target_a(1) target_e(1) target_omega(1)               = 3
 * target_x(1) target_y(1) target_vx(1) target_vy(1)     = 4  (v3: +target velocity)
 * num_bodies(1)                                          = 1
 * body_x[16](16) body_y[16](16)                         = 32
 * body_hard_r[16](16) body_keepout_r[16](16)            = 32
 * SUBTOTAL (pre-ext-3d) = 5+4+5+3+4+1+32+32 = 86
 * ext-3d APPENDED (every pre-existing column index is unchanged):
 * sat_z sat_vz sat_inc sat_raan                          = 4
 * target_z target_vz target_inc target_raan              = 4
 * burn_post_{x,y,z,vx,vy,vz}                             = 6
 * TOTAL = 86 + 14 = 100
 */
#define TRAJ_FLOATS 100

static void fill_traj_row(const TrajectoryRecord* r, float* out) {
    int k = 0;
    out[k++] = r->sim_time;
    out[k++] = r->sat_x;
    out[k++] = r->sat_y;
    out[k++] = r->sat_vx;
    out[k++] = r->sat_vy;
    out[k++] = r->sat_a;
    out[k++] = r->sat_e;
    out[k++] = r->sat_theta;
    out[k++] = r->sat_omega;
    out[k++] = r->fuel;
    out[k++] = (float)r->action;
    out[k++] = r->reward;
    out[k++] = r->delta_v;
    out[k++] = r->min_conj_dist;
    out[k++] = r->target_a;
    out[k++] = r->target_e;
    out[k++] = r->target_omega;
    out[k++] = r->target_x;
    out[k++] = r->target_y;
    out[k++] = r->target_vx;
    out[k++] = r->target_vy;
    out[k++] = (float)r->num_bodies;
    for (int i = 0; i < MAX_BODIES; i++) out[k++] = r->body_x[i];
    for (int i = 0; i < MAX_BODIES; i++) out[k++] = r->body_y[i];
    for (int i = 0; i < MAX_BODIES; i++) out[k++] = r->body_hard_r[i];
    for (int i = 0; i < MAX_BODIES; i++) out[k++] = r->body_keepout_r[i];
    /* ── ext-3d, appended ── */
    out[k++] = r->sat_z;
    out[k++] = r->sat_vz;
    out[k++] = r->sat_inc;
    out[k++] = r->sat_raan;
    out[k++] = r->target_z;
    out[k++] = r->target_vz;
    out[k++] = r->target_inc;
    out[k++] = r->target_raan;
    out[k++] = r->burn_post_x;
    out[k++] = r->burn_post_y;
    out[k++] = r->burn_post_z;
    out[k++] = r->burn_post_vx;
    out[k++] = r->burn_post_vy;
    out[k++] = r->burn_post_vz;
    /* k == TRAJ_FLOATS == 100 */
}

static PyObject* vec_get_trajectory(PyObject* self, PyObject* args) {
    if (PyTuple_Size(args) != 3) {
        PyErr_SetString(PyExc_TypeError,
            "vec_get_trajectory(vec_handle, env_idx, out_float32_array)");
        return NULL;
    }

    PyObject* handle_obj = PyTuple_GetItem(args, 0);
    VecEnv* vec = (VecEnv*)PyLong_AsVoidPtr(handle_obj);
    if (!vec) { PyErr_SetString(PyExc_ValueError, "Invalid vec handle"); return NULL; }

    int env_idx = (int)PyLong_AsLong(PyTuple_GetItem(args, 1));
    if (env_idx < 0 || env_idx >= vec->num_envs) {
        PyErr_SetString(PyExc_ValueError, "env_idx out of range");
        return NULL;
    }
    Orbital* env = vec->envs[env_idx];

    PyArrayObject* arr = (PyArrayObject*)PyTuple_GetItem(args, 2);
    if (!PyArray_ISCONTIGUOUS(arr) || PyArray_TYPE(arr) != NPY_FLOAT) {
        PyErr_SetString(PyExc_ValueError, "out_array must be contiguous float32");
        return NULL;
    }

    /* Use last_traj_records — the number of valid rows including the terminal
     * record (steps+1 for normal episodes, capped at MAX_STEPS). The old code
     * copied last_episode_steps rows, which silently dropped the terminal
     * record from every exported trajectory. Fall back to the legacy count if
     * the field is unset (e.g. mid-episode query). */
    int records = env->last_traj_records;
    if (records <= 0 || records > MAX_STEPS) {
        records = env->last_episode_steps;
        if (records <= 0 || records > MAX_STEPS) records = env->step;
        if (records > MAX_STEPS) records = MAX_STEPS;
    }

    float* out = (float*)PyArray_DATA(arr);
    for (int s = 0; s < records; s++) {
        fill_traj_row(&env->traj_log[s], out + s * TRAJ_FLOATS);
    }

    return PyLong_FromLong(records);
}

/* Phase 5 env-fix F2: per-episode reset-outcome accessor. Returns the
 * rejection-sampler attempts count and gave-up flag for the most recent
 * c_reset on the given env. Called by orbital.py::_save_trajectory at
 * episode boundaries so trajectory exports can record realized init state
 * (not just the kwarg intent). */
static PyObject* vec_get_episode_init_info(PyObject* self, PyObject* args) {
    if (PyTuple_Size(args) != 2) {
        PyErr_SetString(PyExc_TypeError,
            "vec_get_episode_init_info(vec_handle, env_idx) -> (attempts, gave_up)");
        return NULL;
    }

    PyObject* handle_obj = PyTuple_GetItem(args, 0);
    VecEnv* vec = (VecEnv*)PyLong_AsVoidPtr(handle_obj);
    if (!vec) { PyErr_SetString(PyExc_ValueError, "Invalid vec handle"); return NULL; }

    int env_idx = (int)PyLong_AsLong(PyTuple_GetItem(args, 1));
    if (env_idx < 0 || env_idx >= vec->num_envs) {
        PyErr_SetString(PyExc_ValueError, "env_idx out of range");
        return NULL;
    }
    Orbital* env = vec->envs[env_idx];
    return Py_BuildValue("(ii)", env->last_init_attempts, env->last_init_gave_up);
}

/* Last-episode outcome accessor: sim-step count and TERM_* cause code of the
 * most recent completed episode. Lets eval harnesses classify success on the
 * terminal branch that actually fired instead of the terminal reward's sign
 * (which the Φ-clamp can corrupt at wide altitude bands). */
static PyObject* vec_get_episode_result(PyObject* self, PyObject* args) {
    if (PyTuple_Size(args) != 2) {
        PyErr_SetString(PyExc_TypeError,
            "vec_get_episode_result(vec_handle, env_idx) -> (sim_steps, terminal_cause)");
        return NULL;
    }

    PyObject* handle_obj = PyTuple_GetItem(args, 0);
    VecEnv* vec = (VecEnv*)PyLong_AsVoidPtr(handle_obj);
    if (!vec) { PyErr_SetString(PyExc_ValueError, "Invalid vec handle"); return NULL; }

    int env_idx = (int)PyLong_AsLong(PyTuple_GetItem(args, 1));
    if (env_idx < 0 || env_idx >= vec->num_envs) {
        PyErr_SetString(PyExc_ValueError, "env_idx out of range");
        return NULL;
    }
    Orbital* env = vec->envs[env_idx];
    return Py_BuildValue("(ii)", env->last_episode_steps, env->last_terminal_cause);
}

/* ── vec_get_state ──────────────────────────────────────────────────────────
 * Read-only accessor for the chaser and target states of EVERY env in one
 * call. Exists because the 3D-nav wrapper cannot recover the chaser's orbit
 * plane from the observation: the ext-3d block obs[21-28] is SO(3)-invariant
 * by construction, so it carries (Δi_rel, u) and annihilates the inertial node
 * longitude exactly (n3d_REDTEAM MAJOR-5, measured to 1.06e-15).
 *
 * n3d_REDTEAM MAJOR-5: the plane is returned as the UNIT ANGULAR-MOMENTUM
 * VECTOR ĥ, NOT as the scalar Ω. Ω alone is sufficient only at i_target = 0;
 * the moment the target is tilted, recovering i_s from (Δi_rel, Ω_s) requires
 * solving `P sin i + Q cos i = cos Δi_rel`, which is two-valued — 30% of draws
 * at i_t = 51.6° admit >= 2 inclination roots. ĥ is the plane itself: no
 * branch, no solve, and it also honours orbital.h's CONSUMER RULE that no
 * downstream channel may read `omega` or `raan` ALONE.
 *
 * The Cartesian states come from the env's OWN orbit_to_cartesian so the
 * wrapper's reconstruction is on the identical FP path (including the
 * value-gated 2D fast branch) rather than a re-derivation.
 *
 * Mechanics (n3d_REDTEAM MAJOR-6): caller-owned contiguous float64 out-array,
 * mirroring vec_get_trajectory rather than vec_get_episode_init_info — a
 * per-step allocating getter is 1024 x 30 x 8 B of churn per worker per step.
 * Pure read: no RNG draw, no env mutation, no observation write. */
#define STATE_FLOATS 36

static void fill_state_row(const Orbital* env, double* o) {
    const Orbit* s = &env->sat.orbit;
    const Orbit* t = &env->target;
    double hx, hy, hz, x, y, z, vx, vy, vz, ex, ey, ez;

    o[0] = s->a;  o[1] = s->e;  o[2] = s->M;  o[3] = s->theta;  o[4] = s->omega;
    orb_hhat(s, &hx, &hy, &hz);
    o[5] = hx; o[6] = hy; o[7] = hz;
    orbit_to_cartesian(s, &x, &y, &z, &vx, &vy, &vz);
    o[8] = x; o[9] = y; o[10] = z; o[11] = vx; o[12] = vy; o[13] = vz;
    o[14] = env->sat.fuel_mass / (env->sat.dry_mass + env->sat.fuel_mass);

    o[15] = t->a; o[16] = t->e; o[17] = t->M; o[18] = t->theta; o[19] = t->omega;
    orb_hhat(t, &hx, &hy, &hz);
    o[20] = hx; o[21] = hy; o[22] = hz;
    orbit_to_cartesian(t, &x, &y, &z, &vx, &vy, &vz);
    o[23] = x; o[24] = y; o[25] = z; o[26] = vx; o[27] = vy; o[28] = vz;
    o[29] = (double)env->step;

    /* Inertial eccentricity 3-vectors, ELEMENT route (orb_evec), because that
     * is the route obs[23] and obs[28] are built from in fill_observations.
     * 3d_REDTEAM BLOCKER-2: the algebraically identical Cartesian route
     * e = (v x h)/mu - r_hat is a different FP path and breaks the A2 anchor on
     * 87.7% of draws, so the wrapper must not re-derive it — and it CANNOT
     * re-derive the element route from (h_hat, omega) either, because that
     * needs Omega, which is exactly the ill-conditioned quantity at i -> 0
     * (the env pins raan = 0 there, while atan2(h_x, -h_y) on a zero vector
     * returns pi). Hand it over instead. */
    orb_evec(s, &ex, &ey, &ez);
    o[30] = ex; o[31] = ey; o[32] = ez;
    orb_evec(t, &ex, &ey, &ez);
    o[33] = ex; o[34] = ey; o[35] = ez;
}

static PyObject* vec_get_state(PyObject* self, PyObject* args) {
    if (PyTuple_Size(args) != 2) {
        PyErr_SetString(PyExc_TypeError,
            "vec_get_state(vec_handle, out_float64_array)");
        return NULL;
    }

    PyObject* handle_obj = PyTuple_GetItem(args, 0);
    VecEnv* vec = (VecEnv*)PyLong_AsVoidPtr(handle_obj);
    if (!vec) { PyErr_SetString(PyExc_ValueError, "Invalid vec handle"); return NULL; }

    PyArrayObject* arr = (PyArrayObject*)PyTuple_GetItem(args, 1);
    if (!PyArray_Check(arr) || !PyArray_ISCONTIGUOUS(arr)
        || PyArray_TYPE(arr) != NPY_DOUBLE) {
        PyErr_SetString(PyExc_ValueError, "out_array must be contiguous float64");
        return NULL;
    }
    if (PyArray_SIZE(arr) < (npy_intp)vec->num_envs * STATE_FLOATS) {
        PyErr_SetString(PyExc_ValueError,
            "out_array too small: need (num_envs, 36) float64");
        return NULL;
    }

    double* out = (double*)PyArray_DATA(arr);
    for (int i = 0; i < vec->num_envs; i++) {
        fill_state_row(vec->envs[i], out + (size_t)i * STATE_FLOATS);
    }
    return PyLong_FromLong(vec->num_envs);
}

/* ─────────────────────────────────────────────────────────────────────────── */

static int my_init(Env* env, PyObject* args, PyObject* kwargs) {
    /* T3 red-team #4: trajectory logging writes 352 B × every sub-step into a
     * ~1 MB/env buffer (1 GB at num_envs=1024). Only enable when the Python
     * side actually saves trajectories (traj_log_dir set). */
    env->log_enabled    = (int)unpack(kwargs, "log_enabled");
    env->episode_id     = 0;
    env->num_debris_min     = (int)unpack(kwargs, "num_debris_min");
    env->num_debris_max     = (int)unpack(kwargs, "num_debris_max");
    env->e_max_target       = (double)unpack(kwargs, "e_max_target");
    env->init_phase_gap_max = (double)unpack(kwargs, "init_phase_gap_max");
    env->e_max_sat          = (double)unpack(kwargs, "e_max_sat");
    env->same_orbit_init    = (int)unpack(kwargs, "same_orbit_init");
    env->e_mix_easy_frac    = (double)unpack(kwargs, "e_mix_easy_frac");
    env->e_mix_easy_max     = (double)unpack(kwargs, "e_mix_easy_max");
    env->collision_penalty_w = (double)unpack(kwargs, "collision_penalty_w");
    env->enable_action_mask = (int)unpack(kwargs, "enable_action_mask");
    env->valid_init_only = (int)unpack(kwargs, "valid_init_only");
    env->e_target_fixed = (double)unpack(kwargs, "e_target_fixed");
    env->e_sat_fixed = (double)unpack(kwargs, "e_sat_fixed");
    env->phase_gap_fixed = (double)unpack(kwargs, "phase_gap_fixed");
    env->omega_offset_fixed = (double)unpack(kwargs, "omega_offset_fixed");
    env->a_min_override = (double)unpack(kwargs, "a_min_override");
    env->a_max_override = (double)unpack(kwargs, "a_max_override");
    env->log_validation_debug = (int)unpack(kwargs, "log_validation_debug");
    env->max_valid_init_attempts = (int)unpack(kwargs, "max_valid_init_attempts");
    env->gave_up_action          = (int)unpack(kwargs, "gave_up_action");
    env->obs_alt_scale_m         = (double)unpack(kwargs, "obs_alt_scale_m");
    env->phi_orbit_scale_k       = (double)unpack(kwargs, "phi_orbit_scale_k");
    env->lvlh_scale_m            = (double)unpack(kwargs, "lvlh_scale_m");
    env->rendezvous_radius_m     = (double)unpack(kwargs, "rendezvous_radius_m");
    env->rel_vel_tol_ms          = (double)unpack(kwargs, "rel_vel_tol_ms");
    /* T3 corrected-dynamics recovery kwargs (defaults = legacy behavior) */
    env->shaping_mode            = (int)unpack(kwargs, "shaping_mode");
    env->shape_w_lambda          = (double)unpack(kwargs, "shape_w_lambda");
    env->shape_w_match           = (double)unpack(kwargs, "shape_w_match");
    env->shape_dv_ref_ms         = (double)unpack(kwargs, "shape_dv_ref_ms");
    env->shape_gamma             = (double)unpack(kwargs, "shape_gamma");
    env->phase_gap_mode          = (int)unpack(kwargs, "phase_gap_mode");
    env->phase_obs_mode          = (int)unpack(kwargs, "phase_obs_mode");
    env->episode_cap_steps       = (int)unpack(kwargs, "episode_cap_steps");
    env->cap_terminal_reward     = (double)unpack(kwargs, "cap_terminal_reward");
    env->de_max                  = (double)unpack(kwargs, "de_max");
    env->da_max_m                = (double)unpack(kwargs, "da_max_m");
    /* ext-3d kwargs (defaults = legacy 2D behavior, bit-exact) */
    env->dim3_mode               = (int)unpack(kwargs, "dim3_mode");
    env->di_max_rad              = (double)unpack(kwargs, "di_max_rad");
    env->i_target_rad            = (double)unpack(kwargs, "i_target_rad");
    env->raan_target_rad         = (double)unpack(kwargs, "raan_target_rad");
    env->obs_di_scale_rad        = (double)unpack(kwargs, "obs_di_scale_rad");
    env->obs_de_scale            = (double)unpack(kwargs, "obs_de_scale");
    env->shape_match_squash      = (int)unpack(kwargs, "shape_match_squash");
    /* ext-j2 kwarg (default 0 = verbatim legacy propagator, bit-exact) */
    env->j2_mode                 = (int)unpack(kwargs, "j2_mode");
    env->last_terminal_cause     = TERM_NONE;
    env->last_traj_records       = 0;

    /* ── ext-j2 preconditions (j2_A_design §4.1) ─────────────────────────────
     * (1) j2_mode = 1 REQUIRES dim3_mode = 1. Hard error, not a style rule: at
     *     dim3_mode = 0 every orbit is exactly equatorial by construction, so
     *     J2 would take the i = 0 branch for the chaser AND the target and
     *     apply ϖ̇ = +k to both — a real precession of the 2D lineage's ω that
     *     no 2D anchor and no 2D checkpoint covers.
     * (2) j2_mode = 1 REQUIRES num_debris = 0. Inherited from dim3_mode: the
     *     3D obs block occupies body slots 21-32.
     * (3) i_target_rad = 0 under j2_mode = 1 is a WARNING, not an error. At an
     *     equatorial target Δi_rel = i_s regardless of Ω_s, so differential Ω̇
     *     cannot move the plane term at all — measured Δ(dv_pl) = +0.00 m/s and
     *     ΔΦ = −0.00000 over a full 6000-step cap at every Δi (§2.2, the 4th
     *     instance of this project's inert-knob class). It is NOT an error
     *     because anchors J-A3 (equatorial closure) and J-A5 (inertness guard)
     *     are DEFINED at i_t = 0 and must remain runnable; the design's §4.1
     *     "assert i_target_rad > 0" contradicts its own §4.2 on that point. */
    if (env->j2_mode) {
        if (!env->dim3_mode) {
            PyErr_SetString(PyExc_ValueError,
                "j2_mode=1 requires dim3_mode=1 (j2_A_design §1.4/§4.1): at "
                "dim3_mode=0 every orbit is exactly equatorial and J2 would "
                "apply an unanchored varpi-dot precession to the 2D lineage.");
            return -1;
        }
        if (env->num_debris_max > 0) {
            PyErr_SetString(PyExc_ValueError,
                "j2_mode=1 requires num_debris_max=0 (inherited from dim3_mode: "
                "the 3D/J2 obs block occupies body slots 21-32).");
            return -1;
        }
        if (env->i_target_rad == 0.0) {
            fprintf(stderr,
                "[orbital/j2] WARNING: j2_mode=1 with i_target_rad=0. The J2 "
                "plane channel is provably INERT at an equatorial target "
                "(j2_A_design §2.2: d(dv_pl) = 0.00 m/s over a full cap at "
                "every di). Fine for the J-A3/J-A5 anchors; wrong for training.\n");
        }
    }
    return 0;
}

static int my_log(PyObject* dict, Log* log) {
    assign_to_dict(dict, "perf",           log->perf);
    assign_to_dict(dict, "episode_return", log->episode_return);
    assign_to_dict(dict, "episode_length", log->episode_length);
    assign_to_dict(dict, "fuel_used",      log->fuel_used);
    /* Phase 5 env-fix F4 + Phase 5.5: realized-init metrics (epoch means via vec_log) */
    assign_to_dict(dict, "init_attempts_mean",       log->init_attempts_mean);
    assign_to_dict(dict, "init_gave_up_rate",        log->init_gave_up_rate);
    assign_to_dict(dict, "realized_e_target_mean",   log->realized_e_target_mean);
    assign_to_dict(dict, "realized_e_sat_mean",      log->realized_e_sat_mean);
    assign_to_dict(dict, "realized_a_target_mean_m", log->realized_a_target_mean_m);
    assign_to_dict(dict, "realized_a_sat_mean_m",    log->realized_a_sat_mean_m);
    return 0;
}
