#include <Python.h>
#include <numpy/arrayobject.h>
#include "orbital.h"

/* Forward declarations so MY_METHODS can reference them before the function bodies. */
static PyObject* vec_get_trajectory(PyObject* self, PyObject* args);
static PyObject* vec_get_episode_init_info(PyObject* self, PyObject* args);

/* Hook into env_binding.h method table — no trailing comma, the sentinel follows */
#define MY_METHODS \
    {"vec_get_trajectory", (PyCFunction)vec_get_trajectory, METH_VARARGS, \
     "Copy traj_log to numpy array: (vec_handle, env_idx, out_float32_array) -> steps"}, \
    {"vec_get_episode_init_info", (PyCFunction)vec_get_episode_init_info, METH_VARARGS, \
     "Get last-reset info: (vec_handle, env_idx) -> (attempts:int, gave_up:int)"}

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
 * TOTAL = 5+4+5+3+4+1+32+32 = 86
 */
#define TRAJ_FLOATS 86

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
    /* k == TRAJ_FLOATS == 86 */
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

    /* Use last_episode_steps — env->step resets to 0 after terminal */
    int steps = env->last_episode_steps;
    if (steps <= 0 || steps > MAX_STEPS) steps = env->step;
    if (steps > MAX_STEPS) steps = MAX_STEPS;

    float* out = (float*)PyArray_DATA(arr);
    for (int s = 0; s < steps; s++) {
        fill_traj_row(&env->traj_log[s], out + s * TRAJ_FLOATS);
    }

    return PyLong_FromLong(steps);
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

/* ─────────────────────────────────────────────────────────────────────────── */

static int my_init(Env* env, PyObject* args, PyObject* kwargs) {
    env->log_enabled    = 1;  /* always log; Python controls whether to save */
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
