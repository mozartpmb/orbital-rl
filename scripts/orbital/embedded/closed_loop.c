/* closed_loop.c — fully-C closed loop: the PufferLib Ocean orbital environment
 *                 (orbital.h) driven by the exported C policy (policy.c).
 *                 No Python, no torch, no dynamic allocation in the loop.
 *
 * The environment configuration below reproduces, field for field, what
 * binding.c::my_init() sets from the T3 canonical eval command
 * (T3_RECOVERY_CAMPAIGN.md §5/§6, scripts/orbital/t3/t3_ladder.sh::heval):
 *
 *   --e-max-target 0.05 --e-max-sat 0.05 --same-orbit-init 0
 *   --init-phase-gap-max 3.14159 --valid-init-only 1
 *   --shaping-mode 1 --shape-gamma 1.0 --phase-gap-mode 1 --phase-obs-mode 1
 *   --episode-cap-steps 3000 --cap-terminal-reward 0.0
 *   (debris off; all other kwargs at eval_checkpoint.py defaults)
 *
 * Seeding matches env_binding.h::vec_reset exactly — srand(i + seed*num_envs)
 * with i = 0, num_envs = 1 — so the episode stream is bit-identical to the
 * Python harness at the same --seed, which is what makes the 200-episode
 * closed-loop comparison a true like-for-like.
 *
 * Usage:
 *   ./closed_loop [--episodes N] [--seed S] [--trace FILE] [--bench] [--quiet]
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <stdint.h>

#include "orbital.h"
#include "policy.h"

/* High-resolution monotonic clock (see bench_policy.c for the rationale). */
#if defined(__APPLE__)
static inline uint64_t now_ns(void) { return clock_gettime_nsec_np(CLOCK_UPTIME_RAW); }
#else
static inline uint64_t now_ns(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}
#endif

/* Terminal-cause names, indexed by TERM_* from orbital.h */
static const char* const CAUSE_NAME[8] = {
    "none", "success", "collision", "escape",
    "safety_cap", "stranded", "hyperbolic", "gave_up"
};

static void configure_t3_canonical(Orbital* env)
{
    /* --- exercised by the T3 canonical command --------------------------- */
    env->num_debris_min          = 0;
    env->num_debris_max          = 0;
    env->e_max_target            = 0.05;
    env->e_max_sat               = 0.05;
    env->same_orbit_init         = 0;
    env->init_phase_gap_max      = 3.14159;
    env->valid_init_only         = 1;
    env->shaping_mode            = 1;
    env->shape_w_lambda          = 1.0;
    env->shape_w_match           = 0.35;
    env->shape_dv_ref_ms         = 300.0;
    env->shape_gamma             = 1.0;
    env->phase_gap_mode          = 1;
    env->phase_obs_mode          = 1;
    env->episode_cap_steps       = 3000;
    env->cap_terminal_reward     = 0.0;

    /* --- eval_checkpoint.py defaults ------------------------------------- */
    env->e_mix_easy_frac         = 0.0;
    env->e_mix_easy_max          = 0.05;
    env->collision_penalty_w     = 0.0;
    env->enable_action_mask      = 0;
    env->e_target_fixed          = -1.0;
    env->e_sat_fixed             = -1.0;
    env->phase_gap_fixed         = -1.0;
    env->omega_offset_fixed      = -99.0;
    env->a_min_override          = -1.0;
    env->a_max_override          = -1.0;
    env->log_validation_debug    = 0;
    env->max_valid_init_attempts = 4096;
    env->gave_up_action          = 1;        /* "terminate" */
    env->obs_alt_scale_m         = 1.6e6;
    env->phi_orbit_scale_k       = 0.001;
    env->lvlh_scale_m            = 6.371e6;
    env->rendezvous_radius_m     = 30000.0;
    env->rel_vel_tol_ms          = 50.0;
    env->de_max                  = -1.0;
    env->da_max_m                = -1.0;

    env->log_enabled             = 0;        /* trajectory ring buffer off   */
    env->episode_id              = 0;
    env->last_terminal_cause     = TERM_NONE;
    env->last_traj_records       = 0;
}

int main(int argc, char** argv)
{
    int episodes = 200;
    int seed     = 123;
    int bench    = 0;
    int quiet    = 0;
    const char* trace_path = NULL;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--episodes") && i + 1 < argc)   episodes = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--seed") && i + 1 < argc)  seed     = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--trace") && i + 1 < argc) trace_path = argv[++i];
        else if (!strcmp(argv[i], "--bench"))                 bench = 1;
        else if (!strcmp(argv[i], "--quiet"))                 quiet = 1;
        else {
            fprintf(stderr, "usage: %s [--episodes N] [--seed S] [--trace FILE]"
                            " [--bench] [--quiet]\n", argv[0]);
            return 2;
        }
    }

    /* Single up-front allocation; nothing is allocated inside the loop. */
    Orbital* env = (Orbital*)calloc(1, sizeof(Orbital));
    if (!env) { fprintf(stderr, "calloc(Orbital) failed\n"); return 1; }

    static float         obs_buf[OBS_DIM];
    static int           act_buf[1];
    static float         rew_buf[1];
    static unsigned char term_buf[1];
    env->observations = obs_buf;
    env->actions      = act_buf;
    env->rewards      = rew_buf;
    env->terminals    = term_buf;

    configure_t3_canonical(env);

    FILE* trace = NULL;
    if (trace_path) {
        trace = fopen(trace_path, "w");
        if (!trace) { perror("fopen trace"); free(env); return 1; }
        fprintf(trace, "episode,step,action\n");
    }

    /* env_binding.h::vec_reset semantics for num_envs = 1, env index 0. */
    srand((unsigned)(0 + seed * 1));
    c_reset(env);

    policy_state_t pst;
    policy_reset(&pst);

    long   cause_counts[8] = {0};
    long   decisions = 0, sim_substeps = 0;
    double reward_sum = 0.0;
    int    ep = 0, ep_step = 0;
    int    policy_faults = 0;

    uint64_t ns_policy = 0, ns_env = 0;
    const uint64_t t0 = now_ns();

    while (ep < episodes) {
        int action = 0;
        const uint64_t p0 = now_ns();
        const policy_status_t rc = policy_infer(&pst, env->observations,
                                                NULL, NULL, &action);
        ns_policy += now_ns() - p0;
        if (rc != POLICY_OK) {
            /* Flight-software behaviour: a rejected frame commands COAST (the
             * physically safe no-op) and is counted, rather than propagating
             * an undefined action into the actuators. */
            policy_faults++;
            action = 0;
        }

        if (trace) fprintf(trace, "%d,%d,%d\n", ep, ep_step, action);

        const int step_before = env->step;
        env->actions[0] = action;
        const uint64_t e0 = now_ns();
        c_step(env);
        ns_env += now_ns() - e0;
        decisions++;
        ep_step++;

        if (env->terminals[0]) {
            const int cause = env->last_terminal_cause;
            cause_counts[(cause >= 0 && cause < 8) ? cause : 0]++;
            sim_substeps += env->last_episode_steps;
            reward_sum   += (double)env->rewards[0];
            ep++;
            ep_step = 0;
            policy_reset(&pst);          /* episode boundary: zero h and c    */
            if (!quiet && (ep % 50 == 0)) {
                fprintf(stderr, "  episode %d/%d  success=%ld\n",
                        ep, episodes, cause_counts[TERM_SUCCESS]);
            }
        } else {
            sim_substeps += (env->step - step_before);
        }
    }

    const double secs = 1e-9 * (double)(now_ns() - t0);

    if (trace) fclose(trace);

    printf("=== fully-C closed loop (orbital.h env + policy.c) ===\n");
    printf("build          : %s\n", policy_build_variant());
    printf("ckpt sha256    : %s\n", policy_ckpt_sha256());
    printf("config         : T3 canonical (e<=0.05 both, gap +-180 deg, cap 3000, box 30km/50m/s)\n");
    printf("seed           : %d   episodes: %d\n", seed, episodes);
    printf("physical success: %ld/%d (%.1f%%)\n",
           cause_counts[TERM_SUCCESS], episodes,
           100.0 * (double)cause_counts[TERM_SUCCESS] / (double)episodes);
    printf("terminal causes:");
    for (int i = 0; i < 8; i++) {
        if (cause_counts[i]) printf(" %s=%ld", CAUSE_NAME[i], cause_counts[i]);
    }
    printf("\n");
    printf("mean reward    : %.3f\n", reward_sum / (double)episodes);
    printf("decisions      : %ld  (mean %.1f/episode)\n",
           decisions, (double)decisions / (double)episodes);
    printf("sim sub-steps  : %ld  (mean %.1f/episode, 60 s each)\n",
           sim_substeps, (double)sim_substeps / (double)episodes);
    printf("policy faults  : %d\n", policy_faults);
    printf("wall time      : %.4f s\n", secs);
    printf("throughput     : %.0f decisions/s  |  %.0f env sub-steps/s\n",
           (double)decisions / secs, (double)sim_substeps / secs);
    if (bench) {
        const double sp = 1e-9 * (double)ns_policy;
        const double se = 1e-9 * (double)ns_env;
        printf("--- time split (in-loop, instrumented) ---\n");
        printf("  policy_infer : %.4f s (%.1f%%)  -> %.2f us/decision\n",
               sp, 100.0 * sp / secs, 1e6 * sp / (double)decisions);
        printf("  c_step       : %.4f s (%.1f%%)  -> %.2f us/decision, "
               "%.3f us/sub-step\n",
               se, 100.0 * se / secs, 1e6 * se / (double)decisions,
               1e6 * se / (double)sim_substeps);
        printf("  env alone    : %.0f sub-steps/s (policy excluded)\n",
               (double)sim_substeps / se);
    }

    free(env);
    return (cause_counts[TERM_SUCCESS] == episodes) ? 0 : 1;
}
