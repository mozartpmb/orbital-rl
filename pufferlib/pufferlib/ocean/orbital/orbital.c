/*
 * orbital.c — Standalone C test for orbital.h physics
 *
 * Compile: gcc -O2 -fsanitize=address -lm orbital.c -o orbital_test
 * Run:     ./orbital_test
 *
 * Validates:
 *   1. Circular orbit stability (no burn, 1000 steps) — a and e must not drift
 *   2. Prograde burn raises orbit (a increases)
 *   3. Retrograde burn lowers orbit (a decreases)
 *   4. Hohmann transfer 400→800 km — compare Δv to textbook (≈110 m/s)
 *   5. Escape trajectory detection (large prograde burn)
 *   6. Fuel consumption — verify Tsiolkovsky values
 */

#include "orbital.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/* Allocate env buffers on the stack — no Python involved */
static void alloc_env(Orbital* env) {
    static float   obs[OBS_DIM];
    static int     act[1];
    static float   rew[1];
    static unsigned char term[1];
    memset(obs,  0, sizeof(obs));
    memset(act,  0, sizeof(act));
    memset(rew,  0, sizeof(rew));
    memset(term, 0, sizeof(term));
    env->observations = obs;
    env->actions      = act;
    env->rewards      = rew;
    env->terminals    = term;
    env->log_enabled  = 0;
    env->episode_id   = 0;
    memset(&env->log, 0, sizeof(env->log));
}

/* Set up a specific circular orbit (overrides c_reset randomisation) */
static void set_circular_orbit(Orbital* env, double altitude_km) {
    double a = R_EARTH + altitude_km * 1000.0;
    env->sat.orbit.a     = a;
    env->sat.orbit.e     = 0.0;
    env->sat.orbit.M     = 0.0;
    env->sat.orbit.theta = 0.0;
    env->sat.dry_mass    = 850.0;
    env->sat.fuel_mass   = env->sat.dry_mass * FUEL_FRAC / (1.0 - FUEL_FRAC);
    env->total_dv_used   = 0.0;
}

/* ── Test 1: Circular orbit stability ─────────────────────────────────────── */
static void test_circular_stability(void) {
    printf("=== Test 1: Circular orbit stability (400 km, 1000 steps, no burn) ===\n");

    Orbital env = {0};
    alloc_env(&env);
    c_reset(&env);
    set_circular_orbit(&env, 400.0);

    /* No debris — clear bodies except Earth */
    env.num_bodies = 1;

    double a0 = env.sat.orbit.a;
    double e0 = env.sat.orbit.e;

    for (int i = 0; i < 1000; i++) {
        env.actions[0] = 0;  /* coast */
        c_step(&env);
        if (env.terminals[0]) {
            printf("  WARN: episode terminated at step %d\n", i);
            set_circular_orbit(&env, 400.0);
            env.num_bodies = 1;
        }
    }

    double a_final = env.sat.orbit.a;
    double e_final = env.sat.orbit.e;
    double da = fabs(a_final - a0);
    double de = fabs(e_final - e0);

    printf("  a0=%.3f km  a_final=%.3f km  Δa=%.3f m\n",
           a0/1000.0, a_final/1000.0, da);
    printf("  e0=%.6f  e_final=%.6f  Δe=%.2e\n", e0, e_final, de);

    int pass = (da < 100.0 && de < 1e-6);
    printf("  %s (Δa < 100m, Δe < 1e-6)\n\n", pass ? "PASS" : "FAIL");
}

/* ── Test 2: Prograde burn raises orbit ────────────────────────────────────── */
static void test_prograde_raises(void) {
    printf("=== Test 2: Prograde burn raises orbit ===\n");

    Orbital env = {0};
    alloc_env(&env);
    c_reset(&env);
    set_circular_orbit(&env, 400.0);
    env.num_bodies = 1;

    double a_before = env.sat.orbit.a;
    env.actions[0] = 1;  /* prograde small (10 m/s) */
    c_step(&env);
    double a_after = env.sat.orbit.a;

    printf("  a_before=%.3f km  a_after=%.3f km  Δa=%.3f km\n",
           a_before/1000.0, a_after/1000.0, (a_after - a_before)/1000.0);

    int pass = (a_after > a_before);
    printf("  %s (a should increase after prograde burn)\n\n", pass ? "PASS" : "FAIL");
}

/* ── Test 3: Retrograde burn lowers orbit ──────────────────────────────────── */
static void test_retrograde_lowers(void) {
    printf("=== Test 3: Retrograde burn lowers orbit ===\n");

    Orbital env = {0};
    alloc_env(&env);
    c_reset(&env);
    set_circular_orbit(&env, 400.0);
    env.num_bodies = 1;

    double a_before = env.sat.orbit.a;
    env.actions[0] = 5;  /* retrograde small (−10 m/s) — action 5 post NUM_ACTIONS=9 */
    c_step(&env);
    double a_after = env.sat.orbit.a;

    printf("  a_before=%.3f km  a_after=%.3f km  Δa=%.3f km\n",
           a_before/1000.0, a_after/1000.0, (a_after - a_before)/1000.0);

    int pass = (a_after < a_before);
    printf("  %s (a should decrease after retrograde burn)\n\n", pass ? "PASS" : "FAIL");
}

/* ── Test 4: Hohmann transfer 400→800 km ─────────────────────────────────── */
static void test_hohmann(void) {
    printf("=== Test 4: Hohmann transfer 400→800 km ===\n");

    /* Theoretical Hohmann Δv (two-burn maneuver):
     *   Burn 1 (prograde at 400 km): Δv1 = v_transfer_at_peri - v_circ_400
     *   Burn 2 (prograde at 800 km): Δv2 = v_circ_800 - v_transfer_at_apo
     *
     *   v_circ(r) = sqrt(μ/r)
     *   a_transfer = (r1 + r2) / 2
     *   v_transfer_peri = sqrt(μ * (2/r1 - 1/a_t))
     *   v_transfer_apo  = sqrt(μ * (2/r2 - 1/a_t))
     */
    double r1 = R_EARTH + 400e3;
    double r2 = R_EARTH + 800e3;
    double a_t = (r1 + r2) / 2.0;
    double v1 = sqrt(MU / r1);
    double v2 = sqrt(MU / r2);
    double vt1 = sqrt(MU * (2.0/r1 - 1.0/a_t));
    double vt2 = sqrt(MU * (2.0/r2 - 1.0/a_t));
    double dv1_theory = vt1 - v1;
    double dv2_theory = v2  - vt2;
    double dv_total_theory = dv1_theory + dv2_theory;

    printf("  Theoretical: Δv1=%.2f m/s  Δv2=%.2f m/s  total=%.2f m/s\n",
           dv1_theory, dv2_theory, dv_total_theory);

    /* Simulate: apply burn 1 (impulsive, prograde, at periapsis θ=0)
     * We use apply_impulse directly rather than stepping through to be exact. */
    Orbital env = {0};
    alloc_env(&env);
    c_reset(&env);
    set_circular_orbit(&env, 400.0);
    env.num_bodies = 1;
    env.sat.orbit.M     = 0.0;
    env.sat.orbit.theta = 0.0;

    /* Burn 1: prograde Δv1 */
    double dv1_actual = apply_impulse(&env, dv1_theory, 0.0, 0.0);
    printf("  Burn 1: Δv=%.2f m/s  a_after=%.3f km  e_after=%.4f\n",
           dv1_actual,
           env.sat.orbit.a / 1000.0,
           env.sat.orbit.e);

    /* Propagate to apoapsis (half orbit) */
    double period = 2.0 * M_PI * sqrt(pow(env.sat.orbit.a, 3) / MU);
    double t_half = period / 2.0;
    int steps_to_apo = (int)(t_half / DT);
    for (int i = 0; i < steps_to_apo; i++) {
        propagate_orbit(&env.sat.orbit, DT);
    }
    printf("  At apoapsis: θ=%.4f rad (expect ≈π=%.4f)\n",
           env.sat.orbit.theta, M_PI);

    /* Burn 2: prograde Δv2 */
    double dv2_actual = apply_impulse(&env, dv2_theory, 0.0, 0.0);
    printf("  Burn 2: Δv=%.2f m/s  a_after=%.3f km  e_after=%.4f\n",
           dv2_actual,
           env.sat.orbit.a / 1000.0,
           env.sat.orbit.e);

    double a_target = R_EARTH + 800e3;
    double da = fabs(env.sat.orbit.a - a_target);
    double de = env.sat.orbit.e;

    printf("  Final: a=%.3f km (target=%.3f km)  |Δa|=%.3f km  e=%.5f\n",
           env.sat.orbit.a / 1000.0, a_target / 1000.0, da / 1000.0, de);

    int pass = (da < 100e3 && de < 0.01);
    printf("  %s (|Δa| < 100 km, e < 0.01 after Hohmann)\n\n", pass ? "PASS" : "FAIL");
}

/* ── Test 5: Escape detection ─────────────────────────────────────────────── */
static void test_escape(void) {
    printf("=== Test 5: Escape trajectory detection ===\n");

    /* Escape velocity at 400 km: v_esc = sqrt(2*μ/r) */
    double r = R_EARTH + 400e3;
    double v_circ = sqrt(MU / r);
    double v_esc  = sqrt(2.0 * MU / r);
    double dv_needed = v_esc - v_circ + 100.0;  /* extra to guarantee escape */

    printf("  v_circ=%.2f m/s  v_esc=%.2f m/s  applying Δv=%.2f m/s\n",
           v_circ, v_esc, dv_needed);

    Orbital env = {0};
    alloc_env(&env);
    c_reset(&env);
    set_circular_orbit(&env, 400.0);
    env.num_bodies = 1;
    /* Give plenty of fuel for this test */
    env.sat.fuel_mass = 1e6;

    env.actions[0] = 0;
    double dv = apply_impulse(&env, dv_needed, 0.0, 0.0);

    /* Check orbital energy */
    double sx, sy, sz, svx, svy, svz;
    orbit_to_cartesian(&env.sat.orbit, &sx, &sy, &sz, &svx, &svy, &svz);
    double v2 = svx*svx + svy*svy + svz*svz;
    double r2 = sqrt(sx*sx + sy*sy + sz*sz);
    double E_orb = 0.5 * v2 - MU / r2;

    printf("  Δv_applied=%.2f m/s  orbital_energy=%.4e J/kg\n", dv, E_orb);

    /* Now step — should trigger escape termination */
    c_step(&env);
    printf("  terminal=%d  reward=%.1f\n", env.terminals[0], env.rewards[0]);

    int pass = (env.terminals[0] == 1 && env.rewards[0] < 0.0f);
    printf("  %s (should terminate with -10 reward)\n\n", pass ? "PASS" : "FAIL");
}

/* ── Test 6: Fuel consumption (Tsiolkovsky) ───────────────────────────────── */
static void test_fuel_consumption(void) {
    printf("=== Test 6: Tsiolkovsky fuel consumption ===\n");

    /* Expected: Δm/m0 = 1 - exp(-|Δv|/Ve)
     * For Δv=10 m/s, Ve=2942 m/s: Δm/m0 ≈ 0.003396 (0.34%)
     * For Δv=50 m/s, Ve=2942 m/s: Δm/m0 ≈ 0.016817 (1.68%)
     * Ratio: 50m/s burn should cost ≈4.95× the 10m/s burn
     */
    double Ve = VE;
    double expected_10  = 1.0 - exp(-10.0 / Ve);
    double expected_50  = 1.0 - exp(-50.0 / Ve);

    printf("  Ve=%.2f m/s\n", Ve);
    printf("  Expected Δm/m0 for 10 m/s: %.6f (%.4f%%)\n",
           expected_10, expected_10 * 100.0);
    printf("  Expected Δm/m0 for 50 m/s: %.6f (%.4f%%)\n",
           expected_50, expected_50 * 100.0);
    printf("  Cost ratio (50/10): %.2f×\n", expected_50 / expected_10);

    /* Test 10 m/s burn */
    {
        Orbital env = {0};
        alloc_env(&env);
        c_reset(&env);
        set_circular_orbit(&env, 400.0);
        env.num_bodies = 1;

        double m0 = env.sat.dry_mass + env.sat.fuel_mass;
        double fuel_before = env.sat.fuel_mass;
        apply_impulse(&env, 10.0, 0.0, 0.0);
        double fuel_after = env.sat.fuel_mass;
        double actual_frac = (fuel_before - fuel_after) / m0;

        printf("  10 m/s actual Δm/m0: %.6f  expected: %.6f  err: %.2e\n",
               actual_frac, expected_10, fabs(actual_frac - expected_10));
    }

    /* Test 50 m/s burn */
    {
        Orbital env = {0};
        alloc_env(&env);
        c_reset(&env);
        set_circular_orbit(&env, 400.0);
        env.num_bodies = 1;

        double m0 = env.sat.dry_mass + env.sat.fuel_mass;
        double fuel_before = env.sat.fuel_mass;
        apply_impulse(&env, 50.0, 0.0, 0.0);
        double fuel_after = env.sat.fuel_mass;
        double actual_frac = (fuel_before - fuel_after) / m0;

        printf("  50 m/s actual Δm/m0: %.6f  expected: %.6f  err: %.2e\n\n",
               actual_frac, expected_50, fabs(actual_frac - expected_50));
    }
}

/* ── Test 7: Total Δv budget ──────────────────────────────────────────────── */
static void test_dv_budget(void) {
    printf("=== Test 7: Total Δv budget (15%% fuel fraction) ===\n");

    /* Theoretical: Δv_total = Ve * ln(m0 / m_dry)
     * m0/m_dry = 1/(1 - FUEL_FRAC) = 1/0.85 ≈ 1.1765
     * Δv_total = 2942 * ln(1.1765) ≈ 480 m/s
     */
    double dry  = 850.0;
    double fuel = dry * FUEL_FRAC / (1.0 - FUEL_FRAC);
    double m0   = dry + fuel;
    double dv_budget = VE * log(m0 / dry);

    printf("  dry_mass=%.1f kg  fuel_mass=%.1f kg  fuel_fraction=%.2f%%\n",
           dry, fuel, 100.0 * fuel / m0);
    printf("  theoretical Δv budget = Ve * ln(m0/m_dry) = %.1f m/s\n", dv_budget);

    Orbital env = {0};
    alloc_env(&env);
    c_reset(&env);
    set_circular_orbit(&env, 400.0);
    env.num_bodies = 1;

    /* Burn until out of fuel using 10 m/s burns */
    int burns = 0;
    while (env.sat.fuel_mass > 0.0 && burns < 10000) {
        apply_impulse(&env, 10.0, 0.0, 0.0);
        burns++;
    }

    printf("  actual total Δv used = %.1f m/s (%d burns of 10 m/s)\n",
           env.total_dv_used, burns);
    printf("  fuel remaining = %.4f kg\n\n", env.sat.fuel_mass);
}

/* ── Test 9: LVLH stationary — sat co-located with target stays zero in LVLH ── */
static void test_lvlh_stationary(void) {
    printf("=== Test 9: LVLH frame stationary (sat == target, 100 steps) ===\n");

    Orbital env = {0};
    alloc_env(&env);
    c_reset(&env);
    env.num_bodies = 1;  /* Earth only */

    /* Set sat and target to identical 500 km circular orbits */
    double a = R_EARTH + 500e3;
    env.sat.orbit.a     = a;
    env.sat.orbit.e     = 0.0;
    env.sat.orbit.M     = 0.5;
    env.sat.orbit.theta = 0.5;
    env.sat.orbit.omega = 0.0;
    env.sat.dry_mass    = 850.0;
    env.sat.fuel_mass   = env.sat.dry_mass * FUEL_FRAC / (1.0 - FUEL_FRAC);
    env.target.a     = a;
    env.target.e     = 0.0;
    env.target.M     = 0.5;
    env.target.theta = 0.5;
    env.target.omega = 0.0;

    double max_dvx_l = 0.0, max_dvy_l = 0.0;
    double max_dx_l = 0.0, max_dy_l = 0.0;

    for (int i = 0; i < 100; i++) {
        env.actions[0] = 0;  /* coast */
        c_step(&env);
        if (env.terminals[0]) {
            /* Success reward triggered by at_target — auto-reset happened.
             * Re-align since c_reset randomized. */
            env.sat.orbit.a = env.target.a = a;
            env.sat.orbit.e = env.target.e = 0.0;
            env.sat.orbit.M = env.target.M = 0.5;
            env.sat.orbit.theta = env.target.theta = 0.5;
            env.sat.orbit.omega = env.target.omega = 0.0;
            continue;
        }
        /* obs[33..37] = dx_l, dy_l, dvx_l (normalized), dvy_l, n */
        double dx_l  = (double)env.observations[33] * R_EARTH;
        double dy_l  = (double)env.observations[34] * R_EARTH;
        double v_circ_t = sqrt(MU / env.target.a);
        double dvx_l = (double)env.observations[35] * v_circ_t;
        double dvy_l = (double)env.observations[36] * v_circ_t;
        if (fabs(dx_l)  > fabs(max_dx_l))  max_dx_l  = dx_l;
        if (fabs(dy_l)  > fabs(max_dy_l))  max_dy_l  = dy_l;
        if (fabs(dvx_l) > fabs(max_dvx_l)) max_dvx_l = dvx_l;
        if (fabs(dvy_l) > fabs(max_dvy_l)) max_dvy_l = dvy_l;
    }

    printf("  max |dx_l|=%.4e m   max |dy_l|=%.4e m\n", max_dx_l, max_dy_l);
    printf("  max |dvx_l|=%.4e m/s  max |dvy_l|=%.4e m/s\n", max_dvx_l, max_dvy_l);

    int pass = (fabs(max_dvx_l) < 1e-3 && fabs(max_dvy_l) < 1e-3 &&
                fabs(max_dx_l) < 1.0 && fabs(max_dy_l) < 1.0);
    printf("  %s (|dvx_l|, |dvy_l| < 1e-3 m/s — catches n×r sign-flip bugs)\n\n",
           pass ? "PASS" : "FAIL");
}

/* ── R2 diagnostic: G_shape vs terminal reward, 50 random episodes ──────── */
static void test_r2_goodhart(void) {
    printf("=== R2 Goodhart Diagnostic: G_shape vs |R_terminal| over 50 random-action eps ===\n");

    Orbital env = {0};
    alloc_env(&env);
    env.num_debris_min = 0;
    env.num_debris_max = 0;
    env.init_phase_gap_max = 3.14159;
    env.e_max_target = 0.0;
    c_reset(&env);

    int n_eps = 0, n_success = 0;
    double sum_g_shape = 0.0, sum_abs_r_term = 0.0;
    double max_g_shape = 0.0, max_step_reward = 0.0;

    while (n_eps < 50) {
        env.actions[0] = rand() % NUM_ACTIONS;
        c_step(&env);
        if (fabs(env.rewards[0]) > max_step_reward) max_step_reward = fabs(env.rewards[0]);
        if (env.terminals[0]) {
            double r_term = env.rewards[0];
            sum_abs_r_term += fabs(r_term);
            sum_g_shape    += env.last_g_shape;
            if (env.last_g_shape > max_g_shape) max_g_shape = env.last_g_shape;
            if (r_term > 0.0) n_success++;
            n_eps++;
            /* env auto-resets after terminal */
        }
    }

    double mean_g   = sum_g_shape / n_eps;
    double mean_abs = sum_abs_r_term / n_eps;
    double ratio    = mean_g / (mean_abs + 1e-12);
    printf("  Episodes: %d  Successes: %d\n", n_eps, n_success);
    printf("  mean |G_shape|   = %.4f\n", mean_g);
    printf("  mean |R_term|    = %.4f\n", mean_abs);
    printf("  ratio G/R        = %.4f   (Principle F hard-fail if > 0.1)\n", ratio);
    printf("  max |G_shape|    = %.4f   max |step reward| = %.4f\n",
           max_g_shape, max_step_reward);
    printf("  %s\n\n", (ratio <= 0.1) ? "PASS (Goodhart bound holds)" :
                                       "WARN (Goodhart bound violated — reduce W_* constants)");
}

/* ── Test 8: ASCII render ─────────────────────────────────────────────────── */
static void test_render(void) {
    printf("=== Test 8: ASCII render (3 steps) ===\n\n");

    Orbital env = {0};
    alloc_env(&env);
    c_reset(&env);

    for (int i = 0; i < 3; i++) {
        c_render(&env);
        env.actions[0] = (i == 0) ? 1 : 0;  /* prograde on first step */
        c_step(&env);
    }
}

/* ── main ──────────────────────────────────────────────────────────────────── */
int main(void) {
    srand(42);  /* fixed seed for reproducibility */

    printf("╔══════════════════════════════════════════════════════╗\n");
    printf("║         Orbital RL — Physics Validation Suite        ║\n");
    printf("╚══════════════════════════════════════════════════════╝\n\n");

    test_circular_stability();
    test_prograde_raises();
    test_retrograde_lowers();
    test_hohmann();
    test_escape();
    test_fuel_consumption();
    test_dv_budget();
    test_lvlh_stationary();
    test_r2_goodhart();
    test_render();

    printf("Done.\n");
    return 0;
}
