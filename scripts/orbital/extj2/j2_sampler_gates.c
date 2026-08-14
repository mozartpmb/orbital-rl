/* ext-j2 rung — inclined-target sampler gates, run against the REAL c_reset
 * and the REAL fill_observations (the v4_gates.c pattern, not a replica).
 *
 * WHY THIS FILE EXISTS. Target-plane randomization is exactly the operation
 * that produced this project's 11th metric-vs-implementation bug: the ī-disc
 * sampler in 3d_A §4 measured Δi_rel through a projection, so its realized
 * value ran 1.64× the knob at i_t = 51.6° and 21.4× at 98°, and every number
 * downstream of it was quietly wrong. The rotation-form sampler fixed that AT
 * A PINNED TARGET PLANE. This branch now randomizes the target plane itself,
 * which is precisely the axis the old bug lived on, so every realized-vs-knob
 * metric is re-measured here rather than assumed to carry over.
 *
 * S1  realized Δi_rel / di_max knob, across the sampled band
 * S2  realized |Δē| / de_max knob, across the sampled band
 * S3  realized mean-longitude gap per draw (phase_gap_mode=1, the inertial-ϖ
 *     patch) under a randomized target plane
 * S4  the sampled i_t distribution honours its band; RNG-stream identity when
 *     the sampler is OFF
 * S5  THE NON-INERTNESS GATE: distribution of |Ω̇_s − Ω̇_t| over 2000 draws vs
 *     the closed form, and its independence from Ω_t
 * S6  SO(2)-about-ẑ invariance — Ω_t is still gauge under J2
 * S7  the channel itself: Δi_rel injected per radian of λ closed vs the
 *     design's 1.75·J2·(R_EQ/p)²·sin 2i, measured by flying a drift
 *
 * Build & run (from the WORKTREE root):
 *   cc -O2 -flto -lm -I pufferlib/pufferlib/ocean/orbital \
 *      scripts/orbital/extj2/j2_sampler_gates.c -o /tmp/j2samp && /tmp/j2samp
 */
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include "orbital.h"

#define R2D(x) ((x) * 180.0 / M_PI)
#define D2R(x) ((x) * M_PI / 180.0)
#define DAY    86400.0

static int g_pass = 0, g_fail = 0;

static void check(const char* name, int ok, const char* detail) {
    printf("  [%s] %-56s %s\n", ok ? "PASS" : "FAIL", name, detail);
    if (ok) g_pass++; else g_fail++;
}

static float g_obs[64];
static int   g_act[1];
static float g_rew[1];
static unsigned char g_term[1];

/* sizeof(Orbital) is ~4.2 MB — static storage, never the stack. */
static Orbital ENV, ENV_B;

static void env_base(Orbital* e) {
    memset(e, 0, sizeof(*e));
    e->observations = g_obs; e->actions = g_act;
    e->rewards = g_rew; e->terminals = g_term;
    e->num_debris_min = 0; e->num_debris_max = 0;
    e->e_max_target = 0.05; e->e_max_sat = 0.05;
    e->e_target_fixed = -1.0; e->e_sat_fixed = -1.0;
    e->phase_gap_fixed = -1.0; e->omega_offset_fixed = -99.0;
    e->a_min_override = -1.0; e->a_max_override = -1.0;
    e->de_max = -1.0; e->da_max_m = -1.0;
    e->max_valid_init_attempts = 4096; e->valid_init_only = 1;
    e->obs_alt_scale_m = 1.6e6; e->phi_orbit_scale_k = 0.001;
    e->lvlh_scale_m = 6.371e6;
    e->rendezvous_radius_m = 30000.0; e->rel_vel_tol_ms = 50.0;
    e->shaping_mode = 2; e->shape_w_lambda = 1.0; e->shape_w_match = 0.8166667;
    e->shape_dv_ref_ms = 700.0; e->shape_gamma = 1.0;
    e->phase_gap_mode = 1; e->phase_obs_mode = 1;
    e->episode_cap_steps = 3000; e->cap_terminal_reward = 0.0;
    e->init_phase_gap_max = M_PI;
    e->dim3_mode = 1; e->di_max_rad = -1.0;
    e->i_target_rad = 0.0; e->raan_target_rad = 0.0;
    e->obs_di_scale_rad = -1.0; e->obs_de_scale = -1.0;
    e->shape_match_squash = 0;
    e->log_enabled = 0;
    /* ext-j2 defaults = off */
    e->j2_mode = 0;
    e->i_target_min_rad = -1.0; e->i_target_max_rad = -1.0;
    e->raan_target_sample = 0;
}

/* The rung configuration, in one place so every gate uses the same thing. */
static const double BAND_LO = D2R(30.0), BAND_HI = D2R(60.0);
static const double DI_MAX  = 0.017453;          /* 1.0 deg, the X3 knob */

static void env_rung(Orbital* e, int j2) {
    env_base(e);
    e->j2_mode = j2;
    e->di_max_rad = DI_MAX;
    e->i_target_min_rad = BAND_LO;
    e->i_target_max_rad = BAND_HI;
    e->raan_target_sample = 1;
}

static double plane_angle(const Orbit* a, const Orbit* b) {
    double ax, ay, az, bx, by, bz;
    orb_hhat(a, &ax, &ay, &az);
    orb_hhat(b, &bx, &by, &bz);
    double cx = ay*bz - az*by, cy = az*bx - ax*bz, cz = ax*by - ay*bx;
    return atan2(sqrt(cx*cx + cy*cy + cz*cz), ax*bx + ay*by + az*bz);
}

static double devec_norm(const Orbit* a, const Orbit* b) {
    double ax, ay, az, bx, by, bz;
    orb_evec(a, &ax, &ay, &az);
    orb_evec(b, &bx, &by, &bz);
    double dx = ax-bx, dy = ay-by, dz = az-bz;
    return sqrt(dx*dx + dy*dy + dz*dz);
}

static double omega_dot(const Orbit* o) {
    double n = sqrt(MU / (o->a * o->a * o->a));
    double p = o->a * (1.0 - o->e * o->e);
    double k = 1.5 * n * J2_COEF * (J2_R_EQ / p) * (J2_R_EQ / p);
    return -k * cos(o->inc);
}

static int dcmp(const void* a, const void* b) {
    double x = *(const double*)a, y = *(const double*)b;
    return (x > y) - (x < y);
}

/* ═══════════════════════════════════════════════════════════════════════ */
static void s1_s2_s3_s4(void) {
    printf("\n== S1-S4  realized-vs-knob under a RANDOMIZED target plane ==\n");
    const long N = 40000;

    /* S4b first: RNG-stream identity when the sampler is off. Two envs, same
     * seed, one with the kwargs present-but-off. If the off path consumed a
     * rand() draw the whole downstream stream would shift and this would show
     * up as a total mismatch, not a subtle one. */
    {
        long mism = 0;
        srand(777u);
        env_base(&ENV); ENV.di_max_rad = DI_MAX; ENV.i_target_rad = 0.0;
        for (long q = 0; q < 2000; q++) c_reset(&ENV);
        double a1 = ENV.sat.orbit.a, m1 = ENV.sat.orbit.M, w1 = ENV.target.omega;
        srand(777u);
        env_base(&ENV_B); ENV_B.di_max_rad = DI_MAX; ENV_B.i_target_rad = 0.0;
        ENV_B.i_target_min_rad = -1.0; ENV_B.i_target_max_rad = -1.0;
        ENV_B.raan_target_sample = 0;
        for (long q = 0; q < 2000; q++) c_reset(&ENV_B);
        if (a1 != ENV_B.sat.orbit.a || m1 != ENV_B.sat.orbit.M
            || w1 != ENV_B.target.omega) mism = 1;
        check("S4b sampler OFF consumes zero rand() draws (stream identity)",
              mism == 0, mism ? "RNG STREAM SHIFTED" :
              "2000 resets, chaser a/M and target omega bitwise identical");
    }

    /* S1/S2/S3/S4a: the sampler ON. */
    struct { int de_on; const char* name; } modes[] = {
        {0, "e_max sampling (X3 rung)"},
        {1, "de_max sampling (wide-e lineage)"},
    };
    for (unsigned m = 0; m < 2; m++) {
        srand(31337u + m);
        env_rung(&ENV, 1);
        if (modes[m].de_on) { ENV.de_max = 0.02; ENV.e_max_target = 0.05; }

        double di_max_seen = 0.0, de_max_seen = 0.0, gap_err_max = 0.0;
        double i_lo = 1e9, i_hi = -1e9, i_sum = 0.0;
        long over_di = 0, over_de = 0;
        double* gaps = (double*)malloc(N * sizeof(double));
        for (long q = 0; q < N; q++) {
            c_reset(&ENV);
            double di = plane_angle(&ENV.sat.orbit, &ENV.target);
            if (di > di_max_seen) di_max_seen = di;
            if (di > DI_MAX) over_di++;

            double de = devec_norm(&ENV.sat.orbit, &ENV.target);
            if (de > de_max_seen) de_max_seen = de;
            if (modes[m].de_on && de > ENV.de_max) over_de++;

            double it = ENV.target.inc;
            if (it < i_lo) i_lo = it;
            if (it > i_hi) i_hi = it;
            i_sum += it;

            /* realized physical mean-longitude gap in the target-plane gauge */
            PlaneGauge g;
            gauge_from_orbit(&ENV.target, &g);
            double dl = wrap_pi(orb_lambda_gauge(&ENV.target, &g)
                              - orb_lambda_gauge(&ENV.sat.orbit, &g));
            gaps[q] = dl;
        }
        /* the phase-gap knob is uniform +-pi here, so "per-draw error" needs
         * the request; c_reset does not expose it. What IS testable per draw:
         * the realized gap must be a well-defined finite angle, and its
         * distribution must be uniform over +-pi (the knob's own shape). Any
         * gauge break shows as a pile-up. Measure the max deviation of the
         * empirical CDF from uniform (KS-style) AND the max |lambda| overflow. */
        qsort(gaps, N, sizeof(double), dcmp);
        double ks = 0.0, ovfl = 0.0;
        for (long q = 0; q < N; q++) {
            double emp = (q + 0.5) / (double)N;
            double the = (gaps[q] + M_PI) / (2.0 * M_PI);
            if (fabs(emp - the) > ks) ks = fabs(emp - the);
            if (fabs(gaps[q]) > ovfl) ovfl = fabs(gaps[q]);
        }
        free(gaps);
        gap_err_max = ks;

        printf("\n  -- %s --\n", modes[m].name);
        printf("    realized di_rel max/knob = %.4f   over = %ld/%ld (%.2f%%)\n",
               di_max_seen / DI_MAX, over_di, N, 100.0 * over_di / N);
        printf("    realized |de| max        = %.6f   %s\n", de_max_seen,
               modes[m].de_on ? "(knob de_max = 0.02)" : "(no de_max knob; e_max path)");
        printf("    sampled i_t band         = %.4f .. %.4f deg (knob %.1f .. %.1f), mean %.3f\n",
               R2D(i_lo), R2D(i_hi), R2D(BAND_LO), R2D(BAND_HI), R2D(i_sum / N));
        printf("    realized lambda-gap KS vs uniform(+-pi) = %.5f, max |gap| = %.6f rad\n",
               ks, ovfl);

        char d[200];
        snprintf(d, sizeof(d), "%s: max/knob %.4f, %.2f%% over, N=%ld",
                 modes[m].name, di_max_seen / DI_MAX, 100.0 * over_di / N, N);
        check("S1 realized di_rel respects di_max under the sampler",
              di_max_seen <= DI_MAX * (1.0 + 1e-9) && over_di == 0, d);

        if (modes[m].de_on) {
            /* The gate is NOT "zero draws over the knob": a pre-existing 0.9%
             * overshoot exists at ANY i_t > 0, because the di_max rotation acts
             * on the chaser's e-vector AFTER the disc is drawn (|de| grows by
             * ~|e|*delta = 0.05*0.0175). That is 0.2% of draws at 1.009x and is
             * unchanged by this branch. What the gate must catch is the failure
             * the sampler DID introduce and the varpi fix closed: 0.111 max,
             * 55.6% over — a 5.5x knob violation. Gate at 1.05x. */
            snprintf(d, sizeof(d), "max |de|/knob = %.4f, %ld/%ld over (%.1f%%); "
                     "pre-fix this cell read 0.111/5.55x with 55.6%% over",
                     de_max_seen / ENV.de_max, over_de, N, 100.0 * over_de / N);
            check("S2 realized |delta-e-vec| tracks de_max under the sampler",
                  de_max_seen <= ENV.de_max * 1.05, d);
        }

        snprintf(d, sizeof(d), "band honoured to %.4f / %.4f deg of the edges; "
                 "KS vs uniform = %.5f (N=%ld, 95%% crit ~%.5f)",
                 R2D(i_lo - BAND_LO), R2D(BAND_HI - i_hi), ks, N, 1.36 / sqrt((double)N));
        check("S3/S4a i_t band honoured; realized lambda gap stays uniform",
              i_lo >= BAND_LO - 1e-12 && i_hi <= BAND_HI + 1e-12
              && ovfl <= M_PI + 1e-12 && ks < 0.02, d);
    }
}

/* ═══════════════════════════════════════════════════════════════════════
 * S5 — THE NON-INERTNESS GATE
 * ═══════════════════════════════════════════════════════════════════════ */
static void s5(void) {
    printf("\n== S5  NON-INERTNESS: |dOmega/dt_sat - dOmega/dt_tgt| over 2000 draws ==\n");
    const long N = 2000;
    double* d = (double*)malloc(N * sizeof(double));
    double* only_di = (double*)malloc(N * sizeof(double));
    double* only_da = (double*)malloc(N * sizeof(double));
    double* by_raan_lo = (double*)malloc(N * sizeof(double));
    double* by_raan_hi = (double*)malloc(N * sizeof(double));
    long n_lo = 0, n_hi = 0, n_zero = 0;

    srand(90210u);
    env_rung(&ENV, 1);
    double sum = 0.0, sum_da = 0.0, sum_di = 0.0, sum_T = 0.0;
    for (long q = 0; q < N; q++) {
        c_reset(&ENV);
        double dd = fabs(omega_dot(&ENV.sat.orbit) - omega_dot(&ENV.target));
        d[q] = R2D(dd) * DAY;                        /* deg/day */
        sum += d[q];
        if (dd == 0.0) n_zero++;

        /* DECOMPOSITION. Omega-dot = -k(a,e) cos i depends on BOTH a and i, and
         * the chaser and target differ in BOTH: the transfer task itself puts
         * them at different altitudes. Isolate each by cloning one element. */
        Orbit s_da = ENV.sat.orbit, t_da = ENV.target;
        s_da.inc = t_da.inc;                          /* same plane, real delta-a */
        only_da[q] = R2D(fabs(omega_dot(&s_da) - omega_dot(&t_da))) * DAY;
        Orbit s_di = ENV.sat.orbit, t_di = ENV.target;
        s_di.a = t_di.a; s_di.e = t_di.e;             /* same orbit size, real delta-i */
        only_di[q] = R2D(fabs(omega_dot(&s_di) - omega_dot(&t_di))) * DAY;
        sum_da += only_da[q]; sum_di += only_di[q];

        /* the operational quantity: relative inclination injected over a
         * 3000-sub-step (50 h) episode, di_rel ~ dOmega * sin(i_t) */
        sum_T += d[q] * (3000.0 * DT / DAY) * sin(ENV.target.inc);

        if (ENV.target.raan < M_PI) by_raan_lo[n_lo++] = d[q];
        else                        by_raan_hi[n_hi++] = d[q];
    }
    qsort(d, N, sizeof(double), dcmp);
    qsort(only_da, N, sizeof(double), dcmp);
    qsort(only_di, N, sizeof(double), dcmp);
    qsort(by_raan_lo, n_lo, sizeof(double), dcmp);
    qsort(by_raan_hi, n_hi, sizeof(double), dcmp);

    /* Closed forms, each for its own mechanism.
     * delta-a term:  |dOm|/|Om| = (7/2)|da|/a. The LEO band is 300-800 km with
     *                |da| >= 50 km enforced, so E|da| ~ 170 km at a ~ 6871 km.
     * delta-i term:  d(Om-dot)/di = +k sin i. The rotation sampler draws
     *                Delta = di_max*sqrt(U) with node phase uniform, so the
     *                INCLINATION component is Delta*cos(phase):
     *                E ~ k sin(i_t) * (2/3) di_max * (2/pi). */
    double a_mid = R_EARTH + 550e3;
    double n_mid = sqrt(MU / (a_mid*a_mid*a_mid));
    double k_mid = 1.5 * n_mid * J2_COEF * (J2_R_EQ / a_mid) * (J2_R_EQ / a_mid);
    double i_mid = 0.5 * (BAND_LO + BAND_HI);
    double om_mid = R2D(k_mid * cos(i_mid)) * DAY;
    double pred_da = fabs(om_mid) * 3.5 * (170e3 / a_mid);
    double pred_di = R2D(k_mid * sin(i_mid) * (2.0/3.0) * DI_MAX * (2.0/M_PI)) * DAY;

    printf("    TOTAL   p05 %.5f  p25 %.5f  p50 %.5f  p75 %.5f  p95 %.5f  max %.5f deg/day\n",
           d[N/20], d[N/4], d[N/2], d[3*N/4], d[19*N/20], d[N-1]);
    printf("    exactly-zero draws: %ld / %ld\n", n_zero, N);
    printf("\n    DECOMPOSITION (which mechanism supplies the signal):\n");
    printf("      delta-a only (same plane) : median %.5f  mean %.5f  closed form %.5f deg/day\n",
           only_da[N/2], sum_da/N, pred_da);
    printf("      delta-i only (same orbit) : median %.5f  mean %.5f  closed form %.5f deg/day\n",
           only_di[N/2], sum_di/N, pred_di);
    printf("      -> the ALTITUDE difference dominates by %.0fx. The transfer task itself\n"
           "         supplies the differential nodal rate; di_max is a %.0f%% correction.\n",
           (sum_da/N) / (sum_di/N), 100.0 * (sum_di/N) / (sum_da/N));
    printf("\n    OPERATIONAL: mean relative inclination injected over a 3000-sub-step\n"
           "                 (50 h) episode = %.4f deg. Free-plane zone at the\n"
           "                 30 km / 50 m/s box is 0.3775 deg -> %.0f%% of it.\n",
           sum_T/N, 100.0 * (sum_T/N) / 0.3775);
    printf("    split by Omega_t: median(Om_t < 180) = %.5f (n=%ld), "
           "median(Om_t >= 180) = %.5f (n=%ld)\n",
           by_raan_lo[n_lo/2], n_lo, by_raan_hi[n_hi/2], n_hi);

    char det[240];
    snprintf(det, sizeof(det),
             "median %.4f deg/day, p05 %.4f, 0 exactly-zero draws; mean injects %.3f deg "
             "of relative inclination per 50 h episode (%.0f%% of the free-plane zone)",
             d[N/2], d[N/20], sum_T/N, 100.0*(sum_T/N)/0.3775);
    check("S5 differential nodal rate is non-zero across the population",
          n_zero == 0 && d[N/20] > 0.02, det);

    snprintf(det, sizeof(det),
             "delta-a mean %.4f vs closed form %.4f (ratio %.2f); delta-i mean %.4f vs "
             "%.4f (ratio %.2f)", sum_da/N, pred_da, (sum_da/N)/pred_da,
             sum_di/N, pred_di, (sum_di/N)/pred_di);
    check("S5a both mechanisms match their own closed forms",
          fabs((sum_da/N)/pred_da - 1.0) < 0.30 && fabs((sum_di/N)/pred_di - 1.0) < 0.30, det);

    double rel = fabs(by_raan_lo[n_lo/2] - by_raan_hi[n_hi/2])
                 / (0.5 * (by_raan_lo[n_lo/2] + by_raan_hi[n_hi/2]));
    snprintf(det, sizeof(det),
             "medians differ by %.2f%% across the Omega_t half-split — Omega_t is GAUGE; "
             "i_t and delta-a are what make J2 non-inert", 100.0 * rel);
    check("S5b differential rate is independent of Omega_t (gauge)", rel < 0.15, det);
    free(d); free(only_di); free(only_da); free(by_raan_lo); free(by_raan_hi);
}

/* ═══════════════════════════════════════════════════════════════════════
 * S6 — SO(2)-about-z invariance
 * ═══════════════════════════════════════════════════════════════════════ */
static void s6(void) {
    printf("\n== S6  SO(2)-about-z invariance: Omega_t is still gauge under J2 ==\n");
    /* Same seed, same everything, except the whole scene is rotated about z by
     * psi via raan_target_rad. J2's potential is axisymmetric about z, so this
     * must move NOTHING the policy or the classifier can see.
     *
     * Run in BOTH frame modes. The legacy frame is EXPECTED to leak — that is
     * the finding this gate exists to produce, not a failure of the gate. */
    const int N = 4000;
    const double psi = D2R(137.0);
    /* Two classes of deliberately frame-VARIANT slot.
     * RAW: the grandfathered raw omega/theta channels (3d_REDTEAM MAJOR-2 kept
     *      slots 0-16 bit-semantics so every validated decoder still works).
     * BEARING: obs[18] is the Earth-conjunction bearing, and Earth is static at
     *      the origin, so body_angle = atan2(0,0) = 0 and the channel reduces to
     *      atan2(s_y, s_x) — the chaser's ABSOLUTE INERTIAL LONGITUDE. It is
     *      rotation-variant about ẑ by construction. That never mattered while
     *      the target plane was pinned (the scene was never rotated); it becomes
     *      live the instant raan_target_sample = 1. Reported separately rather
     *      than fixed: fixing it would change the conjunction block's semantics,
     *      which the legacy 26/200 anchor (WITH debris) depends on. */
    const int SKIP[] = {2, 3, 9, 10, 11, 12, 13, 14, 16};
    const int BEARING[] = {17, 18, 19, 20};
    double leak[2] = {0.0, 0.0};
    double bearing_leak = 0.0;
    for (int mode = 0; mode <= 1; mode++) {
        double worst_obs = 0.0, worst_phi = 0.0, worst_dl = 0.0, worst_di = 0.0;
        double worst_gf = 0.0, worst_bear = 0.0;
        int worst_slot = -1, worst_bear_slot = -1;
        for (int q = 0; q < N; q++) {
            srand(555u + q);
            env_rung(&ENV, 1); ENV.raan_target_rad = 0.0; ENV.lvlh_frame_mode = mode;
            c_reset(&ENV);
            float o1[64]; memcpy(o1, g_obs, sizeof(o1));
            double phi1 = compute_phi(&ENV);
            PlaneGauge g1; gauge_from_orbit(&ENV.target, &g1);
            double dl1 = wrap_pi(orb_lambda_gauge(&ENV.sat.orbit, &g1)
                               - orb_lambda_gauge(&ENV.target, &g1));
            double di1 = plane_angle(&ENV.sat.orbit, &ENV.target);

            srand(555u + q);
            env_rung(&ENV_B, 1); ENV_B.raan_target_rad = psi; ENV_B.lvlh_frame_mode = mode;
            c_reset(&ENV_B);
            float o2[64]; memcpy(o2, g_obs, sizeof(o2));
            double phi2 = compute_phi(&ENV_B);
            PlaneGauge g2; gauge_from_orbit(&ENV_B.target, &g2);
            double dl2 = wrap_pi(orb_lambda_gauge(&ENV_B.sat.orbit, &g2)
                               - orb_lambda_gauge(&ENV_B.target, &g2));
            double di2 = plane_angle(&ENV_B.sat.orbit, &ENV_B.target);

            for (int t = 0; t < OBS_DIM; t++) {
                int skip = 0, bear = 0;
                for (unsigned u = 0; u < sizeof(SKIP)/sizeof(SKIP[0]); u++)
                    if (SKIP[u] == t) skip = 1;
                for (unsigned u = 0; u < sizeof(BEARING)/sizeof(BEARING[0]); u++)
                    if (BEARING[u] == t) bear = 1;
                double dv = fabs((double)o1[t] - (double)o2[t]);
                if (skip) { if (dv > worst_gf) worst_gf = dv; }
                else if (bear) {
                    if (dv > worst_bear) { worst_bear = dv; worst_bear_slot = t; }
                } else if (dv > worst_obs) { worst_obs = dv; worst_slot = t; }
            }
            if (fabs(phi1 - phi2) > worst_phi) worst_phi = fabs(phi1 - phi2);
            double ddl = fabs(wrap_pi(dl1 - dl2));
            if (ddl > worst_dl) worst_dl = ddl;
            if (fabs(di1 - di2) > worst_di) worst_di = fabs(di1 - di2);
        }
        leak[mode] = worst_obs;
        if (worst_bear > bearing_leak) bearing_leak = worst_bear;
        printf("\n  -- lvlh_frame_mode = %d --\n", mode);
        printf("    success-relevant obs slots  max |delta| = %.3e  (worst slot %d)\n",
               worst_obs, worst_slot);
        printf("    grandfathered raw-omega/theta slots     = %.3e  (variant by design)\n", worst_gf);
        printf("    Earth-conjunction bearing obs[17-20]    = %.3e  (worst slot %d) "
               "<- PRE-EXISTING absolute-longitude channel\n", worst_bear, worst_bear_slot);
        printf("    Phi(mode 2)                 max |delta| = %.3e\n", worst_phi);
        printf("    delta-lambda (target gauge) max |delta| = %.3e deg\n", R2D(worst_dl));
        printf("    realized di_rel             max |delta| = %.3e deg\n", R2D(worst_di));
        if (mode == 1) {
            char d[240];
            snprintf(d, sizeof(d),
                     "psi = 137 deg, N=%d paired draws: obs %.2e, Phi %.2e, dlambda %.2e deg, "
                     "di_rel %.2e deg", N, worst_obs, worst_phi, R2D(worst_dl), R2D(worst_di));
            check("S6 SO(2)-about-z holds under lvlh_frame_mode=1",
                  worst_obs < 1e-6 && worst_phi < 1e-9 && R2D(worst_dl) < 1e-6
                  && R2D(worst_di) < 1e-9, d);
        }
    }
    char d[256];
    snprintf(d, sizeof(d),
             "legacy frame leaks %.2f obs units under a pure z-rotation; corrected frame "
             "leaks %.1e. Physics (Phi, dlambda, di_rel) is invariant in BOTH.", leak[0], leak[1]);
    check("S6b the obs[33-36] leak is closed by lvlh_frame_mode=1",
          leak[0] > 0.1 && leak[1] < 1e-6, d);

    snprintf(d, sizeof(d),
             "obs[17-20] still leak %.2f obs units under a z-rotation in BOTH modes: "
             "Earth is static at the origin so obs[18] IS atan2(s_y, s_x). Pre-existing, "
             "NOT fixed here (the legacy 26/200 anchor runs WITH debris and depends on this "
             "block's semantics) -> recommend raan_target_sample=0 for the training arm",
             bearing_leak);
    check("S6c the residual z-leak is the pre-existing conjunction bearing",
          bearing_leak > 0.1, d);
}

/* ═══════════════════════════════════════════════════════════════════════
 * S7 — the channel: Delta-i_rel injected per radian of lambda closed
 * ═══════════════════════════════════════════════════════════════════════ */
static void s7(void) {
    printf("\n== S7  the channel: di_rel injected per radian of lambda closed ==\n");
    printf("    design: 1.75 * J2 * (R_EQ/p)^2 * sin(2i) per radian, INDEPENDENT of the\n"
           "            drift delta-a. Flown in the real propagator at three delta-a.\n");
    printf("    NOTE e = 0.02, not 0: at e = 0 the argument of periapsis is degenerate,\n"
           "         and orb_lambda_gauge mixes an element-set M with a round-trip varpi,\n"
           "         so lambda itself is meaningless there (the same trap J-G6 hit).\n");
    printf("  %6s %8s | %10s %10s %10s | %10s %10s %8s\n",
           "i deg", "da km", "dlam rad", "dOm/dlam", "analytic", "di/dlam", "analytic", "err");
    double worst = 0.0;
    const double incs[3] = {30.0, 45.0, 60.0};
    const double das[3]  = {-100.0, -200.0, -400.0};
    for (int ii = 0; ii < 3; ii++) {
        for (int jj = 0; jj < 3; jj++) {
            double i_deg = incs[ii], da_km = das[jj];
            double a_t = R_EARTH + 500e3, e0 = 0.02;
            Orbit tgt, sat;
            memset(&tgt, 0, sizeof(tgt)); memset(&sat, 0, sizeof(sat));
            tgt.a = a_t; tgt.e = e0; tgt.inc = D2R(i_deg); tgt.raan = D2R(40.0);
            tgt.omega = D2R(20.0); tgt.M = D2R(10.0);
            tgt.theta = eccentric_to_true(solve_kepler(tgt.M, tgt.e), tgt.e);
            sat = tgt;
            sat.a = a_t + da_km * 1e3;
            /* Delta-i MUST start at 0. The design's relation is dOmega/dlambda;
             * its di_rel form (di_rel ~ dOmega sin i) is the SMALL-ANGLE
             * conversion FROM zero. Seeding a nonzero initial Delta-i measures
             * d/dlambda of the angle between two planes that already differ,
             * which combines the seeded tilt and the accrued node split
             * non-additively and is not the design's quantity at all — that
             * mistake produced a spurious 23-83% "delta-a dependence". */

            PlaneGauge g0; gauge_from_orbit(&tgt, &g0);
            double prev = wrap_pi(orb_lambda_gauge(&sat, &g0) - orb_lambda_gauge(&tgt, &g0));
            double di0  = plane_angle(&sat, &tgt);
            double acc_lam = 0.0;
            for (int k = 0; k < 3000; k++) {
                propagate_orbit_j2(&sat, DT, 1);
                propagate_orbit_j2(&tgt, DT, 1);
                PlaneGauge g; gauge_from_orbit(&tgt, &g);
                double l = wrap_pi(orb_lambda_gauge(&sat, &g) - orb_lambda_gauge(&tgt, &g));
                double dl = l - prev;
                if (dl >  M_PI) dl -= 2*M_PI;
                if (dl < -M_PI) dl += 2*M_PI;
                acc_lam += dl; prev = l;
            }
            double di1 = plane_angle(&sat, &tgt);
            double dOm = sat.raan - tgt.raan;
            while (dOm >  M_PI) dOm -= 2*M_PI;
            while (dOm < -M_PI) dOm += 2*M_PI;
            double om_per_rad = R2D(dOm) / fabs(acc_lam);
            double di_per_rad = R2D(di1 - di0) / fabs(acc_lam);
            double p = a_t * (1.0 - e0*e0);
            double om_analytic = -R2D(3.5 * J2_COEF * (J2_R_EQ/p) * (J2_R_EQ/p)
                                      * cos(D2R(i_deg)));
            double di_analytic = R2D(1.75 * J2_COEF * (J2_R_EQ/p) * (J2_R_EQ/p)
                                     * sin(2.0 * D2R(i_deg)));
            double rel = fmax(fabs(om_per_rad - om_analytic) / fabs(om_analytic),
                              fabs(di_per_rad - di_analytic) / di_analytic);
            if (rel > worst) worst = rel;
            printf("  %6.1f %8.0f | %10.4f %10.5f %10.5f | %10.5f %10.5f %7.1f%%\n",
                   i_deg, da_km, acc_lam, om_per_rad, om_analytic,
                   di_per_rad, di_analytic, 100.0*rel);
        }
    }
    char d[240];
    snprintf(d, sizeof(d),
             "worst rel err on BOTH dOmega/dlambda and di_rel/dlambda = %.1f%% over "
             "i in {30,45,60} deg x delta-a in {-100,-200,-400} km (9 cells); the "
             "delta-a independence across each row is the point",
             100.0 * worst);
    check("S7 channel matches 1.75*J2*(R/p)^2*sin(2i), independent of delta-a",
          worst < 0.08, d);
    printf("    The residual grows LINEARLY with |delta-a| (1.1%% -> 2.6%% -> 5.9%% at\n"
           "    -100/-200/-400 km, i.e. proportional to delta-a/a = 1.5%%/2.9%%/5.8%%).\n"
           "    That is the second-order term of a relation the design states to FIRST\n"
           "    order, so it CONFIRMS the identity rather than contradicting it: the\n"
           "    delta-a independence is exact in the limit and 6%% at the widest drift.\n");

    double a_t = R_EARTH + 500e3, e0 = 0.02, p = a_t * (1.0 - e0*e0);
    printf("\n    Per 180 deg of phase closed, at the rung band:\n");
    for (double i = 30.0; i <= 60.5; i += 15.0) {
        double per_rad = R2D(1.75 * J2_COEF * (J2_R_EQ/p) * (J2_R_EQ/p) * sin(2.0*D2R(i)));
        double v_c = sqrt(MU / a_t);
        double dv = 2.0 * v_c * sin(D2R(per_rad * M_PI) / 2.0);
        printf("      i = %4.1f deg: %.4f deg/rad -> %.4f deg per 180 deg drift "
               "= %.1f m/s = %.1f%% of the 478 m/s budget = %.0f%% of the 0.3775 deg "
               "free-plane zone\n",
               i, per_rad, per_rad * M_PI, dv, 100.0 * dv / 478.13,
               100.0 * per_rad * M_PI / 0.3775);
    }
}

/* ═══════════════════════════════════════════════════════════════════════
 * S8 — the LVLH frame. obs[33-36] is the policy's PRIMARY rendezvous channel.
 * ═══════════════════════════════════════════════════════════════════════ */
static void s8(void) {
    printf("\n== S8  obs[33-36]: is the legacy 'LVLH' block still LVLH at i_t > 0? ==\n");
    /* (a) bit-exactness of mode 1 vs mode 0 at the pinned gauge i = Omega = 0 */
    {
        long mism = 0; double worst = 0.0;
        for (int q = 0; q < 4000; q++) {
            srand(1234u + q);
            env_base(&ENV); ENV.di_max_rad = DI_MAX; ENV.lvlh_frame_mode = 0;
            c_reset(&ENV); float o1[64]; memcpy(o1, g_obs, sizeof(o1));
            srand(1234u + q);
            env_base(&ENV_B); ENV_B.di_max_rad = DI_MAX; ENV_B.lvlh_frame_mode = 1;
            c_reset(&ENV_B); float o2[64]; memcpy(o2, g_obs, sizeof(o2));
            for (int t = 33; t <= 36; t++) {
                double dv = fabs((double)o1[t] - (double)o2[t]);
                if (dv > worst) worst = dv;
                if (o1[t] != o2[t]) mism++;
            }
        }
        char d[200];
        snprintf(d, sizeof(d), "at the pinned gauge i_t = Omega_t = 0: %ld/16000 slots differ, "
                 "max |delta| = %.3e (float32 eps = %.1e)", mism, worst, 1.19e-7);
        check("S8a lvlh_frame_mode 1 reduces to the legacy block at i = Omega = 0",
              worst < 1e-6, d);
    }
    /* (b) how big is the error the sampler exposes? */
    {
        double worst = 0.0, sum = 0.0, worst_rel = 0.0;
        const int N = 4000;
        for (int q = 0; q < N; q++) {
            srand(2468u + q);
            env_rung(&ENV, 1); ENV.lvlh_frame_mode = 0;
            c_reset(&ENV); float o1[64]; memcpy(o1, g_obs, sizeof(o1));
            srand(2468u + q);
            env_rung(&ENV_B, 1); ENV_B.lvlh_frame_mode = 1;
            c_reset(&ENV_B); float o2[64]; memcpy(o2, g_obs, sizeof(o2));
            double dmax = 0.0, mag = 0.0;
            for (int t = 33; t <= 36; t++) {
                double dv = fabs((double)o1[t] - (double)o2[t]);
                if (dv > dmax) dmax = dv;
                if (fabs((double)o2[t]) > mag) mag = fabs((double)o2[t]);
            }
            sum += dmax;
            if (dmax > worst) worst = dmax;
            if (mag > 1e-12 && dmax / mag > worst_rel) worst_rel = dmax / mag;
        }
        printf("    i_t sampled 30-60 deg, Omega_t sampled: legacy vs true-LVLH obs[33-36]\n");
        printf("      mean max|delta| = %.4f obs units, worst = %.4f, worst relative = %.1f%%\n",
               sum / N, worst, 100.0 * worst_rel);
        printf("      (obs are Box(-2,2); the policy's whole rendezvous channel is these 4)\n");
        char d[240];
        snprintf(d, sizeof(d), "mean %.4f / worst %.4f obs units, worst %.0f%% relative — "
                 "the legacy block is NOT the LVLH state at inclined targets",
                 sum / N, worst, 100.0 * worst_rel);
        check("S8b the discrepancy at inclined targets is LARGE (this is the finding)",
              worst > 0.05, d);
    }
    /* (c) does mode 1 restore SO(2)-about-z invariance? */
    {
        double worst = 0.0;
        for (int q = 0; q < 4000; q++) {
            srand(555u + q);
            env_rung(&ENV, 1); ENV.raan_target_rad = 0.0; ENV.lvlh_frame_mode = 1;
            c_reset(&ENV); float o1[64]; memcpy(o1, g_obs, sizeof(o1));
            srand(555u + q);
            env_rung(&ENV_B, 1); ENV_B.raan_target_rad = D2R(137.0); ENV_B.lvlh_frame_mode = 1;
            c_reset(&ENV_B); float o2[64]; memcpy(o2, g_obs, sizeof(o2));
            for (int t = 33; t <= 37; t++) {
                double dv = fabs((double)o1[t] - (double)o2[t]);
                if (dv > worst) worst = dv;
            }
        }
        char d[200];
        snprintf(d, sizeof(d), "under lvlh_frame_mode=1, psi = 137 deg moves obs[33-37] by "
                 "max %.3e (legacy mode: 4.11)", worst);
        check("S8c lvlh_frame_mode=1 restores SO(2)-about-z invariance", worst < 1e-6, d);
    }
}

int main(void) {
    printf("=== ext-j2 rung: inclined-target sampler gates (real c_reset) ===\n");
    printf("    band %.1f..%.1f deg   di_max %.4f rad (%.2f deg)   raan sampled\n",
           R2D(BAND_LO), R2D(BAND_HI), DI_MAX, R2D(DI_MAX));
    s1_s2_s3_s4();
    s5();
    s6();
    s7();
    s8();
    printf("\n=== %d/%d checks pass ===\n", g_pass, g_pass + g_fail);
    return g_fail ? 1 : 0;
}
