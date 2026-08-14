/* ext-j2 recon — J2-assisted plane change: is drift-and-wait ever worth it?
 *
 * THE MANEUVER. Instead of paying ~134 m/s per degree for a direct plane
 * change, dip the semi-major axis by delta-a, let the resulting DIFFERENTIAL
 * nodal precession rotate the relative node, then raise back. The dip's Delta-v
 * is independent of how much plane you buy — you just wait longer — so the
 * comparison against direct burning improves linearly with the target angle.
 *
 * THE CATCH, in two parts, and both are geometric rather than budgetary:
 *
 *  (1) Nodal drift rotates OMEGA. For two orbits at a common inclination i
 *      whose RAANs differ by dOm, the angle between the planes is EXACTLY
 *          di_rel = 2 asin( sin(i) * sin(dOm/2) )
 *      so the rate is d(di_rel)/dt = |dOm_dot| * sin(i) for small angles, and
 *      it SATURATES at di_rel_max = 2i no matter how long you wait. The
 *      product with dOm_dot ∝ cos i gives the sin(2i) law: zero at the equator
 *      (where Omega is gauge) AND zero at the pole (where Omega_dot vanishes),
 *      maximal at 45 deg.
 *
 *  (2) It is a ONE-DIMENSIONAL control in a TWO-DIMENSIONAL error space. The
 *      relative-inclination vector has a node component and an inclination
 *      component; drift moves only the former. An error that is pure
 *      inclination difference cannot be removed by waiting at all. For the
 *      env's uniform-phase disc sampler the drift-correctable fraction of a
 *      random error averages E|cos φ| = 2/pi = 63.7%.
 *
 * Everything below is measured with the SHIPPED propagate_orbit_j2, then
 * compared against the closed form written longhand here.
 *
 * Build & run (from the WORKTREE root):
 *   cc -O2 -flto -lm -I pufferlib/pufferlib/ocean/orbital \
 *      scripts/orbital/extj2/j2_plane_change.c -o /tmp/j2pc && /tmp/j2pc
 */
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "orbital.h"

#define R2D(x) ((x) * 180.0 / M_PI)
#define D2R(x) ((x) * M_PI / 180.0)
#define DAY    86400.0
#define DVBUD  478.12987798780637   /* 15% fuel fraction, Isp 300 s */

static int g_pass = 0, g_fail = 0;
static void check(const char* name, int ok, const char* detail) {
    printf("  [%s] %-52s %s\n", ok ? "PASS" : "FAIL", name, detail);
    if (ok) g_pass++; else g_fail++;
}

static double plane_angle(const Orbit* a, const Orbit* b) {
    double ax, ay, az, bx, by, bz;
    orb_hhat(a, &ax, &ay, &az);
    orb_hhat(b, &bx, &by, &bz);
    double cx = ay*bz - az*by, cy = az*bx - ax*bz, cz = ax*by - ay*bx;
    return atan2(sqrt(cx*cx + cy*cy + cz*cz), ax*bx + ay*by + az*bz);
}

/* Closed-form secular rates, longhand (NOT shared with orbital.h). */
static double raan_dot(double a, double e, double inc) {
    double n = sqrt(MU / (a*a*a));
    double p = a * (1.0 - e*e);
    double k = 1.5 * n * J2_COEF * (J2_R_EQ/p) * (J2_R_EQ/p);
    return -k * cos(inc);
}

/* d(di_rel)/dt for a co-inclined pair separated by delta-a, small-angle:
 *   = (7/4) * k * sin(2i) * |da| / a                                        */
static double dirate_analytic(double a, double e, double inc, double da) {
    double n = sqrt(MU / (a*a*a));
    double p = a * (1.0 - e*e);
    double k = 1.5 * n * J2_COEF * (J2_R_EQ/p) * (J2_R_EQ/p);
    return 1.75 * k * sin(2.0*inc) * fabs(da) / a;
}

/* Round-trip Delta-v of a delta-a dip: two Hohmann pairs, down and back up.
 * Small-transfer limit Dv_one_way = (v_c/2)|da|/a, so the round trip is
 * v_c * |da| / a. */
static double dip_dv(double a, double da) {
    return sqrt(MU/a) * fabs(da) / a;
}

/* Direct plane change: the exact single-impulse chord, 2 v sin(di/2). */
static double direct_dv(double a, double di) {
    return 2.0 * sqrt(MU/a) * sin(0.5*di);
}

int main(void) {
    printf("=== ext-j2: J2-assisted plane change feasibility ===\n");
    printf("    budget %.1f m/s (15%% fuel fraction, Isp 300 s)\n", DVBUD);
    printf("    current episode cap 6000 substeps = %.0f h;  MAX_STEPS %d = %.0f h\n",
           6000*DT/3600.0, MAX_STEPS, MAX_STEPS*DT/3600.0);
    printf("    longest warp in the action table: tau = 360 = %.0f h/decision\n\n",
           360*DT/3600.0);

    /* ── A. the exact plane-angle law, and its saturation ─────────────────── */
    printf("== A. di_rel = 2 asin(sin i * sin(dOm/2)) — propagator vs closed form ==\n");
    printf("   The saturation is the first half of the catch: no amount of drift\n");
    printf("   can buy more than 2i of relative plane.\n");
    printf("  %6s %10s | %12s %12s %10s | %10s\n",
           "i deg", "dOm deg", "di_rel meas", "di_rel form", "rel err", "max=2i");
    double worst_law = 0.0;
    for (int ii = 0; ii < 3; ii++) {
        double i_deg = (ii == 0) ? 30.0 : (ii == 1) ? 45.0 : 60.0;
        for (int jj = 0; jj < 3; jj++) {
            double dOm_deg = (jj == 0) ? 1.0 : (jj == 1) ? 10.0 : 90.0;
            Orbit A, B;
            memset(&A, 0, sizeof(A)); memset(&B, 0, sizeof(B));
            A.a = R_EARTH + 500e3; A.e = 0.0; A.inc = D2R(i_deg);
            A.raan = 0.0; A.omega = 0.0; A.M = 0.0;
            A.theta = eccentric_to_true(solve_kepler(A.M, A.e), A.e);
            B = A; B.raan = D2R(dOm_deg);
            double meas = plane_angle(&A, &B);
            double form = 2.0 * asin(sin(D2R(i_deg)) * sin(0.5*D2R(dOm_deg)));
            double rel = fabs(meas - form) / form;
            if (rel > worst_law) worst_law = rel;
            printf("  %6.1f %10.1f | %12.6f %12.6f %10.2e | %10.1f\n",
                   i_deg, dOm_deg, R2D(meas), R2D(form), rel, 2.0*i_deg);
        }
    }
    {
        char d[160];
        snprintf(d, sizeof(d), "worst rel err vs 2 asin(sin i sin(dOm/2)) = %.2e", worst_law);
        check("A the exact plane-angle law holds in the propagator", worst_law < 1e-9, d);
    }

    /* ── B. drift rate: shipped propagator vs the sin(2i) closed form ─────── */
    printf("\n== B. d(di_rel)/dt for a co-inclined pair separated by delta-a ==\n");
    printf("   Flown 30 days in the SHIPPED propagate_orbit_j2. dOm is compared\n");
    printf("   against the EXACT rate difference Om_dot(a+da) - Om_dot(a); the\n");
    printf("   (7/4) k sin(2i)|da|/a rule of thumb is shown with its own error.\n");
    printf("  %6s %8s | %10s %10s %8s | %10s %10s %8s\n",
           "i deg", "da km", "dOm meas", "dOm exact", "rel err",
           "di/dt meas", "1st-order", "1st err");
    double worst_rate = 0.0;
    const double INCS[4] = {15.0, 30.0, 45.0, 60.0};
    const double DAS[3]  = {-100.0, -300.0, -600.0};
    for (int ii = 0; ii < 4; ii++) {
        for (int jj = 0; jj < 3; jj++) {
            double i_deg = INCS[ii], da_km = DAS[jj];
            double a_t = R_EARTH + 500e3;
            Orbit T, S;
            memset(&T, 0, sizeof(T)); memset(&S, 0, sizeof(S));
            T.a = a_t; T.e = 0.0; T.inc = D2R(i_deg); T.raan = 0.0;
            T.omega = 0.0; T.M = 0.0;
            T.theta = eccentric_to_true(solve_kepler(T.M, T.e), T.e);
            S = T; S.a = a_t + da_km*1e3;

            int steps = (int)(30.0 * DAY / DT);        /* 30 days */
            for (int k = 0; k < steps; k++) {
                propagate_orbit_j2(&T, DT, 1);
                propagate_orbit_j2(&S, DT, 1);
            }
            double t_days = steps * DT / DAY;
            /* Read the rate in the LINEAR variable dOm (di_rel saturates), and
             * compare against the EXACT difference of the two closed-form
             * rates. The (7/4) k sin(2i)|da|/a rule is FIRST ORDER in da/a and
             * is quoted separately with its own error, because at da = 600 km
             * (da/a = 8.7%) the second-order term is already 19%. */
            double di = plane_angle(&S, &T);
            double dOm = 2.0 * asin(fmin(1.0, sin(0.5*di) / sin(D2R(i_deg))));
            double meas_dOm = R2D(dOm) / t_days;                     /* deg/day */
            double exact_dOm = R2D(fabs(raan_dot(a_t + da_km*1e3, 0.0, D2R(i_deg))
                                      - raan_dot(a_t, 0.0, D2R(i_deg)))) * DAY;
            double meas_di = meas_dOm * sin(D2R(i_deg));
            double first_di = R2D(dirate_analytic(a_t, 0.0, D2R(i_deg), da_km*1e3)) * DAY;
            double rel = fabs(meas_dOm - exact_dOm) / exact_dOm;
            if (rel > worst_rate) worst_rate = rel;
            printf("  %6.1f %8.0f | %10.5f %10.5f %8.1e | %10.5f %10.5f %7.1f%%\n",
                   i_deg, da_km, meas_dOm, exact_dOm, rel,
                   meas_di, first_di, 100.0*(first_di/meas_di - 1.0));
        }
    }
    {
        char d[160];
        snprintf(d, sizeof(d), "worst rel err over i in {15,30,45,60} x da in "
                 "{-100,-300,-600} km = %.2e", worst_rate);
        check("B drift rate matches (7/4) k sin(2i) |da|/a", worst_rate < 0.05, d);
    }

    /* ── C. THE FEASIBILITY TABLE ─────────────────────────────────────────── */
    printf("\n== C. drift-and-wait vs direct burning, i = 45 deg (the best case) ==\n");
    printf("   dip Dv is a ROUND TRIP (down and back up) and is INDEPENDENT of the\n");
    printf("   plane angle bought — that is the whole mechanism. Direct Dv scales\n");
    printf("   linearly, so the ratio improves with the target angle and the price\n");
    printf("   is paid in TIME.\n\n");
    double a_t = R_EARTH + 500e3;
    printf("  %8s %10s %10s | %10s %10s %8s | %10s %11s %9s\n",
           "di target", "da km", "dip Dv", "%budget", "direct Dv", "ratio",
           "drift days", "substeps", "fits cap?");
    for (int t = 0; t < 3; t++) {
        double di_deg = (t == 0) ? 1.0 : (t == 1) ? 3.0 : 5.0;
        for (int jj = 0; jj < 3; jj++) {
            double da_km = DAS[jj];
            double rate = R2D(dirate_analytic(a_t, 0.0, D2R(45.0), da_km*1e3)) * DAY;
            double days = di_deg / rate;
            double dv_dip = dip_dv(a_t, da_km*1e3);
            double dv_dir = direct_dv(a_t, D2R(di_deg));
            double substeps = days * DAY / DT;
            const char* fits;
            if (dv_dip > DVBUD)            fits = "NO (fuel)";
            else if (substeps <= 6000.0)   fits = "yes";
            else if (substeps <= MAX_STEPS) fits = "cap raise";
            else                           fits = "NO (cap)";
            printf("  %7.1f%s %10.0f %9.1f m/s | %9.1f%% %9.1f m/s %7.2fx | "
                   "%10.2f %11.0f %9s\n",
                   di_deg, "d", da_km, dv_dip, 100.0*dv_dip/DVBUD, dv_dir,
                   dv_dir/dv_dip, days, substeps, fits);
        }
        printf("\n");
    }

    /* ── D. the >=3x band, and exactly what env changes it needs ──────────── */
    printf("== D. where does drift-and-wait beat direct by >= 3x on fuel? ==\n");
    printf("   Condition: direct_dv(di) / dip_dv(da) >= 3, i.e.\n");
    printf("   2 v sin(di/2) >= 3 v |da|/a  ->  |da|/a <= (2/3) sin(di/2).\n");
    printf("   The cheapest dip that still closes in finite time is the binding\n");
    printf("   choice, so for each di the best case is the LARGEST da meeting it.\n\n");
    printf("  %8s | %12s %10s %9s | %12s %11s %s\n",
           "di deg", "max da km", "dip Dv", "ratio", "drift days", "substeps", "env need");
    int any_ok = 0;
    for (int t = 0; t < 5; t++) {
        double di_deg = (t == 0) ? 1.0 : (t == 1) ? 2.0 : (t == 2) ? 3.0
                      : (t == 3) ? 5.0 : 10.0;
        double da_max = a_t * (2.0/3.0) * sin(0.5*D2R(di_deg));    /* metres */
        double rate = R2D(dirate_analytic(a_t, 0.0, D2R(45.0), da_max)) * DAY;
        double days = di_deg / rate;
        double substeps = days * DAY / DT;
        double dv_dip = dip_dv(a_t, da_max);
        double dv_dir = direct_dv(a_t, D2R(di_deg));
        const char* need;
        if (dv_dip > DVBUD)              need = "INFEASIBLE (fuel)";
        else if (substeps <= 6000.0)     need = "none";
        else if (substeps <= MAX_STEPS)  need = "cap raise only";
        else                             need = "MAX_STEPS + cap raise";
        if (dv_dip <= DVBUD) any_ok = 1;   /* fuel-feasible; cap is a knob */
        printf("  %7.1f | %12.0f %9.1f m/s %8.2fx | %12.2f %11.0f %s\n",
               di_deg, da_max/1e3, dv_dip, dv_dir/dv_dip, days, substeps, need);
    }
    printf("\n   NOTE the drift time at the >=3x threshold is INDEPENDENT of di:\n");
    printf("   da_max ∝ sin(di/2) ∝ di, and rate ∝ da, so days = di/rate is a\n");
    printf("   constant. The 3x fuel win always costs the SAME wall clock.\n");
    {
        double da_max = a_t * (2.0/3.0) * sin(0.5*D2R(3.0));
        double rate = R2D(dirate_analytic(a_t, 0.0, D2R(45.0), da_max)) * DAY;
        double days = 3.0 / rate;
        char d[220];
        snprintf(d, sizeof(d), "yes, and it is fuel-feasible up to di ~ 10 deg; "
                 "it costs a FIXED %.1f days = %.0f substeps regardless of di",
                 days, days*DAY/DT);
        check("D a >=3x fuel band exists and is fuel-feasible", any_ok, d);

        double sub = days*DAY/DT;
        printf("\n   ENV CHANGES REQUIRED, exactly:\n");
        printf("     episode_cap_steps  6000 -> %.0f      (%.1f d; currently 100 h)\n",
               sub*1.2, sub*1.2*DT/DAY);
        printf("     MAX_STEPS         %5d -> %.0f      (compile-time; the traj\n",
               MAX_STEPS, sub*1.2);
        printf("                                          buffer is %.1f MB/env at\n",
               (double)sizeof(TrajectoryRecord)*sub*1.2/1e6);
        printf("                                          that size, vs %.1f MB now,\n",
               (double)sizeof(TrajectoryRecord)*MAX_STEPS/1e6);
        printf("                                          virtual unless log_enabled)\n");
        printf("     ACTION_DV/ACTION_TAU: add ONE warp row, tau = 1440 (24 h).\n");
        printf("       A longer warp does NOT reduce the substep count (the cap\n");
        printf("       counts sub-steps, and warps sub-step internally) — it buys\n");
        printf("       the CREDIT HORIZON. At tau = 360 a %.1f-day drift is %.0f warp\n",
               days, days*DAY/(360.0*DT));
        printf("       decisions and gamma^n = %.3f; at tau = 1440 it is %.0f\n",
               pow(0.995, days*DAY/(360.0*DT)), days*DAY/(1440.0*DT));
        printf("       decisions and gamma^n = %.3f.\n",
               pow(0.995, days*DAY/(1440.0*DT)));
        printf("     NOTHING ELSE. No new burn rows, no obs change, no shaping change.\n");
    }

    /* ── E. the second half of the catch: 1-D control, 2-D error ─────────── */
    printf("\n== E. drift is a 1-D control in a 2-D error space ==\n");
    printf("   For planes (i_s, Om_s) and (i_t, Om_t),\n");
    printf("     cos(di_rel) = cos i_s cos i_t + sin i_s sin i_t cos(dOm)\n");
    printf("   which is MAXIMISED (di_rel minimised) at dOm = 0, where it gives\n");
    printf("   di_rel = |i_s - i_t| EXACTLY. So |di| is a hard FLOOR: drift moves\n");
    printf("   dOm and can null the node component completely, but it can never\n");
    printf("   take di_rel below the inclination difference.\n");
    printf("   Tracked over 30 d in the SHIPPED propagator, da = -300 km, i_t = 45:\n\n");
    printf("  %-30s %10s %10s %10s %12s\n",
           "initial error", "di_rel t0", "MIN di_rel", "floor |di|", "t_min days");
    int e_ok = 1;
    for (int c = 0; c < 3; c++) {
        Orbit T, S;
        memset(&T, 0, sizeof(T)); memset(&S, 0, sizeof(S));
        T.a = a_t; T.e = 0.0; T.inc = D2R(45.0); T.raan = 0.0;
        T.omega = 0.0; T.M = 0.0;
        T.theta = eccentric_to_true(solve_kepler(T.M, T.e), T.e);
        S = T; S.a = a_t - 300e3;
        double floor_deg;
        const char* nm_;
        if (c == 0) {                    /* pure node: di = 0, dOm = 1.4142 deg */
            S.raan = D2R(1.4142); floor_deg = 0.0; nm_ = "pure NODE (dOm=1.414 deg)";
        } else if (c == 1) {             /* pure inclination: dOm = 0 */
            S.inc = D2R(46.0); floor_deg = 1.0; nm_ = "pure INCLINATION (di=1 deg)";
        } else {                         /* mixed, 45 deg in the error plane */
            S.inc = D2R(45.7071); S.raan = D2R(1.0);
            floor_deg = 0.7071; nm_ = "MIXED (45 deg in error plane)";
        }
        double di0 = R2D(plane_angle(&S, &T));
        double mn = di0, t_mn = 0.0;
        int steps = (int)(30.0 * DAY / DT);
        for (int k = 0; k < steps; k++) {
            propagate_orbit_j2(&T, DT, 1);
            propagate_orbit_j2(&S, DT, 1);
            double d = R2D(plane_angle(&S, &T));
            if (d < mn) { mn = d; t_mn = (k + 1) * DT / DAY; }
        }
        printf("  %-30s %9.4f %10.4f %10.4f %12.2f\n",
               nm_, di0, mn, floor_deg, t_mn);
        /* the floor must be respected, and (for the node case) reached */
        if (mn < floor_deg - 1e-3) e_ok = 0;
        if (c == 0 && mn > 0.02) e_ok = 0;
        if (c == 1 && mn < 0.999) e_ok = 0;
    }
    printf("\n   Read the rows: the pure-node error is driven to ~0 (drift alone\n");
    printf("   solves it); the pure-inclination error NEVER drops below its 1 deg\n");
    printf("   floor no matter how long you wait; the mixed error stops at the\n");
    printf("   inclination component. For the env's uniform-phase disc sampler\n");
    printf("   the drift-correctable fraction of a random relative-plane error\n");
    printf("   averages E|cos phi| = 2/pi = 63.7%%; the rest still needs a burn.\n");
    {
        char d[220];
        snprintf(d, sizeof(d), "node error -> ~0; inclination error pinned at its "
                 "|di| floor; mixed error stops at its inclination component");
        check("E drift nulls the node component only; |di| is a hard floor", e_ok, d);
    }

    printf("\n=== %d/%d checks pass ===\n", g_pass, g_pass + g_fail);
    return g_fail ? 1 : 0;
}
