/* ext-3d V4 — the three sampling / frame gates, run against the real c_reset
 * and the real fill_observations (not a replica).
 *
 * G1  SAMPLER (3d_REDTEAM BLOCKER-1). Realized Δi_rel = angle(ĥ_s, ĥ_t) must
 *     satisfy max/knob = 1.000 with 0.0% over, at i_t ∈ {0, 51.6, 90, 98}°.
 *     The deleted ī-disc sampler realized 1.14× / 1.64× / 10.7× / 21.4× there.
 *
 * G2  PHASE GAP (3d_REDTEAM m2). phase_gap_mode=1 must control the realized
 *     mean-longitude gap PER DRAW: max |realized − requested| ≈ 0. The KS test
 *     the design proposed cannot see the failure — the unpatched knob leaves
 *     the realized distribution uniform (KS 0.0083) while decorrelating it from
 *     the request (per-draw error p50 = 90.4°). Gate on the per-draw error.
 *
 * G3  FRAME (3d_REDTEAM MAJOR-1). With i_t = Ω_t = 0 pinned, the residual gauge
 *     freedom is a rotation about ĥ_t. Every success-relevant channel must be
 *     invariant under it. Also reported: the plain equinoctial Δλ = M+ω+Ω under
 *     a general SO(3) rotation (frame-VARIANT, up to 59.8° at Δi = 1°) against
 *     the env's target-plane-gauge Δλ (invariant).
 *
 * Build & run:
 *   cc -O2 -lm -I pufferlib/pufferlib/ocean/orbital \
 *      scripts/orbital/ext3d/v4_gates.c -o /tmp/v4 && /tmp/v4
 */
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "orbital.h"

static float  g_obs[64];
static int    g_act[1];
static float  g_rew[1];
static unsigned char g_term[1];

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
}

static double plane_angle(const Orbit* a, const Orbit* b) {
    double ax, ay, az, bx, by, bz;
    orb_hhat(a, &ax, &ay, &az);
    orb_hhat(b, &bx, &by, &bz);
    double cx = ay*bz - az*by, cy = az*bx - ax*bz, cz = ax*by - ay*bx;
    double dot = ax*bx + ay*by + az*bz;
    return atan2(sqrt(cx*cx + cy*cy + cz*cz), dot);
}

/* ── G1 ─────────────────────────────────────────────────────────────────── */
static void g1(void) {
    printf("\n== G1  relative-plane sampler: realized Δi_rel vs the knob ==\n");
    printf("%8s %9s %10s %12s %12s %10s\n",
           "i_t[deg]", "knob[deg]", "n", "max/knob", "p90/knob", "frac>knob");
    const double its[] = {0.0, 51.6, 90.0, 98.0};
    const double knobs[] = {0.05, 0.25, 0.40, 1.0, 2.0};
    for (int a = 0; a < 4; a++) {
        for (int b = 0; b < 5; b++) {
            static Orbital e; env_base(&e);
            e.i_target_rad = its[a] * M_PI / 180.0;
            e.raan_target_rad = 0.0;
            e.di_max_rad = knobs[b] * M_PI / 180.0;
            long n = 40000, over = 0;
            double mx = 0.0;
            static double buf[40000];
            srand(1000 * a + b + 7);
            for (long k = 0; k < n; k++) {
                c_reset(&e);
                double d = plane_angle(&e.sat.orbit, &e.target);
                buf[k] = d;
                if (d > mx) mx = d;
                if (d > e.di_max_rad * (1.0 + 1e-12)) over++;
            }
            /* p90 */
            for (long i = 1; i < n; i++) {          /* insertion-free: partial */
                (void)i;
                break;
            }
            double p90 = 0.0;
            {   /* cheap p90 by counting */
                double lo = 0.0, hi = e.di_max_rad;
                for (int it = 0; it < 50; it++) {
                    double mid = 0.5*(lo+hi); long c = 0;
                    for (long k = 0; k < n; k++) if (buf[k] <= mid) c++;
                    if (c < (long)(0.9*n)) lo = mid; else hi = mid;
                }
                p90 = 0.5*(lo+hi);
            }
            printf("%8.1f %9.2f %10ld %12.4f %12.4f %9.2f%%\n",
                   its[a], knobs[b], n, mx / e.di_max_rad, p90 / e.di_max_rad,
                   100.0 * over / (double)n);
        }
    }
}

/* ── G2 ─────────────────────────────────────────────────────────────────── */
static void g2(void) {
    printf("\n== G2  phase_gap_mode=1: realized mean-longitude gap, PER DRAW ==\n");
    printf("%8s %8s %10s %14s %14s %14s\n",
           "i_t[deg]", "di[deg]", "n", "err p50[deg]", "err p90[deg]", "err max[deg]");
    const double its[] = {0.0, 51.6};
    const double dis[] = {0.0, 0.40, 1.0, 2.0};
    for (int a = 0; a < 2; a++) {
        for (int b = 0; b < 4; b++) {
            static Orbital e; env_base(&e);
            e.i_target_rad = its[a] * M_PI / 180.0;
            e.di_max_rad = (dis[b] > 0.0) ? dis[b] * M_PI / 180.0 : -1.0;
            e.phase_gap_fixed = -1.0;
            long n = 20000;
            static double err[20000];
            double mx = 0.0;
            srand(50000 + 100*a + b);
            for (long k = 0; k < n; k++) {
                /* Requested gap: reproduce c_reset's own draw is impossible
                 * from outside, so request an EXACT gap instead — the gate is
                 * on |realized − requested| per draw either way. */
                e.phase_gap_fixed = (k % 360) * M_PI / 180.0;
                c_reset(&e);
                PlaneGauge g; gauge_from_orbit(&e.target, &g);
                double realized = wrap_pi(orb_lambda_gauge(&e.sat.orbit, &g)
                                        - orb_lambda_gauge(&e.target, &g));
                /* sign convention (T3): the knob places the TARGET AHEAD, so
                 * realized Δλ = λ_s − λ_t = −gap */
                double d = fabs(wrap_pi(realized + e.phase_gap_fixed));
                err[k] = d;
                if (d > mx) mx = d;
            }
            double p50 = 0.0, p90 = 0.0;
            {   double lo = 0.0, hi = M_PI;
                for (int it = 0; it < 60; it++) {
                    double mid = 0.5*(lo+hi); long c = 0;
                    for (long k = 0; k < n; k++) if (err[k] <= mid) c++;
                    if (c < n/2) lo = mid; else hi = mid;
                }
                p50 = 0.5*(lo+hi);
                lo = 0.0; hi = M_PI;
                for (int it = 0; it < 60; it++) {
                    double mid = 0.5*(lo+hi); long c = 0;
                    for (long k = 0; k < n; k++) if (err[k] <= mid) c++;
                    if (c < (long)(0.9*n)) lo = mid; else hi = mid;
                }
                p90 = 0.5*(lo+hi);
            }
            printf("%8.1f %8.2f %10ld %14.3e %14.3e %14.3e\n",
                   its[a], dis[b], n, p50*180/M_PI, p90*180/M_PI, mx*180/M_PI);
        }
    }
}

/* ── G3 ─────────────────────────────────────────────────────────────────── */
static void rot_orbit(Orbit* o, const double ax[3], double ang) {
    double x, y, z, vx, vy, vz;
    orbit_to_cartesian(o, &x, &y, &z, &vx, &vy, &vz);
    double c = cos(ang), s = sin(ang);
    double p[3] = {x, y, z}, v[3] = {vx, vy, vz}, out[2][3];
    double* src[2] = {p, v};
    for (int q = 0; q < 2; q++) {
        double* u = src[q];
        double dot = ax[0]*u[0] + ax[1]*u[1] + ax[2]*u[2];
        double cr[3] = {ax[1]*u[2] - ax[2]*u[1],
                        ax[2]*u[0] - ax[0]*u[2],
                        ax[0]*u[1] - ax[1]*u[0]};
        for (int i = 0; i < 3; i++)
            out[q][i] = u[i]*c + cr[i]*s + ax[i]*dot*(1.0 - c);
    }
    cartesian_to_elements(out[0][0], out[0][1], out[0][2],
                          out[1][0], out[1][1], out[1][2], o);
}

static void g3(void) {
    printf("\n== G3  frame gates ==\n");
    /* (a) rotations about ĥ_t — the residual gauge freedom at i_t = Ω_t = 0 */
    const int KEEP[] = {0,1,4,5,6,7,8,13,14,15,21,22,23,24,25,26,27,28,33,34,35,36,37};
    const int VAR[]  = {2,3,9,10,11,12,16};
    double worst_keep = 0.0, worst_var = 0.0, worst_phi = 0.0;
    double worst_dist = 0.0, worst_relv = 0.0, worst_dlam = 0.0;
    int wk_slot = -1;
    static Orbital e; env_base(&e);
    e.di_max_rad = 1.0 * M_PI / 180.0;
    srand(31337);
    long n = 4000;
    for (long k = 0; k < n; k++) {
        c_reset(&e);
        float o0[64]; memcpy(o0, g_obs, sizeof(o0));
        double phi0 = compute_phi(&e);
        PlaneGauge gg0; gauge_from_orbit(&e.target, &gg0);
        double dl0 = wrap_pi(orb_lambda_gauge(&e.sat.orbit, &gg0)
                           - orb_lambda_gauge(&e.target, &gg0));
        double sx,sy,sz,svx,svy,svz, tx,ty,tz,tvx,tvy,tvz;
        orbit_to_cartesian(&e.sat.orbit,&sx,&sy,&sz,&svx,&svy,&svz);
        orbit_to_cartesian(&e.target,&tx,&ty,&tz,&tvx,&tvy,&tvz);
        double d0 = sqrt((sx-tx)*(sx-tx)+(sy-ty)*(sy-ty)+(sz-tz)*(sz-tz));
        double rv0 = sqrt((svx-tvx)*(svx-tvx)+(svy-tvy)*(svy-tvy)+(svz-tvz)*(svz-tvz));

        double hx,hy,hz; orb_hhat(&e.target,&hx,&hy,&hz);
        double ax[3] = {hx,hy,hz};
        double ang = (rand()/(double)RAND_MAX) * 2.0 * M_PI;
        rot_orbit(&e.sat.orbit, ax, ang);
        rot_orbit(&e.target,    ax, ang);
        fill_observations(&e);
        double phi1 = compute_phi(&e);
        PlaneGauge gg1; gauge_from_orbit(&e.target, &gg1);
        double dl1 = wrap_pi(orb_lambda_gauge(&e.sat.orbit, &gg1)
                           - orb_lambda_gauge(&e.target, &gg1));
        orbit_to_cartesian(&e.sat.orbit,&sx,&sy,&sz,&svx,&svy,&svz);
        orbit_to_cartesian(&e.target,&tx,&ty,&tz,&tvx,&tvy,&tvz);
        double d1 = sqrt((sx-tx)*(sx-tx)+(sy-ty)*(sy-ty)+(sz-tz)*(sz-tz));
        double rv1 = sqrt((svx-tvx)*(svx-tvx)+(svy-tvy)*(svy-tvy)+(svz-tvz)*(svz-tvz));

        for (unsigned i = 0; i < sizeof(KEEP)/sizeof(int); i++) {
            double d = fabs((double)g_obs[KEEP[i]] - (double)o0[KEEP[i]]);
            if (d > worst_keep) { worst_keep = d; wk_slot = KEEP[i]; }
        }
        for (unsigned i = 0; i < sizeof(VAR)/sizeof(int); i++) {
            double d = fabs((double)g_obs[VAR[i]] - (double)o0[VAR[i]]);
            if (d > worst_var) worst_var = d;
        }
        if (fabs(phi1-phi0) > worst_phi) worst_phi = fabs(phi1-phi0);
        if (fabs(d1-d0)/d0 > worst_dist) worst_dist = fabs(d1-d0)/d0;
        if (fabs(rv1-rv0) > worst_relv) worst_relv = fabs(rv1-rv0);
        if (fabs(wrap_pi(dl1-dl0)) > worst_dlam) worst_dlam = fabs(wrap_pi(dl1-dl0));
    }
    printf("(a) rotation about ĥ_t, di_max=1.0°, n=%ld\n", n);
    printf("    success-relevant obs slots  max |Δ| = %.3e   (worst slot %d)\n",
           worst_keep, wk_slot);
    printf("    grandfathered raw-ω/θ slots max |Δ| = %.3e   (variant BY DESIGN)\n",
           worst_var);
    printf("    Φ(mode 2)                   max |Δ| = %.3e\n", worst_phi);
    printf("    Δλ (target gauge)           max |Δ| = %.3e deg\n",
           worst_dlam*180/M_PI);
    printf("    |Δr| rel / |Δv| abs         max      = %.3e / %.3e m/s\n",
           worst_dist, worst_relv);

    /* (b) general SO(3): env λ (target gauge) vs the plain equinoctial λ */
    printf("(b) general SO(3), target-plane gauge vs plain λ = M+ω+Ω\n");
    printf("%10s %10s %16s %16s %16s\n", "di[deg]", "n",
           "gauge max[deg]", "plain p90[deg]", "plain max[deg]");
    const double dis[] = {0.0, 0.25, 1.0, 2.0};
    for (int b = 0; b < 4; b++) {
        static Orbital f; env_base(&f);
        f.di_max_rad = (dis[b] > 0.0) ? dis[b]*M_PI/180.0 : -1.0;
        srand(999 + b);
        long m = 4000;
        double gmax = 0.0, pmax = 0.0;
        static double pv[4000];
        for (long k = 0; k < m; k++) {
            c_reset(&f);
            PlaneGauge gg; gauge_from_orbit(&f.target, &gg);
            double dg0 = wrap_pi(orb_lambda_gauge(&f.sat.orbit, &gg)
                               - orb_lambda_gauge(&f.target, &gg));
            double dp0 = wrap_pi(orb_lambda(&f.sat.orbit) - orb_lambda(&f.target));
            double u[3];
            double nn;
            do {
                u[0] = 2.0*(rand()/(double)RAND_MAX)-1.0;
                u[1] = 2.0*(rand()/(double)RAND_MAX)-1.0;
                u[2] = 2.0*(rand()/(double)RAND_MAX)-1.0;
                nn = sqrt(u[0]*u[0]+u[1]*u[1]+u[2]*u[2]);
            } while (nn < 1e-6 || nn > 1.0);
            u[0]/=nn; u[1]/=nn; u[2]/=nn;
            double ang = (rand()/(double)RAND_MAX)*2.0*M_PI;
            rot_orbit(&f.sat.orbit, u, ang);
            rot_orbit(&f.target,    u, ang);
            PlaneGauge g2f; gauge_from_orbit(&f.target, &g2f);
            double dg1 = wrap_pi(orb_lambda_gauge(&f.sat.orbit, &g2f)
                               - orb_lambda_gauge(&f.target, &g2f));
            double dp1 = wrap_pi(orb_lambda(&f.sat.orbit) - orb_lambda(&f.target));
            double eg = fabs(wrap_pi(dg1-dg0)), ep = fabs(wrap_pi(dp1-dp0));
            pv[k] = ep;
            if (eg > gmax) gmax = eg;
            if (ep > pmax) pmax = ep;
        }
        double p90 = 0.0;
        {   double lo = 0.0, hi = M_PI;
            for (int it = 0; it < 60; it++) {
                double mid = 0.5*(lo+hi); long c = 0;
                for (long k = 0; k < m; k++) if (pv[k] <= mid) c++;
                if (c < (long)(0.9*m)) lo = mid; else hi = mid;
            }
            p90 = 0.5*(lo+hi);
        }
        printf("%10.2f %10ld %16.3e %16.4f %16.4f\n", dis[b], m,
               gmax*180/M_PI, p90*180/M_PI, pmax*180/M_PI);
    }
}

int main(void) {
    printf("=== ext-3d V4 gates (real c_reset / fill_observations) ===\n");
    g1();
    g2();
    g3();
    return 0;
}
