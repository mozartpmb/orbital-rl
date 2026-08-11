/* bench_policy.c — single-core inference latency benchmark for policy.c.
 *
 * Reports median / p90 / p99 / p99.9 / max per-call wall time over N calls,
 * plus the real-time margin against the 60 s guidance decision period.
 *
 * Observations: by default a deterministic synthetic sweep bounded by the
 * environment's Box(-2, 2) observation space. Pass --obs FILE (raw float32,
 * row-major N x 38, as written by test_parity.py --dump-obs) to drive the
 * benchmark with real harvested observations instead. Timing is insensitive to
 * the data — the graph is dense and branch-free — but measuring on real frames
 * removes the question.
 *
 * The timing array is allocated once before the measured region; the measured
 * region itself performs no allocation, no I/O and no syscalls other than the
 * clock read.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "policy.h"

/* Highest-resolution monotonic clock available.
 * Darwin's clock_gettime(CLOCK_MONOTONIC) quantises to 1 us, which is 2% of a
 * single inference — clock_gettime_nsec_np(CLOCK_UPTIME_RAW) is the raw
 * mach timebase (~41.7 ns tick on Apple silicon). POSIX fallback elsewhere. */
#if defined(__APPLE__)
#include <time.h>
static inline uint64_t now_ns(void)
{
    return clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
}
#else
static inline uint64_t now_ns(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}
#endif

static int cmp_double(const void* a, const void* b)
{
    const double x = *(const double*)a, y = *(const double*)b;
    return (x > y) - (x < y);
}

static double pct(const double* sorted, size_t n, double p)
{
    if (n == 0) return 0.0;
    size_t idx = (size_t)(p * (double)(n - 1) + 0.5);
    if (idx >= n) idx = n - 1;
    return sorted[idx];
}

/* Deterministic bounded synthetic observation (xorshift32, mapped to [-1,1]). */
static void synth_obs(float* obs, uint32_t* rng)
{
    for (int i = 0; i < POLICY_OBS_DIM; i++) {
        uint32_t x = *rng;
        x ^= x << 13; x ^= x >> 17; x ^= x << 5;
        *rng = x;
        obs[i] = (float)((double)x / 2147483648.0 - 1.0);
    }
}

int main(int argc, char** argv)
{
    size_t calls = 100000;
    size_t warmup = 5000;
    const char* obs_path = NULL;
    double period_s = 60.0;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--calls") && i + 1 < argc)   calls  = (size_t)strtoul(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--warmup") && i + 1 < argc) warmup = (size_t)strtoul(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--obs") && i + 1 < argc) obs_path = argv[++i];
        else if (!strcmp(argv[i], "--period") && i + 1 < argc) period_s = atof(argv[++i]);
        else {
            fprintf(stderr, "usage: %s [--calls N] [--warmup N] [--obs FILE] [--period SEC]\n", argv[0]);
            return 2;
        }
    }

    /* ---- observation source (allocated up front, outside the timed region) */
    float*  obs_pool = NULL;
    size_t  n_obs = 0;
    if (obs_path) {
        FILE* f = fopen(obs_path, "rb");
        if (!f) { perror("fopen --obs"); return 1; }
        fseek(f, 0, SEEK_END);
        long bytes = ftell(f);
        fseek(f, 0, SEEK_SET);
        n_obs = (size_t)bytes / (sizeof(float) * POLICY_OBS_DIM);
        if (n_obs == 0) { fprintf(stderr, "--obs file too small\n"); fclose(f); return 1; }
        obs_pool = (float*)malloc(n_obs * POLICY_OBS_DIM * sizeof(float));
        if (!obs_pool) { fprintf(stderr, "oom\n"); fclose(f); return 1; }
        if (fread(obs_pool, sizeof(float), n_obs * POLICY_OBS_DIM, f)
            != n_obs * POLICY_OBS_DIM) {
            fprintf(stderr, "short read\n"); fclose(f); return 1;
        }
        fclose(f);
    } else {
        n_obs = 4096;
        obs_pool = (float*)malloc(n_obs * POLICY_OBS_DIM * sizeof(float));
        if (!obs_pool) { fprintf(stderr, "oom\n"); return 1; }
        uint32_t rng = 0x12345678u;
        for (size_t i = 0; i < n_obs; i++) synth_obs(obs_pool + i * POLICY_OBS_DIM, &rng);
    }

    double* dt = (double*)malloc(calls * sizeof(double));
    if (!dt) { fprintf(stderr, "oom\n"); return 1; }

    policy_state_t st;
    policy_reset(&st);

    volatile int sink = 0;
    float logits[POLICY_ACTIONS];
    float value;
    int action;

    /* ---- warmup (icache/dcache, branch predictors) ------------------------ */
    for (size_t i = 0; i < warmup; i++) {
        policy_infer(&st, obs_pool + (i % n_obs) * POLICY_OBS_DIM, logits, &value, &action);
        sink += action;
        if ((i % 32) == 31) policy_reset(&st);
    }

    /* ---- measured region -------------------------------------------------- */
    const uint64_t g0 = now_ns();
    for (size_t i = 0; i < calls; i++) {
        const float* obs = obs_pool + (i % n_obs) * POLICY_OBS_DIM;
        const uint64_t a = now_ns();
        policy_infer(&st, obs, logits, &value, &action);
        const uint64_t b = now_ns();
        dt[i] = 1e-9 * (double)(b - a);
        sink += action;
        /* Periodic reset keeps the recurrent state bounded over 1e5 calls; it
         * mirrors the episode-boundary reset and costs ~1 us every 512 calls. */
        if ((i % 512) == 511) policy_reset(&st);
    }
    const uint64_t g1 = now_ns();
    const double total = 1e-9 * (double)(g1 - g0);

    /* ---- clock overhead (reported, never subtracted) ---------------------- */
    double clk_med;
    {
        enum { PROBES = 20000 };
        static double probe[PROBES];
        for (size_t i = 0; i < PROBES; i++) {
            const uint64_t p0 = now_ns();
            const uint64_t p1 = now_ns();
            probe[i] = 1e-9 * (double)(p1 - p0);
        }
        qsort(probe, PROBES, sizeof(double), cmp_double);
        clk_med = probe[PROBES / 2];
    }

    qsort(dt, calls, sizeof(double), cmp_double);
    double sum = 0.0;
    for (size_t i = 0; i < calls; i++) sum += dt[i];

    const double med = pct(dt, calls, 0.50);
    const double p99 = pct(dt, calls, 0.99);

    printf("=== policy.c inference benchmark (single core) ===\n");
    printf("build            : %s\n", policy_build_variant());
    printf("ckpt sha256      : %s\n", policy_ckpt_sha256());
    printf("obs source       : %s (%zu frames)\n",
           obs_path ? obs_path : "synthetic deterministic", n_obs);
    printf("calls            : %zu (warmup %zu, excluded)\n", calls, warmup);
    printf("state size       : %zu B   stack scratch: %d B   weights: %zu B rodata\n",
           sizeof(policy_state_t), POLICY_STACK_SCRATCH_BYTES,
           (size_t)(139281u * sizeof(float)));
    printf("--- per-call latency (us) ---\n");
    printf("  min            : %10.3f\n", dt[0] * 1e6);
    printf("  p50 (median)   : %10.3f\n", med * 1e6);
    printf("  mean           : %10.3f\n", (sum / (double)calls) * 1e6);
    printf("  p90            : %10.3f\n", pct(dt, calls, 0.90) * 1e6);
    printf("  p99            : %10.3f\n", p99 * 1e6);
    printf("  p99.9          : %10.3f\n", pct(dt, calls, 0.999) * 1e6);
    printf("  max            : %10.3f\n", dt[calls - 1] * 1e6);
    printf("  clock overhead : %10.4f  (median of 20000 back-to-back reads; not subtracted)\n",
           clk_med * 1e6);
    printf("  note           : max/p99.9 outliers are OS scheduler preemptions on a\n");
    printf("                   non-real-time host, not algorithmic variance — the work\n");
    printf("                   per call is a fixed MAC count with no data-dependent path.\n");
    printf("--- throughput ---\n");
    printf("  wall total     : %.4f s for %zu calls -> %.0f calls/s\n",
           total, calls, (double)calls / total);
    printf("--- real-time margin vs %.0f s decision period ---\n", period_s);
    printf("  duty cycle p50 : %.3e  (1 call per %.0f s)\n", med / period_s, period_s);
    printf("  duty cycle p99 : %.3e\n", p99 / period_s);
    printf("  margin p99     : %.0fx  (period / p99 latency)\n", period_s / p99);
    printf("  budget used p99: %.7f %% of the 60 s epoch\n", 100.0 * p99 / period_s);

    free(dt);
    free(obs_pool);
    return sink == 0x7fffffff ? 1 : 0;   /* keep `sink` observable */
}
