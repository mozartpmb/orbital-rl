/* policy.h — standalone single-precision inference for the orbital-rendezvous
 *            RL policy (PufferLib `Default` + `LSTMWrapper`, T3 canonical).
 *
 * Flight-software contract
 * ------------------------
 *  * No dynamic allocation, ever. All storage is either const .rodata (weights,
 *    ~557 KB) or caller-owned (policy_state_t, 1040 B) plus a bounded stack
 *    frame (POLICY_STACK_SCRATCH_BYTES) inside policy_infer().
 *  * No I/O, no syscalls, no globals with mutable state, no errno use.
 *  * Deterministic: identical (state, obs) always yields identical outputs on a
 *    given build. No RNG, no time dependence, no data-dependent branching that
 *    changes arithmetic order.
 *  * Bounded, branch-count-stable execution: the work per call is a fixed
 *    number of multiply-accumulates independent of input values.
 *  * Fail-safe state commit: the recurrent state is written only after a call
 *    fully succeeds. A rejected (non-finite) observation leaves h/c untouched,
 *    so a single corrupt sensor frame cannot poison the filter memory.
 *  * Reentrant / thread-safe: all mutable state is behind the caller's pointer.
 *
 * Graph (shapes asserted against policy_weights.h at compile time):
 *
 *   obs[38] --Linear(38x128)+b--> GELU --> LSTMCell(128->128) --+--> Linear(128x16) --> logits[16] --> argmax
 *                                              ^      |         |
 *                                              |      v         +--> Linear(128x1)  --> value (unused in control)
 *                                          h,c (carried across decision epochs)
 *
 * Build-time options
 * ------------------
 *   POLICY_GELU_ERF   (default) GELU = 0.5x(1+erf(x/sqrt2)) via erff  — matches
 *                     torch nn.GELU(approximate='none') to float32 rounding.
 *   POLICY_GELU_POLY  same formula, erf via Abramowitz & Stegun 7.1.26 using
 *                     only expf  — removes the erff dependency (|err| <= 1.5e-7).
 *   POLICY_GELU_TANH  torch's approximate='tanh' variant, tanhf only.
 *   POLICY_ACC_DOUBLE accumulate dot products in double (kept off by default:
 *                     the deliverable is a single-precision implementation).
 */
#ifndef ORBITAL_POLICY_H
#define ORBITAL_POLICY_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define POLICY_OBS_DIM   38
#define POLICY_HIDDEN    128
#define POLICY_ACTIONS   16
#define POLICY_GATES     (4 * POLICY_HIDDEN)

/* Bounded stack scratch declared inside policy_infer():
 * enc[H] + gates[4H] + h_new[H] + c_new[H] + logits[A]. Compiler frame overhead
 * is on top of this; measure with -fstack-usage for a flight budget. */
#define POLICY_STACK_SCRATCH_BYTES \
    ((3 * POLICY_HIDDEN + POLICY_GATES + POLICY_ACTIONS) * (int)sizeof(float))

/* Status codes. Negative == failure; the recurrent state is unmodified. */
typedef enum {
    POLICY_OK                = 0,
    POLICY_ERR_NULL          = -1,  /* NULL state or obs pointer                */
    POLICY_ERR_UNINIT        = -2,  /* state never passed through policy_reset  */
    POLICY_ERR_OBS_NONFINITE = -3,  /* NaN/Inf in the observation vector        */
    POLICY_ERR_DIVERGED      = -4   /* NaN/Inf produced internally (should be
                                     * unreachable; trips only on corrupt state
                                     * or a hardware fault)                     */
} policy_status_t;

#define POLICY_STATE_MAGIC 0x4F52424Du /* 'ORBM' — init guard, not a checksum */

/* Recurrent state. Caller-owned; sizeof == 1040 B. Zero-initialising the struct
 * is NOT sufficient — call policy_reset() so the magic guard is set. */
typedef struct {
    float    h[POLICY_HIDDEN];
    float    c[POLICY_HIDDEN];
    uint32_t magic;   /* POLICY_STATE_MAGIC once reset                        */
    uint32_t steps;   /* decision epochs since reset (saturating, diagnostics) */
} policy_state_t;

/* Clear the recurrent state to the episode-start condition (h = c = 0), which
 * is exactly what the torch eval harness does at every episode boundary. */
void policy_reset(policy_state_t* st);

/* One decision epoch.
 *   st      : recurrent state, updated in place on success only
 *   obs     : POLICY_OBS_DIM observation floats, as produced by the C env
 *   logits  : optional out, POLICY_ACTIONS floats (may be NULL)
 *   value   : optional out, critic estimate (may be NULL)
 *   action  : optional out, argmax(logits); ties -> lowest index (may be NULL)
 * Returns POLICY_OK or a negative policy_status_t. */
policy_status_t policy_infer(policy_state_t* st,
                             const float* obs,
                             float* logits,
                             float* value,
                             int* action);

/* Convenience wrapper for the control path: returns the greedy action, or a
 * negative policy_status_t on failure. */
int policy_act(policy_state_t* st, const float* obs);

/* Provenance of the compiled-in weights (from the checkpoint at export time). */
const char* policy_ckpt_sha256(void);
const char* policy_weights_blob_sha256(void);
/* Human-readable identifier of the GELU/accumulation build variant. */
const char* policy_build_variant(void);

#ifdef __cplusplus
}
#endif

#endif /* ORBITAL_POLICY_H */
