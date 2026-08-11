/* policy.c — single-precision, allocation-free inference for the orbital
 *            rendezvous policy. See policy.h for the contract.
 *
 * Reference semantics (pufferlib/pufferlib/models.py):
 *   Default.encode_observations : x.view(B,-1).float(); no normalization
 *   Default.encoder             : Sequential(Linear(38,128), GELU())
 *   LSTMWrapper.forward_eval    : torch.nn.LSTMCell(128,128), state (h,c)
 *   Default.decode_actions      : logits = Linear(128,16)(h); value = Linear(128,1)(h)
 *   eval_checkpoint.py (greedy) : action = argmax(logits)
 *
 * torch LSTMCell equations (docs/aten):
 *   gates = W_ih @ x + b_ih + W_hh @ h + b_hh          (b_ih AND b_hh both added)
 *   [i_pre, f_pre, g_pre, o_pre] = chunk(gates, 4)      (row blocks, in that order)
 *   i = sigmoid(i_pre)   f = sigmoid(f_pre)
 *   g = tanh(g_pre)      o = sigmoid(o_pre)
 *   c' = f * c + i * g
 *   h' = o * tanh(c')
 * W_ih is stored (4H, in) row-major, W_hh is (4H, H) row-major — so gate k of
 * unit j lives at row k*H + j in both, which is why a single fused row loop
 * over 4H rows reproduces the chunking with no reshaping.
 */

#include "policy.h"
#include "policy_weights.h"

#include <math.h>
#include <string.h>

/* ── compile-time shape agreement with the exported weights ───────────────── */
#define POLICY_STATIC_ASSERT(cond, msg) \
    typedef char policy_static_assert_##msg[(cond) ? 1 : -1]
POLICY_STATIC_ASSERT(POLICY_OBS_DIM == POLICY_WEIGHTS_OBS_DIM, obs_dim_mismatch);
POLICY_STATIC_ASSERT(POLICY_HIDDEN  == POLICY_WEIGHTS_HIDDEN,  hidden_mismatch);
POLICY_STATIC_ASSERT(POLICY_ACTIONS == POLICY_WEIGHTS_ACTIONS, action_dim_mismatch);

/* ── accumulator type ─────────────────────────────────────────────────────── */
#ifdef POLICY_ACC_DOUBLE
typedef double policy_acc_t;
#define POLICY_ACC_NAME "acc=double"
#else
typedef float policy_acc_t;
#define POLICY_ACC_NAME "acc=float32"
#endif

/* ── activations ──────────────────────────────────────────────────────────── */

/* Logistic sigmoid. Split at 0 so expf() never sees a large positive argument
 * (overflow-free for every finite input; identical value either branch). */
static inline float policy_sigmoidf(float x)
{
    if (x >= 0.0f) {
        return 1.0f / (1.0f + expf(-x));
    }
    const float ex = expf(x);
    return ex / (1.0f + ex);
}

#if defined(POLICY_GELU_TANH)
#define POLICY_GELU_NAME "gelu=tanh_approx"
/* torch nn.GELU(approximate='tanh'). Not the default torch GELU. */
static inline float policy_gelu(float x)
{
    const float k0 = 0.7978845608028654f;   /* sqrt(2/pi) */
    const float k1 = 0.044715f;
    return 0.5f * x * (1.0f + tanhf(k0 * (x + k1 * x * x * x)));
}
#elif defined(POLICY_GELU_POLY)
#define POLICY_GELU_NAME "gelu=erf_poly_AS7.1.26"
/* Abramowitz & Stegun 7.1.26 erf, |abs err| <= 1.5e-7. expf() only. */
static inline float policy_erff_poly(float x)
{
    const float p  = 0.3275911f;
    const float a1 = 0.254829592f,  a2 = -0.284496736f, a3 = 1.421413741f;
    const float a4 = -1.453152027f, a5 = 1.061405429f;
    const float ax = fabsf(x);
    const float t  = 1.0f / (1.0f + p * ax);
    const float poly = t * (a1 + t * (a2 + t * (a3 + t * (a4 + t * a5))));
    const float y  = 1.0f - poly * expf(-ax * ax);
    return (x >= 0.0f) ? y : -y;
}
static inline float policy_gelu(float x)
{
    const float inv_sqrt2 = 0.70710678118654752f;
    return 0.5f * x * (1.0f + policy_erff_poly(x * inv_sqrt2));
}
#else
#define POLICY_GELU_NAME "gelu=erf_exact"
/* torch nn.GELU() default (approximate='none'): 0.5x(1+erf(x/sqrt(2))). */
static inline float policy_gelu(float x)
{
    const float inv_sqrt2 = 0.70710678118654752f;
    return 0.5f * x * (1.0f + erff(x * inv_sqrt2));
}
#endif

/* ── primitives ───────────────────────────────────────────────────────────── */

/* y[out] = W[out][in] . x[in] + b[out]. W row-major, contiguous rows.
 * Accumulation order is strictly ascending in `in` — fixed and documented so
 * the result is reproducible across builds at a given optimisation level. */
static inline void policy_linear(const float* restrict w,
                                 const float* restrict b,
                                 const float* restrict x,
                                 float* restrict y,
                                 int n_out, int n_in)
{
    for (int o = 0; o < n_out; ++o) {
        const float* restrict row = w + (size_t)o * (size_t)n_in;
        policy_acc_t acc = (policy_acc_t)b[o];
        for (int i = 0; i < n_in; ++i) {
            acc += (policy_acc_t)row[i] * (policy_acc_t)x[i];
        }
        y[o] = (float)acc;
    }
}

/* y[out] += W[out][in] . x[in] (no bias; used for the W_hh contribution). */
static inline void policy_linear_acc(const float* restrict w,
                                     const float* restrict x,
                                     float* restrict y,
                                     int n_out, int n_in)
{
    for (int o = 0; o < n_out; ++o) {
        const float* restrict row = w + (size_t)o * (size_t)n_in;
        policy_acc_t acc = (policy_acc_t)y[o];
        for (int i = 0; i < n_in; ++i) {
            acc += (policy_acc_t)row[i] * (policy_acc_t)x[i];
        }
        y[o] = (float)acc;
    }
}

static inline int policy_all_finite(const float* v, int n)
{
    for (int i = 0; i < n; ++i) {
        if (!isfinite(v[i])) return 0;
    }
    return 1;
}

/* ── public API ───────────────────────────────────────────────────────────── */

void policy_reset(policy_state_t* st)
{
    if (st == (policy_state_t*)0) return;
    memset(st->h, 0, sizeof(st->h));
    memset(st->c, 0, sizeof(st->c));
    st->magic = POLICY_STATE_MAGIC;
    st->steps = 0u;
}

policy_status_t policy_infer(policy_state_t* st,
                             const float* obs,
                             float* logits_out,
                             float* value_out,
                             int* action_out)
{
    if (st == (policy_state_t*)0 || obs == (const float*)0) {
        return POLICY_ERR_NULL;
    }
    if (st->magic != POLICY_STATE_MAGIC) {
        return POLICY_ERR_UNINIT;
    }
    if (!policy_all_finite(obs, POLICY_OBS_DIM)) {
        return POLICY_ERR_OBS_NONFINITE;   /* h/c deliberately left untouched */
    }

    /* Bounded stack scratch — POLICY_STACK_SCRATCH_BYTES total. */
    float enc[POLICY_HIDDEN];
    float gates[POLICY_GATES];
    float h_new[POLICY_HIDDEN];
    float c_new[POLICY_HIDDEN];
    float logits[POLICY_ACTIONS];

    /* 1. encoder: Linear(38 -> 128) + GELU */
    policy_linear(POLICY_ENC_W, POLICY_ENC_B, obs, enc,
                  POLICY_HIDDEN, POLICY_OBS_DIM);
    for (int i = 0; i < POLICY_HIDDEN; ++i) {
        enc[i] = policy_gelu(enc[i]);
    }

    /* 2. LSTM cell. gates = W_ih.x + b_ih + W_hh.h + b_hh, chunked i,f,g,o. */
    policy_linear(POLICY_LSTM_W_IH, POLICY_LSTM_B_IH, enc, gates,
                  POLICY_GATES, POLICY_HIDDEN);
    for (int k = 0; k < POLICY_GATES; ++k) {
        gates[k] += POLICY_LSTM_B_HH[k];
    }
    policy_linear_acc(POLICY_LSTM_W_HH, st->h, gates,
                      POLICY_GATES, POLICY_HIDDEN);

    {
        const float* gi = gates + 0 * POLICY_HIDDEN;
        const float* gf = gates + 1 * POLICY_HIDDEN;
        const float* gg = gates + 2 * POLICY_HIDDEN;
        const float* go = gates + 3 * POLICY_HIDDEN;
        for (int j = 0; j < POLICY_HIDDEN; ++j) {
            const float i_t = policy_sigmoidf(gi[j]);
            const float f_t = policy_sigmoidf(gf[j]);
            const float g_t = tanhf(gg[j]);
            const float o_t = policy_sigmoidf(go[j]);
            const float c_t = f_t * st->c[j] + i_t * g_t;
            c_new[j] = c_t;
            h_new[j] = o_t * tanhf(c_t);
        }
    }

    /* 3. heads */
    policy_linear(POLICY_DEC_W, POLICY_DEC_B, h_new, logits,
                  POLICY_ACTIONS, POLICY_HIDDEN);
    float value;
    policy_linear(POLICY_VAL_W, POLICY_VAL_B, h_new, &value, 1, POLICY_HIDDEN);

    /* 4. health gate before committing recurrent state */
    if (!policy_all_finite(h_new, POLICY_HIDDEN) ||
        !policy_all_finite(c_new, POLICY_HIDDEN) ||
        !policy_all_finite(logits, POLICY_ACTIONS)) {
        return POLICY_ERR_DIVERGED;
    }

    /* 5. greedy action: argmax, ties resolved to the lowest index (matches
     *    torch.argmax on CPU for the 1-D contiguous case). */
    int best = 0;
    float best_v = logits[0];
    for (int a = 1; a < POLICY_ACTIONS; ++a) {
        if (logits[a] > best_v) { best_v = logits[a]; best = a; }
    }

    /* 6. commit */
    memcpy(st->h, h_new, sizeof(h_new));
    memcpy(st->c, c_new, sizeof(c_new));
    if (st->steps != 0xFFFFFFFFu) st->steps++;

    if (logits_out) memcpy(logits_out, logits, sizeof(logits));
    if (value_out)  *value_out = value;
    if (action_out) *action_out = best;
    return POLICY_OK;
}

int policy_act(policy_state_t* st, const float* obs)
{
    int action = 0;
    const policy_status_t rc = policy_infer(st, obs, (float*)0, (float*)0, &action);
    return (rc == POLICY_OK) ? action : (int)rc;
}

const char* policy_ckpt_sha256(void)        { return POLICY_WEIGHTS_CKPT_SHA; }
const char* policy_weights_blob_sha256(void){ return POLICY_WEIGHTS_BLOB_SHA; }
const char* policy_build_variant(void)      { return POLICY_GELU_NAME " " POLICY_ACC_NAME; }
