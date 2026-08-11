# Makefile — embedded orbital policy (C port of the T3 canonical checkpoint)
#
#   make all        policy.o, libpolicy shim, bench_policy, closed_loop
#   make shim       shared library for test_parity.py / run_closed_loop.py (ctypes)
#   make bench      single-core inference latency benchmark
#   make loop       fully-C env + policy closed-loop evaluator
#   make variants   the three GELU builds side by side (shim only)
#   make clean
#
# The env include path points at the in-repo PufferLib Ocean environment; the
# closed-loop binary compiles orbital.h (header-only, static inline) directly
# into the same translation unit as the policy — no Python in the loop.

CC       ?= cc
ENV_DIR  ?= ../../../pufferlib/pufferlib/ocean/orbital

CFLAGS_BASE  := -std=c99 -O2 -Wall -Wextra -Wshadow -Wconversion -Wsign-conversion \
                -Wcast-align -Wpointer-arith -Wstrict-prototypes -Wmissing-prototypes \
                -Wno-unused-parameter -fno-common
LDLIBS       := -lm

# The environment header is third-party-ish w.r.t. this deliverable: compile it
# without the strict pedantic set so our own warnings stay signal.
CFLAGS_ENV   := -std=c99 -O2 -Wall -Wno-unused-function -Wno-unused-variable

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
  SHLIB_EXT := dylib
  SHLIB_FLAGS := -dynamiclib
else
  SHLIB_EXT := so
  SHLIB_FLAGS := -shared -fPIC
endif

SHIM := libpolicy.$(SHLIB_EXT)

.PHONY: all shim bench loop variants poly sanitize clean

all: $(SHIM) bench_policy closed_loop

policy.o: policy.c policy.h policy_weights.h
	$(CC) $(CFLAGS_BASE) -c policy.c -o policy.o

$(SHIM): policy.c policy.h policy_weights.h
	$(CC) $(CFLAGS_BASE) $(SHLIB_FLAGS) -fPIC policy.c -o $(SHIM) $(LDLIBS)

shim: $(SHIM)

bench_policy: bench_policy.c policy.o policy.h
	$(CC) $(CFLAGS_BASE) bench_policy.c policy.o -o bench_policy $(LDLIBS)

bench: bench_policy

closed_loop: closed_loop.c policy.o policy.h $(ENV_DIR)/orbital.h
	$(CC) $(CFLAGS_ENV) -I$(ENV_DIR) closed_loop.c policy.o -o closed_loop $(LDLIBS)

loop: closed_loop

# GELU/accumulation variants — used by test_parity.py --variants
variants: libpolicy_erf.$(SHLIB_EXT) libpolicy_poly.$(SHLIB_EXT) \
          libpolicy_tanh.$(SHLIB_EXT) libpolicy_accdouble.$(SHLIB_EXT)

libpolicy_erf.$(SHLIB_EXT): policy.c policy.h policy_weights.h
	$(CC) $(CFLAGS_BASE) $(SHLIB_FLAGS) -fPIC -DPOLICY_GELU_ERF policy.c -o $@ $(LDLIBS)

libpolicy_poly.$(SHLIB_EXT): policy.c policy.h policy_weights.h
	$(CC) $(CFLAGS_BASE) $(SHLIB_FLAGS) -fPIC -DPOLICY_GELU_POLY policy.c -o $@ $(LDLIBS)

libpolicy_tanh.$(SHLIB_EXT): policy.c policy.h policy_weights.h
	$(CC) $(CFLAGS_BASE) $(SHLIB_FLAGS) -fPIC -DPOLICY_GELU_TANH policy.c -o $@ $(LDLIBS)

libpolicy_accdouble.$(SHLIB_EXT): policy.c policy.h policy_weights.h
	$(CC) $(CFLAGS_BASE) $(SHLIB_FLAGS) -fPIC -DPOLICY_ACC_DOUBLE policy.c -o $@ $(LDLIBS)

# Dependency-minimal build (expf/tanhf only — no erff): same binaries, POLY GELU
policy_poly.o: policy.c policy.h policy_weights.h
	$(CC) $(CFLAGS_BASE) -DPOLICY_GELU_POLY -c policy.c -o policy_poly.o

bench_policy_poly: bench_policy.c policy_poly.o policy.h
	$(CC) $(CFLAGS_BASE) bench_policy.c policy_poly.o -o bench_policy_poly $(LDLIBS)

closed_loop_poly: closed_loop.c policy_poly.o policy.h $(ENV_DIR)/orbital.h
	$(CC) $(CFLAGS_ENV) -I$(ENV_DIR) closed_loop.c policy_poly.o -o closed_loop_poly $(LDLIBS)

poly: bench_policy_poly closed_loop_poly

# AddressSanitizer + UndefinedBehaviorSanitizer build of the full C loop.
# `make sanitize && ASAN_OPTIONS=detect_leaks=0 ./closed_loop_asan --episodes 25`
sanitize: closed_loop_asan
closed_loop_asan: closed_loop.c policy.c policy.h policy_weights.h $(ENV_DIR)/orbital.h
	$(CC) -std=c99 -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer \
	      -I$(ENV_DIR) closed_loop.c policy.c -o closed_loop_asan $(LDLIBS)

clean:
	rm -f policy.o policy_poly.o bench_policy bench_policy_poly \
	      closed_loop closed_loop_poly closed_loop_asan \
	      libpolicy*.dylib libpolicy*.so
	rm -rf closed_loop_asan.dSYM
