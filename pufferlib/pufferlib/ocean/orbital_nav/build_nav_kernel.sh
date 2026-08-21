#!/bin/bash
# Build the J2 STM C kernel. Run from anywhere; writes nav_j2_kernel.so next
# to the source. The .so is a build artifact and is gitignored on purpose —
# it is platform-specific, and a stale checked-in binary silently running
# instead of the source you are reading is exactly the failure this project
# does not need.
#
# -ffp-contract=off is LOAD-BEARING, not boilerplate: with contraction on, the
# compiler fuses a*b+c into an FMA, which rounds ONCE instead of twice. That
# changes results in the last bits, and the differential fuzz gate compares
# against numpy — which does not contract. Leaving it on would fail the gate
# for a reason that has nothing to do with the port being wrong.
set -euo pipefail
D="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
gcc -O3 -ffp-contract=off -fno-fast-math -shared -fPIC \
    -o "$D/nav_j2_kernel.so" "$D/nav_j2_kernel.c" -lm
echo "built $D/nav_j2_kernel.so"
