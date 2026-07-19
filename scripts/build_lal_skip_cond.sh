#!/usr/bin/env bash
# Build the CONDITIONAL logic-driven layer-skip .so from demos/mini_skip_cond.lal.
# Emits ONLY the strong `lal_layer_skip` symbol (no main), mirroring
# scripts/build_lal_filter.sh's --filter-only approach. The emitted symbol only
# returns a skip when the hidden-state confidence exceeds the .lal `when` threshold.
set -e
cd "$(dirname "$0")/.."
python3 compiler/lal.py demos/mini_skip_cond.lal _ build/mini_skip_cond.c --skip-only
gcc -O3 -fPIC -shared -I. -o prebuilt/mini_skip_cond.so build/mini_skip_cond.c -lm
echo "[*] built prebuilt/mini_skip_cond.so (conditional layer skip)"
