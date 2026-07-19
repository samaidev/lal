#!/usr/bin/env bash
# Build the logic-driven layer-skip .so from demos/mini_skip.lal.
# Emits ONLY the strong `lal_layer_skip` symbol (no main), mirroring
# scripts/build_lal_filter.sh's --filter-only approach.
set -e
cd "$(dirname "$0")/.."
python3 compiler/lal.py demos/mini_skip.lal _ build/mini_skip.c --skip-only
gcc -O3 -fPIC -shared -I. -o prebuilt/mini_skip.so build/mini_skip.c -lm
echo "[*] built prebuilt/mini_skip.so (logic-driven layer skip)"
