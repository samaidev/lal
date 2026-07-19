#!/usr/bin/env bash
# Build the LAL logic-layer sampling filter .so from demos/mini_antirepeat.lal.
# Emits ONLY the strong `lal_filter_topk` symbol (no main), mirroring
# scripts/build_lal_skip.sh's --skip-only approach.
set -e
cd "$(dirname "$0")/.."
mkdir -p build
python3 compiler/lal.py demos/mini_antirepeat.lal _ build/mini_antirepeat.c --filter-only
gcc -O3 -fPIC -shared -I. -o prebuilt/mini_antirepeat.so build/mini_antirepeat.c -lm
echo "[*] built prebuilt/mini_antirepeat.so (LAL logic-layer sampling filter)"
