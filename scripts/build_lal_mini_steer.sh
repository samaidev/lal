#!/usr/bin/env bash
# Build the REAL mini-transformer steering hook (demos/mini_steer.lal, which
# carries a direction vector computed from the trained mini model) into a
# runtime-loadable .so for mini_server / qwen_server.
#
# Usage: scripts/build_lal_mini_steer.sh [INPUT.lal] [OUTPUT.so]
# Defaults: demos/mini_steer.lal -> prebuilt/mini_steer.so
set -e
LAL="${1:-demos/mini_steer.lal}"
OUT="${2:-prebuilt/mini_steer.so}"
mkdir -p build prebuilt
echo "[*] compiling $LAL -> build/mini_steer.c"
python3 compiler/lal.py "$LAL" _ build/mini_steer.c --filter-only
echo "[*] building $OUT (-fPIC -shared)"
gcc -O3 -fPIC -shared -I. -o "$OUT" build/mini_steer.c
echo "[*] done: $OUT"
echo "    load with: ./prebuilt/mini_server --prompt 'I feel' --lal-steer $OUT"
