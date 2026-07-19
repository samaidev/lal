#!/usr/bin/env bash
# Build a LAL per-layer hook (.lal with runtime_context(layer=...)) into a
# runtime-loadable .so for qwen_server.
#
# The .so is dlopen'd by qwen_server via --lal-steer, so activation-steering /
# per-layer-read rules can be changed WITHOUT rebuilding the server.
#
# Usage:
#   scripts/build_lal_steer.sh [INPUT.lal] [OUTPUT.so]
# Defaults: demos/qwen_steer.lal -> prebuilt/qwen_steer.so
set -e

LAL="${1:-demos/qwen_steer.lal}"
OUT="${2:-prebuilt/qwen_steer.so}"

mkdir -p build prebuilt

echo "[*] compiling $LAL -> build/qwen_steer.c"
python3 compiler/lal.py "$LAL" _ build/qwen_steer.c --filter-only

echo "[*] building $OUT (-fPIC -shared)"
gcc -O3 -fPIC -shared -I. -o "$OUT" build/qwen_steer.c

echo "[*] done: $OUT"
echo "    load with: ./prebuilt/qwen_server ... --lal-steer $OUT"
