#!/usr/bin/env bash
# Build a LAL `filter` rule into a runtime-loadable .so for qwen_server.
#
# The .so is dlopen'd by qwen_server via --lal-filter, so sampling rules can
# be changed WITHOUT rebuilding the server.
#
# Usage:
#   scripts/build_lal_filter.sh [INPUT.lal] [OUTPUT.so]
# Defaults: demos/qwen_filter.lal -> prebuilt/qwen_filter.so
set -e

LAL="${1:-demos/qwen_filter.lal}"
OUT="${2:-prebuilt/qwen_filter.so}"

mkdir -p build prebuilt

echo "[*] compiling $LAL -> build/qwen_filter.c"
python3 compiler/lal.py "$LAL" _ build/qwen_filter.c --filter-only

echo "[*] building $OUT (-fPIC -shared)"
gcc -O3 -fPIC -shared -I. -o "$OUT" build/qwen_filter.c

echo "[*] done: $OUT"
echo "    load with: ./prebuilt/qwen_server ... --lal-filter $OUT"
