#!/bin/bash
# quality_test.sh — Automated quality comparison across configurations
#
# Tests GPT-2 server with different flags and prompts, outputs comparison table.
# Usage: ./scripts/quality_test.sh
#
# Prerequisites:
#   - prebuilt/gpt2_weights.bin (float weights, run tools/export_gpt2_weights.py)
#   - prebuilt/gpt2_binary.bin (binary weights, run scripts/export_binary_gpt2.py)
#   - build/filter.o (level-2 filter, run: lalc demos/basic/sampling_filter.lal _ --filter-only build/filter.c && gcc -c build/filter.c -o build/filter.o)

set -e
cd "$(dirname "$0")/.."
mkdir -p build

PROMPTS=(
  "The capital of France is"
  "Hello, how are"
  "Once upon a time"
  "Machine learning is"
)
N_TOKENS=20
PORT=9000

echo "=== LAL Quality Test Matrix ==="
echo "Prompts: ${#PROMPTS[@]}, N_TOKENS=$N_TOKENS"
echo ""

# Build servers if missing
if [ ! -f build/gpt2_server_nofilter ]; then
  echo "[*] building no-filter server..."
  gcc -O3 -mavx2 -mfma -I. -Wno-unused-function \
    tools/server/gpt2_server.c runtime/lal_runtime.c -lm \
    -o build/gpt2_server_nofilter
fi

if [ ! -f build/gpt2_server_filtered ] && [ -f build/filter.o ]; then
  echo "[*] building filtered server..."
  gcc -O3 -mavx2 -mfma -I. -Wno-unused-function \
    tools/server/gpt2_server.c runtime/lal_runtime.c build/filter.o -lm \
    -o build/gpt2_server_filtered
fi

# Configs to test
declare -a CONFIGS
declare -a CONFIG_NAMES

CONFIGS+=("./build/gpt2_server_nofilter --binary")
CONFIG_NAMES+=("BNN no-filter")

CONFIGS+=("./build/gpt2_server_filtered --binary")
CONFIG_NAMES+=("BNN +filter")

CONFIGS+=("./build/gpt2_server_nofilter --bwn")
CONFIG_NAMES+=("BWN no-filter")

CONFIGS+=("./build/gpt2_server_filtered --bwn")
CONFIG_NAMES+=("BWN +filter")

# Float baseline (if weights exist)
if [ -f prebuilt/gpt2_weights.bin ]; then
  CONFIGS+=("./build/gpt2_server_nofilter")
  CONFIG_NAMES+=("Float baseline")
fi

echo "Configurations: ${#CONFIGS[@]}"
echo ""

# Run tests
for ci in "${!CONFIGS[@]}"; do
  cfg="${CONFIGS[$ci]}"
  name="${CONFIG_NAMES[$ci]}"
  echo "=== $name ==="
  echo "  cmd: $cfg"

  # Start server
  PORT=$((9000 + ci))
  $cfg --port $PORT &
  SERVER_PID=$!
  sleep 3

  # Test each prompt
  for prompt in "${PROMPTS[@]}"; do
    resp=$(curl -s -X POST http://localhost:$PORT/generate \
      -H "Content-Type: application/json" \
      -d "{\"prompt\":\"$prompt\",\"n_tokens\":$N_TOKENS}" 2>/dev/null || echo '{"text":"ERROR","tokens_per_sec":"0"}')

    text=$(echo "$resp" | python3 -c "import sys,json; print(json.load(sys.stdin).get('text','?'))" 2>/dev/null || echo "?")
    tps=$(echo "$resp" | python3 -c "import sys,json; print(json.load(sys.stdin).get('tokens_per_sec','?'))" 2>/dev/null || echo "?")

    echo "  prompt: $prompt"
    echo "    output: $text"
    echo "    speed:  $tps tok/s"
  done

  kill $SERVER_PID 2>/dev/null
  wait $SERVER_PID 2>/dev/null
  echo ""
done

echo "=== Done ==="
