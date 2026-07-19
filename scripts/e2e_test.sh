#!/usr/bin/env bash
# End-to-end test for LAL inference on the local mini transformer.
# Covers: baseline generation, steering (pos/neg), layer-skip acceleration,
# and steering+skip coexistence. Exits non-zero on any failure.
#
# Prereqs (built by `make mini-server mini-train mini-steer mini-skip`):
#   prebuilt/mini_server  prebuilt/mini_model.bin
#   prebuilt/mini_steer.so  prebuilt/mini_steer_neg.so  prebuilt/mini_skip.so
set -euo pipefail
cd "$(dirname "$0")/.."

SRV=prebuilt/mini_server
[ -x "$SRV" ] || { echo "[e2e] build first: make prebuilt/mini_server"; exit 1; }

run() { "$SRV" --prompt "I feel" "$@" 2>/dev/null; }

ok=1
check() { if [ "$1" = "1" ]; then echo "PASS: $2"; else echo "FAIL: $2"; ok=0; fi; }

echo "=== E2E-1 baseline ==="
base=$(run --n 30)
echo "  $base"
[ -n "$base" ] && check 1 "baseline generates text" || check 0 "baseline generates text"

echo "=== E2E-2 steering ==="
pos=$(run --n 30 --lal-steer prebuilt/mini_steer.so)
neg=$(run --n 30 --lal-steer prebuilt/mini_steer_neg.so)
echo "  base: $base"; echo "  pos : $pos"; echo "  neg : $neg"
[ "$pos" != "$base" ] && [ "$neg" != "$base" ] \
  && check 1 "steering changes output both ways" \
  || check 0 "steering changes output both ways"

echo "=== E2E-3 layer-skip acceleration ==="
# Fixed forward count (--bench) => deterministic timing.
b_t=$("$SRV" --prompt "I feel" --bench 30 --stress-layers 96 2>/dev/null >/dev/null; echo $?)
# use python for timing precision
t=$(python3 - "$SRV" <<'PY'
import subprocess, time, statistics, sys
srv=sys.argv[1]
def t_(a,n=4):
    ts=[]
    for _ in range(n):
        t0=time.perf_counter()
        subprocess.run([srv,"--prompt","I feel"]+a,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        ts.append(time.perf_counter()-t0)
    return statistics.median(ts)
b=t_(["--bench","30","--stress-layers","96"])
s=t_(["--bench","30","--stress-layers","96","--lal-skip","prebuilt/mini_skip.so"])
print(f"{b*1000:.1f} {s*1000:.1f} {b/s:.2f}")
PY
)
read bt st sp <<<"$t"
echo "  baseline=${bt}ms  skip=${st}ms  speedup=${sp}x"
skip_out=$(run --n 30 --stress-layers 96 --lal-skip prebuilt/mini_skip.so)
[ "$(python3 -c "print($st < $bt*0.9)")" = "True" ] && check 1 "skip accelerates >=10%" || check 0 "skip accelerates >=10%"
[ -n "$skip_out" ] && check 1 "skip still generates text" || check 0 "skip still generates text"

echo "=== E2E-4 steering + skip coexist ==="
both=$(run --n 30 --lal-steer prebuilt/mini_steer.so --lal-skip prebuilt/mini_skip.so --stress-layers 64)
echo "  $both"
[ -n "$both" ] && check 1 "steering+skip combo runs" || check 0 "steering+skip combo runs"

echo "=== RESULT: $([ $ok = 1 ] && echo ALL PASSED || echo FAILED) ==="
[ $ok = 1 ]
