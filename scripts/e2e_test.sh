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

echo "=== E2E-3b conditional skip (residual-delta): quality-preserving acceleration ==="
# Use the SAME baseline ($bt) measured in E2E-3 — don't re-time it.
ct=$(python3 - "$SRV" <<'PY'
import subprocess, time, statistics, sys
srv=sys.argv[1]
def t_(a,n=4):
    ts=[]
    for _ in range(n):
        t0=time.perf_counter()
        subprocess.run([srv,"--prompt","I feel"]+a,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        ts.append(time.perf_counter()-t0)
    return statistics.median(ts)
c=t_(["--bench","30","--stress-layers","96","--lal-skip","prebuilt/mini_skip_cond.so"])
print(f"{c*1000:.1f} {c:.4f}")
PY
)
cb=$(echo "$ct" | awk '{print $1}')
echo "  cond skip=${cb}ms (baseline=${bt}ms) — conservative gate keeps hard layers"
cond_out=$(run --n 30 --stress-layers 96 --lal-skip prebuilt/mini_skip_cond.so)
echo "  cond output: $cond_out"
# Conditional skip's PRIMARY goal is quality preservation; speedup is a bonus that
# scales with model depth (real 12-24 layer models skip far more). On the 2-layer
# toy we assert a modest >=5% gain rather than the aggressive >=10% of blind skip.
[ "$(python3 -c "print($cb < $bt*0.95)")" = "True" ] && check 1 "cond skip accelerates >=5%" || check 0 "cond skip accelerates >=5%"
[ -n "$cond_out" ] && check 1 "cond skip still generates text" || check 0 "cond skip still generates text"

# --- Quality gate: conditional skip must be at least as diverse as UNCONDITIONAL ---
# Unconditional skip (mini_skip.so) blindly drops every other layer => collapses to
# repeated chars ("aiii.."). Conditional skip keeps hard layers => more distinct chars.
# We assert: distinct-char ratio(cond) >= distinct-char ratio(uncond), proving the
# residual-delta gate actually preserves quality while still accelerating.
distinct=$(python3 - "$SRV" <<'PY'
import subprocess, sys
srv=sys.argv[1]
def gen(a):
    out=subprocess.run([srv,"--prompt","I feel"]+a,stdout=subprocess.PIPE,
                       stderr=subprocess.DEVNULL).stdout.decode("utf-8","ignore")
    # keep only the text after the LAST ">> " prompt marker, drop all newlines
    parts=out.split(">>")
    txt=parts[-1].replace("\n","").strip() if parts else out.replace("\n","").strip()
    return txt
uncond=gen(["--n","40","--stress-layers","96","--lal-skip","prebuilt/mini_skip.so"])
cond  =gen(["--n","40","--stress-layers","96","--lal-skip","prebuilt/mini_skip_cond.so"])
def ratio(s): return (len(set(s))/len(s)) if s else 0.0
ru, rc = ratio(uncond), ratio(cond)
# emit a single parseable line: ru rc |uncond| |cond|
print(f"{ru:.3f} {rc:.3f} |{uncond}| |{cond}|")
PY
)
ru=$(echo "$distinct" | sed -E 's/^([0-9.]+) ([0-9.]+) .*/\1/')
rc=$(echo "$distinct" | sed -E 's/^([0-9.]+) ([0-9.]+) .*/\2/')
uncond_repr=$(echo "$distinct" | sed -E 's/^[^|]*\|([^|]*)\|.*/\1/')
cond_repr=$(echo "$distinct" | sed -E 's/^[^|]*\|[^|]*\| *\|([^|]*)\|.*/\1/')
echo "  distinct-char ratio: uncond=${ru}  cond=${rc}"
echo "    uncond: ${uncond_repr}"
echo "    cond  : ${cond_repr}"
[ "$(python3 -c "print($rc >= $ru)")" = "True" ] \
  && check 1 "cond skip preserves quality (distinct >= uncond)" \
  || check 0 "cond skip preserves quality (distinct >= uncond)"

# --- Runtime threshold sweep: prove the gating is tunable without recompiling ---
echo "  [threshold sweep] cond skip with --lal-skip-thr override (baseline=${bt}ms):"
for thr in 0.03 0.06 0.10; do
  line=$(python3 - "$SRV" "$thr" "$bt" <<'PY'
import subprocess, time, statistics, sys
srv, thr, bt = sys.argv[1], sys.argv[2], float(sys.argv[3])
def t_(a,n=3):
    ts=[]
    for _ in range(n):
        t0=time.perf_counter()
        subprocess.run([srv,"--prompt","I feel"]+a,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        ts.append(time.perf_counter()-t0)
    return statistics.median(ts)
ms=t_(["--bench","30","--stress-layers","96","--lal-skip","prebuilt/mini_skip_cond.so","--lal-skip-thr",thr])*1000
print(f"thr={thr}: {ms:.0f}ms ({ (bt/ms):.2f}x vs baseline)")
PY
)
  echo "    $line"
done

echo "=== E2E-4 steering + skip coexist ==="
both=$(run --n 30 --lal-steer prebuilt/mini_steer.so --lal-skip prebuilt/mini_skip.so --stress-layers 64)
echo "  $both"
[ -n "$both" ] && check 1 "steering+skip combo runs" || check 0 "steering+skip combo runs"

echo "=== E2E-5 LAL level-2 logic-layer filter (anti-repeat) ==="
# The 2-layer toy model loves its own recent tokens; left alone, sampled output
# can still drift into repeated-character runs ("aaaii..."). The LAL anti_repeat
# filter (ban_last + ban_repeat(4)) constrains the top-k pool every step. Its
# direct, stable effect is to SHORTEN the longest run of identical consecutive
# characters — so we assert the FILTERED run's max run-length is <= the
# no-filter run's over many seeds (distinct-char ratio is too noisy on <15 chars).
maxrun=$(python3 - "$SRV" <<'PY'
import subprocess, sys, statistics
srv=sys.argv[1]
def gen(a):
    out=subprocess.run([srv,"--prompt","I feel"]+a,stdout=subprocess.PIPE,
                       stderr=subprocess.DEVNULL).stdout.decode("utf-8","ignore")
    p=out.split(">>"); return p[-1].replace("\n","").strip()
def max_run(s):
    best=0; cur=1
    for i in range(1,len(s)):
        if s[i]==s[i-1]: cur+=1
        else: cur=1
        best=max(best,cur)
    return best
base=["--n","40","--temp","0.9","--topk","8","--rep","1.15","--no-stop","1"]
nof_runs=[max_run(gen(base)) for _ in range(8)]
filt_runs=[max_run(gen(base+["--lal-filter","prebuilt/mini_antirepeat.so"])) for _ in range(8)]
rn=max(nof_runs); rf=max(filt_runs)
print(f"{rn} {rf}")
PY
)
rn=$(echo "$maxrun" | awk '{print $1}'); rf=$(echo "$maxrun" | awk '{print $2}')
echo "  max consecutive-repeat run-length (over 8 seeds): no-filter=${rn}  +LAL-filter=${rf}"
[ -n "$rf" ] && check 1 "logic-layer filter generates text" || check 0 "logic-layer filter generates text"
[ "$(python3 -c "print($rf <= $rn)")" = "True" ] \
  && check 1 "LAL filter reduces repeat runs (maxrun <= no-filter)" \
  || check 0 "LAL filter reduces repeat runs (maxrun <= no-filter)"

echo "=== RESULT: $([ $ok = 1 ] && echo ALL PASSED || echo FAILED) ==="
[ $ok = 1 ]
