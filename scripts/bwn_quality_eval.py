#!/usr/bin/env python3
"""bwn_quality_eval.py — Empirical BWN output-quality test for LAL GPT-2 server.

Starts the server under each requested flag config, sends a fixed set of
prompts, and reports the generated text + tok/s for side-by-side comparison.

Configs (matched against the actual server code paths):
  float            : plain float weights (baseline)
  binary           : --binary          (BNN: XNOR+popcount, full binarization)
  bwn              : --bwn             (sets g_use_bwn; float load path)
  bwn+binary       : --binary --bwn    (loads GB2L + BWN float-x@sign(w) AVX2)
"""
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SERVER = os.path.join(REPO, "build", "gpt2_server")

PROMPTS = [
    "The capital of France is",
    "Once upon a time",
    "Machine learning is",
    "Hello, how are",
    "The meaning of life is",
]
N_TOKENS = 25


def start_server(flags, port):
    log = open(os.path.join(REPO, "build", f"server_{port}.log"), "w")
    proc = subprocess.Popen(
        [SERVER, "--port", str(port)] + flags,
        cwd=REPO, stdout=log, stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    # Wait for readiness (up to 30s — binary load ~1s, float load ~3-5s)
    for _ in range(60):
        time.sleep(0.5)
        try:
            urllib.request.urlopen(f"http://localhost:{port}/", timeout=2)
            return proc
        except Exception:
            if proc.poll() is not None:
                log.flush()
                print(f"  [!] server exited early, code={proc.returncode}")
                return None
    print("  [!] server did not become ready in 30s")
    return None


def gen_once(port, prompt, n_tokens):
    body = json.dumps({"prompt": prompt, "n_tokens": n_tokens}).encode()
    req = urllib.request.Request(
        f"http://localhost:{port}/generate",
        data=body, headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=60) as r:
        data = json.loads(r.read())
    dt = time.time() - t0
    return data, dt


def run_config(name, flags, port):
    print(f"\n{'='*70}\n=== {name}  flags={flags}\n{'='*70}")
    proc = start_server(flags, port)
    if proc is None:
        return None
    try:
        results = []
        for p in PROMPTS:
            try:
                data, dt = gen_once(port, p, N_TOKENS)
                text = data.get("text", "?")
                tps = data.get("tokens_per_sec", "?")
                results.append((p, text, tps, dt))
            except Exception as e:
                results.append((p, f"<ERROR: {e}>", "?", 0))
        for p, text, tps, dt in results:
            print(f"\n[P] {p}")
            print(f"    -> {text!r}")
            print(f"    tps(server)={tps}  wall={dt:.2f}s/{N_TOKENS}tok")
        return results
    finally:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except Exception:
            proc.kill()
        proc.wait(timeout=10)


def main():
    configs = [
        ("FLOAT (baseline)",     []),
        ("BNN  (--binary)",       ["--binary"]),
        ("BWN  (--bwn alone)",    ["--bwn"]),
        ("BWN  (--binary --bwn)", ["--binary", "--bwn"]),
    ]
    if not os.path.exists(SERVER):
        sys.exit(f"server binary not found: {SERVER}")
    if not os.path.exists(os.path.join(REPO, "prebuilt", "gpt2_weights.bin")):
        sys.exit("prebuilt/gpt2_weights.bin missing")
    if not os.path.exists(os.path.join(REPO, "prebuilt", "gpt2_binary.bin")):
        sys.exit("prebuilt/gpt2_binary.bin missing")

    all_results = {}
    port = 8100
    for name, flags in configs:
        all_results[name] = run_config(name, flags, port)
        port += 1
        time.sleep(2)  # let port free up

    # Compact summary table
    print(f"\n\n{'#'*70}\n# SUMMARY\n{'#'*70}")
    print(f"{'Config':<28} {'tok/s':>8}  sample output")
    for name, res in all_results.items():
        if not res:
            print(f"{name:<28} {'FAIL':>8}")
            continue
        # use the 'Once upon a time' prompt output as the sample
        sample = res[1][1]  # PROMPTS[1]
        # avg tps across prompts (server-reported)
        tps_vals = [r[2] for r in res if r[2] != "?"]
        avg_tps = f"{sum(float(t) for t in tps_vals)/len(tps_vals):.1f}" if tps_vals else "?"
        snippet = sample.replace("\n", " ")[:60]
        print(f"{name:<28} {avg_tps:>8}  {snippet!r}")


if __name__ == "__main__":
    main()
