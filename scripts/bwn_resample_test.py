#!/usr/bin/env python3
"""bwn_resample_test.py — re-test BWN with varied sampling configs.

Tests whether BWN garbled output is a SAMPLING issue (noisy logits + high
temperature) rather than a model-quality issue. If greedy/low-temp BWN output
is coherent, the README's "Good" claim holds under proper sampling.
"""
import json, subprocess, time, urllib.request, os, signal
REPO="/workspace/lal"
PROMPTS=["The capital of France is","Once upon a time","Machine learning is","Hello, how are"]

def start(server, flags, port):
    log=open(f"{REPO}/build/r_{port}.log","w")
    p=subprocess.Popen([server,"--port",str(port)]+flags,cwd=REPO,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    for _ in range(60):
        time.sleep(0.5)
        try: urllib.request.urlopen(f"http://localhost:{port}/",timeout=2); return p
        except:
            if p.poll() is not None: return None
    return None

def gen(port, prompt, sampling, n=25):
    body=json.dumps({"prompt":prompt,"n_tokens":n, **sampling}).encode()
    req=urllib.request.Request(f"http://localhost:{port}/generate",data=body,headers={"Content-Type":"application/json"},method="POST")
    with urllib.request.urlopen(req,timeout=60) as r: return json.loads(r.read())

# sampling configs to try on BWN
SAMPLINGS=[
    ("default (t=0.8,k=40)",  {}),
    ("greedy (t=0)",          {"temperature":0}),
    ("low-temp (t=0.3,k=10)", {"temperature":0.3,"top_k":10}),
    ("top-1 (k=1)",           {"top_k":1}),
]

# (label, server binary, server flags)
SERVERS=[
    ("FLOAT+BLAS",  f"{REPO}/build/gpt2_server_blas", []),
    ("BWN",         f"{REPO}/build/gpt2_server",       ["--binary","--bwn"]),
]

port=8500
for sname, srv, sflags in SERVERS:
    print(f"\n{'#'*70}\n# SERVER: {sname}  flags={sflags}\n{'#'*70}")
    p=start(srv, sflags, port)
    if not p: print("  SERVER FAILED"); continue
    try:
        for slabel, sampling in SAMPLINGS:
            print(f"\n  --- sampling: {slabel} ---")
            for pr in PROMPTS:
                try:
                    d=gen(port, pr, sampling)
                    t=d.get('text','?').replace('\n',' ')
                    print(f"    [P] {pr}\n        -> {t!r}  (tps={d.get('tokens_per_sec','?')})")
                except Exception as e: print(f"    [P] {pr} ERROR: {e}")
    finally:
        try: os.killpg(os.getpgid(p.pid), signal.SIGTERM)
        except: p.kill()
        p.wait(timeout=10)
    port += 1; time.sleep(2)
