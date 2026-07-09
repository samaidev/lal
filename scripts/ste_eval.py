#!/usr/bin/env python3
"""ste_eval.py — test BWN with STE-tuned weights vs original binary weights."""
import json, subprocess, time, urllib.request, os, signal
REPO="/workspace/lal"
PROMPTS=["The capital of France is","The capital of Japan is","Once upon a time",
         "Machine learning is","Hello, how are","The weather today is",
         "The capital of Germany is","Water boils at"]
SAMPS=[("greedy(t=0)",{"temperature":0}),("default(t=0.8,k=40)",{})]

def start(env, flags, port):
    log=open(f"{REPO}/build/ste_{port}.log","w")
    e=os.environ.copy(); e.update(env)
    p=subprocess.Popen([f"{REPO}/build/gpt2_server","--port",str(port)]+flags,
        cwd=REPO, env=e, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    for _ in range(80):
        time.sleep(0.5)
        try: urllib.request.urlopen(f"http://localhost:{port}/",timeout=2); return p
        except:
            if p.poll() is not None: return None
    return None

def gen(port, prompt, sampling, n=25):
    body=json.dumps({"prompt":prompt,"n_tokens":n, **sampling}).encode()
    req=urllib.request.Request(f"http://localhost:{port}/generate",data=body,headers={"Content-Type":"application/json"},method="POST")
    with urllib.request.urlopen(req,timeout=60) as r: return json.loads(r.read())

# (label, env, flags)
CASES=[
    ("BWN original binary",      {},                                       ["--binary","--bwn"]),
    ("BWN + STE-tuned weights",  {"LAL_BINARY":"build/gpt2_binary_ste.bin"}, ["--binary","--bwn"]),
    ("BWN + STE + logic (if file exists)", {}, ["--binary","--bwn"]),
]
port=8600
for name, env, flags in CASES:
    # skip logic case if no ste_logic file
    if "logic" in name and not os.path.exists(f"{REPO}/build/gpt2_binary_ste_logic.bin"):
        continue
    if "STE + logic" in name:
        env["LAL_BINARY"]="build/gpt2_binary_ste_logic.bin"
    print(f"\n{'#'*70}\n# {name}\n  env={env} flags={flags}\n{'#'*70}")
    p=start(env, flags, port)
    if not p: print("  SERVER FAILED"); 
    if p:
        try:
            for slabel, sampling in SAMPS:
                print(f"\n  --- {slabel} ---")
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
