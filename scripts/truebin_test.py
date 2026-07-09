#!/usr/bin/env python3
"""truebin_test.py — test TRUE binary-weight modes (all require --binary)."""
import json, subprocess, time, urllib.request, os, signal, glob
REPO="/workspace/lal"; SERVER=f"{REPO}/build/gpt2_server"
PROMPTS=["The capital of France is","Once upon a time","Machine learning is","Hello, how are"]

def start(flags, port):
    log=open(f"{REPO}/build/tb_{port}.log","w")
    p=subprocess.Popen([SERVER,"--port",str(port)]+flags,cwd=REPO,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    for _ in range(60):
        time.sleep(0.5)
        try: urllib.request.urlopen(f"http://localhost:{port}/",timeout=2); return p
        except:
            if p.poll() is not None: return None
    return None

def gen(port, prompt, n=25):
    body=json.dumps({"prompt":prompt,"n_tokens":n}).encode()
    req=urllib.request.Request(f"http://localhost:{port}/generate",data=body,headers={"Content-Type":"application/json"},method="POST")
    with urllib.request.urlopen(req,timeout=60) as r: return json.loads(r.read())

configs=[
    ("FLOAT (baseline)",            []),
    ("BWN  (--binary --bwn)",         ["--binary","--bwn"]),
    ("INT8 (--binary --int8)",       ["--binary","--int8"]),
    ("MIXED-8 (--binary --mixed-int8 8) README-rec", ["--binary","--mixed-int8","8"]),
]
port=8400
for name, flags in configs:
    print(f"\n{'='*68}\n=== {name}\n{'='*68}")
    p=start(flags, port)
    if not p: print("  SERVER FAILED")
    else:
        try:
            for pr in PROMPTS:
                try:
                    d=gen(port, pr)
                    print(f"[P] {pr}\n    -> {d.get('text','?')!r}\n    tps={d.get('tokens_per_sec','?')}")
                except Exception as e: print(f"[P] {pr} ERROR: {e}")
        finally:
            try: os.killpg(os.getpgid(p.pid), signal.SIGTERM)
            except: p.kill()
            p.wait(timeout=10)
    port += 1; time.sleep(2)

print("\n--- mode logs (which weights loaded) ---")
for f in sorted(glob.glob(f"{REPO}/build/tb_*.log")):
    print(f"\n[{os.path.basename(f)}]")
    for line in open(f):
        if any(k in line for k in ["mode","loading","format","mixed int8"]):
            print("  "+line.rstrip())
