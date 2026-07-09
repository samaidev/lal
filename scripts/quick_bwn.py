#!/usr/bin/env python3
import json, subprocess, time, urllib.request, os, signal
REPO="/workspace/lal"; SERVER=f"{REPO}/prebuilt/gpt2_server"
PROMPTS=["The capital of France is","Once upon a time","Machine learning is","Hello, how are"]
def start(env, port):
    log=open(f"{REPO}/build/q_{port}.log","w")
    e=os.environ.copy(); e.update(env)
    p=subprocess.Popen([SERVER,"--binary","--bwn","--port",str(port)],
        cwd=REPO, env=e, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    for _ in range(80):
        time.sleep(0.5)
        try: urllib.request.urlopen(f"http://localhost:{port}/",timeout=2); return p
        except:
            if p.poll() is not None: return None
    return None
def gen(port, prompt, n=25):
    body=json.dumps({"prompt":prompt,"n_tokens":n,"temperature":0}).encode()
    req=urllib.request.Request(f"http://localhost:{port}/generate",data=body,headers={"Content-Type":"application/json"},method="POST")
    with urllib.request.urlopen(req,timeout=60) as r: return json.loads(r.read())
import sys
label=sys.argv[1] if len(sys.argv)>1 else "test"
binpath=sys.argv[2] if len(sys.argv)>2 else "prebuilt/gpt2_binary.bin"
port=8700
print(f"\n=== {label} (binary={binpath}) greedy ===")
p=start({"LAL_BINARY":binpath}, port)
if not p: print("FAILED")
else:
    try:
        for pr in PROMPTS:
            try:
                d=gen(port,pr); print(f"  [P] {pr}\n      -> {d.get('text','?')!r}  (tps={d.get('tokens_per_sec','?')})")
            except Exception as e: print(f"  [P] {pr} ERR {e}")
    finally:
        try: os.killpg(os.getpgid(p.pid), signal.SIGTERM)
        except: p.kill()
        p.wait(timeout=10)
