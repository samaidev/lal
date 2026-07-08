#!/usr/bin/env python3
"""quick_bwn_temp.py — test BWN with temperature sampling (not just greedy)."""
import json, subprocess, time, urllib.request, os, signal, sys
REPO="/workspace/lal"; SERVER=f"{REPO}/prebuilt/gpt2_server"
PROMPTS=["The capital of France is","Once upon a time","Machine learning is","Hello, how are"]
def start(env, port):
    log=open(f"{REPO}/build/qt_{port}.log","w")
    e=os.environ.copy(); e.update(env)
    p=subprocess.Popen([SERVER,"--binary","--bwn","--port",str(port)],
        cwd=REPO, env=e, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    for _ in range(80):
        time.sleep(0.5)
        try: urllib.request.urlopen(f"http://localhost:{port}/",timeout=2); return p
        except:
            if p.poll() is not None: return None
    return None
def gen(port, prompt, temp, n=25):
    body=json.dumps({"prompt":prompt,"n_tokens":n,"temperature":temp}).encode()
    req=urllib.request.Request(f"http://localhost:{port}/generate",data=body,
        headers={"Content-Type":"application/json"},method="POST")
    with urllib.request.urlopen(req,timeout=60) as r: return json.loads(r.read())
label=sys.argv[1] if len(sys.argv)>1 else "test"
binpath=sys.argv[2] if len(sys.argv)>2 else "prebuilt/gpt2_binary.bin"
port=8750
print(f"\n=== {label} (binary={binpath}) ===")
p=start({"LAL_BINARY":binpath}, port)
if not p: print("FAILED"); sys.exit(1)
try:
    for pr in PROMPTS:
        for temp in [0.0, 0.7, 1.0]:
            try:
                d=gen(port,pr,temp)
                t=d.get('text','?').replace('\n',' ')
                print(f"  T={temp:.1f} [P] {pr}\n      -> {t!r}  (tps={d.get('tokens_per_sec','?')})")
            except Exception as e: print(f"  T={temp:.1f} [P] {pr} ERR {e}")
finally:
    try: os.killpg(os.getpgid(p.pid), signal.SIGTERM)
    except: p.kill()
    p.wait(timeout=10)
