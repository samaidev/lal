#!/usr/bin/env python3
"""logic_eval.py — test GB2L2 logic-guided BWN vs original all-binary BWN."""
import json, subprocess, time, urllib.request, os, signal
REPO="/workspace/lal"; SERVER=f"{REPO}/prebuilt/gpt2_server"
PROMPTS=["The capital of France is","Once upon a time","Machine learning is",
         "Hello, how are","The capital of Japan is","Water boils at"]

def start(env, flags, port):
    log=open(f"{REPO}/build/lg_{port}.log","w")
    e=os.environ.copy(); e.update(env)
    p=subprocess.Popen([SERVER,"--port",str(port)]+flags,cwd=REPO,env=e,
        stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    for _ in range(80):
        time.sleep(0.5)
        try: urllib.request.urlopen(f"http://localhost:{port}/",timeout=2); return p
        except:
            if p.poll() is not None: return None
    return None

def gen(port, prompt, t=0, n=25):
    body=json.dumps({"prompt":prompt,"n_tokens":n,"temperature":t}).encode()
    req=urllib.request.Request(f"http://localhost:{port}/generate",data=body,
        headers={"Content-Type":"application/json"},method="POST")
    with urllib.request.urlopen(req,timeout=60) as r: return json.loads(r.read())

# (label, env LAL_BINARY, flags)
CASES=[
    ("FLOAT (baseline)",            None,                                      []),
    ("BWN all-binary (orig GB2L)",   "prebuilt/gpt2_binary.bin",                ["--binary","--bwn"]),
    ("BWN+LOGIC GB2L2 (0-step)",     "build/gpt2_binary_logic2.bin",            ["--binary","--bwn"]),
    ("BWN+LOGIC GB2L2 (no --bwn)",   "build/gpt2_binary_logic2.bin",            ["--binary"]),
]
port=9000
for name, binpath, flags in CASES:
    env = {"LAL_BINARY":binpath} if binpath else {}
    print(f"\n{'#'*68}\n# {name}\n  bin={binpath} flags={flags}\n{'#'*68}")
    p=start(env, flags, port)
    if not p: print("  SERVER FAILED"); 
    if p:
        try:
            for pr in PROMPTS:
                try:
                    d=gen(port, pr)
                    t=d.get('text','?').replace('\n',' ')
                    print(f"  [P] {pr}\n      -> {t!r}  (tps={d.get('tokens_per_sec','?')})")
                except Exception as e: print(f"  [P] {pr} ERR {e}")
        finally:
            try: os.killpg(os.getpgid(p.pid), signal.SIGTERM)
            except: p.kill()
            p.wait(timeout=10)
    port += 1; time.sleep(2)
