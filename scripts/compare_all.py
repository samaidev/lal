#!/usr/bin/env python3
"""compare_all.py — final comparison across all configurations.
Tests FLOAT baseline, original BWN, logic+STE-300 small corpus,
and wikitext+logic+STE-400 (the new combined path).
"""
import json, subprocess, time, urllib.request, os, signal, sys
REPO="/workspace/lal"; SERVER=f"{REPO}/prebuilt/gpt2_server"
PROMPTS=["The capital of France is","Once upon a time",
         "Machine learning is","Hello, how are"]

def start(env, flags, port):
    log=open(f"{REPO}/build/cmp_{port}.log","w")
    e=os.environ.copy(); e.update(env)
    p=subprocess.Popen([SERVER,"--port",str(port)]+flags,cwd=REPO,env=e,
        stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    for _ in range(80):
        time.sleep(0.5)
        try: urllib.request.urlopen(f"http://localhost:{port}/",timeout=2); return p
        except:
            if p.poll() is not None: return None
    return None

def gen(port, prompt, t=0, n=20):
    body=json.dumps({"prompt":prompt,"n_tokens":n,"temperature":t}).encode()
    req=urllib.request.Request(f"http://localhost:{port}/generate",data=body,
        headers={"Content-Type":"application/json"},method="POST")
    with urllib.request.urlopen(req,timeout=60) as r: return json.loads(r.read())

# (label, env LAL_BINARY (or None for float), flags)
CASES=[
    ("FLOAT baseline (OpenBLAS)",       None,                                      []),
    ("BWN orig all-binary (GB2L)",       "prebuilt/gpt2_binary.bin",                ["--binary","--bwn"]),
    ("BWN+LOGIC+STE-300 small (GB2L2)", "build/gpt2_binary_logic_ste300.bin",      ["--binary","--bwn"]),
    ("BWN+LOGIC+wikitext STE-400 (GB2L2)", "build/gpt2_binary_wiki_logic_ste400.bin", ["--binary","--bwn"]),
]
port=8900
results={}
for name, binpath, flags in CASES:
    env = {"LAL_BINARY":binpath} if binpath else {}
    print(f"\n{'#'*72}\n# {name}\n  bin={binpath} flags={flags}\n{'#'*72}", flush=True)
    p=start(env, flags, port)
    if not p:
        print("  SERVER FAILED", flush=True)
        results[name]="FAILED"; port+=1; time.sleep(2); continue
    outputs=[]
    try:
        for pr in PROMPTS:
            try:
                d=gen(port, pr)
                t=d.get('text','?').replace('\n',' ')
                line=f"  [P] {pr}\n      -> {t!r}  (tps={d.get('tokens_per_sec','?')})"
                print(line, flush=True)
                outputs.append((pr, t, d.get('tokens_per_sec','?')))
            except Exception as e:
                print(f"  [P] {pr} ERR {e}", flush=True)
                outputs.append((pr, "ERR", str(e)))
    finally:
        try: os.killpg(os.getpgid(p.pid), signal.SIGTERM)
        except: p.kill()
        p.wait(timeout=10)
    results[name]=outputs
    port+=1; time.sleep(2)

print("\n\n"+"="*72+"\nFINAL SUMMARY\n"+"="*72)
for name, outs in results.items():
    print(f"\n[{name}]")
    if isinstance(outs, str): print(f"  {outs}"); continue
    for pr, txt, tps in outs:
        print(f"  {pr!r:40s} -> {txt!r}")
