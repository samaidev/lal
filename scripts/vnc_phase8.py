#!/usr/bin/env python3
"""SamVNC phase 8 — kill samcommand for real with kill -9, then run commands.

The shell prompt visible in the terminal is NOT a real shell prompt — it's
samcommand's output. samcommand is still running. To get a real shell, we
need to either:
(a) Open a NEW Konsole tab/window (Ctrl+Shift+T in Konsole)
(b) Background samcommand with Ctrl+Z, then run commands
(c) Kill samcommand with Ctrl+\ (SIGQUIT)

Strategy: try Ctrl+\ (SIGQUIT) first — this is stronger than Ctrl+C and
kills most programs. If that doesn't work, try Ctrl+Z to suspend, then
we get a real shell prompt.
"""
import requests, base64, time, sys

BASE = "https://samcmd-studio-ctz168-mo-25517e.t.aitun.cc"
AUTH = "Bearer 56f75ad34c448526adeec8add6372ed8"
HEADERS = {"Content-Type": "application/json", "Authorization": AUTH}

last_seq = 0

def poll(inputs, clip_req=None):
    global last_seq
    body = {"seq": last_seq, "inputs": inputs}
    if clip_req: body["clip_req"] = clip_req
    try:
        r = requests.post(BASE + "/poll", headers=HEADERS, json=body, timeout=8)
        if not r.ok: return None
        resp = r.json()
        if "seq" in resp: last_seq = resp["seq"]
        return resp
    except Exception as e:
        print(f"  poll error: {e}", file=sys.stderr); return None

def save_frame(resp, tag):
    if not resp or "frame" not in resp: return
    try:
        bin_data = base64.b64decode(resp["frame"])
        if len(bin_data) < 24: return
        fmt = bin_data[3]
        payload_len = int.from_bytes(bin_data[18:22], "little")
        payload = bin_data[24:24 + payload_len]
        mime = "png" if fmt == 2 else "jpg"
        path = f"/tmp/vnc_p8_{tag}.{mime}"
        with open(path, "wb") as f: f.write(payload)
        print(f"  saved: {path}", file=sys.stderr)
    except: pass

def send_text(text):
    print(f">>> type: {text[:120]}", file=sys.stderr)
    return poll([{"type": "text", "text": text}])

def tap_key(key, code):
    poll([{"type": "key", "key": key, "code": code, "down": True, "repeat": False}])
    time.sleep(0.05)
    return poll([{"type": "key", "key": key, "code": code, "down": False, "repeat": False}])

def send_combo(keys):
    inputs = []
    for k, c in keys:
        inputs.append({"type": "key", "key": k, "code": c, "down": True, "repeat": False})
    poll(inputs)
    time.sleep(0.08)
    inputs = []
    for k, c in reversed(keys):
        inputs.append({"type": "key", "key": k, "code": c, "down": False, "repeat": False})
    return poll(inputs)

def click(x, y, button=1):
    poll([{"type": "mouse", "x": x, "y": y, "mask": 0, "wheel": 0}])
    time.sleep(0.05)
    poll([{"type": "mouse", "x": x, "y": y, "mask": button, "wheel": 0}])
    time.sleep(0.05)
    return poll([{"type": "mouse", "x": x, "y": y, "mask": 0, "wheel": 0}])

def fetch_clipboard(selection="clipboard"):
    r = poll([], clip_req=selection)
    if r and "clip" in r: return r["clip"]
    return None

def run_one(cmd, wait=2.0):
    send_text(f"{cmd} > /tmp/last.out 2>&1")
    time.sleep(0.2)
    tap_key("Enter", "Enter")
    time.sleep(wait)
    send_text("xclip -selection clipboard < /tmp/last.out")
    time.sleep(0.2)
    tap_key("Enter", "Enter")
    time.sleep(0.8)
    return fetch_clipboard("clipboard")

# Hello
poll([{"type": "hello", "version": 1, "quality": 3, "width": 0, "height": 0, "mode": 0, "fps": 0, "token": ""}])
time.sleep(0.3)

# Focus Konsole
click(800, 500, button=1)
time.sleep(0.4)

# Try Ctrl+\ (SIGQUIT) — stronger than Ctrl+C
print("\n[*] Ctrl+\\ (SIGQUIT)", file=sys.stderr)
send_combo([("Control", "ControlLeft"), ("\\", "Backslash")])
time.sleep(1.0)
r = poll([])
save_frame(r, "01_after_sigquit")

# Try Ctrl+Z (suspend)
print("\n[*] Ctrl+Z (suspend)", file=sys.stderr)
send_combo([("Control", "ControlLeft"), ("z", "KeyZ")])
time.sleep(1.0)
r = poll([])
save_frame(r, "02_after_suspend")

# Now type "jobs" to see what's suspended
print("\n[*] jobs", file=sys.stderr)
out = run_one("jobs -l; echo ---; ps -ef | grep -E 'samcommand|aitun' | grep -v grep")
print(f"\n=== JOBS ===\n{out}\n=== END ===", file=sys.stderr)
with open("/tmp/r_jobs.txt", "w") as f: f.write(out or "")

# Now we should have a real shell. Run uname
print("\n[*] uname", file=sys.stderr)
out = run_one("uname -a; echo ---; nproc; echo ---; free -h")
print(f"\n=== UNAME ===\n{out}\n=== END ===", file=sys.stderr)
with open("/tmp/r_uname2.txt", "w") as f: f.write(out or "")

# ports
print("\n[*] ports", file=sys.stderr)
out = run_one("ss -tlnp 2>/dev/null | grep -E ':80[0-9]' || netstat -tlnp 2>/dev/null | grep -E ':80[0-9]'")
print(f"\n=== PORTS ===\n{out}\n=== END ===", file=sys.stderr)
with open("/tmp/r_ports2.txt", "w") as f: f.write(out or "")

print("\n=== done ===", file=sys.stderr)
