#!/usr/bin/env python3
"""SamVNC phase 9 — use a unique filename per command so old data can't leak."""
import requests, base64, time, sys, os

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
        path = f"/tmp/vnc_p9_{tag}.{mime}"
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

def run_one(cmd, wait=2.0, tag="x"):
    """Use a UNIQUE filename per call so stale data can't leak."""
    fn = f"/tmp/cmd_{tag}_{int(time.time()*1000)}.out"
    # Write to unique file, then cat to clipboard
    send_text(f"{cmd} > {fn} 2>&1")
    time.sleep(0.2)
    tap_key("Enter", "Enter")
    time.sleep(wait)
    # Verify file exists, then cat to clipboard
    send_text(f"test -f {fn} && cat {fn} | xclip -selection clipboard || echo NOFILE | xclip -selection clipboard")
    time.sleep(0.2)
    tap_key("Enter", "Enter")
    time.sleep(1.0)
    clip = fetch_clipboard("clipboard")
    return clip

# Hello
poll([{"type": "hello", "version": 1, "quality": 3, "width": 0, "height": 0, "mode": 0, "fps": 0, "token": ""}])
time.sleep(0.3)

# Focus Konsole
click(800, 500, button=1)
time.sleep(0.4)

# Clear screen
print("\n[*] clear", file=sys.stderr)
send_text("clear")
time.sleep(0.2)
tap_key("Enter", "Enter")
time.sleep(0.5)

# Send Ctrl+\ in case samcommand is back
send_combo([("Control", "ControlLeft"), ("\\", "Backslash")])
time.sleep(0.5)

# Now test: a unique marker command
print("\n[*] echo MARKER test", file=sys.stderr)
out = run_one("echo HELLO_FROM_LAL_$(date +%s); uname -a; echo ---; nproc", wait=1.5, tag="uname")
print(f"\n=== UNAME ===\n{out}\n=== END ===", file=sys.stderr)
with open("/tmp/r_uname3.txt", "w") as f: f.write(out or "")

# Save frame to see what's on screen
r = poll([])
save_frame(r, "01_after_uname")

# Memory + disk
print("\n[*] mem + disk", file=sys.stderr)
out = run_one("free -h; echo ---; df -h /", wait=1.5, tag="mem")
print(f"\n=== MEM ===\n{out}\n=== END ===", file=sys.stderr)
with open("/tmp/r_mem.txt", "w") as f: f.write(out or "")

# CPU info
print("\n[*] cpu info", file=sys.stderr)
out = run_one("grep -m1 'model name' /proc/cpuinfo; grep -m1 flags /proc/cpuinfo | tr ' ' '\\n' | grep -E 'avx|sse|fma' | tr '\\n' ' '; echo", wait=1.5, tag="cpu")
print(f"\n=== CPU ===\n{out}\n=== END ===", file=sys.stderr)
with open("/tmp/r_cpu2.txt", "w") as f: f.write(out or "")

# Ports
print("\n[*] ports", file=sys.stderr)
out = run_one("ss -tlnp 2>/dev/null | grep -E ':80[0-9]' | head -10", wait=1.5, tag="ports")
print(f"\n=== PORTS ===\n{out}\n=== END ===", file=sys.stderr)
with open("/tmp/r_ports3.txt", "w") as f: f.write(out or "")

# samcommand binary
print("\n[*] which samcommand", file=sys.stderr)
out = run_one("which samcommand samcommand-v12 2>&1; ls -la /usr/local/bin/sam* 2>&1", wait=1.5, tag="which")
print(f"\n=== WHICH ===\n{out}\n=== END ===", file=sys.stderr)
with open("/tmp/r_which.txt", "w") as f: f.write(out or "")

print("\n=== done ===", file=sys.stderr)
