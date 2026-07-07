#!/usr/bin/env python3
"""SamVNC phase 5 — diagnose what's on 8090, and find a free port.

Plan:
1. Focus Konsole
2. Kill ALL samcommand + aitun-client processes
3. Run `ss -tlnp | grep 8090` to see what owns 8090
4. Run `pgrep -af samcommand` and `pgrep -af aitun`
5. Pipe all output through xclip so we can fetch via clipboard
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

def send_text(text):
    print(f">>> type: {text[:100]}", file=sys.stderr)
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

def run_cmd(cmd, wait=1.5):
    """Send command, Enter, wait, return clipboard contents."""
    send_text(cmd)
    time.sleep(0.2)
    tap_key("Enter", "Enter")
    time.sleep(wait)
    # Now put output on clipboard
    send_text(f"__out=$(!! 2>&1); echo \"$__out\" | xclip -selection clipboard 2>/dev/null || echo \"$__out\" | xsel --clipboard --input 2>/dev/null")
    # Actually !! would re-run the last command. Better: pipe directly.
    return None

def run_cmd_to_clip(cmd, wait=1.5):
    """Run cmd, append | xclip to capture output."""
    # Wrap in a subshell to handle pipes/redirects
    full = f"({cmd}) 2>&1 | xclip -selection clipboard 2>/dev/null || ({cmd}) 2>&1 | xsel --clipboard --input 2>/dev/null"
    send_text(full)
    time.sleep(0.2)
    tap_key("Enter", "Enter")
    time.sleep(wait)
    return fetch_clipboard("clipboard")

print("=== phase 5: diagnose 8090 ===", file=sys.stderr)

# Hello
poll([{"type": "hello", "version": 1, "quality": 3, "width": 0, "height": 0, "mode": 0, "fps": 0, "token": ""}])
time.sleep(0.3)

# Focus Konsole
click(960, 540, button=1)
time.sleep(0.3)

# Ctrl+C x3 to kill any running process
for _ in range(3):
    send_combo([("Control", "ControlLeft"), ("c", "KeyC")])
    time.sleep(0.4)

# Run: ss -tlnp | grep 8090 → clipboard
print("\n[*] what is on 8090?", file=sys.stderr)
out = run_cmd_to_clip("ss -tlnp 2>/dev/null | grep -E ':8090|:8091' || netstat -tlnp 2>/dev/null | grep -E ':8090|:8091'")
print(f"\n=== PORT 8090/8091 ===\n{out}\n=== END ===", file=sys.stderr)
with open("/tmp/sc_ports.txt", "w") as f: f.write(out or "")

# Run: ps aux | grep -E 'samcommand|aitun|samvnc' → clipboard
print("\n[*] processes", file=sys.stderr)
out = run_cmd_to_clip("ps aux | grep -E 'samcommand|aitun|samvnc|vnc' | grep -v grep")
print(f"\n=== PROCESSES ===\n{out}\n=== END ===", file=sys.stderr)
with open("/tmp/sc_procs.txt", "w") as f: f.write(out or "")

# Run: samcommand --help to see if --port is even a flag
print("\n[*] samcommand --help", file=sys.stderr)
out = run_cmd_to_clip("samcommand --help 2>&1 | head -40")
print(f"\n=== HELP ===\n{out}\n=== END ===", file=sys.stderr)
with open("/tmp/sc_help.txt", "w") as f: f.write(out or "")

print("\n=== done ===", file=sys.stderr)
