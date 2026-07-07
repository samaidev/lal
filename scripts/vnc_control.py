#!/usr/bin/env python3
"""SamVNC remote control — open a terminal and run samcommand with a new port.

Strategy:
1. POST /poll with input events to send keystrokes to the remote X11 desktop
2. Use Alt+F2 → xterm to open a terminal (works on most Linux desktops)
3. Type the samcommand command and Enter
4. Read back the /poll response frames to confirm

We use /poll (HTTP long-poll) because it doesn't need a websocket client.
"""
import requests, json, base64, time, sys

BASE = "https://samcmd-studio-ctz168-mo-25517e.t.aitun.cc"
AUTH = "Bearer 56f75ad34c448526adeec8add6372ed8"
HEADERS = {"Content-Type": "application/json", "Authorization": AUTH}

last_seq = 0

def poll(inputs, wait_ms=200):
    global last_seq
    body = {"seq": last_seq, "inputs": inputs}
    try:
        r = requests.post(BASE + "/poll", headers=HEADERS, json=body, timeout=5)
        if not r.ok:
            print(f"  poll HTTP {r.status_code}", file=sys.stderr)
            return None
        resp = r.json()
        if "seq" in resp:
            last_seq = resp["seq"]
        return resp
    except Exception as e:
        print(f"  poll error: {e}", file=sys.stderr)
        return None

def save_frame(resp, tag=""):
    if not resp or "frame" not in resp:
        return
    try:
        bin_data = base64.b64decode(resp["frame"])
        if len(bin_data) < 24: return
        fmt = bin_data[3]
        payload_len = int.from_bytes(bin_data[18:22], "little")
        payload = bin_data[24:24 + payload_len]
        mime = "png" if fmt == 2 else "jpg"
        path = f"/tmp/vnc_frame_{tag or 'x'}.{mime}"
        with open(path, "wb") as f:
            f.write(payload)
        print(f"  saved: {path} ({len(payload)} bytes, fmt={fmt})", file=sys.stderr)
    except Exception as e:
        print(f"  frame save error: {e}", file=sys.stderr)

def send_text(text):
    print(f">>> type: {text!r}", file=sys.stderr)
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
    print(f">>> click: ({x},{y}) btn={button}", file=sys.stderr)
    poll([{"type": "mouse", "x": x, "y": y, "mask": 0, "wheel": 0}])
    time.sleep(0.05)
    poll([{"type": "mouse", "x": x, "y": y, "mask": button, "wheel": 0}])
    time.sleep(0.05)
    return poll([{"type": "mouse", "x": x, "y": y, "mask": 0, "wheel": 0}])

def hello():
    return poll([{
        "type": "hello", "version": 1, "quality": 1,
        "width": 0, "height": 0, "mode": 0, "fps": 0, "token": ""
    }])

print("=== SamVNC remote control ===", file=sys.stderr)

print("\n[1] hello handshake", file=sys.stderr)
r = hello()
save_frame(r, "01_hello")
time.sleep(0.3)

print("\n[2] Alt+F2 for run dialog", file=sys.stderr)
send_combo([("Alt", "AltLeft"), ("F2", "F2")])
time.sleep(1.2)
r = poll([])
save_frame(r, "02_after_alt_f2")

print("\n[3] type 'xterm'", file=sys.stderr)
send_text("xterm")
time.sleep(0.3)
tap_key("Enter", "Enter")
time.sleep(1.8)
r = poll([])
save_frame(r, "03_after_xterm")

print("\n[4] type samcommand restart command", file=sys.stderr)
cmd = "pkill -f samcommand; sleep 1; samcommand --port 8091 --no-p2p"
send_text(cmd)
time.sleep(0.3)
tap_key("Enter", "Enter")
time.sleep(3.0)
r = poll([])
save_frame(r, "04_after_command")

print("\n=== done ===", file=sys.stderr)
print(f"last_seq = {last_seq}", file=sys.stderr)
