#!/usr/bin/env python3
"""SamVNC phase 2 — kill the running samcommand and restart on a new port.

Based on phase 1 inspection:
- A Konsole terminal is already open showing the running samcommand
- Alt+F2 didn't work, browser opened instead
- We need to: focus the Konsole → Ctrl+C → type the restart command
"""
import requests, base64, time, sys

BASE = "https://samcmd-studio-ctz168-mo-25517e.t.aitun.cc"
AUTH = "Bearer 56f75ad34c448526adeec8add6372ed8"
HEADERS = {"Content-Type": "application/json", "Authorization": AUTH}

last_seq = 0

def poll(inputs):
    global last_seq
    body = {"seq": last_seq, "inputs": inputs}
    try:
        r = requests.post(BASE + "/poll", headers=HEADERS, json=body, timeout=5)
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
        path = f"/tmp/vnc_p2_{tag}.{mime}"
        with open(path, "wb") as f: f.write(payload)
        print(f"  saved: {path}", file=sys.stderr)
    except Exception as e:
        print(f"  frame save error: {e}", file=sys.stderr)

def send_text(text):
    print(f">>> type: {text[:80]}{'...' if len(text) > 80 else ''}", file=sys.stderr)
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

print("=== SamVNC phase 2: restart samcommand ===", file=sys.stderr)

# Step 1: hello
print("\n[1] hello", file=sys.stderr)
r = hello()
save_frame(r, "01_hello")
time.sleep(0.3)

# Step 2: Click on the Konsole terminal window to focus it.
# Desktop is 1920x1080. The Konsole window likely fills most of the screen.
# Click in the middle of the terminal area (where text content is shown).
print("\n[2] click Konsole to focus", file=sys.stderr)
click(960, 540, button=1)
time.sleep(0.5)
r = poll([])
save_frame(r, "02_after_click")

# Step 3: send Ctrl+C to kill the running samcommand in foreground
print("\n[3] Ctrl+C to kill running samcommand", file=sys.stderr)
send_combo([("Control", "ControlLeft"), ("c", "KeyC")])
time.sleep(1.0)
r = poll([])
save_frame(r, "03_after_ctrl_c")

# Step 4: send Ctrl+C again to be sure (in case first one just interrupted aitun-client)
print("\n[4] Ctrl+C again", file=sys.stderr)
send_combo([("Control", "ControlLeft"), ("c", "KeyC")])
time.sleep(1.0)
r = poll([])
save_frame(r, "04_after_ctrl_c_2")

# Step 5: Now type the command to restart samcommand on a new port.
# Wait a bit and ensure prompt is back, then type.
print("\n[5] type restart command", file=sys.stderr)
cmd = "samcommand --port 8091 --no-p2p --verbose"
send_text(cmd)
time.sleep(0.3)
tap_key("Enter", "Enter")
time.sleep(4.0)
r = poll([])
save_frame(r, "05_after_restart")

# Wait more and grab another frame
time.sleep(3.0)
r = poll([])
save_frame(r, "06_final")

print("\n=== done ===", file=sys.stderr)
