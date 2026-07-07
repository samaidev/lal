#!/usr/bin/env python3
"""SamVNC phase 6 — open a FRESH terminal window, not the one running samcommand.

The current Konsole is running samcommand in foreground, so anything we type
goes to samcommand's stdin (which it ignores). We need a new shell.

Strategy:
1. Use the desktop environment's "open terminal" shortcut.
   - KDE Plasma: Ctrl+Alt+T usually opens Konsole
   - Or click the K menu button (bottom-left) → System → Konsole
2. Run our diagnostic commands in the fresh shell.
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
        path = f"/tmp/vnc_p6_{tag}.{mime}"
        with open(path, "wb") as f: f.write(payload)
        print(f"  saved: {path}", file=sys.stderr)
    except Exception as e:
        print(f"  frame save error: {e}", file=sys.stderr)

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
    print(f">>> click: ({x},{y}) btn={button}", file=sys.stderr)
    poll([{"type": "mouse", "x": x, "y": y, "mask": 0, "wheel": 0}])
    time.sleep(0.05)
    poll([{"type": "mouse", "x": x, "y": y, "mask": button, "wheel": 0}])
    time.sleep(0.05)
    return poll([{"type": "mouse", "x": x, "y": y, "mask": 0, "wheel": 0}])

def fetch_clipboard(selection="clipboard"):
    r = poll([], clip_req=selection)
    if r and "clip" in r: return r["clip"]
    return None

def run_to_clip(cmd, wait=2.0):
    """Type cmd, Enter, wait, then pipe output to xclip."""
    full = f"({cmd}) 2>&1 | xclip -selection clipboard 2>/dev/null || true"
    send_text(full)
    time.sleep(0.2)
    tap_key("Enter", "Enter")
    time.sleep(wait)
    return fetch_clipboard("clipboard")

print("=== phase 6: open fresh terminal ===", file=sys.stderr)

# Hello
poll([{"type": "hello", "version": 1, "quality": 3, "width": 0, "height": 0, "mode": 0, "fps": 0, "token": ""}])
time.sleep(0.3)

# Take initial screenshot to find the K menu / taskbar
r = poll([])
save_frame(r, "01_initial")

# Click the K menu (KDE kickoff). Usually bottom-left corner.
# Desktop is 1920x1080. K-menu button is typically at (10, 1055) or so.
print("\n[*] click K menu (bottom-left)", file=sys.stderr)
click(20, 1060, button=1)
time.sleep(1.5)
r = poll([])
save_frame(r, "02_k_menu")

# Type "konsole" to search
print("\n[*] type 'konsole' to search", file=sys.stderr)
send_text("konsole")
time.sleep(1.5)
r = poll([])
save_frame(r, "03_konsole_search")

# Enter to launch
print("\n[*] Enter to launch Konsole", file=sys.stderr)
tap_key("Enter", "Enter")
time.sleep(2.5)
r = poll([])
save_frame(r, "04_konsole_launched")

# Now we should have a fresh Konsole window with a shell prompt.
# Run our diagnostics.
print("\n[*] run uname -a", file=sys.stderr)
out = run_to_clip("uname -a && nproc && free -h && df -h / && cat /proc/cpuinfo | grep -m1 'model name'")
print(f"\n=== SYSTEM INFO ===\n{out}\n=== END ===", file=sys.stderr)
with open("/tmp/sc_sysinfo.txt", "w") as f: f.write(out or "")

# Check what's on 8090
print("\n[*] check 8090", file=sys.stderr)
out = run_to_clip("ss -tlnp 2>/dev/null | grep -E ':8090|:8091|:8092' || netstat -tlnp 2>/dev/null | grep -E ':8090|:8091'")
print(f"\n=== PORTS ===\n{out}\n=== END ===", file=sys.stderr)
with open("/tmp/sc_ports2.txt", "w") as f: f.write(out or "")

# Check samcommand binary
print("\n[*] which samcommand + look for env vars / config", file=sys.stderr)
out = run_to_clip("which samcommand samcommand-v12 2>&1; echo ---; file $(which samcommand) 2>&1; echo ---; strings $(which samcommand) 2>/dev/null | grep -E '8090|PORT|listen' | head -10")
print(f"\n=== SAMCMD INFO ===\n{out}\n=== END ===", file=sys.stderr)
with open("/tmp/sc_samcmd_info.txt", "w") as f: f.write(out or "")

# Try to find how to change port — env var?
print("\n[*] try SAMCOMMAND_PORT env var", file=sys.stderr)
out = run_to_clip("SAMCOMMAND_PORT=8092 samcommand --no-p2p 2>&1 | head -5 & sleep 3; pkill -f 'samcommand.*8092' 2>/dev/null")
print(f"\n=== ENV VAR TEST ===\n{out}\n=== END ===", file=sys.stderr)

print("\n=== done ===", file=sys.stderr)
