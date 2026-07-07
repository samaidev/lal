#!/usr/bin/env python3
"""SamVNC phase 10 — short commands visible on screen, OCR via VLM.

Skip the clipboard entirely. Run short commands whose output we can read
directly from a screenshot. Use unique markers so we know the output is fresh.
"""
import requests, base64, time, sys

BASE = "https://samcmd-studio-ctz168-mo-25517e.t.aitun.cc"
AUTH = "Bearer 56f75ad34c448526adeec8add6372ed8"
HEADERS = {"Content-Type": "application/json", "Authorization": AUTH}

last_seq = 0

def poll(inputs, clip_req=None, timeout=8):
    global last_seq
    body = {"seq": last_seq, "inputs": inputs}
    if clip_req: body["clip_req"] = clip_req
    try:
        r = requests.post(BASE + "/poll", headers=HEADERS, json=body, timeout=timeout)
        if not r.ok: return None
        resp = r.json()
        if "seq" in resp: last_seq = resp["seq"]
        return resp
    except Exception as e:
        print(f"  poll error: {e}", file=sys.stderr); return None

def save_frame(resp, tag):
    if not resp or "frame" not in resp: return None
    try:
        bin_data = base64.b64decode(resp["frame"])
        if len(bin_data) < 24: return None
        fmt = bin_data[3]
        payload_len = int.from_bytes(bin_data[18:22], "little")
        payload = bin_data[24:24 + payload_len]
        mime = "png" if fmt == 2 else "jpg"
        path = f"/tmp/vnc_p10_{tag}.{mime}"
        with open(path, "wb") as f: f.write(payload)
        print(f"  saved: {path}", file=sys.stderr)
        return path
    except Exception as e:
        print(f"  frame save error: {e}", file=sys.stderr); return None

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

def grab_frame(tag, wait=0.5):
    """Just grab a frame without sending input."""
    time.sleep(wait)
    r = poll([])
    return save_frame(r, tag)

# Hello
poll([{"type": "hello", "version": 1, "quality": 3, "width": 0, "height": 0, "mode": 0, "fps": 0, "token": ""}])
time.sleep(0.3)

# Focus Konsole
click(800, 500, button=1)
time.sleep(0.4)

# Send Ctrl+\ first to ensure samcommand is dead
send_combo([("Control", "ControlLeft"), ("\\", "Backslash")])
time.sleep(0.8)

# Clear the screen
send_text("clear")
time.sleep(0.2)
tap_key("Enter", "Enter")
time.sleep(0.8)

# Take a "blank" screenshot
p = grab_frame("01_after_clear")

# Now type a unique marker + uname
print("\n[*] uname command", file=sys.stderr)
send_text("echo MARKER_LAL_001; uname -a; echo END_MARKER_001")
time.sleep(0.3)
tap_key("Enter", "Enter")
time.sleep(1.5)
p = grab_frame("02_uname")

# Read it with VLM
print("\n[*] VLM read", file=sys.stderr)
import subprocess
res = subprocess.run(
    ["z-ai", "vision", "-p",
     "This is a Linux Konsole terminal screenshot. Look for the line starting with 'MARKER_LAL_001'. After that marker, quote the next 5 lines exactly. Stop at END_MARKER_001.",
     "-i", p, "-o", "/tmp/vlm_uname.json"],
    capture_output=True, text=True, timeout=60
)
print(res.stderr[-300:] if res.stderr else "", file=sys.stderr)
try:
    with open("/tmp/vlm_uname.json") as f:
        vlm = json.load(f)
    print(f"\n=== VLM UNAME ===\n{vlm['choices'][0]['message']['content']}\n=== END ===", file=sys.stderr)
except Exception as e:
    print(f"VLM error: {e}", file=sys.stderr)

# Now do nproc + mem + disk
print("\n[*] mem command", file=sys.stderr)
send_text("echo MARKER_LAL_002; nproc; free -h | head -3; df -h / | tail -1; echo END_MARKER_002")
time.sleep(0.3)
tap_key("Enter", "Enter")
time.sleep(1.5)
p = grab_frame("03_mem")

res = subprocess.run(
    ["z-ai", "vision", "-p",
     "Linux terminal screenshot. Quote all lines between MARKER_LAL_002 and END_MARKER_002 (inclusive).",
     "-i", p, "-o", "/tmp/vlm_mem.json"],
    capture_output=True, text=True, timeout=60
)
try:
    with open("/tmp/vlm_mem.json") as f:
        vlm = json.load(f)
    print(f"\n=== VLM MEM ===\n{vlm['choices'][0]['message']['content']}\n=== END ===", file=sys.stderr)
except Exception as e:
    print(f"VLM error: {e}", file=sys.stderr)

# CPU info
print("\n[*] cpu command", file=sys.stderr)
send_text("echo MK3; grep -m1 'model name' /proc/cpuinfo; grep -oE 'avx512[a-z]*|avx2|fma|sse4_2|popcnt' /proc/cpuinfo | head -1 | tr '\\n' ' '; echo END3")
time.sleep(0.3)
tap_key("Enter", "Enter")
time.sleep(1.5)
p = grab_frame("04_cpu")

res = subprocess.run(
    ["z-ai", "vision", "-p",
     "Linux terminal. Quote lines between MK3 and END3.",
     "-i", p, "-o", "/tmp/vlm_cpu.json"],
    capture_output=True, text=True, timeout=60
)
try:
    with open("/tmp/vlm_cpu.json") as f:
        vlm = json.load(f)
    print(f"\n=== VLM CPU ===\n{vlm['choices'][0]['message']['content']}\n=== END ===", file=sys.stderr)
except Exception as e:
    print(f"VLM error: {e}", file=sys.stderr)

# Ports
print("\n[*] ports command", file=sys.stderr)
send_text("echo MK4; ss -tlnp 2>/dev/null | grep -E ':(80|808)' ; echo END4")
time.sleep(0.3)
tap_key("Enter", "Enter")
time.sleep(1.5)
p = grab_frame("05_ports")

res = subprocess.run(
    ["z-ai", "vision", "-p",
     "Linux terminal. Quote lines between MK4 and END4.",
     "-i", p, "-o", "/tmp/vlm_ports.json"],
    capture_output=True, text=True, timeout=60
)
try:
    with open("/tmp/vlm_ports.json") as f:
        vlm = json.load(f)
    print(f"\n=== VLM PORTS ===\n{vlm['choices'][0]['message']['content']}\n=== END ===", file=sys.stderr)
except Exception as e:
    print(f"VLM error: {e}", file=sys.stderr)

# samcommand path
print("\n[*] which samcommand", file=sys.stderr)
send_text("echo MK5; which samcommand; ls /usr/local/bin/ | grep -i sam; echo END5")
time.sleep(0.3)
tap_key("Enter", "Enter")
time.sleep(1.5)
p = grab_frame("06_which")

res = subprocess.run(
    ["z-ai", "vision", "-p",
     "Linux terminal. Quote lines between MK5 and END5.",
     "-i", p, "-o", "/tmp/vlm_which.json"],
    capture_output=True, text=True, timeout=60
)
try:
    with open("/tmp/vlm_which.json") as f:
        vlm = json.load(f)
    print(f"\n=== VLM WHICH ===\n{vlm['choices'][0]['message']['content']}\n=== END ===", file=sys.stderr)
except Exception as e:
    print(f"VLM error: {e}", file=sys.stderr)

print("\n=== done ===", file=sys.stderr)
import json
