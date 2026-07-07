#!/usr/bin/env python3
"""SamVNC phase 4 — instead of fighting VLM to read URL from screen, we'll:
1. Click Konsole to focus
2. Run: `samcommand --port 8091 --no-p2p > /tmp/sc.log 2>&1 &` (background)
3. Wait 4s, then: `cat /tmp/sc.log > /tmp/sc_out.txt`
4. Download /tmp/sc_out.txt via samcommand's /download endpoint (using old token)
   -- but wait, the old samcommand is dead. So we use the tunnel's VNC /download?
   No, SamVNC doesn't have a file download API.

Actually simpler plan:
1. Open the SamVNC clipboard sync (Ctrl+C in remote = fetch remote clipboard)
2. Run `samcommand --port 8091 --no-p2p | tee /tmp/sc.log`
3. After it prints URL/token, Ctrl+C the tee
4. Then run: `cat /tmp/sc.log | xclip -selection clipboard` (or xsel)
5. Back in our script, POST /poll with `clip_req: "clipboard"` to fetch remote CLIPBOARD
6. Parse the URL + token from the returned clip text
"""
import requests, base64, time, sys, json

BASE = "https://samcmd-studio-ctz168-mo-25517e.t.aitun.cc"
AUTH = "Bearer 56f75ad34c448526adeec8add6372ed8"
HEADERS = {"Content-Type": "application/json", "Authorization": AUTH}

last_seq = 0

def poll(inputs, clip_req=None):
    global last_seq
    body = {"seq": last_seq, "inputs": inputs}
    if clip_req:
        body["clip_req"] = clip_req
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
        path = f"/tmp/vnc_p4_{tag}.{mime}"
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
    """Trigger a copy request and return the clip text."""
    r = poll([], clip_req=selection)
    if r and "clip" in r:
        return r["clip"]
    return None

print("=== SamVNC phase 4: clipboard-based URL extraction ===", file=sys.stderr)

# Hello
r = poll([{
    "type": "hello", "version": 1, "quality": 3,
    "width": 0, "height": 0, "mode": 0, "fps": 0, "token": ""
}])
save_frame(r, "01_hello")
time.sleep(0.3)

# Click Konsole to focus
print("\n[*] focus Konsole", file=sys.stderr)
click(960, 540, button=1)
time.sleep(0.4)

# Send Ctrl+C to kill whatever is running
print("\n[*] Ctrl+C x2", file=sys.stderr)
send_combo([("Control", "ControlLeft"), ("c", "KeyC")])
time.sleep(0.6)
send_combo([("Control", "ControlLeft"), ("c", "KeyC")])
time.sleep(0.6)

# Now run samcommand in background, redirect output to a file
print("\n[*] run samcommand → /tmp/sc.log", file=sys.stderr)
cmd = "nohup samcommand --port 8091 --no-p2p > /tmp/sc.log 2>&1 &"
send_text(cmd)
time.sleep(0.3)
tap_key("Enter", "Enter")
time.sleep(4.0)

# Now read the log file via cat, then pipe to xclip to put on clipboard
print("\n[*] cat log → xclip", file=sys.stderr)
cmd2 = "cat /tmp/sc.log | xclip -selection clipboard 2>/dev/null || cat /tmp/sc.log | xsel --clipboard --input 2>/dev/null"
send_text(cmd2)
time.sleep(0.3)
tap_key("Enter", "Enter")
time.sleep(1.0)

# Fetch clipboard
print("\n[*] fetch clipboard", file=sys.stderr)
clip = fetch_clipboard("clipboard")
print(f"\n=== CLIPBOARD CONTENTS ===\n{clip}\n=== END ===", file=sys.stderr)

# Save to file for parsing
if clip:
    with open("/tmp/sc_clip.txt", "w") as f:
        f.write(clip)
    print(f"saved to /tmp/sc_clip.txt", file=sys.stderr)

# Take a frame to see what's on screen
r = poll([])
save_frame(r, "02_final")

print("\n=== done ===", file=sys.stderr)
