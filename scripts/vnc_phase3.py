#!/usr/bin/env python3
"""SamVNC phase 3 — set high quality, take a fresh screenshot, and read URL/token."""
import requests, base64, time, sys

BASE = "https://samcmd-studio-ctz168-mo-25517e.t.aitun.cc"
AUTH = "Bearer 56f75ad34c448526adeec8add6372ed8"
HEADERS = {"Content-Type": "application/json", "Authorization": AUTH}

last_seq = 0

def poll(inputs):
    global last_seq
    body = {"seq": last_seq, "inputs": inputs}
    try:
        r = requests.post(BASE + "/poll", headers=HEADERS, json=body, timeout=8)
        if not r.ok: return None
        resp = r.json()
        if "seq" in resp: last_seq = resp["seq"]
        return resp
    except Exception as e:
        print(f"  poll error: {e}", file=sys.stderr); return None

def save_frame(resp, tag, mime_pref="jpg"):
    if not resp or "frame" not in resp: return
    try:
        bin_data = base64.b64decode(resp["frame"])
        if len(bin_data) < 24: return
        fmt = bin_data[3]
        payload_len = int.from_bytes(bin_data[18:22], "little")
        payload = bin_data[24:24 + payload_len]
        mime = "png" if fmt == 2 else "jpg"
        path = f"/tmp/vnc_p3_{tag}.{mime}"
        with open(path, "wb") as f: f.write(payload)
        print(f"  saved: {path} ({len(payload)} bytes, fmt={fmt})", file=sys.stderr)
    except Exception as e:
        print(f"  frame save error: {e}", file=sys.stderr)

# Re-hello with quality=3 (high) to get sharper text
print("[*] hello with quality=3 (high)", file=sys.stderr)
r = poll([{
    "type": "hello", "version": 1, "quality": 3,
    "width": 0, "height": 0, "mode": 0, "fps": 0, "token": ""
}])
save_frame(r, "01_hello_hq")
time.sleep(0.5)

# Take a few more frames
for i in range(3):
    time.sleep(0.4)
    r = poll([])
    save_frame(r, f"02_idle_{i}")

print("[*] done", file=sys.stderr)
