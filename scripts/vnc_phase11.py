#!/usr/bin/env python3
"""SamVNC phase 11 — open a NEW Konsole window via K-menu.

Konsole at (0,39) size 902x942 — the existing one is occupied by samcommand.
We'll:
1. Click K-menu at (6, 963) — bottom-left
2. Type "konsole" to search
3. Press Enter to launch a NEW Konsole window
4. The new window will have a fresh shell prompt
5. Run uname + free + df + ports
"""
import requests, base64, time, sys, json, subprocess

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
        path = f"/tmp/vnc_p11_{tag}.{mime}"
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
    print(f">>> click: ({x},{y}) btn={button}", file=sys.stderr)
    poll([{"type": "mouse", "x": x, "y": y, "mask": 0, "wheel": 0}])
    time.sleep(0.05)
    poll([{"type": "mouse", "x": x, "y": y, "mask": button, "wheel": 0}])
    time.sleep(0.05)
    return poll([{"type": "mouse", "x": x, "y": y, "mask": 0, "wheel": 0}])

def grab_frame(tag, wait=0.5):
    time.sleep(wait)
    r = poll([])
    return save_frame(r, tag)

def vlm_read(img_path, prompt):
    """Run z-ai vision CLI, return text content."""
    if not img_path:
        return "VLM error: no image path"
    out_json = f"/tmp/vlm_{int(time.time()*1000)}.json"
    try:
        res = subprocess.run(
            ["z-ai", "vision", "-p", prompt, "-i", img_path, "-o", out_json],
            capture_output=True, text=True, timeout=60
        )
        with open(out_json) as f:
            return json.load(f)['choices'][0]['message']['content']
    except Exception as e:
        return f"VLM error: {e}"

# Hello
poll([{"type": "hello", "version": 1, "quality": 3, "width": 0, "height": 0, "mode": 0, "fps": 0, "token": ""}])
time.sleep(0.3)

# Step 1: click K-menu
print("\n[*] click K-menu at (6, 963)", file=sys.stderr)
click(6, 963, button=1)
p = grab_frame("01_k_menu", wait=1.5)

# Step 2: type "konsole" to search
print("\n[*] type 'konsole'", file=sys.stderr)
send_text("konsole")
p = grab_frame("02_konsole_search", wait=1.0)

# Step 3: Enter to launch
print("\n[*] Enter to launch", file=sys.stderr)
tap_key("Enter", "Enter")
p = grab_frame("03_konsole_launched", wait=2.5)

# Step 4: Now there should be a new Konsole window. Move mouse to a different
# location to find it. Click in the new window area to focus it.
# The new Konsole might open on the right side (since left is occupied).
# Click in the middle-right of the screen to find it.
print("\n[*] click new Konsole window", file=sys.stderr)
click(1400, 500, button=1)
time.sleep(0.5)
p = grab_frame("04_after_click_new", wait=0.5)

# Step 5: type uname in the new window
print("\n[*] type uname", file=sys.stderr)
send_text("echo MARKER_LAL_UNAME; uname -a; echo END_UNAME")
time.sleep(0.3)
tap_key("Enter", "Enter")
p = grab_frame("05_uname", wait=1.5)

# Read it
text = vlm_read(p, "Linux terminal screenshot. Look for the line 'MARKER_LAL_UNAME'. After it, quote the next 3-5 lines exactly. Stop at END_UNAME.")
print(f"\n=== UNAME ===\n{text}\n=== END ===", file=sys.stderr)

# Step 6: mem + cpu
print("\n[*] type mem/cpu", file=sys.stderr)
send_text("echo MK_MEM; nproc; free -h; df -h /; echo END_MEM")
time.sleep(0.3)
tap_key("Enter", "Enter")
p = grab_frame("06_mem", wait=1.5)

text = vlm_read(p, "Linux terminal. Quote all lines between MK_MEM and END_MEM (inclusive).")
print(f"\n=== MEM ===\n{text}\n=== END ===", file=sys.stderr)

# Step 7: cpu flags
print("\n[*] type cpu flags", file=sys.stderr)
send_text("echo MK_CPU; grep -m1 'model name' /proc/cpuinfo; grep -oE 'avx512[a-z]*|avx2|fma|sse4_2|popcnt' /proc/cpuinfo | sort -u | tr '\\n' ' '; echo; echo END_CPU")
time.sleep(0.3)
tap_key("Enter", "Enter")
p = grab_frame("07_cpu", wait=1.5)

text = vlm_read(p, "Linux terminal. Quote lines between MK_CPU and END_CPU.")
print(f"\n=== CPU ===\n{text}\n=== END ===", file=sys.stderr)

# Step 8: ports
print("\n[*] type ports", file=sys.stderr)
send_text("echo MK_PORT; ss -tlnp 2>/dev/null | grep -E ':(80|808)' | head -10; echo END_PORT")
time.sleep(0.3)
tap_key("Enter", "Enter")
p = grab_frame("08_ports", wait=1.5)

text = vlm_read(p, "Linux terminal. Quote lines between MK_PORT and END_PORT.")
print(f"\n=== PORTS ===\n{text}\n=== END ===", file=sys.stderr)

# Step 9: samcommand location
print("\n[*] type which samcommand", file=sys.stderr)
send_text("echo MK_WHICH; which samcommand; which samcommand-v12; ls -la /usr/local/bin/sam* 2>&1; echo END_WHICH")
time.sleep(0.3)
tap_key("Enter", "Enter")
p = grab_frame("09_which", wait=1.5)

text = vlm_read(p, "Linux terminal. Quote lines between MK_WHICH and END_WHICH.")
print(f"\n=== WHICH ===\n{text}\n=== END ===", file=sys.stderr)

print("\n=== done ===", file=sys.stderr)
