#!/usr/bin/env python3
"""SamVNC phase 12 — fix Enter key, run commands in the original (free) Konsole.

Discovery: the original Konsole IS at a shell prompt [~]$. samcommand died.
But my Enter key presses might be getting lost. Let me explicitly:
1. Click in the Konsole text area to focus
2. Send Ctrl+C a few times to abort any half-typed command
3. Send a CLEAR line first (Ctrl+U to clear current line)
4. Then type 'uname -a' and send Enter as a SEPARATE message
5. Take screenshot
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
        path = f"/tmp/vnc_p12_{tag}.{mime}"
        with open(path, "wb") as f: f.write(payload)
        print(f"  saved: {path}", file=sys.stderr)
        return path
    except Exception as e:
        print(f"  frame save error: {e}", file=sys.stderr); return None

def send_text(text):
    print(f">>> type: {text[:120]}", file=sys.stderr)
    return poll([{"type": "text", "text": text}])

def tap_key(key, code):
    print(f">>> key: {key}", file=sys.stderr)
    poll([{"type": "key", "key": key, "code": code, "down": True, "repeat": False}])
    time.sleep(0.05)
    poll([{"type": "key", "key": key, "code": code, "down": False, "repeat": False}])

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
    print(f">>> click: ({x},{y})", file=sys.stderr)
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
    if not img_path: return "VLM error: no image"
    out_json = f"/tmp/vlm_{int(time.time()*1000)}.json"
    try:
        subprocess.run(
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

# Click in the middle of the existing Konsole to focus
print("\n[*] focus Konsole", file=sys.stderr)
click(450, 500, button=1)
time.sleep(0.5)

# Send Ctrl+C a few times to abort the half-typed 'konsole' command
print("\n[*] Ctrl+C to abort", file=sys.stderr)
for _ in range(3):
    send_combo([("Control", "ControlLeft"), ("c", "KeyC")])
    time.sleep(0.2)

# Clear the current line with Ctrl+U
print("\n[*] Ctrl+U to clear line", file=sys.stderr)
send_combo([("Control", "ControlLeft"), ("u", "KeyU")])
time.sleep(0.3)

# Take a screenshot to see prompt state
p = grab_frame("01_after_clear", wait=0.5)
print(vlm_read(p, "Linux terminal. What's on the last line? Is there a prompt like [~]$ visible? Quote the last 3 lines."), file=sys.stderr)

# Now type a simple uname command — IMPORTANT: send Enter as a separate poll
print("\n[*] type 'uname -a'", file=sys.stderr)
send_text("uname -a")
time.sleep(0.3)

# Send Enter explicitly
print("\n[*] Enter (separate)", file=sys.stderr)
tap_key("Enter", "Enter")
time.sleep(1.5)

p = grab_frame("02_uname", wait=0.5)
text = vlm_read(p, "Linux terminal screenshot. Quote the last 5 lines exactly. Is the output of 'uname -a' visible? What does it say?")
print(f"\n=== UNAME ===\n{text}\n=== END ===", file=sys.stderr)

# Run nproc + memory
print("\n[*] type 'nproc; free -h; df -h /'", file=sys.stderr)
send_text("nproc; echo ---; free -h; echo ---; df -h /")
time.sleep(0.3)
tap_key("Enter", "Enter")
time.sleep(1.5)
p = grab_frame("03_mem", wait=0.5)
text = vlm_read(p, "Linux terminal. Quote the last 10 lines. I'm looking for output of: nproc, free -h, df -h /")
print(f"\n=== MEM ===\n{text}\n=== END ===", file=sys.stderr)

# CPU
print("\n[*] cpu info", file=sys.stderr)
send_text("grep -m1 'model name' /proc/cpuinfo; grep -oE 'avx512[a-z]+|avx2|fma|sse4_2|popcnt' /proc/cpuinfo | sort -u | tr '\\n' ' '; echo")
time.sleep(0.3)
tap_key("Enter", "Enter")
time.sleep(1.5)
p = grab_frame("04_cpu", wait=0.5)
text = vlm_read(p, "Linux terminal. Quote the last 5 lines. Looking for CPU model name and SIMD flags (avx2, fma, etc.).")
print(f"\n=== CPU ===\n{text}\n=== END ===", file=sys.stderr)

# Ports
print("\n[*] ports", file=sys.stderr)
send_text("ss -tlnp 2>/dev/null | grep -E ':(80|808)' | head -10")
time.sleep(0.3)
tap_key("Enter", "Enter")
time.sleep(1.5)
p = grab_frame("05_ports", wait=0.5)
text = vlm_read(p, "Linux terminal. Quote the last 10 lines. Looking for ss/netstat output showing what's listening on ports 80-808.")
print(f"\n=== PORTS ===\n{text}\n=== END ===", file=sys.stderr)

# samcommand location
print("\n[*] which samcommand", file=sys.stderr)
send_text("which samcommand; which samcommand-v12; ls -la /usr/local/bin/sam* 2>/dev/null")
time.sleep(0.3)
tap_key("Enter", "Enter")
time.sleep(1.5)
p = grab_frame("06_which", wait=0.5)
text = vlm_read(p, "Linux terminal. Quote the last 5 lines. Looking for paths to samcommand binaries.")
print(f"\n=== WHICH ===\n{text}\n=== END ===", file=sys.stderr)

# Now kill any running samcommand + aitun processes
print("\n[*] kill samcommand", file=sys.stderr)
send_text("pkill -9 -f samcommand; pkill -9 -f aitun; sleep 1; ss -tlnp 2>/dev/null | grep -E ':(80|808)' | head -5")
time.sleep(0.3)
tap_key("Enter", "Enter")
time.sleep(2.0)
p = grab_frame("07_after_kill", wait=0.5)
text = vlm_read(p, "Linux terminal. Quote the last 5 lines. Looking for ports that are still listening after pkill.")
print(f"\n=== AFTER KILL ===\n{text}\n=== END ===", file=sys.stderr)

print("\n=== done ===", file=sys.stderr)
