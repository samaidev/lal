#!/usr/bin/env python3
"""SamVNC phase 7 — clear screen, then run ONE simple command at a time.

Key insight: previous attempts sent commands that got mixed with samcommand
output still in the terminal scrollback. By clearing first and running ONE
simple command whose output we capture via clipboard, we can be sure of
what we're reading.
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
        path = f"/tmp/vnc_p7_{tag}.{mime}"
        with open(path, "wb") as f: f.write(payload)
        print(f"  saved: {path}", file=sys.stderr)
    except Exception as e: pass

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

def run_one(cmd, wait=2.0):
    """Run ONE command (no pipes), capture stdout to /tmp/last.out, then
    cat /tmp/last.out to clipboard."""
    # Step 1: write output to file
    send_text(f"{cmd} > /tmp/last.out 2>&1")
    time.sleep(0.2)
    tap_key("Enter", "Enter")
    time.sleep(wait)
    # Step 2: pipe file to xclip
    send_text("xclip -selection clipboard < /tmp/last.out")
    time.sleep(0.2)
    tap_key("Enter", "Enter")
    time.sleep(0.8)
    clip = fetch_clipboard("clipboard")
    return clip

# Hello
poll([{"type": "hello", "version": 1, "quality": 3, "width": 0, "height": 0, "mode": 0, "fps": 0, "token": ""}])
time.sleep(0.3)

# Focus Konsole by clicking in its center
click(800, 500, button=1)
time.sleep(0.4)

# Clear screen
print("\n[*] clear screen", file=sys.stderr)
send_text("clear")
time.sleep(0.2)
tap_key("Enter", "Enter")
time.sleep(0.5)

# Now run: uname
print("\n[*] uname -a", file=sys.stderr)
out = run_one("uname -a")
print(f"\n=== UNAME ===\n{out}\n=== END ===", file=sys.stderr)
with open("/tmp/r_uname.txt", "w") as f: f.write(out or "")

# nproc + meminfo
print("\n[*] nproc + mem", file=sys.stderr)
out = run_one("echo CORES=$(nproc); echo ---; free -h; echo ---; df -h /")
print(f"\n=== SYS ===\n{out}\n=== END ===", file=sys.stderr)
with open("/tmp/r_sys.txt", "w") as f: f.write(out or "")

# cpuinfo
print("\n[*] cpuinfo", file=sys.stderr)
out = run_one("grep -m1 'model name' /proc/cpuinfo; grep -m1 flags /proc/cpuinfo | tr ' ' '\\n' | grep -E 'avx|sse|fma' | tr '\\n' ' '")
print(f"\n=== CPU ===\n{out}\n=== END ===", file=sys.stderr)
with open("/tmp/r_cpu.txt", "w") as f: f.write(out or "")

# what is on 8090
print("\n[*] ss -tlnp", file=sys.stderr)
out = run_one("ss -tlnp 2>/dev/null | grep -E ':(80|808)' || netstat -tlnp 2>/dev/null | grep -E ':(80|808)'")
print(f"\n=== PORTS ===\n{out}\n=== END ===", file=sys.stderr)
with open("/tmp/r_ports.txt", "w") as f: f.write(out or "")

# samcommand location
print("\n[*] which samcommand", file=sys.stderr)
out = run_one("which samcommand; file $(which samcommand) 2>/dev/null; ls -la $(which samcommand) 2>/dev/null")
print(f"\n=== SAMCMD ===\n{out}\n=== END ===", file=sys.stderr)
with open("/tmp/r_samcmd.txt", "w") as f: f.write(out or "")

# Try --help on samcommand binary (it might fork+exec without --help support)
# Look at strings in binary for port config
print("\n[*] strings samcommand for port", file=sys.stderr)
out = run_one("strings $(which samcommand) 2>/dev/null | grep -iE '8090|listen|port|local' | head -20")
print(f"\n=== STRINGS ===\n{out}\n=== END ===", file=sys.stderr)
with open("/tmp/r_strings.txt", "w") as f: f.write(out or "")

# Check samcommand-v12 too (user originally ran that)
print("\n[*] which samcommand-v12", file=sys.stderr)
out = run_one("which samcommand-v12 2>&1; ls -la /usr/local/bin/sam* 2>&1; ls -la ~/samcommand/ 2>&1")
print(f"\n=== V12 ===\n{out}\n=== END ===", file=sys.stderr)
with open("/tmp/r_v12.txt", "w") as f: f.write(out or "")

# Try SAMCOMMAND_PORT env
print("\n[*] env vars in samcommand binary", file=sys.stderr)
out = run_one("strings $(which samcommand) 2>/dev/null | grep -E 'SAMCOMMAND|AITUN|PORT' | head -20")
print(f"\n=== ENVVARS ===\n{out}\n=== END ===", file=sys.stderr)
with open("/tmp/r_envvars.txt", "w") as f: f.write(out or "")

print("\n=== done ===", file=sys.stderr)
