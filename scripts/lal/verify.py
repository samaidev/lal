"""
verify.py — verify the compiled C demo matches the Python reference.

Builds demo.c, runs it on a battery of test inputs, and compares against
the in-process Python reference interpreter.
"""

import os
import sys
import subprocess
import random
sys.path.insert(0, os.path.dirname(__file__))

from lal import parse, run_reference

LAL_FILE = "/home/z/my-project/scripts/lal/demo.lal"
SRC_DIR  = "/home/z/my-project/download/lal-mvp/src"
BUILD_DIR= "/home/z/my-project/download/lal-mvp/build"
C_FILE   = os.path.join(SRC_DIR, "demo.c")
BIN_FILE = os.path.join(BUILD_DIR, "demo_x86_64")

def main():
    with open(LAL_FILE) as f:
        source = f.read()
    concepts, bounds, memories, relates, rules = parse(source)
    concept_map = {c.name: c.vec for c in concepts}

    os.makedirs(BUILD_DIR, exist_ok=True)
    r = subprocess.run(
        ["gcc", "-O3", "-Wall", "-Wextra", "-o", BIN_FILE, C_FILE],
        capture_output=True, text=True
    )
    if r.returncode != 0:
        print("[!] gcc failed:")
        print(r.stderr)
        sys.exit(1)
    print(f"[*] built {BIN_FILE}")

    # Test cases
    test_cases = []
    for name in ["cat", "dog", "car", "vehicle"]:
        test_cases.append((f"=={name}", concept_map[name]))
    random.seed(42)
    for _ in range(16):
        v = [random.uniform(0, 1) for _ in range(8)]
        test_cases.append(("random", v))

    n_pass = 0
    n_fail = 0
    for label, vec in test_cases:
        env = run_reference(concepts, bounds, memories, relates, rules, "classify", vec)
        ref_label = env["best"]
        label_to_idx = {"cat": 0, "dog": 1, "car": 2, "vehicle": 3}
        ref_idx = label_to_idx[ref_label]

        args = [BIN_FILE] + [f"{x:.6f}" for x in vec]
        r = subprocess.run(args, capture_output=True, text=True)
        c_out = int(r.stdout.strip())
        if c_out == ref_idx:
            n_pass += 1
            status = "OK"
        else:
            n_fail += 1
            status = "FAIL"
        print(f"  [{status}] {label:8s} -> ref={ref_label}({ref_idx}) c={c_out}")

    print()
    print(f"=== {n_pass} passed, {n_fail} failed out of {len(test_cases)} ===")
    if n_fail == 0:
        print("[*] VERIFICATION PASSED — C output matches Python reference exactly.")
    else:
        print("[!] VERIFICATION FAILED — outputs diverge.")
        sys.exit(1)

if __name__ == "__main__":
    main()
