"""
verify_syllogism.py — verify the syllogism demo's C output matches the Python reference.
"""
import os, sys, subprocess
sys.path.insert(0, os.path.dirname(__file__))
from lal import parse, run_reference

LAL_FILE = "/home/z/my-project/scripts/lal/syllogism.lal"
SRC_DIR  = "/home/z/my-project/download/lal-mvp/src"
BUILD_DIR= "/home/z/my-project/download/lal-mvp/build"
C_FILE   = os.path.join(SRC_DIR, "syllogism.c")
BIN_FILE = os.path.join(BUILD_DIR, "syllogism_x86_64")

def main():
    with open(LAL_FILE) as f:
        source = f.read()
    concepts, bounds, memories, relates, rules = parse(source)
    concept_map = {c.name: c.vec for c in concepts}

    os.makedirs(BUILD_DIR, exist_ok=True)
    r = subprocess.run(["gcc", "-O3", "-o", BIN_FILE, C_FILE], capture_output=True, text=True)
    if r.returncode != 0:
        print("[!] gcc failed:", r.stderr); sys.exit(1)
    print(f"[*] built {BIN_FILE}")

    # Test queries: each subject concept + some random vectors
    test_cases = []
    for name in ["socrates", "plato", "aristotle", "stone", "human", "mortal"]:
        test_cases.append((name, concept_map[name]))
    import random
    random.seed(7)
    for _ in range(10):
        test_cases.append(("random", [random.uniform(0, 1) for _ in range(8)]))

    n_pass = n_fail = 0
    for label, vec in test_cases:
        env = run_reference(concepts, bounds, memories, relates, rules, "main", vec)
        ref_label = env["best"]
        label_to_idx = {"mortal": 0, "human": 1, "stone": 2, "socrates": 3}
        ref_idx = label_to_idx[ref_label]
        args = [BIN_FILE] + [f"{x:.6f}" for x in vec]
        r = subprocess.run(args, capture_output=True, text=True)
        c_out = int(r.stdout.strip())
        status = "OK" if c_out == ref_idx else "FAIL"
        if status == "OK": n_pass += 1
        else: n_fail += 1
        print(f"  [{status}] {label:10s} -> ref={ref_label}({ref_idx}) c={c_out}")

    print()
    print(f"=== {n_pass} passed, {n_fail} failed out of {len(test_cases)} ===")
    if n_fail == 0:
        print("[*] VERIFICATION PASSED — C output matches Python reference exactly.")
    else:
        print("[!] VERIFICATION FAILED."); sys.exit(1)

if __name__ == "__main__":
    main()
