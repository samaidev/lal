"""
verify_all.py — verify all LAL demos match their Python references.
"""
import os, sys, subprocess
sys.path.insert(0, os.path.dirname(__file__))
from lal import parse, run_reference

DEMOS = [
    {
        "name": "demo",
        "lal": "/home/z/my-project/scripts/lal/demo.lal",
        "rule": "classify",
        "c":   "/home/z/my-project/download/lal-mvp/src/demo.c",
        "bin": "/home/z/my-project/download/lal-mvp/build/demo_x86_64",
        "labels": ["cat", "dog", "car", "vehicle"],
        "test_concepts": ["cat", "dog", "car", "vehicle"],
        "n_random": 16,
    },
    {
        "name": "syllogism",
        "lal": "/home/z/my-project/scripts/lal/syllogism.lal",
        "rule": "main",
        "c":   "/home/z/my-project/download/lal-mvp/src/syllogism.c",
        "bin": "/home/z/my-project/download/lal-mvp/build/syllogism_x86_64",
        "labels": ["mortal", "human", "stone", "socrates"],
        "test_concepts": ["socrates", "plato", "aristotle", "stone", "human", "mortal"],
        "n_random": 10,
    },
    {
        "name": "vsa_ops",
        "lal": "/home/z/my-project/scripts/lal/vsa_ops.lal",
        "rule": "main",
        "c":   "/home/z/my-project/download/lal-mvp/src/vsa_ops.c",
        "bin": "/home/z/my-project/download/lal-mvp/build/vsa_ops_x86_64",
        "labels": ["alice", "bob", "carol"],
        "test_concepts": ["alice", "bob", "carol"],
        "n_random": 10,
    },
    {
        "name": "embed_demo",
        "lal": "/home/z/my-project/scripts/lal/embed_demo.lal",
        "rule": "classify",
        "c":   "/home/z/my-project/download/lal-mvp/src/embed_demo.c",
        "bin": "/home/z/my-project/download/lal-mvp/build/embed_demo_x86_64",
        "labels": ["cat", "dog", "car", "vehicle"],
        "test_concepts": ["cat", "dog", "car", "vehicle"],
        "n_random": 10,
    },
    {
        "name": "recursion_demo",
        "lal": "/home/z/my-project/scripts/lal/recursion_demo.lal",
        "rule": "main",
        "c":   "/home/z/my-project/download/lal-mvp/src/recursion_demo.c",
        "bin": "/home/z/my-project/download/lal-mvp/build/recursion_demo_x86_64",
        "labels": ["target", "base"],
        "test_concepts": ["base", "target"],
        "n_random": 10,
    },
]

def main():
    import random
    total_pass = total_fail = 0
    for demo in DEMOS:
        print(f"\n=== Demo: {demo['name']} ===")
        with open(demo["lal"]) as f:
            src = f.read()
        concepts, bounds, memories, relates, rules = parse(src, demo["lal"])
        cm = {c.name: c.vec for c in concepts}
        # Build
        os.makedirs(os.path.dirname(demo["bin"]), exist_ok=True)
        r = subprocess.run(["gcc", "-O3", "-o", demo["bin"], demo["c"]], capture_output=True, text=True)
        if r.returncode != 0:
            print(f"[!] build failed: {r.stderr}"); sys.exit(1)
        # Test
        n_pass = n_fail = 0
        for cname in demo["test_concepts"]:
            env = run_reference(concepts, bounds, memories, relates, rules, demo["rule"], cm[cname])
            ref_label = env.get("best")
            ref_idx = demo["labels"].index(ref_label) if ref_label in demo["labels"] else -1
            args = [demo["bin"]] + [f"{x:.6f}" for x in cm[cname]]
            c_out = int(subprocess.run(args, capture_output=True, text=True).stdout.strip())
            ok = c_out == ref_idx
            n_pass += ok; n_fail += (not ok)
            print(f"  [{'OK' if ok else 'FAIL'}] {cname:10s} -> ref={ref_label}({ref_idx}) c={c_out}")
        random.seed(42)
        for _ in range(demo["n_random"]):
            v = [random.uniform(0, 1) for _ in range(8)]
            env = run_reference(concepts, bounds, memories, relates, rules, demo["rule"], v)
            ref_label = env.get("best")
            ref_idx = demo["labels"].index(ref_label) if ref_label in demo["labels"] else -1
            args = [demo["bin"]] + [f"{x:.6f}" for x in v]
            c_out = int(subprocess.run(args, capture_output=True, text=True).stdout.strip())
            ok = c_out == ref_idx
            n_pass += ok; n_fail += (not ok)
        print(f"  → {n_pass} passed, {n_fail} failed (incl {demo['n_random']} random)")
        total_pass += n_pass; total_fail += n_fail
    print(f"\n=== TOTAL: {total_pass} passed, {total_fail} failed ===")
    if total_fail == 0:
        print("[*] ALL DEMOS VERIFIED — C output matches Python reference.")
    else:
        print("[!] VERIFICATION FAILED."); sys.exit(1)

if __name__ == "__main__":
    main()
