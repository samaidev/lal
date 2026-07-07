#!/usr/bin/env python3
"""Benchmark: serial CSR (float) vs binary XNOR+popcount (1-bit weights)."""
import subprocess, os, sys
sys.path.insert(0, "/home/z/my-project/scripts/lal")
from lal import parse
import lal
lal._BINARY_MODE = False
with open("/home/z/my-project/scripts/lal/gpt2_mlp.lal") as f:
    src = f.read()
concepts, _, _, _, _ = parse(src, "gpt2_mlp.lal")
king_vec = concepts[0].vec
vals_str = ", ".join(f"{x:.6f}f" for x in king_vec)

DRIVER = """#include <stdio.h>
#include <stdlib.h>
#include <time.h>
extern void rule_mlp_forward(const float* q, int* out);
int main() {
    float q[768] = {""" + vals_str + """};
    int out;
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    long N = 10000;
    for (long i = 0; i < N; i++) rule_mlp_forward(q, &out);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double dt = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec)*1e-9;
    printf("%ld calls in %.4f s = %.0f calls/s (out=%d)\\n", N, dt, N/dt, out);
    return 0;
}
"""

with open("/tmp/drv_bench.c", "w") as f:
    f.write(DRIVER)

print("=== GPT-2 MLP benchmark (10K in-process calls) ===")
for label, src_c, bin_name in [
    ("Serial CSR (float32)", "/home/z/my-project/download/lal-mvp/src/gpt2_mlp_serial.c", "bench_s"),
    ("Binary XNOR+popcount", "/home/z/my-project/download/lal-mvp/src/gpt2_mlp_binary.c", "bench_b"),
]:
    subprocess.run(["gcc", "-O3", "-mavx2", "-mfma", "-Dmain=__src_main",
                    "-c", "-o", f"/tmp/{bin_name}.o", src_c], check=True)
    subprocess.run(["gcc", "-O3", "-mavx2", "-mfma", "-o",
                    f"/home/z/my-project/download/lal-mvp/build/{bin_name}",
                    "/tmp/drv_bench.c", f"/tmp/{bin_name}.o", "-lm"], check=True)
    r = subprocess.run([f"/home/z/my-project/download/lal-mvp/build/{bin_name}"],
                       capture_output=True, text=True)
    print(f"  {label:25s}: {r.stdout.strip()}")

print()
print("=== Binary sizes ===")
for name in ["serial", "binary"]:
    p = f"/home/z/my-project/download/lal-mvp/build/gpt2_mlp_{name}"
    print(f"  {name:10s}: {os.path.getsize(p)/1e6:.2f} MB")

print()
print("=== Weight data sizes ===")
for name in ["serial", "binary"]:
    p = f"/home/z/my-project/download/lal-mvp/src/gpt2_mlp_{name}.c"
    print(f"  {name:10s}: {os.path.getsize(p)/1e6:.2f} MB (C source)")
