#!/usr/bin/env python3
"""Benchmark: serial CSR vs parallel SIMD linear layers (GPT-2 MLP).

Runs the GPT-2 MLP forward pass N times via each binary and reports throughput.
The parallel version uses AVX2 gather+FMA to compute 8 outputs per instruction.
"""
import subprocess
import time
import sys
import os

SERIAL_BIN = "/home/z/my-project/download/lal-mvp/build/gpt2_mlp_serial"
PARALLEL_BIN = "/home/z/my-project/download/lal-mvp/build/gpt2_mlp_parallel"

# Read the king embedding to use as input
sys.path.insert(0, "/home/z/my-project/scripts/lal")
from lal import parse
with open("/home/z/my-project/scripts/lal/gpt2_mlp.lal") as f:
    src = f.read()
concepts, _, _, _, _ = parse(src, "gpt2_mlp.lal")
king_vec = concepts[0].vec  # king
args = [f"{x:.6f}" for x in king_vec]

def bench(bin_path, n_calls):
    t0 = time.perf_counter()
    for _ in range(n_calls):
        subprocess.run([bin_path] + args, capture_output=True, text=True)
    return time.perf_counter() - t0

N = 50
print(f"Benchmarking GPT-2 MLP forward pass ({N} calls each)...")
print()

# Serial
t_serial = bench(SERIAL_BIN, N)
print(f"Serial CSR:      {N} calls in {t_serial:.3f}s = {N/t_serial:.1f} calls/s")

# Parallel
t_parallel = bench(PARALLEL_BIN, N)
print(f"Parallel SIMD:   {N} calls in {t_parallel:.3f}s = {N/t_parallel:.1f} calls/s")

print()
print(f"Speedup: {t_serial/t_parallel:.2f}x (parallel / serial)")
print()
print("Note: subprocess spawn overhead dominates. For a fair compute-only")
print("comparison, see the in-process benchmark below.")

# In-process benchmark: call the C function directly via a driver
print()
print("=== In-process compute-only benchmark (100K calls) ===")

import tempfile, os
driver = """
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
extern void rule_mlp_forward(const float* q, int* out);
int main() {
    float q[768] = {{VALS}};
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
vals_str = ", ".join(f"{x:.6f}f" for x in king_vec)
driver = driver.replace("{VALS}", vals_str)

for label, src_c, bin_name in [
    ("Serial CSR", "/home/z/my-project/download/lal-mvp/src/gpt2_mlp_serial.c", "bench_serial"),
    ("Parallel SIMD", "/home/z/my-project/download/lal-mvp/src/gpt2_mlp_parallel.c", "bench_parallel"),
]:
    drv_path = f"/tmp/drv_{bin_name}.c"
    with open(drv_path, "w") as f:
        f.write(driver)
    bin_path = f"/home/z/my-project/download/lal-mvp/build/{bin_name}"
    r = subprocess.run(
        ["gcc", "-O3", "-mavx2", "-mfma", "-Dmain=__orig_main",
         "-o", bin_path, drv_path, src_c, "-lm"],
        capture_output=True, text=True
    )
    if r.returncode != 0:
        # Compile source with main renamed away, then link with driver
        subprocess.run(["gcc", "-O3", "-mavx2", "-mfma", "-Dmain=__src_main", "-c", "-o", f"/tmp/{bin_name}.o", src_c], check=True)
        subprocess.run(["gcc", "-O3", "-mavx2", "-mfma", "-o", bin_path, drv_path, f"/tmp/{bin_name}.o", "-lm"], check=True)
    r = subprocess.run([bin_path], capture_output=True, text=True)
    print(f"  {label:15s}: {r.stdout.strip()}")
