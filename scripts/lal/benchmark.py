#!/usr/bin/env python3
"""
benchmark.py — compare specialized (LAL-compiled) vs generic (llama.cpp-style)
implementations of the same logic.

Measures:
  1. Binary size
  2. Throughput (queries/second)
  3. That outputs match
"""

import os
import sys
import subprocess
import time
import random

BUILD_DIR = "/home/z/my-project/download/lal-mvp/build"
SRC_DIR   = "/home/z/my-project/download/lal-mvp/src"
SPEC_BIN  = os.path.join(BUILD_DIR, "demo_x86_64")
GEN_BIN   = os.path.join(BUILD_DIR, "demo_generic_x86_64")
SPEC_SRC  = os.path.join(SRC_DIR, "demo.c")
GEN_SRC   = os.path.join(SRC_DIR, "demo_generic.c")

def build():
    os.makedirs(BUILD_DIR, exist_ok=True)
    # Build specialized with -O3 (default)
    r = subprocess.run(["gcc", "-O3", "-Wall", "-o", SPEC_BIN, SPEC_SRC],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print("[!] specialized build failed:", r.stderr)
        sys.exit(1)
    # Build generic with -O3
    r = subprocess.run(["gcc", "-O3", "-Wall", "-o", GEN_BIN, GEN_SRC],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print("[!] generic build failed:", r.stderr)
        sys.exit(1)
    print(f"[*] built both binaries")

def size(path):
    return os.path.getsize(path)

def strip_size(path):
    """Build a stripped copy and return its size."""
    stripped = path + ".stripped"
    subprocess.run(["cp", path, stripped], check=True)
    subprocess.run(["strip", stripped], check=True)
    s = os.path.getsize(stripped)
    os.unlink(stripped)
    return s

def run_once(bin_path, vec):
    args = [bin_path] + [f"{x:.6f}" for x in vec]
    r = subprocess.run(args, capture_output=True, text=True)
    return int(r.stdout.strip())

def bench(bin_path, vec, n_calls):
    """Run n_calls invocations and return total wall-clock seconds."""
    args = [bin_path] + [f"{x:.6f}" for x in vec]
    t0 = time.perf_counter()
    for _ in range(n_calls):
        subprocess.run(args, capture_output=True, text=True)
    return time.perf_counter() - t0

def bench_inproc(bin_path, vec, n_calls):
    """Same as bench but reuses one process — measures pure compute time.

    We achieve this by passing the query on stdin and looping in C.
    But since the C programs use argv, we instead just measure many subprocess
    spawns. The pure-compute time difference will be dwarfed by process spawn
    overhead, so we ALSO measure via a custom loop.
    """
    # For a fair compute-only comparison, write a small driver that calls the
    # classify function N times. We can do this by writing a tiny C harness
    # that links against the same .c file.
    pass

def main():
    build()

    random.seed(0)
    test_vecs = [[random.uniform(0, 1) for _ in range(8)] for _ in range(50)]

    # Verify outputs match
    mismatches = 0
    for v in test_vecs:
        s = run_once(SPEC_BIN, v)
        g = run_once(GEN_BIN, v)
        if s != g:
            mismatches += 1
            print(f"  [MISMATCH] vec={v} spec={s} gen={g}")
    if mismatches:
        print(f"[!] {mismatches} mismatches — outputs differ")
        sys.exit(1)
    print(f"[*] outputs match on all {len(test_vecs)} test cases")

    # Binary size
    print()
    print("=== Binary size ===")
    spec_size = size(SPEC_BIN)
    gen_size  = size(GEN_BIN)
    spec_strip = strip_size(SPEC_BIN)
    gen_strip  = strip_size(GEN_BIN)
    print(f"  specialized (with symbols): {spec_size:7d} bytes")
    print(f"  generic     (with symbols): {gen_size:7d} bytes")
    print(f"  specialized (stripped):     {spec_strip:7d} bytes")
    print(f"  generic     (stripped):     {gen_strip:7d} bytes")
    print(f"  ratio (generic/specialized stripped): {gen_strip/spec_strip:.2f}x")

    # Throughput — process spawn overhead dominates, but ratio is still informative.
    print()
    print("=== Throughput (subprocess spawn + compute) ===")
    n_calls = 200
    vec = test_vecs[0]
    spec_t = bench(SPEC_BIN, vec, n_calls)
    gen_t  = bench(GEN_BIN,  vec, n_calls)
    print(f"  specialized: {n_calls} calls in {spec_t:.3f}s = {n_calls/spec_t:.0f} calls/s")
    print(f"  generic:     {n_calls} calls in {gen_t:.3f}s = {n_calls/gen_t:.0f} calls/s")
    print(f"  ratio (generic/specialized): {gen_t/spec_t:.2f}x slower")

    # Compute-only benchmark: write a small driver that calls classify() many times.
    print()
    print("=== Compute-only throughput (in-process loop) ===")
    write_and_run_compute_bench()

def write_and_run_compute_bench():
    """Write two .c drivers that call classify() in a tight loop and time it."""
    spec_driver = """
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
extern void classify(const float* q, int* out_best);
int main() {
    float q[8] = {1.0f, 0.1f, 0.2f, 0.7f, 0.3f, 0.5f, 0.8f, 0.2f};
    int out;
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    long N = 100000000L;
    for (long i = 0; i < N; i++) {
        classify(q, &out);
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double dt = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
    printf("%ld calls in %.4f s = %.0f calls/s (last out=%d)\\n", N, dt, N/dt, out);
    return 0;
}
"""
    gen_driver = """
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
extern void classify_generic(const float* q, int* out_best);
int main() {
    float q[8] = {1.0f, 0.1f, 0.2f, 0.7f, 0.3f, 0.5f, 0.8f, 0.2f};
    int out;
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    long N = 100000000L;
    for (long i = 0; i < N; i++) {
        classify_generic(q, &out);
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double dt = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
    printf("%ld calls in %.4f s = %.0f calls/s (last out=%d)\\n", N, dt, N/dt, out);
    return 0;
}
"""
    spec_drv_path = os.path.join(BUILD_DIR, "bench_spec_drv.c")
    gen_drv_path  = os.path.join(BUILD_DIR, "bench_gen_drv.c")
    with open(spec_drv_path, "w") as f: f.write(spec_driver)
    with open(gen_drv_path, "w") as f: f.write(gen_driver)

    spec_bench_bin = os.path.join(BUILD_DIR, "bench_spec")
    gen_bench_bin  = os.path.join(BUILD_DIR, "bench_gen")

    # Compile source to .o with main renamed away, then link with driver.
    spec_obj = os.path.join(BUILD_DIR, "demo.o")
    gen_obj  = os.path.join(BUILD_DIR, "demo_generic.o")
    subprocess.run(["gcc", "-O3", "-c", "-Dmain=__src_main_unused", "-o", spec_obj, SPEC_SRC], check=True)
    subprocess.run(["gcc", "-O3", "-c", "-Dmain=__src_main_unused", "-o", gen_obj,  GEN_SRC],  check=True)

    r = subprocess.run(
        ["gcc", "-O3", "-o", spec_bench_bin, spec_drv_path, spec_obj, "-lm"],
        capture_output=True, text=True
    )
    if r.returncode != 0:
        print("[!] spec bench build failed:", r.stderr)
        return
    r = subprocess.run(
        ["gcc", "-O3", "-o", gen_bench_bin, gen_drv_path, gen_obj, "-lm"],
        capture_output=True, text=True
    )
    if r.returncode != 0:
        print("[!] gen bench build failed:", r.stderr)
        return

    print(f"  specialized: ", end="", flush=True)
    r = subprocess.run([spec_bench_bin], capture_output=True, text=True)
    print(r.stdout.strip())

    print(f"  generic:     ", end="", flush=True)
    r = subprocess.run([gen_bench_bin], capture_output=True, text=True)
    print(r.stdout.strip())

if __name__ == "__main__":
    main()
