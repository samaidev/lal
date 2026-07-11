#!/usr/bin/env python3
"""Patch Makefile: add -mf16c to SIMD_FLAGS for x86_64."""
import sys

with open("Makefile", "r") as f:
    c = f.read()

old = "  SIMD_FLAGS ?= -mavx2 -mfma"
new = "  SIMD_FLAGS ?= -mavx2 -mfma -mf16c"

if old not in c:
    print("ERROR: old pattern not found", file=sys.stderr)
    sys.exit(1)

c = c.replace(old, new)
with open("Makefile", "w") as f:
    f.write(c)
print("OK")
