#!/usr/bin/env python3
"""Patch Makefile: add -fopenmp and -lgomp to qwen7b-server target."""
import sys

with open("Makefile", "r") as f:
    c = f.read()

old = (
    "prebuilt/qwen7b_server: tools/server/qwen7b_server.c runtime/lal_runtime.c "
    "runtime/lal_runtime.h runtime/lal_q8_kernel.h runtime/lal_sampling.h "
    "runtime/lal_dequant.h runtime/lal_tokenizer.h\n"
    "\t$(CC) $(CFLAGS) -Wno-unused-function -Wno-unused-variable -I. \\\n"
    "\t        -o $@ tools/server/qwen7b_server.c runtime/lal_runtime.c -lm -lpthread\n"
    "\t@echo \"[*] built qwen7b_server (Qwen2.5-7B, Q8, GPQ8)\""
)

new = (
    "prebuilt/qwen7b_server: tools/server/qwen7b_server.c runtime/lal_runtime.c "
    "runtime/lal_runtime.h runtime/lal_q8_kernel.h runtime/lal_sampling.h "
    "runtime/lal_dequant.h runtime/lal_tokenizer.h\n"
    "\t$(CC) $(CFLAGS) -fopenmp -Wno-unused-function -Wno-unused-variable -I. \\\n"
    "\t        -o $@ tools/server/qwen7b_server.c runtime/lal_runtime.c -lm -lpthread -lgomp\n"
    "\t@echo \"[*] built qwen7b_server (Qwen2.5-7B, Q8, GPQ8, OpenMP)\""
)

if old not in c:
    print("ERROR: old pattern not found", file=sys.stderr)
    sys.exit(1)

c = c.replace(old, new)
with open("Makefile", "w") as f:
    f.write(c)
print("OK")
