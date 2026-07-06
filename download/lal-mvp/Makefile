# Makefile for LAL MVP — builds all demos for x86_64 and arm64.
#
# Targets:
#   make x86        — build all demos for x86_64 (host)
#   make arm64      — cross-compile all demos for arm64 (requires aarch64-linux-gnu-gcc)
#   make verify     — verify all demos match Python reference
#   make bench      — run compute-only benchmark (specialized vs generic)
#   make clean      — remove build artifacts

CC_X86      ?= gcc
CC_ARM64    ?= aarch64-linux-gnu-gcc
CFLAGS      ?= -O3 -Wall -Wextra
LALC        ?= python3 ../../scripts/lal/lal.py

SRC_DIR     := src
BUILD_DIR   := build

# Each demo: (lal_file, rule_name, c_file, bin_stem)
DEMOS := \
	demo:demo.lal:classify:demo \
	syllogism:syllogism.lal:main:syllogism \
	vsa_ops:vsa_ops.lal:main:vsa_ops

.PHONY: all x86 arm64 verify bench clean

all: x86

# Generate all .c files from .lal files
src/demo.c: src/demo.lal
	$(LALC) src/demo.lal classify src/demo.c
src/syllogism.c: src/syllogism.lal
	$(LALC) src/syllogism.lal main src/syllogism.c
src/vsa_ops.c: src/vsa_ops.lal
	$(LALC) src/vsa_ops.lal main src/vsa_ops.c

# x86_64 targets
x86: build/demo_x86_64 build/syllogism_x86_64 build/vsa_ops_x86_64 build/demo_generic_x86_64

build/demo_x86_64: src/demo.c
	@mkdir -p $(BUILD_DIR)
	$(CC_X86) $(CFLAGS) -o $@ $<
build/syllogism_x86_64: src/syllogism.c
	@mkdir -p $(BUILD_DIR)
	$(CC_X86) $(CFLAGS) -o $@ $<
build/vsa_ops_x86_64: src/vsa_ops.c
	@mkdir -p $(BUILD_DIR)
	$(CC_X86) $(CFLAGS) -o $@ $<
build/demo_generic_x86_64: src/demo_generic.c
	@mkdir -p $(BUILD_DIR)
	$(CC_X86) $(CFLAGS) -o $@ $<

# arm64 targets (cross-compile; can't run without qemu-aarch64-static)
arm64: build/demo_arm64 build/syllogism_arm64 build/vsa_ops_arm64

build/%_arm64: src/%.c
	@mkdir -p $(BUILD_DIR)
	$(CC_ARM64) $(CFLAGS) -o $@ $<

# Verify all demos match Python reference
verify:
	python3 ../../scripts/lal/verify_all.py

# Compute-only benchmark (specialized vs generic, demo only)
bench:
	python3 ../../scripts/lal/benchmark.py

clean:
	rm -rf $(BUILD_DIR) src/demo.c src/syllogism.c src/vsa_ops.c
