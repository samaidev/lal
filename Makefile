# LAL — Logic-Assembly Language
PYTHON ?= python3
CC ?= gcc

# Auto-detect SIMD flags: AVX2+FMA on x86_64, NEON on ARM, nothing elsewhere.
# The gpt2_server.c file uses a portable v8f wrapper that picks the right
# intrinsics per platform (see top of that file).
UNAME_M := $(shell uname -m)
ifeq ($(UNAME_M),x86_64)
  SIMD_FLAGS ?= -mavx2 -mfma
else ifeq ($(UNAME_M),i386)
  SIMD_FLAGS ?= -msse4.1
else ifneq (,$(filter $(UNAME_M),arm armv7l armv7-a))
  # ARMv7 has NEON; -mfpu=neon is needed on 32-bit
  SIMD_FLAGS ?= -mfpu=neon
else ifneq (,$(filter $(UNAME_M),aarch64 arm64))
  # AArch64 has NEON + FMA by default
  SIMD_FLAGS ?=
else
  SIMD_FLAGS ?=
endif

CFLAGS ?= -O3 $(SIMD_FLAGS) -Wall
LALC ?= $(PYTHON) compiler/lal.py

.PHONY: all train server demos verify clean

all: demos train

# === GPT-2 training (binary mode, no PyTorch) ===
train: prebuilt/gpt2_train

prebuilt/gpt2_train: models/gpt2.c runtime/lal_runtime.c runtime/lal_runtime.h
	$(CC) $(CFLAGS) -o $@ models/gpt2.c runtime/lal_runtime.c -lm

# === GPT-2 web server (float mode, SIMD-accelerated, with browser frontend) ===
# Binary mode (~3 ms/step in training) is too memory-hungry for the 4 GB
# sandbox, so the server uses float mode + SIMD. ~5× faster than the original
# scalar server (96 ms/token vs. 490 ms/token baseline on x86_64).
# Works on x86_64 (AVX2), ARMv7 (NEON), and AArch64 (NEON+FMA).
server: prebuilt/gpt2_server

prebuilt/gpt2_server: tools/server/gpt2_server.c tools/server/frontend.html \
	              runtime/lal_runtime.c runtime/lal_runtime.h
	$(CC) $(CFLAGS) -Wno-unused-function -Wno-unused-variable -I. \
	      -o $@ tools/server/gpt2_server.c runtime/lal_runtime.c -lm -lpthread

# === Compile LAL demos ===
demos: prebuilt/demos/demo

prebuilt/demos/demo: demos/basic/demo.lal compiler/lal.py
	@mkdir -p prebuilt/demos
	$(LALC) demos/basic/demo.lal classify prebuilt/demos/demo.c
	$(CC) $(CFLAGS) -o $@ prebuilt/demos/demo.c -lm

# === Verify ===
verify:
	$(PYTHON) tools/verify.py

clean:
	rm -rf build/ prebuilt/demos/*.c prebuilt/gpt2_server
