# LAL — Logic-Assembly Language
PYTHON ?= python3
CC ?= gcc

# Auto-detect SIMD flags: AVX2+FMA on x86_64, NEON on ARM, nothing elsewhere.
# The gpt2_server.c file uses a portable v8f wrapper that picks the right
# intrinsics per platform (see top of that file).
UNAME_M := $(shell uname -m)
ifeq ($(UNAME_M),x86_64)
  # Try -march=native first (enables all CPU-specific tuning), fall back to
  # generic AVX2 if native is not supported (cross-compilation, old gcc)
  NATIVE_OK := $(shell echo 'int main(){return 0;}' | $(CC) -march=native -x c - -o /dev/null 2>/dev/null && echo yes)
  ifeq ($(NATIVE_OK),yes)
    SIMD_FLAGS ?= -march=native
  else
    SIMD_FLAGS ?= -mavx2 -mfma -mf16c
  endif
else ifeq ($(UNAME_M),i386)
  SIMD_FLAGS ?= -msse4.1
else ifneq (,$(filter $(UNAME_M),arm armv7l armv7-a))
  # ARMv7 has NEON; -mfpu=neon is needed on 32-bit
  SIMD_FLAGS ?= -march=armv7-a -mfpu=neon -mfloat-abi=softfp
else ifneq (,$(filter $(UNAME_M),aarch64 arm64))
  # AArch64 has NEON + FMA by default
  SIMD_FLAGS ?=
else
  SIMD_FLAGS ?=
endif

CFLAGS ?= -O3 $(SIMD_FLAGS) -Wall -ldl
LALC ?= $(PYTHON) compiler/lal.py

.PHONY: all train server qwen-server float-subset demos verify clean

all: demos train

# === GPT-2 training (binary mode, no PyTorch) ===
train: prebuilt/gpt2_train

prebuilt/gpt2_train: models/gpt2.c runtime/lal_runtime.c runtime/lal_runtime.h
	$(CC) $(CFLAGS) -o $@ models/gpt2.c runtime/lal_runtime.c -lm

# === OpenBLAS auto-detection for the float server ===
# gpt2_server.c auto-enables BLAS matmul (cblas_sgemv) when <cblas.h> is
# present at compile time, via __has_include. We auto-link -lopenblas here so
# `make server` uses it out of the box on systems with libopenblas-dev.
# On systems without it, BLAS_LIBS is empty and the server silently falls back
# to hand-written SIMD. Override: make server BLAS_LIBS=-lopenblas (or = to disable)
BLAS_LIBS := $(shell pkg-config --libs openblas 2>/dev/null)
ifeq ($(BLAS_LIBS),)
  # Fallback: detect -lopenblas by probing whether <cblas.h> preprocesses.
	BLAS_LIBS := $(shell echo '#include <cblas.h>' | $(CC) -E -x c - >/dev/null 2>&1 && echo -lopenblas)
endif

# === GPT-2 web server (float mode, auto OpenBLAS, with browser frontend) ===
# Float mode + (OpenBLAS if available, else hand-written SIMD). ~5× faster
# than the original scalar server (96 ms/token vs. 490 ms/token baseline on
# x86_64). Works on x86_64 (AVX2), ARMv7 (NEON), and AArch64 (NEON+FMA).
server: prebuilt/gpt2_server

prebuilt/gpt2_server: tools/server/gpt2_server.c tools/server/frontend.html \
	runtime/lal_runtime.c runtime/lal_runtime.h
	$(CC) $(CFLAGS) -Wno-unused-function -Wno-unused-variable -I. \
	-o $@ tools/server/gpt2_server.c runtime/lal_runtime.c -lm -lpthread $(BLAS_LIBS)
	@if [ -n "$(BLAS_LIBS)" ]; then \
	        echo "[*] built with OpenBLAS acceleration ($(BLAS_LIBS))"; \
	else \
	        echo "[*] built with hand-written SIMD (install libopenblas-dev for ~2-3x speedup)"; \
	fi

# server-blas: legacy alias. OpenBLAS is now auto-detected by `make server`.
# Kept for backward compatibility with existing docs and scripts.
server-blas: server

# === Float subset extractor for --mixed-precision on memory-constrained devices ===
# Extracts only layers 0 and 11 (24 tensors, ~54 MB) from the full 498 MB
# gpt2_weights.bin, in the same GPW2 format, so the tablet can run
# --mixed-precision without downloading the full float file.
#   LAL_FLOAT_SUBSET=prebuilt/gpt2_float_subset.bin ./gpt2_server --mixed-precision
float-subset: scripts/extract_float_subset prebuilt/gpt2_float_subset.bin

scripts/extract_float_subset: scripts/extract_float_subset.c
	$(CC) -O2 -o $@ $<

prebuilt/gpt2_float_subset.bin: scripts/extract_float_subset prebuilt/gpt2_weights.bin
	./scripts/extract_float_subset prebuilt/gpt2_weights.bin $@

# === Compile LAL demos ===
demos: prebuilt/demos/demo

prebuilt/demos/demo: demos/basic/demo.lal compiler/lal.py
	@mkdir -p prebuilt/demos
	$(LALC) demos/basic/demo.lal classify prebuilt/demos/demo.c
	$(CC) $(CFLAGS) -o $@ prebuilt/demos/demo.c -lm

# === Verify ===
verify: verify-steer verify-skip

verify-steer: build/verify_steer
	./build/verify_steer

build/verify_steer: tests/verify_steer.c compiler/lal.py
	$(CC) $(CFLAGS) -I. -o $@ tests/verify_steer.c -lm

verify-skip: build/verify_skip
	./build/verify_skip

build/verify_skip: tests/verify_skip.c
	$(CC) $(CFLAGS) -I. -o $@ tests/verify_skip.c -ldl

# === End-to-end: real generation through the mini server (.lal .so hot-loaded) ===
e2e: prebuilt/mini_server prebuilt/mini_steer.so prebuilt/mini_steer_neg.so prebuilt/mini_skip.so
	bash scripts/e2e_test.sh

clean:
	rm -rf build/ prebuilt/demos/*.c prebuilt/gpt2_server prebuilt/qwen_server

# === Qwen2.5-0.5B inference server (Q8 quantization, GQA, RoPE, SwiGLU) ===
qwen-server: prebuilt/qwen_server

prebuilt/qwen_server: tools/server/qwen_server.c runtime/lal_runtime.c runtime/lal_runtime.h
	$(CC) $(CFLAGS) -Wno-unused-function -Wno-unused-variable -I. \
	        -o $@ tools/server/qwen_server.c runtime/lal_runtime.c -lm -lpthread
	@echo "[*] built qwen_server (Qwen2.5-0.5B, Q8 default)"

# === Qwen2.5-7B inference server (Q8, GQA, GPQ8 pre-quantized) ===
qwen7b-server: prebuilt/qwen7b_server

prebuilt/qwen7b_server: tools/server/qwen7b_server.c runtime/lal_runtime.c runtime/lal_runtime.h runtime/lal_q8_kernel.h runtime/lal_sampling.h runtime/lal_dequant.h runtime/lal_tokenizer.h
	$(CC) $(CFLAGS) -fopenmp -Wno-unused-function -Wno-unused-variable -I. \
	        -o $@ tools/server/qwen7b_server.c runtime/lal_runtime.c -lm -lpthread -lgomp
	@echo "[*] built qwen7b_server (Qwen2.5-7B, Q8, GPQ8, OpenMP)"

# === Mini local transformer (real model, no internet) for LAL level-1 live test ===
# Trains a tiny char-LM, exports C-loadable weights + a REAL steering direction.
mini-train: prebuilt/mini_model.bin

prebuilt/mini_model.bin: tools/mini_train.py
	$(PYTHON) tools/mini_train.py

prebuilt/mini_server: tools/mini_server.c
	$(CC) $(CFLAGS) -Wno-unused-function -Wno-unused-variable -I. \
	        -o $@ tools/mini_server.c -lm -ldl
	@echo "[*] built mini_server (tiny real transformer, level-1 hook enabled)"

# Build the real-direction steering .so for mini_server (and qwen_server).
mini-steer: prebuilt/mini_steer.so

prebuilt/mini_steer.so: demos/mini_steer.lal compiler/lal.py
	bash scripts/build_lal_mini_steer.sh

# Build the logic-driven layer-skip .so (level-1 acceleration: early-exit / skip).
mini-skip: prebuilt/mini_skip.so

prebuilt/mini_skip.so: demos/mini_skip.lal compiler/lal.py
	bash scripts/build_lal_skip.sh
