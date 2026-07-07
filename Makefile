# LAL — Logic-Assembly Language
PYTHON ?= python3
CC ?= gcc
CFLAGS ?= -O3 -mavx2 -mfma -Wall
LALC ?= $(PYTHON) compiler/lal.py

.PHONY: all train server demos verify clean

all: demos train

# === GPT-2 training (binary mode, no PyTorch) ===
train: prebuilt/gpt2_train

prebuilt/gpt2_train: models/gpt2.c runtime/lal_runtime.c runtime/lal_runtime.h
	$(CC) $(CFLAGS) -o $@ models/gpt2.c runtime/lal_runtime.c -lm

# === GPT-2 web server (float mode, AVX2 SIMD, with browser frontend) ===
# Binary mode (~3 ms/step in training) is too memory-hungry for the 4 GB
# sandbox, so the server uses float mode + AVX2 SIMD. ~5× faster than the
# original scalar server (96 ms/token vs. 490 ms/token baseline).
server: prebuilt/gpt2_server

prebuilt/gpt2_server: tools/server/gpt2_server.c tools/server/frontend.html \
                      runtime/lal_runtime.c runtime/lal_runtime.h
	$(CC) $(CFLAGS) -Wno-unused-function -Wno-unused-variable \
	      -Wno-format-overflow -Wno-restrict -I. \
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
