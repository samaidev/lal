# LAL — Logic-Assembly Language
PYTHON ?= python3
CC ?= gcc
CFLAGS ?= -O3 -mavx2 -mfma -Wall
LALC ?= $(PYTHON) compiler/lal.py

.PHONY: all train infer demos verify clean

all: demos train

# === GPT-2 training (no PyTorch) ===
train: prebuilt/gpt2_train

prebuilt/gpt2_train: models/gpt2.c runtime/lal_runtime.c runtime/lal_runtime.h
	$(CC) $(CFLAGS) -o $@ models/gpt2.c runtime/lal_runtime.c -lm

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
	rm -rf build/ prebuilt/demos/*.c
