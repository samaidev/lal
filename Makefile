# LAL — Logic-Assembly Language
# Top-level Makefile

PYTHON ?= python3
CC ?= gcc
CFLAGS ?= -O3 -mavx2 -mfma -Wall
LALC ?= $(PYTHON) compiler/lal.py

.PHONY: all compile demos train infer verify clean

all: demos

# === Compile LAL demos to C ===
compile:
	$(LALC) demos/basic/demo.lal classify build/demo.c
	$(LALC) demos/basic/syllogism.lal main build/syllogism.c
	$(LALC) demos/basic/vsa_ops.lal main build/vsa_ops.c
	$(LALC) demos/embedding/embed_demo.lal classify build/embed_demo.c
	$(LALC) demos/embedding/gpt2_analogy.lal solve build/gpt2_analogy.c --binarize
	$(LALC) demos/embedding/lal_train_demo.lal train build/lal_train.c --train

# === Build C binaries ===
demos: build/demo build/syllogism build/gpt2_train

build/demo: compile
	$(CC) $(CFLAGS) -o $@ build/demo.c -lm

build/syllogism: compile
	$(CC) $(CFLAGS) -o $@ build/syllogism.c -lm

# === GPT-2 training (no PyTorch) ===
train: build/gpt2_train

build/gpt2_train: runtime/gpt2_train.c
	@mkdir -p build
	$(CC) $(CFLAGS) -o $@ runtime/gpt2_train.c -lm

# === GPT-2 inference ===
infer: build/gpt2

build/gpt2: runtime/gpt2_runtime.c
	@mkdir -p build
	$(CC) $(CFLAGS) -o $@ runtime/gpt2_runtime.c -lm

build/gpt2_bin: runtime/gpt2_runtime.c runtime/gpt2_binary.c
	@mkdir -p build
	$(CC) $(CFLAGS) -o $@ runtime/gpt2_runtime.c runtime/gpt2_binary.c -lm -DBINARY

# === Verify ===
verify:
	$(PYTHON) tools/verify.py

# === Clean ===
clean:
	rm -rf build/
