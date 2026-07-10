#!/usr/bin/env python3
"""Inspect GPQ8 file: list non-layer tensors and layer-0 tensors."""
import struct, sys

def main(path):
    f = open(path, "rb")
    assert f.read(4) == b"GPQ8"
    n = struct.unpack("<i", f.read(4))[0]
    print(f"Total tensors: {n}")
    tensors = []
    for i in range(n):
        klen = struct.unpack("<i", f.read(4))[0]
        key = f.read(klen).decode("utf-8", "replace")
        ndim = struct.unpack("<i", f.read(4))[0]
        shape = [struct.unpack("<i", f.read(4))[0] for _ in range(ndim)]
        qtype = f.read(1)[0]
        dl = struct.unpack("<Q", f.read(8))[0]
        f.seek(f.tell() + dl)
        ns = struct.unpack("<i", f.read(4))[0]
        f.seek(f.tell() + ns * 4)
        tensors.append((key, shape, "F32" if qtype == 0 else "Q8", dl, ns))

    print("\nNon-layer tensors:")
    for k, s, q, d, ns in tensors:
        if "layers." not in k:
            print(f"  {k}: shape={s} {q} dl={d} n_scale={ns}")

    print("\nLayer 0 tensors:")
    for k, s, q, d, ns in tensors:
        if "layers.0." in k:
            print(f"  {k}: shape={s} {q} dl={d} n_scale={ns}")

    # Also check if lm_head.weight exists
    print("\nlm_head search:")
    for k, s, q, d, ns in tensors:
        if "lm_head" in k or "head" in k.lower():
            print(f"  FOUND: {k}: shape={s} {q} dl={d}")

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "prebuilt/qwen7b_weights.bin")
