#!/usr/bin/env python3
"""Inspect a GPQ8 file: list all tensor headers, verify data_len sums match
file size, and check that embed_tokens is F32 with the right shape."""

import struct, sys, os

def inspect(path):
    sz = os.path.getsize(path)
    print(f"[*] file: {path}  size: {sz/1073741824:.3f} GB ({sz} bytes)")
    with open(path, "rb") as f:
        magic = f.read(4)
        assert magic == b"GPQ8", f"bad magic: {magic!r}"
        n = struct.unpack("<i", f.read(4))[0]
        print(f"[*] {n} tensors")
        total_data = 0
        tensors = []
        for i in range(n):
            klen = struct.unpack("<i", f.read(4))[0]
            key = f.read(klen).decode("utf-8", errors="replace")
            ndim = struct.unpack("<i", f.read(4))[0]
            shape = [struct.unpack("<i", f.read(4))[0] for _ in range(ndim)]
            qtype = f.read(1)[0]
            data_len = struct.unpack("<Q", f.read(8))[0]
            cur = f.tell()
            # skip data
            f.seek(cur + data_len)
            n_scale = struct.unpack("<i", f.read(4))[0]
            if n_scale > 0:
                f.seek(f.tell() + n_scale * 4)
            tensors.append((key, shape, qtype, data_len, n_scale, cur))
            total_data += data_len
        end = f.tell()
        print(f"[*] parse end: {end}  file size: {sz}  diff: {sz - end}")
        print(f"[*] total data_len: {total_data/1073741824:.3f} GB")
        # Check embed_tokens
        print()
        print("=== Tensors of interest ===")
        for k, shape, qt, dl, ns, off in tensors:
            if "embed_tokens" in k or "model.norm" in k or k.endswith(".bias"):
                qtype_str = "F32" if qt == 0 else "Q8"
                expected_f32 = 1
                for d in shape: expected_f32 *= d
                expected_f32 *= 4
                expected_q8  = expected_f32 // 4  # int8 = 1 byte per element
                print(f"  {k}: shape={shape} qtype={qtype_str} data_len={dl} "
                      f"(expected F32={expected_f32}, Q8={expected_q8}) "
                      f"n_scale={ns} offset={off}")
        # First 5 layer tensors
        print()
        print("=== Layer 0 tensors (first 12) ===")
        layer0 = [t for t in tensors if ".0." in t[0]][:12]
        for k, shape, qt, dl, ns, off in layer0:
            qtype_str = "F32" if qt == 0 else "Q8"
            print(f"  {k}: shape={shape} qtype={qtype_str} data_len={dl} n_scale={ns}")

if __name__ == "__main__":
    inspect(sys.argv[1] if len(sys.argv) > 1 else "prebuilt/qwen7b_weights.bin")
