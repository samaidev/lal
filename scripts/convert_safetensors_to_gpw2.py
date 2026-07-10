#!/usr/bin/env python3
"""Convert HuggingFace GPT-2 safetensors to LAL GPW2 format.

The official LAL release ships gpt2_weights.bin (474 MB, GPW2 format) via
GitHub Releases. When github.com is unreachable but a HuggingFace mirror
(e.g. hf-mirror.com) is reachable, download the safetensors file from there
and convert it with this script.

GPW2 format (defined in tools/export_gpt2_weights.py):
  [4 bytes]  magic = "GPW2"
  [4 bytes]  n_tensors (uint32 LE)
  for each tensor:
    [4 bytes]  key_len (uint32 LE)
    [N bytes]  key (UTF-8)
    [4 bytes]  ndim (uint32 LE)
    [ndim * 4 bytes]  shape (uint32 LE each)
    [prod(shape) * 4 bytes]  data (float32, row-major)

The safetensors file contains 160 tensors; this script drops the 12
h.N.attn.bias buffers (causal masks, [1,1,1024,1024] each) because LAL
recomputes attention masks at runtime. The remaining 148 tensors match
the official GPW2 release byte-for-byte.

Usage:
  python3 convert_safetensors_to_gpw2.py <input.safetensors> <output.bin>

Example:
  aria2c -x 16 -s 16 -d /tmp -o gpt2.safetensors \\
    https://hf-mirror.com/openai-community/gpt2/resolve/main/model.safetensors
  python3 convert_safetensors_to_gpw2.py /tmp/gpt2.safetensors prebuilt/gpt2_weights.bin
"""
import struct
import json
import os
import sys


def convert(src_path: str, dst_path: str) -> None:
    with open(src_path, 'rb') as f:
        hlen = struct.unpack('<Q', f.read(8))[0]
        header = json.loads(f.read(hlen).decode('utf-8'))
        data_start = 8 + hlen

        # Filter: drop __metadata__ and the attn.bias / attn.masked_bias
        # buffers (causal masks LAL recomputes at runtime). Keep everything
        # else (weights + biases for c_attn, c_proj, c_fc, ln_1, ln_2, etc.).
        # NOTE: use a leading dot in the suffix check to avoid accidentally
        # matching c_attn.bias / c_proj.bias, which are real weight tensors.
        tensors = []
        for key, value in header.items():
            if key == '__metadata__':
                continue
            if key.endswith('.attn.bias') or key.endswith('.attn.masked_bias'):
                continue
            if value.get('dtype') != 'F32':
                raise ValueError(f"{key}: unexpected dtype {value.get('dtype')!r}")
            tensors.append((key, value))

        print(f'[*] converting {len(tensors)} tensors '
              f'(dropped {len(header) - len(tensors) - 1} buffers)')

        with open(dst_path, 'wb') as out:
            out.write(b'GPW2')
            out.write(struct.pack('<I', len(tensors)))
            total_bytes = 0
            for key, value in tensors:
                key_bytes = key.encode('utf-8')
                shape = value['shape']
                offsets = value['data_offsets']
                nbytes = offsets[1] - offsets[0]

                out.write(struct.pack('<I', len(key_bytes)))
                out.write(key_bytes)
                out.write(struct.pack('<I', len(shape)))
                for dim in shape:
                    out.write(struct.pack('<I', dim))

                f.seek(data_start + offsets[0])
                data = f.read(nbytes)
                if len(data) != nbytes:
                    raise IOError(f"{key}: short read {len(data)} / {nbytes}")
                out.write(data)
                total_bytes += nbytes

    size = os.path.getsize(dst_path)
    print(f'[*] wrote {dst_path}: {size:,} bytes ({size / 1024 / 1024:.1f} MB)')
    print(f'[*] total tensor data: {total_bytes:,} bytes')

    if size == 497_763_872:
        print('[*] size matches official GPW2 release')
    else:
        print(f'[!] size differs from official release (497,763,872 bytes)')


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 1
    convert(sys.argv[1], sys.argv[2])
    return 0


if __name__ == '__main__':
    sys.exit(main())
