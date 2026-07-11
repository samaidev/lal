"""Convert Qwen2.5-7B-Instruct safetensors (BF16, 4 shards) to LAL Q8_0 format.

Q8_0 block format (matches llama.cpp ggml):
  - Block size: 32 elements
  - Per block: 1 fp16 scale (2 bytes) + 32 bytes (32 x int8 values)
  - Total: 34 bytes per 32 elements = 1.0625 bytes/elem
  - Scale is INLINE with data (no separate scale array)

Advantages over per-row Q8 (qtype=1):
  - Per-block scale (not per-row) → better accuracy at same bitrate
  - Inline scale → better cache locality during matmul
  - Sequential access pattern → HW prefetcher works efficiently

File format: "GPQ8" magic + n_tensors + per-tensor:
  [key_len][key][ndim][shape][qtype(0=F32,1=Q8,2=Q4_0,3=Q8_0)][data_len][data][scale_len][scale]

For Q8_0:
  - data = packed blocks (out_dim * in_dim / 32 blocks, 34 bytes each)
  - scale_len = 0 (scale is inside each block)
"""
import struct, json, os, sys, glob
import numpy as np

SHARD_DIR = '/root/qwen7b'
OUT_PATH = '/root/lal/prebuilt/qwen7b_q8_0_weights.bin'

def bf16_to_f32(u16_bytes):
    u16 = np.frombuffer(u16_bytes, dtype=np.uint16)
    return (u16.astype(np.uint32) << 16).view(np.float32)

def quantize_q8_0_per_row(f32_data, in_dim, out_dim):
    """Quantize [out_dim, in_dim] float32 to Q8_0 packed format.

    Returns bytes of packed blocks. Each block = 2 bytes fp16 scale + 32 bytes int8.
    """
    w = np.asarray(f32_data, dtype=np.float32).reshape(out_dim, in_dim)
    assert in_dim % 32 == 0, f"in_dim {in_dim} not multiple of 32"
    n_blocks_per_row = in_dim // 32

    # Reshape to blocks: [out_dim, n_blocks_per_row, 32]
    blocks = w.reshape(out_dim, n_blocks_per_row, 32)
    block_max = np.max(np.abs(blocks), axis=2)  # [out_dim, n_blocks_per_row]

    # Scale = max / 127
    scale = block_max / 127.0
    scale = np.where(np.abs(scale) < 1e-8, 1e-8, scale).astype(np.float32)

    # Quantize: q = round(x / scale), clip to [-127, 127]
    inv_scale = 1.0 / scale
    q = np.round(blocks * inv_scale[:, :, None]).clip(-127, 127).astype(np.int8)

    # Build output: for each block, 2 bytes scale (fp16) + 32 bytes int8
    scale_fp16 = scale.astype(np.float16)  # [out_dim, n_blocks_per_row]

    out = np.zeros((out_dim, n_blocks_per_row, 34), dtype=np.uint8)
    out[:, :, :2] = np.frombuffer(scale_fp16.tobytes(), dtype=np.uint8).reshape(
        out_dim, n_blocks_per_row, 2)
    out[:, :, 2:] = q.view(np.uint8).reshape(out_dim, n_blocks_per_row, 32)

    return out.tobytes()

WEIGHT_SUFFIXES = [
    'self_attn.q_proj.weight',
    'self_attn.k_proj.weight',
    'self_attn.v_proj.weight',
    'self_attn.o_proj.weight',
    'mlp.gate_proj.weight',
    'mlp.up_proj.weight',
    'mlp.down_proj.weight',
]

KEEP_FLOAT_SUFFIXES = [
    'input_layernorm.weight',
    'post_attention_layernorm.weight',
]

EMBED_KEY = 'model.embed_tokens.weight'
NORM_KEY = 'model.norm.weight'
LM_HEAD_KEY = 'lm_head.weight'

def is_weight(key):
    return any(key.endswith(s) for s in WEIGHT_SUFFIXES)

def is_norm(key):
    return any(key.endswith(s) for s in KEEP_FLOAT_SUFFIXES) or key == NORM_KEY

def is_bias(key):
    return key.endswith('.bias')

def is_skip(key):
    return 'attn.bias' in key or 'masked_bias' in key

def main():
    shards = sorted(glob.glob(os.path.join(SHARD_DIR, 'model-*.safetensors')))
    print(f'Found {len(shards)} shards')

    all_tensors = []
    for shard_path in shards:
        with open(shard_path, 'rb') as f:
            hlen = struct.unpack('<Q', f.read(8))[0]
            hdr = json.loads(f.read(hlen).decode('utf-8'))
            for k, v in hdr.items():
                if k == '__metadata__': continue
                if is_skip(k): continue
                all_tensors.append((k, v, shard_path))
        print(f'  {os.path.basename(shard_path)}: {len(hdr)-1} tensors')

    print(f'Total: {len(all_tensors)} tensors')
    n_weight = sum(1 for k,_,_ in all_tensors if is_weight(k))
    n_float = sum(1 for k,_,_ in all_tensors if is_norm(k) or is_bias(k) or k == EMBED_KEY or k == LM_HEAD_KEY)
    print(f'  Q8_0 weights: {n_weight}, float32: {n_float}')

    shard_meta = {}
    for p in shards:
        f = open(p, 'rb')
        hlen = struct.unpack('<Q', f.read(8))[0]
        hdr = json.loads(f.read(hlen).decode('utf-8'))
        data_start = 8 + hlen
        shard_meta[p] = (f, data_start, hdr)

    with open(OUT_PATH, 'wb') as out:
        out.write(b'GPQ8')
        out.write(struct.pack('<I', len(all_tensors)))

        total_q8_bytes = 0
        total_f32_bytes = 0

        for idx, (key, meta, shard_path) in enumerate(all_tensors):
            kb = key.encode('utf-8')
            shape = meta['shape']
            offsets = meta['data_offsets']
            dtype = meta['dtype']

            f, data_start, _ = shard_meta[shard_path]
            f.seek(data_start + offsets[0])
            raw = f.read(offsets[1] - offsets[0])

            n_elems = 1
            for d in shape: n_elems *= d

            if dtype == 'BF16':
                f32_array = bf16_to_f32(raw)
            elif dtype == 'F32':
                f32_array = np.frombuffer(raw, dtype=np.float32).copy()
            else:
                print(f'WARN: {key} dtype={dtype}')
                continue

            out.write(struct.pack('<I', len(kb)))
            out.write(kb)
            out.write(struct.pack('<I', len(shape)))
            for d in shape:
                out.write(struct.pack('<I', d))

            if is_weight(key):
                if len(shape) == 2:
                    in_dim, out_dim = shape[1], shape[0]
                else:
                    in_dim, out_dim = shape[-1], shape[0]
                q8_bytes = quantize_q8_0_per_row(f32_array, in_dim, out_dim)
                out.write(struct.pack('<B', 3))  # qtype=3 (Q8_0)
                out.write(struct.pack('<Q', len(q8_bytes)))
                out.write(q8_bytes)
                out.write(struct.pack('<I', 0))  # no separate scale
                total_q8_bytes += len(q8_bytes)
            else:
                f32_bytes = np.asarray(f32_array, dtype=np.float32).tobytes()
                out.write(struct.pack('<B', 0))  # qtype=0 (F32)
                out.write(struct.pack('<Q', len(f32_bytes)))
                out.write(f32_bytes)
                out.write(struct.pack('<I', 0))
                total_f32_bytes += len(f32_bytes)

            if (idx + 1) % 30 == 0:
                print(f'  [{idx+1}/{len(all_tensors)}] {key[:60]}...', flush=True)

        size = os.path.getsize(OUT_PATH)
        print(f'\nDone: {OUT_PATH}')
        print(f'  Size: {size:,} bytes ({size/1024/1024/1024:.2f} GB)')
        print(f'  Q8_0 data: {total_q8_bytes/1024/1024:.0f} MB')
        print(f'  F32 data: {total_f32_bytes/1024/1024:.0f} MB')

    for f, _, _ in shard_meta.values():
        f.close()

if __name__ == '__main__':
    main()
