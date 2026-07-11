"""Convert Qwen2.5-7B-Instruct safetensors (BF16) to LAL Q4_0A format.

Q4_0A = Q4_0 Aligned: 32 bytes/block = 2B fp16 scale + 16B data + 14B padding.
This fixes the 18-byte misalignment that caused 2x bandwidth waste.

32 bytes = exactly half a cache line (64B). Every block starts on a
32-byte boundary → no cache line crossings → 1.7x bandwidth improvement.

File format: "GPQ8" magic + n_tensors + per-tensor:
  [key_len][key][ndim][shape][qtype(4=Q4_0A)][data_len][data][scale_len=0]
"""
import struct, json, os, sys, glob
import numpy as np

SHARD_DIR = '/root/qwen7b'
OUT_PATH = '/root/lal/prebuilt/qwen7b_q4_0a_weights.bin'

def bf16_to_f32(u16_bytes):
    u16 = np.frombuffer(u16_bytes, dtype=np.uint16)
    return (u16.astype(np.uint32) << 16).view(np.float32)

def quantize_q4_0a_per_row(f32_data, in_dim, out_dim):
    """Quantize [out_dim, in_dim] float32 to Q4_0A packed format (32 bytes/block).
    Block layout: [2B fp16 scale][16B packed q4][14B padding] = 32 bytes.
    """
    w = np.asarray(f32_data, dtype=np.float32).reshape(out_dim, in_dim)
    assert in_dim % 32 == 0
    n_blocks_per_row = in_dim // 32
    blocks = w.reshape(out_dim, n_blocks_per_row, 32)
    block_max = np.max(np.abs(blocks), axis=2)
    scale = block_max / -8.0
    scale = np.where(np.abs(scale) < 1e-8, 1e-8, scale).astype(np.float32)
    inv_scale = 1.0 / scale
    q = np.round(blocks * inv_scale[:, :, None]).astype(np.int32) + 8
    q = np.clip(q, 0, 15).astype(np.uint8)
    q_pairs = q.reshape(out_dim, n_blocks_per_row, 16, 2)
    packed = q_pairs[:, :, :, 0] | (q_pairs[:, :, :, 1] << 4)
    scale_fp16 = scale.astype(np.float16)
    # Build 32-byte blocks: [2B scale][16B data][14B padding]
    out = np.zeros((out_dim, n_blocks_per_row, 32), dtype=np.uint8)
    out[:, :, :2] = np.frombuffer(scale_fp16.tobytes(), dtype=np.uint8).reshape(
        out_dim, n_blocks_per_row, 2)
    out[:, :, 2:18] = packed
    # bytes 18-31 are zero (padding)
    return out.tobytes()

WEIGHT_SUFFIXES = [
    'self_attn.q_proj.weight', 'self_attn.k_proj.weight',
    'self_attn.v_proj.weight', 'self_attn.o_proj.weight',
    'mlp.gate_proj.weight', 'mlp.up_proj.weight', 'mlp.down_proj.weight',
]
def is_weight(key): return any(key.endswith(s) for s in WEIGHT_SUFFIXES)
def is_skip(key): return 'attn.bias' in key or 'masked_bias' in key

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
    print(f'  Q4_0A weights: {n_weight}')

    shard_meta = {}
    for p in shards:
        f = open(p, 'rb')
        hlen = struct.unpack('<Q', f.read(8))[0]
        hdr = json.loads(f.read(hlen).decode('utf-8'))
        shard_meta[p] = (f, 8 + hlen, hdr)

    with open(OUT_PATH, 'wb') as out:
        out.write(b'GPQ8')
        out.write(struct.pack('<I', len(all_tensors)))
        total_q4a = 0; total_f32 = 0
        for idx, (key, meta, shard_path) in enumerate(all_tensors):
            kb = key.encode('utf-8')
            shape = meta['shape']; offsets = meta['data_offsets']; dtype = meta['dtype']
            f, data_start, _ = shard_meta[shard_path]
            f.seek(data_start + offsets[0])
            raw = f.read(offsets[1] - offsets[0])
            if dtype == 'BF16': f32_array = bf16_to_f32(raw)
            elif dtype == 'F32': f32_array = np.frombuffer(raw, dtype=np.float32).copy()
            else: print(f'WARN: {key} dtype={dtype}'); continue
            out.write(struct.pack('<I', len(kb))); out.write(kb)
            out.write(struct.pack('<I', len(shape)))
            for d in shape: out.write(struct.pack('<I', d))
            if is_weight(key):
                in_dim, out_dim = (shape[1], shape[0]) if len(shape)==2 else (shape[-1], shape[0])
                q4a_bytes = quantize_q4_0a_per_row(f32_array, in_dim, out_dim)
                out.write(struct.pack('<B', 4))  # qtype=4 (Q4_0A)
                out.write(struct.pack('<Q', len(q4a_bytes))); out.write(q4a_bytes)
                out.write(struct.pack('<I', 0))  # no separate scale
                total_q4a += len(q4a_bytes)
            else:
                f32_bytes = np.asarray(f32_array, dtype=np.float32).tobytes()
                out.write(struct.pack('<B', 0))  # qtype=0 (F32)
                out.write(struct.pack('<Q', len(f32_bytes))); out.write(f32_bytes)
                out.write(struct.pack('<I', 0))
                total_f32 += len(f32_bytes)
            if (idx + 1) % 50 == 0:
                print(f'  [{idx+1}/{len(all_tensors)}] {key[:60]}...', flush=True)
        size = os.path.getsize(OUT_PATH)
        print(f'\nDone: {OUT_PATH}')
        print(f'  Size: {size:,} bytes ({size/1073741824:.2f} GB)')
        print(f'  Q4_0A data: {total_q4a/1073741824:.2f} GB')
        print(f'  F32 data: {total_f32/1073741824:.2f} GB')
    for f, _, _ in shard_meta.values(): f.close()

if __name__ == '__main__':
    main()
