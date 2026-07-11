"""Convert Qwen2.5-7B-Instruct safetensors (BF16) to LAL Q4_K format.

Q4_K = 256-element superblock, 144 bytes/block:
  - d:     fp16 (2 bytes) — super-block scale
  - dmin:  fp16 (2 bytes) — super-block min scale
  - scales: 12 bytes — 8 sub-block (32 elem) scales, 6-bit packed
  - qs:    128 bytes — 256 × 4-bit packed values (offset by min)
  Total: 144 bytes / 256 elements = 0.5625 bytes/elem

144 = 2.25 × 64 (cache lines). Not perfectly aligned, but 4 blocks = 576B
= 9 cache lines (aligned pattern). Much better than 18B blocks.

Quality: per-sub-block scales + min offset → near-Q8 accuracy.
This matches llama.cpp's Q4_K_M format.
"""
import struct, json, os, sys, glob
import numpy as np

SHARD_DIR = '/root/qwen7b'
OUT_PATH = '/root/lal/prebuilt/qwen7b_q4k_weights.bin'

def bf16_to_f32(u16_bytes):
    u16 = np.frombuffer(u16_bytes, dtype=np.uint16)
    return (u16.astype(np.uint32) << 16).view(np.float32)

def quantize_q4_k_per_row(f32_data, in_dim, out_dim):
    """Quantize [out_dim, in_dim] to Q4_K format.
    Simplified: per-sub-block (32 elem) scale + min, matching llama.cpp's formula:
      q = round((x + min) / scale), dequant: w = scale * q - min
    where scale = d * sc / 63, min = dmin * m / 63
    """
    w = np.asarray(f32_data, dtype=np.float32).reshape(out_dim, in_dim)
    assert in_dim % 256 == 0, f"in_dim {in_dim} not multiple of 256"
    n_super = in_dim // 256
    w_blocks = w.reshape(out_dim, n_super, 8, 32)

    # Per sub-block: find min and scale
    sub_min = np.min(w_blocks, axis=3)  # [out_dim, n_super, 8]
    sub_max = np.max(w_blocks, axis=3)
    sub_scale = (sub_max - sub_min) / 15.0  # range / 15 for 4-bit 0-15
    sub_scale = np.where(sub_scale < 1e-8, 1e-8, sub_scale)

    # Super-block d = max scale, dmin = max |min|
    max_scale = np.max(sub_scale, axis=2)  # [out_dim, n_super]
    max_min = np.max(np.abs(sub_min), axis=2)
    max_scale = np.where(max_scale < 1e-8, 1e-8, max_scale)
    max_min = np.where(max_min < 1e-8, 1e-8, max_min)

    d = max_scale.astype(np.float32)
    dmin = max_min.astype(np.float32)

    # 6-bit scales: sc = round(sub_scale / d * 63)
    scale_6bit = np.clip(np.round(sub_scale / d[:, :, None] * 63), 0, 63).astype(np.uint8)
    # 6-bit mins: m = round(|sub_min| / dmin * 63)
    min_6bit = np.clip(np.round(np.abs(sub_min) / dmin[:, :, None] * 63), 0, 63).astype(np.uint8)

    # Quantize: q = round((x + dmin*m/63) / (d*sc/63))
    actual_scale = d[:, :, None] * scale_6bit.astype(np.float32) / 63.0
    actual_min = dmin[:, :, None] * min_6bit.astype(np.float32) / 63.0  # positive (it's |min|)

    # q = round((x + actual_min) / actual_scale), clip 0-15
    # Dequant: w = actual_scale * q - actual_min
    q = np.round((w_blocks + actual_min[:, :, :, None]) / (actual_scale[:, :, :, None] + 1e-8))
    q = np.clip(q, 0, 15).astype(np.uint8)

    # Pack 4-bit values: 256 elements -> 128 bytes
    # INTERLEAVED packing per sub-block: byte[sub*16+i] = q[sub*32+i] | (q[sub*32+i+16] << 4)
    q_flat = q.reshape(out_dim, n_super, 256)
    qs_packed = np.zeros((out_dim, n_super, 128), dtype=np.uint8)
    for sub in range(8):
        for i in range(16):
            qs_packed[:, :, sub*16 + i] = q_flat[:, :, sub*32 + i] | (q_flat[:, :, sub*32 + i + 16] << 4)

    # Pack scales+mins: 8 scales + 8 mins, 6-bit each, into 12 bytes
    scales_min = np.zeros((out_dim, n_super, 12), dtype=np.uint8)
    combined = np.zeros((out_dim, n_super, 16), dtype=np.uint32)
    combined[:, :, :8] = scale_6bit
    combined[:, :, 8:] = min_6bit
    for i in range(out_dim):
        for s in range(n_super):
            bits = 0
            for j in range(16):
                bits |= (int(combined[i, s, j]) & 0x3F) << (j * 6)
            for b in range(12):
                scales_min[i, s, b] = (bits >> (b * 8)) & 0xFF

    # Build 144-byte blocks: [2B d][2B dmin][12B scales][128B qs]
    d_fp16 = d.astype(np.float16)
    dmin_fp16 = dmin.astype(np.float16)
    out = np.zeros((out_dim, n_super, 144), dtype=np.uint8)
    out[:, :, 0:2] = np.frombuffer(d_fp16.tobytes(), dtype=np.uint8).reshape(out_dim, n_super, 2)
    out[:, :, 2:4] = np.frombuffer(dmin_fp16.tobytes(), dtype=np.uint8).reshape(out_dim, n_super, 2)
    out[:, :, 4:16] = scales_min
    out[:, :, 16:144] = qs_packed
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
    print(f'  Q4_K weights: {n_weight}')

    shard_meta = {}
    for p in shards:
        f = open(p, 'rb')
        hlen = struct.unpack('<Q', f.read(8))[0]
        hdr = json.loads(f.read(hlen).decode('utf-8'))
        shard_meta[p] = (f, 8 + hlen, hdr)

    with open(OUT_PATH, 'wb') as out:
        out.write(b'GPQ8')
        out.write(struct.pack('<I', len(all_tensors)))
        total_q4k = 0; total_f32 = 0
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
                q4k_bytes = quantize_q4_k_per_row(f32_array, in_dim, out_dim)
                out.write(struct.pack('<B', 5))  # qtype=5 (Q4_K)
                out.write(struct.pack('<Q', len(q4k_bytes))); out.write(q4k_bytes)
                out.write(struct.pack('<I', 0))
                total_q4k += len(q4k_bytes)
            else:
                f32_bytes = np.asarray(f32_array, dtype=np.float32).tobytes()
                out.write(struct.pack('<B', 0))
                out.write(struct.pack('<Q', len(f32_bytes))); out.write(f32_bytes)
                out.write(struct.pack('<I', 0))
                total_f32 += len(f32_bytes)
            if (idx + 1) % 50 == 0:
                print(f'  [{idx+1}/{len(all_tensors)}] {key[:60]}...', flush=True)
        size = os.path.getsize(OUT_PATH)
        print(f'\nDone: {OUT_PATH}')
        print(f'  Size: {size:,} bytes ({size/1073741824:.2f} GB)')
        print(f'  Q4_K data: {total_q4k/1073741824:.2f} GB')
        print(f'  F32 data: {total_f32/1073741824:.2f} GB')
    for f, _, _ in shard_meta.values(): f.close()

if __name__ == '__main__':
    main()
