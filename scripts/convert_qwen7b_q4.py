"""Convert Qwen2.5-7B-Instruct safetensors (BF16, 4 shards) to LAL GPQ4 format.

Q4_0 block format (mistral.rs / llama.cpp style):
  - Block size: 32 elements
  - Per block: 1 fp16 scale (2 bytes) + 16 bytes (32 x 4-bit values packed 2/byte)
  - Total: 18 bytes per 32 elements = 0.5625 bytes/element
  - vs Q8: 1 byte/element + 4 bytes/row scale
  - Compression: ~1.78x smaller than Q8

The 4-bit values are stored as unsigned (0..15) with an offset of 8, so the
signed value = q4 - 8 (range -8..7). Scale is fp16, applied per block.

File format: "GPQ8" magic (same as before, qtype field distinguishes) +
n_tensors + per-tensor:
  [key_len][key][ndim][shape][qtype(0=F32,1=Q8,2=Q4_0)][data_len][data][scale_len][scale]

For Q4_0:
  - data = packed blocks (out_dim * in_dim / 32 blocks, 18 bytes each)
  - scale_len = 0 (scale is inside each block, not separate)
"""
import struct, json, os, sys, glob
import numpy as np

SHARD_DIR = '/root/qwen7b'
OUT_PATH = '/root/lal/prebuilt/qwen7b_q4_weights.bin'

def bf16_to_f32(u16_bytes):
    u16 = np.frombuffer(u16_bytes, dtype=np.uint16)
    return (u16.astype(np.uint32) << 16).view(np.float32)

def quantize_q4_0_per_row(f32_data, in_dim, out_dim):
    """Quantize [out_dim, in_dim] float32 to Q4_0 packed format.
    
    Returns bytes of packed blocks. Each block = 2 bytes fp16 scale + 16 bytes
    (32 x 4-bit values, 2 per byte, low nibble first).
    
    Block layout: for row r, block b (covering elements [b*32 .. b*32+31]):
      scale_fp16 = max(abs(block)) / -8.0  (so q4 = round(x/scale) + 8 in [0,15])
      qs[16] = pack 32 values: byte[i] = (q[2i] & 0xF) | ((q[2i+1] & 0xF) << 4)
    """
    w = np.asarray(f32_data, dtype=np.float32).reshape(out_dim, in_dim)
    # Ensure in_dim is multiple of 32
    assert in_dim % 32 == 0, f"in_dim {in_dim} not multiple of 32"
    n_blocks_per_row = in_dim // 32
    total_blocks = out_dim * n_blocks_per_row
    
    # Reshape to blocks: [out_dim, n_blocks_per_row, 32]
    blocks = w.reshape(out_dim, n_blocks_per_row, 32)
    block_max = np.max(np.abs(blocks), axis=2)  # [out_dim, n_blocks_per_row]
    
    # Scale = max / -8.0 (Q4_0 uses -8 as the reference)
    # Avoid division by zero
    scale = block_max / -8.0
    scale = np.where(np.abs(scale) < 1e-8, 1e-8, scale).astype(np.float32)
    
    # Quantize: q = round(x / scale) + 8, clip to [0, 15]
    inv_scale = 1.0 / scale  # [out_dim, n_blocks_per_row]
    q = np.round(blocks * inv_scale[:, :, None]).astype(np.int32) + 8
    q = np.clip(q, 0, 15).astype(np.uint8)  # [out_dim, n_blocks_per_row, 32]
    
    # Pack 2 values per byte: byte[i] = q[2i] | (q[2i+1] << 4)
    # Reshape to [out_dim, n_blocks_per_row, 16, 2] then combine
    q_pairs = q.reshape(out_dim, n_blocks_per_row, 16, 2)
    packed = q_pairs[:, :, :, 0] | (q_pairs[:, :, :, 1] << 4)  # uint8 [out_dim, n_blocks_per_row, 16]
    
    # Interleave scale (fp16) and packed bytes per block
    scale_fp16 = scale.astype(np.float16)  # [out_dim, n_blocks_per_row]
    
    # Build output: for each block, 2 bytes scale + 16 bytes data
    # Layout: [out_dim, n_blocks_per_row, 18]
    out = np.zeros((out_dim, n_blocks_per_row, 18), dtype=np.uint8)
    out[:, :, :2] = np.frombuffer(scale_fp16.tobytes(), dtype=np.uint8).reshape(
        out_dim, n_blocks_per_row, 2)
    out[:, :, 2:] = packed
    
    return out.tobytes()

# Weight matrix keys to quantize to Q4
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
    
    # First pass: collect all tensor headers
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
    print(f'  Q4 weights: {n_weight}, float32: {n_float}')
    
    # Open shard file handles for lazy reads
    shard_files = {p: open(p, 'rb') for p in shards}
    shard_headers = {}
    for p in shards:
        with open(p, 'rb') as f:
            hlen = struct.unpack('<Q', f.read(8))[0]
            shard_headers[p] = (hlen, json.loads(open(p, 'rb').read(8 + hlen).decode('utf-8') if False else b'{}'))
    
    # Re-read headers properly
    shard_meta = {}
    for p in shards:
        f = open(p, 'rb')
        hlen = struct.unpack('<Q', f.read(8))[0]
        hdr = json.loads(f.read(hlen).decode('utf-8'))
        data_start = 8 + hlen
        shard_meta[p] = (f, data_start, hdr)
    
    with open(OUT_PATH, 'wb') as out:
        out.write(b'GPQ8')  # same magic, qtype field distinguishes
        out.write(struct.pack('<I', len(all_tensors)))
        
        total_q4_bytes = 0
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
                q4_bytes = quantize_q4_0_per_row(f32_array, in_dim, out_dim)
                out.write(struct.pack('<B', 2))  # qtype=2 (Q4_0)
                out.write(struct.pack('<Q', len(q4_bytes)))
                out.write(q4_bytes)
                out.write(struct.pack('<I', 0))  # no separate scale (in blocks)
                total_q4_bytes += len(q4_bytes)
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
        print(f'  Q4 data: {total_q4_bytes/1024/1024:.0f} MB')
        print(f'  F32 data: {total_f32_bytes/1024/1024:.0f} MB')
    
    for f, _, _ in shard_meta.values():
        f.close()

if __name__ == '__main__':
    main()
