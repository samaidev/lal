#!/usr/bin/env python3
"""Test Q4_K quantization round-trip: quantize → dequantize → compare.
This verifies the converter and kernel are consistent BEFORE running the full model.
"""
import numpy as np
import struct

def quantize_q4_k_block(w256):
    """Quantize 256 floats to Q4_K block (144 bytes). Returns (packed_bytes, dequant_floats)."""
    w = np.asarray(w256, dtype=np.float32).reshape(8, 32)
    sub_min = np.min(w, axis=1)
    sub_max = np.max(w, axis=1)
    sub_scale = (sub_max - sub_min) / 15.0
    sub_scale = np.where(sub_scale < 1e-8, 1e-8, sub_scale)

    d = np.max(sub_scale)
    dmin = np.max(np.abs(sub_min))
    d = max(d, 1e-8)
    dmin = max(dmin, 1e-8)

    scale_6bit = np.clip(np.round(sub_scale / d * 63), 0, 63).astype(np.uint8)
    min_6bit = np.clip(np.round(np.abs(sub_min) / dmin * 63), 0, 63).astype(np.uint8)

    actual_scale = d * scale_6bit.astype(np.float32) / 63.0
    actual_min = dmin * min_6bit.astype(np.float32) / 63.0

    q = np.round((w + actual_min[:, None]) / (actual_scale[:, None] + 1e-8))
    q = np.clip(q, 0, 15).astype(np.uint8)

    # Dequantize
    dequant = actual_scale[:, None] * q.astype(np.float32) - actual_min[:, None]

    # Pack
    q_flat = q.reshape(256)
    q_pairs = q_flat.reshape(128, 2)
    qs_packed = q_pairs[:, 0] | (q_pairs[:, 1] << 4)

    # Pack 6-bit scales+mins
    combined = np.zeros(16, dtype=np.uint32)
    combined[:8] = scale_6bit
    combined[8:] = min_6bit
    bits = 0
    for j in range(16):
        bits |= (int(combined[j]) & 0x3F) << (j * 6)
    scales_min = bytearray(12)
    for b in range(12):
        scales_min[b] = (bits >> (b * 8)) & 0xFF

    # Build 144-byte block
    block = bytearray(144)
    d_fp16 = np.float16(d)
    dmin_fp16 = np.float16(dmin)
    block[0:2] = d_fp16.tobytes()
    block[2:4] = dmin_fp16.tobytes()
    block[4:16] = scales_min
    block[16:144] = qs_packed.tobytes()

    return bytes(block), dequant.reshape(256)

# Test
np.random.seed(42)
w = np.random.randn(256).astype(np.float32) * 0.1
packed, dequant = quantize_q4_k_block(w)
print(f"Original:  {w[:8]}")
print(f"Dequant:   {dequant[:8]}")
print(f"Error:     {np.max(np.abs(w - dequant)):.6f}")
print(f"Rel error: {np.max(np.abs(w - dequant)) / (np.max(np.abs(w)) + 1e-8):.4f}")

# Now simulate the kernel's unpack
def kernel_unpack(packed):
    """Simulate the C kernel's unpacking and dequantization."""
    d_fp16 = np.frombuffer(packed[0:2], dtype=np.float16)[0].astype(np.float32)
    dmin_fp16 = np.frombuffer(packed[2:4], dtype=np.float16)[0].astype(np.float32)
    scales_min_bytes = packed[4:16]

    # Unpack 16 × 6-bit from 12 bytes
    bits = int.from_bytes(scales_min_bytes, 'little')
    scales_mins = [(bits >> (i * 6)) & 0x3F for i in range(16)]
    scale_6bit = scales_mins[:8]
    min_6bit = scales_mins[8:]

    # Unpack q4 values
    qs = packed[16:144]
    q4 = np.zeros(256, dtype=np.uint8)
    for i in range(128):
        q4[2*i] = qs[i] & 0x0F
        q4[2*i+1] = (qs[i] >> 4) & 0x0F

    # Dequantize using kernel formula
    result = np.zeros(256, dtype=np.float32)
    for sub in range(8):
        sb_scale = d_fp16 * scale_6bit[sub] / 63.0
        sb_min = dmin_fp16 * min_6bit[sub] / 63.0
        for i in range(32):
            idx = sub * 32 + i
            result[idx] = sb_scale * q4[idx] - sb_min
    return result

kernel_result = kernel_unpack(packed)
print(f"\nKernel dequant: {kernel_result[:8]}")
print(f"Python dequant: {dequant[:8]}")
print(f"Kernel vs Python error: {np.max(np.abs(kernel_result - dequant)):.6f}")
print(f"Kernel vs Original error: {np.max(np.abs(kernel_result - w)):.6f}")
