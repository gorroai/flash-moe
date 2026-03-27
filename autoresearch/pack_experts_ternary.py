#!/usr/bin/env python3
"""
pack_experts_ternary.py — Encode 4-bit MLX expert weights to ternary 2-bit format.

Ternary quantization: w_q ∈ {-1, 0, +1}, scaled by per-group scale.
  - scale = max(|w|) per group (group_size=128)
  - threshold = 0.5 * scale: |w| < threshold → 0, else sign(w)
  - symmetric: no bias term (halves scale storage vs 4-bit)

Packing: 2 bits per weight in uint32 (16 values per word):
  0b00 = 0 (zero)
  0b01 = +1
  0b10 = -1
  0b11 = unused (never written)

Expert file layout (packed_experts_ternary/layer_XX.bin):
  [ expert_0 | expert_1 | ... | expert_511 ]  linear, no header
  Per expert (EXPERT_SIZE_TERNARY = 3,342,336 bytes = 3.18 MB):
    gate_proj weights  [out=1024, in=4096, 2-bit] = 1,048,576 bytes  offset 0
    gate_proj scales   [out=1024, groups=32, bf16] =    65,536 bytes  offset 1,048,576
    up_proj   weights  [out=1024, in=4096, 2-bit] = 1,048,576 bytes  offset 1,114,112
    up_proj   scales   [out=1024, groups=32, bf16] =    65,536 bytes  offset 2,162,688
    down_proj weights  [out=4096, in=1024, 2-bit] = 1,048,576 bytes  offset 2,228,224
    down_proj scales   [out=4096, groups=8,  bf16] =    65,536 bytes  offset 3,276,800
    Total: 3,342,336 bytes (page-aligned: 3342336/4=835584, 835584/4096=204 ✓)

Source: 4-bit MLX packed_experts/ format.
"""

import numpy as np
import os
import sys
import struct
import argparse
from pathlib import Path
import time

# ──────────────────────────────────────────────────
# Model constants
# ──────────────────────────────────────────────────
HIDDEN_DIM       = 4096
MOE_INTERMEDIATE = 1024
GROUP_SIZE_4BIT  = 64    # 4-bit MLX source group size
GROUP_SIZE_TERN  = 128   # ternary target group size
NUM_EXPERTS      = 512
NUM_LAYERS       = 60

# ──────────────────────────────────────────────────
# 4-bit MLX source layout (from infer.m)
# ──────────────────────────────────────────────────
EXPERT_SIZE_4BIT = 7077888
GATE_W_OFF = 0
GATE_S_OFF = 2097152
GATE_B_OFF = 2228224
UP_W_OFF   = 2359296
UP_S_OFF   = 4456448
UP_B_OFF   = 4587520
DOWN_W_OFF = 4718592
DOWN_S_OFF = 6815744
DOWN_B_OFF = 6946816

# ──────────────────────────────────────────────────
# Ternary layout constants (must match infer.m)
# ──────────────────────────────────────────────────
# gate_proj: out=1024, in=4096
_GATE_W_BYTES = MOE_INTERMEDIATE * (HIDDEN_DIM // 16) * 4   # 1024 * 256 * 4 = 1,048,576
_GATE_S_BYTES = MOE_INTERMEDIATE * (HIDDEN_DIM // GROUP_SIZE_TERN) * 2  # 1024 * 32 * 2 = 65,536
# up_proj:   same shape as gate
_UP_W_BYTES   = _GATE_W_BYTES   # 1,048,576
_UP_S_BYTES   = _GATE_S_BYTES   # 65,536
# down_proj: out=4096, in=1024
_DOWN_W_BYTES = HIDDEN_DIM * (MOE_INTERMEDIATE // 16) * 4   # 4096 * 64 * 4 = 1,048,576
_DOWN_S_BYTES = HIDDEN_DIM * (MOE_INTERMEDIATE // GROUP_SIZE_TERN) * 2  # 4096 * 8 * 2 = 65,536

GATE_W_OFF_T = 0
GATE_S_OFF_T = GATE_W_OFF_T + _GATE_W_BYTES          # 1,048,576
UP_W_OFF_T   = GATE_S_OFF_T + _GATE_S_BYTES           # 1,114,112
UP_S_OFF_T   = UP_W_OFF_T   + _UP_W_BYTES             # 2,162,688
DOWN_W_OFF_T = UP_S_OFF_T   + _UP_S_BYTES             # 2,228,224
DOWN_S_OFF_T = DOWN_W_OFF_T + _DOWN_W_BYTES           # 3,276,800
EXPERT_SIZE_TERNARY = DOWN_S_OFF_T + _DOWN_S_BYTES    # 3,342,336

def verify_layout():
    assert _GATE_W_BYTES == 1048576, _GATE_W_BYTES
    assert _GATE_S_BYTES == 65536, _GATE_S_BYTES
    assert _DOWN_W_BYTES == 1048576, _DOWN_W_BYTES
    assert _DOWN_S_BYTES == 65536, _DOWN_S_BYTES
    assert EXPERT_SIZE_TERNARY == 3342336, EXPERT_SIZE_TERNARY
    # Page-alignment check: EXPERT_SIZE_TERNARY / 4 must be multiple of 4096
    assert (EXPERT_SIZE_TERNARY // 4) % 4096 == 0, \
        f"Not page-aligned for cache-io-split=4: {EXPERT_SIZE_TERNARY//4} % 4096 = {(EXPERT_SIZE_TERNARY//4)%4096}"
    print(f"Layout verified: EXPERT_SIZE_TERNARY={EXPERT_SIZE_TERNARY} bytes ({EXPERT_SIZE_TERNARY/1024/1024:.2f} MB)")
    print(f"  cache-io-split=4 chunk: {EXPERT_SIZE_TERNARY//4} bytes = {EXPERT_SIZE_TERNARY//4//4096} pages ✓")

# ──────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────

def bf16_to_f32(u16: np.ndarray) -> np.ndarray:
    u32 = u16.astype(np.uint32) << 16
    return u32.view(np.float32)

def f32_to_bf16(f32: np.ndarray) -> np.ndarray:
    u32 = f32.view(np.uint32)
    return (u32 >> 16).astype(np.uint16)

def decode_4bit_matrix(data: bytes, w_off: int, s_off: int, b_off: int,
                        out_dim: int, in_dim: int) -> np.ndarray:
    """Decode MLX 4-bit affine quantized weight matrix to float32."""
    n_packed = in_dim // 8
    n_groups = in_dim // GROUP_SIZE_4BIT

    W = np.frombuffer(data, dtype=np.uint32, count=out_dim * n_packed, offset=w_off)
    W = W.reshape(out_dim, n_packed)

    S = np.frombuffer(data, dtype=np.uint16, count=out_dim * n_groups, offset=s_off)
    S = bf16_to_f32(S).reshape(out_dim, n_groups)

    B = np.frombuffer(data, dtype=np.uint16, count=out_dim * n_groups, offset=b_off)
    B = bf16_to_f32(B).reshape(out_dim, n_groups)

    result = np.zeros((out_dim, in_dim), dtype=np.float32)
    for bit in range(8):
        result[:, bit::8] = ((W >> (bit * 4)) & 0xF).astype(np.float32)

    S_exp = np.repeat(S, GROUP_SIZE_4BIT, axis=1)
    B_exp = np.repeat(B, GROUP_SIZE_4BIT, axis=1)
    return result * S_exp + B_exp


def encode_matrix_ternary(W: np.ndarray, group_size: int = GROUP_SIZE_TERN):
    """
    Ternary-quantize weight matrix W using per-group max scaling.

    W: (out_dim, in_dim) float32
    Returns:
      packed: (out_dim, in_dim//16) uint32  — 2 bits per weight, 16 per word
      scales: (out_dim, in_dim//group_size) uint16  — bf16 scale per group
    Encoding: 0b00=0, 0b01=+1, 0b10=-1
    """
    out_dim, in_dim = W.shape
    assert in_dim % group_size == 0
    assert in_dim % 16 == 0

    n_groups = in_dim // group_size
    n_packed = in_dim // 16  # uint32 words per row

    # Compute per-group max scale
    W_grouped = W.reshape(out_dim, n_groups, group_size)
    scales_f32 = np.max(np.abs(W_grouped), axis=2)  # (out_dim, n_groups)
    scales_f32 = np.where(scales_f32 == 0, 1.0, scales_f32)  # avoid div-by-zero

    # Quantize: threshold = 0.5 * scale
    scales_exp = np.repeat(scales_f32, group_size, axis=1)  # (out_dim, in_dim)
    W_norm = W / scales_exp
    # round to {-1, 0, +1}: values in (-0.5, +0.5) → 0
    W_q = np.round(W_norm).clip(-1, 1).astype(np.int8)

    # Pack 16 ternary values per uint32 (2 bits each)
    # Encoding: 0→0b00, +1→0b01, -1→0b10
    # Map: 0→0, 1→1, -1→2 using: code = (1 - q) & 3 gives: 0→1(wrong)... let me use:
    # code = np.where(W_q == 0, 0, np.where(W_q > 0, 1, 2)).astype(np.uint32)
    code = np.where(W_q == 0, np.uint32(0), np.where(W_q > 0, np.uint32(1), np.uint32(2)))
    code = code.reshape(out_dim, n_packed, 16)

    packed = np.zeros((out_dim, n_packed), dtype=np.uint32)
    for i in range(16):
        packed |= code[:, :, i] << (i * 2)

    scales_bf16 = f32_to_bf16(scales_f32)
    return packed, scales_bf16


def encode_expert_ternary(expert_data: bytes) -> bytes:
    """Encode one 4-bit MLX expert to ternary format."""
    # Decode all three projections
    gate_f32 = decode_4bit_matrix(expert_data, GATE_W_OFF, GATE_S_OFF, GATE_B_OFF,
                                   MOE_INTERMEDIATE, HIDDEN_DIM)
    up_f32   = decode_4bit_matrix(expert_data, UP_W_OFF, UP_S_OFF, UP_B_OFF,
                                   MOE_INTERMEDIATE, HIDDEN_DIM)
    down_f32 = decode_4bit_matrix(expert_data, DOWN_W_OFF, DOWN_S_OFF, DOWN_B_OFF,
                                   HIDDEN_DIM, MOE_INTERMEDIATE)

    # Ternary quantize
    gate_w, gate_s = encode_matrix_ternary(gate_f32)
    up_w,   up_s   = encode_matrix_ternary(up_f32)
    down_w, down_s = encode_matrix_ternary(down_f32)

    # Serialize
    buf = bytearray(EXPERT_SIZE_TERNARY)
    buf[GATE_W_OFF_T:GATE_W_OFF_T + _GATE_W_BYTES] = gate_w.tobytes()
    buf[GATE_S_OFF_T:GATE_S_OFF_T + _GATE_S_BYTES] = gate_s.tobytes()
    buf[UP_W_OFF_T:UP_W_OFF_T   + _UP_W_BYTES]     = up_w.tobytes()
    buf[UP_S_OFF_T:UP_S_OFF_T   + _UP_S_BYTES]     = up_s.tobytes()
    buf[DOWN_W_OFF_T:DOWN_W_OFF_T + _DOWN_W_BYTES] = down_w.tobytes()
    buf[DOWN_S_OFF_T:DOWN_S_OFF_T + _DOWN_S_BYTES] = down_s.tobytes()
    return bytes(buf)


# ──────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Encode 4-bit MLX experts to ternary 2-bit format")
    parser.add_argument("--model", default="~/Models/flash_mlx_4bit",
                        help="Model directory (contains packed_experts/)")
    parser.add_argument("--output", default=None,
                        help="Output directory (default: <model>/packed_experts_ternary)")
    parser.add_argument("--layers", default="all",
                        help="Layers to encode: 'all' or comma-separated indices")
    args = parser.parse_args()

    verify_layout()

    model_dir = Path(os.path.expanduser(args.model))
    src_dir = model_dir / "packed_experts"
    out_dir = Path(os.path.expanduser(args.output)) if args.output else model_dir / "packed_experts_ternary"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.layers == "all":
        layers = list(range(NUM_LAYERS))
    else:
        layers = [int(x) for x in args.layers.split(",")]

    print(f"Source: {src_dir}")
    print(f"Output: {out_dir}")
    print(f"Layers: {layers}")
    print(f"Expert size: {EXPERT_SIZE_TERNARY} bytes ({EXPERT_SIZE_TERNARY/1024/1024:.2f} MB) — vs Q3 5.44 MB, 4-bit 6.75 MB")
    print()

    total_t0 = time.time()

    for layer in layers:
        src_path = src_dir / f"layer_{layer:02d}.bin"
        out_path = out_dir / f"layer_{layer:02d}.bin"

        if not src_path.exists():
            print(f"[layer {layer:02d}] MISSING {src_path}, skipping")
            continue

        t0 = time.time()
        src_data = src_path.read_bytes()
        assert len(src_data) == EXPERT_SIZE_4BIT * NUM_EXPERTS, \
            f"Expected {EXPERT_SIZE_4BIT * NUM_EXPERTS} bytes, got {len(src_data)}"

        out_data = bytearray()
        for e in range(NUM_EXPERTS):
            expert_bytes = src_data[e * EXPERT_SIZE_4BIT:(e + 1) * EXPERT_SIZE_4BIT]
            out_data += encode_expert_ternary(expert_bytes)

        assert len(out_data) == EXPERT_SIZE_TERNARY * NUM_EXPERTS
        out_path.write_bytes(out_data)

        elapsed = time.time() - t0
        remaining = (NUM_LAYERS - layer - 1) * elapsed / max(1, layer - layers[0] + 1)
        print(f"[layer {layer:02d}] {elapsed:.1f}s  ETA {remaining/60:.1f}min  "
              f"→ {out_path} ({len(out_data)/1024/1024:.1f} MB)")

    total = time.time() - total_t0
    print(f"\nDone. {len(layers)} layers in {total/60:.1f} min.")
    print(f"Total output: {len(layers) * EXPERT_SIZE_TERNARY * NUM_EXPERTS / 1024**3:.1f} GB")


if __name__ == "__main__":
    main()
