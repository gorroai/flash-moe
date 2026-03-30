#!/usr/bin/env python3
"""
pack_experts_qjl.py — Encode 4-bit MLX expert weights to 1-bit QJL (SRHT) format.

Algorithm per row r of each expert projection matrix:
  1. y = WHT(D ⊙ r)   where D ∈ {-1,+1}^d is a fixed per-dimension random diagonal
  2. Store: norm = ‖r‖₂ (float32),  bits = sign(y) packed as 1 bit per element

Dequant in Metal (gpu_encode_experts_qjl):
  out_i = (π/2 / d) · norm_i · dot(bits_i, WHT(D ⊙ x))

  The WHT(D⊙x) is computed once per expert dispatch, then reused for all rows.

Expert file layout (packed_experts_QJL/layer_XX.bin):
  [ expert_0 | expert_1 | ... | expert_511 ]  — no file header, linear layout
  Per expert (EXPERT_SIZE_QJL = 1,597,440 bytes):
    gate_proj: float32[1024] norms + uint8[1024][512] sign_bits  (528,384 bytes)
    up_proj:   float32[1024] norms + uint8[1024][512] sign_bits  (528,384 bytes)
    down_proj: float32[4096] norms + uint8[4096][128] sign_bits  (540,672 bytes)

D vectors (fixed seed, global across all layers):
  packed_experts_QJL/d_vectors.bin — 640 bytes per layer × 60 layers = 38,400 bytes
  Layout: for each layer (in order):
    D_shared: uint8[512]  — HIDDEN_DIM/8 bytes (for gate/up, in_dim=4096)
    D_down:   uint8[128]  — MOE_INTERMEDIATE/8 bytes (for down, in_dim=1024)

Source: 4-bit MLX packed_experts/ format (unpacked and decoded here).
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
GROUP_SIZE       = 64
NUM_EXPERTS      = 512
NUM_LAYERS       = 60

# ──────────────────────────────────────────────────
# 4-bit MLX expert source layout (from infer.m)
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
# QJL expert layout constants (must match infer.m)
# ──────────────────────────────────────────────────
_GATE_NORMS_SZ = MOE_INTERMEDIATE * 4             # 4,096
_GATE_SIGNS_SZ = MOE_INTERMEDIATE * (HIDDEN_DIM // 8)  # 524,288
_UP_NORMS_SZ   = MOE_INTERMEDIATE * 4
_UP_SIGNS_SZ   = MOE_INTERMEDIATE * (HIDDEN_DIM // 8)
_DOWN_NORMS_SZ = HIDDEN_DIM * 4                   # 16,384
_DOWN_SIGNS_SZ = HIDDEN_DIM * (MOE_INTERMEDIATE // 8)  # 524,288

QJL_GATE_NORMS_OFF = 0
QJL_GATE_SIGNS_OFF = QJL_GATE_NORMS_OFF + _GATE_NORMS_SZ
QJL_UP_NORMS_OFF   = QJL_GATE_SIGNS_OFF + _GATE_SIGNS_SZ
QJL_UP_SIGNS_OFF   = QJL_UP_NORMS_OFF   + _UP_NORMS_SZ
QJL_DOWN_NORMS_OFF = QJL_UP_SIGNS_OFF   + _UP_SIGNS_SZ
QJL_DOWN_SIGNS_OFF = QJL_DOWN_NORMS_OFF + _DOWN_NORMS_SZ
EXPERT_SIZE_QJL    = QJL_DOWN_SIGNS_OFF + _DOWN_SIGNS_SZ  # 1,597,440

D_SHARED_BYTES = HIDDEN_DIM // 8        # 512
D_DOWN_BYTES   = MOE_INTERMEDIATE // 8  # 128
D_RECORD_BYTES = D_SHARED_BYTES + D_DOWN_BYTES  # 640 per layer


# ──────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────

def bf16_to_f32(u16: np.ndarray) -> np.ndarray:
    """Convert uint16 bfloat16 → float32 via bitcast."""
    u32 = u16.astype(np.uint32) << 16
    return u32.view(np.float32)


def decode_4bit_matrix(data: bytes, w_off: int, s_off: int, b_off: int,
                        out_dim: int, in_dim: int) -> np.ndarray:
    """
    Decode MLX 4-bit affine quantized weight matrix to float32.
    Returns ndarray of shape (out_dim, in_dim).
    """
    n_packed  = in_dim // 8
    n_groups  = in_dim // GROUP_SIZE

    W = np.frombuffer(data, dtype=np.uint32, count=out_dim * n_packed, offset=w_off)
    W = W.reshape(out_dim, n_packed)

    S = np.frombuffer(data, dtype=np.uint16, count=out_dim * n_groups, offset=s_off)
    S = bf16_to_f32(S).reshape(out_dim, n_groups)

    B = np.frombuffer(data, dtype=np.uint16, count=out_dim * n_groups, offset=b_off)
    B = bf16_to_f32(B).reshape(out_dim, n_groups)

    # Extract nibbles: shape (out_dim, in_dim)
    result = np.zeros((out_dim, in_dim), dtype=np.float32)
    for bit in range(8):
        result[:, bit::8] = ((W >> (bit * 4)) & 0xF).astype(np.float32)

    # Broadcast scales/biases over groups
    S_exp = np.repeat(S, GROUP_SIZE, axis=1)
    B_exp = np.repeat(B, GROUP_SIZE, axis=1)
    return result * S_exp + B_exp


def wht_batch(X: np.ndarray) -> np.ndarray:
    """
    Apply in-place Walsh-Hadamard transform to every row of X (unnormalized).
    X shape: (n_rows, d) where d must be a power of 2.
    """
    n = X.shape[1]
    assert (n & (n - 1)) == 0, f"in_dim must be power of 2, got {n}"
    X = X.copy()
    h = 1
    while h < n:
        X = X.reshape(X.shape[0], -1, 2 * h)
        a = X[:, :, :h].copy()
        b = X[:, :, h:].copy()
        X[:, :, :h] = a + b
        X[:, :, h:] = a - b
        X = X.reshape(X.shape[0], -1)
        h *= 2
    return X


def encode_matrix_qjl(W: np.ndarray, D_bits: np.ndarray):
    """
    Encode float32 matrix W to 1-bit QJL using SRHT.

    W:      (out_dim, in_dim) float32 — weight matrix rows
    D_bits: (in_dim,) uint8 array of {0,1} — diagonal scrambling bits

    Returns:
      norms:      (out_dim,) float32 — L2 norms of original rows
      sign_packed:(out_dim, in_dim//8) uint8 — packed sign bits of WHT(D⊙r)
    """
    # Compute L2 norms of original rows (before scrambling)
    norms = np.linalg.norm(W, axis=1).astype(np.float32)

    # Apply D scrambling: D_j ∈ {-1, +1}
    # IMPORTANT: cast to float32 BEFORE multiplication to avoid uint8 overflow
    # (uint8: 0*2-1=255, not -1; float32: 0.0*2-1=-1.0)
    D_signs = D_bits.astype(np.float32) * 2 - 1   # {0,1} → {-1.0, +1.0}
    W_scrambled = W * D_signs[np.newaxis, :]

    # Apply WHT
    W_proj = wht_batch(W_scrambled)

    # 1-bit quantize: 1 = positive (≥ 0), 0 = negative
    sign_bits = (W_proj >= 0).astype(np.uint8)

    # Pack bits little-endian (bit 0 of each byte = element 0 of group of 8)
    sign_packed = np.packbits(sign_bits, axis=1, bitorder='little')

    return norms, sign_packed


# ──────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Encode 4-bit MLX experts to 1-bit QJL format")
    parser.add_argument('--model', default='~/Models/flash_mlx_4bit',
                        help='Path to flash_mlx_4bit model directory')
    parser.add_argument('--output', default=None,
                        help='Output directory (default: <model>/packed_experts_QJL)')
    parser.add_argument('--layers', default='all',
                        help='Comma-separated layer indices or "all"')
    parser.add_argument('--seed', type=int, default=42,
                        help='RNG seed for D diagonal vectors')
    args = parser.parse_args()

    model_path = os.path.expanduser(args.model)
    output_dir = args.output or os.path.join(model_path, 'packed_experts_QJL')
    os.makedirs(output_dir, exist_ok=True)

    if args.layers == 'all':
        layer_range = list(range(NUM_LAYERS))
    else:
        layer_range = [int(x.strip()) for x in args.layers.split(',')]

    # ── Generate D vectors (fixed seed, same for all layers) ──────────────────
    rng = np.random.default_rng(args.seed)
    D_shared_bits = rng.integers(0, 2, size=HIDDEN_DIM, dtype=np.uint8)
    D_down_bits   = rng.integers(0, 2, size=MOE_INTERMEDIATE, dtype=np.uint8)
    D_shared_packed = np.packbits(D_shared_bits, bitorder='little')  # 512 bytes
    D_down_packed   = np.packbits(D_down_bits,   bitorder='little')  # 128 bytes

    # Write d_vectors.bin (same D record repeated for all 60 layers)
    d_vec_path = os.path.join(output_dir, 'd_vectors.bin')
    with open(d_vec_path, 'wb') as dvf:
        for _ in range(NUM_LAYERS):
            dvf.write(D_shared_packed.tobytes())
            dvf.write(D_down_packed.tobytes())
    print(f"[d_vectors] Written: {d_vec_path} ({NUM_LAYERS}×{D_RECORD_BYTES} bytes)")
    print(f"  seed={args.seed}, D_shared entropy="
          f"{np.mean(D_shared_bits):.3f}, D_down entropy={np.mean(D_down_bits):.3f}")

    print(f"\nQJL expert size: {EXPERT_SIZE_QJL:,} bytes ({EXPERT_SIZE_QJL/1e6:.2f} MB) per expert")
    print(f"Layer file size: {EXPERT_SIZE_QJL*NUM_EXPERTS/1e6:.0f} MB per layer")
    print(f"Total: {EXPERT_SIZE_QJL*NUM_EXPERTS*NUM_LAYERS/1e9:.1f} GB for {len(layer_range)} layers\n")

    total_t0 = time.time()

    for layer_idx in layer_range:
        src_path = os.path.join(model_path, 'packed_experts', f'layer_{layer_idx:02d}.bin')
        if not os.path.exists(src_path):
            print(f"[SKIP] layer {layer_idx:02d}: {src_path} not found")
            continue

        out_path = os.path.join(output_dir, f'layer_{layer_idx:02d}.bin')
        layer_t0 = time.time()
        print(f"[layer {layer_idx:02d}] Loading {src_path} ...", flush=True)

        with open(src_path, 'rb') as f:
            data = f.read()

        assert len(data) == EXPERT_SIZE_4BIT * NUM_EXPERTS, \
            f"Expected {EXPERT_SIZE_4BIT*NUM_EXPERTS} bytes, got {len(data)}"

        with open(out_path, 'wb') as out_f:
            for eidx in range(NUM_EXPERTS):
                if eidx % 128 == 0:
                    elapsed = time.time() - layer_t0
                    print(f"  expert {eidx:3d}/512  ({elapsed:.0f}s elapsed)", flush=True)

                base = eidx * EXPERT_SIZE_4BIT

                # gate_proj: (out=1024, in=4096)
                gate_W = decode_4bit_matrix(data,
                    base + GATE_W_OFF, base + GATE_S_OFF, base + GATE_B_OFF,
                    MOE_INTERMEDIATE, HIDDEN_DIM)
                gate_norms, gate_signs = encode_matrix_qjl(gate_W, D_shared_bits)

                # up_proj: (out=1024, in=4096) — same D as gate
                up_W = decode_4bit_matrix(data,
                    base + UP_W_OFF, base + UP_S_OFF, base + UP_B_OFF,
                    MOE_INTERMEDIATE, HIDDEN_DIM)
                up_norms, up_signs = encode_matrix_qjl(up_W, D_shared_bits)

                # down_proj: (out=4096, in=1024)
                down_W = decode_4bit_matrix(data,
                    base + DOWN_W_OFF, base + DOWN_S_OFF, base + DOWN_B_OFF,
                    HIDDEN_DIM, MOE_INTERMEDIATE)
                down_norms, down_signs = encode_matrix_qjl(down_W, D_down_bits)

                # Write expert record
                out_f.write(gate_norms.tobytes())   # float32 × 1024  = 4,096 bytes
                out_f.write(gate_signs.tobytes())   # uint8 × 524,288 = 524,288 bytes
                out_f.write(up_norms.tobytes())
                out_f.write(up_signs.tobytes())
                out_f.write(down_norms.tobytes())   # float32 × 4096  = 16,384 bytes
                out_f.write(down_signs.tobytes())   # uint8 × 524,288

        sz = os.path.getsize(out_path)
        layer_t = time.time() - layer_t0
        print(f"  → {out_path}  {sz/1e6:.0f} MB  ({layer_t:.0f}s)\n", flush=True)

    total_t = time.time() - total_t0
    print(f"Done. Total time: {total_t/60:.1f} min")
    print(f"Output: {output_dir}")
    print(f"\nRun inference with:")
    print(f"  ./metal_infer/infer --model ~/Models/flash_mlx_4bit --qjl-experts \\")
    print(f"    --gguf-embedding ~/Models/flash_mlx_4bit/gguf/embedding_q8_0.bin \\")
    print(f"    --gguf-lm-head ~/Models/flash_mlx_4bit/gguf/lm_head_q6.bin \\")
    print(f"    --cache-io-split 4 --prompt 'Hello' --tokens 50 --stream")


if __name__ == '__main__':
    main()
