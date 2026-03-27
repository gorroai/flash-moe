#!/usr/bin/env python3
"""
validate_qjl.py — Verify QJL encoding round-trips correctly.

For a random x vector:
  1. Decode 4-bit expert weights W from packed_experts/layer_00.bin
  2. Compute reference output: y_ref = W @ x  (float32)
  3. Compute QJL estimate:
       y_qjl_i = (π/2 / d) * norm_i * dot(signs_i, WHT(D⊙x))
  4. Compare cosine similarity and relative error
"""
import numpy as np
import sys
import os

HIDDEN_DIM       = 4096
MOE_INTERMEDIATE = 1024
GROUP_SIZE       = 64

# 4-bit source offsets (per-expert, relative to expert base)
EXPERT_SIZE_4BIT = 7077888
GATE_W_OFF = 0;       GATE_S_OFF = 2097152;  GATE_B_OFF = 2228224
UP_W_OFF   = 2359296; UP_S_OFF   = 4456448;  UP_B_OFF   = 4587520
DOWN_W_OFF = 4718592; DOWN_S_OFF = 6815744;  DOWN_B_OFF = 6946816

# QJL offsets
QJL_GATE_NORMS_OFF = 0
QJL_GATE_SIGNS_OFF = QJL_GATE_NORMS_OFF + MOE_INTERMEDIATE * 4
QJL_UP_NORMS_OFF   = QJL_GATE_SIGNS_OFF + MOE_INTERMEDIATE * (HIDDEN_DIM // 8)
QJL_UP_SIGNS_OFF   = QJL_UP_NORMS_OFF   + MOE_INTERMEDIATE * 4
QJL_DOWN_NORMS_OFF = QJL_UP_SIGNS_OFF   + MOE_INTERMEDIATE * (HIDDEN_DIM // 8)
QJL_DOWN_SIGNS_OFF = QJL_DOWN_NORMS_OFF + HIDDEN_DIM * 4
EXPERT_SIZE_QJL    = QJL_DOWN_SIGNS_OFF + HIDDEN_DIM * (MOE_INTERMEDIATE // 8)

MODEL = os.path.expanduser("~/Models/flash_mlx_4bit")
LAYER = 0
EXPERT = 0  # test first expert


def bf16_to_f32(u16):
    return (u16.astype(np.uint32) << 16).view(np.float32)


def decode_4bit(data, w_off, s_off, b_off, out_dim, in_dim):
    n_packed  = in_dim // 8
    n_groups  = in_dim // GROUP_SIZE
    W = np.frombuffer(data, dtype=np.uint32, count=out_dim * n_packed, offset=w_off)
    W = W.reshape(out_dim, n_packed)
    S = bf16_to_f32(np.frombuffer(data, dtype=np.uint16, count=out_dim * n_groups, offset=s_off))
    S = S.reshape(out_dim, n_groups)
    B = bf16_to_f32(np.frombuffer(data, dtype=np.uint16, count=out_dim * n_groups, offset=b_off))
    B = B.reshape(out_dim, n_groups)
    result = np.zeros((out_dim, in_dim), dtype=np.float32)
    for bit in range(8):
        result[:, bit::8] = ((W >> (bit * 4)) & 0xF).astype(np.float32)
    S_exp = np.repeat(S, GROUP_SIZE, axis=1)
    B_exp = np.repeat(B, GROUP_SIZE, axis=1)
    return result * S_exp + B_exp


def wht(x):
    n = len(x)
    x = x.copy()
    h = 1
    while h < n:
        x = x.reshape(-1, 2 * h)
        a = x[:, :h].copy(); b = x[:, h:].copy()
        x[:, :h] = a + b; x[:, h:] = a - b
        x = x.reshape(-1)
        h *= 2
    return x


def qjl_decode(norms, sign_packed, D_bits, x_trans):
    """Compute QJL matvec estimate."""
    d = len(x_trans)
    n_bytes = d >> 3
    out_dim = len(norms)
    # Unpack signs
    sign_bits = np.unpackbits(sign_packed.reshape(out_dim, n_bytes), axis=1, bitorder='little')
    # Convert {0,1} → {-1,+1}
    signs = (sign_bits.astype(np.float32) * 2) - 1
    # dot product
    dots = signs @ x_trans   # (out_dim,)
    scale = (np.pi / 2) / d * norms
    return scale * dots


def main():
    print(f"Loading 4-bit expert layer_{LAYER:02d}.bin ...")
    src_path = f"{MODEL}/packed_experts/layer_{LAYER:02d}.bin"
    with open(src_path, 'rb') as f:
        data4 = f.read(EXPERT_SIZE_4BIT)  # just first expert
    base = 0

    print("Decoding 4-bit gate_proj ...")
    gate_W  = decode_4bit(data4, base+GATE_W_OFF, base+GATE_S_OFF, base+GATE_B_OFF,
                          MOE_INTERMEDIATE, HIDDEN_DIM)

    print("Loading QJL layer_{LAYER:02d}.bin ...")
    qjl_path = f"{MODEL}/packed_experts_QJL/layer_{LAYER:02d}.bin"
    with open(qjl_path, 'rb') as f:
        data_qjl = f.read(EXPERT_SIZE_QJL)  # first expert

    gate_norms = np.frombuffer(data_qjl, dtype=np.float32,
                               count=MOE_INTERMEDIATE, offset=QJL_GATE_NORMS_OFF)
    gate_signs = np.frombuffer(data_qjl, dtype=np.uint8,
                               count=MOE_INTERMEDIATE * (HIDDEN_DIM // 8),
                               offset=QJL_GATE_SIGNS_OFF).copy()
    gate_signs = gate_signs.reshape(MOE_INTERMEDIATE, HIDDEN_DIM // 8)

    print("Loading D vectors ...")
    dvec_path = f"{MODEL}/packed_experts_QJL/d_vectors.bin"
    with open(dvec_path, 'rb') as f:
        f.seek(LAYER * (HIDDEN_DIM//8 + MOE_INTERMEDIATE//8))
        D_shared_packed = np.frombuffer(f.read(HIDDEN_DIM // 8), dtype=np.uint8)
    D_shared_bits = np.unpackbits(D_shared_packed, bitorder='little').astype(np.float32)
    D_signs = D_shared_bits.astype(np.float32) * 2 - 1  # {0,1} → {-1,+1}

    # Random test vector
    rng = np.random.default_rng(123)
    x = rng.standard_normal(HIDDEN_DIM).astype(np.float32)

    # Reference output
    y_ref = gate_W @ x

    # QJL estimate
    x_scrambled = D_signs * x
    x_trans = wht(x_scrambled)
    y_qjl = qjl_decode(gate_norms, gate_signs, D_signs, x_trans)

    # Metrics
    cos_sim = np.dot(y_ref, y_qjl) / (np.linalg.norm(y_ref) * np.linalg.norm(y_qjl) + 1e-8)
    rel_err = np.linalg.norm(y_ref - y_qjl) / (np.linalg.norm(y_ref) + 1e-8)

    print(f"\ngate_proj validation (expert 0, layer 0):")
    print(f"  cosine similarity: {cos_sim:.4f}")
    print(f"  relative error:    {rel_err:.4f}")
    print(f"  y_ref[:5]:  {y_ref[:5]}")
    print(f"  y_qjl[:5]:  {y_qjl[:5]}")
    print(f"  ref norms match: {np.allclose(gate_norms, np.linalg.norm(gate_W, axis=1), rtol=1e-4)}")

    if cos_sim > 0.5:
        print("\nPASS: QJL encoding is numerically consistent.")
    else:
        print("\nFAIL: Cosine similarity too low — check encoding/decoding.")


if __name__ == '__main__':
    main()
