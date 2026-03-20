#!/usr/bin/env python3
"""
extract_weights_gguf.py -- Convert GGUF non-expert tensors into Flash-MoE model_weights.bin/json.

This is a compatibility exporter: it dequantizes GGUF tensors and requantizes matrix
weights into Flash-MoE runtime formats.

Matrix export modes:
  - affine4
      .weight  -> U32 packed 4-bit values
      .scales  -> BF16
      .biases  -> BF16
  - q8_0
      .weight  -> I8
      .scales  -> F16
      .biases  -> F16 zeros (compatibility placeholder; ignored at runtime)
  - source_aware
      Q8_0 / Q6_K source matrices use q8_0, all others use affine4

Vectors and small state tensors are exported as BF16 or F32 to match the current runtime.
Routed experts are intentionally excluded; use ../repack_experts.py --gguf for those.
"""

import argparse
import fnmatch
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import repack_experts as flash_pack


QUANT_SUFFIXES = (".weight", ".bias", ".scales", ".biases")
Q8_COMPAT_SOURCE_TYPES = {"Q8_0", "Q6_K"}

GLOBAL_QUANT = [
    "model.embed_tokens.weight",
    "lm_head.weight",
]

GLOBAL_BF16 = [
    "model.norm.weight",
]

FULL_ATTN_QUANT = [
    "model.layers.{layer}.self_attn.q_proj.weight",
    "model.layers.{layer}.self_attn.k_proj.weight",
    "model.layers.{layer}.self_attn.v_proj.weight",
    "model.layers.{layer}.self_attn.o_proj.weight",
]

FULL_ATTN_BF16 = [
    "model.layers.{layer}.input_layernorm.weight",
    "model.layers.{layer}.post_attention_layernorm.weight",
    "model.layers.{layer}.self_attn.q_norm.weight",
    "model.layers.{layer}.self_attn.k_norm.weight",
]

LINEAR_ATTN_QUANT = [
    "model.layers.{layer}.linear_attn.in_proj_qkv.weight",
    "model.layers.{layer}.linear_attn.in_proj_z.weight",
    "model.layers.{layer}.linear_attn.in_proj_b.weight",
    "model.layers.{layer}.linear_attn.in_proj_a.weight",
    "model.layers.{layer}.linear_attn.out_proj.weight",
]

LINEAR_ATTN_BF16 = [
    "model.layers.{layer}.input_layernorm.weight",
    "model.layers.{layer}.post_attention_layernorm.weight",
    "model.layers.{layer}.linear_attn.conv1d.weight",
    "model.layers.{layer}.linear_attn.dt_bias",
    "model.layers.{layer}.linear_attn.norm.weight",
]

LINEAR_ATTN_F32 = [
    "model.layers.{layer}.linear_attn.A_log",
]

MOE_QUANT = [
    "model.layers.{layer}.mlp.gate.weight",
    "model.layers.{layer}.mlp.shared_expert.gate_proj.weight",
    "model.layers.{layer}.mlp.shared_expert.up_proj.weight",
    "model.layers.{layer}.mlp.shared_expert.down_proj.weight",
    "model.layers.{layer}.mlp.shared_expert_gate.weight",
]


def model_config():
    layer_types = []
    for i in range(60):
        layer_types.append("full_attention" if (i + 1) % 4 == 0 else "linear_attention")

    return {
        "hidden_size": 4096,
        "num_hidden_layers": 60,
        "num_attention_heads": 32,
        "num_key_value_heads": 2,
        "head_dim": 256,
        "vocab_size": 248320,
        "rms_norm_eps": 1e-6,
        "num_experts": 512,
        "num_experts_per_tok": 10,
        "moe_intermediate_size": 1024,
        "shared_expert_intermediate_size": 1024,
        "full_attention_interval": 4,
        "linear_num_value_heads": 64,
        "linear_num_key_heads": 16,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "linear_conv_kernel_dim": 4,
        "partial_rotary_factor": 0.25,
        "rope_theta": 10000000.0,
        "layer_types": layer_types,
    }


def quant_aliases():
    names = list(GLOBAL_QUANT)
    for layer in range(flash_pack.NUM_LAYERS):
        names.extend(name.format(layer=layer) for name in MOE_QUANT)
        if (layer + 1) % 4 == 0:
            names.extend(name.format(layer=layer) for name in FULL_ATTN_QUANT)
        else:
            names.extend(name.format(layer=layer) for name in LINEAR_ATTN_QUANT)
    return names


def bf16_aliases():
    names = list(GLOBAL_BF16)
    for layer in range(flash_pack.NUM_LAYERS):
        if (layer + 1) % 4 == 0:
            names.extend(name.format(layer=layer) for name in FULL_ATTN_BF16)
        else:
            names.extend(name.format(layer=layer) for name in LINEAR_ATTN_BF16)
    return names


def f32_aliases():
    names = []
    for layer in range(flash_pack.NUM_LAYERS):
        if (layer + 1) % 4 != 0:
            names.extend(name.format(layer=layer) for name in LINEAR_ATTN_F32)
    return names


def align_offset(out_f, offset, alignment):
    if offset % alignment != 0:
        pad = alignment - (offset % alignment)
        out_f.write(b"\x00" * pad)
        offset += pad
    return offset


def resolve_standard_name(alias, name_map):
    std_name = name_map.get_name(alias, try_suffixes=QUANT_SUFFIXES)
    if std_name is not None:
        return std_name

    if ".linear_attn.dt_bias" in alias:
        layer = alias.split(".")[2]
        return f"blk.{layer}.ssm_dt.bias"

    raise KeyError(f"Could not map runtime tensor alias to GGUF tensor: {alias}")


def tensor_to_f32(tensor, gguf):
    raw = np.asarray(tensor.data)
    if tensor.tensor_type.name == "F32":
        return np.asarray(raw, dtype=np.float32)
    if tensor.tensor_type.name == "F16":
        # Some GGUF reader paths expose raw bytes for 16-bit tensors.
        if raw.dtype == np.uint8:
            raw = raw.view(np.float16)
        else:
            raw = np.asarray(raw, dtype=np.float16)
        return raw.astype(np.float32)
    if tensor.tensor_type.name == "BF16":
        # BF16 tensors may also arrive as byte-shaped uint8 buffers; reinterpret
        # them first so the logical matrix width stays correct.
        if raw.dtype == np.uint8:
            raw = raw.view(np.uint16)
        else:
            raw = np.asarray(raw, dtype=np.uint16)
        return (raw.astype(np.uint32) << 16).view(np.float32)
    return gguf.dequantize(tensor.data, tensor.tensor_type).astype(np.float32, copy=False)


def resolve_tensor(alias, tensor_map, name_map):
    std_name = resolve_standard_name(alias, name_map)
    tensor = tensor_map.get(std_name)
    if tensor is None:
        raise KeyError(f"GGUF tensor not found for {alias}: {std_name}")
    return std_name, tensor


def add_manifest_entry(manifest, name, offset, arr, dtype):
    manifest["tensors"][name] = {
        "offset": offset,
        "size": int(arr.nbytes),
        "shape": list(arr.shape),
        "dtype": dtype,
    }


def should_keep_matrix_f32(alias, patterns):
    if not patterns:
        return False
    return any(fnmatch.fnmatch(alias, pat) or pat in alias for pat in patterns)


def export_quant_tensor(out_f, offset, alias, tensor, gguf, manifest, matrix_format, keep_f32_patterns):
    if should_keep_matrix_f32(alias, keep_f32_patterns):
        return export_f32_matvec_tensor(out_f, offset, alias, tensor, gguf, manifest)
    if matrix_format == "q8_0":
        return export_q8_0_tensor(out_f, offset, alias, tensor, gguf, manifest)
    if matrix_format == "source_aware" and tensor.tensor_type.name in Q8_COMPAT_SOURCE_TYPES:
        return export_q8_0_tensor(out_f, offset, alias, tensor, gguf, manifest)

    weights = tensor_to_f32(tensor, gguf)
    if weights.ndim == 1:
        weights = weights.reshape(1, weights.shape[0])
    if weights.ndim != 2:
        raise ValueError(f"{alias}: expected 2D matrix, got {weights.shape}")

    out_dim, in_dim = weights.shape
    if in_dim % flash_pack.GROUP_SIZE != 0:
        raise ValueError(
            f"{alias}: input dim {in_dim} is not divisible by group size {flash_pack.GROUP_SIZE}"
        )

    packed_w, packed_s, packed_b = flash_pack.quantize_affine_4bit(weights, out_dim, in_dim)
    for name, arr, dtype in (
        (alias, packed_w, "U32"),
        (alias.replace(".weight", ".scales"), packed_s, "BF16"),
        (alias.replace(".weight", ".biases"), packed_b, "BF16"),
    ):
        offset = align_offset(out_f, offset, 64)
        out_f.write(arr.tobytes())
        add_manifest_entry(manifest, name, offset, arr, dtype)
        offset += arr.nbytes
    return offset


def export_f32_matvec_tensor(out_f, offset, alias, tensor, gguf, manifest):
    weights = np.asarray(tensor_to_f32(tensor, gguf), dtype=np.float32)
    if weights.ndim == 1:
        weights = weights.reshape(1, weights.shape[0])
    if weights.ndim != 2:
        raise ValueError(f"{alias}: expected 2D matrix, got {weights.shape}")

    out_dim, _in_dim = weights.shape
    dummy = np.zeros((out_dim, 1), dtype=np.float16)
    for name, arr, dtype in (
        (alias, weights, "F32"),
        (alias.replace(".weight", ".scales"), dummy, "F16"),
        (alias.replace(".weight", ".biases"), dummy, "F16"),
    ):
        offset = align_offset(out_f, offset, 64)
        out_f.write(arr.tobytes())
        add_manifest_entry(manifest, name, offset, arr, dtype)
        offset += arr.nbytes
    return offset


def export_q8_0_tensor(out_f, offset, alias, tensor, gguf, manifest):
    weights = tensor_to_f32(tensor, gguf)
    if weights.ndim == 1:
        weights = weights.reshape(1, weights.shape[0])
    if weights.ndim != 2:
        raise ValueError(f"{alias}: expected 2D matrix, got {weights.shape}")

    out_dim, in_dim = weights.shape
    if in_dim % flash_pack.Q8_BLOCK_SIZE != 0:
        raise ValueError(
            f"{alias}: input dim {in_dim} is not divisible by Q8 block size {flash_pack.Q8_BLOCK_SIZE}"
        )

    packed_w, packed_s = flash_pack.quantize_q8_0(weights, out_dim, in_dim)
    zero_bias = np.zeros_like(packed_s, dtype=np.float16)
    for name, arr, dtype in (
        (alias, packed_w, "I8"),
        (alias.replace(".weight", ".scales"), packed_s, "F16"),
        (alias.replace(".weight", ".biases"), zero_bias, "F16"),
    ):
        offset = align_offset(out_f, offset, 64)
        out_f.write(arr.tobytes())
        add_manifest_entry(manifest, name, offset, arr, dtype)
        offset += arr.nbytes
    return offset


def export_bf16_tensor(out_f, offset, alias, tensor, gguf, manifest):
    f32 = tensor_to_f32(tensor, gguf)
    bf16 = flash_pack.f32_to_bf16(f32)
    offset = align_offset(out_f, offset, 64)
    out_f.write(bf16.tobytes())
    add_manifest_entry(manifest, alias, offset, bf16, "BF16")
    return offset + bf16.nbytes


def export_f32_tensor(out_f, offset, alias, tensor, gguf, manifest):
    f32 = np.asarray(tensor_to_f32(tensor, gguf), dtype=np.float32)
    if alias.endswith(".linear_attn.A_log"):
        # GGUF stores Qwen3.5's delta-net decay term as ssm_a = -exp(A_log).
        # Flash-MoE's runtime expects the original A_log and computes
        # exp(-exp(A_log) * softplus(...)) internally, so convert back here.
        f32 = np.log(np.maximum(-f32, 1e-30)).astype(np.float32, copy=False)
    offset = align_offset(out_f, offset, 64)
    out_f.write(f32.tobytes())
    add_manifest_entry(manifest, alias, offset, f32, "F32")
    return offset + f32.nbytes


def summarize_export_precision(aliases, tensor_map, name_map, max_source_bits=4):
    """Inspect GGUF source tensor types for the non-expert export set."""
    counts = {}
    offenders = []
    unknown = []

    for alias in aliases:
        std_name, tensor = resolve_tensor(alias, tensor_map, name_map)
        type_name = tensor.tensor_type.name
        counts[type_name] = counts.get(type_name, 0) + 1

        bit_ceiling = flash_pack.quant_type_bit_ceiling(type_name)
        if bit_ceiling is None:
            unknown.append({
                "alias": alias,
                "tensor_name": std_name,
                "type_name": type_name,
            })
        elif bit_ceiling > max_source_bits:
            offenders.append({
                "alias": alias,
                "tensor_name": std_name,
                "type_name": type_name,
                "bit_ceiling": bit_ceiling,
            })

    flash_pack.print_source_precision_summary(
        "non-expert", counts, offenders, unknown, max_source_bits=max_source_bits
    )
    return counts, offenders, unknown


def main():
    parser = argparse.ArgumentParser(
        description="Convert GGUF non-expert tensors into Flash-MoE model_weights.bin/json"
    )
    parser.add_argument("--gguf", required=True,
                        help="Path to GGUF directory or one GGUF shard")
    parser.add_argument("--output", default=".",
                        help="Output directory for model_weights.bin and model_weights.json")
    parser.add_argument("--matrix-format", choices=["source_aware", "affine4", "q8_0"],
                        default="source_aware",
                        help="Dense/shared matrix export format "
                             "(source_aware: Q8_0/Q6_K -> q8_0, others -> affine4)")
    parser.add_argument("--keep-f32-pattern", action="append", default=[],
                        help="Glob or substring for matrix aliases to preserve as F32")
    parser.add_argument("--keep-router-f32", action="store_true",
                        help="Preserve MoE router and shared-expert gate matrices as F32")
    parser.add_argument("--dry-run", action="store_true",
                        help="Resolve and validate tensors without writing output files")
    args = parser.parse_args()

    keep_f32_patterns = list(args.keep_f32_pattern)
    if args.keep_router_f32:
        keep_f32_patterns.extend([
            "model.layers.*.mlp.gate.weight",
            "model.layers.*.mlp.shared_expert_gate.weight",
        ])

    output_dir = Path(os.path.expanduser(args.output))
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading GGUF tensors...")
    gguf, readers, tensor_map, model_dir, gguf_files = flash_pack.load_gguf_tensors(args.gguf)
    name_map = gguf.get_tensor_name_map(gguf.MODEL_ARCH.QWEN35MOE, flash_pack.NUM_LAYERS)

    print(f"GGUF shards: {len(gguf_files)}")
    print(f"Tensors loaded: {len(tensor_map)}")

    quant_names = quant_aliases()
    bf16_names = bf16_aliases()
    f32_names = f32_aliases()
    all_names = quant_names + bf16_names + f32_names

    missing = []
    for alias in all_names:
        try:
            resolve_tensor(alias, tensor_map, name_map)
        except KeyError as exc:
            missing.append(str(exc))

    if missing:
        print("ERROR: unresolved GGUF tensor mappings:")
        for item in missing[:20]:
            print(f"  {item}")
        if len(missing) > 20:
            print(f"  ... and {len(missing) - 20} more")
        sys.exit(1)

    counts, offenders, unknown = summarize_export_precision(
        all_names, tensor_map, name_map, max_source_bits=4
    )
    report_path = output_dir / "gguf_non_expert_precision_report.json"
    flash_pack.write_source_precision_report(
        str(report_path),
        "non-expert",
        counts,
        offenders,
        unknown,
        max_source_bits=4,
    )

    if args.dry_run:
        print(f"Dry run OK: {len(quant_names)} quantized matrices, "
              f"{len(bf16_names)} BF16 tensors, {len(f32_names)} F32 tensors")
        return

    bin_path = output_dir / "model_weights.bin"
    json_path = output_dir / "model_weights.json"
    manifest = {
        "model": str(args.gguf),
        "num_tensors": 0,
        "tensors": {},
        "config": model_config(),
    }

    t0 = time.time()
    offset = 0

    with open(bin_path, "wb") as out_f:
        for idx, alias in enumerate(quant_names, start=1):
            std_name, tensor = resolve_tensor(alias, tensor_map, name_map)
            offset = export_quant_tensor(
                out_f, offset, alias, tensor, gguf, manifest, args.matrix_format, keep_f32_patterns
            )
            if idx % 25 == 0 or idx == len(quant_names):
                print(f"  [quant {idx}/{len(quant_names)}] {offset / 1e9:.2f} GB")

        for idx, alias in enumerate(bf16_names, start=1):
            std_name, tensor = resolve_tensor(alias, tensor_map, name_map)
            offset = export_bf16_tensor(out_f, offset, alias, tensor, gguf, manifest)
            if idx % 100 == 0 or idx == len(bf16_names):
                print(f"  [bf16 {idx}/{len(bf16_names)}] {offset / 1e9:.2f} GB")

        for idx, alias in enumerate(f32_names, start=1):
            std_name, tensor = resolve_tensor(alias, tensor_map, name_map)
            offset = export_f32_tensor(out_f, offset, alias, tensor, gguf, manifest)
            if idx % 100 == 0 or idx == len(f32_names):
                print(f"  [f32 {idx}/{len(f32_names)}] {offset / 1e9:.2f} GB")

    manifest["num_tensors"] = len(manifest["tensors"])
    with open(json_path, "w") as f:
        json.dump(manifest, f, indent=2)

    elapsed = time.time() - t0
    total_bytes = bin_path.stat().st_size
    print(f"Done: {total_bytes / 1e9:.2f} GB in {elapsed:.1f}s "
          f"({total_bytes / max(elapsed, 1e-9) / 1e9:.2f} GB/s)")
    print(f"Binary: {bin_path}")
    print(f"Manifest: {json_path}")


if __name__ == "__main__":
    main()
