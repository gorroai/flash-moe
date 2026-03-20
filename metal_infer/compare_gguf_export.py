#!/usr/bin/env python3
"""
compare_gguf_export.py -- Compare Flash-MoE dense/shared export tensors against GGUF source.

This tool is meant to catch conversion mistakes quickly by comparing:
  - matrix matvec outputs from the exported model_weights.bin/json
  - direct GGUF dequantized tensors

It focuses on dense/shared tensors (the part currently suspected), not routed experts.
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from metal_infer import extract_weights_gguf as gguf_export  # type: ignore


MANIFEST_DTYPES = {
    "U32": np.uint32,
    "I8": np.int8,
    "F16": np.float16,
    "F32": np.float32,
    "BF16": np.uint16,
    "I32": np.int32,
}


def bf16_to_f32(arr_u16):
    arr = np.asarray(arr_u16, dtype=np.uint16)
    return (arr.astype(np.uint32) << 16).view(np.float32)


def alias_layer(alias):
    parts = alias.split(".")
    if len(parts) >= 3 and parts[0] == "model" and parts[1] == "layers":
        return int(parts[2])
    return None


def selected_aliases(layers):
    aliases = []
    for bucket in (gguf_export.quant_aliases(), gguf_export.bf16_aliases(), gguf_export.f32_aliases()):
        for alias in bucket:
            layer = alias_layer(alias)
            if layer is None or layer in layers:
                aliases.append(alias)
    return aliases


def load_manifest(path):
    with open(path) as f:
        return json.load(f)


def load_export_array(weights_path, manifest, name):
    meta = manifest["tensors"][name]
    dtype = MANIFEST_DTYPES[meta["dtype"]]
    shape = tuple(meta["shape"])
    arr = np.memmap(
        weights_path,
        mode="r",
        dtype=dtype,
        offset=meta["offset"],
        shape=shape,
        order="C",
    )
    if meta["dtype"] == "BF16":
        return bf16_to_f32(arr)
    if meta["dtype"] == "F16":
        return np.asarray(arr, dtype=np.float16).astype(np.float32)
    if meta["dtype"] == "F32":
        return np.asarray(arr, dtype=np.float32)
    return np.asarray(arr)


def vec_stats(ref, got):
    ref = np.asarray(ref, dtype=np.float32).reshape(-1)
    got = np.asarray(got, dtype=np.float32).reshape(-1)
    diff = got - ref
    mae = float(np.mean(np.abs(diff)))
    max_abs = float(np.max(np.abs(diff)))
    ref_mean_abs = float(np.mean(np.abs(ref))) + 1e-12
    ref_max_abs = float(np.max(np.abs(ref))) + 1e-12
    dot = float(np.dot(ref, got))
    n1 = float(np.linalg.norm(ref))
    n2 = float(np.linalg.norm(got))
    cos = dot / max(n1 * n2, 1e-12)
    return {
        "mae": mae,
        "max_abs": max_abs,
        "rel_mae": mae / ref_mean_abs,
        "rel_max": max_abs / ref_max_abs,
        "cos": cos,
        "ref_rms": float(np.sqrt(np.mean(ref * ref))),
        "got_rms": float(np.sqrt(np.mean(got * got))),
    }


def unpack_4bit_matrix(packed_u32, out_dim, in_dim):
    shifts = np.arange(8, dtype=np.uint32) * 4
    unpacked = ((packed_u32[..., None] >> shifts) & 0xF).astype(np.float32)
    return unpacked.reshape(out_dim, in_dim)


def exported_affine4_matvec(weights_path, manifest, alias, x):
    w = load_export_array(weights_path, manifest, alias).astype(np.uint32, copy=False)
    s = load_export_array(weights_path, manifest, alias.replace(".weight", ".scales")).astype(np.float32, copy=False)
    b = load_export_array(weights_path, manifest, alias.replace(".weight", ".biases")).astype(np.float32, copy=False)
    out_dim = manifest["tensors"][alias]["shape"][0]
    in_dim = s.shape[1] * 64
    vals = unpack_4bit_matrix(w, out_dim, in_dim).reshape(out_dim, -1, 64)
    xg = np.asarray(x, dtype=np.float32).reshape(-1, 64)
    dot_nibbles = np.einsum("obg,bg->ob", vals, xg, optimize=True)
    sum_x = np.sum(xg, axis=1, dtype=np.float32)[None, :]
    return np.sum(dot_nibbles * s + sum_x * b, axis=1, dtype=np.float32)


def exported_q8_matvec(weights_path, manifest, alias, x):
    w = load_export_array(weights_path, manifest, alias).astype(np.int8, copy=False)
    s = load_export_array(weights_path, manifest, alias.replace(".weight", ".scales")).astype(np.float32, copy=False)
    out_dim = manifest["tensors"][alias]["shape"][0]
    in_dim = w.shape[1]
    vals = w.reshape(out_dim, -1, 32).astype(np.float32)
    xg = np.asarray(x, dtype=np.float32).reshape(-1, 32)
    dot = np.einsum("obg,bg->ob", vals, xg, optimize=True)
    return np.sum(dot * s, axis=1, dtype=np.float32)


def exported_f32_matvec(weights_path, manifest, alias, x):
    w = load_export_array(weights_path, manifest, alias).astype(np.float32, copy=False)
    return w @ np.asarray(x, dtype=np.float32)


def compare_matrix(alias, tensor, gguf, weights_path, manifest, rng, samples):
    ref_w = gguf_export.tensor_to_f32(tensor, gguf)
    if ref_w.ndim == 1:
        ref_w = ref_w.reshape(1, ref_w.shape[0])
    ref_w = np.asarray(ref_w, dtype=np.float32)
    out_dim, in_dim = ref_w.shape

    ref_nonfinite = int(ref_w.size - np.count_nonzero(np.isfinite(ref_w)))
    if ref_nonfinite:
        return {
            "samples": 0,
            "export_dtype": manifest["tensors"][alias]["dtype"],
            "shape": [int(out_dim), int(in_dim)],
            "source_type": tensor.tensor_type.name,
            "mae": math.inf,
            "max_abs": math.inf,
            "rel_mae": math.inf,
            "rel_max": math.inf,
            "cos": -1.0,
            "ref_rms": math.inf,
            "got_rms": math.inf,
            "nonfinite_ref": ref_nonfinite,
            "nonfinite_export": 0,
        }

    dtype = manifest["tensors"][alias]["dtype"]
    if dtype == "F32":
        export_w = load_export_array(weights_path, manifest, alias).astype(np.float32, copy=False)
        export_nonfinite = int(export_w.size - np.count_nonzero(np.isfinite(export_w)))
        if export_nonfinite:
            return {
                "samples": 0,
                "export_dtype": dtype,
                "shape": [int(out_dim), int(in_dim)],
                "source_type": tensor.tensor_type.name,
                "mae": math.inf,
                "max_abs": math.inf,
                "rel_mae": math.inf,
                "rel_max": math.inf,
                "cos": -1.0,
                "ref_rms": float(np.sqrt(np.mean(ref_w * ref_w))),
                "got_rms": math.inf,
                "nonfinite_ref": 0,
                "nonfinite_export": export_nonfinite,
            }

    sample_stats = []
    for _ in range(samples):
        x = (rng.standard_normal(in_dim, dtype=np.float32) * 0.01).astype(np.float32)
        ref = ref_w @ x
        if dtype == "F32":
            got = exported_f32_matvec(weights_path, manifest, alias, x)
        elif dtype == "I8":
            got = exported_q8_matvec(weights_path, manifest, alias, x)
        elif dtype == "U32":
            got = exported_affine4_matvec(weights_path, manifest, alias, x)
        else:
            raise ValueError(f"{alias}: unsupported export dtype for matrix compare: {dtype}")
        if not np.all(np.isfinite(ref)) or not np.all(np.isfinite(got)):
            return {
                "samples": 0,
                "export_dtype": dtype,
                "shape": [int(out_dim), int(in_dim)],
                "source_type": tensor.tensor_type.name,
                "mae": math.inf,
                "max_abs": math.inf,
                "rel_mae": math.inf,
                "rel_max": math.inf,
                "cos": -1.0,
                "ref_rms": math.inf,
                "got_rms": math.inf,
                "nonfinite_ref": int(ref.size - np.count_nonzero(np.isfinite(ref))),
                "nonfinite_export": int(got.size - np.count_nonzero(np.isfinite(got))),
            }
        sample_stats.append(vec_stats(ref, got))

    worst = max(sample_stats, key=lambda item: item["rel_mae"])
    worst["samples"] = samples
    worst["export_dtype"] = dtype
    worst["shape"] = [int(out_dim), int(in_dim)]
    worst["source_type"] = tensor.tensor_type.name
    return worst


def compare_nonmatrix(alias, tensor, gguf, weights_path, manifest):
    ref = np.asarray(gguf_export.tensor_to_f32(tensor, gguf), dtype=np.float32)
    got = load_export_array(weights_path, manifest, alias).astype(np.float32, copy=False)
    if ref.ndim == 1 and got.ndim == 2 and got.shape[0] == 1 and got.shape[1] == ref.shape[0]:
        ref = ref.reshape(1, ref.shape[0])
    if ref.shape != got.shape:
        raise ValueError(f"{alias}: shape mismatch GGUF {ref.shape} vs export {got.shape}")
    stats = vec_stats(ref, got)
    stats["samples"] = 1
    stats["export_dtype"] = manifest["tensors"][alias]["dtype"]
    stats["shape"] = list(ref.shape)
    stats["source_type"] = tensor.tensor_type.name
    return stats


def main():
    parser = argparse.ArgumentParser(description="Compare Flash-MoE dense/shared export vs GGUF source")
    parser.add_argument("--gguf", required=True, help="GGUF directory or shard path")
    parser.add_argument("--weights", required=True, help="Exported model_weights.bin path")
    parser.add_argument("--manifest", required=True, help="Exported model_weights.json path")
    parser.add_argument("--layers", default="0-2", help="Layer range/spec to inspect (default: 0-2)")
    parser.add_argument("--samples", type=int, default=2, help="Random matvec samples per matrix")
    parser.add_argument("--top", type=int, default=20, help="How many worst tensors to print")
    parser.add_argument("--report", help="Optional JSON output path")
    args = parser.parse_args()

    rng = np.random.default_rng(1234)
    layers = set(gguf_export.flash_pack.parse_layers(args.layers))
    weights_path = os.path.expanduser(args.weights)
    manifest = load_manifest(os.path.expanduser(args.manifest))

    print("Loading GGUF...")
    gguf, _readers, tensor_map, _model_dir, gguf_files = gguf_export.flash_pack.load_gguf_tensors(args.gguf)
    name_map = gguf.get_tensor_name_map(gguf.MODEL_ARCH.QWEN35MOE, gguf_export.flash_pack.NUM_LAYERS)
    print(f"GGUF shards: {len(gguf_files)}")
    print(f"Comparing layers: {sorted(layers)}")

    results = []
    aliases = selected_aliases(layers)
    for idx, alias in enumerate(aliases, start=1):
        if alias not in manifest["tensors"]:
            continue
        tensor_name, tensor = gguf_export.resolve_tensor(alias, tensor_map, name_map)
        is_matrix = (
            alias.endswith(".weight")
            and alias.replace(".weight", ".scales") in manifest["tensors"]
            and alias.replace(".weight", ".biases") in manifest["tensors"]
        )
        if is_matrix:
            stats = compare_matrix(alias, tensor, gguf, weights_path, manifest, rng, args.samples)
            compare_kind = "matrix"
        else:
            stats = compare_nonmatrix(alias, tensor, gguf, weights_path, manifest)
            compare_kind = "tensor"

        results.append({
            "alias": alias,
            "layer": alias_layer(alias),
            "tensor_name": tensor_name,
            "compare_kind": compare_kind,
            **stats,
        })

        if idx % 20 == 0 or idx == len(aliases):
            print(f"  [{idx}/{len(aliases)}] {alias}")

    results.sort(key=lambda item: (item["rel_mae"], item["max_abs"]), reverse=True)

    print("\nWorst offenders by relative MAE:")
    for item in results[: args.top]:
        print(
            f"  L{item['layer'] if item['layer'] is not None else '--':>2} "
            f"{item['compare_kind']:6s} {item['alias']}\n"
            f"     src={item['source_type']} export={item['export_dtype']} shape={item['shape']} "
            f"rel_mae={item['rel_mae']:.6f} rel_max={item['rel_max']:.6f} "
            f"mae={item['mae']:.6f} max={item['max_abs']:.6f} cos={item['cos']:.8f}"
        )

    if args.report:
        report_path = os.path.expanduser(args.report)
        Path(report_path).parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w") as f:
            json.dump({
                "gguf": os.path.expanduser(args.gguf),
                "weights": weights_path,
                "manifest": os.path.expanduser(args.manifest),
                "layers": sorted(layers),
                "samples": args.samples,
                "results": results,
            }, f, indent=2)
        print(f"\nWrote report: {report_path}")


if __name__ == "__main__":
    main()
