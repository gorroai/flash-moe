#!/usr/bin/env python3
"""Repack expert weights into contiguous per-layer binary files.

Creates one binary file per layer: packed_experts/layer_XX.bin
Each file = 512 experts x 7,077,888 bytes = ~3.63 GB
Expert E starts at byte offset E * 7,077,888

Within each expert block, 9 components packed in fixed order:
  gate_proj.weight, gate_proj.scales, gate_proj.biases,
  up_proj.weight,   up_proj.scales,   up_proj.biases,
  down_proj.weight,  down_proj.scales,  down_proj.biases

Source modes:
  - safetensors + expert_index.json: byte-for-byte repack from the original MLX layout
  - GGUF: dequantize routed experts per expert, then requantize into Flash-MoE's affine 4-bit format

Usage:
    python repack_experts.py                          # repack all 60 layers from expert_index.json
    python repack_experts.py --layers 0-4             # repack layers 0-4
    python repack_experts.py --layers 0,5,10          # repack specific layers
    python repack_experts.py --dry-run                # verify inputs without writing
    python repack_experts.py --verify-only 0          # verify layer 0 against original safetensors
    python repack_experts.py --gguf /path/to/gguf     # export routed experts from GGUF shards
"""

import argparse
import glob
import json
import os
import time
import sys
import numpy as np

# Component order and expected sizes
COMPONENTS = [
    {"name": "gate_proj.weight",  "offset": 0,       "size": 2097152, "dtype": "U32", "shape": [1024, 512]},
    {"name": "gate_proj.scales",  "offset": 2097152,  "size": 131072,  "dtype": "BF16", "shape": [1024, 64]},
    {"name": "gate_proj.biases",  "offset": 2228224,  "size": 131072,  "dtype": "BF16", "shape": [1024, 64]},
    {"name": "up_proj.weight",    "offset": 2359296,  "size": 2097152, "dtype": "U32", "shape": [1024, 512]},
    {"name": "up_proj.scales",    "offset": 4456448,  "size": 131072,  "dtype": "BF16", "shape": [1024, 64]},
    {"name": "up_proj.biases",    "offset": 4587520,  "size": 131072,  "dtype": "BF16", "shape": [1024, 64]},
    {"name": "down_proj.weight",  "offset": 4718592,  "size": 2097152, "dtype": "U32", "shape": [4096, 128]},
    {"name": "down_proj.scales",  "offset": 6815744,  "size": 131072,  "dtype": "BF16", "shape": [4096, 16]},
    {"name": "down_proj.biases",  "offset": 6946816,  "size": 131072,  "dtype": "BF16", "shape": [4096, 16]},
]

EXPERT_SIZE = 7077888   # bytes per expert
NUM_EXPERTS = 512
NUM_LAYERS = 60
LAYER_SIZE = NUM_EXPERTS * EXPERT_SIZE  # 3,623,878,656 bytes (~3.63 GB)
GROUP_SIZE = 64

Q8_DOWN_OUT_DIM = 4096
Q8_DOWN_IN_DIM = 1024
Q8_BLOCK_SIZE = 32
Q8_DOWN_WEIGHT_SIZE = Q8_DOWN_OUT_DIM * Q8_DOWN_IN_DIM
Q8_DOWN_SCALE_SIZE = Q8_DOWN_OUT_DIM * (Q8_DOWN_IN_DIM // Q8_BLOCK_SIZE) * 2
Q8_DOWN_W_OFF = 0
Q8_DOWN_S_OFF = Q8_DOWN_W_OFF + Q8_DOWN_WEIGHT_SIZE
Q8_DOWN_EXPERT_SIZE = Q8_DOWN_S_OFF + Q8_DOWN_SCALE_SIZE
Q8_DOWN_LAYER_SIZE = NUM_EXPERTS * Q8_DOWN_EXPERT_SIZE

PROJECTIONS = [
    {"prefix": "gate_proj", "out_dim": 1024, "in_dim": 4096},
    {"prefix": "up_proj",   "out_dim": 1024, "in_dim": 4096},
    {"prefix": "down_proj", "out_dim": 4096, "in_dim": 1024},
]

GGUF_EXPERT_TENSOR_PATTERNS = {
    "gate_proj": [
        "blk.{layer}.ffn_gate_exps.weight",
        "model.layers.{layer}.mlp.experts.gate_proj",
        "model.layers.{layer}.moe.gate_proj",
    ],
    "up_proj": [
        "blk.{layer}.ffn_up_exps.weight",
        "model.layers.{layer}.mlp.experts.up_proj",
        "model.layers.{layer}.moe.up_proj",
    ],
    "down_proj": [
        "blk.{layer}.ffn_down_exps.weight",
        "model.layers.{layer}.mlp.experts.down_proj",
        "model.layers.{layer}.moe.down_proj",
    ],
}

QUANT_TYPE_BIT_CEILINGS = {
    "IQ1_M": 1,
    "IQ1_S": 1,
    "TQ1_0": 1,
    "IQ2_S": 2,
    "IQ2_XS": 2,
    "IQ2_XXS": 2,
    "Q2_K": 2,
    "TQ2_0": 2,
    "IQ3_S": 3,
    "IQ3_XXS": 3,
    "Q3_K": 3,
    "IQ4_NL": 4,
    "IQ4_XS": 4,
    "MXFP4": 4,
    "Q4_0": 4,
    "Q4_1": 4,
    "Q4_K": 4,
    "Q5_0": 5,
    "Q5_1": 5,
    "Q5_K": 5,
    "Q6_K": 6,
    "Q8_0": 8,
    "Q8_1": 8,
    "Q8_K": 8,
    "I8": 8,
    "BF16": 16,
    "F16": 16,
    "I16": 16,
    "F32": 32,
    "I32": 32,
    "F64": 64,
    "I64": 64,
}


def parse_layers(spec):
    """Parse layer specification like '0-4' or '0,5,10' or 'all'."""
    return parse_index_spec(spec, NUM_LAYERS, "layer")


def parse_experts(spec):
    """Parse expert specification like '0-4' or '0,5,10' or 'all'."""
    return parse_index_spec(spec, NUM_EXPERTS, "expert")


def parse_index_spec(spec, upper_bound, label):
    """Parse a comma/range spec and validate indices are within [0, upper_bound)."""
    if spec is None or spec == 'all':
        return list(range(upper_bound))

    values = []
    for part in spec.split(','):
        part = part.strip()
        if not part:
            continue
        if '-' in part:
            a, b = part.split('-', 1)
            start = int(a)
            end = int(b)
            values.extend(range(start, end + 1))
        else:
            values.append(int(part))

    values = sorted(set(values))
    if not values:
        raise ValueError(f"No {label} indices parsed from spec: {spec}")
    for value in values:
        if value < 0 or value >= upper_bound:
            raise ValueError(
                f"{label} index {value} out of range [0, {upper_bound - 1}] for spec: {spec}"
            )
    return values


def load_index(index_path):
    """Load expert_index.json and return expert_reads dict + model_path."""
    index_path = os.path.expanduser(index_path)
    with open(index_path) as f:
        idx = json.load(f)
    return idx['expert_reads'], os.path.expanduser(idx['model_path'])


def require_gguf():
    """Import gguf lazily so the safetensors path stays dependency-free."""
    try:
        import gguf  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "ERROR: GGUF export requires the optional 'gguf' Python package.\n"
            "Install it with: python3 -m pip install gguf"
        ) from exc
    return gguf


def collect_gguf_files(path_spec):
    """Resolve a GGUF directory or shard path to an ordered shard list."""
    path_spec = os.path.expanduser(path_spec)
    if os.path.isdir(path_spec):
        files = sorted(glob.glob(os.path.join(path_spec, "*.gguf")))
    elif path_spec.endswith(".gguf"):
        files = sorted(glob.glob(os.path.join(os.path.dirname(path_spec), "*.gguf")))
        if not files:
            files = [path_spec]
    else:
        raise ValueError(f"GGUF path is neither a directory nor a .gguf file: {path_spec}")

    if not files:
        raise FileNotFoundError(f"No .gguf files found under {path_spec}")
    return files


def load_gguf_tensors(path_spec):
    """Load all GGUF shards and build a name -> tensor map."""
    gguf = require_gguf()
    files = collect_gguf_files(path_spec)

    readers = []
    tensor_map = {}
    for gguf_path in files:
        reader = gguf.GGUFReader(gguf_path)
        readers.append(reader)
        for tensor in reader.tensors:
            if tensor.name in tensor_map:
                raise ValueError(f"Duplicate tensor in GGUF shards: {tensor.name}")
            tensor_map[tensor.name] = tensor

    return gguf, readers, tensor_map, os.path.dirname(files[0]), files


def logical_tensor_shape(tensor, gguf):
    """Return the logical tensor shape after undoing GGUF byte packing."""
    if tensor.data.dtype == np.uint8:
        return tuple(gguf.quant_shape_from_byte_shape(tensor.data.shape, tensor.tensor_type))
    return tuple(tensor.data.shape)


def resolve_gguf_expert_tensor(tensor_map, layer_idx, prefix):
    """Find the routed-expert tensor for one layer/projection."""
    candidates = [pat.format(layer=layer_idx) for pat in GGUF_EXPERT_TENSOR_PATTERNS[prefix]]
    for name in candidates:
        tensor = tensor_map.get(name)
        if tensor is not None:
            return name, tensor
    raise KeyError(
        f"Missing GGUF routed expert tensor for layer {layer_idx}, {prefix}. "
        f"Tried: {', '.join(candidates)}"
    )


def quant_type_bit_ceiling(type_name):
    """Return a conservative bit-width class for a GGUF tensor type name."""
    return QUANT_TYPE_BIT_CEILINGS.get(type_name)


def write_source_precision_report(report_path, scope_label, counts, offenders, unknown, max_source_bits=4):
    """Write a JSON report describing GGUF source tensor precision."""
    report_path = os.path.expanduser(report_path)
    report_dir = os.path.dirname(report_path)
    if report_dir:
        os.makedirs(report_dir, exist_ok=True)

    payload = {
        "scope": scope_label,
        "max_source_bits": int(max_source_bits),
        "counts": {name: int(count) for name, count in sorted(counts.items())},
        "summary": {
            "num_types": int(len(counts)),
            "num_offenders": int(len(offenders)),
            "num_unknown": int(len(unknown)),
        },
        "offenders": offenders,
        "unknown": unknown,
    }

    with open(report_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote source precision report: {report_path}")


def print_source_precision_summary(scope_label, counts, offenders, unknown, max_source_bits=4):
    """Print a human-readable summary of GGUF source tensor precision."""
    print(f"GGUF {scope_label} source types:")
    for type_name in sorted(counts):
        bit_ceiling = quant_type_bit_ceiling(type_name)
        suffix = f" (~<={bit_ceiling}-bit class)" if bit_ceiling is not None else " (unknown bit class)"
        print(f"  {type_name}: {counts[type_name]} tensors{suffix}")

    if offenders:
        print(
            f"WARNING: Found {len(offenders)} {scope_label} source tensors above "
            f"{max_source_bits}-bit class:"
        )
        for entry in offenders[:12]:
            details = []
            if "layer" in entry:
                details.append(f"layer {entry['layer']:2d}")
            if "projection" in entry:
                details.append(f"{entry['projection']:9s}")
            if "alias" in entry:
                details.append(entry["alias"])
            details_text = " ".join(details)
            print(
                f"  {details_text} -> {entry['tensor_name']} "
                f"({entry['type_name']}, ~<={entry['bit_ceiling']}-bit class)"
            )
        if len(offenders) > 12:
            print(f"  ... and {len(offenders) - 12} more")
    else:
        print(f"GGUF {scope_label} source check: no tensors exceed {max_source_bits}-bit class")

    if unknown:
        print(
            f"WARNING: Could not classify {len(unknown)} {scope_label} source tensors by bit-width:"
        )
        for entry in unknown[:12]:
            details = []
            if "layer" in entry:
                details.append(f"layer {entry['layer']:2d}")
            if "projection" in entry:
                details.append(f"{entry['projection']:9s}")
            if "alias" in entry:
                details.append(entry["alias"])
            details_text = " ".join(details)
            print(f"  {details_text} -> {entry['tensor_name']} ({entry['type_name']})")
        if len(unknown) > 12:
            print(f"  ... and {len(unknown) - 12} more")


def summarize_gguf_expert_precision(tensor_map, layers, max_source_bits=4):
    """Inspect routed-expert source tensor types and flag anything above the threshold."""
    counts = {}
    offenders = []
    unknown = []

    for layer_idx in layers:
        for proj in PROJECTIONS:
            tensor_name, tensor = resolve_gguf_expert_tensor(tensor_map, layer_idx, proj["prefix"])
            type_name = tensor.tensor_type.name
            counts[type_name] = counts.get(type_name, 0) + 1

            bit_ceiling = quant_type_bit_ceiling(type_name)
            if bit_ceiling is None:
                unknown.append({
                    "layer": layer_idx,
                    "projection": proj["prefix"],
                    "tensor_name": tensor_name,
                    "type_name": type_name,
                })
            elif bit_ceiling > max_source_bits:
                offenders.append({
                    "layer": layer_idx,
                    "projection": proj["prefix"],
                    "tensor_name": tensor_name,
                    "type_name": type_name,
                    "bit_ceiling": bit_ceiling,
                })

    print_source_precision_summary(
        "routed-expert", counts, offenders, unknown, max_source_bits=max_source_bits
    )

    return offenders, unknown, counts


def f32_to_bf16(f32):
    """Convert float32 array to uint16 BF16 bit patterns by truncation."""
    arr = np.asarray(f32, dtype=np.float32)
    return (arr.view(np.uint32) >> 16).astype(np.uint16)


def pack_4bit(vals):
    """Pack 8 x 4-bit values into each uint32, LSB-first."""
    vals = np.asarray(vals, dtype=np.uint8)
    if vals.shape[-1] % 8 != 0:
        raise ValueError(f"Last dimension {vals.shape[-1]} must be divisible by 8")

    flat = vals.reshape(-1, vals.shape[-1])
    packed_cols = vals.shape[-1] // 8
    out = np.zeros((flat.shape[0], packed_cols), dtype=np.uint32)
    for i in range(8):
        out |= flat[:, i::8].astype(np.uint32) << (i * 4)
    return out.reshape(vals.shape[:-1] + (packed_cols,))


def quantize_q8_0(weights, out_dim, in_dim):
    """Quantize one expert projection to GGUF-style Q8_0 blocks."""
    weights = np.asarray(weights, dtype=np.float32)
    if weights.shape == (in_dim, out_dim):
        weights = weights.T
    if weights.shape != (out_dim, in_dim):
        raise ValueError(f"Expected weights shape {(out_dim, in_dim)}, got {weights.shape}")

    if in_dim % Q8_BLOCK_SIZE != 0:
        raise ValueError(f"Input dim {in_dim} must be divisible by Q8 block size {Q8_BLOCK_SIZE}")

    num_blocks = in_dim // Q8_BLOCK_SIZE
    grouped = weights.reshape(out_dim, num_blocks, Q8_BLOCK_SIZE)
    max_abs = np.max(np.abs(grouped), axis=2, keepdims=True)
    degenerate = (max_abs <= 1e-12)
    safe_max = np.where(degenerate, 1.0, max_abs)
    scales = np.where(degenerate, 0.0, safe_max / 127.0).astype(np.float32)
    inv = np.where(degenerate, 0.0, 127.0 / safe_max).astype(np.float32)

    q = np.rint(grouped * inv).astype(np.int32)
    q = np.clip(q, -127, 127).astype(np.int8)
    q[degenerate.repeat(Q8_BLOCK_SIZE, axis=2)] = 0

    return (
        q.reshape(out_dim, in_dim).view(np.uint8),
        scales.squeeze(axis=2).astype(np.float16),
    )


def quantize_affine_4bit(weights, out_dim, in_dim):
    """Quantize one expert projection to Flash-MoE's affine 4-bit layout."""
    weights = np.asarray(weights, dtype=np.float32)
    if weights.shape == (in_dim, out_dim):
        weights = weights.T
    if weights.shape != (out_dim, in_dim):
        raise ValueError(f"Expected weights shape {(out_dim, in_dim)}, got {weights.shape}")

    num_groups = in_dim // GROUP_SIZE
    grouped = weights.reshape(out_dim, num_groups, GROUP_SIZE)
    mins = grouped.min(axis=2, keepdims=True)
    maxs = grouped.max(axis=2, keepdims=True)
    scales = (maxs - mins) / 15.0

    degenerate = (scales == 0.0)
    safe_scales = scales.copy()
    safe_scales[degenerate] = 1.0

    q = np.rint((grouped - mins) / safe_scales).astype(np.int16)
    q = np.clip(q, 0, 15).astype(np.uint8)
    q[degenerate.repeat(GROUP_SIZE, axis=2)] = 0

    packed = pack_4bit(q.reshape(out_dim, in_dim))
    return (
        packed,
        f32_to_bf16(scales.squeeze(axis=2)),
        f32_to_bf16(mins.squeeze(axis=2)),
    )


def verify_component_sizes(expert_reads):
    """Verify that component sizes in the index match expected sizes."""
    expected = {c['name']: c['size'] for c in COMPONENTS}
    for layer_key, comps in expert_reads.items():
        for comp_name, info in comps.items():
            if comp_name not in expected:
                print(f"WARNING: unknown component {comp_name} in layer {layer_key}")
                continue
            if info['expert_size'] != expected[comp_name]:
                print(f"MISMATCH: layer {layer_key}, {comp_name}: "
                      f"index says {info['expert_size']}, expected {expected[comp_name]}")
                return False
    print("Component sizes verified: all match expected layout")
    return True


def open_source_files(expert_reads, model_path, layers):
    """Open all needed safetensors files, return {filename: fd}."""
    needed_files = set()
    for layer_idx in layers:
        layer_key = str(layer_idx)
        if layer_key not in expert_reads:
            print(f"WARNING: layer {layer_idx} not found in expert_reads")
            continue
        for info in expert_reads[layer_key].values():
            needed_files.add(info['file'])

    fds = {}
    for fname in sorted(needed_files):
        path = os.path.join(model_path, fname)
        fds[fname] = os.open(path, os.O_RDONLY)
    print(f"Opened {len(fds)} source safetensors files")
    return fds


def repack_layer(layer_idx, expert_reads, model_path, fds, output_dir, expert_indices=None, dry_run=False):
    """Repack all 512 experts for one layer into a contiguous binary file.

    Returns (bytes_written, elapsed_seconds).
    """
    if expert_indices is None:
        expert_indices = list(range(NUM_EXPERTS))

    layer_key = str(layer_idx)
    if layer_key not in expert_reads:
        print(f"  Layer {layer_idx}: NOT FOUND in index, skipping")
        return 0, 0.0

    layer_info = expert_reads[layer_key]
    out_path = os.path.join(output_dir, f"layer_{layer_idx:02d}.bin")

    if dry_run:
        # Just verify we can compute all offsets
        for expert_idx in expert_indices:
            for comp in COMPONENTS:
                info = layer_info[comp['name']]
                src_offset = info['abs_offset'] + expert_idx * info['expert_stride']
                dst_offset = expert_idx * EXPERT_SIZE + comp['offset']
        subset_bytes = len(expert_indices) * EXPERT_SIZE
        print(
            f"  Layer {layer_idx:2d}: DRY RUN OK — would populate "
            f"{len(expert_indices)} experts ({subset_bytes:,} bytes written) in {out_path}"
        )
        return subset_bytes, 0.0

    t0 = time.monotonic()

    # Pre-allocate the full logical layer size so runtime expert offsets stay unchanged.
    # On APFS this remains sparse when we only write a subset of experts.
    fd_out = os.open(out_path, os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o644)
    os.ftruncate(fd_out, LAYER_SIZE)

    bytes_written = 0

    # Build read plan: group reads by source file for better locality
    # Each entry: (src_fd, src_offset, dst_offset, size)
    read_plan = []
    for expert_idx in expert_indices:
        for comp in COMPONENTS:
            info = layer_info[comp['name']]
            src_fd = fds[info['file']]
            src_offset = info['abs_offset'] + expert_idx * info['expert_stride']
            dst_offset = expert_idx * EXPERT_SIZE + comp['offset']
            read_plan.append((src_fd, src_offset, dst_offset, comp['size']))

    # Sort by (src_fd, src_offset) for sequential read locality
    read_plan.sort(key=lambda x: (x[0], x[1]))

    # Execute reads and writes
    for src_fd, src_offset, dst_offset, size in read_plan:
        data = os.pread(src_fd, size, src_offset)
        if len(data) != size:
            raise IOError(f"Short read: expected {size}, got {len(data)} "
                          f"at offset {src_offset}")
        os.pwrite(fd_out, data, dst_offset)
        bytes_written += size

    os.close(fd_out)
    elapsed = time.monotonic() - t0

    return bytes_written, elapsed


def repack_layer_from_gguf(layer_idx, tensor_map, gguf, output_dir, expert_indices=None, dry_run=False):
    """Export one layer of routed experts from GGUF into Flash-MoE's packed layout."""
    if expert_indices is None:
        expert_indices = list(range(NUM_EXPERTS))

    layer_tensors = {}
    for proj in PROJECTIONS:
        tensor_name, tensor = resolve_gguf_expert_tensor(tensor_map, layer_idx, proj["prefix"])
        logical_shape = logical_tensor_shape(tensor, gguf)
        expected_shape = (NUM_EXPERTS, proj["out_dim"], proj["in_dim"])
        if logical_shape != expected_shape:
            raise ValueError(
                f"Unexpected GGUF tensor shape for {tensor_name}: "
                f"expected {expected_shape}, got {logical_shape}"
            )
        layer_tensors[proj["prefix"]] = tensor

    out_path = os.path.join(output_dir, f"layer_{layer_idx:02d}.bin")
    if dry_run:
        subset_bytes = len(expert_indices) * EXPERT_SIZE
        print(
            f"  Layer {layer_idx:2d}: GGUF DRY RUN OK — would populate "
            f"{len(expert_indices)} experts ({subset_bytes:,} bytes written) in {out_path}"
        )
        return subset_bytes, 0.0

    t0 = time.monotonic()
    fd_out = os.open(out_path, os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o644)
    os.ftruncate(fd_out, LAYER_SIZE)
    bytes_written = 0

    for expert_idx in expert_indices:
        expert_base = expert_idx * EXPERT_SIZE
        for proj in PROJECTIONS:
            tensor = layer_tensors[proj["prefix"]]
            weights = gguf.dequantize(tensor.data[expert_idx], tensor.tensor_type)
            packed_w, packed_s, packed_b = quantize_affine_4bit(
                weights, proj["out_dim"], proj["in_dim"]
            )

            w_comp = next(c for c in COMPONENTS if c["name"] == f"{proj['prefix']}.weight")
            s_comp = next(c for c in COMPONENTS if c["name"] == f"{proj['prefix']}.scales")
            b_comp = next(c for c in COMPONENTS if c["name"] == f"{proj['prefix']}.biases")

            os.pwrite(fd_out, packed_w.tobytes(), expert_base + w_comp["offset"])
            os.pwrite(fd_out, packed_s.tobytes(), expert_base + s_comp["offset"])
            os.pwrite(fd_out, packed_b.tobytes(), expert_base + b_comp["offset"])
            bytes_written += w_comp["size"] + s_comp["size"] + b_comp["size"]

        if ((expert_indices.index(expert_idx) + 1) % 64 == 0 or
                expert_idx == expert_indices[-1]):
            print(f"    expert {expert_indices.index(expert_idx) + 1:3d}/{len(expert_indices)} "
                  f"(model expert {expert_idx})")

    os.close(fd_out)
    return bytes_written, time.monotonic() - t0


def export_q8_downproj_layer_from_gguf(
    layer_idx, tensor_map, gguf, output_dir, expert_indices=None, dry_run=False
):
    """Export one layer's down_proj experts as Q8_0 sidecar data."""
    if expert_indices is None:
        expert_indices = list(range(NUM_EXPERTS))

    tensor_name, tensor = resolve_gguf_expert_tensor(tensor_map, layer_idx, "down_proj")
    logical_shape = logical_tensor_shape(tensor, gguf)
    expected_shape = (NUM_EXPERTS, Q8_DOWN_OUT_DIM, Q8_DOWN_IN_DIM)
    if logical_shape != expected_shape:
        raise ValueError(
            f"Unexpected GGUF tensor shape for {tensor_name}: "
            f"expected {expected_shape}, got {logical_shape}"
        )

    out_path = os.path.join(output_dir, f"layer_{layer_idx:02d}.bin")
    if dry_run:
        subset_bytes = len(expert_indices) * Q8_DOWN_EXPERT_SIZE
        print(
            f"  Layer {layer_idx:2d}: Q8 down_proj DRY RUN OK — would populate "
            f"{len(expert_indices)} experts ({subset_bytes:,} bytes written) in {out_path}"
        )
        return subset_bytes, 0.0

    t0 = time.monotonic()
    fd_out = os.open(out_path, os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o644)
    os.ftruncate(fd_out, Q8_DOWN_LAYER_SIZE)
    bytes_written = 0

    for export_idx, expert_idx in enumerate(expert_indices, start=1):
        expert_base = expert_idx * Q8_DOWN_EXPERT_SIZE
        weights = gguf.dequantize(tensor.data[expert_idx], tensor.tensor_type)
        packed_w, packed_s = quantize_q8_0(
            weights, Q8_DOWN_OUT_DIM, Q8_DOWN_IN_DIM
        )

        os.pwrite(fd_out, packed_w.tobytes(), expert_base + Q8_DOWN_W_OFF)
        os.pwrite(fd_out, packed_s.tobytes(), expert_base + Q8_DOWN_S_OFF)
        bytes_written += Q8_DOWN_EXPERT_SIZE

        if (export_idx % 64 == 0 or export_idx == len(expert_indices)):
            print(f"    q8 down_proj expert {export_idx:3d}/{len(expert_indices)} "
                  f"(model expert {expert_idx})")

    os.close(fd_out)
    return bytes_written, time.monotonic() - t0


def verify_layer(layer_idx, expert_reads, model_path, fds, output_dir, expert_indices=None):
    """Read back expert 0 from packed file and compare to originals."""
    if expert_indices is None:
        expert_indices = [0, 1, 255, 511]

    layer_key = str(layer_idx)
    layer_info = expert_reads[layer_key]
    out_path = os.path.join(output_dir, f"layer_{layer_idx:02d}.bin")

    if not os.path.exists(out_path):
        print(f"  Layer {layer_idx}: packed file not found")
        return False

    fd_packed = os.open(out_path, os.O_RDONLY)

    mismatches = 0
    for expert_idx in expert_indices:
        for comp in COMPONENTS:
            info = layer_info[comp['name']]
            src_fd = fds[info['file']]
            src_offset = info['abs_offset'] + expert_idx * info['expert_stride']
            dst_offset = expert_idx * EXPERT_SIZE + comp['offset']

            original = os.pread(src_fd, comp['size'], src_offset)
            packed = os.pread(fd_packed, comp['size'], dst_offset)

            if original != packed:
                print(f"  MISMATCH: layer {layer_idx}, expert {expert_idx}, {comp['name']}")
                mismatches += 1

    os.close(fd_packed)

    if mismatches == 0:
        print(f"  Layer {layer_idx}: verification PASSED (experts 0, 1, 255, 511)")
    else:
        print(f"  Layer {layer_idx}: verification FAILED ({mismatches} mismatches)")

    return mismatches == 0


def write_layout(output_dir):
    """Write layout.json describing the packed format."""
    layout = {
        "expert_size": EXPERT_SIZE,
        "num_layers": NUM_LAYERS,
        "num_experts": NUM_EXPERTS,
        "components": COMPONENTS,
    }
    path = os.path.join(output_dir, "layout.json")
    with open(path, 'w') as f:
        json.dump(layout, f, indent=2)
    print(f"Wrote {path}")


def write_q8_downproj_layout(output_dir, layers):
    """Write layout.json for Q8_0 down_proj override sidecars."""
    layout = {
        "scope": "down_proj",
        "quant": "q8_0",
        "num_layers": NUM_LAYERS,
        "num_experts": NUM_EXPERTS,
        "layers": list(layers),
        "expert_size": Q8_DOWN_EXPERT_SIZE,
        "layer_size": Q8_DOWN_LAYER_SIZE,
        "block_size": Q8_BLOCK_SIZE,
        "weight": {
            "offset": Q8_DOWN_W_OFF,
            "size": Q8_DOWN_WEIGHT_SIZE,
            "dtype": "I8",
            "shape": [Q8_DOWN_OUT_DIM, Q8_DOWN_IN_DIM],
        },
        "scales": {
            "offset": Q8_DOWN_S_OFF,
            "size": Q8_DOWN_SCALE_SIZE,
            "dtype": "F16",
            "shape": [Q8_DOWN_OUT_DIM, Q8_DOWN_IN_DIM // Q8_BLOCK_SIZE],
        },
    }
    path = os.path.join(output_dir, "layout.json")
    with open(path, "w") as f:
        json.dump(layout, f, indent=2)
    print(f"Wrote {path}")


def main():
    parser = argparse.ArgumentParser(description="Repack expert weights into contiguous per-layer binary files")
    parser.add_argument('--index', default='~/Workspace/ane-research/expert_index.json',
                        help='Path to expert_index.json for safetensors mode')
    parser.add_argument('--gguf', default=None,
                        help='Path to GGUF directory or one GGUF shard for compatibility export')
    parser.add_argument('--layers', default=None,
                        help='Layer spec: "all", "0-4", "0,5,10" (default: all)')
    parser.add_argument('--experts', default=None,
                        help='Expert spec: "all", "0-4", "0,5,10" (default: all)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Verify offsets without writing')
    parser.add_argument('--verify-only', type=int, default=None, metavar='LAYER',
                        help='Verify a specific layer against originals (safetensors mode only)')
    parser.add_argument('--output-dir', default=None,
                        help='Override the packed expert output directory')
    parser.add_argument('--q8-down-proj-layers', default=None,
                        help='In GGUF mode, also export affine 8-bit down_proj sidecars for these layers')
    parser.add_argument('--q8-down-proj-dir', default=None,
                        help='Override the 8-bit down_proj sidecar output directory')
    parser.add_argument('--fail-if-source-exceeds-4bit', action='store_true',
                        help='In GGUF mode, abort if any routed-expert source tensor exceeds 4-bit class')
    args = parser.parse_args()
    use_gguf = args.gguf is not None

    if use_gguf and args.verify_only is not None:
        print("ERROR: --verify-only is only supported in safetensors mode")
        sys.exit(2)

    if use_gguf:
        print("Loading GGUF tensors...")
        try:
            gguf, gguf_readers, tensor_map, model_path, gguf_files = load_gguf_tensors(args.gguf)
        except (FileNotFoundError, ValueError, KeyError) as exc:
            print(f"ERROR: {exc}")
            sys.exit(1)
        print(f"GGUF shards: {len(gguf_files)}")
        print(f"Tensors loaded: {len(tensor_map)}")
        print("Export mode: GGUF compatibility -> Flash-MoE affine 4-bit experts")
        output_dir = os.path.expanduser(args.output_dir) if args.output_dir else os.path.join(model_path, "packed_experts")
    else:
        print("Loading expert index...")
        expert_reads, model_path = load_index(args.index)
        print(f"Model path: {model_path}")
        print(f"Layers in index: {len(expert_reads)}")

        # Verify component sizes
        if not verify_component_sizes(expert_reads):
            print("ABORTING: component size mismatch")
            sys.exit(1)

        output_dir = os.path.expanduser(args.output_dir) if args.output_dir else os.path.join(model_path, "packed_experts")

    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Determine which layers/experts to process
    try:
        if args.verify_only is not None:
            layers = [args.verify_only]
        else:
            layers = parse_layers(args.layers)
        experts = parse_experts(args.experts)
        q8_down_layers = parse_layers(args.q8_down_proj_layers) if args.q8_down_proj_layers else []
    except ValueError as exc:
        print(f"ERROR: {exc}")
        sys.exit(2)

    if q8_down_layers and not use_gguf:
        print("ERROR: --q8-down-proj-layers is only supported in GGUF mode")
        sys.exit(2)

    print(f"Layers to process: {layers[0]}-{layers[-1]} ({len(layers)} layers)")
    print(f"Experts to populate: {experts[0]}-{experts[-1]} ({len(experts)} experts)")
    if q8_down_layers:
        print(f"Q8 down_proj override layers: {q8_down_layers[0]}-{q8_down_layers[-1]} ({len(q8_down_layers)} layers)")

    if use_gguf:
        offenders, unknown, counts = summarize_gguf_expert_precision(
            tensor_map, layers, max_source_bits=4
        )
        report_path = os.path.join(output_dir, "gguf_routed_expert_precision_report.json")
        write_source_precision_report(
            report_path,
            "routed-expert",
            counts,
            offenders,
            unknown,
            max_source_bits=4,
        )
        if offenders and args.fail_if_source_exceeds_4bit:
            print("ABORTING: source routed-expert tensor exceeds 4-bit class")
            sys.exit(1)

    if not args.dry_run and args.verify_only is None:
        total_bytes = len(layers) * len(experts) * EXPERT_SIZE
        logical_total = len(layers) * LAYER_SIZE
        q8_total_bytes = len(q8_down_layers) * len(experts) * Q8_DOWN_EXPERT_SIZE
        q8_logical_total = len(q8_down_layers) * Q8_DOWN_LAYER_SIZE
        print(f"Total expert data to write: {total_bytes / (1024**3):.1f} GB")
        if len(experts) != NUM_EXPERTS:
            print(f"Logical sparse file footprint: {logical_total / (1024**3):.1f} GB across {len(layers)} layers")
        if q8_down_layers:
            print(f"Total Q8 down_proj sidecar data to write: {q8_total_bytes / (1024**3):.1f} GB")
            if len(experts) != NUM_EXPERTS:
                print(
                    f"Logical Q8 sparse footprint: {q8_logical_total / (1024**3):.1f} GB "
                    f"across {len(q8_down_layers)} layers"
                )

        # Check free disk space
        stat = os.statvfs(output_dir)
        free_bytes = stat.f_bavail * stat.f_frsize
        free_gb = free_bytes / (1024**3)
        needed_gb = (total_bytes + q8_total_bytes) / (1024**3)
        print(f"Free disk space: {free_gb:.1f} GB, needed: {needed_gb:.1f} GB")
        if free_bytes < (total_bytes + q8_total_bytes):
            print(f"WARNING: Not enough free space! Need {needed_gb:.1f} GB but only {free_gb:.1f} GB free.")
            print(f"Hint: use --layers to process a subset, e.g. --layers 0-{int(free_gb / 3.63) - 1}")
            sys.exit(1)

    if not use_gguf:
        # Open source files
        fds = open_source_files(expert_reads, model_path, layers)

        if args.verify_only is not None:
            verify_layer(args.verify_only, expert_reads, model_path, fds, output_dir)
            for fd in fds.values():
                os.close(fd)
            return

    # Write layout.json
    write_layout(output_dir)
    q8_down_dir = None
    if q8_down_layers:
        q8_down_dir = (
            os.path.expanduser(args.q8_down_proj_dir)
            if args.q8_down_proj_dir
            else os.path.join(model_path, "packed_experts_q8_down")
        )
        os.makedirs(q8_down_dir, exist_ok=True)
        print(f"Q8 down_proj output directory: {q8_down_dir}")
        write_q8_downproj_layout(q8_down_dir, q8_down_layers)

    # Repack each layer
    t_start = time.monotonic()
    total_written = 0

    for i, layer_idx in enumerate(layers):
        if use_gguf:
            bytes_written, elapsed = repack_layer_from_gguf(
                layer_idx, tensor_map, gguf, output_dir, expert_indices=experts, dry_run=args.dry_run
            )
        else:
            bytes_written, elapsed = repack_layer(
                layer_idx, expert_reads, model_path, fds, output_dir, expert_indices=experts, dry_run=args.dry_run
            )
        total_written += bytes_written

        if not args.dry_run and bytes_written > 0:
            throughput = bytes_written / elapsed / (1024**3) if elapsed > 0 else float('inf')
            overall_elapsed = time.monotonic() - t_start
            overall_throughput = total_written / overall_elapsed / (1024**3) if overall_elapsed > 0 else 0
            eta = (len(layers) - i - 1) * (overall_elapsed / (i + 1))
            print(f"  Layer {layer_idx:2d}: {bytes_written/1024**3:.2f} GB in {elapsed:.1f}s "
                  f"({throughput:.1f} GB/s) | "
                  f"Total: {total_written/1024**3:.1f}/{len(layers)*LAYER_SIZE/1024**3:.1f} GB "
                  f"({overall_throughput:.1f} GB/s avg) | "
                  f"ETA: {eta:.0f}s")

            if not use_gguf:
                # Verify this layer immediately
                verify_experts = experts[:4] if len(experts) > 4 else experts
                if not verify_layer(layer_idx, expert_reads, model_path, fds, output_dir,
                                    expert_indices=verify_experts):
                    print(f"ABORTING: verification failed for layer {layer_idx}")
                    sys.exit(1)

    q8_written = 0
    if q8_down_layers:
        print("\nExporting Q8 down_proj sidecars...")
        q8_start = time.monotonic()
        for i, layer_idx in enumerate(q8_down_layers):
            bytes_written, elapsed = export_q8_downproj_layer_from_gguf(
                layer_idx, tensor_map, gguf, q8_down_dir,
                expert_indices=experts, dry_run=args.dry_run
            )
            q8_written += bytes_written
            if not args.dry_run and bytes_written > 0:
                throughput = bytes_written / elapsed / (1024**3) if elapsed > 0 else float('inf')
                overall_elapsed = time.monotonic() - q8_start
                overall_throughput = q8_written / overall_elapsed / (1024**3) if overall_elapsed > 0 else 0
                eta = (len(q8_down_layers) - i - 1) * (overall_elapsed / (i + 1))
                print(f"  Q8 layer {layer_idx:2d}: {bytes_written/1024**3:.2f} GB in {elapsed:.1f}s "
                      f"({throughput:.1f} GB/s) | "
                      f"Total: {q8_written/1024**3:.1f}/{len(q8_down_layers)*Q8_DOWN_LAYER_SIZE/1024**3:.1f} GB "
                      f"({overall_throughput:.1f} GB/s avg) | "
                      f"ETA: {eta:.0f}s")

    if not use_gguf:
        # Close source files
        for fd in fds.values():
            os.close(fd)

    # Final summary
    total_elapsed = time.monotonic() - t_start
    if not args.dry_run and total_written > 0:
        print(f"\n{'='*60}")
        print(f"DONE: {total_written:,} bytes ({total_written/1024**3:.1f} GB) written")
        if q8_written > 0:
            print(f"Q8 down_proj sidecars: {q8_written:,} bytes ({q8_written/1024**3:.1f} GB) written")
        print(f"Time: {total_elapsed:.1f}s")
        print(f"Throughput: {total_written/total_elapsed/1024**3:.1f} GB/s")
        print(f"Output: {output_dir}")
    elif args.dry_run:
        print(f"\nDRY RUN complete: {len(layers)} layers validated")


if __name__ == '__main__':
    main()
