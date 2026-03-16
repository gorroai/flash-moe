"""
stream_infer.py — Streaming inference engine for Qwen3.5 MoE models.
Loads weights layer-by-layer from safetensors during inference.
Proves flash offloading works and measures overhead vs fully-loaded baseline.

Usage:
    uv run stream_infer.py --model mlx-community/Qwen3.5-35B-A3B-4bit --tokens 20 --mode baseline
    uv run stream_infer.py --model mlx-community/Qwen3.5-35B-A3B-4bit --tokens 20 --mode stream
    uv run stream_infer.py --model mlx-community/Qwen3.5-35B-A3B-4bit --tokens 20 --mode layerwise
"""

import argparse
import time
import sys
import json
import re
import os
from pathlib import Path
from collections import defaultdict, OrderedDict
from concurrent.futures import ThreadPoolExecutor

import subprocess
import psutil
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.utils
import mlx_lm
from safetensors import safe_open


def parse_safetensors_header(filepath):
    """Parse a safetensors file header. Returns (header_dict, data_start_offset)."""
    import struct
    with open(filepath, 'rb') as f:
        header_len = struct.unpack('<Q', f.read(8))[0]
        header = json.loads(f.read(header_len))
        data_start = 8 + header_len
    return header, data_start


def read_tensors_direct(filepath, tensor_names, header_cache, file_handle_cache=None):
    """Read specific tensors from a safetensors file using direct I/O (not mmap).
    Returns dict of tensor_name -> mx.array (real Metal allocations, not mmap-backed).

    Handles dtype mapping:
    - U32 -> numpy uint32 -> mx uint32
    - F32 -> numpy float32 -> mx float32
    - F16 -> numpy float16 -> mx float16
    - I32 -> numpy int32 -> mx int32
    - BF16 -> read as uint16, shift to float32, cast to bfloat16

    If file_handle_cache is provided (dict), file handles are kept open across calls
    to avoid ~3456 open/close cycles per token (72 tensors × 48 layers).
    """
    if filepath not in header_cache:
        header_cache[filepath] = parse_safetensors_header(filepath)
    header, data_start = header_cache[filepath]

    NP_DTYPE = {
        'U32': np.uint32,
        'F32': np.float32,
        'F16': np.float16,
        'I32': np.int32,
        'I64': np.int64,
        'U8': np.uint8,
    }

    # Sort tensors by file offset for sequential I/O
    sorted_names = sorted(tensor_names, key=lambda n: header[n]['data_offsets'][0])

    # Use cached file handle if available, otherwise open (and optionally cache)
    owned_f = None
    if file_handle_cache is not None:
        if filepath not in file_handle_cache:
            file_handle_cache[filepath] = open(filepath, 'rb')
        f = file_handle_cache[filepath]
    else:
        owned_f = open(filepath, 'rb')
        f = owned_f

    result = {}
    try:
        for name in sorted_names:
            meta = header[name]
            off = meta['data_offsets']
            byte_len = off[1] - off[0]

            f.seek(data_start + off[0])
            raw = f.read(byte_len)

            dtype_str = meta['dtype']
            shape = meta['shape']

            if dtype_str in NP_DTYPE:
                np_arr = np.frombuffer(raw, dtype=NP_DTYPE[dtype_str]).reshape(shape)
                result[name] = mx.array(np_arr)
            elif dtype_str == 'BF16':
                # bfloat16 = top 16 bits of float32. Read as uint16,
                # shift left 16 bits to get float32, then convert to bfloat16.
                np_uint16 = np.frombuffer(raw, dtype=np.uint16).reshape(shape)
                np_f32 = (np_uint16.astype(np.uint32) << 16).view(np.float32)
                result[name] = mx.array(np_f32).astype(mx.bfloat16)
            else:
                raise ValueError(f"Unsupported safetensors dtype: {dtype_str}")
    finally:
        if owned_f is not None:
            owned_f.close()

    return result


def read_expert_slices_direct(filepath, tensor_name, expert_indices, header_cache, file_handle_cache=None):
    """Read specific expert slices from a stacked [num_experts, ...] tensor using direct I/O.

    Args:
        filepath: path to safetensors file
        tensor_name: full tensor name in the safetensors file
        expert_indices: list/array of expert indices to read (e.g., [3, 17, 42, ...])
        header_cache: dict for caching parsed headers
        file_handle_cache: optional dict mapping filepath -> open file handle.
            If provided, file handles are kept open across calls to avoid
            repeated open/close overhead (~3456 cycles/token).

    Returns: mx.array of shape [len(expert_indices), ...rest_dims]
    """
    if filepath not in header_cache:
        header_cache[filepath] = parse_safetensors_header(filepath)
    header, data_start = header_cache[filepath]

    meta = header[tensor_name]
    shape = meta['shape']  # e.g., [256, 1024, 384]
    dtype_str = meta['dtype']
    tensor_offsets = meta['data_offsets']
    tensor_start = data_start + tensor_offsets[0]

    num_experts = shape[0]
    expert_shape = shape[1:]  # e.g., [1024, 384]

    # Calculate bytes per expert
    NP_DTYPE = {
        'U32': (np.uint32, 4),
        'F32': (np.float32, 4),
        'F16': (np.float16, 2),
        'BF16': (np.uint16, 2),  # Read as uint16
        'I32': (np.int32, 4),
    }
    np_dtype, elem_size = NP_DTYPE[dtype_str]
    expert_elems = 1
    for d in expert_shape:
        expert_elems *= d
    expert_bytes = expert_elems * elem_size

    # Use cached file handle if available, otherwise open (and optionally cache)
    owned_f = None
    if file_handle_cache is not None:
        if filepath not in file_handle_cache:
            file_handle_cache[filepath] = open(filepath, 'rb')
        f = file_handle_cache[filepath]
    else:
        owned_f = open(filepath, 'rb')
        f = owned_f

    # Read each selected expert's data
    expert_arrays = []
    try:
        for idx in expert_indices:
            offset = tensor_start + int(idx) * expert_bytes
            f.seek(offset)
            raw = f.read(expert_bytes)
            np_arr = np.frombuffer(raw, dtype=np_dtype).reshape(expert_shape)
            expert_arrays.append(np_arr)
    finally:
        if owned_f is not None:
            owned_f.close()

    # Stack into [num_selected, ...] array
    stacked = np.stack(expert_arrays, axis=0)

    if dtype_str == 'BF16':
        # Convert uint16 bit pattern to bfloat16 via float32
        np_f32 = (stacked.astype(np.uint32) << 16).view(np.float32)
        return mx.array(np_f32).astype(mx.bfloat16)
    else:
        return mx.array(stacked)


def _read_single_expert_attrs(expert_idx, layer_idx, expert_file_map, header_cache):
    """Read all 9 tensor attributes for a single expert using coalesced I/O.

    Instead of 9 individual seek+read operations, groups reads by file, sorts by
    offset, and merges adjacent reads within a 64KB gap into larger sequential reads.
    This reduces ~9 seeks per expert to ~1-3 larger reads.

    Each thread opens its own file handles (no shared seek positions), reads all
    3 projections x 3 attributes, and returns a dict of results.

    Args:
        expert_idx: which expert to read (e.g. 42)
        layer_idx: which layer (for constructing tensor names)
        expert_file_map: dict of full tensor name -> filepath
        header_cache: dict for cached headers (read-only after init, thread-safe)

    Returns:
        (expert_idx, results_dict, io_stats_dict)
        results_dict: (proj_name, attr_name) -> mx.array of shape [1, ...expert_dims]
        io_stats_dict: {"bytes_read": int, "seek_count": int, "io_time_s": float, "array_time_s": float}
    """
    NP_DTYPE = {
        'U32': (np.uint32, 4),
        'F32': (np.float32, 4),
        'F16': (np.float16, 2),
        'BF16': (np.uint16, 2),
        'I32': (np.int32, 4),
    }

    COALESCE_GAP = 65536  # 64KB: merge reads with gaps smaller than this

    # Phase 1: Collect all (file, offset, size, metadata) tuples without doing I/O
    read_specs = []  # list of (filepath, abs_offset, byte_len, proj_name, attr_name, expert_shape, dtype_str, np_dtype)
    for proj_name in ("gate_proj", "up_proj", "down_proj"):
        prefix = f"language_model.model.layers.{layer_idx}.mlp.switch_mlp.{proj_name}"
        for attr_name in ("weight", "scales", "biases"):
            full_key = f"{prefix}.{attr_name}"
            if full_key not in expert_file_map:
                continue
            filepath = expert_file_map[full_key]

            header, data_start = header_cache[filepath]
            meta = header[full_key]
            shape = meta['shape']
            dtype_str = meta['dtype']
            tensor_offsets = meta['data_offsets']
            tensor_start = data_start + tensor_offsets[0]
            expert_shape = shape[1:]

            np_dtype, elem_size = NP_DTYPE[dtype_str]
            expert_elems = 1
            for d in expert_shape:
                expert_elems *= d
            expert_bytes = expert_elems * elem_size

            abs_offset = tensor_start + int(expert_idx) * expert_bytes
            read_specs.append((filepath, abs_offset, expert_bytes, proj_name, attr_name, expert_shape, dtype_str, np_dtype))

    # Phase 2: Group by file, sort by offset, merge adjacent reads
    by_file = defaultdict(list)
    for spec in read_specs:
        by_file[spec[0]].append(spec)

    results = {}
    local_handles = {}
    io_bytes = 0
    io_seeks = 0
    io_time = 0.0
    array_time = 0.0

    try:
        for filepath, specs in by_file.items():
            # Sort by file offset
            specs.sort(key=lambda s: s[1])

            # Merge into coalesced read groups
            # Each group: (start_offset, total_len, [(local_offset, byte_len, proj_name, attr_name, expert_shape, dtype_str, np_dtype), ...])
            groups = []
            for spec in specs:
                _, abs_offset, byte_len, proj_name, attr_name, expert_shape, dtype_str, np_dtype = spec
                if groups:
                    g_start, g_len, g_items = groups[-1]
                    g_end = g_start + g_len
                    gap = abs_offset - g_end
                    if gap <= COALESCE_GAP:
                        # Extend the current group to cover this read
                        new_end = abs_offset + byte_len
                        groups[-1] = (g_start, new_end - g_start, g_items)
                        local_offset = abs_offset - g_start
                        g_items.append((local_offset, byte_len, proj_name, attr_name, expert_shape, dtype_str, np_dtype))
                        continue
                # Start a new group
                groups.append((abs_offset, byte_len, [(0, byte_len, proj_name, attr_name, expert_shape, dtype_str, np_dtype)]))

            # Open file handle (reuse within this thread)
            if filepath not in local_handles:
                local_handles[filepath] = open(filepath, 'rb')
            f = local_handles[filepath]

            # Phase 3: Execute coalesced reads and split into individual tensors
            for g_start, g_len, g_items in groups:
                t_io = time.time()
                f.seek(g_start)
                raw_chunk = f.read(g_len)
                io_time += time.time() - t_io
                io_bytes += g_len
                io_seeks += 1

                t_arr = time.time()
                for local_offset, byte_len, proj_name, attr_name, expert_shape, dtype_str, np_dtype in g_items:
                    raw = raw_chunk[local_offset:local_offset + byte_len]
                    np_arr = np.frombuffer(raw, dtype=np_dtype).reshape(expert_shape)

                    if dtype_str == 'BF16':
                        np_f32 = (np_arr.astype(np.uint32) << 16).view(np.float32)
                        results[(proj_name, attr_name)] = mx.array(np_f32).astype(mx.bfloat16)
                    else:
                        results[(proj_name, attr_name)] = mx.array(np_arr)
                array_time += time.time() - t_arr
    finally:
        for fh in local_handles.values():
            fh.close()

    io_stats = {
        "bytes_read": io_bytes,
        "seek_count": io_seeks,
        "io_time_s": io_time,
        "array_time_s": array_time,
    }
    return expert_idx, results, io_stats


class ExpertCache:
    """LRU cache for expert weight slices keyed by (layer, expert_id).

    Stores per-attribute arrays so that partially-populated entries can be
    built incrementally (one proj/attr at a time) then read back as a batch.

    Each entry is a dict mapping "proj_name.attr_name" -> mx.array, e.g.
    {"gate_proj.weight": mx.array(...), "gate_proj.scales": mx.array(...), ...}.
    """

    def __init__(self, max_entries=256):
        self.max_entries = max_entries
        self.cache = OrderedDict()  # (layer_idx, expert_id) -> {attr_key: mx.array}
        self.hits = 0
        self.misses = 0

    def get_attr(self, layer_idx, expert_id, proj_name, attr_name):
        """Return a single cached array or None."""
        key = (layer_idx, expert_id)
        if key in self.cache:
            self.cache.move_to_end(key)
            entry = self.cache[key]
            attr_key = f"{proj_name}.{attr_name}"
            return entry.get(attr_key)
        return None

    def put_attr(self, layer_idx, expert_id, proj_name, attr_name, array):
        """Store a single attribute array for an expert."""
        key = (layer_idx, expert_id)
        attr_key = f"{proj_name}.{attr_name}"
        if key in self.cache:
            self.cache.move_to_end(key)
            self.cache[key][attr_key] = array
        else:
            if len(self.cache) >= self.max_entries:
                self.cache.popitem(last=False)  # evict LRU
            self.cache[key] = {attr_key: array}

    def has_expert(self, layer_idx, expert_id):
        """Check whether all 9 attributes (3 projs x 3 attrs) are cached."""
        key = (layer_idx, expert_id)
        if key not in self.cache:
            return False
        return len(self.cache[key]) >= 9  # gate/up/down x weight/scales/biases

    def touch(self, layer_idx, expert_id):
        """Move entry to end (most recently used) to prevent eviction."""
        key = (layer_idx, expert_id)
        if key in self.cache:
            self.cache.move_to_end(key)

    def record_hit(self):
        self.hits += 1

    def record_miss(self):
        self.misses += 1

    @property
    def hit_rate(self):
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


def get_mem_gb():
    return psutil.Process().memory_info().rss / (1024 ** 3)


def check_memory_pressure():
    """Check macOS system memory free percentage. Returns (level, free_pct)."""
    try:
        result = subprocess.run(
            ["memory_pressure"],
            capture_output=True, text=True, timeout=5
        )
        for line in result.stdout.split("\n"):
            if "free percentage" in line.lower():
                pct = int(line.split(":")[-1].strip().rstrip("%"))
                if pct < 10:
                    return "critical", pct
                elif pct < 25:
                    return "warn", pct
                return "normal", pct
        return "unknown", -1
    except Exception:
        return "unknown", -1


def fmt_time(seconds):
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"


def resolve_model_path(model_id):
    """Resolve a HF model ID to a local path."""
    p = Path(model_id)
    if p.exists():
        return p
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(model_id))


def build_weight_index(model_path):
    """Build a mapping: layer_num -> [(tensor_name, file_path)].
    Also returns 'global' key for non-layer tensors (embed, norm, lm_head)."""
    index_path = model_path / "model.safetensors.index.json"
    with open(index_path) as f:
        idx = json.load(f)

    layer_weights = defaultdict(list)
    for name, filename in idx["weight_map"].items():
        filepath = model_path / filename
        match = re.search(r'layers\.(\d+)\.', name)
        if match:
            layer_num = int(match.group(1))
            layer_weights[layer_num].append((name, str(filepath)))
        else:
            layer_weights["global"].append((name, str(filepath)))

    return layer_weights


def load_layer_from_safetensors(weight_index, layer_num, model, file_cache=None):
    """Load a single layer's weights from safetensors into the model.
    Uses mx.load() which handles bfloat16 natively via mmap.
    Returns the load time in seconds."""
    entries = weight_index.get(layer_num, [])
    if not entries:
        return 0.0

    t0 = time.time()

    # Group by file for efficiency
    by_file = defaultdict(list)
    for name, filepath in entries:
        by_file[filepath].append(name)

    weights = []
    for filepath, names in by_file.items():
        # mx.load() returns lazy mmap'd arrays — actual I/O at mx.eval()
        if file_cache is not None and filepath in file_cache:
            all_tensors = file_cache[filepath]
        else:
            all_tensors = mx.load(filepath)
            if file_cache is not None:
                file_cache[filepath] = all_tensors

        for name in names:
            if name not in all_tensors:
                continue
            # Sanitize: remove "language_model." prefix to match model's param paths
            san_name = name
            if san_name.startswith("language_model."):
                san_name = san_name[len("language_model."):]
            weights.append((san_name, all_tensors[name]))

    # Apply to model (language_model level)
    model.language_model.load_weights(weights, strict=False)
    # Force evaluation so weights are actually loaded from disk
    mx.eval(model.language_model.model.layers[layer_num].parameters())

    return time.time() - t0


def manual_forward(model, input_ids, cache):
    """Run the model's forward pass layer-by-layer, returning logits.
    This replicates the model's own forward but gives us per-layer control."""
    lm = model.language_model
    text_model = lm.model
    layers = text_model.layers

    # Embed
    h = text_model.embed_tokens(input_ids)

    # Create masks (same logic as the model)
    from mlx_lm.models.qwen3_5 import create_attention_mask, create_ssm_mask
    fa_mask = create_attention_mask(h, cache[text_model.fa_idx])
    ssm_mask = create_ssm_mask(h, cache[text_model.ssm_idx])

    # Process layers
    for i, (layer, c) in enumerate(zip(layers, cache)):
        mask = ssm_mask if layer.is_linear else fa_mask
        h = layer(h, mask=mask, cache=c)

    # Norm
    h = text_model.norm(h)

    # LM head
    if lm.args.tie_word_embeddings:
        logits = text_model.embed_tokens.as_linear(h)
    else:
        logits = lm.lm_head(h)

    return logits


def manual_forward_layerwise(model, input_ids, cache, weight_index=None, file_cache=None):
    """Same as manual_forward but returns per-layer timing info.
    If weight_index is provided, reloads weights from safetensors per layer."""
    lm = model.language_model
    text_model = lm.model
    layers = text_model.layers

    h = text_model.embed_tokens(input_ids)
    mx.eval(h)

    from mlx_lm.models.qwen3_5 import create_attention_mask, create_ssm_mask
    fa_mask = create_attention_mask(h, cache[text_model.fa_idx])
    ssm_mask = create_ssm_mask(h, cache[text_model.ssm_idx])

    layer_timings = []

    for i, (layer, c) in enumerate(zip(layers, cache)):
        load_time = 0.0
        if weight_index is not None:
            load_time = load_layer_from_safetensors(weight_index, i, model, file_cache)

        mask = ssm_mask if layer.is_linear else fa_mask

        t_compute = time.time()
        h = layer(h, mask=mask, cache=c)
        mx.eval(h)
        compute_time = time.time() - t_compute

        layer_timings.append({
            "layer": i,
            "is_linear": layer.is_linear,
            "load_ms": load_time * 1000,
            "compute_ms": compute_time * 1000,
        })

    h = text_model.norm(h)

    if lm.args.tie_word_embeddings:
        logits = text_model.embed_tokens.as_linear(h)
    else:
        logits = lm.lm_head(h)
    mx.eval(logits)

    return logits, layer_timings


def generate_baseline(model, tokenizer, prompt, max_tokens):
    """Generate using mlx_lm's built-in stream_generate. Reference baseline."""
    t_start = time.time()
    token_times = []
    generated_tokens = []
    peak_mem = get_mem_gb()

    t_gen_start = time.time()

    for i, response in enumerate(mlx_lm.stream_generate(
        model, tokenizer, prompt=prompt, max_tokens=max_tokens,
    )):
        t_now = time.time()

        if i == 0:
            ttft_ms = (t_now - t_gen_start) * 1000
            print(f"  [{fmt_time(t_now - t_start)}] Token 1/{max_tokens}... ttft={ttft_ms:.0f}ms")
        else:
            token_times.append(t_now - t_prev)

        t_prev = t_now
        generated_tokens.append(response.token)

        if (i + 1) % 5 == 0 or i == max_tokens - 1:
            cur_mem = get_mem_gb()
            peak_mem = max(peak_mem, cur_mem)
            elapsed = t_now - t_gen_start
            tps = (i + 1) / elapsed if elapsed > 0 else 0
            print(f"  [{fmt_time(t_now - t_start)}] Token {i+1}/{max_tokens}... "
                  f"{tps:.1f} tok/s (mem: {cur_mem:.1f} GB)")

        if i + 1 >= max_tokens:
            break

    total_time = time.time() - t_gen_start
    total_tokens = len(generated_tokens)
    text = tokenizer.decode(generated_tokens)

    return {
        "text": text,
        "tokens": total_tokens,
        "total_time": total_time,
        "tok_sec": total_tokens / total_time if total_time > 0 else 0,
        "ttft_ms": ttft_ms,
        "peak_mem_gb": peak_mem,
    }


def generate_manual(model, tokenizer, prompt, max_tokens, weight_index=None, mode="stream"):
    """Generate tokens using manual layer-by-layer forward pass.
    If weight_index is provided (mode=stream), reloads weights from safetensors each layer."""
    t_start = time.time()

    input_ids = mx.array(tokenizer.encode(prompt))[None, :]
    cache = model.make_cache()

    generated_tokens = []
    token_times = []
    all_layer_timings = []
    peak_mem = get_mem_gb()
    # Cache file handles for mx.load() — avoids re-parsing safetensors headers
    file_cache = {} if mode == "stream" else None

    for token_idx in range(max_tokens):
        t_token_start = time.time()

        if mode == "layerwise" or mode == "stream":
            logits, layer_timings = manual_forward_layerwise(
                model, input_ids, cache,
                weight_index=weight_index if mode == "stream" else None,
                file_cache=file_cache,
            )
            all_layer_timings.append(layer_timings)
        else:
            logits = manual_forward(model, input_ids, cache)
            layer_timings = None

        # Greedy sample
        next_token = mx.argmax(logits[:, -1, :], axis=-1)
        mx.eval(next_token)

        token_id = next_token.item()
        generated_tokens.append(token_id)

        t_token_end = time.time()
        token_time = t_token_end - t_token_start
        token_times.append(token_time)

        cur_mem = get_mem_gb()
        peak_mem = max(peak_mem, cur_mem)

        # Safety: check system memory pressure every 5 tokens
        if (token_idx + 1) % 5 == 0:
            pressure, free_pct = check_memory_pressure()
            if pressure == "critical":
                print(f"\n  ABORT: System memory free={free_pct}% (critical). "
                      f"Stopping to protect system stability.")
                break

        # Progress
        if token_idx == 0:
            ttft_ms = token_time * 1000
            load_ms = sum(lt["load_ms"] for lt in layer_timings) if layer_timings else 0
            compute_ms = sum(lt["compute_ms"] for lt in layer_timings) if layer_timings else 0
            print(f"  [{fmt_time(t_token_end - t_start)}] Token 1/{max_tokens}: "
                  f"ttft={ttft_ms:.0f}ms (load={load_ms:.0f}ms compute={compute_ms:.0f}ms)")
        elif (token_idx + 1) % 5 == 0 or token_idx == max_tokens - 1:
            elapsed = t_token_end - t_start
            avg_tps = (token_idx + 1) / elapsed
            load_ms = sum(lt["load_ms"] for lt in layer_timings) if layer_timings else 0
            compute_ms = sum(lt["compute_ms"] for lt in layer_timings) if layer_timings else 0
            print(f"  [{fmt_time(elapsed)}] Token {token_idx+1}/{max_tokens}: "
                  f"{avg_tps:.1f} tok/s (load={load_ms:.0f}ms compute={compute_ms:.0f}ms "
                  f"mem={cur_mem:.1f}GB)")

        # Next iteration input
        input_ids = next_token.reshape(1, 1)

    total_time = time.time() - t_start
    total_tokens = len(generated_tokens)
    text = tokenizer.decode(generated_tokens)

    # Aggregate layer timing stats (skip first token — prompt processing is different)
    if all_layer_timings and len(all_layer_timings) > 1:
        gen_timings = all_layer_timings[1:]  # skip prompt token
        avg_load = np.mean([sum(lt["load_ms"] for lt in tt) for tt in gen_timings])
        avg_compute = np.mean([sum(lt["compute_ms"] for lt in tt) for tt in gen_timings])

        # Per-layer breakdown
        num_layers = len(gen_timings[0])
        per_layer_load = [np.mean([tt[i]["load_ms"] for tt in gen_timings]) for i in range(num_layers)]
        per_layer_compute = [np.mean([tt[i]["compute_ms"] for tt in gen_timings]) for i in range(num_layers)]
    else:
        avg_load = 0
        avg_compute = 0
        per_layer_load = []
        per_layer_compute = []

    return {
        "text": text,
        "tokens": total_tokens,
        "total_time": total_time,
        "tok_sec": total_tokens / total_time if total_time > 0 else 0,
        "ttft_ms": token_times[0] * 1000 if token_times else 0,
        "peak_mem_gb": peak_mem,
        "avg_load_ms_per_token": avg_load,
        "avg_compute_ms_per_token": avg_compute,
        "per_layer_load_ms": per_layer_load,
        "per_layer_compute_ms": per_layer_compute,
    }


def clear_layer_weights(model, layer_num):
    """Replace all leaf parameters in a layer with tiny dummy arrays to free memory.
    This is the critical step that keeps DRAM usage bounded: after computing a layer,
    we throw away its ~1.3GB of weights so the next layer can be loaded."""
    layer = model.language_model.model.layers[layer_num]
    dummy_weights = []
    for name, param in mlx.utils.tree_flatten(layer.parameters()):
        dummy_weights.append((f"model.layers.{layer_num}.{name}", mx.zeros((1,), dtype=param.dtype)))
    model.language_model.load_weights(dummy_weights, strict=False)
    mx.clear_cache()


def clear_expert_weights(model, layer_num):
    """Clear only MoE expert weights (switch_mlp), preserving attention/norms/shared_expert.
    Expert weights are ~1.27GB per layer; non-expert weights are ~50MB and stay pinned."""
    layer = model.language_model.model.layers[layer_num]
    if not hasattr(layer.mlp, 'switch_mlp'):
        return  # Dense layer, nothing to clear
    switch = layer.mlp.switch_mlp
    dummy_weights = []
    for proj_name in ["gate_proj", "up_proj", "down_proj"]:
        proj = getattr(switch, proj_name)
        for attr_name in ["weight", "scales", "biases"]:
            if hasattr(proj, attr_name):
                full_name = f"model.layers.{layer_num}.mlp.switch_mlp.{proj_name}.{attr_name}"
                dummy_weights.append((full_name, mx.zeros((1,), dtype=getattr(proj, attr_name).dtype)))
    if dummy_weights:
        model.language_model.load_weights(dummy_weights, strict=False)
    mx.clear_cache()  # Return freed Metal memory to OS


def compute_moe_direct(x, indices, expert_tensors, group_size=64, bits=4, mode="affine"):
    """Compute MoE SwitchGLU forward pass directly via mx.gather_qmm.

    Bypasses model weight mutation entirely -- no load_weights, no clear_expert_weights.
    Replicates the SwitchGLU.__call__ + QuantizedSwitchLinear.__call__ logic.

    Args:
        x: hidden states [batch, seq_len, hidden_dim]
        indices: remapped expert indices [batch, seq_len, top_k] with values in 0..num_unique-1
        expert_tensors: dict with keys like "gate_proj.weight", "gate_proj.scales", etc.
            Each value is [num_unique_experts, ...] stacked tensor.
        group_size: quantization group size (default 64)
        bits: quantization bits (default 4)
        mode: quantization mode (default "affine")

    Returns:
        output: [batch, seq_len, top_k, hidden_dim]
    """
    # Replicate SwitchGLU.__call__: expand dims for gather_qmm batch indexing
    x = mx.expand_dims(x, (-2, -3))

    # For single-token inference (indices.size < 64), no sorting needed
    do_sort = indices.size >= 64
    idx = indices
    inv_order = None
    if do_sort:
        # Replicate _gather_sort
        *_, M = indices.shape
        flat_indices = indices.flatten()
        order = mx.argsort(flat_indices)
        inv_order = mx.argsort(order)
        x = x.flatten(0, -3)[order // M]
        idx = flat_indices[order]

    # up_proj: x -> intermediate
    x_up = mx.gather_qmm(
        x,
        expert_tensors["up_proj.weight"],
        expert_tensors["up_proj.scales"],
        expert_tensors.get("up_proj.biases"),
        rhs_indices=idx,
        transpose=True,
        group_size=group_size,
        bits=bits,
        mode=mode,
        sorted_indices=do_sort,
    )

    # gate_proj: x -> intermediate
    x_gate = mx.gather_qmm(
        x,
        expert_tensors["gate_proj.weight"],
        expert_tensors["gate_proj.scales"],
        expert_tensors.get("gate_proj.biases"),
        rhs_indices=idx,
        transpose=True,
        group_size=group_size,
        bits=bits,
        mode=mode,
        sorted_indices=do_sort,
    )

    # SwiGLU activation: silu(gate) * up
    x_act = nn.silu(x_gate) * x_up

    # down_proj: intermediate -> hidden
    out = mx.gather_qmm(
        x_act,
        expert_tensors["down_proj.weight"],
        expert_tensors["down_proj.scales"],
        expert_tensors.get("down_proj.biases"),
        rhs_indices=idx,
        transpose=True,
        group_size=group_size,
        bits=bits,
        mode=mode,
        sorted_indices=do_sort,
    )

    # Unsort if we sorted
    if do_sort:
        # Replicate _scatter_unsort
        out = out[inv_order]
        out = mx.unflatten(out, 0, indices.shape)

    return out.squeeze(-2)


def generate_offload(model, tokenizer, prompt, max_tokens, weight_index, model_path, lazy_eval=False):
    """Generate tokens with explicit per-layer weight streaming for models larger than DRAM.

    For each token:
      1. Embed (weights already pinned in DRAM)
      2. For each of 48 layers:
         a. Load layer weights from safetensors (~1.3GB)
         b. Run layer forward pass
         c. Clear layer weights (replace with tiny dummies)
      3. Norm + lm_head (weights already pinned)
      4. Sample next token

    Peak DRAM usage: ~1GB global + ~1.3GB one layer = ~2.3GB active weights.
    The rest is KV cache and activations.

    If lazy_eval=True, skip mx.eval(layers[i].parameters()) after load_weights.
    This leaves weights as lazy mmap'd references. The subsequent layer forward
    pass + mx.eval(h) should only page in the expert weights actually accessed
    by the router (~40MB for 8/256 experts) instead of all ~1.3GB per layer.
    """
    t_start = time.time()

    input_ids = mx.array(tokenizer.encode(prompt))[None, :]
    cache = model.make_cache()

    lm = model.language_model
    text_model = lm.model
    layers = text_model.layers
    num_layers = len(layers)

    generated_tokens = []
    token_times = []
    all_layer_timings = []
    peak_mem = get_mem_gb()
    file_cache = {}  # Cache mx.load() dicts across tokens to avoid re-parsing safetensors headers

    from mlx_lm.models.qwen3_5 import create_attention_mask, create_ssm_mask

    for token_idx in range(max_tokens):
        t_token_start = time.time()

        # --- Embed ---
        h = text_model.embed_tokens(input_ids)
        mx.eval(h)

        # --- Create masks ---
        fa_mask = create_attention_mask(h, cache[text_model.fa_idx])
        ssm_mask = create_ssm_mask(h, cache[text_model.ssm_idx])

        layer_timings = []

        # --- Per-layer: load, compute, clear ---
        for i in range(num_layers):
            layer = layers[i]
            c = cache[i]

            # (a) Load this layer's weights from safetensors
            t_load = time.time()
            entries = weight_index.get(i, [])
            if entries:
                by_file = defaultdict(list)
                for name, filepath in entries:
                    by_file[filepath].append(name)

                layer_weights = []
                for filepath, names in by_file.items():
                    # mx.load() returns lazy mmap'd arrays — cache the dict so we
                    # only parse each safetensors file header once across all tokens
                    if filepath not in file_cache:
                        file_cache[filepath] = mx.load(filepath)
                    all_tensors = file_cache[filepath]
                    for name in names:
                        if name in all_tensors:
                            san_name = name
                            if san_name.startswith("language_model."):
                                san_name = san_name[len("language_model."):]
                            layer_weights.append((san_name, all_tensors[name]))

                lm.load_weights(layer_weights, strict=False)
                if not lazy_eval:
                    # Eager: pre-materialize all ~1.3GB of layer weights from SSD
                    mx.eval(layers[i].parameters())
                # else: lazy — skip mx.eval(params), let the forward pass
                # trigger lazy mmap eval so only accessed expert pages are read
                del layer_weights

            load_time = time.time() - t_load

            # (b) Compute this layer
            mask = ssm_mask if layer.is_linear else fa_mask

            t_compute = time.time()
            h = layer(h, mask=mask, cache=c)
            mx.eval(h)
            compute_time = time.time() - t_compute

            # (c) Clear this layer's weights — free ~1.3GB
            t_clear = time.time()
            clear_layer_weights(model, i)
            clear_time = time.time() - t_clear

            layer_timings.append({
                "layer": i,
                "is_linear": layer.is_linear,
                "load_ms": load_time * 1000,
                "compute_ms": compute_time * 1000,
                "clear_ms": clear_time * 1000,
            })

        all_layer_timings.append(layer_timings)

        # --- Norm + LM head (already pinned) ---
        h = text_model.norm(h)
        if lm.args.tie_word_embeddings:
            logits = text_model.embed_tokens.as_linear(h)
        else:
            logits = lm.lm_head(h)
        mx.eval(logits)

        # --- Sample ---
        next_token = mx.argmax(logits[:, -1, :], axis=-1)
        mx.eval(next_token)

        token_id = next_token.item()
        generated_tokens.append(token_id)

        t_token_end = time.time()
        token_time = t_token_end - t_token_start
        token_times.append(token_time)

        cur_mem = get_mem_gb()
        peak_mem = max(peak_mem, cur_mem)

        # Safety: check system memory pressure every 5 tokens
        if (token_idx + 1) % 5 == 0:
            pressure, free_pct = check_memory_pressure()
            if pressure == "critical":
                print(f"\n  ABORT: System memory free={free_pct}% (critical). "
                      f"Stopping to protect system stability.")
                break

        # Progress
        total_load = sum(lt["load_ms"] for lt in layer_timings)
        total_compute = sum(lt["compute_ms"] for lt in layer_timings)
        total_clear = sum(lt["clear_ms"] for lt in layer_timings)

        if token_idx == 0:
            ttft_ms = token_time * 1000
            print(f"  [{fmt_time(t_token_end - t_start)}] Token 1/{max_tokens}: "
                  f"ttft={ttft_ms:.0f}ms (load={total_load:.0f}ms compute={total_compute:.0f}ms "
                  f"clear={total_clear:.0f}ms mem={cur_mem:.1f}GB)")
        elif (token_idx + 1) % 5 == 0 or token_idx == max_tokens - 1:
            elapsed = t_token_end - t_start
            avg_tps = (token_idx + 1) / elapsed
            print(f"  [{fmt_time(elapsed)}] Token {token_idx+1}/{max_tokens}: "
                  f"{avg_tps:.2f} tok/s (load={total_load:.0f}ms compute={total_compute:.0f}ms "
                  f"clear={total_clear:.0f}ms mem={cur_mem:.1f}GB)")

        # Next iteration input
        input_ids = next_token.reshape(1, 1)

    total_time = time.time() - t_start
    total_tokens = len(generated_tokens)
    text = tokenizer.decode(generated_tokens)

    # Aggregate layer timing stats (skip first token — prompt processing is different)
    if all_layer_timings and len(all_layer_timings) > 1:
        gen_timings = all_layer_timings[1:]
        avg_load = np.mean([sum(lt["load_ms"] for lt in tt) for tt in gen_timings])
        avg_compute = np.mean([sum(lt["compute_ms"] for lt in tt) for tt in gen_timings])
        avg_clear = np.mean([sum(lt["clear_ms"] for lt in tt) for tt in gen_timings])

        num_layers_t = len(gen_timings[0])
        per_layer_load = [np.mean([tt[i]["load_ms"] for tt in gen_timings]) for i in range(num_layers_t)]
        per_layer_compute = [np.mean([tt[i]["compute_ms"] for tt in gen_timings]) for i in range(num_layers_t)]
    else:
        avg_load = 0
        avg_compute = 0
        avg_clear = 0
        per_layer_load = []
        per_layer_compute = []

    return {
        "text": text,
        "tokens": total_tokens,
        "total_time": total_time,
        "tok_sec": total_tokens / total_time if total_time > 0 else 0,
        "ttft_ms": token_times[0] * 1000 if token_times else 0,
        "peak_mem_gb": peak_mem,
        "avg_load_ms_per_token": avg_load,
        "avg_compute_ms_per_token": avg_compute,
        "avg_clear_ms_per_token": avg_clear,
        "per_layer_load_ms": per_layer_load,
        "per_layer_compute_ms": per_layer_compute,
    }


def split_layer_entries(entries):
    """Split a layer's weight index entries into non-expert and expert entries.

    Expert entries are the stacked expert weight tensors inside SwitchGLU:
        switch_mlp.{gate_proj,up_proj,down_proj}.{weight,scales,biases}
    Everything else (layer norms, attention/SSM, router, shared expert) is non-expert.

    Returns (non_expert_entries, expert_entries).
    """
    expert_pattern = re.compile(r'\.switch_mlp\.(gate_proj|up_proj|down_proj)\.(weight|scales|biases)$')
    non_expert = []
    expert = []
    for name, filepath in entries:
        if expert_pattern.search(name):
            expert.append((name, filepath))
        else:
            non_expert.append((name, filepath))
    return non_expert, expert


def generate_offload_selective(model, tokenizer, prompt, max_tokens, weight_index, model_path):
    """Generate tokens with selective expert loading for MoE models.

    At startup: pre-load all non-expert weights (~2.3GB) for all layers into DRAM.
    For each token, for each layer:
      Phase 2: Run attention + router (weights already resident, no loading needed)
      Phase 3: Load ONLY the 8 selected expert slices (~40MB), run MoE computation
      Phase 4: Clear only expert weights (keep attention/norms/shared_expert resident)

    This reduces per-token I/O to just expert slices (~40MB/layer), eliminating the
    ~38s of non-expert loading overhead from the previous implementation.
    """
    t_start = time.time()

    input_ids = mx.array(tokenizer.encode(prompt))[None, :]
    cache = model.make_cache()

    lm = model.language_model
    text_model = lm.model
    layers = text_model.layers
    num_layers = len(layers)

    generated_tokens = []
    token_times = []
    all_layer_timings = []
    peak_mem = get_mem_gb()

    from mlx_lm.models.qwen3_5 import create_attention_mask, create_ssm_mask

    # === Pre-load all non-expert weights using DIRECT I/O (not mmap) ===
    # mmap-backed arrays get evicted from page cache when expert weights are loaded.
    # Direct I/O creates real Metal allocations that stay pinned in DRAM.
    t_preload = time.time()
    header_cache = {}  # filepath -> (header_dict, data_start_offset)
    all_nonexpert_weights = []

    # Group all non-expert tensor names by filepath for sequential I/O
    file_to_names = defaultdict(list)
    file_to_san = {}  # (filepath, name) -> sanitized_name
    for layer_i in range(num_layers):
        entries = weight_index.get(layer_i, [])
        non_expert, _ = split_layer_entries(entries)
        for name, filepath in non_expert:
            file_to_names[filepath].append(name)
            san_name = name
            if san_name.startswith("language_model."):
                san_name = san_name[len("language_model."):]
            file_to_san[(filepath, name)] = san_name

    # Read from each file sequentially (sorted by offset within file)
    for filepath, names in sorted(file_to_names.items()):
        tensors = read_tensors_direct(filepath, names, header_cache, file_handle_cache=None)
        for name in names:
            if name in tensors:
                san_name = file_to_san[(filepath, name)]
                all_nonexpert_weights.append((san_name, tensors[name]))
        print(f"    read {len(names)} tensors from {Path(filepath).name} ({get_mem_gb():.1f}GB)", flush=True)

    print(f"    Total non-expert weights to load: {len(all_nonexpert_weights)}")
    lm.load_weights(all_nonexpert_weights, strict=False)
    # Skip mx.eval — direct I/O arrays are already real Metal allocations (not mmap-backed).
    # BF16 astype results will be lazily evaluated during first forward pass (cheap).
    del all_nonexpert_weights
    preload_time = time.time() - t_preload
    print(f"  Pre-loaded non-expert weights for {num_layers} layers in {preload_time:.1f}s ({get_mem_gb():.1f}GB)")

    # === File handle cache: keep safetensors files open for duration of inference ===
    # Avoids ~3456 open/close cycles per token (72 tensor reads × 48 layers).
    file_handle_cache = {}

    # === LRU cache for expert weight slices ===
    # 1536 entries = 48 layers × 8 experts × 4 tokens worth. ~7.6GB max.
    # Better tok/s per GB of memory than 3072 entries.
    expert_cache = ExpertCache(max_entries=1536)

    # === Cache quantization parameters from model config (read once) ===
    # These are needed by compute_moe_direct for mx.gather_qmm calls.
    with open(model_path / "config.json") as f:
        _cfg = json.load(f)
    _qcfg = _cfg.get("quantization", _cfg.get("quantization_config", {}))
    qparams = {
        "group_size": _qcfg.get("group_size", 64),
        "bits": _qcfg.get("bits", 4),
        "mode": _qcfg.get("mode", "affine"),
    }
    del _cfg, _qcfg

    # === I/O instrumentation counters (per-token, reset each token) ===
    io_stats_history = []  # list of per-token dicts

    for token_idx in range(max_tokens):
        t_token_start = time.time()

        # Per-token I/O counters
        token_io_bytes = 0
        token_io_seeks = 0
        token_io_time = 0.0
        token_array_time = 0.0
        token_moe_compute_time = 0.0

        # --- Embed ---
        h = text_model.embed_tokens(input_ids)
        mx.eval(h)

        # --- Create masks ---
        fa_mask = create_attention_mask(h, cache[text_model.fa_idx])
        ssm_mask = create_ssm_mask(h, cache[text_model.ssm_idx])

        layer_timings = []

        # --- Per-layer: selective load, compute (clearing deferred to end of token) ---
        for i in range(num_layers):
            layer = layers[i]
            c = cache[i]

            entries = weight_index.get(i, [])
            _, expert_entries = split_layer_entries(entries)

            # ====== Phase 2: Run attention + router (weights already resident) ======
            t_attn = time.time()

            x_normed = layer.input_layernorm(h)
            mask = ssm_mask if layer.is_linear else fa_mask
            if layer.is_linear:
                r = layer.linear_attn(x_normed, mask, c)
            else:
                r = layer.self_attn(x_normed, mask, c)
            h_mid = h + r
            mx.eval(h_mid)

            # Run router to discover which experts are needed
            h_post = layer.post_attention_layernorm(h_mid)
            gates = layer.mlp.gate(h_post)
            gates = mx.softmax(gates, axis=-1, precise=True)
            k = layer.mlp.top_k
            inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
            scores = mx.take_along_axis(gates, inds, axis=-1)
            scores = scores / scores.sum(axis=-1, keepdims=True)
            mx.eval(inds)

            attn_router_time = time.time() - t_attn

            # ====== Phase 3: Selective expert loading + compute ======
            t_expert = time.time()

            # Extract UNIQUE expert IDs across all positions
            inds_np = np.array(inds.tolist())
            unique_experts = np.unique(inds_np)
            num_unique = len(unique_experts)
            unique_list = unique_experts.tolist()  # Python list for indexing

            # Build remap table: original expert ID -> index in sliced tensor
            remap = np.zeros(layer.mlp.num_experts, dtype=np.int32)
            remap[unique_experts] = np.arange(num_unique)
            remapped_inds = mx.array(remap[inds_np])

            # Build a map from expert tensor name -> filepath for this layer
            expert_file_map = {}
            for name, filepath in expert_entries:
                expert_file_map[name] = filepath

            # --- LRU cache: determine which experts need disk reads ---
            uncached_list = []
            for idx in unique_list:
                if expert_cache.has_expert(i, idx):
                    expert_cache.record_hit()
                    # Touch the entry so it won't be evicted during put_attr
                    # for uncached experts below
                    expert_cache.touch(i, idx)
                else:
                    expert_cache.record_miss()
                    uncached_list.append(idx)

            # Read only uncached experts from disk (threaded: one expert per thread)
            if uncached_list:
                # Pre-populate header_cache for all expert files (thread-safe after this)
                for filepath in set(expert_file_map.values()):
                    if filepath not in header_cache:
                        header_cache[filepath] = parse_safetensors_header(filepath)

                # Parallel read: each thread reads all 9 attrs for one expert
                with ThreadPoolExecutor(max_workers=min(4, len(uncached_list))) as executor:
                    futures = [
                        executor.submit(
                            _read_single_expert_attrs,
                            expert_idx, i, expert_file_map, header_cache
                        )
                        for expert_idx in uncached_list
                    ]
                    for future in futures:
                        eidx, attrs, io_stats = future.result()
                        # Accumulate I/O stats from this expert read
                        token_io_bytes += io_stats["bytes_read"]
                        token_io_seeks += io_stats["seek_count"]
                        token_io_time += io_stats["io_time_s"]
                        token_array_time += io_stats["array_time_s"]
                        for (proj_name, attr_name), arr in attrs.items():
                            expert_cache.put_attr(i, eidx, proj_name, attr_name, arr)

            # Assemble [num_unique, ...] weight tensors into a dict for direct computation
            expert_tensors = {}
            for proj_name in ["gate_proj", "up_proj", "down_proj"]:
                for attr_name in ["weight", "scales", "biases"]:
                    slices = []
                    for idx in unique_list:
                        arr = expert_cache.get_attr(i, idx, proj_name, attr_name)
                        if arr is not None:
                            slices.append(arr)
                        else:
                            raise RuntimeError(
                                f"Expert cache miss: layer={i} expert={idx} "
                                f"{proj_name}.{attr_name} not found after loading"
                            )
                    if slices:
                        expert_tensors[f"{proj_name}.{attr_name}"] = mx.stack(slices, axis=0)

            # Force-eval the assembled expert weight tensors
            mx.eval(*expert_tensors.values())

            # Run expert MoE computation directly via mx.gather_qmm (no model weight mutation)
            t_moe = time.time()
            y = compute_moe_direct(
                h_post, remapped_inds, expert_tensors,
                group_size=qparams["group_size"],
                bits=qparams["bits"],
                mode=qparams["mode"],
            )
            y = (y * scores[..., None]).sum(axis=-2)

            # Run shared expert (already loaded in phase 1)
            shared_y = layer.mlp.shared_expert(h_post)
            shared_y = mx.sigmoid(layer.mlp.shared_expert_gate(h_post)) * shared_y
            y = y + shared_y

            h = h_mid + y
            mx.eval(h)
            token_moe_compute_time += time.time() - t_moe

            # Delete stacked expert tensors (they're copies from cache, not needed after compute).
            # The actual mx.clear_cache() is done once per token after all layers complete.
            del expert_tensors

            expert_time = time.time() - t_expert

            layer_timings.append({
                "layer": i,
                "is_linear": layer.is_linear,
                "attn_router_ms": attn_router_time * 1000,
                "expert_ms": expert_time * 1000,
                "clear_ms": 0.0,
                "load_ms": expert_time * 1000,  # total I/O for compat (only experts now)
                "compute_ms": attn_router_time * 1000,  # compute portion for compat
            })

        all_layer_timings.append(layer_timings)

        # Free Metal memory from stacked expert tensor copies accumulated across 48 layers.
        # Direct MoE doesn't mutate model weights (no dummy-weight clearing needed), but
        # the Metal allocator still holds dead stacked-tensor memory (~40MB/layer x 48 = ~1.9GB).
        # Without clear_cache(), memory pressure triggers OS paging that slows file I/O.
        t_clear = time.time()
        mx.clear_cache()
        batch_clear_time = time.time() - t_clear

        # --- Norm + LM head (already pinned) ---
        h = text_model.norm(h)
        if lm.args.tie_word_embeddings:
            logits = text_model.embed_tokens.as_linear(h)
        else:
            logits = lm.lm_head(h)
        mx.eval(logits)

        # --- Sample ---
        next_token = mx.argmax(logits[:, -1, :], axis=-1)
        mx.eval(next_token)

        token_id = next_token.item()
        generated_tokens.append(token_id)

        t_token_end = time.time()
        token_time = t_token_end - t_token_start
        token_times.append(token_time)

        cur_mem = get_mem_gb()
        peak_mem = max(peak_mem, cur_mem)

        # Store I/O stats for this token
        io_stats_history.append({
            "bytes_read": token_io_bytes,
            "seek_count": token_io_seeks,
            "io_time_s": token_io_time,
            "array_time_s": token_array_time,
            "moe_compute_time_s": token_moe_compute_time,
            "token_time_s": token_time,
        })

        # Safety: check system memory pressure every 5 tokens
        if (token_idx + 1) % 5 == 0:
            pressure, free_pct = check_memory_pressure()
            if pressure == "critical":
                print(f"\n  ABORT: System memory free={free_pct}% (critical). "
                      f"Stopping to protect system stability.")
                break

        # Progress
        total_attn_router = sum(lt["attn_router_ms"] for lt in layer_timings)
        total_expert = sum(lt["expert_ms"] for lt in layer_timings)
        total_clear = batch_clear_time * 1000

        cache_hr = expert_cache.hit_rate

        if token_idx == 0:
            ttft_ms = token_time * 1000
            print(f"  [{fmt_time(t_token_end - t_start)}] Token 1/{max_tokens}: "
                  f"ttft={ttft_ms:.0f}ms (attn+router={total_attn_router:.0f}ms "
                  f"expert={total_expert:.0f}ms clear={total_clear:.0f}ms "
                  f"cache_hr={cache_hr:.0%} mem={cur_mem:.1f}GB)")
        elif (token_idx + 1) % 5 == 0 or token_idx == max_tokens - 1:
            elapsed = t_token_end - t_start
            avg_tps = (token_idx + 1) / elapsed
            print(f"  [{fmt_time(elapsed)}] Token {token_idx+1}/{max_tokens}: "
                  f"{avg_tps:.2f} tok/s (attn+router={total_attn_router:.0f}ms "
                  f"expert={total_expert:.0f}ms clear={total_clear:.0f}ms "
                  f"cache_hr={cache_hr:.0%} mem={cur_mem:.1f}GB)")

        # Next iteration input
        input_ids = next_token.reshape(1, 1)

    # Close cached file handles
    for fh in file_handle_cache.values():
        try:
            fh.close()
        except Exception:
            pass

    total_time = time.time() - t_start
    total_tokens = len(generated_tokens)
    text = tokenizer.decode(generated_tokens)

    # === I/O instrumentation summary (skip first token — prompt processing is different) ===
    if len(io_stats_history) > 1:
        gen_io = io_stats_history[1:]  # skip prompt token
        avg_bytes = np.mean([s["bytes_read"] for s in gen_io])
        avg_seeks = np.mean([s["seek_count"] for s in gen_io])
        avg_io_ms = np.mean([s["io_time_s"] for s in gen_io]) * 1000
        avg_arr_ms = np.mean([s["array_time_s"] for s in gen_io]) * 1000
        avg_moe_ms = np.mean([s["moe_compute_time_s"] for s in gen_io]) * 1000
        avg_tok_ms = np.mean([s["token_time_s"] for s in gen_io]) * 1000
        avg_other_ms = avg_tok_ms - avg_io_ms - avg_arr_ms - avg_moe_ms
        total_bytes = sum(s["bytes_read"] for s in gen_io)
        total_seeks = sum(s["seek_count"] for s in gen_io)

        print(f"\n  === I/O Instrumentation Summary (tokens 2-{total_tokens}) ===")
        print(f"  Avg bytes/token:  {avg_bytes/1024/1024:.1f} MB")
        print(f"  Avg seeks/token:  {avg_seeks:.0f}")
        print(f"  Avg file I/O:     {avg_io_ms:.1f} ms/token")
        print(f"  Avg arr build:    {avg_arr_ms:.1f} ms/token")
        print(f"  Avg MoE compute:  {avg_moe_ms:.1f} ms/token")
        print(f"  Avg other:        {avg_other_ms:.1f} ms/token (attn+router+clear+overhead)")
        print(f"  Avg total:        {avg_tok_ms:.1f} ms/token")
        print(f"  Total disk read:  {total_bytes/1024/1024:.1f} MB across {total_seeks} seeks")
        if avg_io_ms > 0:
            bw = (avg_bytes / 1024 / 1024) / (avg_io_ms / 1000)
            print(f"  Effective disk BW: {bw:.0f} MB/s")
    else:
        avg_bytes = 0
        avg_seeks = 0
        avg_io_ms = 0
        avg_arr_ms = 0
        avg_moe_ms = 0

    # Aggregate layer timing stats (skip first token — prompt processing is different)
    if all_layer_timings and len(all_layer_timings) > 1:
        gen_timings = all_layer_timings[1:]
        avg_load = np.mean([sum(lt["load_ms"] for lt in tt) for tt in gen_timings])
        avg_compute = np.mean([sum(lt["compute_ms"] for lt in tt) for tt in gen_timings])
        avg_clear = np.mean([sum(lt["clear_ms"] for lt in tt) for tt in gen_timings])
        avg_expert = np.mean([sum(lt["expert_ms"] for lt in tt) for tt in gen_timings])
        avg_attn_router = np.mean([sum(lt["attn_router_ms"] for lt in tt) for tt in gen_timings])

        num_layers_t = len(gen_timings[0])
        per_layer_load = [np.mean([tt[j]["load_ms"] for tt in gen_timings]) for j in range(num_layers_t)]
        per_layer_compute = [np.mean([tt[j]["compute_ms"] for tt in gen_timings]) for j in range(num_layers_t)]
    else:
        avg_load = 0
        avg_compute = 0
        avg_clear = 0
        avg_expert = 0
        avg_attn_router = 0
        per_layer_load = []
        per_layer_compute = []

    return {
        "text": text,
        "tokens": total_tokens,
        "total_time": total_time,
        "tok_sec": total_tokens / total_time if total_time > 0 else 0,
        "ttft_ms": token_times[0] * 1000 if token_times else 0,
        "peak_mem_gb": peak_mem,
        "preload_time": preload_time,
        "avg_load_ms_per_token": avg_load,
        "avg_compute_ms_per_token": avg_compute,
        "avg_clear_ms_per_token": avg_clear,
        "avg_expert_ms_per_token": avg_expert,
        "avg_attn_router_ms_per_token": avg_attn_router,
        "per_layer_load_ms": per_layer_load,
        "per_layer_compute_ms": per_layer_compute,
        "expert_cache_hits": expert_cache.hits,
        "expert_cache_misses": expert_cache.misses,
        "expert_cache_hit_rate": expert_cache.hit_rate,
        "expert_cache_entries": len(expert_cache.cache),
        "avg_io_bytes_per_token": avg_bytes,
        "avg_io_seeks_per_token": avg_seeks,
        "avg_io_ms_per_token": avg_io_ms,
        "avg_arr_ms_per_token": avg_arr_ms,
        "avg_moe_compute_ms_per_token": avg_moe_ms,
    }


def load_model_no_weights(model_path):
    """Create model architecture with quantization but load NO weights at all.
    For offload mode: weights are loaded per-layer during inference, not at init.
    This avoids mmap'ing 61GB of safetensors which causes OS thrashing on <DRAM machines."""
    model_path = Path(model_path)

    # 1. Load config
    with open(model_path / "config.json") as f:
        config = json.load(f)

    # 2. Create model architecture (empty — no weights)
    from mlx_lm.utils import _get_classes
    model_class, model_args_class = _get_classes(config)
    model_args = model_args_class.from_dict(config)
    model = model_class(model_args)

    # 3. Apply quantization (sets up quantized module structure, still no weights)
    qconfig = config.get("quantization", config.get("quantization_config", {}))
    if qconfig:
        nn.quantize(model, bits=qconfig["bits"], group_size=qconfig["group_size"])

    model.eval()

    # 4. Load ONLY global weights (embed_tokens, norm, lm_head) — ~1GB total
    weight_index = build_weight_index(model_path)
    global_entries = weight_index.get("global", [])
    if global_entries:
        by_file = defaultdict(list)
        for name, filepath in global_entries:
            by_file[filepath].append(name)

        global_weights = []
        for filepath, names in by_file.items():
            all_tensors = mx.load(filepath)
            for name in names:
                if name in all_tensors:
                    global_weights.append((name, all_tensors[name]))
            # Don't keep reference to all_tensors — let it be GC'd
            del all_tensors

        model.load_weights(global_weights, strict=False)
        # Force-eval global weights so they're resident in DRAM
        lm = model.language_model
        text_model = lm.model
        mx.eval(text_model.embed_tokens.parameters())
        mx.eval(text_model.norm.parameters())
        if hasattr(lm, 'lm_head'):
            mx.eval(lm.lm_head.parameters())
        del global_weights

    # 5. Load tokenizer
    from mlx_lm.utils import load_tokenizer
    eos_ids = config.get("eos_token_id", [])
    if not isinstance(eos_ids, list):
        eos_ids = [eos_ids]
    tokenizer = load_tokenizer(model_path, eos_token_ids=eos_ids)

    return model, tokenizer


def load_model_custom(model_path):
    """Custom model loader that bypasses mlx_lm.load() overhead.
    Loads model architecture + lazy weights directly. Fast even for 100GB+ models."""
    import glob

    model_path = Path(model_path)

    # 1. Load config
    with open(model_path / "config.json") as f:
        config = json.load(f)

    # 2. Load all weights lazily (mmap, no actual I/O)
    weight_files = sorted(glob.glob(str(model_path / "model*.safetensors")))
    weights = {}
    for wf in weight_files:
        weights.update(mx.load(wf))

    # 3. Create model architecture
    from mlx_lm.utils import _get_classes
    model_class, model_args_class = _get_classes(config)
    model_args = model_args_class.from_dict(config)
    model = model_class(model_args)

    # 4. Sanitize weights (transforms HF format to MLX format)
    if hasattr(model, "sanitize"):
        weights = model.sanitize(weights)

    # 5. Apply quantization
    qconfig = config.get("quantization", config.get("quantization_config", {}))
    if qconfig:
        nn.quantize(model, bits=qconfig["bits"], group_size=qconfig["group_size"])

    # 6. Load weights into model (still lazy — no actual I/O)
    model.eval()
    model.load_weights(list(weights.items()), strict=False)

    # 7. Load tokenizer separately
    from mlx_lm.utils import load_tokenizer
    eos_ids = config.get("eos_token_id", [])
    if not isinstance(eos_ids, list):
        eos_ids = [eos_ids]
    tokenizer = load_tokenizer(model_path, eos_token_ids=eos_ids)

    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(description="Streaming inference engine")
    parser.add_argument("--model", required=True, help="HuggingFace model ID or local path")
    parser.add_argument("--tokens", type=int, default=20, help="Tokens to generate")
    parser.add_argument("--prompt", default="Explain the theory of relativity in simple terms.",
                        help="Prompt for generation")
    parser.add_argument("--mode", choices=["baseline", "layerwise", "stream", "lazy", "offload", "offload_lazy",
                                          "offload_selective"],
                        default="stream",
                        help="baseline=mlx_lm native, layerwise=manual forward with timing, "
                             "stream=reload weights from safetensors per layer, "
                             "lazy=load with lazy=True (mmap, OS handles paging), "
                             "offload=per-layer load/compute/clear for models larger than DRAM, "
                             "offload_lazy=like offload but skips mx.eval(params) so only "
                             "accessed expert pages are read via lazy mmap (~40MB vs ~1.3GB/layer), "
                             "offload_selective=run router first, then load only 8/256 selected "
                             "expert slices per layer (~40MB vs ~1.3GB, ~32x I/O reduction)")
    parser.add_argument("--max-mem-gb", type=float, default=40.0,
                        help="Abort if RSS exceeds this (GB)")
    args = parser.parse_args()

    t_start = time.time()
    mem_before = get_mem_gb()

    print(f"[{fmt_time(0)}] Mode: {args.mode}")
    print(f"[{fmt_time(0)}] Loading model: {args.model}")
    print(f"[{fmt_time(0)}] Memory before load: {mem_before:.1f} GB")

    # Resolve model path (for safetensors access)
    model_path = resolve_model_path(args.model)

    if args.mode in ("offload", "offload_lazy", "offload_selective"):
        # Offload mode: create empty model, load ONLY global weights (~1GB).
        # Layer weights are loaded/cleared per-layer during inference.
        # This is the only mode that works for model_size > DRAM.
        # offload_lazy variant skips mx.eval(params) to test lazy mmap expert paging.
        # offload_selective variant runs router first, then loads only selected expert slices.
        print(f"[{fmt_time(time.time() - t_start)}] Using offload loader (no layer weights)...")
        model, tokenizer = load_model_no_weights(model_path)
        # Cap wired memory — we only need ~2-3GB active at any time
        wired_gb = min(args.max_mem_gb * 0.4, 20)  # conservative: 40% of limit or 20GB max
        mx.set_wired_limit(int(wired_gb * 1024**3))
        mode_note = ""
        if args.mode == "offload_lazy":
            mode_note = " [lazy_eval=True]"
        elif args.mode == "offload_selective":
            mode_note = " [selective expert loading]"
        print(f"[{fmt_time(time.time() - t_start)}] Loaded (global weights only). "
              f"wired limit={wired_gb:.0f}GB{mode_note}")
    elif args.mode == "lazy":
        # Lazy mode: load all weights via mmap, let OS handle paging.
        # WARNING: thrashes on models larger than DRAM. Use offload instead.
        print(f"[{fmt_time(time.time() - t_start)}] Using custom loader (lazy mmap)...")
        model, tokenizer = load_model_custom(model_path)
        # Pin only embed_tokens, norm, and lm_head in DRAM
        lm = model.language_model
        text_model = lm.model
        mx.eval(text_model.embed_tokens.parameters())
        mx.eval(text_model.norm.parameters())
        if hasattr(lm, 'lm_head'):
            mx.eval(lm.lm_head.parameters())
        # Cap wired memory to leave room for the rest of the system
        wired_gb = min(args.max_mem_gb * 0.6, 28)  # ~60% of limit or 28GB max
        mx.set_wired_limit(int(wired_gb * 1024**3))
        print(f"[{fmt_time(time.time() - t_start)}] Loaded. Pinned essentials, "
              f"wired limit={wired_gb:.0f}GB")
    else:
        # Full load: everything in DRAM
        model, tokenizer = mlx_lm.load(str(model_path))
        mx.eval(model.parameters())

    mem_after_load = get_mem_gb()
    t_loaded = time.time() - t_start
    print(f"[{fmt_time(t_loaded)}] Model loaded. Memory: {mem_after_load:.1f} GB "
          f"(+{mem_after_load - mem_before:.1f} GB)")

    if args.mode not in ("lazy", "offload", "offload_lazy", "offload_selective") and mem_after_load > args.max_mem_gb:
        print(f"ABORT: Memory {mem_after_load:.1f} GB exceeds limit {args.max_mem_gb} GB")
        sys.exit(1)

    # Count parameters
    total_params = sum(p.size for n, p in mlx.utils.tree_flatten(model.parameters()))
    params_b = total_params / 1e9

    # Build weight index for streaming/offload modes
    weight_index = build_weight_index(model_path) if args.mode in ("stream", "offload", "offload_lazy", "offload_selective") else None

    # Generate
    print(f"[{fmt_time(time.time() - t_start)}] Generating {args.tokens} tokens ({args.mode})...")
    print(f"[{fmt_time(time.time() - t_start)}] Prompt: {args.prompt[:80]}{'...' if len(args.prompt) > 80 else ''}")

    if args.mode == "baseline":
        result = generate_baseline(model, tokenizer, args.prompt, args.tokens)
    elif args.mode == "offload_selective":
        # Selective offload: run attention+router first, then load only 8/256 expert slices.
        # ~32x I/O reduction vs full offload (~40MB vs ~1.3GB per layer).
        result = generate_offload_selective(model, tokenizer, args.prompt, args.tokens,
                                            weight_index, model_path)
    elif args.mode in ("offload", "offload_lazy"):
        # Offload mode: explicit per-layer load -> compute -> clear cycle.
        # Only way to run models larger than DRAM without OS thrashing.
        # offload_lazy: skip mx.eval(params) so lazy mmap only reads accessed experts.
        result = generate_offload(model, tokenizer, args.prompt, args.tokens,
                                  weight_index, model_path,
                                  lazy_eval=(args.mode == "offload_lazy"))
    elif args.mode == "lazy":
        # Lazy mode: all weights mmap'd, OS handles paging
        result = generate_manual(model, tokenizer, args.prompt, args.tokens,
                                 weight_index=None, mode="layerwise")
    else:
        result = generate_manual(model, tokenizer, args.prompt, args.tokens,
                                 weight_index=weight_index, mode=args.mode)

    # Summary
    peak_mem = max(result["peak_mem_gb"], get_mem_gb())
    print(f"\n[{fmt_time(time.time() - t_start)}] Done. {result['tokens']} tokens in "
          f"{result['total_time']:.1f}s")
    print(f"Generated: {result['text'][:200]}{'...' if len(result['text']) > 200 else ''}")
    print()

    # Mode-specific stats
    if args.mode in ("layerwise", "stream", "lazy", "offload", "offload_lazy", "offload_selective"):
        print(f"Per-token breakdown (generation phase, excluding prompt):")
        print(f"  Avg weight load: {result.get('avg_load_ms_per_token', 0):.1f}ms")
        print(f"  Avg compute:     {result.get('avg_compute_ms_per_token', 0):.1f}ms")
        if result.get('avg_clear_ms_per_token', 0) > 0:
            print(f"  Avg clear:       {result.get('avg_clear_ms_per_token', 0):.1f}ms")
        if args.mode == "offload_selective":
            print(f"  Avg non-expert load: {result.get('avg_load_nonexpert_ms_per_token', 0):.1f}ms")
            print(f"  Avg attn+router:     {result.get('avg_attn_router_ms_per_token', 0):.1f}ms")
            print(f"  Avg expert load+run: {result.get('avg_expert_ms_per_token', 0):.1f}ms")

        if result.get("per_layer_compute_ms"):
            linear_times = []
            fa_times = []
            layers = model.language_model.model.layers
            for i, t in enumerate(result["per_layer_compute_ms"]):
                if layers[i].is_linear:
                    linear_times.append(t)
                else:
                    fa_times.append(t)

            if linear_times:
                print(f"  Avg linear_attn layer: {np.mean(linear_times):.2f}ms "
                      f"({len(linear_times)} layers)")
            if fa_times:
                print(f"  Avg full_attn layer:   {np.mean(fa_times):.2f}ms "
                      f"({len(fa_times)} layers)")

        if result.get("per_layer_load_ms"):
            avg_load_per_layer = np.mean(result["per_layer_load_ms"])
            total_load = sum(result["per_layer_load_ms"])
            print(f"  Avg safetensors load/layer: {avg_load_per_layer:.2f}ms")
            print(f"  Total load per token: {total_load:.0f}ms")
        print()

    # Compute active params (MoE heuristic)
    active_b = params_b
    model_name = args.model.split("/")[-1] if "/" in args.model else args.model
    match = re.search(r'A(\d+)B', model_name, re.IGNORECASE)
    if match:
        active_b = float(match.group(1))

    # Machine-parseable result line
    print(f"RESULT model={model_name} params_B={params_b:.1f} active_B={active_b:.1f} "
          f"tok_sec={result['tok_sec']:.2f} ttft_ms={result['ttft_ms']:.0f} "
          f"mem_gb={peak_mem:.1f} tokens={result['tokens']} mode={args.mode}")


if __name__ == "__main__":
    main()
