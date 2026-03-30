#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "autoresearch" / "config.json"
LINEAR_GATE_RE = re.compile(r"^blk\.(\d+)\.attn_gate\.weight$")
LINEAR_OUT_RE = re.compile(r"^blk\.(\d+)\.ssm_out\.weight$")
LINEAR_NUM_K_HEADS = 16
LINEAR_NUM_V_HEADS = 64
LINEAR_HEAD_DIM = 128
GGUF_Q8_0_BLOCK = 32


def load_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def resolve_path(path_text: str | None) -> Path | None:
    if not path_text:
        return None
    path = Path(os.path.expanduser(path_text))
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def import_gguf(llama_cpp_root: Path | None) -> tuple[Any, Any]:
    if llama_cpp_root:
        gguf_py = llama_cpp_root / "gguf-py"
        if gguf_py.exists():
            sys.path.insert(0, str(gguf_py))
    from gguf import GGUFReader  # type: ignore
    from gguf.constants import GGML_QUANT_SIZES  # type: ignore

    return GGUFReader, GGML_QUANT_SIZES


def discover_shards(source: Path) -> list[Path]:
    if source.is_dir():
        shards = sorted(source.glob("*.gguf"))
    else:
        shards = sorted(source.parent.glob("*.gguf"))
    if not shards:
        raise SystemExit(f"No GGUF shards found under {source}")
    return shards


def grouped_to_tiled_v_head(grouped_v_head: int) -> int:
    num_v_per_k = LINEAR_NUM_V_HEADS // LINEAR_NUM_K_HEADS
    k_head = grouped_v_head // num_v_per_k
    v_index_within_k = grouped_v_head % num_v_per_k
    return v_index_within_k * LINEAR_NUM_K_HEADS + k_head


def untile_v_rows(chunk: bytes, in_dim: int, out_dim: int, type_size: int) -> bytes:
    expected_out = LINEAR_NUM_V_HEADS * LINEAR_HEAD_DIM
    if out_dim != expected_out:
        raise SystemExit(f"Unexpected row-reordered out_dim {out_dim}; expected {expected_out}")

    row_size = (in_dim // GGUF_Q8_0_BLOCK) * type_size
    if len(chunk) != out_dim * row_size:
        raise SystemExit(
            f"Unexpected row-reordered byte size {len(chunk)} for in_dim={in_dim}, out_dim={out_dim}"
        )

    dst = bytearray(len(chunk))
    rows_per_head = LINEAR_HEAD_DIM
    head_span = rows_per_head * row_size
    src = memoryview(chunk)

    for grouped_v_head in range(LINEAR_NUM_V_HEADS):
        tiled_v_head = grouped_to_tiled_v_head(grouped_v_head)
        src_off = tiled_v_head * head_span
        dst_off = grouped_v_head * head_span
        dst[dst_off:dst_off + head_span] = src[src_off:src_off + head_span]

    return bytes(dst)


def untile_v_columns(chunk: bytes, in_dim: int, out_dim: int, type_size: int) -> bytes:
    expected_in = LINEAR_NUM_V_HEADS * LINEAR_HEAD_DIM
    if in_dim != expected_in:
        raise SystemExit(f"Unexpected column-reordered in_dim {in_dim}; expected {expected_in}")

    row_size = (in_dim // GGUF_Q8_0_BLOCK) * type_size
    if len(chunk) != out_dim * row_size:
        raise SystemExit(
            f"Unexpected column-reordered byte size {len(chunk)} for in_dim={in_dim}, out_dim={out_dim}"
        )

    blocks_per_head = LINEAR_HEAD_DIM // GGUF_Q8_0_BLOCK
    bytes_per_head = blocks_per_head * type_size
    dst = bytearray(len(chunk))
    src = memoryview(chunk)

    for row in range(out_dim):
        row_off = row * row_size
        row_src = src[row_off:row_off + row_size]
        row_dst = memoryview(dst)[row_off:row_off + row_size]
        for grouped_v_head in range(LINEAR_NUM_V_HEADS):
            tiled_v_head = grouped_to_tiled_v_head(grouped_v_head)
            src_off = tiled_v_head * bytes_per_head
            dst_off = grouped_v_head * bytes_per_head
            row_dst[dst_off:dst_off + bytes_per_head] = row_src[src_off:src_off + bytes_per_head]

    return bytes(dst)


def main() -> int:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=str(DEFAULT_CONFIG))
    pre_args, _ = pre.parse_known_args()
    cfg = load_config(Path(pre_args.config))

    parser = argparse.ArgumentParser(
        description="Extract GGUF Q8_0 linear-attention gate/out tensors into a standalone raw overlay blob"
    )
    parser.add_argument("--config", default=pre_args.config)
    parser.add_argument("--gguf", default=cfg.get("gguf_source"), help="GGUF shard or directory")
    parser.add_argument("--llama-cpp-root", default=cfg.get("llama_cpp_root"), help="Local llama.cpp root for gguf-py")
    parser.add_argument("--out-bin", default=cfg.get("gguf_linear_bin", "autoresearch/gguf/linear_q8_0.bin"))
    parser.add_argument("--out-json", default=cfg.get("gguf_linear_json", "autoresearch/gguf/linear_q8_0.json"))
    args = parser.parse_args()

    gguf_path = resolve_path(args.gguf)
    if not gguf_path:
        raise SystemExit("Missing GGUF source path")
    llama_cpp_root = resolve_path(args.llama_cpp_root)
    out_bin = resolve_path(args.out_bin)
    out_json = resolve_path(args.out_json)
    assert out_bin is not None
    assert out_json is not None

    GGUFReader, quant_sizes = import_gguf(llama_cpp_root)
    shards = discover_shards(gguf_path)

    collected: list[tuple[dict[str, Any], bytes]] = []

    for shard in shards:
        reader = GGUFReader(str(shard), "r")
        for tensor in reader.tensors:
            gate_match = LINEAR_GATE_RE.match(tensor.name)
            out_match = LINEAR_OUT_RE.match(tensor.name)
            if not gate_match and not out_match:
                continue

            layer = int((gate_match or out_match).group(1))
            qtype = tensor.tensor_type
            block_size, type_size = quant_sizes[qtype]
            if qtype.name != "Q8_0":
                raise SystemExit(f"Expected {tensor.name} to be Q8_0, got {qtype.name}")
            if int(block_size) != GGUF_Q8_0_BLOCK:
                raise SystemExit(f"Expected Q8_0 block size 32 for {tensor.name}, got {block_size}")

            shape = [int(x) for x in tensor.shape]
            in_dim = shape[0]
            out_dim = shape[1]
            chunk = tensor.data.tobytes()

            if gate_match:
                role = "gate"
                runtime_name = f"model.layers.{layer}.linear_attn.in_proj_z.weight"
                chunk = untile_v_rows(chunk, in_dim=in_dim, out_dim=out_dim, type_size=int(type_size))
                source_layout = "gguf_qwen35_tiled_v_rows"
                output_layout = "flash_moe_grouped_v_rows"
            else:
                role = "out"
                runtime_name = f"model.layers.{layer}.linear_attn.out_proj.weight"
                chunk = untile_v_columns(chunk, in_dim=in_dim, out_dim=out_dim, type_size=int(type_size))
                source_layout = "gguf_qwen35_tiled_v_cols"
                output_layout = "flash_moe_grouped_v_cols"

            collected.append(
                (
                    {
                        "layer": layer,
                        "role": role,
                        "gguf_name": tensor.name,
                        "runtime_name": runtime_name,
                        "size": len(chunk),
                        "shape": shape,
                        "tensor_type": qtype.name,
                        "block_size": int(block_size),
                        "type_size": int(type_size),
                        "source_shard": shard.name,
                        "source_tensor_data_offset": int(tensor.data_offset),
                        "source_layout": source_layout,
                        "output_layout": output_layout,
                    },
                    chunk,
                )
            )

    collected.sort(key=lambda item: (item[0]["layer"], item[0]["role"]))
    if not collected:
        raise SystemExit(f"No linear gate/out tensors found in {gguf_path}")

    ordered_chunks: list[bytes] = []
    remapped_entries: list[dict[str, Any]] = []
    cursor = 0
    for entry, chunk in collected:
        updated = dict(entry)
        updated["offset"] = cursor
        remapped_entries.append(updated)
        ordered_chunks.append(chunk)
        cursor += len(chunk)

    out_bin.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_bin.write_bytes(b"".join(ordered_chunks))
    meta = {
        "family": "linear_attention_aux",
        "tensor_type": "Q8_0",
        "block_size": GGUF_Q8_0_BLOCK,
        "entries": remapped_entries,
        "output_path": str(out_bin),
        "source_gguf": str(gguf_path),
    }
    out_json.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote {out_bin} ({cursor} bytes)")
    print(f"Wrote {out_json} ({len(remapped_entries)} tensors)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
