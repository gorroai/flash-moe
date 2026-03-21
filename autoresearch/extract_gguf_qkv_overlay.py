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
QKV_RE = re.compile(r"^blk\.(\d+)\.attn_qkv\.weight$")
LINEAR_NUM_K_HEADS = 16
LINEAR_NUM_V_HEADS = 64
LINEAR_HEAD_DIM = 128


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


def untile_qwen35_linear_v_rows(chunk: bytes, in_dim: int, out_dim: int, type_size: int) -> bytes:
    q_dim = LINEAR_NUM_K_HEADS * LINEAR_HEAD_DIM
    k_dim = LINEAR_NUM_K_HEADS * LINEAR_HEAD_DIM
    v_dim = LINEAR_NUM_V_HEADS * LINEAR_HEAD_DIM
    if out_dim != q_dim + k_dim + v_dim:
        raise SystemExit(
            f"Unexpected attn_qkv out_dim {out_dim}; expected {q_dim + k_dim + v_dim}"
        )

    row_size = (in_dim // 32) * type_size
    if len(chunk) != out_dim * row_size:
        raise SystemExit(
            f"Unexpected attn_qkv byte size {len(chunk)} for in_dim={in_dim}, out_dim={out_dim}"
        )

    num_v_per_k = LINEAR_NUM_V_HEADS // LINEAR_NUM_K_HEADS
    qk_bytes = (q_dim + k_dim) * row_size
    v_src = memoryview(chunk)[qk_bytes:]
    v_dst = bytearray(v_dim * row_size)

    for grouped_v_head in range(LINEAR_NUM_V_HEADS):
        k_head = grouped_v_head // num_v_per_k
        v_index_within_k = grouped_v_head % num_v_per_k
        tiled_v_head = v_index_within_k * LINEAR_NUM_K_HEADS + k_head

        src_off = tiled_v_head * LINEAR_HEAD_DIM * row_size
        dst_off = grouped_v_head * LINEAR_HEAD_DIM * row_size
        span = LINEAR_HEAD_DIM * row_size
        v_dst[dst_off:dst_off + span] = v_src[src_off:src_off + span]

    return chunk[:qk_bytes] + bytes(v_dst)


def main() -> int:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=str(DEFAULT_CONFIG))
    pre_args, _ = pre.parse_known_args()
    cfg = load_config(Path(pre_args.config))

    parser = argparse.ArgumentParser(description="Extract GGUF Q8_0 attn_qkv tensors into a standalone raw overlay blob")
    parser.add_argument("--config", default=pre_args.config)
    parser.add_argument("--gguf", default=cfg.get("gguf_source"), help="GGUF shard or directory")
    parser.add_argument("--llama-cpp-root", default=cfg.get("llama_cpp_root"), help="Local llama.cpp root for gguf-py")
    parser.add_argument("--out-bin", default=cfg.get("gguf_qkv_overlay_bin", "autoresearch/gguf/attn_qkv_q8_0.bin"))
    parser.add_argument("--out-json", default=cfg.get("gguf_qkv_overlay_json", "autoresearch/gguf/attn_qkv_q8_0.json"))
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
            m = QKV_RE.match(tensor.name)
            if not m:
                continue
            layer = int(m.group(1))
            qtype = tensor.tensor_type
            block_size, type_size = quant_sizes[qtype]
            if qtype.name != "Q8_0":
                raise SystemExit(f"Expected {tensor.name} to be Q8_0, got {qtype.name}")
            if int(block_size) != 32:
                raise SystemExit(f"Expected Q8_0 block size 32 for {tensor.name}, got {block_size}")

            shape = [int(x) for x in tensor.shape]
            in_dim = shape[0]
            out_dim = shape[1]
            chunk = untile_qwen35_linear_v_rows(
                tensor.data.tobytes(),
                in_dim=in_dim,
                out_dim=out_dim,
                type_size=int(type_size),
            )
            collected.append(
                (
                    {
                        "layer": layer,
                        "gguf_name": tensor.name,
                        "runtime_name": f"model.layers.{layer}.linear_attn.in_proj_qkv.weight",
                        "size": len(chunk),
                        "shape": shape,
                        "tensor_type": qtype.name,
                        "block_size": int(block_size),
                        "type_size": int(type_size),
                        "source_shard": shard.name,
                        "source_tensor_data_offset": int(tensor.data_offset),
                        "source_row_layout": "gguf_qwen35_tiled_v",
                        "output_row_layout": "flash_moe_grouped_v",
                    },
                    chunk,
                )
            )

    collected.sort(key=lambda item: item[0]["layer"])
    if not collected:
        raise SystemExit(f"No blk.*.attn_qkv.weight tensors found in {gguf_path}")

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
        "family": "attn_qkv",
        "tensor_type": "Q8_0",
        "block_size": 32,
        "source_row_layout": "gguf_qwen35_tiled_v",
        "output_row_layout": "flash_moe_grouped_v",
        "entries": remapped_entries,
        "output_path": str(out_bin),
        "source_gguf": str(gguf_path),
    }
    out_json.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote {out_bin} ({cursor} bytes)")
    print(f"Wrote {out_json} ({len(remapped_entries)} layers)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
