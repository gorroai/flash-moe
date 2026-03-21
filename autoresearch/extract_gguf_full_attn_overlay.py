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
FULL_ATTN_RE = re.compile(r"^blk\.(\d+)\.(attn_q|attn_k|attn_v|attn_output)\.weight$")
ROLE_INFO = {
    "attn_q": ("q", "self_attn.q_proj.weight"),
    "attn_k": ("k", "self_attn.k_proj.weight"),
    "attn_v": ("v", "self_attn.v_proj.weight"),
    "attn_output": ("o", "self_attn.o_proj.weight"),
}
ROLE_ORDER = {"q": 0, "k": 1, "v": 2, "o": 3}


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


def main() -> int:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=str(DEFAULT_CONFIG))
    pre_args, _ = pre.parse_known_args()
    cfg = load_config(Path(pre_args.config))

    parser = argparse.ArgumentParser(
        description="Extract GGUF Q8_0 full-attention q/k/v/o tensors into a standalone raw overlay blob"
    )
    parser.add_argument("--config", default=pre_args.config)
    parser.add_argument("--gguf", default=cfg.get("gguf_source"), help="GGUF shard or directory")
    parser.add_argument("--llama-cpp-root", default=cfg.get("llama_cpp_root"), help="Local llama.cpp root for gguf-py")
    parser.add_argument(
        "--out-bin",
        default=cfg.get("gguf_full_attn_bin", cfg.get("gguf_full_attn_overlay_bin", "autoresearch/gguf/full_attn_q8_0.bin")),
        help="Output overlay blob path",
    )
    parser.add_argument(
        "--out-json",
        default=cfg.get("gguf_full_attn_json", cfg.get("gguf_full_attn_overlay_json", "autoresearch/gguf/full_attn_q8_0.json")),
        help="Output overlay manifest path",
    )
    parser.add_argument(
        "--roles",
        default="q,k,v,o",
        help="Comma-separated subset of roles to extract from {q,k,v,o}",
    )
    args = parser.parse_args()

    gguf_path = resolve_path(args.gguf)
    if not gguf_path:
        raise SystemExit("Missing GGUF source path")
    llama_cpp_root = resolve_path(args.llama_cpp_root)
    out_bin = resolve_path(args.out_bin)
    out_json = resolve_path(args.out_json)
    assert out_bin is not None
    assert out_json is not None
    selected_roles = {role.strip() for role in args.roles.split(",") if role.strip()}
    invalid_roles = selected_roles - set(ROLE_ORDER)
    if invalid_roles:
        raise SystemExit(f"Unsupported roles: {sorted(invalid_roles)}")
    if not selected_roles:
        raise SystemExit("No roles selected")

    GGUFReader, quant_sizes = import_gguf(llama_cpp_root)
    shards = discover_shards(gguf_path)

    collected: list[tuple[dict[str, Any], bytes]] = []
    skipped_non_q8: list[tuple[str, str]] = []

    for shard in shards:
        reader = GGUFReader(str(shard), "r")
        for tensor in reader.tensors:
            match = FULL_ATTN_RE.match(tensor.name)
            if not match:
                continue

            layer = int(match.group(1))
            family = match.group(2)
            qtype = tensor.tensor_type
            block_size, type_size = quant_sizes[qtype]
            if qtype.name != "Q8_0":
                skipped_non_q8.append((tensor.name, qtype.name))
                continue
            if int(block_size) != 32:
                raise SystemExit(f"Expected Q8_0 block size 32 for {tensor.name}, got {block_size}")

            role, runtime_suffix = ROLE_INFO[family]
            if role not in selected_roles:
                continue
            chunk = tensor.data.tobytes()
            shape = [int(x) for x in tensor.shape]
            collected.append(
                (
                    {
                        "layer": layer,
                        "role": role,
                        "gguf_name": tensor.name,
                        "runtime_name": f"model.layers.{layer}.{runtime_suffix}",
                        "size": len(chunk),
                        "shape": shape,
                        "tensor_type": qtype.name,
                        "block_size": int(block_size),
                        "type_size": int(type_size),
                        "source_shard": shard.name,
                        "source_tensor_data_offset": int(tensor.data_offset),
                    },
                    chunk,
                )
            )

    collected.sort(key=lambda item: (item[0]["layer"], ROLE_ORDER[item[0]["role"]]))
    if not collected:
        raise SystemExit(f"No Q8_0 full-attention tensors found in {gguf_path}")

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
        "family": "full_attention",
        "tensor_type": "Q8_0",
        "block_size": 32,
        "selected_roles": sorted(selected_roles, key=lambda role: ROLE_ORDER[role]),
        "entries": remapped_entries,
        "output_path": str(out_bin),
        "source_gguf": str(gguf_path),
        "skipped_non_q8": [{"name": name, "tensor_type": qtype} for name, qtype in skipped_non_q8],
    }
    out_json.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote {out_bin} ({cursor} bytes)")
    print(f"Wrote {out_json} ({len(remapped_entries)} tensors)")
    if skipped_non_q8:
        print("Skipped non-Q8 full-attention tensors:")
        for name, qtype in skipped_non_q8:
            print(f"  {name}: {qtype}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
