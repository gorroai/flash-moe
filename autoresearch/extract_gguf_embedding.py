#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "autoresearch" / "config.json"
EMBEDDING_NAME = "token_embd.weight"


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

    parser = argparse.ArgumentParser(description="Extract the GGUF Q8_0 embedding into a standalone raw blob")
    parser.add_argument("--config", default=pre_args.config)
    parser.add_argument("--gguf", default=cfg.get("gguf_source"), help="GGUF shard or directory")
    parser.add_argument("--llama-cpp-root", default=cfg.get("llama_cpp_root"), help="Local llama.cpp root for gguf-py")
    parser.add_argument("--out-bin", default=cfg.get("gguf_embedding_bin", "autoresearch/gguf/embedding_q8_0.bin"))
    parser.add_argument("--out-json", default=cfg.get("gguf_embedding_meta", "autoresearch/gguf/embedding_q8_0.json"))
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

    tensor = None
    tensor_shard = None
    for shard in shards:
        reader = GGUFReader(str(shard), "r")
        for item in reader.tensors:
            if item.name == EMBEDDING_NAME:
                tensor = item
                tensor_shard = shard
                break
        if tensor is not None:
            break

    if tensor is None or tensor_shard is None:
        raise SystemExit(f"Tensor {EMBEDDING_NAME!r} not found in {gguf_path}")

    qtype = tensor.tensor_type
    block_size, type_size = quant_sizes[qtype]
    if qtype.name != "Q8_0":
        raise SystemExit(f"Expected {EMBEDDING_NAME} to be Q8_0, got {qtype.name}")
    if int(block_size) != 32:
        raise SystemExit(f"Expected Q8_0 block size 32, got {block_size}")

    out_bin.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    tensor.data.tofile(out_bin)

    metadata = {
        "name": tensor.name,
        "tensor_type": qtype.name,
        "block_size": int(block_size),
        "type_size": int(type_size),
        "shape": [int(x) for x in tensor.shape],
        "n_bytes": int(tensor.n_bytes),
        "source_gguf": str(gguf_path),
        "source_shard": tensor_shard.name,
        "source_tensor_data_offset": int(tensor.data_offset),
        "output_path": str(out_bin),
    }
    out_json.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote {out_bin} ({tensor.n_bytes} bytes)")
    print(f"Wrote {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
