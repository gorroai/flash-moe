#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fnmatch
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "autoresearch" / "config.json"


def load_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def resolve_path(path_text: str | None) -> Path | None:
    if not path_text:
        return None
    return Path(path_text).expanduser()


def import_gguf(llama_cpp_root: Path | None) -> tuple[Any, Any]:
    if llama_cpp_root:
        gguf_py = llama_cpp_root / "gguf-py"
        if gguf_py.exists():
            sys.path.insert(0, str(gguf_py))
    try:
        from gguf import GGUFReader  # type: ignore
        from gguf.constants import GGML_QUANT_SIZES  # type: ignore
    except Exception as exc:  # pragma: no cover - surfaced to user
        raise SystemExit(
            "Failed to import gguf tooling. Set llama_cpp_root in autoresearch/config.json "
            "or install the Python gguf package."
        ) from exc
    return GGUFReader, GGML_QUANT_SIZES


def discover_shards(source: Path) -> list[Path]:
    if source.is_dir():
        shards = sorted(source.glob("*.gguf"))
    elif source.suffix == ".gguf":
        shards = sorted(source.parent.glob("*.gguf"))
    else:
        raise SystemExit(f"GGUF source must be a .gguf file or a directory: {source}")
    if not shards:
        raise SystemExit(f"No GGUF shards found under {source}")
    return shards


def tensor_template(name: str) -> str:
    return re.sub(r"^blk\.\d+\.", "blk.*.", name)


def classify_residency(name: str) -> str:
    if name == "output.weight":
        return "lm_head"
    if name == "token_embd.weight":
        return "embedding"
    if ".ffn_" in name and "_exps.weight" in name:
        return "streamed_expert"
    if ".ffn_" in name and "_shexp.weight" in name:
        return "shared_expert"
    return "resident_dense"


def classify_family(name: str) -> str:
    template = tensor_template(name)
    if template == "output.weight":
        return "lm_head"
    if template == "token_embd.weight":
        return "embedding"
    if ".attn_" in template:
        return "attention"
    if ".ssm_" in template:
        return "ssm"
    if ".ffn_" in template and "_exps.weight" in template:
        return "routed_expert"
    if ".ffn_" in template and "_shexp.weight" in template:
        return "shared_expert"
    if "ffn_gate_inp" in template:
        return "router"
    if "norm.weight" in template or template.endswith(".bias"):
        return "norm_or_bias"
    return "other"


def priority_bucket(name: str) -> str:
    residency = classify_residency(name)
    if residency == "lm_head":
        return "first"
    if residency in {"embedding", "resident_dense", "shared_expert"}:
        return "early"
    return "late"


def gib(byte_count: int) -> float:
    return byte_count / (1024 ** 3)


def build_inventory(
    source: Path,
    shards: list[Path],
    gguf_reader_cls: Any,
    quant_sizes: Any,
    expected_blocks: dict[str, int],
) -> dict[str, Any]:
    tensors: list[dict[str, Any]] = []
    quant_counts: Counter[str] = Counter()
    quant_bytes: Counter[str] = Counter()
    quant_meta: dict[str, tuple[int, int]] = {}
    template_counts: Counter[str] = Counter()
    template_bytes: Counter[str] = Counter()
    template_quants: dict[str, Counter[str]] = defaultdict(Counter)
    outliers: list[dict[str, Any]] = []
    block_mismatches: list[dict[str, Any]] = []

    for shard in shards:
        reader = gguf_reader_cls(str(shard), "r")
        for tensor in reader.tensors:
            qtype = tensor.tensor_type
            qname = qtype.name
            block_size, type_size = quant_sizes[qtype]
            shape = [int(x) for x in tensor.shape]
            name = tensor.name
            template = tensor_template(name)
            n_bytes = int(tensor.n_bytes)
            item = {
                "name": name,
                "template": template,
                "family": classify_family(name),
                "residency": classify_residency(name),
                "priority": priority_bucket(name),
                "shard": shard.name,
                "tensor_type": qname,
                "block_size": int(block_size),
                "type_size": int(type_size),
                "shape": shape,
                "n_bytes": n_bytes,
                "n_gib": gib(n_bytes),
                "data_offset": int(tensor.data_offset),
                "field_offset": int(tensor.field.offset),
                "n_elements": math.prod(shape),
            }
            tensors.append(item)

            quant_counts[qname] += 1
            quant_bytes[qname] += n_bytes
            quant_meta.setdefault(qname, (int(block_size), int(type_size)))
            template_counts[template] += 1
            template_bytes[template] += n_bytes
            template_quants[template][qname] += 1

            expected_block = expected_blocks.get(qname)
            if expected_block is not None and expected_block != int(block_size):
                block_mismatches.append(
                    {
                        "name": name,
                        "tensor_type": qname,
                        "expected_block_size": expected_block,
                        "actual_block_size": int(block_size),
                    }
                )

            if qname in {"Q6_K", "Q5_K", "BF16"}:
                outliers.append(item)

    template_rows = []
    for template, count in template_counts.items():
        sample = next(t for t in tensors if t["template"] == template)
        template_rows.append(
            {
                "template": template,
                "family": sample["family"],
                "residency": sample["residency"],
                "priority": sample["priority"],
                "count": count,
                "n_bytes": template_bytes[template],
                "n_gib": gib(template_bytes[template]),
                "quant_types": dict(sorted(template_quants[template].items())),
                "example_tensor": sample["name"],
            }
        )

    quant_rows = []
    for qname, count in quant_counts.most_common():
        block_size, type_size = quant_meta[qname]
        quant_rows.append(
            {
                "tensor_type": qname,
                "count": count,
                "n_bytes": quant_bytes[qname],
                "n_gib": gib(quant_bytes[qname]),
                "block_size": int(block_size),
                "type_size": int(type_size),
            }
        )

    inventory = {
        "source": str(source),
        "shards": [shard.name for shard in shards],
        "shard_count": len(shards),
        "tensor_count": len(tensors),
        "quant_types": quant_rows,
        "template_summary": sorted(template_rows, key=lambda row: (row["priority"], row["template"])),
        "outliers": sorted(outliers, key=lambda item: item["name"]),
        "block_mismatches": block_mismatches,
        "tensors": sorted(tensors, key=lambda item: item["name"]),
    }
    return inventory


def render_quant_table(rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| Quant | Block | Bytes/Block | Tensors | Total GiB |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in sorted(rows, key=lambda item: item["n_bytes"], reverse=True):
        lines.append(
            f"| {row['tensor_type']} | {row['block_size']} | {row['type_size']} | "
            f"{row['count']} | {row['n_gib']:.3f} |"
        )
    return lines


def render_template_table(rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| Template | Residency | Quant | Count | Total GiB |",
        "|---|---|---|---:|---:|",
    ]
    for row in rows:
        quant_desc = ", ".join(f"{name} x{count}" for name, count in row["quant_types"].items())
        lines.append(
            f"| `{row['template']}` | {row['residency']} | {quant_desc} | "
            f"{row['count']} | {row['n_gib']:.3f} |"
        )
    return lines


def render_match_table(rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| Tensor | Shard | Quant | Block | Shape | Bytes |",
        "|---|---|---|---:|---|---:|",
    ]
    for row in rows:
        shape = "x".join(str(dim) for dim in row["shape"])
        lines.append(
            f"| `{row['name']}` | `{row['shard']}` | {row['tensor_type']} | {row['block_size']} | "
            f"`{shape}` | {row['n_bytes']} |"
        )
    return lines


def select_matches(tensors: list[dict[str, Any]], patterns: list[str]) -> list[dict[str, Any]]:
    return [
        tensor
        for tensor in tensors
        if any(fnmatch.fnmatch(tensor["name"], pattern) for pattern in patterns)
    ]


def render_markdown(inventory: dict[str, Any], match_rows: list[dict[str, Any]] | None = None) -> str:
    lines = ["# GGUF Q3 Tensor Sweep", ""]
    lines.append("Metadata-only sweep. This does not instantiate the full GGUF model.")
    lines.append("")
    lines.append(f"- Shards scanned: {inventory['shard_count']}")
    lines.append(f"- Tensors scanned: {inventory['tensor_count']}")
    lines.append(f"- Source: `{inventory['source']}`")
    lines.append("")
    lines.append("## Quant Types")
    lines.extend(render_quant_table(inventory["quant_types"]))
    lines.append("")

    if inventory["block_mismatches"]:
        lines.append("## Block Size Mismatches")
        for mismatch in inventory["block_mismatches"]:
            lines.append(
                f"- `{mismatch['name']}` expected block {mismatch['expected_block_size']} "
                f"but GGUF reports {mismatch['actual_block_size']}"
            )
        lines.append("")

    if inventory["outliers"]:
        lines.append("## Outliers")
        lines.extend(render_match_table(inventory["outliers"]))
        lines.append("")

    lines.append("## Key Templates")
    key_rows = [
        row
        for row in inventory["template_summary"]
        if row["template"] in {
            "output.weight",
            "token_embd.weight",
            "blk.*.attn_gate.weight",
            "blk.*.attn_qkv.weight",
            "blk.*.attn_q.weight",
            "blk.*.attn_k.weight",
            "blk.*.attn_v.weight",
            "blk.*.attn_output.weight",
            "blk.*.ssm_out.weight",
            "blk.*.ffn_down_shexp.weight",
            "blk.*.ffn_gate_shexp.weight",
            "blk.*.ffn_up_shexp.weight",
            "blk.*.ffn_down_exps.weight",
            "blk.*.ffn_gate_exps.weight",
            "blk.*.ffn_up_exps.weight",
        }
    ]
    lines.extend(render_template_table(key_rows))
    lines.append("")
    lines.append("## Suggested Start Order")
    lines.append("- `output.weight` first: isolated LM head path and exact `Q6_K` kernel work.")
    lines.append("- `Q8_0` resident dense tensors next: attention, SSM, and shared expert tensors.")
    lines.append("- Keep `packed_experts_Q3/` separate for routed-expert experiments, matching the 2-bit workflow.")
    lines.append("- Short PPL plus a short generation smoke test on every iteration.")
    lines.append("")

    if match_rows is not None:
        lines.append("## Requested Matches")
        lines.extend(render_match_table(match_rows))
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Metadata-only GGUF tensor sweep for hybrid Flash-MoE experiments.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to autoresearch config JSON.")
    parser.add_argument("--gguf", help="GGUF shard or directory to inspect. Defaults to config gguf_source.")
    parser.add_argument("--llama-cpp-root", help="Local llama.cpp checkout for gguf-py imports.")
    parser.add_argument("--tensor", action="append", default=[], help="Shell-style tensor match pattern. Repeat as needed.")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of Markdown.")
    parser.add_argument("--write-json", help="Write full inventory JSON to this path.")
    parser.add_argument("--write-markdown", help="Write Markdown summary to this path.")
    args = parser.parse_args()

    config = load_config(Path(args.config))
    gguf_source = resolve_path(args.gguf or config.get("gguf_source"))
    if gguf_source is None:
        raise SystemExit("No GGUF source configured.")

    llama_cpp_root = resolve_path(args.llama_cpp_root or config.get("llama_cpp_root"))
    gguf_reader_cls, quant_sizes = import_gguf(llama_cpp_root)
    shards = discover_shards(gguf_source)
    expected_blocks = {
        str(name): int(block_size)
        for name, block_size in config.get("gguf_quant_types", {}).items()
    }
    inventory = build_inventory(gguf_source, shards, gguf_reader_cls, quant_sizes, expected_blocks)

    match_rows = None
    if args.tensor:
        match_rows = select_matches(inventory["tensors"], args.tensor)
        if not match_rows:
            raise SystemExit(f"No tensors matched: {args.tensor}")

    if args.write_json:
        out = Path(args.write_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(inventory, indent=2, sort_keys=True) + "\n")

    markdown = render_markdown(inventory, match_rows=match_rows)
    if args.write_markdown:
        out = Path(args.write_markdown)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(markdown)

    if args.json:
        payload: dict[str, Any] = inventory if match_rows is None else {"matches": match_rows}
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(markdown, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
