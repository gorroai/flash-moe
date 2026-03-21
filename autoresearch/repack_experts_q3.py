#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "autoresearch" / "config.json"
NUM_EXPERTS = 512

IQ3_XXS_EXPERT_PROJ_SIZE = 1_605_632
EXPERT_SIZE_Q3_HYBRID = 5_439_488
GATE_W_OFF_Q3 = 0
UP_W_OFF_Q3 = GATE_W_OFF_Q3 + IQ3_XXS_EXPERT_PROJ_SIZE
DOWN_W_OFF_Q3 = UP_W_OFF_Q3 + IQ3_XXS_EXPERT_PROJ_SIZE

IQ4_XS_EXPERT_PROJ_SIZE = 2_228_224
Q5_K_EXPERT_PROJ_SIZE = 2_883_584
EXPERT_SIZE_Q3_OUTLIER = 7_340_032
GATE_W_OFF_Q3_OUTLIER = 0
UP_W_OFF_Q3_OUTLIER = GATE_W_OFF_Q3_OUTLIER + IQ4_XS_EXPERT_PROJ_SIZE
DOWN_W_OFF_Q3_OUTLIER = UP_W_OFF_Q3_OUTLIER + IQ4_XS_EXPERT_PROJ_SIZE

GATE_RE = re.compile(r"^blk\.(\d+)\.ffn_gate_exps\.weight$")
UP_RE = re.compile(r"^blk\.(\d+)\.ffn_up_exps\.weight$")
DOWN_RE = re.compile(r"^blk\.(\d+)\.ffn_down_exps\.weight$")
OUTLIER_LAYER = 27


def load_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def resolve_path(path_text: str | None) -> Path | None:
    if not path_text:
        return None
    path = Path(os.path.expanduser(path_text))
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def import_gguf(llama_cpp_root: Path | None) -> Any:
    if llama_cpp_root:
        gguf_py = llama_cpp_root / "gguf-py"
        if gguf_py.exists():
            sys.path.insert(0, str(gguf_py))
    from gguf import GGUFReader  # type: ignore
    return GGUFReader


def discover_shards(source: Path) -> list[Path]:
    if source.is_dir():
        shards = sorted(source.glob("*.gguf"))
    else:
        shards = sorted(source.parent.glob("*.gguf"))
    if not shards:
        raise SystemExit(f"No GGUF shards found under {source}")
    return shards


def parse_layers(spec: str | None, include_outlier: bool) -> list[int]:
    if not spec or spec == "all":
        layers = list(range(60))
        if not include_outlier:
            layers.remove(OUTLIER_LAYER)
        return layers
    layers: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-", 1)
            layers.extend(range(int(lo), int(hi) + 1))
        else:
            layers.append(int(part))
    layers = sorted(set(layers))
    if OUTLIER_LAYER in layers and not include_outlier:
        raise SystemExit("Layer 27 requires --include-outlier-layer")
    return layers


def collect_target_tensors(shards: list[Path], gguf_reader_cls: Any, layers: list[int]) -> dict[int, dict[str, Any]]:
    wanted = {
        f"blk.{layer}.ffn_gate_exps.weight": (layer, "gate")
        for layer in layers
    }
    wanted.update({
        f"blk.{layer}.ffn_up_exps.weight": (layer, "up")
        for layer in layers
    })
    for layer in layers:
        wanted[f"blk.{layer}.ffn_down_exps.weight"] = (layer, "down")
    if OUTLIER_LAYER in layers:
        wanted[f"blk.{OUTLIER_LAYER}.ffn_down_exps.weight"] = (OUTLIER_LAYER, "down")

    found: dict[int, dict[str, Any]] = {}
    for shard in shards:
        reader = gguf_reader_cls(str(shard), "r")
        for tensor in reader.tensors:
            hit = wanted.get(tensor.name)
            if not hit:
                continue
            layer, role = hit
            found.setdefault(layer, {})[role] = tensor
    return found


def validate_iq3_tensor(layer: int, role: str, tensor: Any) -> None:
    qname = tensor.tensor_type.name
    if qname != "IQ3_XXS":
        raise SystemExit(
            f"Layer {layer} {role} tensor is {qname}, not IQ3_XXS. "
            "The first hybrid repacker only supports non-outlier IQ3_XXS layers."
        )
    expert_size = tensor.n_bytes // NUM_EXPERTS
    if tensor.n_bytes % NUM_EXPERTS != 0 or expert_size != IQ3_XXS_EXPERT_PROJ_SIZE:
        raise SystemExit(
            f"Layer {layer} {role} tensor has unexpected expert size {expert_size} bytes"
        )
    if tuple(int(x) for x in tensor.shape) != (4096, 1024, 512):
        raise SystemExit(
            f"Layer {layer} {role} tensor shape {list(tensor.shape)} does not match expected [4096, 1024, 512]"
        )
    if tensor.data.shape != (NUM_EXPERTS, 1024, 1568):
        raise SystemExit(
            f"Layer {layer} {role} byte shape {tensor.data.shape} does not match expected (512, 1024, 1568)"
        )


def validate_iq4_tensor(layer: int, role: str, tensor: Any) -> None:
    qname = tensor.tensor_type.name
    if qname != "IQ4_XS":
        raise SystemExit(
            f"Layer {layer} {role} tensor is {qname}, not IQ4_XS. "
            "The outlier repacker expects exact GGUF IQ4_XS bytes."
        )
    expert_size = tensor.n_bytes // NUM_EXPERTS
    if tensor.n_bytes % NUM_EXPERTS != 0 or expert_size != IQ4_XS_EXPERT_PROJ_SIZE:
        raise SystemExit(
            f"Layer {layer} {role} tensor has unexpected expert size {expert_size} bytes"
        )
    if tuple(int(x) for x in tensor.shape) != (4096, 1024, 512):
        raise SystemExit(
            f"Layer {layer} {role} tensor shape {list(tensor.shape)} does not match expected [4096, 1024, 512]"
        )
    if tensor.data.shape != (NUM_EXPERTS, 1024, 2176):
        raise SystemExit(
            f"Layer {layer} {role} byte shape {tensor.data.shape} does not match expected (512, 1024, 2176)"
        )


def validate_iq4_down_tensor(layer: int, role: str, tensor: Any) -> None:
    qname = tensor.tensor_type.name
    if qname != "IQ4_XS":
        raise SystemExit(
            f"Layer {layer} {role} tensor is {qname}, not IQ4_XS. "
            "The normal-layer down repacker expects exact GGUF IQ4_XS bytes."
        )
    expert_size = tensor.n_bytes // NUM_EXPERTS
    if tensor.n_bytes % NUM_EXPERTS != 0 or expert_size != IQ4_XS_EXPERT_PROJ_SIZE:
        raise SystemExit(
            f"Layer {layer} {role} tensor has unexpected expert size {expert_size} bytes"
        )
    if tuple(int(x) for x in tensor.shape) != (1024, 4096, 512):
        raise SystemExit(
            f"Layer {layer} {role} tensor shape {list(tensor.shape)} does not match expected [1024, 4096, 512]"
        )
    if tensor.data.shape != (NUM_EXPERTS, 4096, 544):
        raise SystemExit(
            f"Layer {layer} {role} byte shape {tensor.data.shape} does not match expected (512, 4096, 544)"
        )


def validate_q5_tensor(layer: int, role: str, tensor: Any) -> None:
    qname = tensor.tensor_type.name
    if qname != "Q5_K":
        raise SystemExit(
            f"Layer {layer} {role} tensor is {qname}, not Q5_K. "
            "The outlier repacker expects exact GGUF Q5_K bytes."
        )
    expert_size = tensor.n_bytes // NUM_EXPERTS
    if tensor.n_bytes % NUM_EXPERTS != 0 or expert_size != Q5_K_EXPERT_PROJ_SIZE:
        raise SystemExit(
            f"Layer {layer} {role} tensor has unexpected expert size {expert_size} bytes"
        )
    if tuple(int(x) for x in tensor.shape) != (1024, 4096, 512):
        raise SystemExit(
            f"Layer {layer} {role} tensor shape {list(tensor.shape)} does not match expected [1024, 4096, 512]"
        )
    if tensor.data.shape != (NUM_EXPERTS, 4096, 704):
        raise SystemExit(
            f"Layer {layer} {role} byte shape {tensor.data.shape} does not match expected (512, 4096, 704)"
        )


def write_layout_json(output_dir: Path) -> None:
    payload = {
        "format": "q3_streamed_experts",
        "default_format": "q3_gguf_iq3_xxs_iq4_xs",
        "expert_size": EXPERT_SIZE_Q3_HYBRID,
        "num_experts": NUM_EXPERTS,
        "components": [
            {"name": "gate_proj.weight", "quant": "IQ3_XXS", "offset": GATE_W_OFF_Q3, "size": IQ3_XXS_EXPERT_PROJ_SIZE},
            {"name": "up_proj.weight", "quant": "IQ3_XXS", "offset": UP_W_OFF_Q3, "size": IQ3_XXS_EXPERT_PROJ_SIZE},
            {"name": "down_proj.weight", "quant": "IQ4_XS", "offset": DOWN_W_OFF_Q3, "size": IQ4_XS_EXPERT_PROJ_SIZE},
        ],
        "outlier_layers": {
            str(OUTLIER_LAYER): {
                "format": "q3_outlier_iq4_xs_q5_k",
                "expert_size": EXPERT_SIZE_Q3_OUTLIER,
                "components": [
                    {"name": "gate_proj.weight", "quant": "IQ4_XS", "offset": GATE_W_OFF_Q3_OUTLIER, "size": IQ4_XS_EXPERT_PROJ_SIZE},
                    {"name": "up_proj.weight", "quant": "IQ4_XS", "offset": UP_W_OFF_Q3_OUTLIER, "size": IQ4_XS_EXPERT_PROJ_SIZE},
                    {"name": "down_proj.weight", "quant": "Q5_K", "offset": DOWN_W_OFF_Q3_OUTLIER, "size": Q5_K_EXPERT_PROJ_SIZE},
                ],
            }
        },
    }
    (output_dir / "layout.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def repack_layer_iq3(model_path: Path, output_dir: Path, layer: int, gate_tensor: Any, up_tensor: Any, down_tensor: Any, verify: bool) -> None:
    del model_path
    dst_layer_path = output_dir / f"layer_{layer:02d}.bin"
    t0 = time.monotonic()

    fd_out = os.open(dst_layer_path, os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o644)
    try:
        os.ftruncate(fd_out, NUM_EXPERTS * EXPERT_SIZE_Q3_HYBRID)
        for expert in range(NUM_EXPERTS):
            gate_chunk = gate_tensor.data[expert].tobytes()
            up_chunk = up_tensor.data[expert].tobytes()
            down_chunk = down_tensor.data[expert].tobytes()

            if len(gate_chunk) != IQ3_XXS_EXPERT_PROJ_SIZE:
                raise SystemExit(f"Layer {layer} expert {expert} gate size mismatch: {len(gate_chunk)}")
            if len(up_chunk) != IQ3_XXS_EXPERT_PROJ_SIZE:
                raise SystemExit(f"Layer {layer} expert {expert} up size mismatch: {len(up_chunk)}")
            if len(down_chunk) != IQ4_XS_EXPERT_PROJ_SIZE:
                raise SystemExit(f"Layer {layer} expert {expert} down size mismatch: {len(down_chunk)}")

            expert_off = expert * EXPERT_SIZE_Q3_HYBRID
            os.pwrite(fd_out, gate_chunk, expert_off + GATE_W_OFF_Q3)
            os.pwrite(fd_out, up_chunk, expert_off + UP_W_OFF_Q3)
            os.pwrite(fd_out, down_chunk, expert_off + DOWN_W_OFF_Q3)

        if verify:
            for expert in (0, 1, 255, 511):
                expert_off = expert * EXPERT_SIZE_Q3_HYBRID
                gate_chunk = gate_tensor.data[expert].tobytes()
                up_chunk = up_tensor.data[expert].tobytes()
                down_chunk = down_tensor.data[expert].tobytes()
                if os.pread(fd_out, IQ3_XXS_EXPERT_PROJ_SIZE, expert_off + GATE_W_OFF_Q3) != gate_chunk:
                    raise SystemExit(f"Verification failed: layer {layer} expert {expert} gate")
                if os.pread(fd_out, IQ3_XXS_EXPERT_PROJ_SIZE, expert_off + UP_W_OFF_Q3) != up_chunk:
                    raise SystemExit(f"Verification failed: layer {layer} expert {expert} up")
                if os.pread(fd_out, IQ4_XS_EXPERT_PROJ_SIZE, expert_off + DOWN_W_OFF_Q3) != down_chunk:
                    raise SystemExit(f"Verification failed: layer {layer} expert {expert} down")
    finally:
        os.close(fd_out)

    elapsed = time.monotonic() - t0
    gib = (NUM_EXPERTS * EXPERT_SIZE_Q3_HYBRID) / (1024 ** 3)
    print(f"Layer {layer:02d}: wrote {gib:.2f} GiB to {dst_layer_path} in {elapsed:.1f}s")


def repack_layer_outlier(model_path: Path, output_dir: Path, layer: int, gate_tensor: Any, up_tensor: Any, down_tensor: Any, verify: bool) -> None:
    dst_layer_path = output_dir / f"layer_{layer:02d}.bin"
    t0 = time.monotonic()

    fd_out = os.open(dst_layer_path, os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o644)
    try:
        os.ftruncate(fd_out, NUM_EXPERTS * EXPERT_SIZE_Q3_OUTLIER)
        for expert in range(NUM_EXPERTS):
            gate_chunk = gate_tensor.data[expert].tobytes()
            up_chunk = up_tensor.data[expert].tobytes()
            down_chunk = down_tensor.data[expert].tobytes()

            if len(gate_chunk) != IQ4_XS_EXPERT_PROJ_SIZE:
                raise SystemExit(f"Layer {layer} expert {expert} gate size mismatch: {len(gate_chunk)}")
            if len(up_chunk) != IQ4_XS_EXPERT_PROJ_SIZE:
                raise SystemExit(f"Layer {layer} expert {expert} up size mismatch: {len(up_chunk)}")
            if len(down_chunk) != Q5_K_EXPERT_PROJ_SIZE:
                raise SystemExit(f"Layer {layer} expert {expert} down size mismatch: {len(down_chunk)}")

            expert_off = expert * EXPERT_SIZE_Q3_OUTLIER
            os.pwrite(fd_out, gate_chunk, expert_off + GATE_W_OFF_Q3_OUTLIER)
            os.pwrite(fd_out, up_chunk, expert_off + UP_W_OFF_Q3_OUTLIER)
            os.pwrite(fd_out, down_chunk, expert_off + DOWN_W_OFF_Q3_OUTLIER)

        if verify:
            for expert in (0, 1, 255, 511):
                expert_off = expert * EXPERT_SIZE_Q3_OUTLIER
                gate_chunk = gate_tensor.data[expert].tobytes()
                up_chunk = up_tensor.data[expert].tobytes()
                down_chunk = down_tensor.data[expert].tobytes()
                if os.pread(fd_out, IQ4_XS_EXPERT_PROJ_SIZE, expert_off + GATE_W_OFF_Q3_OUTLIER) != gate_chunk:
                    raise SystemExit(f"Verification failed: layer {layer} expert {expert} gate")
                if os.pread(fd_out, IQ4_XS_EXPERT_PROJ_SIZE, expert_off + UP_W_OFF_Q3_OUTLIER) != up_chunk:
                    raise SystemExit(f"Verification failed: layer {layer} expert {expert} up")
                if os.pread(fd_out, Q5_K_EXPERT_PROJ_SIZE, expert_off + DOWN_W_OFF_Q3_OUTLIER) != down_chunk:
                    raise SystemExit(f"Verification failed: layer {layer} expert {expert} down")
    finally:
        os.close(fd_out)

    elapsed = time.monotonic() - t0
    gib = (NUM_EXPERTS * EXPERT_SIZE_Q3_OUTLIER) / (1024 ** 3)
    print(f"Layer {layer:02d}: wrote {gib:.2f} GiB to {dst_layer_path} in {elapsed:.1f}s")


def main() -> int:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=str(DEFAULT_CONFIG))
    pre_args, _ = pre.parse_known_args()
    cfg = load_config(Path(pre_args.config))

    parser = argparse.ArgumentParser(
        description="Create packed_experts_Q3/ expert files with exact GGUF streamed bytes"
    )
    parser.add_argument("--config", default=pre_args.config)
    parser.add_argument("--model", default=cfg.get("model"), help="Base MLX model directory containing packed_experts/")
    parser.add_argument("--gguf", default=cfg.get("gguf_source"), help="GGUF shard or directory")
    parser.add_argument("--llama-cpp-root", default=cfg.get("llama_cpp_root"), help="Local llama.cpp root for gguf-py")
    parser.add_argument("--output", default=cfg.get("packed_experts_q3_dir", "packed_experts_Q3"), help="Output directory name or path")
    parser.add_argument("--layers", default="0", help="Layer spec: 0, 0-4, 0,5,10, 27, or all (default: 0)")
    parser.add_argument("--include-outlier-layer", action="store_true", help="Allow layer 27 and write it with exact IQ4_XS gate/up plus Q5_K down")
    parser.add_argument("--no-verify", action="store_true", help="Skip readback verification")
    args = parser.parse_args()

    model_path = resolve_path(args.model)
    gguf_path = resolve_path(args.gguf)
    llama_cpp_root = resolve_path(args.llama_cpp_root)
    output_arg = Path(os.path.expanduser(args.output))

    if not model_path or not model_path.exists():
        raise SystemExit("Missing or invalid --model path")
    if not gguf_path or not gguf_path.exists():
        raise SystemExit("Missing or invalid --gguf path")
    output_path = output_arg if output_arg.is_absolute() else model_path / output_arg
    output_path.mkdir(parents=True, exist_ok=True)
    write_layout_json(output_path)

    layers = parse_layers(args.layers, include_outlier=args.include_outlier_layer)

    GGUFReader = import_gguf(llama_cpp_root)
    shards = discover_shards(gguf_path)
    tensors = collect_target_tensors(shards, GGUFReader, layers)

    for layer in layers:
        pair = tensors.get(layer, {})
        gate_tensor = pair.get("gate")
        up_tensor = pair.get("up")
        if gate_tensor is None or up_tensor is None:
            raise SystemExit(f"Missing GGUF gate/up tensors for layer {layer}")
        if layer == OUTLIER_LAYER:
            validate_iq4_tensor(layer, "gate", gate_tensor)
            validate_iq4_tensor(layer, "up", up_tensor)
            down_tensor = pair.get("down")
            if down_tensor is None:
                raise SystemExit("Missing GGUF down tensor for layer 27 outlier")
            validate_q5_tensor(layer, "down", down_tensor)
            repack_layer_outlier(model_path, output_path, layer, gate_tensor, up_tensor, down_tensor, verify=not args.no_verify)
        else:
            validate_iq3_tensor(layer, "gate", gate_tensor)
            validate_iq3_tensor(layer, "up", up_tensor)
            down_tensor = pair.get("down")
            if down_tensor is None:
                raise SystemExit(f"Missing GGUF down tensor for layer {layer}")
            validate_iq4_down_tensor(layer, "down", down_tensor)
            repack_layer_iq3(model_path, output_path, layer, gate_tensor, up_tensor, down_tensor, verify=not args.no_verify)

    print(f"Wrote layout manifest: {output_path / 'layout.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
