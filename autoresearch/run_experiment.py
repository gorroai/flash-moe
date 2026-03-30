#!/usr/bin/env python3
"""Run a fixed Flash-MoE benchmark suite for autoresearch."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any


DEFAULT_PROMPT = "Explain quantum computing in one concise paragraph."
DEFAULT_GEN_TOKENS = 64
DEFAULT_PPL_MAX = 5.75


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def expand_path(value: str | None, base: Path) -> str | None:
    if not value:
        return None
    path = Path(os.path.expanduser(value))
    if not path.is_absolute():
        path = base / path
    return str(path)


def load_config(path: str) -> dict[str, Any]:
    cfg_path = Path(os.path.expanduser(path))
    if not cfg_path.is_absolute():
        cfg_path = repo_root() / cfg_path
    if not cfg_path.exists():
        return {}
    with cfg_path.open() as f:
        return json.load(f)


def first_existing_dir(candidates: list[str | None]) -> str | None:
    for candidate in candidates:
        if not candidate:
            continue
        p = Path(os.path.expanduser(candidate))
        if p.is_dir():
            return str(p)
    return None


def first_existing_file(candidates: list[str | None]) -> str | None:
    for candidate in candidates:
        if not candidate:
            continue
        p = Path(os.path.expanduser(candidate))
        if p.is_file():
            return str(p)
    return None


def detect_model_path(config: dict[str, Any]) -> str | None:
    return first_existing_dir(
        [
            os.environ.get("FLASH_MOE_MODEL"),
            expand_path(config.get("model"), repo_root()),
            os.path.expanduser("~/Models/flash_mlx_4bit"),
            os.path.expanduser("~/Models/mlx-community-Qwen3.5-397B-A17B-4bit"),
        ]
    )


def detect_ppl_tokens(config: dict[str, Any], key: str, env_key: str, fallback_name: str) -> str | None:
    return first_existing_file(
        [
            os.environ.get(env_key),
            expand_path(config.get(key), repo_root()),
            str(repo_root() / fallback_name),
        ]
    )


def detect_optional_file(config: dict[str, Any], key: str, env_key: str) -> str | None:
    return first_existing_file(
        [
            os.environ.get(env_key),
            expand_path(config.get(key), repo_root()),
        ]
    )


def maybe_load_json(path: str | None) -> dict[str, Any] | None:
    if not path:
        return None
    p = Path(os.path.expanduser(path))
    if not p.is_file():
        return None
    with p.open() as f:
        return json.load(f)


def commit_hash(cwd: Path) -> str:
    proc = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=cwd,
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        return "UNKNOWN"
    return proc.stdout.strip() or "UNKNOWN"


def run_command(
    cmd: list[str],
    cwd: Path,
    timeout_s: int,
    log_path: Path,
) -> dict[str, Any]:
    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd,
            cwd=cwd,
            text=True,
            capture_output=True,
            timeout=timeout_s,
            check=False,
        )
        timed_out = False
        stdout = proc.stdout
        stderr = proc.stderr
        returncode = proc.returncode
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        returncode = 124

    elapsed_s = time.time() - t0
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        f"$ {' '.join(cmd)}\n\n=== STDOUT ===\n{stdout}\n\n=== STDERR ===\n{stderr}\n",
        encoding="utf-8",
    )
    return {
        "cmd": cmd,
        "returncode": returncode,
        "timed_out": timed_out,
        "elapsed_s": elapsed_s,
        "stdout": stdout,
        "stderr": stderr,
        "log_path": str(log_path),
    }


def parse_generation(text: str) -> dict[str, float]:
    m = re.search(r"decode:\s*([0-9.]+)\s*t/s,\s*prefill:\s*([0-9.]+)\s*t/s", text)
    if not m:
        raise ValueError("could not parse decode/prefill throughput")
    return {
        "decode_tok_s": float(m.group(1)),
        "prefill_tok_s": float(m.group(2)),
    }


def parse_ppl(text: str) -> dict[str, float]:
    ppl = re.search(r"Perplexity:\s*([0-9.]+)", text)
    tps = re.search(r"Time:\s*[0-9.]+\s*s\s+\(([0-9.]+)\s+tok/s\)", text)
    ce = re.search(r"Cross-entropy:\s*([0-9.]+)", text)
    if not ppl or not tps:
        raise ValueError("could not parse perplexity output")
    out = {
        "ppl": float(ppl.group(1)),
        "ppl_tok_s": float(tps.group(1)),
    }
    if ce:
        out["cross_entropy"] = float(ce.group(1))
    return out


def compute_vs_baseline(current: dict[str, Any], baseline: dict[str, Any] | None) -> dict[str, float] | None:
    if not baseline:
        return None

    out: dict[str, float] = {}
    cur_gen = current.get("generation") or {}
    base_gen = baseline.get("generation") or {}
    cur_ppl = current.get("perplexity") or {}
    base_ppl = baseline.get("perplexity") or {}

    for key in ("decode_tok_s", "prefill_tok_s"):
        cur_v = cur_gen.get(key)
        base_v = base_gen.get(key)
        if isinstance(cur_v, (int, float)) and isinstance(base_v, (int, float)):
            out[f"{key}_abs"] = cur_v - base_v
            out[f"{key}_pct"] = ((cur_v / base_v) - 1.0) * 100.0 if base_v else 0.0

    if isinstance(cur_ppl.get("ppl"), (int, float)) and isinstance(base_ppl.get("ppl"), (int, float)):
        cur_v = float(cur_ppl["ppl"])
        base_v = float(base_ppl["ppl"])
        out["ppl_abs"] = cur_v - base_v
        out["ppl_pct"] = ((cur_v / base_v) - 1.0) * 100.0 if base_v else 0.0

    cur_full = current.get("full_perplexity") or {}
    base_full = baseline.get("full_perplexity") or {}
    if isinstance(cur_full.get("ppl"), (int, float)) and isinstance(base_full.get("ppl"), (int, float)):
        cur_v = float(cur_full["ppl"])
        base_v = float(base_full["ppl"])
        out["full_ppl_abs"] = cur_v - base_v
        out["full_ppl_pct"] = ((cur_v / base_v) - 1.0) * 100.0 if base_v else 0.0

    return out


def main() -> int:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default="autoresearch/config.json", help="Config JSON path")
    pre_args, _ = pre.parse_known_args()
    config = load_config(pre_args.config)
    root = repo_root()

    parser = argparse.ArgumentParser(description="Run a fixed Flash-MoE autoresearch benchmark")
    parser.add_argument("--config", default=pre_args.config, help="Config JSON path")
    parser.add_argument("--model", default=detect_model_path(config), help="Flash-MoE model directory")
    parser.add_argument(
        "--ppl-tokens",
        default=detect_ppl_tokens(config, "short_ppl_tokens", "FLASH_MOE_PPL_TOKENS", "ppl_tokens.bin"),
        help="Short ppl token file path",
    )
    parser.add_argument(
        "--full-ppl-tokens",
        default=detect_ppl_tokens(config, "full_ppl_tokens", "FLASH_MOE_FULL_PPL_TOKENS", "ppl_tokens_2k.bin"),
        help="Full/periodic ppl token file path",
    )
    parser.add_argument(
        "--baseline-result",
        default=expand_path(config.get("baseline_result"), root),
        help="Baseline JSON result path",
    )
    parser.add_argument(
        "--gguf",
        default=expand_path(config.get("gguf_source"), root),
        help="Reference GGUF path for hybrid experiments",
    )
    parser.add_argument(
        "--gguf-lm-head",
        default=detect_optional_file(config, "gguf_lm_head_bin", "FLASH_MOE_GGUF_LM_HEAD"),
        help="Optional raw Q6_K GGUF LM head blob",
    )
    parser.add_argument(
        "--gguf-embedding",
        default=detect_optional_file(config, "gguf_embedding_bin", "FLASH_MOE_GGUF_EMBEDDING"),
        help="Optional raw Q8_0 GGUF embedding blob",
    )
    parser.add_argument(
        "--gguf-full-attn-bin",
        default=detect_optional_file(config, "gguf_full_attn_bin", "FLASH_MOE_GGUF_FULL_ATTN_BIN"),
        help="Optional raw Q8_0 GGUF full-attention overlay blob",
    )
    parser.add_argument(
        "--gguf-full-attn-json",
        default=detect_optional_file(config, "gguf_full_attn_json", "FLASH_MOE_GGUF_FULL_ATTN_JSON"),
        help="Optional full-attention overlay manifest JSON",
    )
    parser.add_argument(
        "--gguf-qkv-bin",
        default=detect_optional_file(config, "gguf_qkv_overlay_bin", "FLASH_MOE_GGUF_QKV_BIN"),
        help="Optional raw Q8_0 GGUF linear-attention qkv overlay blob",
    )
    parser.add_argument(
        "--gguf-qkv-json",
        default=detect_optional_file(config, "gguf_qkv_overlay_json", "FLASH_MOE_GGUF_QKV_JSON"),
        help="Optional qkv overlay manifest JSON",
    )
    parser.add_argument(
        "--gguf-linear-bin",
        default=detect_optional_file(config, "gguf_linear_bin", "FLASH_MOE_GGUF_LINEAR_BIN"),
        help="Optional raw Q8_0 GGUF linear gate/out overlay blob",
    )
    parser.add_argument(
        "--gguf-linear-json",
        default=detect_optional_file(config, "gguf_linear_json", "FLASH_MOE_GGUF_LINEAR_JSON"),
        help="Optional linear gate/out overlay manifest JSON",
    )
    parser.add_argument("--smoke-prompt", default=config.get("smoke_prompt", "What is Apple Neural Engine?"), help="Short smoke-test prompt")
    parser.add_argument("--smoke-tokens", type=int, default=int(config.get("smoke_tokens", 24)), help="Short smoke-test token count")
    parser.add_argument("--prompt", default=config.get("prompt", DEFAULT_PROMPT), help="Generation benchmark prompt")
    parser.add_argument("--gen-tokens", type=int, default=int(config.get("gen_tokens", DEFAULT_GEN_TOKENS)), help="Generation token count")
    parser.add_argument("--2bit", dest="two_bit", action="store_true", help="Run the 2-bit expert path")
    parser.add_argument("--q3-experts", dest="q3_experts", action="store_true", help="Run the Q3 GGUF expert path")
    parser.add_argument("--qjl-experts", dest="qjl_experts", action="store_true", help="Run the 1-bit QJL SRHT expert path")
    parser.add_argument("--cache-io-split", dest="cache_io_split", type=int, default=0, help="Fan out expert preads (0=default)")
    parser.add_argument("--ppl-max", type=float, default=float(config.get("short_ppl_max", DEFAULT_PPL_MAX)), help="Maximum allowed short perplexity")
    parser.add_argument("--full-ppl-max", type=float, default=float(config.get("full_ppl_max", DEFAULT_PPL_MAX)), help="Maximum allowed periodic full perplexity")
    parser.add_argument("--logs-dir", default="autoresearch/logs", help="Directory for build/gen/ppl logs")
    parser.add_argument("--build-timeout", type=int, default=300, help="Build timeout in seconds")
    parser.add_argument("--gen-timeout", type=int, default=300, help="Generation timeout in seconds")
    parser.add_argument("--ppl-timeout", type=int, default=600, help="PPL timeout in seconds")
    parser.add_argument("--skip-build", action="store_true", help="Skip the build step")
    parser.add_argument("--skip-smoke", action="store_true", help="Skip the short smoke generation")
    parser.add_argument("--skip-ppl", action="store_true", help="Skip perplexity benchmark")
    parser.add_argument("--full-check", action="store_true", help="Also run periodic/full perplexity benchmark")
    parser.add_argument("--save-baseline", action="store_true", help="Write the successful result JSON to --baseline-result")
    parser.add_argument("--json", action="store_true", help="Emit only JSON")
    args = parser.parse_args()

    logs_dir = root / args.logs_dir

    result: dict[str, Any] = {
        "ok": False,
        "commit": commit_hash(root),
        "mode": config.get("mode", "default"),
        "model": args.model,
        "gguf": args.gguf,
        "two_bit": args.two_bit,
        "q3_experts": args.q3_experts,
        "qjl_experts": args.qjl_experts,
        "cache_io_split": args.cache_io_split,
        "gguf_embedding": args.gguf_embedding,
        "gguf_full_attn_bin": args.gguf_full_attn_bin,
        "gguf_full_attn_json": args.gguf_full_attn_json,
        "gguf_lm_head": args.gguf_lm_head,
        "gguf_qkv_bin": args.gguf_qkv_bin,
        "gguf_qkv_json": args.gguf_qkv_json,
        "gguf_linear_bin": args.gguf_linear_bin,
        "gguf_linear_json": args.gguf_linear_json,
        "smoke_prompt": args.smoke_prompt,
        "smoke_tokens": args.smoke_tokens,
        "ppl_tokens": args.ppl_tokens,
        "full_ppl_tokens": args.full_ppl_tokens,
        "baseline_result": args.baseline_result,
        "prompt": args.prompt,
        "gen_tokens": args.gen_tokens,
        "ppl_max": args.ppl_max,
        "full_ppl_max": args.full_ppl_max,
        "build": None,
        "smoke_generation": None,
        "generation": None,
        "perplexity": None,
        "full_perplexity": None,
        "baseline": None,
        "vs_baseline": None,
        "quality_pass": None,
        "score": 0.0,
        "failure_stage": None,
    }

    if not args.model:
        result["failure_stage"] = "setup"
        result["error"] = "No model directory found. Set FLASH_MOE_MODEL or pass --model."
        print(json.dumps(result, indent=None if args.json else 2))
        return 1

    if not args.skip_ppl and not args.ppl_tokens:
        result["failure_stage"] = "setup"
        result["error"] = "No ppl token file found. Set FLASH_MOE_PPL_TOKENS or pass --ppl-tokens."
        print(json.dumps(result, indent=None if args.json else 2))
        return 1

    if args.full_check and not args.full_ppl_tokens:
        result["failure_stage"] = "setup"
        result["error"] = "No full ppl token file found. Set FLASH_MOE_FULL_PPL_TOKENS or pass --full-ppl-tokens."
        print(json.dumps(result, indent=None if args.json else 2))
        return 1

    if not args.skip_build:
        build = run_command(
            ["make", "-C", "metal_infer", "infer"],
            cwd=root,
            timeout_s=args.build_timeout,
            log_path=logs_dir / "build.log",
        )
        result["build"] = {
            "returncode": build["returncode"],
            "timed_out": build["timed_out"],
            "elapsed_s": round(build["elapsed_s"], 3),
            "log_path": build["log_path"],
        }
        if build["returncode"] != 0:
            result["failure_stage"] = "build"
            result["error"] = "build failed"
            print(json.dumps(result, indent=None if args.json else 2))
            return 1

    if not args.skip_smoke:
        smoke_cmd = [
            "./metal_infer/infer",
            "--model",
            args.model,
            "--prompt",
            args.smoke_prompt,
            "--tokens",
            str(args.smoke_tokens),
            "--stream",
        ]
        if args.two_bit:
            smoke_cmd.append("--2bit")
        if args.q3_experts:
            smoke_cmd.append("--q3-experts")
        if args.qjl_experts:
            smoke_cmd.append("--qjl-experts")
        if args.cache_io_split:
            smoke_cmd.extend(["--cache-io-split", str(args.cache_io_split)])
        if args.gguf_embedding:
            smoke_cmd.extend(["--gguf-embedding", args.gguf_embedding])
        if args.gguf_full_attn_bin:
            smoke_cmd.extend(["--gguf-full-attn-bin", args.gguf_full_attn_bin])
        if args.gguf_full_attn_json:
            smoke_cmd.extend(["--gguf-full-attn-json", args.gguf_full_attn_json])
        if args.gguf_lm_head:
            smoke_cmd.extend(["--gguf-lm-head", args.gguf_lm_head])
        if args.gguf_qkv_bin:
            smoke_cmd.extend(["--gguf-qkv-bin", args.gguf_qkv_bin])
        if args.gguf_qkv_json:
            smoke_cmd.extend(["--gguf-qkv-json", args.gguf_qkv_json])
        if args.gguf_linear_bin:
            smoke_cmd.extend(["--gguf-linear-bin", args.gguf_linear_bin])
        if args.gguf_linear_json:
            smoke_cmd.extend(["--gguf-linear-json", args.gguf_linear_json])

        smoke = run_command(
            smoke_cmd,
            cwd=root,
            timeout_s=args.gen_timeout,
            log_path=logs_dir / "smoke.log",
        )
        result["smoke_generation"] = {
            "returncode": smoke["returncode"],
            "timed_out": smoke["timed_out"],
            "elapsed_s": round(smoke["elapsed_s"], 3),
            "log_path": smoke["log_path"],
        }
        if smoke["returncode"] != 0:
            result["failure_stage"] = "smoke_generation"
            result["error"] = "smoke generation failed"
            print(json.dumps(result, indent=None if args.json else 2))
            return 1

    gen_cmd = [
        "./metal_infer/infer",
        "--model",
        args.model,
        "--prompt",
        args.prompt,
        "--tokens",
        str(args.gen_tokens),
        "--stream",
    ]
    if args.two_bit:
        gen_cmd.append("--2bit")
    if args.q3_experts:
        gen_cmd.append("--q3-experts")
    if args.qjl_experts:
        gen_cmd.append("--qjl-experts")
    if args.cache_io_split:
        gen_cmd.extend(["--cache-io-split", str(args.cache_io_split)])
    if args.gguf_embedding:
        gen_cmd.extend(["--gguf-embedding", args.gguf_embedding])
    if args.gguf_full_attn_bin:
        gen_cmd.extend(["--gguf-full-attn-bin", args.gguf_full_attn_bin])
    if args.gguf_full_attn_json:
        gen_cmd.extend(["--gguf-full-attn-json", args.gguf_full_attn_json])
    if args.gguf_lm_head:
        gen_cmd.extend(["--gguf-lm-head", args.gguf_lm_head])
    if args.gguf_qkv_bin:
        gen_cmd.extend(["--gguf-qkv-bin", args.gguf_qkv_bin])
    if args.gguf_qkv_json:
        gen_cmd.extend(["--gguf-qkv-json", args.gguf_qkv_json])
    if args.gguf_linear_bin:
        gen_cmd.extend(["--gguf-linear-bin", args.gguf_linear_bin])
    if args.gguf_linear_json:
        gen_cmd.extend(["--gguf-linear-json", args.gguf_linear_json])

    gen = run_command(
        gen_cmd,
        cwd=root,
        timeout_s=args.gen_timeout,
        log_path=logs_dir / "generation.log",
    )
    if gen["returncode"] != 0:
        result["failure_stage"] = "generation"
        result["error"] = "generation benchmark failed"
        result["generation"] = {
            "returncode": gen["returncode"],
            "timed_out": gen["timed_out"],
            "elapsed_s": round(gen["elapsed_s"], 3),
            "log_path": gen["log_path"],
        }
        print(json.dumps(result, indent=None if args.json else 2))
        return 1

    try:
        gen_metrics = parse_generation(gen["stdout"] + "\n" + gen["stderr"])
    except ValueError as exc:
        result["failure_stage"] = "generation"
        result["error"] = str(exc)
        result["generation"] = {
            "returncode": gen["returncode"],
            "timed_out": gen["timed_out"],
            "elapsed_s": round(gen["elapsed_s"], 3),
            "log_path": gen["log_path"],
        }
        print(json.dumps(result, indent=None if args.json else 2))
        return 1

    result["generation"] = {
        **gen_metrics,
        "returncode": gen["returncode"],
        "timed_out": gen["timed_out"],
        "elapsed_s": round(gen["elapsed_s"], 3),
        "log_path": gen["log_path"],
    }

    if args.skip_ppl:
        result["quality_pass"] = True
        result["score"] = gen_metrics["decode_tok_s"]
        result["ok"] = True
        print(json.dumps(result, indent=None if args.json else 2))
        return 0

    ppl_cmd = [
        "./metal_infer/infer",
        "--model",
        args.model,
        "--ppl",
        args.ppl_tokens,
    ]
    if args.two_bit:
        ppl_cmd.append("--2bit")
    if args.q3_experts:
        ppl_cmd.append("--q3-experts")
    if args.qjl_experts:
        ppl_cmd.append("--qjl-experts")
    if args.cache_io_split:
        ppl_cmd.extend(["--cache-io-split", str(args.cache_io_split)])
    if args.gguf_embedding:
        ppl_cmd.extend(["--gguf-embedding", args.gguf_embedding])
    if args.gguf_full_attn_bin:
        ppl_cmd.extend(["--gguf-full-attn-bin", args.gguf_full_attn_bin])
    if args.gguf_full_attn_json:
        ppl_cmd.extend(["--gguf-full-attn-json", args.gguf_full_attn_json])
    if args.gguf_lm_head:
        ppl_cmd.extend(["--gguf-lm-head", args.gguf_lm_head])
    if args.gguf_qkv_bin:
        ppl_cmd.extend(["--gguf-qkv-bin", args.gguf_qkv_bin])
    if args.gguf_qkv_json:
        ppl_cmd.extend(["--gguf-qkv-json", args.gguf_qkv_json])
    if args.gguf_linear_bin:
        ppl_cmd.extend(["--gguf-linear-bin", args.gguf_linear_bin])
    if args.gguf_linear_json:
        ppl_cmd.extend(["--gguf-linear-json", args.gguf_linear_json])

    ppl = run_command(
        ppl_cmd,
        cwd=root,
        timeout_s=args.ppl_timeout,
        log_path=logs_dir / "ppl.log",
    )
    if ppl["returncode"] != 0:
        result["failure_stage"] = "perplexity"
        result["error"] = "perplexity benchmark failed"
        result["perplexity"] = {
            "returncode": ppl["returncode"],
            "timed_out": ppl["timed_out"],
            "elapsed_s": round(ppl["elapsed_s"], 3),
            "log_path": ppl["log_path"],
        }
        print(json.dumps(result, indent=None if args.json else 2))
        return 1

    try:
        ppl_metrics = parse_ppl(ppl["stdout"] + "\n" + ppl["stderr"])
    except ValueError as exc:
        result["failure_stage"] = "perplexity"
        result["error"] = str(exc)
        result["perplexity"] = {
            "returncode": ppl["returncode"],
            "timed_out": ppl["timed_out"],
            "elapsed_s": round(ppl["elapsed_s"], 3),
            "log_path": ppl["log_path"],
        }
        print(json.dumps(result, indent=None if args.json else 2))
        return 1

    result["perplexity"] = {
        **ppl_metrics,
        "returncode": ppl["returncode"],
        "timed_out": ppl["timed_out"],
        "elapsed_s": round(ppl["elapsed_s"], 3),
        "log_path": ppl["log_path"],
    }

    if args.full_check:
        full_ppl_cmd = [
            "./metal_infer/infer",
            "--model",
            args.model,
            "--ppl",
            args.full_ppl_tokens,
        ]
        if args.two_bit:
            full_ppl_cmd.append("--2bit")
        if args.q3_experts:
            full_ppl_cmd.append("--q3-experts")
        if args.qjl_experts:
            full_ppl_cmd.append("--qjl-experts")
        if args.cache_io_split:
            full_ppl_cmd.extend(["--cache-io-split", str(args.cache_io_split)])
        if args.gguf_embedding:
            full_ppl_cmd.extend(["--gguf-embedding", args.gguf_embedding])
        if args.gguf_full_attn_bin:
            full_ppl_cmd.extend(["--gguf-full-attn-bin", args.gguf_full_attn_bin])
        if args.gguf_full_attn_json:
            full_ppl_cmd.extend(["--gguf-full-attn-json", args.gguf_full_attn_json])
        if args.gguf_lm_head:
            full_ppl_cmd.extend(["--gguf-lm-head", args.gguf_lm_head])
        if args.gguf_qkv_bin:
            full_ppl_cmd.extend(["--gguf-qkv-bin", args.gguf_qkv_bin])
        if args.gguf_qkv_json:
            full_ppl_cmd.extend(["--gguf-qkv-json", args.gguf_qkv_json])
        if args.gguf_linear_bin:
            full_ppl_cmd.extend(["--gguf-linear-bin", args.gguf_linear_bin])
        if args.gguf_linear_json:
            full_ppl_cmd.extend(["--gguf-linear-json", args.gguf_linear_json])

        full_ppl = run_command(
            full_ppl_cmd,
            cwd=root,
            timeout_s=args.ppl_timeout,
            log_path=logs_dir / "ppl_full.log",
        )
        if full_ppl["returncode"] != 0:
            result["failure_stage"] = "full_perplexity"
            result["error"] = "full perplexity benchmark failed"
            result["full_perplexity"] = {
                "returncode": full_ppl["returncode"],
                "timed_out": full_ppl["timed_out"],
                "elapsed_s": round(full_ppl["elapsed_s"], 3),
                "log_path": full_ppl["log_path"],
            }
            print(json.dumps(result, indent=None if args.json else 2))
            return 1
        try:
            full_ppl_metrics = parse_ppl(full_ppl["stdout"] + "\n" + full_ppl["stderr"])
        except ValueError as exc:
            result["failure_stage"] = "full_perplexity"
            result["error"] = str(exc)
            result["full_perplexity"] = {
                "returncode": full_ppl["returncode"],
                "timed_out": full_ppl["timed_out"],
                "elapsed_s": round(full_ppl["elapsed_s"], 3),
                "log_path": full_ppl["log_path"],
            }
            print(json.dumps(result, indent=None if args.json else 2))
            return 1

        result["full_perplexity"] = {
            **full_ppl_metrics,
            "returncode": full_ppl["returncode"],
            "timed_out": full_ppl["timed_out"],
            "elapsed_s": round(full_ppl["elapsed_s"], 3),
            "log_path": full_ppl["log_path"],
        }

    result["quality_pass"] = ppl_metrics["ppl"] <= args.ppl_max
    if args.full_check and result["full_perplexity"]:
        result["quality_pass"] = result["quality_pass"] and (result["full_perplexity"]["ppl"] <= args.full_ppl_max)
    result["score"] = gen_metrics["decode_tok_s"] if result["quality_pass"] else 0.0

    baseline = maybe_load_json(args.baseline_result)
    if baseline:
        result["baseline"] = {
            "path": args.baseline_result,
            "commit": baseline.get("commit"),
            "score": baseline.get("score"),
            "generation": baseline.get("generation"),
            "perplexity": baseline.get("perplexity"),
            "full_perplexity": baseline.get("full_perplexity"),
        }
        result["vs_baseline"] = compute_vs_baseline(result, baseline)

    result["ok"] = True

    if args.save_baseline and result["ok"]:
        baseline_path = Path(os.path.expanduser(args.baseline_result))
        if not baseline_path.is_absolute():
            baseline_path = root / baseline_path
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        baseline_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(result, indent=None if args.json else 2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
