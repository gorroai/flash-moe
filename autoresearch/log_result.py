#!/usr/bin/env python3
"""Append an autoresearch JSON result to autoresearch/results.tsv."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


RESULTS_HEADER = (
    "commit\tscore\tdecode_tok_s\tdecode_vs_4bit_pct\tprefill_tok_s\t"
    "ppl\tppl_vs_4bit\tfull_ppl\tstatus\tdescription\n"
)


def fmt_num(value: object, digits: int = 2) -> str:
    if value is None:
        return ""
    if isinstance(value, (int, float)):
        return f"{value:.{digits}f}"
    return str(value)


def main() -> int:
    parser = argparse.ArgumentParser(description="Append an autoresearch result to results.tsv")
    parser.add_argument("--json-path", default="autoresearch/last_result.json", help="Benchmark JSON path")
    parser.add_argument("--results", default="autoresearch/results.tsv", help="Results TSV path")
    parser.add_argument("--status", required=True, choices=["keep", "discard", "crash"], help="Row status")
    parser.add_argument("--description", required=True, help="Short experiment description")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    json_path = root / args.json_path
    results_path = root / args.results

    with json_path.open() as f:
        data = json.load(f)

    vs_base = data.get("vs_baseline") or {}
    gen = data.get("generation") or {}
    ppl = data.get("perplexity") or {}
    full_ppl = data.get("full_perplexity") or {}

    results_path.parent.mkdir(parents=True, exist_ok=True)
    if not results_path.exists():
        results_path.write_text(RESULTS_HEADER, encoding="utf-8")

    row = [
        data.get("commit", ""),
        fmt_num(data.get("score"), 2),
        fmt_num(gen.get("decode_tok_s"), 2),
        fmt_num(vs_base.get("decode_tok_s_pct"), 2),
        fmt_num(gen.get("prefill_tok_s"), 2),
        fmt_num(ppl.get("ppl"), 2),
        fmt_num(vs_base.get("ppl_abs"), 2),
        fmt_num(full_ppl.get("ppl"), 2),
        args.status,
        args.description.replace("\t", " ").strip(),
    ]

    with results_path.open("a", encoding="utf-8") as f:
        f.write("\t".join(row) + "\n")

    print(f"Appended result to {results_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
