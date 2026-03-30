#!/usr/bin/env python3
"""Initialize a Flash-MoE autoresearch run.

Creates a fresh `autoresearch/<tag>` branch and initializes
`autoresearch/results.tsv` with the expected header.
"""

from __future__ import annotations

import argparse
import datetime as dt
import subprocess
import sys
from pathlib import Path


RESULTS_HEADER = (
    "commit\tscore\tdecode_tok_s\tdecode_vs_4bit_pct\tprefill_tok_s\t"
    "ppl\tppl_vs_4bit\tfull_ppl\tstatus\tdescription\n"
)


def run(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=cwd,
        text=True,
        capture_output=True,
        check=False,
    )


def default_tag() -> str:
    today = dt.datetime.now()
    return today.strftime("%b").lower() + str(today.day)


def main() -> int:
    parser = argparse.ArgumentParser(description="Initialize a Flash-MoE autoresearch run")
    parser.add_argument("--tag", default=default_tag(), help="Run tag, e.g. mar20")
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Allow initialization from a dirty worktree",
    )
    parser.add_argument(
        "--results",
        default="autoresearch/results.tsv",
        help="Results TSV path",
    )
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    branch = f"autoresearch/{args.tag}"
    results_path = repo / args.results
    baselines_dir = repo / "autoresearch" / "baselines"

    status = run(["git", "status", "--porcelain"], repo)
    if status.returncode != 0:
        print(status.stderr.strip() or "git status failed", file=sys.stderr)
        return 1
    if status.stdout.strip() and not args.allow_dirty:
        print(
            "Worktree is dirty. Commit/stash changes first, or re-run with --allow-dirty.",
            file=sys.stderr,
        )
        return 1

    existing = run(["git", "rev-parse", "--verify", branch], repo)
    if existing.returncode == 0:
        print(f"Branch {branch} already exists. Pick a new tag.", file=sys.stderr)
        return 1

    checkout = run(["git", "checkout", "-b", branch], repo)
    if checkout.returncode != 0:
        print(checkout.stderr.strip() or "git checkout failed", file=sys.stderr)
        return 1

    results_path.parent.mkdir(parents=True, exist_ok=True)
    baselines_dir.mkdir(parents=True, exist_ok=True)
    if not results_path.exists():
        results_path.write_text(RESULTS_HEADER, encoding="utf-8")

    print(f"Initialized branch: {branch}")
    print(f"Results file: {results_path}")
    print(f"Baselines dir: {baselines_dir}")
    print("Next step:")
    print("  python3 autoresearch/run_experiment.py --json --save-baseline > autoresearch/last_result.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
