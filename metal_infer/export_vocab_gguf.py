#!/usr/bin/env python3
"""Export GGUF tokenizer metadata to Flash-MoE's simple vocab.bin format.

This writes the decode-only vocabulary format expected by infer.m:
  uint32 num_entries
  uint32 max_id
  repeated:
    uint16 byte_len
    uint8[byte_len] token_utf8

It does not export tokenizer merges; this file is only for token decoding.
"""

import argparse
import os
import struct
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import repack_experts as flash_pack


def main():
    parser = argparse.ArgumentParser(
        description="Export GGUF tokenizer metadata to Flash-MoE vocab.bin"
    )
    parser.add_argument(
        "--gguf",
        required=True,
        help="Path to a GGUF directory or one GGUF shard",
    )
    parser.add_argument(
        "--output",
        default="vocab.bin",
        help="Output vocab.bin path",
    )
    args = parser.parse_args()

    gguf = flash_pack.require_gguf()
    gguf_files = flash_pack.collect_gguf_files(args.gguf)
    first_shard = gguf_files[0]
    out_path = Path(os.path.expanduser(args.output))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Reading tokenizer metadata from: {first_shard}")
    start = time.time()
    reader = gguf.GGUFReader(first_shard)

    field = reader.fields.get("tokenizer.ggml.tokens")
    if field is None:
        raise SystemExit("ERROR: tokenizer.ggml.tokens metadata not found in GGUF")

    num_entries = len(field.data)
    max_id = num_entries - 1
    print(f"Exporting {num_entries} tokens to: {out_path}")

    with open(out_path, "wb") as f:
        f.write(struct.pack("<II", num_entries, max_id))
        for idx in range(num_entries):
            token = field.contents(idx)
            token_bytes = token.encode("utf-8")
            if len(token_bytes) > 0xFFFF:
                raise ValueError(f"Token {idx} is too long for vocab.bin: {len(token_bytes)} bytes")
            f.write(struct.pack("<H", len(token_bytes)))
            if token_bytes:
                f.write(token_bytes)

            if idx > 0 and idx % 50000 == 0:
                print(f"  {idx}/{num_entries} tokens...")

    elapsed = time.time() - start
    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"Done in {elapsed:.1f}s")
    print(f"File size: {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
