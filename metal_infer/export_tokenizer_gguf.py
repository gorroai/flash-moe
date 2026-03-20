#!/usr/bin/env python3
"""Export GGUF tokenizer metadata to the compact tokenizer.bin format.

This creates the binary tokenizer format consumed by tokenizer.h/infer.m:
  Header:
    magic: "BPET"
    version: uint32
    vocab_size: uint32
    num_merges: uint32
    num_added: uint32
  Vocab section:
    repeated: uint32 token_id, uint16 str_len, bytes[str_len]
  Merges section:
    repeated: uint16 len_a, bytes[len_a], uint16 len_b, bytes[len_b]
  Added tokens section:
    repeated: uint32 token_id, uint16 str_len, bytes[str_len]

For GGUF, we derive:
  - vocab from tokenizer.ggml.tokens
  - merges from tokenizer.ggml.merges
  - added tokens from tokenizer.ggml.token_type == CONTROL/USER_DEFINED
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


def parse_merge_pair(merge_text):
    parts = merge_text.split(" ", 1)
    if len(parts) != 2:
        raise ValueError(f"Unexpected GGUF merge format: {merge_text!r}")
    return parts[0], parts[1]


def main():
    parser = argparse.ArgumentParser(
        description="Export GGUF tokenizer metadata to tokenizer.bin"
    )
    parser.add_argument(
        "--gguf",
        required=True,
        help="Path to a GGUF directory or one GGUF shard",
    )
    parser.add_argument(
        "--output",
        default="tokenizer.bin",
        help="Output tokenizer.bin path",
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

    vocab_field = reader.fields.get("tokenizer.ggml.tokens")
    merges_field = reader.fields.get("tokenizer.ggml.merges")
    types_field = reader.fields.get("tokenizer.ggml.token_type")
    model_field = reader.fields.get("tokenizer.ggml.model")
    pre_field = reader.fields.get("tokenizer.ggml.pre")

    if vocab_field is None or merges_field is None:
        raise SystemExit("ERROR: GGUF is missing tokenizer.ggml.tokens or tokenizer.ggml.merges")

    vocab_size = len(vocab_field.data)
    merge_count = len(merges_field.data)

    added = []
    if types_field is not None:
        control = int(gguf.TokenType.CONTROL)
        user_defined = int(gguf.TokenType.USER_DEFINED)
        for token_id in range(len(types_field.data)):
            token_type = types_field.contents(token_id)
            if token_type in (control, user_defined):
                added.append((token_id, vocab_field.contents(token_id)))

    print(f"Tokenizer model: {model_field.contents() if model_field else 'unknown'}")
    if pre_field is not None:
        print(f"Pretokenizer:    {pre_field.contents()}")
    print(f"Vocab size:      {vocab_size}")
    print(f"Merges:          {merge_count}")
    print(f"Added tokens:    {len(added)}")
    print(f"Writing:         {out_path}")

    with open(out_path, "wb") as f:
        f.write(b"BPET")
        f.write(struct.pack("<I", 1))
        f.write(struct.pack("<I", vocab_size))
        f.write(struct.pack("<I", merge_count))
        f.write(struct.pack("<I", len(added)))

        for token_id in range(vocab_size):
            token_text = vocab_field.contents(token_id)
            token_bytes = token_text.encode("utf-8")
            if len(token_bytes) > 0xFFFF:
                raise ValueError(f"Token {token_id} is too long: {len(token_bytes)} bytes")
            f.write(struct.pack("<I", token_id))
            f.write(struct.pack("<H", len(token_bytes)))
            f.write(token_bytes)

            if token_id > 0 and token_id % 50000 == 0:
                print(f"  vocab {token_id}/{vocab_size}")

        for merge_idx in range(merge_count):
            left, right = parse_merge_pair(merges_field.contents(merge_idx))
            left_b = left.encode("utf-8")
            right_b = right.encode("utf-8")
            if len(left_b) > 0xFFFF or len(right_b) > 0xFFFF:
                raise ValueError(f"Merge {merge_idx} token is too long")
            f.write(struct.pack("<H", len(left_b)))
            f.write(left_b)
            f.write(struct.pack("<H", len(right_b)))
            f.write(right_b)

            if merge_idx > 0 and merge_idx % 50000 == 0:
                print(f"  merges {merge_idx}/{merge_count}")

        for token_id, token_text in added:
            token_bytes = token_text.encode("utf-8")
            if len(token_bytes) > 0xFFFF:
                raise ValueError(f"Added token {token_id} is too long: {len(token_bytes)} bytes")
            f.write(struct.pack("<I", token_id))
            f.write(struct.pack("<H", len(token_bytes)))
            f.write(token_bytes)

    elapsed = time.time() - start
    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"Done in {elapsed:.1f}s")
    print(f"File size: {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
