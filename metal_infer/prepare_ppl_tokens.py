#!/usr/bin/env python3
"""Prepare ground truth tokens for perplexity evaluation.

Tokenizes text (WikiText-2 or custom file) and writes prompt_tokens.bin format
that the C inference engine can load via load_prompt_tokens().

Binary format: [uint32 count][uint32 id0][uint32 id1]...

Usage:
    # WikiText-2 test set (default, requires `datasets` package)
    python prepare_ppl_tokens.py --output ppl_tokens.bin

    # Limit token count (recommended: 500 for quick test, 2000+ for accurate PPL)
    python prepare_ppl_tokens.py --output ppl_tokens.bin --max-tokens 500

    # Custom text file
    python prepare_ppl_tokens.py --text-file eval.txt --output ppl_tokens.bin

    # Specify tokenizer explicitly
    python prepare_ppl_tokens.py --tokenizer /path/to/tokenizer.json --output ppl_tokens.bin
"""
import argparse
import struct
import sys
import os


def load_wikitext2():
    """Download and return WikiText-2 test set text."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: `datasets` package required for WikiText-2.", file=sys.stderr)
        print("  pip install datasets", file=sys.stderr)
        sys.exit(1)

    print("Loading WikiText-2 (test split)...")
    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    text = '\n\n'.join(dataset['text'])
    print(f"  Text length: {len(text):,} chars")
    return text


def tokenize_with_hf(text, model_name="Qwen/Qwen3.5-397B-A17B"):
    """Tokenize using HuggingFace tokenizer."""
    from transformers import AutoTokenizer
    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    ids = tokenizer.encode(text, add_special_tokens=False)
    return ids


def tokenize_with_json(text, tokenizer_json_path):
    """Tokenize using tokenizers library directly from tokenizer.json."""
    try:
        from tokenizers import Tokenizer
    except ImportError:
        print("ERROR: `tokenizers` package required.", file=sys.stderr)
        print("  pip install tokenizers", file=sys.stderr)
        sys.exit(1)

    print(f"Loading tokenizer: {tokenizer_json_path}")
    tokenizer = Tokenizer.from_file(tokenizer_json_path)
    encoded = tokenizer.encode(text)
    return encoded.ids


def write_tokens_bin(ids, output_path):
    """Write tokens in prompt_tokens.bin format: [uint32 count][uint32 id0]..."""
    with open(output_path, 'wb') as f:
        f.write(struct.pack('<I', len(ids)))
        for token_id in ids:
            f.write(struct.pack('<I', token_id))
    print(f"Wrote {output_path}: {len(ids)} tokens ({os.path.getsize(output_path):,} bytes)")


def main():
    parser = argparse.ArgumentParser(description='Prepare tokens for perplexity evaluation')
    parser.add_argument('--output', '-o', default='ppl_tokens.bin',
                        help='Output binary file (default: ppl_tokens.bin)')
    parser.add_argument('--text-file', default=None,
                        help='Custom text file (default: WikiText-2 test)')
    parser.add_argument('--tokenizer', default=None,
                        help='Path to tokenizer.json (default: use HF AutoTokenizer)')
    parser.add_argument('--model-name', default='Qwen/Qwen3.5-397B-A17B',
                        help='HF model name for tokenizer (default: Qwen/Qwen3.5-397B-A17B)')
    parser.add_argument('--max-tokens', type=int, default=None,
                        help='Limit number of tokens (default: all)')
    args = parser.parse_args()

    # Load text
    if args.text_file:
        print(f"Loading text: {args.text_file}")
        with open(args.text_file, 'r', encoding='utf-8') as f:
            text = f.read()
        print(f"  Text length: {len(text):,} chars")
    else:
        text = load_wikitext2()

    # Tokenize
    if args.tokenizer:
        ids = tokenize_with_json(text, args.tokenizer)
    else:
        ids = tokenize_with_hf(text, args.model_name)

    print(f"Tokenized: {len(ids):,} tokens")

    # Truncate if requested
    if args.max_tokens and len(ids) > args.max_tokens:
        ids = ids[:args.max_tokens]
        print(f"Truncated to {len(ids):,} tokens")

    # Write binary
    write_tokens_bin(ids, args.output)


if __name__ == '__main__':
    main()
