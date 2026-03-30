#!/usr/bin/env python3
"""Measure perplexity of Qwen3.5-397B-A17B using MLX.

This provides the MLX baseline PPL to compare against the C engine's
4-bit and 2-bit quantized inference.

Usage:
    # MLX baseline (default model path)
    python measure_ppl.py --mlx

    # MLX with custom model path
    python measure_ppl.py --mlx --model /path/to/mlx-model

    # Custom text file
    python measure_ppl.py --mlx --text-file eval.txt

    # Limit tokens (recommended for 397B: start with 500)
    python measure_ppl.py --mlx --max-tokens 500

    # Compare all results
    python measure_ppl.py --results
"""
import argparse
import json
import math
import os
import struct
import sys
import time
from pathlib import Path


def load_wikitext2_text():
    """Download and return WikiText-2 test set text."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: `datasets` package required. pip install datasets", file=sys.stderr)
        sys.exit(1)
    print("Loading WikiText-2 (test split)...")
    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    text = '\n\n'.join(dataset['text'])
    print(f"  Text: {len(text):,} chars")
    return text


def measure_ppl_mlx(model_path, text, max_tokens=None):
    """Measure perplexity using MLX framework."""
    try:
        import mlx.core as mx
        import mlx.nn as nn
        from mlx_lm import load
    except ImportError:
        print("ERROR: mlx and mlx_lm required. pip install mlx mlx-lm", file=sys.stderr)
        sys.exit(1)

    print(f"Loading MLX model: {model_path}")
    model, tokenizer = load(model_path)

    # Tokenize
    ids = tokenizer.encode(text)
    if max_tokens and len(ids) > max_tokens:
        ids = ids[:max_tokens]
    print(f"Tokens: {len(ids):,} (scoring {len(ids)-1} predictions)")

    input_ids = mx.array(ids)[None, :]  # [1, seq_len]
    num_eval = len(ids) - 1

    # Process in chunks to avoid OOM on 397B model
    # For MoE models, we process token-by-token (autoregressive)
    # since the model streams experts from disk
    chunk_size = 512  # context window per chunk
    stride = 256      # overlap

    total_nll = 0.0
    tokens_scored = 0
    t0 = time.time()

    seq_len = input_ids.shape[1]

    prev_end = 0
    for begin_loc in range(0, seq_len - 1, stride):
        end_loc = min(begin_loc + chunk_size, seq_len)
        trg_len = end_loc - prev_end
        input_chunk = input_ids[:, begin_loc:end_loc]

        # Forward pass
        logits = model(input_chunk)
        mx.eval(logits)

        # Score only new tokens (avoid double-counting overlap)
        # logits[:, :-1] predicts labels[:, 1:]
        shift_logits = logits[:, -trg_len-1:-1, :]  # [1, trg_len, vocab]
        shift_labels = input_ids[:, end_loc-trg_len:end_loc]  # [1, trg_len]

        # Cross-entropy per token
        log_probs = nn.log_softmax(shift_logits, axis=-1)  # [1, trg_len, vocab]
        # Gather log-probs for target tokens
        target_log_probs = mx.take_along_axis(
            log_probs, shift_labels[:, :, None], axis=-1
        ).squeeze(-1)  # [1, trg_len]

        nll = -mx.sum(target_log_probs).item()
        total_nll += nll
        tokens_scored += trg_len
        prev_end = end_loc

        elapsed = time.time() - t0
        avg_nll = total_nll / tokens_scored
        print(f"\r  [{tokens_scored}/{num_eval}] PPL={math.exp(avg_nll):.2f} "
              f"CE={avg_nll:.4f} ({tokens_scored/elapsed:.1f} tok/s)", end='', flush=True)

        if end_loc >= seq_len:
            break

    elapsed = time.time() - t0
    avg_nll = total_nll / tokens_scored
    ppl = math.exp(avg_nll)

    print()
    return {
        'perplexity': ppl,
        'cross_entropy': avg_nll,
        'tokens': tokens_scored,
        'time': elapsed,
        'tokens_per_sec': tokens_scored / elapsed if elapsed > 0 else 0,
    }


def print_results():
    """Print saved results from results/perplexity.json."""
    results_file = Path(__file__).parent / "results" / "perplexity.json"
    if not results_file.exists():
        print("No results found. Run measurements first.")
        return

    with open(results_file) as f:
        results = json.load(f)

    sorted_results = sorted(results.items(), key=lambda x: x[1].get('perplexity', float('inf')))

    print()
    print("=" * 65)
    print("PERPLEXITY RESULTS (sorted by best PPL)")
    print("=" * 65)
    for rank, (key, entry) in enumerate(sorted_results, 1):
        ppl = entry.get('perplexity', 0)
        ce = entry.get('cross_entropy', 0)
        tokens = entry.get('tokens', 0)
        tps = entry.get('tokens_per_sec', 0)
        marker = " <-- best" if rank == 1 else ""
        print(f"  #{rank} {key}")
        print(f"     PPL: {ppl:.2f}  CE: {ce:.4f}  tokens: {tokens}  "
              f"({tps:.1f} tok/s){marker}")
        print()
    print("=" * 65)


def save_result(key, result, results_dir=None):
    """Save result to results/perplexity.json."""
    from datetime import datetime
    if results_dir is None:
        results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    results_file = results_dir / "perplexity.json"

    existing = {}
    if results_file.exists():
        with open(results_file) as f:
            existing = json.load(f)

    existing[key] = {
        'perplexity': round(result['perplexity'], 2),
        'cross_entropy': round(result['cross_entropy'], 4),
        'tokens': result['tokens'],
        'tokens_per_sec': round(result['tokens_per_sec'], 1),
        'time_seconds': round(result['time'], 1),
        'timestamp': datetime.now().isoformat(),
    }

    with open(results_file, 'w') as f:
        json.dump(existing, f, indent=2)
    print(f"Saved to {results_file}")


def main():
    parser = argparse.ArgumentParser(description='Measure perplexity for Flash-MoE')
    parser.add_argument('--mlx', action='store_true',
                        help='Measure MLX baseline PPL')
    parser.add_argument('--model', default=os.path.expanduser(
                            '~/Models/mlx-community-Qwen3.5-397B-A17B-4bit'),
                        help='MLX model path')
    parser.add_argument('--text-file', default=None,
                        help='Custom text file (default: WikiText-2 test)')
    parser.add_argument('--max-tokens', type=int, default=None,
                        help='Limit number of tokens')
    parser.add_argument('--results', action='store_true',
                        help='Print all saved results')
    parser.add_argument('--save', action='store_true', default=True,
                        help='Save results to results/perplexity.json')
    args = parser.parse_args()

    if args.results:
        print_results()
        return

    if not args.mlx:
        parser.print_help()
        print("\nFor C engine PPL, use:")
        print("  # Prepare tokens")
        print("  python metal_infer/prepare_ppl_tokens.py --max-tokens 500 -o ppl_tokens.bin")
        print()
        print("  # 4-bit PPL")
        print("  ./metal_infer/infer --model ~/Models/flash_mlx_4bit --ppl ppl_tokens.bin")
        print()
        print("  # 2-bit PPL")
        print("  ./metal_infer/infer --model ~/Models/flash_mlx_4bit --ppl ppl_tokens.bin --2bit")
        return

    # Load text
    if args.text_file:
        with open(args.text_file, 'r', encoding='utf-8') as f:
            text = f.read()
        print(f"Text: {args.text_file} ({len(text):,} chars)")
    else:
        text = load_wikitext2_text()

    # Measure
    result = measure_ppl_mlx(args.model, text, args.max_tokens)

    # Print results
    print(f"\n=== MLX Perplexity Results ===")
    print(f"Tokens evaluated: {result['tokens']}")
    print(f"Cross-entropy:    {result['cross_entropy']:.4f} nats")
    print(f"Perplexity:       {result['perplexity']:.2f}")
    print(f"Time:             {result['time']:.1f} s ({result['tokens_per_sec']:.2f} tok/s)")
    print(f"Model:            {args.model}")

    # Save
    if args.save:
        key = f"mlx-4bit"
        save_result(key, result)


if __name__ == '__main__':
    main()
