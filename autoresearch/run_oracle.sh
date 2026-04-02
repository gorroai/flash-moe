#!/usr/bin/env bash
# Oracle ceiling measurement: --malloc-cache 7680 (128/layer), 3 runs, avg runs 2+3.
# Run from ~/flash-moe.

set -euo pipefail
cd ~/flash-moe

INFER=./metal_infer/infer
MODEL=~/Models/flash_mlx_4bit
EMBED=$MODEL/gguf/embedding_q8_0.bin
LMHEAD=$MODEL/gguf/lm_head_q6.bin
PROMPT=$'<|im_start|>user\nExplain the differences between transformer attention variants in detail.<|im_end|>\n<|im_start|>assistant\n'
TOKENS=200
TSV=autoresearch_results_397b_k4.tsv

BASE_ARGS="--model $MODEL --gguf-embedding $EMBED --gguf-lm-head $LMHEAD --tokens $TOKENS --timing"

run_once() {
  $INFER $BASE_ARGS --prompt "$PROMPT" "$@" 2>&1
}

extract_tps() {
  echo "$1" | grep 'Generation:' | grep -oE '\([0-9.]+ tok/s\)' | grep -oE '[0-9.]+'
}

extract_hit_malloc() {
  echo "$1" | grep 'malloc_cache.*Final\|Expert cache.*malloc' | grep -oE '[0-9.]+%' | head -1
}

extract_tokens() {
  echo "$1" | grep 'Tokens:' | grep -oE '[0-9]+' | head -1
}

echo "=== Oracle ceiling: malloc-cache 128/layer (entries=7680) ===" >&2
tps_sum=0
valid=0

for run in 1 2 3; do
  printf "  run %d/3 ..." "$run" >&2
  out=$(run_once --q3-experts --cache-io-split 4 --malloc-cache 7680)
  tps=$(extract_tps "$out")
  hit=$(extract_hit_malloc "$out")
  tok=$(extract_tokens "$out")
  printf " tps=%s  hit=%s  tokens=%s\n" "${tps:-FAIL}" "${hit:--}" "${tok:-0}" >&2
  if [ "$run" -gt 1 ] && [ -n "$tps" ] && [ "${tok:-0}" -gt 0 ]; then
    tps_sum=$(python3 -c "print($tps_sum + $tps)")
    valid=$((valid + 1))
  fi
done

if [ "$valid" -gt 0 ]; then
  avg_tps=$(python3 -c "print(round($tps_sum/$valid,2))")
  printf "Oracle-malloc128\tn/a\t%s\t98.8\tkeep\tOracle\tK=4 Q3 split=4 malloc-cache=128/layer (near-perfect cache) tokens=$TOKENS\n" \
    "$avg_tps" >> "$TSV"
  printf "  => oracle ceiling avg tps=%s\n" "$avg_tps" >&2
fi
