#!/usr/bin/env bash
# Experiments Exp00–Exp05 for 397B campaign.
# Run from ~/flash-moe. No approvals needed through Exp05.

set -euo pipefail
cd ~/flash-moe

INFER=./metal_infer/infer
MODEL=~/Models/flash_mlx_4bit
EMBED=$MODEL/gguf/embedding_q8_0.bin
LMHEAD=$MODEL/gguf/lm_head_q6.bin
PROMPT=$'<|im_start|>user\nExplain the differences between transformer attention variants in detail.<|im_end|>\n<|im_start|>assistant\n'
TOKENS=200
TSV=autoresearch_results_397b.tsv

BASE_ARGS="--model $MODEL --gguf-embedding $EMBED --gguf-lm-head $LMHEAD --tokens $TOKENS --timing --k 10"

ensure_header() {
  if [ ! -f "$TSV" ]; then
    printf "exp_id\tcommit\tdecode_tok_s\thit_rate_pct\tstatus\tlever\tdescription\n" > "$TSV"
  fi
}

run_once() {
  # Returns output; caller parses
  $INFER $BASE_ARGS --prompt "$PROMPT" "$@" 2>&1
}

extract_tps() {
  # "Generation:     4.3 s (6.80 tok/s)"
  echo "$1" | grep 'Generation:' | grep -oE '\([0-9.]+ tok/s\)' | grep -oE '[0-9.]+'
}

extract_hit() {
  # "[predict] hits=N misses=M rate=X%"
  echo "$1" | grep '\[predict\]' | grep -oE 'rate=[0-9.]+%' | grep -oE '[0-9.]+'
}

extract_tokens() {
  # "Tokens:         30 generated"
  echo "$1" | grep 'Tokens:' | grep -oE '[0-9]+' | head -1
}

avg_runs_2_3() {
  local exp_id="$1" lever="$2" desc="$3"
  shift 3
  local tps_sum=0 hit_sum=0 valid=0

  for run in 1 2 3; do
    printf "  run %d/3 ..." "$run" >&2
    local out; out=$(run_once "$@")
    local tps; tps=$(extract_tps "$out")
    local hit; hit=$(extract_hit "$out")
    local tok; tok=$(extract_tokens "$out")
    printf " tps=%s  hit=%s%%  tokens=%s\n" "${tps:-FAIL}" "${hit:--}" "${tok:-0}" >&2
    if [ "$run" -gt 1 ] && [ -n "$tps" ] && [ "${tok:-0}" -gt 0 ]; then
      tps_sum=$(python3 -c "print($tps_sum + $tps)")
      hit_sum=$(python3 -c "print($hit_sum + ${hit:-0})")
      valid=$((valid + 1))
    fi
  done

  if [ "$valid" -eq 0 ]; then
    printf "%s\tn/a\t0.00\t0.00\tdiscard\t%s\t%s — all warm runs failed\n" \
      "$exp_id" "$lever" "$desc" >> "$TSV"
    echo "  FAILED — no valid warm runs" >&2
    return 1
  fi

  local avg_tps avg_hit
  avg_tps=$(python3 -c "print(round($tps_sum/$valid,2))")
  avg_hit=$(python3 -c "print(round($hit_sum/$valid,2))")
  printf "%s\tn/a\t%s\t%s\tkeep\t%s\t%s\n" \
    "$exp_id" "$avg_tps" "$avg_hit" "$lever" "$desc" >> "$TSV"
  printf "  => avg tps=%s  predict_hit=%s%%\n" "$avg_tps" "$avg_hit" >&2
}

ensure_header

# ─────────────────────────────────────────────────────────────
echo "=== Exp00: Baseline — 4-bit, no predict, cache-io-split 4, default cache ===" >&2
avg_runs_2_3 "Exp00" "Baseline" \
  "397B 4-bit no-predict split=4 tokens=$TOKENS" \
  --cache-io-split 4

# ─────────────────────────────────────────────────────────────
echo "=== Exp01: cache-io-split sweep (4-bit, no predict) ===" >&2
for split in 1 2 4 8 16; do
  printf "\n--- split=%s ---\n" "$split" >&2
  avg_runs_2_3 "Exp01-split${split}" "Lever0-split" \
    "397B 4-bit no-predict split=${split} tokens=$TOKENS" \
    --cache-io-split "$split"
done

BEST_SPLIT=$(python3 - <<'EOF'
import csv
rows = [r for r in csv.DictReader(open('autoresearch_results_397b.tsv'), delimiter='\t')
        if r['exp_id'].startswith('Exp01-split')]
if not rows:
    print(4)
else:
    best = max(rows, key=lambda r: float(r['decode_tok_s'] or 0))
    import re; print(re.search(r'split=(\d+)', best['description']).group(1))
EOF
)
echo "==> Best split: $BEST_SPLIT" >&2

# ─────────────────────────────────────────────────────────────
echo "=== Exp02: Q3 quality gate (split=$BEST_SPLIT, no predict) ===" >&2
avg_runs_2_3 "Exp02-Q3" "Lever-Q3" \
  "397B Q3 no-predict split=${BEST_SPLIT} tokens=$TOKENS" \
  --q3-experts --cache-io-split "$BEST_SPLIT"

# ─────────────────────────────────────────────────────────────
echo "=== Exp03: --predict on vs off (Q3, split=$BEST_SPLIT) ===" >&2
avg_runs_2_3 "Exp03-predict-off" "Lever1-predict" \
  "397B Q3 predict=off split=${BEST_SPLIT} tokens=$TOKENS" \
  --q3-experts --cache-io-split "$BEST_SPLIT"

avg_runs_2_3 "Exp03-predict-on" "Lever1-predict" \
  "397B Q3 predict=on split=${BEST_SPLIT} tokens=$TOKENS" \
  --q3-experts --predict --cache-io-split "$BEST_SPLIT"

# ─────────────────────────────────────────────────────────────
# cache-entries = bank_size × 60 layers
echo "=== Exp04: cache-entries sweep (Q3, predict, split=$BEST_SPLIT) ===" >&2
for bank in 64 128 176 256 384 512; do
  entries=$((bank * 60))
  printf "\n--- bank=%s (cache-entries=%s) ---\n" "$bank" "$entries" >&2
  avg_runs_2_3 "Exp04-bank${bank}" "Lever2-bank" \
    "397B Q3 predict split=${BEST_SPLIT} bank=${bank} entries=${entries} tokens=$TOKENS" \
    --q3-experts --predict --cache-io-split "$BEST_SPLIT" --cache-entries "$entries" || true
done

# ─────────────────────────────────────────────────────────────
echo "=== Exp05: split cross-validation at extreme banks ===" >&2
for bank in 64 256; do
  entries=$((bank * 60))
  for split in 4 "$BEST_SPLIT"; do
    [ "$split" = "$BEST_SPLIT" ] && [ "$split" = "4" ] && continue
    printf "\n--- bank=%s split=%s ---\n" "$bank" "$split" >&2
    avg_runs_2_3 "Exp05-bank${bank}-split${split}" "Lever0-split-xval" \
      "397B Q3 predict bank=${bank} split=${split} tokens=$TOKENS" \
      --q3-experts --predict --cache-io-split "$split" --cache-entries "$entries" || true
  done
done

echo "" >&2
echo "=== Exp00–Exp05 complete. Review $TSV before Exp06. ===" >&2
cat "$TSV"
