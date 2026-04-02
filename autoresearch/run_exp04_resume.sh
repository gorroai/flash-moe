#!/usr/bin/env bash
# Resume Exp04 from bank=64 (bank=32 hung and was killed), then run Exp05.
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
BEST_SPLIT=4

BASE_ARGS="--model $MODEL --gguf-embedding $EMBED --gguf-lm-head $LMHEAD --tokens $TOKENS --timing"

run_once() {
  $INFER $BASE_ARGS --prompt "$PROMPT" "$@" 2>&1
}

extract_tps() {
  echo "$1" | grep 'Generation:' | grep -oE '\([0-9.]+ tok/s\)' | grep -oE '[0-9.]+'
}

extract_hit() {
  echo "$1" | grep '\[predict\]' | grep -oE 'rate=[0-9.]+%' | grep -oE '[0-9.]+'
}

extract_tokens() {
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
    echo "  FAILED" >&2; return 1
  fi

  local avg_tps avg_hit
  avg_tps=$(python3 -c "print(round($tps_sum/$valid,2))")
  avg_hit=$(python3 -c "print(round($hit_sum/$valid,2))")
  printf "%s\tn/a\t%s\t%s\tkeep\t%s\t%s\n" \
    "$exp_id" "$avg_tps" "$avg_hit" "$lever" "$desc" >> "$TSV"
  printf "  => avg tps=%s  predict_hit=%s%%\n\n" "$avg_tps" "$avg_hit" >&2
}

# ── Exp04 resume: bank=64 through 1024 (bank=32 HANG-logged already) ────────
echo "=== Exp04 RESUME: cache-entries sweep from bank=64 (K=4, Q3, predict, split=$BEST_SPLIT) ===" >&2
for bank in 64 128 256 512 1024; do
  entries=$((bank * 60))
  printf "\n--- bank=%s/layer (entries=%s) ---\n" "$bank" "$entries" >&2
  avg_runs_2_3 "Exp04-bank${bank}" "Lever2-bank" \
    "K=4 Q3 predict split=${BEST_SPLIT} bank=${bank}/layer entries=${entries} tokens=$TOKENS" \
    --q3-experts --predict --cache-io-split "$BEST_SPLIT" --cache-entries "$entries" || true
done

# ── Exp05: split cross-check at best bank ───────────────────────────────────
BEST_BANK=$(python3 - <<'EOF'
import csv, re
rows = [r for r in csv.DictReader(open('autoresearch_results_397b_k4.tsv'), delimiter='\t')
        if r['exp_id'].startswith('Exp04-bank') and r['status'] == 'keep']
if not rows:
    print(64)
else:
    best = max(rows, key=lambda r: float(r['decode_tok_s'] or 0))
    print(re.search(r'bank=(\d+)/layer', best['description']).group(1))
EOF
)
echo "==> Best bank: $BEST_BANK/layer" >&2
entries_best=$((BEST_BANK * 60))

echo "=== Exp05: split cross-check at bank=$BEST_BANK/layer ===" >&2
for split in 1 2 4 8 16; do
  [ "$split" = "$BEST_SPLIT" ] && continue
  printf "\n--- split=%s ---\n" "$split" >&2
  avg_runs_2_3 "Exp05-bank${BEST_BANK}-split${split}" "Lever0-split-xval" \
    "K=4 Q3 predict bank=${BEST_BANK}/layer split=${split} tokens=$TOKENS" \
    --q3-experts --predict --cache-io-split "$split" --cache-entries "$entries_best" || true
done

echo "" >&2
echo "=== Exp04 resume + Exp05 complete. Review $TSV. ===" >&2
cat "$TSV"
