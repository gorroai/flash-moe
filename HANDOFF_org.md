# Handoff: Export Pipeline & UX Improvements

## Summary

Made the inference engine self-contained — all model files can live in a single `--model` directory, and added a clean `--stream` output mode.

## Changes

### 1. `repack_experts.py` — Export to custom directory
- Added `--output` flag to write packed experts to any directory (was hardcoded to `<model_path>/packed_experts`)
- Changed default `--index` path from hardcoded `/Users/danielwoods/...` to `expert_index.json` (relative)

### 2. `expert_index.json` — Updated model path
- Changed `model_path` from original author's HuggingFace cache to `/Users/anemll/Models/mlx-community-Qwen3.5-397B-A17B-4bit`

### 3. `metal_infer/export_vocab.py` — NEW
- Generates `vocab.bin` from `tokenizer.json` (this script was missing from the repo)
- Implements GPT-2 byte-level BPE decoding (`Ġ` → space, `Ċ` → newline, etc.) so decoded tokens display correctly
- Usage: `python3 metal_infer/export_vocab.py <tokenizer.json> [output.bin]`

### 4. `metal_infer/infer.m` — `--model` as single source of truth
- Default paths for `model_weights.bin`, `model_weights.json`, `vocab.bin`, and `tokenizer.bin` now check `<model_path>/` first, then fall back to legacy `metal_infer/` and `./` locations
- No more need to pass `--weights`, `--manifest`, `--vocab` separately when all files are under `--model`

### 5. `metal_infer/infer.m` — `--stream` flag
- New `--stream` mode: outputs only generated text + a one-line summary
- Suppresses: engine banner, prefill progress, `[prompt]` tokens, per-token `[gen X/Y]` progress, `[eos]` messages, and full statistics block
- Ends with: `decode: XX.XX t/s, prefill: XX.XX t/s`

## Full Export Pipeline

```bash
MODEL=/Users/anemll/Models/mlx-community-Qwen3.5-397B-A17B-4bit
OUT=~/Models/flash_mlx_4bit

# 1. Repack expert weights (~209 GB, ~60 layers)
python3 repack_experts.py --output $OUT/packed_experts

# 2. Extract non-expert weights (~5.5 GB)
python3 metal_infer/extract_weights.py --model $MODEL --output $OUT

# 3. Export vocab (token decoding)
python3 metal_infer/export_vocab.py $MODEL/tokenizer.json $OUT/vocab.bin

# 4. Export tokenizer (prompt encoding)
python3 metal_infer/export_tokenizer.py $MODEL/tokenizer.json $OUT/tokenizer.bin
```

## Running

```bash
# Full output with timing
./metal_infer/infer --model ~/Models/flash_mlx_4bit --prompt "Hello" --tokens 100 --timing

# Clean streaming output
./metal_infer/infer --model ~/Models/flash_mlx_4bit --prompt "Hello" --tokens 100 --stream
```

## Files in export directory

```
~/Models/flash_mlx_4bit/
  model_weights.bin      # Non-expert weights (5.5 GB, mmap'd)
  model_weights.json     # Tensor manifest
  vocab.bin              # Vocabulary for token decoding
  tokenizer.bin          # BPE tokenizer for prompt encoding
  packed_experts/        # 60 layer files (~3.63 GB each, ~209 GB total)
    layout.json
    layer_00.bin
    ...
    layer_59.bin
```
