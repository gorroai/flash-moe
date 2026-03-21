# Model Download And Conversion

This guide covers both layers of setup:

1. convert the upstream MLX 4-bit model into the base Flash-MoE runtime layout
2. optionally extract GGUF-backed artifacts into the same model directory for hybrid experiments

The canonical runtime directory is a single `--model` tree such as:

```bash
./metal_infer/infer --model /path/to/flash_mlx_4bit --prompt "Hello" --tokens 32
```

## Output Layout

After the base MLX conversion, your model directory should look like:

```text
flash_mlx_4bit/
  model_weights.bin
  model_weights.json
  vocab.bin
  tokenizer.bin
  packed_experts/
    layout.json
    layer_00.bin
    ...
    layer_59.bin
```

After the optional GGUF hybrid conversion, it can also contain:

```text
flash_mlx_4bit/
  gguf/
    embedding_q8_0.bin
    embedding_q8_0.json
    lm_head_q6.bin
    lm_head_q6.json
    full_attn_q8_0.bin
    full_attn_q8_0.json
    attn_qkv_q8_0.bin
    attn_qkv_q8_0.json
    linear_q8_0.bin
    linear_q8_0.json
    q3-tensor-inventory.json
  packed_experts_Q3/
    layout.json
    layer_00.bin
    ...
    layer_59.bin
```

`flash_mlx_4bit/gguf/` is the canonical location for extracted GGUF artifacts. The repo-local `autoresearch/gguf` path is only a compatibility symlink.

## Support Matrix

These conversion paths are implemented today:

| Family | Source | Output | Status |
|---|---|---|---|
| Base runtime weights | MLX safetensors | `model_weights.bin/json` | implemented |
| Base routed experts | MLX safetensors | `packed_experts/` | implemented |
| Embedding | GGUF `Q8_0` | `gguf/embedding_q8_0.*` | implemented |
| LM head | GGUF `Q6_K` | `gguf/lm_head_q6.*` | implemented |
| Linear-attn `attn_qkv` | GGUF `Q8_0` | `gguf/attn_qkv_q8_0.*` | implemented |
| Full-attn `q/k/v/o` | GGUF `Q8_0` | `gguf/full_attn_q8_0.*` | implemented |
| Linear-attn `attn_gate` + `ssm_out` | GGUF `Q8_0` | `gguf/linear_q8_0.*` | implemented |
| Routed experts | GGUF `IQ3_XXS/IQ4_XS/Q5_K` | `packed_experts_Q3/` | implemented |
| Shared experts | GGUF mostly `Q8_0`, layer 27 `BF16` | no script yet | not yet scripted |

Important: shared-expert GGUF conversion is not automated in this repo yet. Do not assume the shared expert family is already covered by the existing scripts.

## Storage And Time Expectations

- MLX source download: about `224 GB`
- Converted base Flash-MoE output: about `214.5 GB`
- Optional GGUF hybrid artifacts:
  - resident overlays: a few extra GB total
  - `packed_experts_Q3/`: another full routed-expert tree
- Recommended free disk space during conversion: at least `450 GB`

The slowest base step is usually repacking the 60 routed expert layers.

## Prerequisites

From the repo root:

```bash
xcode-select -p
python3 --version
```

Install the Python dependencies used by the setup scripts:

```bash
python3 -m pip install --upgrade numpy huggingface_hub
```

Install the Hugging Face CLI if needed:

```bash
curl -LsSf https://hf.co/cli/install.sh | bash
hf --help
```

If needed:

```bash
export HF_HUB_DOWNLOAD_TIMEOUT=30
```

## Step 1: Choose Paths

From the repo root, define:

```bash
REPO=/Users/anemll/SourceRelease/GITHUB/ML_playground/flash-moe-org
MODEL=$HOME/Models/mlx-community-Qwen3.5-397B-A17B-4bit
OUT=$HOME/Models/flash_mlx_4bit
GGUF=$HOME/Models/Qwen3.5/Qwen3.5-397B-A17B-GGUF-UD-Q3_K_XL/Qwen3.5-397B-A17B-UD-Q3_K_XL-00001-of-00005.gguf
LLAMA_CPP=$HOME/SourceRelease/GITHUB/ML_playground/llama.cpp

mkdir -p "$MODEL" "$OUT" "$OUT/gguf"
```

`MODEL` is the untouched MLX download. `OUT` is the directory you pass to `--model`.

## Step 2: Download The MLX Source Model

```bash
hf download mlx-community/Qwen3.5-397B-A17B-4bit --local-dir "$MODEL"
```

You should end up with:

```text
config.json
model.safetensors.index.json
model-00001-of-00046.safetensors
...
model-00046-of-00046.safetensors
tokenizer.json
tokenizer_config.json
```

## Step 3: Point `expert_index.json` At The MLX Download

`repack_experts.py` reads its source path from `expert_index.json`.

```bash
MODEL="$MODEL" python3 - <<'PY'
import json
import os

path = "expert_index.json"
with open(path) as f:
    data = json.load(f)

data["model_path"] = os.environ["MODEL"]

with open(path, "w") as f:
    json.dump(data, f, indent=2)
    f.write("\n")

print(f"Updated {path} -> {data['model_path']}")
PY
```

Sanity check:

```bash
python3 repack_experts.py --dry-run
```

## Step 4: Build The Base Routed-Expert Files

This writes the native MLX-derived routed expert layout used by plain 4-bit inference:

```bash
python3 repack_experts.py --output "$OUT/packed_experts"
```

Optional checks:

```bash
python3 repack_experts.py --verify-only 0 --output "$OUT/packed_experts"
python3 repack_experts.py --layers 0-4 --output "$OUT/packed_experts"
```

## Step 5: Build The Base Non-Expert Weight Blob

```bash
python3 metal_infer/extract_weights.py --model "$MODEL" --output "$OUT"
```

This writes:

```text
$OUT/model_weights.bin
$OUT/model_weights.json
```

## Step 6: Export `vocab.bin` And `tokenizer.bin`

```bash
python3 metal_infer/export_vocab.py "$MODEL/tokenizer.json" "$OUT/vocab.bin"
python3 metal_infer/export_tokenizer.py "$MODEL/tokenizer.json" "$OUT/tokenizer.bin"
```

## Step 7: Build The Inference Binary

```bash
make -C metal_infer infer
```

Optional:

```bash
make -C metal_infer chat
```

## Step 8: Smoke Test The Base 4-bit MLX Runtime

```bash
./metal_infer/infer --model "$OUT" --prompt "What is Apple Neural Engine?" --tokens 24 --stream
```

Timing:

```bash
./metal_infer/infer --model "$OUT" --prompt "What is Apple Neural Engine?" --tokens 200 --timing
```

Full PPL:

```bash
./metal_infer/infer --model "$OUT" --ppl "$REPO/ppl_tokens_2k.bin"
```

## Step 9: Sweep The GGUF Metadata

The GGUF scripts only inspect metadata and individual tensor payloads. They do not load the full model into memory.

```bash
python3 autoresearch/sweep_gguf_tensors.py \
  --gguf "$GGUF" \
  --llama-cpp-root "$LLAMA_CPP" \
  --write-markdown "$REPO/docs/gguf-q3-tensor-sweep.md"
```

To also keep the JSON inventory in the model directory:

```bash
python3 autoresearch/sweep_gguf_tensors.py \
  --gguf "$GGUF" \
  --llama-cpp-root "$LLAMA_CPP" \
  --write-json "$OUT/gguf/q3-tensor-inventory.json"
```

## Step 10: Extract GGUF Resident Artifacts

These write standalone blobs into the model-local `gguf/` directory.

LM head:

```bash
python3 autoresearch/extract_gguf_lm_head.py \
  --gguf "$GGUF" \
  --llama-cpp-root "$LLAMA_CPP" \
  --out-bin "$OUT/gguf/lm_head_q6.bin" \
  --out-json "$OUT/gguf/lm_head_q6.json"
```

Embedding:

```bash
python3 autoresearch/extract_gguf_embedding.py \
  --gguf "$GGUF" \
  --llama-cpp-root "$LLAMA_CPP" \
  --out-bin "$OUT/gguf/embedding_q8_0.bin" \
  --out-json "$OUT/gguf/embedding_q8_0.json"
```

Linear-attention `attn_qkv`:

```bash
python3 autoresearch/extract_gguf_qkv_overlay.py \
  --gguf "$GGUF" \
  --llama-cpp-root "$LLAMA_CPP" \
  --out-bin "$OUT/gguf/attn_qkv_q8_0.bin" \
  --out-json "$OUT/gguf/attn_qkv_q8_0.json"
```

Full-attention `q/k/v/o`:

```bash
python3 autoresearch/extract_gguf_full_attn_overlay.py \
  --gguf "$GGUF" \
  --llama-cpp-root "$LLAMA_CPP" \
  --out-bin "$OUT/gguf/full_attn_q8_0.bin" \
  --out-json "$OUT/gguf/full_attn_q8_0.json"
```

Linear-attention `attn_gate` + `ssm_out`:

```bash
python3 autoresearch/extract_gguf_linear_overlay.py \
  --gguf "$GGUF" \
  --llama-cpp-root "$LLAMA_CPP" \
  --out-bin "$OUT/gguf/linear_q8_0.bin" \
  --out-json "$OUT/gguf/linear_q8_0.json"
```

## Step 11: Repack GGUF Streamed Experts

This writes exact GGUF-streamed expert bytes into `packed_experts_Q3/`.

Full routed-expert repack:

```bash
python3 autoresearch/repack_experts_q3.py \
  --model "$OUT" \
  --gguf "$GGUF" \
  --llama-cpp-root "$LLAMA_CPP" \
  --output "$OUT/packed_experts_Q3" \
  --layers all \
  --include-outlier-layer
```

Partial bring-up examples:

```bash
python3 autoresearch/repack_experts_q3.py \
  --model "$OUT" \
  --gguf "$GGUF" \
  --llama-cpp-root "$LLAMA_CPP" \
  --output "$OUT/packed_experts_Q3" \
  --layers 0-4

python3 autoresearch/repack_experts_q3.py \
  --model "$OUT" \
  --gguf "$GGUF" \
  --llama-cpp-root "$LLAMA_CPP" \
  --output "$OUT/packed_experts_Q3" \
  --layers 27 \
  --include-outlier-layer
```

Current routed-expert layout:

- normal layers:
  - `gate_proj.weight = IQ3_XXS`
  - `up_proj.weight = IQ3_XXS`
  - `down_proj.weight = IQ4_XS`
- outlier layer `27`:
  - `gate_proj.weight = IQ4_XS`
  - `up_proj.weight = IQ4_XS`
  - `down_proj.weight = Q5_K`

Repack hygiene:

- if you are rebuilding the same on-disk Q3 layout, you do not need to delete `packed_experts_Q3/` first
- if the layout format changed, rebuild into an empty directory or delete/rename the old `packed_experts_Q3/` first

## Step 12: Shared-Expert Status

The shared-expert GGUF family is not yet scripted in this repo.

Current GGUF shared-expert types from the sweep:

- `blk.*.ffn_gate_shexp.weight`: mostly `Q8_0`, one `BF16` outlier
- `blk.*.ffn_up_shexp.weight`: mostly `Q8_0`, one `BF16` outlier
- `blk.*.ffn_down_shexp.weight`: mostly `Q8_0`, one `BF16` outlier

So today:

- routed streamed experts: implemented
- LM head and other resident overlays: implemented
- shared experts: not yet converted by an automated script

## Step 13: Run Common Configurations

Base 4-bit MLX:

```bash
./metal_infer/infer \
  --model "$OUT" \
  --prompt "What is Apple Neural Engine?" \
  --tokens 24 \
  --stream
```

GGUF LM head on top of the base model:

```bash
./metal_infer/infer \
  --model "$OUT" \
  --gguf-lm-head "$OUT/gguf/lm_head_q6.bin" \
  --prompt "What is Apple Neural Engine?" \
  --tokens 24 \
  --stream
```

Q3 GGUF experts:

```bash
./metal_infer/infer \
  --model "$OUT" \
  --q3-experts \
  --prompt "What is Apple Neural Engine?" \
  --tokens 24 \
  --stream
```

Q3 GGUF experts plus GGUF LM head:

```bash
./metal_infer/infer \
  --model "$OUT" \
  --q3-experts \
  --gguf-lm-head "$OUT/gguf/lm_head_q6.bin" \
  --prompt "What is Apple Neural Engine?" \
  --tokens 24 \
  --stream
```

All currently implemented GGUF resident overlays on top of the base model:

```bash
./metal_infer/infer \
  --model "$OUT" \
  --gguf-embedding "$OUT/gguf/embedding_q8_0.bin" \
  --gguf-lm-head "$OUT/gguf/lm_head_q6.bin" \
  --gguf-qkv-bin "$OUT/gguf/attn_qkv_q8_0.bin" \
  --gguf-qkv-json "$OUT/gguf/attn_qkv_q8_0.json" \
  --gguf-full-attn-bin "$OUT/gguf/full_attn_q8_0.bin" \
  --gguf-full-attn-json "$OUT/gguf/full_attn_q8_0.json" \
  --gguf-linear-bin "$OUT/gguf/linear_q8_0.bin" \
  --gguf-linear-json "$OUT/gguf/linear_q8_0.json" \
  --prompt "What is Apple Neural Engine?" \
  --tokens 24 \
  --stream
```

All currently implemented GGUF files together:

```bash
./metal_infer/infer \
  --model "$OUT" \
  --q3-experts \
  --gguf-embedding "$OUT/gguf/embedding_q8_0.bin" \
  --gguf-lm-head "$OUT/gguf/lm_head_q6.bin" \
  --gguf-qkv-bin "$OUT/gguf/attn_qkv_q8_0.bin" \
  --gguf-qkv-json "$OUT/gguf/attn_qkv_q8_0.json" \
  --gguf-full-attn-bin "$OUT/gguf/full_attn_q8_0.bin" \
  --gguf-full-attn-json "$OUT/gguf/full_attn_q8_0.json" \
  --gguf-linear-bin "$OUT/gguf/linear_q8_0.bin" \
  --gguf-linear-json "$OUT/gguf/linear_q8_0.json" \
  --prompt "What is Apple Neural Engine?" \
  --tokens 24 \
  --stream
```

This is the current “all GGUF” stack in this repo:

- routed experts via `--q3-experts`
- embedding
- LM head
- QKV overlay
- full-attention overlay
- linear overlay

It still does not include shared experts, because the shared-expert GGUF converter is not implemented yet.

## Step 14: Get Full PPL

Use `ppl_tokens_2k.bin` for the main comparison check.

Base 4-bit MLX:

```bash
./metal_infer/infer --model "$OUT" --ppl "$REPO/ppl_tokens_2k.bin"
```

Q3 GGUF experts:

```bash
./metal_infer/infer --model "$OUT" --q3-experts --ppl "$REPO/ppl_tokens_2k.bin"
```

Q3 GGUF experts plus GGUF LM head:

```bash
./metal_infer/infer \
  --model "$OUT" \
  --q3-experts \
  --gguf-lm-head "$OUT/gguf/lm_head_q6.bin" \
  --ppl "$REPO/ppl_tokens_2k.bin"
```

All implemented resident overlays plus the base experts:

```bash
./metal_infer/infer \
  --model "$OUT" \
  --gguf-embedding "$OUT/gguf/embedding_q8_0.bin" \
  --gguf-lm-head "$OUT/gguf/lm_head_q6.bin" \
  --gguf-qkv-bin "$OUT/gguf/attn_qkv_q8_0.bin" \
  --gguf-qkv-json "$OUT/gguf/attn_qkv_q8_0.json" \
  --gguf-full-attn-bin "$OUT/gguf/full_attn_q8_0.bin" \
  --gguf-full-attn-json "$OUT/gguf/full_attn_q8_0.json" \
  --gguf-linear-bin "$OUT/gguf/linear_q8_0.bin" \
  --gguf-linear-json "$OUT/gguf/linear_q8_0.json" \
  --ppl "$REPO/ppl_tokens_2k.bin"
```

All currently implemented GGUF files together:

```bash
./metal_infer/infer \
  --model "$OUT" \
  --q3-experts \
  --gguf-embedding "$OUT/gguf/embedding_q8_0.bin" \
  --gguf-lm-head "$OUT/gguf/lm_head_q6.bin" \
  --gguf-qkv-bin "$OUT/gguf/attn_qkv_q8_0.bin" \
  --gguf-qkv-json "$OUT/gguf/attn_qkv_q8_0.json" \
  --gguf-full-attn-bin "$OUT/gguf/full_attn_q8_0.bin" \
  --gguf-full-attn-json "$OUT/gguf/full_attn_q8_0.json" \
  --gguf-linear-bin "$OUT/gguf/linear_q8_0.bin" \
  --gguf-linear-json "$OUT/gguf/linear_q8_0.json" \
  --ppl "$REPO/ppl_tokens_2k.bin"
```

## Full Copy-Paste Base Pipeline

From the repo root:

```bash
python3 -m pip install --upgrade numpy huggingface_hub
curl -LsSf https://hf.co/cli/install.sh | bash

REPO=/Users/anemll/SourceRelease/GITHUB/ML_playground/flash-moe-org
MODEL=$HOME/Models/mlx-community-Qwen3.5-397B-A17B-4bit
OUT=$HOME/Models/flash_mlx_4bit

mkdir -p "$MODEL" "$OUT"

hf download mlx-community/Qwen3.5-397B-A17B-4bit --local-dir "$MODEL"

MODEL="$MODEL" python3 - <<'PY'
import json
import os

path = "expert_index.json"
with open(path) as f:
    data = json.load(f)
data["model_path"] = os.environ["MODEL"]
with open(path, "w") as f:
    json.dump(data, f, indent=2)
    f.write("\n")
print(f"Updated {path} -> {data['model_path']}")
PY

python3 repack_experts.py --output "$OUT/packed_experts"
python3 metal_infer/extract_weights.py --model "$MODEL" --output "$OUT"
python3 metal_infer/export_vocab.py "$MODEL/tokenizer.json" "$OUT/vocab.bin"
python3 metal_infer/export_tokenizer.py "$MODEL/tokenizer.json" "$OUT/tokenizer.bin"
make -C metal_infer infer
./metal_infer/infer --model "$OUT" --prompt "Hello" --tokens 16 --stream
```

## Troubleshooting

- `ERROR: ... model.safetensors.index.json not found`
  The `MODEL` path is wrong or the download did not finish.

- `Short read` or missing shard errors during `repack_experts.py`
  One or more `.safetensors` shards are missing from the MLX download.

- `repack_experts.py --dry-run` points at the wrong directory
  Update `expert_index.json` again and re-run the dry run.

- `infer` cannot find `model_weights.bin`, `model_weights.json`, `vocab.bin`, or `tokenizer.bin`
  Make sure those files are directly under `"$OUT"` and `packed_experts/` is inside `"$OUT"`.

- `--q3-experts` looks unchanged after a repack
  Check whether you changed the on-disk Q3 layout. If yes, rebuild `packed_experts_Q3/` into an empty directory.

- You want one self-contained runtime tree
  Put both the base MLX artifacts and the optional GGUF artifacts under the same `"$OUT"` directory and pass only `--model "$OUT"`.
