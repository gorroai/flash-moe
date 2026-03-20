# Model Download And Conversion

This guide turns the upstream MLX model into the file layout expected by Flash-MoE.

It assumes you are starting from a fresh clone of this repo and want a self-contained output directory that works with:

```bash
./metal_infer/infer --model /path/to/flash_mlx_4bit --prompt "Hello" --tokens 32
```

## What You Will End Up With

After conversion, your output directory will contain:

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

## Storage And Time Expectations

- Source download from Hugging Face: about 224 GB
- Converted Flash-MoE output: about 214.5 GB
- Recommended free disk space during conversion: at least 450 GB
- Longest step: repacking the 60 expert layers into contiguous `layer_XX.bin` files

You need enough space for both the original Hugging Face snapshot and the converted output at the same time.

## Prerequisites

From the repo root, make sure you have:

```bash
xcode-select -p
python3 --version
```

Install the small Python dependencies used by the setup scripts:

```bash
python3 -m pip install --upgrade numpy huggingface_hub
```

Install the Hugging Face CLI if you do not already have it:

```bash
curl -LsSf https://hf.co/cli/install.sh | bash
hf --help
```

If you hit timeouts on a slow connection, increasing the Hub timeout can help:

```bash
export HF_HUB_DOWNLOAD_TIMEOUT=30
```

## Step 1: Choose Your Source And Output Directories

Pick one directory for the original MLX download and one for the converted Flash-MoE export:

```bash
MODEL=$HOME/Models/mlx-community-Qwen3.5-397B-A17B-4bit
OUT=$HOME/Models/flash_mlx_4bit

mkdir -p "$MODEL" "$OUT"
```

`MODEL` is the untouched Hugging Face snapshot. `OUT` is the final directory you will pass to `--model`.

## Step 2: Download The MLX Model

Download the full `mlx-community/Qwen3.5-397B-A17B-4bit` repository into your chosen source directory:

```bash
hf download mlx-community/Qwen3.5-397B-A17B-4bit --local-dir "$MODEL"
```

When the download finishes, you should see files like these under `"$MODEL"`:

```text
config.json
model.safetensors.index.json
model-00001-of-00046.safetensors
...
model-00046-of-00046.safetensors
tokenizer.json
tokenizer_config.json
```

## Step 3: Point `expert_index.json` At Your Download

`repack_experts.py` reads the source model path from `expert_index.json`. If your model is not stored at the path already checked into this repo, update it before repacking.

You can edit the file manually, or use this command from the repo root:

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

Quick sanity check:

```bash
python3 repack_experts.py --dry-run
```

If the path is correct, the script should print the model path, validate component sizes, and report that it would write all 60 packed layers.

## Step 4: Repack The Expert Weights

This converts the scattered expert tensors from the MLX safetensors shards into contiguous per-layer binaries:

```bash
python3 repack_experts.py --output "$OUT/packed_experts"
```

What this produces:

- `packed_experts/layer_00.bin` through `packed_experts/layer_59.bin`
- `packed_experts/layout.json`

Optional verification for one layer after the repack:

```bash
python3 repack_experts.py --verify-only 0 --output "$OUT/packed_experts"
```

Useful partial-run examples:

```bash
python3 repack_experts.py --layers 0-4 --output "$OUT/packed_experts"
python3 repack_experts.py --layers 5-9 --output "$OUT/packed_experts"
```

## Step 5: Extract Non-Expert Weights

This script builds the `model_weights.bin` and `model_weights.json` files used by the C inference engine. It skips the vision tower and skips routed expert tensors by default.

```bash
python3 metal_infer/extract_weights.py --model "$MODEL" --output "$OUT"
```

Expected outputs:

```text
$OUT/model_weights.bin
$OUT/model_weights.json
```

## Step 6: Export `vocab.bin`

This creates the token ID to decoded-string table used for output text decoding:

```bash
python3 metal_infer/export_vocab.py "$MODEL/tokenizer.json" "$OUT/vocab.bin"
```

Expected output:

```text
$OUT/vocab.bin
```

## Step 7: Export `tokenizer.bin`

This creates the compact tokenizer format used for prompt encoding:

```bash
python3 metal_infer/export_tokenizer.py "$MODEL/tokenizer.json" "$OUT/tokenizer.bin"
```

Expected output:

```text
$OUT/tokenizer.bin
```

## Step 8: Build The Inference Binary

Build the inference engine from the repo root:

```bash
make -C metal_infer infer
```

If you also want the interactive chat client:

```bash
make -C metal_infer chat
```

## Step 9: Smoke Test The Converted Model

Run a short generation test against the converted directory:

```bash
./metal_infer/infer --model "$OUT" --prompt "Hello" --tokens 16 --stream
```

For a timing breakdown:

```bash
./metal_infer/infer --model "$OUT" --prompt "Hello" --tokens 16 --timing
```

## Full Copy-Paste Pipeline

From the repo root:

```bash
python3 -m pip install --upgrade numpy huggingface_hub
curl -LsSf https://hf.co/cli/install.sh | bash

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
  One or more `.safetensors` shards are missing from the Hugging Face download.

- `repack_experts.py --dry-run` points at the wrong directory
  Update `expert_index.json` again and re-run the dry run before the full repack.

- `infer` cannot find `model_weights.bin`, `model_weights.json`, `vocab.bin`, or `tokenizer.bin`
  Make sure all four files are directly under `"$OUT"` and `packed_experts/` is inside `"$OUT"`.

- You want a single self-contained runtime directory
  Use the converted `OUT` directory with `--model`. That is the directory the engine is designed to consume.
