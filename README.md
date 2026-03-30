# Flash-MoE: Qwen3.5-397B-A17B on Apple Silicon

> **ANEMLL fork** — Work in Progress. Primarily targeting **Apple M5 Max 128GB**.
>
> Forked from [danveloper/flash-moe](https://github.com/danveloper/flash-moe) — **[Read the original paper](paper/flash_moe.pdf)** for the full story of how an AI and a human built this in 24 hours.
>
> **[Read our paper →](https://github.com/iluvclubs/flash-moe/releases/tag/v1.0)** — Beyond the DRAM Wall: 20.34 tok/s on M5 Max, 4.67× over baseline.

Pure C/Metal inference engine that runs **Qwen3.5-397B-A17B** (a 397 billion parameter Mixture-of-Experts model) on Apple Silicon. The entire model streams from SSD through a custom Metal compute pipeline. Features high-quality [Unsloth](https://github.com/unslothai/unsloth) Q3 expert quantization with optimized IQ3_XXS/IQ4_XS/Q5_K dequant kernels (llama.cpp GGUF-compatible), Metal 4 NAX tensor matmul support (M5+), and improved SSD cache throughput via page-aligned pread fanout (`--cache-io-split`, adapted from [ncdrone/rustane](https://github.com/ncdrone/rustane)). No Python. No frameworks. Just C, Objective-C, and hand-tuned Metal shaders.

---

## M5 Max Autoresearch Results (March 2026)

> **43 experiments · ~24 hours · 4.67× improvement over the M3 Max baseline**
> **[Download paper (PDF)](https://github.com/iluvclubs/flash-moe/releases/tag/v1.0)**

Using a Claude Code autoresearch loop — identical methodology to [danveloper/flash-moe](https://github.com/danveloper/flash-moe) — we ran 43 Metal optimization experiments on M5 Max 128GB and reached **20.34 tok/s**, breaking the theoretical 2-bit I/O ceiling (18.6 tok/s) using 4-bit weights.

### Optimization Trajectory

| Experiment | Optimization | tok/s | Δ% |
|---|---|---|---|
| Baseline (Exp1) | M5 Max 4-bit, cache-io-split 4 | 12.5 | — |
| Exp9 | **Temporal expert prediction** | 16.4 | **+31%** |
| Exp16 | Q3 GGUF experts + temporal | 18.7 | +14% |
| Exp27 | CMD2 pre-encode (GDN layers) | 19.1 | +2.7% |
| Exp41 | Horizontal Q/K/V projection fusion | 19.9 | +4.0% |
| **Exp42** | **CMD2 pre-encode (all 60 layers)** | **20.34** | **+2.4%** |

### Key Finding: Beating the Hardware Ceiling

The theoretical SSD I/O limit for 2-bit experts is **18.6 tok/s** (943 MB/token ÷ 17.5 GB/s).
We achieved **20.34 tok/s with 4-bit experts** — faster than the 2-bit physical limit — by overlapping SSD reads with GPU compute via temporal expert prediction.

**Temporal expert prediction** was the single biggest win: exploiting 27% cross-token routing correlation to pre-fetch the next layer's experts while the GPU is still computing the current one. This alone contributed **+55% decode speed** (10.61 → 16.40 tok/s in isolation, Exp9).

### vs. Baseline

| Machine | Best decode | vs. M3 Max |
|---------|-------------|------------|
| M3 Max 48GB — Dan Woods ([danveloper/flash-moe](https://github.com/danveloper/flash-moe)) | 4.36 tok/s | baseline |
| M5 Max 128GB — Anemll fork (this repo, pre-autoresearch) | 12.9 tok/s | 3.0× |
| **M5 Max 128GB — autoresearch/mar25 (this repo)** | **20.34 tok/s** | **4.67×** |

### Best Inference Command

```bash
./metal_infer/infer \
  --model ~/Models/flash_mlx_4bit \
  --q3-experts \
  --predict \
  --cache-io-split 4 \
  --gguf-embedding ~/Models/flash_mlx_4bit/gguf/embedding_q8_0.bin \
  --gguf-lm-head ~/Models/flash_mlx_4bit/gguf/lm_head_q6.bin \
  --prompt "<|im_start|>user\nYour prompt<|im_end|>\n<|im_start|>assistant\n" \
  --tokens 200 --timing
```

### Failed Experiments (43 total, ~30% keep rate)

- **1-bit QJL (SRHT):** PPL catastrophic (5647) — too lossy for generative LLM
- **Ternary 2-bit:** PPL failed (11.49) — 84% sparsity too aggressive
- **K=3 experts:** PPL failed (6.54) — model requires K=4 minimum
- **Cross-layer prediction:** near-zero hit rate
- **Encoder fusion:** zero improvement (encoder overhead only ~1μs)
- **Shared memory optimizations:** hurt occupancy — M5 Max L2 handles it automatically

---

## Anemll Fork: Q3 GGUF + Hybrid Quantization

> The following sections document the Anemll fork's Q3 GGUF hybrid quantization work,
> which provides the foundation the autoresearch experiments built on.

## What's New (This Fork)

### Performance (M5 Max 128GB)

| Config | Decode | PPL | Expert I/O | Expert size |
|--------|--------|-----|------------|-------------|
| **Q3 GGUF + cache-io-split 4** (recommended) | **12.9 tok/s** | **3.81** | 27 ms/tok | 163 GB |
| 4-bit MLX experts | 9.5 tok/s | **3.64** | 45 ms/tok | 209 GB |
| 2-bit MLX experts | **14.5 tok/s** | 5.71 | 21 ms/tok | 120 GB |

| Machine | Decode tok/s | vs Original |
|---------|-------------|-------------|
| M3 Max 48GB (original) | 4.36 | baseline |
| M5 Max 128GB (this fork) | 12.9 | 3.0× |
| **M5 Max 128GB (autoresearch/mar25)** | **20.34** | **4.67×** |

### Perplexity (WikiText-2, 2000 tokens)

| Configuration | PPL | CE (nats) | Decode | Expert size |
|--------------|-----|-----------|--------|-------------|
| **Q3 GGUF experts** (recommended) | **3.81** | — | **12.9 tok/s** | 163 GB |
| 4-bit MLX + GGUF Q6 LM head | **3.62** | 1.286 | 9.5 tok/s | 209 GB |
| 4-bit MLX experts | **3.64** | 1.292 | 9.5 tok/s | 209 GB |
| Full GGUF resident stack* | **3.49** | 1.250 | 5.1 tok/s | 209 GB |
| 2-bit MLX experts | 5.71 | 1.742 | 14.5 tok/s | 120 GB |

*Full GGUF resident stack = Q8_0 embedding + Q6_K LM head + Q8_0 attention overlays (4-bit experts). Best PPL (3.49) but 2x slower decode — the Q8_0 attention weights are 2x larger, doubling GPU memory bandwidth for 54% of per-token time.

Q3 GGUF is the best speed/quality tradeoff: near-4-bit quality (PPL 3.81 vs 3.64) at **36% faster decode** (12.9 vs 9.5 tok/s). GGUF Q6 LM head improves PPL for free (1.4ms/tok, 2% of decode time). 2-bit is fastest but PPL degrades 57%. See [Perplexity Evaluation](#perplexity-evaluation) for how to run.

### Per-Token Decode Breakdown (M5 Max, 4-bit)

```
Component                    ms/layer   MB/layer   Layers   ms/tok    %
────────────────────────────────────────────────────────────────────────
GDN: QKV+gate proj            0.43       1.1 MB     45     20.7    22%
GDN: BLAS recurrence           0.02       —          45      0.9     1%
Full: QKV proj                 0.43       1.0 MB     15      6.9     7%
CMD2: o_proj+norm+routing      0.29       0.5 MB     60     17.5    18%
Expert I/O (SSD)               0.74      27.0 MB     60     44.7    47%
Expert compute (CMD3)          0.02       —           60      1.2     1%
LM head                         —         2.0 MB      1      1.5     2%
────────────────────────────────────────────────────────────────────────
Total                                                       95.9   100%
Per token: ~96 ms, ~1.7 GB read (1.62 GB experts + 89 MB dense)
```

**With Q3 GGUF experts + cache-io-split 4 (recommended):**

```
Component                    ms/layer   MB/layer   Layers   ms/tok    %
────────────────────────────────────────────────────────────────────────
GDN: QKV+gate proj            0.46       1.1 MB     45     20.7    27%
CMD2: o_proj+norm+routing      0.30       0.5 MB     60     18.8    24%
Expert I/O (SSD)               0.44      21.8 MB     60     26.7    35%
Expert compute (CMD3)          0.02       —           60      1.3     2%
LM head                         —         2.0 MB      1      1.4     2%
────────────────────────────────────────────────────────────────────────
Total                                                       76.8   100%
Per token: ~77 ms, ~1.3 GB read (1.31 GB Q3 experts + 89 MB dense) = 12.9 tok/s
```

### Recommended Configuration

For best speed + quality balance, we recommend Q3 GGUF experts with GGUF embedding and LM head:

```bash
./metal_infer/infer \
  --model ~/Models/flash_mlx_4bit \
  --q3-experts \
  --gguf-embedding ~/Models/flash_mlx_4bit/gguf/embedding_q8_0.bin \
  --gguf-lm-head ~/Models/flash_mlx_4bit/gguf/lm_head_q6.bin \
  --prompt "What is Apple Neural Engine?" \
  --tokens 140 --stream --cache-io-split 4
```

- **`--q3-experts`** — Unsloth GGUF 3-bit experts (IQ3_XXS/IQ4_XS, 23% smaller, ~36% faster)
- **`--gguf-lm-head`** — Unsloth Q6_K LM head (better quality, negligible speed impact)
- **`--gguf-embedding`** — Unsloth Q8_0 embedding (better quality, negligible speed impact)
- **`--cache-io-split 4`** — Page-aligned pread fanout (improved SSD throughput)

See [docs/model-download-and-convert.md](docs/model-download-and-convert.md) for full download and conversion guide.

### WIP: Hybrid MLX + GGUF/Unsloth Quantization

Mixed-source quantization for **Qwen3.5-397B-A17B** combining:
- **MLX 4-bit** ([mlx-community/Qwen3.5-397B-A17B-4bit](https://huggingface.co/mlx-community/Qwen3.5-397B-A17B-4bit)) for dense/attention weights
- **Unsloth GGUF 3-bit** ([unsloth/Qwen3.5-397B-A17B-GGUF](https://huggingface.co/unsloth/Qwen3.5-397B-A17B-GGUF) `UD-Q3_K_XL`) for routed and shared experts

**Why Unsloth 3-bit?** Unsloth's dynamic quantization (IQ3_XXS/IQ4_XS) is not uniform 3-bit — it's a high-quality mixed-precision format that selectively uses higher precision where it matters most. Layer 27 is the key example: Unsloth keeps attention weights at BF16, shared experts at BF16, and down_proj routed experts at Q5_K (5-bit), while other layers use IQ3_XXS/IQ4_XS. This selective precision preserves output quality while reducing expert I/O by ~23% vs uniform 4-bit.

| Expert format | Expert size | I/O per token | Decode (M5 Max) |
|--------------|-------------|---------------|-----------------|
| MLX 4-bit | 6.75 MB | 1.62 GB | 9.5 tok/s |
| **Unsloth Q3** | **5.44 MB** | **1.30 GB** | **11.2 tok/s** |
| MLX 2-bit | 3.75 MB | 0.90 GB | 14.5 tok/s |

Current status:
- Q3 routed experts fully working: **12.9 tok/s** with `--cache-io-split 4` (36% faster than 4-bit)
- IQ3_XXS/IQ4_XS/Q5_K GPU dequant kernels implemented (vendored from llama.cpp)
- GGUF metadata-only tensor sweep across all 5 shards ([docs/gguf-q3-tensor-sweep.md](docs/gguf-q3-tensor-sweep.md))
- Q6_K LM head and Q8_0 embedding integrated (quality gain, no speed impact)
- Q8_0 full attention and linear attention overlays integrated (quality gain, ~2x decode overhead)
- Full GGUF resident stack measured: PPL **3.49** (best quality) at 5.1 tok/s ([docs/gguf-hybrid-bringup-log.md](docs/gguf-hybrid-bringup-log.md))
- Layer 27 outliers handled (BF16 attention, Q5_K experts) — Unsloth's selective precision
- Per-layer mixed quantization support (`--skip-layers`)
- Shared expert GGUF conversion not yet scripted (stays MLX 4-bit for now)

**GGUF resident tensors (dense/persistent weights):**

The Unsloth GGUF also provides higher-quality dense tensors that can replace MLX 4-bit originals. Current integration status:

| Tensor family | GGUF format | Status | Impact |
|--------------|------------|--------|--------|
| LM head | Q6_K | Integrated | Minimal speed impact (1.4ms/tok), quality improvement |
| Embedding | Q8_0 | Integrated | Quality improvement |
| Full attention Q/K/V/O | Q8_0 | Integrated | Slower (~2x dense matmul), better quality |
| Linear attention QKV | Q8_0 | Integrated | Slower (~2x dense matmul), better quality |
| Shared experts | Q8_0 / BF16 (L27) | Not yet scripted | Stays MLX 4-bit for now |

**Note:** The Q8_0 dense/attention overlays improve quality but slow down decode significantly — the dense matmuls (CMD1+CMD2) are 54% of per-token time, and Q8_0 tensors are 2x the size of 4-bit, doubling GPU memory bandwidth for those operations. The LM head (Q6_K) is the sweet spot: only 2% of per-token time, so the quality gain is essentially free. See [docs/q3-vs-4bit-clean-timing-report.md](docs/q3-vs-4bit-clean-timing-report.md) for detailed comparisons.

### New Features

- **`--ppl`** — Perplexity measurement on ground truth tokens
- **`--stream`** — Clean streaming output (tokens only, no debug)
- **`--timing`** — Per-token decode breakdown (GDN vs full attention, expert I/O, LM head)
- **`--2bit`** — 2-bit expert quantization (44% smaller, +27% faster, PPL 5.71 — suitable for short tool calls but not long-form coding/reasoning)
- **`--nax`** — NAX tensor matmul (Metal 4 / M5+, experimental)
- **`--skip-layers`** — Per-layer mixed quantization (keep sensitive layers at 4-bit)
- **`--cache-io-split N`** — Fan out expert preads across page-aligned chunks
- **Self-contained `--model` path** — All files auto-resolved from one directory
- **`export_vocab.py`** — Generate vocab.bin from tokenizer.json
- **`prepare_ppl_tokens.py`** — Tokenize WikiText-2 for PPL evaluation

### M5 Max: NAX (Metal 4 Neural Accelerator)

Apple M5 chips include NAX — dedicated matrix multiply hardware inside the GPU, accessible via MetalPerformancePrimitives tensor API. We built working NAX kernels ([nax_gemm.metal](metal_infer/nax_gemm.metal)) and benchmarked them:

- LM head GEMM in isolation: **4.5x** speedup (7ms vs 32ms in sandbox)
- Single-token decode (M=1): FMA kernel already 1.4ms on M5 Max native — NAX adds tile padding overhead, no net gain
- NAX will benefit **batched prefill** (M>1) when implemented — ready via `--nax` flag
- Metal 4.0 shader compilation, PSO creation, and cooperative tensor store all working

---

## Original Design

> The following sections describe the original Flash-MoE architecture from [danveloper/flash-moe](https://github.com/danveloper/flash-moe).

![Progress](progress.png)

### Original Results (M3 Max)

| Configuration | tok/s | Quality | Notes |
|--------------|-------|---------|-------|
| 4-bit experts, FMA kernel | **4.36** | Excellent | Full tool calling. 209GB on disk. |
| 4-bit experts, baseline | 3.90 | Excellent | Before FMA kernel optimization. |
| 2-bit experts, trust OS | 5.74 | Good* | 120GB on disk. *Breaks JSON/tool calling. |
| 2-bit peak single token | 7.05 | Good* | Warm cache burst. *Not suitable for tool use. |

### Hardware (Original)

- **Machine**: MacBook Pro, Apple M3 Max
- **Chip**: 16-core CPU (12P + 4E), 40-core GPU, 16-core ANE
- **Memory**: 48 GB unified (~400 GB/s bandwidth)
- **SSD**: 1TB Apple Fabric, **17.5 GB/s sequential read** (measured)

## Architecture

The model has 60 transformer layers: 45 GatedDeltaNet (linear attention) + 15 standard full attention. Each layer has 512 experts, of which K=4 are activated per token (plus one shared expert). Hidden dimension is 4096.

### Key Techniques

1. **SSD Expert Streaming** — Expert weights (209GB at 4-bit) are read from NVMe SSD on demand via parallel `pread()` with GCD dispatch groups. Only the K=4 active experts per layer are loaded (~6.75MB each). The OS page cache manages caching — no custom cache needed ("Trust the OS" principle). Inspired by Apple's "LLM in a Flash" paper.

2. **FMA-Optimized Dequant Kernel** — The inner loop of the 4-bit dequantized matrix-vector multiply rearranges the math from `(nibble * scale + bias) * x` to `fma(nibble, scale*x, bias*x)`. Pre-computing `scale*x` and `bias*x` lets the GPU fused multiply-add unit do dequant+multiply in one instruction. 12% faster than the naive formulation.

3. **Metal Compute Shaders** — Hand-written Metal kernels for:
   - 4-bit and 2-bit dequantized matrix-vector multiply (tiled, SIMD-reduced, shared input cache, FMA-optimized)
   - Fused SwiGLU activation
   - RMS normalization (two-pass: sum-of-squares reduction + apply)
   - Batched GPU attention (Q@K^T, softmax, scores@V) for full attention layers
   - GPU RoPE (fused with Q deinterleave and K normalization)
   - MoE combine + residual + sigmoid gate (fused kernel)

4. **Deferred GPU Expert Compute** — CMD3 (expert forward pass) is submitted without waiting. The GPU executes it while the CPU prepares the next layer. The combine + residual + norm are also on GPU, feeding directly into the next layer's attention projections.

5. **Accelerate BLAS for Linear Attention** — The GatedDeltaNet recurrence uses `cblas_sscal`, `cblas_sgemv`, and `cblas_sger` for the 64-head x 128x128 state matrix update. 64% faster than scalar code.

6. **Trust the OS** — No custom expert cache. The OS page cache (~35GB) manages expert data caching via standard LRU. Every custom caching approach we tested (Metal LRU, malloc cache, LZ4 compressed cache) was slower due to GPU memory pressure or overhead. The page cache achieves ~71% hit rate naturally.

### Pipeline Per Layer

```
CMD3(prev) -> CMD1: attention projections + delta-net  [0.46ms GPU on M5 Max]
           -> CPU: flush results                       [0.01ms CPU]
           -> CMD2: o_proj + norm + routing + shared    [0.29ms GPU]
           -> CPU: softmax + topK routing               [0.003ms]
           -> I/O: parallel pread K=4 experts           [0.74ms SSD]
           -> CMD3: expert forward + combine + norm     [0.02ms encode, DEFERRED]
```

### Memory Budget

```
Component                          Size
Dense weights (mmap'd)           5,520 MB   5.52 GB
Delta-net state (45 layers)        189 MB   0.19 GB  (fixed, O(1) — no growth with context)
Expert data buffers (8x2)          113 MB   (4-bit) / 63 MB (2-bit)
Attention + logits + scratch         3 MB
Total active RAM                ~5.8 GB
Remaining for page cache        ~42 GB    (of 48 GB)
```

### Context Length Scaling

GatedDeltaNet replaces KV cache for 45/60 layers with fixed O(1) state. Only 15 full attention layers need KV cache (2 KV heads x 256 dim each):

```
Context    KV cache   Total RAM   Page cache remaining
8K          0.3 GB     6.1 GB     41.9 GB
128K        4.0 GB     9.9 GB     38.1 GB
256K        8.1 GB    13.9 GB     34.1 GB
512K       16.1 GB    21.9 GB     26.1 GB
```

## Quick Start

```bash
cd metal_infer
make

# Recommended: Q3 GGUF experts + GGUF embedding/LM head + cache-io-split
./infer --model ~/Models/flash_mlx_4bit \
  --q3-experts \
  --gguf-embedding ~/Models/flash_mlx_4bit/gguf/embedding_q8_0.bin \
  --gguf-lm-head ~/Models/flash_mlx_4bit/gguf/lm_head_q6.bin \
  --cache-io-split 4 \
  --prompt "Explain quantum computing" --tokens 100 --stream

# 4-bit MLX experts (baseline)
./infer --model ~/Models/flash_mlx_4bit --prompt "Explain quantum computing" --tokens 100 --stream

# 2-bit experts (fastest, lower quality)
./infer --model ~/Models/flash_mlx_4bit --prompt "Hello" --tokens 100 --stream --2bit

# Per-layer timing breakdown
./infer --model ~/Models/flash_mlx_4bit --prompt "Hello" --tokens 20 --timing

# Interactive chat with tool calling
./chat
```

### Perplexity Evaluation

```bash
# Prepare ground truth tokens (WikiText-2)
python3 metal_infer/prepare_ppl_tokens.py \
  --tokenizer ~/Models/mlx-community-Qwen3.5-397B-A17B-4bit/tokenizer.json \
  --max-tokens 2000 -o ppl_tokens_2k.bin

# 4-bit PPL
./metal_infer/infer --model ~/Models/flash_mlx_4bit --ppl ppl_tokens_2k.bin

# 2-bit PPL
./metal_infer/infer --model ~/Models/flash_mlx_4bit --ppl ppl_tokens_2k.bin --2bit

# Q3 GGUF experts PPL
./metal_infer/infer --model ~/Models/flash_mlx_4bit --q3-experts --ppl ppl_tokens_2k.bin
```

### Model Downloads

Two source models from HuggingFace:

| Model | Size | Purpose |
|-------|------|---------|
| [mlx-community/Qwen3.5-397B-A17B-4bit](https://huggingface.co/mlx-community/Qwen3.5-397B-A17B-4bit) | ~224 GB | Dense weights, base 4-bit experts, tokenizer |
| [unsloth/Qwen3.5-397B-A17B-GGUF](https://huggingface.co/unsloth/Qwen3.5-397B-A17B-GGUF) (UD-Q3_K_XL) | ~166 GB | Q3 experts, Q6_K LM head, Q8_0 embedding |

```bash
# Install HuggingFace CLI
pip install huggingface_hub
# or: curl -LsSf https://hf.co/cli/install.sh | bash

# 1. MLX 4-bit model (required)
hf download mlx-community/Qwen3.5-397B-A17B-4bit \
  --local-dir ~/Models/mlx-community-Qwen3.5-397B-A17B-4bit

# 2. Unsloth GGUF Q3 (recommended for best speed/quality)
hf download unsloth/Qwen3.5-397B-A17B-GGUF \
  --include "Qwen3.5-397B-A17B-UD-Q3_K_XL-*" \
  --local-dir ~/Models/Qwen3.5/Qwen3.5-397B-A17B-GGUF-UD-Q3_K_XL
```

### Conversion Step 1: MLX Base (Required)

This creates the base runtime from the MLX 4-bit model:

```bash
MODEL=~/Models/mlx-community-Qwen3.5-397B-A17B-4bit
OUT=~/Models/flash_mlx_4bit
mkdir -p $OUT

# Extract dense weights (~5.5 GB, ~2 min)
python3 metal_infer/extract_weights.py --model $MODEL --output $OUT

# Export vocabulary and tokenizer
python3 metal_infer/export_vocab.py $MODEL/tokenizer.json $OUT/vocab.bin
python3 metal_infer/export_tokenizer.py $MODEL/tokenizer.json $OUT/tokenizer.bin

# Repack 4-bit experts (~209 GB, ~25 min)
python3 repack_experts.py --output $OUT/packed_experts

# Build and test
make -C metal_infer infer
./metal_infer/infer --model $OUT --prompt "Hello" --tokens 20 --stream
```

**Optional: 2-bit experts** (44% smaller, faster but PPL degrades from 3.64 to 5.71):

```bash
python3 metal_infer/repack_experts_2bit.py --model $OUT --output $OUT/packed_experts_2bit
./metal_infer/infer --model $OUT --prompt "Hello" --tokens 20 --stream --2bit
```

### Conversion Step 2: GGUF Overlays (Recommended)

This extracts high-quality Unsloth tensors to overlay on the MLX base. Minimum recommended: **Q3 experts + LM head + embedding**.

```bash
GGUF=~/Models/Qwen3.5/Qwen3.5-397B-A17B-GGUF-UD-Q3_K_XL/Qwen3.5-397B-A17B-UD-Q3_K_XL-00001-of-00005.gguf
OUT=~/Models/flash_mlx_4bit
mkdir -p $OUT/gguf

# Q3 streamed experts (~163 GB, ~20 min)
python3 autoresearch/repack_experts_q3.py \
  --model $OUT --gguf $GGUF \
  --llama-cpp-root ~/SourceRelease/GITHUB/ML_playground/llama.cpp \
  --output $OUT/packed_experts_Q3 --layers all --include-outlier-layer

# Q6_K LM head (better quality, negligible speed impact)
python3 autoresearch/extract_gguf_lm_head.py \
  --gguf $GGUF \
  --llama-cpp-root ~/SourceRelease/GITHUB/ML_playground/llama.cpp \
  --out-bin $OUT/gguf/lm_head_q6.bin --out-json $OUT/gguf/lm_head_q6.json

# Q8_0 embedding (better quality)
python3 autoresearch/extract_gguf_embedding.py \
  --gguf $GGUF \
  --llama-cpp-root ~/SourceRelease/GITHUB/ML_playground/llama.cpp \
  --out-bin $OUT/gguf/embedding_q8_0.bin --out-json $OUT/gguf/embedding_q8_0.json

# Test recommended configuration
./metal_infer/infer --model $OUT \
  --q3-experts \
  --gguf-embedding $OUT/gguf/embedding_q8_0.bin \
  --gguf-lm-head $OUT/gguf/lm_head_q6.bin \
  --cache-io-split 4 \
  --prompt "What is Apple Neural Engine?" --tokens 100 --stream
```

See [docs/model-download-and-convert.md](docs/model-download-and-convert.md) for the complete guide including optional Q8_0 attention overlays (Steps 9-14).

### Model Directory Structure (`~/Models/flash_mlx_4bit/`)

```
flash_mlx_4bit/
  model_weights.bin          # Non-expert weights (5.5 GB, mmap'd) — from MLX
  model_weights.json         # Tensor manifest
  vocab.bin                  # Vocabulary for token decoding
  tokenizer.bin              # BPE tokenizer for prompt encoding
  packed_experts/            # MLX 4-bit experts (209 GB, 60 × 3.4 GB)
    layout.json
    layer_00.bin ... layer_59.bin
  packed_experts_2bit/       # Optional: 2-bit experts (120 GB, 60 × 1.9 GB)
    layer_00.bin ... layer_59.bin
  packed_experts_Q3/         # Unsloth GGUF Q3 experts (163 GB, 60 × 2.7 GB)
    layout.json
    layer_00.bin ... layer_59.bin
  gguf/                      # Unsloth GGUF resident overlays
    embedding_q8_0.bin       # Q8_0 embedding (higher quality)
    lm_head_q6.bin           # Q6_K LM head (higher quality, free speed)
    lm_head_q6.json
    attn_qkv_q8_0.bin        # Q8_0 attention projections (optional, slower)
    full_attn_q8_0.bin       # Q8_0 full attention weights (optional, slower)
    ...
```

### Mixed Q3/Q4 Quantization Explained

The engine uses a **hybrid quantization** strategy — different bit widths for different weight families, mixing MLX and GGUF sources:

```
┌─────────────────────────────────────────────────────────────────┐
│ Weight Family         │ Source  │ Format       │ Size    │ Notes│
├───────────────────────┼────────┼──────────────┼─────────┼──────┤
│ Routed experts (512)  │ GGUF   │ IQ3_XXS/IQ4  │ 163 GB  │ SSD  │
│   gate_proj, up_proj  │        │ IQ3_XXS      │         │      │
│   down_proj           │        │ IQ4_XS       │         │      │
│   layer 27 (outlier)  │        │ IQ4_XS/Q5_K  │         │      │
├───────────────────────┼────────┼──────────────┼─────────┼──────┤
│ Shared expert         │ MLX    │ 4-bit        │ 0.4 GB  │ GPU  │
│ Attention (QKV, O)    │ MLX    │ 4-bit        │ 3.9 GB  │ GPU  │
│ Embedding             │ GGUF   │ Q8_0         │ 0.6 GB  │ GPU  │
│ LM head               │ GGUF   │ Q6_K         │ 0.6 GB  │ GPU  │
│ Norms, gates          │ MLX    │ BF16/F32     │ 0.1 GB  │ GPU  │
└─────────────────────────────────────────────────────────────────┘

Why this mix?
- Routed experts: Q3 GGUF saves 23% I/O vs 4-bit (SSD-bound, biggest win)
- LM head: Q6_K from GGUF improves quality for free (only 2% of decode time)
- Embedding: Q8_0 from GGUF improves first-token quality
- Dense attention: stays MLX 4-bit (Q8_0 would be 2x slower, 54% of decode time)
- Shared expert: stays MLX 4-bit (GGUF shared conversion WIP)
```

Unsloth's GGUF is not uniform 3-bit — it uses **importance-aware mixed precision** (IQ3_XXS for less sensitive weights, IQ4_XS/Q5_K for more sensitive ones). Layer 27 is a key outlier where Unsloth keeps attention at BF16 and down_proj experts at Q5_K, preserving quality where the model is most sensitive.

## Project Structure

```
metal_infer/
  infer.m              # Complete inference engine (~7500 lines)
  shaders.metal        # Metal compute kernels (~1300 lines)
  nax_gemm.metal       # NAX tensor matmul kernels (Metal 4.0, M5+)
  nax_bench.m/.metal   # NAX standalone benchmark
  chat.m               # Interactive chat TUI with tool calling
  tokenizer.h          # C BPE tokenizer (single-header, 449 lines)
  Makefile             # Build system
  extract_weights.py   # Creates model_weights.bin from safetensors
  export_vocab.py      # Creates vocab.bin from tokenizer.json
  export_tokenizer.py  # Creates tokenizer.bin from tokenizer.json
  repack_experts_2bit.py  # 4-bit -> 2-bit expert requantization
  prepare_ppl_tokens.py   # Tokenize text for perplexity evaluation

repack_experts.py      # 4-bit expert packing from safetensors
measure_ppl.py         # MLX baseline perplexity measurement
docs/                  # Experiment logs, GGUF bringup notes, timing reports
  model-download-and-convert.md  # Full setup guide (Steps 1-14)
  gguf-q3-tensor-sweep.md        # GGUF tensor inventory
  q3-vs-4bit-clean-timing-report.md  # Q3 vs 4-bit comparison
  gguf-hybrid-bringup-log.md     # Incremental GGUF integration log
```

## What We Tried (and What Worked)

### Kept
| Approach | Result | Impact |
|----------|--------|--------|
| FMA dequant kernel | GPU compute -12% | **+12% tok/s** |
| Trust OS page cache | Deleted Metal LRU -> +38% | **Foundational** |
| GPU combine+norm in CMD3 | Eliminates CPU round-trip | **Pipeline** |
| BLAS delta-net (Accelerate) | cpu_attn 0.78->0.28ms | **+64% attn** |
| F_NOCACHE for 2-bit | +3% from avoiding page thrash | **2-bit only** |
| GPU fused attention (RoPE) | +2% for full-attn layers | **Small** |
| C BPE tokenizer | 180ms vs 3500ms startup | **20x startup** |
| Deferred CMD3 execution | GPU/CPU overlap | **Pipeline** |
| Per-layer mixed quant | Keep sensitive layers at 4-bit | **Quality** |
| GGUF Q3 streamed experts | -23% expert I/O vs 4-bit | **WIP** |
| **Temporal expert prediction** | **10.61 → 16.40 tok/s (+55%)** | **autoresearch/mar25** |

### Discarded (58+ experiments, highlights)
| Approach | Result | Why |
|----------|--------|-----|
| NAX for M=1 decode | -8% | Tile padding overhead > GEMM speedup for matvec |
| LZ4 expert compression | -13% | Decompress overhead > warm cache savings |
| F_RDADVISE prefetch | net 0% | Unified memory: SSD DMA slows GPU -73% |
| Temporal prediction (early attempt) | -18% | 25% hit rate in original impl — later fixed in autoresearch |
| MLP routing predictor | 31% accuracy | Worse than temporal baseline |
| GPU LUT dequant kernel | -2% | Indirect register access serializes |
| Spin-poll GPU wait | -23% | CPU thermal competes with GPU |
| mmap expert files | -5x | Per-page fault overhead on cold data |
| MTP speculative decoding | break-even | MoE I/O scales per-token (unlike dense) |

## Safety

This is a primary development machine. The engine explicitly controls memory:
- Non-expert weights: 5.5GB (mmap'd, read-only)
- Metal scratch buffers: ~200MB
- Total: ~6GB, leaving 42GB for OS + page cache
- No OOM risk. Expert data streams from SSD on demand.
- No custom caches. Trust the OS.
