# GorroAI Pro — Qwen3.5-397B Flash-MoE on M5 Max
## UPDATED: Full benchmark complete, HTTP wrapper phase added
## Claude Code Project Instructions

---

## Project Overview

This is a scientific benchmarking and commercial deployment project running
Qwen3.5-397B-A17B (a 397 billion parameter Mixture-of-Experts model) locally
on an Apple M5 Max MacBook Pro using the flash-moe inference engine.

**End goal:** Deploy as GorroAI Pro — a premium tier of the existing GorroAI
API (api.gorroai.com), offering frontier-level local inference with no cloud,
no data leaving the machine, HIPAA-friendly positioning.

**Immediate goal:** Complete benchmarking, publish results to r/LocalLLaMA
and X (@Saboo_Shubham_), credit prior authors, establish M5 Max as the
fastest flash-moe benchmark publicly documented.

---

## Hardware

- **Machine:** Apple MacBook Pro M5 Max (2025)
- **RAM:** 128GB unified memory
- **OS:** macOS (latest)
- **SSD:** ~1TB, ~5 GB/s sustained write, ~17.5 GB/s read bursts
- **Shell:** zsh with pyenv, Python 3.12.13

---

## Prior Art & Credits (MUST credit in all posts)

1. **Dan Woods (@danveloper)** — Original flash-moe author
   - Repo: https://github.com/danveloper/flash-moe
   - Baseline: 4.36 tok/s on M3 Max 48GB
   - Used Claude Code autoresearch pattern (90 experiments) to build it in 24h

2. **Anemll fork** — M5-optimized fork we are using
   - Repo: https://github.com/Anemll/flash-moe
   - Added: Metal 4 NAX tensor matmul (M5+), --cache-io-split, Q3 GGUF experts
   - Already documented M5 Max 128GB results in README

3. **Daniel Pacary (@danpacary)** — Ran Kimi-K2 1T params on M4 Max
   - Also used Claude Opus autoresearch, 100 experiments, 0.005 → 1.7 tok/s

4. **Allen Lee (@allenwlee)** — Independent benchmarker documenting JANG_2L
   - Useful code quality benchmarks showing Qwen3.5 quality ceiling

---

## Existing Setup (Already Done)

All paths are on the local machine, not in this repo.

```
~/flash-moe/                          # Anemll fork, cloned, built
~/Models/
  mlx-community-Qwen3.5-397B-A17B-4bit/   # 209GB MLX source model
  flash_mlx_4bit/                          # Converted runtime
    model_weights.bin                      # 5.52 GB dense weights
    model_weights.json
    vocab.bin
    tokenizer.bin
    packed_experts/                        # 202.5 GB, 60 layer bins
      layout.json
      layer_00.bin ... layer_59.bin
```

**Binary already built:**
```bash
~/flash-moe/metal_infer/infer   # main inference binary
~/flash-moe/metal_infer/chat    # chat binary (optional, build with make chat)
```

**GorroAI (existing production service):**
- llama-server serving MiniMax M2.5 230B at ~62 tok/s
- Cloudflare tunnel at api.gorroai.com
- LaunchAgents: com.gorroai.llama.plist, com.gorroai.cloudflared.plist
- Pause before running flash-moe benchmarks to free Metal/GPU resources

---

## Benchmark Results So Far

| Config | tok/s | Expert I/O ms/tok | Notes |
|--------|-------|-------------------|-------|
| M3 Max 48GB — original baseline | 4.36 | ~45 ms | Dan Woods |
| M5 Max 128GB — 4-bit, no split | 9.90 | 47.1 ms | Our run |
| M5 Max 128GB — 4-bit, cache-io-split 4 | **12.66** | 27.6 ms | Our run |
| M5 Max 128GB — 2-bit, cache-io-split 4 | ??? | ??? | NEXT STEP |

**Key finding:** `--cache-io-split 4` reduced Expert I/O from 47.1ms → 27.6ms
per token (41% reduction), confirming M5 Max SSD fanout optimization works.

**TTFT with cache-io-split 4:** 1053ms (vs 2296ms without)

---

## Immediate Next Steps

### Step 1: Generate 2-bit experts (IN PROGRESS)
```bash
cd ~/flash-moe
python3 metal_infer/repack_experts_2bit.py \
  --model ~/Models/flash_mlx_4bit \
  --output ~/Models/flash_mlx_4bit/packed_experts_2bit
```

### Step 2: Benchmark 2-bit
```bash
./metal_infer/infer \
  --model ~/Models/flash_mlx_4bit \
  --prompt "What is Apple Neural Engine?" \
  --tokens 200 --timing --cache-io-split 4 --2bit
```
Target: ~14.5 tok/s (per README). If achieved = 3.3x baseline.

### Step 3: Final benchmark table
Run each config 3x, take average. Document:
- 4-bit baseline (no split)
- 4-bit + cache-io-split 4
- 2-bit + cache-io-split 4

### Step 4: Reddit post
Target: r/LocalLLaMA (post Tuesday-Thursday morning EST)
See REDDIT_POST.md for draft template.

### Step 5: X post
Tag: @danveloper, @allenwlee, Dan Pacary
Hashtags: #LocalLLaMA #MLX #MoE #AppleSilicon

---

## Future Phase: Autoresearch Optimization

**Goal:** Use Claude Code autoresearch loop to find M5-specific Metal optimizations,
similar to how Dan Woods built the original (90 experiments, 19% hit rate).

**Method:**
1. Establish 3-run baseline for each config
2. Feed flash-moe codebase + benchmark results to Claude Code
3. Run automated experiment loop: read codebase → propose optimization →
   implement → benchmark 3x → commit if improvement, revert if not
4. Target: push past 14.5 tok/s ceiling on 2-bit

**Potential optimization targets (from timing breakdown):**
- Dense/attn CMD1: 29ms (37%) — largest single component
- Expert I/O SSD: 27.6ms (35%) — already improved with cache-io-split
- o_proj+shared CMD2: 19ms (24%)
- Expert compute CMD3: 1.3ms (1.6%) — already well optimized

**This phase becomes a separate scientific paper / follow-up Reddit post.**

---

## Future Phase: GorroAI Pro Deployment

**Architecture:**
- Separate endpoint from existing GorroAI (do NOT mix with llama-server)
- Smart routing: simple queries → MiniMax 230B (fast), complex → Qwen 397B
- Time-based fallback: Qwen 397B at night (low traffic), MiniMax during peak
- New RapidAPI listing: "GorroAI Pro — 397B Local Inference"

**Pricing angle:**
- Standard: MiniMax 230B, current pricing
- Pro: Qwen3.5-397B, premium tier, privacy-first positioning
- Target market: healthcare developers (HIPAA), legal tech, privacy-conscious devs

**Key differentiator:**
Owner is a physician → unique credibility for HIPAA-friendly local AI API pitch.

---

## Commands Reference

**Pause GorroAI before benchmarking:**
```bash
launchctl unload ~/Library/LaunchAgents/com.gorroai.llama.plist
launchctl unload ~/Library/LaunchAgents/com.gorroai.cloudflared.plist
```

**Resume GorroAI after benchmarking:**
```bash
launchctl load ~/Library/LaunchAgents/com.gorroai.llama.plist
launchctl load ~/Library/LaunchAgents/com.gorroai.cloudflared.plist
```

**Base 4-bit inference:**
```bash
cd ~/flash-moe
./metal_infer/infer --model ~/Models/flash_mlx_4bit \
  --prompt "Your prompt here" --tokens 200 --timing
```

**Optimized 4-bit (recommended):**
```bash
./metal_infer/infer --model ~/Models/flash_mlx_4bit \
  --prompt "Your prompt here" --tokens 200 --timing --cache-io-split 4
```

**2-bit (fastest, lower quality):**
```bash
./metal_infer/infer --model ~/Models/flash_mlx_4bit \
  --prompt "Your prompt here" --tokens 200 --timing --cache-io-split 4 --2bit
```

**Chat mode:**
```bash
./metal_infer/chat --model ~/Models/flash_mlx_4bit --cache-io-split 4
```

**Rebuild binary:**
```bash
cd ~/flash-moe/metal_infer && make
```

---

## Repository Structure

```
flash-moe/
  metal_infer/
    infer           # main inference binary (built)
    infer.m         # main inference source
    main.m          # Metal shader entry
    extract_weights.py
    export_vocab.py
    export_tokenizer.py
    repack_experts_2bit.py
  repack_experts.py
  expert_index.json   # points to ~/Models/mlx-community-Qwen3.5-397B-A17B-4bit
  docs/
    model-download-and-convert.md
  autoresearch/       # experiment loop infrastructure (future phase)
```

---

## Science & Ethics Notes

- This is a replication + extension study. Always credit Dan Woods and Anemll fork.
- Be transparent that Anemll fork already documented M5 Max numbers in README.
- Our contribution: independent verification + first public Reddit post with
  M5 Max results + commercial deployment pathway documentation.
- Future autoresearch phase is original contribution.
- Code quality ceiling is Qwen3.5 model-level (not runtime) per Allen Lee's benchmarks.
  Sonnet 4.6 still wins on code quality. Be honest about this in posts.

---

## CRITICAL WARNINGS (Added after painful experience)

### Do NOT run two llama-server instances simultaneously
The LaunchAgent auto-starts on reboot. If you manually start a second
llama-server, both load the 230B model and you get GPU OOM errors.
Always check first:
```bash
ps aux | grep llama-server
```
Kill all before starting a new one:
```bash
pkill -f llama-server
```

### Do NOT run flash-moe and MiniMax simultaneously
They cannot share GPU memory on 128GB. Always pause GorroAI first:
```bash
launchctl unload ~/Library/LaunchAgents/com.gorroai.llama.plist
launchctl unload ~/Library/LaunchAgents/com.gorroai.cloudflared.plist
```
Restore after benchmarking:
```bash
launchctl kickstart -k gui/$(id -u)/com.gorroai.llama
launchctl kickstart -k gui/$(id -u)/com.gorroai.cloudflared
curl http://localhost:8000/health
```

### Correct chat prompt format for Qwen3.5
Plain prompts hit EOS immediately. Always use:
```
<|im_start|>user
Your prompt here<|im_end|>
<|im_start|>assistant

```

---

## Phase 3: HTTP API Wrapper (GorroAI Pro enabler)

**Problem:** flash-moe is CLI only. No HTTP server. Cannot serve API requests.

**Goal:** Build an OpenAI-compatible HTTP wrapper around the infer binary
so flash-moe can be exposed as a proper API endpoint.

**Requirements:**
- OpenAI-compatible /v1/chat/completions endpoint
- Streaming support (SSE)
- Handle Qwen3.5 chat format internally
- Queue incoming requests (single inference at a time)
- Written in Python FastAPI or Go

**This is a Claude Code project — start a new session with this CLAUDE.md
and ask Claude Code to build the HTTP wrapper.**

---

## Phase 4: GorroAI Pro Deployment

- Cannot run both models simultaneously on one machine
- Options: sequential routing, second machine, or Mac Studio Ultra
- Target market: healthcare devs (HIPAA), legal tech, privacy-conscious devs
- Owner is a physician — unique credibility for HIPAA-friendly pitch
- New RapidAPI listing: "GorroAI Pro — 397B Local Inference"

---

## Phase 5: Apple Neural Engine (ANE) Co-processing Research

**Status:** Future research — 2-3 month project

**The insight:** ANE provides ~16 TFLOPS of FP16 compute that sits completely
idle during flash-moe inference. Current pipeline is sequential:
attention (GPU) then experts (GPU). ANE co-processing would make it parallel:
attention (ANE) simultaneously with experts (GPU).

**Architecture:**
- Attention layers = static, predictable computation = fits ANE constraints
- MoE routing + experts = dynamic top-K = must stay on GPU
- True parallelism: max(attention_time, expert_time) vs current sum
- Theoretical ceiling: could push toward 25+ tok/s from current 19+

**Why ANE works for attention but not experts:**
- ANE requires CoreML/MIL format — no dynamic control flow
- Attention has fixed computation graph per token
- MoE routing is dynamic (top-K selection changes per token) — incompatible

**What needs building:**
1. Convert attention layers to CoreML format
2. Build Metal/CoreML dispatch bridge
3. Synchronize ANE output back into GPU pipeline
4. Validate quality (PPL) matches GPU-only baseline

**Current bottleneck from autoresearch (supports this approach):**
- CMD1 (attention): 28.7ms = 37% of decode time
- Expert I/O: solved by temporal prediction (near zero)
- CMD1 + CMD2 (GPU compute) = ~85% of remaining time
- Offloading CMD1 to ANE = largest remaining single optimization

**Prior art:** Nobody has published ANE + GPU co-processing for MoE inference.
This is original research territory.

**Other future research directions (from NotebookLM analysis):**

- Self-speculative decoding: K=1 draft from same model (no second model needed)
  Software-only change, could try in autoresearch loop
- DeepSeek-V3 671B: same flash-moe methodology, new model, 37B active params
  Nobody has published M5 Max benchmarks — obvious next Reddit post
- 1-bit/ternary expert quantization: halve I/O payload again
  Autoresearch quality gate would catch regressions automatically
- Multi-device Thunderbolt: two MacBooks, 30 layers each
  Requires second M-series machine
- TurboQuant KV cache: MLX implementation already exists
  Could integrate into flash-moe for long context inference

---

## NotebookLM Research Analysis (March 25, 2026)

### Theoretical Performance Ceiling

**Hard I/O limit (2-bit experts): 18.6 tok/s**
- 2-bit experts: 943 MB data per token from SSD
- M5 Max SSD max throughput: 17.5 GB/s
- Math: 943 MB / 17.5 GB/s = 53.9ms minimum = 18.6 tok/s hard ceiling

**Key finding: We already beat the 2-bit theoretical ceiling with 4-bit**
- 4-bit + temporal prediction = 19.11 tok/s
- 2-bit theoretical ceiling = 18.6 tok/s
- Temporal prediction effectively made 4-bit faster than 2-bit's physical limit
- This is a genuinely surprising result — highlight in paper

**ANE compute ceiling: 25+ tok/s**
- Requires ANE co-processing AND 1-bit/ternary experts together
- 1-bit experts needed to bring I/O below SSD bandwidth limit
- Neither alone is sufficient

### Why 25+ tok/s Requires Both ANE + 1-bit Experts
- ANE ceiling (25+ tok/s) mathematically exceeds 2-bit I/O ceiling (18.6 tok/s)
- Therefore 1-bit/ternary quantization is a prerequisite for ANE to matter
- Research order: validate 1-bit quality first, then ANE co-processing

### Routing Assumptions and Failed Experiments
- Variable K (K=0,2,3 for middle layers): blank responses — failed
- Two-pass overlapped execution: 6.0 tok/s, broke quality — failed
- Speculative routing: 53% accuracy, cache pollution, 38% slower — failed
- Cross-layer prediction: near zero correlation — failed (Exp19 in autoresearch)
- K=3: immediate quality collapse (EOS after few words)
- K=4 is the minimum viable routing depth

### Promising Untried Strategies
- Self-speculative decoding K=1: use 397B model itself as draft
  No second model needed, fraction of I/O for draft tokens
- Correlated co-activation bundling: group frequently co-activating experts
  on SSD for larger sequential reads — initial attempt failed due to duplication
  but advanced clustering could work

### ANE Co-processing Risks (Not Yet Addressed)
1. Metal/CoreML dispatch bridge latency — unknown, could negate parallelism gains
2. PPL validation pending — ANE FP16 may differ from GPU pipeline
3. Unified memory bus contention — ANE + GPU + SSD all share same bus
   macOS memory compressor previously caused 1-2 GB/s bandwidth loss
   Simultaneous ANE/GPU/SSD could create new contention bottleneck

### Paper Narrative
The unexpected result (beating 2-bit theoretical ceiling with 4-bit via
temporal prediction) is the strongest finding. It reframes the optimization
story: instead of just "faster hardware", the algorithmic improvement
(temporal prediction) was the key insight that unlocked performance beyond
what naive hardware scaling could achieve.

---

## Theoretical Performance Roadmap (Paper Section)

### The Stacked Optimization Path to 37 tok/s

| Step | Config | tok/s | Status |
|------|--------|-------|--------|
| 0 | M3 Max 48GB baseline | 4.36 | Done |
| 1 | M5 Max + cache-io-split 4 | 12.99 | Done |
| 2 | M5 Max + Q3 + temporal prediction | 19.11 | Done |
| 3 | M5 Max + 1-bit + ANE co-processing | 25+ | Theoretical |
| 4 | Dual M5 Max Thunderbolt + all above | 37+ | Theoretical |

### Key Narrative: Hardware Limits Are Not Fixed

The temporal prediction result is the central insight of the paper.
Naive hardware analysis predicts a hard 18.6 tok/s I/O ceiling for 2-bit experts.
Yet 4-bit + temporal prediction achieved 19.11 tok/s — breaking the theoretical limit.

This proves that co-designing algorithms with hardware can bypass apparent physical limits.
The same principle applies at every level of the roadmap.

### Requirements for Each Step

**Step 3 (25+ tok/s) requires simultaneously:**
- 1-bit/ternary quantization (quality gate unknown)
- ANE co-processing without memory bus contention (unvalidated)
- Metal/CoreML dispatch bridge with acceptable latency (unbuilt)

**Step 4 (37+ tok/s) requires additionally:**
- Two M5 Max MacBooks connected via Thunderbolt
- Apple Fabric bus latency characterization (completely unknown)
- Layer splitting across devices without hidden state sync overhead
- All Step 3 requirements working simultaneously

### Thunderbolt Architecture Details
- Split 60 layers evenly: 30 layers per device
- Each device handles half the compute bottleneck
- Apple's unified memory model extends across Thunderbolt in theory
- Inter-device hidden state transfer latency is uncharacterized
- No published benchmarks for this configuration exist

### Paper Framing
Title: "Beyond the DRAM Wall: ..."
Central claim: algorithmic innovation (temporal prediction) broke the
theoretical hardware ceiling — proving that naive hardware limits are
not the true performance ceiling when algorithms and hardware are
co-designed.

---

## M5 Max GPU Architecture Findings (Previously Unpublished)

Discovered through 32+ autoresearch experiments. These characteristics are not
documented anywhere in Apple's public documentation or academic literature.

### GPU Scheduling
- Encoder boundaries are REQUIRED for hardware pipelining
- Monolithic single-encoder approach HURTS performance
- M5 Max GPU uses encoder boundaries to speculate between dispatches
- Encoder overhead: ~1μs per pair (not 5-10μs as previously assumed)
- Aggressive encoder fusion is unnecessary and counterproductive

### Caching
- M5 Max L2 cache automatically caches vectors of at least 28KB
- Manual threadgroup shared memory loading is counterproductive
- Allocating 14KB threadgroup memory drops GPU to 50% occupancy
- Allocating 32KB threadgroup memory hits hard Metal limits
- Let the L2 cache do its job — don't fight it

### Occupancy
- Kernel fusion that reduces parallel threadgroups DESTROYS performance
- Example: fusing residual_add + rms_norm dropped from 28 TGs to 1 TG
- Always preserve threadgroup count when fusing kernels
- Thread count tweaking cannot bypass memory bandwidth bottlenecks
- 64 threads vs 32 threads per TG = zero improvement when memory-bound

### Metal Optimization Rules (Derived from Experiments)
- Use sx/bx precompute instead of dequant-first FMA
- Dequant-first FMA dropped throughput from 18.67 to 18.14 tok/s
- Store input vector x in threadgroup shared memory
- Use 256 threads per TG, processing 64 groups in parallel per row
- Interval-based FP32 accumulation every 128 elements prevents underflow

---

## QJL Fallback Hierarchy (If 1-bit Fails PPL Gate)

1. 1-bit QJL (current experiment — Exp33+)
   - Target: ~1.52 MB per expert vs 5.44 MB Q3
   - Risk: PPL likely fails at strict 1-bit
   - Uses SRHT for GPU-efficient dequantization

2. Ternary quantization (-1, 0, +1)
   - More representational capacity than 1-bit
   - Untried — no published results for MoE experts
   - Could halve I/O vs 2-bit while preserving quality

3. Skip 2-bit entirely
   - Standard 2-bit failed: PPL spiked to 8.09, read corruptions
   - Chunk-alignment issues with 2-bit format
   - Never retry 2-bit without fixing alignment first

4. Q3 + temporal prediction (current best)
   - Proven: 19.11 tok/s, PPL 5.58
   - Q3 reduced expert payload to 5.44 MB (from 6.6 MB at 4-bit)
   - Temporal prediction overlaps I/O with GPU compute completely
   - CMD2 pre-encoding closed the final submission gap
   - This combination beats the theoretical 2-bit hardware ceiling

---

## NotebookLM Session Summary (March 27, 2026)

Sources loaded: 26 (flash-moe paper, TurboQuant, Apple LLM in Flash,
DeepSeek-V3, Speculative Decoding, Qwen2.5, MoE routing papers,
results.tsv, CLAUDE.md, Reddit post)

Key synthesis outputs:
- Complete ANE co-processing blueprint (4 steps)
- QJL 1-bit implementation spec (4 steps with exact math)
- M5 Max GPU architecture reverse-engineering
- Theoretical roadmap 4.36 → 19.11 → 25+ → 37 tok/s
- sx/bx Metal optimization pattern
- Paper abstract draft
- Related work section candidates
- Business case for HIPAA-compliant local AI API

---

## Final Autoresearch Results (March 27-28, 2026)

### Final Best: 20.34 tok/s (Exp42)

**43 experiments completed over ~24 hours**

| Experiment | Config | tok/s | Status |
|------------|--------|-------|--------|
| Baseline | M3 Max 48GB (Dan Woods) | 4.36 | Reference |
| Exp1 | M5 Max 4-bit no split | 12.48 | Keep |
| Exp16 | Q3 experts + temporal prediction | 18.67 | Keep |
| Exp27 | CMD2 pre-encode GDN layers | 19.11 | Keep |
| Exp33 | 1-bit QJL (SRHT) | FAIL PPL=5647 | Discard |
| Exp35 | Ternary 2-bit symmetric | FAIL PPL=11.49 | Discard |
| Exp36 | K=3 experts | FAIL PPL=6.54 | Discard |
| Exp41 | Horizontal Q/K/V projection fusion | 19.87 | Keep |
| **Exp42** | **CMD2 pre-encode full-attention layers** | **20.34** | **FINAL BEST** |
| Exp43 | CMD1 pre-encode GDN (during cmd2_wait) | 19.51 | Discard |

### Final Performance Stats
- **Best tok/s:** 20.34 (clean machine, no background processes)
- **Sustained under load:** 18-19 tok/s
- **PPL:** 3.70 (full 2K context, unchanged from baseline)
- **Improvement over M3 Max baseline:** 4.67x
- **Beat theoretical 2-bit I/O ceiling (18.6 tok/s):** YES — with 4-bit + algorithms

### Key Algorithmic Wins
1. **cache-io-split 4** — parallel SSD fanout, 41% I/O reduction
2. **Q3 GGUF experts** — smaller payload, surprisingly better PPL than 4-bit
3. **Temporal prediction** — overlap expert I/O with GPU compute
4. **CMD2 pre-encode (GDN)** — eliminate CMD1→CMD2 submission gap
5. **Horizontal Q/K/V fusion** — single kernel reads x once for all projections
6. **CMD2 pre-encode (full-attn)** — extend Exp27 to all 60 layers

### Failed Experiments (Important for Paper)
- 1-bit QJL: PPL catastrophic (5647) — too lossy for generative LLM
- Ternary 2-bit: PPL failed (11.49) — 84% sparsity too aggressive
- K=3 experts: PPL failed (6.54) — model trained with K=4
- Cross-layer prediction: near-zero hit rate
- Encoder fusion: zero improvement (encoder overhead only ~1μs)
- Shared memory optimizations: hurt occupancy, L2 handles it automatically

### Updated Abstract Numbers
- Final tok/s: **20.34** (not 19.11 as in draft)
- Improvement: **4.67x** over M3 Max baseline
- Experiments: **43 total**, ~30% keep rate

### Best Inference Command
```bash
cd ~/flash-moe
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

---

## NEXT TASK: Merge Anemll iOS-App Batched Prefill Branch

### Goal
Merge the Anemll iOS-App branch batched prefill optimizations with our autoresearch/mar25 decode optimizations to get BOTH fast prefill AND fast decode.

### Current State
- Our branch: autoresearch/mar25
- Our decode: 20.34 tok/s ✅
- Our prefill: 5.52 tok/s ❌ (slow)
- Anemll iOS-App branch: batched prefill achieving 19.7-20.5 tok/s prefill ✅
- Their decode: ~10.9 tok/s ❌ (no our optimizations)

### What Anemll iOS-App Branch Added
- Batched projections: GPU batch GEMM reads weight matrices once for N tokens
- Batched linear attention: Custom Metal kernels for conv1d, delta-net, gated RMS norm
- Batched full attention: prefill_causal_attn Metal kernel with Flash Attention style online softmax
- Skip routed experts during prefill: intermediate tokens use shared expert only (K=0)
- Experts at full-attention only mode: K=0 at linear layers, K=4 at full-attention layers
- New CLI flags: --pfb N, --prefill-skip-experts, --prefill-experts-full-only, --prefill-k N

### New Metal Kernels in iOS-App Branch
- dequant_gemm_4bit_batch — batched 4-bit dequant GEMM
- prefill_causal_attn — Flash Attention with online softmax
- prefill_q_rope_norm_bf16 — fused Q deinterleave + RMS norm + RoPE
- conv1d_step_batched — batched conv1d for linear attention
- rms_norm_qk_batched — batched Q/K RMS norm
- compute_decay_beta_batched — batched decay/beta for delta-net
- gated_delta_net_step_batched — batched delta-net recurrence
- gated_rms_norm_batched — batched gated RMS norm
- prefill_rms_norm — batched input RMS norm
- prefill_residual_norm — batched residual + post-attn norm
- prefill_swiglu — batched SwiGLU activation
- prefill_combine — batched MoE combine + residual

### Performance Targets After Merge
- Prefill: ~20 tok/s (from 5.52 tok/s — 3.6x improvement)
- Decode: ~20.34 tok/s (preserve our current best)
- TTFT for 200 token prompt: ~10s (from ~36s)

### Recommended Merge Config
Per Anemll's own testing on M5 Max 128GB 4-bit model:
--prefill-experts-full-only --pfb 128
- 45 linear layers (75%): fully batched, K=0 — no SSD I/O
- 15 full-attention layers (25%): batched + K=4 experts
- Result: 1.9x faster prefill with identical output quality

### Merge Strategy
1. git remote add anemll-ios https://github.com/Anemll/flash-moe
2. git fetch anemll-ios iOS-App
3. git checkout -b combined-prefill-decode
4. git merge anemll-ios/iOS-App
5. Resolve conflicts in infer.m and shaders.metal carefully
6. Preserve ALL our decode optimizations (temporal prediction, CMD2 pre-encode, fused Q/K/V)
7. Add ALL their prefill optimizations (batched kernels, skip experts)
8. Test decode: should still hit ~20 tok/s
9. Test prefill: should hit ~20 tok/s with --prefill-experts-full-only --pfb 128

### Paper Update After Merge
Update arxiv_draft.tex to include:
- Batched prefill results (new Section or add to Results)
- Credit Anemll iOS-App branch for batched prefill methodology
- Combined system: ~20 tok/s decode AND ~20 tok/s prefill
- Update abstract to mention both contributions
- Add Anemll team to acknowledgments

- Paper draft: ~/flash-moe/paper/arxiv_draft.tex
