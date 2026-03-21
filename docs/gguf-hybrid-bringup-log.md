# GGUF Hybrid Bring-Up Log

This file is the working notebook for the incremental GGUF hybridization effort.

Keep it updated as the implementation changes so the project can resume without rebuilding context from chat history.

## Mission

Bring selected GGUF tensors into Flash-MoE incrementally, using the existing 4-bit runtime as the base.

Current intent:

- start with persistent resident tensors, not streamed experts
- preserve GGUF quantization format and block size exactly
- replace one tensor or one layer family at a time
- validate each step with short PPL and a short generation smoke test
- compare speed and quality against the existing 4-bit baseline

## Hard Rules

- Do not load the full GGUF model into memory.
- Use metadata-only GGUF inspection.
- Do not mass-convert the model.
- Keep the current runtime split:
  - resident tensors stay mmap-backed at startup
  - routed experts stay on the existing streamed path until explicitly changed
- Use local `llama.cpp` as the layout and math reference, not as the runtime.
- Copy or adapt exact block structs and kernel math for supported tensor types.
- Preserve GGUF block sizes exactly. Do not normalize them into MLX-style group sizes.

## Canonical GGUF Types In Scope

These are the current GGUF tensor types we expect to support:

| Type | Block Size | Notes |
|---|---:|---|
| `Q8_0` | 32 | Resident dense tensors and embeddings |
| `Q6_K` | 256 | LM head |
| `Q5_K` | 256 | Outlier expert/down tensor in `blk.27` |
| `IQ4_XS` | 256 | Expert tensors |
| `IQ3_XXS` | 256 | Expert tensors |

## Current Architecture Decision

The current preferred approach is:

1. Keep GGUF tensor payloads in their original quantized layout.
2. Copy selected tensor bytes into Flash-MoE-managed artifacts without requantizing them into the old 4-bit format.
3. Add manifest-driven lookup for resident GGUF-backed tensors.
4. Copy or adapt the exact `llama.cpp` block structs and Metal kernel math for each supported tensor type.
5. Validate one tensor path at a time.

This means "copy/adapt", not "call into `llama.cpp`".

Why:

- calling the full `ggml` runtime does not fit Flash-MoE's execution model
- preserving GGUF bytes reduces format-conversion risk
- bit-exact block layout is critical for correctness
- resident tensors let us validate the math before touching flash-sensitive expert streaming

## Confirmed References

Local `llama.cpp` already has the exact structs and Metal kernels we need as reference implementations:

- `QK_K = 256` in [/Users/anemll/SourceRelease/GITHUB/ML_playground/llama.cpp/ggml/src/ggml-common.h#L89](/Users/anemll/SourceRelease/GITHUB/ML_playground/llama.cpp/ggml/src/ggml-common.h#L89)
- `QK8_0 = 32` in [/Users/anemll/SourceRelease/GITHUB/ML_playground/llama.cpp/ggml/src/ggml-common.h#L230](/Users/anemll/SourceRelease/GITHUB/ML_playground/llama.cpp/ggml/src/ggml-common.h#L230)
- `block_q5_K` in [/Users/anemll/SourceRelease/GITHUB/ML_playground/llama.cpp/ggml/src/ggml-common.h#L334](/Users/anemll/SourceRelease/GITHUB/ML_playground/llama.cpp/ggml/src/ggml-common.h#L334)
- `block_q6_K` in [/Users/anemll/SourceRelease/GITHUB/ML_playground/llama.cpp/ggml/src/ggml-common.h#L346](/Users/anemll/SourceRelease/GITHUB/ML_playground/llama.cpp/ggml/src/ggml-common.h#L346)
- `block_iq3_xxs` in [/Users/anemll/SourceRelease/GITHUB/ML_playground/llama.cpp/ggml/src/ggml-common.h#L390](/Users/anemll/SourceRelease/GITHUB/ML_playground/llama.cpp/ggml/src/ggml-common.h#L390)
- `block_iq4_xs` in [/Users/anemll/SourceRelease/GITHUB/ML_playground/llama.cpp/ggml/src/ggml-common.h#L439](/Users/anemll/SourceRelease/GITHUB/ML_playground/llama.cpp/ggml/src/ggml-common.h#L439)
- `kernel_mul_mv_q6_K_f32` in [/Users/anemll/SourceRelease/GITHUB/ML_playground/llama.cpp/ggml/src/ggml-metal/ggml-metal.metal#L7709](/Users/anemll/SourceRelease/GITHUB/ML_playground/llama.cpp/ggml/src/ggml-metal/ggml-metal.metal#L7709)
- `kernel_mul_mv_q5_K_f32` in [/Users/anemll/SourceRelease/GITHUB/ML_playground/llama.cpp/ggml/src/ggml-metal/ggml-metal.metal#L7601](/Users/anemll/SourceRelease/GITHUB/ML_playground/llama.cpp/ggml/src/ggml-metal/ggml-metal.metal#L7601)
- `kernel_mul_mv_iq3_xxs_f32` in [/Users/anemll/SourceRelease/GITHUB/ML_playground/llama.cpp/ggml/src/ggml-metal/ggml-metal.metal#L8049](/Users/anemll/SourceRelease/GITHUB/ML_playground/llama.cpp/ggml/src/ggml-metal/ggml-metal.metal#L8049)
- `kernel_mul_mv_iq4_xs_f32` in [/Users/anemll/SourceRelease/GITHUB/ML_playground/llama.cpp/ggml/src/ggml-metal/ggml-metal.metal#L8702](/Users/anemll/SourceRelease/GITHUB/ML_playground/llama.cpp/ggml/src/ggml-metal/ggml-metal.metal#L8702)

## Current GGUF Inventory Snapshot

From the metadata-only sweep:

- `output.weight` is `Q6_K`
- `token_embd.weight` is `Q8_0`
- most resident dense and shared expert tensors are `Q8_0`
- most routed experts are `IQ4_XS` or `IQ3_XXS`
- `blk.27.ffn_down_exps.weight` is a `Q5_K` outlier
- seven `blk.27` tensors are `BF16` outliers
- the persistent path is still dominated by `Q8_0`, so the next resident targets are the large dense families such as `blk.*.attn_qkv.weight`, `blk.*.attn_gate.weight`, and `blk.*.ssm_out.weight`

See the current sweep report in [/Users/anemll/SourceRelease/GITHUB/ML_playground/flash-moe-org/docs/gguf-q3-tensor-sweep.md](/Users/anemll/SourceRelease/GITHUB/ML_playground/flash-moe-org/docs/gguf-q3-tensor-sweep.md).

## Start Order

1. `output.weight` first
2. `Q8_0` resident dense tensors
3. shared expert resident tensors
4. streamed expert tensors later via `packed_experts_Q3/`

Rationale:

- LM head is isolated and always resident
- it avoids SSD streaming changes
- it gives a clean correctness target for short PPL
- it lets us prove the GGUF kernel plumbing before widening scope

## Validation Contract

For every incremental change:

- run a short PPL check
- run a short generation smoke test
- compare throughput versus the 4-bit baseline
- avoid merging multiple tensor-type changes in one step unless earlier single-type tests are already stable

## Starting Point Snapshot

The official locked checkpoint for this GGUF line is:

![GGUF Hybrid Starting Point](/Users/anemll/SourceRelease/GITHUB/ML_playground/flash-moe-org/docs/gguf-hybrid-starting-point.svg)

Use this snapshot as the locked resident-tensor reference for future quality and throughput comparisons.

## Logging Rule

When the GGUF hybrid effort changes meaningfully, append a dated entry here that records:

- what changed
- why it changed
- what tensor type or tensor family it affected
- how it was validated
- whether the change is considered kept, discarded, or still exploratory

## Entries

### 2026-03-20 - Established hybrid GGUF bring-up rules

Status: exploratory

Recorded decisions:

- the repo's autoresearch config now targets incremental hybrid GGUF work
- GGUF inspection must remain metadata-only
- `packed_experts_Q3/` is reserved for future streamed expert artifacts
- the work should mirror the existing 2-bit mixed-quant workflow
- short PPL and smoke generation should be run on every iteration

Validation:

- updated autoresearch config and documentation

### 2026-03-20 - Completed initial Q3 GGUF tensor sweep

Status: kept

What was learned:

- quantization in this model is per tensor, with per-block packing inside each tensor
- the LM head is `Q6_K`
- resident tensors are mostly `Q8_0`
- routed expert tensors are mostly `IQ4_XS` and `IQ3_XXS`
- there are real outliers that must be preserved exactly, including `Q5_K` and `BF16` tensors in `blk.27`

Validation:

- metadata-only sweep across all 5 GGUF shards
- tensor-specific checks for `output.weight` and `blk.27.*`

### 2026-03-20 - Decided to copy/adapt GGUF tensor support instead of calling `llama.cpp`

Status: kept

Decision:

- preserve GGUF tensor payloads as GGUF-style quantized data
- copy or adapt exact block structs and Metal kernel logic from local `llama.cpp`
- do not link the Flash-MoE runtime to the full `ggml` execution stack

Why:

- this keeps the runtime close to the current Flash-MoE structure
- it minimizes conversion risk
- it avoids changing streaming behavior while resident-tensor support is being proven

Next target:

- implement `Q6_K` support for `output.weight` first

### 2026-03-20 - Implemented first GGUF `Q6_K` LM head path

Status: exploratory

What changed:

- added `autoresearch/extract_gguf_lm_head.py` to copy `output.weight` into a standalone raw `Q6_K` artifact
- added optional `--gguf-lm-head` runtime plumbing in `metal_infer/infer.m`
- added a dedicated `Q6_K` CPU fallback and Metal kernel for the LM head path
- wired the autoresearch harness to pass the extracted LM head artifact when present

Validation:

- initial smoke test failed with all-zero logits, which exposed a real decoding bug
- fixed the bug by switching the `Q6_K` block scale decode from bf16 to fp16
- post-fix smoke generation produced sane non-zero logits and output
- short autoresearch benchmark result:
  - decode `7.79 tok/s`
  - prefill `4.21 tok/s`
  - short PPL `5.45`
  - versus 4-bit baseline: slower decode, better prefill, slightly better short PPL

Takeaway:

- correctness looks good enough to continue iterating
- performance is not yet competitive with the baseline, so this is a bring-up milestone rather than a kept optimization win

### 2026-03-20 - Added a cheap smoke prompt ahead of PPL

Status: kept

What changed:

- the autoresearch harness now runs a short smoke generation with `What is Apple Neural Engine?` before the more expensive PPL benchmark

Why:

- this catches catastrophic output regressions earlier
- it keeps GGUF bring-up iterations cheap when the path is obviously broken

### 2026-03-20 - Full 2k-token PPL comparison for the first LM head experiment

Status: kept

What was measured:

- 4-bit baseline:
  - full PPL `3.64`
  - cross-entropy `1.2920`
  - full-PPL throughput `8.58 tok/s`
- GGUF `Q6_K` LM head over the 4-bit base:
  - full PPL `3.62`
  - cross-entropy `1.2856`
  - full-PPL throughput `8.29 tok/s`
- plain `--2bit` comparison:
  - full PPL `5.71`
  - cross-entropy `1.7415`
  - full-PPL throughput `11.21 tok/s`

Takeaway:

- the GGUF LM head slightly improves full PPL versus the 4-bit baseline
- decode and PPL throughput are still slower than the current 4-bit baseline
- `--2bit` is much faster but remains materially worse on quality
- this four-way result set is the initial GGUF starting point snapshot

Assumption:

- the `--2bit` comparison was run as plain 2-bit experts without the GGUF LM head overlay

### 2026-03-20 - Added ETA to live PPL progress output

Status: kept

What changed:

- the PPL progress line in `metal_infer/infer.m` now shows an ETA computed from the current average throughput and remaining tokens

Validation:

- verified on the short `ppl_tokens.bin` run that progress lines now print values such as `ETA 00:47`

### 2026-03-20 - Fourth full-PPL comparison: `--2bit` plus GGUF LM head

Status: kept

What was measured:

- `--2bit` experts plus GGUF `Q6_K` LM head:
  - full PPL `5.66`
  - cross-entropy `1.7333`
  - full-PPL throughput `11.40 tok/s`

Comparison versus plain `--2bit`:

- PPL improved from `5.71` to `5.66`
- cross-entropy improved from `1.7415` to `1.7333`
- throughput improved from `11.21 tok/s` to `11.40 tok/s`

Takeaway:

- the better LM head recovers a small amount of quality for the 2-bit path
- the dominant quality loss still comes from the 2-bit expert tensors, not the LM head

### 2026-03-20 - Added GGUF `Q8_0` embedding path and locked the resident-tensor checkpoint

Status: kept

What changed:

- added `autoresearch/extract_gguf_embedding.py` to copy `token_embd.weight` into a standalone raw `Q8_0` artifact
- added optional `--gguf-embedding` runtime plumbing alongside the existing GGUF LM head path
- ran the resident-tensor comparison with the embedding plus LM head overlays
- updated the checkpoint SVG to reflect the resident-tensor baseline rather than only the LM-head starting point

What was measured:

- 4-bit + GGUF embedding + GGUF LM head:
  - smoke sane
  - generation `8.51 decode tok/s`, `4.59 prefill tok/s`
  - short PPL `5.62`, cross-entropy `1.7260`, `9.06 tok/s`
  - full PPL `3.61`, cross-entropy `1.2829`, `8.35 tok/s`

Comparison versus the 4-bit baseline:

- full PPL improved slightly from `3.64` to `3.61`
- cross-entropy improved slightly from `1.2920` to `1.2829`
- decode throughput was slightly slower, while prefill improved

Takeaway:

- the persistent GGUF overlay works for both embedding and LM head
- the resident-tensor checkpoint is now stable enough to use as the new baseline for further `Q8_0` work

### 2026-03-20 - Recorded the 2-bit resident-tensor counterpart

Status: kept

What was measured:

- `--2bit` + GGUF embedding + GGUF `Q6_K` LM head:
  - smoke sane
  - generation `8.33 decode tok/s`, `4.17 prefill tok/s`
  - short PPL `8.31`, cross-entropy `2.1180`, `7.99 tok/s`
  - full PPL `5.72`, cross-entropy `1.7445`, `9.37 tok/s`

Takeaway:

- the 2-bit resident-tensor overlay is materially worse on quality than the 4-bit resident checkpoint
- it is useful as a comparison point, but the persistent path should continue forward from the 4-bit resident baseline

### 2026-03-20 - GGUF `Q8_0` QKV bridge for linear attention

Status: kept

What changed:

- `autoresearch/extract_gguf_qkv_overlay.py` no longer raw-copies `blk.*.attn_qkv.weight`
- the extractor now untile-reorders only the V rows from GGUF's Qwen3.5 tiled-head layout back into Flash-MoE's grouped-V runtime layout
- the bridge preserves GGUF `Q8_0` blocks exactly; it only permutes whole row spans
- `autoresearch/run_experiment.py` now accepts `--gguf-qkv-bin` and `--gguf-qkv-json`

Why:

- local `llama.cpp` applies `_reorder_v_heads(...)` to Qwen3.5 `linear_attn.in_proj_qkv`
- a raw GGUF byte copy produced broken semantics even on the CPU fallback path
- the failure mode was layout mismatch, not just missing `Q8_0` kernel math

What was measured for the corrected `QKV`-only overlay:

- smoke:
  - prompt output became sane again
  - smoke throughput `5.98 decode tok/s`, `1.50 prefill tok/s`
- generation:
  - `7.90 decode tok/s`
  - `3.90 prefill tok/s`
- short PPL:
  - PPL `5.41`
  - cross-entropy `1.6890`
  - throughput `8.04 tok/s`
- full 2k-token PPL:
  - PPL `3.52`
  - cross-entropy `1.2595`
  - throughput `7.22 tok/s`

Comparison versus the 4-bit baseline:

- short PPL improved from `5.51` to `5.41`
- full PPL improved from `3.64` to `3.52`
- generation decode slowed from `8.85` to `7.90 tok/s`
- generation prefill improved from `3.17` to `3.90 tok/s`

Additional combined resident-set checkpoint:

- embedding + LM head + corrected `QKV` overlay:
  - smoke sane
  - generation `8.06 decode tok/s`, `3.86 prefill tok/s`
  - short PPL `5.41`
  - cross-entropy `1.6890`
  - short-PPL throughput `8.02 tok/s`

Takeaway:

- the first persistent attention-family `Q8_0` overlay is now valid
- `attn_qkv.weight` is not raw-drop-in compatible with Flash-MoE; it needs a Qwen3.5-specific row-layout bridge
- once corrected, the tensor is quality-positive relative to the current 4-bit baseline
- the next persistent target should stay on the attention side, but not assume every linear-attention tensor is plug-compatible

Post-fix local verification on the checked-in path:

- found and fixed a real performance bug in `metal_infer/infer.m`
- root cause: `fast_batch_matvec(...)` was skipping `MATVEC_KIND_GGUF_Q8_0` specs from the GPU batch and sending them through `cpu_q8_0_matvec(...)`
- consequence: the exact local smoke command for the `QKV` overlay could fall to about `0.7 decode tok/s`
- after the fix, the exact smoke command is now:
  - 4-bit baseline: `6.84 decode tok/s`, `2.29 prefill tok/s`
  - 4-bit + corrected `QKV` overlay: `8.39 decode tok/s`, `3.41 prefill tok/s`
- direct `QKV`-only short PPL rerun:
  - PPL `5.33`
  - cross-entropy `1.6735`
  - throughput `5.48 tok/s`
- direct `QKV`-only 64-token generation rerun:
  - `4.51 decode tok/s`
  - `1.85 prefill tok/s`

Measurement note:

- `autoresearch/run_experiment.py` currently picks up optional overlay defaults from `autoresearch/config.json`
- for isolated `QKV`-only verification, prefer direct `./metal_infer/infer ... --gguf-qkv-bin ... --gguf-qkv-json ...` commands or explicitly blank the other overlay args

Checkpoint:

- snapshot SVG: `docs/gguf-qkv-bridge-checkpoint.svg`

### 2026-03-20 - Full-attention `Q8_0` block bring-up: `q+k+v+o` fixed on the kept path

Status: keep

What changed:

- added `autoresearch/extract_gguf_full_attn_overlay.py` to extract resident GGUF full-attention `Q8_0` tensors
- the extractor supports role subsets via `--roles q,k,v,o`
- `autoresearch/run_experiment.py` accepts `--gguf-full-attn-bin` and `--gguf-full-attn-json`
- `metal_infer/infer.m` now wraps the full-attention GGUF overlay as its own Metal buffer and tracks per-spec GGUF source selection instead of assuming all `Q8_0` specs come from the linear-attention `qkv` blob
- `metal_infer/infer.m` now supports GGUF `Q8_0` `o_proj` directly inside fused `CMD2`
- the full-attention deferred expert path now uploads CPU-computed `h_mid` into `buf_h_mid` before GPU combine when fused `CMD2` was not used
- long-context GGUF `o_proj` no longer enters the unvalidated full-attention GPU-attention branch; it stays on CPU attention plus fused `CMD2`

What was measured after the fix:

- `q`-only overlay:
  - smoke stayed sane
  - short PPL `5.59`
  - cross-entropy `1.7201`
  - short-PPL throughput `8.95 tok/s`
- `q+k+v` block overlay:
  - smoke stayed sane
  - smoke throughput `6.05 decode tok/s`, `2.37 prefill tok/s`
  - short PPL `5.29`
  - cross-entropy `1.6662`
  - short-PPL throughput `8.75 tok/s`
- full `q+k+v+o` block overlay, final kept path:
  - smoke is sane
  - smoke throughput `6.04 decode tok/s`, `1.53 prefill tok/s`
  - short PPL `5.28`
  - cross-entropy `1.6633`
  - short-PPL throughput `7.95 tok/s`
  - full 2k-token PPL `3.48`
  - full 2k-token cross-entropy `1.2467`
  - full 2k-token throughput `6.28 tok/s`
- raw `o`-only overlay:
  - smoke failed immediately with `<|im_end|>`-style output
  - smoke throughput `8.86 decode tok/s`, `3.54 prefill tok/s`
  - short PPL `17990.40`
  - cross-entropy `9.7976`
  - short-PPL throughput `8.49 tok/s`
- candidate `o`-column head-order bridges:
  - `tiled -> grouped`: smoke failed
  - `grouped -> tiled`: smoke failed

Cross-check against local `llama.cpp`:

- `Qwen3_5MoeTextModel` only applies Qwen3.5-specific reorder logic to tensors whose names contain `linear_attn.`; full-attention `attn_output.weight` falls through unchanged
- that matched the final result: the real failures were runtime-path issues, not a missing Qwen3.5 export transform

Takeaway:

- full attention should be treated as a resident block, not as isolated tensors picked ad hoc
- the `q+k+v` block is valid and slightly better than the short-PPL 4-bit baseline (`5.29` vs `5.51`)
- the kept `q+k+v+o` path is now valid end to end on smoke, short PPL, and full 2k-token PPL
- quality is now slightly better than the recorded 4-bit 2k-token baseline (`3.48` vs `3.64`)
- speed is slower than the current native 4-bit path because GGUF `o_proj` still avoids the full-attention GPU-attention fast path
- the new full-attention GGUF Metal buffer path and fused `Q8_0 o_proj` path are solid groundwork for later speed work

Next step:

- treat this as the resident full-attention baseline to improve from
- if we want speed next, the likely target is re-enabling a validated GPU-attention path for GGUF `Q8_0 o_proj`
- keep layer 27 on the native BF16 path unless we add explicit mixed-format handling for that outlier

Checkpoint:

- snapshot SVG: `docs/gguf-full-attn-split-checkpoint.svg`
  - updated to the kept full-attention checkpoint with the final smoke, short-PPL, and 2k-PPL metrics
  - refreshed after direct local re-verification from the checked-in binary

### 2026-03-20 - Measured the full resident GGUF stack together

Status: keep

Scope:

- `embedding_q8_0`
- `lm_head_q6`
- `attn_qkv_q8_0`
- `full_attn_q8_0`
- base experts remained on the normal 4-bit path

Command shape:

- `./metal_infer/infer --model ... --gguf-embedding ... --gguf-lm-head ... --gguf-qkv-bin ... --gguf-qkv-json ... --gguf-full-attn-bin ... --gguf-full-attn-json ... --ppl ...`

What was measured:

- short PPL on `ppl_tokens.bin`:
  - cross-entropy `1.6450`
  - PPL `5.18`
  - throughput `4.46 tok/s`
- full 2k-token PPL on `ppl_tokens_2k.bin`:
  - cross-entropy `1.2496`
  - PPL `3.49`
  - throughput `5.10 tok/s`

Comparison:

- versus recorded plain 4-bit baseline:
  - short PPL improved from `5.51` to `5.18`
  - full 2k-token PPL improved from `3.64` to `3.49`
- versus the kept full-attention-only resident checkpoint:
  - quality stayed essentially tied on full PPL (`3.49` vs `3.48`)
  - quality improved on short PPL (`5.18` vs `5.28`)
  - throughput is lower because every resident GGUF overlay is active at once

Takeaway:

- all currently kept resident GGUF overlays compose cleanly
- the combined resident stack is quality-positive versus the plain 4-bit baseline
- the next problem is speed optimization, not correctness

### 2026-03-20 - Measured the full resident GGUF stack together on `--2bit`

Status: keep

Scope:

- `embedding_q8_0`
- `lm_head_q6`
- `attn_qkv_q8_0`
- `full_attn_q8_0`
- experts on `--2bit`

Command shape:

- `./metal_infer/infer --model ... --2bit --gguf-embedding ... --gguf-lm-head ... --gguf-qkv-bin ... --gguf-qkv-json ... --gguf-full-attn-bin ... --gguf-full-attn-json ... --ppl ...`

What was measured:

- short PPL on `ppl_tokens.bin`:
  - cross-entropy `2.1052`
  - PPL `8.21`
  - throughput `8.55 tok/s`
- full 2k-token PPL on `ppl_tokens_2k.bin`:
  - cross-entropy `1.7009`
  - PPL `5.48`
  - throughput `6.89 tok/s`

Comparison:

- versus recorded plain `--2bit` baseline:
  - full 2k-token PPL improved from `5.71` to `5.48`
- versus the earlier `--2bit` plus GGUF LM head checkpoint:
  - full 2k-token PPL improved from `5.66` to `5.48`
- versus the full resident GGUF stack on 4-bit experts:
  - quality is still materially worse (`5.48` vs `3.49` full PPL)
  - speed is higher (`6.89 tok/s` vs `5.10 tok/s` on full 2k-token PPL)

Takeaway:

- all currently kept resident GGUF overlays also compose cleanly on `--2bit`
- the resident GGUF stack helps the 2-bit path, but the dominant quality loss still comes from the 2-bit experts
- this is the correct combined-stack `--2bit` baseline for future hybrid work

### 2026-03-20 - Finished the resident linear overlay for `attn_gate` and `ssm_out`

Status: keep

Scope:

- `blk.*.attn_gate.weight`
- `blk.*.ssm_out.weight`
- no shared experts
- no streamed experts

Implementation:

- added a dedicated resident GGUF `Q8_0` linear overlay path in the runtime
- extracted both tensor families into:
  - `autoresearch/gguf/linear_q8_0.bin`
  - `autoresearch/gguf/linear_q8_0.json`
- applied the inverse Qwen3.5 bridge needed to map GGUF layout back into Flash-MoE layout:
  - `attn_gate` uses the inverse V-row tiling bridge
  - `ssm_out` uses the inverse V-column tiling bridge

Command shape:

- smoke:
  - `./metal_infer/infer --model ... --gguf-linear-bin autoresearch/gguf/linear_q8_0.bin --gguf-linear-json autoresearch/gguf/linear_q8_0.json --prompt "What is Apple Neural Engine?" --tokens 24 --stream`
- short PPL:
  - `./metal_infer/infer --model ... --gguf-linear-bin ... --gguf-linear-json ... --ppl ppl_tokens.bin`
- full PPL:
  - `./metal_infer/infer --model ... --gguf-linear-bin ... --gguf-linear-json ... --ppl ppl_tokens_2k.bin`

What was measured:

- smoke:
  - output sane
  - decode `7.10 tok/s`
  - prefill `1.82 tok/s`
- short PPL on `ppl_tokens.bin`:
  - cross-entropy `1.7029`
  - PPL `5.49`
  - throughput `8.07 tok/s`
- full 2k-token PPL on `ppl_tokens_2k.bin`:
  - cross-entropy `1.2734`
  - PPL `3.57`
  - throughput `7.27 tok/s`

Comparison:

- versus the recorded plain 4-bit baseline:
  - short PPL improved from `5.51` to `5.49`
  - full 2k-token PPL improved from `3.64` to `3.57`
- versus the previous best resident full-attention checkpoint:
  - isolated linear overlay is a smaller quality win than the full-attention block
  - but it is still clearly positive and composes cleanly

Takeaway:

- the remaining non-shared persistent linear-attention tensors are now working as a resident GGUF `Q8_0` overlay
- this is a keep on both smoke and PPL
- with this checkpoint, the remaining blocked work in the persistent family is shared experts, which stays out of scope until explicitly reopened

Checkpoint:

- snapshot SVG: `docs/gguf-resident-stack-checkpoint.svg`
- notebook: `docs/gguf-hybrid-bringup-log.md`

### 2026-03-20 - Re-measured the full resident GGUF stack after adding the linear overlay

Status: keep

Scope:

- `embedding_q8_0`
- `lm_head_q6`
- `attn_qkv_q8_0`
- `full_attn_q8_0`
- `linear_q8_0` for `attn_gate` and `ssm_out`
- base experts remained on the normal 4-bit path

Command shape:

- `./metal_infer/infer --model ... --gguf-embedding ... --gguf-lm-head ... --gguf-qkv-bin ... --gguf-qkv-json ... --gguf-full-attn-bin ... --gguf-full-attn-json ... --gguf-linear-bin ... --gguf-linear-json ... --ppl ...`

What was measured:

- short PPL on `ppl_tokens.bin`:
  - cross-entropy `1.6488`
  - PPL `5.20`
  - throughput `6.31 tok/s`
- full 2k-token PPL on `ppl_tokens_2k.bin`:
  - cross-entropy `1.2290`
  - PPL `3.42`
  - throughput `5.43 tok/s`

Comparison:

- versus recorded plain 4-bit baseline:
  - short PPL improved from `5.51` to `5.20`
  - full 2k-token PPL improved from `3.64` to `3.42`
- versus the earlier resident stack without the linear overlay:
  - short PPL changed from `5.18` to `5.20`
  - full 2k-token PPL improved from `3.49` to `3.42`
  - throughput improved slightly on full 2k-token PPL (`5.10` to `5.43 tok/s`) under the clean sequential rerun

Takeaway:

- all currently allowed resident GGUF overlays now compose cleanly, including `attn_gate` and `ssm_out`
- the best full resident 4-bit checkpoint is now `3.42` full PPL
- persistent-tensor bring-up is effectively complete for the non-shared path

Checkpoint:

- snapshot SVG: `docs/gguf-resident-stack-checkpoint.svg`
- notebook: `docs/gguf-hybrid-bringup-log.md`

### 2026-03-20 - Added per-change resident GGUF delta chart

Status: keep

What changed:

- added `docs/gguf-resident-delta-by-change.svg`
- the chart compares each measured resident checkpoint against the plain 4-bit baseline
- it shows:
  - full 2k-token PPL improvement
  - full-PPL throughput loss

Notes:

- the chart uses measured checkpoints only
- the embedding row is recorded as `embedding + LM head` because that is the checkpoint we actually measured
- the stack rows are separated from the single-change rows so the per-change view stays honest
