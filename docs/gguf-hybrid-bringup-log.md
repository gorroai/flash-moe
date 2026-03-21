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

### 2026-03-20 - Re-opened shared-expert optimization in the program

Status: keep

What changed:

- `program.md` now explicitly allows shared-expert overlays and shared-expert kernel work
- the old hard stop has been removed

Constraints that still remain:

- shared experts should be treated as their own experiment family
- they should be validated separately from routed streamed experts
- cache sensitivity still matters, so start with small isolated experiments instead of broad rollouts
- streamed routed experts remain a later step unless the human changes priorities again

### 2026-03-20 - First streamed Q3 hybrid expert bring-up

Status: keep

Scope:

- routed expert `gate/up` only
- exact GGUF `IQ3_XXS` bytes preserved for streamed `ffn_gate_exps` and `ffn_up_exps`
- existing native 4-bit `down` bytes kept unchanged
- layer `27` intentionally left on the 4-bit fallback path for now

Implementation:

- added `metal_infer/gguf_iq_shared.h` with vendored `IQ3_XXS` tables and block definitions from local `llama.cpp`
- added GPU `IQ3_XXS` expert matvec support in `metal_infer/shaders.metal`
- extended mixed-quant expert plumbing in `metal_infer/infer.m` with explicit `--q3-experts`
- added `autoresearch/repack_experts_q3.py` to write `packed_experts_Q3/`

Hybrid format:

- `gate`: `IQ3_XXS`, `1,605,632` bytes per expert
- `up`: `IQ3_XXS`, `1,605,632` bytes per expert
- `down`: existing native 4-bit blob, `2,359,296` bytes per expert
- total: `5,570,560` bytes per expert
- versus current 4-bit expert blob `7,077,888` bytes:
  - `21.3%` less streamed data per expert

First artifact test:

- command:
  - `python3 autoresearch/repack_experts_q3.py --layer 0`
- result:
  - wrote `packed_experts_Q3/layer_00.bin`
  - spot-check verification passed for experts `0`, `1`, `255`, and `511`

First decode test:

- command:
  - `./metal_infer/infer --model /Users/anemll/Models/flash_mlx_4bit --q3-experts --prompt "What is Apple Neural Engine?" --tokens 24 --stream`
- runtime state:
  - `1` layer at `Q3-hybrid`, `59` layers at `4-bit`
- output:
  - sane answer
  - decode `8.30 tok/s`
  - prefill `3.57 tok/s`

First short PPL:

- command:
  - `./metal_infer/infer --model /Users/anemll/Models/flash_mlx_4bit --q3-experts --ppl ppl_tokens.bin`
- result with only `layer 00` in Q3-hybrid:
  - cross-entropy `1.6907`
  - PPL `5.42`
  - throughput `8.09 tok/s`

Five-layer widen check:

- command:
  - `python3 autoresearch/repack_experts_q3.py --layers 0-4`
- decode result:
  - `5` layers at `Q3-hybrid`, `55` layers at `4-bit`
  - sane answer
  - decode `6.85 tok/s`
  - prefill `2.97 tok/s`
- short PPL result:
  - cross-entropy `1.6749`
  - PPL `5.34`
  - throughput `6.59 tok/s`

Takeaway:

- the first streamed `IQ3_XXS` hybrid path is decoding correctly
- per-layer fallback is working as intended
- next widening step should be more non-outlier layers, not new format work

### 2026-03-20 - Streamed Q3 widen to 59 hybrid layers

Status: keep

Scope:

- widened the streamed routed-expert path to all non-outlier layers
- `59` layers use exact GGUF `IQ3_XXS` for `gate/up`
- `1` layer (`27`) still falls back to the existing `4-bit` packed file

Artifact state:

- command:
  - `python3 autoresearch/repack_experts_q3.py --layers 5-26,28-59`
- result:
  - `packed_experts_Q3/` now contains `59` streamed Q3-hybrid layer files
  - `layer_27.bin` is intentionally absent in this checkpoint

Validation:

- smoke decode:
  - command:
    - `./metal_infer/infer --model /Users/anemll/Models/flash_mlx_4bit --q3-experts --prompt "What is Apple Neural Engine?" --tokens 24 --stream`
  - runtime state:
    - `59` layers at `Q3-hybrid`, `1` layer at `4-bit`
  - result:
    - sane answer
    - decode `7.87 tok/s`
    - prefill `3.24 tok/s`

- short PPL:
  - command:
    - `./metal_infer/infer --model /Users/anemll/Models/flash_mlx_4bit --q3-experts --ppl ppl_tokens.bin`
  - result:
    - cross-entropy `1.7328`
    - PPL `5.66`
    - throughput `9.44 tok/s`

- full 2k-token PPL:
  - command:
    - `./metal_infer/infer --model /Users/anemll/Models/flash_mlx_4bit --q3-experts --ppl ppl_tokens_2k.bin`
  - result:
    - cross-entropy `1.3253`
    - PPL `3.76`
    - throughput `8.96 tok/s`

Takeaway:

- the widened `59`-layer streamed hybrid path stays very close to the plain `4-bit` baseline on quality
- the remaining missing piece is the `blk.27` outlier, not the main `IQ3_XXS` family

### 2026-03-20 - Layer 27 streamed outlier support

Status: keep

Scope:

- added explicit streamed outlier support for `blk.27`
- preserved exact GGUF `IQ4_XS` bytes for streamed `ffn_gate_exps` and `ffn_up_exps`
- kept `blk.27` `down` on the existing native `4-bit` bytes for this checkpoint
- no caching model changes; still one packed file per layer and one expert read per selected expert

Implementation:

- extended `metal_infer/gguf_iq_shared.h` with exact `IQ4_XS` block definitions and lookup tables
- added GPU `IQ4_XS` expert matvec support in `metal_infer/shaders.metal`
- aligned the streamed outlier runtime path in `metal_infer/infer.m`
- extended `autoresearch/repack_experts_q3.py` with `--include-outlier-layer`

Artifact state:

- command:
  - `python3 autoresearch/repack_experts_q3.py --layer 27 --include-outlier-layer`
- result:
  - wrote `packed_experts_Q3/layer_27.bin`
  - size `3.25 GiB`
  - `packed_experts_Q3/` now covers all `60` layers

Validation:

- smoke decode:
  - command:
    - `./metal_infer/infer --model /Users/anemll/Models/flash_mlx_4bit --q3-experts --prompt "What is Apple Neural Engine?" --tokens 24 --stream`
  - runtime state:
    - `59` layers at `Q3-hybrid`, `1` layer at `Q3-outlier`
  - result:
    - sane answer
    - decode `9.62 tok/s`
    - prefill `3.27 tok/s`

- short PPL:
  - command:
    - `./metal_infer/infer --model /Users/anemll/Models/flash_mlx_4bit --q3-experts --ppl ppl_tokens.bin`
  - result:
    - cross-entropy `1.7379`
    - PPL `5.69`
    - throughput `10.40 tok/s`

- full 2k-token PPL:
  - command:
    - `./metal_infer/infer --model /Users/anemll/Models/flash_mlx_4bit --q3-experts --ppl ppl_tokens_2k.bin`
  - result:
    - cross-entropy `1.3313`
    - PPL `3.79`
    - throughput `10.01 tok/s`

Comparison versus the previous `59`-layer checkpoint:

- quality moved slightly worse:
  - full PPL `3.76 -> 3.79`
- throughput improved:
  - full-PPL speed `8.96 -> 10.01 tok/s`

Takeaway:

- the full `60`-layer streamed Q3 path is decoding correctly
- the `IQ4_XS` outlier does not break output quality or correctness
- the next meaningful streamed step is `GGUF Q5_K` down support for `blk.27`, not more `IQ3_XXS` plumbing

### 2026-03-20 - Layer 27 outlier revalidation and GPU-path check

Status: keep CPU fallback, discard GPU outlier dispatch

Corrections to the earlier layer-27 note:

- `packed_experts_Q3/layer_27.bin` now stores exact GGUF bytes for all three routed projections:
  - `ffn_gate_exps.weight` = `IQ4_XS`
  - `ffn_up_exps.weight` = `IQ4_XS`
  - `ffn_down_exps.weight` = `Q5_K`
- the stable runtime configuration still keeps `layer 27` on the CPU expert path
- the first attempt to run the outlier layer through the generic GPU multi-expert path was reverted after measurement

Byte verification:

- re-checked `layer_27.bin` directly against the GGUF source tensors for experts `0`, `1`, `2`, `255`, and `511`
- all three stored projections matched exactly for every sampled expert
- this ruled out repack/layout corruption and narrowed the question to runtime execution choices

Kept validation (`--q3-experts`, CPU fallback for the outlier layer):

- smoke decode:
  - command:
    - `./metal_infer/infer --model /Users/anemll/Models/flash_mlx_4bit --q3-experts --prompt "What is Apple Neural Engine?" --tokens 24 --stream`
  - result:
    - sane answer
    - representative decode `6.11 tok/s`
    - representative prefill `2.67 tok/s`

- short PPL:
  - command:
    - `./metal_infer/infer --model /Users/anemll/Models/flash_mlx_4bit --q3-experts --ppl ppl_tokens.bin`
  - result:
    - cross-entropy `1.6888`
    - PPL `5.41`
    - throughput `6.52 tok/s`

- full 2k-token PPL:
  - command:
    - `./metal_infer/infer --model /Users/anemll/Models/flash_mlx_4bit --q3-experts --ppl ppl_tokens_2k.bin`
  - result:
    - cross-entropy `1.3077`
    - PPL `3.70`
    - throughput `3.84 tok/s`

Discarded follow-up:

- change:
  - removed the `!g_use_q3_outlier` guard in `metal_infer/infer.m` so `layer 27` would use the generic GPU multi-expert path
- outcome:
  - quality stayed the same on short PPL:
    - cross-entropy `1.6888`
    - PPL `5.41`
  - speed got much worse:
    - smoke decode `2.46 tok/s`
    - smoke prefill `1.34 tok/s`
    - short-PPL throughput `3.53 tok/s`
- decision:
  - reverted immediately

Takeaway:

- the on-disk `IQ4_XS/Q5_K` outlier format is correct
- the current GPU outlier expert kernels are correctness-ready but not performance-ready for this streamed workload
- keep the CPU fallback for `layer 27` until a faster GPU outlier path is proven

### 2026-03-20 - IQ3 streamed expert launch tuning

Status: keep

Change:

- switched the streamed expert `IQ3_XXS` path to a llama.cpp-style launch shape
- kernel now uses:
  - `64` threads per threadgroup
  - `2` simdgroups per threadgroup
  - `4` output rows per simdgroup
  - threadgroup staging for the `iq3xxs_grid` and `ksigns` tables
- runtime dispatch now applies the tuned launch only to streamed expert projections whose kind is `IQ3_XXS`
- non-expert matvec paths and other quant types stayed on their previous launch geometry

Implementation points:

- `metal_infer/shaders.metal`
  - `dequant_matvec_iq3_xxs`
- `metal_infer/infer.m`
  - `expert_threads_per_threadgroup`
  - `expert_threadgroup_memory_length`
  - `expert_dispatch_matvec`

Validation:

- smoke decode:
  - command:
    - `./metal_infer/infer --model /Users/anemll/Models/flash_mlx_4bit --q3-experts --prompt "What is Apple Neural Engine?" --tokens 24 --stream`
  - result:
    - sane answer
    - decode `5.95 tok/s`
    - prefill `2.06 tok/s`

- short PPL:
  - command:
    - `./metal_infer/infer --model /Users/anemll/Models/flash_mlx_4bit --q3-experts --ppl ppl_tokens.bin`
  - result:
    - cross-entropy `1.6888`
    - PPL `5.41`
    - throughput `5.43 tok/s`

- 200-token timing:
  - command:
    - `./metal_infer/infer --model /Users/anemll/Models/flash_mlx_4bit --q3-experts --prompt "What is Apple Neural Engine?" --tokens 200 --timing`
  - result:
    - generation `5.93 tok/s`
    - dense/attn `44.2 ms/token`
    - `o_proj+shared` `26.9 ms/token`
    - expert I/O `42.4 ms/token`
    - expert compute `51.7 ms/token`

Comparison versus the previous streamed-Q3 timing checkpoint:

- decode improved:
  - `4.86 -> 5.93 tok/s`
- expert I/O improved slightly:
  - `47.0 -> 42.4 ms/token`
- expert compute improved materially:
  - `72.8 -> 51.7 ms/token`
- short-PPL quality stayed unchanged:
  - `5.41 -> 5.41`

Reference 4-bit timing on the same 200-token prompt:

- command:
  - `./metal_infer/infer --model /Users/anemll/Models/flash_mlx_4bit --prompt "What is Apple Neural Engine?" --tokens 200 --timing`
- result:
  - generation `7.43 tok/s`
  - expert I/O `53.3 ms/token`
  - expert compute `2.4 ms/token`

Takeaway:

- the streamed `IQ3_XXS` path is now materially faster without changing quality
- the remaining gap versus native `4-bit` is no longer primarily SSD I/O
- the next likely speed limiter is the non-optimized outlier/runtime mix, especially the `layer 27` `IQ4_XS/Q5_K` path that still avoids the GPU fast path

### 2026-03-20 - Layer-27 outlier GPU path (IQ4_XS gate/up + Q5_K down)

Status: keep

Change:

- ported the layer-27 outlier kernels to a llama.cpp-style launch shape
- `IQ4_XS` now runs with:
  - `64` threads per threadgroup
  - `2` simdgroups per threadgroup
  - `2` output rows per simdgroup
  - threadgroup lookup staging for the nonlinear value table
- `Q5_K` now runs with:
  - `64` threads per threadgroup
  - `2` simdgroups per threadgroup
  - `1` output row per simdgroup
  - packed scale/min unpacking adapted from ggml
- runtime dispatch now computes threadgroups by projection kind:
  - `IQ3_XXS`: `8` rows per threadgroup
  - `IQ4_XS`: `4` rows per threadgroup
  - `Q5_K`: `2` rows per threadgroup
- removed the `layer 27` GPU exclusion, so the outlier layer now runs through the normal GPU expert path

Implementation points:

- `metal_infer/shaders.metal`
  - `dequant_matvec_iq4_xs`
  - `dequant_matvec_q5_k`
- `metal_infer/infer.m`
  - `expert_threads_per_threadgroup`
  - `expert_rows_per_threadgroup`
  - `expert_threadgroup_memory_length`
  - `expert_dispatch_matvec`
  - routed expert GPU-path guard in the streamed expert path

Validation:

- smoke decode:
  - command:
    - `./metal_infer/infer --model /Users/anemll/Models/flash_mlx_4bit --q3-experts --prompt "What is Apple Neural Engine?" --tokens 24 --stream`
  - result:
    - sane answer
    - decode `6.88 tok/s`
    - prefill `3.74 tok/s`

- short PPL:
  - command:
    - `./metal_infer/infer --model /Users/anemll/Models/flash_mlx_4bit --q3-experts --ppl ppl_tokens.bin`
  - result:
    - cross-entropy `1.6888`
    - PPL `5.41`
    - throughput `9.45 tok/s`

- 200-token timing:
  - command:
    - `./metal_infer/infer --model /Users/anemll/Models/flash_mlx_4bit --q3-experts --prompt "What is Apple Neural Engine?" --tokens 200 --timing`
  - result:
    - generation `11.25 tok/s`
    - dense/attn `31.3 ms/token`
    - `o_proj+shared` `20.3 ms/token`
    - expert I/O `33.6 ms/token`
    - expert compute `1.3 ms/token`

Comparison versus the previous streamed-Q3 checkpoint:

- smoke decode improved:
  - `5.34 -> 6.88 tok/s`
- short-PPL throughput improved:
  - `5.66 -> 9.45 tok/s`
- 200-token generation improved:
  - `5.87 -> 11.25 tok/s`
- expert I/O improved:
  - `43.2 -> 33.6 ms/token`
- expert compute collapsed:
  - `51.7 -> 1.3 ms/token`
- short-PPL quality stayed unchanged:
  - `5.41 -> 5.41`

Takeaway:

- the outlier layer was the main remaining performance drag in the streamed-Q3 path
- once `IQ4_XS` and `Q5_K` moved onto the tuned GPU path, streamed Q3 became clearly faster than the current plain 4-bit baseline on this prompt family
- the current Q3 path is now in a good place for longer-context and full-PPL follow-up measurements

## 2026-03-20 21: Exact GGUF routed-expert migration completed

Change:

- finished the remaining routed expert repack so `packed_experts_Q3/` now contains all `60` layers in the new exact-GGUF layout
- normal layers now use:
  - `gate_proj.weight` = `IQ3_XXS`
  - `up_proj.weight` = `IQ3_XXS`
  - `down_proj.weight` = `IQ4_XS`
- outlier layer `27` remains:
  - `gate/up` = `IQ4_XS`
  - `down` = `Q5_K`
- updated the runtime label from `Q3-mixed` to `Q3-GGUF` so the output matches the actual finished layout

Validation:

- directory state:
  - `packed_experts_Q3/` contains `60` layer files
  - runtime reports:
    - `59 layers at Q3-GGUF`
    - `1 layer at Q3-outlier`
    - `0 layers at 4-bit`

- smoke decode:
  - command:
    - `./metal_infer/infer --model /Users/anemll/Models/flash_mlx_4bit --q3-experts --prompt "What is Apple Neural Engine?" --tokens 24 --stream`
  - result:
    - sane answer
    - decode `7.02 tok/s`
    - prefill `2.00 tok/s`

- short PPL:
  - command:
    - `./metal_infer/infer --model /Users/anemll/Models/flash_mlx_4bit --q3-experts --ppl ppl_tokens.bin`
  - result:
    - cross-entropy `1.7227`
    - PPL `5.60`
    - throughput `10.21 tok/s`

Comparison versus the prior partially migrated state:

- partial state (`21` Q3-GGUF layers, `39` 4-bit fallback layers):
  - short PPL `5.25`
  - throughput `9.81 tok/s`
- full exact-GGUF routed state:
  - short PPL `5.60`
  - throughput `10.21 tok/s`

Takeaway:

- the temporary native-4bit `down_proj` fallback is now fully gone from the routed expert path
- full exact-GGUF routed experts are slightly faster on the short PPL pass than the partial rollout
- quality moved back near the original 4-bit baseline (`5.51`), which means the exact-GGUF conversion is now the honest apples-to-apples streamed-expert measurement

- full 2k-token PPL:
  - command:
    - `./metal_infer/infer --model /Users/anemll/Models/flash_mlx_4bit --q3-experts --ppl ppl_tokens_2k.bin`
  - result:
    - cross-entropy `1.3372`
    - PPL `3.81`
    - throughput `10.11 tok/s`

Comparison versus the recorded plain 4-bit baseline:

- full 2k PPL moved from `3.64` to `3.81`
- full-PPL throughput improved from `8.58 tok/s` to `10.11 tok/s`

## 2026-03-20 23: Artifact location and repack hygiene

Decision:

- GGUF-derived generated artifacts should live under the model directory:
  - `/Users/anemll/Models/flash_mlx_4bit/gguf/`
- repo-local `autoresearch/gguf` is now treated as a compatibility path, not the canonical storage location

Notes:

- `autoresearch/config.json` now points GGUF artifact outputs at the model-local directory by default
- existing commands that still use `autoresearch/gguf/...` can keep working through a compatibility symlink

Repack rule for `packed_experts_Q3/`:

- if the on-disk Q3 layout has **not** changed and you are just regenerating the same format, you do **not** need to delete the old directory first
- if the on-disk Q3 layout **has** changed, you should rebuild into an empty directory or delete/rename the old `packed_experts_Q3/` first
- reason:
  - stale layer files with the old layout keep the same names
  - the runtime will still open them
  - mixing old-format and new-format layer files is unsafe unless the fallback is intentionally to plain `packed_experts/`

Current example of when cleanup was required:

- moving from the temporary hybrid layout:
  - `gate/up = IQ3_XXS`
  - `down = native 4-bit`
- to the final exact-GGUF layout:
  - `gate = IQ3_XXS`
  - `up = IQ3_XXS`
  - `down = IQ4_XS`

In that case, old `packed_experts_Q3/layer_XX.bin` files were incompatible and had to be replaced.

## 2026-03-20 22: Clean serial timing rerun supersedes earlier noisy I/O read

Context:

- an earlier timing comparison suggested `Q3-GGUF` had higher expert I/O than plain `4-bit`
- that comparison was contaminated by concurrent local testing on the same machine
- reran the comparison cleanly, one inference at a time, alternating order:
  - `4bit_a -> q3_a -> q3_b -> 4bit_b`

Commands:

- `./metal_infer/infer --model /Users/anemll/Models/flash_mlx_4bit --prompt "What is Apple Neural Engine?" --tokens 200 --timing`
- `./metal_infer/infer --model /Users/anemll/Models/flash_mlx_4bit --q3-experts --prompt "What is Apple Neural Engine?" --tokens 200 --timing`

Per-run results:

- `4bit_a`
  - expert I/O `49.0 ms`
  - total/token `104.1 ms`
  - generation `9.55 tok/s`
  - TTFT `1569 ms`
- `4bit_b`
  - expert I/O `52.2 ms`
  - total/token `106.2 ms`
  - generation `9.36 tok/s`
  - TTFT `1652 ms`
- `q3_a`
  - expert I/O `43.1 ms`
  - total/token `95.5 ms`
  - generation `10.39 tok/s`
  - TTFT `1401 ms`
- `q3_b`
  - expert I/O `31.2 ms`
  - total/token `83.2 ms`
  - generation `11.91 tok/s`
  - TTFT `2337 ms`

Averages:

- plain `4-bit`
  - expert I/O `50.6 ms`
  - total/token `105.15 ms`
  - generation `9.46 tok/s`
  - TTFT `1610.5 ms`
- `Q3-GGUF`
  - expert I/O `37.15 ms`
  - total/token `89.35 ms`
  - generation `11.15 tok/s`
  - TTFT `1869.0 ms`

Takeaway:

- the earlier “Q3 has worse expert I/O” reading should be treated as invalid
- on clean serial reruns, `Q3-GGUF` is faster on expert I/O and faster overall decode than plain `4-bit` for this prompt family
- the main remaining caveat is TTFT variance, which still swings noticeably run to run
- the working interpretation is now:
  - `Q3-GGUF` improves steady-state decode throughput
  - `Q3-GGUF` slightly worsens full 2k PPL versus plain `4-bit`
  - TTFT remains noisy and should not be over-interpreted from one or two runs

## 2026-03-20 24: Streamed Q3 and shared-expert SVG checkpoint

Added a dedicated streamed-side snapshot:

- `docs/gguf-q3-streamed-shared-checkpoint.svg`

What it summarizes:

- plain `4-bit` baseline
- final routed `Q3-GGUF` expert path
- `Q3-GGUF` experts plus GGUF `Q6_K` LM head
- clean serial timing rerun averages for `4-bit` vs `Q3-GGUF`
- current shared-expert status

Key numbers captured in the SVG:

- plain `4-bit`
  - full 2k PPL `3.64`
  - full-PPL throughput `8.58 tok/s`
  - clean serial decode average `9.46 tok/s`
  - clean serial expert I/O average `50.6 ms`
- `Q3-GGUF` routed experts
  - full 2k PPL `3.81`
  - full-PPL throughput `10.11 tok/s`
  - clean serial decode average `11.15 tok/s`
  - clean serial expert I/O average `37.15 ms`
- `Q3-GGUF` routed experts plus GGUF LM head
  - full 2k PPL `3.79`
  - full-PPL throughput `10.14 tok/s`

Shared-expert status captured in the same SVG:

- `blk.*.ffn_gate_shexp.weight`: `Q8_0 x59`, `BF16 x1`
- `blk.*.ffn_up_shexp.weight`: `Q8_0 x59`, `BF16 x1`
- `blk.*.ffn_down_shexp.weight`: `Q8_0 x59`, `BF16 x1`
- status:
  - not scripted yet
  - should be handled as a separate experiment family from routed experts

## 2026-03-20 25: Full PPL for the complete implemented GGUF stack

Measured the full currently implemented GGUF stack in one run:

- routed experts:
  - `--q3-experts`
- resident overlays:
  - `--gguf-embedding`
  - `--gguf-lm-head`
  - `--gguf-qkv-bin/json`
  - `--gguf-full-attn-bin/json`
  - `--gguf-linear-bin/json`

Command:

- `./metal_infer/infer --model /Users/anemll/Models/flash_mlx_4bit --q3-experts --gguf-embedding /Users/anemll/Models/flash_mlx_4bit/gguf/embedding_q8_0.bin --gguf-lm-head /Users/anemll/Models/flash_mlx_4bit/gguf/lm_head_q6.bin --gguf-qkv-bin /Users/anemll/Models/flash_mlx_4bit/gguf/attn_qkv_q8_0.bin --gguf-qkv-json /Users/anemll/Models/flash_mlx_4bit/gguf/attn_qkv_q8_0.json --gguf-full-attn-bin /Users/anemll/Models/flash_mlx_4bit/gguf/full_attn_q8_0.bin --gguf-full-attn-json /Users/anemll/Models/flash_mlx_4bit/gguf/full_attn_q8_0.json --gguf-linear-bin /Users/anemll/Models/flash_mlx_4bit/gguf/linear_q8_0.bin --gguf-linear-json /Users/anemll/Models/flash_mlx_4bit/gguf/linear_q8_0.json --ppl /Users/anemll/SourceRelease/GITHUB/ML_playground/flash-moe-org/ppl_tokens_2k.bin`

Result:

- cross-entropy `1.2828`
- PPL `3.61`
- throughput `6.19 tok/s`
- time `322.7 s`

Comparison versus the recorded plain MLX 4-bit baseline:

- plain `4-bit`
  - full 2k PPL `3.64`
  - full-PPL throughput `8.58 tok/s`
- full implemented GGUF stack
  - full 2k PPL `3.61`
  - full-PPL throughput `6.19 tok/s`

Takeaway:

- the full implemented GGUF stack is now slightly better than the plain MLX 4-bit baseline on full PPL
- the quality gain is small:
  - `3.64 -> 3.61`
- the full-stack path is still materially slower on the long PPL run:
  - `8.58 -> 6.19 tok/s`
- so the current best reading is:
  - routed `Q3-GGUF` alone is the throughput-oriented path
  - the full stacked GGUF configuration is the quality-oriented path among the currently implemented options
