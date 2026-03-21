# Flash-MoE Autoresearch Harness

This directory adapts Flash-MoE to a Karpathy-style autoresearch workflow.

The harness does not invent research ideas by itself. Instead, it gives an external coding agent a stable setup:

- a branch initializer
- a fixed benchmark suite
- a machine-readable result format
- a hybrid-GGUF config
- a metadata-only GGUF tensor sweep tool
- a persistent GGUF bring-up log
- a `program.md` loop contract

## Files

- `init_run.py` creates `autoresearch/<tag>` and initializes `autoresearch/results.tsv`
- `run_experiment.py` builds the engine, runs generation and perplexity, and emits a JSON result
- `log_result.py` appends a JSON result to `autoresearch/results.tsv`
- `config.json` holds the current autoresearch parameters for the hybrid GGUF effort
- `sweep_gguf_tensors.py` inspects GGUF shards by metadata only and records exact tensor quantization
- `extract_gguf_qkv_overlay.py` copies `blk.*.attn_qkv.weight` into a standalone `Q8_0` overlay and applies the required Qwen3.5 row-layout bridge
- `extract_gguf_linear_overlay.py` copies `blk.*.attn_gate.weight` and `blk.*.ssm_out.weight` into a standalone `Q8_0` overlay and applies the required inverse Qwen3.5 row/column bridges
- `docs/gguf-hybrid-bringup-log.md` records intent, supported types, and important decisions as the bring-up evolves
- `program.md` tells the agent how to run the continuous keep/discard loop

## Benchmark Philosophy

Karpathy's original autoresearch loop uses a fixed 5-minute training budget and optimizes validation loss.

That does not map cleanly to Flash-MoE, because this repo is an inference engine rather than a trainer. For this repo, the harness uses a fixed workload instead, tuned for hybrid GGUF bring-up:

- Smoke check:
  - fixed sanity prompt
  - short token budget
  - catches obviously broken output paths before the more expensive PPL run
- Generation benchmark:
  - fixed prompt
  - fixed token count
  - objective metric: `decode_tok_s`
- Quality benchmark:
  - short PPL on `ppl_tokens.bin` every run
  - optional periodic full PPL on `ppl_tokens_2k.bin`
  - quality metric: perplexity relative to the current 4-bit baseline

The harness computes:

- `score = decode_tok_s` if the quality gate passes
- `score = 0.0` if the quality gate fails or the run crashes

This keeps the optimization target simple: go faster, but not by breaking model quality.

## Hybrid GGUF Focus

The current research mode is incremental hybridization:

- keep the existing 4-bit MLX model as the runtime base
- replace one tensor or one layer family at a time with GGUF-backed quantized weights
- preserve the original GGUF block size exactly
- mirror the existing 2-bit mixed-quant workflow
- use `packed_experts_Q3/` for new Q3 expert artifacts
- start with always-resident tensors such as the LM head or persistent dense layers before streamed experts

Use the local `llama.cpp` checkout only for metadata inspection and debugging. Do not load the full GGUF model into memory.

Keep the bring-up notebook current in [docs/gguf-hybrid-bringup-log.md](/Users/anemll/SourceRelease/GITHUB/ML_playground/flash-moe-org/docs/gguf-hybrid-bringup-log.md), especially when the set of supported tensor types or the migration plan changes.

Configured GGUF block sizes in [config.json](/Users/anemll/SourceRelease/GITHUB/ML_playground/flash-moe-org/autoresearch/config.json):

- `Q8_0`: block size `32`
- `Q6_K`: block size `256`
- `Q5_K`: block size `256`
- `IQ4_XS`: block size `256`
- `IQ3_XXS`: block size `256`

## Quick Start

From the repo root:

```bash
export FLASH_MOE_MODEL=$HOME/Models/flash_mlx_4bit
python3 autoresearch/extract_gguf_embedding.py
python3 autoresearch/extract_gguf_full_attn_overlay.py
python3 autoresearch/extract_gguf_lm_head.py
python3 autoresearch/extract_gguf_linear_overlay.py
python3 autoresearch/extract_gguf_qkv_overlay.py
python3 autoresearch/init_run.py --tag mar20 --allow-dirty
python3 autoresearch/sweep_gguf_tensors.py --write-markdown docs/gguf-q3-tensor-sweep.md
python3 autoresearch/run_experiment.py --json --save-baseline > autoresearch/last_result.json
cat autoresearch/last_result.json
```

Then launch your coding agent in this repo and point it at `program.md`.

## Environment Variables

- `FLASH_MOE_MODEL`
  - path to the converted Flash-MoE model directory
- `FLASH_MOE_PPL_TOKENS`
  - optional override for the short perplexity token file
- `FLASH_MOE_FULL_PPL_TOKENS`
  - optional override for the periodic/full perplexity token file
- `FLASH_MOE_GGUF_EMBEDDING`
  - optional override for the extracted GGUF embedding blob
- `FLASH_MOE_GGUF_FULL_ATTN_BIN`
  - optional override for the extracted GGUF full-attention overlay blob
- `FLASH_MOE_GGUF_FULL_ATTN_JSON`
  - optional override for the full-attention overlay manifest JSON
- `FLASH_MOE_GGUF_LM_HEAD`
  - optional override for the extracted GGUF LM head blob
- `FLASH_MOE_GGUF_QKV_BIN`
  - optional override for the extracted GGUF QKV overlay blob
- `FLASH_MOE_GGUF_QKV_JSON`
  - optional override for the QKV overlay manifest JSON
- `FLASH_MOE_GGUF_LINEAR_BIN`
  - optional override for the extracted GGUF linear gate/out overlay blob
- `FLASH_MOE_GGUF_LINEAR_JSON`
  - optional override for the linear gate/out overlay manifest JSON

If `FLASH_MOE_MODEL` is unset, the harness tries a few common local paths.

For GGUF inspection, the harness defaults to the configured local `llama.cpp` checkout and performs metadata-only reads across the shard set.

## Results File

The harness expects an untracked TSV at:

```text
autoresearch/results.tsv
```

Header:

```text
commit	score	decode_tok_s	decode_vs_4bit_pct	prefill_tok_s	ppl	ppl_vs_4bit	full_ppl	status	description
```

`status` should be one of:

- `keep`
- `discard`
- `crash`

## Baseline Comparison

`run_experiment.py` can save and compare against a 4-bit baseline JSON:

```bash
python3 autoresearch/run_experiment.py --json --save-baseline > autoresearch/last_result.json
```

Subsequent runs include `vs_baseline` deltas in their JSON output, including:

- `decode_tok_s_pct`
- `prefill_tok_s_pct`
- `ppl_abs`

For periodic full checks:

```bash
python3 autoresearch/run_experiment.py --json --full-check > autoresearch/last_result.json
```

For tensor-level debugging:

```bash
python3 autoresearch/extract_gguf_embedding.py
python3 autoresearch/extract_gguf_full_attn_overlay.py --roles q,k,v
python3 autoresearch/extract_gguf_linear_overlay.py
python3 autoresearch/extract_gguf_lm_head.py
python3 autoresearch/extract_gguf_qkv_overlay.py
python3 autoresearch/run_experiment.py --json --gguf-embedding autoresearch/gguf/embedding_q8_0.bin
python3 autoresearch/run_experiment.py --json --gguf-embedding '' --gguf-lm-head '' --gguf-qkv-bin '' --gguf-qkv-json '' --gguf-full-attn-bin autoresearch/gguf/full_attn_qkv_only_q8_0.bin --gguf-full-attn-json autoresearch/gguf/full_attn_qkv_only_q8_0.json
python3 autoresearch/run_experiment.py --json --gguf-embedding '' --gguf-lm-head '' --gguf-qkv-bin autoresearch/gguf/attn_qkv_q8_0.bin --gguf-qkv-json autoresearch/gguf/attn_qkv_q8_0.json
python3 autoresearch/sweep_gguf_tensors.py --tensor output.weight
python3 autoresearch/sweep_gguf_tensors.py --tensor token_embd.weight
python3 autoresearch/sweep_gguf_tensors.py --tensor blk.0.attn_qkv.weight
python3 autoresearch/sweep_gguf_tensors.py --tensor 'blk.27.*'
```
