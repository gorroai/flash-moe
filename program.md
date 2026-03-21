# Flash-MoE Autoresearch

This repo is adapted to run in a Karpathy-style autoresearch loop for systems optimization.

The current goal is to build a hybrid GGUF path inside Flash-MoE:

- keep the existing 4-bit model as the base
- replace one tensor or one layer family at a time with GGUF-backed quantized weights
- preserve GGUF block sizes exactly
- mirror the existing 2-bit mixed-quant test flow
- use `packed_experts_Q3/` for new Q3 expert artifacts
- track short-PPL progress and speed against the current 4-bit baseline
- preserve any Qwen3.5 GGUF row or column layout transforms that local `llama.cpp` applies for the tensor family under test
- for full-attention work, treat the resident family as a block (`q+k+v+o`) with native fallback for BF16 outliers instead of optimizing isolated tensors in a vacuum

The project notebook for this effort is:

- `docs/gguf-hybrid-bringup-log.md`

Update it whenever the GGUF plan, supported tensor types, or validation status changes in a meaningful way.

## Shared Experts

The human has explicitly re-opened shared-expert optimization.

- Shared-expert overlays are now allowed.
- Shared-expert kernel work is now allowed.
- Treat shared experts as their own experiment family and validate them separately from routed streamed experts.
- Keep cache sensitivity in mind: shared-expert work can still interact with resident memory pressure and page-cache behavior.
- Prefer small, isolated shared-expert experiments before broad rollouts.
- Streamed routed experts are still a later step unless the human explicitly changes priorities again.

## Setup

To set up a fresh run, work with the human to:

1. Agree on a short run tag, typically based on today's date, for example `mar20`.
2. Start from a clean git state if possible. If the worktree is dirty, stop and ask the human whether to continue with `--allow-dirty`.
3. Initialize the run:

```bash
python3 autoresearch/init_run.py --tag mar20 --allow-dirty
```

This creates a new branch named `autoresearch/<tag>` and initializes `autoresearch/results.tsv`.

4. Read these files for context:
   - `program.md`
   - `autoresearch/README.md`
   - `autoresearch/config.json`
   - `docs/gguf-hybrid-bringup-log.md`
   - `autoresearch/run_experiment.py`
   - `metal_infer/infer.m`
   - `metal_infer/shaders.metal`
   - `docs/io-and-gpu-exploration.md`
   - `docs/optimization-experiments-q4.md`

5. Verify benchmark assets exist:
   - A converted model directory, ideally via `FLASH_MOE_MODEL=/path/to/flash_mlx_4bit`
   - `ppl_tokens.bin` at repo root, or `FLASH_MOE_PPL_TOKENS=/path/to/ppl_tokens.bin`
   - Optionally `ppl_tokens_2k.bin` for periodic full checks
   - The reference GGUF file from `autoresearch/config.json`
   - Local `llama.cpp` checkout for metadata-only GGUF inspection

6. Sweep the GGUF shards before editing anything:

```bash
python3 autoresearch/sweep_gguf_tensors.py --write-markdown docs/gguf-q3-tensor-sweep.md
python3 autoresearch/sweep_gguf_tensors.py --tensor output.weight
```

This must stay metadata-only. Do not load the full GGUF model into memory.

7. Run the baseline first, before making any edits:

```bash
python3 autoresearch/run_experiment.py --json --save-baseline > autoresearch/last_result.json
cat autoresearch/last_result.json
```

8. Record the baseline in `autoresearch/results.tsv` with status `keep`:

```bash
python3 autoresearch/log_result.py --status keep --description "baseline 4-bit"
```

## Scope

Prefer changing only:

- `metal_infer/infer.m`
- `metal_infer/shaders.metal`
- `metal_infer/main.m`
- `packed_experts_Q3/` layout tooling
- shared-expert resident overlay or kernel tooling
- new GGUF support files that directly serve hybrid quantization experiments

Avoid editing the autoresearch harness itself unless the human explicitly asks for harness work.

Do not modify:

- `ppl_tokens.bin`
- historical `results.tsv` files already tracked in the repo
- model artifacts under `~/Models/`
- the GGUF block sizes from the source format
- the metadata-only safety rule for GGUF inspection

## Benchmark Contract

Use this command for every experiment:

```bash
python3 autoresearch/run_experiment.py --json > autoresearch/last_result.json
```

The harness:

- builds `metal_infer/infer`
- runs a fixed generation benchmark
- runs a short perplexity benchmark
- mirrors the 2-bit test pattern: quick smoke generation plus short PPL every run
- compares the result to the saved 4-bit baseline
- computes a `score`

The primary objective is `score`, which is:

- `decode_tok_s` if the quality gate passes
- `0.0` if the quality gate fails or the run crashes

Higher `score` is better.

Every few kept experiments, or before accepting a larger architecture change, run:

```bash
python3 autoresearch/run_experiment.py --json --full-check > autoresearch/last_result.json
```

## Results TSV

`autoresearch/results.tsv` is tab-separated with this header:

```text
commit	score	decode_tok_s	decode_vs_4bit_pct	prefill_tok_s	ppl	ppl_vs_4bit	full_ppl	status	description
```

Use:

- `keep` for improvements that advance the branch
- `discard` for valid runs that do not improve the best kept score
- `crash` for build/runtime failures

Do not commit `autoresearch/results.tsv`. It should stay untracked.

## Research Log

Keep `docs/gguf-hybrid-bringup-log.md` current.

Use it to record:

- the current GGUF migration plan
- exact quantization types and block sizes in scope
- major decisions, especially copy/adapt versus conversion choices
- validation outcomes for each new tensor path

This file is the durable memory for the hybrid GGUF effort.

## Experiment Loop

Once setup is complete, loop forever:

1. Inspect the current branch and latest kept result.
2. Make one focused experimental change.
3. Commit it.
4. Run:

```bash
python3 autoresearch/run_experiment.py --json > autoresearch/last_result.json
```

5. Parse `autoresearch/last_result.json`.
6. If the run crashed:
   - log a `crash` row in `autoresearch/results.tsv`
   - example:

```bash
python3 autoresearch/log_result.py --status crash --description "short description"
```

   - then revert to the previous kept commit
7. If the run completed but `score` is not better than the best kept score:
   - log a `discard` row
   - example:

```bash
python3 autoresearch/log_result.py --status discard --description "short description"
```

   - revert to the previous kept commit
8. If `score` improved:
   - log a `keep` row
   - example:

```bash
python3 autoresearch/log_result.py --status keep --description "short description"
```

   - advance the branch from this commit
9. Repeat immediately without asking the human whether to continue.

## Decision Rules

- Simpler wins if the score is meaningfully equal.
- Quality regressions are not acceptable just because throughput improves.
- Small wins that add a lot of fragile complexity are usually not worth keeping.
- If a run hangs or exceeds the harness timeout, treat it as a failure.
- If several attempts around one idea crash, abandon the idea and move on.

## Ideas That Are In Scope

- GGUF tensor parsing and metadata plumbing
- dequant kernel changes for exact GGUF block sizes
- LM head hybridization first
- 2-bit-style mixed-quant plumbing for new GGUF-backed tensors
- `packed_experts_Q3/` per-layer expert experiments
- persistent dense tensors before streamed experts
- command scheduling and overlap
- SSD I/O strategy
- buffer layout and alignment
- expert loading path
- CPU/GPU synchronization removal
- attention and delta-net plumbing

## GGUF Rules

Treat these block sizes as mandatory:

- `Q8_0`: `32`
- `Q6_K`: `256`
- `Q5_K`: `256`
- `IQ4_XS`: `256`
- `IQ3_XXS`: `256`

Do not normalize them into the existing MLX group size scheme just to make the code easier. Preserve the source format and adapt the kernels around it.

Use the local `llama.cpp` checkout only for metadata inspection and tensor debugging. Do not instantiate or load the full GGUF model.

Keep the experimental workflow close to the existing 2-bit path:

- separate artifact directory
- per-layer or per-tensor substitution
- short PPL on every run
- quick inference smoke test on every run

Start with the simplest targets:

- LM head
- always-resident dense tensors
- shared or persistent dense layers
- streamed experts only after the always-resident path is proven out
- use `packed_experts_Q3/` for streamed-expert Q3 artifacts

## Ideas That Already Failed

Read `docs/optimization-experiments-q4.md` before retrying old directions.

Especially avoid re-trying without a genuinely new angle:

- naive custom expert caches
- naive temporal expert prediction
- `F_RDADVISE` prefetch during GPU compute
- `dispatch_io`
- `aio_read`
- `mmap` for cold expert reads
- compression that adds CPU decode overhead to warm-cache reads

## Never Stop Rule

After the baseline is recorded, do not ask the human whether to continue. Keep running the keep/discard loop until the human interrupts you.
