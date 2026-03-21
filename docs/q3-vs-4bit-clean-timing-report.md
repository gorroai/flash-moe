# Q3-GGUF vs 4-bit Clean Timing Report

Date: `2026-03-20`

## Goal

Resolve whether the finished streamed `Q3-GGUF` expert path really has worse expert I/O than the original `4-bit` path.

An earlier ad hoc timing result suggested `Q3-GGUF` was slower on `Expert I/O (SSD)`. That comparison happened while other local tests were also running, so it was not trustworthy.

## Method

Ran clean serial timing comparisons on the same machine, one inference at a time, with the same prompt and `200` generated tokens.

Order:

- `4bit_a`
- `q3_a`
- `q3_b`
- `4bit_b`

Commands:

```bash
./metal_infer/infer \
  --model /Users/anemll/Models/flash_mlx_4bit \
  --prompt "What is Apple Neural Engine?" \
  --tokens 200 \
  --timing

./metal_infer/infer \
  --model /Users/anemll/Models/flash_mlx_4bit \
  --q3-experts \
  --prompt "What is Apple Neural Engine?" \
  --tokens 200 \
  --timing
```

Raw logs:

- `/tmp/flash_moe_seq_logs_clean/4bit_a.log`
- `/tmp/flash_moe_seq_logs_clean/4bit_b.log`
- `/tmp/flash_moe_seq_logs_clean/q3_a.log`
- `/tmp/flash_moe_seq_logs_clean/q3_b.log`

## Results

| Run | Expert I/O | Total/token | Decode | TTFT |
|---|---:|---:|---:|---:|
| `4bit_a` | `49.0 ms` | `104.1 ms` | `9.55 tok/s` | `1569 ms` |
| `4bit_b` | `52.2 ms` | `106.2 ms` | `9.36 tok/s` | `1652 ms` |
| `q3_a` | `43.1 ms` | `95.5 ms` | `10.39 tok/s` | `1401 ms` |
| `q3_b` | `31.2 ms` | `83.2 ms` | `11.91 tok/s` | `2337 ms` |

Average by mode:

| Mode | Expert I/O avg | Total/token avg | Decode avg | TTFT avg |
|---|---:|---:|---:|---:|
| `4-bit` | `50.6 ms` | `105.15 ms` | `9.46 tok/s` | `1610.5 ms` |
| `Q3-GGUF` | `37.15 ms` | `89.35 ms` | `11.15 tok/s` | `1869.0 ms` |

## Interpretation

The clean rerun does **not** support the idea that `Q3-GGUF` has structurally worse expert I/O.

Instead:

- `Q3-GGUF` showed lower average `Expert I/O (SSD)` than plain `4-bit`
- `Q3-GGUF` also showed better average steady-state decode throughput
- the earlier “Q3 I/O is worse” result was most likely contaminated by concurrent activity or run-state noise

## Why Alignment Is Not The Problem

The current `Q3-GGUF` routed expert layout is cleanly aligned:

- `4-bit` expert size: `7,077,888` bytes
- `Q3-GGUF` expert size: `5,439,488` bytes
- both are aligned to `4096` and `16384`
- the `IQ3_XXS`, `IQ4_XS`, and `Q5_K` component offsets are also aligned to `4096` and `16384`

So the current evidence does **not** point to:

- bad per-expert stride alignment
- padding inefficiency
- page-boundary misalignment

## Important Timing Caveat

`Expert I/O` in the timing output is not a pure disk-service metric. In the runtime it measures the whole expert-load phase until async preads complete, so overlap effects and nearby pipeline work can move this number somewhat.

That means:

- it is still a useful operational metric
- but it should be interpreted as “expert-load wait in pipeline context,” not raw SSD latency

## Bottom Line

- The finished streamed `Q3-GGUF` path is a valid throughput win over plain `4-bit` on this prompt family.
- The earlier claim that `Q3-GGUF` had worse expert I/O should be considered superseded.
- The main remaining tradeoff is quality:
  - plain `4-bit` full 2k PPL: `3.64`
  - `Q3-GGUF` full 2k PPL: `3.81`
