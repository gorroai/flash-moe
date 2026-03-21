# Cache I/O Split Experiment

Date: 2026-03-20

Test machine for the measurements below:

- MacBook Pro, Apple M5 Max
- unified-memory Apple Silicon system

This note documents the experimental `--cache-io-split N` flag added to `metal_infer/infer`.

Credit:

- this experiment was inspired by Daniel Pacary's "rustane" cached-read fanout work
- reference repo: [ncdrone/rustane](https://github.com/ncdrone/rustane)
- this implementation is not a code import from rustane; it is a Flash-MoE-native experiment built on top of the existing async routed-expert `pread()` path

Intent:

- force a higher-fanout routed-expert `pread()` workflow without changing quantization or routing
- test whether more concurrent page-cache-backed reads improve steady-state decode throughput
- mimic the spirit of recent "rustane"-style cached-read fanout experiments while staying inside the current Flash-MoE pipeline

What the flag does:

- only affects the async routed-expert load path
- splits each routed expert blob into `N` page-aligned chunks
- dispatches those chunk reads concurrently through the existing async `pread()` path
- keeps the current storage layout, OS page cache policy, and Metal expert kernels unchanged

Current implementation details:

- page-aligned split size uses `16 KiB` boundaries
- split count is clamped to `[1, 8]`
- active only when `--cache-io-split N` is passed with `N > 1`
- default behavior remains `1`, which is equivalent to the old one-`pread`-per-expert path
- when active, startup prints:
  - `[tiered-io] Experimental cache fanout: split routed expert preads into N page-aligned chunks`

Hardware note:

- the measurements in this document were taken on an `M5 Max`
- do not assume the same best split value on `M3 Max` or `M4 Max` without rerunning the sweep

Recommended value right now:

- `--cache-io-split 4`

Reason:

- in every clean timing sweep below, `4` was the best value tested
- `8` helped less than `4`, suggesting the machine is getting most of the benefit by `4` and then paying extra scheduling overhead beyond that

## Commands

Q3 routed experts only:

```bash
./metal_infer/infer \
  --model /Users/anemll/Models/flash_mlx_4bit \
  --q3-experts \
  --cache-io-split 4 \
  --prompt "What is Apple Neural Engine?" \
  --tokens 200 \
  --timing
```

Plain 4-bit baseline with the same experimental fanout:

```bash
./metal_infer/infer \
  --model /Users/anemll/Models/flash_mlx_4bit \
  --cache-io-split 4 \
  --prompt "What is Apple Neural Engine?" \
  --tokens 200 \
  --timing
```

All currently implemented GGUF files plus routed Q3 experts:

```bash
./metal_infer/infer \
  --model /Users/anemll/Models/flash_mlx_4bit \
  --q3-experts \
  --cache-io-split 4 \
  --gguf-embedding /Users/anemll/Models/flash_mlx_4bit/gguf/embedding_q8_0.bin \
  --gguf-lm-head /Users/anemll/Models/flash_mlx_4bit/gguf/lm_head_q6.bin \
  --gguf-qkv-bin /Users/anemll/Models/flash_mlx_4bit/gguf/attn_qkv_q8_0.bin \
  --gguf-qkv-json /Users/anemll/Models/flash_mlx_4bit/gguf/attn_qkv_q8_0.json \
  --gguf-full-attn-bin /Users/anemll/Models/flash_mlx_4bit/gguf/full_attn_q8_0.bin \
  --gguf-full-attn-json /Users/anemll/Models/flash_mlx_4bit/gguf/full_attn_q8_0.json \
  --gguf-linear-bin /Users/anemll/Models/flash_mlx_4bit/gguf/linear_q8_0.bin \
  --gguf-linear-json /Users/anemll/Models/flash_mlx_4bit/gguf/linear_q8_0.json \
  --prompt "What is Apple Neural Engine?" \
  --tokens 200 \
  --timing
```

## Results

### Q3 routed experts sweep

Warm-cache sequential sweep, `200` generated tokens, same prompt, `--q3-experts`:

| Split | Decode tok/s | Expert I/O ms/token | Expert compute ms/token | Total ms/token | TTFT ms |
|---|---:|---:|---:|---:|---:|
| `1` | `11.04` | `36.1` | `1.5` | `89.9` | `2617` |
| `2` | `12.07` | `29.0` | `1.5` | `82.2` | `1709` |
| `4` | `13.33` | `25.6` | `1.3` | `74.4` | `1531` |
| `8` | `12.04` | `27.3` | `1.6` | `82.3` | `1541` |

Takeaway:

- `4` is the best point in this sweep
- compared with split `1`, split `4` improved:
  - decode throughput by about `20.7%`
  - expert I/O time by about `29.1%`
  - total decode time per token by about `17.2%`

### All currently implemented GGUF files

Warm-cache sequential comparison, `200` generated tokens:

| Split | Decode tok/s | Expert I/O ms/token | Expert compute ms/token | Total ms/token | TTFT ms |
|---|---:|---:|---:|---:|---:|
| `1` | `7.39` | `40.5` | `23.9` | `134.1` | `2323` |
| `4` | `7.95` | `31.2` | `23.2` | `124.5` | `2660` |

Takeaway:

- the full current GGUF stack also benefits
- split `4` reduced routed expert I/O stall by about `22.9%`
- decode improved by about `7.6%`

### Plain 4-bit routed experts

Warm-cache sequential comparison, `200` generated tokens:

| Split | Decode tok/s | Expert I/O ms/token | Expert compute ms/token | Total ms/token | TTFT ms |
|---|---:|---:|---:|---:|---:|
| `1` | `9.34` | `46.8` | `1.6` | `106.3` | `1365` |
| `4` | `10.93` | `32.9` | `1.7` | `90.7` | `1477` |

Takeaway:

- this is not only a Q3/GGUF effect
- plain 4-bit also benefits strongly from split `4`
- compared with split `1`, split `4` improved:
  - decode throughput by about `17.0%`
  - expert I/O time by about `29.7%`
  - total decode time per token by about `14.7%`

## Interpretation

The current evidence supports three conclusions:

1. More page-cache-backed read fanout does help on this machine.
2. The best tested setting is `4`, not `8`.
3. The improvement is showing up in end-to-end inference, not only in a microbenchmark.

This experiment does not prove raw SSD bandwidth is higher. It shows that the current pipeline sees lower routed-expert load stall when each expert read is split into a few concurrent page-aligned chunk reads.

Most likely explanation:

- more concurrent cache-backed reads let the OS service warm expert pages at higher effective bandwidth
- the benefit saturates around `4`
- pushing to `8` starts adding queueing/scheduling overhead with less additional gain

## Current status

- flag is implemented and buildable
- recommended experimental value is `--cache-io-split 4`
- this should still be treated as experimental until it is repeated across more prompts and longer runs
