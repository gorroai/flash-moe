# Warm Cache `pread()` Microbench

This repo now includes a standalone microbenchmark for the exact question:

- what is the maximum **effective cached expert-load bandwidth** on this box?
- how does it scale with thread count?
- how much does page-aligned chunk fanout help by itself, outside the full inference pipeline?

The tool is intentionally small and narrow:

- no tokenizer
- no attention
- no Metal kernels
- no inference loop
- just warm-page-cache `pread()` over packed expert files

That makes it the right tool for measuring **cached expert streaming bandwidth**, not end-to-end tokens/second.

## Build

```bash
cd /Users/anemll/SourceRelease/GITHUB/ML_playground/flash-moe-org/metal_infer
make cachebench
```

## Basic Usage

Default mode is the original 4-bit routed expert tree:

```bash
./cachebench \
  --model /Users/anemll/Models/flash_mlx_4bit \
  --layer 0 \
  --experts 128 \
  --threads 8 \
  --split 4 \
  --warmup 2 \
  --passes 5
```

Q3 routed experts:

```bash
./cachebench \
  --model /Users/anemll/Models/flash_mlx_4bit \
  --q3-experts \
  --layer 0 \
  --experts 128 \
  --threads 8 \
  --split 4 \
  --warmup 2 \
  --passes 5
```

2-bit routed experts:

```bash
./cachebench \
  --model /Users/anemll/Models/flash_mlx_4bit \
  --2bit \
  --layer 0 \
  --experts 128 \
  --threads 8 \
  --split 4 \
  --warmup 2 \
  --passes 5
```

Exact file override:

```bash
./cachebench \
  --file /Users/anemll/Models/flash_mlx_4bit/packed_experts_Q3/layer_00.bin \
  --threads 8 \
  --split 4 \
  --experts 128
```

## What The Threads Actually Do

`split=1`:

- one task per expert
- workers read different experts
- no worker is re-reading the same expert chunk on purpose

`split>1`:

- each expert is divided into `N` page-aligned chunks
- workers read different chunks of those experts
- this is not duplicate download work; it is more concurrent cached `pread()` work over the same total bytes

So for `128` experts and `split=4`, the benchmark schedules:

- `128 * 4 = 512` tasks
- same total bytes as `split=1`
- more in-flight page-cache-backed reads

That mirrors the same basic idea as `--cache-io-split` in `infer`, but without the rest of the model pipeline around it.

## Interpreting The Numbers

The tool prints:

- `GiB/s` and `GB/s`
- `us/expert`
- per-pass timings
- average, best, and worst pass

These are **effective cached-read bandwidth** numbers. They are not raw NAND bandwidth.

That is why `cachebench` can report numbers much higher than `iostat` shows for the physical SSD:

- warm page-cache hits are served from memory
- `iostat` only measures real disk traffic
- `cachebench` measures bytes delivered to the benchmarked read path

## M5 Max Results

Measured on this machine:

- `MacBook Pro`
- `Apple M5 Max`
- `128 GB` unified memory

Configuration:

- `layer_00.bin`
- `128` shuffled experts per pass
- `2` warmup passes
- `5` timed passes

### Q3-GGUF Routed Experts

| Threads | Split | Avg GB/s | Avg GiB/s | Avg us/expert |
|---|---:|---:|---:|---:|
| 1 | 1 | 19.49 | 18.15 | 279.1 |
| 2 | 1 | 33.91 | 31.58 | 160.4 |
| 4 | 1 | 44.95 | 41.86 | 121.0 |
| 8 | 1 | 62.46 | 58.17 | 87.1 |
| 8 | 4 | **72.07** | **67.12** | **75.5** |

### Original 4-bit Routed Experts

| Threads | Split | Avg GB/s | Avg GiB/s | Avg us/expert |
|---|---:|---:|---:|---:|
| 8 | 1 | 63.72 | 59.35 | 111.1 |
| 8 | 4 | **71.08** | **66.20** | **99.6** |

### 2-bit Routed Experts

| Threads | Split | Avg GB/s | Avg GiB/s | Avg us/expert |
|---|---:|---:|---:|---:|
| 8 | 4 | **76.92** | **71.64** | **51.1** |

## Takeaways

- This M5 Max can deliver about `~72 GB/s` effective cached `pread()` bandwidth on the Q3 routed expert path with `8` workers and `split=4`.
- The scaling curve is close to the pattern reported in `rustane`: one worker is near raw-storage speed, then multi-worker warm-cache service climbs sharply.
- The standalone microbench is faster than full inference because it removes:
  - attention work
  - GPU expert compute
  - unified-memory contention from the rest of the pipeline

## Relation To `infer --cache-io-split`

This tool is the best way to answer:

- how fast can the cached routed-expert read path go by itself?

`infer --cache-io-split` answers a different question:

- how much of that cached-read capability survives once the full inference pipeline is running?

Use both:

- `cachebench` for max cached-read bandwidth
- `infer --timing --cache-telemetry` for end-to-end impact

## Credit

This microbenchmark was added after comparing notes with Daniel Pacary’s `rustane` experiments on warm-page-cache `pread()` scaling. The implementation here is repo-local and tailored to Flash-MoE’s packed expert files and flags.
