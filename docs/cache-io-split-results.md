# Cache I/O Split Quick Results ( M5 MAX 128G 400B model)

Best current setting:

- `--cache-io-split 4`
- current implementation: persistent worker pool + `8` I/O workers

Terminology:

- `split=1` means **no chunk fanout** inside the current implementation:
  - one contiguous `pread()` task per expert
  - same total bytes
  - no extra chunking
- omitting `--cache-io-split` is the same as using `split=1`
- but `split=1` is **not** automatically the same as the older pre-feature measurements, because the base async I/O implementation changed when the persistent worker pool was added

Measured on:

- MacBook Pro, Apple M5 Max

Credit:

- inspired by Daniel Pacary's "rustane" cached page-cache fanout experiments
- reference repo: [ncdrone/rustane](https://github.com/ncdrone/rustane)

## Before Persistent Pool

First split-fanout prototype:

- split expert reads into page-aligned chunks
- launched chunk reads with per-chunk GCD async tasks

Warm-cache `200`-token results:

| Configuration | Split-1 decode tok/s | Split-4 decode tok/s | Split-1 expert I/O ms | Split-4 expert I/O ms |
|---|---:|---:|---:|---:|
| Plain `4-bit` | `9.34` | `10.93` | `46.8` | `32.9` |
| `Q3-GGUF` experts | `11.04` | `13.33` | `36.1` | `25.6` |
| Full current GGUF stack | `7.39` | `7.95` | `40.5` | `31.2` |

Takeaway:

- the original split prototype showed a strong win
- but it still paid noticeable per-chunk scheduling overhead

## After Persistent Pool

Current implementation:

- split expert reads into page-aligned chunks
- runs the async routed-expert path through the persistent worker pool
- pool widened to `8` I/O workers

Warm-cache `200`-token results:

| Configuration | Split-1 decode tok/s | Split-4 decode tok/s | Split-1 expert I/O ms | Split-4 expert I/O ms |
|---|---:|---:|---:|---:|
| Plain `4-bit` | `9.81` | `10.95` | `46.8` | `34.8` |
| `Q3-GGUF` experts | `11.71` | `12.52` | `32.9` | `27.5` |
| Full current GGUF stack | `8.79` | `9.41` | `36.6` | `27.5` |

Takeaway:

- split `4` improved every tested routed-expert configuration
- split `8` was worse than split `4` in the Q3 sweep
- the best absolute decode result in the current sweep is `12.52 tok/s` on `Q3-GGUF` with `split=4`
- the current implementation also benefits from the lower-overhead persistent worker pool even before extra fanout

Current decode summary, default vs explicit fanout:

| Configuration | No `--cache-io-split` (`split=1`) | `--cache-io-split 4` |
|---|---:|---:|
| Plain `4-bit` | `9.81 tok/s` | `10.95 tok/s` |
| `Q3-GGUF` experts | `11.71 tok/s` | `12.52 tok/s` |
| Full current GGUF stack | `8.79 tok/s` | `9.41 tok/s` |

## Repeated A/B Average

To make sure the persistent-pool version was not accidentally slower, the `Q3-GGUF` path was rerun `3` times each for:

- old implementation at commit `ec53c39`
- current local persistent-pool implementation
- `split=1`
- `split=4`

Each measured run used:

- a short warmup immediately beforehand
- the same prompt
- `200` generated tokens
- alternating old/new run order to reduce drift

Average results:

| Impl | Split | Decode tok/s | Expert I/O ms | Total ms/token | TTFT ms |
|---|---:|---:|---:|---:|---:|
| old | `1` | `11.33` | `33.7` | `87.6` | `1513` |
| old | `4` | `11.61` | `26.7` | `85.7` | `1547` |
| new | `1` | `11.80` | `32.6` | `84.1` | `1546` |
| new | `4` | `12.17` | `27.9` | `81.5` | `1543` |

What this means:

- the persistent-pool version is not slower on average
- it improved both the baseline path and the split-4 path
- the apparent “split bonus” got smaller mainly because `split=1` also got faster

Most important comparison:

- old `split=4`: `11.61 tok/s`
- new `split=4`: `12.17 tok/s`

So the current implementation remains the one to keep.

## What Changed

Comparing the two implementations:

- the persistent pool raised the `split=1` baseline across all tested configurations
- that means some of the original win was really “less scheduling overhead,” not only “more chunk fanout”
- `split=4` still remains the best tested setting
- the marginal gain from `split=1 -> split=4` is smaller now because the base async path is healthier
- repeated old-vs-new averages confirm the new implementation is faster overall, not slower

Short version:

- before persistent pool: bigger apparent split-fanout win, but noisier and more overhead-heavy
- after persistent pool: better base path, cleaner implementation, `split=4` still best

Recommended test command:

```bash
./metal_infer/infer \
  --model /Users/anemll/Models/flash_mlx_4bit \
  --q3-experts \
  --cache-io-split 4 \
  --prompt "What is Apple Neural Engine?" \
  --tokens 200 \
  --timing
```

## Standalone Cachebench

For true warm-page-cache `pread()` bandwidth, use the standalone microbench instead of full inference:

```bash
cd /Users/anemll/SourceRelease/GITHUB/ML_playground/flash-moe-org/metal_infer
make cachebench
./cachebench --q3-experts --layer 0 --experts 128 --threads 8 --split 4 --warmup 2 --passes 5
```

Direct `split=1` vs `split=4` cachebench comparison on this M5 Max:

| Layout | Threads | Split | Avg GB/s | Avg GiB/s | Avg us/expert |
|---|---:|---:|---:|---:|---:|
| `Q3-GGUF` | `8` | `1` | `62.46` | `58.17` | `87.1` |
| `Q3-GGUF` | `8` | `4` | `72.07` | `67.12` | `75.5` |
| `4-bit` | `8` | `1` | `63.72` | `59.35` | `111.1` |
| `4-bit` | `8` | `4` | `71.08` | `66.20` | `99.6` |
| `2-bit` | `8` | `4` | `76.92` | `71.64` | `51.1` |

Current M5 Max cached-read headline numbers:

| Layout | Threads | Split | Avg GB/s | Avg GiB/s |
|---|---:|---:|---:|---:|
| `Q3-GGUF` | `8` | `4` | `72.07` | `67.12` |
| `4-bit` | `8` | `4` | `71.08` | `66.20` |
| `2-bit` | `8` | `4` | `76.92` | `71.64` |

Fresh short rerun sanity check for `Q3-GGUF`, `8` threads, `split=4`, `3` timed passes:

- `73.46 GB/s`
- `68.41 GiB/s`
- `74.0 us/expert`

This is effective cached bandwidth, not raw NAND bandwidth. Full details and thread-scaling results are in [cache-pread-microbench.md](./cache-pread-microbench.md).
