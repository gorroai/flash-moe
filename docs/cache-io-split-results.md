# Cache I/O Split Quick Results ( M5 MAX 128G 400B model)

Best current setting:

- `--cache-io-split 4`

Measured on:

- MacBook Pro, Apple M5 Max

Credit:

- inspired by Daniel Pacary's "rustane" cached page-cache fanout experiments
- reference repo: [ncdrone/rustane](https://github.com/ncdrone/rustane)

Best observed gains at `200` decode tokens:

| Configuration | Baseline decode tok/s | Split-4 decode tok/s | Baseline expert I/O ms | Split-4 expert I/O ms |
|---|---:|---:|---:|---:|
| Plain `4-bit` | `9.34` | `10.93` | `46.8` | `32.9` |
| `Q3-GGUF` experts | `11.04` | `13.33` | `36.1` | `25.6` |
| Full current GGUF stack | `7.39` | `7.95` | `40.5` | `31.2` |

Short summary:

- split `4` improved every tested routed-expert configuration
- split `8` was worse than split `4` in the Q3 sweep
- the strongest win was on `Q3-GGUF` routed experts

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
