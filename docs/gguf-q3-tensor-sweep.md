# GGUF Q3 Tensor Sweep

Metadata-only sweep. This does not instantiate the full GGUF model.

- Shards scanned: 5
- Tensors scanned: 1098
- Source: `/Users/anemll/Models/Qwen3.5/Qwen3.5-397B-A17B-GGUF-UD-Q3_K_XL/Qwen3.5-397B-A17B-UD-Q3_K_XL-00001-of-00005.gguf`

## Quant Types
| Quant | Block | Bytes/Block | Tensors | Total GiB |
|---|---:|---:|---:|---:|
| IQ3_XXS | 256 | 98 | 118 | 90.344 |
| IQ4_XS | 256 | 136 | 61 | 64.812 |
| Q8_0 | 32 | 34 | 459 | 8.447 |
| Q5_K | 256 | 176 | 1 | 1.375 |
| Q6_K | 256 | 210 | 1 | 0.777 |
| F32 | 1 | 4 | 451 | 0.480 |
| BF16 | 1 | 2 | 7 | 0.219 |

## Outliers
| Tensor | Shard | Quant | Block | Shape | Bytes |
|---|---|---|---:|---|---:|
| `blk.27.attn_k.weight` | `Qwen3.5-397B-A17B-UD-Q3_K_XL-00003-of-00005.gguf` | BF16 | 1 | `4096x512` | 4194304 |
| `blk.27.attn_output.weight` | `Qwen3.5-397B-A17B-UD-Q3_K_XL-00003-of-00005.gguf` | BF16 | 1 | `8192x4096` | 67108864 |
| `blk.27.attn_q.weight` | `Qwen3.5-397B-A17B-UD-Q3_K_XL-00003-of-00005.gguf` | BF16 | 1 | `4096x16384` | 134217728 |
| `blk.27.attn_v.weight` | `Qwen3.5-397B-A17B-UD-Q3_K_XL-00003-of-00005.gguf` | BF16 | 1 | `4096x512` | 4194304 |
| `blk.27.ffn_down_exps.weight` | `Qwen3.5-397B-A17B-UD-Q3_K_XL-00003-of-00005.gguf` | Q5_K | 256 | `1024x4096x512` | 1476395008 |
| `blk.27.ffn_down_shexp.weight` | `Qwen3.5-397B-A17B-UD-Q3_K_XL-00003-of-00005.gguf` | BF16 | 1 | `1024x4096` | 8388608 |
| `blk.27.ffn_gate_shexp.weight` | `Qwen3.5-397B-A17B-UD-Q3_K_XL-00003-of-00005.gguf` | BF16 | 1 | `4096x1024` | 8388608 |
| `blk.27.ffn_up_shexp.weight` | `Qwen3.5-397B-A17B-UD-Q3_K_XL-00003-of-00005.gguf` | BF16 | 1 | `4096x1024` | 8388608 |
| `output.weight` | `Qwen3.5-397B-A17B-UD-Q3_K_XL-00002-of-00005.gguf` | Q6_K | 256 | `4096x248320` | 834355200 |

## Key Templates
| Template | Residency | Quant | Count | Total GiB |
|---|---|---|---:|---:|
| `blk.*.attn_gate.weight` | resident_dense | Q8_0 x45 | 45 | 1.494 |
| `blk.*.attn_k.weight` | resident_dense | BF16 x1, Q8_0 x14 | 15 | 0.033 |
| `blk.*.attn_output.weight` | resident_dense | BF16 x1, Q8_0 x14 | 15 | 0.527 |
| `blk.*.attn_q.weight` | resident_dense | BF16 x1, Q8_0 x14 | 15 | 1.055 |
| `blk.*.attn_qkv.weight` | resident_dense | Q8_0 x45 | 45 | 2.241 |
| `blk.*.attn_v.weight` | resident_dense | BF16 x1, Q8_0 x14 | 15 | 0.033 |
| `blk.*.ffn_down_shexp.weight` | shared_expert | BF16 x1, Q8_0 x59 | 60 | 0.253 |
| `blk.*.ffn_gate_shexp.weight` | shared_expert | BF16 x1, Q8_0 x59 | 60 | 0.253 |
| `blk.*.ffn_up_shexp.weight` | shared_expert | BF16 x1, Q8_0 x59 | 60 | 0.253 |
| `blk.*.ssm_out.weight` | resident_dense | Q8_0 x45 | 45 | 1.494 |
| `token_embd.weight` | embedding | Q8_0 x1 | 1 | 1.006 |
| `output.weight` | lm_head | Q6_K x1 | 1 | 0.777 |
| `blk.*.ffn_down_exps.weight` | streamed_expert | IQ4_XS x59, Q5_K x1 | 60 | 64.062 |
| `blk.*.ffn_gate_exps.weight` | streamed_expert | IQ3_XXS x59, IQ4_XS x1 | 60 | 46.234 |
| `blk.*.ffn_up_exps.weight` | streamed_expert | IQ3_XXS x59, IQ4_XS x1 | 60 | 46.234 |

## Suggested Start Order
- `output.weight` first: isolated LM head path and exact `Q6_K` kernel work.
- `Q8_0` resident dense tensors next: attention, SSM, and shared expert tensors.
- Keep `packed_experts_Q3/` separate for routed-expert experiments, matching the 2-bit workflow.
- Short PPL plus a short generation smoke test on every iteration.
