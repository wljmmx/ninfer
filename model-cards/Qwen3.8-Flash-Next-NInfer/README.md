---
library_name: ninfer
pipeline_tag: image-text-to-text
inference: false
license: apache-2.0
base_model: Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
tags:
  - ninfer
  - qwen3.8
  - qwen4-exp
  - sparse-attention
  - qsa
  - multimodal
  - moe
  - gguf
  - cuda
  - rtx-4090
---

# Qwen3.8-Flash-Next-NInfer

`Qwen4ExpForCausalLM` converted to the NInfer v3 container — the only artifact in this repository
that runs the **QSA** route (`qsa_prepare` → `qsa_indexer` → `qsa_sparse_gqa`). Twelve of its 48 text
layers are `indexed_attention`; the rest are `linear_attention`.

Text, vision, MTP and DFlash2 draft are one file; there is no separate draft artifact to load.

## Artifact

| | |
|---|---|
| file | `qwen3_8_flash_next_v5_e2.ninfer` |
| bytes | 123,931,846,144 (115.4 GiB) |
| sha256 | `0d621e204a5f5156c08d5042adb0bc738555ed58ae86189167370e183ddfdccb` |
| container | v3, artifact id `91f7429b090f4107a60e5b38c68ce290` |
| architecture | `Qwen4ExpForCausalLM` (`model_type: qwen4_exp_text`) |
| components | text, vision, mtp, dflash2 |

## Geometry

| | |
|---|---|
| text layers | 48 (36 linear_attention + 12 indexed_attention) |
| hidden / vocab | 2,560 / 248,320 |
| attention | 24 query heads, 2 KV heads, head_dim 256 |
| experts | 512 per layer, top-10, intermediate 640 |
| indexer | budget 2,048 keys, 4 heads × 128, compression ratio 4 |
| context | 262,144 positions |
| draft | DFlash2, 5 layers, 7 draft tokens recommended (1–15) |

## Conversion source

Unsloth's GGUF conversions of `Qwen/Qwen3.8-Flash-Next` and of
`incoai/Qwen3.8-Flash-Next-DFlash2` (Apache-2.0):

- weights — `Qwen3.8-Flash-Next-UD-Q4_K_XL-0000{1..4}-of-00004.gguf` plus `mmproj-BF16.gguf`
- draft — `Qwen3.8-Flash-Next-DFlash2-BF16.gguf`

`source.revision` in the manifest is `null`: the download into the local model folder did not record
a repository revision, and a guessed one would be worse than an empty one.

Formats inside the container: bf16 633, fp32 72, q4_g64_fp16 314, q5_g64_fp16 501, q6_g64_fp16 50,
q8_g32_fp16 24; layouts `contiguous_le_v1` 705, `row_split_k128_v1` 889.

## Running it

From a packaged release folder (`scripts\serve_flashnext_2k.bat` on port 11439):

```
ninfer-serve qwen3_8_flash_next_v5_e2.ninfer ^
  --host 127.0.0.1 --port 11439 --max-context 2048 ^
  --max-concurrency 1 --max-pending-requests 16 --prefill-chunk 1024 --kv-dtype bf16
```

`--kv-dtype bf16` is explicit on purpose: `qsa_sparse_gqa` admits bf16 and nothing else unless the
low-bit QSA profiles are opted in with `NINFER_QSA_SPARSE_GQA_QUANT_KV=1`, and those profiles are
covered by the operator tests (`test_sparse_gqa --kv-cache fp8|int8|nvfp4|k8v4|rk*`) rather than by
this artifact.

Perplexity over a fixed corpus:

```
ninfer-perplexity qwen3_8_flash_next_v5_e2.ninfer --text <corpus> --context 512 --stride 256 ^
  --kv-dtype bf16
```

## Gate

`tools/release/gate_qsa.ps1` measures this artifact (context 512, stride 256, bf16) and compares
against `15.533755882784398` within 0.1%. It also refuses any artifact that declares no
`indexed_attention` layer, so it cannot pass without touching the QSA kernels.

Recorded 2026-11-19 on the sm_89 port at `5caf0f0b`; the value repeats to delta 0.000% in about a
minute.

## Notes on this port

- Built and gated on RTX 4090 (sm_89) / Windows 11 / CUDA 13.5; the artifact itself is
  architecture-neutral.
- 512 experts per layer with a 115 GiB artifact means the experts stream; give
  `NINFER_EXPERT_CACHE_MB` a real budget (the gate uses 12,288) or the first tokens pay for the
  upload.
- Every field in `artifact-manifest.json` above the `geometry` block is measured from the file or
  taken from the conversion record; only `source.revision` is unknown and is left `null` on purpose.
