---
library_name: ninfer
pipeline_tag: image-text-to-text
inference: false
license: apache-2.0
base_model: Qwen/Qwen3.6-35B-A3B
base_model_relation: quantized
tags:
  - ninfer
  - qwen3.6
  - moe
  - expert-streaming
  - multimodal
  - gguf
  - cuda
  - rtx-4090
---

# Qwen3.6-35B-A3B-e2-NInfer

`Qwen3_5MoeForCausalLM` converted to the NInfer v3 container with **every routed expert as its own
object**, which is what the expert-streaming cache pages. It is the artifact `tools/gate.cmd`
measures, and the profile `scripts\serve_35b_e2_2k.bat` starts.

This is a layout variant of `Qwen3.6-35B-A3B-NInfer`, not a different model: same base weights, same
component set, a different tensor decomposition.

## Artifact

| | |
|---|---|
| file | `qwen3_6_35b_a3b_e2.ninfer` |
| bytes | 22,794,248,704 (21.2 GiB) |
| sha256 | `f81e9f2666ae1baf383275ad26f8424bf35fcdc854aa50d305bf2df963ff9a22` |
| container | v3, artifact id `c5065fc4b63448b7907e579c41ea2034` |
| architecture | `Qwen3_5MoeForCausalLM` (`model_type: qwen3_5_moe_text`) |
| components | text, vision, mtp, dflash |

## What "e2" changes

| | fused (`qwen3_6_35b_a3b.ninfer`) | this artifact |
|---|---|---|
| tensors | 934 | **21,844** |
| objects | 940 | 21,850 |
| `row_split_k128_v1` | 386 | 21,296 |
| q4_g64 / q5_g64 | 95 / 91 | 10,295 / 9,526 |

The routed experts are split into individual objects so the host/device expert cache can move or
drop one expert at a time; the fused artifact keeps them in grouped rows and has to move a whole
group. Runtime flags that matter for it: `NINFER_EXPERT_CACHE_MB` (the gate uses 12,288) and the
MoE CPU activation precision (`NINFER_MOE_CPU_INT8`).

## Conversion source

Unsloth's GGUF conversions of `Qwen/Qwen3.6-35B-A3B`, with the `z-lab/Qwen3.6-35B-A3B-DFlash`
draft folded in (Apache-2.0).

`source.revision` is `null` and `conversion.record` is `null`: the conversion record was not kept
next to this artifact, and guessing a revision would be worse than an empty field. The base model and
the GGUF vendor are recorded from the operator's note, not read out of the file.

## Running it

From a packaged release folder (`scripts\serve_35b_e2_2k.bat` on port 11438):

```
ninfer-serve qwen3_6_35b_a3b_e2.ninfer ^
  --host 127.0.0.1 --port 11438 --max-context 2048 ^
  --max-concurrency 1 --max-pending-requests 16 --prefill-chunk 1024
```

## Gate

`tools/gate.cmd` runs the KV seam tests and a MoE perplexity on this artifact, expecting
**7.776858** within 0.5%. `gate.cmd record` re-baselines it into `.temp\kv\gate_golden_ppl.txt`, which
the third step then compares bit for bit.

Measured on the sm_89 port at `5caf0f0b`.
