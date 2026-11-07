# QSA family — naming and layout contract (T095)

This directory is the local home of the upstream QSA family
(`src/ops/softmax_attention/qsa/`, upstream a754fcf3). The family was absent from this
tree (zero `qsa|Qwen4` hits) before T095, so the names below are taken over verbatim from
the upstream standardisation commit `d492ee78` (`[Qwen4Exp] Remove FlashAttention
dependency and standardize QSA naming`) mapped onto the local C++ layout.

## Split by responsibility (upstream `ops` three-way split)

| upstream (vLLM, after d492ee78) | local files | responsibility |
|---|---|---|
| `nvidia/ops/qsa.py` | `sparse_gqa.cpp`, `sparse_gqa_*.cuh`, `sparse_gqa_launch.cu` | sparse paged GQA main kernel |
| `nvidia/ops/qsa_indexer.py` | `indexer.cpp`, `indexer_*.cuh`, `indexer_launch.cu` | token selection (weight-free) |
| `nvidia/ops/qsa_prepare.py` | `prepare.cpp`, `prepare_*.cuh`, `prepare_launch.cu` | norm + RoPE + cache write |
| `nvidia/indexer_qsa.py` (`QSAIndexer`) | `indexer.cpp` (no separate module) | indexer driver |
| `common/qsa_cache.py` | `QsaIndexCacheView` in `include/ninfer/ops/qsa.h` | index cache ownership + metadata |
| `common/qsa.py` (`Qwen4ExpQSAMetadata*`) | `QsaBatchView`, `QsaBlockSelection` | platform-independent metadata |
| — | `tail_extract.cpp`, `tail_extract.cu` | raw index-key tail hand-off |
| — | `input_projection.cpp`, `input_projection_bf16.cu`, `input_projection_q8.cu` | merged Q/K/V/gate/index projections |

`Qwen4ExpQSABackend` / `Qwen4ExpQSAImpl` (upstream `nvidia/qsa.py`) are model execution
objects, not ops; they belong to the qwen4-exp execution line (T191+), not here.

## Naming rules

1. `qsa_*` (snake_case) = ops (`qsa_input_proj`, `qsa_prepare`, `qsa_indexer`,
   `qsa_sparse_gqa`, `qsa_tail_extract`).
2. `Qsa*` (PascalCase) = view/parameter structs (`QsaProjectionView`,
   `QsaPrepareParameters`, `QsaBatchView`, `QsaCacheView`, `QsaIndexCacheView`,
   `QsaRawRecordReadView`, `QsaRawRecordWriteView`, `QsaTailStateView`,
   `QsaBlockSelection`, `Qsa*ExecutionEnvelope`).
3. `kQsa*`/`kIndexer*`/`kSparse*` = constants, matching the local `k` prefix habit.
4. `indexer` always means "which tokens" (weight-free); `prepare` always means "ready Q/K"
   (norm + RoPE + KV encode); `cache` never names the indexer.

## Contracts carried over from d492ee78

- **KV ownership is explicit**: `forward_includes_kv_cache_update = False` upstream is
  localised as "`qsa_prepare` is the only writer of the main/index caches for this layer;
  `qsa_indexer` and `qsa_sparse_gqa` are read-only consumers of that same stream step".
  No op re-writes K/V behind the attention call.
- **No explanatory overrides**: the ops inherit nothing from a foreign attention base
  class. The only shared code is the local CUDA primitives (`ops/common/*`,
  `ops/kv_cache/*`, `ops/softmax_attention/common/*`).
- **Kernel naming**: `qsa_*` device kernels stay inside `ninfer::ops::detail::qsa`.

Directory-level `sources.cmake` registers the translation units in `ninfer_ops`; the
public surface lives in `include/ninfer/ops/qsa.h`.

## Quantized KV is opt-in (default off)

`qsa_prepare` and `qsa_sparse_gqa` accept Int8Group64, Fp8E4M3Row256, Nvfp4Group16 and
Fp8KeyNvfp4Value planes, selected solely by the caller's `cache.storage`. That capability
is compiled and qualified on sm_89, but it is **not** a default: the quantized path drops
the BF16 K rounding boundary (K enters the Hadamard/encoder directly from norm+RoPE) and
rotates the V basis, so its output is not bitwise equal to the BFloat16 path. Nothing on
the 27B/35B line selects QSA at all today; when a model line does, it must stay on
`KvCacheStorage::BFloat16` until a dedicated PPL (bitwise-where-claimed) + task-suite
evaluation clears the quantized arm.

