# RTX 4090 (sm_89) performance baseline — Qwen3.8-27B

Build: **wljmmx/ninfer `96b18187`** (`build-win`, `CMAKE_CUDA_ARCHITECTURES=89`), CUDA 13.4, Windows 11, single 24 GB RTX 4090.
Artifact: `qwen3_8_27b_v3.ninfer` (20,437,521,664 bytes, container v3); engine-reported weights 15.92 GiB.

Method follows the [UDPSendToFailed/ninfer-4090](https://github.com/UDPSendToFailed/ninfer-4090) standard
product benchmark matrix (`ninfer_bench`) and its `--wddm-evictable-budget` context-ceiling procedure.
This page is the reference point for later optimisation (KV tiering, sparsification, scheduling).

---

## 1. Standard product benchmark matrix (`ninfer_bench`)

`-r 5 --warmup 1 --prefill-chunk 1024 -o json`, greedy, corpus `bench/fixtures/bench_corpus.ids`:

```powershell
ninfer_bench --weights qwen3_8_27b_v3.ninfer --corpus bench/fixtures/bench_corpus.ids \
  -p 512,2048,4096 --kv-dtype int8
ninfer_bench ... -pg 32768,128 --kv-dtype rk4v4-e8 --spec mtp --draft-tokens 7 --lm-head-draft --max-ctx 40960
ninfer_bench ... -pg 2048,128  --kv-dtype int8 --spec mtp --draft-tokens 7 --lm-head-draft
ninfer_bench ... -pg 2048,128  --kv-dtype rk4v4-e8 --spec mtp --draft-tokens 7 --lm-head-draft
ninfer_bench ... -n 128 --kv-dtype rk4v4-e8 --spec mtp --draft-tokens 4 --lm-head-draft
ninfer_bench ... -n 128 --kv-dtype int8
```

| Test | `ninfer_bench` case | This build | ninfer-4090 (reference) |
|---|---|---:|---:|
| Prefill `pp512` (chunk 1024, INT8 KV) | `pp512` | **prefill 2386 tok/s** | 1,971.5 ± 6.4 tok/s |
| Prefill `pp2048` (chunk 1024, INT8 KV) | `pp2048` | **prefill 2664 tok/s** | 2,146.3 ± 3.0 tok/s |
| Prefill `pp4096` (chunk 1024, INT8 KV) | `pp4096` | **prefill 2626 tok/s** | 2,637.6 ± 3.3 tok/s |
| `pp32768+tg128` MTP7, rk4v4-e8 | `pp32768+tg128` | **prefill 2297 / decode 221.9 tok/s** (acc 100.0%, len 8.00) | 272.5 ± 0.7 tok/s (acc 100%, len 8.00) |
| `pp2048+tg128` MTP7, INT8 KV | `pp2048+tg128` | **prefill 2636 / decode 206.8 tok/s** (acc 88.0%, len 7.11) | 220.8 ± 23.5 tok/s (acc 88.0%, len 7.11) |
| `pp2048+tg128` MTP7, rk4v4-e8 | `pp2048+tg128` | **prefill 2635 / decode 206.1 tok/s** (acc 88.0%, len 7.11) | 226.0 ± 23.2 tok/s (acc 88.0%, len 7.11) |
| `tg128` MTP4, rk4v4-e8 (cold corpus) | `tg128` | **decode 87.0 tok/s** (acc 38.5%, len 2.51) | 79.5 ± 8.7 tok/s (acc 28.4%) |
| `tg128` MTP0, INT8 KV, CUDA Graph | `tg128` | **decode 48.7 tok/s** | 51.9 ± 2.2 tok/s |

Decoder acceptance is engine-reported (`speculative.acceptance_rate` / `acceptance_length`).
MTP7 ran at 100% acceptance with a licensed length of 8.00 on the deep-context case.

Do not read the two columns as a controlled A/B: the runs are not simultaneous, the desktop shares the
card, and this build measures the v3 artifact (`qwen3_8_27b_v3.ninfer`) while the reference README used
`qwen3_8_27b.ninfer`. Re-measure both on one machine for a like-for-like comparison.

---

## 2. Capacity · speed · precision by KV layout

| KV layout | PPL (ctx 4096) | vs int8 | KV bytes/token | KV payload @8192 | Ceiling (strict) | Ceiling (WDDM) | decode tok/s @8192 | prefill tok/s @8192 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `bf16` | 7.0035 | +0.02% | 63.5 KiB | 508 MiB | 38944 | 222028 | 47.5 | 2530 |
| `int8` | 7.0024 | +0.00% | 33.0 KiB | 264 MiB | 85568 | 135338 | 46.3 | 2510 |
| `fp8` | 7.0043 | +0.03% | 32.2 KiB | 258 MiB | 120288 | 261328 | 47.8 | 2540 |
| `k8v4` | 7.0162 | +0.20% | 25.1 KiB | 201 MiB | 108384 | 259016 | 47.8 | 2490 |
| `rk8v4` | 7.0127 | +0.15% | 25.0 KiB | 200 MiB | 103424 | 261328 | 48.0 | 2510 |
| `rk4v4` | 7.0450 | +0.61% | 17.0 KiB | 136 MiB | 142112 | 261328 | 48.0 | 2500 |
| `rk4v4-e8` | 7.0731 | +1.01% | 17.0 KiB | 136 MiB | 168896 | 261328 | 48.1 | 2490 |
| `rk2v4-e8` | 7.6048 | +8.60% | 13.0 KiB | 104 MiB | 189728 | 261328 | 47.9 | 2480 |

Sources:

- **Precision** — perplexity, `context 4096 / stride 1024`, 11,654 scored tokens, after the rank-compressed
  sign-extension fix (`194f7565`).
- **KV bytes/token** — measured from the engine's runtime reservation delta between 8192 and 16384, and matching
  the plane layout in `src/core/paged_kv_storage.h`.
- **Speed** — `ninfer-serve` in isolation (`--no-prefix-reuse --device-state-slots 0 --host-context-mib 0`,
  `--max-concurrency 1`, greedy, no speculation), 8192 context. Decode is weight-bandwidth bound, so KV layout
  moves it by less than 7%; the real difference between layouts is capacity.

---

## 3. Context ceilings

### 3.1 Two procedures, two different quantities

| Column | Procedure | Meaning |
|---|---|---|
| **strict** | binary search on `--max-context` / `--kv-capacity`, no eviction budget | engine budgets against the free device memory CUDA reports: safe, no over-commit |
| **WDDM** | same search, plus `--wddm-evictable-budget` | engine may budget against total VRAM minus a DWM floor, so WDDM can evict background apps and the pool may over-commit |

### 3.2 Measured ceilings

| KV layout | Ceiling (strict) | Ceiling (WDDM) | ninfer-4090 (no speculation) |
|---|---:|---:|---:|
| `bf16` | 38944 | 222028 | — |
| `int8` | 85568 | 135338 | 223,000 |
| `fp8` | 120288 | 261328 | — |
| `k8v4` | 108384 | 259016 | — |
| `rk8v4` | 103424 | 261328 | 294,000 |
| `rk4v4` | 142112 | 261328 | 433,000 |
| `rk4v4-e8` | 168896 | 261328 | 433,000 |
| `rk2v4-e8` | 189728 | 261328 | 567,000 |

### 3.3 Two findings worth recording

1. **The WDDM ceiling is not monotonic in KV size, so it must not be used to rank layouts.**
   Measured here `bf16` = 222,028 (64 KiB/token) while `int8` = 135,338 (33 KiB/token), and `fp8` = 261,328
   at essentially `int8`'s bytes per token. Under an over-committing budget the binding constraint is the
   eviction/over-commit path, not the KV footprint. Rank layouts with the **strict** column, or with the
   budget-normalised form in §3.4.
2. **Five layouts saturate at the same 261,328 tokens** — the model's position capacity (262,144, see the
   [Qwen3.5 model reference](../maintainer/qwen3_5-model.md)). `fp8`, `rk8v4`, `rk4v4`, `rk4v4-e8` and
   `rk2v4-e8` all stop there, so beyond that point the limit is the architecture's position capacity rather
   than memory. Any ceiling above 262,144 needs the position/RoPE path extended before it describes usable text.

### 3.4 Budget-normalised ceiling (comparable across layouts)

`max_context(B) = (B − fixed overhead) / measured bytes per token`, using each layout's measured overhead
(268–300 MiB). This removes the desktop-VRAM fluctuation that makes the raw strict column non-comparable
between runs.

| KV layout | bytes/token | 2.5 GiB | 3.0 GiB | 3.5 GiB |
|---|---:|---:|---:|---:|
| `bf16` | 63.5 KiB | 36,499 | 44,756 | 53,012 |
| `int8` | 33.0 KiB | 70,953 | 86,840 | 102,728 |
| `fp8` | 32.2 KiB | 68,869 | 85,151 | 101,433 |
| `k8v4` | 25.1 KiB | 88,364 | 109,252 | 130,140 |
| `rk8v4` | 25.0 KiB | 93,778 | 114,749 | 135,721 |
| `rk4v4` | 17.0 KiB | 137,825 | 168,666 | 199,506 |
| `rk4v4-e8` | 17.0 KiB | 138,056 | 168,896 | 199,736 |
| `rk2v4-e8` | 13.0 KiB | 180,079 | 220,409 | 260,739 |

---

## 4. Method and reproduction

- Isolation: `--no-prefix-reuse --device-state-slots 0 --host-context-mib 0 --max-concurrency 1 --greedy --prefill-chunk 1024`.
- Machine: single RTX 4090 24 GB, Windows 11, desktop sharing the card. The strict ceiling depends on how much
  VRAM the desktop holds at that moment; the WDDM ceiling depends on the eviction floor. Always report the
  granted budget with a ceiling — the engine prints it as `capacity | KV ... | runtime ... | free ...`.
- Ceilings binary-searched to ±1024 (strict) and ±2048 (WDDM) tokens.
- Raw records: `ninfer_bench` JSON reports (`-o json --output-file ...`) plus one engine startup log per probe.
- `ninfer_bench` needs the FFmpeg/curl runtime DLLs beside the executable, as the apps do.

## 5. How to use this as a baseline

1. **Capacity** — compare like with like: strict against strict, WDDM against WDDM. Never mix the two.
2. **Speed** — compare `decode_output_tok_s` and `prefill_tok_s` at the same context and the same speculation
   configuration. Decode is weight-bandwidth bound, so a KV-layout change alone should not move it much.
3. **Precision** — compare PPL at the same `context`/`stride`, and also report the `vs int8` delta.
4. **KV tiering / sparsification** — report the same columns plus `kv_payload_bytes` and TTFT, so capacity gain
   and latency cost stay separable.
