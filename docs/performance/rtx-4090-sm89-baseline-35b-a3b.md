# RTX 4090 (sm_89) performance baseline — Qwen3.6-35B-A3B

Build: **wljmmx/ninfer `ba33b85a`** — the shipped release binaries (`R:\infer`,
`v1.0.0-rtx4090`; `ninfer_bench.exe`), CUDA 13.4, Windows 11, single 24 GB RTX 4090 with the
desktop sharing the card.
Artifact: `qwen3_6_35b_a3b_e2.ninfer` (`R:\model`), 2.49 GB read; engine weights 2.49 GB;
MoE 40 banks × 256 experts, top-8, DFlash draft model.

This page is the 35B-A3B counterpart of the [Qwen3.8-27B baseline](rtx-4090-sm89-baseline.md)
and follows the same matrix, method and reporting rules.

---

## 1. Standard product benchmark matrix (`ninfer_bench`)

`-r 5 --warmup 1 --prefill-chunk 1024`, greedy, corpus `bench/fixtures/bench_corpus.ids`:

```powershell
set NINFER_EXPERT_CACHE_MB=19456   # full residency: every expert on the device
ninfer_bench --weights qwen3_6_35b_a3b_e2.ninfer --corpus bench/fixtures/bench_corpus.ids `
  -r 5 --warmup 1 --prefill-chunk 1024 -p 512,2048,4096 --kv-dtype int8
ninfer_bench ... -pg 2048,128 --kv-dtype int8      --spec dflash --draft-tokens 7 --lm-head-draft
ninfer_bench ... -pg 2048,128 --kv-dtype rk4v4-e8  --spec dflash --draft-tokens 7 --lm-head-draft
ninfer_bench ... -n 128       --kv-dtype rk4v4-e8  --spec dflash --draft-tokens 4 --lm-head-draft
ninfer_bench ... -pg 8192,128 --kv-dtype int8
ninfer_bench ... -pg 8192,128 --kv-dtype rk4v4-e8
ninfer_bench ... -n 128       --kv-dtype int8
```

| Test | `ninfer_bench` case | This build |
|---|---|---|
| Prefill `pp512` (chunk 1024, INT8 KV) | `pp512` | **prefill 8,671 tok/s** |
| Prefill `pp2048` | `pp2048` | **prefill 11,499 tok/s** |
| Prefill `pp4096` | `pp4096` | **prefill 11,434 tok/s** |
| `pp2048+tg128` DFlash7, INT8 KV | `pp2048+tg128` | **prefill 2,225 / decode 522.8 tok/s** (acc 88.0%, len 7.11) |
| `pp2048+tg128` DFlash7, rk4v4-e8 | `pp2048+tg128` | **prefill 2,270 / decode 546.6 tok/s** (acc 88.0%, len 7.11) |
| `pp8192+tg128` no speculation, INT8 KV | `pp8192+tg128` | **prefill 12,377 / decode 236.1 tok/s** |
| `pp8192+tg128` no speculation, rk4v4-e8 | `pp8192+tg128` | **prefill 12,141 / decode 234.6 tok/s** |
| `tg128` DFlash4, rk4v4-e8 (cold corpus) | `tg128` | **decode 234.6 tok/s** (acc 18.0%, len 1.72) |
| `tg128` no speculation, INT8 KV | `tg128` | **decode 248.7 tok/s** |

Three properties of this matrix are load-bearing:

1. **The pool must be pinned to compare runs.** With the default auto-curved pool (~19 GiB but a
   few hundred experts unslotted) part of every layer runs on the CPU half: the same binary
   measured 374–632 tok/s at 2k/4k prefill with a **3.3× per-repetition spread**
   (pp4096: 3.78 s … 12.35 s). Pinned at 19,456 MiB the spread collapses under 5% and the table
   above is what the hardware does. Deployment profiles that need a larger KV pool trade this
   prefill away — state the pool with any number quoted from this page.
2. **DFlash costs prefill and pays back at decode.** The draft model runs its own pass over the
   prompt: 2,225 vs 11,499 tok/s at the same 2k prompt, while decode at 88% acceptance is
   522–547 tok/s against 236–249 without speculation. Prefill-bound workloads should leave the
   draft off; decode-bound ones should not.
3. **The cold-corpus DFlash4 point is weak by design.** Bench corpus continuation is close to
   unpredictable, so acceptance is 18% (len 1.72) and decode lands at the no-spec level. It is
   reported to keep the speculation table honest, not as a recommendation.

The `pp32768+tg128` row is **not measured** in this pass. The full-residency pool leaves enough
KV for rk4v4-e8 at 32k, but the long prefill exceeded this pass's wall-clock budget (the run was
stopped after 19 minutes without completing a warm-up plus three measured passes), and the
streaming profile is still being tuned with the CPU-partition line. It is left as unmeasured
rather than estimated; the dense 27B page carries the long-context reference point.

---

## 2. KV layout: precision and speed at 8,192 context

Perplexity at `context 4096 / stride 1024` over the varied MoE corpus (4,801 scored tokens, the
same corpus the MoE gates score), default auto-curved pool, one deterministic pass per layout:

| KV layout | PPL (ctx 4096) | vs int8 |
|---|---:|---:|
| `k8v4` | 6.478748 | −0.02% |
| `fp8` | 6.479183 | −0.02% |
| `int8` | 6.480351 | — |
| `bf16` | 6.486000 | +0.09% |
| `rk8v4` | 6.520409 | +0.62% |
| `rk4v4` | 6.528934 | +0.75% |
| `rk4v4-e8` | 6.591483 | +1.71% |
| `rk2v4-e8` | 6.819138 | +5.23% |

The ordering and the sign of the deltas match the dense model's table (rank-compressed layouts
lose precision monotonically); the absolute level differs because the corpus and the model do.

Speed at 8,192 context, no speculation: `int8` 12,377 prefill / 236.1 decode, `rk4v4-e8`
12,141 / 234.6. A KV layout therefore moves prefill by under 2% and decode by under 1% here —
as on the dense model, the layout is a capacity-versus-accuracy choice, not a speed choice.

Bytes per token and context ceilings are model geometry × layout width; the 27B page documents
the measurement procedure (`reservation delta between two capacities`, then binary search under
`strict` and `--wddm-evictable-budget`). They are not re-measured for the 35B in this pass and
should be quoted only after that procedure is run on this artifact.

---

## 3. Method and reproduction

- Machine: single RTX 4090 24 GB, Windows 11, desktop sharing the card. Report the granted
  budget with any ceiling; the engine prints `capacity | KV ... | runtime ... | free ...`.
- Pool: `NINFER_EXPERT_CACHE_MB=19456` for every row in §1 (full residency). §2 runs with the
  default curve pool because perplexity is a single-pass deterministic measurement; the pool
  size still selects how much of each layer runs on the CPU half, so quote it with the table.
- Repetitions: `-r 5 --warmup 1` for every row (the 32k row would have used `-r 3`).
- `ninfer_bench` writes a JSON report (`-o json --output-file ...`) with per-repetition timings;
  the means in §1 come from those reports, and the table does not hide the spread — see the
  pinned-versus-curved note above.
- Chunking: `--prefill-chunk 1024` throughout; KV layouts are stated per row.

## 4. How to use this as a baseline

1. Pin the pool before comparing anything: the same binary moves 5× at 2k prefill between the
   curved and the pinned pool.
2. Compare decode at the same speculation setting and the same acceptance; DFlash and no-spec
   are different products, not a faster/slower switch.
3. Compare prefill at the same context and residency; the DFlash draft pass is part of prefill
   whenever the draft is on.
4. For KV layouts, compare PPL at the same `context`/`stride` and the same pool, and report the
   `vs int8` delta rather than the absolute level.
