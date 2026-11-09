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
| `pp32768+tg128` no speculation, rk4v4-e8 (`-r 3`, max-ctx 40960) | `pp32768+tg128` | **prefill 724.5 / decode 45.4 tok/s** |
| `tg128` DFlash4, rk4v4-e8 (cold corpus) | `tg128` | **decode 234.6 tok/s** (acc 18.0%, len 1.72) |
| `tg128` no speculation, INT8 KV | `tg128` | **decode 248.7 tok/s** |

Three properties of this matrix are load-bearing:

1. **The pool must be pinned to compare runs.** With the default auto-curved pool (~19 GiB but a
   few hundred experts unslotted) part of every layer runs on the CPU half: the same binary
   measured 374–632 tok/s at 2k/4k prefill with a **3.3× per-repetition spread**
   (pp4096: 3.78 s … 12.35 s). Pinned at 19,456 MiB the spread collapses under 5% and the table
   above is what the hardware does. Deployment profiles that need a larger KV pool trade this
   prefill away — state the pool with any number quoted from this page. Note that the DFlash rows
   are **not** at 19,456 MiB: DFlash reserves 431 MiB of KV against 177 MiB without it, and the pin
   gives way to 18,685 MiB, which is what puts 423 experts on the host tier (§1b).
2. **DFlash's own prefill pass is ~1% of a chunk; the 4.6× in the table is residency, not the
   draft.** Measured with `NINFER_DFLASH_TIMING=1`, the whole draft append for a 1,024-token chunk
   (feature projection + six K/V context projections + key RMSNorm + RoPE + draft KV append) costs
   **1.14 ms / 0.99 ms** against a target chunk of 88–94 ms (§1b). The 2,225 tok/s row above was
   taken with the pool at 18,685 MiB, where 423 of the 10,240 experts sit on the host tier — and
   that same pool measures 1,731.6 tok/s with the draft switched **off**. The draft pays back at
   decode (522–547 tok/s at 88% acceptance against 236–249 without speculation), and it is the
   decode side that justifies it.
3. **The cold-corpus DFlash4 point is weak by design.** Bench corpus continuation is close to
   unpredictable, so acceptance is 18% (len 1.72) and decode lands at the no-spec level. It is
   reported to keep the speculation table honest, not as a recommendation.

### 1b. Where a DFlash prefill chunk actually goes

Two env-gated probes carry this section (`NINFER_DFLASH_TIMING`, `NINFER_PREFILL_TIMING`; both are
off by default and cost nothing when unset). The numbers below come from this tree at `55eac7f6`
plus the probes; they are the only delta against the shipped `R:\infer` binaries, and with both
variables unset the two produce the same tokens (§ "Token identity" below).

- `NINFER_DFLASH_TIMING=1` splits one `dflash_append_context` call into CUDA-event intervals
  (`src/models/qwen3_5/execution/draft.cpp`, `AppendTiming`) and prints one line per call. It also
  brackets the target-side feature copies (`src/models/qwen3_5/execution/text.cpp`,
  `FeatureCopyTiming`) and stands down on a capturing stream, because recording a timing event
  inside a capture invalidates it.
- `NINFER_PREFILL_TIMING=1` prints the device time of each whole prefill chunk from the Program's
  own timer, with no extra synchronisation, so the two can be read against each other.

Segment table for the draft append (`pp2048`, INT8 KV, chunk 1024, `--spec dflash
--draft-tokens 7 --lm-head-draft`, pool at the all-resident point):

| Segment (per 1,024-token chunk) | Chunk 1 | Chunk 2 |
|---|---:|---:|
| feature projection `[2048→2048]`, K = 8 × 2048 (`project`) | 0.54 ms | 0.49 ms |
| context RMSNorm | 0.01 ms | 0.01 ms |
| 6 × context K+V projections (Q8, `ops::linear_pair`) | 0.44 ms | 0.40 ms |
| key RMSNorm × 6 | 0.03 ms | 0.03 ms |
| RoPE × 6 | 0.03 ms | 0.03 ms |
| draft KV append × 6 (`kv_cache_append_prefix`) | 0.10 ms | 0.04 ms |
| **total draft append** | **1.14 ms** | **0.99 ms** |
| target-side feature copies (8 taps × 4 MiB) | 0.42 ms | 0.18 ms |
| target chunk, same shape, draft **off** | 88–94 ms | 88–94 ms |

So the draft pass is **1.2–1.4%** of a chunk. The feature projection is a single 34.4 G-MAC GEMM
(0.5 ms ≈ 137 TFLOPS) and the six K/V projections are 25.8 G MAC (0.44 ms ≈ 117 TFLOPS), i.e. the
append moves 60 G MAC at ~67 TOPS of int8 work against the 4090's ~660 TOPS peak — the small-N
dequant shapes are what limit it, and even at peak the whole append would return under 1 ms of an
88 ms chunk. There is nothing to win here.

**What actually costs 4.6–6.6× is which tier the experts are on.** Same binary, same corpus,
`-p 2048 --kv-dtype int8 --prefill-chunk 1024 -r 3 --warmup 1`:

| Configuration | `pp2048` | Pool | Experts unslotted |
|---|---:|---:|---:|
| no speculation, `NINFER_EXPERT_CACHE_MB=19456` | **11,376.9 ± 159.5** | 19,351 MiB | 0 of 10,240 |
| no speculation, pool pinned to the DFlash run's size | 1,731.6 ± 269.1 | 18,406 MiB | 423 of 10,240 |
| `--spec dflash` (auto split) | **1,849.0 ± 160.5** | 18,406 MiB | 423 of 10,240 |

The last two rows differ only by the draft, and they differ by 7% inside a ±16% spread. DFlash adds
528 MiB of draft weights and 254 MiB of draft local KV. The no-spec plan already sat 151 MiB above
the 19,200 MiB all-resident point, so the draft pushes the pool off it: 423 experts go to the host
tier and every layer that routes to one takes the split path for that chunk. That is the entire
"DFlash costs prefill" effect on this card.

Forcing the pool back to 19,200 MiB by spending the startup planning allowance does **not** recover
it, and the attempt is recorded here because it is the obvious next idea: the pool plus KV then
needs 19,631 MiB, `nvidia-smi` reports 23,764 MiB of 24,564 used with the SM at 21% utilisation,
and the identical command measures anywhere from 40 to 2,910 tok/s (`pp1024` came in at 890 ms and at
14,500 ms per chunk on two runs). On a 24 GB card the model, the draft weights, the draft KV and the
all-resident expert set do not fit at 2k context at the same time; the draft KV is BFloat16 over the
4,096-token sliding window and is the only one of the four that a future change could shrink.

**Token identity.** The probes record events and nothing else: a 64-token greedy run of the same
prompt with `NINFER_DFLASH_TIMING` unset and set produced byte-identical text
(`SHA-256 2EED0A1D6B3D…`), and this tree's binaries produce byte-identical text to the shipped
`R:\infer` binaries on both the draft-on and draft-off arms. One property is *not* covered by that
check and predates this pass: on this artifact a greedy DFlash run and a greedy no-speculation run
diverge at the **first** token (`秋风轻…` vs `秋风悄…`), in the shipped release binary as well. The
first token is the prefill bonus sample, so the draft path changes what the prefill publishes; it is
a verification/bonus-token question that needs its own investigation and is not a prefill-speed
matter.

The `pp32768+tg128` row is measured **without speculation** under the same full-residency pool:
prefill 724.5 tok/s, decode 45.4 tok/s (`-r 3`, rk4v4-e8, `--max-ctx 40960`). Long-context
attention, not the expert tier, is the binding cost there — the same command at 8,192 context
measures 12,141 / 234.6. The DFlash variant of this row is not published: the draft pass over a
32k prompt plus the draft KV pushes the pool off full residency on a 24 GB card, and the
identical command measured anywhere from 40 to 2,910 tok/s depending on where the pool landed.

### 1c. MTP on this artifact: measured, and why it loses

The 35B artifact carries an `mtp` component, so "use MTP instead of DFlash" is a fair question and
it was measured rather than argued (`--spec mtp --draft-tokens 7 --lm-head-draft`, `-pg 2048,128
--kv-dtype int8`, `-r 5 --warmup 1 --prefill-chunk 1024`, `NINFER_EXPERT_CACHE_MB=19456`):

| Mode | prefill tok/s | decode tok/s | acceptance | pool | resident experts |
|---|---:|---:|---:|---:|---:|
| no speculation | **11,499** | 236.1 | — | 19,351 MiB | 10,240 / 10,240 |
| `--spec dflash` 7 | 2,225 | **522.8** | 88.0% | 18,685 MiB | 9,817 / 10,240 |
| `--spec mtp` 7 | 127.8 ± 46.6 | 76.1 ± 4.1 | 88.0% | 18,766 MiB | 5,774 / 10,496 |

MTP is not a substitute here: it loses prefill to DFlash by 17× and decode to the *no-speculation*
row, at the same acceptance rate. The cause is residency again, but of a different shape. Startup
reports `banks=41 slot=3407872` for MTP against `banks=40 slot=1966080` without it: the MTP head
adds a 41st expert bank whose expert is 3.25 MiB against 1.875 MiB for the 40 main banks, and the
pool sizes **every** slot from the widest bank (`req.max_entry_bytes`, `core/expert_cache.cpp:246`).
The ask therefore becomes 10,496 × 3.25 MiB ≈ 33.2 GiB against the 18.8 GiB the card grants, so
5,774 of 10,496 experts stay resident and most layers take the split path for most chunks — which
is the same mechanism §1b describes, at three times the miss rate.

What could change this, in order of how much it would buy (none of it is implemented, and only the
first two are pure arithmetic on the layouts):

1. **Size slots per bank instead of from the global maximum.** The 40 main banks would be billed
   19,200 MiB (as today's all-resident point) and the MTP bank 830 MiB. That is 20,030 MiB, still
   above the 18,766 MiB granted, so the MTP bank would have to stream in part — much better than
   55% residency, still not full residency.
2. **Narrow the draft KV.** It is BFloat16 over the sliding window; `int8`/`fp8` halves its 254 MiB.
   That closes roughly a sixth of the DFlash gap (§1b) and does nothing for the MTP gap, so it is a
   partial measure at best.
3. **Switch the pool between prefill and decode.** Prefill wants the all-resident 19,351 MiB;
   decode wants the draft's 782 MiB. The idle-window `resize` path already moves pool and KV at a
   request boundary (§"Runtime pool/KV resize" in the release notes), but a captured whole-step graph
   freezes the expert offsets, so this needs a capture/no-capture trade first. Unproven.

Until one of those lands, the choice is by workload: no speculation for prefill-heavy traffic
(11,499 / 236), DFlash when decode throughput is the point (2,225 / 523), and MTP only on the dense
27B, where the head costs no residency (3,309 / 223.8).

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
- Pool: `NINFER_EXPERT_CACHE_MB=19456` for every row in §1 (full residency) — except the DFlash
  rows, where the same pin resolves to 18,685 MiB because DFlash's KV reservation is larger; the
  resolution is printed as `[budget][split] ... pool=<MiB>` and `[budget] experts: ... unslotted=N
  of 10,240`. §2 runs with the default curve pool because perplexity is a single-pass
  deterministic measurement; the pool size still selects how much of each layer runs on the CPU
  half, so quote it with the table.
- Repetitions: `-r 5 --warmup 1` for every row (the 32k row would have used `-r 3`). The §1b
  residency A/B is `-r 3 --warmup 1`.
- `ninfer_bench` writes a JSON report (`-o json --output-file ...`) with per-repetition timings;
  the means in §1 come from those reports, and the table does not hide the spread — see the
  pinned-versus-curved note above.
- Chunking: `--prefill-chunk 1024` throughout; KV layouts are stated per row.
- Timing probes: `NINFER_DFLASH_TIMING=1` and `NINFER_PREFILL_TIMING=1` (§1b) are diagnostic only.
  Both add synchronisation and must never be left on for a throughput number; every number in §1
  and §2 was taken with them unset.

## 4. How to use this as a baseline

1. Pin the pool before comparing anything: the same binary moves 5× at 2k prefill between the
   curved and the pinned pool. Print `unslotted=` with the number — 423 missing experts and 0
   missing experts are different products at the same nominal pool.
2. Compare decode at the same speculation setting and the same acceptance; DFlash and no-spec
   are different products, not a faster/slower switch.
3. Compare prefill at the same context, **the same pool and the same `unslotted=` count**: the
   draft's own pass is 1.2–1.4% of a chunk (§1b), so a prefill gap between DFlash and no-spec that
   is larger than that is residency, not the draft.
4. For KV layouts, compare PPL at the same `context`/`stride` and the same pool, and report the
   `vs int8` delta rather than the absolute level.
