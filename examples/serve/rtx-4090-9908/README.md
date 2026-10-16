# RTX 4090 (sm_89) serve matrix - nine launchers on port 9908

Nine Windows launchers covering **{128K, 256K, 1M} x {rk4v4-e8, rk4v4, int8}**, all listening on
**port 9908**, all with the optimisations this release can offer turned on by default.

| Launcher | Context | Concurrency | KV | Speculative |
|---|---|---|---|---|
| `run_v3_serve_128k_c8_rk4v4-e8.bat` | 128K (131072) | 8 | `rk4v4-e8` | dflash2 (draft 7) |
| `run_v3_serve_128k_c8_rk4v4.bat` | 128K (131072) | 8 | `rk4v4` | dflash2 (draft 7) |
| `run_v3_serve_128k_c8_int8.bat` | 128K (131072) | 8 | `int8` | dflash2 (draft 7) |
| `run_v3_serve_256k_c8_rk4v4-e8.bat` | 256K (262144) | 8 | `rk4v4-e8` | dflash2 (draft 7) |
| `run_v3_serve_256k_c8_rk4v4.bat` | 256K (262144) | 8 | `rk4v4` | dflash2 (draft 7) |
| `run_v3_serve_256k_c8_int8.bat` | 256K (262144) | 8 | `int8` | dflash2 (draft 7) |
| `run_v3_serve_1m_c4_rk4v4-e8.bat` | 1M (1048576) | 4 | `rk4v4-e8` | dflash2 (draft 7) |
| `run_v3_serve_1m_c4_rk4v4.bat` | 1M (1048576) | 4 | `rk4v4` | dflash2 (draft 7) |
| `run_v3_serve_1m_c4_int8.bat` | 1M (1048576) | 4 | `int8` | dflash2 (draft 7) - **cannot start, see below** |

`SERVE_MATRIX_9908.md` is the same matrix with the measurements behind it, in Chinese.

## Where these files run

The launchers are **shipped inside the release folder**, i.e. next to `ninfer-serve.exe`, with the
artifact one level up (`..\models\qwen3_8_27b_v3.ninfer`). Each one `cd`s to its own directory and
refuses to run if `ninfer-serve.exe` is not beside it, so copy them back there if you take them out
of the release. `NINFER_MODEL` overrides the artifact path.

## What they turn on by default

| Switch | Effect |
|---|---|
| `--kv-evict auto` | eviction contract, decided once at startup from the resolved arena, the Host budget and the KV dtype. The startup log prints the decision (`eviction \| contract on \| window 4 sink 1 pages \| host demotion by default`) or the condition that failed. |
| `--kv-idle-reclaim-ms 60000` | idle shared-prefix demotion to the pinned Host arena, fetched back on reuse. This is the engine default; it was measured output-neutral (greedy text byte-identical off vs on), first fetch +30..80 ms against a 17-19 s recompute. |
| `--wddm-evictable-budget` | size the arena against total VRAM - 512 MB rather than currently-free VRAM. |
| `--spec dflash2 --draft-tokens 7 --lm-head-draft` | speculative decoding with the DFlash2 draft and an LM-head draft. |

`--kv-capacity auto` keeps the 1 GiB sizing headroom and takes the largest legal arena; pin an
explicit capacity instead on a card you share.

## dflash2 at 1M

The draft's trained range is **262,144 tokens**, and that does **not** cap `--max-context`: the hard
refusal is gone (appendix FS, plan S-A), and a batch whose draft window would cross 262,144 tokens is
routed to the **ordinary round** instead (`decode.cpp` `decode_raw`, decided per batch), so the draft
is never executed outside its trained range. Within 256K positions decode keeps the draft's
acceleration; past that it is plain decode. Measured on this build:
`--spec dflash2 --max-context 1048576 --max-concurrency 4` starts in ~66 s, listens and serves a real
completion. Use `--spec mtp --draft-tokens 3` if you would rather have a draft that never stands down
across a 1M range.

## 1M x int8 cannot start on a 24 GB card

int8 needs ~33 KiB per token and `--kv-capacity` must be >= `--max-context`, so a 1M int8 arena needs
~34.5 GiB before anything else is reserved. Startup fails after ~8 s:

```text
FATAL server failed during startup | minimum Engine runtime reservation requires
36979631360 bytes in addition to 1073741824 bytes of automatic headroom, but only
25219825664 bytes are available
```

MTP and no-speculation fail identically while 1M x `rk4v4` without speculation starts, so the cause is
the KV dtype, not dflash2. The launcher keeps the requested command, states this in its header, and
carries a ready alternative at the end of the file - the same profile at 512K (arena 558,720 tokens),
measured to start and serve. 512K is the largest int8 profile verified on this card.

## Reading the startup log

`ninfer-serve` reports what it could not give you rather than failing quietly:

- `capacity | KV N tokens, ..., auto | pages P/Q` - how much KV the arena actually holds.
- `concurrency | the arena holds N full-length ... lane(s) but --max-concurrency is M` - the extra
  requests queue; under pressure the engine pauses lanes and pushes their snapshots into the Host
  arena.
- `queue | a queued request can wait up to about N ms ... beyond --pending-timeout-ms` - raise the
  timeout, lower the concurrency, or free device memory.
- `preemption | ... one full preemption of every lane needs N GiB` - a pause that cannot keep its
  snapshot is recovered by recomputing the context (+22..33% wall time); raise `--host-context-mib`.

Background on the KV layouts, the retention-quality curve and the two startup switches:
`docs/zh-CN/kv-cache-and-eviction.md`.
