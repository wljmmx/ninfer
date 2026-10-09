# sm_89 fallback table for the upstream bf16 shapes

Upstream targets **sm_120a (RTX 5090)**. Its bf16 linear shapes serve the large-token tail
with TMA schedules (`Bf16A16TmaMmaSchedule`), which do not exist on sm_89. Each shape wraps
that tail in `#if defined(NINFER_ENABLE_TMA)` and, in the `#else`, falls back to **this
shape's own cp.async schedule** (a coarse, conservative choice). This table records the
**like-for-like** Ada replacement for every TMA instance, so stage 2 of
`upstream_absorption_plan.md` becomes a mechanical edit plus measurement rather than a
design exercise.

## Mapping rule

The replacement follows the inheritance already declared in
`src/ops/linear/bf16/bf16_schedule.cuh` (the TMA schedule derives from the cp.async one
with identical tile geometry):

```
Bf16A16TmaMmaSchedule<R,T,K,WR,WT,S,M[,Raster[,RGR[,ConsumerKUnroll]]]>
  : Bf16A16MmaSchedule<R,T,K,WR,WT,S,M, cg, cg, PingPong, Raster, Tma128, RGR>

Ada fallback:
Bf16A16MmaSchedule<R,T,K,WR,WT,S,M, cg, cg, PingPong, Raster, Xor64, RGR>
```

Only the swizzle changes (**Tma128** producer staging -> cp.async **Xor64**); tile geometry,
stages, raster, min-blocks-per-SM and any outer wrapper (`Bf16ScheduleInstance`,
`Bf16RowTailSchedule`, `Bf16KTailSchedule`) are preserved. `ConsumerKUnroll` has no cp.async
analogue and is dropped. Each fallback satisfies the cp.async schedule's static asserts on
sm_89 (`kBlockRows % kWarpRows == 0`, `kBlockTokens % kWarpTokens == 0`, `kBlockK % 64 == 0`,
`2 <= kStages <= 8`, `kSharedBytes <= 99 KiB`).

## Status

- **Recorded, not wired in.** The `#else` branches still keep their conservative reuse; the
  fallbacks below are proposed replacements, verified by static reasoning only - not yet
  compiled or benchmarked on device.
- **Stage 2 is deferred** (no target model family): every shape here is inert for the
  current baseline, qwen3.8-27b (hidden = 5120).
- Regenerate with `python tools/port/gen_bf16_fallback_table.py`.

`token range` is the token interval that selects the instance; `inf` is the final `return`
(no upper bound). Only TMA-selected steps get a fallback row; a non-TMA step interleaved
inside the guard still advances the interval.

## Coverage

Sixteen shape files carry an inline TMA ladder (listed below). Three further files
(`n14336_k5120.cu`, `n256_k5120.cu`, `n5120_k6144.cu`) reach TMA through the pre-existing
aliases in `bf16_instances.cuh`; they predate this merge and already ship an `#else`
fallback. `n48_k2560.cu` and `n96_k2560.cu` contain no TMA ladder.

---


## Per-shape ladders

### `n10240_k320.cu`

| token range | TMA instance (sm_120a) | Ada fallback (sm_89) |
|---|---|---|
| (48, 64] | `Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<64, 32, 64, 16, 16, 3, 2>, 320>` | `Bf16ScheduleInstance<Bf16A16MmaSchedule<64, 32, 64, 16, 16, 3, 2, Cache::cg, Cache::cg, Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast, Bf16MmaSwizzle::Xor64, 1>, 320>` |
| (128, inf] | `Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<64, 64, 64, 32, 16, 2, 3, Bf16MmaRaster::Grouped, 8>, 320>` | `Bf16ScheduleInstance<Bf16A16MmaSchedule<64, 64, 64, 32, 16, 2, 3, Cache::cg, Cache::cg, Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::Grouped, Bf16MmaSwizzle::Xor64, 8>, 320>` |

current `#else` fallback: `S2` (`mma`)

### `n1152_k1152.cu`

| token range | TMA instance (sm_120a) | Ada fallback (sm_89) |
|---|---|---|
| (128, 256] | `Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<32, 64, 64, 16, 8, 3, 1>, 1152>` | `Bf16ScheduleInstance<Bf16A16MmaSchedule<32, 64, 64, 16, 8, 3, 1, Cache::cg, Cache::cg, Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast, Bf16MmaSwizzle::Xor64, 1>, 1152>` |
| (256, 512] | `Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<64, 64, 64, 32, 16, 3, 2>, 1152>` | `Bf16ScheduleInstance<Bf16A16MmaSchedule<64, 64, 64, 32, 16, 3, 2, Cache::cg, Cache::cg, Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast, Bf16MmaSwizzle::Xor64, 1>, 1152>` |
| (512, 1024] | `Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<64, 128, 64, 32, 32, 3, 1>, 1152>` | `Bf16ScheduleInstance<Bf16A16MmaSchedule<64, 128, 64, 32, 32, 3, 1, Cache::cg, Cache::cg, Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast, Bf16MmaSwizzle::Xor64, 1>, 1152>` |
| (1024, 2048] | `Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<128, 128, 64, 64, 32, 3, 1>, 1152>` | `Bf16ScheduleInstance<Bf16A16MmaSchedule<128, 128, 64, 64, 32, 3, 1, Cache::cg, Cache::cg, Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast, Bf16MmaSwizzle::Xor64, 1>, 1152>` |
| (2048, inf] | `Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<64, 64, 64, 32, 16, 3, 2>, 1152>` | `Bf16ScheduleInstance<Bf16A16MmaSchedule<64, 64, 64, 32, 16, 3, 2, Cache::cg, Cache::cg, Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast, Bf16MmaSwizzle::Xor64, 1>, 1152>` |

current `#else` fallback: `S5` (`mma`)

### `n1152_k1536.cu`

| token range | TMA instance (sm_120a) | Ada fallback (sm_89) |
|---|---|---|
| (128, 256] | `Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<64, 32, 64, 16, 16, 3, 2>, 1536>` | `Bf16ScheduleInstance<Bf16A16MmaSchedule<64, 32, 64, 16, 16, 3, 2, Cache::cg, Cache::cg, Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast, Bf16MmaSwizzle::Xor64, 1>, 1536>` |
| (256, 512] | `Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<64, 64, 64, 32, 16, 3, 2>, 1536>` | `Bf16ScheduleInstance<Bf16A16MmaSchedule<64, 64, 64, 32, 16, 3, 2, Cache::cg, Cache::cg, Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast, Bf16MmaSwizzle::Xor64, 1>, 1536>` |
| (512, 1024] | `Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<64, 128, 64, 32, 32, 3, 1>, 1536>` | `Bf16ScheduleInstance<Bf16A16MmaSchedule<64, 128, 64, 32, 32, 3, 1, Cache::cg, Cache::cg, Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast, Bf16MmaSwizzle::Xor64, 1>, 1536>` |
| (1024, 2048] | `Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<128, 128, 64, 64, 32, 3, 1>, 1536>` | `Bf16ScheduleInstance<Bf16A16MmaSchedule<128, 128, 64, 64, 32, 3, 1, Cache::cg, Cache::cg, Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast, Bf16MmaSwizzle::Xor64, 1>, 1536>` |
| (2048, inf] | `Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<64, 64, 64, 32, 16, 3, 2>, 1536>` | `Bf16ScheduleInstance<Bf16A16MmaSchedule<64, 64, 64, 32, 16, 3, 2, Cache::cg, Cache::cg, Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast, Bf16MmaSwizzle::Xor64, 1>, 1536>` |

current `#else` fallback: `S4` (`mma`)

### `n1152_k4304.cu`

| token range | TMA instance (sm_120a) | Ada fallback (sm_89) |
|---|---|---|
| (116, 128] | `Bf16KTailSchedule<Bf16ScheduleInstance< Bf16A16TmaMmaSchedule<32, 32, 64, 16, 8, 4, 1, Bf16MmaRaster::TokenFast, 1, 1>, 4304>>` | `Bf16KTailSchedule<Bf16ScheduleInstance< Bf16A16MmaSchedule<32, 32, 64, 16, 8, 4, 1, Cache::cg, Cache::cg, Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast, Bf16MmaSwizzle::Xor64, 1>, 4304>>` |
| (128, 256] | `Bf16KTailSchedule< Bf16A16TmaMmaSchedule<64, 40, 64, 16, 8, 5, 1, Bf16MmaRaster::TokenFast, 1, 1>>` | `Bf16KTailSchedule< Bf16A16MmaSchedule<64, 40, 64, 16, 8, 5, 1, Cache::cg, Cache::cg, Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast, Bf16MmaSwizzle::Xor64, 1>>` |
| (256, 512] | `Bf16KTailSchedule<Bf16ScheduleInstance< Bf16A16TmaMmaSchedule<64, 64, 64, 32, 16, 4, 1, Bf16MmaRaster::RowFast, 1, 1>, 4304>>` | `Bf16KTailSchedule<Bf16ScheduleInstance< Bf16A16MmaSchedule<64, 64, 64, 32, 16, 4, 1, Cache::cg, Cache::cg, Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::RowFast, Bf16MmaSwizzle::Xor64, 1>, 4304>>` |
| (512, 1024] | `Bf16KTailSchedule<Bf16ScheduleInstance< Bf16A16TmaMmaSchedule<64, 128, 64, 32, 32, 3, 1, Bf16MmaRaster::Grouped, 4, 1>, 4304>>` | `Bf16KTailSchedule<Bf16ScheduleInstance< Bf16A16MmaSchedule<64, 128, 64, 32, 32, 3, 1, Cache::cg, Cache::cg, Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::Grouped, Bf16MmaSwizzle::Xor64, 4>, 4304>>` |
| (1024, 2048] | `Bf16KTailSchedule< Bf16A16TmaMmaSchedule<128, 128, 64, 32, 32, 3, 1, Bf16MmaRaster::TokenFast, 1, 1>>` | `Bf16KTailSchedule< Bf16A16MmaSchedule<128, 128, 64, 32, 32, 3, 1, Cache::cg, Cache::cg, Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast, Bf16MmaSwizzle::Xor64, 1>>` |
| (2048, inf] | `Bf16KTailSchedule<Bf16ScheduleInstance< Bf16A16TmaMmaSchedule<128, 112, 64, 32, 16, 2, 1, Bf16MmaRaster::TokenFast, 1, 1>, 4304>>` | `Bf16KTailSchedule<Bf16ScheduleInstance< Bf16A16MmaSchedule<128, 112, 64, 32, 16, 2, 1, Cache::cg, Cache::cg, Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast, Bf16MmaSwizzle::Xor64, 1>, 4304>>` |

current `#else` fallback: `S4` (`mma`)

### `n12800_k2560.cu`

| token range | TMA instance (sm_120a) | Ada fallback (sm_89) |
|---|---|---|
| (48, 64] | `Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<128, 64, 128, 32, 32, 2, 1>, 2560>` | `Bf16ScheduleInstance<Bf16A16MmaSchedule<128, 64, 128, 32, 32, 2, 1, Cache::cg, Cache::cg, Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast, Bf16MmaSwizzle::Xor64, 1>, 2560>` |
| (64, 96] | `Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<128, 32, 64, 32, 16, 3, 1>, 2560>` | `Bf16ScheduleInstance<Bf16A16MmaSchedule<128, 32, 64, 32, 16, 3, 1, Cache::cg, Cache::cg, Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast, Bf16MmaSwizzle::Xor64, 1>, 2560>` |
| (96, 128] | `Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<128, 48, 64, 32, 16, 2, 1>, 2560>` | `Bf16ScheduleInstance<Bf16A16MmaSchedule<128, 48, 64, 32, 16, 2, 1, Cache::cg, Cache::cg, Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast, Bf16MmaSwizzle::Xor64, 1>, 2560>` |
| (128, 512] | `Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<128, 64, 64, 32, 32, 3, 1>, 2560>` | `Bf16ScheduleInstance<Bf16A16MmaSchedule<128, 64, 64, 32, 32, 3, 1, Cache::cg, Cache::cg, Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast, Bf16MmaSwizzle::Xor64, 1>, 2560>` |
| (512, inf] | `Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<128, 128, 64, 64, 32, 3, 1>, 2560>` | `Bf16ScheduleInstance<Bf16A16MmaSchedule<128, 128, 64, 64, 32, 3, 1, Cache::cg, Cache::cg, Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast, Bf16MmaSwizzle::Xor64, 1>, 2560>` |

current `#else` fallback: `S4` (`mma`)

### `n13952_k2560.cu`

| token range | TMA instance (sm_120a) | Ada fallback (sm_89) |
|---|---|---|
| (48, 64] | `Bf16A16TmaMmaSchedule<128, 64, 128, 32, 32, 2, 1>` | `Bf16A16MmaSchedule<128, 64, 128, 32, 32, 2, 1, Cache::cg, Cache::cg, Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast, Bf16MmaSwizzle::Xor64, 1>` |
| (64, 96] | `Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<128, 32, 128, 16, 16, 2, 1>, 2560>` | `Bf16ScheduleInstance<Bf16A16MmaSchedule<128, 32, 128, 16, 16, 2, 1, Cache::cg, Cache::cg, Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast, Bf16MmaSwizzle::Xor64, 1>, 2560>` |
| (96, 128] | `Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<128, 48, 64, 32, 16, 2, 1>, 2560>` | `Bf16ScheduleInstance<Bf16A16MmaSchedule<128, 48, 64, 32, 16, 2, 1, Cache::cg, Cache::cg, Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast, Bf16MmaSwizzle::Xor64, 1>, 2560>` |
| (128, 512] | `Bf16A16TmaMmaSchedule<64, 64, 64, 32, 32, 3, 2>` | `Bf16A16MmaSchedule<64, 64, 64, 32, 32, 3, 2, Cache::cg, Cache::cg, Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast, Bf16MmaSwizzle::Xor64, 1>` |
| (512, inf] | `Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<64, 128, 64, 32, 32, 3, 1>, 2560>` | `Bf16ScheduleInstance<Bf16A16MmaSchedule<64, 128, 64, 32, 32, 3, 1, Cache::cg, Cache::cg, Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast, Bf16MmaSwizzle::Xor64, 1>, 2560>` |

current `#else` fallback: `S4` (`mma`)

### `n1664_k2560.cu`

| token range | TMA instance (sm_120a) | Ada fallback (sm_89) |
|---|---|---|
| (96, 128] | `Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<64, 32, 64, 16, 16, 3, 2>, 2560>` | `Bf16ScheduleInstance<Bf16A16MmaSchedule<64, 32, 64, 16, 16, 3, 2, Cache::cg, Cache::cg, Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast, Bf16MmaSwizzle::Xor64, 1>, 2560>` |
| (128, 512] | `Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<128, 48, 64, 32, 16, 2, 1>, 2560>` | `Bf16ScheduleInstance<Bf16A16MmaSchedule<128, 48, 64, 32, 16, 2, 1, Cache::cg, Cache::cg, Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast, Bf16MmaSwizzle::Xor64, 1>, 2560>` |
| (512, inf] | `Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<128, 80, 64, 32, 16, 3, 1>, 2560>` | `Bf16ScheduleInstance<Bf16A16MmaSchedule<128, 80, 64, 32, 16, 3, 1, Cache::cg, Cache::cg, Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast, Bf16MmaSwizzle::Xor64, 1>, 2560>` |

current `#else` fallback: `S4` (`mma`)

### `n248320_k2560.cu`

| token range | TMA instance (sm_120a) | Ada fallback (sm_89) |
|---|---|---|
| (32, 64] | `Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<128, 32, 128, 32, 16, 2, 1>, 2560>` | `Bf16ScheduleInstance<Bf16A16MmaSchedule<128, 32, 128, 32, 16, 2, 1, Cache::cg, Cache::cg, Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast, Bf16MmaSwizzle::Xor64, 1>, 2560>` |
| (64, 96] | `Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<128, 48, 128, 32, 16, 2, 1>, 2560>` | `Bf16ScheduleInstance<Bf16A16MmaSchedule<128, 48, 128, 32, 16, 2, 1, Cache::cg, Cache::cg, Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast, Bf16MmaSwizzle::Xor64, 1>, 2560>` |
| (96, inf] | `Bf16A16TmaMmaSchedule<128, 128, 64, 64, 32, 3, 1>` | `Bf16A16MmaSchedule<128, 128, 64, 64, 32, 3, 1, Cache::cg, Cache::cg, Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast, Bf16MmaSwizzle::Xor64, 1>` |

current `#else` fallback: `S3` (`sliced_k_mma`)

### `n2560_k2560.cu`

| token range | TMA instance (sm_120a) | Ada fallback (sm_89) |
|---|---|---|
| (64, 96] | `Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<64, 32, 64, 16, 8, 3, 1>, 2560>` | `Bf16ScheduleInstance<Bf16A16MmaSchedule<64, 32, 64, 16, 8, 3, 1, Cache::cg, Cache::cg, Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast, Bf16MmaSwizzle::Xor64, 1>, 2560>` |
| (128, 512] | `Bf16ScheduleInstance< Bf16A16TmaMmaSchedule<128, 64, 64, 32, 32, 2, 2, Bf16MmaRaster::TokenFast, 1>, 2560>` | `Bf16ScheduleInstance< Bf16A16MmaSchedule<128, 64, 64, 32, 32, 2, 2, Cache::cg, Cache::cg, Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast, Bf16MmaSwizzle::Xor64, 1>, 2560>` |
| (512, 2048] | `Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<128, 128, 64, 64, 32, 3, 1>, 2560>` | `Bf16ScheduleInstance<Bf16A16MmaSchedule<128, 128, 64, 64, 32, 3, 1, Cache::cg, Cache::cg, Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast, Bf16MmaSwizzle::Xor64, 1>, 2560>` |
| (2048, inf] | `Bf16ScheduleInstance< Bf16A16TmaMmaSchedule<128, 64, 64, 64, 32, 2, 2, Bf16MmaRaster::RowFast, 1>, 2560>` | `Bf16ScheduleInstance< Bf16A16MmaSchedule<128, 64, 64, 64, 32, 2, 2, Cache::cg, Cache::cg, Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::RowFast, Bf16MmaSwizzle::Xor64, 1>, 2560>` |

current `#else` fallback: `S4` (`mma`)

### `n2560_k4608.cu`

| token range | TMA instance (sm_120a) | Ada fallback (sm_89) |
|---|---|---|
| (24, 64] | `Bf16RowTailSchedule< Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<48, 32, 192, 16, 16, 3, 1>, 4608>>` | `Bf16RowTailSchedule< Bf16ScheduleInstance<Bf16A16MmaSchedule<48, 32, 192, 16, 16, 3, 1, Cache::cg, Cache::cg, Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast, Bf16MmaSwizzle::Xor64, 1>, 4608>>` |
| (64, 96] | `Bf16ScheduleInstance< Bf16A16TmaMmaSchedule<64, 32, 128, 16, 16, 3, 1, Bf16MmaRaster::TokenFast, 1, 18>, 4608>` | `Bf16ScheduleInstance< Bf16A16MmaSchedule<64, 32, 128, 16, 16, 3, 1, Cache::cg, Cache::cg, Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast, Bf16MmaSwizzle::Xor64, 1>, 4608>` |
| (96, 128] | `Bf16RowTailSchedule< Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<48, 64, 128, 16, 16, 3, 1>, 4608>>` | `Bf16RowTailSchedule< Bf16ScheduleInstance<Bf16A16MmaSchedule<48, 64, 128, 16, 16, 3, 1, Cache::cg, Cache::cg, Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast, Bf16MmaSwizzle::Xor64, 1>, 4608>>` |
| (128, 256] | `Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<64, 64, 64, 32, 16, 4, 1, Bf16MmaRaster::Grouped, 4>, 4608>` | `Bf16ScheduleInstance<Bf16A16MmaSchedule<64, 64, 64, 32, 16, 4, 1, Cache::cg, Cache::cg, Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::Grouped, Bf16MmaSwizzle::Xor64, 4>, 4608>` |
| (256, 512] | `Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<128, 64, 64, 32, 32, 3, 1>, 4608>` | `Bf16ScheduleInstance<Bf16A16MmaSchedule<128, 64, 64, 32, 32, 3, 1, Cache::cg, Cache::cg, Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast, Bf16MmaSwizzle::Xor64, 1>, 4608>` |
| (512, inf] | `Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<128, 128, 64, 64, 32, 3, 1>, 4608>` | `Bf16ScheduleInstance<Bf16A16MmaSchedule<128, 128, 64, 64, 32, 3, 1, Cache::cg, Cache::cg, Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast, Bf16MmaSwizzle::Xor64, 1>, 4608>` |

current `#else` fallback: `S3` (`sliced_k_mma`)

### `n2560_k6144.cu`

| token range | TMA instance (sm_120a) | Ada fallback (sm_89) |
|---|---|---|
| (64, 96] | `Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<64, 32, 128, 32, 16, 2, 2>, 6144>` | `Bf16ScheduleInstance<Bf16A16MmaSchedule<64, 32, 128, 32, 16, 2, 2, Cache::cg, Cache::cg, Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast, Bf16MmaSwizzle::Xor64, 1>, 6144>` |
| (96, 128] | `Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<64, 32, 128, 16, 16, 2, 2>, 6144>` | `Bf16ScheduleInstance<Bf16A16MmaSchedule<64, 32, 128, 16, 16, 2, 2, Cache::cg, Cache::cg, Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast, Bf16MmaSwizzle::Xor64, 1>, 6144>` |
| (128, 512] | `Bf16ScheduleInstance< Bf16A16TmaMmaSchedule<128, 64, 64, 32, 32, 2, 1, Bf16MmaRaster::Grouped, 4>, 6144>` | `Bf16ScheduleInstance< Bf16A16MmaSchedule<128, 64, 64, 32, 32, 2, 1, Cache::cg, Cache::cg, Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::Grouped, Bf16MmaSwizzle::Xor64, 4>, 6144>` |
| (512, inf] | `Bf16A16TmaMmaSchedule<128, 128, 64, 64, 32, 3, 1>` | `Bf16A16MmaSchedule<128, 128, 64, 64, 32, 3, 1, Cache::cg, Cache::cg, Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast, Bf16MmaSwizzle::Xor64, 1>` |

current `#else` fallback: `S4` (`mma`)

### `n320_k10240.cu`

| token range | TMA instance (sm_120a) | Ada fallback (sm_89) |
|---|---|---|
| (128, 512] | `Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<32, 64, 64, 16, 8, 4, 1>, 10240>` | `Bf16ScheduleInstance<Bf16A16MmaSchedule<32, 64, 64, 16, 8, 4, 1, Cache::cg, Cache::cg, Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast, Bf16MmaSwizzle::Xor64, 1>, 10240>` |
| (512, inf] | `Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<32, 80, 64, 16, 8, 5, 1>, 10240>` | `Bf16ScheduleInstance<Bf16A16MmaSchedule<32, 80, 64, 16, 8, 5, 1, Cache::cg, Cache::cg, Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast, Bf16MmaSwizzle::Xor64, 1>, 10240>` |

current `#else` fallback: `S4` (`sliced_k_mma`)

### `n324_k10240.cu`

| token range | TMA instance (sm_120a) | Ada fallback (sm_89) |
|---|---|---|
| (128, 512] | `Bf16RowTailSchedule< Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<32, 64, 64, 16, 8, 4, 1>, 10240>>` | `Bf16RowTailSchedule< Bf16ScheduleInstance<Bf16A16MmaSchedule<32, 64, 64, 16, 8, 4, 1, Cache::cg, Cache::cg, Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast, Bf16MmaSwizzle::Xor64, 1>, 10240>>` |
| (512, inf] | `Bf16RowTailSchedule< Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<64, 48, 64, 16, 8, 4, 1>, 10240>>` | `Bf16RowTailSchedule< Bf16ScheduleInstance<Bf16A16MmaSchedule<64, 48, 64, 16, 8, 4, 1, Cache::cg, Cache::cg, Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast, Bf16MmaSwizzle::Xor64, 1>, 10240>>` |

current `#else` fallback: `S4` (`sliced_k_mma`)

### `n3456_k1152.cu`

| token range | TMA instance (sm_120a) | Ada fallback (sm_89) |
|---|---|---|
| (16, 96] | `Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<64, 32, 64, 16, 16, 3, 2>, 1152>` | `Bf16ScheduleInstance<Bf16A16MmaSchedule<64, 32, 64, 16, 16, 3, 2, Cache::cg, Cache::cg, Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast, Bf16MmaSwizzle::Xor64, 1>, 1152>` |
| (96, 128] | `Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<64, 64, 64, 32, 16, 3, 2>, 1152>` | `Bf16ScheduleInstance<Bf16A16MmaSchedule<64, 64, 64, 32, 16, 3, 2, Cache::cg, Cache::cg, Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast, Bf16MmaSwizzle::Xor64, 1>, 1152>` |
| (128, 256] | `Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<96, 64, 64, 48, 32, 2, 2>, 1152>` | `Bf16ScheduleInstance<Bf16A16MmaSchedule<96, 64, 64, 48, 32, 2, 2, Cache::cg, Cache::cg, Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast, Bf16MmaSwizzle::Xor64, 1>, 1152>` |
| (256, 512] | `Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<96, 128, 64, 32, 32, 3, 1>, 1152>` | `Bf16ScheduleInstance<Bf16A16MmaSchedule<96, 128, 64, 32, 32, 3, 1, Cache::cg, Cache::cg, Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast, Bf16MmaSwizzle::Xor64, 1>, 1152>` |
| (512, 1024] | `Bf16ScheduleInstance< Bf16A16TmaMmaSchedule<64, 128, 64, 32, 32, 2, 2, Bf16MmaRaster::TokenFast, 1, 6>, 1152>` | `Bf16ScheduleInstance< Bf16A16MmaSchedule<64, 128, 64, 32, 32, 2, 2, Cache::cg, Cache::cg, Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast, Bf16MmaSwizzle::Xor64, 1>, 1152>` |
| (1024, inf] | `Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<96, 64, 64, 48, 32, 2, 2>, 1152>` | `Bf16ScheduleInstance<Bf16A16MmaSchedule<96, 64, 64, 48, 32, 2, 2, Cache::cg, Cache::cg, Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast, Bf16MmaSwizzle::Xor64, 1>, 1152>` |

current `#else` fallback: `S1` (`mma`)

### `n4304_k1152.cu`

| token range | TMA instance (sm_120a) | Ada fallback (sm_89) |
|---|---|---|
| (32, 96] | `Bf16RowTailSchedule< Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<64, 32, 64, 32, 16, 3, 2>, 1152>>` | `Bf16RowTailSchedule< Bf16ScheduleInstance<Bf16A16MmaSchedule<64, 32, 64, 32, 16, 3, 2, Cache::cg, Cache::cg, Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast, Bf16MmaSwizzle::Xor64, 1>, 1152>>` |
| (96, 128] | `Bf16RowTailSchedule< Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<64, 64, 64, 32, 16, 3, 2>, 1152>>` | `Bf16RowTailSchedule< Bf16ScheduleInstance<Bf16A16MmaSchedule<64, 64, 64, 32, 16, 3, 2, Cache::cg, Cache::cg, Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast, Bf16MmaSwizzle::Xor64, 1>, 1152>>` |
| (128, 256] | `Bf16RowTailSchedule< Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<64, 64, 64, 32, 16, 2, 2>, 1152>>` | `Bf16RowTailSchedule< Bf16ScheduleInstance<Bf16A16MmaSchedule<64, 64, 64, 32, 16, 2, 2, Cache::cg, Cache::cg, Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast, Bf16MmaSwizzle::Xor64, 1>, 1152>>` |
| (256, 512] | `Bf16RowTailSchedule< Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<128, 128, 64, 64, 32, 3, 1>, 1152>>` | `Bf16RowTailSchedule< Bf16ScheduleInstance<Bf16A16MmaSchedule<128, 128, 64, 64, 32, 3, 1, Cache::cg, Cache::cg, Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast, Bf16MmaSwizzle::Xor64, 1>, 1152>>` |
| (512, 1024] | `Bf16RowTailSchedule< Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<64, 64, 64, 32, 16, 3, 2>, 1152>>` | `Bf16RowTailSchedule< Bf16ScheduleInstance<Bf16A16MmaSchedule<64, 64, 64, 32, 16, 3, 2, Cache::cg, Cache::cg, Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast, Bf16MmaSwizzle::Xor64, 1>, 1152>>` |
| (1024, 2048] | `Bf16RowTailSchedule<Bf16A16TmaMmaSchedule<96, 64, 64, 48, 32, 2, 2>>` | `Bf16RowTailSchedule<Bf16A16MmaSchedule<96, 64, 64, 48, 32, 2, 2, Cache::cg, Cache::cg, Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast, Bf16MmaSwizzle::Xor64, 1>>` |
| (2048, inf] | `Bf16RowTailSchedule< Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<64, 128, 64, 64, 32, 2, 2>, 1152>>` | `Bf16RowTailSchedule< Bf16ScheduleInstance<Bf16A16MmaSchedule<64, 128, 64, 64, 32, 2, 2, Cache::cg, Cache::cg, Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast, Bf16MmaSwizzle::Xor64, 1>, 1152>>` |

current `#else` fallback: `S0` (`mma`)

### `n4608_k4608.cu`

| token range | TMA instance (sm_120a) | Ada fallback (sm_89) |
|---|---|---|
| (32, 64] | `Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<64, 32, 128, 16, 16, 3, 1>, 4608>` | `Bf16ScheduleInstance<Bf16A16MmaSchedule<64, 32, 128, 16, 16, 3, 1, Cache::cg, Cache::cg, Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast, Bf16MmaSwizzle::Xor64, 1>, 4608>` |
| (64, 96] | `Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<96, 32, 128, 16, 16, 3, 1>, 4608>` | `Bf16ScheduleInstance<Bf16A16MmaSchedule<96, 32, 128, 16, 16, 3, 1, Cache::cg, Cache::cg, Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast, Bf16MmaSwizzle::Xor64, 1>, 4608>` |
| (96, 128] | `Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<64, 64, 128, 32, 16, 3, 1>, 4608>` | `Bf16ScheduleInstance<Bf16A16MmaSchedule<64, 64, 128, 32, 16, 3, 1, Cache::cg, Cache::cg, Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast, Bf16MmaSwizzle::Xor64, 1>, 4608>` |
| (128, 256] | `Bf16ScheduleInstance< Bf16A16TmaMmaSchedule<128, 64, 64, 32, 32, 4, 1, Bf16MmaRaster::Grouped, 4>, 4608>` | `Bf16ScheduleInstance< Bf16A16MmaSchedule<128, 64, 64, 32, 32, 4, 1, Cache::cg, Cache::cg, Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::Grouped, Bf16MmaSwizzle::Xor64, 4>, 4608>` |
| (256, 512] | `Bf16A16TmaMmaSchedule<128, 128, 64, 64, 32, 3, 1>` | `Bf16A16MmaSchedule<128, 128, 64, 64, 32, 3, 1, Cache::cg, Cache::cg, Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast, Bf16MmaSwizzle::Xor64, 1>` |
| (512, inf] | `Bf16ScheduleInstance< Bf16A16TmaMmaSchedule<96, 64, 64, 48, 32, 2, 2, Bf16MmaRaster::TokenFast, 1, 9>, 4608>` | `Bf16ScheduleInstance< Bf16A16MmaSchedule<96, 64, 64, 48, 32, 2, 2, Cache::cg, Cache::cg, Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast, Bf16MmaSwizzle::Xor64, 1>, 4608>` |

current `#else` fallback: `S5` (`mma`)
