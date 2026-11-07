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

- **Wired in and measured on sm_89 (RTX 4090).** The `#else` branches of all sixteen
  ladders below run the recorded cp.async fallback per token range. Verified by
  `ninfer_linear_bf16_a16_test` and by interleaved A/B `linear_bench --sweep 512:4096:512`
  (min, repeat 200, flush 96 MiB): every wired rung is a win or neutral except the two
  documented deviations in `n1152_k1152.cu` and `n1152_k4304.cu` (see below).
- **Measured deviations.** `n1152_k1152`: the 32x64 rung regressed +21% at (128, 256] and
  the 64x128 / 128x128 rungs wave-quantize (9 row tiles x tokens/128 = 144 blocks = 2
  unbalanced waves on 128 SMs), so that shape runs S5 to 256 and the 64x64 F7 beyond.
  `n1152_k4304`: the 64x64 / 64x128 rungs regressed +23%/+14% over (256, 1024] and the
  128x128 rung quantizes at 2048 tokens, so those spans keep S4 while (116, 256] and
  (1280, 1792] / (2048, inf] keep the wired fallbacks.
- **Stage 2 is deferred** (no target model family): every shape here is inert for the
  current baseline, qwen3.8-27b (hidden = 5120).
- Regenerate with `python tools/port/gen_bf16_fallback_table.py`. The generator reads the
  `#if NINFER_ENABLE_TMA` ladder only; the two measured deviations live in the `#else`
  branches and are not regenerated.

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

## T004 初稿：上游新增 bf16 形状的 sm_89 fallback（待 T016 复核/实测）

> 范围：`upstream/master`（a754fcf3）中**本地树尚无同名 shape**、且不在 T001 判死组
> （`*/dgx_spark/*`，本次排除 22 个 dgx_spark bf16 梯度）里的 bf16 TMA 梯度：
> `linear/add/bf16` ×1、`linear/split/bf16/rtx5090` ×3、`linear/split/bf16` 共享 ×1，
> 共 5 个 shape / 20 行。映射规则同上，仅换生产者 swizzle（Tma128→Xor64）并丢掉
> ConsumerKUnroll；**未编译、未实测**，行内 token 区间只表达上游的分档意图。
>
> 生成命令：
> `python tools/port/gen_bf16_fallback_table.py --shapes <upstream>/src/ops/linear --recursive
> --instances <upstream>/src/ops/linear/kernels/bf16/bf16_instances.cuh --draft
> --only split/bf16/rtx5090 --only split/bf16/shapes/ --only add/bf16`
>
> 上游这批文件没有 `#if defined(NINFER_ENABLE_TMA)`（上游只按目录/目标选编译单元），搬入时
> 需先补门控，`#else` 分支按本表落 cp.async 回退，再由 T016 实测确认/修正。
>
> 新增入口 `--recursive`（遍历 shapes/）、`--instances`（解析 `Bf16A16TmaR64T64K64S3` 这类别名）、
> `--only`（路径过滤）、`--draft`（只出 banner + 梯度表），默认输出与旧版逐字一致。

### `n5120_k6144.cu` — `linear/add/bf16`

`src/ops/linear/add/bf16/shapes/n5120_k6144.cu`

| token range | TMA instance (sm_120a) | Ada fallback (sm_89) |
|---|---|---|
| (64, 96] | `Bf16A16TmaMmaSchedule<64, 32, 64, 32, 16, 3>` | `Bf16A16MmaSchedule<64, 32, 64, 32, 16, 3, Cache::cg, Cache::cg, Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast, Bf16MmaSwizzle::Xor64, 1>` |
| (96, 128] | `Bf16A16TmaMmaSchedule<64, 64, 128, 32, 32, 2>` | `Bf16A16MmaSchedule<64, 64, 128, 32, 32, 2, Cache::cg, Cache::cg, Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast, Bf16MmaSwizzle::Xor64, 1>` |
| (128, 256] | `Bf16A16TmaMmaSchedule<64, 64, 64, 32, 32, 3>` | `Bf16A16MmaSchedule<64, 64, 64, 32, 32, 3, Cache::cg, Cache::cg, Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast, Bf16MmaSwizzle::Xor64, 1>` |
| (256, inf] | `Bf16A16TmaMmaSchedule<64, 128, 64, 32, 32, 2>` | `Bf16A16MmaSchedule<64, 128, 64, 32, 32, 2, Cache::cg, Cache::cg, Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast, Bf16MmaSwizzle::Xor64, 1>` |

current `#else` fallback: `None`（上游无门控，搬入时补）

### `n1152_k2560.cu` — `linear/split/bf16/rtx5090`

`src/ops/linear/split/bf16/rtx5090/shapes/n1152_k2560.cu`

| token range | TMA instance (sm_120a) | Ada fallback (sm_89) |
|---|---|---|
| (128, 256] | `Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<64, 32, 64, 16, 16, 3>, 2560>` | `Bf16ScheduleInstance<Bf16A16MmaSchedule<64, 32, 64, 16, 16, 3, Cache::cg, Cache::cg, Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast, Bf16MmaSwizzle::Xor64, 1>, 2560>` |
| (256, 512] | `Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<64, 64, 64, 32, 32, 3>, 2560>` | `Bf16ScheduleInstance<Bf16A16MmaSchedule<64, 64, 64, 32, 32, 3, Cache::cg, Cache::cg, Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast, Bf16MmaSwizzle::Xor64, 1>, 2560>` |
| (512, 1024] | `Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<64, 128, 64, 32, 32, 2>, 2560>` | `Bf16ScheduleInstance<Bf16A16MmaSchedule<64, 128, 64, 32, 32, 2, Cache::cg, Cache::cg, Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast, Bf16MmaSwizzle::Xor64, 1>, 2560>` |
| (1024, inf] | `Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<128, 128, 64, 64, 32, 3>, 2560>` | `Bf16ScheduleInstance<Bf16A16MmaSchedule<128, 128, 64, 64, 32, 3, Cache::cg, Cache::cg, Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast, Bf16MmaSwizzle::Xor64, 1>, 2560>` |

current `#else` fallback: `None`（上游无门控，搬入时补）

### `n13952_k2560.cu` — `linear/split/bf16/rtx5090`

`src/ops/linear/split/bf16/rtx5090/shapes/n13952_k2560.cu`

| token range | TMA instance (sm_120a) | Ada fallback (sm_89) |
|---|---|---|
| (48, 64] | `Bf16A16TmaMmaSchedule<128, 64, 128, 32, 32, 2, 1>` | `Bf16A16MmaSchedule<128, 64, 128, 32, 32, 2, 1, Cache::cg, Cache::cg, Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast, Bf16MmaSwizzle::Xor64, 1>` |
| (64, 96] | `Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<128, 32, 128, 16, 16, 2, 1>, 2560>` | `Bf16ScheduleInstance<Bf16A16MmaSchedule<128, 32, 128, 16, 16, 2, 1, Cache::cg, Cache::cg, Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast, Bf16MmaSwizzle::Xor64, 1>, 2560>` |
| (96, 128] | `Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<128, 48, 64, 32, 16, 2, 1>, 2560>` | `Bf16ScheduleInstance<Bf16A16MmaSchedule<128, 48, 64, 32, 16, 2, 1, Cache::cg, Cache::cg, Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast, Bf16MmaSwizzle::Xor64, 1>, 2560>` |
| (128, 512] | `Bf16A16TmaMmaSchedule<64, 64, 64, 32, 32, 3, 2>` | `Bf16A16MmaSchedule<64, 64, 64, 32, 32, 3, 2, Cache::cg, Cache::cg, Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast, Bf16MmaSwizzle::Xor64, 1>` |
| (512, inf] | `Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<64, 128, 64, 32, 32, 3, 1>, 2560>` | `Bf16ScheduleInstance<Bf16A16MmaSchedule<64, 128, 64, 32, 32, 3, 1, Cache::cg, Cache::cg, Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast, Bf16MmaSwizzle::Xor64, 1>, 2560>` |

current `#else` fallback: `None`（上游无门控，搬入时补）

### `n1664_k2560.cu` — `linear/split/bf16/rtx5090`

`src/ops/linear/split/bf16/rtx5090/shapes/n1664_k2560.cu`

| token range | TMA instance (sm_120a) | Ada fallback (sm_89) |
|---|---|---|
| (96, 128] | `Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<64, 32, 64, 16, 16, 3, 2>, 2560>` | `Bf16ScheduleInstance<Bf16A16MmaSchedule<64, 32, 64, 16, 16, 3, 2, Cache::cg, Cache::cg, Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast, Bf16MmaSwizzle::Xor64, 1>, 2560>` |
| (128, 512] | `Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<128, 48, 64, 32, 16, 2, 1>, 2560>` | `Bf16ScheduleInstance<Bf16A16MmaSchedule<128, 48, 64, 32, 16, 2, 1, Cache::cg, Cache::cg, Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast, Bf16MmaSwizzle::Xor64, 1>, 2560>` |
| (512, inf] | `Bf16ScheduleInstance<Bf16A16TmaMmaSchedule<128, 80, 64, 32, 16, 3, 1>, 2560>` | `Bf16ScheduleInstance<Bf16A16MmaSchedule<128, 80, 64, 32, 16, 3, 1, Cache::cg, Cache::cg, Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast, Bf16MmaSwizzle::Xor64, 1>, 2560>` |

current `#else` fallback: `None`（上游无门控，搬入时补）

### `n14336_k5120.cu` — `linear/split/bf16`

`src/ops/linear/split/bf16/shapes/n14336_k5120.cu`

| token range | TMA instance (sm_120a) | Ada fallback (sm_89) |
|---|---|---|
| (32, 64] | `Bf16A16TmaMmaSchedule<64, 64, 64, 32, 32, 3>` | `Bf16A16MmaSchedule<64, 64, 64, 32, 32, 3, Cache::cg, Cache::cg, Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast, Bf16MmaSwizzle::Xor64, 1>` |
| (96, 128] | `Bf16A16TmaMmaSchedule<64, 128, 64, 32, 32, 2>` | `Bf16A16MmaSchedule<64, 128, 64, 32, 32, 2, Cache::cg, Cache::cg, Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast, Bf16MmaSwizzle::Xor64, 1>` |
| (128, 192] | `Bf16A16TmaMmaSchedule<64, 64, 64, 32, 32, 3>` | `Bf16A16MmaSchedule<64, 64, 64, 32, 32, 3, Cache::cg, Cache::cg, Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast, Bf16MmaSwizzle::Xor64, 1>` |
| (192, inf] | `Bf16A16TmaMmaSchedule<64, 128, 64, 32, 32, 2>` | `Bf16A16MmaSchedule<64, 128, 64, 32, 32, 2, Cache::cg, Cache::cg, Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast, Bf16MmaSwizzle::Xor64, 1>` |

current `#else` fallback: `None`（上游无门控，搬入时补）

## T004 登记：上游新增的量化 / 新家族形状（无机械映射，待移植时实测确定）

上游 305 提交把 `src/ops/**/shapes/` 重排为「按 family + n/k 拆文件」：上游 shape 文件 268 个，
本地有同名对应 117 个，上游独有 151 个。其中带 TMA 梯度的 81 个（按调度类名统计）：非判死组
（`*/dgx_spark/*`）13 个，另有 2 个 bf16 shape 经 `bf16_instances.cuh` 别名走 TMA（调度类名统计不到），
合计 **15 个需要 sm_89 回退**。

- 5 个 bf16（`linear/add/bf16` ×1、`linear/split/bf16/rtx5090` ×3、`linear/split/bf16` 共享 ×1）已在上一节
  按 bf16 映射规则出初稿（20 行）。
- 其余 10 个量化 shape 与 4 个新 kernel 家族**没有可验证的机械映射**，下表只登记处置方式，不写死精确映射。
- `dgx_spark` 的 68 个 TMA shape 属 T001 判死组（Spark/DGX），不生成初稿。

| 家族 | 上游新增（非判死） | TMA 梯度 | sm_89 回退处置 | 依据 / 备注 |
|---|---|---|---|---|
| `linear/add/fp8` | 2（`n5120_k6144`、`n5120_k17408`） | `Fp8A8TmaMmaSchedule` / `Fp8A8TmaSplitKSchedule` | **待移植时实测确定** | 候选规则：`Fp8A8TmaMmaSchedule : Fp8A8MmaSchedule<…>`（`kernels/fp8/fp8_schedule.cuh:175`，差别仅在 `kTmaSwizzle` / 生产者线程数），与 bf16 同构；但本表只实测过 bf16，fp8 需另行验证 |
| `linear/split/fp8` | 2（`n14336_k5120/a8`、`n16384_k5120/a8`） | 同上 | **待移植时实测确定** | 与 add 共用 fp8 A8 调度族 |
| `linear/swiglu/fp8` | 1（`n34816_k5120/a8`） | 同上 | **待移植时实测确定** | 同上 |
| `linear/add/nvfp4` | 2（`n5120_k6144_tma`、`n5120_k17408_tma`） | `Nvfp4A4TmaMmaSchedule` | **待移植时实测确定（大概率不可移植）** | NVFP4 权重/张量核在 sm_89 不存在，回退只能是「去量化到 bf16/FP8 再走 cp.async 路径」，属另一条数值路径，须实测精度与性能 |
| `linear/split/nvfp4` | 2（`n14336_k5120/a4_tma`、`n16384_k5120/a4_tma`） | 同上 | 同上 | 同上 |
| `linear/swiglu/nvfp4` | 1（`n34816_k5120/a4_tma`） | 同上 | 同上 | 同上 |
| `linear/{projection,split}/q8`、`q4`、`q5`、`q6` | 0 | — | 不涉及 | 上游新增的 q4/q8 TMA shape 全在 `dgx_spark`（判死）；`rtx5090` 侧无新增 TMA shape |
| QSA（`softmax_attention/qsa`） | 42 个文件（`indexer` / `prepare` / `sparse_gqa` / `tail_extract`） | 0 | **待移植时实测确定** | 新算法家族，无 bf16 TMA 梯度，本表机制不适用；先按 sm_89 可执行性（SM89 的 ldsm/mma.sync + cp.async）逐 kernel 评估 |
| PLE（`ops/ple`） | 7 个文件（`conv` / `gate` / `lookup` / `state_extract`） | 0 | **待移植时实测确定** | 同上，且访存形态（lookup/gather）与 linear 不同 |
| GDN（`linear_attention/gated_delta_net`） | 97 个文件 | 11（全在 `input_conv`：fp8 ×1、nvfp4 ×3、q8-dgx ×7） | **待移植时实测确定** | `control/bf16` 的 bf16 梯度（`n64_k2048`、`n96_k5120`、rtx5090 `n96_k2560`）**不带 TMA**，故不产生本表行；带 TMA 的部分都是量化输入投影 |
| MoE execution（`ops/moe`） | 129 个文件 | 8（7 个 `dgx_spark` + `swiglu/kernel/q8_down.cuh`、`swiglu/dispatch.h`） | **待移植时实测确定** | `topk_softmax_swiglu_moe` / `topk_softmax_route` 的 shape 目录只有 `dispatch.cpp`（无 TMA 梯度）；D2/D3 已本地落地的是 CPU 专家路径，与这些 CUDA 融合 shape 无关 |

> 判定口径：只有「同族调度类继承关系可读且仅差 swizzle」才写机械映射；其余一律登记为
> 待实测，避免在计划里写入未验证的精确类型。T016 复核时应把这些行替换为实测结论或删除。
