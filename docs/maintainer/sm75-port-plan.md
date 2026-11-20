# Porting NInfer to RTX 2080 Ti (sm_75, Turing)

This branch (`sm75-2080ti`) wires the RTX 2080 Ti class (Turing, `sm_75`) into NInfer's
build system, runtime capability gate, and CMake presets. The **Turing tensor-core kernel
port itself is not complete on this branch**: the configure step for `rtx2080ti` fails
with an explicit guard, and the gap analysis below records exactly what must be ported
before the target can build.

## What this branch already lands

- **Build target `rtx2080ti`** (`cmake/NinferPlatform.cmake`): maps to
  `CMAKE_CUDA_ARCHITECTURES=75`; added to the `NINFER_TARGET` cache strings and the
  `win-2080ti-release` preset (`CMakePresets.json`).
- **Runtime capability gate** (`src/models/qwen3_5/program/planning/startup.cpp`): the
  Qwen3.5 family runtime now accepts compute capability `75` alongside `89` and `120`,
  with the same explicit-rejection behaviour for NVFP4 weights/KV as on sm_89.
- **Feature gates** (root `CMakeLists.txt`): the `rtx2080ti` build defines
  `NINFER_ARCH_SM75=1` and `NINFER_ENABLE_K8V4=1`, and does **not** define the
  Blackwell-only `NINFER_ARCH_SM120` / `NINFER_ENABLE_TMA` / `NINFER_ENABLE_NVFP4`
  macros, so every Blackwell-only path is compiled out exactly as on sm_89.
- **Configure-time guard** (`cmake/NinferPlatform.cmake`): selecting `rtx2080ti` fails
  configure with a pointer to this document, instead of a build that dies deep in ptxas.
- **Documentation**: `README.md` and `docs/maintainer/build-system.md` name the third
  target and its skeleton state.

## Why sm_75 cannot build today

A full `rtx2080ti` build attempt (CUDA 13.5, MSVC 2022, this tree) reduced every failure
to three hardware-missing features. Turing (sm_75) predates all of them:

| Missing on sm_75 | Required by | First seen in build |
|---|---|
| `cp.async` / `cp.async.commit_group` / `cp.async.wait_group` (sm_80+) | the entire sm_89 fallback pipeline strategy (`src/ops/common/cuda/pipeline.cuh`) | `w4a8_gemm.cu`, every `cp_async*` call site |
| `mma.sync ... m16n8k16 ... bf16` (sm_80+) | every BF16 GEMM/attention/MoE kernel (`mma_bf16` in `src/ops/common/cuda/mma.cuh`) | `bf16_attn_input_gemm_mma.cu`, `fp8_attn_input_a16_gemm.cu`, `sampling.cu` |
| `mma.sync ... m16n8k32 ... e4m3` (sm_89+) | the FP8 A16/A8 GEMM family and K8V4 value-plane decode (`mma_fp8_e4m3`) | `fp8_attn_input_a8.cu`, `fp8_attn_input_a16_small_t.cu`, `block_scaled/fp8/launch.cu` |
| `__reduce_max_sync` (sm_80+) | warp reduction in `src/ops/kernel/sampling_device.cuh` (lines 184, 186, 223) | `launcher/sampling.cu`, `launcher/speculative_round.cu` |

`ldmatrix` (sm_75+), INT8 `mma.sync m16n8k32 s8` (sm_75+), and `__dp4a` (sm_75+) are all
available on Turing and are **not** part of the gap.

The structural point: the sm_89 port's fallback model is "cp.async + ldmatrix +
mma.sync", and its three pillars are all post-Turing. Turing has no BF16 or FP8 tensor
cores at all — its tensor cores are FP16, TF32, and INT8 — so the port cannot simply
re-target the existing kernels; the arithmetic plane changes.

## Port strategy (work items)

The goal is the same product surface as the RTX 4090 target: every groupwise-int path
(q4/q5/q6/q8 weights), FP8-weight path, BF16/INT8/FP8/K8V4 KV, and the full Qwen3.5
Dense + MoE model family, running on a 22 GB 2080 Ti.

### S1 — Portable sampler reduction (small, low risk)

Replace the three `__reduce_max_sync` calls in `src/ops/kernel/sampling_device.cuh` with
the `__shfl_down_sync` tree reduction already used in `sampling_block_max_key` (lines
62–81). Identical semantics, runs on every architecture. This unblocks
`launcher/sampling.cu` and `launcher/speculative_round.cu` immediately.

### S2 — Turing pipeline primitive (the load-bearing change)

`src/ops/common/cuda/pipeline.cuh` currently exposes `cp_async` / `cp_async_zfill` /
`cp_commit` / `cp_wait` (Ampere+) and the TMA bulk path (sm_90+, already gated behind
`NINFER_ENABLE_TMA`). Add a `NINFER_ARCH_SM75` branch that provides the same interface
over the pre-Ampere `ld.global → st.shared` double-buffered pipeline (equivalent to the
CUDA `__pipeline_memcpy_async` / `__pipeline_commit` / `__pipeline_wait_prior` intrinsics,
which are already present in the file at lines 123–131 but unused). Every `cp_async*`
call site in the tree then compiles for sm_75 without per-file edits.

### S3 — Arithmetic-plane fallback (the large item)

The BF16 and FP8 tensor-core planes do not exist on Turing. Two candidate planes are
available on sm_75 and already have MMA wrappers in `mma.cuh`:

- **FP16 tensor cores** (`mma.sync m16n8k16 f16`, `mma_f16`): same tile shape as the
  BF16 path. BF16 operands convert to FP16 losslessly for values in the FP16 range; the
  exponent-range difference (FP16 5-bit vs BF16 8-bit) means extreme logits/activations
  need the same scaling discipline the existing kernels already apply.
- **TF32 tensor cores** (`mma.sync m16n8k8 tf32`, `mma_tf32`): higher precision, smaller
  K per instruction; a natural fit for the BF16 GEMM family.

Per the project rule that a fallback changing the arithmetic profile must be re-qualified
(see `sm89-port-sync.md`, Tier C), the chosen plane must be qualified against the
independent mathematical oracle for every Op it touches — kernel parity or plausible
output is not acceptance. NVFP4 has no sm_75 path by design (the dispatch already
rejects it with an explicit error, same as sm_89).

### S4 — FP8-weight path

Turing has no FP8 tensor cores. The FP8 A16/A8 GEMM family (and the K8V4 value-plane
decode) must either dequantize to the FP16/TF32 plane of S3 or route to the groupwise-int
path. The groupwise-int artifact remains the primary 2080 Ti artifact, mirroring the
RTX 4090 guidance in `README.md`.

### S5 — Shape ladders and measurement

The measured sm_89 ladders (`sm89-fallback-table.md`) are **not** valid on sm_75:
different tensor-core throughput, no `cp.async` (the S2 pipeline has different latency),
and a different SM count (68 vs 128). Unmeasured shapes ride the generic rungs. Every
ladder value that is claimed for sm_75 must be measured on an actual RTX 2080 Ti before
it is recorded; the development box here has an RTX 4090, which cannot execute sm_75
binaries.

### S6 — Verification

- Op-level: the existing Op test suites (`tests/ops/`) against their oracles, registered
  for the `rtx2080ti` build with the same NVFP4-only-skip pattern used for sm_89
  (`tests/ops/tests.cmake` lines 106–112).
- End-to-end: a groupwise-int artifact loaded and served on a physical RTX 2080 Ti, with
  the PPL/quality gates the model cards define.

## Status

| Item | State on `sm75-2080ti` |
|---|---|
| Build target, presets, feature gates | **Landed** |
| Configure-time guard | **Landed** |
| Runtime capability gate (startup.cpp) | **Landed** |
| S1 sampler reduction | Pending |
| S2 Turing pipeline primitive | Pending |
| S3 arithmetic-plane fallback + qualification | Pending |
| S4 FP8-weight path | Pending |
| S5 ladder measurement on a real 2080 Ti | Pending (needs the hardware) |
| S6 test/e2e verification | Pending |

Remove the configure-time guard in `cmake/NinferPlatform.cmake` once S1–S4 are landed and
the tree compiles for `sm_75`; keep S5/S6 evidence in the per-shape record before the
target is called ported. (When the port lands, the `win-2080ti-release` preset and the
README target description become fully operative and the "skeleton" wording is removed.)
