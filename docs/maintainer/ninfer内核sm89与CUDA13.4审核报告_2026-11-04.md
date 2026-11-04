# NInfer 内核 sm_89/CUDA 13.4 调优审核与 BUG 修复执行清单

- **审核对象**：`D:\TeleAgentWork\ninfer`，提交 `5635ddd`（2026-11-04 14:41 +0800，master 最新）
- **审核目标**：sm_89（RTX 4090）+ CUDA 13.4 调优完成度核实；接口/逻辑 BUG 核实
- **审核方式**：CodeGraph 索引（1,770 文件 / 38,299 节点 / 120,671 边）+ 项目自带门控检查器 + 双通道子代理全量审读 + 人工抽验（关键发现均已在源码逐行确认）
- **约束**：本清单为只读审核产物，未修改任何项目代码；所有伪代码为参考逻辑，落地时按实际上下文调整

## 一、总体结论

**sm_89 移植质量整体较高**：主路径算子（qwen3.8-27B 的 Q4/Q5 projection、Q8 lm_head、int8/rk/k8v4 attention、w4a8 int8-mma、sparse MoE cp.async+mma）全部运行 Ada 原生内核；fail-loud 门控（TMA/NVFP4/PDL）在 sm_89 上退化正确（`check_blackwell_guards.py`：1127 文件零未门控调用点）；CUDA 13.4 适配到位（`-Xptxas=-v`、`--split-compile=0`、`__maxnreg__(120)`、`cudaFuncSetAttribute` 超过 48KB smem、无废弃 API）。

**两类实质缺口**：
1. **bf16 线性形状的 sm_89 大 token 区间调优未完成**——16 个 TMA-ladder 文件的 `#else` 回退仍是"单档保守复用"（6~7 档 tile 折叠为 1 档），项目自己在 `docs/maintainer/sm89-fallback-table.md` 记录的 like-for-like Ada 替换**全部未接线**（该文档自述 "Recorded, not wired in"）。对当前基线 qwen3.8-27B（hidden=5120）惰性，但 2560/4608/1152 系（Qwen3.6 系/DFlash）大 token prefill 段 tile 减半以上。
2. **提交 5635ddd 的同族 BUG 残留**——该提交修复了 wide-batch 内核的 `score_offset` 缺失，但 int8/bf16 **tiled** 内核的同类路径未同步修复（已逐行核实）；另有 2 个该提交新约定未覆盖的独立缺陷（见 BUG 清单）。

**各目录调优完成率**（按内核加权）：

| 目录 | 状态 | 完成率 | 主要缺口 |
|---|---|---:|---|
| linear/bf16 16 个 TMA-ladder 文件 | ❌ 保守回退 | 15% | fallback-table 未接线（K1） |
| linear/bf16 k5120/k6144 三件套 | ⚠️ 粗粒度 | 55% | 单档 R32T32K256S3 通吃（K2） |
| launcher/kernel（mtp_pack 等） | ❌/⚠️ | 85% | mtp_pack 标量未向量化（K3）；24 文件缺 launch_bounds（K4） |
| softmax_attention | ⚠️ | 92% | tiled 内核缺 lb；B1/B3 逻辑缺陷 |
| linear q4/q5/q6/q8 | ⚠️ | 80% | 档位边界沿用 5090 测量无 4090 复测（K8） |
| sparse_moe / moe / kv_cache / rmsnorm_rope / w4a8 | ✅ | 95% | 仅缺 lb 声明 |
| fp8 / nvfp4 桩 / act_quant | ✅ / ✅ / ⚠️ | 100% / 100% / 40% | act_quant 为调试探针（K9） |
| CUDA 13.4 基建 | ✅ | 98% | pipe_* 死代码（K10）；NVFP4 MoE 断线宏（K6） |

---

## 二、执行顺序总表（agent 顺序执行）

按"风险从低到高 + 收益从高到低 + 独立项优先"排序。每项独立可执行，项内步骤完整。

| 序 | 任务 | 类型 | 改动面 | 预期收益/风险 | 依赖 |
|---|---|---|---|---|---|
| 1 | B1 tiled 内核 page_score 补 score_offset | BUG-P1 | 2 行 ×2 文件 | 修复 5635ddd 同族残留 | 无 |
| 2 | B3A cached 路径 page_score 静默丢弃显式化 | BUG-P2 | ~10 行 | 消除 API 行为不一致 | 无 |
| 3 | K3 mtp_pack 两内核向量化 | 调优 | 2 内核 | MTP 每步实跑，带宽段 3~4× | 无 |
| 4 | K4 24 个零-launch_bounds 内核补声明 | 调优 | ~24 处单行 | 寄存器溢出防护、调度稳定 | 无 |
| 5 | K1 16 个 bf16 ladder 文件接线 fallback-table | 调优 | 16 文件 | 2560 系大 token 段 tile 6 档恢复 | 无 |
| 6 | K2 k5120/k6144 三件套区间化 | 调优 | 3 文件 | hidden=5120 主路径 bf16 变体 | 可参考 K1 模式 |
| 7 | K5 causal_conv1d 标量 prefill 内核收敛 | 调优 | 1 内核 | GDN 路径统一 | 无 |
| 8 | K6 NVFP4 MoE 断线宏修复 | 调优(sm_120a) | 1~3 行 | sm_120a 侧 fail-loud 一致性 | 无 |
| 9 | B2 NVFP4 KV B>1/W>16 路径（短期诊断化） | BUG-P1(sm_120a) | ~8 行 | 规划期可诊断失败 | 无 |
| 10 | K7 rope CTA residency 常数 4090 复测 | 测量 | 常数 | 消除移植未复测参数 | 需 4090 实机 |
| 11 | K8 q4/q8 shape 档位边界 4090 sweep | 测量 | 配置表 | 建立 sm89 专属边界机制 | 需 4090 实机 |
| 12 | K9 act_quant 向量化（可选） | 调优 | 1 内核 | 非生产路径，低优先 | 无 |
| 13 | K10 pipe_* 死代码清理（可选） | 清理 | 1 文件 | 减噪 | 无 |

统一验证流程（每项完成后执行）：
```powershell
# 1. 编译（sm_89 / MSVC vcpkg 路线；或 Linux+Ninja）
cmake -S . -B build-win -G Ninja -DCMAKE_BUILD_TYPE=Release `
  -DCMAKE_CUDA_ARCHITECTURES=89 `
  -DCMAKE_TOOLCHAIN_FILE=<vcpkg-root>/scripts/buildsystems/vcpkg.cmake `
  -DVCPKG_TARGET_TRIPLET=x64-windows
cmake --build build-win -j
# 2. Blackwell 门控守卫（防止新增未门控 TMA 引用）
python tools\port\check_blackwell_guards.py
# 3. 测试（该项关联的目标，见各任务"验证"节）
ctest --test-dir build-win --timeout 180 --output-on-failure -R <目标正则>
# 4. 基准（性能相关项）
#    bench/inference/ninfer_bench 相关 case，详见 docs/performance/rtx-4090-sm89-baseline.md
```

---

## 三、BUG 清单（含修复伪代码）

### B1【P1】int8/bf16 tiled 内核 page_score override 未加行偏移

- **位置**：`src/ops/softmax_attention/dense/causal_cache/int8/tiled_mma.cuh:97`、`src/ops/softmax_attention/dense/causal_cache/bf16/tiled_mma.cuh:60`
- **性质**：5635ddd 修复了 `int8/batch_wide_mma.cuh:107-108` 与 `common/mxfp8_tiled_mma.cuh:91-93`（改为 `page_score + metadata.score_offset()`），但 int8/bf16 的 tiled 内核仍是旧约定——调用方绑定得分矩阵基址时，所有请求的票都写入第 0 行，页得分热图丢行，污染 KV 驱逐决策。
- **证据（当前代码）**：
  ```cpp
  // int8/tiled_mma.cuh:97（bf16/tiled_mma.cuh:60 同构）
  unsigned* const counters = page_score != nullptr ? page_score : metadata.counters();
  ```
  对照已修复版（batch_wide_mma.cuh:107-108）：
  ```cpp
  unsigned* const counters =
      page_score != nullptr ? page_score + metadata.score_offset() : metadata.counters();
  ```
  `PagedKVBatchMetadata::score_offset()`（`src/ops/kernel/paged_kv_address.cuh:102`）= `table_rows[0] * table_stride`；`PagedKVDirectMetadata::score_offset()`（同文件 :42）恒为 0——因此 direct 元数据路径行为完全不变，仅批式（`table_rows[0] != 0`）路径受影响。
- **触发条件**：批式入口（`int8_kv_append_attention`/`bf16_kv_append_attention`，`launch.cu:478-479` 以 `make_quantized_causal_cache_view` 构造 batch metadata）+ 执行行号 r≠0 + tiled 家族（int8/rk：W>256；bf16：W>256）+ 调用方绑定 page_score 基址。注意 rk 5 种布局共用 int8 tiled 入口，一并受益。
- **修复伪代码（两处同构，直接照抄）**：
  ```diff
  - unsigned* const counters = page_score != nullptr ? page_score : metadata.counters();
  + unsigned* const counters =
  +     page_score != nullptr ? page_score + metadata.score_offset() : metadata.counters();
  ```
  同步更新两处过时注释（int8 版注释仍称 "resolved exactly like block_table"、bf16 版 "P6 2c: the counter planes…"——改为与 batch_wide 一致的 "The caller's override binds the matrix base, so it takes the same row offset the metadata's own plane would"）。
- **验证**：`ctest -R causal_cache`；并建议在 `tests/ops/softmax_attention/causal_cache.cpp` 的 `check_scores`（:2622-2628 现仅断言"非全零"）上增加 r≠0 行的行号断言：绑定 `[pool_rows×4096]` 矩阵、执行第 2 行请求后断言第 2 行非零而第 0 行为零。

### B2【P1，仅 sm_120a/NINFER_ENABLE_NVFP4 构建】Nvfp4Group16 KV 在 B>1 且 W>16 无 wide-batch 路径

- **位置**：`src/ops/softmax_attention/dense/causal_cache/nvfp4/plan.cpp:19`（`(batch > 1 && width > 16)` 直接 throw "invalid plan inputs"）、`nvfp4/launch.cu:128-135`（缺 rk 式 early 分支）、`causal_softmax_attention.cpp:285-298`（early-return-0 名单漏 `Nvfp4Group16`，导致 workspace 查询在 width 循环中途撞 throw）、`nvfp4/plan.cpp:50-52`
- **性质**：5635ddd 提交说明明确"每个 rk 布局当初都栽在这个形状上"——nvfp4 是同型拒绝条件但未随 rk 一起接入 wide 内核。NVFP4 KV 模型（仅 5090）无法做批式宽上下文 prefill/verify。
- **触发条件**：sm_120a 构建 + `--kv-dtype` 为 Nvfp4Group16 + batch>1 且 width>16（如 B=2/4 × W=32/64/128/256）。
- **修复伪代码（短期：把规划期异常变成可诊断的显式契约）**：
  ```cpp
  // causal_softmax_attention.cpp —— workspace capacity 的 wide-batch 专属名单（L285-298）
  // 在 switch 的 default 之前显式拦截：
  if (cache_storage == KvCacheStorage::Nvfp4Group16
      && batch_size > 1 && max_width > kMaximumVerifyTokens) {
      throw std::invalid_argument(
          "causal_softmax_attention: NVFP4 KV does not support batch>1 with width>16 yet; "
          "use width<=16 verify batches or BF16/INT8/FP8 KV storage");
  }
  ```
  **根治方案**（工作量大，建议独立排期）：仿 rk 接入模式——
  ```cpp
  // nvfp4/launch.cu —— 仿 rk_launch.cu:110-185 的 early 分支
  if (q.ne[3] > 1 && q.ne[2] > 16) {
      nvfp4_kv_append_launch(...);                 // 先 standalone append
      nvfp4_kv_batch_wide_attention(...);          // 复用 5635ddd 的 wide 内核骨架，
                                                    // 读回 codec 走 nvfp4/tile_io（与 tiled 相同）
      return;
  }
  // 并把 Nvfp4Group16 加入 causal_softmax_attention.cpp L285-298 的 early-return-0 名单
  // （wide 内核不保留 split-K partials，workspace 为 0——与 INT8/rk 语义一致）
  ```
- **验证**：NVFP4 不可在 4090 测试，需 5090 实机；短期方案可仅做编译验证 + 用 `BFloat16` 布局回归确认 early-return-0 名单未破坏其他路径。

### B3【P2】page_score 对 bf16/fp8/nvfp4/k8v4 的 cached 与 nvfp4 批式路径被静默丢弃

- **位置**：`src/ops/softmax_attention/dense/causal_cache/causal_softmax_attention.cpp:473-477`（bf16 cached）、`479-483`（fp8）、`491-495`（nvfp4）、`504-508`（k8v4）、`398-402`（nvfp4 批式 append）
- **性质**：dispatcher 在 `:466-471` 先对 page_score 做严格校验（dtype/连续性，校验失败会 throw），随后除 `Int8Group64`（`:485-489` 透传）外全部静默丢弃——校验通过却 no-op，且同一 op 参数的行为随 KV 布局 fork。bf16 的 tiled 内核实际具备记录能力（`bf16/tiled_mma.cuh:60` 有 page_score 入参），只是 `bf16_kv_cached_attention` 签名没接。
- **修复伪代码（方案 A：显式拒绝，防御闭环，推荐先做）**：
  ```cpp
  // causal_softmax_attention.cpp cached 分发（每个不支持透传的分支前）：
  if (cache.storage == KvCacheStorage::BFloat16) {
      if (page_score != nullptr) {
          throw std::invalid_argument(
              std::string(op) + ": page_score is only supported for INT8/rk KV cached attention");
      }
      detail::bf16_kv_cached_attention(q, positions, scale, cache, envelope, workspace, out,
                                       execution);
      return;
  }
  // fp8 / nvfp4 / k8v4 分支同构；nvfp4 批式 append 分支（:398-402）同样处理。
  ```
  **方案 B（彻底，工作量较大）**：给 `bf16_kv_cached_attention`/`fp8_kv_cached_attention`/`k8v4_kv_cached_attention`/`nvfp4_kv_cached_attention` 加 `Tensor* page_score` 形参并透传到 tiled 内核（bf16 tiled 与 mxfp8 tiled 内核已支持，仅需接管线）。
- **验证**：`ctest -R softmax_attention`；新增负向用例：bf16 KV + 非空 page_score → 期待 throw。

---

## 四、待修改内核清单（含优化方案与伪代码）

### K1【❌ 最大缺口】16 个 bf16 TMA-ladder 形状文件接入 fallback-table 的 like-for-like Ada 回退

- **范围**：`src/ops/linear/bf16/shapes/` 下 16 个文件（现状全部为单档保守回退）：

  | 文件 | `#else` 行号 | 当前回退 | fallback-table 对应小节 |
  |---|---:|---|---|
  | n10240_k320.cu | L30-34 | `mma<S2>` | `### n10240_k320.cu` |
  | n1152_k1152.cu | L40-44 | `mma<S5>` | `### n1152_k1152.cu` |
  | n1152_k1536.cu | L38-42 | `mma<S4>` | `### n1152_k1536.cu` |
  | n1152_k4304.cu | L52-56 | `mma<S4>` | `### n1152_k4304.cu` |
  | n12800_k2560.cu | L43-47 | `mma<S4>` | `### n12800_k2560.cu` |
  | n13952_k2560.cu | L40-44 | `mma<S4>` | `### n13952_k2560.cu` |
  | n1664_k2560.cu | L35-39 | `mma<S4>` | `### n1664_k2560.cu` |
  | n248320_k2560.cu | L27-31 | `sliced_k_mma<S3>` | `### n248320_k2560.cu` |
  | n2560_k2560.cu | L41-45 | `mma<S4>` | `### n2560_k2560.cu` |
  | n2560_k4608.cu | L43-47 | `sliced_k_mma<S3>` | `### n2560_k4608.cu` |
  | n2560_k6144.cu | L35-39 | `mma<S4>` | `### n2560_k6144.cu` |
  | n320_k10240.cu | L30-34 | `sliced_k_mma<S4>` | `### n320_k10240.cu` |
  | n324_k10240.cu | L35-39 | `sliced_k_mma<S4>` | `### n324_k10240.cu` |
  | n3456_k1152.cu | L33-37 | `mma<S1>` | `### n3456_k1152.cu` |
  | n4304_k1152.cu | L36-40 | `mma<S0>` | `### n4304_k1152.cu` |
  | n4608_k4608.cu | L45-49 | `mma<S5>` | `### n4608_k4608.cu` |

- **改法**：机械替换。映射规则（`docs/maintainer/sm89-fallback-table.md:16-23`）：
  ```
  Bf16A16TmaMmaSchedule<R,T,K,WR,WT,S,M[,Raster[,RGR[,ConsumerKUnroll]]]>
    → Bf16A16MmaSchedule<R,T,K,WR,WT,S,M, Cache::cg, Cache::cg,
                         Bf16MmaFragmentPipeline::PingPong, Raster,
                         Bf16MmaSwizzle::Xor64, RGR>
  ```
  tile 几何/stages/raster/min-blocks 全保留，仅 `Tma128`→`Xor64` swizzle；`ConsumerKUnroll` 丢弃；外层包装（`Bf16ScheduleInstance`/`Bf16RowTailSchedule`/`Bf16KTailSchedule`）原样保留。每个 TMA 实例的 token 区间在 `#else` 分支恢复为对应档位。
- **完整示例（n2560_k6144.cu，其余 15 个文件按各自表小节同构操作）**——
  现状（L30-39）：
  ```cpp
  #if defined(NINFER_ENABLE_TMA)
      if (tokens <= 96)  return launch_bf16_tma_mma<S5>;
      if (tokens <= 128) return launch_bf16_tma_mma<S6>;
      if (tokens <= 512) return launch_bf16_tma_mma<S7>;
      return launch_bf16_tma_mma<S8>;
  #else
      // Ada fallback: no TMA on sm_89; reuse this shape's own cp.async MMA
      // schedule for the token ranges the TMA instances covered.
      return launch_bf16_mma<S4>;
  #endif
  ```
  修改（在匿名命名空间补 4 个 Ada 调度别名，`#else` 分支恢复区间化）：
  ```cpp
  namespace {
  // ... 既有 S0..S8 定义保持不变 ...
  using A5 = Bf16ScheduleInstance<
      Bf16A16MmaSchedule<64, 32, 128, 32, 16, 2, 2, Cache::cg, Cache::cg,
                         Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast,
                         Bf16MmaSwizzle::Xor64, 1>,
      6144>;
  using A6 = Bf16ScheduleInstance<
      Bf16A16MmaSchedule<64, 32, 128, 16, 16, 2, 2, Cache::cg, Cache::cg,
                         Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast,
                         Bf16MmaSwizzle::Xor64, 1>,
      6144>;
  using A7 = Bf16ScheduleInstance<
      Bf16A16MmaSchedule<128, 64, 64, 32, 32, 2, 1, Cache::cg, Cache::cg,
                         Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::Grouped,
                         Bf16MmaSwizzle::Xor64, 4>,
      6144>;
  using A8 = Bf16A16MmaSchedule<128, 128, 64, 64, 32, 3, 1, Cache::cg, Cache::cg,
                               Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast,
                               Bf16MmaSwizzle::Xor64, 1>;
  } // namespace

  Bf16Launch select_bf16_n2560_k6144(std::int32_t tokens) {
      // ... 既有 <=64 及以下分支保持不变 ...
  #if defined(NINFER_ENABLE_TMA)
      // ... 既有 TMA 分支保持不变 ...
  #else
      // Ada fallback: like-for-like cp.async ladders from
      // docs/maintainer/sm89-fallback-table.md (tile geometry preserved, Xor64 swizzle).
      if (tokens <= 96)  return launch_bf16_mma<A5>;
      if (tokens <= 128) return launch_bf16_mma<A6>;
      if (tokens <= 512) return launch_bf16_mma<A7>;
      return launch_bf16_mma<A8>;
  #endif
  }
  ```
- **注意（static asserts，来自 fallback-table.md:28-30）**：每个新调度需满足 `kBlockRows % kWarpRows == 0`、`kBlockTokens % kWarpTokens == 0`、`kBlockK % 64 == 0`、`2 <= kStages <= 8`、`kSharedBytes <= 99 KiB`——表中全部条目已由 `tools/port/gen_bf16_fallback_table.py` 预验证，机械套用即可；编译期 static_assert 会对任何违规立即报错。
- **验证**：`cmake --build build-win -j`（sm_89 编译通过即验证 static asserts）；`python tools/port/check_blackwell_guards.py`；`ctest -R bf16`；性能对比用 `bench/ops/linear_bench.cu`（参照 `docs/maintainer/linear-benchmark.md` 的测量规则），重点形状 n2560_*（Qwen3.6 系 hidden=2560 FFN），对比大 token 段（≥512）新旧吞吐。
- **执行提示**：16 文件彼此独立，可逐文件提交；建议先做 `n2560_k2560.cu`（表节 5 档，最典型）打通流程，再机械推进其余 15 个。

### K2【⚠️】k5120/k6144 三件套（alias 派生）区间化细化

- **范围**：`src/ops/linear/bf16/shapes/n14336_k5120.cu:30-35`、`n256_k5120.cu:19-22`、`n5120_k6144.cu`（28-32）
- **现状**：有区间但粒度粗，大 token 段单一 `R32T32K256S3` 小 tile（32×32），算术强度远低于 TMA 原本的 64/128 宽 tile。例（n14336_k5120.cu）：
  ```cpp
  #else
      if (tokens <= 64)  return launch_bf16_mma<Bf16ScheduleInstance<Bf16A16MmaR64T32K64S3, 5120>>;
      if (tokens <= 96)  return launch_bf16_mma<Bf16ScheduleInstance<Bf16A16MmaR64T32K64S3, 5120>>;
      return launch_bf16_mma<Bf16ScheduleInstance<Bf16A16MmaR32T32K256S3, 5120>>;
  #endif
  ```
  对照 TMA 段：`R64T64K64S3`(≤64) / `R64T128K64S2`(≤128,∞)——Ada 段 64..96 用了减半 tile（R64T**32**），96..∞ 用 32 宽小 tile。
- **改法**：按 `sm89-port-sync.md:67-73` 的自我示范 + fallback-table 映射规则，补 64/128 宽 Ada 实例：
  ```cpp
  namespace {
  using AdaT64 = Bf16ScheduleInstance<
      Bf16A16MmaSchedule<64, 64, 64, 32, 32, 3, 1, Cache::cg, Cache::cg,
                         Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast,
                         Bf16MmaSwizzle::Xor64, 1>,
      5120>;   // 对应 TmaR64T64K64S3（64..192 区间 like-for-like）
  using AdaT128 = Bf16ScheduleInstance<
      Bf16A16MmaSchedule<64, 128, 64, 32, 32, 2, 1, Cache::cg, Cache::cg,
                         Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast,
                         Bf16MmaSwizzle::Xor64, 1>,
      5120>;   // 对应 TmaR64T128K64S2（>192 区间）
  }
  Bf16Launch select_bf16_n14336_k5120(std::int32_t tokens) {
      // ... <=32 以下不变 ...
  #if defined(NINFER_ENABLE_TMA)
      // ... 不变 ...
  #else
      if (tokens <= 64)  return launch_bf16_mma<Bf16ScheduleInstance<Bf16A16MmaR64T32K64S3, 5120>>;
      if (tokens <= 192) return launch_bf16_mma<AdaT64>;
      return launch_bf16_mma<AdaT128>;
  #endif
  }
  ```
  `n256_k5120.cu`（>1024 段）与 `n5120_k6144.cu`（>64 段）同构处理。
- **影响面说明**：qwen3.8-27B 主体 FFN 走 Q4 权重（`model-cards/Qwen3.8-27B-NInfer/README.md:122-123`），此三形状影响 bf16 权重变体与 MTP/辅助投影段，非 decode 主热路径——优先级低于 K3/K1，但改动小。
- **验证**：同 K1；`bench/ops/linear_bench.cu` 用 N=14336/N=256/N=5120、K=5120/6144 的大 token 段对比。

### K3【❌】mtp_pack 两个内核向量化（MTP 每步实跑，主路径收益最高）

- **位置**：`src/ops/kernel/mtp_pack.cuh:13-24`（`mtp_pack_fc_input_kernel`）、`:26-54`（`mtp_split_attn_in_kernel`）；launcher 在 `src/ops/launcher/mtp_pack.cu:13-41`
- **现状**：标量 bf16 逐元素访存、无向量化、无 `__launch_bounds__`；split 内核每元素做 `idx % kMtpAttnRows`（14336 非 2 幂，编译为真实除法指令）。MTP3 spec decode 每轮都跑（pack→attn→split），是带宽敏感段。
- **优化方案**：`kMtpQRows=6144`、`kMtpKvRows=1024` 均 8 元素对齐 → `uint4`（8×bf16）向量化；除法用 `blockDim` 分摊 token 维消除。
- **伪代码（pack 内核完整参考）**：
  ```cpp
  __global__ __launch_bounds__(256) void mtp_pack_fc_input_kernel(
      const __nv_bfloat16* __restrict__ embedding_norm,
      const __nv_bfloat16* __restrict__ hidden_norm,
      __nv_bfloat16* __restrict__ out, std::int32_t rows) {
      static_assert(kMtpQRows % 8 == 0, "rows must be 8-element aligned");
      const int vec      = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
      const int n_vec    = rows >> 3;                    // 6144/8 = 768
      if (vec >= n_vec) { return; }
      const std::int64_t in_v  = static_cast<std::int64_t>(blockIdx.y) * n_vec + vec;
      const std::int64_t out_v = static_cast<std::int64_t>(blockIdx.y) * (2 * n_vec) + vec;
      reinterpret_cast<uint4*>(out)[out_v] =
          reinterpret_cast<const uint4*>(embedding_norm)[in_v];
      reinterpret_cast<uint4*>(out)[out_v + n_vec] =
          reinterpret_cast<const uint4*>(hidden_norm)[in_v];
  }
  // launcher：grid = dim3(ceil_div(rows/8, 256), tokens)，block = 256
  ```
  **split 内核参考**（q/k/gate/v 四路拆分，同样 uint4 化；行号 = vec*8 起连续 8 个，天然消除 `% kMtpAttnRows`——若需 per-token 列，把 blockIdx.y 设 token、x 设行向量并保留行内除法为 2 幂 `kMtpKvRows` 的移位）：
  ```cpp
  __global__ __launch_bounds__(256) void mtp_split_attn_in_kernel(
      const __nv_bfloat16* __restrict__ attn_in, __nv_bfloat16* __restrict__ q,
      __nv_bfloat16* __restrict__ k, __nv_bfloat16* __restrict__ gate,
      __nv_bfloat16* __restrict__ v, std::int32_t tokens) {
      const int token = static_cast<int>(blockIdx.y);
      const int vec   = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
      const int n_vec = kMtpQRows >> 3;                  // 14336 拆段宽度均 8 对齐
      if (vec >= n_vec) { return; }
      const int64_t src = (int64_t)token * (4 * n_vec) + vec;   // 四段连续布局时
      reinterpret_cast<uint4*>(q)[(int64_t)token * n_vec + vec]    = reinterpret_cast<const uint4*>(attn_in)[src];
      reinterpret_cast<uint4*>(k)[(int64_t)token * n_vec + vec]    = reinterpret_cast<const uint4*>(attn_in)[src + n_vec];
      reinterpret_cast<uint4*>(gate)[(int64_t)token * n_vec + vec] = reinterpret_cast<const uint4*>(attn_in)[src + 2 * n_vec];
      reinterpret_cast<uint4*>(v)[(int64_t)token * n_vec + vec]    = reinterpret_cast<const uint4*>(attn_in)[src + 3 * n_vec];
  }
  // 注：落地前先读 mtp_pack.cuh:26-54 确认 attn_in 的实际段布局与常量名（kMtpAttnRows=14336
  // 是否拆为 q/k/gate/v 各 14336、或共用一个 K 界），按真实布局对齐伪代码；
  // 关键收益点：uint4 访存 + 消除非 2 幂取模，而非段序本身。
  ```
- **验证**：`ctest -R mtp`；`bench/ops/mtp_pack_bench.cu` 前后对比（预期带宽段 3~4×）；端到端 `ninfer_bench` 的 `pp2048+tg128` MTP case。

### K4【⚠️】24 个零-launch_bounds 内核文件补 `__launch_bounds__` 声明

- **背景**：281 个 `__global__` 内核中 166 个带声明（59%）；缺声明者集中在以下主路径相关文件。补声明是机械单行改动，价值在固定寄存器预算、防止 occupancy 随编译器版本漂移（配合已有 `-Xptxas=-v` 报表核对）。
- **清单与目标值**（block 尺寸均为源码常量，直接采用）：

  | 文件 | 内核数 | 建议声明 |
  |---|---:|---|
  | `src/ops/sparse_moe/decode/sparse_moe_decode_kernels.cu` | 6 | `__launch_bounds__(288)`（9 warp）/`__launch_bounds__(256)` 按各内核实际 block |
  | `src/ops/kernel/rope.cuh`（rope_fixed_kernel / rope_fixed_split_kernel） | 2 | `__launch_bounds__(1024)`（block 动态 64~1024，用最大值上界声明） |
  | `src/ops/kernel/scatter.cuh`（scatter_bf16x8/x2/scalar） | 3 | `__launch_bounds__(128)` |
  | `src/ops/kernel/mtp_pack.cuh` | 2 | `__launch_bounds__(256)`（随 K3 一起做） |
  | `src/ops/launcher/sampling.cu`（increment_token_counts_kernel） | 1 | `__launch_bounds__(256)` |
  | `src/ops/softmax_attention/dense/causal_cache/int8/tiled_mma.cuh` | 1 | 对齐兄弟内核 `__launch_bounds__(kThreads, kMinBlocks)`（同 `int8/grouped_mma.cuh:34`） |
  | `src/ops/softmax_attention/dense/causal_cache/k8v4/tiled_mma` | 1 | 同上 |
  | `src/ops/softmax_attention/common/mxfp8_tiled_mma.cuh` | 1 | 同上（fp8/k8v4 tiled 共用） |
  | `src/ops/linear/q5/q5_rowsplit_gemv.cuh` | 1 | 对齐同族 SIMT 系列 |
  | `src/ops/w4a8/w4a8_gemm.cu`（dequant_q4_i8_folded / _transpose 辅助内核 :483,:504） | 2 | 按各内核 block 常量 |
  | 其余 launcher 侧轻量内核（cast.cuh 3、position.cuh 2、scalar.cuh 4、gdn_gating.cuh 1、mtp_round.cuh 1、prepare_*.cuh 2、logits_topk_softmax.cu 1、causal_conv1d.cuh 11） | ~25 | 按 block 常量批量补；小内核影响低，顺手处理 |

- **伪代码（通用模式）**：
  ```cpp
  // 改前
  __global__ void scatter_bf16x8_kernel(...) {
  // 改后（block 常量已在文件内 static constexpr）
  __global__ __launch_bounds__(128) void scatter_bf16x8_kernel(...) {
  // 或与调度参数联动（softmax_attention 族）：
  __global__ __launch_bounds__(Schedule::kThreads, Schedule::kMinBlocks)
  void causal_int8_tiled_kernel(...) {
  ```
- **验证**：全量编译 + `cmake --build` 日志中 `-Xptxas=-v` 输出对比（确认无新 spill）；`ctest` 全绿。若某内核声明后出现寄存器强制溢出（spill bytes > 0 且性能回退），按报表回退该项并记录。

### K5【⚠️】causal_conv1d 标量 prefill 内核收敛到 pairs 版

- **位置**：`src/ops/kernel/causal_conv1d.cuh:74-99`（`causal_conv1d_prefill_kernel`，标量 bf16 四点累加 :96-99）；同文件 `:104-135` 已有 `prefill_pairs_kernel`（bfloat162 配对版）
- **改法**：确认 prefill 调用点（`src/ops/launcher/causal_conv1d.cu` 内 11 内核的路由条件）后，将标量版路由切到 pairs 版；若 pairs 版谓词不覆盖标量版的形状（奇数宽度等），为 pairs 版补谓词而不是保留双实现。
  ```cpp
  // launcher 路由伪代码：
  // if (宽度满足 pairs 谓词) launch prefill_pairs_kernel;  // bf16x2 累加
  // else                     launch scalar 保留（残余形状）
  // 并以 static_assert/注释标明 pairs 版为规范路径，标量版仅残尾。
  ```
- **验证**：`ctest -R conv1d`；`bench/ops/causal_conv1d_silu_bench.cu` 对比；GDN（qwen3.6 系）启用模型的 perplexity 回归。

### K6【⚠️ sm_120a 侧】NVFP4 MoE 断线宏 `NINFER_CUDA_HAS_BLOCK_SCALE_MMA`

- **位置**：`src/ops/moe/kernels/cuda/block_scaled/nvfp4/launch.cu:9`——`#if NINFER_CUDA_HAS_BLOCK_SCALE_MMA`；全仓 grep 确认该宏**零定义**（CMakeLists/cmake/ 均无），意味着 sm_120a 构建的 NVFP4 MoE 也永远走 throw 桩。
- **修复方案（二选一，伪代码）**：
  ```cpp
  // 方案 1：若意图是 sm_120a 启用（块缩放 MMA kind::mxf4nvf4 需 >=100），
  // 在 CMakeLists.txt:27-29 的 120a 分支补定义：
  if(CMAKE_CUDA_ARCHITECTURES STREQUAL "120a")
      add_compile_definitions(NINFER_ARCH_SM120=1 NINFER_ENABLE_TMA=1 NINFER_ENABLE_NVFP4=1
                              NINFER_ENABLE_K8V4=1 NINFER_CUDA_HAS_BLOCK_SCALE_MMA=1)  // ← 新增
  // 方案 2：若该路径尚未准备好，改用既有宏使门控语义显式：
  #if defined(NINFER_ENABLE_NVFP4)   // 替换 NINFER_CUDA_HAS_BLOCK_SCALE_MMA
  ```
- **注意**：对 sm_89 构建零影响（该分支两个方向都不编译块缩放路径）；先与上游确认意图再选方案，防止在 5090 上启用未验证内核。
- **验证**：sm_89：编译 + `check_blackwell_guards.py`；sm_120a：需 5090 实机跑 `diag-moe`。

### K7【⚠️】rope CTA residency 常数 4090 复测

- **位置**：`src/ops/launcher/rope.cu:16-17`——`kLargeBlockResidentCtasPerSm = 6`，注释自证"sm_120a 测量"（4090 与 5090 每 SM 线程上限同为 1536，6×256=1536 理论适用，但属移植未复测参数）。
- **方法**（需 4090 实机）：
  ```powershell
  # 1. 以 -Xptxas=-v 确认 rope 内核寄存器占用 → 计算理论 CTA/SM 上限
  # 2. bench/ops/rope_bench.cu 分别用 4/5/6 常数跑典型头数（qwen3.8: 64 头 128 维）
  # 3. 选实测最优者改常数并更新注释为 "measured on sm_89"
  ```

### K8【⚠️ 结构性】q4/q8 shape 档位边界 4090 sweep + `NINFER_ARCH_SM89` 参数覆盖机制

- **位置**：`src/ops/linear/q4/shapes/*.cu`（10 个注册项）、`q8/shapes/n248320_k5120.cu` 等——注释自证区间边界为 "cold CUDA Graph measurements on RTX 5090"（如 q8 lm_head `tokens<=33`/`<=48` 档位衔接处 C40 只覆盖 33 一个值，疑为 5090 冷测量怪癖）。另有全局事实：`NINFER_ARCH_SM89` 在 CMakeLists.txt:32 定义但 src/ 零引用——sm_89 无任何专属形状参数修正机制。
- **优化方案**：
  1. 在 4090 上跑形状 sweep（工具链齐备：`bench/ops/linear_bench.cu` + `docs/maintainer/linear-tuning.md` 方法论），产出各形状 token 边界的 4090 实测值；
  2. 为边界引入 sm_89 覆盖机制，伪代码：
     ```cpp
     // select 函数内的边界常量改为编译期分档：
     constexpr int kSlicedKEdge = [] {
     #if defined(NINFER_ARCH_SM89)
         return 96;   // 4090 实测值
     #else
         return 128;   // 5090 实测值（保持不变）
     #endif
     }();
     ```
     或建运行时 shape policy 表（启动时按 `execution.multiprocessor_count`==128 判 4090）——前者更贴合现有"编译期选择"风格，推荐。
- **验证**：sweep 前后 `linear_bench` 各形状报告归档对比；`docs/performance/rtx-4090-sm89-baseline.md` 基准复核。

### K9【⚠️ 低优先】act_quant 向量化

- **位置**：`src/ops/act_quant/act_quant_a8.cu:13-71`——标量 bf16 两遍 amax+roundtrip。仅在 `NINFER_ACT_A8` 环境变量启用时运行的精度探针，非生产路径。
- **改法**：若保留该探针，改 bf16x2/uint4 归一化遍历（模式同 K3）；若计划下线，直接在 CMake 加开关默认排除。伪代码从略（与 K3 同构）。

### K10【可选清理】pipeline.cuh `pipe_*` 死代码

- **位置**：`src/ops/common/cuda/pipeline.cuh:52-64`——`pipe_copy/pipe_commit/pipe_wait` 包装 `cuda_pipeline.h` 老接口，全仓无外部调用者（主路径全部直用内联 PTX `cp.async`）。下次清理时删除即可，无行为影响。

---

## 五、已排除项（审核过但不成立的怀疑点，防止重复排查）

- bf16 21 个 shape 文件 token 区间无 gap（`<=N` 链 + 无条件兜底 return）；
- sm_89 分发无未定义引用（TMA 声明与调用点全部在门控内；`#else` 回退内核全部无条件编译）；
- 各 `#else` 回退调度的 K/行整除性逐一验算通过（含 `Bf16RowTailSchedule`/`Bf16KTailSchedule` 谓词路径）；
- rk 5 布局 codec 模板（PackedV/PackedK/E8Root/V8Root/E8Lattice）在 grouped/parallel/tiled/wide 四路径映射一致；
- launcher 侧 rmsnorm/rope/sampling/scatter/mtp_pack 无 int32 溢出（偏移全 int64，超界显式 throw）；
- `launch_bf16_mma_partitions` grid.x 溢出有 int64 检查；240k token 靠 token 切片兜底 grid.y；
- fp8/k8v4 的 B>1∧W>16 由 mxfp8 tiled 内核承接（含 batch 维 partial 平面），workspace 查询与执行对称；
- sliding-window/context attention 仅支持 bf16 为独立 cyclic/context 缓存的设计约束（显式校验）；
- RK2V4E8 codec 在 standalone append、fused append、read-back 三处对齐；
- `n1152_*` 系 `tokens%4` throw 为刻意 P 域限制（视觉 patch 最小 4）；
- CodeGraph impact 链核查（`select_bf16_launch` 18 符号、`make_int8_kv_causal_plan` 17 符号、`kv_cache_append_batch_launch` 5 调用者、`single_row_paged_kv_batch_view` 13 符号）无断链；
- `Nvfp4Group16`/NVFP4 线性算子/PDL/TMA 在 sm_89 的 throw/退化路径均为刻意设计（README:43 明示 4090 加载 groupwise-int artifact）。

## 六、审核方法备注

- 项目自检器 `tools/port/check_blackwell_guards.py`：1127 文件扫描通过（零未门控 TMA 调用点）；
- CodeGraph 索引：`codegraph init -y` 于项目根目录（1,770 文件 / 38,299 节点 / 120,671 边，9.9s），后续代码变更后 `codegraph sync` 增量更新；
- B1/B2/B3 三项 BUG 的代码证据均经主审核人逐行复核（`tiled_mma.cuh` ×2、`batch_wide_mma.cuh`、`mxfp8_tiled_mma.cuh`、`paged_kv_address.cuh`、`causal_softmax_attention.cpp`、`nvfp4/plan.cpp`）；K1 16 文件 `#else` 现状由主审核人逐一提取核对；`NINFER_CUDA_HAS_BLOCK_SCALE_MMA` 零定义与 `mtp_pack` 标量实现均经 grep/源码直读确认。
