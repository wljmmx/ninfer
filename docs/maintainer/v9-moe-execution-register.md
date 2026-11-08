# v9 Phase 5 · MoE complete execution 组（T120–T125）登记

- 范围：v9 §7.6 的 T120–T125（上游 a754fcf3 线，本地基线 de5d4288，共同基线 81c8ce09）
- 本文件只做**结论 + 证据**：每条给出「搬入 / 不搬入 / 部分搬入」，以及本地对应的文件与行号
- 口径：本地目标 rtx4090（sm_89 / Ada），无 dgx_spark 目标；上游目标 rtx5090（sm_120a）与
  dgx_spark（sm_121a）。跨目标内容一律按 `tools/port/check_blackwell_guards.py` 与
  `docs/maintainer/sm89-fallback-table.md` 的规则处理。

| 条目 | 上游提交 | 裁决 | 本地落点 |
|---|---|---|---|
| T120 | 88105a00 + a4e3f4be + 95a651aa | 不搬入（登记） | 本文件 §T120 |
| T121 | 757194af | 搬入（判据＋MoE 用例） | `tests/ops/op_check.h`、`tests/ops/op_tester.h`、`tests/ops/test_moe_block_scaled_fp8.cpp` |
| T122 | a08d5133 + 7ea6cb6e | 部分搬入：scratch 限定搬入；共享输出消费已成本地形态 | `src/ops/moe/operands.*`、`src/ops/moe/workspace.h`、4 个 entry |

---

## T120 complete sigmoid / clamped-swiglu execution ＋ 专家子集 —— 不搬入

**裁决：零代码改动。** 四条独立理由，任一条都足以拦住；四条同时成立。

### 1. 执行体整体在 `NINFER_TARGET_SPARK` 门内，本地没有这个目标

- 上游 `src/ops/moe/topk_sigmoid_swiglu_moe/entry.cpp`（a754fcf3）:32、:52、:137 三处
  `#if NINFER_TARGET_SPARK`；非 Spark 构建里这个算子只有参数校验，没有任何执行体。
- 上游 `src/ops/moe/swiglu/entry.cpp`:22–43 只有 SPARK 分支，`#else` 直接
  `throw std::invalid_argument("swiglu: no registered profile for this target")`。
- 本地 `cmake/NinferPlatform.cmake`:1–26 只定义 rtx4090(sm_89) 与 rtx5090(sm_120a)，
  注释明写 “There is no DGX Spark target here”；全仓无 `NINFER_TARGET_SPARK` 宏。
- T001 已把 Spark/DGX 96 条封口为「不搬入本地树」（`.temp/upstream-scan/verdict-deadlist-sealed.txt`
  :49–133）。搬入这批文件在本机构建里等于新增 4 千行永不编译的死代码。

### 2. 几何不同：上游只注册 h4096/i1024(2048)/e288/k8，本地只编 h2560/i640/e512/k10

- 上游 `topk_sigmoid_swiglu_moe/entry.cpp`:85–87 把 `hidden==4096`、`shared.intermediate==1024`、
  `experts==288`、`top_k==8` 写成硬校验，随后只分派
  `moe_sigmoid::{nvfp4,fp8}_h4096_i1024_e288_k8()`。
- 本地 block-scaled 专家的形状是编译期常量：`src/ops/moe/kernels/cuda/block_scaled/launch.h`:7
  `kHidden = 2560, kIntermediate = 640, kExperts = 512, kTopK = 10`。
- 两个几何不是参数化关系：本地 `Groups`/`Storage`/`Plan` 都按 kExperts 定长（`:25–46`），
  换成 288 专家是重写而不是移植；而按 L03 的结论，按专家数做编译期特化在本地 10240 专家/40 bank
  的形态下本来就不可行。

### 3. 依赖一个本地不存在的独立 clamped swiglu 算子

- a4e3f4be 新建 `src/ops/moe/swiglu/{entry.cpp,dispatch.h,kernel/…,dgx_spark/…}`（共享专家 FFN，
  带显式 clamp limit）与 `block_scaled/clamped/*`，并改动 `src/ops/linear/swiglu/product_epilogue.cuh`、
  `nvfp4_a4_quantize.cuh` 的 Spark 路径。
- 本地：`src/ops/moe/swiglu` 目录不存在；全仓 `clamp_limit` 命中 0；无 `ops::swiglu`。
- 本地的共享专家是**在路由内核里 joined** 的：`include/ninfer/ops/moe.h`:130–135 的
  `ScaledSwiGLUTerm` 契约（“adds coefficients[t]*F_shared(x[:,t]) exactly once”）、
  `src/ops/moe/routed_swiglu/cuda/entry.cpp` 把 `term` 直接交给 grouped/block-scaled 内核。
  引入上游那套「独立 swiglu 先算共享、再回头加路由贡献」会把一条已经单遍、单次 round 的路径
  拆成两遍，并把该路径的 PPL 契约重新洗一遍 —— 收益是零（本地没有 Spark 的 TMA/双缓冲前提）。

### 4. 专家子集（95a651aa）在本地没有消费方，其配套 EP 夹具已封口

- 本地 `topk_softmax_swiglu_moe` 在生产路径没有调用点（`src/` 仅 `core/nvtx.h` 里出现名字）；
  生产路径是 split layer 的 `ops::routed_swiglu` + CPU 半区
  （`src/models/qwen3_5/execution/ffn.cpp`:1255），而该路径**已经带** `ExpertSubset`
  （`moe.h`:81–86；GPU half 用 mask、CPU half 跑 mask 保留的路由，见 `moe.h`:184–185 的
  `missed_routes` 注释）。
- 95a651aa 另加 `tests/ops/moe/ep_fixture.*`、`test_ep.cpp`、global↔local 专家 id 映射，
  这是多 rank 专家并行的形态；T001 已把 TP2/分布式 18 条封口。
- 给一个没有调用方的算子加子集参数，是把「可编译」当「可用」，与本地纪律（功能必须真正接线）
  相冲突。

### 保留（只记录，不实现）

- T044 的 **padding 语义下沉**（router 输出 `-1/0`、所有消费者统一跳过）仍是本地批组织若要
  引入 padding 行时的正确形态；本地当前批是紧凑的，无 padding 行可跳。
- 上游 `warning_once(归因后的原因)`（af4db2a6）可直接搬到本地 feature-gate 日志，属日志条。

---

## T121 近零误差判据修正 —— 搬入

**裁决：搬入判据本体，并把它用在本地的 MoE 用例上。**

- `tests/ops/op_check.h`：`ReductionCriterion` 增加可选 `absolute_rms`（默认 0.0）。非零时
  范数判据改为 `RMSE <= absolute_rms + relative_l2 * reference_rms`（`reduction_rms_limit`），
  零时保持原来的纯相对形式 —— 所有既有 `{rel, gross_abs, gross_rel}` 初始化语义不变。
- `tests/ops/op_tester.h`：`OP_ERROR_STATS kind=reduction` 增加
  `rms_abs_limit / rms_limit / rms_ratio` 三个字段，判据变化在日志里可见（否则「为什么这条过了」
  无法用日志回答）。
- `tests/ops/test_moe_block_scaled_fp8.cpp`：`kA8` 从 `{0.08, 1e-7, 0.12}` 改成
  `{0.08, 1e-7, 0.12, 1e-7}`。该 fixture 的第一块 gate/up 行 tile 是**全零**（`Matrix` 构造里
  `nr == 0` 时 scale 取 0），SiLU(0)=0 的那些行对相对范数没有信号，正是上游这条修正的适用面。
- 上游同时改的 `clamped_ep_fixture.*` 本地不存在（T120 裁决），未搬。

**验证**：`ninfer_moe_block_scaled_fp8_test` 退出码 0（T=1/2/3/5/9 全过，含新增的近零尾行）。

---

## T122 共享输出消费 routed contributions ＋ scratch 限定 —— 后半搬入，前半已是本地形态

### 7ea6cb6e「aligned scratch 限定在算子调用内」—— 搬入

- 本地原实现（`src/ops/moe/operands.cpp` 的 `workspace(const MemoryArena&, size_t)`）把
  **调用方游标所在的偏移**当作 scratch 原点：容量查询假设 256 字节对齐原点，而执行侧要求
  `operand::aligned(span.data, 256)`，两条契约不一致；游标不在原点时直接抛
  `"MoE: workspace is not aligned"`。调用方还必须自己保证「用完把游标还回去」。
- 现在：`workspace(MemorySpan)` 只校验设备与落盘范围；每个 entry 先
  `aligned_scratch(workspace, capacity)` 从调用方 arena 里切一块**对齐子 arena**
  （`src/ops/moe/workspace.h`），所有切片分配改从该子 arena 取，函数返回时
  `workspace.scope()` 把调用方游标还原。零字节不分配（仍带设备名，供操作数校验读同一答案）。
- 落点：`routed_swiglu/cuda/entry.cpp`、`routed_swiglu/cpu/entry.cpp`、`topk_softmax_route/entry.cpp`、
  `topk_softmax_swiglu_moe/entry.cpp`。CPU 半区同样处理（它原来也要求
  `"workspace cursor must be 256-byte aligned"`），否则 split 层两侧契约不同。
- **回归用例**：`tests/ops/test_moe_block_scaled_fp8.cpp` 新增 `run(fixture, 3, 8)` ——
  先把调用方游标推到 8 字节（非 256 对齐）再调用，断言调用成功且游标被还原。
  用例强度做过 A/B：把 `operands.*` 与四个 entry 退回旧实现重建，该用例以
  `MoE: workspace is not aligned` 退出 1；换回新实现退出 0。

### a08d5133「共享输出消费 routed contributions」—— 不搬入（本地已是等价且更省的单遍）

- 上游把方向定为：路由先把贡献写成 FP32[H,T]＋可选有效性平面 `valid_tokens`
  （`routed_swiglu/cuda/contribution.h` 新增 `RoutedSwiGLUContribution`），再由**共享 FFN**
  在自己的 epilogue 里把路由贡献加进来（`swiglu_ffn::launch_*_ffn_join`），因此只 round 一次。
  有效性平面存在的原因是该提交配套的 clamped 路径会整块跳过路由（数学零但 values 不可读）。
- 本地是**反向的单遍**：共享 term 由路由内核在 epilogue 里消费
  （`moe.h`:130–135 的 `ScaledSwiGLUTerm`；`routed_swiglu/cuda/entry.cpp` 把 `term` 直接交给
  grouped/block-scaled 内核，`block_scaled::launch` 的 shared 分支在 `block_scaled/launch.cu`:183–185）。
  两条路径的算术同为「路由和 + 系数×共享项，一次 round」，没有 FP32 往返可省。
- 本地没有「数学零但 values 不可读」的行：split 层的 GPU 半区用 mask、CPU 半区只跑 mask 保留的
  路由（`moe.h`:184–185），所以有效性平面在本地无对位；为它新增一套平面只会让两侧契约分叉。
  与 `74671a4e` 建立的 CPU 执行半区口径一致：CPU 侧收到的就是「要被算的项」。
- 结论：该提交的**目标**（一次 round、无中间往返）本地已达成，代码不搬。

**验证**：MoE 相关五个测试全 exit=0（`moe_block_scaled_fp8`、`moe_cpu_nvfp4`、`moe_int8_mma`、
`sparse_moe`、`moe_weight_views`）。

---
