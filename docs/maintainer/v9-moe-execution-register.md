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
| T123 | a9c68b44 + ff17188e | 不搬入（登记） | 本文件 §T123 |
| T124 | 67ded232 + 0857a495 | 不搬入（机制已在本地形态） | 本文件 §T124 |
| T125 | 组门禁 | 通过（见数据） | 本文件 §T125 |

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

## T123 token 序保持 ＋ 稀疏 FP8 route 压缩 —— 不搬入

**裁决：零代码改动。** 两半各自不成立，原因不同。

### a) a9c68b44 `token_ordered_groups`：本地几何下判据恒假

- 上游新增 `fully_covered_groups()`（`block_scaled/h4096_i2048_e288_k8/launch.cu`）只在
  「每个被选专家都覆盖全部 T 个 token、每专家恰好一个 job」时，把 packed 索引退化成
  `offsets[e] + a/kTopK`（density 分支 `jobs[2]==1 && offsets[kExperts]==assignments/kTopK`；
  非 density 分支要求 `load_range == [tokens, tokens]`）。
- 本地几何 kTopK=10、kExperts=512（`block_scaled/launch.h`:7）：总 assignment 数是 10T，
  判据要求「每个被选专家都有 T 行」⇒ 每个 token 必须选同一组 10 个专家。常规 top-10 路由下
  每专家平均 10T/512 行；本地 fixture 的路由是 `route = j*(kExperts/kTopK) + 3*t`
  （`tests/ops/test_moe_block_scaled_fp8.cpp`:185），逐 token 不同，判据恒假。
  搬进去只是一个永不进入的分支。
- 上游只把该开关写进 `routed_swiglu/cuda/{fp8,nvfp4}/dgx_spark/shapes/h4096_i2048_e288_k8/dispatch.cpp`；
  本地的形状目录只有 h2560_i640_e512_k10 与 h2048_i512_e256_k8。

### b) ff17188e 稀疏 FP8 route 压缩：目标内核本地不存在

- 主体是 `block_scaled/h4096_i2048_e288_k8/fp8/vector_pair.cuh` 的 167 行重写（成对 route 的
  工作压缩）＋该形状的 `launch.cu/h` 与 dispatch。
- 本地 FP8 的向量路径是另一套内核：`block_scaled/fp8/launch.cu`:101–122 的
  `vector_gate` / `vector_down`（`vector_activation[kIntermediate/4, T*kTopK]`），
  没有「成对 route 工作区间」这个数据结构；`vector_pair.cuh` 本地不存在。
  按 L02 的准入口径（融合必须与不融合时的选核共用同一套 ncols/nchannels/n_experts 口径），
  在两个不同内核之间搬这段没有意义。

### c) 顺带记录：本地「确定性」的真正位置

- 本地单块分组 `block_scaled/launch.cu`:79–125 的 rank 由 `atomicAdd(counts+e,1)` 决定，
  同一专家内的 packed 顺序跨线程不确定。要做到「token 序保持」须把 rank 改成按 assignment
  序的前缀和 —— 那是**数值/确定性改动**，必须按 T028 同时报 KL（分长度）＋PPL 逐位＋任务套件
  ＋交错 A/B。本条没有实测窗口，登记为后续候选，不在本轮伪装成移植。

---

## T124 dense FP8 down contributions 融合 ＋ Q8 shared GU tiles 缩减 —— 不搬入

**裁决：零代码改动。** 前半的机制本地已有（不是缺），后半的目标内核本地不存在。

### a) 67ded232 dense down contributions：本地已经是「down 直接落贡献」

- 上游为 h4096_i2048_e288_k8/fp8 新增 `dense_down.cuh`，把 down 的输出与 contributions 的
  累加合成一步。
- 本地 FP8 专家的 down 本来就是这一步：`block_scaled/fp8/launch.cu`:30–49 的
  `grouped_mma<false, Columns, Atomic>` 直接把结果写进 `s.contributions`，或在
  `plan.atomic_output` 时（`:93–97`）直接原子累加进 `s.routed_sum`；随后
  `block_scaled/launch.cu`:186–192 的 `reduce` 只做最后合并与写回。
- gate 侧的 act+quant 也已融合：gate 内核的 epilogue 直接产出 `product_codes/product_scales`
  （`launch.cu`:43–47），没有「先写 FP32 再量化」的第二趟。T044 记的「act+quant 融合会移动数值、
  必须过 PPL 逐位」在这里不构成待办：本地没有可融合的第二趟。

### b) 0857a495 Q8 shared GU tiles：改的是本地没有的 TMA 阶梯

- 该提交只改 `src/ops/moe/swiglu/dgx_spark/q8.cu`：把 (65,128] 的
  `Q8A16TmaMmaSchedule<32,128,…>` 换成新增的 `T64Rows32 = Q8A16TmaMmaSchedule<32,64,32,16,…>`。
- 本地：无 `src/ops/moe/swiglu`；sm_89 无 TMA（`sm89-fallback-table.md`:1–30 把 TMA 阶梯整体
  映射到 cp.async 回退，守卫脚本的 `tma` 规则对未门控调用直接判违规）。
- 本地 Q8 共享 GU 走 `groupwise_int` 的 `SharedPlan.columns`（`groupwise_int/launch.h`:32–36），
  按 token 区间选 MMA 列 tile，是另一套调度。若要在本地形态下重挑 (65,128] 的 tile，那是
  **本地调优条目**：按 T028 同时报 KL（分长度）＋PPL 逐位＋任务套件＋交错 A/B，
  而不是把上游的 TMA 实例名抄过来。

### c) T044 裁决沿用

- TMA 子项**不采纳**（sm_89 无此指令，上游门槛写死 `is_device_capability(90)`）；
  act+quant 融合若要真动，必须按功能改动走 PPL 逐位＋dump 门禁 —— 本地当前无需动。

---

## T125 组门禁（T120–T124）—— 通过

执行环境：RTX 4090 / sm_89 / Windows 11 / CUDA 13.4；本机全量测试与门禁同会话连跑。
原始日志与 CSV：`.temp/t125/`（`ctest_full.log`、`regress_fast.log`、`regress_moe.log`、
`guards.log`、`bench_sparse_moe.{log,csv}`、`softmax_attn.log`、各 MoE 测试日志）。

### 1. 构建

- 全量目标构建零错误；本轮最后改动后删掉 `src/ops/moe` 下全部 **28 个 .obj** 重编，
  再到 `ninja: no work to do`（即改动面所在 TU 是干净重编出来的，不是旧 obj 复用）。

### 2. 守卫复检

- `python tools/port/check_blackwell_guards.py` → `ok: 1255 file(s) scanned, no ungated
  Blackwell-only call sites`（exit 0）。

### 3. 全量测试（ctest，176 条，1150 s）

- 失败 6 条，与 T119 基线逐条对账后**无新增**：
  - 既有失败、本轮同集合（5 条）：`ninfer_artifact_reader_test`、`ninfer_bench_support_test`、
    `ninfer_qwen3_5_loading_test`、`ninfer_qwen3_5_frontend_test`(0xc0000409)、
    `ninfer_kv_cache_append_test`(0xc0000409)。
  - `ninfer_fused_rmsnorm_test`：T119 失败、本轮通过（flaky，方向变好）。
  - `ninfer_softmax_attention_test`：全量跑里 **180 s 超时**（本轮整机负载：总时长 1150 s vs
    T119 的 629 s）。单独复跑 `--kv-dtype rk4v4-e8` **114.8 s 通过，exit 0**
    （日志末行 `PASS causal_softmax_attention rk4v4-e8 public-contract correctness (114.791 s)`）。
    判为超时门槛 + 机器负载，不是本批回归；本批未触碰 attention。
- MoE 相关六个程序复跑：`moe_block_scaled_fp8`、`moe_cpu_nvfp4`、`moe_int8_mma`、`sparse_moe`、
  `moe_weight_views` 全 exit=0；`qwen3_5_moe_real_test` exit=77（无工件，SKIP，与本批无关）。

### 4. regress `--tier fast`（双 PPL 的稠密侧 + 字节门 + 精度面）

```
default PPL 7.441868  expect 7.441868   PASS      ← 逐位
selector-off dump 636e38de95046781  expect 636e38de95046781   PASS   ← 逐位
accuracy: 9 hit / 2 miss，全部 PASS
regressions: 0
```

### 5. regress `--tier moe`（双 PPL 的 MoE 侧，含 decode 哈希）

```
ctx 1024 stride 512   PPL 7.111129  expect 7.111129   PASS
ctx 512  stride 256   PPL 7.777808  expect 7.777808   PASS
ctx 4096 stride 2048  PPL 6.479183  expect 6.479183   PASS
decode d093ac2070ab3292  expect d093ac2070ab3292   PASS   '2, 3, 5, 7, 11<|im_end|>'
regressions: 0
```

结论：T122 的 scratch 生命周期改动对生产 MoE 路径**数值中性**（三点逐位不变、decode 哈希不变），
这是本组唯一的算子侧改动，故 PPL 逐位即为该改动的验收。

### 6. MoE bench

`ninfer_sparse_moe_bench --codec all --sweep 1:2048:256 --repeat 200 --warmup 5 --flush-mib 96
--cache cold --execution graph`（RTX 4090，理论 1008.1 GB/s，冷缓存，graph replay）：

| codec | T=1 min | T=257 min | T=1025 min | T=1 唯一权重带宽 |
|---|---|---|---|---|
| q4-q5 | 35.840 us | 920.576 us | 1301.504 us | 510.3 GB/s (50.6%) |
| q4-q6 | 34.816 us | 977.920 us | 1343.488 us | 554.2 GB/s (55.0%) |
| q8-q8 | 50.176 us | 1494.016 us | 1995.776 us | 608.1 GB/s (60.3%) |

注：该 bench 覆盖的是公开 `SparseMoe` 合约（h2048/e256/k8）；本轮改动的 `routed_swiglu`
（h2560/e512/k10）在本地**没有独立 bench**，因此本条**不声称任何提速**，也不引用 bench 数字
作为它的性能证据 —— 与「无提速声明」一致。

### 7. T028 三件套适用性判定

- T121 只动测试判据（不动算子），T122 只动 scratch 生命周期（不动算术，逐位门为证）：
  两条都**不是提速型改动**，因此不触发「KL 分长度 + 套件 + 性能 A/B」；组门禁按
  「regress fast + 双 PPL + MoE bench」口径执行完毕。
- 遗留候选（转后续条目，均需 T028 全套）：T123c 的确定性 rank 改造（atomicAdd → 前缀和）、
  T124b 的本地 Q8 共享 GU tile 重挑（需先给该 op 建 bench）。

---
