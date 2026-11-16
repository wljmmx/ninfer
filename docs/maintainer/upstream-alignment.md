# 上游对齐状态与下一版本目标

本文件回答三件事：**本地对齐到上游哪个版本**、**对到了什么程度**、**下一轮对齐的目标与完成定义**。
本地的"对齐"是**内容等价**，不是历史合并：上游那层 Linux/SM100 的执行设施在 Windows 自研平台层上重写，
提交历史独立演进（本地 `master` 与 `upstream/master` 不存在祖先关系，`git rev-list` 的 ahead/behind 只反映历史，
不反映内容缺口）。

## 1. 当前对齐基线

| 项 | 值 |
|---|---|
| 上游基线（**已逐条核对，非全部吸收**） | `77c2e5e2` — `fix(runtime): recover preempted requests without stalling`（2026-11-07）；逐条处置见 §7 台账 |
| 基线之前 | `de5d4288`（v8 完工）→ `77c2e5e2` 共 **1565** 提交；其中 `a754fcf3` 之前的 1197 条经 `cc3938c9`（2026-10-09）并入本地历史，其余按 §7 台账逐条标注 |
| 吸收路径 | 判死 132（Spark 96 / GLM 18 / TP2 18，双计已标注）；A 类整搬；M 类三方融合（T049–T083）；语义对表（T145–T168）；参考核对（T021–T047 / L01–L14）；模型集成登记（T191–T202） |
| 本地版本 | `git describe` → 见 `VERSION`（`v1.0.0-rtx4090-<N>-g<hash>`） |
| 对齐证据 | `docs/maintainer/v9-*.md` 各登记、`v9-closeout-register.md` 的 305 条对账、`.temp/upstream-scan/` 留档 |

> **更正（2026-11-09）**：本表原先记的「上游基线 `a754fcf3` **已吸收**、基线前 305 条按 v9 计划吸收完毕」**不成立**——
> 依据与本轮逐目录核对见 §6。自今日起本文件以 `77c2e5e2` 为增量基线，逐条处置记录见 §7。

## 2. 上游现状（本轮核对）

`git ls-remote upstream` / `git fetch upstream` 于 2026-11-08：

| 项 | 值 |
|---|---|
| `upstream/master` | `77c2e5e2` — `fix(runtime): recover preempted requests without stalling` |
| 相对基线新增 | **63** 提交（`a754fcf3..77c2e5e2`） |
| `upstream/dev` | `9eff5314` |
| `upstream/feat/qwen4-27b-ops` | `e1874e94`（QSA 算子优化分支，未入 master） |

### 63 提交的构成与 sm_89 适用性

| 类别 | 条数 | 内容 | 4090 判定 |
|---|---|---|---|
| `perf(linear)` | 43 | MXFP8 / NVFP4 / Block-FP8 / TMA / BF16 模板在各 token 范围与 tile 形状上的调度调优 | **不适用**：MXFP8/NVFP4/TMA 属 SM100 路径；本地 T004 守卫表已禁 `tma/tma_bulk/cluster/tp2`，本地 0 违规 |
| `perf(moe)` | 5 | A16 EP 专家流水、CSF 专家流、A16 乘积归一、M128 共享 FFN 延迟崖 | **不适用**（EP/A16/CSF 为 Blackwell 执行器）；仅 `38e1559d` 的"共享 FFN 在小 batch 的延迟"现象可作**观测参考** |
| `perf(swiglu)` | 3 | Block-FP8 shared/dense A8 FFN 流水与 FP32 贡献合并 | **不适用**（本地无 A8 FFN 路径） |
| `perf(kpool)` | 3 | 原生 sparse MLA 因子调度、indexer 窗口与图开销 | **不适用**（本地无 MLA/kpool 执行器） |
| `perf(kda)` / `perf(mhc)` / `perf(rmsnorm)` / `perf(embedding)` | 4 | KDA 卷积 epilogue、MHC 小列调度、Spark plain norm、Spark BF16 gather | **不适用**（Blackwell 指令集） |
| `feat(ops)` / `feat(weights)` | 2 | 原生 FP4-Local 推理核 + FP4-Local 权重导入 | **不适用**：Ada 无 FP4 张量核（T013/T108 已判死 NVFP4 路线） |
| `fix(moe)` | 1 | `454c5e7a` A16 乘积无微小输入损失归一 | **不适用**（A16 EP 专属）；本地 CPU 专家 int16 路径有自己的归一与门禁 |
| `style(ops)` | 1 | native projection coverage 格式化 | 无需对齐 |
| `fix(runtime)` | 1 | **`77c2e5e2` preemption 恢复 + 关闭顺序 + 观测字段** | **需吸收**，见 §3 |

**结论：62/63 在 sm_89 上不适用（架构或守卫原因），唯一需要吸收的是 `77c2e5e2`。**

## 3. 下一对齐目标（v10）

| 项 | 设定 |
|---|---|
| 目标上游版本 | **`77c2e5e2`**（2026-11-07） |
| 范围 | 只吸收 `77c2e5e2`，其余 62 条登记为 N/A（§2 表，逐类给出原因，不逐条搬） |
| 拆分 | **a** 零 frontier 请求直接恢复进保存的生命周期（恢复容量保留到第一个新提交单元）；**b** 关闭顺序：先停 Engine 再 join HTTP worker，信号交给控制线程；**c** 观测字段：`direct/snapshot/replay` 恢复计数、replay frontier、service 归属与 last progress；**d** 回归夹具：异步 binding/preemption、blocked-replay 关闭 |
| 与本地既有工作的关系 | T060（`970a5f58` 轮转/防跳过/压力期推迟 fresh admission）、T061（`ea100705` prepare→release→publish）已覆盖压力期准入；本轮是**恢复路径**与**关闭顺序**，不重叠 |
| 完成定义（DoD） | 1) 四条子项全部落地且有测试；2) 35B default 930 MiB / 冻结 12 GiB PPL 逐位不变；3) 27B regress fast/moe 0 回归、dump `636e38de95046781` 不变；4) 中断/取消在 c8 常驻压力下 30 分钟无 stall（按 `77c2e5e2` 的验证口径）；5) 观测字段在 `/monitor/snapshot` 与服务日志可读；6) 登记回填 v10 计划 |
| 启动条件 | 无外部依赖，可立即开工；若 `upstream/master` 继续前移，目标按"最近一次已核对且含需吸收项"的提交冻结，不追 HEAD |

## 4. 分支与模型线

| 分支 | 状态 |
|---|---|
| `upstream/feat/qwen4-27b-ops` | 未并入 master；787 文件/+33445 全为 QSA 算子优化（四头 k8v4/nvfp4/int8/bf16、sparse attention 跨 KV 格式、20480 整向量 rmsnorm）。**qwen4-27b 不是独立模型格式**，是 Qwen4Exp 架构的 27B 配置，本地转换器已覆盖。并入时按 v10.1 立项，走 T028 三件套（KL+PPL 逐位 / 任务套件 / 同会话交错 A/B） |
| `upstream/dev` | 不跟踪（非发布线） |

## 5. 本文件以外的依据

- 逐条对账：`v9-closeout-register.md`（305 条覆盖路径）
- 参考核对原文：`H:\TeleAgentWorks\.temp\upstream-scan\ref-review\{S,W,L}-REVIEW.md`
- 守卫与判死：`.temp/upstream-scan/{guards-violations.txt,verdict-deadlist-sealed.txt}`
- 每次对齐的门禁：`v9-dashboard-register.md` 的 T131 门禁节 + 各 phase-gate 留档

## 6. 2026-11-09 核对：上游 Qwen4Exp 的本地化缺口（更正 §1 的基线说法）

上一轮把"已吸收 a754fcf3（305 提交）"写进了本文，实测**不成立**：
`git merge-base --is-ancestor a754fcf3 master` 为假，本地最后一次吸收上游是
`cc3938c9`（2026-10-09）。相对 `upstream/master`（`77c2e5e2`）本地 **领先 810 / 落后 368**。
下面是这次逐目录对账的结果（差距只列 Qwen4Exp 线，其余按 §2 判定 N/A）。

| 上游内容 | 本地状态 | 影响 |
|---|---|---|
| `src/models/qwen4_exp/`（114 文件：config/definition/load/execution(gdn,qsa,ple,mtp)/program/session/state/parallel/wire） | **仅 12 个 PLE 文件**，模型层整层缺失 | Qwen4Exp 的权威实现不在树内；本地走的是在 `qwen3_5` 树上外挂（M1/M2），不是上游形态 |
| `src/ops/softmax_attention/qsa/{indexer,sparse_gqa,prepare}` | **缺失 43 文件中的 39**（本地只有 gated_residual 之外的旧 QSA 片段） | QSA 稀疏注意力无可用内核，Qwen4Exp 的 12 个 QSA 层无法执行 |
| `src/ops/gated_residual/{mix,norm_combine}` | 缺 17 文件（`mix/{implementation,select,up_gemv_mix}`、`norm_combine/launch.cuh`、`rtx5090/`、`dgx_spark/`） | HC 四流混合与 norm-combine 只有部分实现 |
| `src/ops/ple/*` | 8/8 齐（本地已移植） | PLE 卷积/状态提取可用 |
| 分页 NF4、expert cache/offload、paged KV 混合行跨度、校准优先级 | 缺失（在 368 提交内） | 上游的 PLE 表 NF4 磁盘路径与 5090 专家缓存不可用 |
| `upstream/dev`（`41da21f4`） | 不跟踪 | 分布式 Spark 集合通信、native FP4 流水（非发布线） |

### 6.1 上游的量化选择与 sm_89 的冲突

`model-cards/Qwen3.8-Flash-Next-NInfer/artifact-manifest.json` 给出上游 recipe 的对象统计：
`nvfp4 96`、`bf16 916`、`q8_g32_fp16 96`、`fp32 72`、`fp8_e4m3 2`、`nf4_g16 128`、`q4_g64 1`。
`docs/maintainer/qwen4_exp-model.md` 写明取舍：**Main routed experts 导入 NVIDIA NVFP4、允许 A4**；
MTP experts Block-FP8 A8；GDN/QSA 大投影 Q8 A16；PLE 表 NF4。

- NVFP4 的 A4 路径需要 FP4 张量核（SM100/SM120），4090（sm_89）没有；本地 T013/T108 已判死该路线。
- Q8 A16 的 GEMM 是 `q8_g32` 行缩放，本地 linear 只注册了部分形状；`14976x5120` 一类的 5090 专用 tile 缺失。
- `upstream/feat/qwen4-27b-ops` 的 18 条提交全部标注 "RTX 5090"（四头 k8v4/nvfp4/int8/bf16 KV、
  20480 整向量 rmsnorm、fp8 四流 residual mixing），属 sm_120a 调优，sm_89 不适用。

**结论**：上游代码是完整的，但它对 4090 的可用性取决于"有没有 sm_89 版本的专家与线性内核"。
本地已有 q4/q5/q6 的 groupwise-int 专家路径（本次扩展到 Qwen4Exp 形状），因此走
"保留 GGUF 的 q4/q5/q6 专家、补齐所需内核"这条路，不需要重转专家，也不需要 NVFP4。

### 6.2 本次已落地的内核侧（`a72a704a`）

- groupwise-int 全链路按 `ShapeKind` 选形状（工作区/分组/token/grouped/shared），新增
  `SwiGLUH2560I640E512K10` 的 q4_q5、q4_q6 两条 ShapePolicy。
- `CudaSwiGLUExperts` 增加可选 `gate` / `up` 视图，支持"一专家两对象"的拆分 bank；
  同时给两种形态会被拒绝，不做优先级猜测。
- 新形状暴露并修掉三个在 2048 下不可见的缺陷：`dot_fp32_rows` 的组四元组越行读、
  `grouped_integer_down` 的 16 字节 scale 暂存错位（20 字节行距）与越界读、
  `reduce_assignments` 的 16 字节向量错位（H=2560/256=10 元素）。
- 验证：新增 `ninfer_moe_qwen4exp_split_test`（512 专家拆分 bank、T=1/8/64、对同一批码字的
  double 参考），compute-sanitizer 0 错误；`ctest -R "moe|qwen4"` 10/10 通过。

### 6.3 仍未做

1. 模型层接线（`ffn.cpp` 仍以 "M5 wires the split form" 拒绝三 bank）：
   `ExpertStream` 只支持 gate_up+down 两 bank，Qwen4Exp 的 121 GB 工件必须走三 bank 流式。
2. QSA/indexer（12 层）与 HC、PLE 注入进 `run_layers`——依赖 6.1 表中缺失的上游内核。
3. 上游 368 提交的吸收仍按 §3 的 v10 计划另立，与本节的"能跑起来"目标分开推进。

## 7. 增量对齐台账（2026-11-09）

**固定基线＝`77c2e5e2`（`upstream/master`，2026-11-07）。** 基线以内的 1565 条上游提交已逐条写入台账：
这是**已逐条核对（triaged）**，**不是逐条吸收（absorbed）**——`absorbed` 只表示"有书面依据说明内容已在本地"，
其余七条分级各自带原因。此后每一轮只看基线之后的新提交，旧的 1565 条不再重读。

| 项 | 值 |
|---|---|
| 台账 | `docs/maintainer/upstream-dispositions.tsv` |
| 列 | `commit`（8 位短哈希）/ `date`（`%ad --date=short`）/ `subject`（原样，去制表符，超 120 字符截断并以 `...` 结尾）/ `class` / `evidence` / `note`；UTF-8 无 BOM、制表符分隔、首行表头 |
| 行数 | **1565**，与 `git rev-list upstream/master` 逐条对应（无重复、无遗漏）；2026-11-16 销账只改 44 行的 class，行数不变 |
| 分级（2026-11-16 销账后） | absorbed **1313** / dead **155** / deferred **23** / na-arch **62** / na-doc **9** / na-hardware **2** / todo **1**（销账前：1306 / 151 / 0 / 62 / 0 / 2 / 35 + needs-review 9，见 §9） |
| 生成方式 | 先 `git log upstream/master --format="%h%x09%ad%x09%s" --date=short` 取全量行，再按 §7.2 规则表顺序匹配、先命中先用地打级；证据列只引用下列文档或本地提交号，不自造哈希 |
| 报告脚本 | `powershell -NoProfile -ExecutionPolicy Bypass -File tools/upstream/alignment_report.ps1` |

报告脚本以自身位置（`tools/upstream/`）推出仓库根，任意工作目录都能跑；`upstream` remote 或 `upstream/master` ref 缺失时以退出码 **2** 明确报错（不静默通过）；
`baseline..upstream/master` 只要有一条缺处置记录就以退出码 **1** 退出并打印 `NO ROW` 清单，因此可以直接当对齐门禁用（0＝干净）。

**规则：每一轮只处理 `baseline..upstream/master` 中缺处置记录的新提交，旧提交不再重读。**

> 2026-11-16 销账（§9）只重定 44 行的 `class`（35 todo + 9 needs-review），**基线仍 `77c2e5e2`、行数仍 1565、
> baseline 之后新增提交仍为 0**；因此"每个基线后新提交都必须有处置行"这条门禁的语义不变，变的只是旧行的分级。

### 7.1 分级口径

| class | 判据 | 台账里的落级来源 |
|---|---|---|
| absorbed | 内容已在本地树内，或已有对表记录 | `cc3938c9`（2026-10-09 合并上游至 `81c8ce09`，1197 条）、本地提交 message 里的 `[upstream <hash>]`（移植或对表登记）、v9 各 register 判「搬入 / 跟随 / 等价」的条目 |
| na-arch | sm_89 指令集上不成立 | 按 §2 的逐类判定：MXFP8 / NVFP4 / Block-FP8 / TMA / cluster / A16 / EP / CSF / MLA / kpool / A8-FFN |
| na-hardware | 只有 sm_120a（RTX 5090）才有的对象 | §6.1 |
| dead | 判死不吸收 | T001 封口池（Spark 96 / GLM 18 / TP2 18）、T013/T108（FP4-Local），以及 register 明写「不搬入 / 不跟随」的条目 |
| todo | 待吸收，缺口已定位 | §3 的 `77c2e5e2`、§6.3 的 `qwen4_exp`/QSA/`gated_residual` 缺口、qwen4exp 挂起项（T192–T198）、§6 的 305 条内查不到落地依据者 |
| deferred | **确实未做，且被外部条件挂起**——note 必须写出挂起条件与解锁判据；条件不出现就不开工 | 2026-11-16 新增：qwen4_exp 运行时 14 条、KDA/预算/传输/夹具类 9 条（§9） |
| na-doc | **上游 docs-only 提交**，本地文档集独立演进，无代码对象可吸收 | 2026-11-16 新增：原 needs-review 的 9 条上游 `docs` 提交（§9） |
| needs-review | 规则表放不下，需人工定处置 | 证据列**留空**；2026-11-16 后该列为 0 条 |

### 7.2 规则表（前缀 / 路径 -> 分级 -> 证据）

顺序匹配，先命中先用：

| 匹配 | class | evidence |
|---|---|---|
| 是 `81c8ce09` 的祖先 | absorbed | `local:cc3938c9` |
| 某本地提交的 message 带 `[upstream <hash>]` | absorbed | `local:<该本地提交>` |
| v9 各 register 逐条判定过（T071–T074、T120–T168、T192–T199） | 按 register | `v9-<area>-register#T<nnn>` |
| `a754fcf3..77c2e5e2`（63 条） | 按 §2 表 | `upstream-alignment#2:<行名>` |
| 主题含 `spark` / `dgx` / `glm` / `tp2` / `distributed` / `tensor-parallel` / `sharded` / `partitioned vocab` | dead | `v9-closeout-register#T214:T001` |
| 主题含 `FP4-Local` | dead | `v9-closeout-register#T214:T013/T108` |
| 触及 `src/ops/ple` | absorbed | `upstream-alignment#6`（8/8 已移植） |
| 触及 `src/models/qwen4_exp` 运行时（program/execution/state/parallel） | deferred | `v9-qwen4exp-register#T192-T198`（QSA/gated_residual 算子按 §8 已移植，走 absorbed） |
| 主题含 `mxfp8` / `nvfp4` / `block-fp8` / `tma` / `cluster` / `a16` / `csf` / `mla` / `kpool` / `a8 ffn` | na-arch | `upstream-alignment#2:perf(<area>)` |
| `perf(linear\|moe\|swiglu\|kpool\|kda\|mhc\|rmsnorm\|embedding)`、`style(ops)` | na-arch | `upstream-alignment#2:<原行名>` |
| `q5090` 前缀、`docs` 提交里提到的 RTX 5090 | na-hardware | `upstream-alignment#6.1` |
| 在 `81c8ce09..a754fcf3` 内且以上皆不命中 | todo | `upstream-alignment#6` |
| 其余（上游 `docs` 提交） | na-doc | 本地同主题文档（§9.3 逐条对应） |

### 7.3 当前实测（2026-11-09 跑 `tools/upstream/alignment_report.ps1`）

```
== upstream alignment report ==
register : H:\TeleAgentWorks\ninfer-4090-port\ninfer\docs\maintainer\upstream-dispositions.tsv
           1565 row(s), 6 class(es)
upstream : upstream/master @ 77c2e5e2  2026-11-07  fix(runtime): recover preempted requests without stalling
baseline : 77c2e5e2  2026-11-07  fix(runtime): recover preempted requests without stalling
           pinned at upstream commit 1565 of 1565; 0 commit(s) after it

nothing to triage: the register already covers the upstream tip.
classes over the whole register:
  absorbed       1306
  dead           151
  na-arch        62
  todo           35
  needs-review   9
  na-hardware    2

ok: every upstream commit up to the tip has a disposition row.
```

> 以上是 **2026-11-09 的运行快照**（六类）。2026-11-16 销账后为七类，数字以 §7 表与 §9 为准；
> 本轮只改台账、未重跑脚本，脚本复跑放到下一次对齐时执行（预期输出应与 §9 的计数逐项相等）。

读法：

- 63 条 `a754fcf3..77c2e5e2` 的 class 与 §2 完全对齐（60 na-arch / 2 dead / 1 todo＝`77c2e5e2`）。其中 `361dbbd1`、`4441cf47` 两条 Spark 提交在 §2 里按 `perf(rmsnorm)`、`perf(embedding)` 记为不适用，故落 na-arch 而非 305 段里 Spark 惯用的 dead——以 §2 为准。
- `absorbed` 1306（销账后 1313，见 §9）＝`cc3938c9` 合并带入的 1197 条 + 109 条有本地移植/对表依据的条目。**这仍然是 `cc3938c9` 的口径，不重新主张 §6 已否定的「305 条已吸收」。**
- `needs-review` 9 条全部是上游 `docs` 提交（`b963bf96`、`19c3a2ae`、`a78bf9ec`、`af26313c`、`c72ba1b9`、`05327b22`、`6e772ee3`、`f1ace85e`、`fbd285b5`）；2026-11-16 一次性落 `na-doc`，该列清零（§9）。
- `dead` 151（销账后 155，见 §9）＝T001 封口池与 T013/T108，加 register 明写「不搬入 / 不跟随」的条目（T120、T123、T124、T145 五条、T149、T151、T163、T127 的 Docker 条）。
- 每次上游前移，先跑脚本拿 `NO ROW` 清单，补行（同一规则表），再看退出码是否回到 0。

## 8. 更正 6.1 的两行：QSA/HC 内核**本地已有**，缺的是格式与接线（2026-11-10）

§6 的表按**路径**比对上游目录，把命名不同当成了缺失。逐文件核对后更正：

| 上游路径 | 本地实际 | 结论 |
|---|---|---|
| `qsa/indexer/{score,select,work_map,plan,launch}`, `qsa/prepare/{kernel,norm_rope,cache}`, `qsa/sparse_gqa/{kernel,tile,prefill,simt,epilogue}` | 本地是**扁平同名**文件：`qsa/indexer_{score,select,launch}.cu(h)`、`qsa/prepare_{kernel,cache}.cuh`、`qsa/sparse_gqa_{kernel,tile,prefill,simt}.cuh`，且已进 `qsa/sources.cmake`、随 `ninfer_ops` 编译 | **已移植**，不是缺失；本地没有的是上游的 `rtx5090/`、`dgx_spark/` 两个硬件变体目录（sm_89 本来就不适用） |
| `gated_residual/mix/*`, `norm_combine/*` | 本地 `gated_residual/mix/{launch,plan,operands,mix.cpp}` + `mix/kernel/{up_mma_mix,up_tma_mix,output,shared_layout,math}` + `kernel/{combine,norm}` | **已移植**（含 MMA 与 TMA 两条），同样缺 `rtx5090/`、`dgx_spark/` 变体 |
| `ple/*`、`models/qwen4_exp/ple/*` | 8 + 12 文件齐 | 已移植 |

因此 M4 的真缺口不是"移植内核"，而是下面三类**本地化**工作：

1. **格式接纳**（sm_89 路线的核心）：
   - QSA 输入投影只认 `Q8 [6144,6144]` 或 BF16；工件是 **Q4 [12288,2560]**（query+gate 同父）+ Q4/Q5 的 K/V、Q4 index_key；
   - HC `mix` 只认 **BF16 contiguous**；工件存 **Q5**（`[320,10240]`/`[10240,320]`/`[4,10240]`）；
   - PLE 表是 **q4 136 B/行**，`PleTableSource` 只吃 NF4 90 B/行。
2. **模型层接线**：`text.cpp` 的 indexed 层仍打印 "QSA not wired yet (M4)" 并退回 full attention；HC 的四流混合、PLE 的第 2 层注入都没有进 `run_layers`。
3. **sm_89 路径选择**：`up_tma_mix` 走 TMA（sm_90+），4090 必须走 `up_mma_mix`/SIMT；QSA 的三个内核要确认没有 `cp.async.bulk`/`wgmma`/cluster 依赖（本地已按 sm_89 编过，仍需按形状实测）。

### 8.1 单独立项：`ninfer_qwen3_5_moe_real_test` 的 0xC0000005

- 现象：`exercise_maximum_configuration`（256K 最大配置 + "routed experts in host RAM" 回退）崩 `0xC0000005`，最后一行日志固定在
  `[budget][split] free=20478 MiB kv=3118 MiB pool=16336 MiB safety=1024 MiB by:curve: affordable`。
- 归因：A2 时把 `expert_stream.{h,cpp}` 与 `ffn.cpp` 回退到基线 `98c3f24d` 重建，崩点与日志逐字相同 → **基线既有**，与 A1–A4 无关。
- 处置：登记为本文件外的独立项（不阻塞 Qwen4Exp 路线），需要时按"最大配置 + host 专家回退"单独复现与定位。

### 8.2 A6：dense 因果注意力接纳 24q/2kv@256（GQA-12，2026-11-10）

Qwen4Exp 的 12 个 indexed 层是 `num_attention_heads=24, num_key_value_heads=2, head_dim=256`（组 12），
而 dense 因果注意力只承认 `(24,4)`/`(16,2)`。真机第一处失败：

```
error: causal_softmax_attention workspace: unsupported head geometry
```

**根因不是白名单常量本身，而是四处独立假设连成的链**：

1. `causal_softmax_attention.cpp:36` `require_causal_geometry` 只列了两对（容量查询和运行时共用）；
2. `bf16/plan.cpp` 原来的 `const int kv_heads = heads == 24 ? 4 : 2` **由 q 头数反推 kv 头数**，规划永远描述不出 24q/2kv 的 cache；
3. `bf16/launch.cu` / `bf16/batch_wide_launch.cu` 的模板分派只看 `p.query_heads == 24`，因此选到 `CausalD256H24Kv4`；
4. `bf16/template_launch.cuh:18` 的 `cache.kv_heads != Geometry::KVHeads` 在最后一刻以 "operands do not match" 拒绝。

内核本身与组大小**无关**：`grouped_mma.cuh` 的行轴是 `width * GroupSize`（token/组内头打成 packed row），
kv 头数只进 `grid.x = KVHeads * ceil(width*GroupSize/M)` 与 `head = blockIdx.x % KVHeads`；
`batch_wide_mma.cuh` 是一个 CTA 一个 `(query tile, query head)`，`kv_head = q_head / GroupSize`。
共享内存与寄存器里都没有 group 项：grouped 固定 32 KiB（`union qkv[2*32*256]` 与 reduction 同尺寸，
≤48 KiB 用静态 smem，`minBlocks=2`），batch-wide Tiled64 为 `(64+2*64)*256*2 = 96 KiB`/CTA、128 线程。
所以 sm_89 不需要新 tile，只是同一 KV 头被 12 个 CTA 重读（24q/4kv 是 6 个），属带宽注记。

改动：新增 `CausalD256H24Kv2`；`make_bf16_kv_causal_plan` 与 `bf16_kv_workspace_bytes` 改为
**kv 头数与 q 头数分别来自 profile**（不再推导），plan 记录两者、`group()` 由二者求出；
两处分派按 `(q 头数, kv 头数)` 选模板；几何白名单只加这一条，并在 24/2 时把非 BF16 KV 存储按名拒绝
（int8/fp8/k8v4/nvfp4 的模板仍只有 `(24,4)`/`(16,2)`，不该走到 "invalid operands"）。

验证（`--kv-dtype bf16`，`NINFER_OP_REPORT_STATS=1`，27 例）：最差 `rel_l2=1.694e-3`（上限 `2.8e-3`，
ratio 0.605）、`max_abs` 3.06e-5…9.73e-4；既有 24q/4kv 的最差 ratio 0.624、16q/2kv 0.629 —— 同带，判据未放宽。
覆盖 T=1/2/3/5/6/7/15/16/17、单 token 对 8192 keys、T=2048/keys=6144 前缀、32768 rung 上下沿与 40008 keys、
graph 重放 3 例、双请求 batch、4 档 SM 的容量/分配镜像。

真机第一处失败前移到 `error: q5 linear: unsupported shape`；临时探针（已完整回退并重编）显示为 LM head
`n=248320, k=2560, T=1`：`startup.cpp:413-415 linear_scratch(output_head)` → `linear.cpp:124` →
`q5_dispatch.cpp:117`。规划能走到 head，说明 48 层（含第 3 层 24q/2kv）的注意力容量查询全部通过。

基线既有问题（与 A6 无关，A/B 复现）：`ninfer_softmax_attention_test` 的 packed 套件崩 `0xC0000409`，
末两行为 `packed_softmax_attention T=11 / T=194 ... actual=3.38953e+38`。

## 9. 2026-11-16 销账结果

目的：在"按祖先合并上游"之前，把台账里 35 条 `todo` 与 9 条 `needs-review` 一次性落到终态，使每一行都能回答
"合并时要不要为它单独做一次内容对齐"。

### 9.1 口径与边界（先写清楚，避免把"定级"读成"已实测"）

- 本轮**只改两份文档**（本文件 + `upstream-dispositions.tsv`）：不改代码、不编译、不运行程序、不执行 git 命令。
- 定级依据 = **本地树可核事实**（文件路径/行号、各 v9 register 的判定）+ 提交 subject。
  **commit→path 明细本机不可得**（本轮禁止 git：`infer-upstream` 虽有 `.git` 也不读其历史，只用它的工作树核对对象存在性），
  因此不能落到具体本地文件的判定，一律按 subject + register 给出机制原因与解锁判据，不虚构文件归属与哈希。
- 行数仍 **1565**、基线仍 `77c2e5e2`；`baseline..upstream/master` 新提交 **0**。本轮只重定 44 行的 `class`
  （35 `todo` + 9 `needs-review`），其余 **1521 行一字未动**，顺序未重排。
- 新增两个 class（`deferred`、`na-doc`），§7.1 已各补一行判据，因此**分级从六类变七类，计数必然变化**。

### 9.2 before / after

| class | before | after | 变化 | 说明 |
|---|---|---|---|---|
| absorbed | 1306 | **1313** | +7 | 本轮找到本地等价物（见 9.3） |
| dead | 151 | **155** | +4 | 多 rank / NCCL / rank 执行 / attention DP 放置 |
| deferred | 0 | **23** | +23（新类） | qwen4_exp 运行时 14 + 其余 9；每条都带挂起条件与解锁判据 |
| na-arch | 62 | **62** | 0 | 未触碰 |
| na-doc | 0 | **9** | +9（新类） | 原 `needs-review` 的 9 条上游 docs 提交 |
| na-hardware | 2 | **2** | 0 | 未触碰 |
| todo | 35 | **1** | −34 | 仅剩 `77c2e5e2`（见 9.4） |
| needs-review | 9 | **0** | −9 | 本类清空 |
| 合计 | 1565 | **1565** | 0 | 行数不变 |

### 9.3 每类代表条目（证据都在本地树内，路径可 rg 到）

**absorbed +7**

| 提交 | 本地依据 | 判据 |
|---|---|---|
| `f8790a78` vendor tomlplusplus 3.4.0 | `third_party/tomlplusplus/toml.hpp:1154-1156`（`TOML_LIB_MAJOR/MINOR/PATCH = 3/4/0`）、`cmake/Dependencies.cmake:118` | 版本逐位相同 |
| `35aede69` share ops and core libraries | `src/core/CMakeLists.txt:12`（`ninfer_core`）、`src/ops/CMakeLists.txt:2`（`ninfer_ops`）、`apps/CMakeLists.txt:3` | 共享库分层已在，apps 直接链接 |
| `066714cc` unify target capabilities / roofline | `src/core/device_roofline.h:1-71`、`bench/ops/ninfer_bench_common.h:87-101`（`device_peak_bw_gbs`） | roofline 从"固定 1792"改为**设备查询**（V10-11），11 处 bench 已改用；能力位宏在单 target 树不采用 |
| `2cf04452` separate planning and allocation budgets | `include/ninfer/types.h:80-93`（planning allowance / KV sizing reserve 两账分立）、`src/core/memory_budget.h:1-29`、`src/runtime/engine/context_cache/joint_planning.h:11` | 概念等价；**无同名 `StartupMemoryBudget` 类**（上游 `src/runtime/engine/startup_memory.h`），合并时按语义而非文件名对 |
| `637cf4a4` standardize adp naming | `src/serve/monitor.cpp:288`（`{"adp", ...}`）、`docs/maintainer/distributed-execution.md:19-26` | 命名已在；实现是 `size=1` 占位 |
| `76c39559` / `81671499` bench 原生 prefill/decode 计时与前缀预热 | `bench/inference/infer_bench_support.cpp:298`（`prefill_seconds`）、`:639-704`（`decode_graph_prime`）、`infer_bench_support.h:33/126-127` | decode 计时分组 + `graph_prime` 字段入报告 |

**dead +4**（全部并入 `v9-closeout-register#T214:T001` 的多 rank 封口池）

| 提交 | 机制原因 |
|---|---|
| `b281592d` bound NCCL memory | NCCL 显存上界只在多 rank 执行存在；单卡无 NCCL |
| `1b6c7232` measurement boundaries across ranks | "across ranks" 无对象：本地 `tp/ep/dp/adp` 在 `monitor.cpp:279-288` 恒 `size=1` |
| `5cc161ca` attention DP placement foundations | attention DP 放置需要多 rank/多副本；本地 `placements` 恒空 |
| `71cdd615` separate native sessions and rank execution | rank 执行是多 rank 设施，单卡无该层 |

**deferred +23**（每条 note 都写了挂起条件 + 解锁判据）

- qwen4_exp 运行时 **14 条**：`f2620157`、`088e8da4`、`4f31ad4a`、`985121d4`、`afe43480`、`439e1075`、
  `3444964f`、`1408277a`、`9cdbf221`、`5e581e92`、`2507d770`、`e07307e0`、`7131c0b9`、`f8a774f7`。
  统一口径：上游 `src/models/qwen4_exp/` 是 75 文件整层（`infer-upstream` 工作树），本地只有 **28** 个文件
  （`rg --files src/models/qwen4_exp` 口径：config、text_config、load（含 load/text）、execution{qsa,hc,ple_layer}、ple），
  **缺 program/、state/、startup、model、weights.h**；本地运行走的是 qwen3_5 树上的自研
  M1/M2 外挂（`v9-qwen4exp-register` §"M1/M2 已落地"、`text.cpp:541` 的 `mtp_hc_stem`）；解锁 =
  qwen4exp 工件就绪 + T013 可行性账通过 + 主路径数值缺陷（PPL 1.49e6，同 register §4）关闭，即 T192–T198 复评。
- 其余 9 条：`2c94555b`（KDA 输入准备已在 `src/ops/linear_attention/kimi_delta_attention/chunked/prepare.cu`，
  但 ReplaySSM 只接 GDN（`include/ninfer/ops/gdn_replay.h`），KDA 无消费方——GLM5Next 已判死）、
  `5551d9b1`（公共前端抽取，T149 不跟随，等 qwen4-exp 前端立项）、`e0114c70`/`996d9214`（qwen4_exp program 层，
  本地无该对象）、`d4a78e03`（serving bench 无流量账，只 ops bench 有 `effective_gbs`）、`c996baa8`（标定表可用但
  本地只有热序，无 model 侧默认序注入，T161）、`6a42de35`（`copy_bytes` 传输层未收口，register A2）、
  `ae54c1a9`（7 个扁平头仍在 `src/ops/common/`，register A5 前置）、`a0d27b22`（无 `tokens_per_row`，
  `src/core/paged_kv_cache.cpp:39-41` 固定 64 页，register A1）。

**na-doc +9**：`b963bf96`、`19c3a2ae`、`a78bf9ec`、`af26313c`、`c72ba1b9`、`05327b22`、`6e772ee3`、
`f1ace85e`、`fbd285b5`。全部是上游 `docs` 提交（无代码对象），本地文档集独立演进；台账已逐条指向本地的
同主题文档（`README.md`/`model-cards/`、`engine-architecture.md`、`resource-scheduling-and-context-cache.md`、
`paged-kv-cache.md`、`AGENTS.md`/`CONTRIBUTING.md`、`op-development.md`、`qwen3_5-model.md`、`storage-layouts.md`）。

### 9.4 仍留在 `todo` 的唯一一条：`77c2e5e2`

不落终态的理由（不是漏做，是**按现有定义落不进去**）：

- `deferred` 要求"被**外部条件**挂起"，而 §3 明确写"启动条件：无外部依赖，可立即开工"——它是排期问题，不是阻塞问题。
- `absorbed` 要求"内容已在本地"：本地确有 preemption 恢复的骨架（`include/ninfer/types.h:823`
  `GenerationRecoveryRoute{None,Snapshot,Replay}`、`src/runtime/engine/engine_core.h:2245-2246/2313/2402`、
  `src/serve/request_log.cpp:347/655` 的 `preemptions`/`preemption_index` 遥测），但 §3 的四条子项里
  "零 frontier 请求直接恢复（`Direct` 路由不存在）""关闭顺序""d 回归夹具"都还没落地，写 `absorbed` 会掩盖 v10 的
  实际工作量。
- 结论：**留 `todo`，由用户决定是并入 v10 立项还是明确降级为 `deferred`（若决定降级，解锁判据＝§3 的 DoD 六条）。**

### 9.5 合并前的读法（给"按祖先合并"用）

1. `absorbed` 只保证"内容等价"，**不保证同文件同形态**——`2cf04452`/`066714cc`/`637cf4a4` 是概念等价、无同名对象，
   合并时按语义判断，不要按路径覆盖。
2. `dead` 的 4 条都来自多 rank 封口池；若将来单卡线要接 NCCL/rank，这 4 条与 T001 需一起复议，不是永久不可逆。
3. `deferred` 的 23 条都带解锁判据，**判据不出现就不开工**；其中 14 条 qwen4_exp 运行时是同一条外部依赖，
   应作为一个批次（T192–T198）处理，不要逐条 cherry-pick。

## 10. 2026-11-16 增量 64 条（`77c2e5e2..454b922f`）

上游前移到 `454b922f`（2026-11-16）。本轮只做**增量登记**：把 `77c2e5e2..454b922f` 的 64 条按 §7.1/§7.2
同一口径补进台账，**不改代码、不编译、不运行程序**；`git` 只用 `log/show/rev-list` 只读。

> §1/§2 的"上游基线"行仍写 `77c2e5e2`，本轮不改写旧表（避免与 §9 的计数口径打架）；
> 台账已覆盖到 `454b922f`，基线改写与 `alignment_report.ps1` 复跑放到下一次对齐时一次做完。

### 10.1 计数（before → after）

| class | §9 后 | 本轮 | 合计 | 说明 |
|---|---|---|---|---|
| absorbed | 1313 | 0 | **1313** | 本轮没有一条能给出本地等价物证据（见 10.4） |
| dead | 155 | +15 | **170** | Spark/GLM 线 11 条 + 多 rank 通信 4 条 |
| deferred | 23 | +19 | **42** | 每条 note 都写了解锁判据（10.5） |
| na-arch | 62 | +14 | **76** | NVFP4/A16/FP8-A8/mhc/FP4 流水 |
| na-doc | 9 | +3 | **12** | 上游 docs 提交 |
| na-hardware | 2 | +11 | **13** | 5090 形状/变体目录 |
| todo | 1 | +2 | **3** | 新增 `8e464812`、`4461c968`（见 10.2） |
| needs-review | 0 | 0 | **0** | 无不可定级条目 |
| 合计 | 1565 | +64 | **1629** | 文件 1630 行（表头 + 1629 行） |

新 64 条：`dead 15 / deferred 19 / na-arch 14 / na-hardware 11 / na-doc 3 / todo 2`。

### 10.2 必须吸收（本轮 `todo`，缺口已定位到文件/行）

| 提交 | 为什么必须吸收 | 建议落点 |
|---|---|---|
| `8e464812` `fix(grammar)` backport #1003/#942 | 修正两处语法缓存/ token 边界缺陷：① 字符预算路径下 `has_token_edges_` 只在 `has_char_budget_rules_` 时计算（`grammar_matcher.cc:508`），token 边存在但被漏判；② 字面量后紧跟 token 级自由文本区（如 `"\n\n"` 后 `any_tokens`）时，整 token 被拒（新增 `FindMidTokenExcludeEdgeRow`，回移边界）。**本地 9 个改动文件与提交前像逐字节相同**（`git hash-object` 对比 `8e464812^`：`grammar_matcher.cc c85497c1`、`earley_parser.h a3ef3e71`、`earley_parser.cc`、`grammar_impl.h a2045b40`、`grammar_compiler.cc 0acf00cf`、`grammar_functor.cc 661348c8`、`container.h`、`json_serializer.h`、`tests/test_grammar.cpp` 全 SAME），因此可原样打补丁 | `third_party/xgrammar/cpp/{grammar_matcher.cc,earley_parser.{h,cc},grammar_impl.h,grammar_compiler.cc,grammar_functor.cc,support/{container.h,json_serializer.h}}` + `tests/test_grammar.cpp` |
| `4461c968` `fix(frontend)` 非法工具历史 | 历史里的 `tool_calls[].arguments` 目前**直接 `Json::parse`**（本地 `src/models/qwen3_5/frontend/chat_template.cpp:227-228`），坏 JSON 抛的是 nlohmann 异常而不是带 `messages[i].tool_calls[j]` 身份的 `invalid_argument`，落不到 400 `invalid_prompt`。上游把它做成共享解析（空参数保留为空对象、非对象拒绝、报告字节位置），并补 5 个坏例用例 | `src/models/qwen3_5/frontend/chat_template.{h,cpp}`（本地无 `models/common/frontend` 共享层，先落在 qwen 前端，等共享层建立再上提）+ `tests/models/qwen3_5/test_frontend.cpp` |

### 10.3 建议吸收（`deferred` 里优先级最高的 7 组）

| 提交 | 为什么 | 建议落点 | 解锁判据 |
|---|---|---|---|
| `54466d12` 并发执行容量 | 本地并发被硬钉在 **8**（`include/ninfer/types.h:23 kMaximumConcurrency = 8`，`src/runtime/engine/scheduler.h:27-29/48-49` 用它开数组，`model_instance.cpp:899` 校验上限）。上游删掉固定行上限，改为 C（驻留）/B（本轮成员）/P（物理行）/L（ADP 局部行）/W（阶段列）合同 + 共享 `models/common/batch`（新增 `batch.{h,cpp}`，桶 `1..8,10,12,14,16,20,…`），padding 语义写进新 `docs/designs/concurrency-contract.md`。**单卡 >8 并发是本地 serving 的直接上限**，但它一次改 651 文件（+14602/−5994） | 新增 `src/models/common/batch.{h,cpp}`；`types.h`、`scheduler.h`、`engine_core.h`、`src/models/qwen3_5/{program,session}/*` | 先有 `models/common` 层与 Work/Decision/Commit 分层；建议切片顺序：① batch 桶+padding 表 → ② 容量改配置校验 → ③ C>8 的 Graph 桶 |
| `3b76b98c` 回收/保留代价记账 | 单卡显存压力下的核心决策（删 checkpoint / Host 保留 / capture / pause 用同一代价模型比较）。上游新增 `context_cache/recovery_cost.h`(636 行) + `recovery_value.h`(193) + `runtime/contract/recovery.h`(90) + `models/common/context/identity.h`(73)，+4516/−505 | `src/runtime/engine/context_cache/recovery_cost.h`、`resource_manager.h`、`src/models/qwen3_5/program/checkpoint*` | 现有单文件 `context_cache/resource_manager.h`（91 KB）先拆出 ReclaimSelection 与代价模型；否则无处安放 |
| `e8059c32` 压力下保留当前输入 | 输入 checkpoint 被冷历史挤掉就退化为整段重算。上游加 `CaptureStorage{Automatic,Device,Host}`、`protected_points`、`replaces_input` 与专用 `less_rank`（`reclaim_selection.h`） | 同上；`src/models/qwen3_5/program/checkpoint_export.cpp` | 与 `3b76b98c` 同批（同一文件的同一决策点），不要单独 cherry-pick |
| `77418cfe` 去除重复 prompt 工作与 prefill 分裂 | 本地**确有**该机制：`rewrite_execution_frontiers` 分布在 10 个文件（9 个源文件 + 1 个测试；`src/models/qwen3_5/frontend/processor.h:113/133`、`prepared_prompt.h:91`、`program/prefix_identity.*`、`program/prefill.cpp`、`planning/request_plan.cpp` 等），强制 prefill 在重写前沿分裂；上游整条删掉（69 文件，−867 行） | `src/models/qwen3_5/frontend/{processor,prepared_prompt,prompt_layout}.*`、`src/models/qwen3_5/program/prefix_identity.*` | 本地前端与 identity 结构对齐后再删；删前先量一次同会话交错 A/B（prefill 分块数） |
| `13ebb92f` + `52e7cd7e` Jinja 生命周期/渲染开销 | 每条请求都渲染 chat template，Windows 单线程 CPU 上是真实成本。上游把异常 break/continue 换成 `loop_flow`、帧改成池化 `render_scopes`、`render()` 直写字符串，并修了捕获悬垂/迭代器关闭/`divisibleby` 零除与 INT64_MIN 取模。**本地已分叉**：`runtime.h:64` 用 `value_object` 环境、`capture_scope` 返回 `weak_ptr`（`:80`），`break_statement` 仍抛异常（`:262-279`），全树无 `scope_analysis`/`render_scopes` | `third_party/llama-jinja/jinja/{parser,runtime,value,string,json,unicode}.cpp`、`tests/text/test_jinja.cpp` | 按本地分叉重写（不能 `git show \| patch`）；先把上游新增的语义用例（宏捕获/loop 元数据/零除与 INT64_MIN 取模）在本地补齐再动结构 |
| `0386059d` JSONL 饱和负载 | 本地并发压测的饱和夹具仍是 `long_decode_aime26_15`（`tools/bench/run_serve_concurrency.py:34`），上游换成结构化 JSONL 场景并扩容种子（`saturation_seeds`），更贴近 grammar 约束下的 decode 饱和 | `tools/bench/run_serve_concurrency.py` | 与 `54466d12` 同批改口径；否则点集与并发能力不匹配（本地 C≤8，上游无上限） |
| `1fdb3716` 多模态前缀查找/恢复 | ① 视觉单元在部分重放时按整图计 `vision_items/vision_patches`；② 前缀短名单键不再要求 `can_checkpoint(f)`（分支可以落在 media 中间）。本地两个对象都缺：无 `src/models/glm5next`，也无 `src/models/qwen4_exp/session/`（qwen3_5 走自己的 `program/vision_control.cpp`，未触及） | qwen4_exp 运行时批次（T192–T198）内一并吸收 | qwen4_exp 工件与主路径数值缺陷关闭（同 §9 的 14 条 deferred） |

### 10.4 特别判断的 13 条：本地现状 → 结论

| 提交 | 本地现状（可核证据） | 结论 |
|---|---|---|
| `54466d12` | `types.h:23 = 8`；`scheduler.h:27-29`；无 `src/models/common` | 相关（卡吞吐上限），**deferred**；整包 651 文件不能当补丁 |
| `3b76b98c` | 无 `context_cache/reclaim_selection.h`、`recovery_cost.h`；`resource_manager.h` 单文件 91 KB | 相关，**deferred**（先拆代价模型） |
| `1fdb3716` | 无 `glm5next`、无 `qwen4_exp/session`；qwen3_5 有 `program/vision_control.*` 与 `prefix_identity.*` 但未被该提交触及 | **deferred**（qwen4_exp 批次） |
| `77418cfe` | 本地保留 `rewrite_execution_frontiers`（10 文件，`processor.h:113`） | 相关，**deferred**（删机制需先对齐前端） |
| `8e464812` | 9 个文件与 `8e464812^` 逐字节相同（`hash-object` 全 SAME） | 相关，**todo**（可直接打） |
| `13ebb92f` | `runtime.h:80` weak 捕获、`:262` 异常 break；与 `13ebb92f^` 全 DIFF | 相关，**deferred**（重写非搬运） |
| `52e7cd7e` | 无 `render_scopes`/帧池；`runtime.h` 仍是 `deque` 之外的旧分叉形态 | 相关，**deferred**（与上条同一改造） |
| `4461c968` | `chat_template.cpp:227` 直接 `Json::parse`；无 `models/common/frontend` | 相关，**todo**（小改，落 qwen 前端） |
| `e8059c32` | 无 `reclaim_selection.h`/`CaptureStorage`/`protected_points` | 相关，**deferred**（随 `3b76b98c`） |
| `0d32c2a6` | 只改 `replay/rtx5090/record.cu`；本地 `gated_delta_net/` 只有 `chunked/` + 单层 `replay.cpp`（无 `record/` 子目录），全树 `rtx5090`/`dgx_spark` 目录各 **0** 个 | 不适用，**na-hardware**；`cp.async` 窗口预取思路可作 sm_89 参考，本轮不排期 |
| `51e30a8c` | `WarpAddresses` 的 `static_assert` 要求 `Fp8E4m3fnGroup128Ue8m0`；本地 QSA 白名单无该格式（`qsa/storage.h:17-25`），`sparse_gqa_tile.cuh:32-40/74-93` 仍走共享内存地址数组 | 不适用，**na-hardware** |
| `d01e04ba` | 上游四头 = 27B-5090 设计（`docs/designs/qwen4-27b-rtx5090.md:41` "24 query / 4 KV"）；本地产物 `num_key_value_heads=2`（`R:\model\qwen3_8_flash_next_v5\*conversion.json`），`qsa/sparse_gqa.cpp:23` 只收 (24,2)/(12,1) | 不适用，**na-hardware**；其中的 compact/merge 长序列内核若要在 24/2 上要收益，需单独立项评估 |
| `0386059d` | `run_serve_concurrency.py:34` 仍是 `long_decode_aime26_15`；语料侧已有 `scenario_structured_jsonl`（`run_serve_corpus.py:79`） | 相关，**deferred**（与并发容量同批） |

### 10.5 `deferred` 19 条的挂起条件分组

判据不出现就不开工；同一组按批推进，不逐条 cherry-pick。

| 组 | 条数 | 提交 | 共同解锁判据 |
|---|---|---|---|
| 并发容量 | 2 | `54466d12`、`0386059d` | `models/common/batch` + C/B/P/L/W 合同落地 |
| 上下文记账 | 3 | `3b76b98c`、`e8059c32`、`1fdb3716` | `resource_manager.h` 拆出 ReclaimSelection/代价模型；qwen4_exp 批次开工 |
| 前端/文本 | 3 | `13ebb92f`、`52e7cd7e`、`77418cfe` | 本地 Jinja 分叉重写完成 + 前端 identity 对齐 |
| draft/采样算子 | 8 | `2366b2a2`、`b5008303`、`6a753af3`、`ad49b325`、`31198bbd`、`dc4e789b`、`8c713a83`、`bf74f1cf` | 本地 draft 切到上游 dflash2 形态（bf16 finish / H4096 prepare / gathered 候选） |
| 滑窗几何 | 1 | `9be36f5d` | draft 改用 16q/4kv（本地现为 32q/8kv，`sliding_window_attention.cpp:17`） |
| 运行时/bench | 2 | `41da21f4`、`9eff5314` | 本地 `engine_core.h` 单体拆分 + bench 结构与上游 `bench/inference/options.cpp` 对齐 |

### 10.6 定级存疑/混合提交

`needs-review` 仍为 **0**：64 条都命中 §7.2 规则。但有 3 条是**混合提交**，按规则表先命中 `na-hardware`，
其中夹着一段对本地可能有用的共享改动，已在台账 note 里点名，合并时不要整包丢弃：

1. `f918cbdf`：除 5090 FP8 形状外，放宽了 row-scaled FP8 的 partial 父视图（上游 `weight_view.cpp:495` 删除
   "requires a complete parent"）；本地 `src/core/weight_view.cpp:315-319` **仍拒绝** partial RowScale/BlockScale 父视图。
2. `d01e04ba`：四头几何外，`compact.cuh`/`merge.cuh` 是通用的长序列 split 合并内核。
3. `0d32c2a6`：5090 变体外，`RecordWindow` + `cp_async` 的"取状态时顺带取记录输入"是通用写法（sm_89 支持 `cp.async`）。

### 10.7 下一次 fetch 之后的流程

1. `git fetch upstream` → `git log --format="%h%x09%ad%x09%s" --date=short <pinned>..upstream/master` 取全量；
   **目标冻结在"最近一次已核对且含需吸收项"的提交，不追 HEAD**（§3 的启动条件）。
2. 逐条 `git show --stat`；触及 runtime/context/frontend/grammar/bench 的必须看实质 diff，不按标题定级。
   涉及第三方树的，先 `git hash-object` 对比前像，逐字节相同才允许写"可直接打补丁"。
3. 按 §7.2 规则表定级；新类别先补 §7.1 判据再落行。`todo` 只用于"必须本批吸收且缺口已定位"，
   其余一律 `deferred` + 解锁判据。
4. 新行插在表头之后、保持上游日志新→旧、每行 6 列、note ≤200 字符、不改旧行。
5. 跑 `tools/upstream/alignment_report.ps1`：`NO ROW` 必须为空、退出码回 0；
   本轮未跑（只改文档），预期输出应等于 §10.1 的七类计数。
6. 复跑 `docs/maintainer/upstream-dispositions.tsv` 的自制校验（行数、6 列、无重复、note 长度、
   新旧行字节不变）——本轮结果见 10.8。
7. 收口：把本轮 `todo` 的落点并进 v10/v10.1 立项，或按 §9.4 的先例降级为 `deferred` 并写判据。

### 10.8 本轮校验结果

```
file   : docs/maintainer/upstream-dispositions.tsv
lines  : 1630  (header + 1629 rows) = 1566 + 64  ✅
cols   : 每行 6 列，tab 分隔，UTF-8 无 BOM，LF 换行  ✅
dup    : 0（1629 个 commit 唯一）  ✅
order  : 前 64 行 = `git log 77c2e5e2..454b922f` 逐行同序  ✅
old    : 既有 1565 行与 HEAD 版本逐字节相同  ✅
class  : dead 170 / absorbed 1313 / na-arch 76 / na-hardware 13 / na-doc 12 / deferred 42 / todo 3
```

