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
其余五条分级各自带原因。此后每一轮只看基线之后的新提交，旧的 1565 条不再重读。

| 项 | 值 |
|---|---|
| 台账 | `docs/maintainer/upstream-dispositions.tsv` |
| 列 | `commit`（8 位短哈希）/ `date`（`%ad --date=short`）/ `subject`（原样，去制表符，超 120 字符截断并以 `...` 结尾）/ `class` / `evidence` / `note`；UTF-8 无 BOM、制表符分隔、首行表头 |
| 行数 | **1565**，与 `git rev-list upstream/master` 逐条对应（无重复、无遗漏） |
| 分级 | absorbed **1306** / dead **151** / na-arch **62** / todo **35** / needs-review **9** / na-hardware **2** |
| 生成方式 | 先 `git log upstream/master --format="%h%x09%ad%x09%s" --date=short` 取全量行，再按 §7.2 规则表顺序匹配、先命中先用地打级；证据列只引用下列文档或本地提交号，不自造哈希 |
| 报告脚本 | `powershell -NoProfile -ExecutionPolicy Bypass -File tools/upstream/alignment_report.ps1` |

报告脚本以自身位置（`tools/upstream/`）推出仓库根，任意工作目录都能跑；`upstream` remote 或 `upstream/master` ref 缺失时以退出码 **2** 明确报错（不静默通过）；
`baseline..upstream/master` 只要有一条缺处置记录就以退出码 **1** 退出并打印 `NO ROW` 清单，因此可以直接当对齐门禁用（0＝干净）。

**规则：每一轮只处理 `baseline..upstream/master` 中缺处置记录的新提交，旧提交不再重读。**

### 7.1 分级口径

| class | 判据 | 台账里的落级来源 |
|---|---|---|
| absorbed | 内容已在本地树内，或已有对表记录 | `cc3938c9`（2026-10-09 合并上游至 `81c8ce09`，1197 条）、本地提交 message 里的 `[upstream <hash>]`（移植或对表登记）、v9 各 register 判「搬入 / 跟随 / 等价」的条目 |
| na-arch | sm_89 指令集上不成立 | 按 §2 的逐类判定：MXFP8 / NVFP4 / Block-FP8 / TMA / cluster / A16 / EP / CSF / MLA / kpool / A8-FFN |
| na-hardware | 只有 sm_120a（RTX 5090）才有的对象 | §6.1 |
| dead | 判死不吸收 | T001 封口池（Spark 96 / GLM 18 / TP2 18）、T013/T108（FP4-Local），以及 register 明写「不搬入 / 不跟随」的条目 |
| todo | 待吸收，缺口已定位 | §3 的 `77c2e5e2`、§6.3 的 `qwen4_exp`/QSA/`gated_residual` 缺口、qwen4exp 挂起项（T192–T198）、§6 的 305 条内查不到落地依据者 |
| needs-review | 规则表放不下，需人工定处置 | 证据列**留空** |

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
| 触及 `src/models/qwen4_exp`、`src/ops/softmax_attention/qsa`、`src/ops/gated_residual` | todo | `upstream-alignment#6.3` |
| 主题含 `mxfp8` / `nvfp4` / `block-fp8` / `tma` / `cluster` / `a16` / `csf` / `mla` / `kpool` / `a8 ffn` | na-arch | `upstream-alignment#2:perf(<area>)` |
| `perf(linear\|moe\|swiglu\|kpool\|kda\|mhc\|rmsnorm\|embedding)`、`style(ops)` | na-arch | `upstream-alignment#2:<原行名>` |
| `q5090` 前缀、`docs` 提交里提到的 RTX 5090 | na-hardware | `upstream-alignment#6.1` |
| 在 `81c8ce09..a754fcf3` 内且以上皆不命中 | todo | `upstream-alignment#6` |
| 其余（当前全部是上游 `docs` 提交） | needs-review | 空 |

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

读法：

- 63 条 `a754fcf3..77c2e5e2` 的 class 与 §2 完全对齐（60 na-arch / 2 dead / 1 todo＝`77c2e5e2`）。其中 `361dbbd1`、`4441cf47` 两条 Spark 提交在 §2 里按 `perf(rmsnorm)`、`perf(embedding)` 记为不适用，故落 na-arch 而非 305 段里 Spark 惯用的 dead——以 §2 为准。
- `absorbed` 1306＝`cc3938c9` 合并带入的 1197 条 + 109 条有本地移植/对表依据的条目。**这仍然是 `cc3938c9` 的口径，不重新主张 §6 已否定的「305 条已吸收」。**
- `needs-review` 9 条全部是上游 `docs` 提交（`b963bf96`、`19c3a2ae`、`a78bf9ec`、`af26313c`、`c72ba1b9`、`05327b22`、`6e772ee3`、`f1ace85e`、`fbd285b5`），证据列留空，等人工一次性定处置。
- `dead` 151＝T001 封口池与 T013/T108，加 register 明写「不搬入 / 不跟随」的条目（T120、T123、T124、T145 五条、T149、T151、T163、T127 的 Docker 条）。
- 每次上游前移，先跑脚本拿 `NO ROW` 清单，补行（同一规则表），再看退出码是否回到 0。
