# 上游对齐状态与下一版本目标

本文件回答三件事：**本地对齐到上游哪个版本**、**对到了什么程度**、**下一轮对齐的目标与完成定义**。
本地的"对齐"是**内容等价**，不是历史合并：上游那层 Linux/SM100 的执行设施在 Windows 自研平台层上重写，
提交历史独立演进（本地 `master` 与 `upstream/master` 不存在祖先关系，`git rev-list` 的 ahead/behind 只反映历史，
不反映内容缺口）。

## 1. 当前对齐基线

| 项 | 值 |
|---|---|
| 上游基线（**已吸收**） | `a754fcf3` — `fix(runtime): prevent request starvation under context pressure`（2026-11-07） |
| 基线之前 | `de5d4288`（v8 完工）→ `a754fcf3` 共 **305** 提交，按 v9 计划吸收完毕 |
| 吸收路径 | 判死 132（Spark 96 / GLM 18 / TP2 18，双计已标注）；A 类整搬；M 类三方融合（T049–T083）；语义对表（T145–T168）；参考核对（T021–T047 / L01–L14）；模型集成登记（T191–T202） |
| 本地版本 | `git describe` → 见 `VERSION`（`v1.0.0-rtx4090-<N>-g<hash>`） |
| 对齐证据 | `docs/maintainer/v9-*.md` 各登记、`v9-closeout-register.md` 的 305 条对账、`.temp/upstream-scan/` 留档 |

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
