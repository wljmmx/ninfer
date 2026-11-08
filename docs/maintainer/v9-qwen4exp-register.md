# v9 Phase 7 qwen4-exp 模型集成登记（T191–T202）

日期：2026-11-08
上游基线：upstream/master = a754fcf3
前置对表：T013 qwen4_exp-model（.temp\upstream-scan\design-parity\T013-qwen4exp-model.md）——qwen4-exp 真正新增 HC/QSA/PLE 三样、转换规格已完整转录（总容量 100.666 GiB，固定 GPU 10.257616 GiB）、4090 三拦路虎（无 NVFP4 硬件、63.28 GiB experts 需分层、PLE 27 GiB 随机行读）；T073 已登记本地无 Qwen4Exp 转换路径。

## 前置事实核查（2026-11-08）

| 项 | 结果 | 证据 |
|---|---|---|
| qwen4-exp 源权重工件 | **不存在** | `D:\ninfer-rtx4090\models\` 仅有 qwen3.6-35B-A3B（b/e2）与 qwen3.8-27B（v3） |
| 转换器 | **不存在** | `tools\convert\` 只有 qwen3_5.py 等，无 qwen4_exp.py（T073 已登记） |
| 可行性 | **未证** | T013：100.666 GiB 权重、4090 无 NVFP4 张量核（T114 已判死 NVFP4 MoE）、PLE 27 GiB 随机行读 |

## T191 qwen4exp 权重转换＋paged NF4 —— 待工件（阻塞）

- 上游 d8c0fa6f：qwen4exp 权重转换器 + paged NF4 布局。
- 本地：无源权重、无转换器 → **无法执行**。登记待工件前置链：
  1. 取得 qwen4-exp 源权重（HF 格式）与 T013 转录的转换规格（几何表/18 行 parent 行序/recipe/bank 字节/PLE 分页 90B 行/280B controls/hash）；
  2. 移植上游 `tools/convert/qwen4_exp.py`（参考 L05 NVFP4 判 dtype、L08 量化器/校准、L12 工具链健壮性五连）；
  3. 先跑**容量可行性账**（100.666 GiB → 4090 24 GiB：experts 分层比例、PLE 随机读带宽、KV 262144 ctx），不可行则登记降级路线（如 INT4/INT8 替换 NVFP4）；
  4. 工件校验（sha256/结构/量化元数据）。
- 判定：阻塞，待工件。

## T192–T198 —— 挂起（依赖 T191 工件）

| 条 | 上游 | 挂起理由 |
|---|---|---|
| T192 | f2620157 GDN shapes＋GDN 重组 | 依赖 qwen4exp 工件与 T104–T108 的 GDN 算子（本地已有 96x2560 control，但 qwen4exp shapes 需工件校准） |
| T193 | 088e8da4＋4f31ad4a 加载/资源＋执行成本 trace | 依赖 T192 |
| T194 | 985121d4＋afe43480＋439e1075 eager text/vision/mtp | 依赖 T193；vision 按 4090 显存核对（T013 已警示） |
| T195 | 3444964f＋1408277a CUDA graph＋resident main/draft | 依赖 T194；本地图经验（T163）可参考 |
| T196 | 9cdbf221＋5e581e92 a8 MTP experts 转换修复＋frontier bound | 依赖 T195 |
| T197 | 2507d770＋e07307e0＋7131c0b9 MTP 三连 | 依赖 T196 与 T010 结论；本地 MTP 从未在 4090 线跑过（T010） |
| T198 | f8a774f7 应用标定 priorities | 依赖 T197 与 T161；T161 已判定"本轮不并入"，且需与 qwen4exp f8a774f7 同源表共用一次数据整理 |

结论：以上 7 条全部挂起，解锁条件统一为「qwen4exp 工件就绪 + T013 可行性账通过」。

## T199 两 rank 三条 —— 判死（单卡无场景）

- 上游 56a8cf90＋dcc576ba＋27df5a34：两 rank 相关（与 T001 判死池合并，dcc576ba/27df5a34 已在 T001 双计）。
- 本地：单卡 RTX 4090，无 rank 拓扑 → **判死，与 T001 池合并**。
- 判定：判死登记。

## T200 docs 四条 —— 随组登记（T013 已对表）

- 64d10fbf＋30e3a92f＋dc8b58fb＋84d1cf02 的文档结论已在 T013 对表（qwen4_exp-model.md 268 行主取证、转换规格完整转录、HC/QSA/PLE 三新增）。
- 判定：登记完成（T013），不重复搬运。

## T201 组门禁

- **35B-A3B 回归 PPL 逐位**：default 4801 / 2.051275 / 7.777808；冻结 12 GiB 4801 / 2.049253 / 7.762104（最近实测留档，Phase 6 门禁未移动；本批无代码改动）。
- **qwen4exp 端到端**：无工件 → 记录「待工件」，不伪造数据。
- 判定：回归门禁 PASS；qwen4exp 端到端待工件。

## T202 absorb/m5 并主线＋推送

按用户纪律：不建分支、不推送。Phase 7 无代码提交（全部登记/挂起/判死），提交仅本 register 一条；master 线性。

## 结论汇总

| 条 | 判定 | 动作 |
|---|---|---|
| T191 | 待工件（阻塞） | 登记前置链（权重/转换器/可行性账/校验） |
| T192–T198 | 挂起 | 解锁条件=工件就绪+可行性通过 |
| T199 | 判死 | 与 T001 池合并 |
| T200 | 完成 | T013 已对表 |
| T201 | 回归 PASS / qwen4exp 待工件 | 记录 |
| T202 | 不推送 | 登记 |

## 补充核查与新增功能（2026-11-08 18:50）

### 转换器已就位（本地无功能→已增加）
- `tools/convert/qwen4_exp.py`（626 行，upstream d8c0fa6f 的文件）+ `tools/convert/qwen_resources.py`（30 行，其依赖）已移植，`py_compile` 通过，提交 `f377aa93`。
- 运行前置（非新增要求）：依赖 `torch`（与本地既有 qwen3_5.py/methods.py 同一前置，本机 Python 环境未装 torch——转换在任何环境都需此依赖，不是 qwen4 特有）。
- 覆盖：Qwen4ExpForConditionalGeneration / Qwen4ExpForCausalLM 两种架构，含 checkpoint 轴与逻辑参数映射。

### qwen4-27b：与 qwen4-exp 共用转换器（已确认）
- `upstream/feat/qwen4-27b-ops` 分支的 `tools/convert/` 与 `upstream/master` **完全一致**（目录列表相同、qwen4_exp.py blob 相同），即 qwen4-27b 不是独立模型格式，而是 Qwen4Exp 架构的 27B 配置；本地转换器已同时覆盖。
- 该分支 787 文件 / +33445 的差异**全部是 QSA 算子优化**（四头 k8v4/nvfp4/int8/bf16 调优、sparse attention 跨 KV 格式调优、20480 整向量 rmsnorm、kpool fixture），不是模型集成功能。
- 评估结论：qwen4-27b 的"模型转换/注册"功能本地已具备（转换器+QSA 基础算子+GDN+PLE 层）；剩余为**性能优化批次**（可选，不阻塞功能），建议在 qwen4-27b 工件出现后按 T028 纪律分批吸收（优先 20480 rmsnorm 整向量、QSA 四头 bf16/int8 调优）。

### 仍缺的本地功能（需工件）
- 模型主体：text/vision/mtp 执行、程序、参数映射、HF→ninfer 工件注册（T192–T198）——依赖 qwen4exp 工件；PLE 磁盘层（src/models/qwen4_exp/ple/）与 QSA/GDN 算子已就位，工件到位即可接。
