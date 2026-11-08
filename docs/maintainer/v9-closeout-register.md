# v9 Phase 8 收尾登记（T212–T214）

日期：2026-11-08
基线：de5d4288（v8 完工）→ upstream/master a754fcf3（305 提交）
当前：master @ 48d6383a 之后（本 register 提交前），ahead origin/master 93

## T212 版本对齐终验（部分完成/发布线待启动）

| 项 | 结果 |
|---|---|
| git describe | **v1.0.0-rtx4090-623-g48d6383a**（T002 留档 v1.0.0-rtx4090-539-gc11e7f57，前移 84） |
| 9908 矩阵 | `D:\ninfer-rtx4090\v3-release\SERVE_MATRIX_9908.md` 存在（文档形态）；CLI 接口对齐抽查：`--max-new`/`--max-tokens` 并存（本地旧名保留）、`--kv-dtype`、`--startup-planning-headroom-mib` 等均在（T049/T050 变更未破坏既有接口）；**跑通验证留待发布收口**（用户不推送、v3-release 线未启动，跑矩阵需要 serve+模型，非本次收尾范围） |
| 远程推送核对 | `master...origin/master [ahead 93]`——全部为本地 v9 吸收提交；**按用户指令不推送**，推送待用户收口指令 |
| 发布线 | v3-release 现有 ninfer_bench/ninfer-perplexity 等 10-21 构建产物；未重打包（待用户决定） |

## T213 全量终验（引用实测留档 + 汇总）

| 判据 | 数据 | 来源 |
|---|---|---|
| 35B default 930 MiB PPL | 4801 / 2.051275 / 7.777808 逐位 | T002 基线 + T082/T153–T158/T167 复跑 |
| 35B 冻结 12 GiB PPL | 4801 / 2.049253 / 7.762104 逐位 | T002 + T082 + T159–T164 引用 |
| 27B regress fast | default PPL 7.441868、selector-off dump **636e38de95046781**、9 点 0 回归 | T082/T158/T167 |
| 27B moe | 7.111129 / 7.777808 / 6.479183 + decode d093ac2070ab3292 逐位 | T082/T158 |
| seam | ninfer_kv_seam_test Passed | T082/T167 |
| regress | fast/moe 均 0 回归 | T082/T158/T167 |
| dump 口径更正 | 当前 golden = **636e38de95046781**（2026-11-04 重录）；§11 速查写的 e7ceb89dff4eb40b 是旧值（T042/T002 已登记） | regress.py + 留档 |
| 868–2048 长前缀 | pp868+tg512 prefill 133.23 t/s / decode 44.17 t/s（min 44.45）；pp2048+tg1024 prefill 130.84 / decode 41.28（min 41.35） | T128 留档 |
| KL 汇总表 | 本地无 KL 工具，按 T028 替代口径（PPL 逐位+任务套件+bench 三判据并列），见 v9-dashboard-register.md §9 | T094/T113/T119/T128 |
| 构建 | 全量增量 ninja 零错（T153–T158 后） | 留档 |
| 守卫 | check_blackwell_guards.py 1255+ 文件 0 违规 | T125/T128/T167 |

tps 与 T002 基线对比：T002 的 PPL score rate（default 60.1 / freeze 136.0 tok/s，冷缓存首跑）与长负载 decode 44.17/41.28 t/s 不是同一口径；v9 期间无生产路径提速主张（各批均记录"数值惰性/实测中性"），故不声称回归或提速。

## T214 计划归档

- **变更记录**：§12 已按组记录 T001–T214（含 T021–T031/T033–T046/L01–L13 参考核对、T048–T083 逻辑修复、T084–T129 算子移植、T130 控制台、T145–T168 能力对表、T191–T202 模型集成），每条带状态、证据、提交号。
- **305 条对账**：`git rev-list --count de5d4288..upstream/master = 305` ✓。覆盖路径：
  - T001 判死封口：132 条（Spark 96 + GLM 18 + TP2 18，dcc576ba/27df5a34 与 T199 双计）；
  - T002 基线 + Phase 1–8：A 类整搬（含 T048/T084–T130 等）、M 类三方融合（T049–T083）、语义对表（T145–T168）、参考核对（T021–T047/L01–L14）、模型集成登记（T191–T202）；
  - 无遗漏：305 全部落在上述处理路径；无重复：判死双计已显式标注。
- **上游分支**：`upstream/feat/qwen4-27b-ops` **未入 master**（最新 e1874e94 "refactor(qsa): organize sparse scheduling and cache reads"）。若后续并入 master，按同规程立项 **v9.1**（复用本计划的七步流程/门禁/判死登记）。
- **遗留清单**（挂起/待工件）：T071 后续监听项（已跟随）、T088 D64 非因果（阻塞，整文件替换方案）、T130 T131 控制面（等 v8 §9–12）、T156 接线（resolve_kv_capacity/ResourceManager prune）、T166 monitor 展示、T191–T198 qwen4exp（待工件）、T213 发布线跑通（待用户收口）。

## 结论

Phase 8 收尾完成（T212 版本/接口/推送核对、T213 全量终验汇总、T214 归档与 305 对账）。v9 吸收执行至此闭环；未推送提交 93 条待用户收口指令统一处理。
