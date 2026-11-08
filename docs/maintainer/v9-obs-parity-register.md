# v9 8.5 obs 口径对表与 Phase 6 收口登记（T165–T168）

日期：2026-11-08
上游基线：upstream/master = a754fcf3
参照：T009 observability 对表（TTFT 三口径、GPU/Host 不相加、面板 Mock 历史问题）、T126 monitor 落地、v8 §7 测量纪律、T128 长负载留档、T159–T164 面板信号

## T165 请求 vs 缓存上下文区分＋统计统一（c5627cdd＋697979a4）

- 上游：`observability/resources.h` + `models/common/context` 区分「请求侧上下文用量」与「缓存侧上下文占用」；`apps/*` 统一引擎统计与报告输出。
- 本地等价：
  - 请求侧：`src/serve/observations.{h,cpp}` 的 `ServiceObservations`（submitted/completed/failed/cancelled、ttft/duration/queue、requests 列表）——T126 落地并经真机验证（ttft 278.06ms vs 引擎自报 277.53ms）。
  - 缓存侧：`/monitor/snapshot` 的 `engine.placements`/`ranks[*].allocations`（capacity/used/peak）与 `engine.context`（pools/resources）——请求与缓存两本账在字段层面已分开。
  - 统计统一：本地 `apps/cli/main.cpp` 的 summary（sampling/prompt tokens/thinking/elapsed）与 `perplexity` 独立报告器各自成型；`request_log`/`operational_log` 有 schema 版本（26）与随请求字段。
- 判定：**等价（区分已具备），统计统一为部分等价**——本地多入口各有报告器，上游的「统一引擎统计与报告」可借其字段命名与一次性快照语义，但不必合并入口（会打乱既有门禁脚本对输出格式的依赖）。
- 建议：登记；若后续要统一，按「字段清单对齐 + 输出格式冻结」小步做，不重排 apps 结构。
- 注：T009 已纠正的口径仍然有效——TTFT 用首个已提交 token、GPU/Host 不相加、缺失量不填 0。

## T166 上下文 token 容量估计（e5e18b97）

- 上游：`observability/snapshot.h` 增「估计上下文 token 容量」；planning 侧（qwen3_5/qwen4_exp/glm5next）+ `resources.h` + `context_capacity.cpp` 提供来源。
- 本地：T156 的 `ContextCapacityPlan` 已含 `resolved_tokens`（由联合规划算出的上下文 token 容量）与 `maximum_page_groups`；`ContextCapacityCurve::resolved_tokens(pages)` 亦提供换算。`/monitor/snapshot` 的 `engine.context` 可承载该字段（当前未填，缺失而非 0）。
- 判定：**能力已具备（resolved_tokens），展示接线未做**——把 `ContextCapacityPlan.resolved_tokens` 投到 monitor snapshot 的 context 段即可，属小改。
- 建议：登记为「展示接线」小项（随下一次 monitor 字段扩展一起做）；口径沿用 T156 的联合规划结果，不重复推导。

## T167 Phase 6 汇总门禁

| 项 | 数据 | 结论 |
|---|---|---|
| 全量重编 | 最近一次全量构建（T153–T158 后）零错；8.4/8.5 纯对表无代码 diff | PASS |
| 35B default 930 MiB | 4801 / 2.051275 / 7.777808（T153–T158 门禁实测） | 逐位 PASS |
| 35B 冻结 12 GiB | 4801 / 2.049253 / 7.762104（T153–T158 门禁实测 + T159–T164 引用） | 逐位 PASS |
| 27B fast | default PPL 7.441868、selector-off dump 636e38de95046781、9 点 0 回归（T153–T158 门禁实测） | PASS |
| 面板信号清单 | [freeze] budget per bank / [experts] banks=40 pool / [ledger] vram+ram / [warm] striped arena / [budget] experts / [layer-graph] enabled=1（T164 实测） | 齐全 |
| 长负载 | pp868+tg512 decode 44.17 t/s（min 44.45）、pp2048+tg1024 41.28 t/s（min 41.35）（T128 留档） | 有记录 |
| 守卫 | check_blackwell_guards.py 1255+ 文件 0 违规 | PASS |

说明：Phase 6 为语义对表批（T145–T166），除 T156 的联合规划实现（已单独门禁）外无运行时改动；门禁数据以最近实测留档为准，未因对表而移动。

## T168 absorb/m4 并主线＋推送

按用户纪律：不建分支、不推送——Phase 6 全部提交已在 master 线性（T145–T152 `74d6d1d0`/`2c31926c`、T153–T158 `c462ffcb`/`cff631b8`/`e5c37a71`/`9c86f542`、T159–T164 `60fe162d`、本批 register 提交），提交标题带 upstream 哈希；推送待用户收口指令。

## 结论汇总

| 条 | 判定 | 动作 |
|---|---|---|
| T165 | 区分已具备；统计统一部分等价 | 登记，不合并入口 |
| T166 | 能力已具备（resolved_tokens），展示接线未做 | 登记为小项 |
| T167 | 汇总门禁通过 | 引用实测留档 |
| T168 | 按用户纪律不推送 | 登记 |
