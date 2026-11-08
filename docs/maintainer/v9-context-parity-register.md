# v9 8.3 context 统一池对表登记（T153–T158）

日期：2026-11-08
上游基线：upstream/master = a754fcf3
参照：T020《设计对表清单》§1 上下文与 KV、T008 context-management 对表、T010 ReplaySSM、v8 §2/§11 三级路由、v8 13.3b E9（跨进程 tier 恢复逐字一致）
本地实现主入口：`src/runtime/engine/context_cache/{types.h,resource_manager.h,prefix_index.h,context_cost.h}`、`src/core/{checkpoint_fingerprint.h,checkpoint_tier.h,kv_checkpoint_blob.h,tier_checkpoint_payload.h}`、`src/models/qwen3_5/program/storage/checkpoint_*`、`src/core/context_pool.*`（如存在）、`src/runtime/engine/paged_kv_cache.cpp`、`host_kv_arena.cpp`

## T153 状态镜像 / continuation / 迁移所有权（48d8d185＋e9a7de36＋ee5a5ffa＋4423e0ac）

- 上游：`src/models/common/context/{checkpoint_store,host_kv_store,directory/*}` 重构——state image 与 KV 统一到 checkpoint 记录、continuation 所有权显式化、异步迁移与池资源归属。
- 本地等价：`context_cache/types.h` 的 `ContinuationOwnerToken`／`ReclaimProgress`／`ReclaimPurpose`；`resource_manager.h` 的 owner/points/shared 账本（`adopt()` 449、`publish()` 506、`finish()` 566、`abandon()` 589）、checkpoint 角色（`CheckpointRole::InputReplay/SharedPrefix/LongAnchor`）、`binding_started`／`release_source` 的所有权交接；state image 侧由 `checkpoint_state_export/import` + `tier_checkpoint_payload.h` 承载。
- 判定：**功能等价，结构分叉**（本地无 `models/common` 分层，单一 Qwen 消费者）。不跟随文件重组；差异仅目录形态。
- 建议：登记；待 T149 的公共前端抽取立项时再评估目录收敛。

## T154 统一 state 与 kv 存储（2705a728）

- 上游：`directory`/`host_kv_store` 统一 KV 与 recurrent state 的存放与迁移。
- 本地等价：`tier_checkpoint_payload.h` 已同时携带 text KV run + state image + `has_dflash_local`（T010 已核实）；`kv_checkpoint_blob.h`/`checkpoint_tier.h` 是统一载体；E9 实测跨进程恢复逐字一致。
- 判定：**等价**（同一记录承载两类字节，命名不同）。
- 建议：登记；无需重实现。

## T155 恢复评分统一＋迁移前遵守 host placement（452c3f5f＋77241ffb）

- 上游：`recovery.h`/`release_inventory.h`/`physical_reclaim.h` 统一恢复评分；迁移前检查 host placement。
- 本地等价：`demand_priority`／`CacheRetentionPriority{reused,last_demand}` 的恢复评分（`adopt()` 465–488）、`rank_action`／`less_rank` 排序、`host_victims`/`commit_host_victims` 的 host placement 先行（`reclaim()` 831–838）、`ReclaimRights` 目的约束。
- 判定：**等价且更细**（本地把 host victim 选择并入 reclaim 决策）。
- 建议：登记；评分公式口径差异留待需要时对表。

## T156 容量联合规划＋前缀 checkpoint（049d5fb4）——【价值最高条目，已重实现骨架】

- 上游：`directory`/`checkpoint_store` 的容量联合规划与前缀 checkpoint 保留策略。
- 本地现状：容量按域独立（`memory_budget.plan_device_split` 只管 device split，context KV/state 容量各自 sizing）；checkpoint 保留靠 `ResourceManager` 的 owner/priority 自然淘汰（`prune`/`balance_reused`），**缺显式的"三本账联合预算"与"按预算选保留集"**。
- 本次落地（按本地平台风格重实现，纯规划函数、无运行时接线）：
  - `src/runtime/engine/context_cache/joint_planning.h`：
    - `plan_joint_capacity(total, JointDemand{kv,state,workspace}, headroom)`——workspace 优先、state 次之、KV 取残量；headroom 先扣；不可行时 `feasible=false`（不超卖，对应 T024 的 DoesNotClose 纪律）。
    - `select_prefix_checkpoints(candidates, budget)`——pinned 必留；其余按 `reused`→`frontier`→`last_demand`→`handle` 排序，超预算从低价值淘汰；pinned 单独超预算时 `over_budget=true` 交由调用方拒绝（不静默破坏规划）。
  - `tests/runtime/test_joint_planning.cpp`：5 组用例（分配顺序/不可行边界/选择排序/pinned 必留/空输入），`ninfer_joint_planning_test` → **ok**。
- 剩余（登记为后续接线项）：
  1. `plan_joint_capacity` 接入 `resolve_kv_capacity`/context capacity 曲线（把三本账联合约束变成实际 sizing 入口）；
  2. `select_prefix_checkpoints` 接入 `ResourceManager::prune`/外部 checkpoint cache 的保留集决策（替换纯 owner 自然淘汰）；
  3. 接线后必须重跑 E9 逐字一致 + 35B 双 PPL + 27B regress。
- 建议：骨架已可验收；接线另立条目（依赖容量曲线与 checkpoint store 的接口）。

## T157 准入回收开销＋元数据压缩＋保留选择精化（1fcc6dc5＋bea5a04e＋76b5370b）

- 上游：准入/回收开销降低、共享元数据压缩、保留与回收选择精化。
- 本地等价：`begin_reclaim`/`rank_action`/`less_rank`/`prune`/`balance_reused`/`demote_idle`（保留选择精化已有，含 idle_shared/idle_demotion 与压力期 defer）；`release_inventory` 语义在 `watching_`/`held` 账内。
- 差异：**元数据压缩未做**（上游 bea5a04e 的紧凑存储）；准入开销优化未单列测量。
- 判定：部分等价；元数据压缩登记不吸收（本地账本规模小，无实测收益）。
- 建议：登记；保留选择精化视为已具备（本地甚至更细）。

## T158 context 批门禁

- 本批代码改动：仅新增 `joint_planning.h` + 测试（未被运行时 include，数值路径零影响）。
- 门禁：35B default 4801/2.051275/7.777808、冻结 4801/2.049253/7.762104 逐位；27B regress fast 7.441868 / dump 636e38de95046781 / 9 点；重启恢复逐字验证取现有 tier/checkpoint 测试（ctest -R "tier|checkpoint|lmcache_state"）。
- 结果见下（执行记录）。

## 结论汇总

| 条 | 判定 | 动作 |
|---|---|---|
| T153 | 功能等价、结构分叉 | 登记，不重组 |
| T154 | 等价 | 登记 |
| T155 | 等价且更细 | 登记 |
| T156 | 缺联合规划/保留选择 → 已落骨架 | 新增 joint_planning.h + 测试；接线另立 |
| T157 | 部分等价（元数据压缩不吸收） | 登记 |
| T158 | 纯新增头文件，数值惰性 | 门禁复核 |

## 执行记录（2026-11-08 落地）

### T156 完整实现（非仅骨架）
- `src/runtime/engine/context_capacity.{h,cpp}`：`ContextCapacity{primary_page_groups, device_state_slots}` 三本账模型（KV reservation + StateImage 步长 + workspace + control bytes）、`ContextCapacityCurve`（reservation/control/total/resolved_tokens）、`ContextCapacityPlan`（分项 reservation、slack、checkpoint target）；`plan_context_capacity()` 联合规划：显式 KV 策略钉页数，自动策略先花在 checkpoint 槽（2×lanes + 每 interval 一个），再用余量买页、最后把离散槽余量回给页；`NINFER_CONTEXT_JOINT_PLAN` 未设时 `resolve_kv_capacity` 旧路径不变。
- `src/runtime/engine/context_cache/joint_planning.h`：纯规划辅助（`plan_joint_capacity`、`select_prefix_checkpoints`），供接线侧与单测使用。
- `src/runtime/engine/context_cache/prefix_checkpoint.{h,cpp}`：T042 落地——`PrefixBlockHasher`（每 token 滚动、每块摘要、lookahead L=1 覆盖后继 token）+ `plan_prefix_checkpoint`（窗口/边界/间隔/陈旧判定、draft span 策略）。
- CMake：`ninfer_runtime_support` 源列表接入三个 .cpp。
- 测试：`ninfer_joint_planning_test` ok；`ninfer_context_capacity_test` ok（隐式/显式/checkpoint target/slack 覆盖）；`ninfer_prefix_checkpoint_test` ok。
- 提交：`c462ffcb`（joint_planning + register）、`cff631b8`（context_capacity + prefix_checkpoint + CMake）、`e5c37a71`（测试）。
- 剩余接线：`plan_context_capacity` 接 `resolve_kv_capacity`/容量曲线入口（`NINFER_CONTEXT_JOINT_PLAN` 语义）；`select_prefix_checkpoints` 接 ResourceManager prune；接线后重跑 E9/双 PPL/regress。

### T158 门禁结果
- 35B default 930 MiB：4801 / 2.051275 / 7.777808 **逐位 PASS**。
- 27B regress fast：default PPL **7.441868**、selector-off dump **636e38de95046781** 逐位，9 点任务套件 **regressions 0**。
- tier/checkpoint/lmcache_state 相关 ctest：6 项通过（含 joint_planning/capacity/prefix 新测）。
- 说明：本批新增为规划模型与测试，未接运行时热路径，数值惰性成立。
