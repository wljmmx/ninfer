# CPU 路径 / 预取 / 容量规划 跟踪台账

> 用途：把散落在多处结论里的待办，收成一张可逐项推进、可逐项关账的表。
> 规则：每一项的"状态"只能由"证据"列里的数字或行号支撑；没有证据的写 `❓未验证`，
> 不要写"应该"、"大概"。tok/s 不作为证据（bench 双模态：同一二进制 8.7/7.3/4.3/3.3 都出现过）。
> 最后核对：2026-11-16（本机 RTX 4090 / sm_89 / Win11 / CUDA 13.5）。

## 0. 状态图例

| 记号 | 含义 |
|---|---|
| ✅ | 已验：有本机实测数字或代码行号，默认路径不受影响 |
| 🟡 | 部分：机制在，但缺消费者/缺测量/默认关闭 |
| ⬜ | 未做 |
| ❓ | 未验证：本轮没有证据，且写清"用什么能验" |
| N/A | 本机不可达（硬件或池几何决定），不是待办 |

---

## 1. 2×2 矩阵：预测器 / 端口 / 命中率可读

判据来源（代码）：
- 流式 vs 驻留：`ffn.cpp:1544` `resident = !streaming || moe.stream->fully_resident()`
- 驻留判据：`expert_stream.cpp` `fully_resident()` = **每个 run 的 `resident_capacity(bank) >= count_`**
- 预测器入口：`ffn.cpp:957 predict_next_layer()`；split 路径在 `ffn.cpp:2004`（`set_predicted`）、
  融合路径在 `ffn.cpp:2576/2658/2689`；`prev` 信号在 `text.cpp:2504 stage_history()`
- 端口：`ffn.cpp:1300` 附近的 `[ctl] prefetch port: ... -> N slot(s)/bank (reason)`

| 格 | 预测器 | 端口 | 命中率可读 | 本机可达 |
|---|---|---|---|---|
| **e2 × 流式** | 🟡 有信号无放置件：`[pf-pred] written=64 ids=640 drained=64 dropped=0`（环完整），但 `[pf-region] hits=0 calls=640 (0.0%)`；`[pf-ctl] cutoff=0 armed=N reason=depth`（8 槽）/ `armed=Y`（16 槽） | ✅ `[ctl] prefetch port` 正常打印 cold/hit/cpu/runs/banks/slots→N | 🟡 计数器可读（`hits/calls`），但恒 0-0.2%，**语义上等于不可读** | ✅ 可达 |
| **e2 × 驻留** | — 无需预测（整层驻留） | ✅ 同一端口 | — | **N/A（硬件）**：`count_` = 73728/144 = 512 专家/bank；整层驻留需 73728 slot × 1.25 MiB ≈ 90,000 MiB ≈ 88 GiB，24 GB 卡不可达 |
| **非 e2 × 流式** | ⬜ "层元数据"信号没有生产：`predict_next_layer` 只有 residual(28.8%) / prev(49.4%) 两条 | ✅ | 🟡 同左（无填充→0） | ❓ 本轮未跑非 e2 制品（`R:\model\qwen3_6_35b_a3b`）；验法：同命令换权重，看 `[pf-pred]` 是否有 ids |
| **非 e2 × 驻留** | ⬜ 双缓冲换层未实现；"换层窗口内无 miss"这个前提本身未验证 | ✅ | — | ❓ 验法：35B 权重 + `[gh1] no_miss_pct` 是否 100%，否则这一格的前提不成立 |

**结论一句话**：四格里只有 1 格在本机可达且有实测；1 格是硬件不可达；2 格没有验证过（不是"未实现"就等于"不可能"）。

---

## 2. 工作项（按依赖排序）

### P0 场地可用（阻塞一切，必须先做）

| ID | 任务 | 验收 | 状态 | 证据 |
|---|---|---|---|---|
| P0.1 | 竞技场改成"分配成功多少就按多少定槽数"，删掉固定的 21 GiB 上限（`model_instance.cpp` `kMappedArenaCeiling`） | 默认门禁连跑 3 次都过 `6.024313` / `20.673595` | ✅ | 分配在前、定尺寸在后：`allocate_host_arena()` 按 GiB 步退让，块字节数交给 tier，slot 由块推。默认 3×2 全逐位（`6.024313` ×3、`20.673595` ×3，exit=0），竞技场 22093–22200 MiB 漂移不影响数值；回执 `[host] arena allocated 22100 MiB (asked 22100, 0 refusals) device_accessible=1 - the tier is sized from this`；显式 `NINFER_EXPERT_HOST_ARENA_MB=16384` → 13107 slots，仍 `6.024313` |
| P0.2 | 若单块映射不可行：分块注册（每块 `cudaHostRegister` + 设备地址表），把"块"变成 tier 的显式概念 | 同上，且 20 GiB 级竞技场可用 | N/A | 不需要（P0.1 已达标）：单块 `cudaHostAllocMapped` 在 22093–22200 MiB 上 6/6 成功（`[host] arena allocated 22100 MiB (asked 22100, 0 refusals)`），未触发过退让；分块注册与设备地址表按任务约定不动 |
| P0.3 | 分配失败在分配点就 fail-loud（现在会跑到 `qsa/prepare_launch.cu:333` 伪装成 kernel bug） | 日志首行指名 arena 与尺寸 | ✅ | `NINFER_EXPERT_HOST_ARENA_MB=60000` → 首行 `[host] arena allocation failed: asked 60000 MiB, refused at every step down to 60000 MiB after 1 attempt(s) (out of memory) - the host expert tier cannot exist, and there is no pageable stand-in for it`，exit=1，13 行日志里无 `prepare_launch.cu:333`；`arena()` 里 registered/pageable 兜底全部删除 |

### P1 容量规划对齐规格

规格：`device: expert_pool_mb / kv_pages_mb / prefetch_mb / slru_band_mb`、
`host: expert_arena_mb / kv_host_mb / prefetch_readahead_mb / second_chance_mb`、
`ssd: 无专家预算；kv_spill = {enabled, 上限, 重读计数}`、`cold_range = 总专家 − 池 − 暖区`（观测值）。

| ID | 任务 | 验收 | 状态 | 证据 |
|---|---|---|---|---|
| P1.1 | 列名/列序改成 `*_mb` 与规格一致（`prefetch_readahead`） | `[plan]` 三行与规格逐列对应 | ✅ | 现为 device `expert_pool_mb \| kv_pages_mb \| prefetch_mb \| slru_band_mb \| staging_mb \| claimed`；host `expert_arena_mb \| kv_host_mb \| prefetch_readahead_mb \| second_chance_mb \| claimed`（主机列序按规格）；默认门禁仍 `6.024313` / `20.673595` |
| P1.2 | **prefetch 未接通强制 0** 落成规则（消费者未接通 → 恒 0 并在行内标注原因） | 默认 `[plan] prefetch=0*` 且 `[pf-region]` 不再被 carve | ✅ | 规则在 `core::device_prefetch_consumer_present()`（恒 false，依据 `[pf-pred] Device form: ... fill=host-only (0 placed by the Device)`），`TierPlan::set` 对 `prefetch_mb` 强制 0；行内原因 `prefetch_mb=0*(no consumer: device-side placement absent ...)`；`NINFER_PREFETCH_SLOTS=16` 单独设置 → exit=1 `cannot arm the Device prefetch region: it has no consumer`；默认门禁 `6.024313` / `20.673595` |
| P1.3 | `kv_spill = {enabled, 上限, 重读计数}` | `[plan] ssd` 行给出三元组 | ✅ | `[plan] ssd cap=none \| kv_spill={enabled=0, cap=none, reread=none}`；`enabled` 取唯一创建该 tier 的开关（`program_impl.cpp` 只在 `NINFER_KV_SPILL_MB` 设置时挂文件），`cap` 无文件即 none、有文件时给文件 extent 容量并标"not a plan claim"，`reread` 写 none 并说明树里只有 `kv_tier_stats()` 的 spilled/fetched 页（另一个问题），不编 0 |
| P1.4 | `cold_range` 进表 | `[plan]` 出现 `cold_range=NN`（= 总专家 − 池 − 暖区） | ✅ | `[plan] ssd ... \| cold_range=38415 experts (73728 - 18000 - 17313)`；三项同取 `attach_expert_cache` 的 placement（与 `[budget] experts: ... unslotted=55728 of 73728` 同源），故 `[plan]` 报告点移到 placement 定稿之后 |

### P2 预取（先定生死，再谈优化）

| ID | 任务 | 验收 | 状态 | 证据 |
|---|---|---|---|---|
| P2.1 | 按 P1.2 强制 0，删掉"env 能武装一个没消费者的区" | 默认 0 槽，无 `[pf-region]` carve 行 | ⬜ | 现在 `NINFER_PREFETCH_SLOTS=16` 仍能武装 |
| P2.2 | device 端放置件（原 S4/S5）：放置状态机上 device，填充不回 host | `hits/calls` 从 0 抬起；`[pf-mirror] region uploads` 随 `prefetch_generation` 收敛 | ⬜ | `[pf-region] hits=0/640`；`region uploads=144`（仅镜像） |
| P2.3 | 形态与策略解耦：`NINFER_PREFETCH_MODE=vram` 与武装分离会漏（有槽无填充看起来像"没有调用者"） | 不设 MODE 时 `[pf-ctl]` 仍说清 armed | 🟡 | `[pfv] enter #1 bank=3 vram=0` → `skip: mode is not vram` |
| P2.4 | 复活判据重测：`prev` ≥66%（现 49.4%）、每节点 ≤0.0117 s（现 0.0162 s） | 达标则开，否则写"永久关闭" | ⬜ | 16 槽武装：命中 0.2%、速度 −40% |

### P3 CPU 半定价（决策 2/3）

| ID | 任务 | 验收 | 状态 | 证据 |
|---|---|---|---|---|
| P3.1 | 统一 scope：copy 侧也记 tier 读，或从 span 减掉读 | 冷 tier 下 `est copy` 与 `span` 同量级，`m*` 回到 ~4.8 | ⬜ | span 8186 us/路由 vs copy 12.6 us/路由（4 路由层） |
| P3.2 | 只有 P3.1 之后才允许实测速率接回决策 | `leg=half` 比例不随冷热翻转 | 🟡 | 已接、默认 `NINFER_CPU_LEG_ACT=off`；`leg=half` 1389→5 |
| P3.3 | 面板可达：`[ctl] cpu share` / `[ctl] cpu leg` 在 e2 路径一次都没打出来 | `NINFER_SPLIT_TIMING=1` 下 e2 路径可见 | ⬜ | 816 行 `[cpu-path]` vs 0 行 `[ctl] fetch=/cpu share` |

### P4 决策 1（KV split）

| ID | 任务 | 验收 | 状态 | 证据 |
|---|---|---|---|---|
| P4.1 | E8 与 parallel 的 gate 调用点（`launch.cu` `rk_e8_grouped:122` / `rk_e8v2_grouped:149` / `rk_root_grouped:175` / `rk_parallel_grouped:390`） | E8 模型开 `NINFER_KV_CPU_PARTITION` 真进入 CPU 腿 | ⬜ | 5 族 dtype 已全部可命名（`eb1a6faa`），但 3 族无消费者 → "允许但不可用" |
| P4.2 | `make_int8_kv_causal_plan` 的 partition/key_tile 透出给 models | 决策 1 能自称选择者，knob 降级为覆盖 | ⬜ | 几何只在 ops dispatch；决策 1 仍零调用 |
| P4.3 | 真宿主驻留页（demote 后注意力直接回读） | 镜像不再依赖 D2H；每层少一次流同步 | ⬜ | 现在是测试性归属 + 每层 `cudaStreamSynchronize` |
| P4.4 | 记录"恒为 DeviceCopy"的量化结论 | 文档写明：33,792 B ≈1.35 µs vs CPU 275 µs；链路需 <0.12 GB/s 才翻 | ⬜ | 已算，未落文 |

### P5 测试与清理

| ID | 任务 | 验收 | 状态 |
|---|---|---|---|
| P5.1 | `ctest` 全量跑一次（注册 208，`build-win` 只有部分可执行） | 给出通过/失败清单 | ⬜ |
| P5.2 | `diag-cpu-partial` A/B 扩到 5 族 | 每族 engage + 偏差 | ⬜ |
| P5.3 | `host_pages.h` 的 `packed_k` 已无生产消费者 | 删除或标注"仅测试用" | ⬜ |

---

## 3. 可独立关账的 7 项（做完即划掉，互不依赖）

1. P1.1 列名对齐 `*_mb`
2. P1.4 `cold_range` 进表（取 `unslotted`）
3. P1.3 `kv_spill` 三元组
4. P1.2/P2.1 prefetch 强制 0
5. P4.1 E8/parallel 的 4 处 gate 调用点
6. P3.3 `[ctl] cpu` 面板在 e2 路径可达
7. P5.3 `packed_k` 清理

## 4. 口径与陷阱（写下来，避免下一轮再踩）

1. **竞技场不是计算**：16384 / 18432 / 21504 / 22500 MiB 都给 `6.024313`，所以它的尺寸只影响命中，不影响数值。
2. **映射悬崖会漂**：同一天从 ~22 GiB 漂到 ~20 GiB，固定上限不可依赖；且"先探测后分配"不成立（探测 22528 成功、真分配 22527 失败，探测本身还花 3.7 s）。
3. **tok/s 双模态**：臂内散布可大于臂间差；速度结论必须 min-of-N + 同会话交错，且不能单独作回归证据。
4. **span 与 copy 不同口径**：span 含 tier fill，copy 侧把这次读约掉了；不统一 scope 之前，任何"用实测速率定价"的结论都不成立。
5. **`[ctl]` 面板在 e2 路径不打印**：`report()` 只在 `streamed_ffn` 被调用，看到 0 行不等于"值为 0"。
6. **设备别名早一层就换数值**：`arena_device_base()` 非空决定 split 路径走"Device 直读 host tier"还是第一层走 CPU 腿。块预先分配但别名在构造时就发布，ppl114 给 `6.017490`（17203 slots 与基线同）；别名等到第一次 `arena()` 再发布，回到 `6.024313`。所以"竞技场尺寸不影响数值"成立的前提是**别名的可见时机也不变**。

## 5. 变更记录

| 日期 | 事件 |
|---|---|
| 2026-11-16 | 建表。矩阵：1 格实测、1 格 N/A（硬件）、2 格待验。P0 由当晚 `mapped allocation of 20941 MiB failed` 升级为最高优先级 |
| 2026-11-16 | P0.1 关账（`cbc2ad49`）：分配在前、定尺寸在后，默认 3×2 门禁逐位。P0.2 记 N/A（单块在 22 GiB 上 6/6 成功）。P0.3 关账：失败在分配点抛出，pageable/registered 兜底删除 |
| 2026-11-16 | P1 四项关账：P1.1 列名/列序（`aea2b93c`）、P1.2 无消费者强制 0 + env fail-loud（`b9f557e9`）、P1.3 `kv_spill` 三元组（`54568858`）、P1.4 `cold_range` 进表。`[plan]` 三行逐列对齐规格，跨架构不变（只读 `Instance` 几何与 placement） |
