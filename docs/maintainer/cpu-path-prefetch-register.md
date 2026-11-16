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
| **e2 × 驻留** | — 无需预测（整层驻留） | ✅ 同一端口 | — | **本机不可达，两个独立原因**：① 125B 上 `count_` = 512 专家/bank，整层驻留需 73728 slot × 1.25 MiB ≈ 88 GiB；② 35B-e2 现在能跑了（P6.1 已修，`overall 6.701375`），但池只装 **10040/10240** 专家（池 18879 MiB，每 bank 差 5 槽 × 1920 KiB = 375 MiB），`fully_resident()` 仍为 false |
| **非 e2 × 流式** | ⬜ "层元数据"信号没有生产：`predict_next_layer` 只有 residual(28.8%) / prev(49.4%) 两条 | ✅ | 🟡 同左（无填充→0） | ❓ 35B plain 能跑（`overall 6.701375`），但它无池行、未确认有专家路由；验法：同命令加 `[pf-pred]`/`[gh1]` 观察是否有 ids |
| **非 e2 × 驻留** | ⬜ 双缓冲换层未实现；"换层窗口内无 miss"这个前提本身未验证 | ✅ | — | ❓ 验法：35B plain + `[gh1] no_miss_pct` 是否 100%，否则这一格的前提不成立 |

**结论一句话**：四格里只有 1 格在本机可达且有实测；1 格是硬件不可达；2 格没有验证过（不是"未实现"就等于"不可能"）。

---

## 1.5 工作目录约定（2026-11-16 立，避免再出现"找不到文件"）

- 仓库唯一入口：`H:\TeleAgentWorks\.temp\r` —— 它是 **junction**，指向 `H:\TeleAgentWorks\ninfer-4090-port\infer`。
  两条路径是同一个仓库；**命令与编辑一律用 junction 路径**，因为原始路径在卷视图里会间歇性只显示成 `.temp`（Storage Space 池的已知现象：`Test-Path .git` 为真而 `dir` 里没有）。
- 命名别混：项目叫 **ninfer**，仓库目录名是 **infer**，上层目录是 **ninfer-4090-port**。写路径逐段核对，别漏 `n`。
- 临时产物**只放一处**：`H:\TeleAgentWorks\.temp\runs\<YYYY-MM-DD>\<项名>\`。不要再往 `.temp` 根或别处摊 `out-*` / `gate*` / `p*.log`。
- 当日已清理：`.temp` 根下 **796** 个散落条目移入 `runs\2026-11-16\`；一次性探针可执行与 70 MB 级 `.obj` 已删除；根目录现在只剩原有工具/语料与 `runs\`。

---

## 2. 工作项（按依赖排序）

### P0 场地可用（阻塞一切，必须先做）

| ID | 任务 | 验收 | 状态 | 证据 |
|---|---|---|---|---|
| P0.1 | 竞技场改成"分配成功多少就按多少定槽数"，删掉固定的 21 GiB 上限（`model_instance.cpp` `kMappedArenaCeiling`） | 默认门禁连跑 3 次都过 `6.024313` / `20.673595` | ✅ | 分配在前、定尺寸在后：`allocate_host_arena()` 先按 commit 余额夹取候选，一次 `cudaHostAlloc(Mapped)` 成功多少就是多少，块字节数交给 tier，slot 由块推。默认 3×2 全逐位（`6.024313` ×3、`20.673595` ×3，exit=0）；回执 `[host] arena allocated 22351 MiB (asked 22351, commit headroom 23449) device_accessible=1 - the tier is sized from this`；夹取实例：`NINFER_PREFETCH=1 NINFER_PREFETCH_SLOTS=8192`（候选 32657）→ `allocated 23385 MiB (asked 32657, commit headroom 23385)`；显式 `NINFER_EXPERT_HOST_ARENA_MB=16384` → 13107 slots，仍 `6.024313` |
| P0.2 | 若单块映射不可行：分块注册（每块 `cudaHostRegister` + 设备地址表），把"块"变成 tier 的显式概念 | 同上，且 20 GiB 级竞技场可用 | N/A | 不需要（P0.1 已达标）：单块 `cudaHostAllocMapped` 在默认 22331–22538 MiB 上 6/6 成功（`[host] arena allocated 22538 MiB (asked 22538, commit headroom 23325)`），退让与分块都未需要；且探针显示失败后 VirtualAlloc 也拿不到 commit（1455），分块不解决根因（见陷阱 7） |
| P0.3 | 分配失败在分配点就 fail-loud（现在会跑到 `qsa/prepare_launch.cu:333` 伪装成 kernel bug） | 日志首行指名 arena 与尺寸 | ✅ | `NINFER_EXPERT_HOST_ARENA_MB=60000` → `[host] arena allocation failed: asked 60000 MiB (committed ask 60000 MiB, headroom 23408 MiB) (out of memory) - the host expert tier cannot exist, there is no pageable stand-in for it, and no smaller retry after a refusal`，exit=1，13 行日志里无 `prepare_launch.cu:333`；`arena()` 里 registered/pageable 兜底全部删除 |

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
| P2.1 | 按 P1.2 强制 0，删掉"env 能武装一个没消费者的区" | 默认 0 槽，无 `[pf-region]` carve 行 | ✅ | 同 P1.2：`device_prefetch_consumer_present()` 恒 false，`TierPlan::set` 强制 0；`NINFER_PREFETCH_SLOTS=16` 单独设置 → exit=1 `cannot arm the Device prefetch region: it has no consumer`，不再静默武装 |
| P2.2 | device 端放置件（原 S4/S5）：放置状态机上 device，填充不回 host | `hits/calls` 从 0 抬起；`[pf-mirror] region uploads` 随 `prefetch_generation` 收敛 | ⬜ | `[pf-region] hits=0/640`；`region uploads=144`（仅镜像） |
| P2.3 | 形态与策略解耦：`NINFER_PREFETCH_MODE=vram` 与武装分离会漏（有槽无填充看起来像"没有调用者"） | 不设 MODE 时 `[pf-ctl]` 仍说清 armed | 🟡 | `[pfv] enter #1 bank=3 vram=0` → `skip: mode is not vram` |
| P2.4 | 复活判据重测：`prev` ≥66%（现 49.4%）、每节点 ≤0.0117 s（现 0.0162 s） | 达标则开，否则写"永久关闭" | ⬜ | 16 槽武装：命中 0.2%、速度 −40% |

### P3 CPU 半定价（决策 2/3）

| ID | 任务 | 验收 | 状态 | 证据 |
|---|---|---|---|---|
| P3.1 | 统一 scope：copy 侧也记 tier 读，或从 span 减掉读 | 冷 tier 下 `est copy` 与 `span` 同量级，`m*` 回到 ~4.8 | ⬜ | span 8186 us/路由 vs copy 12.6 us/路由（4 路由层） |
| P3.2 | 只有 P3.1 之后才允许实测速率接回决策 | `leg=half` 比例不随冷热翻转 | 🟡 | 已接、默认 `NINFER_CPU_LEG_ACT=off`；`leg=half` 1389→5 |
| P3.3 | 面板可达：`[ctl] cpu share` / `[ctl] cpu leg` 在 e2 路径一次都没打出来 | `NINFER_SPLIT_TIMING=1` 下 e2 路径可见 | ✅ | 面板在 split 入口可达：`[ctl] cpu share: routes cpu=532/13818 (3.9%) last=0 experts \| rate span=440.1us@1 compute=236.2us@1 ...`、`[ctl] cpu leg: layer-level=0/0 (half=0 copy=0 nodata=0) \| acted=0 layer(s)/0 route(s) \| last m=0 m*=-1.0`；同一次 e2 运行 `NINFER_CPU_PATH_POLICY=on NINFER_SPLIT_TIMING=1` 下 `[ctl] fetch=` 20 行 / `[ctl] cpu share` 20 行 / `[ctl] cpu leg` 20 行 对 816 行 `[cpu-path]`（此前 0 行）；`PolicyController::layers` 现在每条 FFN 路径每层恰好自增一次（融合走 `step()`、split 走 `note_layer()`），默认门禁 `6.024313` / `20.673595` 且 0 行面板（只剩载入期的 `[ctl] prefetch:` / `[ctl] host split:` 两行） |

### P4 决策 1（KV split）

| ID | 任务 | 验收 | 状态 | 证据 |
|---|---|---|---|---|
| P4.1 | E8 与 parallel 的 gate 调用点（`launch.cu` `rk_e8_grouped:122` / `rk_e8v2_grouped:149` / `rk_root_grouped:175` / `rk_parallel_grouped:390`） | E8 模型开 `NINFER_KV_CPU_PARTITION` 真进入 CPU 腿 | ✅ | 4 处调用点已加：三个 E8 grouped 直接传家族（`Rk4v4E8` / `Rk4v2E8` / `Rk2v4E8`），parallel 由 `rk_parallel_host_dtype<PackedK,E8Lattice,E8Root,V8Root>()` 从 dispatch 的 codec 布尔推家族（int8K+int4V→rk8v4、int4K+int4V→rk4v4、+E8Lattice→rk4v4-e8、+E8V→rk4v2-e8、E8Root→rk2v4-e8）。实测（35B-e2 / `--no-cuda-graph` / prompt 330 tokens / split 0）：`[kv-cpu] split 0 -> CPU partial (dtype=rk4v4-e8 q_heads=16 kv_heads=2 width=2 window=330 active=6 range=[0,32) key_tile=32); the GPU slot is overwritten before the merge`，`rk4v2-e8` / `rk2v4-e8` 同形（三族均 engage，exit=0）。parallel 段（width 9..16）**此前对所有家族都没有 gate**：现已在 `rk_parallel_grouped` 调用，11-token prompt 上得到 `[kv-cpu] NINFER_KV_CPU_PARTITION=0: this call's width exceeds the CPU leg's pinned staging (8 tokens)`；它按设计进不去 CPU 腿（`kStageWidth=8`，且 producer 对同宽度也拒绝），故只到 gate 为止。为使该行可辨，gate 的 width 判决移到 view 判决之前并改了措辞（producer 与 gate 原来印同一句）。默认两道门禁逐位 `6.024313` / `20.673595`，且默认（knob 未设）**0 行** `[kv-cpu]`。仍有两个只在 armed 时出现的限制（非本项范围，记下来）：① CUDA graph 开着时 mirror 的 D2H+`cudaStreamSynchronize` 使捕获失效 → `cudaErrorStreamCaptureInvalidated` 崩（exit `0xC0000409`），所以上面用 `--no-cuda-graph`；② mirror 的 record 缓冲"已为另一几何定尺寸"就拒绝，故只有窗口首次的那次 decode 能 engage |
| P4.2 | `make_int8_kv_causal_plan` 的 partition/key_tile 透出给 models | 决策 1 能自称选择者，knob 降级为覆盖 | 🟡 | 通道已通、决策 1 仍零调用。`src/ops/softmax_attention/dense/causal_cache/int8/plan_probe.h` 一槽 thread-local，**唯一写入点**是 `rk_kv_cpu_split_gate`（传的就是 launch 用的同一个 `partition` 对象与 `Instance::Schedule::kKeyRows`，家族取该实例的 `DType` 经 `rk_host_kv_codecs().name`），写在 knob 判定之前，所以描述的是"跑了什么"而不是"knob 想跑什么"；models 侧 `text.cpp::attn_mix` 在注意力调用返回后只读打印一次。实测（qwen3_8_27b_v3 / rk4v4 / `--no-cuda-graph` / `NINFER_SPLIT_TIMING=1 NINFER_KV_CPU_PARTITION=0 NINFER_QSA_FORCE_FULL=1`，`runs/2026-11-16/p42-kv-plan/probe.log`）：`[kv-cpu] split 0 -> CPU partial (dtype=rk4v4 q_heads=24 kv_heads=4 width=2 window=127 active=2 range=[0,64) key_tile=32)` 与 `[kv-plan] layer 0: partition(capacity=2 target=64 key_shift=6) key_rows=32 family=rk4v4 q_heads=24 kv_heads=4 width=2 batch=1` 四项对齐（capacity 2 = active 2；key_rows 32 = key_tile 32；家族同名；q/kv/width 同值）。默认门禁逐位 `6.024313` / `20.673595`，默认（env 未设）0 行 `[kv-plan]`。**仍差什么才能成为选择者**：① 被定价的页不是宿主驻留的（`kv_host_split_mirror.h` 仍是 D2H 拷贝），决策 1 的"省掉 copy-in"侧没有真输入；② 一次 split 的 `partition_bytes` 要 `rk_partial_split_range(window, split, active, key_tile)` → `rk_partial_count_ops().kv_bytes`，其中 window 仍来自宿主 positions 镜像，所以决策点必须站在镜像生产者里，而不是本通道；③ gate 现在按 `NINFER_KV_CPU_PARTITION` 选 split，把它换成选择者需要 ② 的两个输入。本项给出的 value 是：`capacity`（=本次调用的 active）与 `key_rows`/`key_shift` 从今以后不必猜 |
| P4.3 | 真宿主驻留页（demote 后注意力直接回读） | 镜像不再依赖 D2H；每层少一次流同步 | ⬜ | 现在是测试性归属 + 每层 `cudaStreamSynchronize` |
| P4.4 | 记录"恒为 DeviceCopy"的量化结论 | 文档写明：33,792 B ≈1.35 µs vs CPU 275 µs；链路需 <0.12 GB/s 才翻 | ⬜ | 已算，未落文 |

### P5 测试与清理

| ID | 任务 | 验收 | 状态 |
|---|---|---|---|
| P5.1 | `ctest` 全量跑一次（注册 208，`build-win` 只有部分可执行） | 给出通过/失败清单 | ⬜ |
| P5.2 | `diag-cpu-partial` A/B 扩到 5 族 | 每族 engage + 偏差 | ⬜ |
| P5.3 | `host_pages.h` 的 `packed_k` 已无生产消费者 | 删除或标注"仅测试用" | ⬜ |

---

### P6 35B-e2 制品跑不起来（2026-11-16 新发现，阻塞 e2×驻留验证）

| ID | 任务 | 验收 | 状态 | 证据 |
|---|---|---|---|---|
| P6.1 | 查清 35B-e2 为何构造期没有建池 / 流未挂载 | 该制品能跑到 `overall` 行且 `[experts] runs=3` | ✅ | 断点：**加载计划与 FFN 视图判据不一致**。该制品是 repack 布局（一个专家一个对象：`text/layers/0/moe/experts/0/gate` → `weight/expert/000003/000/gate_up` 行 [0,I)、`up` 同一对象行 [I,2I)；对照 plain 35B 的 `experts/0/gate` → `weight/000003` 行 [0,I)、专家 1 的 gate 在同一对象行 [2I,3I)）。于是 `parameters.cpp:524` 的 `resident`（`banks.whole_bank()`）为 false，层里**必然**建 `ExpertStream`（`:571`）；而 `load_options.h:64` 的 `stream_experts = expert_cache_bytes != 0` 因为 19.6 GiB 驻留计划装得下、`expert_cache_bytes` 仍为 0，`model_instance.cpp:1589` 的 `if (options.expert_cache_bytes != 0)` 整段不执行 → 无池、无 attach。修法：`bind_moe` 用专家 0 与专家 1 的**对象身份**判定布局（`text.cpp`，`Bindings::experts_per_object`），`LoadPlan::experts_per_object()` 透出，`model_instance.cpp:1416` 在该判据为真时先按 `kExpertCacheAuto` 重规划再加载。实测：`[budget] experts: this artifact keeps one object per expert, so no parent holds a resident bank view; the run streams them (19.6 GiB resident -> 2.3 GiB)`，`[experts] banks=40 slot=1966080 windows=40 pool=18879MiB`，`overall 1.902313 6.701375`（exit=0，与 plain 35B 逐位相同 → 同一权重、只换对象布局）。**与验收口径的差异**：本制品 `runs=1` 不是 3，因为 repack 里 gate/up 在每个专家对象内是融合的（`fused_form()` 为真）；`runs=3` 的三bank形态是 8B 制品（`experts/gate\|up\|down` 三个 bank 对象，`[experts] host tier ... form=gate/up/down (one tier bank per run)`）。plain 35B 不受影响：仍 `weights ready \| 19.6 GiB`、无新行、`6.701375`；默认两道门禁逐位 `6.024313` / `20.673595`。**驻留仍未达**：池 18879 MiB 只装 10040/10240 专家（每 bank 差 5 槽 × 1920 KiB），故 e2×整层驻留这一格仍不可达 |

---

## 3. 可独立关账的项目（做完即划掉，互不依赖）

已关账（本轮）：

1. ✅ P1.1 列名对齐 `*_mb`
2. ✅ P1.4 `cold_range` 进表
3. ✅ P1.3 `kv_spill` 三元组
4. ✅ P1.2 prefetch 强制 0（`NINFER_PREFETCH_SLOTS` 单独设置已 fail-loud）
5. ✅ P3.3 `[ctl] cpu` 面板在 e2 路径可达
6. ✅ P6.1 35B-e2 建池/挂载（repack 布局没有驻留 bank 视图，改为强制流式 + auto 池）
7. ✅ P4.1 E8/parallel 的 4 处 gate 调用点（三族 E8 均 engage）

待关账：

1. ~~P4.1 E8/parallel 的 4 处 gate 调用点~~（本轮关账）
2. P5.3 `packed_k` 清理
3. P2.3 形态与策略解耦（`[pfv]/[pf-ctl]` 说清楚缺哪一半）
4. ~~P6.1 35B-e2 为何不建池 / 流未挂载~~（本轮关账）

## 4. 口径与陷阱（写下来，避免下一轮再踩）

1. **竞技场不是计算**：16384 / 18432 / 21504 / 22500 MiB 都给 `6.024313`，所以它的尺寸只影响命中，不影响数值。
2. **映射悬崖会漂**：同一天从 ~22 GiB 漂到 ~20 GiB，固定上限不可依赖；且"先探测后分配"不成立（探测 22528 成功、真分配 22527 失败，探测本身还花 3.7 s）。
3. **tok/s 双模态**：臂内散布可大于臂间差；速度结论必须 min-of-N + 同会话交错，且不能单独作回归证据。
4. **span 与 copy 不同口径**：span 含 tier fill，copy 侧把这次读约掉了；不统一 scope 之前，任何"用实测速率定价"的结论都不成立。
5. **`[ctl]` 面板在 e2 路径不打印**：`report()` 只在 `streamed_ffn` 被调用，看到 0 行不等于"值为 0"。
6. **设备别名早一层就换数值**：`arena_device_base()` 非空决定 split 路径走"Device 直读 host tier"还是第一层走 CPU 腿。块预先分配但别名在构造时就发布，ppl114 给 `6.017490`（17203 slots 与基线同）；别名等到第一次 `arena()` 再发布，回到 `6.024313`。所以"竞技场尺寸不影响数值"成立的前提是**别名的可见时机也不变**。
7. **一次被拒的映射是这个进程的终点**：钉住内存由系统 commit 担保，WDDM 还把 Device 分配记进同一份账。实测（独立探针，持 21.5 GiB Device 内存）：avail commit 23.3 GiB，请求 44 GiB 失败后 commit `used=limit=91.1 GiB`、avail 0，之后 40/32/24/20/16/12 GiB **全部** `resource already mapped`；引擎内同样表现是 42 次尝试全败。"失败再逐级退让"因此不成立，只能**先按 `ullAvailPageFile`（减 1 GiB）夹取再请求一次**；`NINFER_EXPERT_HOST_ARENA_MB` 是显式值，不夹取，拒了就报错。
8. **主机读-ahead 武装会换 PPL**：`NINFER_PREFETCH=1 NINFER_PREFETCH_SLOTS=8192` 下 ppl114 = `6.063844`（默认 `6.024313`）。与第 6 条同源：tier 内容改变哪些专家走 CPU 腿。预取线的 A/B 必须以 `6.024313` 臂为准，不能跨臂比数值。

## 5. 变更记录

| 日期 | 事件 |
|---|---|
| 2026-11-16 | 建表。矩阵：1 格实测、1 格 N/A（硬件）、2 格待验。P0 由当晚 `mapped allocation of 20941 MiB failed` 升级为最高优先级 |
| 2026-11-16 | P0.1 关账（`cbc2ad49`）：分配在前、定尺寸在后，默认 3×2 门禁逐位。P0.2 记 N/A（单块在 22 GiB 上 6/6 成功）。P0.3 关账：失败在分配点抛出，pageable/registered 兜底删除 |
| 2026-11-16 | P1 四项关账：P1.1 列名/列序（`aea2b93c`）、P1.2 无消费者强制 0 + env fail-loud（`b9f557e9`）、P1.3 `kv_spill` 三元组（`54568858`）、P1.4 `cold_range` 进表。`[plan]` 三行逐列对齐规格，跨架构不变（只读 `Instance` 几何与 placement） |
| 2026-11-16 | 修正两处被证据推翻的实现：P1.1 只调表行没调枚举，`slot()` 按枚举索引，`prefetch_readahead` 的 160 MiB grant 被当作"从别行 carve"静默拒绝（已按规格顺序重排枚举 + 编译期 `static_assert`）；P0.1 的"失败后按 GiB 退让"被探针证明不可恢复（见陷阱 7），改为按 `ullAvailPageFile` 夹取后单次请求 |
| 2026-11-16 | P3.3 关账：`PolicyController::note_layer()` 成为"不跑 step() 的路径"的唯一计数入口，split 入口每层调用一次并在 `NINFER_SPLIT_TIMING` 下调用 `report()`。e2 实测 20/20/20 行（fetch/cpu share/cpu leg）对 816 行 `[cpu-path]`；默认门禁逐位，面板 0 行。`report()` 的数字语义未改，`[ctl] cpu share` 的 `rate span=..@N compute=..@M` 与 `acted=N 层/M 路由` 原样可读 |
| 2026-11-16 | P6.1 关账：35B-e2 是 repack 布局（每专家一个对象 `weight/expert/<layer>/<e>/gate_up`），`parameters.cpp` 因此必建 `ExpertStream`，而加载计划按"装得下就驻留"给了 0 池 → 层 0 抛 `expert stream is not attached`。新增 `Bindings::experts_per_object`（用专家 0/1 的对象身份判定，而不是绑定名——两种布局都有 `experts/<e>/gate` 名字）+ `LoadPlan::experts_per_object()`，`construct_model_once` 在该判据为真时按 auto 重规划。实测 `19.6 GiB resident -> 2.3 GiB`、`[experts] ... pool=18879MiB`、`overall 6.701375`（与 plain 逐位相同）；plain 35B 仍 19.6 GiB 驻留；默认门禁 `6.024313` / `20.673595` |
| 2026-11-16 | P4.1 关账：`launch.cu` 四处 gate 调用点补齐（`rk_e8_grouped` / `rk_e8v2_grouped` / `rk_root_grouped` 直接传家族，`rk_parallel_grouped` 用 `rk_parallel_host_dtype` 从 codec 布尔推出家族）。三族 E8 在 35B-e2 上实测 engage（`dtype=rk4v4-e8/rk4v2-e8/rk2v4-e8`，各 exit=0）；width 9..16 段历史上对所有家族都没有 gate，现在会到 gate 并按 8-token staging 边界拒绝。gate 的 width 判决移到 view 判决之前并改写措辞，使 producer 与 gate 的拒绝可分辨。默认门禁逐位，默认 0 行 `[kv-cpu]` |
| 2026-11-16 | P4.2 通道关账（决策仍未接）：新增 `int8/plan_probe.h`（thread-local 一槽，唯一写入点 `rk_kv_cpu_split_gate`，参数即 launch 的 partition 与 `Schedule::kKeyRows`），`text.cpp::attn_mix` 在 `NINFER_SPLIT_TIMING` 下只读打印一次 `[kv-plan]`。实测两行对齐（`capacity=2`=`active=2`、`key_rows=32`=`key_tile=32`、`family=rk4v4`=`dtype=rk4v4`）；默认门禁逐位 `6.024313` / `20.673595`。`decide_cpu_kv_split` 仍零调用：缺的是宿主驻留页与"从这个通道之外的 window/字节"两件事，见 P4.2 行 |
