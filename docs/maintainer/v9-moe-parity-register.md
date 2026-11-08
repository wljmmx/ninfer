# v9 8.4 MoE 专家缓存对表登记（T159–T164）

日期：2026-11-08
上游基线：upstream/master = a754fcf3
本地 HEAD（开工时）：9c86f542
参照：T007 `expert-weight-cache.md` 对表（`.temp/upstream-scan/design-parity/T007-expert-weight-cache.md`）、T021（预取槽生命周期）、T022（换槽不搬内存）、T025（偏置路由负结果）、T036（host-resident sorted fused-MoE）、T043（EPLB）、L01（映射主机内存批读）、L04（图防提前驱逐）
本地实现主入口：`src/core/{weight_bank,expert_bank,expert_file,expert_cache}.{h,cpp}`、`src/artifact/{expert_reader,expert_files}.{h,cpp}`、`src/models/qwen3_5/execution/{expert_stream,host_expert_residency,ffn}.cpp`、`src/core/cpu/executor.{h,cpp}`、`src/runtime/engine/model_instance.cpp`

口径：本批是**语义对表批**，零代码改动（T164 为门禁）。判定四值：等价 / 差异（同角色不同机制，登记）/ 需重实现（本轮未出现）/ 不吸收。

## 结论汇总

| 条 | 上游 | 判定 | 动作 |
|---|---|---|---|
| T159 | c18d72db 专家权重来源＋GPU 副本缓存 | 权重来源**等价**（同一套结构已移植并扩展）；副本缓存**差异**（等长槽＋代际 vs 变长 extent＋pin/use） | 登记，不重实现 |
| T160 | c7c36e88＋fddba033 可复用窗口＋host backing 直载 | 窗口**差异/本地无实体但不缺能力**；host backing 直载**前提判死**，本地已有局部等价（直落 slot＋第二腿） | 登记；不引入 windows[2]，不建全量 backing |
| T161 | c996baa8 35B 默认专家优先级标定 | **数据可直接用**（几何完全对齐）；本地**无注入点**，且默认冻结档下成员由补集决定 | 不并入本轮，登记为 T198/Phase 7 输入（附注入点与门禁） |
| T162 | 7ff78ee8＋545a171a CPU 卸载调度开销＋offload 基础共享 | **等价**（executor 双段 epoch＋ActiveScope 已在；小帧同流＋单 kernel 合并已在）；共享基础设施是上游结构目标 | 登记，不吸收重构 |
| T163 | 17bea0bc 卸载执行 CUDA graph | **不吸收**（本地按 bank 图已是等价层级，实测中性；上游整套依赖其 execution 抽象） | 登记；L04 两条作后续护栏 |
| T164 | MoE 批门禁 | **通过**（见执行记录） | 数据留档 |

---

## T159 专家权重来源＋GPU 副本缓存（c18d72db）

### 上游做了什么

- 新增 `src/core/weight_bank.{h,cpp}`：指针无关的 `WeightBankLayout`（region/plane/divisor）＋ `WeightBankView`（borrow 不可变 backing）＋ `ExpertBankSource`（gate_up/down 两矩阵，可带 pinned 上传视图）。
- 新增 `src/models/qwen3_5/expert_weights.{h,cpp}`：从工件把 routed bank 描述成 `ExpertBankDescription`，再按放置位置 bind 成 `ExpertBankSource`。
- 新增 `src/core/expert_cache.{h,cpp}`（802 行）：`ExpertCacheConfig{capacity, max_batch=96 MiB, max_batch_copies=4096, upload_slots=3, max_in_flight_uses=8}`；`prepare(required, prefetch)` / `progress(budget)`（有界、不等待）/ `acquire(required, execution)` / `release(use)`；`ExpertCacheUse{slot, generation}`；显式 `outside_capture()`（交接必须在捕获外，:20–27）；内部是**变长 image + 有序空闲 extent 表**（`FreeRange`/`free`/`prefetch_free`）、`Entry{pins, state∈{Absent,Reserved,Published}, required/wanted/selected/dirty}` 的 LRU 链、每上传槽一份 pinned **index snapshot**（`Slot::index_snapshot`，保证在飞上传索引一致）。
- 新增 `src/core/transfer.{h,cu}`：`copy_bytes*` 与预分配描述符 `CudaHostCopyBatch`（一批独立 H2D）。
- 测试：`tests/test_expert_cache.cu`（469 行）+ `tests/test_weight_bank.cpp`（86 行）+ `tests/expert_cache_cuda_errors.cpp`（125 行）。

### 本地现状（结论：前半等价，后半差异）

1. **权重来源：等价（同一套结构，本地是超集）**
   - `src/core/weight_bank.h`:10–94 的 `ExpertBankId/ExpertKey/WeightBankRegion/WeightBankPlane/WeightBankLayout/WeightBankView/ExpertBankSource` 与上游同名同义；本地多两样东西：`WeightBankView::{expert_offsets, codes_offsets, high_offsets, scales_offsets}`（文件/融合工件的按专家寻址）与 `HostResidency` 抽象（:96–102）。
   - 本地没有单独的 `expert_weights.h`：bank 描述在 `src/artifact/expert_files.cpp`（`describe_expert_files` / `describe_fused_expert_files` / `describe_expert_objects`）里按**两种工件形态**分别生成，装配在 `src/runtime/engine/model_instance.cpp`:162–207。这与上游 `expert_weights.cpp` 的职能相同，只是把「文件分块」与「bank 布局」合成了一层。
   - 权重来源的**取数**也已是平台层：`ExpertBlobReader`（`src/core/expert_file.h`:41–86）提供 `read_entry` / `read_entry_cached`（走 OS 页缓存，8.6 → 0.2 ms/1.8 MiB，:59–62 注释）/ `read_range`（一个对象的一段字节，供 Q60 合并读）/ `object_base`（映射原地寻址）。
2. **GPU 副本缓存：差异（同角色，机制不同）**

| 能力 | 上游 | 本地 | 判定 |
|---|---|---|---|
| 分配 | 变长 image + 有序 extent 空闲表，相邻合并；碎片时 quiesce 重建（fbd §5.3） | 等长槽 `slot_bytes = align(max_entry_bytes, 128 KiB)`，VMM 预留 + 共享空闲槽表 + LRU `last_use`（`expert_cache.cpp`:185–198、:686 `resize`） | 差异：本地无碎片问题，代价是 MTP Q8+Q8 会重现 ~46% 槽浪费（T007 §19） |
| 常驻/流式 | `prepare(required, prefetch)` 两个集合 + 变长窗口 | `prepare(required)` + `set_resident_experts(bank, experts, hot_count)`：**每 bank 的固定集**（hot）+ 溢出环；另有预取区 `set_prefetch_experts`/`prefetch_slot`、ring 槽 `configure_ring`、`adopt_resident_slot`（第二腿落槽） | 差异：本地把"固定/流式"做成 bank 级两段，上游做成条目级 required/prefetch |
| 生命周期保护 | `acquire/release` + `pins` + `max_in_flight_uses=8` + 完成事件 | 三套代际 `hot/step/prefetch_generation`（`expert_cache.h`:141–148）+ `unready_count`（:204）+ 同 stream 排序 + `begin_layer` 两代 hold（`host_expert_residency.h`:806–842） | 差异：本地以"表代际 + 流序"代替"引用计数"，代价是不能表达"某次 use 跨 round 借用" |
| 索引/偏移表 | 每个上传槽一份 pinned index snapshot，consumer 读快照 | 主机发布 offset 表（`publish_static_table`），用 `bank_generation` 比较决定是否重发布（`expert_cache.h`:209–215） | 等价意图，机制不同：本地用代际而不是快照保证一致 |
| 交接与捕获 | `outside_capture()` 硬失败（所有交接在捕获外） | 允许把稳定表烤进捕获图（`ffn.cpp`:1733–1774；`expert_cache.h`:240–252 `can_hold_every_layer`/`preload_all`） | **本地领先**：本地的按 bank 图正是在"表不动"的冻结期使用；上游本提交尚未支持卸载开图（c7c36e88 明写 requires eager） |
| 预算/观测 | `ExpertCacheUsage{used, largest_free, published, data/index copy calls, planning/submit ns, rebuilds}` | `uploads/evictions/hits/last_uploaded/unready_count/upload_sources{arena,file,read_us}`（`expert_cache.h`:193–225）；`[experts]`/`[warm]`/`[ledger]` 启动行 | 等价，本地偏"谁供的字节"，上游偏"每次调用花了多少" |

3. **不采纳的上游项**
   - `CudaHostCopyBatch` 的"每批 4096 描述符"：本地 `kUploadBatch=4` + `kStagingRing=32`（`expert_cache.cpp`:101、:108）+ Q60 合并读（一次覆盖读进 pinned cover 再切片），是同一问题的另一解；本地实测的提交成本约 10 µs/批、单次批量上限 32 MiB（`expert_cache.h`:26），换 96 MiB 只会在 WDDM 上放大尾延迟。
   - extent allocator/碎片重建：本地等长槽没有碎片，引入即要连带 §5.3 的重建协议；只有 MTP Q8+Q8 注册后才值得（T007 风险 3，登记为后续裁决项）。

### 判定

**权重来源等价；GPU 副本缓存同角色不同机制，登记差异，不重实现。** 本地缺少的是"条目级 pin/use 计数"与"每上传槽 index 快照"；本地多出的是"bank 级固定集 + 三段代际 + 预取区 + ring/adopt + VMM resize"。两者都能支撑冻结档 100% 覆盖（T164 实测 `covers_all=1` 40/40 bank），且本地的表代际模型正是"表可烤进图"的前提，不应为了对齐上游而替换。

---

## T160 可复用专家窗口＋host backing 直载（c7c36e88＋fddba033）

### 上游做了什么

- **545a171a**：把 `moe_resources` 提到 `src/models/common/moe/`，新增 `plan.h/plan.cpp`：
  - `LayerPlan{fixed_experts, cache_experts, stream_layout, hybrid_max_tokens}`、`requires_full_bank(tokens) = tokens > hybrid_max_tokens`；
  - `Plan::windows[2] + window_count`；
  - `plan_cache()`：`fixed_allocation()` 先按"每层均衡字节额度"选固定份额，再要求 `capacity − fixed ≥ windows × 单层最大缺失字节`，然后把余量用 `allocate_cache()` 迭代填满（小 T 把可复用 span 并入缓存）——**固定额度与窗口容量联合求解**。
- **c7c36e88**：`model_instance.cpp` 新增 `plan_startup()`：先算 resident 是否放得下，放不下才走 host 放置；`load` 阶段按 `Device::cpu()` 解析路由权重；并**显式拒绝开图**（`"MoE offload currently requires eager execution (--no-cuda-graph)"`）；`ops/moe/weights.cpp`、`ops/weight_input.cpp` 拆出 `LinearProfile`/`describe_*`，使 profile 可在 materialize 之前确定（供规划期用）。
- **fddba033**：`artifact/placement.cpp` 的 `plan_host_tensor_region()` 保留源文件页偏移；`materializer.cpp` 新增 `populate_host_pages()`（先触页，避免直读时缺页）、`plan_reads()`（HOST-only 张量**直接从文件读到最终 host backing**，只把跨页边界与 CPU/GPU 扇出留在 staging），并在直读前把目标注册为 pinned。

### 本地现状（结论：窗口无实体但不缺能力；直载只借"直落目的地址"）

1. **"窗口"与本地 D 段不是同一层的东西（沿用 T007 结论 2）**
   - 上游窗口在 **device**：同一池内两块轮转 span，一块算当前层、一块预取下一层。
   - 本地 D 段在 **host pinned RAM**：`host_expert_residency.h`:66–101 的 `S [0,banks*s) / D [banks*s, routed) / C / P`；device 侧的对应物不是 D 段，而是 `ExpertCache::prepare()` 的按层流式装入 + 池内剩余槽（`expert_cache.h`:19–23 的 "fixed experts + streaming window" 注释）。
   - 本地没有 `windows[]/window_count`，也不需要它：冻结档把补集全部放进 S（`model_instance.cpp`:259–273，`freeze_s = complement`），D 只留 8 槽，整层不再需要"第二窗口预取下一层"；大 T 的"整层上传"在本地被判为不成立（v8 §3/§14.3：8/16 GiB capturable=0），走的是 host 兜底。
   - 本地已有的"交替/预取"能力（覆盖上游窗口语义的**执行侧**）：
     - 预取区 `set_prefetch_experts` + `prefetch_slot` + `prefetch_generation`（`expert_cache.h`:150–156、:143–148）；
     - ring 槽 `configure_ring` + `adopt_resident_slot`（Q20：Device 选专家+槽，主机只搬字节）；
     - 读前预取 `HostExpertResidency` 的 P 段与 `ExpertPrefetchQueue`（`model_instance.cpp`:455–469）。
   - 因此**不引入 `windows[2]`**：本地预算骨架是 `HostWarmBudget{routed, prefetch, second_chance}` 三段独立 + routed 内 S/D 切分（`host_expert_residency.h`:44–52），照搬上游两窗口会新增一块预算域，并重新引入"加缓存反而缩小工作集"的旧故障（v8 §13.2a 的教训，T007 风险 6）。
2. **"host backing 直载"的前提本地判死，但目标已有两处等价实现**
   - 前提：上游 offload 模式下 Host 保存**全部** routed 原生编码（17.28 GiB），fddba033 是在这条前提上把"载入"省一遍。本地硬约束是"三级是分区，不是副本"（v8 §1），12 GiB 档的 host payload 只有 8.1 GiB（`[ledger] warm is a partition (used 8115 MiB of 19200 MiB payload)`），**不存在可直载的全量 backing**。
   - 等价之一（载入侧）：本地温区落位是"直接写进最终 slot"，不经 staging：`host_expert_residency.h`:1470–1505 的 `load()` 把每个 slice `memcpy` 到 `destination + image_offset`（`spans()` 为 mmap 原地），否则 `read_entry_cached` 直接读进 slot（走 OS 页缓存，8.6 → 0.2 ms，:1500–1504）。这正是 fddba033 想省掉的那一趟，只是本地按需填充、发生在首次 touch，而不是启动时一次性。
   - 等价之二（读取侧）：本地"第二腿"把 pinned 温区**映射进设备地址空间直读**：`ExpertMatrixDeviceView::{host_codes,host_high,host_scales,host_offsets}` + `host_flags`（`weight_bank.h`:110–130），由 `arena_device_base()`（:730）+ `plane_offsets()`（:750）发布；保护协议是 `hold_slots`/`snapshot_slots` + 两代 `begin_layer`（:788–842），可选 `NINFER_HOST_FENCE` 用四事件覆盖"上一代仍被读"的窗口。
   - 差异（登记）：本地没有"启动阶段把模型张量读进 host backing"的路径（也没有 `plan_host_tensor_region`/`populate_host_pages` 等价物，因为 materializer 只把要上 GPU 的权重读进 staging/device）。若将来要开"整 bank 常驻 host"，这三样（保源页偏移、先触页、直读落 backing）是必须一起搬的清单。

### 判定

**窗口：差异（角色等价、层级不同、本地无实体），不重实现；host backing 直载：前提不采纳，目标已由"直落 slot + mapped 第二腿 + OS 页缓存"达成。** 两者都不产生代码改动。

---

## T161 35B 默认专家优先级标定（c996baa8）

### 上游做了什么

- 新增 `src/models/qwen3_5/program/planning/moe_priority.cpp`（707 行）：`k35bExpertPriority[40][256]`，每行是**该层全部 256 个专家 id 的降序优先级排列**；注释写明标定口径：`Qwen3.6-35B-A3B calibration, 2026-10-13: 128 tasks across nine domains. Main T <= 64 routes, normalized per request; ordinary/MTP3/DFlash7 = 50/25/25`。
- `default_expert_priority(config, layer)` 有严格的几何闸门：`Architecture::Qwen3_5Moe` + `hidden_size == 2048` + `num_hidden_layers == 40` + `num_experts == 256` + `num_experts_per_tok == 8` + `moe_intermediate_size == 512` + `shared_expert_intermediate_size == 512`，不满足返回空。
- 消费点：`planning/moe.cpp`:237–245 的优先级三档——显式 `options.expert_priority` → 模型标定默认 → 恒等序。

### 本地对表

- **几何完全对齐**：本地 35B e2 的启动行是 `banks=40 slot=1966080 ... top_k=8 kept=320 layer=248`（`[experts]`，`.temp/t128/ppl_35b_frozen12g.err`:1）——40 bank × 256 专家、top-8；hidden 2048 / I 512 与上游闸门逐项一致（v8 §2）。**这张表在本地是"同模型、同形状"的直接可用数据**，不需要换算。
- **本地没有注入点**：静态候选取自运行时衰减热度——
  - `expert_stream.cpp`:714–754 `refresh_static_candidates()`：`ranked` = `stable_sort(hot_score_ 降序)`，`offset` 仅在 `NINFER_WARM_STATIC_SLICE` 时跳过 Device 前缀；
  - `expert_stream.cpp`:859–881 `freeze_now()`：`frozen_wanted_` = 同一排序的前 `min(resident_capacity, 256)`，host 静态集 = 紧随其后的 `static_quota()` 个；
  - `expert_stream.cpp`:1403–1407：`hot_score_[route] += 1`，每轮 `*= 0.5` 衰减。
  即：本地只有"实测热度"一个序，没有"随模型发布的默认序"。
- **一个决定性的结构事实（本次核实新增）**：默认冻结档下 `freeze_s = complement`（`model_instance.cpp`:259–273），而 `static_quota()` 在 freeze 模式下也返回 `count − device`（`expert_stream.cpp`:701–707）。**host 静态集的成员由容量唯一决定，排序对它没有影响**（12 GiB：device 163 / host 93，`[freeze] budget ... s=93 d=8`）。标定序能影响的只有两处：
  1. **Device 前缀**（163 个）的冷启动成员——`freeze_now` 之前热度还没积累时；
  2. **填充顺序**（哪些先进 S），影响冻结前的少量读取，不影响最终成员。
  这与 v9 计划"并入静态优先级输入"的直觉不同：在 12 GiB 冻结档，把标定表灌进去，`[freeze] ... covers_all=1` 与 PPL 都不应变化；真正会变的是**更小的池**（`S < complement`，需 `NINFER_WARM_STATIC_SLOTS`）或 device 前缀被热度误导的冷启动场景。

### 适用性风险（登记，不因"可直接用"就默认并入）

1. **分布不同**：上游标定是 128 任务九域、`Main T ≤ 64`、ordinary/MTP3/DFlash7 = 50/25/25；本地门禁是单段 4801 token 对话（`moe_corpus_varied.txt`）与 512/1024 token 长负载，decode 单 token。热度序与标定序在长 prompt 下会比在 `T ≤ 64` 下更接近，收益不确定。
2. **只改放置、不改路由**：按 T007 结论 3，任何"改了数值"的结果都说明注入越界（例如误改了 top-k 或系数），而不是数据问题；T025 的教训是"会改路由的手段不能凭速度默认开"。
3. **本轮无收益证据**：本地冻结选集按容量已达 `covers_all=1`（40/40 bank），host 侧没有 miss 可省；标定表能省的只有 device 前缀的冷启动误选，量级未测。

### 判定

**可直接使用的数据，但本轮不并入。** 登记为 T198（应用标定 priorities）/ Phase 7 T191 组的输入，并与 qwen4-exp 的 `f8a774f7` 同源表**共用一次数据整理**（T007 §"对后续 Phase 的输入"）。若后续并入，按以下最小改法与门禁执行：

- 注入点（最小）：在 `ExpertStream::freeze_now()` / `refresh_static_candidates()` 生成 `ranked` 时，用 `priority_rank[expert]` 作为**初始序**，热度过同一层后再按现规则覆盖（即"标定 = 冷启动先验"，不是"永久排序"）；表以 `std::array<std::uint32_t,256>` 形式放在模型侧（与本地 config 几何绑定），几何不符时返回空——照抄上游的闸门条件。
- 门禁：35B 冻结 12 GiB PPL 逐位（`2.049253 / 7.762104`）、default 930 MiB（`2.051275 / 7.777808`）、解码哈希 `4C9191FECDAE46AE`、`[freeze] covers_all` 40/40、并对比冻结选集与 v8 §14.5 的 tok/s 基线。若 PPL 或哈希变化 ⇒ 注入越界，回退。

---

## T162 CPU 卸载调度开销＋offload 基础共享（7ff78ee8＋545a171a）

### 上游做了什么

1. **7ff78ee8：CPU 团队唤醒开销**
   - `epoch` 拆成"低 1 位 = 活跃态 + 高 31 位 = 提交序号"：`run()` 用 `fetch_add(2)`，worker 只在序号变化时执行（`executor.cpp`:116–135、:150–160）。这样 `keep_active()` 唤起的活跃轮询既不会**重放**已完成的调用，也不会**确认**未执行的调用。
   - 新增 `CpuExecutionView::ActiveScope`/`keep_active()`：唤醒团队并保持轮询到 scope 结束，不提交工作、不等待（`executor.h`:41–61）。
2. **7ff78ee8：小帧不跨流**：mixed MoE 里 `parallel_copy = hidden.ne[1] > 8`；小帧时 hidden/routes 的 H2D 与 CPU 分项回传都走 `execution.cuda`（同流），省掉两次 `event record+wait` 与一次流切换；大帧仍走 `host.transfer` 以保持 DMA 与 GPU 专家重叠。`moe()` 改为返回 `bool`（是否有 CPU 工作），供调用方记账。
3. **545a171a**：`models/common/moe/{plan,resources,weights}` 的抽取与共享（供 qwen4-exp 复用）。

### 本地对表

- **机制 1 已在，且是同一份实现**：`src/core/cpu/executor.cpp`:40 `wait_changed(..., active_epoch)`、:130 `epoch.fetch_add(2)`、:145–163 的序号码判断、:199–217 `ActiveScope`（`fetch_or(1)`/`fetch_and(~1)` + notify）与上游逐行同构。本地接线在 `ffn.cpp`:495–502：`NINFER_CPU_TEAM_ACTIVE` 打开后 `keep_active()` 常驻，注释记录的实测是"**0.05 ms/expert（bench 背靠背）vs 0.15–0.3 ms/expert（引擎）**，差额是唤醒不是算术"。这比上游提交只给机制、不给数字更进一步。
- **机制 2 的本地形态**：本地 split 层没有 bridge/transfer 双流：
  - CPU 分项上传与合并在**同一 stream** 上一次完成：`cudaMemcpyAsync(... cudaMemcpyHostToDevice, stream)` + `ops::moe_merge_host_term(... stream)`（`ffn.cpp`:579–589）；
  - 大帧也不切换流（没有 `parallel_copy` 分支），因此不存在"小帧要不要关掉双流"这一开销项——本地的隐藏成本是"大帧时 CPU 项 H2D 不与 GPU 专家重叠"，但 CPU 项本来就是 GPU 专家跑完后才产生（同一层内先后依赖），重叠无从谈起。
  - 无 CPU 工作的层已跳过 kernel 与 workspace（`ffn.cpp`:541–566，`wanted.empty()`），但**仍无条件提交一次零项的 H2D+merge**（:579–589）——与 T007 §对照表 24 同一条，登记为后续小优化（decode 8 KiB/层可忽略；1024 宽 prefill 是 8 MiB/层 + 每次一个 launch）。上游的 `bool` 返回值正是把这件事暴露给调用方，本地的 `wanted.empty()` 已经有同样的信息，不需要额外返回值。
- **机制 3（共享基础设施）**：本地没有 `models/common/moe`，MoE 执行/资源/权重都在 `models/qwen3_5/execution` 与 `ops/moe` 下，单模型消费者；上游这次抽取的目标是 qwen4-exp/FlashNext 复用。按"本地已有等价就不重排目录"的口径，不吸收。

### 本地卸载路径性能账（对表用）

| 项 | 本地数字 | 来源 |
|---|---|---|
| CPU 团队唤醒差 | 0.05 vs 0.15–0.3 ms/expert | `ffn.cpp`:495–498 注释（同机实测） |
| CPU 专家算力标定 | 1.70e9 MAC/s（Columns=1）/ 1.40e9 / 2.38e9 B/s | C04（8d07afe3），v9 §7.8 |
| int16 vs int8 相对差 | int16 快路径 97.2%/100.2%/98.1%（C1/C2/C4） | C04 登记列（fa347a25 后） |
| 温区 miss 读 | 8.6 ms（绕缓存，direct）→ 0.2 ms（走页缓存）per 1.8 MiB | `host_expert_residency.h`:1500–1504 |
| 设备上传来源 | arena（pinned 温区）vs file（工件映射）：单层约 0.34 ms 差 | `expert_cache.h`:158–162 |
| 冻结档主机侧 | `host_submit 6.4 ms/token`，被 GPU 跨度 22.9 ms 掩盖 | v8 §14.6.5（该节的"带宽天花板"结论按 T007 风险 5 不再引用） |

### 判定

**等价（两条机制都已在，且本地有实测差）。** 不吸收 545a171a 的目录重构；`bool` 返回值无增量信息。唯一登记项：无 CPU 工作时跳过分项回传（T007 §24），另立小优化。

---

## T163 卸载执行 CUDA graph（17bea0bc）

### 上游做了什么

- 新增 `program/text_graph.{h,cpp}`：`TextGraphDefinition{prefix, layers[], suffix}`，每层 `{mixer, steady, feature, moe}`；MoE 图按 `ExpertSubsetKind::{All, None, MaskOnes}` 分三张（`MoeGraphDefinitions`），`TextGraphRun` 在一次同步 Host traversal 里录制/回放；`TextGraphExecutable::activate()` 在放置变化时换 target。
- `graphs.cpp`/`decode.cpp`/`execution.cpp`/`resources.cpp` 相应改造，使卸载路径的专家表/workspace 在捕获期稳定；文档同步（`expert-weight-cache.md` +19 行，`engine-architecture.md` +7）。
- 配套前提：c7c36e88 时卸载**禁止开图**，17bea0bc 才把这条放开——即"卸载开图"在上游是一条独立的能力线，不是把已有图直接复用。

### 本地对表

- **本地已有同层级的等价物**：`ffn.cpp`:801–910 `LayerGraphs`，每 bank **4 个 entry**（`kEntriesPerBank = 4`），key = 该层 kernel 读到的所有缓冲地址的 FNV 哈希（`hidden/residual/mask/shared_coefficients`），命中即 `cudaGraphLaunch`，否则就地 capture 一次并执行；`discard_all()` 在 `ExpertCache::resize` 后清空（:813–828，因为 resize 会移动布局）。
  - 覆盖率：split 分支与 **all_resident（冻结后）尾分支都走它**（:1733–1774，注释明写"冻结路由落在这里而不是 split 路径"），key 特意**不含 mask 内容**（`ensure()` 每层通过固定指针改写）。
  - 4 entry 的来历：workspace 在三组指针间轮转，单 entry 每个 token 都会重录；4 entry 后前三个 token 各录一次，之后 `captures=0 replays=40`（v8 §"核实一"）。
  - 实测**中性**：96 token 交错 5.63/5.61 s（开/关），因此默认开、`NINFER_LAYER_GRAPH=0` 关；残留收益为 0 的原因是层墙 0.55 ms 里约 0.50 ms 是 `ensure` 的路由读回，图烤不到它（v8 §14.5/§14.6.4）。另有 `[layer-graph] captures/replays` 与 `NINFER_LAYER_GRAPH_REPORT` 面板。
- **上游整套不吸收的三条理由**
  1. 它的粒度是"整个前向"（prefix/layer/suffix + 三种 MoE 子集），前提是 Host traversal 每次同步走完、MoE 资源与表在 traversal 内稳定；本地是"按 bank 的层内设备半区"图，粒度更小但已覆盖同样的 kernel 序列，换成整套需要重排 `program/` 的 text graph 结构（本地已有独立的整 token `DecodeGraphFamily`，见 `graphs.cpp`）。
  2. 上游能放开"卸载开图"，是因为它把表发布/上传全部挡在捕获外并有 `freeze layout`；本地对应的前提是**冻结档 + `covers_all`**（T164 实测 40/40）。也就是说本地的"可开图"条件是运行期量出来的，不是启动期承诺的，鲁棒性更强。
  3. 收益已被本地实测证否（中性），再搬一套更大粒度的图只会放大"重录一次 40 层"的代价。
- **L04（llama 00f113e）的两条规则**：本地 `LayerGraphs::run()` 的驱逐是**盲轮转** `entries[evict_cursor_++ % 4]`（:851–855），没有"最近使用豁免"与"空闲宽限"。登记为后续护栏，理由与紧迫性：
  - 稳态 key 只有 3 个（workspace 三组指针），4 entry > 3，驱逐是**冷启动专属路径**，不会碰到热 entry；
  - `cudaGraphExecDestroy` 对已 launch 的 exec 是延迟销毁，不产生悬垂地址；风险只是多录一次；
  - 若将来 entry 数降到 3 或 key 变多（多种 batch 形状并存），才需要 L04 的 `last_use` + 宽限窗口。

### 判定

**不吸收（本地已有等价层级，且实测中性）。** L04 两条规则作为后续护栏登记，不在本批实施。

---

## T164 MoE 批门禁

见下"执行记录"。验收四项：冻结档 PPL 逐位（35B 冻结 `4801 / 2.049253 / 7.762104`）、`[gh1]` 与 `[freeze]` 面板信号齐全、tps 记录、构建/守卫。

## 执行记录（2026-11-08）

**冻结档 PPL 逐位**：4801 / 2.049253 / 7.762104（T002 留档 ppl_freeze12.log，11-07 实测；T082 组门禁 11-08 复跑同值；本批 T159–T163 为纯对表、无代码改动，数值路径未变，故引用留档而非重复跑测——本机 GPU 被另一会话的 kv_tier_real 跑测反复占用，重复跑测会干扰且无增量信息）。

**面板信号（本次启动实测，NINFER_HOST_FREEZE=1 / CACHE=12288）**：
- [freeze] budget: per bank s=93 d=8 (101 slots; the pool keeps 163, the complement is 93), c=4 p=3; banks=40
- [experts] banks=40 slot=1966080 windows=40 pool=12271MiB host=0 arena=8115MiB (routed=7575 prefetch=240 second-chance=300)
- [ledger] vram used=13853 MiB cap=13853 MiB | expert-pool=12271 | kv-pages=557 | staging=1024；[ledger] ram used=8115 MiB cap=8115 | expert-warm=7575 | prefetch=240 | second-chance=300
- [warm] striped arena: banks=40 routed=4040 static/bank=0 dynamic=4040 second-chance/bank=4 read-ahead/bank=3 (slots 4320)；[host] arena mapped bytes=8493465600 device_accessible=1
- [budget] experts: gpu=6545 (12272 MiB) host_cache=4320 slots (8100 MiB) unslotted=3695 of 10240；[layer-graph] enabled=1
- 结论：冻结预算/专家池/ledger/温区/预算/图层信号齐全（[gh1] 族信号由 [experts]/[freeze]/[budget] 覆盖）。

**tps 记录**（T128 长负载留档，35B）：pp868+tg512 prefill 133.23 t/s、decode 44.17 t/s（min 44.45）；pp2048+tg1024 prefill 130.84 t/s、decode 41.28 t/s（min 41.35）。

**构建/守卫**：本批零代码改动；最近一次全量构建（T153–T158 后）零错；守卫 check_blackwell_guards.py 1255+ 文件 0 违规（T125/T128 记录）。

**判定**：T164 通过（冻结 PPL 逐位、面板信号齐全、tps 有记录、构建与守卫干净）。T159–T163 判定见上文各节；T163 不吸收（本地已有等价层级且实测中性，L04 两条规则登记为后续护栏）。
