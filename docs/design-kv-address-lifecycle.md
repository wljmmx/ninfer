# KV 地址生命周期：统一设计（替换按点释放的补丁）

现状问题：`KVAddressSpaceStore` 只有"能不能 release"的即时判断（`can_release` / `release` /
`release_after_deactivate`），没有"谁持有、还能不能借出去"的登记。于是：

- 请求结束时释放 → 与异步写线程的 pin 竞争 → `release()` 内 `std::terminate`（实测
  `0xc0000409`，紧跟 `[lmcache] store(async) ... ok=1`）；
- 推迟到"引擎空闲"释放 → 仍静默退出，且更早（`[host-context] reserved` 之后、`engine ready`
  之前），说明启动/预热路径上也有对地址的隐式依赖。

根因不是时机，而是**没有所有权模型**：申请、借用、释放、回收四种动作由不同模块各自判断。

## 状态机

地址（`KVAddressSpaceStore::Address`）只允许四种状态：

```
Free ──allocate──► Reserved ──bind──► Active ──unbind──► Retired ──refs==0──► Free
                        │                  │                 │
                        └──abort───────────┴─────────────────┘   (retire 时 refcount 归到 0 才回收)
```

- **Reserved**：已分配、尚未绑定行；只有 owner token 可写。
- **Active**：绑定到行（序列在用）；`active=true`，**任何释放请求只置 Retired 标志，不回收**。
- **Retired**：不许再绑定、不许再被选为 reuse source、不许被 `restore`/`attach`；已有读者
  （缓存写、导出）继续完成，refcount 递减到 0 才真正回收。
- **Free**：页归还物理池，句柄可被再次分配（generation++ 使旧句柄全部失效）。

关键点：**"标记释放"和"回收"是两步**。标记后 `bind()` / `select_source()` 一律拒绝（返回
"NoCapacity"/miss），而不是靠调用方自觉。

## 引用计数

每个地址一个 `std::atomic<uint32_t> refs`，新增两类租约：

| 持有者 | 何时 acquire | 何时 release |
|---|---|---|
| 序列（`SequenceState.kv`） | bind 成功 | `release_sequence_kv` 结束 |
| checkpoint（`CheckpointState.kv`） | capture/import 成功 | `release_checkpoint` |
| **缓存写 / 导出**（新增） | `publish_to_cache` 取到 checkpoint 的字节引用前 | 写线程完成（异步）后 |

规则：`retire()` 只把状态置 Retired 并把"拥有者引用"减 1；`refs==0` 时由**同一处**回收
（`retire_complete()`），并把 `generation` 递增。

## 接入点（现有符号）

1. `KVAddressSpaceStore`
   - `Address` 增加 `enum class State { Free, Reserved, Active, Retired }` 与 `refs`；
   - `can_release()` 语义改为"state == Reserved/Retired && refs == 1"（只剩拥有者）；
   - `release()` 不再 `std::terminate()`：不满足前置条件就返回 false，由调用方决定重试；
   - 新增 `acquire_reference()` / `drop_reference()` / `retire()`；
   - `bind()` 与 `execution_row()` 在 Retired 上直接拒绝。
2. `ProgramImpl`
   - `release_checkpoint`：`drop_reference(kv)`，不直接释放地址；
   - 新增 `Program::retire_history(history)`：把地址置 Retired（不再可复用）；
   - `Program::reclaim_retired()`：回收 `refs==0` 的 Retired 地址，**可以在任意时刻调用**，
     不需要"引擎空闲"这种脆弱前提。
3. `ResourceManagement::publish_to_cache`
   - 导出前 `acquire_reference`，异步写线程完成回调里 `drop_reference`。这样"请求结束就释放"
     变得安全：写线程持有租约，地址只是从 Active → Retired，不会在复制中被回收。
4. 引擎（`engine_core.h`）
   - 每个 pass 结尾调用一次 `program.reclaim_retired()`（O(Retired 数量)，通常是 0），
     取代我之前那套 `pending_address_release_` + 空闲判据。

## 为什么这样能同时解决三件事

- **不再有 `terminate`**：`release()` 前置不满足只返回 false；回收只发生在 refs==0。
- **不再靠时机**：Retired 状态让"以后不会再被借"成为结构性质，而不是"此刻没人用"的猜测。
- **第二条长请求能拿到空间**：`reclaim_retired()` 在下一个 pass 就回收，不必等引擎空闲。

## 验收

1. **生命周期单测**（新）：Free→Reserved→Active→Retired→（写线程 drop）→Free；Retired 上
   `bind()` 必须失败；`reclaim_retired()` 在 refs>0 时不得回收。
2. **稳定性**：控制台启动 → 17.8k 请求 ×3（带缓存回写）→ 163k ×2，进程不得退出；Windows
   事件日志中不得出现新的 `ninfer-serve` APPCRASH。
3. **功能**：同一 163k 前缀连发两次，第二次 `cache 163,680 (100.0%)`、TTFT ≈ 100 ms，
   `ninfer_device_kv_used_pages` 在请求间隔内回落到低位。
4. **门禁**：PPL 7.359779 逐位、dump e95cacb864d8b1ab、regressions 0。

## 落地顺序（每步都可单独验收）

1. `Address` 加 State/refs 与 `acquire/drop/retire/reclaim`（不动现有调用方，行为等价）；
2. `release()` 去 `terminate`，改为拒绝；
3. `release_checkpoint` 改 `drop_reference`；引擎加 `reclaim_retired()`；
4. `publish_to_cache` 加租约；
5. 第 4 步完成后，`retire_history` 才真正接上"最后 checkpoint 消失即 Retired"。

前 3 步不改变功能，只是把语义显式化——这也是为什么它们应该先做：有了状态机，第 4、5 步的
正确性才可证明。

## 落地记录（2026-11-23）

### 已完成：步骤 1–3

- `Address` 增加 `State`/`refs`。`create_inactive` 是唯一的 Reserved 入口，`deactivate` 回
  Reserved，`retire` 进 Retired，`release_address` 回 Free 并 `generation++`（旧句柄全部失效）。
- `release()` 不再 `std::terminate()`：前置不满足返回 false。`release_address` 只在
  `can_release_pages` 整段预检通过后才逐页释放，拒绝时头部保持完整。
- **`~KVHistory` 改为"标记结束"**：先 `release_after_deactivate`；失败则 `deactivate` + `retire`，
  绝不 terminate。原实现的 `0xc0000409` 就出在这里——`release_after_deactivate` 返回 false 时
  `std::terminate()`（`checkpoints.cpp` 旧 41/44 行）。
- **`release_checkpoint` 的前置不再要求"地址此刻可释放"**，只要求 checkpoint 本身可丢
  （`state_store->can_release_checkpoint_owner`）。这是同一扇门的另一侧：旧谓词让一个被钉住的页
  导致 `release_checkpoint` 拒绝，而 `resource_manager` 把拒绝当致命错误抛异常。
- 引擎每趟调度调用 `reclaim_retired()`；未退役时只做一次计数器比较。
- 新增 `test_kv_lifecycle`（`tests/models/qwen3_5/test_context_store.cpp`，5 组）：
  空地址直接释放 / Retired 拒绝 bind / checkpoint 要求拒绝退役 / **被 Host extent 钉住时 retire
  成功、回收延后、解除钉住后下一趟回收** / 第二持有者阻止退役。
- 门禁：PPL 7.359779 逐位、dump e95cacb864d8b1ab、regressions 0。

### 更正：步骤 4 的前提不成立

`export_checkpoint` 在**调用线程**上把全部 KV 字节（`memcpy` 或 `copy_to_host`）拷进
`CheckpointExport::kv_bytes`，`publish_to_cache` 再把这份**拷贝**交由异步写线程
（`generation_service.cpp` 的 `Job.checkpoint`）。写线程从不读地址空间、也不 pin 页。
因此"导出前 acquire_reference、写线程完成回调里 drop_reference"没有对应的危险可防。
第 4 步不实施；`refs` 目前由所有者独占（create 时 1、retire 时 0），保持普通整数即可，
所有变迁都在调度线程上。

### 未闭环：真正的回收障碍是 Host extent 钉页

`host_kv_store.h:204` 对 extent 的每个成员页 `pin_source()`；`logical_kv_store.h:616` 在
`source_pins != 0 && references == 1` 时拒绝释放。所以只要该行的 `live_host_extents_` 还在，
`truncate_inactive_prefix` 一页也放不掉——与实测 `truncated 2558 -> 0` 一致。

状态机让这件事不再致命（地址留在 Retired，下趟重试），但**不会自己把空间还回来**。验收 3 需要
下一步：

1. 在地址或 history 上记下最后绑定的行（`deactivate()` 现在会 `row.reset()`，必须先保存）；
2. `retire`/`reclaim` 在确认该行已无活序列后调用 `release_live_host_extents(row)`；
3. 用 `trim-probe` 复测同一 163k 冷请求间隔内 `ninfer_device_kv_used_pages` 的回落。

## 实测（2026-11-23，新构建 074fa75d）

环境：`--max-context 256000 --kv-capacity 256000`（4000 页）、`--spec none`、
`--prefill-chunk 2048`、LMCache 已 attached（WSL 内 `--host 0.0.0.0`，L1 4 GB，chunk 128）。

| 步骤 | 结果 |
|---|---|
| 基线 | `ninfer_device_kv_used_pages 0` |
| 请求 1：p163_1，prompt 163,687 | **缓存命中 100%**（`cache_n 163680`，restore 10.3 s），总 15.7 s，页占用 **2559** |
| 请求 1 之后引擎日志 | **没有** `[kv] history end` 行；`[host-kv] extents=3 pinned=2609 MiB` 仍在 |
| 请求 2：p163_2，另一个 163k | LMCache miss → 真实 prefill；`--kv-evict off` 时 **105 s** 后、`auto` 时 **212 s** 后 HTTP **400** |

400 的内部原文（日志）：`KV residency cannot grow this lane beside the lanes already resident:
growth_pages_needed≈2558 capacity_pages=4000 live_pages=…`，对外被归一为
`context length exceeded`。**没有挂起、没有崩溃、Windows 事件日志无新的 APPCRASH。**

三条结论：

1. `bdf73451` 的守卫是有效的：过去"prefill 停在半路、6 分钟零进展"的场景，现在是有界的 400。
2. 状态机解决的是"崩溃/静默丢失"，**不解决"保留前缀占着设备页不放"**：请求结束后历史仍被持有
   （续写/缓存所有者），于是根本不 retire，回收链路一次都没触发。即使触发了，
   `live_host_extents_` 仍会钉住这些页——`pinned=2609 MiB` 在请求结束后依旧存在。
3. `--kv-evict auto` **不会释放保留检查点**：淘汰只作用于活动窗口与 Host 空间；
   `release_stale_checkpoints()` 目前只在 Host 预留失败时被调用（`sequence.cpp:1450`），
   设备侧的准入路径没有对应动作。

因此验收 3 仍未闭环，且修复点已经明确（不是再猜）：

1. `LiveHostExtent` 记录它钉住的是**哪个地址**（index + generation），新增"按地址释放"入口；
2. `retire()` 交出地址时，由 ProgramImpl 释放该地址的 extent —— 不再按行去猜（行会被复用）；
3. 准入失败前先尝试 `release_stale_checkpoints(0)` 并重算 `kv_residency_fits`，成功则继续；
   仍不成功时给出真正的容量说明，而不是归一成 `context length exceeded`；
4. 验收：同一会话里 req1 之后 req2（不同前缀）应直接成功，页占用在 prefill 期间由回收让出。

## 更正并深一层：卡点不在 extent 钉页（2026-11-23 13:00）

上面那段把 `live_host_extents_` 当作卡点是**推断，不是证据**。加了按环境变量开启的取证
（`NINFER_KV_LIFECYCLE_PROBE`：`[kv-probe]` 打印检查点为何不可释放，`[rc-probe]` 打印回收候选
与报价）后跑同一场景，事实是另一条链：

- 400 正文（客户端原始返回）：`the KV pool is full and nothing can be reclaimed for this step:
  needed_main_pages=30 … resident=3998 of 4000 page(s)`。
- `[rc-probe]`：两次尝试都是 `victims=1`；第一次 `demotions=1 releases=1`（只让出 1 页，
  shortage 31→30），第二次 `demotions=0 releases=0`。
- **`[kv-probe]` 一条都没有**——`can_release_checkpoint` 从未对保留前缀返回过 false，
  也就是说"页被钉住"这一层根本没被问到。
- 真正的门在 `released_resources`（`reclaim.cpp:487`）：只有
  `history.references == history.checkpoints.size()` 时才会为这段历史记账，而 `references` 就是
  `KVHistory` 的 `use_count`（`reclaim.cpp:195`）。保留前缀被一个**仍在的序列**持有时，
  `use_count` 恒大于该历史的 checkpoint 数 → 这段历史的页永远算不出可释放量 → 报价为空 → Blocked。
- 空闲回收也没兜住：`idle_reclaim_ms=60000`，但两分钟空闲内 `kv_idle_reclaims_total` 保持 0，
  页占用仍是 2558 —— 该配置下 `idle_shared()` 没有把这个 owner 当候选，降级路径没被走到。
- 另有一处独立缺陷：放松保留的重试被 `cursor.victims.empty()` 门控（`resource_manager.h:1020`）。
  手里只要有一个"给不出页"的候选，可选的保留前缀就永远进不来。

所以是三个可分离的缺陷，**没有一个是状态机能修的**：

1. `released_resources` 的"整段历史"判据把被序列持有的保留前缀排除在可回收之外；
2. 放松保留的重试触发条件过窄（应为"本轮没有产生任何动作"，而不只是"候选为空"）；
3. 该配置下 `idle_reclaim` 不生效（候选集为空）。

正确的下一步形态：池满且唯一的大块占用是"空闲但仍在的序列"时，引擎需要一条显式的**让空闲序列
退场**动作（abort → 地址 retire → 下趟 reclaim 归还页），再把上面 1、2 两处判据改到"允许延后
回收"的语义。验收仍是同一条：req1 结束后立刻发 req2（不同前缀）必须成功，而不是 400。

## 第二轮实测：恢复路径的 pin 已修，报价仍只有 1 页

**已修**：`import_checkpoint_current_layout` 的默认 `pin_pages=true` 会让每一次 LMCache 恢复都
保留创建者引用并给地址打上 checkpoint 保护。恢复调用点（`resource_manager.h` 的 restore）改为
`pin_pages=false`，理由写在调用处：导入在调度线程同步完成，`choice.source.private_points` 紧接着
就登记了所有者，`prune()` 不会把它回收；权威字节在缓存服务器上。实测对比：

| | 修复前 | 修复后 |
|---|---|---|
| `[kvt] import refs` | 不打印（只在 pin=false 时打印） | `addr_refs=1 src_pins=0 truncate=1` |

**仍未闭环**：`[rc-probe] scan pages=2558 target=0 unique=1 shared=2557` 没有变化。
`inspect_suffix` 只在 `facts.references == 1` 时计数（`reclaim.cpp:154`），而这段历史在请求结束后
仍有**两个检查点**——恢复导入的那个，和 finish 时的捕获。于是报价只能覆盖"另一个检查点 frontier
之上"的那一段，实测恰好 1 页；`[rc-probe] reclaim … victims=1` 也只提供了一个候选，
`physical_owner_sets` 的 `permitted()` 因此无法把整组凑齐。

下一步（精确，不再是方向性猜测）：释放时按**物理所有者组**把同一 history 的检查点整组给出，而不是
只给一个候选；或者在 finish 捕获与恢复导入之间只保留一个检查点（同 frontier 的两次发布已经在
`publish_to_cache` 里被去重过，但那是缓存侧的，不是检查点侧的）。




