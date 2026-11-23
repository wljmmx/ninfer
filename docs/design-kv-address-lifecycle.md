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

## 第三轮实测：两处已改，剩下一处是所有权重复

**已改（提交 a22db520 + 本轮）**：

1. 恢复导入 `pin_pages=false`（见上表）。
2. 放松保留的重试条件从"候选为空"改为"严格顺序没有产生任何动作"
   （`resource_manager.h` 的 `reclaim`）。本轮没有观测到它被触发（修复 1 之后候选从 1 个变成 3 个，
   严格顺序直接给出了报价），保留它作为兜底。

**严格复现仍然失败**（重启清空 → 只恢复一个 163k 前缀 → 发一个缓存外的冷 163k）：

```
[rc-probe] reclaim main=31 … victims=1 relaxed=0 frontiers=[163694 ]
[rc-probe] scan main=1 pages=2558 target=0 unique=1 shared=2557 inactive=1 row=-1 committed=163694
[rc-probe] history refs=2 ckpts=2
→ HTTP 400, resident=3998 of 4000
```

`refs=2 ckpts=2` 是关键：同一个 `KVHistory` 上有**两个检查点**——恢复导入的那个，和 finish 时捕获
的那个。两者指向同一个地址，所以页引用是 2；释放其中任何一个，另一个仍持有全部 2558 页，
`inspect_suffix` 因此只把最上面 1 页算成可释放。

这不是状态机或记账算术的问题，而是**所有权重复**：`checkpoint_import.cpp` 的注释写明了约定
（"the caller either adopts this into a record, which takes the reference, or hands it to a request,
which takes the ownership"），恢复路径却把导入的检查点留在了 owner 里，请求结束时又捕获了一个
覆盖同一前缀的检查点。

修复需要按这条约定做：请求接管恢复出来的前缀时，源检查点应当被**消费**（或由 owner 释放），而不是
与 finish 捕获并存。两种可选实现都要动所有权账：

- 让恢复源成为 consuming（`consumable()` / `take_private` 路径），finish 只留下一个检查点；
- 或者在 `detach_checkpoint` 成功后释放同一 history 上 frontier 不大于新的、且无 owner 引用的
  重复检查点（需要 Program 暴露"同一 history 的检查点集合"，并确认 manager 里的引用计数不会因此
  留下失效句柄）。

验收不变：恢复命中之后立刻发一个缓存外的冷 163k，必须成功而不是 400。

## 第四轮实测：整组候选已接上，卡点落到"第二个地址"

按上面的第二种思路，新增只读接口 `Program::checkpoints_sharing_history()`，`reclaim_order` 用它把
同一 history 的检查点成组给出（兄弟候选遵守同一套 retention 过滤器）。`checkpoints_sharing_history`
`[kvt] import refs` 保持 `addr_refs=1`，但严格场景仍 400，而且数字变了：

```
[rc-probe] reclaim main=31 … victims=1 relaxed=0 frontiers=[163694 ]
[rc-probe] scan … unique=1 shared=2557 inactive=1 row=-1 committed=163694
[rc-probe] history refs=1 ckpts=1        ← 这一次历史上只有一个检查点
```

`refs=1 ckpts=1` 却仍然 `shared=2557`，说明第二份页引用不在检查点账上，而在**另一个地址**上：
恢复前缀被请求绑定时走的是 active prefix view（`commit_active_prefix_view`），它为目标地址
`retain_reference()`，而目标地址不在 `KVHistory.text` 里——检查点被释放时只有 `KVHistory` 的地址
会退引用，这个视图地址就留了下来。

所以下一处要查的是 prefix-view 的目标地址在请求结束时的去向（`detach_checkpoint` 把
`state.kv->text` 设成哪个地址、另一个地址由谁释放）。它比前面的修复更靠所有权核心，需要先把
"哪个地址是 history 的、哪个是视图的"在日志里打印出来，再决定释放点。

## 第五轮实测：两个地址，一个没人释放

`debug_page_holders()`（只读）在拒绝现场打出了答案：

```
[addr-probe] scan slot=0 pages=2558 committed=163680 checkpoint=0      active=0 state=1 refs=1
[addr-probe] scan slot=1 pages=2558 committed=163694 checkpoint=163694 active=0 state=1 refs=1
[addr-probe] scan holders=2
```

两个**已占用的地址槽**持有同一批 2558 个逻辑页：

- slot 1 是 history 的地址（`committed=163694`，带 checkpoint 保护），正常；
- slot 0 是恢复导入出来的那个地址（`committed=163680`，保护已被清成 0），请求结束后**仍然是
  Reserved、仍然持有 1 份页引用**，没有任何检查点再指向它。

所以第二份页引用不在检查点账上，而在一个"谁也放不掉"的地址上。`~KVHistory` 也不够：它只释放
`history->text`，而这个地址已经不在任何 history 的 `text` 里了。

下一处要读的是绑定事务里前缀视图的归属：`binding.cpp` 的 `borrow_text` 分支让
`binding_history` 直接借用检查点的 history（其 `text` 是 slot 0），而视图的 destination（slot 1）
记在 `reserved_kv` 上；两条记录谁在结束时释放谁，就是那次要读清楚的一段。释放点选定后，验收仍是
同一条：恢复命中后立刻发一个缓存外的冷 163k 必须成功。

## 第六轮实测：`release_checkpoint` 整个运行里一次都没被调用

在 `release_checkpoint` 里加了取证（打印释放后的 `history use_count` 与 `text` 地址页数），然后跑
严格场景——**一行都没打出来**。结合前一轮的 `holders=2`，结论很明确：

- 这次请求的绑定走的是非借用分支（`prepare_prefix_fork`，源是 slot 0、目标是 slot 1），目标地址
  因共享同一批逻辑页而 +1 引用；
- 源检查点本应在自己不再被需要时被释放，从而让它的 history 析构、把 slot 0 交回；
- 但实际上**没有任何代码路径调用 `release_checkpoint`**，源检查点连同 slot 0 一直活到进程结束。

试过的两条改动都没有命中并已撤回（不留未验证代码在热路径）：在 `commit_binding` 里释放
`retired_checkpoints`（该场景下这个列表是空的）、以及此前按地址归属释放 Host extent。

下一步要查的是"源检查点的引用计数为什么一直不为零"：`release_unreferenced` 在 `references(handle)`
非零时直接返回，而 `references` 汇总目录与 `waiting_.held`；需要打印该 handle 在 catalog / owners /
waiting 三处的命中情况，才能判断是谁把它钉住不放。

## 第七轮实测：连"尝试释放"都没有发生

给 `release_unreferenced` 加了 `[ref-probe]`（打印 owners / waiting / shared 三处命中），再跑严格
场景：**一条都没有**。也就是说源检查点既没有被直接释放（`released ckpt` 无输出），也没有任何一次
"因仍有引用而跳过"的尝试——它是直接从 manager 的账上消失的。

到这里可以确定的三件事（都有日志或代码位置支撑）：

1. 恢复出来的前缀在请求结束后留下两个地址槽（slot 0 `committed=163680` 无保护、slot 1
   `committed=163694` 带保护），二者共享全部 2558 个逻辑页；
2. 页引用因此是 2，`inspect_suffix` 只把最上面 1 页算成可释放，报价因此是 1 页；
3. 没有任何一条路径尝试释放 slot 0 的所有者。

剩下要回答的是"谁把源检查点从账上抹掉了"——`forget()` 会 `index_.erase` 而不释放，调用点是
`binding_started(retired)`、`prune()` 与 `adopt()`。下一步要在 `forget()` 里打印调用来源与该
handle 是否仍被任何 owner 持有，才能定位到具体的那一次。

本轮所有新增代码都是环境变量开启的只读取证（`[kv-probe]` / `[rc-probe]` / `[addr-probe]` /
`[ref-probe]`），热路径上没有任何未验证的行为改动；两条试过的定点修改（`commit_binding` 释放
`retired_checkpoints`、按地址释放 Host extent）都因未命中而撤回。

## 第八轮：整条因果链闭合——孤儿地址让"借用"永久失效

`forget()` 与 `waiting_` 两条线都没有输出（`forget-probe` 零条、`retain_source` 在冷路径上根本没被
调用），说明源检查点不是被"忘记"或"等待表"丢掉的。但这一轮跑出了更关键的现象：

```
p163_1（恢复命中）: 13.6 s   → device_kv_used_pages 2559
p163_1（同一前缀再发）: HTTP 400, resident=3998 of 4000
```

**同一个前缀的第二次请求也失败**。原因是 `binding.cpp:170-188` 的 `movable()`：

- 复用前缀时，若源地址可"借用"（`borrow_text`），请求直接接管那个地址，不新增页；
- `movable()` 要求尾页 `address_references == 1` 且无 source pin；
- 而上一轮留下的孤儿地址让每个页都有 **2** 份引用（两个地址共享同一批逻辑页），
  `movable()` 因此恒为 false；
- 借用被拒绝 → 每次复用都走 `prepare_prefix_fork`，**复制一份完整的 2558 页** → 24 GB 卡上
  163k 前缀的复用需要两倍容量，必然 400。

所以这不是"回收算得少"的问题，而是：**孤儿地址 → 引用翻倍 → 借用永久失效 → 复用退化成复制**。
也解释了为什么池子一旦被一个 163k 前缀占满，之后连相同前缀都进不来。

修复的唯一入口仍然是那个孤儿地址（slot 0）。现在它有两个明确判据可用于验收：修复后
`[addr-probe] holders` 应回到 1，且 `movable()` 恢复为真、复用不再需要第二份页。

## 第九轮：不是同一个 history，而是同一批物理页

`debug_checkpoint_table()` 打出了台账本身，纠正了上一轮的判断：

```
[ckpt-table] slot=0 frontier=163680 use=1 pins=0 pages=2558 active=0 releasable=1
[ckpt-table] slot=1 frontier=163694 use=1 pins=0 pages=2558 active=0 releasable=1
```

两个检查点**各有自己的地址**（`use=1`，即不同的 `KVHistory`），却因为 fork 复制了目录而共享
同一批逻辑页。所以"按 history 分组"永远只能找到 1 个候选——改成按**共享物理页**分组后：

```
[rc-probe] reclaim victims=2 … plan allowed=1 … releases=0
```

候选变成 2 个，但 `allowed` 仍是 1：`allowed` 那一步要求 manager 侧"仍有引用"，而这个伙伴只被
Program 记账、不在 manager 的目录里。把它放宽（只要求 valid + 未排除）确实让报价出现了
（`allowed=2 owners=3 releases=3`），但提交时被一致性校验拒绝：

```
HTTP 500  "quoted cache release changed before commit"
```

比原来的 400 更糟，因此**该放宽已撤回**，状态码回到 400。

结论：伙伴检查点缺少 manager 侧的记账，是比"分组"更靠上层的问题——它必须先在 owner/catalog 里
被正确登记（或由 `abandon` 真正释放），否则一切按组释放都会在提交校验处被拒。下一轮从
`replace_points` / `adopt` 的登记行为入手，而不是继续改回收侧的过滤条件。

## 第十轮：按设计把回收接回生命周期（已闭环）

上一轮已经把结论说清楚了：manager 的目录不是"能否丢弃检查点"的权威，Program 的所有权
（`pins`、state owner）才是。设计文档里回收本来就该是三步——**标记 → 回收 → 度量**，而不是
"预先报价 → 准入校验 → 提交"。回收路径之前一直走后者，所以一个页被两个地址共享时报价永远只有
1 页。

实现（`Program::retire_checkpoint_group`，`checkpoints.cpp`）：

```cpp
const auto before = physical_usage().occupied.main_kv_pages;
for (const auto handle : handles) { if (valid_checkpoint(handle)) release_checkpoint(handle); }
reclaim_retired();
return before > after ? before - after : 0U;
```

接入点放在 `ResourceManager::reclaim` 的兜底处（严格顺序与放松顺序都没给出动作之后）：把候选与
**共享物理页**的伙伴合成一组，跳过 pinned 与 excluded，整组交给上面这个操作，然后按"实际归还的
页数"返回 `Changed`。`release_checkpoint` 自己会拒绝仍在使用的检查点，所以这里不需要第二套授权
判断——这正是设计里"标记释放永远允许、回收延后"的语义。

实测（`--kv-capacity 256000`，4000 页）：

| 场景 | 改前 | 改后 |
|---|---|---|
| 同一 163k 前缀第二次请求 | HTTP 400（43 s），或复制第二份 2558 页 | **3.0 s 命中，页占用保持 2559** |
| 命中后发一个缓存外的不同 163k | HTTP 400（`resident=3998 of 4000`） | **128 s 跑完，1605 tok/s，无 400** |

日志侧同时出现 `[kv-probe] released ckpt 2: history use_count=2 text_pages=2558`，说明重复的检查点
确实被释放、地址随后被回收，而不是靠报价"算出"了可释放。

门禁：PPL 7.359779 逐位、dump e95cacb864d8b1ab、regressions 0。












