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
