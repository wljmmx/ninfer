# 需求侧降级：让一条长前缀和新请求共存

状态：未实现。起点提交 `bdf73451`（master，已推送，门禁通过）。

## 要解决的问题

256K + MTP 下 KV 是"主 KV + 草稿 KV"两份，24 GB 卡上只能同时容一个 163k 级前缀。
恢复出来的前缀（2558 页主 + 2558 页草稿）常驻在 Device，第二个同尺寸请求会：

```
[lmcache] restore declined: 2558 page(s) + 585 reserved, 2123 of 4682 free
→ 冷 prefill 163k（约 80 s）→ 池子正好占满 → 还差 1 页 → 立刻 400
```

`bdf73451` 已经保证它不再挂死（`requests_paused` 恒 0），但调用方白等 80 秒，且本可以走恢复。

## 方案

需求侧（admission）主动把保留前缀降级到 Host，再决定是否够用。

### 1. `src/runtime/engine/context_cache/resource_manager.h`

现状：`demote_idle(Program&, std::span<const Handle>)` 已实现完整动作链
（选句柄 → `plan_reclaim` → `begin_kv_batch` → `start_demote`），但只由空闲扫描调用，
且 `idle_shortage()` 一次只挑一类 KV（main 优先，否则 backend）。

改动：

- 抽出候选集合：`std::vector<Handle> retained_demotion_candidates(Program&) const`
  —— 取 `retained_prefixes(program)` 里的 `optional`（`pinned` 必须排除，注释见 V10-02），
  并像 `demote_idle` 一样用 `find_shared_by_handle` 把同一 owner 的 `points` 一起带上
  （appendix IC：缺了它们模型会拒掉几乎所有 page group）。
- 新增公有入口：

```cpp
// Demand-time counterpart of the idle sweep: give back the retained prefixes' Device pages now,
// because a request is about to need them. `DemotionClass::Auto` demotes the main run first and the
// draft run in a second call (idle_shortage's rule); the caller retries until it returns Stopped.
enum class DemotionNow { Stopped, Started, Blocked };
DemotionNow demote_retained_now(Program& program);
```

实现直接复用 `demote_idle` 的骨架，只是把 `idle` 换成
`retained_demotion_candidates(program)`，并把 `IdleDemotion` 映射到 `DemotionNow`
（`Nothing → Stopped`，`Deferred/Blocked` 原样）。

注意：`demote_idle` 里的 `prune(program)` 与 `enabled_` 判定要保留；没有
`checkpoint_cache_` 或 arena 未启用时返回 `Stopped`，不要抛。

### 2. `src/runtime/engine/engine_core.h` — `try_admit_one`

在真正 `reserve`/绑定之前插一次容量预检：

```cpp
// A prompt that cannot fit beside the retained prefixes should say so before it pays for a
// prefill, and the retained prefixes are the thing to give back: LMCache already holds their bytes,
// the local copy is only the fast path.
const auto pages = kv_pages_for_frontier(prompt_tokens);
if (pages > instance_.program->checkpoint_pool_headroom().available_pages) {
    for (int pass = 0; pass < 2; ++pass) {   // main, then backend
        if (resources_.demote_retained_now(*instance_.program) !=
            typename ResourceManagement::DemotionNow::Started) { break; }
    }
    if (pages > instance_.program->checkpoint_pool_headroom().available_pages) {
        complete_error(request, std::make_exception_ptr(RequestError(
            RequestErrorKind::ContextLengthExceeded,
            "the prompt needs " + std::to_string(pages) + " page(s) but only " +
            std::to_string(instance_.program->checkpoint_pool_headroom().available_pages) +
            " are free even after demoting the retained prefixes; lower max_context")));
        return false;
    }
}
```

`kv_pages_for_frontier()` 与 `checkpoint_pool_headroom()` 都在 `program.h` 已经可见。

### 3. 验收（照既有纪律）

- 同一 163k prompt 连发两次：第二次应当 `restore 命中`（约 10 s），而不是 80 s 后 400；
  若仍不够，则必须**准入即拒**（秒级 400），`ninfer_requests_paused` 恒 0。
- 引擎存活：拒绝后普通小请求仍 200（本轮已验证该性质）。
- 门禁：`py -3 H:\TeleAgentWorks\.temp\kv\regress.py --tier fast`
  → PPL `7.359779` 逐位、dump `e95cacb864d8b1ab`、regressions 0。
- 构建注意：改 `engine_core.h` 后 `ninja` 不传导头依赖，需删
  `build-win\src\runtime\**\*.obj` 再编；门禁用的是 `ninfer.exe` / `ninfer-perplexity.exe`，
  必须一并重链（它们链 `infer_engine.lib`）。

## 本轮已经落地的两处相关修复（不要回退）

- `dc716904`：恢复暂存 `bad_alloc` → 该片失败 → `restore failed` → 冷 prefill，
  不再把 worker 打死（原先此后所有请求 503）。
- `bdf73451`：单请求且无可暂停对象时不再 `pause_resident(self)`，改带数字拒绝。
