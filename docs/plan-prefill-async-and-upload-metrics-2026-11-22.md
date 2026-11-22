# Prefill 时间去向、异步化与真实上传带宽指标

> **更正（2026-11-23）**：本轮实测用的 `qwen3_8_27b_v3` 是**稠密**模型，不是 MoE。因此：
> - `[graph] skipped: layer needs host fetch` 不是"专家池装不下全部层"，而是 `every_layer`
>   要求存在专家池、稠密模型永远没有池 → 整步图被误关。已在 `5a32c297` 修复，实测 boundary
>   6.8 s → 3.2 s，图已成功捕获（`[graph] capture measured -30.7 MiB`）。
> - `submit` 27 s **没有**随图开启下降（27.1 s），所以它不是逐 kernel 启动开销，而是捕获区域
>   之外的 host 工作、且随上下文增长。这是下一项要查的。
> - 页面「上传带宽」在稠密模型上没有"专家权重"含义；引擎完全不发布该量。要做真实指标，应定义
>   为 **H2D 字节/秒**（KV 页恢复 + 权重上传），而不是"专家/层"。
> - 「计划四」中关于 split 层与专家预取的条目只对 MoE 工件成立，稠密工件不适用。

起点：`8e7427bc`（master）。数据来自 2026-11-22 23:16–23:23 的一次 163k 冷 prefill
（`--spec none`、256K、`--kv-capacity auto`、RTX 4090、qwen3.8-27B）。

## 实测：142.9 s 里 GPU 只占一半

```
req#2 done | prompt 163,702 | cache 0 (0.0%) | TTFT 2m 21.0s
engine-wall 2m 18.6s
  boundary  40.1 s   ← checkpoint 捕获 + 导出 + LMCache 回写（host 804%）
  submit    27.2 s   ← 提交开销
  devwait   1m 13.1 s ← GPU 计算
prefill 1.63k tok/s（端点自报），实测全程 1181 tok/s
```

吞吐曲线（5 s 窗口，chunk = 2048 token）：

```
2.87k → 2.05k → 1.64k/2.05k → 1.23k 稳定
对应每窗口 7 → 5 → 4/5 → 3 个 chunk
```

**结论：单 chunk 变慢是 attention 随前缀线性增长（真实算力成本），但 40.1 s 的
boundary 和 27.2 s 的 submit 是流水线堵塞，不是算力。**

## 计划一：prefill 流水线解耦

1. **异步回写（先做，已有开关）**：`NINFER_LMCACHE_ASYNC_WRITE=1` 已实现（把 checkpoint
   字节交给 writer 线程，请求只付一次 memcpy）。当前默认关闭。验收：同一 163k 冷请求，
   `boundary` 应从 40.1 s 降到亚秒级，`req#2 done` 总时长下降约 40 s。
2. **捕获移出请求路径**：现在 checkpoint 捕获在 prefill 尾部同步完成（`capture-pending 1`
   期间 host 804% 持续 40 s）。`prefill.cpp` 的 `advance_prefill` 在 `capture_pending` 时
   直接返回，说明捕获本身已经能与后续 pass 分离；需要的是让"请求完成"不等捕获，把捕获交给
   引擎的后台 pass（`context_owner_` 已有这条通道）。
3. **submit 27.2 s**：27.2 s / ~80 chunk ≈ 340 ms per chunk，明显高于一次 kernel 提交。
   下一步在 `PrefillStepResult` 的 submit 相位打点里细分（CUDA Graph 录制 vs 真实提交），
   确认是否每 chunk 都在重新构图（`[graph] skipped: layer needs host fetch` 曾在启动
   日志出现过——一旦构图被跳过，每 chunk 都要逐层提交，这正好解释 340 ms）。
4. **验收口径**：同 prompt 冷跑三次取中位数，要求
   `boundary < 2 s`、`submit < 5 s`、`engine-wall` 下降 ≥ 30%，且门禁
   PPL 7.359779 / dump e95cacb864d8b1ab / regressions 0 不变。

## 计划二：上传带宽的真实指标

页面「上传带宽」= 系列 `upload`，单位 MB/层，语义是 MoE 专家权重每层 host→device 流式上传。
**当前引擎不发布**，只有 demo 模拟实例有值（console.html:1023 注释明确写了）。

要变成真实指标：

1. 引擎侧在专家流式路径上累计 H2D 字节（`expert cache` 的窗口上传），发布两个量：
   `ninfer_expert_upload_bytes_total`（counter）与 `ninfer_expert_upload_mib_per_layer`（gauge，
   最近一个统计窗口）；
2. `/monitor/snapshot` 带上 `expert_upload`，控制台 `liveSample()` 把 `s.upload` 从"留空"
   改成读它（单位从 MB/层换算成 GB/s 需要除以窗口秒数）；
3. 验收：prefill 期间该值非零且与 `nvidia-smi dmon` 的 H2D 趋势一致；idle 时为 0；
   无 MoE 模型时为"不适用"而不是 0。

## 计划三：缓存释放补完（承接前一轮）

决定性证据：

```
[trim-probe] truncated 2558 -> 0 page(s), pool allocated 2559 -> 2558
```

逻辑地址截断成功，物理池只回 1 页；而 abort 走 `clear_lane → release_sequence_kv`
（sequence.cpp:568-587）时一次掉 2112 页。差别在于 `release_sequence_kv` 在
`refresh_history_requirements` 之前调用了：

```cpp
const auto row = text_kv_addresses->bound_row(sequence.kv->text);
release_live_host_extents(row);        // live_host_extents_[row] → HostKVExtentStore::release
```

`live_host_extents_` 以**行号**为键（program_impl.h:808），记录该行 demotion 产生的 extent，
登记在 demotion 时（`demote_absent_pages_to_host`）。checkpoint 释放路径拿不到这个行号，
所以 Host extent 一直挂着，物理页无法回池。

下一步（按可行性排序）：

1. 在 `CheckpointState`/`KVHistory` 里记下"创建该地址时的行号"，`release_checkpoint`
   释放最后一个 checkpoint 时用它调用一次 `release_live_host_extents(row)`；
2. 若无行号（地址由恢复创建），则退化为"该地址上没有 active 序列时，释放所有未绑定行的
   extent"——需要 `LiveHostExtent` 增加 address 归属字段才能安全区分；
3. 验收：`[trim-probe] truncated N -> 0, pool allocated X -> X-N`，且第二个 163k 请求
   `cache 163,680 (100.0%)`、TTFT ~12 s、`requests_paused` 恒 0。

## 计划四：无图时的批量提交兜底（本轮新增要求）

代码事实（`ffn.cpp` / `model_instance.cpp`）：

- 整步 CUDA Graph 在 24 GB 卡上**结构性关闭**：`model_instance.cpp:2157` 要求专家池装下全部层
  （`every_layer`），12 GB 池做不到，于是 `[graph] skipped: layer needs host fetch`，
  `graph_ok = false`。
- **按层图已经存在**：`LayerGraphs`（`ffn.cpp:814-926`）默认开启（`NINFER_LAYER_GRAPH` 未设即 on），
  按 bank + 参数地址做 key 缓存 capture，命中就 `cudaGraphLaunch`；在 `ffn.cpp:2546`
  （全驻留快速路径）与 `ffn.cpp:2670`（一般 MoE 路径）被调用。
- **缺口**：需要 host fetch 的 split 路径无法 capture（capture 内不允许 stream sync），
  该层只能逐 kernel 提交，而且 fetch 在关键路径上。实测 `submit 28.3 s / ~80 chunk`
  ≈ 350 ms/chunk，与"每 chunk 等一次专家上传"吻合。

因此"无图时的批量提交兜底"不是再造图，而是：

1. **把 split 层切成两段各录一张图**：fetch 之前与之后分别 capture，fetch 留在两张图之间，
   每层从"逐 kernel"变成"3 次提交"；
2. **让 fetch 离开关键路径**：`ExpertPrefetchQueue`（`expert_prefetch.cpp:218`）已存在，
   由控制器按 `value > 0.5` 启用（`ffn.cpp:2458`）；先确认 27B 这套权重上它是否被打开，
   打开后上传与计算重叠，350 ms/chunk 应显著下降；
3. **取证口径**：`NINFER_LAYER_GRAPH_DEBUG=1`（`[graph-key]`）与
   `NINFER_LAYER_GRAPH_REPORT=1`（captures/replays）——`captures` 持续增长而 `replays` 不动
   说明 key 每次变、图没命中，问题在参数地址而不是图；
4. **验收**：同一冷 163k，`submit` 从 28.3 s 降到 < 8 s，`prefill` ≥ 1.9k tok/s，门禁三项不变。

## 已知的其它结论（不要重复踩）

- 容量方案不成立：`--kv-capacity 327680`（=5120 页，2×2558）被拒
  `kv_capacity is outside the usable range`——24 GB 卡装不下两个 163k 前缀。
- 逐页 demotion 不可行：一次事务只搬 1 页（2559 → 2558）。
- 单请求不能自我暂停：已在 `bdf73451` 改为带数字拒绝。
