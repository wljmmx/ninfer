# v12：长上下文 TTFT、预填充稳定性与 Qwen4Exp 接线

本文件是 v12 的执行计划与跟踪台账。每一项都有**问题、做法、判据（DoD）、证据**四栏；状态只有四种：
`未开始` / `进行中` / `已完成（附证据）` / `阻塞（附原因）`。

## 0. 依据与范围

- **实测依据**：2026-11-24 11:36–11:54 的真实前台运行日志
  `H:\TeleAgentWorks\.temp\kv\console-20261124-503.log`（27B gguf、`max-context 262144`、
  并发 4、pending 16、chunk 1024、`rk4v4-e8`、MTP7、`kv-capacity auto`），发布 `1663-g841b532a`。
- **代码依据**：`src/ops/softmax_attention`（dense/sliding_window/qsa）、
  `src/models/qwen4_exp/*`、`src/models/qwen3_5/execution/{parameters,expert_stream,ffn}.cpp`、
  `src/runtime/engine/model_instance.cpp`、`src/core/{expert_cache,vmm_pool}.cpp`。
- **范围**：本计划只管三件事——Qwen4Exp 的接线、TTFT/预填充的调度与内存账、长上下文的注意力与 KV。
  503 雪崩本身不在此列（见 §1.5，已由 954dd187 / b7e8b56d 修复）。

## 1. 已核实的事实

### 1.1 FlashAttention2 的覆盖

| 路径 | 结论 | 证据 |
|---|---|---|
| 稠密 27B | 已覆盖，且分块 + 在线 softmax + split-KV 齐全 | prefill：`dense/context`（`PageMapped=true`、页 64、`SplitKv` 路由，`key_block` 32/64、split 上限 32–64 按 `max_context` 分档）；decode：`dense/causal_cache`（每 warp 16 行 query 走 QK→在线 softmax→PV，`fp8` 与 `k8v4` 各一套） |
| MoE 35B | 已覆盖（同一套注意力） | MoE 只替换 FFN，注意力仍走 `dense/context`；同一二进制对照：27B pp4096→pp32768 −18%（3,215→2,643），35B −94%（11,434→724.5），说明 MoE 的长上下文问题不在注意力 |
| Qwen4Exp | 不走 FA2 | 注意力是 QSA（`indexer` + `sparse_gqa`），`qwen4_exp/execution/qsa.cpp` 只产出规划几何，`startup.cpp` 已接 KV 计划；sparse 主体与层执行未完成 |

**结论**：把 FA2 再移植一遍没有收益；长上下文要么落在调度/内存账上（§1.2–1.4），要么在 Qwen4Exp 的 QSA 上。

### 1.2 TTFT 的真实构成（日志）

| 请求 | prompt | TTFT | queue | prepare | 纯 prefill |
|---|---:|---:|---:|---:|---:|
| req#3 | 17,622 | 15.5 s | 2.2 s | 25.8 ms | 2.70k tok/s |
| req#2 | 18,993 | 176.8 s | 96.7 s | 61.5 s | 2.98k tok/s |
| req#4 | 27,163 | 143.4 s | 125.2 s | 2.7 s | 2.80k tok/s |
| req#5 | 19,334 | 150.7 s | 77.6 s | 35.3 ms | 2.65k tok/s |

预填充速率全程稳定在 2.65–2.98k；19k token 的纯计算约 7 秒，TTFT 却是 150 秒——
**95% 的时间在排队与调度等待**。

### 1.3 预填充被解码车道挤占

5 秒窗口里经常只有一个 chunk 推进（1,024 tok / 5 s = 205 tok/s），只有解码空档才出现
12,288 tok / 5 s 的爆发（2.46k）。按分钟统计的推进量是 39k → 13k → 86k → 27k → 19k，
**波动 6.6 倍**。这是"预填充波动"的直接来源，不是显存碎片。

### 1.4 前缀缓存 0 命中

四条成功请求的 `cache` 全是 `0 (0.0%)`，54 个 tools 的相同前缀一次也没复用。

### 1.5 503 雪崩

11:42:46 两条 `500 KV address-space handle is stale` 后 `running` 归 0，11:42:51 起连续 36 条
`503 failed during prepare`，直到日志结束未恢复。**已在 954dd187（队列工作量计入 deadline、超限
429）与 b7e8b56d（物理内存准入）修复**，本轮不重复处理，只在 §3 的门禁里保留一条回归项。

## 2. 工作项

### W：Qwen4Exp 接线（优先）

| 项 | 状态 | 问题 | 做法 | 判据 |
|---|---|---|---|---|
| W1 三 bank 专家流 | 已完成（代码级） | `ExpertStream` 有 split 形态，但需要模型侧构造并绑定 | `parameters.cpp:682` 已按 gate/up/down 三个 `ExpertMatrixBankDesc` 构造；`model_instance.cpp` 已按 `runs[i].split` 调 `attach_split()` 绑三个 bank | 代码就位；待 W4 端到端证明 |
| W2 PLE / HC 注入 | 已完成（代码级） | `run_layers` 需要 PLE 表与四流残差 | `text.cpp` 已接 `PleStepWindow`（3 处）与 HC 残差折叠 | 同上 |
| W3 QSA 规划 | 部分完成 | QSA 的 KV 页与索引计划要进 startup | `startup.cpp:220/463` 已调用 `qsa_geometry` / `qsa_plan_inputs`；`qsa.cpp` 只产出几何，sparse 执行未落地 | 缺 QSA 内核与层执行 |
| W4 端到端跑通 115 GB 工件 | 已完成（能执行） | 依次卡在 `vmm_pool` OOM、`cudaErrorStreamCaptureUnsupported` | 三层修复：池只映射 live 段；`ExpertCache::all_experts_resident()` 取代份额判据；prefill 栈捕获增加 `layer_stack_capturable()` 开关（`layer_capture.h`，引擎在池建立后写入）；另在 `ffn.cpp` 的 per-bank 图缓存上加 `fully_resident()` 守卫 | 同命令（512 ctx、`rk4v4-e8`）连续出数：窗口 1 PPL 14.59、窗口 2 PPL 16.02，7–8 tok/s（44 万专家里约 5.6 万驻留，其余流式）。数值不作为精度结论，仅证明执行链路通 |
| W4a 容量计划与已加载权重 | 已关闭（环境） | 自动池曾 `cuMemCreate` OOM | 停掉占 10,924 MiB 的常驻 serve 后，`gpu=18000 (22500 MiB) host_cache=5058 unslotted=55728 of 73728`，池正常建立 | 无规划缺陷 |
| W4b split 层与图捕获 | 已完成 | 捕获里执行 split 的 host 取数 → `ffn.cpp:372` 断言 | 三个入口分别加守卫：`model_instance` 的整步图用 `all_experts_resident()`；`ffn.cpp` 的 per-bank 图用 `stream->fully_resident()`；`text.cpp` 的 prefill 栈图用 `layer_stack_capturable()` | 同一份工件不再崩；预填充按普通路径执行 |

### P0：TTFT 与稳定性

| 项 | 状态 | 问题 | 做法 | 判据 |
|---|---|---|---|---|
| P0-1 KV 句柄失效 | 已完成 | 日志里 2 条 `500 stale handle` 打崩了后续全部请求 | `tools/serve/replay_burst.ps1`：同形状 4 并发 × 23,090 token + 54 tools | 4/4 返回 200，0×500、0×503 |
| P0-2 失败后不恢复 | 已完成 | 一次 KV 错误后 admission 永久拒绝 | 同一脚本的 follow-up：burst 之后立刻发一条小请求 | 200，468 ms 墙钟（引擎内 TTFT 117 ms） |
| P0-3 预填充车道 | 已定位根因：绑定串行化 | `--max-concurrency 4` 时 `throughput` 稳定报 `running 2 (prefill 2) | waiting 2`，4 条 23k 请求分两波 | 结论不是 KV、不是显存、也不是车道不够：新增的准入探针（`NINFER_PREFILL_BATCH` 打开时打印）在 72 次准入里给出 `tx_busy=56, no_lane=0, kv_wait=0, phys_wait=0, ok=5`。也就是说每条新请求都在等 `Program::has_context_transaction()` 释放——上下文事务（KV 页转移 + 147 MiB 状态备份 + 绑定安装）一次只能有一个，23k prompt 的绑定因此把车道数压到 ~2。修法是把事务按 lane 分段、或把它缩短到只覆盖 KV 发布，而不是调 chunk/并发 | 4 条 23k 并发 TTFT ≤10 s，且 decode 不劣化 >10% |
| P0-4 prepare 61.5 s | 未复现 | 昨天 req#2 的 prepare 是 61.5 s | 本轮 4 条同规模请求的 prepare 为 19.6–204 ms | 保持观察；若再现再拆段 |
| P0-5 前缀缓存 0% | 已撤销（前提不成立） | 昨天四条成功请求 `cache 0 (0.0%)` | 昨天的启动日志实际是 `checkpoint cache | attached`（不是 off）；四条请求是 72/14/3/11 条消息的四段不同对话，本就没有可复用前缀。今天我用四条**完全相同**的 prompt 回放，才出现 `cache 23,083 (100.0%, checkpoint)`。另：`checkpoint cache` 指 LMCache 外部缓存，由 `NINFER_LMCACHE_RESTORE` 决定，不是 CLI 参数，前台面板不可见 | 无缺陷可修 |
| P0-6 KV 预留按实际输出 | 已关闭（假设不成立） | 疑 `--default-max-tokens 32768` 按最大输出预留导致并发下降 | 同一 burst 跑 `-MaxTokens 64` 与 `-MaxTokens 0`（服务端默认 32768）两组：19.3/19.1/20.9/19.0 s 对 19.1/19.6/20.6/19.0 s，无差异 | KV 预留不是本轮 TTFT 的主因；问题在 P0-3 的车道数 |

### P1：长上下文

| 项 | 状态 | 问题 | 做法 | 判据 |
|---|---|---|---|---|
| P1-1 SplitKv 是否启用 | 未开始 | 27B 32k 2,643 vs 4k 3,215（−18%），不知是注意力还是调度 | `context/launch.cu` 的 SplitKv 路由开关做同会话 A/B | 明确 −18% 的归属；若已启用则到此为止 |
| P1-2 32k 的 KV 精度 | 未开始 | 4k 有 fp8 +0.13% / rk4v4-e8 +3.33%，32k 空白 | 32k 补测 fp8 / rk4v4-e8 | 若 fp8 在 32k 仍 ≤+0.5%，可直接换更大池 |

### P2：不投入

块级复用（KVBoost）、块稀疏（PBS-Attn）、层选择性稀疏（CutAttn）、FlashInfer 整体引入、FA3、
PagedAttention 重做（已是核心）、FPGA。理由见 v12 建立时的评审结论：需要 32k 精度基线与逐位门禁，
且与现有内核重叠。

## 3. 顺序与门禁

执行顺序：**W4 → P0-2 → P0-1 → P0-6 → P0-5 → P0-3 → P0-4 → P1-1 → P1-2**。
每项完成后回填本文件的"状态"与证据（命令 + 原始输出摘要）。

统一门禁（任何改动都要复跑，除非改动完全不触数值路径）：

- `python H:\TeleAgentWorks\.temp\kv\regress.py --tier fast`：default PPL `7.441868` 逐位、
  selector-off dump `636e38de95046781` 逐位；
- `--tier moe`：35B 三点 `7.111129 / 7.777808 / 6.479183` 与 decode digest `d093ac2070ab3292`；
- `ctest -R "expert_cache|resize|kv_tier"`；
- 发布冒烟两份（`SMOKE.txt` / `SMOKE-UI.txt`）；
- 故障回放：`powershell -NoProfile -File tools\serve\replay_burst.ps1`（退出码 0 才通过）。

## 4. 记录

- 2026-11-25：建立本文件。审计出 W1–W3 代码已就位、W4 卡在 `VmmPool` 把增长上限当已映射显存；
  修复为"构造时只映射 live 段"，进入验证。
- 2026-11-25：`VmmPool` 惰性映射落地（`VmmPool(device, capacity, segment, mapped_bytes)`；
  `ExpertCache` 传 `capacity_bytes`）。验证：`ninfer_expert_cache_test`、
  `ninfer_expert_cache_per_bank_test` 通过；Qwen4Exp 在 `NINFER_EXPERT_CACHE_MB=4096` 下越过池构造、
  进入执行（随后撞上 W4b）。自动池仍 OOM → 新开 W4a。
- 2026-11-25：W4b 取证：4 GiB 池运行日志无 `[graph] skipped`，崩在 `ffn.cpp:372` 的
  `cudaStreamSynchronize`（capture 内非法），池 `windows=0`——捕获守卫没有覆盖这条构造路径。
- 2026-11-25：§1.5 的 503 雪崩确认由 954dd187 / b7e8b56d 修复，P0-1/P0-2 降级为"复跑确认"。
- 2026-11-25：门禁与 W4a 被环境阻塞：本机常驻 `ninfer-serve`（pid 65468）占 10,924 MiB，
  27B 权重加载与自动池都因此 `cudaMalloc/cuMemCreate` OOM。**不擅自停该服务**；空卡后再复跑
  `--tier fast` 与 W4a。本文件的状态以此为准，不把环境导致的失败记成代码结论。
- 2026-11-25（获授权停 pid 65468 后）：`--tier fast` 通过——`default PPL 7.359779` 逐位、
  `selector-off dump e95cacb864d8b1ab` 逐位、9 个 salient 点全 PASS、`regressions: 0`；
  自动池 OOM 消失（W4a 关闭）。W4b 三层守卫落地后，Qwen4Exp 115 GB 工件首次跑出困惑度数值。
  注意：这轮的 PPL 只跑了两窗就被停掉（为释放显卡跑门禁），不作为精度结论。
- 2026-11-25：P0 回放（`tools/serve/replay_burst.ps1`，4 并发 × 23,090 token + 54 tools）：
  全部 200，0×500/0×503，follow-up 468 ms。新增两条结论——(a) **此条后经核对更正，见下条**：
  `cache 0%` 曾被归因于 `checkpoint cache | off`，实际昨天是 `attached`，0% 来自四条互不相同的对话；
  (b) `--default-max-tokens` 的有无对本轮
  TTFT 无影响，真正的限制是 `running 2 (prefill 2) | waiting 2`，即 **prefill 并行度只有 2**，
  4×23k 被分成两波（17–21 s）。P0-3 据此重写，P0-6 关闭。
- 2026-11-25：P0-3 三组对照（chunk 1024 / chunk 512 / `--max-concurrency 8`）显示车道数恒为 2，
  TTFT 分别 19.0–20.9 s / 23.1–24.7 s / 18.6–19.7 s。排除「合并矩形」与「并发上限」两种解释；
  约束落在准入与绑定（`waiting` 与 `materializing` 同时出现）。下一步不是调参，而是给准入加
  计时与拒绝原因探针。
- 2026-11-25：更正两条。(a) **P0-5 撤销**：昨天启动日志是 `checkpoint cache | attached`，四条请求是
  四段不同对话，0% 命中属正常；`checkpoint cache` 由 `NINFER_LMCACHE_RESTORE` 控制，不是 CLI 参数。
  (b) **P0-3 的约束含控制器**：`--max-concurrency 8` 时引擎告警“arena 只能容纳 4 条全长 lane”，
  且 `[ctl] engine prefill merge: on (controller)`；`NINFER_PREFILL_BATCH=all` 显示只有 2 条被绑定、
  其余 `waiting` 常驻。并发数量是引擎按 arena 现算的，下一步要在准入处打印“本次只绑 2 条”的原因位。
- 2026-11-25：准入原因位落地（`[admit] calls/ok/tx_busy/no_lane/kv_wait/phys_wait`，随
  `NINFER_PREFILL_BATCH` 打印）。4×23k 回放给出 `tx_busy=56/72`，其余三项为 0——**绑定被
  `Program` 级上下文事务串行化**，每个绑定的 KV 转移 + 状态备份 + 安装一次只能有一个，
  这是车道数停在 2 的原因，与 KV 容量无关。P0-3 的修法据此改为“事务分段/缩短”。

