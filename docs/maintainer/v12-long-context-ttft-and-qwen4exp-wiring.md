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
| W4 端到端跑通 115 GB 工件 | 进行中 | 首次运行：加载 3.24 GiB 权重、PLE 表、KV 计划、容量计划全部通过，最后 `vmm_pool: cuMemCreate failed: CUDA_ERROR_OUT_OF_MEMORY` | 第一层根因：`VmmPool` 构造即映射整个**增长上限**（17.5 GiB），与"只映射 capacity"的设计相反。已改为构造函数接收 `mapped_bytes`，只映射 live 段，尾段留给 `resize()` 的 `acquire` | 单元测试与 4 GiB 池路径已通过（见下）；自动池仍 OOM，第二层根因见下 |
| W4a 容量计划与已加载权重 | 待复测（可能不是计划问题） | 自动池 `achieved_pool_mb=17514` 仍 `cuMemCreate` OOM，规划日志自称 `free=19526 MiB` | `nvidia-smi` 显示本机有常驻 `ninfer-serve`（pid 65468）占 **10,924 MiB**，同时 27B 的 15.9 GiB 权重加载也 `cudaMalloc insufficient`。即 OOM 可能来自**卡被占用**，而非计划超配 | 停掉常驻 serve 后在空卡上复跑：若不再 OOM 则本项关闭；若仍 OOM 再查 free 是否把 WDDM 可回收内存算成可用 |
| W4b split 层与图捕获 | 未开始 | 4 GiB 池下进入执行后崩在 `ffn.cpp:372 cudaStreamSynchronize: cudaErrorStreamCaptureUnsupported`，说明捕获已开而该层要走 split 的 host 取数 | 捕获守卫在 `model_instance.cpp:2172`，判据是 `expert_cache == nullptr \|\| can_hold_every_layer()`；`windows=0` 的 Qwen4Exp 池没被这个判据拦住（日志无 `[graph] skipped`）。查 qwen4exp 构造是否在池建立前就做了图决策，或走了另一条构造分支 | 该工件启动时打印 `[graph] skipped: layer needs host fetch` 并正常跑完困惑度 |

### P0：TTFT 与稳定性

| 项 | 状态 | 问题 | 做法 | 判据 |
|---|---|---|---|---|
| P0-1 KV 句柄失效 | 待复跑 | 日志里 2 条 `500 stale handle` 打崩了后续全部请求 | 用同形状请求（72/14 messages + 54 tools、4 并发）重放，确认 841b532a 的 lane 修复是否覆盖此路径；必要时按 `NINFER_KV_*` 探针定位 | 连续 200 请求 0×500 |
| P0-2 失败后不恢复 | 待复跑 | 一次 KV 错误后 admission 永久拒绝 | 错误请求隔离 + arena 重新绑定，不把坏句柄留在共享状态 | 注入一次 500 后，下一条请求 200 |
| P0-3 预填充饿死 | 未开始 | 解码占边界时 prefill 得不到执行 | 每 N 个 decode step 强制放行一个 prefill chunk（N 可配，默认 2），或等待中请求的 prefill 提权 | 19k prompt TTFT 150 s → <30 s，同期 decode 劣化 <10% |
| P0-4 prepare 61.5 s | 未开始 | 同形状请求的 prepare 差 2,000 倍（26 ms vs 61.5 s） | 拆解模板渲染 / 工具 schema / tokenize 三段，定位后修 | 同形状 prompt prepare <200 ms |
| P0-5 前缀缓存 0% | 未开始 | 54 tools 的相同前缀不命中 | 查 prefix index 的 key 是否含 tools/thinking 段、`--kv-evict auto` 是否把前缀页当冷页驱逐、checkpoint 是否落在请求边界 | 第二次相同前缀命中 >90%，TTFT 降一个数量级 |
| P0-6 KV 预留按实际输出 | 未开始 | `--default-max-tokens 32768` + `kv-capacity auto` 下 `running 2 / waiting 2`，疑似按最大输出预留 | 先用 `--default-max-tokens 4k` 重跑同批请求验证假设，再改成按声明/按需增长 | 并发 4 时 running 达到 4；TTFT 同比例下降 |

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
- 发布冒烟两份（`SMOKE.txt` / `SMOKE-UI.txt`）。

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
