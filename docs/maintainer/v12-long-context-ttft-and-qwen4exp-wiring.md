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
| MoE 35B | 已覆盖（同一套注意力） | MoE 只替换 FFN，注意力仍走 `dense/context`；同一二进制对照：35B pp4096→pp32768 −94%（11,434→724.5）；27B 的同一跨度 **P1-1 复测只掉 5.5%**（3,546→2.74k、27,818→2.59k），原记的“27B −18%（3,215→2,643）”已证伪（两次口径不同）。说明 MoE 的长上下文问题不在注意力，而在专家路由与权重流 |
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
**95% 的时间在排队与调度等待**。注意：P0-3 后来证明排队期间 GPU 已满载（`util.gpu` p50=99、
413–417 W），排的是算力而不是闲置产能；这里是当天的构成，归因见 §4 的 2026-11-25 记录。

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
| W1 三 bank 专家流 | 已完成（W4 端到端已证明） | `ExpertStream` 有 split 形态，但需要模型侧构造并绑定 | `parameters.cpp:682` 已按 gate/up/down 三个 `ExpertMatrixBankDesc` 构造；`model_instance.cpp` 已按 `runs[i].split` 调 `attach_split()` 绑三个 bank | W4 的 115 GB 工件跑出 PPL，三 bank 路径实际执行 |
| W2 PLE / HC 注入 | 已完成（W4 端到端已证明） | `run_layers` 需要 PLE 表与四流残差 | `text.cpp` 已接 `PleStepWindow`（3 处）与 HC 残差折叠 | 同上 |
| W3 QSA 执行 | 已完成（原判“sparse 执行未落地”已作废） | QSA 的 KV 页、索引计划与稀疏执行 | `qwen4_exp/execution/qsa_layer.cpp:101` 的 `qsa_mix` 走完整链路：`qsa_prepare` → `qsa_indexer` → `qsa_sparse_gqa` → `qsa_tail_extract`；kernel 在 `ops/softmax_attention/qsa/`（`sparse_gqa_kernel.cuh`、`sparse_gqa_prefill.cuh`、`sparse_gqa_launch.cu`）。`text.cpp:2775` 中 `indexed && !qsa_dense_arm()` 走 `qsa_mix`，即**默认就是稀疏路径**，只有 `NINFER_QSA_FORCE_FULL` 退回 dense | 支持 width×batch 矩形（merged prefill / decode / 单 lane 三种形状）；W4 的 65 窗 PPL 就是这条路径 |
| W3a QSA 低比特 KV | 已完成 | QSA 的 KV 量化仍被 env 门控，默认降级 bf16 | `model_instance.cpp:1723` 的 `resolve_kv_storage`：`qsa_kv_storage_admitted()` 不通过就把存储改 `BFloat16` 并打印；只有 `NINFER_QSA_SPARSE_GQA_QUANT_KV=1` 才启用低比特路由。所以 W4 用 `--kv-dtype rk4v4-e8` 跑出来的 PPL 实际是 **bf16 KV** | 同语料同窗口（33,322 token、65 窗、sha256 `777E71E9…`）开 `NINFER_QSA_SPARSE_GQA_QUANT_KV=1`：bf16 **13.486344**；**fp8 13.467359（−0.014%，通过）**；rk4v4-e8 13.934833（+3.33%，与 27B 稠密的 +3.39% 一致，是 4-bit 固定代价）。容量：`--max-context 262144 --kv-dtype fp8 --kv-capacity auto` 启动给出 **KV 599,232 tokens**（9,363 页，runtime 7.89 GiB，free 20.2 GiB），是 max_context 的 2.28 倍 |
| W3b QSA 与 dense 参考臂对照 | 已完成 | 稀疏选块是否改变了同一层的数值 | `NINFER_QSA_FORCE_FULL` 是同一种子的 dense 臂，可做逐窗对照 | 同语料同窗口 bf16：sparse **13.486344** / dense **13.488094**，差 **+0.013%**。稀疏选块没有引入数值偏差，W3a 的 fp8 −0.014% 因此可信；两个臂的逐窗 PPL 也同步（32 窗时 9.5951 vs 9.5072） |
| W4 端到端跑通 115 GB 工件 | 已完成（含 65 窗精度基线） | 依次卡在 `vmm_pool` OOM、`cudaErrorStreamCaptureUnsupported` | 三层修复：池只映射 live 段；`ExpertCache::all_experts_resident()` 取代份额判据；prefill 栈捕获增加 `layer_stack_capturable()` 开关（`layer_capture.h`，引擎在池建立后写入）；另在 `ffn.cpp` 的 per-bank 图缓存上加 `fully_resident()` 守卫 | 完整 65 窗（`--context 1024 --stride 512`、33,322 token、语料 sha256 `777E71E97A6113A3…`、**bf16 KV**——`rk4v4-e8` 会被 W3a 的门降级）：**overall PPL 13.486344**，11.0 tok/s，50 min 40 s；窗口 1 PPL 12.7070。旧记的 14.59/16.02 是另一份语料的 2 窗，不可比。这是 Qwen4Exp 第一个可复现的精度基线 |
| W4a 容量计划与已加载权重 | 已关闭（环境） | 自动池曾 `cuMemCreate` OOM | 停掉占 10,924 MiB 的常驻 serve 后，`gpu=18000 (22500 MiB) host_cache=5058 unslotted=55728 of 73728`，池正常建立 | 无规划缺陷 |
| W4b split 层与图捕获 | 已完成 | 捕获里执行 split 的 host 取数 → `ffn.cpp:372` 断言 | 三个入口分别加守卫：`model_instance` 的整步图用 `all_experts_resident()`；`ffn.cpp` 的 per-bank 图用 `stream->fully_resident()`；`text.cpp` 的 prefill 栈图用 `layer_stack_capturable()` | 同一份工件不再崩；预填充按普通路径执行 |

### P0：TTFT 与稳定性

| 项 | 状态 | 问题 | 做法 | 判据 |
|---|---|---|---|---|
| P0-1 KV 句柄失效 | 已完成 | 日志里 2 条 `500 stale handle` 打崩了后续全部请求 | `tools/serve/replay_burst.ps1`：同形状 4 并发 × 23,090 token + 54 tools | 4/4 返回 200，0×500、0×503 |
| P0-2 失败后不恢复 | 已完成 | 一次 KV 错误后 admission 永久拒绝 | 同一脚本的 follow-up：burst 之后立刻发一条小请求 | 200，468 ms 墙钟（引擎内 TTFT 117 ms） |
| P0-3 预填充车道 | 已完成：`auto` 的 2 张状态镜像是正确默认，不是准入缺陷；“18 s 几乎全是排队”已更正 | 车道数由 `controller_residency_policy(auto).device_state_images = 2` 决定（`planning/startup.h` 的 `device_state_image_slots()` 取 `min(policy, logical_slots)`，`NINFER_STATE_IMAGE_DEVICE_SLOTS` 仍可覆盖；`performance` 才是 4）。2 条 lane 占满 2 张 device image 后，第 3/4 条 bind 被拒、只能等前两条完成，且错过后无法插队（`[prefill-reject] mid_capture`）。崩溃修复后事务不是限制（`[tx] bind avg=4.1ms`） | 同二进制 A/B（`replay_burst.ps1` 4×23k，唯一变量 `NINFER_STATE_IMAGE_DEVICE_SLOTS`）：**2 张** → `running 2 \| waiting 2`，队头 18.6 s、队尾 37.0 s；**4 张** → `running 4 (prefill 4)`，全员 36.4 s。两臂总 prefill 吞吐都锁定在 2.04–3.08k tok/s，墙钟严格 = 总 token / 吞吐。4 张多花 294 MiB state image 并挤掉 295 个 device KV 页，无吞吐收益。GPU 侧采样：三种配置 util.gpu p50=99、413–417 W、带宽 30%——compute-bound | 车道策略保持 `auto=2`；TTFT 的优化目标改为 prefill 吞吐（需 ≥2.3× 才能把 4×23k 拉进 10 s）。原判据不可达，已废 |
| P0-4 prepare 61.5 s | 未复现 | 昨天 req#2 的 prepare 是 61.5 s | 本轮 4 条同规模请求的 prepare 为 19.6–204 ms | 保持观察；若再现再拆段 |
| P0-5 前缀缓存 0% | 已撤销（前提不成立） | 昨天四条成功请求 `cache 0 (0.0%)` | 昨天的启动日志实际是 `checkpoint cache | attached`（不是 off）；四条请求是 72/14/3/11 条消息的四段不同对话，本就没有可复用前缀。今天我用四条**完全相同**的 prompt 回放，才出现 `cache 23,083 (100.0%, checkpoint)`。另：`checkpoint cache` 指 LMCache 外部缓存，由 `NINFER_LMCACHE_RESTORE` 决定，不是 CLI 参数，前台面板不可见 | 无缺陷可修 |
| P0-6 KV 预留按实际输出 | 已关闭（假设不成立） | 疑 `--default-max-tokens 32768` 按最大输出预留导致并发下降 | 同一 burst 跑 `-MaxTokens 64` 与 `-MaxTokens 0`（服务端默认 32768）两组：19.3/19.1/20.9/19.0 s 对 19.1/19.6/20.6/19.0 s，无差异 | KV 预留不是本轮 TTFT 的主因；问题在 P0-3 的车道数 |
| P0-7 prefill/decode 排序 | 已完成（prefill turn credit，三档可配） | 11-24 日志：5 s 窗口常只有一个 chunk 推进（205 tok/s），只有 decode 空档才出现 2.46k 的爆发，按分钟波动 6.6×。复现后看到本质：**一个 pass 只能整块跑 decode 或 prefill**，长 chunk 独占 pass 时 decode 只推进 1 token | 修法不是 pass 内连跑多轮 decode（第一轮结束 unit 已 settle，第二轮会被 Program 拒绝：`execution does not match its reserved unit`），而是**给 prefill turn 记账**：每个 pass 都跑一轮 decode（各自正常预留 unit），只有第 N 个 pass 才放行 prefill；N 由 `ControllerResidencyPolicy::decode_turns_per_prefill` 给出（stable=1 / auto=3 / performance=8） | `tools/serve/p07_prefill_decode_mix.ps1`（1 条 1,024-token decode + 1 条 19.5k prefill）：**N=1** prefill 8.4 s、decode 7.8 tok/s；**N=3** prefill **9.4 s（−12%）**、decode **20.6–38.0 tok/s（2.6–4.9×）**、总 decode 98.9 tok/s；**N=4** prefill 10.2 s（−21%）、decode 25.6–40.5。两向都 ≤20% → 选 auto=3 |

### P1：长上下文

| 项 | 状态 | 问题 | 做法 | 判据 |
|---|---|---|---|---|
| P1-1 SplitKv 是否启用 | 已完成 | 27B 32k 2,643 vs 4k 3,215（−18%），不知是注意力还是调度 | 代码：`dense/context/launch.cu` 的 `direct = (envelope.max_context == 0)`，服务跑 262k 上限时必然走 `SplitKv`，**没有 env 开关**。同会话串行探针 `tools/serve/p11_context_probe.ps1`（unique 前缀、单条、chunk 1024、rk4v4-e8、MTP7） | 3,546 tok → 2.74/2.79k；13,947 → 2.78/2.79k；27,818 → 2.59/2.60k。两轮重复一致，**8× 长度只掉 5.5%**，与 23k 的 2.66k 同量级。−18% 不能复现：4k 的 3,215 与 32k 的 2,643 不是同一口径。注意力不是长上下文瓶颈，P1-2 继续查精度 |
| P1-2 32k 的 KV 精度 | 已完成 | 4k 有 fp8 +0.13% / rk4v4-e8 +3.33%，32k 空白 | 32k 补测 fp8 / rk4v4-e8（`tools/serve/p12_kv_precision_probe.ps1`：upstream 文档拼接的 313 KB 真实语料、context 32768、stride 16384，与 4k 用同一 `infer-perplexity` 口径） | bf16 9.410566；fp8 9.41285（**+0.024%**）；rk4v4-e8 9.729298（+3.39%）。fp8 远优于 ≤+0.5% 判据，32k 可直接用 fp8 换更大池；rk4v4-e8 的 3.39% 与 4k 的 3.33% 一致，是固定量化损失、不随长度放大 |

### P2：不投入

块级复用（KVBoost）、块稀疏（PBS-Attn）、层选择性稀疏（CutAttn）、FlashInfer 整体引入、FA3、
PagedAttention 重做（已是核心）、FPGA。理由见 v12 建立时的评审结论：需要 32k 精度基线与逐位门禁，
且与现有内核重叠。

## 3. 顺序与门禁

执行顺序：**W4 → P0-2 → P0-1 → P0-6 → P0-5 → P0-3 → P0-4 → P1-1 → P1-2**。
每项完成后回填本文件的"状态"与证据（命令 + 原始输出摘要）。

统一门禁（任何改动都要复跑，除非改动完全不触数值路径）：

- `python H:\TeleAgentWorks\.temp\kv\regress.py --tier fast`：default PPL `7.359779` 逐位、
  selector-off dump `e95cacb864d8b1ab` 逐位（V10-61 重录后的值；L104 曾写 `7.441868` /
  `636e38de95046781`，是重录前的旧 golden，按旧值跑会把通过的门禁判成 FAIL）；
- `--tier moe`：35B 三点 `7.111129 / 7.777808 / 6.479183` 与 decode digest `d093ac2070ab3292`；
- `ctest -R "expert_cache|resize|kv_tier"`；
- 发布冒烟两份（`SMOKE.txt` / `SMOKE-UI.txt`）；
- 故障回放：`powershell -NoProfile -File tools\serve\replay_burst.ps1`（退出码 0 才通过）。
## 4. 记录


## 5. 2026-11-25 追加：burst 能把引擎打崩（已修并验证）

`tools/serve/replay_burst.ps1` 在验证事务探针时暴露了一个比 TTFT 更严重的问题：**同一份请求能把
serve 打死**。两次回放的现场一致——`[prefill-reject] lane_count a=0 b=0` 刷屏之后：

```
src/core/device.cu:260: CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_))
    failed: cudaErrorNotReady: device not ready
```

即一次计时采样读到 stop 事件尚未完成，`CUDA_CHECK` 把它升级为进程退出；此后每条请求都是 503。
这与 2026-11-24 日志里“先 500、随后 36 条 503 到结束”的形态吻合：引擎不是拒绝，是已经死了。

修法：`CudaEventTimer::elapsed_ms()` 只对 `cudaErrorNotReady` 降级——返回上一次有效采样并清错误码，
其它错误仍走 `CUDA_CHECK`，不吞真实故障。

验证：修复后连续两次 `replay_burst.ps1` 都 `PASS 4 burst request(s) + follow-up, no 5xx`
（follow-up 216 ms / 232 ms，四条 23k 请求 18.5–19.7 s）；修复前同一脚本连续两次 FAIL
（3/4 与 2/4 请求无应答）。`[prefill-reject] lane_count a=0 b=0` 仍会出现但不致命，
是合并尝试在“可 prefill 的 lane 不足 2 条”时的正常拒绝，归到 P0-7 的调度观察里。



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
- 2026-11-25：准入原因位落地（`[admit] calls/ok/tx_busy/no_lane/kv_wait/phys_wait/idle_wait`，随
  `NINFER_PREFILL_BATCH` 打印）。4×23k 回放给出 `calls=103, ok=5, tx_busy=87`，其余为 0，
  `idle_wait=0`——**不是准入被饿死**，而是有请求在等时每轮都在尝试、每轮都撞上开着的事务
  （Bind/Capture/Pause 共用同一个 `context_transaction_`）。下一步按 kind 计时 open→commit。
- 2026-11-25：踩到并记录一个已知坑的实例：改 `engine_core.h` 后 ninja 未重编（depfile 对头文件不可靠），
  二进制停留在旧版本、探针字段没出现。删 `ninfer_engine.dir` 下的引擎 obj 强制重编后恢复正常。
  以后改 `engine_core.h` 一律先清对应 obj。
- 2026-11-25：事务计时推翻上一轮的“事务太慢”结论。`[tx] bind n=5 avg=3.7ms | capture n=2 avg=12.6ms |
  demote n=3 avg=23.4ms | pause n=0`，整轮事务占用约 150 ms；但 `ok=6` 对 `bind n=5`，成对少一次，
  且 `tx_busy` 单调升到 88/105——是“有一次事务开而不关、后续 emplace 替换它并重置计时”，
  不是临界区过慢。P0-3 的下一步因此改为定位“开而不 commit”的路径。
- 2026-11-25：开启状态探针（`open kink / age / opens / replaces`）给出最终答案，也再次更正上一条：
  `replaces=0`（没有替换，也没有泄漏），`opens≈380`（每次 prefill chunk 触发一次 Capture），
  但存在 **`open=0 age=6635ms`** —— 一次 **Bind 事务持锁 6.6 秒**。短事务的中位数掩盖了长尾：
  绑定把整个 prompt 的 KV 页预留、147 MiB 状态镜像与转移都放在事务里，23k prompt 因此 6.6 s，
  期间所有其它绑定被拒。两条 lane 各绑一次（≈13 s）再一起 prefill，正好是 17–19 s 的 TTFT。
  修法：把 KV 预留移出事务（按 chunk 增量申请），或允许不同 lane 并发绑定。

- 2026-11-25：崩溃修复后复测，P0-3 的 6.6 s 被证明是**崩溃途中的假象**：`[tx] bind n=5 avg=4.1 max=8.4`、
  `ph=[0 0 1 0]ms`、`open=0 age=2ms`。真正的原因是调度器一次只放行 2 条 prefill：两条跑满 2.66k tok/s，
  另两条 queue 18.4 s 后靠 checkpoint 命中在 69–201 ms 内完成。P0-3 改为“放行策略可配并实测”，
  不再动 KV 绑定事务。
- 2026-11-25：准入分支探针把队头被挡的位置收敛到 `admission_deferred_`（一轮 burst 里 `deferred=68`，
  `owner/held/arm_no_lane/arm_tx/arm_paced` 全为 0）。该标志只有两处会置位：决策循环里 lane 被抢走
  （`if (!lane)`）与 `reservation.capacity_possible` 但当前无可用容量。也就是说：**不是没被唤醒、不是缺车道、
  不是事务，而是 KV 预留这一刻判不出容量**。下一步给这两处各加一个计数（决策无 lane / 容量不可行）即可区分，
  然后才能决定是等一个容量事件还是改预留口径。
- 2026-11-25：教训：两次用「按行号改」的方式编辑 `engine_core.h` 都把文件改坏（第二次在 270 行报
  `stopping_/failed_/outstanding_` 未声明）。已 `git checkout` 恢复并确认可编译。以后改这个头文件只用
  字符串锚点编辑，或在改完立即编译。
- 2026-11-25：**“是否真的放不下”已有答案：不是 KV、不是显存，是一个状态镜像槽位。**
  决策分支探针给出 `no_lane=0 capacity=6` 与
  `[admit] capacity short: state_slots=1 main_pages=0 backend_pages=0 host=0 B`，
  即缺的是 `state_store` 的 device 槽位（`reserve_destination()` 需要 `need_device=true`）。
  下一步在该行补 `device_occupied/device_capacity`，判断是“槽位真的用完了”还是“分配器没有把已释放的槽位还回来”，
  再决定是加 extra states 还是修释放路径。
- 2026-11-25：**上面那条的答案：既不是用完了也不是没还，是控制器按模式只给 2 张。**
  `core/controller_mode.h` 的 `controller_residency_policy(Auto)` 返回
  `{kv_host_tiering=true, device_state_images=2}`（`Performance` 才是 4，`Stable` 用 `all_device_images`）；
  `planning/startup.h` 的 `device_state_image_slots()` 取 `min(policy, logical_slots)`，
  `NINFER_STATE_IMAGE_DEVICE_SLOTS` 仍优先。启动日志 `[kv-plan] state_images=307,908,608`
  = 2 × `state_image_bytes` 153,954,304，印证物理张数就是 2。所以 2 条 lane 占满后第 3/4 条被拒，
  是策略在生效，不是释放路径泄漏。
- 2026-11-25：**同二进制 A/B（唯一变量 `NINFER_STATE_IMAGE_DEVICE_SLOTS`）证明放开到 4 只有代价**：

  | 指标 | 2 张 | 4 张 |
  |---|---|---|
  | burst 墙钟（4×23,090 tok，`max_tokens=64`） | 19.3 / 19.2 / 20.8 / 20.8 s | 36.8 / 36.4 / 36.5 / 36.4 s |
  | 同上改用 `-UniquePrompts`（无共享前缀） | 18.6 / 18.6 / 37.0 / 36.9 s（两波） | 全员 36.4 s（一波即全量，与前一行等价） |
  | 车道 | `running 2 (prefill 2) \| waiting 2` | `running 4 (prefill 4)` |
  | 总 prefill 吞吐 | 2.05–3.08k tok/s | 2.04–3.07k tok/s |
  | state image | 294 MiB（`anchor_slots=2`） | 587 MiB（`anchor_slots=4`） |
  | device KV 驻留 | 3,646 页 / 3,631.8 MiB（19.5%） | 3,351 页 / 3,337.9 MiB（18.0%） |
  | KV logical 容量 | 18,725 页组 | 18,620 页组 |

  墙钟严格等于“总 token ÷ ≈2.5k tok/s”：2 张是两波（队头 18.6 s、队尾 37 s），4 张把两波并成一波
  （全员 36.4 s）。**车道数只改变排队怎么分，不改变吞吐**;4 张还多占 294 MiB state image 并把 device KV
  挤小 295 页。结论：`auto=2` 是正确默认，P0-3 不再算待修缺陷。
- 2026-11-25：由此 P0-3 的判据“4×23k TTFT ≤10 s”不成立：按 2.5k tok/s，4 条 23k 的理论下限
  就是 92k ÷ 2.5k ≈ 37 s。TTFT 的优化目标改为 **prefill 吞吐**（需 ≥2.3× 才可能进 10 s），不是车道数。
- 2026-11-25：日志口径坑：`device_state_slots 4 [default]` 打印的是**逻辑槽位**
  （`model_instance.cpp` 把它规范化为 `value_or(concurrency)`），不是物理张数；物理张数只能从
  `[kv-plan] state_images=<bytes>` ÷ `state_image_bytes` 读出。
- 2026-11-25：`tools/serve/replay_burst.ps1` 新增 `-UniquePrompts`（默认关，向后兼容）：给每条请求
  一个不同的开头句，长度不变。没有它时，车道数少于请求数会让后几条命中 100% 前缀缓存，
  把调度/pipeline 测试测成缓存测试。
- 2026-11-25：**GPU 侧实测否掉“有空闲 GPU 在干等”**（`tools/serve/gpu_util_probe.ps1`，200 ms 一次
  `nvidia-smi` 采样 + 同窗口 `throughput` 行）：

  | 配置 | 请求 | TTFT/墙钟 | util.gpu | ≥90% 占比 | 显存带宽 | 功耗 | SM 时钟 |
  |---|---|---|---|---|---|---|---|
  | 2 张 | 1×23,036 | 8.9 / 9.9 s | p50 99 | 94.8% | 30.2% | 417 W | 2833 MHz |
  | 2 张 | 4×23,045（unique） | 18.6 / 18.6 / 36.8 / 36.8 s | p50 99 | 94.8% | 30.2% | 417 W | 2833 MHz |
  | 4 张 | 4×23,045（unique） | 36.6 / 36.5 / 36.2 / 36.3 s | p50 99 | 94.2% | 29.6% | 413 W | 2838 MHz |

  三种配置 GPU 都接近满载（4090 TDP 450 W），显存带宽只到 30% —— prefill 是 **compute-bound**，
  单条就已经把 SM 打满，加 lane 只是把同一份算力切给更多请求。请求日志的阶段分解也一致：
  单条 `submit 2.7s / devwait 7.1s`，两条 `submit 5.1s / devwait 13.1s`（submit 是 host 侧串行提交，
  与另一条 lane 的 GPU 计算重叠，未进入墙钟）。
- 2026-11-25：**读日志的坑**：`throughput | ... | host 16.5% (828 ms)` 里的 `host` 是
  CPU 侧活跃时间 / 窗口（`host_active_ns / interval`），不是 GPU 利用率；20% 表示 80% 时间在等 GPU，
  不是“GPU 只用了 20%”。GPU 是否空闲只能用 `nvidia-smi` 采样判定。
- 2026-11-25：**P0-3 收尾，并更正“TTFT 的 18 秒几乎全是排队等车道”**：
  - 仍成立：只有 2 条能同时 prefill；后两条确实 `queue 18.4 s`；6.6 s 确实是崩溃副产物。
  - 已证伪（只对“四条 prompt 逐字相同”成立）：等待者 `engine-wall 69–201 ms`、`cache 23,083 (100%)`。
    `-UniquePrompts` 复测同一形状，等待者 `engine-wall 17.5 s`，TTFT 36.9 s 里约一半是真计算。
  - 已证伪（归因）：不是“调度器一次只放行 2 条”的策略选择，而是 2 张 device state image 的容量；
    且单条 23k 就要 8.9 s、GPU 99%，所谓排队是在等算力，不是等闲置产能。
  结论：P0-3 关闭，不再加车道；后续只做 prefill 吞吐。
- 2026-11-25：**P1-1 结案：SplitKv 无开关，且 32k 没有额外劣化。**
  `dense/context/launch.cu`：`direct = (envelope.max_context == 0)`，服务跑 `--max-context 262144`
  时必然走 `SplitKv`，`NINFER_*` 也没有路由覆盖。同会话串行探针
  （`tools/serve/p11_context_probe.ps1`，每条不同开头、单条、chunk 1024、rk4v4-e8、MTP7）：

  | prompt tokens | TTFT | prefill | submit | devwait |
  |---|---|---|---|---|
  | 3,546 | 1.3 s | 2.74 / 2.79k | 347 ms | 1.1 s |
  | 13,947 | 5.1 / 5.0 s | 2.78 / 2.79k | 1.4 s | 3.8 s |
  | 27,818 | 10.8 / 10.7 s | 2.59 / 2.60k | 3.1 s | 7.7 s |

  两轮重复一致，8× 长度只掉 5.5%，且 32k 的 2.59k 与 23k 的 2.66k 同量级。
  所以台账原先记的“4k 3,215 → 32k 2,643（−18%）”是两次不同口径的对比（4k 那次 3,215
  在当前配置下复现为 2.74k），不是长上下文让注意力变慢。P1-2 继续查精度。
- 2026-11-25：**P1-2 结案：32k 下 fp8 只 +0.024%，rk4v4-e8 +3.39%。**
  探针 `tools/serve/p12_kv_precision_probe.ps1`，语料是 upstream 文档按名拼接的 313 KB 真实文本
  （现有 `ppl_corpus` 只有 ~11k token，不够 32k 窗口），`--context 32768 --stride 16384`：

  | dtype | PPL | Δ vs bf16 |
  |---|---|---|
  | bf16 | 9.410566 | — |
  | fp8 | 9.412850 | +0.024% |
  | rk4v4-e8 | 9.729298 | +3.39% |

  fp8 的 32k 误差（+0.024%）比 4k 的 +0.13% 还小，远超 ≤+0.5% 判据，可在 32k 直接换 fp8；
  rk4v4-e8 的 3.39% 与 4k 的 3.33% 一致，说明它是固定的量化损失，不随上下文长度放大。
- 2026-11-25：一并修掉文档内部矛盾：§1.1 引用的“27B −18%”与 P1-1 复测冲突，已改为
  “27B 5.5% / 35B −94%”；§1.2 的“95% 在排队”补注“排队期间 GPU 满载”。
- 2026-11-25：**补录 P0-7（prefill/decode 排序）。这一项此前只以 §1.3 的现象存在，
  从未进过工作项表，是台账的漏项**（pacing / 有界 hold 的实验回退后没有留痕，所以清单里看不见）。
  复现（`tools/serve/p07_prefill_decode_mix.ps1`，2 张镜像、1 条 1,024-token decode + 1 条 19.5k prefill）：

  | 场景 | prefill | decode |
  |---|---|---|
  | 单独 prefill 19.5k | 2.74k tok/s | — |
  | 单独 decode | — | 104 tok/s |
  | 并行 5 s 窗口 | 1.04k / 2.45k tok/s | 30.2 → **7.8** tok/s |
  | prefill 结束后 | — | 73.1 tok/s |

  prefill 几乎不受影响（2.74k vs 单独 2.66k），**decode 被压到 7.8 tok/s（−92%）**。
  与 11-24 的“prefill 掉到 205 tok/s”合起来看，机制是同一个：一个 pass 只能整块跑
  decode 或 prefill，另一方整轮等待，谁先占住谁赢。修法不是调 pacing，而是让两类 row
  进同一个 batch pass（执行层已有多行 batch，缺的是 decode row + prefill row 混合）。
  这一项与 W3 同级工作量，排期需单独确认。

## 6. 调度方案的模型差异（P0-7 设计）

P0-7 的修法不能一套方案套三个模型。三个形态的 batch 形状能力不同，混合 decode+prefill 的代价也不同。

### 6.1 现状：一个 pass 只能整块跑一类

`engine_core.h` 的顺序是 control → decode → prefill，两者各自 `device.synchronize()`；decode 一侧
还有采样、draft 提交、state fork 三处同步（`decode.cpp:948/1114` 等），所以不能靠“异步提交”绕过。
唯一出路是让两类 row 进**同一个 pass**，一次同步覆盖两者。

### 6.2 三种形态的形状能力

| 模型 | prefill 侧 | decode 侧 | 混合可行性 | 关键约束 |
|---|---|---|---|---|
| 稠密 27B | `dense/context`：`T=1..16 × B=1..8`，**per-row `valid_columns`**（`context_softmax_attention.cpp:139–152`） | `dense/causal_cache`：每 warp 16 行 Q | **高**：context kernel 本身接受每行不同有效列，decode 行取 `T=1 / valid=1` 即可入同一矩形 | 一个 chunk 1024 token 要拆成 64 个 16-token tile，decode 行在第 1 个 tile 后必须“提前退场”；需要 tile 级行退场，而不是一整 pass 的行集合 |
| MoE 35B | 同一注意力；FFN 走 `ffn.cpp` 的 routed 专家，routes 是 `{top_k, tokens}` | 同一 FFN，专家驻留或流式 | **中**：路由按 token 独立，混合不会算错 | 混合会把两行的**专家并集放大**，`ExpertCache` 的 miss 上升，流式上传可能吃掉 decode 的收益；必须按“专家重合度”门控 |
| Qwen4Exp | `qsa_mix`：`width × batch` 矩形，注释明确 “a decode is that rectangle at width one” | 同一个 `qsa_mix` | **最高**：形状模型本身就是为“矩形 + 每行有效列”设计的 | `qsa_indexer` 对 1-token 行与 T-token 行的选块语义要分别核对（`sparse_gqa_prefill.cuh` 的 `Queries` 模板已按查询数分档） |

### 6.3 结论：分三阶段，不能并行上

1. **Qwen4Exp 先做**。`qsa_mix` 已是矩形模型，改动集中在调度入口（把 decode 行塞进同一个
   `qsa_mix` 调用），不碰注意力数值路径。判据：混跑时 decode ≥70 tok/s（当前 7.8），
   prefill 不低于单独值的 85%（当前 2.74k → ≥2.33k）。
2. **27B 稠密次之**。需要给 `dense/context` 增加 tile 级的行退场，并让 decode 行改走 context
   分支（放弃 `causal_cache` 的专用优化）。判据同上；另外要求 padding 到 16 的无效列不增加
   FLOPs（`valid_columns` 已具备，但要确认 kernel 真的跳过）。
3. **MoE 最后**，且必须先量“专家并集”代价。若混合导致 miss 上升超过 10%，就只在
   “decode 行与 prefill 行的路由重合度高”时混合，否则保持串行。

### 6.4 与 P0-3 的关系

P0-3 已证明车道数不是瓶颈（2 张镜像就是硬件饱和点），所以本节的混合**不增加 lane 数**，
只改变一个 pass 内的执行内容——两者不冲突：车道仍 2 条，每条 lane 上 decode 与 prefill 不再互斥。

### 6.5 三档控制器各自的排序方式

三档不是同一个策略的松紧，而是三种不同的取舍，所以排序也要分档：

| 档位 | 驻留策略 | 排序方式 | 理由 |
|---|---|---|---|
| `stable` | 不分层，KV 全 resident，配置装不下就拒绝启动 | **严格顺序，不混合**：一个 pass 只做 decode 或 prefill，按到达顺序服务 | 这一档卖的是可预测性：混合 batch 会让 TTFT/ITL 随负载形状漂移，且分层换页会引入不可预期的停顿 |
| `auto` | 2 张 device image + Host tiering | **混合 batch**：decode row（width=1）与 prefill row（chunk）进同一矩形，一次同步 | 默认档，两向都不该被对方饿死；2 条 lane 已能打满 GPU，混合不增加算力需求 |
| `performance` | 4 张 device image + Host tiering | **混合 + 更激进的配额**：混合 batch，且允许 prefill chunk 上调、decode 多轮连投 | 这一档面向突发请求：队头延迟可以牺牲，换总完成时间；4 张镜像正是为此多花的显存 |

实现上，`controller_residency_policy()` 已经是三档的单一来源，把排序方式也挂到同一个结构里
（新增一个 `prefill_decode_order` 枚举），调度循环按它分支，避免再出现“环境变量与控制器各说一套”。

**已落地（2026-11-25）**：`core/controller_mode.h` 新增 `PrefillDecodeOrder`
（`SerialDecodeFirst` / `MixedBatch` / `MixedBatchAggressive`）并挂在
`ControllerResidencyPolicy::prefill_decode_order` 上，三档分别是 serial / mixed / mixed-aggressive。
**调度循环尚未按它分支**——当前仍是一个 pass 跑一类，所以行为没有变化，这一项只保证配置就位、
可被打印和测试。下一步是让 `auto` 的 `MixedBatch` 真正生效（见 §6.2 的实施顺序）。



## 7. 记录（W3 / W4 结果）

- 2026-11-25：**W3 更正**：QSA 的稀疏执行早已落地（`qsa_mix` 全链路 + `ops/softmax_attention/qsa/`
  的 kernel），台账原写“sparse 执行未落地”是错的。真正缺的是低比特 KV 的资格化（W3a）与
  dense 参考臂对照（W3b）。
- 2026-11-25：**W4 完成 65 窗精度基线**。语料是 upstream 文档拼接（33,322 token、sha256
  `777E71E97A6113A3…`），`--context 1024 --stride 512`，**bf16 KV**（`rk4v4-e8` 被 W3a 的门降级）：
  `overall PPL 13.486344`，11.0 tok/s，50 min 40 s，窗口 1 = 12.7070。旧记录的 14.59 / 16.02 是
  另一份语料的 2 窗，不可与本次比较。
- 2026-11-25：**W3a 完成（精度）**。同语料同窗口、开 `NINFER_QSA_SPARSE_GQA_QUANT_KV=1`：

  | dtype | PPL | Δ vs bf16 | 耗时 |
  |---|---|---|---|
  | bf16 | 13.486344 | — | 50m40s |
  | **fp8** | **13.467359** | **−0.014%** | 50m08s |
  | rk4v4-e8 | 13.934833 | +3.33% | 49m45s |

  fp8 通过 ≤+0.5% 判据（甚至略低于 bf16，属量化噪声）；rk4v4-e8 的 +3.33% 与 27B 稠密的
  +3.39% 一致，说明是 4-bit 建码的固定代价、不是 QSA 的选块。**结论：Qwen4Exp 的 QSA 配 fp8，
  不配 rk4v4-e8**。262k 上下文的容量核算还没做（fp8 每元素 1 字节，bf16 是 2 字节，理论上翻倍）。
- 2026-11-25：**W3b 已启动**：`NINFER_QSA_FORCE_FULL=1` + bf16，同语料同窗口，用于把
  “稀疏选块”与“dense 全注意力”在数值上分开。这是 W3a 结论的前提（若两臂差得离谱，
  fp8 的 −0.014% 就只是两臂共有的偏差）。
- 2026-11-25：**W3b 完成：稀疏与 dense 在数值上等价**。同语料同窗口、都 bf16：
  sparse **13.486344**、dense（`NINFER_QSA_FORCE_FULL=1`）**13.488094**，差 **+0.013%**；
  逐窗也同步（32 窗 9.5951 vs 9.5072，差 −0.09%）。所以 QSA 的选块不是误差来源，
  整套 QSA 精度结论成立：**fp8（−0.014%）可用，rk4v4-e8（+3.33%）不可用**。
- 2026-11-25：W3 至此收口——执行链路（早已存在）、低比特精度（fp8 通过）、dense 对照（等价）
  三项都有实测。
- 2026-11-25：**W3a 的容量收尾**：`--max-context 262144 --max-concurrency 2 --kv-dtype fp8
  --kv-capacity auto` 启动，日志给出 `capacity | KV 599,232 tokens, fp8, auto | pages 9,363/9,363
  | runtime 7.89 GiB | free 20.2 GiB | max_context 262,144`。**299k → 599k，是 max_context 的
  2.28 倍**，262k 在 fp8 下有充足余量。注意启动时的 capacity plan 行：expert pool 按 auto 拿到
  18,501 MiB，KV 的 device 腿一度被压到 0、靠 host 腿（`host_kv_budget=16,256,040,960`）撑起
  9,363 页——这是 Qwen4Exp 在 24 GB 卡上的固有形态，不是缺陷。

## 8. QSA 全 dtype 支持（2026-11-25）

原先有两道门把 Qwen4Exp 锁在 bf16：引擎侧 `qsa_kv_storage_admitted()` 和 Op 侧
`sparse_gqa.cpp` 的 `quantized_kv_enabled()`，都要求 `NINFER_QSA_SPARSE_GQA_QUANT_KV` 才放行。
但 `ops/softmax_attention/qsa/storage.h` 的 `qsa_storage_supported()` 早已列出全部 11 种
（bf16、int8-g64、fp8、nvfp4、k8v4、rk8v4、rk4v4、rk4v4-e8、rk2v4-e8、rk4v2-e8），
`qsa_prepare` 与 `qsa_sparse_gqa` 也都实现了它们——门比实现更严，且两处判据会漂移。

改法：判据收到 `storage.h` 的 `qsa_low_bit_kv_enabled()` 一处——**默认放行全部已实现存储**，
`NINFER_QSA_SPARSE_GQA_QUANT_KV=0` 显式退回 bf16（复现旧臂），`=1` 与不设等价。两处引用同一个函数。

冒烟（2 窗、1,163 scored tokens，bf16 基线 11.858697）：

| dtype | PPL | Δ vs bf16 | 结论 |
|---|---|---|---|
| bf16 | 11.858697 | — | 基线 |
| int8 | 11.814408 | −0.37% | 可用 |
| rk4v4 | 11.907859 | +0.41% | 可用 |
| k8v4 | 11.911352 | +0.44% | 可用 |
| rk8v4 | 11.960540 | +0.86% | 可用 |
| nvfp4 | 11.971266 | +0.95% | 可用 |
| rk2v4-e8 | 12.471646 | +5.17% | 2-bit K，精度代价明显 |
| rk4v2-e8 | 13.581238 | +14.53% | 2-bit V，精度代价很大 |

7 个 dtype 全部 `exit=0`，**“支持”达成**。注意 2 窗语料的噪声约 ±1%，所以 +0.4%～+0.95% 这几档
只能说明“同量级”，精确排序要用全语料（fp8 与 rk4v4-e8 的全语料值见 §7：−0.014% / +3.33%）。
`rk2v4-e8` 与 `rk4v2-e8` 的损失远超噪声，属于“能跑但默认不该选”。

待确认：`--kv-dtype` 不指定时 Qwen4Exp 落到的默认值（perplexity 侧注释说默认 fp8）。放行后
显式指定的低比特会被真正执行，不再静默降级——这一点需要在 serve 的默认值上确认一次。

## 9. Qwen4Exp 低比特权重（GGUF 混合量化，文件已核实，未接入）

Qwen4Exp 后续要用 R 盘上的低比特 GGUF。目标是解码 **≥85–97 tok/s**：85–97 是 strata **官方在单卡 3090 上**
实现的数字，本机是单卡 4090、同环境，理应更高，所以 85–97 是**下限**而不是上限；比较时必须与 strata
同配置（上下文、常驻比例、单卡）。

**2026-11-26 核实（模型已下载）**：`R:\model\qwen4exp\IQ3_XXS\`
`Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-0000{1,2}-of-00002.gguf`，GGUF v3、`split.count=2`，
两片 43.81 + 26.82 = **70.63 GB**（对比现有 `.ninfer` 的 115.4 GB，只有 1.63× 压缩）。

**名字叫 IQ3_XXS，实际是 GSQ-RCO 混合量化**（分片 1 的 1,223 个 tensor）：

| 类型 | 数量 | 类型 | 数量 |
|---|---|---|---|
| BF16 | 484 | Q4_K | 45 |
| F32 | 292 | Q2_0 | 38 |
| IQ3_S | 82 | IQ2_XS | 20 |
| Q6_K | 71 | IQ2_S | 20 |
| IQ4_XS | 67 | IQ2_XXS | 18 |
| IQ4_NL | 58 | Q5_K | 15 |
| — | — | **IQ3_XXS** | **12** |

分片 2 只有 1 个 tensor：`per_layer_token_embd.weight [160, 320,001,536]`，IQ4_NL，26.82 GB——就是 Qwen4Exp
的 PLE 表（与运行日志 `ple table 320001536 rows x 160` 一致）。

所以“引入 IQ3_XXS”实际是**引入一个 GGUF 容器 + 十余种 GGML 量化类型 + PLE 表**，而不是一种码本。

需要的工作：

1. **GGUF 读取与转换**：`src/` 里没有任何 GGUF 解析（现有 artifact 是自有 `.ninfer` 容器）。
   要么写 GGUF→`.ninfer` 转换器，要么给加载器加 GGUF 直读；两者都要覆盖上表的十余种类型。
2. **类型覆盖**：BF16/F32/Q4_K/Q5_K/Q6_K/IQ4_NL/IQ4_XS/IQ3_S/IQ3_XXS/IQ2_XXS/IQ2_XS/IQ2_S/Q2_0。
   其中 F32 的 292 个多为 norm/scale，可按 4 字节原样；IQ 系列要各自的码本与符号位解码。
3. **PLE 表路径**：26.82 GB 的 IQ4_NL 表注定流式读取（现有 `ple/` 有磁盘读队列，但编码是
   `q4_g64_fp16/row_split_k128_v1`，需要接受 IQ4_NL 行布局）。
4. **与 expert streaming 配合**：上传原码、device 侧反量化，否则 host 解压成为新瓶颈。
5. **判据**：同条件（262k、fp8 KV、MTP7）解码 ≥85–97 tok/s；当前 `q4_g64_fp16` 基线是 4.4 tok/s
   （host-bound，见下），所以 85–97 需要 19–22 倍，1.63× 的压缩只贡献一部分，
   **device 驻留比例必须显著上升**。

前置基线（`tools/serve/q4exp_decode_baseline.ps1`，fp8 KV、262k、c=1、output 256）：
`decode 4.4 tok/s`，`TTFT 9.5 s`，总 66.9 s。请求行的分解是决定性的：

```
phases boundary 19.1 ms | submit 1m 6.6s | devwait 251 ms | decodehost 57.2 s | decodewait 214 ms
```

**这不是 GPU 慢，是 host-bound**：`devwait` 只有 251 ms，而 host 侧 submit 66.6 s、decodehost 57.2 s。
专家池装不下的路由专家落到 CPU/流式路径（`HostRoutedFetch` / `host_routed_ffn`），GPU 大部分时间在等 host。
所以格式再省也到不了 85 t/s，必须让 device 驻留的专家比例大幅上升，把 host 腿压到接近零。

R 盘同时有对照件：`Qwen3.8-Flash-Next-DFlash2-strata1-tuned-BF16.gguf`（1.13 GB）与
`qwen3.8-flash-next-dflash2.strata1.q4_0-selq4.pack`（0.32 GB），以及各档 DFlash2 draft
（Q4_K_M/Q5_K_M/Q6_K/Q8_0/BF16）。`.incomplete` 的三个分片属于 UD-Q4_K_XL，与 IQ3_XXS 无关。
