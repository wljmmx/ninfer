# 交接：控制台 / KV 驻留 / 图 的工作状态（2026-11-20）

本文件记录一轮长会话的结论与未完成项，目的是让接手的人不必重新推导。
所有数字都是本机（RTX 4090 24 GB / sm_89 / CUDA 13.5 / Win11）实测，且是串行单进程测得的。

## 1. 当前运行形态

```
R:\infer-release-20261119\ninfer-serve.exe R:\model\qwen3_8_27b_v3\qwen3_8_27b_v3_gguf_dflash2.ninfer
  --host 0.0.0.0 --port 11438 --max-context 256000 --max-concurrency 4
  --prefill-chunk 2048 --kv-dtype rk4v4-e8 --spec mtp --draft-tokens 7 --lm-head-draft
  --kv-evict off --kv-capacity 256000 --kv-residency-slack-mib 1536 --wddm-evictable-budget
```

- 池：4,000 / 16,000 页（**全驻留**），`free 7.15 GiB`
- 图：`[graph] capture measured +0.0 MiB` → **captured**
- `/monitor/info`：`model.ffn=dense`、`model.resident_tokens=256000`
- 控制台：`python lmcache_console.py --host 0.0.0.0 --engine-host 0.0.0.0 --port 8099`

## 2. 已确认的结论（含证据）

### 2.1 图与分层互斥，且这是正确行为

```
[graph] skipped: KV residency may need a Host fetch inside the step
                 (the fetch's stream sync is illegal inside a capture)
```

只要计划是分层的（logical > resident），引擎就跳过捕获，并把图预算转成驻留页。
`--kv-evict off` 单独关不掉它——分层来自容量计划，不是 evict 契约。
让逻辑容量等于驻留容量（`--kv-capacity <tokens>`）即可捕获。

### 2.2 图的真实收益：+3.3%，不值得为它做动态切换

| 用例（`ninfer_bench`，同接受率） | 图关 | 图开 | 差 |
|---|---|---|---|
| pp2048+tg128 | prefill 3456.32 ± 6.02 / decode 147.53 ± 0.06 | 3457.80 ± 5.82 / 152.40 ± 0.13 | decode +3.3%，prefill 0% |
| tg512 | 39.40 ± 0.05 | 40.69 ± 0.06 | +3.3% |
| 并发 4 · pp2048+tg128 | batch 136.42 ± 0.18 | 137.61 ± 0.14 | +0.9% |

捕获一次约 **2.0 s**（`CUDA graphs ready | 2.0s`）。因此“按负载动态切换工作点”被否决：
收益 3% 撑不起 2 s 捕获 + 排空。

### 2.3 draft 15 是 decode 退化的唯一原因

| 配置（pp2048+tg128，rk4v4-e8） | prefill | decode | 接受率 |
|---|---|---|---|
| 基准（spec 7） | 3,112 | 188.5 | 82.6% |
| MTP 15 | 3,462.78 ± 2.62 | **151.52 ± 2.51** | 54.3% |
| **MTP 7** | **3,522.24 ± 6.60** | **195.33 ± 0.65** | 77.7% |

dense 27B 用 7；MoE 的默认值必须单独标定（记录值：35B MTP7 decode 471.1 / INT8 522.8）。

### 2.4 RST 与“间歇掉线”的根因

cpp-httplib 默认 keep-alive **5 s / 100 次**，服务端从未覆盖：客户端池化连接空闲超过 5 s，
服务端已关闭，客户端再写就是 Windows RST（10054）。
现在显式设置：keep-alive 300 s、1000 次、read 60 s、write 600 s、TCP_NODELAY。
实测：idle 1/3/6/10/30/70 s 全部复用成功；20 次复用 GET + 3 次同连接生成 + 4 路 SSE（40 s）零失败。

300 s 不再加大，是因为该实现是**线程每连接**（约 21 个 worker），长空闲连接会占住线程。

### 2.5 稠密 / MoE 判据

引擎发布 `model.ffn = dense | moe | unknown`，按规范架构名的表驱动（`Qwen3_5ForCausalLM` → dense，
`Qwen3_5MoeForCausalLM` / `Qwen4ExpForCausalLM` → moe，未知不默认 dense）。
**不要**用名字子串判断：`Qwen4Exp` 是 MoE 但名字里没有 “Moe”。
控制台已不再有 “e2 / fused 工件形态” 这个概念（下拉已删除）。

### 2.6 宿主与容量

- `NINFER_KV_RESIDENCY_SLACK_MB` 已升格为 `--kv-residency-slack-mib`（默认 auto = device/12，512 MiB..4 GiB）；
- `--wddm-evictable-budget` 把预算从“当前空闲”改成 `总显存 − 权重 − 512 MiB DWM 地板`，本机预算 2.2 GiB → 5.9 GiB；
- 但可用池强烈依赖启动那一刻的显存占用：同一条命令实测出现过 1,380 / 3,576 / 4,274 / 4,531 页。
- 实测上限：**5,302 页（339k）可分配但不可用**（换页风暴，出现过 243 s 的单步）；**4,531 页（290k）是全速区**。

## 3. 已完成的结构性改动

| 提交 | 内容 |
|---|---|
| `464a0b2b` | 准入按**驻留容量**拒绝：`maximum context length is 97216 tokens, but the prepared prompt has 163678 tokens`，0.04 s / 400 |
| `33e09cf7` | `/monitor/info` 发布 `model.resident_tokens` |
| `10c68b9f` | 输出预算钳制到剩余空间，并在日志中可见：`[budget] output budget clamped 100000 -> 47519 (resident 97216, prompt 49697)` |
| `d4652924` | 引擎发布 `model.ffn` |
| `2fa5f541` | HTTP 连接寿命显式化（RST 修复） |
| `f25b3139` | minidump 崩溃取证（`ninfer-crash-*.dmp` / `ninfer-terminate-*.dmp`） |
| `fb75fa0c` | 控制台/引擎支持局域网，同源白名单只认本机地址 |

## 4. 未完成

1. **删除步内 page-in**（`src/models/qwen3_5/program/storage/sequence.cpp`）。准入 + 钳制之后该路径已不可达，
   删除后仍保留“KV residency cannot serve this request”文案的那条分支可以一并清掉。
   验收：跑一条 163k，`page-in` 计数恒 0；门禁逐位。
2. **MoE draft 标定**：`ninfer_bench --weights <35B> --spec mtp --draft-tokens 4,7,11 -pg '2048,128' -r 3 --warmup 1 --kv-dtype rk4v4-e8`。
3. **P6**：TTFT 15k / 64k / 163k 三档对照；控制器三档 × 稠密/MoE × (G/P) 规则表。
4. **P7**：面板 LMCache 勾选是否真正 `[lmcache] attached`、目录默认值、宽度对齐、配色与中英混排。

## 5. 施工纪律（本轮踩过的坑）

- **不要改 `include/ninfer/types.h`**：它被 276 个 `.cu` 包含，改一次接近全量重编（>1 h）。
  要加字段优先走 `monitor.cpp`、`engine.cpp`、`src/serve/*` 这类 C++ TU。
  `include/ninfer/engine.h` 经核实**不被任何 `.cu` 包含**，加访问器是廉价改动。
- 性能测试**串行单进程**。本轮两次数据被并行 bench 污染（±1935、±28.5），已作废。
- 门禁依赖 `D:\infer-rtx4090\n4_q1.txt`（dump 阶段），这个文件曾丢失导致 push 被挡。
- 门禁需要独占 GPU：push 前先停引擎，否则 perplexity 会拿不到显存。

## 6. 补充实测（同日，串行单进程）

### 6.1 MoE 的 draft 标定（pp2048+tg128，r=3，warmup=1，rk4v4-e8）

| 形态 | draft | prefill | decode | 接受率 |
|---|---|---|---|---|
| 27B 稠密 | 7 | 3,522.24 | **195.33** | 77.7% |
| 27B 稠密 | 15 | 3,462.78 | 151.52 | 54.3% |
| 35B MoE | 4 | 1,741.01 | 271.81 | 80.2% |
| 35B MoE | **7** | **2,856.03** | **432.72** | 82.6% |
| 35B MoE | 11 | 2,816.68 | 417.76 | 70.6% |

结论：7 对两种形态都是实测最优；draft 4 会把 MoE 的 prefill 打到 1,741（−39%）。
面板默认因此为 7，标签写明两边来源。

### 6.2 TTFT 随上下文的衰减（现网 256k 全驻留，MTP 7）

| prompt | TTFT | prefill | 相对 16k |
|---|---|---|---|
| 16,058 | 6.0 s | 2.91k tok/s | — |
| 59,615 | 25.9 s | 2.42k tok/s | −17% |
| 163,678 | 108.9 s | 1.55k tok/s | −47% |

队列只有 16–50 ms，TTFT 就是 prefill 本身。这不是回归，是长上下文 prefill 的固有成本。

### 6.3 LMCache 勾选

引擎侧正常：勾选启动后日志为
`[lmcache] attached | page 983040 B, 64 token(s), chunk 128 token(s) (= 2 page(s))` +
`checkpoint cache | attached`。
问题是前端勾选状态不持久化，刷新/重启后回到未勾选，于是 `lmcache=false`。现已写入
localStorage，实测刷新后仍勾选。

## 7. 未完成（更新）

1. 拆“步内取页”路径：`src/runtime/engine/engine_core.h:1092/1098` 是判定与文案，
   `src/models/qwen3_5/program/storage/sequence.cpp` 是机制。属拆特性，需要一次
   编译 + 163k 实测（确认 page-in 计数恒 0）+ 门禁，必须在一个完整窗口内完成。
2. P7：目录默认值、日志框外的宽度对齐、配色与中英混排。

已完成并从计划里移除：MoE draft 标定、TTFT 三档、LMCache 勾选持久化、`resident_tokens` 发布。
## 8. 决策（2026-11-20 定稿）：保留 KV 分层

**结论：不拆步内取页，保留 `--kv-evict` 与 absent-page 机制。**

原因是一次读代码纠正了此前的错误前提：

- `engine_core.h:2597` 与 `2612` 的两处 `refuse_kv_residency*` **不是步内拷贝**，而是
  worker 准入循环里的守卫（单请求 + 聚合各一处）。注释写明：没有聚合守卫，两个各自合法的
  请求会互相抢占、彼此暂停、永不恢复（作者实测过非终止）。删掉它等于重新引入死锁。
- 真正的逐页取页在 program 的 sequence/attention 路径（absent page 的 copy），它是一个
  **特性**，不是缺陷路径。

因此“运行期零缺页”只能通过取消分层实现，而那是一次产品决策（放弃超额订阅），不是补丁。
本日选择保留分层：

| 后果 | 说明 |
|---|---|
| 长请求必须独占驻留 | 256k × 4 在这张卡上同时只跑 1 条，其余排队 |
| 分层时图被跳过 | 已实测；全驻留配置（`--kv-capacity <tokens>`）才能 captured |
| 超配额请求在准入被拒 | `464a0b2b`（单请求）+ `engine_core.h` 守卫（聚合） |
| 输出预算按剩余空间截断 | `10c68b9f`，日志 `[budget] output budget clamped` |

若将来要改为“取消分层”（工作点 B），必须一次做完：删 `--kv-evict` 的 on/auto、删 absent-page
机制与 eviction window 安装、删两处 `refuse_kv_residency*` 守卫——中间状态会死锁。