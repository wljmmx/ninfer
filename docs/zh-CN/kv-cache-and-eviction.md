# KV 缓存：类型全集、显存策略与两个启动开关（RTX 4090 / sm_89）

> 本文面向在 **RTX 4090（Ada，sm_89）** 上部署 NInfer 的运维/开发同学。英文原版说明见仓库根 `README.md`
> 与 `docs/`，本文只讲 4090 上容易踩坑的三件事：**可选哪些 KV 类型**、**显存不够时引擎会做什么**、
> **本版新增的两个启动开关**。

## 1. 可选 KV 类型全集（`--kv-dtype`）

`--kv-dtype` 共接受 **10 种**取值：`bf16`、`int8`、`fp8`、`nvfp4`、`k8v4`、`rk8v4`、`rk4v4`、
`rk4v4-e8`、`rk2v4-e8`、`rk4v2-e8`。

其中 **9 种在 sm_89 上可用**；`nvfp4` 需要 Blackwell（sm_120a）的 NVFP4 张量核，在 4090 上
**启动即被拒绝**（引擎会给出明确错误并提示改用 BF16/INT8/FP8）：

```text
Nvfp4Group16 KV storage requires a Blackwell (sm_120a) build with NVFP4 tensor cores;
choose BF16, INT8 or FP8 KV on RTX 4090 (sm_89).
```

| `--kv-dtype` | sm_89 可用 | 每令牌 KV 字节 | PPL | vs `int8` |
|---|---|---|---|---|
| `bf16` | ✅ | 64 KiB | 6.472085 | +0.02% |
| `int8` | ✅ | 33 KiB | 6.470616 | +0.00% |
| `fp8` | ✅ | 32 KiB | 6.480522 | +0.15% |
| `nvfp4` | ❌（需 Blackwell sm_120a） | — | **不可用** | — |
| `k8v4` | ✅ | 25 KiB | 6.478967 | +0.13% |
| `rk8v4` | ✅ | 25 KiB | 6.500340 | +0.46% |
| `rk4v4` | ✅ | 17 KiB | 6.519109 | +0.75% |
| `rk4v4-e8` | ✅ | 17 KiB | 6.690658 | +3.40% |
| `rk2v4-e8` | ✅ | 13 KiB | 7.067984 | +9.23% |
| `rk4v2-e8` | ✅ | 13 KiB（与 rk2v4-e8 同量级） | 7.028697 | +8.62% |

口径：本机（单卡 RTX 4090 24 GB）、v3 artifact `qwen3_8_27b_v3.ninfer`、同一 20,000 字符切片、
`--context 4096 --stride 1024`（与上游 README 的 4090 KV 表同协议）。上游表未包含 `rk4v2-e8`，
本表补齐。**解码是权重带宽受限，KV 类型对解码速度影响 <7%**，所以这里的取舍本质是
**容量 vs 精度**：`rk2v4-e8`/`rk4v2-e8` 最省显存但精度代价明显，`int8`/`fp8` 是精度最优的一档。

## 2. 显存不够时引擎会做什么（默认行为，无需开关）

1. **先排队，不抢占**：新到的请求不会把正在跑的请求挤掉；竞技场装不下的并发就等（`--max-pending-requests`
   上限、`--pending-timeout-ms` 绝对截止）。
2. **暂停 → 快照进主机**：轮边界上，最老的驻留者可位移更年轻者；被暂停车道的**快照写进主机竞技场**
   （默认 **8 GiB + 8 个 StateImage ≈ 9.15 GiB**）。恢复时**从快照取回**；只有在主机也放不下时才会
   退化为**重算**（实测重算比取回贵约 60×）。
3. **掩码页的处置（需开启逐出契约）**：契约会把窗口外的 KV 页从注意力读取中掩掉，然后按 fate 处置 ——
   本版默认 fate 是**降级到主机（保字节日后仍可取回）**，而不是丢弃。

## 3. 两个启动开关（本版新增）

> 二者都是**启动参数**：进程启动时定，**改任一都需要重启 serve**。
> 环境变量 `NINFER_KV_EVICT_*` / `NINFER_KV_IDLE_RECLAIM_MS` 仍然被尊重且**优先级更高**（实验/复现通道）。

### 3.1 `--kv-evict off|on|auto`（默认 `off`）

配套 `--kv-evict-window N`（无条件保留的尾部页数，默认 4）、`--kv-evict-sink N`（头部恒保留页数，默认 1）。

`auto` 由引擎在**启动期一次判定**（容量解析之后），四个条件全满足才开启，并把结论与原因写进启动日志：

| 条件 | 判据 | 依据 |
|---|---|---|
| A 容量真的不够 | 竞技场装不下 `--max-concurrency` 条满长车道 | 启动顾问同一判据 |
| B 质量在可接受带内 | 需要的保留比例 ≥ 58%（实测：77% 逐位无损、70% +0.15%、64% +0.92%、58% +1.78%） | 本机保留–质量曲线 |
| C 主机装得下一次完整抢占 | `host_capacity_bytes ≥ 每条车道字节 × 并发` | 不够则会退化为重算 |
| D 该 KV 类型有缺页路径 | `nvfp4` 没有（且本机不可运行） | 引擎能力表 |

不满足时保持 `off`，并在启动日志里**指明哪一条不满足**与对应杠杆（降并发 / 减小 `--kv-sizing-headroom-mib`（旧名
`--kv-capacity-headroom-mib`，保留一版期） /
加大 `--host-context-mib` / 或显式 `--kv-evict on` 自行承担质量代价）。

### 3.2 `--kv-idle-reclaim-ms N`（**默认 60000**，`0` = 关）

空闲一定时间后，把**空闲的共享前缀保留项**的设备副本降级到主机（字节保留），后续同前缀请求
**取回即可复用**。实测（同一贪心多轮对话，关 vs 开）：生成文本**逐字一致**；开时降级 190 页、
取回 187 页，复用轮 TTFT 由 261–311 ms 变为 341–347 ms（**首次取回 +30~80 ms**），相对整段重算
（17–19 s）仍是 ~60× 的便宜。

## 4. 快速自检（与上游 README 的示例一致）

```bash
# 每种 KV 类型都跑一遍官方 PPL（覆盖所有可用类型）
for kv in bf16 int8 fp8 k8v4 rk8v4 rk4v4 rk4v4-e8 rk2v4-e8 rk4v2-e8; do
  ./build-win/apps/ninfer-perplexity models/qwen3_8_27b_v3.ninfer \
    --text notes.txt --context 4096 --stride 1024 --kv-dtype $kv
done

# 服务端：带逐出契约自动判定 + 空闲降级（默认已开）
./build-win/apps/ninfer-serve models/qwen3_8_27b_v3.ninfer \
  --max-context 48000 --kv-capacity auto --max-concurrency 8 \
  --kv-dtype rk4v4-e8 --kv-evict auto --kv-idle-reclaim-ms 60000
```

启动日志里应能看到 `eviction | contract on | window 4 sink 1 pages | host demotion by default`
（或指明未开启的原因），以及 `concurrency | the arena holds N full-length ... lane(s)` 的容量提示。

## 5. `NINFER_KV_PAGE_IN`（**默认关**）—— 宿主驻留页的按需回填

### 5.1 它做什么

竞技场小于逻辑上下文时，某些页在 Host tier 而不在 Device。打开本开关后，**每一步在注意力读取该行之前**：

1. 算出这一步的**被寻址集合**（有驱逐窗口时 = sink + 最新窗口 + 选择器保留集；没有窗口 = 整段存活历史）；
2. 其中任何**没有 Device 副本**的页，先 `ensure_host_resident()`，再从 Host 拷进一个 Device 空槽，
   并把该页重新 `publish` 回这一行的页表项（`kAbsentPage = -1` 的语义不变——本步不读的页仍然是 -1）；
3. 没有空槽时，把**本步不读的、逻辑页号最小（即最老）**的一页交给 Host（拷出 + 降级）腾出槽位。

传输走既有的 `DeviceKVPagePool::copy_to_host/copy_from_host`（底层同一个 `cudaMemcpyAsync` 通路，
与 `core::copy_bytes` 的方向语义一致），失败就计数，**不重试、不假设**：拿不回来的页数会变成一个点名
剩余上限的拒绝，而不是让注意力去读一个已被回收的物理页。

### 5.2 它解除了哪三处拒绝

| 位置 | 关（默认） | 开 |
|---|---|---|
| 容量下注的选择 A（专家全驻留、KV 退到 Host） | 判为"不可执行（没有常驻 KV 腿）" | 可以取 A，竞技场按剩余字节建成更小 |
| `NINFER_KV_EVICT_CAPACITY` 对全注意力模型 | 启动直接报错 | 允许，只要一步的被寻址集合装得下 |
| 一步的守卫 / 准入 | "不能从 Host tier 流式读" | 报出 `prompt_tokens` / `resident_tokens`：竞技场本身才是上限 |

### 5.3 它**没有**解除的那条限制（重要）

一个页组横跨**所有层**的 K/V plane（`state/decoder_state.cpp:66-81`），而全注意力层**每步读整段历史**。
因此"某一步的被寻址集合"就是整段存活历史，回填不能让它变小——竞技场必须装得下**单条请求的历史**。
超出时在准入处拒绝（`KV residency cannot serve this request: prompt_tokens=… resident_tokens=…`），
而不是静默算错。要真正跑更长的上下文，只能给 KV 更多页（`NINFER_KV_RESIDENCY_SLACK_MB` 调小 /
`--max-context` 调小）。

> 装了窗口（`NINFER_KV_EVICT_WINDOW` / `_SINK` / `_KEEP_*`）时，上面这句要被 §5.6 取代：
> 一步只寻址 sink + 最新窗口 + selector 的 keep 集，准入也按这个集合定价。

### 5.4 需要 Host tier

回填要有地方可读：residency 计划会自己把 `host_kv_payload_bytes` 定进 Host 预算；显式声明竞技场
（`NINFER_KV_EVICT_CAPACITY`）时要自己给 Host 预算（serve：`--host-context-mib`）。两者都没有时**在构造期**
报错，而不是等到第一步。

### 5.5 验证（本机实测）

* 27B dense / bf16 / `--context 8192`，竞技场声明 48 页（3072 token，< 8192）：
  `NINFER_KV_PAGE_IN_PROBE=4` 把刚回填好的 4 页再交还 Host 并再次回填（`[kv-page-in] probe:
  staged=0 demoted=4 refetched=4 missing=0`），生成的 8 个 token 与**全驻留**运行逐位相同；
  3900 token 的提示词在准入被拒（`resident_tokens=3072`）。`ctest -R kv_page_in`。
* 35B e2 / `--max-context 65536` / `--controller-mode performance`（A）+ `NINFER_KV_RESIDENCY_SLACK_MB=19900`：
  `choice=A(expert-resident)`，竞技场 `resident=499/1024 page(s) (48.7%)`、`host_kv=656.2 MiB`；
  23 token 提示词的 8 个 token 与 1024 页运行一致，20.0 tok/s（1024 页：A 24.8 / B 25.0 tok/s）；
  43520 token 提示词在准入被拒（`resident_tokens=31936`）。
* 关掉时门禁逐位不变：27B `7.359779` / `e95cacb864d8b1ab`；35B `4801 / 2.055310 / 7.809254`。

### 5.6 多行准入按"一步读的页集"计价（V10-84，`NINFER_KV_READ_PRICING` 默认开）

§5.3 说的是"没装窗口"的情形。装了窗口后 `stage_step_kv_pages()` 的寻址判据是
`page < sink || page ≥ end − eff_window || page ∈ keep`，V10-79/81 的 page-in 与跨行借用都按这个集合工作。
准入与每步检查此前仍按整段历史定价，于是四行批次买到的竞技场远大于 Σ(sink+window)——
V10-81 的 `borrowed=0` 就是这个原因。V10-84 让两个点改按读集计价：

```
price(行) = fixed + pending(行) + growth(行)      price(新人) = fixed + ceil(min(prompt, chunk) / 64)
fixed    = sink + window + 显式 keep 列表 + 3 × selector 预算      // 与 page_in 的判据同源
pending  = mapped_pages − ceil(committed_frontier / 64)           // 窗口按"已提交"施加，映射跑在前面
growth   = ceil((max(frontier,cursor) + unit)/64) − ceil(max(frontier,cursor)/64)
```

* 只在四件事同时成立时生效：`NINFER_KV_PAGE_IN=1`、竞技场分层、装了窗口、存储内核支持 absent pages；
  `NINFER_KV_EVICT_KEEP_RATIO`（keep 是比例）或 `NINFER_KV_EVICT_DEFER` + selector（掩码被推迟）时
  **自动退回整段历史**。腿关时逐字节不变。
* 不变量：计价集 ⊇ 寻址集，且 Σ 计价 ≤ 竞技场 ⇒ 一步要读的页永远装得下，`page_in` 一定能在
  "本步不读的页"里找到槽（先本行、再按 V10-81 跨行借）。真被打破时走既有精确拒绝，绝不读回收页。
* 实测（27B dense / bf16 / 4 lanes / window 8 / sink 1 / chunk 128）：4×6000 token（Σ 历史 376 页）
  在声明 96 页竞技场准入并跑完，四行 token 与 512 页全驻留**逐位一致**（`spilled=341 fetched=2`，
  probe `demoted=2 refetched=2 missing=0 borrowed=0`）；同二进制 `NINFER_KV_READ_PRICING=0` 第 2 行被拒。
  V10-81 的配置（4×1500 / chunk 1024 / 100 页）买到的仍是 100 页，token 与当时一致。
  `NINFER_KV_BORROW` 的跨行借用在 96/64 页下仍未触发，而这是**预期**：不变量 `Σ计价 ≤ 竞技场` 同时保证 `R−(S−A) ≥ A−(C−R)`，本行 spare 与各行读集已够用，借用只在界被破坏时（`NINFER_KV_EVICT_DEFER` 的掩码被推迟、Host tier 满、被抢占行的 replay）才需要动手。所以 `borrowed=0` 不能当缺陷读。计数链（V10-86 ①）：4 车道 × 24 驻留页恰好填满 96 页池（`pool=96/128`），每步 `spare=0`，但 `addressed_missing=0`——池满、本行无页可让，却没有页需要换回，根本进不了 `reserve` 失败那条分支。借用要同时满足"被寻址页缺副本"与"无空槽"，而换入与降级在顺序步内是一对一腾槽，两者互斥；只有并发步或抢占 replay 才可能同时成立。
* 未闭环：被抢占行的 replay 仍按整段历史预留（`recovery->coverage`）；Host tier 必须装得下被降级的页
  （512 MiB ≈ 128 页，4×6000 需要 ~340 页）；竞技场下限 = `sink + window + 2 × chunk + 1` 页/行，
  所以窗口尺度的竞技场需要小 `--prefill-chunk`；Control 步仍按整段历史（Program 看不到 control token 数）。
  详见 `.temp/v10-84/report.md`。
* DFlash2 的换入腿（V10-86 ②，已闭环）：该 drafter 的草稿 KV 全在 cyclic 本地缓存（`draft_context.h:14`），分页的 `full` 半从不规划，故 `backend_kv_cache()` 返回空（`sequence.cpp:604-607`）、`state.kv->backend` 也为空——**没有可分页的后端空间，换入腿对它没有对象**。`NINFER_KV_PAGE_IN_TRACE=1` 实测：DFlash2 `cache=none bundle=0`，MTP `cache=paged bundle=1`。27B/8192 上下文/48 页竞技场/4 页探针两臂复跑：MTP `fetched=8 spilled=8`、DFlash2 `fetched=4 spilled=4`，**两臂 token 都与全驻留逐位一致**（`381 21103 1083 279 24112 1495 1518 279`）。测试按结构区分期望（MTP 2×probe、DFlash2 1×probe），不放低阈值。

## 6. 容量下注 A/B 的代价（V10-82）

下注的三条规则（`core/capacity_bet.h`）本来是**优先级**，不是价目表。V10-82 给它接了两笔实测成本：

* **量的来源**：`core/transfer_rate.h` 在构造引擎时（仅当 `NINFER_KV_PAGE_IN` 打开）用本机 H2D
  通路测两种真实负载——一个 KV 页（本机 1280 KiB，52 份/轮）与一个专家槽（1888 KiB，35 份/轮），
  每轮 4 次预热、3 轮取最好，打印 `[plan] transfer leg: page_in=… expert=… free=…`。
  本机实测 **23.1 / 23.2 GB/s**（多次构造 22.9–23.5）。
* **公式**（`core/transfer_cost.h`，纯函数）：A 每步 `(被寻址页 − 驻留页) × 页字节 / 页速率`；
  B 每步 `(每步路由槽 − 驻留槽) × 槽字节 / 槽速率`。`[plan] capacity …` 行尾追加
  `cost_a_ms= cost_b_ms= page_in_gbps= expert_gbps= cheaper=`。
* **实测结论（35B e2，ctx 32768，prompt 24000，decode 96）**：A（池 18879 MiB、KV 397/512 页）
  每步回填 **0 页 / 95 步**，8.705 ms/token 对全驻留 9.064（更快，不是更慢）——因为 §5.3 那条准入
  限制让"被寻址集合永远装得下"。B 的**斜率**对得上（0.0449 ms/MiB 实测 vs 0.0455 预测，1.1%），
  但每步另有一笔 **22.9 ms 的固定开销**在第一次 miss 就出现（40 层各 upload+sync，模型未计价）。
* 因此目标函数只在**精确的那一格**动手：B 的账单严格为 0（零 miss 就是零拷贝）时才把字母改成 B；
  否则只打印 `cheaper=` 并附 `not_applied=unpriced_fixed_stream_cost`。腿关时该行与 V10-82 之前逐字相同。
* 未闭环：A 的真实代价是**上下文上限**（驻留页 × 页 token），不是每步毫秒；固定开销还没有可测的价目。
  详见 `.temp/v10-82/report.md` 的预测/实测表。

