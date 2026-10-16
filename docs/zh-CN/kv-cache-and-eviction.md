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

不满足时保持 `off`，并在启动日志里**指明哪一条不满足**与对应杠杆（降并发 / 减小 `--kv-capacity-headroom-mib` /
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
