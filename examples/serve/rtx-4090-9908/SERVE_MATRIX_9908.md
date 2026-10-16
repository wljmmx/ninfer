# 9908 服务矩阵（RTX 4090 / sm_89 · Qwen3.8-27B v3）

本目录下的 9 个 `run_v3_serve_*.bat` 覆盖 **3 个上下文档 × 3 种 KV 类型**，统一 **端口 9908**，
并默认打开本版可用的优化项。

| 脚本 | 上下文 | 并发 | KV 类型 | 投机解码 |
|---|---|---|---|---|
| `run_v3_serve_128k_c8_rk4v4-e8.bat` | 128K（131072 令牌） | 8 | `rk4v4-e8` | dflash2（draft 7）|
| `run_v3_serve_128k_c8_rk4v4.bat` | 128K（131072 令牌） | 8 | `rk4v4` | dflash2（draft 7）|
| `run_v3_serve_128k_c8_int8.bat` | 128K（131072 令牌） | 8 | `int8` | dflash2（draft 7）|
| `run_v3_serve_256k_c8_rk4v4-e8.bat` | 256K（262144 令牌） | 8 | `rk4v4-e8` | dflash2（draft 7）|
| `run_v3_serve_256k_c8_rk4v4.bat` | 256K（262144 令牌） | 8 | `rk4v4` | dflash2（draft 7）|
| `run_v3_serve_256k_c8_int8.bat` | 256K（262144 令牌） | 8 | `int8` | dflash2（draft 7）|
| `run_v3_serve_1m_c4_rk4v4-e8.bat` | 1M（1048576 令牌） | 4 | `rk4v4-e8` | dflash2（draft 7）|
| `run_v3_serve_1m_c4_rk4v4.bat` | 1M（1048576 令牌） | 4 | `rk4v4` | dflash2（draft 7）|
| `run_v3_serve_1m_c4_int8.bat` | 1M（1048576 令牌） | 4 | `int8` | dflash2（draft 7）|

## 默认打开的优化项

| 开关 | 作用 | 默认值/说明 |
|---|---|---|
| `--kv-evict auto` | 逐出契约：把窗口外的 KV 页掩掉，其设备副本按既定 fate **降级到主机**（保字节、可取回） | 启动期一次判定（容量不足 ∧ 保留比例在实测 <2% 带内 ∧ 主机装得下一次完整抢占 ∧ 该 KV 类型有缺页路径）；**结论与原因打印在启动日志**，不满足时保持关闭 |
| `--kv-idle-reclaim-ms 60000` | 空闲的共享前缀保留项降级到主机，后续同前缀请求取回复用 | 也是本版默认值；实测输出中性（关/开贪心生成逐字一致），首次取回 +30~80 ms |
| `--wddm-evictable-budget` | 按「总显存 − 512 MB」而非「当前空闲」给竞技场定容 | 共享显卡上更不容易因桌面占用而缩水 |
| `--spec dflash2 --draft-tokens 7 --lm-head-draft`（**全九行**） | 投机解码 + LM-head 草稿 | 1M 行位置越过 256K 后自动回退（见下） |

## 逐行实测（本机 RTX 4090 24 GB，发布目录自己的二进制，2026-10-16）

| 行 | 起服 | 竞技场容量 | 真实生成请求 |
|---|---|---|---|
| 128K × C8 × `rk4v4-e8` | ✅ 2m 52.6s | 913,536 令牌（14,274/16,384 页） | 健康检查通过（未发请求） |
| 128K × C8 × `rk4v4` | ✅ 178.9s | 806,080 令牌（12,595 页） | ✅ 24 令牌正常返回 |
| 128K × C8 × `int8` | ✅ 171.9s | 415,232 令牌（6,488 页） | ✅ 正常返回 |
| 256K × C8 × `rk4v4-e8` | ✅ 166.4s | 913,536 令牌（14,274 页） | ✅ 正常返回 |
| 256K × C8 × `rk4v4` | ✅ 159.6s | 806,016 令牌（12,594 页） | ✅ 正常返回 |
| 256K × C8 × `int8` | ✅ 2m 50.7s | 415,232 令牌（6,488 页） | 健康检查通过（未发请求） |
| 1M × C4 × `rk4v4-e8` | ✅ 65.8s（直接跑 .bat 为 70.1s） | 1,229,184 令牌（19,206 页） | ✅ 正常返回 |
| 1M × C4 × `rk4v4` | ✅ 67.3s | 1,084,608 令牌（16,947 页） | ✅ 正常返回 |
| 1M × C4 × `int8` | ❌ 8.1s FATAL（见下） | — | — |
| *替代*：int8 × C4 × 512K | ✅ 77.9s | 558,720 令牌（8,730 页） | ✅ 正常返回 |

容量一栏是引擎自报的 `capacity |` 行；`auto` 按「加载权重后的剩余显存」取最大合法竞技场，
所以同一上下文在不同 KV 类型、甚至不同时刻都会不同。九行里 8 行起服并回归健康，7 行还完成了
一次真实生成请求；剩下那一行见下节。

## 1M 那三行的 dflash2 行为（位置越过 256K 后回退）

dflash2 草稿头的训练范围是 **262,144 令牌（256K）**。早期版本在 `max_context` 超过它时
直接拒绝启动；**该硬拒绝已被附录 FS（S-A 方案）移除**，现在改为**按批次**处理：
某个批次的草稿窗口一旦跨过 262,144（`decode.cpp` 的 `decode_raw` 路由：
`frontier + extent >= draft->max_position_embeddings` 即转 ordinary round），
该批次就交给 **ordinary round**（草稿**永不在训练范围外运行**）。
⇒ 位置还在 256K 以内的请求照常拿草稿加速，越过 256K 之后就是**无投机**的普通解码。
本机实测：`--spec dflash2 --max-context 1048576 --max-concurrency 4` **能启动、能服务**
（见上表；上一版“1M 不能用 dflash2”的说法已作废）。
若要让草稿在整个 1M 范围都不退场，把脚本里的投机行换成 `--spec mtp --draft-tokens 3`
（MTP 不是被降级机制匹配的 masked draft，不受该回退约束）。

## 唯一一个在本卡跑不起来的单元格：1M × int8

`run_v3_serve_1m_c4_int8.bat` 保留了你要的组合（4 并发 + 1M + int8），但它在本机**起不来**：
int8 每令牌约 33 KiB，而 `--kv-capacity` 必须 ≥ `--max-context`，于是 1M 的 int8 竞技场地板就是
约 **34.5 GiB**。实测约 8 秒后失败：

```
FATAL server failed during startup | minimum Engine runtime reservation requires
36979631360 bytes in addition to 1073741824 bytes of automatic headroom, but only
25219825664 bytes are available
```

归因实验：同一 1M + int8，换成 **MTP**（需 39.5 GB）和**完全不开投机**（需 37.0 GB）都是同样的
FATAL，而 1M + rk4v4 不开投机能正常启动 ⇒ **是 KV 类型的地板，不是 dflash2**。
本卡实测能起并服务的最大 int8 口径是 **512K（竞技场 558,720 令牌，4 并发）**；该替代命令已
写在脚本末尾（删掉行首 `rem ` 即可用）。

## 启动日志里的 `queue |` 告警

竞技场只装得下很少几条满长车道时，引擎会警告“排队请求最长要等 N ms，超过
`--pending-timeout-ms`（默认 30000）会被 QueueTimeout 拒绝”。这不是错误，而是容量
事实，可选对策：调大 `--pending-timeout-ms`、调小 `--max-concurrency`、给竞技场更多显存，
或让 `--kv-evict auto` 真的生效（驻留可以小于上下文）。本矩阵九行的实测最坏等待：
128K 约 50–101 s、256K 约 202 s、1M 约 1,210 s，**都超过默认超时** ⇒ 高并发长上下文
实际会退化为“排队 + 部分被拒”，要真正并发起来必须给竞技场更多显存或开启契约。

## `--kv-capacity auto` 的含义

引擎按「权重加载后剩余显存」取**最大合法竞技场**（并保留 1 GiB 定容余量）。
在 24 GB 卡上同时被桌面占用时，实际竞技场通常只有几 GB，**远小于** `max-context × 并发`，
于是启动日志会给出 `concurrency | the arena holds N full-length ... lane(s) but --max-concurrency is M`
的提示：**装不下的并发会排队**，压力下引擎会**暂停车道并把快照推进主机竞技场**（默认 8 GiB + 8 个 StateImage），
恢复时从快照取回、必要时才重算。要获得更多驻留，可减小 `--kv-capacity-headroom-mib`、
或把显示输出接到核显以释放显存。
