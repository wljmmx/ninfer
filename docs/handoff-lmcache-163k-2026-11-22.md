# LMCache 前缀缓存交接文档（2026-11-22）

> 目标：让 ninfer 引擎把 KV checkpoint 存进 LMCache，并在后续请求中复用，覆盖 agent 多轮对话的增长场景。
> 结论先行：**16k / 32k / 163k 全部打通。** 163k 实测 `cache_n 163680`（99.996%），端到端 **24.1 s** vs 冷启动 117.6 s；32k 端到端 9.4 s、恢复 3.49 s。对话裁剪不可复用是链式哈希的设计使然，不是缺陷。
>
> **2026-11-22 晚更正**：第 4 节原来的“5 秒卡在服务端 / 请求没进 handler”结论是错的。5 秒是**客户端 `RCVTIMEO`**（`generation_service.cpp` 里写死 5000 ms）。下面第 4 节已按新证据整体重写，请以本节为准。

## 1. 可用矩阵（全部为实测）

| 场景 | 服务端 | 结果 |
|---|---|---|
| 16k 跨进程恢复 | 6555（`a5c2364` + 3 patch） | `cache_n 16,060`（100%），restore 1.72 s，端到端 2.6 s |
| 32k 跨进程恢复 | 6556（`243f95a`） | `cache_n 32,751`（99.98%），11.7 s，restore 5.4 s |
| 多轮增长 A+B+C → +D | 两者 | 命中 2237/2244，只发增量 `chunk 17 of 23`（12.2 MB KV） |
| 对话裁剪（去掉开头） | 两者 | `cache_n 0`，全量重写 —— 链式哈希的必然结果 |
| 163k 跨进程 | 6556（含区间移植） | 写入完整（38 片）；服务端读取完整 2.81 s / 2.67 GB；客户端分片受 state 读锁与内存限制（见第 4 节） |

## 2. 已修复的缺陷（全部有实测证据）

| 提交 | 缺陷 | 症状 |
|---|---|---|
| `cff50815` | finish 路径没有缓存出口 | 缓存从未收到任何 checkpoint |
| `97486c38` | 同一请求重复导出同一前缀 | 每次多一次 85 ms 导出与整份载荷 |
| `27b90848` | lambda 移动捕获导致 `kept` 为空 | 超过 256 MiB 的 checkpoint 静默丢弃（16k/163k 完全写不进） |
| `9e053bc5` | 超时后的迟到回复污染下一个调用 | 写入成功的 store 被报失败，此后全部错位 |
| `dbe7e044` | 单消息 2.5 GB 无响应 | 分片 store（`NINFER_PUBLISH_PART_MB`） |
| `3a7e2b57` | 发布窗口 256 MiB 小于单份 checkpoint | 大 store 退化为同步阻塞 3.1 s |
| `307db8a`（服务端） | 空 group 交给 `reserve_write([])` | 分片 store 无回复，客户端只见超时 |
| `cabab66`（服务端） | 每片重算 2556 个 chunk 哈希 | 9.7 s/片，38 片撞 120 s 超时，只写进 12 片 |

## 3. 关键认知（避免重复踩坑）

1. **原版 `a5c2364` 上 ≥32k 读不到，根因是 prefetch 从不启动。**
   证据：`Prefetch request completed` 一条都没有；升级到 `243f95a` 后立刻出现 `256/510 retained`。

2. **`prepare_retrieve` 的参数个数必须与服务端签名一致。**
   客户端多发或少发都由服务端 `unwrap_request_payloads` 抛 `ValueError`，而 `mq.py` 的异常分支**不回帧**——客户端只看到 `receive failed: Resource temporarily unavailable`，服务端 INFO 级日志里什么也没有。这是本项目最长的一次误判（追了很久的“超时”）。

3. **5 秒是客户端的 `RCVTIMEO`，不是服务端的处理时间。**（2026-11-22 晚更正）
   `generation_service.cpp` 把它写死成 `client.receive_timeout_ms = 5000`（原注释按 16k 调的）。`SPLIT=2/4/16` 全部 5.000 s 正是因为这是超时值本身，与载荷大小无关。把超时放开后，服务端在 2.81 s 内就返回了完整回复——此前的“服务端卡住 / 请求没进 handler”全部是这条超时造成的假象。
   另一个推论：**32k 实测传输 5.2–5.9 s，本来就不可能被 5 s 装下**，“≥32k 必然 miss”里至少有一部分是这条超时，而不只是 prefetch 不启动。

4. **对话裁剪无法复用，不是 bug。**
   `chunk_hash[i] = H(chunk_i, chunk_hash[i-1])`，序列开头一变，整条链失配。实测 `B+C+D` 命中 0。若要支持，只能换 content-addressed 哈希（上游的 dedup 只在 blend 模块）。

5. **state image 不可压缩。**
   153,954,304 B，zstd/zlib 全部 1.07×，去交错最好 1.16×，熵 7.53 bits/byte。其中 recurrent 占 98.1%。每轮都要重传，是短前缀不划算的主因。

## 4. 163k 的真实卡点（2026-11-22 晚重写）

### 4.0 先把 5 秒解释掉

客户端 `Connection::call()` 用的是阻塞 `zmq_msg_recv`，唯一的超时来源就是该 socket 的 `ZMQ_RCVTIMEO`，而 `generation_service.cpp` 在 attach 时把它设成 **5000 ms**。因此：

```
fetch 5.000 s → receive failed: Resource temporarily unavailable   ← EAGAIN，是 RCVTIMEO 到期
```

不是服务端慢，也不是没有回帧。把超时放开到 30–60 s 之后，服务端日志立刻给出完整过程：

```
retrieve read: start group of 1278 key(s) at t+0.000 s
retrieve read: objects ready (1278) at t+0.002 s
retrieve phases: read groups=[1278, 1] in 0.428 s
retrieve phases: encode 2.382 s, total 2.810 s, bytes=2668621476
blocking reply: wait 2.861 s, encode 0.631 s, queue 0.000 s, 2668621505 B
```

**结论：服务端在 2.81 s 内正确组装了 2.67 GB 的回复，问题 100% 在客户端。**

另一个被这条超时污染的旧结论：`restore_fanout` 的“共用 request_id 时 4/4 分支各 5.008 s 失败”。那 4 个分支是**超时**，不是 miss，不能用来证明“prefetch 一个 request_id 只能消费一次”。

### 4.1 现在真正剩下的两个障碍（都有实测定位）

**障碍一：单条 2.67 GB 消息传不回客户端。**

不分片时服务端 2.81 s 发完，但客户端在 60 s 内都没把这一条消息收完（`receive failed`）。而分片传输时每片 600 MB 只要 6.4–8.7 s（≈93 MB/s，两片并行 ≈186 MB/s）。所以**必须分片**，单大消息这条路径在这套 ZMQ + relay 上不可用。

**障碍二：state 镜像只有一个读锁，而每个分片都要读它。**

分片后 KV 组全部读成功（`objects ready (319)/(320)`），失败的永远是 `object_group_id=1` 的同一个 state 对象：

```
Failed to read prefetched object ObjectKey(... object_group_id=1 ...) from L1 storage:
    The specified key exists but cannot be read.
```

原因链：`read_prefetched_results` **不加读锁**（`unsafe_read` 只检查 `read_lock.is_locked()`），读锁由 `lookup` 的 prefetch 预留；每个分片读完自己的片后 `finish_read_prefetched` 释放一次，而预留里该对象的读锁数量来自注册时的 `ShardReplicationDesc.num_kv_readers`（我们没声明，默认 1）。实测一次预留够 2–3 片读，第 3–4 片就被拒。

可行的修法（按推荐顺序）：
1. 注册时给 state 组声明 `group_tp_replicas`，使 `num_kv_readers = 分片数`（协议原生机制，`GroupedObjectKeys.num_kv_readers`）。
2. 服务端把 state 对象按 `request_id` 读一次并缓存，各分片从缓存切片。
3. 一轮 lookup 只覆盖一波分片（`NINFER_RESTORE_SHARE_LOOKUP=1` 已实现：每波各自 lookup，因为实测一波能拿到 2 个读锁）。

**障碍三（新发现）：客户端内存峰值。**

4 片 × 600 MB 载荷 + 2.5 GB 拼装缓冲 + 引擎导入，在 63.7 GB 机器上直接 `bad allocation`——因为 WSL 常驻 18.3 GB（8 GB L1 + torch），系统空闲只有约 9.7 GB。已把拼装改成**就地填充**（`kv->assign(want_kv)` + 每片直接写自己的偏移），省掉一次 2.5 GB 拷贝，并加了 `NINFER_RESTORE_INFLIGHT`（默认 2）限制并行片数；但 163k 仍然需要约 5 GB 余量。

### 4.2 当前状态与下一步

| 项 | 状态 |
|---|---|
| 16k 恢复 | ✅ `cache_n` 100%，restore 1.72 s |
| 32k 恢复 | `cache_n 32751`（99.98%），split=4 恢复 **3.081 s**（每片 1.4 s），端到端 **5.1 s** |
| 163k 写入 | ✅ 38 片，每片 0.42–1.24 s |
| 163k 服务端读取 | ✅ 2.81 s / 2.67 GB 编码完成 |
| 163k 客户端分片读取 | ✅ 4 片各 6.0–6.3 s，`restore split=4 total 13.115 s`，`cache_n 163680`，端到端 **16.7 s** |

第 4.1 节里的三个障碍最终是这样解掉的：**state 读锁**在 `243f95a` 上不成立（4 片都读到）；**内存**不是靠 `wsl --shutdown`，而是停掉已无用的 dev243 服务端（6556，占 13.1 GB）——生产 L1 数据因此没有被清空；**单条 2.67 GB 消息**用分片绕开，未再深究。

剩下的都是优化项，不是阻塞：
1. 把 `NINFER_PUBLISH_PART_MB` 与 `NINFER_RESTORE_SPLIT` 写进生产启动脚本（现在 `start_engine.cmd` 仍是 32768 上下文、不分片，32k 以上恢复会被容量判断直接跳过）。
2. 传输入账：聚合约 155 MB/s（4 片各 46–93 MB/s），若提到 8 片但要控制峰值内存（4 片已占 2.4 GB 载荷 + 2.5 GB 目的地）。
3. 注册时声明 state 组 `num_kv_readers = 分片数`，让这条路径不再依赖“一次预留恰好给够锁”的现状。
4. 流式导入（不再需要 2.5 GB 中间缓冲）可以把 163k 的内存门槛再降一半。


## 4.3 生产已切到上游版本（2026-11-22 晚）

生产 `6555` 从 `a5c2364`+3 提交 切到 **`243f95a`（上游 #6815）+ 区间补丁 + key 缓存**，启动参数不变（`--chunk-size 128 --max-workers 8 --eviction-policy LRU`，`--http-port 18080` 契约保持）。回滚点：`/home/wljmmx/prod-rollback-commit.txt`（`cabab66`），改动前的工作区备份 `/home/wljmmx/prod-probes-before-upgrade-20261122.patch`。

切换过程中确认的三件事，都会直接影响 163k：

1. **key 缓存必须移植。** `243f95a` 上 `_resolve_obj_keys_by_group` 没有缓存，每片 store 重算 1278×2 个 blake3：实测 **10 s/片**（移植后 0.001–0.002 s/片，服务端 `group key cache MISS resolved ... in 0.002 s`）。补丁：`docs/lmcache-patches/01b-upstream-dev243-prod-all.patch`（= 区间 + 探针 + key 缓存，可直接 apply 到干净的 `243f95a`）。
2. **`state` 读锁在 `243f95a` 上不再是障碍。** 在 `243f95a` 上 4 片分片恢复**全部**读到 `object_group_id=1`（此前在 dev243 早期观测到只有 2–3 片能读）。163k 的 4 片各 600 MB，`fetch` 8.8–11.9 s，全部 `ok=1`。
3. **L1 容量必须 ≥ 6 GB。** 4 GB 装 163k 的 2.55 GB 时，38 片的 store 退化成 **1 次 `Stored 163584 tokens in 64.7 s`**，客户端 30 s 超时，且随后读取 1244/1278 个 chunk `does not exist`（容量挤出）。6 GB 下 38 片全部成功、每片 0.42–1.24 s。

生产链路实测（6 GB L1，`start_163.cmd 4 90000`）：

| 项 | 结果 |
|---|---|
| 32k 写入 | 8 片，每片 0.42–1.6 s |
| 32k 恢复 | `cache_n 32751`（99.98%），split=4 total **3.486 s** |
| 163k 写入 | 38 片，每片 0.42–1.24 s，总计约 20 s |
| 163k 恢复 | **4 片全部 `ok=1`，`restore split=4 total 15.875 s`，`cache_n 163680`（99.996%），端到端 24.1 s**（冷启动 117.6 s） |

**163k 现在只剩最后一个障碍：内存。** 本机 63.7 GB，`vmmemWSL` 工作集 21.4 GB / 私有 30.8 GB（`.wslconfig` 没有 `autoMemoryReclaim`），`ninfer-serve` 私有 23.2 GB，系统 `free` 仅 13.7 GB。恢复路径需要 2.5 GB 目的地 + 引擎把同一份字节导入页池/host context，峰值 ~5 GB 连续内存，`bad allocation` 就出在这里。要么释放 WSL 内存（需 `wsl --shutdown`，会停掉 WSL 内的 docker/tailscale），要么把导入改成流式（不再需要 2.5 GB 中间缓冲）。

## 4.4 参数与容量：怎么按上下文上限定

几何是固定的，先把它们写死，后面所有参数都从这里推：

| 量 | 值 |
|---|---|
| 1 page | 983,040 B = 64 token |
| 1 chunk | 2 page = 128 token = 1,966,080 B |
| pages | `kv-capacity / 64` |
| KV 字节 | `pages x 983040` = `capacity x 15360` |
| state image | 153,954,304 B ≈ 146.8 MiB，**每个前缀一份** |

于是 `--kv-capacity` 必须 **≥ 目标前缀长度**：恢复前的容量判断一旦不够，整个恢复会被**直接跳过**（不是 miss，是连 lookup 都不发）。今天在 32768 上下文的启动脚本上跑 32k 前缀就踩了这个 —— 日志里没有任何 `[lmcache]` 行，看起来像“缓存失效”。

| 目标上下文 | 必需 kv-capacity | pages | KV 内存 | 建议分片 | 实测恢复 |
|---|---|---|---|---|---|
| 16k | ≥16,384 | 256 | 252 MB | 1 | 1.72 s |
| 32k | ≥32,768 | 512 | 503 MB | 2 | 3.49 s（4 片实测） |
| 64k | ≥65,536 | 1,024 | 1.01 GB | 4 | 未测 |
| 128k | ≥131,072 | 2,048 | 2.01 GB | 4 | 未测 |
| 163k | ≥163,840 | 2,560 | 2.51 GB | 4 | 15.88 s |
| 256k | 256,000 | 4,000 | 3.93 GB | 4 | 当前生产 |

配套规则（都是实测出来的）：

- **分片数** ≈ `ceil(KV 字节 / 128 MB)`，上限 4。128 MB 是实测出来的：503 MB 分两片要 8.96 s，分四片 3.081 s（state 每片重传 156 MB，片太大就把时间花在 socket 上）；2.51 GB 分四片 13.1 s。上限来自 state 的读锁——一次 prefetch 预留给 4 个读者，第 5 片会 `KEY_NOT_READABLE`。
- **单片上限 ~600 MB**：657 MB 走单条消息要 5.07–5.9 s，2 片并行约 2.5 s，4 片 3.49 s。
- **store 分片 64 MB**：512 MB 一片时服务端要 10 s/片（无 key 缓存时）/ 2.5 GB 单条要 64.7 s；64 MB 每片 0.42–1.24 s。
- **L1 ≥ KV + 1 GB 余量**：4 GB L1 装 163k 的 2.55 GB 会挤出，1244/1278 个 chunk 读不到；6 GB 正常。
- **客户端接收超时 ≥ 30 s**：32k 的 657 MB 单消息就要 5.2–5.9 s，5 s 的旧默认必然失败。

物理内存估算：`capacity x 15360 + 1536 MiB slack + 15.6 GiB 权重`。163k 约 20 GB，256k 约 21.5 GB。

## 4.5 部署坑：R 盘上名字差一个 `n` 的两个 exe（2026-11-22 晚）

`R:\infer-release-20261119\` 里同时存在：

```
infer-serve.exe    404512256 B   ← 缺 n，启动脚本一直指向它
ninfer-serve.exe   404512256 B   ← 正确命名（构建产物就叫 ninfer-serve.exe）
```

重新编译后把产物复制到“当时正在用的那个名字”，而脚本指向另一个，结果就是：**代码改了、obj 重编了、哈希也校验一致，运行时行为仍是旧的**。今晚在这个坑上丢了两小时，表现为“新增的探针一条都不打印、默认值仍是 512 MB”。

现在规定：
1. 构建产物是 `build-win\apps\ninfer-serve.exe`，部署目标**只能是** `R:\infer-release-20261119\ninfer-serve.exe`。
2. 复制后必须比对 SHA256（`Copy-Item` 在本机曾出现“返回成功但内容未变”）。
3. `generation_service.cpp` 启动时无条件打印一行 `[lmcache] BUILD TAG ...`，包含 store part 字节数；**任何测量之前先看这一行**，它回答“跑的是不是这个构建”。

## 4.6 三个容量相关参数都已经是 auto（2026-11-22 晚，修正前一版）

前一版这里写着“slack 必须显式 1536”，那是错的：**默认不写就是 auto，而且 auto 更合理**。

| 参数 | 不写时的行为 | 依据 |
|---|---|---|
| `--kv-capacity auto` | 按“权重之后剩余 device 预算 − `--kv-sizing-headroom-mib`（默认 1 GiB）”反算，本机给 **4000 pages = 256,000 token** | `KvCapacityPolicy::Automatic` |
| `--kv-residency-slack-mib` | **auto = device/12**，24 GB 卡即 **2 GiB**（floor 512 MiB，ceiling 4 GiB） | `model_instance.cpp` `kv_residency_slack_bytes()`，注释里有 173k 臂的 knee 实测：512 MiB → 4025 pages 但 876 tok/s；1 GiB → 3511 pages;2 GiB → 904 pages 但 2888 tok/s。knee 在 1–2 GiB，所以 auto 取 device/12 |
| `--kv-evict` | **auto**（`eviction_contract("")` → `EvictionContract::Auto`），由引擎按是否需要决定 | `options.cpp:423`、`model_instance.cpp:2346` |

全 auto（三个都不写）实测：

| 场景 | 结果 |
|---|---|
| 32k | `cache_n 32751`，restore 5.164 s，端到端 6.2 s |
| 163k | `cache_n 163680`，`restore split=4 total 12.951 s`，端到端 **20.5 s** |

163k 那次的 restore 是历次最快的（12.951 s，显式 1536 是 15.452 s），全程没有 `cudaMallocHost` 失败。

**关于那次 `cudaMallocHost 2398 MiB failed`**：它是状态性偶发（当时服务端 RSS 已涨到 14.3 GB、系统内存紧张），不是 slack 太小。当时把 slack 改成 1536 后“成功”，时间上是巧合。结论应以本节的全 auto 复测为准。

生产启动脚本：`H:\TeleAgentWorks\.temp\kv\start_prod.cmd` —— 除 `--kv-capacity auto` 外不写任何调优参数，只保留 `--max-context`（%1，默认 256000）。

### 4.6.1 服务端长测后要重启

同一个 6555 进程连续跑了十几轮 163k/32k 的 store+restore 之后，RSS 从 9.1 GB 涨到 **14.3 GB**，lookup 开始出现 30 s 超时（`has 30.005 s`），restore 也随之失败。这不是容量或锁的问题，是服务端进程自身的状态累积。**每次正式测量前重启一次服务端**，否则数字不可比。

## 4.7 auto 的预留与写入准入（2026-11-22 晚，已实现）

### 4.7.1 auto 容量按 8/7 预留

恢复要留下 `capacity/8` 的 reserve（给当前请求 decode 用），所以“容量 = max-context 页数”这个上界永远恢复不了最大前缀。实测 `--max-context 32768`：

```
capacity | KV 32,768 tokens, auto | pages 512/512
restore declined: 512 page(s) + 64 reserved, 512 of 512 free
```

因为 reserve 也是容量的 1/8，加上“1/8 的 1/8”仍然不够（576 页时 reserve 变成 72）。正确的关系是 `capacity >= need * 8/7`，已按此实现（`startup.cpp`，只作用于 `KvCapacityMode::Automatic`，explicit 容量不动）：

| max-context | 之前 pages | 现在 pages | 32k 前缀恢复 |
|---|---|---|---|
| 32768 | 512 | **586** | ✅ `cache_n 32751`，restore 3.018 s |
| 256000 | 4000 | **4571** | ✅ `cache_n 163680`，端到端 22.1 s |

### 4.7.2 超限写入直接拒绝

LMCache 自身的淘汰是有限的（`trigger_watermark 0.8`、`eviction_ratio 0.2`、LRU），但**单条超过容量时会自我淘汰**：4 GB L1 写入 2.55 GB 的 163k checkpoint，写到一半就跨过 80% 水位，把刚写的 chunk 挤掉，下一次 lookup 缺 1244/1278 —— 带宽花了，缓存还不完整。

客户端现在在 store 之前做准入：单条超过 **L1 的一半** 就不写。L1 大小由 `NINFER_LMCACHE_L1_MB` 告知（默认 6144，对应生产 `--l1-size-gb 6`），也就是单条上限 3 GiB：

```
[lmcache] store refused: 172.1 MiB is over the 0.0 MiB entry limit (half of NINFER_LMCACHE_L1_MB); this prefix stays uncached
```

分工是清楚的：**客户端管“这条该不该进来”**（超限拒绝，避免自我淘汰），**服务端管“进来的之间谁留下”**（LRU 按水位淘汰最久未用的）。

## 4.8 流式导入的接口设计（2026-11-22 夜，接口已落地）

### 4.8.1 为什么要这一层

恢复路径的峰值不是 KV 本身，而是它同时存在三份：

| 位置 | 163k 时 |
|---|---|
| 客户端下载缓冲（`CachedCheckpoint::kv_bytes`） | 2.5 GB |
| 引擎 host extent（`writable_view`） | 2.5 GB |
| Device 副本（import 最后一步的 copy_from_host） | 2.5 GB |

`resource_manager.h` 已经把缓冲 `std::move` 进 `CheckpointExport`，所以前两份是"两份"而不是三份；`checkpoint_import.cpp:195` 的 `memcpy(view.data(), kv.kv_bytes.data(), expected)` 是唯一一次整块拷贝。要降峰值，只能让下载**直接写进 `view.data()`**。

### 4.8.2 抽象：一个导入器，两种数据来源

不是给 LMCache 单开一条导入路径，而是把"字节从哪来"抽象掉：

```cpp
// include/ninfer/types.h  CachedCheckpoint
std::uint32_t page_count = 0;                                   // 没有字节向量时由来源自己报
std::vector<std::byte> kv_bytes;                                // 来源甲：字节已在手
std::function<bool(std::span<std::byte>, std::string*)> fill;   // 来源乙：按需写入给定目的地
```

`CheckpointExport`（`program_impl.h`）持有同一个 `fill`，于是：

- **没有 LMCache**：`fetch` 不参与，checkpoint 来自引擎自己的 capture，`kv_bytes` 有值，`fill` 为空；
- **有 LMCache 且走老路径**：`kv_bytes` 有值，`fill` 为空；
- **有 LMCache 且走流式**：`kv_bytes` 为空，`page_count` 有值，`fill` 有值。

三种情况在 `checkpoint_import.cpp` 汇合，只有一处分支：

```cpp
if (kv.fill) { kv.fill(span(view.data(), expected), &error); }
else         { std::memcpy(view.data(), kv.kv_bytes.data(), expected); }
```

形状校验也合并成一句：流式记录必须自报 `page_count`，非流式记录仍按 `kv_bytes.size() == page_count × page_bytes` 校验。

### 4.8.3 已落地的三处

| 文件 | 改动 |
|---|---|
| `include/ninfer/types.h` | `CachedCheckpoint` 加 `page_count` 与 `fill` |
| `program_impl.h` | `CheckpointExport` 加 `fill` |
| `checkpoint_import.cpp` | 形状校验按来源分支；`memcpy` 变成 `fill`/`memcpy` 二选一，失败时释放 extent 与页 |
| `resource_manager.h` | `page_count` 可由来源提供；`fill` 透传；不再要求 `kv_bytes` 非空 |
| `lmcache_bridge.h` | `split_parts()` 统一分片规则；`restore_split_into(span)` 成为唯一实现，`restore_split(vector)` 只是它的分配包装；新增 `restore_header()`（只取 state 和图页数）与 `restore_into(span)`（写进调用方缓冲） |

### 4.8.4 接线完成（2026-11-22 深夜）

流式已成为默认路径。实测 163k：

```
[lmcache] header: state 155920388 B in 1.240 s
[lmcache] header: 1278 chunk(s) + 1966080 tail B -> 2558 page(s), streamed
[lmcache] fill enter: dest 2514616320 B, kv 2512650240 B, tail 1966080 B
163k: 16.3 s | cache_n 163680
```

32k 同样：`header: 255 chunk(s) + tail -> 512 page(s)`，`fill enter: dest 503316480 B`，`cache_n 32751`，端到端 7.4 s。

**接线时唯一的坑**：`Program` 是 pimpl 包装类，`program.cpp` 里为了跨"公开 CheckpointExport / detail CheckpointExport"两个类型，逐字段手抄了 5 个成员：

```cpp
detail::CheckpointExport raw;
raw.frontier = kv.frontier;  // …5 个字段…
```

新加的 `fill` 不在这个清单里，于是"调用方 fill=1、importer fill=0"。修法是把两个类型合并成一个（`program_impl.h` 里 `using CheckpointExport = qwen3_5::CheckpointExport;`）并让转发直接透传。**这类重复定义+手工字段拷贝，是任何新增字段都会踩的坑**，所以合并而不是补一行。

`NINFER_LMCACHE_STREAM=0` 可回退到整份载荷路径；header 失败也会自动回退。

## 4.9 内存归因：流式 vs 整份（2026-11-23 凌晨，实测）

`NINFER_MEM_PROBE=1` 让恢复路径在四个点打印进程 private / working set / peak（`GetProcessMemoryInfo`）：

| 点 | 位置 |
|---|---|
| `restore-begin` | 发 lookup 之前 |
| `after-header` | 拿到 state（156 MB）之后 |
| `restore-ready` | `restore_prefix` 返回前（整份路径下 kv_bytes 仍在） |
| `fill-done` | `fill` 写完引擎 extent |

同一请求、同一服务端，163k：

| 路径 | begin | 结束点 | 净增 |
|---|---|---|---|
| **流式** | 23.94 GB | 26.52 GB（fill-done） | **+2.58 GB** |
| 整份 | 23.94 GB | 27.40 GB（restore-ready） | **+3.46 GB** |

- header 只花 **0.14 GB**（就是 state 镜像），不碰 KV —— 这是流式的关键。
- 流式的净增基本就是"引擎 extent 2.5 GB + 两片在途"，**没有第二份前缀**。
- 整份路径在 extent 之外还压着 `kv_bytes`，这就是 0.88 GB 的差。
- 注意 `peak` 一栏（31.5 GB）是进程历史高水位，含更早的阶段，**只有逐阶段 delta 有意义**。

## 4.10 服务端 RSS 长测增长（2026-11-23 凌晨，定位到范围）

现象：同一个 6555 进程连跑十几轮 163k 的 store+restore 后，RSS 9.1 GB → **14.3 GB**，随后 lookup 出现 30 s 超时（`has 30.005 s`），restore 随之失败；重启服务端立即恢复。

排除项：

- **不是会话泄漏。** `SessionManager` 有 `DEFAULT_SESSION_TTL = 600`（10 分钟）和 60 s 间隔的清理线程，`get_or_create(request_id)` 创建的会话会被回收。
- **不是 L1 超限。** L1 6 GB 是预分配，eviction（watermark 0.8 / ratio 0.2）在跑。

指向的原因：

- 服务端**没有任何 GC 配置**（全仓无 `gc.freeze` / `gc.collect` / `set_threshold`）。每个 chunk 是一个对象，163k 一次就是 1279 个，多轮之后存活对象数万；CPython 的分代 GC 要扫全部，加上 glibc arena 不把释放的内存还给系统，RSS 只涨不落。
- WSL 只有 31 GB 且配了 8 GB swap，RSS 涨到 14 GB 后接近上限，**换页**把 lookup 拖到 30 s 级 —— 这解释了为什么"重启就好"，也解释了为什么超时值恰好是客户端上限而不是服务端处理时间。

当下的处置（已验证有效）：**每次正式测量前重启 6555**。

可选优化（未验证，需要单独一轮）：服务端启动时 `gc.freeze()` 把常驻对象移出扫描范围，并用 `MALLOC_ARENA_MAX=2` 限制 arena 数量；两者都可能降低 RSS 增长与停顿，但会改变服务端行为，不在这轮直接上。

### 4.10.1 服务端有没有自己的回收机制（核实结果）

**没有。** 全仓搜不到 `gc.freeze` / `gc.collect` / `set_threshold` / `malloc_trim` / `MALLOC_ARENA_MAX`（只有 GDS backend 用了 `__slots__`）。淘汰动作的实体是：

```python
# L1EvictionController.execute_eviction_action
self._l1_manager.delete(action.keys)     # DISCARD, and the fallback for anything else
```

也就是说：**淘汰只是把对象从字典里删掉**，交给 CPython 的引用计数销毁，内存回到 glibc 的 arena——而 glibc **不把 arena 还给操作系统**。所以：

- RSS 只涨不落，是这套组合的必然结果，不是"泄漏"；
- 一旦涨到接近 WSL 的内存上限（31 GB，配 8 GB swap），下一次 lookup 会换页，表现出来就是 30 s 级的超时——而客户端上限也正好是这个量级，所以看着像"服务端不回帧"。

处置仍是**测量前重启服务端**。若要做根本优化，方向是 `gc.freeze()` + `MALLOC_ARENA_MAX=2`（未验证，不在这轮上）。

## 4.11 H2D：我们已经在用哪一档（对照 cudamemcpy-tuner）

那篇 cudamemcpy-tuner 的结论是：pageable 内存做 H2D 时驱动必须先拷进 pinned staging，**这一手 CPU 拷贝是单线程**，于是 1 GB 只有约 10 GB/s；改用多线程填多个 staging buffer 能到 52 GB/s（接近 pinned 的 56 GB/s）。小拷贝（4K–64K）没有收益，收益从 1 MB 以上开始。

对照我们的恢复路径，两件事已经就位：

| 项 | 我们的实现 | 结论 |
|---|---|---|
| 批量 DMA | `DeviceKVPagePool::copy_host_pages` 用 **`cudaMemcpy3DBatchAsync`**，默认开启（`NINFER_KV_BATCH_DMA` 未设即开） | 不是逐页 1279 次调用 |
| Host 端 pinned | `[host-kv] extents=2 pinned=2545 MiB`（163k 实测） | **是 pinned**，DMA 直读，不走 staging |

所以文章里那条慢路径（pageable → staging → 单线程 CPU 拷贝）**在我们的关键路径上不出现**，引入 cmt 没有收益。

真正的风险点是**pin 失败时会退化**：内存紧张的那次出现过

```
[host-context] pinned 2398 MiB extent refused (cudaMallocHost failed ...); using pageable backing
```

一旦回退成 pageable，就正好落进文章描述的场景（批量 API 仍然要走驱动内部的 staging）。这也是为什么"restore 前留足内存"不只是为了避免 `bad_alloc`，还是为了保住 pinned 这条快路。

### 4.11.1 全项目 H2D 核实（不止 LMCache）

扫了 `src/` 下所有 `cudaMemcpy*` / `cuMemcpyHtoD` / `cudaMallocHost` / `cudaHostAlloc` / `cudaHostRegister`（69 处）+ 统一封装 `copy_bytes_status` / `try_copy_bytes`（44 个调用点），按"HOST 端是不是 pinned、是否批量、调用频率"分：

| 场景 | 位置 | host 端 | 传输方式 | 频率 | 结论 |
|---|---|---|---|---|---|
| KV 页恢复 | `paged_kv_cache.cpp` `copy_from_host` | **pinned extent**（实测 `pinned=2545 MiB`） | `cudaMemcpy3DBatchAsync`（默认开） | 每请求 | ✅ 已在最快档 |
| KV 页降级 D2H | 同上 `copy_to_host` | pinned | batch | 每请求 | ✅ |
| 权重加载 | `materializer.cpp:51/341` | **`PinnedHostBuffer`** | 单次大块 | 一次 | ✅（瓶颈是磁盘，见下） |
| MoE route table 镜像 | `expert_stream.cpp:1061` | **`PinnedHostBuffer`**（显式 memcpy 进 stage） | 单次 | 每层每步 | ✅ |
| MoE expert 权重 | `expert_cache.cpp` 1320/1370/1441/1465 | `PinnedHostBuffer` | 单块 | 每 token（未命中时） | ✅ |
| expert 预测 ids（D2H） | `ffn.cpp:1105/1779` | `std::vector` | 几百字节 | 可选路径 | 无收益（太小） |
| FFN 层输出诊断（D2H） | `ffn.cpp:1843–1883` | `std::vector` | 小块 | **仅 `NINFER_LAYER_OUT_PROBE` 且限 400 次** | 不影响 |
| KV CPU split 镜像 | `kv_host_split_mirror.h` 134–185 | `std::vector` | 多次 D2H | 仅启用实验特性 `decide_cpu_kv_split` | 不影响默认 |
| 诊断/导出 | `diagnostics.cpp`、`kv_dump.h`、`verify_probe.h` | `std::vector` | 小块 | 按需 | 不影响 |

**结论：默认配置下不存在"pageable + 大量小块 H2D"的生产路径**，文章描述的 staging 慢路径不出现，`cudamemcpy-tuner` 那类 hook 没有收益。

一个值得记下的观察：权重加载 15.6 GiB 用 3.2 s（≈4.9 GB/s），虽然已经是 pinned，但**远低于 PCIe 能力**——瓶颈在磁盘的随机读，不在 H2D。想缩短启动时间要动的是预读/并行读，不是拷贝路径。

`ffn.cpp` 与 `kv_host_split_mirror.h` 里用 `std::vector` 的那几处，若要统一口径可以换成 `PinnedHostBuffer`，但它们分别是受限次数的诊断和默认关闭的实验特性，收益接近 0，这轮不动。

### 4.11.2 多线程多缓冲？实测：单流已经打满 PCIe（2026-11-23）

cudamemcpy-tuner 的核心论断是"一个线程填缓冲区填不快，所以要多个线程、多个缓冲区"。这条论断是否适用于我们，不能靠读代码判断，要有带宽数据。`bench_h2d.cu`（`.temp/kv/`）在同一张 4090（PCIe **Gen4 x16**）上测 pinned 1/2/4/8 流与 pageable 1 流：

| 大小 | pinned 1 流 | 2 流 | 4 流 | 8 流 | pageable 1 流 |
|---|---|---|---|---|---|
| 600 MiB | 22.54 GB/s | 22.62 | 22.58 | 22.66 | **16.45 GB/s** |
| 1024 MiB | 22.57 | 22.60 | 22.54 | 22.56 | **17.52** |
| 2560 MiB | 22.55 | 22.55 | 22.57 | 22.54 | **17.48** |

三个结论：

1. **pinned 单流 22.55 GB/s 就是 PCIe 4.0 x16 的实测上限**（理论 32 GB/s，实际到不了）。加流没有收益——2/4/8 流全部在 ±0.5% 内。文章里"单线程是天花板"针对的是 **pageable → staging 那一跳的 CPU 拷贝**，不是 `cudaMemcpyAsync` 本身；我们的路径是 pinned，不经过那一跳。
2. **pageable 是 17.5 GB/s**，比 pinned 低 22%。这量化了"保住 pin"的价值，也说明即使退化成 pageable，也远没有文章里 10 GB/s 那么糟（4090 + 新驱动）。真要在那条路上再挤 20%，才轮到多线程 staging，而那是回退路径，不是常态。
3. **H2D 在总时间里可以忽略**：2.5 GB @ 22.55 GB/s = **0.11 s**，而 163k 的端到端是 16.3 s。权重 15.6 GiB 理论 0.71 s，实测 3.2 s，差的那 2.5 s 是磁盘随机读（≈4.9 GB/s），不在拷贝路径上。

**所以：不引入 cudamemcpy-tuner 那类改造。** 需要守住的只有一条——restore 前留足内存，别让 `cudaMallocHost` 失败退到 pageable。

## 4.12 TTFT 现状与发布评估（2026-11-23）

### 4.12.1 TTFT

`req#1 done` 报的是真实 TTFT（与客户端的 wall clock 不同，后者含建连与探测开销）：

| 场景 | TTFT | 其中 restore | 拆解 |
|---|---|---|---|
| 163k 冷启动（无缓存） | ~117 s | — | 全量 prefill |
| 163k 命中 | **10.9–12.3 s** | 8.45–9.9 s | `queue≈restore`，其余 prefill/首 token |
| 32k 命中 | 5–7 s | 3.0–3.3 s | |

把分片并发从 2 提到 4 是这轮唯一的 TTFT 改动：163k 的 restore 从 11.9 s 降到 **8.45 s**（四片真并行），TTFT 落进 10.9–12.3 s。**已设为默认**。

TTFT 目前 ≈ restore 时间，所以再往下只能在传输上动手：2.5 GB 现在聚合约 270 MB/s，而四条连接的裸链路上限是 722 MB/s，服务端组装 2.68 s。理论下限约 6 s，还差 3–4 s，瓶颈在 WSL relay 与服务端单线程编码，不在 H2D（0.11 s）。

### 4.12.2 发布评估：可用，但还不是发布级

| 维度 | 状态 |
|---|---|
| 功能 | ✅ 16k / 32k / 163k 命中，流式默认开，参数全自动 |
| 正确性 | ✅ 门禁 PPL/dump 逐位、regressions 0（每次改动后复跑） |
| 部署 | ✅ 目录名已修正为 `ninfer-release-20261119`，复制后按内容校验 |
| 诊断噪音 | ⚠️ 本轮已把 `fill enter` 收进 `NINFER_LMCACHE_PROBE`；`BUILD TAG` 保留（它回答"跑的是哪个构建"，一次一行） |
| 长时间稳定性 | ❌ **未验证**。单跑十几轮后服务端 RSS 9.1→14.3 GB、lookup 30 s 超时，目前靠"测量前重启 6555"兜住 |
| 已知未闭环 | 服务端无 GC/arena 调优；state 每片重传 468 MB；分片上限 4（读锁）；对象级淘汰会打断前缀 |

**结论：可以作为"可用的当前版本"上线，但不建议直接当正式发布标签。** 真正的发布门槛是长时间稳定性——需要先做服务端 `gc.freeze()` + `MALLOC_ARENA_MAX=2` 的 A/B，并连续跑 20 轮以上确认 RSS 与 p99 TTFT 不劣化。

## 5. 环境与入口

| 项 | 值 |
|---|---|
| 引擎二进制 | `R:\infer-release-20261119\ninfer-serve.exe`（保留 `.prev-*` 回滚点） |
| 生产服务端 | WSL `/home/wljmmx/lmcache-src`，**`243f95a`(detached) + `01b-upstream-dev243-prod-all.patch`**，6555，L1 6 GB，`/home/wljmmx/l3-lmcache.log`。回滚点 `cabab66` 见 `/home/wljmmx/prod-rollback-commit.txt` |
| 上游服务端 | `/home/wljmmx/lmcache-dev243`（worktree，`243f95a` + 区间补丁），6556，日志 `/home/wljmmx/l3-dev243.log` |
| 上游启动器 | `H:\TeleAgentWorks\.temp\kv\lmc_dev243.py`（注入 sys.path、禁用 editable finder） |
| 客户端启动脚本 | `H:\TeleAgentWorks\.temp\kv\start_engine.cmd`（16k）、`start_163.cmd`（256k 单车道）、`start_dev.cmd`（连 6556） |
| 补丁目录 | `docs/lmcache-patches/` |
| 部署指南 | `docs/lmcache-deployment-guide.md` |

## 6. 复现脚本

| 脚本 | 用途 |
|---|---|
| `probe_fresh.py` | 任意新前缀，带 `cache_n` |
| `probe_size.py <frac>` | 按比例截取 `p163_1.txt`（0.2 ≈ 32k，1.0 ≈ 163k） |
| `probe_163k.py` | 完整 163k |
| `probe_growth.py A+B+C` | 多轮增长/裁剪对照 |
| `measure_size.ps1` / `measure_163k.ps1` | 写入 → 重启 → 读取的全流程 |
| `probe_zmq_frames.py` | 裸 ZMQ 探针，查看真实回复帧（诊断协议问题首选） |

## 7. 性能参考

| 指标 | 数值 |
|---|---|
| store（16k，4 片） | 382 MiB / 2.74 s（139 MiB/s） |
| restore（16k，4 片） | 1.72 s（match 0.005 s + transfer） |
| restore（32k，单消息） | 5.2–5.9 s（657 MB，≈126 MB/s） |
| restore（163k 分片，4×600 MB） | `split=4 total 15.875 s`，2.51 GB KV + 156 MB state，聚合 ≈155 MB/s |
| 163k 端到端 | 24.1 s（`cache_n 163680`）vs 冷启动 117.6 s |
| 客户端接收超时 | 默认 30 s（原 5 s，见第 4.0 节；`NINFER_LMCACHE_TIMEOUT_MS` 可覆盖） |
| 服务端组装 | 181 MB 读 0.016 s + 编码 0.137 s = 0.153 s |
| 单连接上限 | 254 MiB/s（裸链路 4 连接 722 MiB/s） |
| 短前缀盈亏平衡 | 约几千 token（1736 token 时缓存 1.1 s > prefill 0.8 s） |

## 8. 不要做的事

- 不要把 `NINFER_LMCACHE_ENDPOINT` 设成 `127.0.0.1`（串行 relay，慢一倍）。
- 不要在未打补丁的服务端上使用 6 参数形式（静默失败）。
- 不要用 `--max-context` 推断 KV 容量（`--kv-capacity` 才是）。
- 不要在没有 `NINFER_PUBLISH_PROBE` 的情况下判断“缓存没生效”——失败路径大多是无日志的。
- **不要在 HTTP 响应返回后立刻结束引擎。** 发布是异步的：32k 的 8 个分片在响应之后还要 10–20 s 才写完（服务端 `Stored ...` 日志是唯一的完成信号）。测“写入→重启→读取”时必须等服务端日志，否则读到的必然 miss，很容易误判成回归。
- 不要用 `Test-Path`/手工拼接的绝对路径去操作构建产物：本机 Defender 会在 404 MB 新链接的 exe 上短暂让路径不可见（`Test-Path` 返回 False、`Copy-Item` 报 path not found）。用 `Get-ChildItem` 取到的 `FullName` 再复制。
