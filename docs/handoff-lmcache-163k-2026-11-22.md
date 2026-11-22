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

## 4.6 `--kv-capacity auto` 的两个前提（2026-11-22 晚）

引擎支持 `--kv-capacity auto`（`KvCapacityPolicy::Automatic`）：按“权重加载后剩余的 device 预算 − `--kv-sizing-headroom-mib`”反算页数，实测在本机给出 **4000 pages = 256,000 token**（正好覆盖 256k 上下文），日志里 `kv_capacity auto [argv]`、`[kv-plan] ... mode=auto`。

但 auto 只解决容量，另外两件事必须显式：

1. **`--kv-residency-slack-mib 1536` 不能交给 auto。** auto 的默认是 `device/12`（本机约 381 MiB），KV 会把 device 吃满，restore 之后的 decode kernel 直接 `cudaErrorMemoryAllocation`：
   ```
   [host-context] pinned 2398 MiB extent refused (cudaMallocHost failed ... 2398 MiB requested)
   scalar.cu:11: CUDA_CHECK(cudaGetLastError()) failed: cudaErrorMemoryAllocation
   ```
   原因是 4090 上还有 15.6 GiB 权重和 WDDM 可驱逐预算，余量必须按“权重之后还剩多少”给，而不是按 device 总量除以 12。
2. **`--kv-sizing-headroom-mib`（默认 1 GiB）决定 auto 留多少不用于 KV**。想让容量更大就调小它，但和上面那条余量叠加，别同时压到 0。

实测（auto + slack 1536）：163k `cache_n 163680`、`restore split=4 total 15.452 s`、端到端 23.0 s。与显式 256000 的 16.7 s 差在服务端状态，不在容量策略。

生产启动脚本：`H:\TeleAgentWorks\.temp\kv\start_prod.cmd`（`--kv-capacity auto --kv-residency-slack-mib 1536`，分片/part/超时全部由客户端推导）。

### 4.6.1 服务端长测后要重启

同一个 6555 进程连续跑了十几轮 163k/32k 的 store+restore 之后，RSS 从 9.1 GB 涨到 **14.3 GB**，lookup 开始出现 30 s 超时（`has 30.005 s`），restore 也随之失败。这不是容量或锁的问题，是服务端进程自身的状态累积。**每次正式测量前重启一次服务端**，否则数字不可比。

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
