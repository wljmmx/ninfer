# LMCache 前缀缓存交接文档（2026-11-22）

> 目标：让 ninfer 引擎把 KV checkpoint 存进 LMCache，并在后续请求中复用，覆盖 agent 多轮对话的增长场景。
> 结论先行：**16k 与 32k 已可用；163k 写入完整但读取卡在 5 秒；对话裁剪不可复用（设计使然）。**

## 1. 可用矩阵（全部为实测）

| 场景 | 服务端 | 结果 |
|---|---|---|
| 16k 跨进程恢复 | 6555（`a5c2364` + 3 patch） | `cache_n 16,060`（100%），restore 1.72 s，端到端 2.6 s |
| 32k 跨进程恢复 | 6556（`243f95a`） | `cache_n 32,751`（99.98%），11.7 s，restore 5.4 s |
| 多轮增长 A+B+C → +D | 两者 | 命中 2237/2244，只发增量 `chunk 17 of 23`（12.2 MB KV） |
| 对话裁剪（去掉开头） | 两者 | `cache_n 0`，全量重写 —— 链式哈希的必然结果 |
| 163k 跨进程 | 6556（含区间移植） | 写入完整（38 片，服务端 `Stored 163584 tokens`）；读取 5 s 超时 |

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

3. **5 秒不是超时。**
   `SPLIT=2/4/16`（每片 1.25 GB / 600 MB / 150 MB）全部是 5.00 s，说明是常量。服务端 `blocking reply` 计时探针显示：`ping`/`lookup` 有回复日志，`prepare_retrieve` **一条都没有**——handler 在 5 秒内没有返回。

4. **对话裁剪无法复用，不是 bug。**
   `chunk_hash[i] = H(chunk_i, chunk_hash[i-1])`，序列开头一变，整条链失配。实测 `B+C+D` 命中 0。若要支持，只能换 content-addressed 哈希（上游的 dedup 只在 blend 模块）。

5. **state image 不可压缩。**
   153,954,304 B，zstd/zlib 全部 1.07×，去交错最好 1.16×，熵 7.53 bits/byte。其中 recurrent 占 98.1%。每轮都要重传，是短前缀不划算的主因。

## 4. 163k 的精确卡点

```
写入：publish part 0..37 全部执行，服务端连续 Stored 163584 tokens   ✅
查询：lookup 正常（0.06 s），服务端有 Prefetch request completed 日志  ✅
读取：fetch 5.000 s → receive failed                              ❌
服务端：prepare_retrieve 的 blocking reply 日志从未出现            ❌
```

两条未解：

1. **服务端**：`prepare_retrieve` 处理 320 个对象（600 MB）超过 5 秒没有返回。需要 `py-spy dump` 抓栈，确认卡在 `read_prefetched_results` 还是 `encode_grouped_tensors`。
2. **客户端**：`RCVTIMEO` 明确设为 120 s，但实际 5.00 s 就返回 EAGAIN。说明这个 socket 上有别的机制在 5 秒打断接收（`wait_prefetch` 残留状态、`ZMQ_HANDSHAKE_IVL`、或 `ROUTER_MANDATORY` 的丢弃路径）。

**下一步建议顺序**：
1. `pip install py-spy`，读取进行到 3 秒时 `py-spy dump --pid <server>`（本机 `ptrace_scope=1` 且 sudo 需密码，py-spy 会 Permission Denied，需要放宽或改用 `faulthandler`）；
2. 若栈在编码/读对象，考虑把每片降到 100 MB 以下（`NINFER_RESTORE_SPLIT=24`）再测；
3. 若栈在等锁，查 `read_prefetched_results` 与 prefetch 的锁生命周期。

### 4.1 最后一次观测（2026-11-22 12:2x）

在 `InlineTransferStrategy.prepare_retrieve` 与 handler 都加了分段计时后，跑一次 163k 读取：

- 服务端 `retrieve plan` 计数 = **0**（handler 里的日志一条都没打）
- 服务端无新的 `Error handling operation prepare_retrieve`
- 客户端仍然是 `fetch 5.000–5.017 s → receive failed`

也就是说：**`prepare_retrieve` 这条请求没有到达 handler**。可能的落点是 ZMQ 层（`ROUTER_MANDATORY=1` 下 identity 不匹配会静默丢弃）或 `_handle_request` 的解码分支。

**这改变了下一步的方向**：不再是“handler 慢”，而是“请求没进 handler”。建议先用 `probe_zmq_frames.py` 手工发一条 `prepare_retrieve`（带真实 key 与 instance_id），确认服务端在该路径上是否回帧；若不回，问题在传输层而非业务逻辑。

### 4.2 5 秒的归属仍未定

- 客户端 `RCVTIMEO` = 120 s，但 5.00 s 就 EAGAIN；
- `SPLIT=2/4/16` 全部 5.00 s，与载荷大小无关；
- 服务端无对应日志。

优先怀疑客户端 socket 上的残留状态（`wait_prefetch` 之后复用同一连接），可在 `call()` 里打印每次 `zmq_getsockopt(ZMQ_RCVTIMEO)` 的实际值来确认。

## 5. 环境与入口

| 项 | 值 |
|---|---|
| 引擎二进制 | `R:\infer-release-20261119\ninfer-serve.exe`（保留 `.prev-*` 回滚点） |
| 生产服务端 | WSL `/home/wljmmx/lmcache-src`，6555，8 GB L1，`/home/wljmmx/l3-lmcache.log` |
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
| restore（32k，单消息） | 5.4 s |
| 服务端组装 | 181 MB 读 0.016 s + 编码 0.137 s = 0.153 s |
| 单连接上限 | 254 MiB/s（裸链路 4 连接 722 MiB/s） |
| 短前缀盈亏平衡 | 约几千 token（1736 token 时缓存 1.1 s > prefill 0.8 s） |

## 8. 不要做的事

- 不要把 `NINFER_LMCACHE_ENDPOINT` 设成 `127.0.0.1`（串行 relay，慢一倍）。
- 不要在未打补丁的服务端上使用 6 参数形式（静默失败）。
- 不要用 `--max-context` 推断 KV 容量（`--kv-capacity` 才是）。
- 不要在没有 `NINFER_PUBLISH_PROBE` 的情况下判断“缓存没生效”——失败路径大多是无日志的。
