# LMCache 服务端部署与补丁应用指南

> 适用：ninfer（RTX 4090 / sm_89 / Windows 11）作为客户端，LMCache 作为 L3 前缀缓存。
> 最后更新：2026-11-22

## 1. 为什么有这份文档

我们对 LMCache 服务端做了两类改动：

1. **生产环境（6555）**：在 `a5c2364` 上打了 3 个提交（区间参数、空组跳过、key 解析缓存）；
2. **上游验证环境（6556）**：在 `243f95a` 工作树上做了区间移植与诊断探针（未提交）。

`pip install -e .` 或 `git pull` 都会覆盖这些改动，而症状是**静默的**——缓存写成功、读不到，或请求直接超时。所有补丁已导出到 `docs/lmcache-patches/`，升级后用它们重新应用。

## 2. 补丁清单

| 文件 | 内容 | 目标版本 |
|---|---|---|
| `0001-engine_driven-optional-chunk-state-ranges-on-prepare.patch` | `prepare_retrieve` 增加 `chunk_begin/end`、`state_begin/end` | `a5c2364`（生产） |
| `0002-engine_driven-skip-empty-groups-in-a-grouped-store.patch` | 分组 store 跳过空 group（否则 `reserve_write([])` 抛异常、无回复） | `a5c2364` |
| `0003-engine_driven-cache-resolved-keys-so-a-split-store-d.patch` | 按 `(request_id, end, groups)` 缓存 key 解析（9.7 s → 0.003 s） | `a5c2364` |
| `01-upstream-dev243-ranges-and-probes.patch` | 把上述区间能力移植到 `243f95a`，外加 reply 计时探针 | `243f95a` |
| `02-prod-diagnostic-probes.patch` | 生产环境的诊断日志（finish_write 失败计数等），可选 | `a5c2364` |

## 3. 应用方式

### 3.1 生产环境（6555）

```bash
cd /home/wljmmx/lmcache-src
git checkout a5c2364              # 确保基线
git am /mnt/h/TeleAgentWorks/ninfer-4090-port/infer/docs/lmcache-patches/000*.patch
```

或手动 `git apply`。应用后必须重启服务端，改动才会生效。

### 3.2 上游验证环境（6556）

```bash
cd /home/wljmmx/lmcache-dev243     # worktree，检出 243f95a
git checkout -- .                   # 清掉上次的残留
git apply /mnt/h/TeleAgentWorks/ninfer-4090-port/infer/docs/lmcache-patches/01-upstream-dev243-ranges-and-probes.patch
```

**注意**：`243f95a` 需要先生成 gRPC 文件，否则导入即失败：

```bash
/home/wljmmx/lmc-venv/bin/python lmcache/v1/transport/grpc/_proto_gen/_generate.py
```

## 4. 启动参数

### 4.1 生产（6555，8 GB L1）

```bash
nohup /home/wljmmx/lmc-venv/bin/python /home/wljmmx/lmc-venv/bin/lmcache server \
  --transport zmq --host 0.0.0.0 --port 6555 --http-port 18080 \
  --supported-transfer-mode engine_driven --chunk-size 128 \
  --max-workers 8 --l1-size-gb 8 --eviction-policy LRU \
  >> /home/wljmmx/l3-lmcache.log 2>&1 &
```

### 4.2 上游验证（6556）

用 `H:\TeleAgentWorks\.temp\kv\lmc_dev243.py` 作为入口（它把 `sys.path` 指向 worktree 并禁用 editable finder），参数同上，端口 6556 / http 18081。

## 5. 客户端环境变量

| 变量 | 作用 | 说明 |
|---|---|---|
| `NINFER_LMCACHE_ENDPOINT` | 服务端地址 | 必须用 WSL 网卡 IP（如 `tcp://10.222.222.1:6555`），`127.0.0.1` 是串行 relay，慢一倍 |
| `NINFER_LMCACHE_CHUNK_TOKENS` | chunk 大小 | 必须与服务端 `--chunk-size` 一致（128） |
| `NINFER_LMCACHE_RESTORE` | 启用恢复 | 1 |
| `NINFER_LMCACHE_NO_RANGES` | 只用 2 个 payload | **未打区间补丁的服务端必须设 1**；否则服务端在解码阶段抛错且不回帧，表现为 `receive failed` |
| `NINFER_RESTORE_SPLIT` | 恢复分片数 | 需要服务端支持区间；连接数会自动提到该值 |
| `NINFER_PUBLISH_PART_MB` | store 分片大小 | 默认 512；163k 需降到 64 |
| `NINFER_PUBLISH_WINDOW_MB` | 异步发布窗口 | 默认 1024；小于单份 checkpoint 会退化为同步阻塞 |
| `NINFER_PUBLISH_PROBE` | 诊断日志 | `[publish]`、`[lmcache] store/restore` 明细 |

## 6. 验证步骤

`a5c2364`（6555）：16k 前缀应在 2.6 s 内命中。

`243f95a`（6556）：32k 前缀应命中 99.98%，约 11 s。

任何一步如果出现 `prepare_retrieve missed` 且**没有** `Prefetch request completed` 日志，就是 prefetch 没有启动——旧版本（`a5c2364`）的已知缺陷，升级到 `243f95a` 才能解决。

如果出现 `receive failed` 且服务端无异常日志，先检查 `NINFER_LMCACHE_NO_RANGES` 是否与服务端的签名匹配。

## 7. 回滚

客户端二进制：`R:\infer-release-20261119\ninfer-serve.exe` 旁边保留了 `.prev-*` 与 `.bak` 副本。

服务端：`cd /home/wljmmx/lmcache-src && git checkout a5c2364 -- .` 后重启，即回到未打补丁状态（16k 仍可用）。

## 8. 已知限制

| 前缀长度 | 状态 | 原因 |
|---|---|---|
| ≤16k | 可用（生产 6555，100%） | — |
| ≤32k | 可用（上游 6556，99.98%） | — |
| 163k | 写入完整、读取卡 5 s | 见交接文档第 4 节 |
| 任意长度、对话被裁剪 | 不可复用 | chunk hash 是链式的，序列开头变化会导致整条链失配 |

## 9. 排查清单

1. `[lmcache] attached ... N connection(s)` 是否出现？没有就是 attach 失败，先看 `NINFER_LMCACHE_NO_RANGES` 与服务端签名是否匹配。
2. 服务端日志有没有 `Prefetch request completed: X/Y retained`？没有就是 prefetch 没跑。
3. `store incomplete` 是否非零？非零说明 finish_write 有 key 没完成。
4. `[publish] finish produced NO checkpoint` 是否出现？出现说明 capture 层没产出，缓存路径不会被执行。
