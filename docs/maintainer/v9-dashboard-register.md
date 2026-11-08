# v9 Phase 5 · dashboard 组（T126–T129）登记

- 范围：v9 §7.7 的 T126–T129（上游 `a754fcf3` 线，本地基线 `de5d4288`，共同基线 `81c8ce09`）
- 仓库：`H:\TeleAgentWorks\ninfer-4090-port\infer`（master，本组起点 `2c3d5513`）
- 纪律：不建分支、不推送；每项单独提交，提交标题带 upstream 哈希

| 编号 | 上游提交 | 结论 | 摘要 |
|---|---|---|---|
| T126 | `5b834fc3` + `43fda581` | 搬入（引擎侧本地适配） | `/monitor/info|snapshot|events` 真实落地；`dashboard/` 整搬 |
| T127 | `f008a3d4` + `e70ff85a` + `169d8c1d` + `b1388224` + `b57eaddc` | 跟随（Docker 不采纳） | 布局/主题/TTFT/响应式跟随；Docker/Compose 域外 |
| T128 | 组门禁 | 见文末 | 全量重编＋PPL 逐位＋seam＋regress＋长负载 |
| T129 | 收口 | 登记 | 提交已在 master 线性，未推送（用户指令） |

## T126 监控快照/事件＋面板持久历史 —— 引擎侧本地适配，Web 资产整搬

### 三方比对

- 共同基线 `81c8ce09`：`serve/metrics.cpp` 只有 Prometheus 文本导出；没有 `/monitor/*` 路由，
  没有 `ServiceObservations`，没有 SSE 事件源；`Engine` 只暴露累计 `RuntimeStats`。
- 本地 `2c3d5513`：与共同基线同形态，但多了 Windows 端私有计数（`state_*` / `main_kv_*` /
  `backend_kv_*` / `idle_reclaims*`），且 `docs/performance/methodology.md` 已按 T009 把
  Host/GPU 不相加写成纪律。
- 上游 `5b834fc3`：92 文件、+2216/−380。引入 `include/ninfer/observability/{snapshot,execution,
  request,resources}.h`、`RuntimeStats.requests`、`EngineDescription/EngineSnapshot/
  SnapshotOptions`、`Engine::snapshot()`、`product::observation_json/observation_delta/
  startup_resources_json`、`serve/observations.*`（服务侧计数＋事件环）与 `serve/monitor.cpp`
  （三条路由）；`/metrics` 改成消费 `ServiceObservations`。
- 上游 `43fda581`：新增 `dashboard/`（Vue 前端＋Node 采集器/HTTP API/SQLite 持久历史），39 文件、
  +7969 行，纯 Web 资产。

### 本地落地（引擎侧）

本机是单机单卡 Windows 移植线，`Engine` 没有 `EngineDescription`/`EngineSnapshot`/
`SnapshotOptions`，模型程序也没有上游的 `Execution{result,resolution,retirement}` 分段。
按 T009 的边界表，**只发布本机真正测量到的量**，缺的字段不出现（不以 0 伪装）：

| 文件 | 内容 |
|---|---|
| `src/serve/observations.h/.cpp` | `ServiceObservations`：实例 id／启动时刻／请求登记表／服务侧计数与 TTFT·duration·queue 直方图；`MonitorEvents`（≤4 订阅者、每订阅者 64 事件/64 KiB 有界队列，溢出即断开）；`service_observation_json` |
| `src/serve/monitor.cpp` | `GET /monitor/info`（身份＋模型/rank 描述＋`product::configuration_view` 脱敏配置＋启动预算）、`GET /monitor/snapshot`（`engine`＝`runtime_stats`/`memory_summary`/`media_cache_summary`，`service`＝上表）、`GET /monitor/events`（SSE，复用 `SseTransport`/`prepare_sse_response`） |
| `src/serve/http_server.*` | 注册三条路由；请求生命周期（preparing／first_token／scheduling／done／rejected／failed／response_failed）同时喂 `Metrics`（/metrics 不变）与 `ServiceObservations`；`worker_count += MonitorEvents::kMaxSubscribers`；`attach/listen/stop` 管理事件环生命周期 |
| 三个协议适配器 | `first_token_observer(req_id)` 带上请求 id，事件才能与登记表对齐 |
| `tests/test_monitor.cpp` | 计数/直方图桶/登记表/事件序列/订阅上限/关闭语义，11 条断言 |

口径对齐 T009（对外数字可直接引用）：

- `runtime.host_work.*` 是互斥 Host 相位纳秒；`device_wait_ns` 是阻塞墙钟，**不加**进相位和；
  没有 "GPU+Host" 汇总字段。
- `service.ttft` 在首个已提交 token 观测一次，含 prepare（提交前既成事实）与提交后区间；
  不是 prepare/Vision/prefill 的相加。
- 资源字段保持"显式容量／逻辑占用／物理快照"三分：`kv_capacity*`、`host_context_*`、
  `free_after_startup_bytes` 各自独立，`reserved` 是 `occupied` 的子集。
- 本机不测的量一律缺失：无 `executions` 分段（`{}`）、无 `decode_by_batch`（`[]`）、无
  `prefix_cache` 明细（`{}`）、无 `physical_memory`（`null`，不引 NVML）、无 `experts`/`ple`
  rank 块（`null`）。

### 与上游的已知差异

1. `/metrics` 仍由 `Metrics` 渲染（本地未改成消费 `ServiceObservations`）：两条路径用同一组
   观测输入，Prometheus 输出与门禁基线逐字节不变，避免为监控端点改动既有导出格式。
2. `engine.requests` 为空数组：本机 `Engine` 没有公开的 per-request 快照（上游有
   `RequestSnapshot`）。在线请求走 `service.requests`（id/协议/HTTP id/接收时刻/是否仍在准备）。
3. `engine.ranks[0].hardware` 取 `cuda:<device>`，不是设备名：本机没有把 CUDA 设备属性
   引入 serve 层。

## T127 布局/主题/TTFT/响应式 —— 跟随（Docker 不采纳）

- `169d8c1d` 主题精选＋默认 Paper：`web/theme.ts`、`web/themes/{paper,ember,blueprint,fjord,
  phosphor}.json`、`config.example.json`、`README.md`。
- `b1388224` 遥测栏与 TTFT 分布：新增 `web/components/TtftDistribution.vue`，改 `App.vue`、
  `TelemetryRail.vue`、`split.css`。
- `f008a3d4` 响应式与字体：`App.vue`、`TelemetryRail.vue`、`TtftDistribution.vue`、
  `RequestTable.vue`、`SplitLayout.vue`、`base.css`、`split.css`、`charts/timeseries.ts`。
- `b57eaddc` 布局/图表配置响应式：`charts/timeseries.ts`、`RequestTable.vue`、
  `SplitLayout.vue`、`base.css`、`split.css`、`theme.ts`。
- `e70ff85a` Docker：**域外不采纳**。`dashboard/Dockerfile`、`dashboard/compose.yaml`、
  `dashboard/.dockerignore` 未落地；`README.md` 取 `169d8c1d` 版本（无 Docker 章节），
  `package.json`/`package-lock.json` 保持 `43fda581` 版本（不引入容器脚本与依赖归类调整）。
  根 `.dockerignore` 同样未改。

前端构建验证：本机 Node 24.18 / npm 11.16，`npm ci --ignore-scripts`（130 包）后
`npm run build`（`vue-tsc --noEmit` → `tsc -p tsconfig.server.json` → `vite build`）全部通过，
产物落在 `dashboard/dist/`（已 gitignore，未入库）。

## T128 Phase 5 汇总门禁

执行环境：RTX 4090 / sm_89 / Windows 11 / CUDA 13.4（引擎自报 13.5 runtime+driver）；
`HEAD` = `9a3fa1b8`（T127），工作树干净（仅 `.codegraph/` 未跟踪）。原始日志与报表：
`.temp/t128/`（`build_full.log`、`ppl_35b_*.log`、`regress_fast.log`、`regress_moe.log`、
`seam.log`、`guards.log`、`ctest_serve.log`、`bench_long_35b.json`、`bench_sparse_moe.log`、
`monitor_*.json`、`monitor_events*.txt`）。

### 1. 构建

```
ninja -C build-win -j 8      → 188 个目标，BUILD_EXIT=0，零错误
ninja -C build-win -j 8      → ninja: no work to do
```

本批未改公共头 `include/ninfer/*`，故不触发 depfile 布局问题；改动面是 `src/serve` 与
`tests`，增量重建覆盖 `ninfer_serve`、`ninfer-serve` 与全部链接 serve 的测试目标。

前端：`npm run build`（类型检查＋服务端 tsc＋vite）通过，见 T127 节。

### 2. 35B 双 PPL（逐位）

语料 `moe_corpus_varied.txt`，ctx 512 / stride 256，4801 评分 token：

| 档位 | tokens | mean_nll | PPL | 期望 | 判定 |
|---|---|---|---|---|---|
| default 930 MiB（int16 默认档） | 4801 | 2.051275 | 7.777808 | 4801 / 2.051275 / 7.777808 | 逐位 PASS |
| 冻结 12 GiB（HOST_FREEZE/WARM_STATIC/EXPERT_FREEZE=2） | 4801 | 2.049253 | 7.762104 | 4801 / 2.049253 / 7.762104 | 逐位 PASS |

score rate：default 73.4 tok/s、冻结 148.4 tok/s（仅记录，不作性能结论）。

### 3. 27B regress

`--tier fast`：

```
default PPL 7.441868  expect 7.441868           PASS（逐位）
selector-off dump 636e38de95046781  expect 636e38de95046781   PASS（逐位）
accuracy: 9 hit / 2 miss，11 条全 PASS
regressions: 0
```

`--tier moe`：

```
ctx 1024 stride 512   PPL 7.111129  expect 7.111129   PASS
ctx 512  stride 256   PPL 7.777808  expect 7.777808   PASS
ctx 4096 stride 2048  PPL 6.479183  expect 6.479183   PASS
decode d093ac2070ab3292  expect d093ac2070ab3292   PASS  '2, 3, 5, 7, 11<|im_end|>'
regressions: 0
```

### 4. seam 与守卫

- `ctest -R ninfer_kv_seam_test` → **Passed 0.13 s**（1/1，100%）。
- `python tools/port/check_blackwell_guards.py` →
  `ok: 1258 file(s) scanned, no ungated Blackwell-only call sites`（exit 0；较 T125 的 1255
  增加 3，为本批新增 `observations.h/.cpp`、`monitor.cpp`）。

### 5. serve 面回归（本批实际改动面）

`ctest -R "monitor|metrics|request_log|http_transport|http_error_handler|serve_options|public_api"`
→ **7/7 通过**（2.54 s），含新增 `ninfer_monitor_test`。

### 6. 端点真机验证（非 mock）

`ninfer-serve` 载入 `qwen3_8_27b_v3.ninfer`（`--max-context 2048 --max-concurrency 1`），
两个真实 chat 请求下：

| 端点 | 实测 |
|---|---|
| `/monitor/info` | 3941 B；`model.architecture=Qwen3_5ForCausalLM`、`prefill_chunk=1024`、`kv_cache=bfloat16`、`context_token_capacity_estimate=2048`；`ranks=1`、`configuration` 52 项、`resources.ranks[0].budget` 齐全 |
| `/monitor/snapshot` | 4549 B；`engine.status=ready`、`runtime.prompt_tokens=111`、`computed_prefill_tokens=111`、`committed_decode_tokens=10`、`decode_rounds=10`、`host_work.decode_host_ns=898800`、`host_work.device_wait_ns=347331300`、`device_main_kv_occupied_pages=3`、`service.submitted=1/completed=1`、`service.ttft.sum_seconds=0.1138` |
| `/monitor/events` | 首帧 `event: connected`；真实请求依次产生 `request_preparing`(seq 4) → `first_token`(seq 5, ttft 0.388 s) → `request_completed`(seq 6, prompt 56 / completion 6)；空闲期每 5 s 一条 `: keep-alive` |

结论：端点是引擎真实计数与请求生命周期的投影，**不是 mock**；与本文件的字段边界一致。

### 7. 868–2048 长负载（35B e2 / bf16 KV / CUDA Graph / C=1）

`ninfer_bench --weights qwen3_6_35b_a3b_e2.ninfer -pg "868,512;2048,1024" -r 3 --warmup 1
--max-ctx 4096 -o json`（min 口径另按最快一轮折算）：

| 组合 | prefill tok/s（mean±sd） | decode tok/s（mean±sd） | decode min 折算 | total s（mean） |
|---|---|---|---|---|
| pp868+tg512 | 133.23 ± 0.46 | 44.17 ± 0.35 | 44.45 | 18.13 |
| pp2048+tg1024 | 130.84 ± 3.83 | 41.28 ± 0.05 | 41.35 | 40.50 |

说明：`ninfer_bench` 的 `-r` 给的是重复均值/标准差，无 min 字段；上表 min 列由最快一轮
`decode_seconds` 折算（512/11.5198、1024/24.7669）。本轮未做交错 A/B 对照（dashboard 组
不改算术），故不声称任何提速。

### 8. Phase 5 bench 数据汇总

| 组 | 代表负载 | 数据 | 判定 |
|---|---|---|---|
| T120–T125 MoE execution | `ninfer_sparse_moe_bench --codec all --sweep 1:2048:256 --repeat 200 --flush-mib 96 --cache cold --execution graph` | q4-q5 min 35.840 / 928.768 / 1317.888 us（T=1/257/1025），单权重带宽 510.3 GB/s；q4-q6 34.816 / 995.328 / 1362.944 us，554.2 GB/s；q8-q8 55.296 / 1496.064 / 1993.728 us，552.8 GB/s | 与 T125 同量级（q8 T=1 由 50.176→55.296 us，属轮次波动；本轮无相关代码改动） |
| T120–T125 数值面 | 35B 三点 + 27B MoE 三点 | 见第 2、3 节，全部逐位 | PASS |
| T126–T127 dashboard | 27B 真实 chat + monitor 端点 | 第 6 节；无算子改动 | 数值面由 27B/35B 逐位门覆盖 |
| 长负载 | 35B pp868+tg512 / pp2048+tg1024 | 第 7 节 | 记录 |

### 9. KL 对比表 —— 按 T028 替代口径

本地无 KL 工具（T094/T113/T119 已登记），因此按 T028 的替代口径执行：
**数值（PPL 逐位）＋ 语义（任务套件命中数）＋ 性能（bench）三判据并列**。

| 判据 | 口径 | 本批结果 |
|---|---|---|
| 数值 | 27B default PPL、selector-off dump；35B default/冻结双档 | 全部逐位不变（7.441868 / 636e38de95046781 / 2.051275·7.777808 / 2.049253·7.762104） |
| 语义 | regress `--tier fast` 9 点任务套件 | 9 hit / 2 miss（与基线一致），regressions 0 |
| 性能 | 长负载 + 稀疏 MoE bench | 第 7、8 节；dashboard 组无数值/算子改动，不主张提速 |

结论：本批唯一影响运行时的是 serve 侧新增观测路径（只读计数，不在请求热路径上加锁以外的
工作），数值面逐位不变、任务套件无退化，按 T028 纪律可判"质量门禁通过"。

## T129 收口

Phase 5 全部提交在 master 线性，提交标题带 upstream 哈希；按用户纪律**不建分支、不推送**。

