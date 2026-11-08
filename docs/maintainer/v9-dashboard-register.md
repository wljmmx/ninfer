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

## T128 Phase 5 汇总门禁

见本文件末尾"T128 门禁数据"。

## T129 收口

Phase 5 全部提交在 master 线性，提交标题带 upstream 哈希；按用户纪律**不建分支、不推送**。
