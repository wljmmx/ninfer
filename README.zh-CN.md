# NInfer

> 精选检查点，单卡推理性能最大化。

[English](README.md) · **简体中文**

NInfer 是面向 Qwen3.5 Dense 与 MoE 架构、从零实现的 C++/CUDA 推理引擎，运行在单张
NVIDIA GeForce RTX 5090（Blackwell，`sm_120a`）或 RTX 4090（Ada，`sm_89`）上。它通过本地
CLI 或 OpenAI / Anthropic 兼容的 HTTP API 处理文本、图像与视频提示。运行时刻意保持专一：
一张 GPU、一个常驻模型、启动时固定的 1–8 条执行车道。

官方工件共五个。快速开始示例使用 Qwen3.8-27B NVFP4；在 RTX 4090 上请改用 groupwise-int
工件（`qwen3_8_27b.ninfer`），因为 NVFP4 张量核路径仅限 Blackwell。

| 模型 | 权重格式 | 工件 | 下载与模型卡 |
|---|---|---|---|
| Qwen3.6-27B | `groupwise-int` | `qwen3_6_27b.ninfer` | [Qwen3.6-27B](https://huggingface.co/neroued/Qwen3.6-27B-NInfer) |
| Qwen3.6-27B | `nvfp4` | `qwen3_6_27b_nvfp4.ninfer` | [Qwen3.6-27B NVFP4](https://huggingface.co/neroued/Qwen3.6-27B-nvfp4-NInfer) |
| Qwen3.8-27B | `groupwise-int` | `qwen3_8_27b.ninfer` | [Qwen3.8-27B](https://huggingface.co/neroued/Qwen3.8-27B-NInfer) |
| Qwen3.8-27B | `nvfp4` | `qwen3_8_27b_nvfp4.ninfer` | [Qwen3.8-27B NVFP4](https://huggingface.co/neroued/Qwen3.8-27B-nvfp4-NInfer) |
| Qwen3.6-35B-A3B | `groupwise-int` | `qwen3_6_35b_a3b.ninfer` | [Qwen3.6-35B-A3B](https://huggingface.co/neroued/Qwen3.6-35B-A3B-NInfer) |

每个 v3 `.ninfer` 工件都携带模型配置、编码权重、逻辑绑定与前端资源；运行时结合已实现的模型
与算子能力使用这些事实。你也可以[转换自己的权重](docs/weight-conversion.md)，复用官方配方
或选择其他受支持的格式组合。

当前引擎要求 v3 工件。既有的官方 v2 下载可以在本地
[升级](docs/weight-conversion.md#upgrade-an-existing-v2-artifact)，无需重新下载权重。

## 近期更新

Windows RTX 4090 线现在同时提供**源码构建**与**可直接运行的发布目录**
（`R:\infer`，`v1.0.0-rtx4090`，附带匹配的 `symbols\`）。当前构建（`ba33b85a`，2026-11-08）
新增：

- **运行时控制器**。参数总线登记 22 个运行期参数（带边界与拒绝原因）；三个有实测支撑的
  判据在线生效——并发（专家覆盖率低于 80% 且车道数大于 1 时串行化：8 GiB 档、四路 256
  token 负载，中位 75.7 s 对 87.4 s）、预取（按 usefulness 判定，读前读实测为零即关闭）、
  专家池慢调（每 60 s 最多 512 MiB 一格，3% 回退护栏，只在空闲窗口执行）。
  `--controller-mode auto|stable|performance` 选择策略；serve 暴露 `/control/state`、
  `/control/mode`、`/control/freeze`。整步图捕获仍是构造期决策，运行期如实登记为
  no-executor，而不是假装可翻转。
- **前缀复用有了索引**。KV tier 会持久化 frontier 记录，更长的请求可以恢复自己前缀里
  最长的缓存点，而不再要求整段 prompt 完全相同：397+103 token 命中 397 点前缀
  （TTFT 327 ms 对冷跑 3,957 ms），397+3,703 token 同样命中该点（1,781 ms 对 7,035 ms），
  输出 token 与冷跑逐位一致。
- **运行期池/KV 重分配通过负载验收**（繁忙时拒绝、空闲时零扰动）。KV 边沿在构造宽度内双向弹性，
  控制器可驱动（L1 慢调 60s/格、3% 回退、只在空闲窗口）。首次验收留下的两个 open 均关闭：
  queue→lane hand-off 不会再让 shrink 穿过 idle 判决（判决与变更多次共用 queue 锁；1000 次压测
  尝试、引擎侧“带活请求接受”计数为 0）；降档往返不再发散（tier 用例钉死全驻留，专家落点不再是
  数值变量）。过程中另修三个真实缺陷：已发布 offset 表未失效、增量发布基线被破坏、idle 双读被
  统计相位穿透。
- **q4 分档按实测收敛**。12 个 shape span 改用 sm_89 实测最优核（这些 span 上 −9%…−38%）；
  其余梯子保留 5090 实测值。
- **确定性不变**。默认 PPL `7.441868`、selector-off dump `636e38de95046781`、decode 哈希
  `d093ac2070ab3292` 全部逐位保持。grouped-MoE 困惑度三点按纪律重录
  （`7.108737 / 7.766837 / 6.469093`），因为默认 CPU 专家路径改为镜像设备侧舍入口径；
  位移 0.11–0.17%，经 A/B 实测归因，不作为漂移接受。
- **RTX 4090 基线**已用发布二进制 + `R:\model` 工件重测，并新增 Qwen3.6-35B-A3B。

## 快速开始

NInfer 支持两种单卡目标：

- **RTX 5090（Blackwell，`sm_120a`）**——64 位 Linux（已验证）与完整格式矩阵，包含 NVFP4
  权重与 NVFP4 KV。
- **RTX 4090（Ada，`sm_89`）**——64 位 Linux 或 Windows（MSVC 2022 及以上）。全部
  groupwise-int 路径原生运行，另有经普通 FP8 张量核 MMA 的 FP8 权重，以及完整的 groupwise
  KV 集合：BF16 / INT8 / FP8 / K8V4，加上秩压缩家族（rk8v4、rk4v4、rk4v4-e8、rk2v4-e8、
  rk4v2-e8）。实测的逐布局表见 docs/zh-CN/kv-cache-and-eviction.md。NVFP4 权重与 NVFP4 KV
  需要 Blackwell 张量核，会以明确错误拒绝；在 RTX 4090 上请加载 groupwise-int 工件。

两种目标都需要支持所选架构的 CUDA 12.8 或更新版本、CMake 3.28 或更新版本、C++20 主机
编译器、Ninja、FFmpeg 开发库（`libavformat`、`libavcodec`、`libavutil`、`libswscale`）以及
`libcurl >= 7.85`。`sm_120a` 的验证开发工具链是 CUDA 13.1；`sm_89` 验证的是 CUDA 13.x。
构建只接受 `CMAKE_CUDA_ARCHITECTURES=89` 或 `120a`（默认 `89`）。

构建产品二进制：

```bash
git clone https://github.com/wljmmx/ninfer.git
cd ninfer

cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build -j
```

Linux 上 FFmpeg 与 libcurl 开发包通过 `pkg-config` 解析。Windows 上推荐使用
[vcpkg](https://vcpkg.io) 清单模式与 `x64-windows` 三元组；`vcpkg.json` 固定了依赖集合：

```powershell
cmake -S . -B build-win -G Ninja `
  -DCMAKE_BUILD_TYPE=Release `
  -DCMAKE_CUDA_ARCHITECTURES=89 `
  -DCMAKE_TOOLCHAIN_FILE=<vcpkg-root>/scripts/buildsystems/vcpkg.cmake `
  -DVCPKG_TARGET_TRIPLET=x64-windows
cmake --build build-win -j
```

也可以显式提供外部 FFmpeg 构建（任何带头文件与导入库的共享 FFmpeg 均可）：

```powershell
cmake ... -DFFMPEG_INCLUDE_DIRS=<ffmpeg>/include `
          -DFFMPEG_LIBRARY_DIRS=<ffmpeg>/lib `
          -DFFMPEG_LIBRARIES="avformat;avcodec;avutil;swscale"
```

测试与基准默认不参与构建。`cmake --preset release` 配置相同的产品构建；`cmake --preset dev`
额外启用测试与基准，并查找 Python 3 解释器。`win-release` / `win-dev` 预设选择 RTX 4090 的
MSVC 布局。两个预设都使用 `build/`（或 `build-win/`），并显式重置构建选项。机器相关的编译器
与 Python 路径应写入被忽略的 `CMakeUserPresets.json`。
详见[构建组织与配置](docs/maintainer/build-system.md)。

没有安装目标，也不做二进制分发打包；从源码构建目录直接运行 NInfer。Python 工具独立于 CMake
运行；独立的 HBM 探测工具有自己的[构建命令](tools/README.md#standalone-hbm-probe)。

用 Hugging Face CLI 下载示例所需工件：

```bash
hf download neroued/Qwen3.8-27B-nvfp4-NInfer \
  qwen3_8_27b_nvfp4.ninfer \
  --local-dir models
```

启动一个带两条执行车道与前缀缓存的长驻文本/智能体服务：

```bash
./build/apps/ninfer-serve models/qwen3_8_27b_nvfp4.ninfer \
  --max-context 240000 \
  --kv-capacity 240000 \
  --max-concurrency 2 \
  --kv-dtype fp8 \
  --device-state-slots 2 \
  --spec mtp --draft-tokens 3 \
  --lm-head-draft \
  --preserve-thinking
```

每个请求有 240,000 token 的逻辑上限。共享的 240,000 token Device KV 池服务常驻请求与保留
前缀。请求随执行推进获取 KV 页；在压力下调度器可以暂停请求并在之后恢复。该配置额外提供两个
Device StateImage，以及默认共享 pinned Host 预算：8 GiB 加八个模型 StateImage，用于保留
状态、KV 与暂停快照。

发送一个 OpenAI 风格请求：

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.8-27b",
    "messages": [{"role": "user", "content": "Reply with one short sentence."}],
    "max_tokens": 64
  }'
```

用 32,768 token 分配运行一次性 CLI 请求：

```bash
./build/apps/ninfer models/qwen3_8_27b_nvfp4.ninfer \
  --prompt "Explain prefill and decode, then give a concise conclusion." \
  --max-context 32768 \
  --max-new 8192 \
  --kv-dtype fp8 \
  --spec mtp --draft-tokens 3 \
  --lm-head-draft
```

回答内容写入 stdout。人类可读的启动/运行诊断，以及 CLI 自有的推理、计时、吞吐、内存与
投机解码报告写入 stderr；推理内容与结果报告保持无前缀的产品输出。在终端上，权重物化使用
一行临时进度条，随后是紧凑的 Engine-ready 摘要。重定向的 stderr 收到持续可读的进度，不含
终端控制序列。用 `--log-level debug` 获取完整启动细节。选项与本地输入错误仍保持直接命令
诊断。结构化图像/视频输入使用 `--messages FILE` 与 `--vision`；见 [CLI 指南](docs/cli.md)
与[已提交示例](examples/cli/)。

## 资源感知的长上下文复用

可复用检查点把 KV 与精确 token 前沿处的完整续写状态组合在一起。引擎保留已完成的会话端点
与稳定的输入边界，供多轮与智能体复用。未激活的检查点共享 Device 与 pinned Host 容量；
压力下先回收保留资源，再暂停常驻请求。被暂停的请求从快照恢复，或重放已提交 token 重建状态。

算法见[资源调度与上下文缓存](docs/maintainer/resource-scheduling-and-context-cache.md)；
[Serve TTFT 基准](tools/bench/ttft/)覆盖公共 HTTP 下的热复用、Host 恢复、逐出、共享前缀、
调度边界与多模态加载。

## 性能

已发布测量基于 RTX 5090。[性能索引](docs/performance.md)链接到逐模型运行记录与
[测量规则](docs/performance/methodology.md)。下表摘自这些详细结果。Qwen3.8 使用 FP8 E4M3
row-256 KV；Qwen3.6 使用 INT8 group-64 KV。

### 并发 MTP3 解码

饱和解码使用 CUDA Graphs、MTP3，每个活跃请求一次 8,192 token 生成。吞吐按实际解码 batch
等于配置并发的完整区间内的聚合已提交解码 token 计算。验收覆盖完整请求波次；这些速率是稳态
解码（tok/s）。

| 模型配置 | C=1 tok/s / 接受率 | C=2 tok/s / 接受率 | C=4 tok/s / 接受率 | C=8 tok/s / 接受率 |
|---|---:|---:|---:|---:|
| [Qwen3.6-27B](docs/performance/qwen3.6-27b.md#decode-saturation) `groupwise-int` | 185.8 / 68.2% | 247.0 / 69.0% | 309.5 / 68.4% | 535.0 / 68.3% |
| [Qwen3.6-27B](docs/performance/qwen3.6-27b.md#decode-saturation) `nvfp4` | 202.4 / 69.3% | 399.7 / 71.4% | 699.7 / 69.3% | 1,146.9 / 68.6% |
| [Qwen3.6-35B-A3B](docs/performance/qwen3.6-35b-a3b.md#decode-saturation) `groupwise-int` | 642.5 / 68.6% | 907.2 / 66.3% | 1,213.5 / 69.6% | 1,380.7 / 68.0% |
| [Qwen3.8-27B](docs/performance/qwen3.8-27b.md#decode-saturation) `groupwise-int` | 136.5 / 44.4% | 253.3 / 45.2% | 398.1 / 46.1% | 582.4 / 46.4% |
| [Qwen3.8-27B](docs/performance/qwen3.8-27b.md#decode-saturation) `nvfp4` | 147.7 / 46.2% | 291.0 / 48.7% | 522.2 / 45.8% | 922.4 / 46.1% |

### 单请求服务

串行服务语料使用 CUDA Graphs、1,024 token 预填充块与五个固定种子（预热后）。下表为每个已
发布配置保留一个短预填充、一个极端预填充与一个结构化输出 MTP3 点；完整上下文与场景矩阵
从下方各模型链接。

| 模型配置 | 7,680 token 预填充 | 260,096 token 预填充 | 结构化 MTP3 解码 |
|---|---:|---:|---:|
| [Qwen3.6-35B-A3B](docs/performance/qwen3.6-35b-a3b.md#single-request-speculative-decode) `groupwise-int` | 17,705.4 tok/s | 5,247.0 tok/s | 779.6 tok/s |
| [Qwen3.6-27B](docs/performance/qwen3.6-27b.md#single-request-speculative-decode) `groupwise-int` | 3,218.1 tok/s | 1,614.8 tok/s | 193.0 tok/s |
| [Qwen3.6-27B](docs/performance/qwen3.6-27b.md#single-request-speculative-decode) `nvfp4` | 11,191.5 tok/s | 2,510.6 tok/s | 252.2 tok/s |
| [Qwen3.8-27B](docs/performance/qwen3.8-27b.md#single-request-speculative-decode) `groupwise-int` | 3,331.9 tok/s | 2,139.4 tok/s | 214.7 tok/s |
| [Qwen3.8-27B](docs/performance/qwen3.8-27b.md#single-request-speculative-decode) `nvfp4` | 12,819.1 tok/s | 4,016.4 tok/s | 231.7 tok/s |

### RTX 4090（sm_89）基线

使用发布二进制（`R:\infer\ninfer_bench.exe`）与 `R:\model` 工件，在单张 24 GB RTX 4090、
Windows 11、桌面共享该卡的条件下测量。Qwen3.8-27B 运行 MTP7/MTP4；Qwen3.6-35B-A3B 运行
自己的 DFlash 草稿模型。MoE 行把专家池固定在 19,456 MiB（全驻留）：自动曲线池会让部分专家
落到 CPU，实测单轮离散度可达 3.3 倍。DFlash 行达不到这个固定值——DFlash 的 KV 预留是 431 MiB，
无草稿时为 177 MiB，因此同一固定值实际解析为 18,685 MiB，仍有 423/10,240 个专家留在 host
层；表下的段落把这部分与草稿自身的开销分开。35B 列另有无投机的 8,192 token 点，便于对照有无
草稿 pass 的预填充与解码。

| `ninfer_bench` 用例 | Qwen3.8-27B | Qwen3.6-35B-A3B |
|---|---:|---:|
| `pp512`（INT8 KV，chunk 1024） | 预填充 2,531 tok/s | 预填充 8,671 tok/s |
| `pp2048` | 预填充 3,233 tok/s | 预填充 11,499 tok/s |
| `pp4096` | 预填充 3,215 tok/s | 预填充 11,434 tok/s |
| `pp32768+tg128`（spec 7，rk4v4-e8） | 2,643 / 232.1 tok/s（接受率 100%，长度 8.00） | 未测 ¹ |
| `pp2048+tg128`（spec 7，INT8 KV） | 3,309 / 223.8（接受率 88.0%，长度 7.11） | 2,225 / 522.8（接受率 88.0%，长度 7.11） |
| `pp2048+tg128`（spec 7，rk4v4-e8） | 3,112 / 188.5（接受率 82.6%，长度 6.74） | 2,270 / 546.6（接受率 88.0%，长度 7.11） |
| `pp8192+tg128`（无投机，INT8 KV） | — | 12,377 / 236.1 |
| `pp8192+tg128`（无投机，rk4v4-e8） | — | 12,141 / 234.6 |
| `tg128`（spec 4，rk4v4-e8） | 解码 65.1（接受率 23.0%，长度 1.91） | 解码 234.6（接受率 18.0%，长度 1.72） |
| `tg128`（无投机，INT8 KV） | 解码 45.7 | 解码 248.7 |

¹ MoE 的 32k 点需要全驻留池（rk4v4-e8 下 KV 可容纳，但长预填充超出本轮预算）或仍在调优的
流式配置；这里报“未测”而不是估算。dense 27B 行是长上下文参考。

35B 列有两个需要直说的性质。**DFlash 自身的预填充开销很小，表里那 4.6 倍是驻留问题，不是草稿。**
草稿的 context append（特征投影 + 六层 context K/V 投影 + key RMSNorm + RoPE + 草稿 KV append）
在 1,024 token 的块上实测 1.14 ms 与 0.99 ms，而目标块是 88–94 ms，即 1.2–1.4%；目标侧的特征
拷贝再加 0.2–0.4 ms。那 4.6 倍的来源是：DFlash 多出 528 MiB 草稿权重与 254 MiB 草稿 KV，在 24 GB
卡上这 782 MiB 超过了无草稿方案本来只剩 151 MiB 的余量（19,200 MiB 全驻留点之上），于是 423/10,240
个专家落到 host 层，每个路由到它们的层在那一块上走 split 路径。在**同一个池**（18,406 MiB，423 个
未驻留）下实测：无投机 1,731.6 tok/s，DFlash 1,849.0 tok/s——草稿在噪声内，表里的 2,225 tok/s
就是同一池状态。DFlash 仍在解码端收回（88% 接受率下 522–547 tok/s，无投机为 236–249），这才是
它的正当理由。此外 MoE 预填充随驻留程度变化：同一二进制在自动曲线池下 2k/4k 只测得 374–632 tok/s，
因为每层都有一部分落在 CPU 半边。

完整记录、命令行、逐布局精度表与上下文天花板流程：
[Qwen3.8-27B 基线](docs/performance/rtx-4090-sm89-baseline.md) ·
[Qwen3.6-35B-A3B 基线](docs/performance/rtx-4090-sm89-baseline-35b-a3b.md)。

解码受权重带宽约束，KV 布局对它影响小于 7%；布局选择本质是容量与精度的权衡。逐布局 PPL、
每 token 字节数与两种天花板流程见上述两个基线页。

## 评测

能力分数通过 NInfer 的 OpenAI 兼容服务路由测量，开启思考、MTP3 与 EvalScope 1.9.0
（0-shot，规则评分，每题一个样本）：

| 模型配置 | AIME 2025 | AIME 2026 | GPQA-Diamond | ERQA | RealWorldQA |
|---|---:|---:|---:|---:|---:|
| [Qwen3.6-27B groupwise-int](model-cards/Qwen3.6-27B-NInfer/README.md) | 86.67% | 93.33% | 86.87% | — | — |
| [Qwen3.6-27B NVFP4](model-cards/Qwen3.6-27B-nvfp4-NInfer/README.md) | 93.33% | 93.33% | 84.34% | — | — |
| [Qwen3.6-35B-A3B groupwise-int](model-cards/Qwen3.6-35B-A3B-NInfer/README.md) | 90.00% | 90.00% | 85.35% | — | — |
| [Qwen3.8-27B groupwise-int](model-cards/Qwen3.8-27B-NInfer/README.md) | 96.67% | 96.67% | 87.37% | 66.25% | 82.22% |
| [Qwen3.8-27B NVFP4](model-cards/Qwen3.8-27B-nvfp4-NInfer/README.md) | 96.67% | 96.67% | 90.40% | 66.25% | 83.53% |

Qwen3.6 各行使用温度 0.6 与 presence penalty 1.0；Qwen3.8 各行使用温度 1.0 与
presence penalty 0.0。多模态评测使用 `--vision` 与 81,920 token 上下文上限。文本评测使用
262,144 token，Qwen3.8-27B NVFP4 例外（252,928 token，以适配 RTX 5090 减去权重后的空间）。
每题一个样本；模型卡包含正确/总数与评测说明。

## 启动说明

GPU 驻留由进程启动时固定。`--spec` 选择投机解码驻留，`--vision` 独立选择视觉驻留。
Qwen3.6-35B-A3B DFlash 可与 Vision 组合；它加速多模态预填充之后的生成文本解码，而不是
视觉编码本身。

## Docker

在装有 NVIDIA Container Toolkit 的主机上构建运行时镜像：

```bash
docker build --tag ninfer:local .
```

挂载下载的模型并运行相同的示例服务配置：

```bash
docker run --rm \
  --gpus '"device=0"' \
  --publish 8080:8080 \
  --volume "$PWD/models:/models:ro" \
  ninfer:local \
  ninfer-serve /models/qwen3_8_27b_nvfp4.ninfer \
  --host 0.0.0.0 \
  --max-context 240000 \
  --kv-capacity 240000 \
  --max-concurrency 2 \
  --kv-dtype fp8 \
  --device-state-slots 2 \
  --spec mtp --draft-tokens 3 \
  --lm-head-draft \
  --preserve-thinking
```

## 能力与边界

官方工件提供以下能力，可选组件在启动时启用：

- 带思考与非思考提示模式的文本生成；
- 图像、多图、视频与混合多模态消息；
- 分块预填充、精确 batch CUDA Graph 解码，以及启动期固定批大小的批量解码；
- 草稿窗口一至五的 MTP 投机解码；
- BF16、INT8、FP8、NVFP4 与 K8V4 KV 存储；
- 离线因果困惑度评分；
- 私有与共享的精确前缀复用，带 Device/Host 状态与 KV 保留；
- 模型感知的采样默认值与显式采样器覆盖；
- OpenAI Responses Core、OpenAI Chat Completions 与 Anthropic Messages，包含流式、工具、
  本地响应状态、token 计数与用量记账。

35B-A3B 目标额外支持 DFlash：文本与图像/视频 Vision 提示下草稿窗口一至十五。带 DFlash2
伴生权重的 Qwen3.8-27B 工件支持 `--spec dflash2 --draft-tokens 7`，走同一 Text/Vision
Engine 路径，草稿数 1..15，提案头可选完整或优化版本。

产品边界保持刻意小：

- 每个 Engine 一张 RTX 5090 与一个常驻模型；
- 一至八条常驻执行车道，带边界 FIFO 入口；
- 资源压力抢占，配合快照或 token 重放恢复；
- 无优先级/QoS、权重卸载、多 GPU 或分布式服务；
- 活跃请求与保留前缀共享一个启动期固定的 KV 池；
- 模型架构与格式/形状组合使用显式实现的原生路径；
- 解析出的工具调用返回给客户端；NInfer 不执行工具；
- 树内 C++ 头文件不作为已安装 SDK 分发。

`--max-context` 是每条序列的逻辑上限。`--kv-capacity` 决定活跃请求与保留前缀共享的
Main Text KV 池大小；`auto` 在启动期按权重之后剩余的内存解析最大合法容量，并保留 1 GiB
规划余量。显式容量在进程生命周期内固定。

## 文档

- [文档索引](docs/README.md)
- [CLI](docs/cli.md)
- [HTTP 服务](docs/serving.md)
- [性能](docs/performance.md)
- [RTX 4090 基线 — Qwen3.8-27B](docs/performance/rtx-4090-sm89-baseline.md)
- [RTX 4090 基线 — Qwen3.6-35B-A3B](docs/performance/rtx-4090-sm89-baseline-35b-a3b.md)
- [困惑度评测](docs/perplexity.md)
- [权重转换与自定义配方](docs/weight-conversion.md)
- [资源调度与上下文缓存](docs/maintainer/resource-scheduling-and-context-cache.md)
- [Serve TTFT 基准](tools/bench/ttft/)
- [CLI 示例](examples/cli/)
- [中文说明：KV 缓存与显存](docs/zh-CN/kv-cache-and-eviction.md)
- [中文说明：RTX 4090 发布说明](docs/zh-CN/release-notes-v1.0.0-rtx4090.md)
- [贡献指南](CONTRIBUTING.md)

运行相关 `--help` 获取当前选项契约。

## 支持

NInfer 是个人兴趣项目。如果你觉得它有用并希望支持其持续开发，可以
[在 Ko-fi 上支持本项目](https://ko-fi.com/neroued)。

支持完全自愿，不是购买或投资，不附带财务回报、承诺的服务或功能，也不包含项目决策权。
项目方向、优先级、技术选型与发布节奏仍由维护者独立决定。

## 许可

NInfer 以 [Apache License 2.0](LICENSE) 许可发布。

已发布工件派生自 [Qwen/Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B)、
[Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) 与
[Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)。Qwen3.6-27B NVFP4
工件还使用来自
[rdtand/Qwen3.6-27B-PrismaSCOUT-Blackwell-NVFP4-BF16-vllm](https://huggingface.co/rdtand/Qwen3.6-27B-PrismaSCOUT-Blackwell-NVFP4-BF16-vllm)
的固定打包权重。Qwen3.8-27B NVFP4 工件还使用来自
[unsloth/Qwen3.8-27B-NVFP4](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4) 的固定
混合 FP8/NVFP4 权重。这些源仓库以 Apache-2.0 分发。内置依赖在 `third_party/` 下保留各自的
许可证文件。
