<!-- v9 吸收登记：Phase 4 frontend/serving 组（T071–T074），三方比对 = 81c8ce09 / de5d4288 线 / upstream a754fcf3 -->
# v9 吸收登记：frontend / serving

本文件记录 v9 计划 Phase 4 `frontend/serving` 组里**不能整块跟随**的上游提交：三方比对结论、
本地现状与前置条件。能跟随的条目直接进代码与测试，这里只登记缺口与核对结论。

| 编号 | 上游提交 | 结论 | 摘要 |
|---|---|---|---|
| T071 | `bc70d7c8` 原生工具调用参数流 | 登记缺口 | 本地没有 `OutputEvent` / `ToolArgumentsDelta`，参数流需要先落地事件形态 |
| T072 | `58733a86` JPEG 解码对齐 Pillow | 部分跟随 | AVIO 缓冲泄漏已跟随；JPEG 像素对齐需 libjpeg-turbo |
| T073 | `4de9e425` 专用 flash next chat 模板 | 不适用 | 本地没有 Qwen4Exp 转换器，模板无人引用 |

## T071 `bc70d7c8` — 原生工具调用参数流

三方比对：

- 共同基线 `81c8ce09`：输出只有 `OutputDelta{text, reasoning}`，
  `OutputSession::commit_preview()` 返回固定两槽的 `PublishedOutput`，工具调用由
  `ToolCallOutputDecoder` 在终端一次性解析（`feed` 只返回可发布的正文，`finish` 返回
  `Terminal{content, tool_calls, diagnostics}`）。
- 本地 `de5d4288`..HEAD：同一形态，另加语法约束下的工具串自动机（T068/`1ae7c9e8`），
  但 `OutputSink::publish(OutputDelta)`、CLI `apps/cli/main.cpp` 与
  `generation_service.cpp` 的接收器仍然只认正文。
- 上游 `bc70d7c8`：`publish(OutputEvent)`，`OutputEvent = variant<OutputDelta,
  ToolCallStart, ToolArgumentsDelta, ToolCallEnd>`；`ToolCallOutputDecoder::feed` 直接返回
  `vector<OutputEvent>`，只有需要推理的参数才缓冲；参数编码在解码前选定，参数名唯一性由
  推测前瞻/提交/回滚共享的事务式语法状态保证。

结论：**不兼容，登记缺口**。本地缺少的是事件形态本身，而不是某个解析分支；按上游语义做
"只改解析器"的局部移植会退化成另一套半流式实现。

落地前置条件（按依赖顺序）：

1. `include/ninfer/types.h`：`ToolCallStatus`、`ToolCallStart`、`ToolArgumentsDelta`、
   `ToolCallEnd`、`OutputEvent`，`OutputSink::publish(OutputEvent)`；`GeneratedToolCall`
   带 `status`，不完整调用不进入历史。
2. `OutputSession`：`PublishedOutput` 从两槽正文数组改为 `vector<OutputEvent>`，
   `take_tool_calls()`/`finish` 的终态校验改为与流式事件逐项对账
   （上游 `serve/tool_stream_state.h` 提供该对账）。
3. `tool_call_parser`：增量解析改为事件产出，参数名唯一性与 JSON 语法放进
   `text::GrammarSession` 的事务状态（推测前瞻/提交/回滚共用），
   `tool_grammar` 的 `basic_arguments` 改为 `StringDispatch` over 流式 raw/JSON 参数编码
   —— 这正是 T068 已作为 vendored API 预留、但本地尚未接线的形态。
4. `Engine` 事件环、CLI、Chat Completions / Responses / Anthropic Messages、
   `request_log` / `operational_log` 逐层透传 `OutputEvent`；协议层各自做 wire id 与
   不完整调用的投影。
5. 回归：上游 `tests/test_tool_streaming.cpp`（452 行）与 decoder 单测需要同步移植。

在 1–3 落地前，本地保持"终端一次性解析 + 完整调用才发布"的现状；这与上游对**完整调用**
的语义一致，差别只在流式可见性与不完整调用的投影。

## T072 `58733a86` — JPEG 解码对齐 Pillow

三方比对：本地 `src/media/decode/decode.cpp` 与上游父版本同构（自建 AVIO + FFmpeg
解码 + swscale RGB 转换 + EXIF 朝向），所以该提交的**非 JPEG 部分**可以直接跟随：

- 自定义 AVIO 的 32 KiB 缓冲由 `av_malloc` 分配、`avio_alloc_context` 接管，
  `avio_context_free` 只释放上下文、不释放该缓冲；每次 `Decoder` 析构漏 32 KiB。
  已按上游补 `av_freep(&io_->buffer)`（`cleanup()`）。
- `apply_orientation` 只是提取函数，无语义变化，不跟随。

JPEG 部分（`decode/jpeg.cpp`、`JPEG::JPEG` 链接、Pillow 像素金标）**登记缺口**：上游把
JPEG 从 FFmpeg 路径剥离，改用 libjpeg 解码以获得与 Pillow（libjpeg-turbo）一致的色度
上采样与 YCbCr→RGB 取整。本地 Windows 依赖树只有 FFmpeg shared 9.0.2，没有 `jpeglib.h`，
`deps/vcpkg` 的 libjpeg-turbo 也未安装，`find_package(JPEG)` 无法成立。

前置条件：引入 libjpeg-turbo（vcpkg 或预编译前缀）、在 `cmake/Dependencies.cmake` 与
`vcpkg.json` 登记、`src/media/CMakeLists.txt` 链 `JPEG::JPEG`，再移植 `jpeg.cpp`
（含 CMYK/YCCK 与截断 JPEG 的 `JWRN_JPEG_EOF` 处理）和上游金标用例。在此之前本地 JPEG 仍走
FFmpeg MJPEG，像素与 Pillow 存在色度上采样差异。
## T073 `4de9e425` — 专用 flash next chat 模板

三方比对：上游把 `qwen3_8_flash_next.jinja` 作为 **Qwen4Exp 转换器**的默认模板
（`tools/convert/qwen4_exp.py` 覆盖 `chat_template.jinja` 资源，并更新
`docs/weight-conversion.md` 的模板表）。本地 `tools/convert` 只有 `qwen3_5.py`，没有
`qwen4_exp.py`；`tools/chat_templates` 只有 `qwen3_6.jinja` 与 `qwen3_8.jinja`，也没有
Qwen4Exp 的模型规格与转换路径。

结论：**不适用，模板单独落地是死代码**（198 行 Jinja 不会被任何转换器或启动路径引用，
测试也无从建立）。登记为"随 Qwen4Exp 转换器一起吸收"，届时连同模板表与
`tests/text/test_chat_templates.py` 的 flash-next 用例一并移植。