<!-- v9 吸收登记：Phase 4 frontend/serving 组（T071–T074），三方比对 = 81c8ce09 / de5d4288 线 / upstream a754fcf3 -->
# v9 吸收登记：frontend / serving

本文件记录 v9 计划 Phase 4 `frontend/serving` 组里**不能整块跟随**的上游提交：三方比对结论、
本地现状与前置条件。能跟随的条目直接进代码与测试，这里只登记缺口与核对结论。

| 编号 | 上游提交 | 结论 | 摘要 |
|---|---|---|---|
| T071 | `bc70d7c8` 原生工具调用参数流 | 已跟随 | T071a–f 全部落地（a0bb385b / 56b7fb90 / 9fd10749 / ce56ff8a / 3be8716c） |
| T072 | `58733a86` JPEG 解码对齐 Pillow | 部分跟随 | AVIO 缓冲泄漏已跟随；JPEG 像素对齐需 libjpeg-turbo |
| T073 | `4de9e425` 专用 flash next chat 模板 | 不适用 | 本地没有 Qwen4Exp 转换器，模板无人引用 |
| T074 | `d602894a` / `4ec903a0` | 跟随 + 核对 | 布局异常安全已跟随；direct I/O 对齐按 Windows 契约核对 |

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

结论：**已跟随（2026-11-08）**。事件形态、增量解析、StringDispatch 一期语法、
协议投影与上游回归均已落地，逐子项记录见文末"缺口补充规划"。

已完成内容（对应上游语义）：

1. `include/ninfer/types.h`：`ToolCallStatus`、`ToolCallStart`、`ToolArgumentsDelta`、
   `ToolCallEnd`、`OutputEvent`，`OutputSink::publish(OutputEvent)` 与 `OutputDelta` 便捷重载；
   `GeneratedToolCall` 带 `status`，不完整调用不进入历史。
2. `OutputSession`：`PublishedOutput` 为 `vector<OutputEvent>`，提交时正文与 ToolCall* 事件
   同批产出；`take_tool_calls()` 由增量 decoder 的唯一聚合提供，`serve/tool_stream_state.h`
   供协议层逐项对账。
3. `tool_call_parser`：`feed` 返回 `vector<OutputEvent>`，参数编码在解码前选定；raw string
   逐字节流式、JsonValue 按词法流式、只有推断类型参数缓冲到分隔符。参数名唯一性由
   `tool_name_constraint`（`GrammarStateConstraint`）与静态 matcher 共享
   lookahead/accept/confirm/discard 事务；`tool_grammar` 的 `basic_arguments` 使用 T068
   (`1ae7c9e8`) 预留的 `StringDispatch` 同代表达面（声明名 trie + 共享开放名后缀 +
   `StringConstraints` / `XMLToolCallingConverter::ConvertJSONValue`）。
4. `Engine` 事件环、CLI、Chat Completions / Responses / Anthropic Messages、
   `request_log` / `operational_log` 逐层透传 `OutputEvent`；协议层各自负责 wire id 与
   不完整调用投影。
5. 回归：上游 `tests/test_tool_streaming.cpp`（452 行）与 decoder 单测已移植为
   `ninfer_qwen3_5_tool_streaming_test` / `ninfer_tool_call_parser_test`。

与上游的已知差异：本仓库保留 `ToolCallParseFallbackReason` 的 `duplicate_parameter` /
`trailing_content` 取值和 `empty_arguments_omitted` 计数（请求日志 schema 仍为 29），
新 decoder 不再产生这两个 fallback；其余语义与 `bc70d7c8` 一致。

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
## T074 `d602894a` / `4ec903a0` — workspace 布局与 direct I/O 对齐

**`d602894a`（跟随）**：共同基线的 `WorkspaceLayoutBuilder::alloc/alloc_bytes` 先写
`cursor_` 再做溢出检查，失败时留下半更新的 cursor/peak；本地与上游父版本逐字一致，因此
按上游改为局部计算成功后再提交（`src/core/layout.cpp`），并在 `tests/test_arena.cpp`
补布局用例：失败分配保持 peak、后续分配不复用对齐填充、scope 回退后 cursor 正确。

**`4ec903a0`（核对，不引入 per-file 契约）**：上游把该修复建立在
`core/io/file.h` 的 `FileView{offset_alignment, memory_alignment}` 与
`direct_read_window()` 上，对齐来源是 Linux `statx.stx_dio_offset_align`。本地没有这层
抽象：Windows 路径是 `artifact/file_io` 的 `FILE_FLAG_NO_BUFFERING` +
`ReadMode::Direct`，上游的 Host 原地读（`host_destination`）本地也不存在。

本地对齐参数核对（`FILE_FLAG_NO_BUFFERING` 要求文件偏移、缓冲区地址、读写长度都按卷扇区
对齐，Windows 卷扇区为 512/4096）：

| 参数 | 本地来源 | 结论 |
|---|---|---|
| 读偏移 | `materializer.cpp` 按 `kPayloadAlignment=4096` 下取整，块步长 `slot_bytes` 也是 4096 倍数 | 对齐 |
| 读长度 | `align_up(remaining, 4096)`，尾部短块按 EOF 允许的短读处理 | 对齐 |
| 缓冲区地址 | staging 槽为 `PinnedHostBuffer`（`cudaMallocHost`，页对齐）；上游的 pinned slack 用于 per-file 512 契约，本地恒 4096 不需要 | 对齐 |
| 入口校验 | `InputFile::read_direct/begin_direct_read` 对 offset/size/address 三重 `%4096` 校验 | 一致 |

结论：本地常量 4096 同时覆盖 Windows 最小扇区 512 与常见 4K 扇区，比 per-file 契约更严格，
不需要改成按文件查询；反而不能放宽到 512，否则 4K 扇区卷上的无缓冲读会失败。该结论以注释
写入 `artifact/file_io.cpp` 的 `read_direct` 前，防止后续把它"优化"成 per-file 对齐。

验证：`ninfer_arena_test` 通过（含新增布局用例）；`ninfer_media_decode_test` 通过；全量
`cmake --build build-win -j 4` 收敛（`-j 8` 曾遇一次瞬态 `LNK1000`，重试通过）。
`ninfer_artifact_materialization_test` 在 MSVC 下不构建（依赖 GNU ld `--wrap`），
`ninfer_artifact_reader_test` 的 `ios_base::failbit` 在本次改动前的 HEAD 上同样复现，
属既有环境问题，与本次修改无关。
## 缺口补充规划（2026-11-08）

T071/T072 不再停留在“登记缺口”，按下列内部拆解执行；编号不占 T 序位，完成情况写回本表。
每个子任务独立可编译、可复跑，不把“半流式”中间态长期留在树上。

| 子任务 | 依赖 | 内容 | 验收 | 状态 |
|---|---|---|---|---|
| T071a | T068 | types.h 增 ToolCallStatus/ToolCallStart/ToolArgumentsDelta/ToolCallEnd/OutputEvent，OutputSink 增 publish(OutputEvent) 并保留 OutputDelta 便捷重载；GeneratedToolCall 带 status | 全量构建零错；ninfer_public_api_test | ☑ `a0bb385b`：全量重编通过（唯一失败为他人未提交的 `ninfer_ple_bench`），public_api exit=0 |
| T071b | T071a | PublishedOutput 改 vector<OutputEvent>；commit 时正文与 ToolCall* 同批产出；take_tool_calls/finish 与事件逐项对账；engine_core 发布环、request_record 跟随 | 正文逐字节等价；frontend/tool 单测 | ☑ `56b7fb90`：正文投影逐字节等价；新增 `serve/tool_stream_state.h`；tool/constraint 单测通过 |
| T071c | T071b | tool_call_parser.feed 返回事件；参数名唯一性/JSON 语法进 GrammarSession 事务状态；tool_grammar 的 basic_arguments 接 StringDispatch（T068 预留 API） | 增量/重复参数/回滚用例；掩码逐位不变 | ☑ `9fd10749`：GrammarVocabulary + GrammarStateConstraint + tool_name_constraint；basic_arguments 用声明名 trie/开放名后缀 + StringConstraints + ConvertJSONValue；parser/constraint/oracle 全过 |
| T071d | T071c | engine.cpp、generation_service.cpp、CLI、OpenAI chat/responses、Anthropic、request_log/operational_log 透传；协议层各自 wire id 与不完整调用投影 | serve 单测；openai/anthropic schema；SSE 事件序比对 | ☑ `ce56ff8a`：serve/openai/anthropic/request_log 全过；`invalid_tool_output`→500 |
| T071e | T071d | 移植上游 tests/test_tool_streaming.cpp 与 decoder 单测 | 新增用例全过、无回归 | ☑ `3be8716c`：`ninfer_qwen3_5_tool_streaming_test` exit=0；decoder 支持头随 parser 单测移植 |
| T071f | T071e | 本文档改“已跟随”，docs/serving.md 与 types.h 契约注释同步 | 文档与代码一致 | ☑ 本次提交：register/serving/cli/constrained-decoding/engine-architecture/logging 同步 |
| T072a | — | 引入 libjpeg-turbo（vcpkg/离线包，代理 127.0.0.1:10808），vcpkg.json + Dependencies.cmake + media/CMakeLists 链 JPEG::JPEG | find_package(JPEG) 成立、构建零错 |
| T072b | T072a | jpeg.cpp 移植（CMYK/YCCK、JWRN_JPEG_EOF、色度上采样对齐 Pillow） | ninfer_media_decode_test 全过 |
| T072c | T072b | JPEG 像素金标（灰度/CMYK/渐进式/截断） | 逐像素容差内 |

执行窗口纪律：构建前确认 ninja/cl/link 空闲且 `ninja -n` 只剩本批目标；同工作树里 ops 进程有
未提交改动时不动 `include/ninfer/types.h` 等共享头（会触发对方全量重编）；提交只按本组路径
`git commit -- <paths>`，不吞并 qsa 等未提交改动；T071a 与 b–d 在同一窗口内一次完成。

2026-11-08 08:20 状态：代码未开始。当时工作树有 18 个 qsa 文件未提交、构建目录待 CMake 重跑，
且本组要改共享头，故按上面的纪律等待窗口。

2026-11-08 收口状态：T071a–f 全部完成，6 笔本地提交（`a0bb385b`、`56b7fb90`、`9fd10749`、
`ce56ff8a`、`3be8716c`、本次文档提交），未 push。全量构建（`cmake --build build-win -j 8`）
除 `ninfer_ple_bench` 外收敛；该目标因他人未提交的 `bench/ops/ple/fixture.cuh` 引用不存在的
`core/buffer.h` 而失败，与本组改动无关（基线 HEAD 同样失败）。

测试：`ninfer_public_api_test`、`ninfer_tool_call_parser_test`、
`ninfer_qwen3_5_tool_constraints_test`、`ninfer_qwen3_5_tool_schema_oracle_test`、
`ninfer_openai_schema_test`、`ninfer_openai_responses_test`、`ninfer_anthropic_schema_test`、
`ninfer_qwen3_5_tool_streaming_test`、`ninfer_request_log_test` 全部 exit=0。
`ninfer_qwen3_5_frontend_test` 在本次改动前的 HEAD 上就以 0xC0000409 退出（插桩定位到
`test_bpe_merge_order`，早于全部工具用例），属既有环境问题；该文件的渠道投影逻辑已随 T071b
更新，待该故障排除后可直接复跑。