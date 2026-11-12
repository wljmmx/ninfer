# 模型层分离：Qwen4Exp 与 Qwen3.5MoE 的身份边界（第一轮）

日期：2026-11-13
仓库：`H:\TeleAgentWorks\ninfer-4090-port\ninfer`
环境：RTX 4090 / sm_89 / Windows 11 / CUDA 13.5（`H:\TeleAgentWorks\.temp\cuda135`）
范围：只做模型层分离，不修数值。授权面 `src/models/qwen4_exp`、`src/models/qwen3_5`、
`src/models/registry.*`、`src/models/CMakeLists.txt`、`tests`、`docs`。

## 1. 结论

1. **身份入口**：`models/registry.*` 的 `Architecture` + `Identity` 是唯一的族身份表；
   `src/models/qwen3_5` 内 `Architecture::Qwen4Exp` 出现次数为 **0**（`rg -c` 已核）。
2. **本轮搬移**（纯搬移，无语义改动）：族自己的绑定词汇、QSA 几何与选块执行、HC 四流执行、
   MTP 草稿头的族侧绑定、逐平面导出诊断，共 6 个新文件落在 `src/models/qwen4_exp/` 下；
   `qwen3_5` 侧改为通过 `models/load_rules.h` 索取"族表"，不再书写族的绑定名。
3. **验收**：35B 同语料 PPL **9.619649**（= 9.61965，未变）；Qwen4Exp（`qwen3_8_flash_next_mtp`）
   仍能加载并出 8 个 token；`ninfer_qsa_plan_test` 通过；`ninfer_model_loading`、
   `ninfer_model_runtime`、`ninfer`、`ninfer-perplexity` 构建通过。
4. **未搬移项**在第 5 节逐条列出并给出计数与理由，没有一项是"忘了"。

## 2. 身份边界表

三层：**族专属**（只有该族的工件会走）、**共享**（引擎/内核，族身份不进入）、
**族专属但仍在共享头**（本轮未搬，属下一轮的明确工作项）。

| 层 | 内容 | 位置 | 判据 / 证据 |
|---|---|---|---|
| 族专属 | 架构解析与族身份 | `qwen3_5/config.cpp`（`identity`）、`qwen4_exp/config.cpp`（`identity` + `complete`）、`registry.cpp` | `resolve_architecture` 只认 `Qwen4ExpForCausalLM/ForConditionalGeneration + qwen4_exp_text` |
| 族专属 | 工件绑定词汇：流式残差的分组 norm、修正对、n-gram 嵌入、草稿头尾折 | **`qwen4_exp/load/text.cpp`（新）**，经 `models/load_rules.h` 的 `TextBindingRules` 表调用 | `bind_hc`/`bind_ple`/`bind_final_correction`/`mtp_norm` 的族实现全在此文件 |
| 族专属 | PLE 磁盘层与 PLE 文本阶段 | `qwen4_exp/ple/*`、`qwen4_exp/execution/ple_layer.*` | 表/行读/哈希/预取，独立静态库 `ninfer_ple` |
| 族专属 | PLE 表的开启与 eos | `qwen4_exp/load.cpp`（`open_ple_table`、`ple_eos_token`） | 由 `LoadPlan` 在工件仍打开时调用 |
| 族专属 | QSA 几何与容量 | **`qwen4_exp/execution/qsa.{h,cpp}`（新）** | 24 Q 头 / 2 KV 头 / 4-token 块 / 每页 16 块，规划侧 `startup.cpp` 调它 |
| 族专属 | QSA 选块执行 | **`qwen4_exp/execution/qsa_layer.cpp`（新）** | `TextContext::qsa_mix`、`qsa_tail_view` + 三个 QSA 诊断开关（dump/sync/meta） |
| 族专属 | HC 四流残差执行 | **`qwen4_exp/execution/hc.cpp`（新）** | `hc_stem`、`hc_mix_into`、`hc_combine`、`final_hidden` |
| 族专属 | 四流 MTP 草稿头的族侧绑定 | `qwen4_exp/load/text.cpp`（`mtp_norm`、`mtp_final_correction`） | `mtp/final_hc/norm` 与无注入的修正对 |
| 共享 | 文本工件解析（宿主侧） | `qwen3_5/config.cpp`（`text()`/`moe_text` 结构检查） | 结构字段（`hc_count`/`indexer_*`/`ple_*`）决定解析出的形状，与族名无关 |
| 共享 | 前端（tokenizer、模板、工具调用、输出会话） | `qwen3_5/frontend/*` | 不读族身份 |
| 共享 | Program / 事务 / KV 分层 / 状态镜像 | `qwen3_5/program/*`、`qwen3_5/state/*` | 只按 config 的层类型分派 |
| 共享 | 文本块执行引擎（注意力、GDN、MoE 尾巴、run_layers） | `qwen3_5/execution/{text,attention,gdn,ffn,mtp,expert_*}.cpp` | `qwen3_5::execution::TextContext` 是共享类；族实现以成员定义的形式落在族目录 |
| 共享 | 内核 | `src/ops/**`（含 `qsa`、`gated_residual`、`sparse_moe`） | 只吃张量与枚举，无族名 |
| 共享 | 运行时 | `src/runtime/**`、`src/core/**` | 只经 `LoadPlan`/`Model`/`Program` 接口 |
| 族专属但仍在共享头 | `TextConfig` 的 `hc`/`indexer`/`ple`/`gdn_gate` 字段 | `qwen3_5/config.h` | 结构数据，不含族名；解析后只由形状驱动 |
| 族专属但仍在共享头 | `HcWeights`/`PleWeights`/`HcParameters`/`QsaParameters`/`PleParameters` | `qwen3_5/weights.h`、`qwen3_5/execution/parameters.h` | 作为 `BlockParameters` 的可选成员被共享执行路径持有 |
| 族专属但仍在共享头 | `TextContext` 的 HC/QSA/PLE 成员与声明 | `qwen3_5/execution/text.h` | 本轮只迁走了实现；声明仍在共享类上（见 §5.2） |

族标识在 `qwen3_5` 内的现存数量（`rg -c Qwen4Exp`，全部为注释或结构说明）：

| 文件 | 处数 | 性质 |
|---|---|---|
| `execution/parameters.cpp` / `.h` | 11 / 9 | 族的参数装填与结构说明（§5.3） |
| `execution/text.h` | 5 | 成员声明上的注释（§5.2） |
| `weights.h` | 4 | 权重组记录中的族结构（§5.3） |
| `load/bindings.h`、`config.h` | 2 / 2 | 词汇说明 |
| `execution/ffn.cpp`、`expert_stream.{h,cpp}` | 2 / 2 / 1 | MoE 三分账行窗口的尺寸说明（§5.4） |
| `program/program_impl.{h,cpp}`、`transactions/binding.cpp` | 3 | PLE 历史的持有与提交（§5.5） |
| `state/decoder_state.h`、`workspace.h`、`execution/mtp.cpp`、`execution/text.cpp` | 各 1–2 | QSA 平面/四流根的说明 |

## 3. 本轮搬移清单

全部是"同样的语句换文件"，没有改一行表达式。调用点同步改限定名或 include。

| 新文件 | 行数 | 来源 | 说明 |
|---|---|---|---|
| `src/models/load_rules.h` | 48 | 新增 | 族绑定词汇的接口：`block_norms`、`block_corrections`、`final_correction`、`block_ple`、`mtp_norm`、`mtp_final_correction` |
| `src/models/qwen4_exp/load/text.h` | 14 | 新增 | 族侧 `bind_hc` 与 `binding_rules()` |
| `src/models/qwen4_exp/load/text.cpp` | 93 | `qwen3_5/load/text.cpp` 的 `bind_hc`/`bind_ple`/HC norm 别名/final norm 别名 + `qwen3_5/load/mtp.cpp` 的 final norm | 族的绑定名只此一份 |
| `src/models/qwen4_exp/execution/qsa.{h,cpp}` | 61 + 112 | `qwen3_5/execution/qsa.{h,cpp}` | 命名空间改为 `qwen4_exp::execution`；`startup.cpp` 三处调用改限定名 |
| `src/models/qwen4_exp/execution/qsa_layer.cpp` | 276 | `qwen3_5/execution/text.cpp` 的 `qsa_tail_view`/`qsa_mix` + `qsa_dump_*`/`qsa_mark`/`qsa_meta_probe`/`qsa_sync_probe`/`qsa_dense_read` | 选块与稀疏 GQA 读的全部执行代码 |
| `src/models/qwen4_exp/execution/hc.cpp` | 65 | `qwen3_5/execution/text.cpp` 的 `hc_stem`/`hc_mix_into`/`hc_combine`/`final_hidden` | 四流残差 |
| `src/models/qwen3_5/execution/diagnostics.{h,cpp}` | 18 + 125 | `text.cpp` 的 `nan_probe_tensor` + `qsa_dump_tensor`/`qsa_dump_wants_layer`/`fnv1a64` | 两侧都要用的逐平面导出，移到共享诊断单元（不是族逻辑） |
| `src/models/qwen4_exp/{loading,execution}_sources.cmake` | 9 + 8 | 新增 | 每个族列自己的源文件；`qwen3_5/*.cmake` 不再出现 `../qwen4_exp/*` |
| `tests/models/qwen4_exp/test_qsa_plan.cpp` | 141 | `tests/models/qwen3_5/test_qsa_plan.cpp` | 测试随实现迁移，注册从 `qwen3_5/tests.cmake` 移到 `qwen4_exp/tests.cmake` |

`src/models/qwen3_5/execution/text.cpp` 由 3491 行降到 3055 行（−436），
`qwen3_5/load/text.cpp` 的族绑定名全部换成族表调用。

### 3.1 顺带修复的构建阻塞（非数值改动）

本轮开始时工作树**不可构建**：`config.h` 已把门控改成 `GdnGateKind gdn_gate`
（上一轮把族身份写进 `Identity::output_gate` 后的一半改动），而
`execution/text.cpp` 仍在读 `config_.output_gate_type`（字符串），`config.cpp` 对 `identity()`
的调用又因为参数 `Architecture` 在 `ninfer::models` 里而触发 ADL 歧义
（`qwen3_5::identity` 与 `models::identity` 同名）。

处理：

- `text.cpp` 的 GDN 输出门改为按 `config_.gdn_gate` 的枚举分派（`Silu` → `silu_gated_rmsnorm`，
  `Sigmoid` → `sigmoid_gated_rmsnorm`），名字到枚举的解析与"与该族身份不符就报错"已在
  `config.cpp` 的 `gdn_gate(...)` 完成，因此行为与上一轮登记的修复完全一致：
  35B 走 `Silu`（工件无该键），Qwen4Exp 走 `Sigmoid`（工件声明 `sigmoid`）。
- `config.cpp` 的三处 `identity(...)` 改为完全限定 `::ninfer::models::identity(...)`，
  即回到"按架构分派到族表"的原意（先前的半改若按 `qwen3_5::identity` 解析，
  Qwen4Exp 会拿到 3.5 族的 `silu` 并被解析期拒绝，实机报
  `Qwen3.5 config: output_gate_type 'sigmoid' is not the 'silu' gate this architecture declares`）。

## 4. 验收

| 项 | 命令 / 工件 | 结果 |
|---|---|---|
| 族分支清零 | `rg -c "Architecture::Qwen4Exp" src/models/qwen3_5` | **0** |
| 构建 | `ninfer_model_loading`、`ninfer_model_runtime`、`ninfer`、`ninfer-perplexity`（Ninja, Release, sm_89） | 全部 exit 0 |
| 35B PPL | `ninfer-perplexity qwen3_6_35b_a3b.ninfer --text .temp\ppl_text.txt --context 512 --stride 256 --kv-dtype bf16`，`NINFER_EXPERT_CACHE_MB=14000` | **PPL 9.619649 / mean NLL 2.263808 / 114 tokens**（报告 `profiles/perplexity/qwen3.6-35b-a3b/6e28…/20261112-233219/report.json`） |
| Qwen4Exp 加载出字 | `ninfer qwen3_8_flash_next_mtp.ninfer --prompt "Hello" --max-new 8 --spec none --greedy --print-token-ids` | exit 0，`generated ids 68 168697 1322 268 2022 102411 74751 977`；日志含 `[ple] window: tokens=51 … table rows=816` |
| QSA 计划测试（迁移后） | `ninfer_qsa_plan_test` | `OK qsa_plan (page group 49152 B, pool@32 1572864 B, tail@8 74880 B, indexer 0 B, gqa 6315936 B)` |

### 4.1 预存在的红（不是本轮引入）

`ninfer_qwen4_exp_config_bindings_test` 在 `plan_load` 处退出 1：
`text: missing resource ple_hash_controls`。

证据：`src/models/qwen4_exp/load.cpp`（含 `open_ple_table` 读取
`text/ple_hash_controls`）在本树是**未跟踪文件**，`HEAD` 里没有；测试夹具
`tests/models/qwen4_exp/test_config_bindings.cpp` 的 `resource(...)` 调用只有
tokenizer/tokenizer_config/generation_config/chat_template 四项，没有 PLE 哈希控制量
（280 字节 = 3 个乘子 + 16 个词表尺寸 + 16 个偏移）。即：上一轮的 PLE 接线新增了这条路径，
夹具没有同步。本轮只把该测试的 `output_gate_type` 断言改成 `gdn_gate`（否则连编译都不过），
没有伪造哈希控制量的内容。

## 5. 未搬移项与下一轮顺序

### 5.0 仍然存在的三条跨族包含边（调用点，不是分支）
`qwen3_5` 侧还包含三个族头文件，每一处都是"调用族模块"而不是"写族逻辑"，但仍是耦合：

| 边 | 原因 | 收口方式 |
|---|---|---|
| `qwen3_5/load.cpp` → `qwen4_exp/load.h` | `LoadPlan` 持 `shared_ptr<const PleTableSource>`，`ple_table()/ple_eos()` 被 `src/runtime/engine/model_instance.cpp` 读取 | 把 PLE 表移到 `LoadPlan` 的类型擦除外壳（`PleLoadHandle`），或让 runtime 从族侧取；两者都要动 `src/runtime`（不在授权面） |
| `qwen3_5/program/planning/startup.cpp` → `qwen4_exp/execution/qsa.h` | 规划器为 indexed 层向族询价（几何、索引面、尾部状态、两个 Ops 的 scratch） | 与 5.2 的族层钩子一起收口：规划走 `FamilyPlanRules` |
| `qwen3_5/{execution/text.cpp,execution/parameters.cpp,program/program_impl.cpp,program/storage/ple_state.cpp}` → `qwen4_exp/execution/ple_layer.h` | 共享执行/程序持有族的 `PleTextStage` 并驱动它的 window | 见 5.5 |

### 5.1 `src/models/qwen3_5` 仍是"共享文本引擎 + 3.5 族"
`frontend`、`program`、`state`、`execution` 的共享部分与 3.5 族住在同一棵树里。真正的分层应是
`models/text`（共享）+ `models/qwen3_5`（仅身份/绑定）+ `models/qwen4_exp`。本轮不做，因为
`Runtime`（不在授权面）直接调用 `models::qwen3_5::plan_load/materialize_model`，改树名会连带
改 `src/runtime/engine/model_instance.cpp`。

### 5.2 `TextContext` 的族成员声明
HC/QSA/PLE 的实现已搬走，但声明（`hc_stem`/`hc_mix_into`/`hc_combine`/`final_hidden`/
`qsa_mix`/`qsa_tail_view`/`hc_`/`ple_`/`active_qsa_*`）仍在 `qwen3_5/execution/text.h`。
它们是 `run_layers` 与各 pass 入口的私有成员；迁出需要先把 `run_layers` 拆成
"共享块循环 + 族层钩子"，而这会动到 workspace 的预留顺序（`startup.cpp` 按同一配方同序预留），
属于下一轮要单独验证的改动，本轮不做。

### 5.3 `execution::Parameters` 的族结构（`parameters.cpp/h`，11 + 9 处）
`HcParameters`/`QsaParameters`/`PleParameters`/`MtpParameters::final_hc` 作为
`BlockParameters` 的可选成员被共享执行路径持有；装填逻辑（banked experts 的三账本、
索引器四矩阵、PLE 行块、四流草稿头）仍在 `parameters.cpp`。它已由 `Identity` 的几个布尔量
（`banked_experts`/`aliased_activations`/`four_stream_residual`）驱动，没有 `Architecture::` 分支；
下一轮应把"族参数装填"做成同 `TextBindingRules` 的 `ParametersRules`。

### 5.4 MoE 行窗口（`ffn.cpp` 2 处、`expert_stream.h/cpp` 3 处）
"每个专家三账本、按行窗口取块"是共享 MoE 执行对 **banked 工件**的寻址方式，由
`Identity::banked_experts` 打开，不是族名分支。要把它变成族专属，需要把 `expert_stream`
的池/表/逐账本步长拆成族实现，代价远大于本轮，且会动放置与驱逐的数值路径（本轮禁改数值）。

### 5.5 PLE 程序状态（`program/storage/ple_state.*`、`program_impl.*`、`transactions/binding.cpp`）
`detail::PleProgramState` 由共享 `Program` 持有（九列历史、哈希尾、window 提交）。
它是族专属的*值*，但归属是共享 Program 的生命周期；迁出需要 Program 接受族状态接口
（`PleState` 抽象），属下一轮。

### 5.6 `state/decoder_state.*` 的 QSA 平面
`qsa_index_`/`qsa_tail_layer`/`has_qsa` 是 KV 缓存的一部分（与主缓存的页空间共址），
拆分会切开 `PagedKVCache` 的绑定与 `StateImage` 的镜像；本轮保留。

### 5.7 建议顺序
1. 抽 `models/text`（共享引擎），`runtime` 侧只改一处调用；
2. `ParametersRules`（族参数装填）+ 5.3；
3. `run_layers` 的族层钩子（先钉住 workspace 预留顺序，再动）；
4. PLE 状态接口（5.5）与 QSA 平面（5.6）；
5. 补 `test_config_bindings` 的 `ple_hash_controls` 夹具（4.1）。
