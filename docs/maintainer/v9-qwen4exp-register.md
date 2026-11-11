# v9 Phase 7 qwen4-exp 模型集成登记（T191–T202）

日期：2026-11-08
上游基线：upstream/master = a754fcf3
前置对表：T013 qwen4_exp-model（.temp\upstream-scan\design-parity\T013-qwen4exp-model.md）——qwen4-exp 真正新增 HC/QSA/PLE 三样、转换规格已完整转录（总容量 100.666 GiB，固定 GPU 10.257616 GiB）、4090 三拦路虎（无 NVFP4 硬件、63.28 GiB experts 需分层、PLE 27 GiB 随机行读）；T073 已登记本地无 Qwen4Exp 转换路径。

## 前置事实核查（2026-11-08）

| 项 | 结果 | 证据 |
|---|---|---|
| qwen4-exp 源权重工件 | **不存在**（T191 立项基线快照；2026-11-08 起模型统一位于 `R:\model\`，qwen4exp 目录已有 MTP 2.66 GiB 与 UD-Q4_K_XL 第 1 分片） |
| 转换器 | **不存在** | `tools\convert\` 只有 qwen3_5.py 等，无 qwen4_exp.py（T073 已登记） |
| 可行性 | **未证** | T013：100.666 GiB 权重、4090 无 NVFP4 张量核（T114 已判死 NVFP4 MoE）、PLE 27 GiB 随机行读 |

## T191 qwen4exp 权重转换＋paged NF4 —— 待工件（阻塞）

- 上游 d8c0fa6f：qwen4exp 权重转换器 + paged NF4 布局。
- 本地：无源权重、无转换器 → **无法执行**。登记待工件前置链：
  1. 取得 qwen4-exp 源权重（HF 格式）与 T013 转录的转换规格（几何表/18 行 parent 行序/recipe/bank 字节/PLE 分页 90B 行/280B controls/hash）；
  2. 移植上游 `tools/convert/qwen4_exp.py`（参考 L05 NVFP4 判 dtype、L08 量化器/校准、L12 工具链健壮性五连）；
  3. 先跑**容量可行性账**（100.666 GiB → 4090 24 GiB：experts 分层比例、PLE 随机读带宽、KV 262144 ctx），不可行则登记降级路线（如 INT4/INT8 替换 NVFP4）；
  4. 工件校验（sha256/结构/量化元数据）。
- 判定：阻塞，待工件。

## T192–T198 —— 挂起（依赖 T191 工件）

| 条 | 上游 | 挂起理由 |
|---|---|---|
| T192 | f2620157 GDN shapes＋GDN 重组 | 依赖 qwen4exp 工件与 T104–T108 的 GDN 算子（本地已有 96x2560 control，但 qwen4exp shapes 需工件校准） |
| T193 | 088e8da4＋4f31ad4a 加载/资源＋执行成本 trace | 依赖 T192 |
| T194 | 985121d4＋afe43480＋439e1075 eager text/vision/mtp | 依赖 T193；vision 按 4090 显存核对（T013 已警示） |
| T195 | 3444964f＋1408277a CUDA graph＋resident main/draft | 依赖 T194；本地图经验（T163）可参考 |
| T196 | 9cdbf221＋5e581e92 a8 MTP experts 转换修复＋frontier bound | 依赖 T195 |
| T197 | 2507d770＋e07307e0＋7131c0b9 MTP 三连 | 依赖 T196 与 T010 结论；本地 MTP 从未在 4090 线跑过（T010） |
| T198 | f8a774f7 应用标定 priorities | 依赖 T197 与 T161；T161 已判定"本轮不并入"，且需与 qwen4exp f8a774f7 同源表共用一次数据整理 |

结论：以上 7 条全部挂起，解锁条件统一为「qwen4exp 工件就绪 + T013 可行性账通过」。

## T199 两 rank 三条 —— 判死（单卡无场景）

- 上游 56a8cf90＋dcc576ba＋27df5a34：两 rank 相关（与 T001 判死池合并，dcc576ba/27df5a34 已在 T001 双计）。
- 本地：单卡 RTX 4090，无 rank 拓扑 → **判死，与 T001 池合并**。
- 判定：判死登记。

## T200 docs 四条 —— 随组登记（T013 已对表）

- 64d10fbf＋30e3a92f＋dc8b58fb＋84d1cf02 的文档结论已在 T013 对表（qwen4_exp-model.md 268 行主取证、转换规格完整转录、HC/QSA/PLE 三新增）。
- 判定：登记完成（T013），不重复搬运。

## T201 组门禁

- **35B-A3B 回归 PPL 逐位**：default 4801 / 2.051275 / 7.777808；冻结 12 GiB 4801 / 2.049253 / 7.762104（最近实测留档，Phase 6 门禁未移动；本批无代码改动）。
- **qwen4exp 端到端**：无工件 → 记录「待工件」，不伪造数据。
- 判定：回归门禁 PASS；qwen4exp 端到端待工件。

## T202 absorb/m5 并主线＋推送

按用户纪律：不建分支、不推送。Phase 7 无代码提交（全部登记/挂起/判死），提交仅本 register 一条；master 线性。

## 结论汇总

| 条 | 判定 | 动作 |
|---|---|---|
| T191 | 待工件（阻塞） | 登记前置链（权重/转换器/可行性账/校验） |
| T192–T198 | 挂起 | 解锁条件=工件就绪+可行性通过 |
| T199 | 判死 | 与 T001 池合并 |
| T200 | 完成 | T013 已对表 |
| T201 | 回归 PASS / qwen4exp 待工件 | 记录 |
| T202 | 不推送 | 登记 |

## 补充核查与新增功能（2026-11-08 18:50）

### 转换器已就位（本地无功能→已增加）
- `tools/convert/qwen4_exp.py`（626 行，upstream d8c0fa6f 的文件）+ `tools/convert/qwen_resources.py`（30 行，其依赖）已移植，`py_compile` 通过，提交 `f377aa93`。
- 运行前置（非新增要求）：依赖 `torch`（与本地既有 qwen3_5.py/methods.py 同一前置，本机 Python 环境未装 torch——转换在任何环境都需此依赖，不是 qwen4 特有）。
- 覆盖：Qwen4ExpForConditionalGeneration / Qwen4ExpForCausalLM 两种架构，含 checkpoint 轴与逻辑参数映射。

### qwen4-27b：与 qwen4-exp 共用转换器（已确认）
- `upstream/feat/qwen4-27b-ops` 分支的 `tools/convert/` 与 `upstream/master` **完全一致**（目录列表相同、qwen4_exp.py blob 相同），即 qwen4-27b 不是独立模型格式，而是 Qwen4Exp 架构的 27B 配置；本地转换器已同时覆盖。
- 该分支 787 文件 / +33445 的差异**全部是 QSA 算子优化**（四头 k8v4/nvfp4/int8/bf16 调优、sparse attention 跨 KV 格式调优、20480 整向量 rmsnorm、kpool fixture），不是模型集成功能。
- 评估结论：qwen4-27b 的"模型转换/注册"功能本地已具备（转换器+QSA 基础算子+GDN+PLE 层）；剩余为**性能优化批次**（可选，不阻塞功能），建议在 qwen4-27b 工件出现后按 T028 纪律分批吸收（优先 20480 rmsnorm 整向量、QSA 四头 bf16/int8 调优）。

### 仍缺的本地功能（需工件）
- 模型主体：text/vision/mtp 执行、程序、参数映射、HF→ninfer 工件注册（T192–T198）——依赖 qwen4exp 工件；PLE 磁盘层（src/models/qwen4_exp/ple/）与 QSA/GDN 算子已就位，工件到位即可接。

## 新增：GGUF 直转（2026-11-08，只发布 GGUF 时的转换能力）

用户要求：qwen4exp 若只有 GGUF，也要能转成 `.ninfer`。已落地一条可运行的转换链路，而不是继续登记“待工件”。

### 实现
- `tools/convert/sources/gguf.py`（新，numpy-only）：GGUF v2/v3 容器 + 元数据（全部值类型）+ 有界分块读取；反量化覆盖 F32/F16/BF16/F64、I8/I16/I32/I64、Q4_0/Q4_1/Q5_0/Q5_1/Q8_0 与 K 系 Q2_K/Q3_K/Q4_K/Q5_K/Q6_K（llama.cpp gguf-py 反量化的移植，MIT）。未支持类型按名拒绝，不猜。
- `tools/convert/gguf_qwen4exp.py`（新）：把 GGUF 以转换器已有的 duck-typed source 形式暴露（config/root/weight_map/describe/read_flat），因此 `tools/convert/qwen4_exp.py` 的轴映射与量化路径原样复用。三件必须做对的事：
  1. 命名：`blk.N.*` → HF 逻辑名，取自 llama.cpp QWEN4EXP 张量表（conversion/qwen4exp.py + gguf-py TensorNameMap）的逆映射；
  2. 合并张量拆回：expert `ffn_gate_up_exps`（或 gate/up 二分）、indexer `q_proj|k_proj` → `index_qk_proj`、linear-attn `attn_qkv_gate` → `in_proj_qkv|in_proj_z`、`ssm_beta/alpha`、`hc_*_down_inject` → down|inject；
  3. V 头顺序：llama.cpp 对线性注意力按 tiled 存 V 头（`_LinearAttentionVReorderBase`），转换时做逆置换回 grouped；`nk == nv` 时恒等。
- 配置重建：GGUF 元数据（`qwen4exp.block_count/embedding_length/attention.*/expert_*/ssm.*/hyper_connection.*/attention.indexer.*/ple.*/attention.compress_ratios/rope.freq_base`）合成 HF `config.json` 形状；GGUF 不携带的字段（`rope_parameters` 细节、`indexer_compress_ratio`、vision/mtp 子树）由 `--gguf-config` 侧车 config.json 覆盖。PLE 哈希常量是 GGUF **元数据** u64 数组（不是张量），按 int64 精确写入工件。
- CLI：`--model X.gguf`、`--gguf-config`、`--gguf-root`；按 `architectures` 分派 qwen3_5 / qwen4_exp 构建器；`.gguf` 只允许 `--components text`（vision 在 mmproj、MTP 有独立转换，明确拒绝而非产出残缺工件）。
- 新增 recipe `qwen4_exp`（Q4 路由专家/注意力 V 头投影、Q5 投影、Q8 词表）。
- 顺带修复（Windows 可用性，均为真实缺陷）：
  - `tools/artifact/win_compat.py` 的 dup 句柄在 `os.close` 后不释放 → 工件发布时硬链接/删除报 WinError 32（转换在最后一步必失败）；现随 fd 关闭一并关闭；新增 `file_io.pread` 可移植实现，`safetensors.py` 不再直接调 POSIX-only `os.pread`。
  - `tools/convert/model.py` 补 `proposal_domain`、`proposal.py` 按其分派（上游 qwen4exp 构建器需要）。
  - `qwen_resources.load_qwen_resources` 调用参数与本地 `load_resources` 不匹配（移植后从未跑过）→ 已修。
  - `tools/chat_templates/qwen3_8_flash_next.jinja` 缺失（qwen4_exp 构建器引用）→ 从上游取回。
  - `tools/convert/__main__.py` 的 `file.py:func` 解析会把 Windows 盘符当函数名 → 仅在后缀为标识符时切分。

### 验证
- `tests/convert/test_gguf_qwen4exp.py`（新，4 项全过）：容器/反量化对拍 gguf-py 自身（随机块覆盖 Q2_K/Q4_K/Q5_K/Q6_K/Q8_0/Q4_0/F16，含有界读）；名称/形状映射断言（conv 的 (C,1,taps)、专家 bank (E,2I,H)、indexer 融合、I64 控制量）；合并张量拆回与 V 头逆置换逐元素断言；真实转换（合成 Qwen4Exp GGUF → `--recipe qwen4_exp --device cpu` 产出 v3 工件 64 objects/59 tensors，PLE 控制量逐字节、bf16 张量逐字相等）。
- `python -m pytest tests/convert -q`：28 passed / 1 skipped（含既有 27 项）。
- 未验证部分（如实登记）：真实 100 GiB 级 qwen4exp 权重不存在，容量/带宽实账仍待工件；运行时装载属 T192–T198。GGUF 已量化源再转只保留该文件的近似精度（文档已写明）。

## 真实工件转换完成（2026-11-09）

源：`R:\model\qwen4exp\UD-Q4_K_XL\Qwen3.8-Flash-Next-UD-Q4_K_XL-0000{1..4}-of-00004.gguf`（4 片 103.7 GiB；第 1 片为**元数据片**：67 个 KV、0 张量；第 4 片是单张 27 GiB 的 `per_layer_token_embd`，IQ4_NL）。
侧车：`R:\model\qwen4exp\hf\{config.json,tokenizer.json,tokenizer_config.json,generation_config.json,chat_template.jinja}`（取自 Qwen/Qwen3.8-Flash-Next；tokenizer.json 完整 12.8 MB，首次下载截断后断点续传补齐）。

命令：`python -m tools.convert --model <首片> --recipe qwen4_exp --components text --gguf-config ...\hf\config.json --gguf-root ...\hf --device cpu --name qwen3.8-flash-next-ud-q4k-xl --out R:\model\qwen3_8_flash_next\qwen3_8_flash_next.ninfer`

结果（conversion.json）：**1163 张量 / 1168 对象 / 4 个文件 / 121.25 GB**，用时 **2890.8 s**；格式分布 q5 436、bf16 353、q4 253、q6 48、q8 1、fp32 72；device=cpu（本机 torch 2.15.0+cpu）。工件 inspect：容器 **version 3**，config 与官方一致（mrope_section [11,11,10]、partial_rotary_factor 0.25、ple_shard_shape [2500012,160]（128 分片）、image/video_token_id），bindings 1308。

**装载验证（如实登记失败点）**：ninfer.exe 打开该工件在 planning 108 us 后失败——`error: Qwen3.5 config: text config: unknown member attention_bias`。引擎目前只有 Qwen3.5 的配置/程序路径，**没有 Qwen4Exp 架构入口（T192–T198 未做）**，因此本工件**无法端到端推理，无 tok/s 可测**。已实测的只有组件级：ninfer_ple_lookup/conv/gate/state_extract/disk 五项全部通过（8.8/14.1/5.9/3.8/0.24 s）。

容量账（按工件配置计算，非实测 tps）：满注意力层 12 层，KV = 2 x 2 kv_head x 256 x 2 B x 12 = **24 KiB/token**（2K=48 MiB、8K=192 MiB、32K=768 MiB、128K=3.0 GiB、256K=6.0 GiB）；PLE 表 q4 约 **25.3 GiB**；路由专家 120.8 G 元素约 **69 GiB** -> 24 GB 卡必须走专家流式 + PLE 页式。

## M1/M2 已落地（2026-11-09，提交 f6f8840a）

- M1：Qwen4Exp 配置键全部放行、架构注册（复用 MoE plan）、绑定别名（final_hc / attention_hc / moe_hc / output_gate / indexer 四项 / PLE 七项 / 三专家 bank 展开为每专家行块），未知或歧义按逻辑参数名报错。
- M2：Use 别名（attention_input / moe_input / experts/product）、参数模型支持「融合 bank」与「三独立 bank」两种描述（读错行即报错，旧的 2*I 行假设已删）、mixer 判据改 `attention_mixer()`（indexed 层不再落进 GDN 分支，暂走 full attention 并打印 QSA 未接线）。
- 真实工件：48 层 / 1308 绑定 / 74,892 逻辑参数全部解析，卡在 `text/layers/0: input projection: unsupported logical projection geometry`（ops 的融合投影白名单）。测试 `ctest -R qwen4_exp` 2/2 通过。

## 路线修正：MoE 并非"无可用核"

前一条"MoE 只注册 NVFP4/FP8、本工件 q4/q6 被拒"需要精确化：`ops/moe/operands.cpp:31-45` 的 `validate_route_profile` **本身就接受 hidden 2560 / E 512 / top-10**，`validate_expert_profile` 对 hidden 2560 接受 **FP8 block128x128（AllowA8）** 或 NVFP4。也就是说：
- 4090 上没有 NVFP4，但 **FP8 块缩放路线是有核的**；
- 转换器已支持 `fp8_e4m3fn_block128x128_bf16`（`tools/artifact/formats.py:69`）；
- 因此**不需要写新 MoE 核**，最短路径是「专家三 bank 用 FP8 块缩放重转 + 投影通用化（M3）+ HC/PLE/QSA 接线（M4）」。
剩余硬点：PLE 表行格式（工件 q4 136 B/行，`PleTableSource` 只吃 NF4 90 B/行，需转码或扩读取器）与 24 GB 卡上的专家流式（69 GiB）+ 表分页（40.5 GiB）。

## T203 QSA 接线：S2/S3 门按算子契约重述（2026-11-10）

`qsa_prepare` 的数值契约是「归一化与旋转之间没有可表示的 BF16 中间量」（`include/ninfer/ops/qsa.h`），
而既有 dense 路径是 `rmsnorm → BF16 → rope → BF16`。两者因此**不可能逐字节相等**，且这是契约差而非接线错：
已定位到唯一差异源，并把门重述为可判定、可重复的形式。

差异定位（位置 1..50，两个 KV 头，每对 (i, i+32)）：
- 相位差 |Δφ| 均值 4e-4…9e-4 rad，**不随位置线性增长**（token 1 为 3.95e-4、token 50 为 9.15e-4，2.3×）——
  频率/theta 错会随位置线性放大，实测没有；
- 角抖动 × 对幅值（扰动向量长度）最大 **1.41 ulp**（BF16 相对 ulp 2^-7），与「输入多/少一次舍入」相符；
- 归一化输出（`dim≥64`，不旋转）**0 差异**，V 平面 **0 差异**，位置 0（旋转为恒等）**逐字节相同**。

| 门 | 判据（重述后） | 实测（layer 3 = 首个 indexed 层，51 列前缀） | 结论 |
|---|---|---|---|
| S2-A | `dim≥64` 逐元素相同 | 0 / 786,432 差异 | 过 |
| S2-B | 差异只在旋转维 `dim<64` | 1,325 差异全部在 `dim<64`，最大 \|Δu16\| 239（近抵消对） | 过 |
| S2-C | 位置 0 逐字节相同 | 两个 KV 头的第 0 行完全相同 | 过 |
| S2-D1 | 相位差不随位置增长 | 3.95e-4 → 9.15e-4 rad（2.3×，非 50×） | 过 |
| S2-D2 | 角抖动 × 幅值 ≤ 2 BF16 ulp | 1.41 ulp | 过 |
| S2-V | V 平面 0 差异 | 0 / 26,112 | 过 |
| S3 | 同一 prepare 缓存上 sparse 与只读 dense ≤2 BF16 ulp，且单可见行 token 相同 | p99 2.87e-3、峰 7.81e-3 × 行幅值（预算 1.56e-2）；token 0 相同；247,079/313,344 逐字节相同；0 NaN | 过 |

复现方式（不依赖人工结论）：`NINFER_QSA_KV_DUMP=<dir> NINFER_QSA_LAYER_DUMP=<dir>
NINFER_QSA_DUMP_LAYER=0` 跑两次——`NINFER_QSA_FORCE_FULL=1`（dense 参考臂）与
`NINFER_QSA_DENSE_READ=1`（prepare 写 + 只读 dense 读）——再以 `.temp\qsa_gates.py s2 <dense> <prepare>`
判 S2；S3 用 `NINFER_QSA_DENSE_READ` 与稀疏臂两组 `attention_layer*_T<列>.bin` 交 `qsa_gates.py s3`。
脚本逐条打印上表判据与数值，任一不满足即非零退出。

未因此放宽的部分：S2 的「主 KV 逐字节相等」与 S3 的「层输出与 dense 位级相等」两条**按原措辞不成立**，
原因就是上表的契约差（另加 sparse 与 dense 两个软最大核的归约顺序差，≤2 ulp）。若要恢复位级相等，
唯一办法是在 `qsa_prepare` 的 norm 与旋转之间插入一次 BF16 舍入——2026-11-10 决定**不做**，保持算子契约。

## T204 dense 侧 24Q/2KV/D256 的量化 KV 路由（独立工作项，未开工）

状态：**只立方案**。QSA 的 rk 端到端门被这一项阻塞：`--kv-dtype rk*` 在第一个 attention 层被拒，
因此「prepare 写 vs dense append 写」的同层主 KV 双转储比对无法在本机执行。

### 现状证据（2026-11-10 实测）

| 项 | 事实 | 位置 |
|---|---|---|
| 拒绝点 | `24q/2kv + 非 BF16` 直接抛 `the 24q/2kv D256 head geometry is served by the BF16 KV route only` | `src/ops/softmax_attention/dense/causal_cache/causal_softmax_attention.cpp:48`（`require_storage_supports_geometry`） |
| 几何类型 | `CausalGeometry<256,24,2>` **已存在**（注释即写明是 Qwen4Exp indexed 层的 GQA-12 配对） | `src/ops/softmax_attention/common/causal_geometry.h:17` |
| 实例覆盖 | 只有 **BF16 族**为 `CausalD256H24Kv2` 实例化 | `bf16/launch.cu:80`、`bf16/batch_wide_launch.cu:48` |
| 分派形状 | 五个量化族**只按 `query_heads` 分支**，24 头一律落 `CausalD256H24Kv4`（KVHeads=4），故操作数不匹配 | 例：`int8/batch_wide_launch.cu:95-99`；同型见 `int8/launch.cu:335/371/433/560`、`fp8/launch.cu:66/101`、`fp8/tiled_launch.cu:22`、`nvfp4/launch.cu:71/120`、`nvfp4/tiled_launch.cu:14`、`k8v4/launch.cu:71/120`、`k8v4/tiled_launch.cu:25` |
| rk 解码算术 | **已具备**：int8 目录同时承载 int8 与 rk8v4/rk4v4/rk4v4-e8/rk2v4-e8/rk4v2-e8（`PackedK`/`PackedV`/`E8Lattice`/`E8Root`/`V8Root` 五个模板开关） | `int8/batch_wide_mma.cuh`、`int8/grouped_mma.cuh`、`int8/batch_wide_launch.cu:100-147` |

结论：缺的**不是解码算术**，而是 `(24,2)` 这一对在每个量化族的模板实例化与 `kv_heads` 分派。

### 改动清单

| 文件 | 改动 | 量级 |
|---|---|---|
| `dense/causal_cache/int8/batch_wide_launch.cu` | 6 个入口（int8 / rk8v4 / rk4v4 / rk4v4-e8 / rk4v2-e8 / rk2v4-e8）的分派由 `query_heads==24 ? H24Kv4 : H16Kv2` 改为 `kv_heads==2 ? H24Kv2 : (query_heads==24 ? H24Kv4 : H16Kv2)`，并为 `CausalD256H24Kv2` 实例化对应模板 | 6 个实例 + 1 处分派 |
| `dense/causal_cache/int8/launch.cu`（prefill：grouped / parallel_grouped / tiled） | 同上 3 处分派 + 实例 | 3 处 |
| `dense/causal_cache/fp8/launch.cu`、`fp8/tiled_launch.cu` | 同上 2 处 | 2 处 |
| `dense/causal_cache/nvfp4/launch.cu`、`nvfp4/tiled_launch.cu` | 同上 2 处 | 2 处 |
| `dense/causal_cache/k8v4/launch.cu`、`k8v4/tiled_launch.cu` | 同上 2 处 | 2 处 |
| `dense/causal_cache/causal_softmax_attention.cpp:48` | 把「24/2 只服务 BF16」的整段拒绝改为按族逐项放开（保留未实例化组合的按名拒绝） | 1 处表 |
| `cpu/partial.h:87`（`rk_partial<24,4>`/`<16,2>` 注释处） | 确认 CPU 侧部分和 oracle 是否需要 `<24,2>` | 1 处 |

### 风险

- **tile 假设**：量化核按 head 映射分块（tile 16、`rpk ∈ {6,8}`）。GQA-12 不整除 16，很可能触发
  `static_assert` 或需要改 `grouped_mma.cuh` / `batch_wide_mma.cuh` 的 tile 选择；
  **一旦改动落入这两个文件，就是 prefill + decode 两套新核**，工作量与验证面都上一个量级。
- **基线移动**：任何 tile/归约顺序改动都会移动 BF16 的位级 golden（PPL 7.441868、dump、regress），需重录并复跑门禁。
- rk 变体在 `PackedV` 下的 zero-fill 宽度（`rk4_zero_packed_v`）与 masked 列处理已有前车之鉴（宽度 >256 时 16B 存会 `MisalignedAddress`），新实例要覆盖到。

### 分阶段与门（每阶段独立可回滚）

| 阶段 | 内容 | 门 |
|---|---|---|
| S1 | 只加实例与 `kv_heads` 分派，**不**放开拒绝 | 现有 BF16 golden 逐位复跑（default PPL 7.441868、`dump` e7ceb89d…、`regress` 0）；`ctest -R "qsa|linear_"` 全绿 |
| S2 | 放开 24/2 的量化存储（先 int8 与 rk8v4/rk4v4） | `--kv-dtype rk8v4 --prompt Hello --max-new 8` 两次 token 逐位一致；再跑 `rk4v4`（`NINFER_EXPERT_CACHE_MB=8192`） |
| S3 | prepare 写 vs dense append 写同层主 KV 双转储 | `NINFER_QSA_FORCE_FULL=1` 与默认臂各转储一次，`main_key_layer0.bin`/`main_value_layer0.bin`/`index_key_layer0.bin` 哈希比对（rk 应与 append 逐字节相等，故差异元素数应为 0） |
| S4 | 长前缀与选择性 A/B | 2200/4152 前缀无非法访存、无新增 NaN 探针命中；4096 前缀 `NINFER_QSA_FORCE_FULL` A/B 记录 top-1 与速度 |

### S2 实测（2026-11-11，`NINFER_QSA_SPARSE_GQA_QUANT_KV=1`、`NINFER_EXPERT_CACHE_MB=8192`）

ids 门（`--prompt Hello --max-new 8 --greedy --print-token-ids`，各跑两次）：

| 臂 | run1 | run2 |
|---|---|---|
| `--kv-dtype rk8v4` | `8239 2780 39435 7520 55076 42228 37554 9825` | 同 run1（逐位一致） |
| `--kv-dtype rk4v4` | `97110 194972 4487 5936 33930 76336 1775 46849` | 同 run1（逐位一致） |
| BF16 基线 | `160864 206077 99011 1525 90856 128702 186459 3794` | 不变 |

主 K 平面（layer 0，61 列 × 2 KV head = 122 行 / 31232 元素，页面主序 `(page, token, head)`）：

| 平面 | 字节 | sha256 |
|---|---|---|
| BF16 参考 | 2097152 | `144f830ab85afad1a1f1746b611b496ad6780e1661ee6fa1e9778b486063317d` |
| rk8v4 K | 1048576 | `3879a11e5401123583631c986dda936f8555f8f7cd4524f5bd335f153c5f6b2b` |
| rk8v4 K scale | 32768 | `7216fbc74ceb061c1ccae191a6a7e3d7c4c051359288bd5e6ab6bd4b5cf1b3fe` |
| rk4v4 K | 524288 | `6eae562f30cd050a4c42fc1454f7357bf6f810f530ea0e6c05e9ad88bb5001c9` |
| rk4v4 K scale | 32768 | `9f031cc8b5e4dda2672a1911ec8a10731ee56006d944f3d3d9c0a20ad61f52c3` |

逆 Hadamard（/16）后逐元素比：

| 臂 | 差异元素 | max \|d\| | p99 | 单码步内的行 |
|---|---|---|---|---|
| rk8v4（码步 = absmax/127） | 30208 / 31232 | 11.0598（112.0% 平面 absmax 9.875） | 4.0991（41.5%） | 105 / 122：max 0.0545（0.55%）、p99 0.0357（0.36%） |
| rk4v4（码步 = absmax/7） | 30206 / 31232 | 9.2570（93.7%） | 3.8601（39.1%） | 105 / 122：max 0.9923（10.0%）、p99 0.6252（6.3%） |

**未解释项（不是量化误差）**：17 行内容与该列的 BF16 键完全无关——rk8v4 为行 `18,22,53–59,69,96,103,117–121`，rk4v4 为 `14,18,50,53–59,79,103,117–121`（两臂共有的坏行：`18,53–59,103,117–121`，即 token 26–29 与 58–60）。
判定依据：① 把这些行的码按 `H256/16`、无旋转、`H64/8`、`H16/4`、`H4/2` 五种约定去拟合 **全部 122 行** BF16 键，最高命中仅 2%（偶然水平）；② 两臂彼此也不一致（同一行 int8 与 int4 解码 max 差 6.7，而邻行只有 0.67）。量化器本身解释了另外 105 行（int8 单码步内 0.55% absmax），ids 门也不受影响（两次逐位一致），但 S3 的「prepare 写 vs dense append 写逐字节」比对必须先落在这 17 行上定位。

### 预估与前置

- 若 S1 的模板实例化不触发 tile 断言：约 0.5–1 天（机械实例 + 分派 + 复跑 golden）。
- 若触发 tile 断言（GQA-12 vs tile 16）：另加 1–2 天用于 `grouped_mma.cuh`/`batch_wide_mma.cuh` 的分块改造与两套路径复验。
- 前置：本项是 **rk 端到端门** 的唯一障碍（QSA 侧读侧 5/5 oracle 已过、写侧 2/5 逐字节已过、E8 写侧 K 平面 1024/1024 子空间码已对齐）；不解决它就无法在真实模型上验证 rk KV 的端到端正确性。
