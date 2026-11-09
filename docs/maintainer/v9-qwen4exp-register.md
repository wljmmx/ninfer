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
