# v9 Phase 6 · 架构批（T145–T152）对表登记

- 范围：v9 §8.1–§8.2（ops 重组路径跟随 + 核心架构对表）
- 仓库：`H:\TeleAgentWorks\ninfer-4090-port\ninfer`（master，本组起点 `5ca26483`，工作树仅未跟踪 `.codegraph/`）
- 上游：`upstream/master` = `a754fcf3`；merge-base `81c8ce09`
- 底稿：`.temp/upstream-scan/design-parity/DESIGN-PARITY-CHECKLIST.md`（T020），配套 `conflict-grades.txt`（T003）、`path-map.txt`（T005）、`docs/maintainer/sm89-fallback-table.md`（T016）
- 纪律：**本批不做代码搬运**。逐能力判定为「本地已有等价（登记差异）／本地缺且需要（另立重实现）／不需要（留痕）」；不建分支、不推送
- 复核：首版登记 `74d6d1d0`；本版逐条复算证据（14 条路径计数、include 计数、行号抽查），修正 3 处统计口径（T005 命中率、族计数、`ae54c1a9` 残留）与 6 处行号引用

## 0 结论总表

| 编号 | 上游 | 判定 | 建议 | 一句话结论 |
|---|---|---|---|---|
| T145 | ops 重组 14 连 | 口径不同（本地迁移前形态 + 已局部跟随） | 只建映射，重组另立项 | 14 条的家族归属与 T005 一致（修 1 处误归）；本地 11 族同名、16 族本地专有、7 族上游专有；清理提交 `ae54c1a9` 未跟随（7 个扁平头仍在）；35B-A3B 工件路径 `moe/experts/<e>/{gate,up,down}` 与 14 条无关 |
| T146 | `a0d27b22` | 条件缺失（表达方式不同） | 登记，随 QSA 接入 host 时吸收 | 本地已用「每页 16 行硬编码」表达 4 token/行，但几何层无 `tokens_per_row`，Device/Host 两条布局仍按 64 行 |
| T147 | `6a42de35` | 部分等价（底座同源、传输层未收口） | 登记，重实现另立项 | `memory.h` 已与上游同源，`CudaExecutionView` 已本地化；缺 `HostBuffer/PinnedHostBuffer` 归属与 `copy_bytes*` 统一传输层，119 处 `cudaMemcpyAsync` 散落 |
| T148 | `74671a4e`＋`8e3dff94` | 等价 + 本地领先 | 记录（不重实现） | `src/core/cpu/{system,executor,task_pool}` 与 `moe/routed_swiglu/cpu` 已按本地平台层落地（Windows 拓扑自研、int16 默认、多 nvfp4 形状） |
| T149 | `8cf26bb6`＋`9cfd5646` | 功能等价、结构分叉 | 不跟随文件拆分／公共前端抽取 | 本地 `ffn.cpp` 合体 + 参数层 `std::variant` 已分离 dense/MoE；无 `models/common`（单一 Qwen 消费者，抽出去没有第二个模型受益） |
| T150 | `066714cc` | 本地缺失统一层（能力/roofline） | 登记，另立 Phase 5 条目 | 上游有 `platform.h.in` 能力宏 + `bench/common/roofline.h`；本地 11 处硬编码 5090 的 1792 GB/s，4090 的 DRAM_% 口径偏低 |
| T151 | `f30e661f`＋`91972187` | 结构分叉 | 不跟随，映射登记 | 本地 46 个 `test_*.cpp`（一 op 一可执行）vs 上游家族 suite；测试目标保持不破 |
| T152 | 组门禁 | 纯对表无 diff | 记录 | 本组仅新增/修改文档，`ninja -n` 无工作，PPL 与 dump 基线不变 |

## T145 ops 重组 14 连 —— 只建立映射，不重组

### 14 条的家族归属（按提交改动文件统计，非 T005 路径条目）

口径：每行三个可复现数字——**后置路径**＝该提交 `--name-status` 的去重后置路径数；**本地同名**＝这些路径在本树 `git ls-files` 中存在者；**改名对（命中）**＝`R` 条目数与其 old→new 对在 T005 `path-map.txt` 中逐对命中者。计数脚本见 `.temp/t145_local_exist.py`、`.temp/t145_family_hits.py`。

| # | 提交 | 后置路径 / 本地同名 | 改名对（命中） | 上游落点族（Top） | 本地对应与判定 |
|---|---|---|---|---|---|
| 1 | `90d25edd` | 417 / 145 | 237（195） | `linear`235、`elementwise`1 | `linear` 在，但**无 `kernels/`、`projection/`、`split/` 三层**，也无 `linear_split.h`；分段投影在 `linear`+`linear_add/pair/swiglu/topk`。口径不同 |
| 2 | `7ee63999` | 148 / 32 | 0（—） | `linear`115、`common`15（无改名，纯合并） | `src/ops/common/{cuda,host,numeric,quant}` 已按同布局存在；旧扁平头未清（见下）。等价 |
| 3 | `20d3a13f` | 101 / 72 | 18（4） | `normalization`13、`elementwise`2、`embedding`1 | `normalization/rmsnorm` 在；`layer_norm`/`l2norm`/`embedding` 仍走 `launcher/*`+`kernel/*`。口径不同 |
| 4 | `1844d861` | 55 / 40 | 6（1） | `ple`3、`gated_residual`2、`dynamic_grouped_conv`1 | 三族同名且内部布局一致。等价 |
| 5 | `4eb2c042` | 118 / 56 | 9（6） | `linear`6、tests 2 | 分段投影入口各自独立（`linear_add/pair/topk`），不并入 `linear/split`。口径不同 |
| 6 | `d53eb7ae` | 157 / 72 | 28（28） | `linear`28（add/swiglu/topk/pair 子目录） | `linear_add`、`linear_swiglu`、`linear_topk`、`linear_pair` 四族并列。等价（未合并） |
| 7 | `6733be72` | 95 / 66 | 8（8） | `rope`5、`kv_cache`3 | `kv_cache/append` 在；`rope` 无家族（`kernel/rope.cuh`+`launcher/rope.cu`+`rmsnorm_rope`）。部分等价 |
| 8 | `c012155c` | 132 / 66 | 58（36） | `softmax_attention`58 | 同名族 `dense/qsa/sliding_window/common/cpu`，无 `kernels/` 层。等价 |
| 9 | `4366c7b6` | 125 / 27 | 62（21） | `linear_attention`62 | 同名族但内部布局不同（`gated_delta_net/`、`kimi_delta_attention/` 无上游的 cases 分片）。等价（布局差异） |
| 10 | `8cc45088` | 98 / 63 | 6（6） | `sampling`5、include 1 | 无 `sampling/` 家族：`kernel/sampling*.cuh` + `launcher/sampling.cu` + `candidate_selector/`。口径不同 |
| 11 | `e6585d15` | 28 / 12 | 3（0） | `sparse_moe`3 | 同名族存在（与 `moe/` 并存，前者已是遗留）。等价 |
| 12 | `ae54c1a9` | 17 / 10 | 0（—） | 删 `common`7、`linear`5，改 `sparse_moe`2 | **未跟随**：本地仍保留该提交删除的 7 个扁平头（见下）。不适用（清理待立项） |
| 13 | `610f79e9` | 84 / 45 | 3（1） | `moe`2、bench 1 | **已跟随**：`moe/{kernels,routed_swiglu,topk_softmax_route,topk_softmax_swiglu_moe,operands.*,weights.cpp}` 同构；仅缺 `elementwise/sum_contributions`（本地 residual 在 MoE epilogue）。等价 |
| 14 | `3124dffe` | 156 / 149 | 3（1） | tests 2、eval 1 | 无 `src/ops` 改名；测试改名未跟随。不适用 |

合计：14 条去重后置路径 **1346**，本地同名 **631（47%）**；改名对 **441**，`path-map` 逐对命中 **307（70%）**——未命中的 134 条是同一提交内「先新增后改名」的多跳路径，T005 表按 tip-to-tip 生成，天然不收录中间跳。

### T005 映射表复核

- 逐对核对（`t145_family_hits.py`）：441 个改名对中 **307 个（70%）** 可在 `path-map.txt` 中按 old→new 直接命中；未命中的 134 个集中在 `c012155c`（22/58）、`4366c7b6`（41/62）、`20d3a13f`（14/18）、`90d25edd`（42/237），全部是同一提交内新增后再改名的多跳路径，或目录整搬（`linear/<dtype>`→`linear/kernels/<dtype>` 后又并层）。**0 处家族归属冲突**：按目标家族聚合，14 条没有一个目标族在表里缺席。
- **修正 1 处**（人工复核）：`include/ninfer/ops/vision_pos_embed.h` 被 T005 启发式归到 `indexing`（同提交新增 `indexing.h`），实际能力是 `embedding_interpolate_add`，应归 `embedding`。多对一启发式的误归，不影响 14 条的重组语义。
- 其余低置信条目（`dynamic_grouped_conv/{add,prepare}`、`gated_delta_net/input_conv`、`gdn_input_proj→linear_attention/.../projection_a4.cu`、`sparse_moe/sources.cmake→moe/sources.cmake`）家族均正确。

### 本地 src/ops 家族对照（`git ls-files` 口径）

- 上游 HEAD 有 **18 族**：`common`、`dynamic_grouped_conv`、`elementwise`、`embedding`、`gated_residual`、`indexing`、`kernel`、`kv_cache`、`linear`、`linear_attention`、`moe`、`normalization`、`ple`、`reduction`、`rope`、`sampling`、`sequence`、`softmax_attention`。
- **同名 11 族**：`common`、`dynamic_grouped_conv`、`gated_residual`、`kernel`、`kv_cache`、`linear`、`linear_attention`、`moe`、`normalization`、`ple`、`softmax_attention`。
- **本地专有 16 族**（迁移前切法或本地新增）：`act_quant`、`attn_input_proj`、`candidate_selector`、`context_kv_materialize`、`gdn_control`、`gdn_input_proj`、`launcher`、`linear_add`、`linear_pair`、`linear_swiglu`、`linear_topk`、`rmsnorm`、`rmsnorm_rope`、`sparse_moe`、`w4a8`、`wrapper`。
- **上游专有 7 族**：`elementwise`、`embedding`、`indexing`、`reduction`、`rope`、`sampling`、`sequence`。功能在本地存在，只是载体是 `kernel/*.cuh` + `launcher/*.cu`（单文件核 + 宿主入口）。（`collective`、`mhc` 只出现在设计文档，上游 HEAD 已无目录，不列为差异。）

### `ae54c1a9` 的清理**没有**跟随（本批最重要的实测差异）

- 上游该提交删除 `src/ops/common/{math.cuh,math.h,memory.cuh,mma.cuh,score_id_order.cuh,token_slices.h,warp.cuh}` 与 `linear/kernels/{fp8,q4,q5,q6,q8}/*_storage.cuh`。
- 本地这 7 个扁平头**全部仍在**（`src/ops/common/`），并且仍是主要 include 面：`ops/common/math.cuh` 被 60 处引用、`ops/common/memory.cuh` 被 63 处引用，而新副本 `ops/common/numeric/math.cuh` 只有 8 处。
- 结论：家族**目录**已跟随（`numeric/`、`cuda/`、`quant/`、`host/` 都在），**清理未跟随**，代价约 120 处 include 改写。这正是「只建映射不重组」的理由之一：任何清理都必须与这 120 处同批完成，否则新旧两套头继续并存。

### 35B-A3B 工件路径映射确认

- 本地转换器与上游**路径模板一致**：`<layer>/moe/experts/<e>/{gate,up,down}`（`tools/convert/qwen3_5.py:657–676`；上游 `tools/convert/qwen3_5.py:658–663` 同模板）。
- **唯一差异在 group（对象归属，不在路径）**：上游把同层全部专家 gate/up 收进一个对象 `self.group(*gate_up)`（`:680–681`），本地**每专家各自成组** `self.group(ep+"gate", ep+"up")` / `self.group(ep+"down")`（`:683–684`，注释 `:677–682` 记录三档统一的 3× 提交数收益）。
- 读取侧两种形态都支持：`src/artifact/expert_files.h:19`（per-expert）与 `:27`（fused bank），本地 35B 工件走 per-expert。
- **14 条对工件路径零影响**：13 条只改 `src/ops`、`include/ninfer/ops`、`tests/ops`；`3124dffe` 触到的两个 `tools/` 文件是删死码（`row_split.py` 的 `gather_row_planes`、`methods.py`），不涉及对象命名或 schema。

### 结论

只建立映射，不执行重组。本地 `wrapper/launcher/kernel` 三件套是迁移前的基本算子组织方式，重组会同时改动 L1 公共头与 35B 热路径的 include 面（T005 已判定「上游已整树重构，不能按 M 类融合」），收益仅是可读性；维持现状并把 11 个同名族作为后续吸收的锚点。重组提案的硬前置：① `ae54c1a9` 的 7 个扁平头清理（约 120 处 include）；② `linear` 的 dtype-first 布局迁到 `kernels/projection/split`（370 条路径，含 35B 热路径的 `linear` 对象归属）；③ 测试目标改名（T151）。

## T146 paged KV 混合行跨度（`a0d27b22`）

- 上游：新增 `KVPlaneGeometry::tokens_per_row`（只接受 1 或 4；span=4 必须 PageMajor，且 `page_tokens % span == 0`），把「一页多少行」从 `page_tokens` 中拆出来，使同一页里可以同时存在 span=1 与 span=4 的 plane（`paged_kv_cache.{h,cpp}`、`host_kv_arena.cpp:60–64`）。`kv_plane_rows_per_page()` 同时服务 Device 池规划（`paged_kv_cache.cpp:96–107`）、绑定校验（`:233`）与 Host 页布局。
- 本地几何：`KVPlaneGeometry` 无该字段（`src/core/paged_kv_cache.h:68–75`），且**显式拒绝非 64 页**（`src/core/paged_kv_cache.cpp:40`）。Device 池按 `kPagedKVPageSize` 建 4 维张量（`:97`、`:102`），绑定校验同（`:248`）；Host 页布局直接乘 `geometry.page_tokens`（`src/core/host_kv_arena.cpp:64`）。
- 本地已用**硬编码等价值**实现 QSA 索引页：`v.plane(cache.index.keys, BF16, {128, 16, 1, pages})`（`src/ops/softmax_attention/qsa/prepare.cpp:72`），16 = 64/4 即 tokens_per_row=4，只是没有升格为几何参数。
- 融合点：一处 helper + 两处布局规划（Device `plan_device_kv_page_pool`、Host `plan_host_kv_page_layout`）+ 一处硬编码（`prepare.cpp:72` 改由几何推导）。span=1 的现有路径零行为变化。
- 现状边界：`cache.index.keys` 只在 `src/ops/softmax_attention/qsa/` 内出现，尚未接入 Engine 的 Host context 路径（`grep index.keys` 在 ops 外零命中），因此跨 Device/Host 的 4 行几何**还没有被真正需要**；提前引入会多一个无人消费的字段。
- 判定：**条件缺失**（表达方式不同，功能等价）。建议登记，待 QSA 接入 Host context（T193/T154 线）时随行吸收，不在 T146 动代码。

## T147 分离内存所有权与 CUDA 执行（`6a42de35`）

上游把 `arena.cu` 拆成 9 个文件：`arena.{h,cpp}` 只做游标分配、`buffer.{h,cpp,cu}` 持 `HostBuffer/PinnedHostBuffer/DeviceBuffer`、`memory.h` 描述位置、`transfer.{h,cpp,cu}` 提供 `copy_bytes*`/`CudaCopyBatch`、`device.*` 持 stream。文档合同见 `engine-architecture.md` §2.6（本次新增 21 行）。

| 能力 | 上游 | 本地 | 判定 |
|---|---|---|---|
| 位置描述 `Device`/`MemorySpan`/`CudaHostView` | `memory.h` | `src/core/memory.h:10/50/75`，与上游同源（已吸收） | 等价 |
| Device 归属 | `buffer.h` `DeviceBuffer`（带 `Device` 入参） | `src/core/arena.h:19` `DeviceBuffer`（单 `cudaMalloc`，`arena.cu:61`）；无 `Device` 入参（单卡） | 等价（单卡退化） |
| Host / Pinned 归属 | `buffer.h` `HostBuffer`/`PinnedHostBuffer` | 只有 `arena.h:93` `PinnedHostBuffer`；普通 Host 用 `std::vector`/`MemoryArena` | 部分等价 |
| CUDA 注册 | `CudaHostRegistration(ConstMemorySpan, Device)` | `src/core/cuda_host.h:14` `CudaHostRegistration(void* host, size_t bytes, int cuda_ordinal)` | 部分等价（裸指针，丢位置标签） |
| arena 只管游标 | `arena.cpp` | `src/core/memory_arena.cpp`（`arena.h` `MemoryArena`） | 等价 |
| 执行视图 | `CudaExecutionView`（`device.h`） | `src/core/device.h:43` `DeviceExecutionView` + `:59` `using CudaExecutionView` | 等价（已本地化） |
| 统一传输 | `transfer.h`：`copy_bytes`/`copy_bytes_async`/`copy_rows_async`/`copy_host_2d_async`/`CudaCopyBatch` | **无**；`src/` 内 113 处 `cudaMemcpyAsync` 与 12 处 `cudaMemcpy2DAsync` 直连（`ffn.cpp` 19、`text.cpp` 16、`state_image.cpp` 11、`paged_kv_cache.cpp` 11） | 本地缺失 |
| 平台层自有所有权 | 无对应物 | `vmm_pool.{h,cpp}`（VA 预留 + 段回收）、`host_kv_arena`、`kv_spill_file`、`lmcache_bridge`、`memory_budget/ledger` | 本地领先 |

- 判定：**部分等价**。最危险的一点已被吸收（`memory.h` 的位置标签 + `DeviceExecutionView` 单一执行环境），本地平台层的内存所有权是自研的、覆盖更宽（可回收显存、落盘、LMCache）；缺的是**传输收口**（113+12 处直连）与 `CudaHostRegistration` 的 span 化签名。
- 建议：登记，不重实现。传输层收口属"另立项"级（125 处调用点 + 35B 热路径 stream 序），当前没有正确性缺陷，只有可审计性问题；`CudaHostRegistration` 签名可在下一次触碰 `cuda_host.*` 时顺带修正。
- 结论一句话：本地平台层内存所有权**已自研且更宽**，M 类冲突不成立（`arena.cu` 已被本地改造成核心组件，不能按上游删除/拆分融合）。

## T148 统一 CPU 执行基础 + CPU routed swiglu（`74671a4e`＋`8e3dff94`）

- 本地**已落地**，且是 Windows 自研平台层的直接产物：
  - `src/core/cpu/system.h:35` `CpuTopology`、`:79` `CpuFeatures`（含 `intersect` `:90`）、`system_windows.cpp`（Windows 拓扑/绑核实现，替代上游 hwloc 假设）。
  - `src/core/cpu/executor.h:39` `CpuExecutionView`、`:61` `keep_active()`、`CpuWorkerContext::barrier`（`:28`）、`task_pool.{h,cpp}`。
  - `src/ops/moe/routed_swiglu/cpu/{q4_q5,q4_q6,q8_q8}/shapes/h2048_i512_e256_k8/dispatch.cpp` 与上游**同路径**；本地另加 `nvfp4/shapes/h2560_i640_e512_k10`（qwen4-exp 几何）与 CUDA 侧同构 dispatch。
- 差异：
  1. 数值档：上游 q4/q5/q6（int4 权重）；本地 `moe/kernels/cpu/groupwise_int` 支持 int8/**int16 默认**（`launch.cpp:85` 读 `NINFER_MOE_CPU_INT8`：未设/16 → int16，`=8` → int8，`=0` → float），精度收益与性能见 v9 C04 记录。
  2. 旧线程池仍在：`src/core/host_worker_pool.{h,cpp}` 只服务 `frontend/media_cache.cpp:3`（上游 74671a4e 已删）；属**叶节点**，不参与数值路径。
  3. 选择策略：本地 `src/models/qwen3_5/execution/cpu_path_policy.h`（代价决策 + 开关）比上游"按列数选混合/整层"更硬，且有实测。
- 判定：**等价 + 本地领先**。建议记录，不重实现；`host_worker_pool` 可在 media singleflight 改造时并入 `TaskPool`（非本批）。
- 结论一句话：CPU 半区已是"统一基础 + routed swiglu + 本地 int16 默认"，与上游同构，差异是数值档与 Windows 拓扑实现。

## T149 qwen3_5 dense/MoE 分离 + 共享 qwen 基础（`8cf26bb6`＋`9cfd5646`）

- 上游 `8cf26bb6`：从 `ffn.cpp` 拆出 `dense_mlp.{cpp,h}` 与 `moe.{cpp,h}`，`parameters` 去掉合并字段。
- 本地：**参数层已分离、文件层合体**：
  - `src/models/qwen3_5/execution/parameters.h:46` `std::variant<DenseParameters, ops::SparseMoeWeights, RoutedMoeParameters>`；
  - `ffn.cpp:1978/1982/2051/2059` 按 variant 分派（sparse MoE / routed MoE 各两处）；
  - `ffn.cpp` 127 KB 单文件，共享同一套放置、预取、CPU 兜底与图捕获逻辑（35B 热路径的主体）。
- 上游 `9cfd5646`：抽出 `src/models/common/{bindings.cpp,bindings.h,startup.h,weights.cpp,weights.h}` + `common/{context,moe,qwen}` 三目录，把 qwen3_5 的 26 个 frontend 文件迁到 `common/qwen/frontend/`。
- 本地：`src/models/` 只有 `qwen3_5/`（frontend 32 文件，`src/models/qwen3_5/frontend/`）与 `qwen4_exp/ple/`；无 `models/common`。
- 判定：**功能等价、结构分叉**。
  - dense/MoE 拆分：本地文件边界更粗，但**没有第二个消费者**需要单独编译 unit；拆开反而要把共享的 expert 放置状态外置。**不跟随**，登记。
  - 公共前端抽取：本地只有一种 Qwen 前端（GLM 判死、qwen4-exp 前端未开工），抽出去没有第二个模型受益，且会动 32 个文件的 include 面与测试。**不吸收（记录）**；待 qwen4-exp 前端立项时再做（T191 前置）。
- 结论一句话：本地以"参数 variant + 单 ffn 文件"实现同一分离，不跟随上游文件级重构。

## T150 统一目标能力与 roofline 引用（`066714cc`）

- 上游：新增 `src/core/platform.h.in`（`NINFER_TARGET_RTX5090`/`NINFER_TARGET_SPARK` + `NINFER_SHARED_PHYSICAL_MEMORY`/`NINFER_EXPERT_OFFLOAD` 两个独立能力位）与 `bench/common/roofline.h`（按 target 选 `memory_gbs`/`sustained_read_gbs`/bf16/fp8/mxfp8/nvfp4 TFLOPS，未建立项为 NaN）；`NinferTargets.cmake` 加 `${PROJECT_BINARY_DIR}/generated`。
- 本地：
  - `cmake/NinferPlatform.cmake:8–30` 已有 target 轴（`rtx4090`/`rtx5090`，无 spark，注释明确"adapted from upstream 8ed68325"）；`cmake/NinferTargets.cmake` 缺 generated include 根。
  - **无** `src/core/platform.h.in`、无能力宏；**无** `bench/common/roofline.h`（`bench/common/` 只有 `fixture_data.cuh`）。
  - roofline 常量仍是 **RTX 5090 的 1792 GB/s**，硬编码在 **11 处**：`bench/ops/ninfer_bench_common.h:32`（`kRooflineGBs`）、`linear_bench.cu:43`、`bf16_linear_add_bench.cu:33`、`context_softmax_attention_bench.cu:41`、`gdn_input_proj_bench.cu:34`、`kv_cache_append_bench.cu:38`、`packed_softmax_attention_bench.cu:38`、`prepare_masked_block_bench.cu:28`、`q4_linear_swiglu_bench.cu:137`、`q5_linear_add_bench.cu:36`、`sliding_window_attention_bench.cu:41`；`bench/README.md:298` 仍写"fixed RTX 5090 memory reference"。
  - 本地等价物是另一层：`docs/maintainer/sm89-fallback-table.md`（逐 shape Tier A/B/C 能力表）+ `linear-benchmark.md:346` 的 1792 GB/s 固定规格说明。
- 差异（Windows 分支必须登记）：
  1. 4090 的 DRAM 参考不是 1792；当前 bench 的 `DRAM_%` 在 4090 上系统性偏低（分母过大），**不能用于跨机比较**。
  2. 本地无 NVFP4 张量核（Ada），`nvfp4_f32_tflops` 必须缺失（NaN）而不能照抄 5090 的 1676。
  3. 本地无 Spark target，`NINFER_SHARED_PHYSICAL_MEMORY` 恒 0；`NINFER_EXPERT_OFFLOAD` 恒 1（35B 线依赖）。
  4. 上游"能力位与 target 身份分离"的原则可吸收，但收益在**编译期判定**，本地目前用 CMake 变量 + `#if defined(NINFER_WINDOWS)` 承担。
- 判定：**本地缺失统一层**。建议另立 Phase 5 条目（改 11 个 bench 文件 + `bench/README` + 新增 `roofline.h`），本批**不动代码**：会改动全部 bench 输出数字，须与 T004/T016 的审计清单一起重录。T147 的 `platform.h.in` 可同批引入。
- 结论一句话：roofline 引用未按 target 化，4090 分支仍用 5090 的 1792 GB/s，属登记在案的 Phase 5 口径缺口。

## T151 测试套件重组（`f30e661f`＋`91972187`）

- 上游：`tests/ops/` 从"一 op 一文件"改为"一家族一 runner + cases"（`elementwise/main.cpp` + `tests.cmake`、`linear/{add,projection,split,swiglu}/{cases_*.cpp,runner.cpp,test.cpp}`），共享 `tests/ops/suite.{cpp,h}`、`float_codec.h`、`kv_cache_storage.h`；`91972187` 删 `linear-test-reorganization-plan.md`。
- 本地：`tests/ops/` 顶层 57 个文件、其中 **46 个 `test_*.cpp`**（递归含家族子目录为 85 个 `test_*.cpp`），家族目录只有 `gated_residual/`、`gdn_control/`、`linear/`、`linear_add/`、`linear_pair/`、`linear_swiglu/`、`ple/`、`rmsnorm/`、`softmax_attention/`（部分跟随），目标注册在 `tests/ops/tests.cmake`（扁平列表 24 个 basic op）+ 各家族 `tests.cmake` + `tests/cmake/NinferTests.cmake:8` `ninfer_add_test` / `:38` `ninfer_add_op_test`（`:40` `SKIP_RETURN_CODE 77`、oracle 关闭 fast-math）；上游已把 `ninfer_add_op_test` 改成 `FORMATS` 分区（`tests/cmake/NinferTests.cmake` 新增 `cmake_parse_arguments`），本地未跟随。
- 判定：**结构分叉**。本地"一 op 一目标"并行度高、失败定位直接；suite 化的收益是链接次数与 case 表复用，代价是本地 46 个目标的重命名 + CTest 名变化（会打断既有门禁脚本与 `-R` 过滤）。
- 建议：**不跟随**，只登记映射（`test_<op>.cpp` ↔ `<family>/<op>.cpp`）。T151 的验收就是"本地测试目标不破"：本批未改 `tests/`，目标集合不变。
- 结论一句话：跟随映射登记，测试目标零改动。

## T152 架构批小结门禁

- 本组产出仅文档：新增本登记 + 回填 v9 计划 T145–T152 行与变更记录，**零代码 diff**。
- 证明：`git diff --stat` 只含 `docs/`；`cmake --build build-win -- -n` 只输出 `[0/2] Re-checking globbed directories...` 与 `[1/2] Re-running CMake...`，**零个编译/链接步骤**（dry-run 因不真正执行 CMake 而在此停住，恰好说明其后没有源文件需要重编）。
- 因此 35B 双档 PPL（`4801/2.051275/7.777808`、`4801/2.049253/7.762104`）、27B regress fast（PPL `7.441868`、selector-off dump `636e38de95046781`）、moe 三点与 decode hash 均沿用 T126–T129 组门禁记录，**不重跑**：没有对象被重编，重跑只会复现同一基线。
- 结论：**纯对表无 diff**，构建与门禁基线未变化。

## 后续立项（本批只登记，不执行）

| 条目 | 内容 | 触发条件 |
|---|---|---|
| A1 | paged KV `tokens_per_row`（T146 融合点 5 处） | QSA index 页接入 Host context |
| A2 | 统一传输层 `copy_bytes*`/`CudaCopyBatch` + `CudaHostRegistration` span 化（T147） | 多 stream/异步合并需求出现，或 119 处调用点需要审计 |
| A3 | `platform.h.in` 能力位 + `bench/common/roofline.h` 按 target 取值（T150） | Phase 5 bench 基线重录批次 |
| A4 | `models/common/qwen` 前端抽取（T149） | qwen4-exp 前端开工（T191） |
| A5 | ops 家族重组（T145） | 吸收稳定后单独立项，须先冻结 35B 工件与 include 面 |
