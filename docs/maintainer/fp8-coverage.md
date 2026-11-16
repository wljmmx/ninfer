# FP8 A8 覆盖度：各 GEMM 精度路径 × 计算量（V10-37）

> **性质**：只读分析。本轮未编译、未运行、未执行任何 git 写操作。
> **口径**：所有 FLOPs 数字都是按「层数 × 每层 GEMM 形状」的**静态估算**，不是实测。本文不写 tok/s、不写收益百分比（第 7 节给出唯一允许的引用来源与出处）。
> **模型**：35B = Qwen3.6-35B-A3B（H=2048，40 层：10 全注意力 + 30 GDN），27B = Qwen3.8-27B（H=5120，64 层：16 + 48）。几何取自 `docs/maintainer/qwen3_5-model.md:59-72`。

---

## 0. 结论摘要

1. **精度选择是四层门**，不是一处开关：工件元数据 `activation_policy` → `LinearPolicy` → 权重 `QType` → `(形状注册, token 阈值, 编译宏)`。任一层不满足就静默退回 A16（第 1 节）。
2. **FP8 的线性几何全是 H=5120**：注册表只有 6 个（`fp8_dispatch.cpp:9-10`），A8 激活量化只认 K∈{5120, 6144, 17408}（`fp8_a8.cu:88-97`）。**35B（H=2048）在 FP8-A8 上是零覆盖**；而且 `weight_input.cpp:148-152` 的单父格式白名单把 MoE 几何的单父**限定为 Q8**，35B 即使注册了新形状也会在权重准备阶段被拒。
3. **27B 的 A8 覆盖是 95.0% 的每 token GEMM FLOPs，但当前跑的是 int8（W4A8），不是 FP8**：官方配方把 Q/K/gate/up 定到 Q4、其余 Q5（`official_recipes.py:61-86`），四角色自然落进「成对 Q4+Q5」形态（`weight_input.cpp:155-159`），而注意力/GDN 的成对形态**强制 `AllowA8`**（`attention.cpp:63-64`、`gdn.cpp:134-135`），于是默认开启的 W4A8 在 `T%128==0` 时接管。**FP8-A8 的同批形状代码也齐（95.0%），但本地没有任何 FP8 工件**（三份 `conversion.json` 的 formats 汇总里没有 `fp8_*`），所以 FP8-A8 GEMM 实际调用次数为 0。
4. **35B 的 A8 是零**：dense 投影是 Q8（A16 数学，`q8_a16_mma.cuh:1-10`）、专家是 Q4/Q5（A16）、头是 Q6（A16）。唯一现成入口是 **MoE int8 MMA**，内核已写但默认关（`grouped_swiglu.cu:127-132`）。
5. **FP8 A8 对解码是"看权重格式"的事，不是普适收益**：A8 只换激活精度，不减权重字节。按 bpw 算：Q4 g64 ≈ 4.25、Q5 ≈ 5.25、Q6 ≈ 6.25、**Q8 g32 ≈ 8.06、FP8 row ≈ 8.00**。解码是权重带宽界（`docs/performance/rtx-4090-sm89-baseline.md:110` 原文 "Decode is weight-bandwidth bound"）。所以：
   - **27B（Q4/Q5）换 FP8 工件 = 解码变慢**，只在 prefill-only / 打分类批量部署里划算；
   - **35B 的 dense 投影本来就是 Q8（8.06 bpw），换 FP8 是字节中性**，纯赚 prefill 的 tensor 吞吐 —— 这是本表里唯一"FP8 不付解码代价"的窗口。
   想在 Q4/Q5 权重上拿 A8，正确路线是 int8（W4A8 / MoE int8 MMA），不是 FP8。
6. `lm_head` 不在候选里：27B 的头形状 `uses_a8` 恒 false（`n248320_k5120.cu:100`），是明确决定而非遗漏；35B 的头连几何都没有。

### 0.1 A8 现状矩阵（一眼看完）

| 机制 | 27B（Qwen3.8） | 35B（Qwen3.6-A3B） | 触发条件 | 证据 |
|---|---|---|---|---|
| FP8-A8 GEMM（E4M3 权重 × E4M3 激活） | 代码就绪 95.0%，**工件无 FP8 → 0 次** | **不可用**（无形状 + 单父白名单只认 Q8） | FP8 工件 + `AllowA8` + T≥形状阈值 | `fp8_dispatch.cpp:36-45`、`weight_input.cpp:148-152` |
| int8-A8 `W4A8`（Q4/Q5 权重 × int8 激活） | **默认开**，dense 95.0% 覆盖，仅 `T%128==0` | 不适用（无 Q4/Q5 成对投影） | Q4/Q5 成对或 Q4 swiglu + 形状白名单 + T%128==0 | `w4a8_gemm.cu:1139-1229` |
| int8-A8 `MoE int8 MMA`（Q4 专家 × int8 激活） | 无 MoE | 内核已写，**默认关** | `NINFER_MOE_INT8_MMA=1` + Q4 gate/up + Q4 非 split 银行 | `grouped_swiglu.cu:127-132, 163-199` |
| FP8 KV 的 MXFP8 QK | 实测页在用（FP8 row-256 KV） | 实测页用 INT8 group-64 KV | `--kv-dtype` | `mxfp8_tiled_mma.cuh:298`、`docs/performance.md:32-33` |

---

## 1. 四层门：精度选择发生在哪里

### 1.1 第一层——工件声明（不是代码）

每个逻辑权重在每个数学输入上的许可来自工件 directory 的 `activation_policy`：

- `src/models/qwen3_5/load/prepare.cpp:147-176`：`A16Only/AllowA8/AllowA4` 三选一映射；缺字段直接抛错（`prepare.cpp:149-151`）。
- 转换侧写入口：`tools/convert/recipe.py:147-213`（`activation_policy=`），官方配方见 `official_recipes.py:229, 255`。

**含义**：内核齐全 ≠ 会跑 A8。本地 27B/35B 的三份配方（`official_recipes.py:89-94`、`:175-200`、`:97-172`）**都不写 `AllowA8`**，所以 FP8-A8 即使有内核也没有许可；27B 的 int8-A8 是**例外**——成对形态在代码里被强制 `AllowA8`（见 1.2）。

### 1.2 第二层——LinearPolicy 与"强制放行"的两个点

`include/ninfer/ops/linear.h:28-37`：`A16Only` / `AllowA8` / `AllowA4`；`allows_a8()` 对后两者为真。

| 位置 | 行为 |
|---|---|
| `execution/linear.h:10-26` | `project/project_add/project_swiglu` 一律传工件给的 `p.policy` |
| `execution/attention.cpp:57-68` | 分离 → 各自 policy；**成对 → 硬编码 `AllowA8`**；单父 → `single.policy` |
| `execution/gdn.cpp:126-139` | 同上；成对 → 硬编码 `AllowA8` |
| `execution/parameters.cpp:486-489, 532-536, 574-577, 638-644` | MoE 路由专家与共享专家**一律硬编码 `A16Only`** |

### 1.3 第三层——权重 QType 分派

`src/ops/linear/linear.cpp:87-112` 是纯 GEMM 的唯一分派点：

```
Q4_G64_FP16 → q4     Q5 → q5     Q6 → q6     Q8_G32_FP16 → q8
BF16 → bf16          NVFP4 → nvfp4            FP8_E4M3FN_ROW_BF16 → fp8
```

`q4/q5/q6/q8` 都是 **A16 数学**：权重反量化成 BF16 再做 BF16 MMA（`q8_a16_mma.cuh:1-10` 原文 "staged with cp.async before dequantization into a swizzled BF16 shared tile … m16n8k16 BF16 MMA"）。**Q8 不是 A8**，这一点在 35B 上是决定性的。

### 1.4 第四层——形状注册、token 阈值、父形态白名单、编译宏

**a) FP8 形状注册表**（`fp8_dispatch.cpp:9-17`）只有 6 个几何：
`14336×5120 / 16384×5120 / 34816×5120 / 5120×6144 / 5120×17408 / 248320×5120`。不在表内直接 `throw "fp8 linear: unsupported shape"`。

**b) A8 阈值**（同一个算子族里也不统一）：

| 形状 | 角色 | A8 起始 | 证据 |
|---|---|---:|---|
| 14336×5120 | 27B attn in | T ≥ 17 | `fp8/shapes/n14336_k5120.cu:56` |
| 16384×5120 | 27B gdn in | T ≥ 17 | `.../n16384_k5120.cu:57` |
| 34816×5120 | 27B MLP gate/up | T ≥ 5 | `.../n34816_k5120.cu:57` |
| 5120×6144 | 27B attn/gdn out | T ≥ 17 | `.../n5120_k6144.cu:51` |
| 5120×17408 | 27B MLP down | T ≥ 17 | `.../n5120_k17408.cu:59` |
| 248320×5120 | 27B lm_head | **永不** | `.../n248320_k5120.cu:100` |

另有独立阈值：`linear_swiglu` T≥5（`fp8_linear_swiglu_plan.cpp:26`）、`linear_add` K=6144→T≥17 / K=17408→**T≥20**（`fp8_linear_add_plan.cpp:28-29`）、`attn_input_proj` / `gdn_input_proj` T≥17（`fp8_attn_input_plan.cpp:24`、`fp8_gdn_input_plan.cpp:23`）。

**c) A8 激活量化只认三种 K**：`fp8_a8.cu:88-97`（5120 / 6144 / 17408），其它 K 抛 `unsupported K`。这是 35B 进不来的第二道硬门。

**d) 父形态白名单**（最容易被忽略的一层）：
`src/ops/weight_input.cpp:115-160` 决定四角色是"单父 / 成对 / 分离"，并且：
- 几何必须是 dense（27B：attn 6144/1024/6144/1024，gdn 2048/2048/6144/6144）或 moe（35B：attn 4096/512/4096/512，gdn 2048/2048/4096/4096），否则 → 分离（四个独立 Linear）；
- 单父且连续 → `supported = (moe && Q8) || (dense && (NVFP4 || FP8 || (attention && BF16)))`，**不满足直接抛 `"input projection: unsupported single-parent format"`（`weight_input.cpp:148-152`）**；
- 非连续 → 必须是 dense，且**只接受 Q4+Q5 成对**（`:155-159`）。

推论（都可以反推工件形态，不必先读工件）：

- **35B 必然是单个连续 Q8 父**：moe 几何若父不连续会走到 `:155` 的 `require(dense, "unsupported multi-parent geometry")` 直接失败；单父又只允许 Q8。
- **35B 若换 FP8 单父会被 `:148-152` 拒掉**，必须先改白名单（这是 N3 的前置条件，不是"注册形状就行"）。
- **27B 的 Q8 单父反而被拒**（attention 只允许 NVFP4/FP8/BF16；非 attention 的 dense 几何只允许 NVFP4/FP8，连 BF16 都不行）。所以在"工件 formats 只有 q4/q5/q6/q8"的前提下，27B 的投影**只能是成对 Q4+Q5**，这也正是它天然吃到 int8-A8 的原因。

**e) 编译宏**：`NINFER_ENABLE_TMA`（sm_89 走 `#else` 的 cp.async 回退，`n14336_k5120.cu:41-53`）、`NINFER_ENABLE_NVFP4` / `NINFER_ENABLE_K8V4`（注意力侧，`causal_softmax_attention.cpp:344-373`）。sm_89 不是"没 TMA 就没 A8"。

### 1.5 一个容易误读的点

`AllowA8` 只是**许可**，不是执行承诺。仓库自己写明了：`docs/maintainer/linear-benchmark.md:470-471`——"policy 只是许可，长期 benchmark 不把许可本身冒充为低精度执行。当前所有预置 suite 显式使用 `A16Only`"。判断"是否真的 A8"必须同时看形状表、阈值与父形态，不能只看 policy。

---

## 2. 表 A：路径 × 精度选择点

| # | 路径（算子） | 精度选择点（文件:行） | 支持精度 | 默认（本地工件） | 选择条件 |
|---|---|---|---|---|---|
| A1 | 纯 GEMM `ops::linear`（lm_head、MTP、DFlash、vision、共享专家单矩阵） | `src/ops/linear/linear.cpp:87-112`（QType 分派）；`src/ops/linear/fp8/fp8_dispatch.cpp:36-45`（A16/A8） | BF16 / Q4 / Q5 / Q6 / Q8（均 A16 反量化）/ NVFP4（A16 或 A4）/ FP8（A16 或 A8） | 随工件：27B 头 Q8（`official_recipes.py:93-94` 传入 Q8 词表）、35B 头 Q6、MTP/DFlash Q8、draft 头 Q4、vision Q4/Q5/Q6/Q8（`official_recipes.py:19-58, 61-86, 175-200`） | qtype 分派命中 + FP8 时形状∈6 几何 + T≥该形状阈值 + `AllowA8` + 调用方 workspace（`fp8_dispatch.cpp:39-40`）。便捷重载**强制 A16Only**（`linear.cpp:166-169`） |
| A2 | 融合头 `linear_topk`（仅 draft/proposal） | `src/ops/linear_topk/linear_topk.cpp:45-67`（profile 白名单）、`:252-262`（launch） | Q8 full（248320×5120）、FP8 full（248320×5120，**A16 数学**：`fp8.cu:114-129` 用 `fp8_e4m3x2_to_bf16x2_bits` + `mma_bf16`）、Q4 optimized（131072×5120）、Q5 full（248320×K，K≤5120 且 %128==0） | 随头工件：27B draft Q4；正常头走 A1 | `kLinearTopKFullRows=248320`、`kLinearTopKHidden=5120`（`linear_topk_workspace.h:15-18`）→ **35B（K=2048）只剩 Q5 档**；Q4 档另需 id_map |
| A3 | 稠密 MLP gate/up `linear_swiglu` | `src/ops/linear_swiglu/fp8/fp8_linear_swiglu_plan.cpp:20-26` | Q4/Q5/Q6/Q8/BF16（A16）、FP8-A16、FP8-A8、NVFP4-A16/A4、W4A8-int8 | 27B：Q4 34816×5120 → **W4A8-int8 优先**，否则 A16；35B 无此路径（无稠密 MLP） | FP8：T≥5 **且形状必须 34816×5120**（`fp8_linear_swiglu_plan.cpp:46` 硬编码）且 `AllowA8`；W4A8：Q4 + 34816×5120 + T%128==0（`w4a8_gemm.cu:1176-1181`），并优先于 FP8（`ffn.cpp:2880-2885`） |
| A4 | 残差投影 `linear_add`（MLP down / attn out / gdn out） | `src/ops/linear_add/fp8/fp8_linear_add_plan.cpp:20-30` | 同上 + FP8-A8（K=6144 与 17408 两档） | 27B：Q5 → **W4A8-int8 优先**（`w4a8_supported` 不检查 qtype，`:1157-1164`），否则 A16；35B 无此形状 | 形状必须 5120×6144 或 5120×17408；FP8 阈值 17 / **20**；W4A8：n=5120 且 k∈{6144,17408} 且 T%128==0 |
| A5 | 注意力输入投影 `attn_input_proj` | `src/ops/attn_input_proj/fp8/fp8_attn_input_plan.cpp:18-25`；成对门控 `q4_q5/q4_q5_attn_input_plan.cpp:99-105`；形态判定 `src/ops/weight_input.cpp:115-160` | 单父：BF16/Q8/FP8(A16/A8)/NVFP4(A16/A4)（**按几何分别白名单**）；分离：四路独立 Linear；成对 Q4+Q5：A16 或 W4A8-int8 | 27B：**成对 Q4(q,k)+Q5(gate,v)** → int8；35B：**Q8 单父 9216×2048**（`wrapper/attn_input_proj.cpp:151, 193`） | FP8 只支持 14336×5120 单父 + T≥17；成对形态由 `attention.cpp:63-64` 强制 `AllowA8`；35B 的 moe 几何单父只允许 Q8（`weight_input.cpp:149`） |
| A6 | GDN 输入投影 `gdn_input_proj`（qk+v+z 融合 + 卷积） | `src/ops/gdn_input_proj/fp8/fp8_gdn_input_plan.cpp:17-24`；成对门控 `q4_q5/q4_q5_gdn_input_plan.cpp:147-148` | 单父：**只有 FP8 或 NVFP4**（非 attention 的 dense 几何在 `weight_input.cpp:150-151` 既无 Q8 也无 BF16 单父档，Moe 几何则只允许 Q8）；成对 Q4(qk)+Q5(value+z)：A16 或 W4A8-int8 | 27B：成对 Q4+Q5 → int8；35B：Q8 单父 12288×2048（`wrapper/gdn_input_proj.cpp:341-359`，按 moe 几何的白名单） | FP8 只支持 16384×5120 + T≥17；`gdn.cpp:135` 对成对形态强制 `AllowA8` |
| A7 | 路由专家 `sparse_moe` / `topk_softmax_swiglu_moe` | `src/ops/moe/operands.cpp:31-72`（几何/格式白名单）；`src/ops/moe/routed_swiglu/cuda/dispatch.cpp:6-26`（实现族） | Q4/Q5、Q4/Q6、Q8/Q8（**BF16 激活** = A16）；block-scaled FP8 E4M3 128×128 + A8（仅 H=2560/I=640/E=512/k=10）；NVFP4 + A4（同几何） | 35B：Q4 gate/up + Q5 down（层 34/38/39 为 Q6 down）→ h2048 整数族 A16；policy **硬编码 A16Only**（`parameters.cpp:486-489/533-536/574-577/638-644`） | 几何白名单只有 `(2048,E256,k8)` 与 `(2560,E512,k10)`（`operands.cpp:32-34`）；块缩放 FP8 还要求 I=640 且两侧 policy 都 `AllowA8`（`operands.cpp:44-51`） |
| A8 | MoE 共享专家（SwiGLU 矩阵对） | `src/ops/moe/operands.cpp:74-89`；profile `parameters.cpp:486-489, 641-644` | H=2048：Q8 SwiGLU（A16）；H=2560：**仅 BF16 且仅 A16Only** | 35B：Q8 1024×2048 + 2048×512 | 无 A8 选项（`operands.cpp:79-80` 把 `A16Only` 写进校验） |
| A9 | 注意力核心 `causal_softmax_attention` / `_cached` | `src/ops/softmax_attention/dense/causal_cache/causal_softmax_attention.cpp:424-506`（append）、`:565-633`（cached） | BF16 MMA；**FP8 KV → MXFP8 QK**（`common/mxfp8_tiled_mma.cuh:298`）+ **FP16 PV**（`:473`）；INT8 KV → s8 QK + f16 PV；rk*/E8 复用 int8 核；NVFP4/K8V4 在 sm_89 编不进来 | `KvCacheStorage::BFloat16`（`include/ninfer/types.h:258`）；实测页 27B 用 FP8 row-256、35B 用 INT8 g64（`docs/performance.md:32-33`） | `--kv-dtype` / 容量规划；与 LinearPolicy 无关 |
| A10 | Qwen4Exp QSA（indexer 投影 + sparse GQA） | `src/ops/softmax_attention/qsa/input_projection.cpp:36-58`（BF16/Q8 白名单） | BF16、Q8 投影；sparse GQA 跟 A9 的 KV 家族 | 工件 Q8（`official_recipes.py:259-318`） | 形状固定 k=2560；Q8 需 RowSplit/G32/FP16 scale |
| A11 | 横切：W4A8（int8 激活 × Q4/Q5 权重） | `w4a8_gemm.cu:1139-1155`（开关）、`:1157-1229`（四条 profile） | int8 激活 + Q4/Q5 → s8 MMA | 默认**开**（`NINFER_W4A8=0` 关）；接线 `ffn.cpp:2880-2893`、`text.cpp:2199-2204/2478-2484`、两个 q4_q5 plan | n=5120 且 k∈{6144,17408}；或 swiglu 34816×5120 Q4；或成对 attn/gdn（k=5120）；**且 T%128==0**。`w4a8_supported` 不校验 qtype（Q4/Q5 都可），swiglu 校验 Q4 |
| A12 | 探针 `act_quant_a8_roundtrip_bf16` | `src/ops/act_quant/act_quant_a8.cu:113-150` | int8 往返（只改数值，不提速） | **默认关**（`NINFER_ACT_A8` 未设即 no-op，`:141`） | 叫用于 `ffn.cpp:2839, 2887` |

---

## 3. 表 B：计算量 × A8 覆盖 × 候选优先级

**估算口径**：每 token FLOPs = 2·N·K × 层数，只算 GEMM；不含 attention 的 O(L) 核心项、KV append、卷积、norm、路由 top-k、采样。"占比"是**该模型 body+head 每 token GEMM FLOPs 占比**，不是耗时占比（解码受带宽约束，见第 7 节）。

| 路径 | 占比 27B / 35B | 已 A8？ | A8 候选优先级 | 需要的改动（内核 / 格式 / 接线） |
|---|---|---|---|---|
| 路由专家（35B 8×gate_up+down） | — / **34.2%** | 否（A16）；int8 MMA 内核已写但**默认关** | **P1（int8）** | 只需把 `NINFER_MOE_INT8_MMA` 的 PPL 门禁跑通再默认化；**不要**换 FP8 权重（专家是 Q4/Q5，4.25/5.25 → 8.0 bpw，方向相反） |
| 稠密 MLP gate/up | **44.6%** / — | 是（int8 W4A8，T%128；FP8 代码就绪但无工件） | P4（27B 无缺口） | 仅 FP8 工件（34816×5120 单父 + `AllowA8`）；注意 Q4→FP8 会增加解码字节（仅 prefill-only 值得） |
| 稠密 MLP down | **22.3%** / — | 是（int8 W4A8；FP8 T≥20 就绪） | P4 | 同上（5120×17408） |
| GDN 输入投影 | 15.7% / **25.7%** | 27B 是（int8）；**35B 否**（Q8 单父，Q8=A16） | **P1（35B）** | 35B：Q8→FP8 是**字节中性**（8.06→8.00），优先走 FP8 单父 —— 需 (12288,2048) 几何 + K=2048 量化档 + **改 `weight_input.cpp:149` 放行 moe 单父的非 Q8 格式** |
| GDN 输出投影（5120×6144 / 2048×4096） | 5.9% / **8.6%** | 27B 是；35B 否 | P1（35B，与下面共用形状） | 35B 需 (2048,4096) 几何 + K=4096 量化档 |
| 注意力输入投影 | 4.6% / **6.4%** | 27B 是（int8 成对）；35B 否 | P2（35B） | 35B 需 (9216,2048) 几何 + K=2048 量化 + 同上白名单 |
| 注意力输出投影 | 2.0% / 2.9% | 与 GDN 输出同形状同算子 | P2（35B） | 与 GDN 输出共用 (2048,4096)，一次注册两处受益 |
| MoE 共享专家 | — / 4.3% | 否（Q8 A16） | P5（不建议） | 1024×2048/2048×512 太小；且 Q8→FP8 虽是字节中性，收益落在尾延迟里 |
| `lm_head`（248320×K） | 5.0% / **17.3%** | **否；27B 显式关闭**（`n248320_k5120.cu:100`） | P5（不推荐先做） | 需新几何 (248320,2048)；27B 先要推翻"为什么关"（logits 精度） |
| 注意力核心（FP8 KV 的 QK） | 6.3% @L=8192 / **22.8%** @L=8192 | **是**（MXFP8 QK，实测页在用） | 已完成 | 无；PV 仍 FP16（`mxfp8_tiled_mma.cuh:473`），做 A8 PV 属新数值路径 |
| MTP / DFlash / vision 投影 | 未计（≤1 层 + 视觉步） | 否（Q4/Q5/Q6/Q8，A16） | P6 | 形状碎、占比小；视觉 prefill-only |

**每 token GEMM FLOPs 原始估算（GFLOP/token）**

| 路径 | 27B 形状 × 层数 | 27B | 35B 形状 × 层数 | 35B |
|---|---|---:|---|---:|
| attn in | 14336×5120 ×16 | 2.35 | 9216×2048 ×10 | 0.38 |
| attn out | 5120×6144 ×16 | 1.01 | 2048×4096 ×10 | 0.17 |
| gdn in | 16384×5120 ×48 | 8.05 | 12288×2048 ×30 | 1.51 |
| gdn out | 5120×6144 ×48 | 3.02 | 2048×4096 ×30 | 0.50 |
| MLP gate/up | 34816×5120 ×64 | 22.82 | —（MoE） | — |
| MLP down | 5120×17408 ×64 | 11.41 | — | — |
| 共享专家 | — | — | (1024×2048 + 2048×512) ×40 | 0.25 |
| 路由专家 ×8 | — | — | 8×(1024×2048 + 2048×512) ×40 | 2.01 |
| router | — | — | 257×2048 ×40 | 0.04 |
| lm_head | 248320×5120 ×1 | 2.54 | 248320×2048 ×1 | 1.02 |
| **合计** | | **51.20** | | **5.88** |
| 注意力核心（另计） | 4·H_q·D·L·层数，L=8192 | +3.22 | 同式 | +1.34 |

---

## 4. 结论：Top-N 候选

### N1（首选）35B 的路由专家 int8 MMA 默认化 —— 34.2%，内核已在

- **理由**：35B 在 A8 上是零，而它最大的单块计算量就是路由专家（34.2%）。这条路线**不动权重格式**（Q4 gate/up + Q5 down 保持 4.25/5.25 bpw），只把激活换成 int8，因此完全不碰解码字节。
- **现状**：`grouped_swiglu.cu:127-132` 原文 "Off by default; the split PPL gate has to pass against bf16 before it can default" —— 代码完成，缺门禁结论。
- **要的改动**：跑 PPL 门禁（default 逐位 + dump + regress），通过后默认化。两个已知边界：① 它只覆盖 **Q4 gate/up**（`:163-164` 条件），Q5 down 仍走 bf16；② 它挂在 grouped（prefill）tile 上，decode 走 `token_swiglu`，没有 int8 变体 —— 所以这条收益是 **prefill 段**的。

### N2 35B 的 dense 投影走 FP8 单父 —— 47.9%，且字节中性

- **理由**：35B 的 attn in/out + gdn in/out + 共享专家共 47.9%，全部是 Q8 单父（`weight_input.cpp:149` 只允许 Q8），Q8 = A16 数学 → 今天**无任何 A8 入口**。关键在于 **Q8 g32 ≈ 8.06 bpw 与 FP8 row ≈ 8.00 bpw 基本相等**，所以这是唯一"换 FP8 不让解码变慢"的窗口；而 W4A8 只认 Q4/Q5，对 35B 完全不适用。
- **要的改动**（四处，按依赖顺序）：
  1. `weight_input.cpp:148-152` 放行 moe 几何的 FP8 单父（一行白名单，但会改变权重准备阶段的拒绝面，需要正反例测试）；
  2. `fp8_dispatch.cpp:9-10` 注册 `(9216,2048)`、`(12288,2048)`、`(2048,4096)`；
  3. `fp8_a8.cu:88-97` 增加 K=2048 / 4096 的量化几何（`Fp8ActivationGeometry` 只要求 K%32==0，两者都满足）；
  4. 按 (N,K) 实例化 `Fp8A8MmaSchedule` / `Fp8A16MmaSchedule` + `n*_k*.cu` 形状文件 + workspace 查询；转换器给 35B 一条 FP8 单父配方。
- **顺序建议**：先做形状 2+3+4 中的 gdn_in（单个 25.7%，收益最大、形状最少），再补 attn in/out。

### N3 27B 的 FP8 工件（零内核改动，仅 prefill-only 部署有价值）

- **理由**：27B 的 FP8-A8 代码覆盖 95.0%，缺的只有工件——三份 `conversion.json` 的 formats 汇总都没有 `fp8_*`。配方侧已有先例：`official_recipes.py:233-256` 的 27B nvfp4 变体把**所有非 MLP 投影**写成 FP8 + `AllowA8`，MLP 层 <56 走 NVFP4、≥56 走 FP8。
- **要的改动**：只做转换（四角色必须并成**单父**：14336 / 16384 / 5120×6144 / 5120×17408），并在 Use 上写 `AllowA8`；内核、阈值、workspace 全部现成。
- **代价**：27B 的 MLP/投影现在是 Q4/Q5（4.25/5.25 bpw），换 FP8 是 +3 bpw 左右的解码代价（不给百分比，理由见第 7 节）；T≤16 还会退回 A16。**只在 prefill-only / 打分类批量部署里划算。**

### N4 27B 的 `lm_head`：先查清为什么关，不要直接开

- 27B 头形状 `uses_a8` 恒 false（`n248320_k5120.cu:100`）。这是唯一用 `return false` 硬编码的 A8 拒绝，且头是采样唯一入口——A8 的行级 scale 与采样链路的数值耦合在仓库内没有任何实测支撑。
- **先做的事**：`git log -S "uses_a8" -- src/ops/linear/fp8/shapes/n248320_k5120.cu` 找到引入 commit 及其 PPL 记录；有证据后再谈 35B 的 (248320,2048)。

### N5 已完成，不要重开

FP8 KV 的 QK 已经是 MXFP8-A8，27B/35B 的实测页在用（`docs/performance.md:32-33`）。唯一可议的是 PV 仍 FP16。

---

## 5. 明确"不改"的路径及理由

| 路径 | 不改的理由 |
|---|---|
| Q4/Q5/Q6/Q8 的 A16 分派主路 | 三份工件的全部权重格式；改任一条都会移动门禁 golden，而收益面已被 int8 / FP8 覆盖，属重复投入 |
| 路由专家的 **FP8 权重** | 专家是解码带宽主项；Q4/Q5（4.25/5.25）→ FP8（8.0）是**反向**。要 A8 就在 Q4 上做 int8（N1） |
| MoE 共享专家的 A8 | 35B 只有 4.3%，且是 Q8 小矩阵（1024×2048），A8 只是多一次量化开销 |
| `linear_topk` 的 FP8 档 | 它已是 A16 数学（`fp8.cu:114-129`），只服务 draft/proposal 头；改 A8 要重做 top-k 的 key 比较数值，风险/收益不对称 |
| QSA / vision 的 A8 | QSA 投影 k=2560 只有 BF16/Q8；vision 形状碎（1152/3456/4304/4608）且 prefill-only。没有"视觉 token 占比高"的实测前不动 |
| `act_quant_a8_roundtrip_bf16` | 默认 no-op，只用于精度实验，不应被当成"已有 A8" |

---

## 6. 无法确定的路径（缺什么证据）

| 待定结论 | 缺的证据 | 只读补法 |
|---|---|---|
| 每份工件里每个权重的**实际** qtype 与父形态 | 本地 `conversion.json` 只给 formats **计数**（27B q56：q4=144, q5=282, q6=5, q8=8, bf16=582, fp32=96, int32=1），不给 per-tensor 映射 | `tools/artifact/inspect.py` 读工件 directory（本轮按"不运行"约束未执行） |
| 27B 是否真的落在"成对 Q4+Q5" | 逻辑上可反推（`weight_input.cpp:148-159`：Q8 单父在 dense 几何被拒，非连续必须成对 Q4+Q5），但"工件四角色确实是 Q4/Q4/Q5/Q5"仍缺 per-tensor 证据；formats 计数（q4=144, q5=282）不足以证明 | 读工件目录里 `text/layers/0/{attention,gdn}/{query,key,gate,value,z}` 的 format 与 parent |
| 35B 的 FFN 走驻留 `topk_softmax_swiglu_moe` 还是流式 `routed_swiglu` | 取决于 expert bank 是否驻留（`parameters.cpp:519-579`） | 启动日志 `[experts] banks=… pool=…` 行 |
| FP8-A16 在 T≤16 的真实代价 | 仓库内**无** FP8-A16 vs A8 的同机实测文档 | 需要新测量（本轮禁止） |
| `lm_head` 关闭 A8 的原始原因 | 代码无注释，`docs/` 未见记录 | `git log -S "uses_a8"` |
| 各路径真实耗时占比（而非 FLOPs） | 无逐算子 profile 落盘文档 | 需要 NCU/引擎自报统计（本轮禁止运行） |
| W4A8 在非对齐尾块的占比 | 取决于真实请求长度分布；`prefill_chunk` 默认 1024（`include/ninfer/types.h:257`），尾块通常非 128 对齐 | 需要真实流量统计 |

---

## 7. 未测项与"不许写"的边界

1. **本文所有百分比都是 FLOPs 估算**，不是耗时占比。解码受权重带宽约束（`docs/performance/rtx-4090-sm89-baseline.md:110`），FLOPs 占比与耗时占比可以差一个量级：35B 路由专家 FLOPs 34.2% 但它是解码绝对主项；反之 attn core 在短上下文几乎不占时间。
2. **本文不自造任何 tok/s 或收益百分比**，只在必须区分"实测 vs 估算"时引用仓库内已有实测文档（下面两条），并给出处：
   - FP8 KV 布局：`docs/performance/rtx-4090-sm89-baseline.md:88`（`fp8` 布局 PPL `7.066428`、相对 int8 `+0.13%`、decode `47.8 tok/s`、prefill `2540 tok/s` @8192）；
   - 精度侧的一次对照：`docs/maintainer/v9-architecture-parity-register.md:118`（split 投影 int8 6.012279 / int16 6.149038 / bf16 6.122722，用于说明"更精确 ≠ PPL 更低"）。
3. **FP8-A8 GEMM 的吞吐在仓库里没有任何记录**。`linear-benchmark.md:470-471` 明确预置 suite 全用 `A16Only`；`linear-benchmark.md:496-507` 记的是 NVFP4-A4（5090）。因此本表不引用任何 FP8-A8 速度数字。
4. 仓库外工作区存在若干 FP8 实测笔记（`H:\TeleAgentWorks\fp8_stage_a_b_final.md`、`fp8_impl_review_sm89.md`、`fp8_plan_revised.md`、`w4a8_production_measurements.md`）。它们含逐形状 TFLOP/s 结论，但**不在仓库内**，按本轮口径不作为本文结论的引用来源；若后续要写收益，须先把这些测量落进仓库文档。

---

## 8. 改动边界与读取路径说明

- 新增：`docs/maintainer/fp8-coverage.md`（本文件）。
- 未改动任何源码、未改构建、未改 v10 计划文件、未执行 git 写操作。
- 读取路径提示：本轮 H 盘卷视图出现间歇性不可见（`H:\TeleAgentWorks\infer-4090-port\ninfer` 一度 `Test-Path=False`，同时 `H:\TeleAgentWorks\.temp\r` 正常）。后续读取统一走 junction `.temp\r`（指向同一仓库），文件内容不受影响。
