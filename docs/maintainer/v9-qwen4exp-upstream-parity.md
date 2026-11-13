# Qwen4Exp 与上游实现的逐项对拍（2026-11-13）

日期：2026-11-13
仓库：`H:\TeleAgentWorks\infer-4090-port\ninfer`（目录名 `ninfer`）
上游：`H:\TeleAgentWorks\infer-4090-port\infer-upstream`（只读参考）
工件：`R:\model\qwen3_8_flash_next_mtp\qwen3_8_flash_next_mtp.ninfer`（122.87 GB，4 分片，text+mtp）
源：`R:\model\qwen4exp\UD-Q4_K_XL\Qwen3.8-Flash-Next-UD-Q4_K_XL-0000{1..4}-of-00004.gguf`
环境：RTX 4090 / sm_89 / Windows 11 / CUDA 13.5

## 0. 结论

| 项 | 结果 |
|---|---|
| 目标 PPL（114 token 英文，ctx512/stride256，bf16 KV，cache 14000） | **256558（NLL 12.455）**，未达标（目标 ≤20） |
| 本轮修复后的最好值 | **19839（NLL 9.895）**＝+GDN 值头序补偿 |
| 同语料 35B 回归 | **9.619649**（未变） |
| 引擎算术 | 6 个抽样层与工件**逐位一致（BF16 级 2e-3…5e-3）**，不是引擎算错 |
| 已定位缺陷 | 工件的 GDN **value-head 轴**带 llama.cpp tiling，且被**叠加两次**；引擎与上游都按 HF grouped 约定配对 K 头 |
| 仍未解 | 补偿后 PPL 仍差 4 个量级；见 §5 |

一句话：**引擎按文档合同是对的，工件在 GDN 值头轴上违反该合同**；补偿能吃掉 2.5 nats，但还有第二处（不在本轮已核对的任何张量里）。

## 1. 环节对照表

上游＝`infer-upstream/src/models/qwen4_exp/**`；我们＝`infer/src/models/{qwen4_exp,qwen3_5}/**`。
"一致"指：同一份工件权重、同一条数学合同下，两者产出在 BF16 舍入级相同（§2 有逐项数字）。

| # | 环节 | 上游怎么做 | 我们怎么做 | 一致 | 证据/位置 |
|---|---|---|---|---|---|
| 1 | 配置解析 | `config.cpp:45-60` 硬校验 `output_gate_type=sigmoid`、`tie_word_embeddings=false`、`norm_topk_prob=true`、`mamba_ssm_dtype=float32`；`rotary_dim=head_dim*partial_rotary_factor`；`budget/compression` 得块数；层类型只认 `linear_attention`/`indexed_attention` | 由 `qwen3_5::parse_config` 解析同名字段，`qwen4_exp::complete` 追加 hc_count==4、FFN 必须是 MoE；gate 名解析成 `GdnGateKind`（`qwen3_5/config.cpp`，`text.cpp:2455` 分派） | 是 | sigmoid 门、tie=false、rotary 64、块预算 512 全部按同一语义生效 |
| 2 | RoPE / 位置 | HF `Qwen4ExpTextRotaryEmbedding`：`inv_freq=theta^(-2i/64)`，`recomposition_frequencies` 按 `mrope_section=[11,11,10]` 交错后 `cat(freqs,freqs)`；`apply_rotary` 是 **rotate_half 配对 (d, d+32)** | `src/ops/softmax_attention/qsa/norm_rope.cuh:59-141` 同一配对（lane±32/N 取 mate，`coefficients[(d)%32]`），`prepare_launch.cu:85-87` 频率 `theta^(-i/32)`；文本只有一条位置轴，交错是恒等 | 是 | QSA 第一层 `query` 复算 rel **2.33e-3**（含 norm+rope） |
| 3 | Embedding | `ops::embedding` 查表 → `hc_expand_norm`（四条流复制 + 逐流 offset RMSNorm） | `hc.cpp:19-36` 同一 Op，权重取 `layers[0].input_norm`（绑定到 `attention_hc/norm`，`qwen4_exp/load/text.cpp:26-33`） | 是 | 工件行反量化 vs 引擎 x：rel 1.3e-3（旧轮）；GGUF 逐行：corr **1.00000**（本轮，row 0/760） |
| 4 | 层类型判定 | `config.cpp:112-120` 由 `layer_types` 顺序建 `{kind, compact_index}` | `text.cpp:2570-2585` 用 `config_.layer_types[layer]`，QSA 用 compact 索引选 cache/state 平面 | 是 | 抽 3 个 QSA 层（3/47）与 3 个 GDN 层（0/1/46）都算对；层序对 GGUF exact（§2） |
| 5 | GDN | `execution/gdn.cpp`：`gdn_control_proj(g,beta)` → `linear_split(qkv,z)` → `causal_conv1d_silu_split`（4 tap，dilation=1）→ `gated_delta_net(scale=1/√128, normalize_qk=true)` → `sigmoid_gated_rmsnorm`（直接 gamma）→ `linear(out)` | `text.cpp:2217-2484` 同一调用链；三相位（序列 / 记录 / 快照）与上游一一对应 | 是 | L0/L1/L46：g、beta **rel 0/1e-5**，proj 9e-4…1.2e-3，conv 8e-5…2.5e-3，o 2e-3…3.9e-3，on 4e-3…5.1e-3 |
| 6 | HC（combine/injection/顺序） | `execution/text.cpp:58-85`：每块先 `hc_mix(attention)` → mixer → `hc_combine_norm(moe_hc.norm)` → `hc_mix(moe)`；`:87-104` 层末 `hc_combine_norm(next.attention_hc.norm)`，PLE 层末只做 `hc_combine` 不带 norm | `text.cpp:2570-2677` 同一顺序；`hc.cpp:38-60` 两个入口 | 是 | combine 恒等式 rel 1.1e-3…1.7e-3；`normalized`=offsetNorm(respost, 对应 norm) rel **1.66e-3**（L0/L1/L2/L3 全对） |
| 7 | PLE（取值/注入位置/行对齐） | `execution/ple.cpp`：`linear` → key(4×2560)/value(2560) 同父两段 → `ple_gate` → `ple_conv_add_norm(next.norm, gate.rms_epsilon)`；注入点在 `ple_layer_ids` 规范化后的层**入口**（`text.cpp:106-118` + `prefix/suffix`） | `ple_layer.cpp:173-241` 同一父布局（`key.nb[2]=12800*2`）、同一 Op；`text.cpp:2593-2631` 在该块 mixer 之前调用，并用该块的 `input_norm` 重算 normalized | 是 | ids→表行、embedding **rel 0**，projected 1.1e-4，gated 1.7e-3，residual-out 2.3e-3，normalized 2.5e-3；卷积布局 tap-major(4,C) |
| 8 | QSA（选块/scale/mask） | `execution/qsa.cpp`：`linear_split` → `qsa_prepare` → `qsa_indexer` → `qsa_sparse_gqa` → `linear(out)`；scale=1/√256，块=4 token，选块预算 512，尾块读可见 token | `qsa_layer.cpp:99-298` 同一顺序；`qsa.cpp` 只做几何/plan | 是 | L3/L47：prepared q 2.3e-3，gated attention 4.9e-3，o_proj 5.5e-3（L47 逐行 corr 1.0） |
| 9 | MoE（路由/行窗/共享） | `moe::execute` 归一化 top-10 softmax + shared sigmoid 门；`norm_topk_prob=true` | `mlp_tail`→banked 专家 + shared；`next_projection_hints` 预取 | 是 | 整块复算 rel 3.3e-3（旧轮）；router 对 GGUF **rel 0**，专家 0…7 rel 0.11-0.13（q4 噪声），无专家错位 |
| 10 | head（tied/顺序） | `finish()`：`sample=hc_mix(final_hc)`，无额外 final RMSNorm；head 与 embedding 未绑 | `hc.cpp:62-71` 同；`project` 直连 `lm_head_` | 是 | sample vs final_hc 1.7e-3；head 对 GGUF 为同一对象（q5），logits rms 2.40、熵 8.96 |

**代码级差异（不影响数值，记录在案）**：`qsa_layer.cpp:179` 把 indexer 头数写成字面量 4（配置里是 `indexer_n_heads`）；`ple_layer.cpp:234` 传 `next_epsilon` 给 `ple_conv_add_norm`，上游传 `gate.rms_epsilon`（本工件两者都是 1e-6）。

## 2. 数值复算（全部以工件权重独立复算，脚本在 `.temp/qwen4exp/`）

布局坑（两次踩到，记下来）：GDN/HC 的平面是**特征连续、列优先**（`[rows, cols]` 读法 `reshape(cols,rows).T`），
QSA 的投影缓冲是**token 连续**（`[T][N]`，直接 `reshape(-1, N)`），HC 的 10240 轴是 **s-major**（`reshape(S,H,T)`）。
按错布局读会把 rel 恒等的两个张量读成"完全不相关"，本轮前半天就是被这两点误导。

| 复算对象 | 方法 | 结果 |
|---|---|---|
| GDN L0 / L1 / L46 | control→proj→conv→delta rule→gated norm→out，从引擎 dump 的 h 起算 | rel 2e-3…5.1e-3，corr ≥0.99999 |
| QSA L3 / L47 | q/k/v/gate 投影 + 逐头 offset norm + rope + dense causal（T=114 全块入选）+ sigmoid 门 + o_proj | 2.3e-3 / 4.9e-3 / 5.5e-3（L47 逐行 corr 1.00000） |
| HC mix / combine / norm | 从残差 + 构件权重 | 1.7e-3 / 1.1e-3…1.7e-3 / **1.66e-3** |
| PLE | hash→表行→projection→gate→dilated conv→add→norm | 0 / 1.1e-4 / 1.7e-3 / 2.3e-3 / 2.5e-3 |
| 工件 vs GGUF 源 | 逐张量（gguf-py 反量化；Q8_0 另用手工块解码核对） | 嵌入 corr 1.00000；router rel 0；层序 L0/L1/L5/L24/L40/L47 全 rel 0；HC norm 1.8e-3；HC up/down/inject 0.064/0.064/0.079（q5 噪声）；shared 0.07、专家 0.11-0.13（q4/q5 噪声）；QSA q/gate 去交错 0.13/0.13、K 0.14、V 0.074、out 0.058 |
| 工件 vs GGUF：**唯一不一致** | GDN 值头轴 | 见 §3 |

## 3. 已定位缺陷：GDN value-head 轴被 tiling 了两次

**症状链**：36/48 层是 GDN；delta rule 里 V 头 h 用 K 头 `h/(VH/KH)`（HF grouped）。工件的 V/Z/a_log/dt_bias/out 列
却是 llama.cpp 的 tiled 顺序，所以每个 V 头都读到**别的 K 头**的键 → 线性注意力的输出方向错，量级不变。

**证据（逐头相关，工件 vs GGUF）**：`corr(art V head i, gguf V head g)=0.998`，其中 `g=(i%16)*3+i//16`（16=K 头数，3=V/K）；
同表里"同位"只有 0.00，"tiled 位"也只有 0.00 → 不是噪声，是置换。Z、out_proj 列、a_log、dt_bias 同一置换
（dt_bias/ssm_a 是 F32，逐值可对：`artifact[i] = gguf[(i%16)*3+i//16]`，bf16 舍入内相等）。

**为什么是"两次"**：llama.cpp 的 `_LinearAttentionVReorderBase._reorder_v_heads` 把 HF grouped→tiled 写进 GGUF；
本工件又是 GGUF 的 `m` 置换（`art[i]=gguf[m(i)]`，`m(t)=3*(t%16)+t/16`）。所以 `art = m∘m(HF)`，要还原 HF 必须再作用两次逆置换。

**修复**（`src/models/qwen3_5/execution/text.cpp:2042-2090`、`gdn_mix` 内两处）：
`NINFER_GDN_V_TILED=k` 时，在循环前把 v/z/g/beta 按 `(m⁻¹)^k` 聚成 grouped，循环后把 `on` 按 `m^k` 散回，
o_proj 的列序因此与工件对齐；k=0（默认）完全不变。用 `cudaMemcpy2DAsync` 逐头搬，每层 4×(48+48) 次拷贝。

| 命令（同语料/同参数） | 无补偿 | k=1 | k=2 |
|---|---|---|---|
| 114 token 英文 | PPL **256558** / NLL 12.455 | — | PPL **19839** / NLL 9.895 |
| "the "×120 | PPL 60157 / NLL 11.005 | PPL 160740 / NLL 11.988 | PPL **12256** / NLL **9.414** |

k=2 一致优于 k=1 和 k=0，与"两次 tiling"的推导吻合，这是该缺陷成立的旁证。

## 4. 回归

| 项 | 结果 |
|---|---|
| `ctest -R "gdn\|qwen3_5_dense\|kv_capacity\|expert_cache\|mtp_stem"` | **17/17 通过**（5 个 dense 真机用例按设计 skip） |
| 35B 同语料 PPL | **9.619649**（NLL 2.263808，未变；开关默认关） |
| 本目标基线复跑 | **256558.1011**（逐位复现注册表数值） |

## 5. 未闭环

1. **PPL ≤20 未达成**（补偿后 19839）。已排除引擎算术：§2 的六层 + 嵌入 + head + MoE/HC/PLE 全部在 BF16 级对齐工件，
   而工件除 GDN 值头轴外逐张量与 GGUF 一致（含层序与专家身份）。剩下的一处因此只可能在"工件语义 vs HF 语义"这一层，
   即转换链路，`tools/convert`（他人未提交）不在本轮授权面内。
2. **残差里的 token 身份逐层流失**：`cos(residual_t, embedding(ids_t))` = 0.91(L0) → 0.56(PLE 入口) → 0.41(PLE 出口)
   → 0.30(L3) → 0.03(L47，与打乱后 0.028 无差别)。`the`×120 的 NLL 9.41（正确模型应 ~0.1）说明模型基本不会"复读"，
   不是标定/温度问题（温度扫描只会让 NLL 更差，top-1 恒为 1.8%）。
3. **下一步建议**（按优先级）：
   - 转换侧：确认 `gguf_qwen3_5.py` 的 `tiled_to_grouped` 方向。本文件的 GGUF 已经是 tiled（llama.cpp 写入），
     再作用一次会把 grouped 的意图变成 double-tiled；正确做法是对该文件**不作用**该置换，或改为作用逆置换一次。
     复现判据：`artifact[i]==gguf[m(i)]`（m 见 §3）即说明方向反了。
   - 转换侧：`ssm_out` 列、conv 的 V 通道必须与 V 行用**同一个**置换；本工件三者一致（都 double-tiled），
     所以引擎侧补偿只需 4 处。
   - 若重新转换后 PPL 仍不达标：从 PLE 入手（它在入口就把身份从 0.91 砍到 0.41，且注增量 0.0108 与残差 0.0154 同量级，
     是唯一"注入量≈残差量"的阶段），比对 HF `_short_conv` 的 dilation=3 九列历史与 `apply_mask_to_padding_states`。
4. **本轮未做**：`--spec mtp` 判据（主路径未达标前不可判定）、27B 同语料跑分。

## 6. 本轮改动清单

| 文件 | 改动 |
|---|---|
| `src/models/qwen3_5/execution/text.cpp` | `gdn_v_tiling()/gather_heads()` 与 `gdn_mix` 内两处值头序补偿（默认关）；`residual_dump()` 及其调用（`NINFER_RESIDUAL_DUMP`，带层过滤、块输入/归一化/sample 平面） |
| `src/models/qwen4_exp/execution/ple_layer.cpp` | `NINFER_PLE_DUMP` 平面导出（上一轮遗留，本轮用于 PLE 复算） |
| `docs/maintainer/v9-qwen4exp-upstream-parity.md` | 本文件 |

诊断开关：`NINFER_GDN_V_TILED=k`（值头序补偿）、`NINFER_RESIDUAL_DUMP[_LAYERS|_ONLY]`、`NINFER_PLE_DUMP`、
`NINFER_GDN_DUMP[_LAYERS]`、`NINFER_QSA_LAYER_DUMP/NINFER_QSA_DUMP_LAYER`。全部默认关闭，不影响 35B/27B 路径。
