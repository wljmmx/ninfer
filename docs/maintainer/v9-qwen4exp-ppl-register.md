# Qwen4Exp（Qwen3.8-Flash-Next）文本路径 PPL 二分登记

日期：2026-11-12
仓库：`H:\TeleAgentWorks\ninfer-4090-port\ninfer`，HEAD `5f1c3ea6`
工件：`R:\model\qwen3_8_flash_next_mtp\qwen3_8_flash_next_mtp.ninfer`（122.87 GB，4 分片，text+mtp）
环境：RTX 4090 / sm_89 / Windows 11 / CUDA 13.5

## 1. 基线与复现命令

语料固定为 114 token 的普通英文（`H:\TeleAgentWorks\.temp\ppl_text.txt`），窗口/步长固定：

```powershell
$env:NINFER_EXPERT_CACHE_MB='14000'
build-win/apps/ninfer-perplexity.exe R:\model\qwen3_8_flash_next_mtp\qwen3_8_flash_next_mtp.ninfer `
  --text <114-token file> --context 512 --stride 256 --kv-dtype bf16
```

| 项 | 数值 |
|---|---|
| PPL | **1.48457e+06**（mean NLL 14.210635） |
| 均匀基线 | `ln(248320) = 12.4229` |
| 结论 | 比均匀分布还差 1.79 nats ⇒ logits 无信息 |

对照：同一语料、同一命令，`qwen3_6_35b_a3b`（非 HC 模型）得 **PPL 9.61965**。这条对照是本轮所有
"正常量级"判断的锚点，也说明打分链路本身在非 HC 模型上可用。

## 2. 二分表（每档均为同一命令、同一语料）

| # | 臂 | PPL | mean NLL | 判读 |
|---|---|---|---|---|
| 0 | 基线（cache 14000，bf16 KV） | 1.48457e6 | 14.210635 | 复现任务给定数值 |
| 1 | `NINFER_PLE_OFF=1` | 1.02993e6 | 13.845000 | PLE 不是主因（关掉更差） |
| 2 | `NINFER_QSA_FORCE_FULL=1`（indexed 层走 dense 参考臂） | 8.57158e5 | 13.661378 | QSA indexer 不是主因 |
| 3 | `NINFER_EXPERT_CACHE_MB=8192` | 1.44970e6 | 14.186864 | 与 #0 同量级；ids 分叉见 §6 |
| 4 | `NINFER_MOE_HOST_ONLY=1`（整层路由项走 CPU 参考） | 1.48457e6 | 14.210635 | 与 #0 **逐位相同** ⇒ 设备/主机两条 MoE 路径读到同一份（错）数据，或 MoE 与主因无关 |
| 5 | **GDN 输出门改为 sigmoid（本轮修复，见 §3）** | **1.09633e5** | **11.604896** | NLL 首次低于 `ln V`；13.5× 改善 |
| 6 | #5 + `NINFER_PLE_OFF=1` | 1.71490e5 | 12.918 | PLE 仍不是剩余主因 |
| 7 | #5 + `NINFER_QSA_FORCE_FULL=1` | 1.34612e5 | 11.792 | QSA 仍不是剩余主因 |
| 8 | #5 + 诊断补丁（打分读**正确的** 2560 宽 final hidden，见 §4） | 2.56558e5 | 12.455 | **去掉打分缺陷后仍差 4 个量级** ⇒ 模型路径还有至少一个缺陷 |
| 9 | #8 + `NINFER_PLE_OFF=1` | 4.19200e5 | 12.946 | PLE 关闭仍更差 |

逐层健康度（临时 RMS 探针，已移除；`--prompt "Hello"`，51 列 prefill chunk，cache 14000）：

| 量 | 数值 |
|---|---|
| embedding RMS（layer 0 输入） | 5.06e-3，max 4.8e-2 |
| layer 0 注意力 mix 输出（block input） | RMS 0.78，max 8.7 |
| layer 0 注意力乘积（o_proj 输出） | RMS 0.24 |
| layer 0 MoE 乘积 | RMS 0.056 |
| layer 46 residual（四流） | RMS 1.58–1.78 |
| 结论 | 无 NaN/无爆量；逐层残差单调增长到 O(1)，不存在"指数爆炸"或"塌陷" |

## 3. 根因一（已修）：linear-attention 的输出门激活被固定成 SiLU

**证据链**

- 工件配置：`text_config.output_gate_type = "sigmoid"`（`config.json` 与工件内 config 一致）。
- HF 参考 `Qwen4ExpTextRMSNormGated`（`.temp/hf_qwen4_exp.py:173,186`）：`activation =
  config.output_gate_type or config.hidden_act`，作用为 `hidden * ACT2FN[activation](gate)`。
- 上游（只读参考）`src/models/qwen4_exp/execution/gdn.cpp:91` 使用 `ops::sigmoid_gated_rmsnorm`，
  且 `src/models/qwen4_exp/config.cpp:50` 直接 `expect(c, "output_gate_type", true)`。
- 本地 `src/models/qwen3_5/execution/text.cpp` 在 36 个 GDN 层上**一律**调用
  `ops::silu_gated_rmsnorm`，`config_.output_gate_type` 只被解析、从未被消费。

**为什么错这一处就能把 PPL 打到 1e6**：SiLU 门与 sigmoid 门在 z<0 时符号相反、在 |z| 大时相差一个
|z| 因子，等于把每一层线性注意力的输出乘上一个逐元素随机增益；36 层连乘后 residual 里几乎没有
可用的 token 信息，logits 退化为噪声（mean NLL 14.2 ≈ 均匀 + 1.8），greedy 输出即词形噪声。

**修复**（`text.cpp` `gdn_mix`）：按配置名分派，`sigmoid` → `sigmoid_gated_rmsnorm`，空/`silu` →
`silu_gated_rmsnorm`（Qwen3.5 工件没有该键，行为不变），其它值按名报错。

**改前/改后（同命令）**：PPL 1.48457e6 → 1.09633e5；mean NLL 14.210635 → 11.604896。

## 4. 根因二（已证、**未修**：修复点在本轮授权面之外）：causal scoring 把 4 流 frame 当成 2560 宽 hidden

`ninfer-perplexity` 走 `ProgramImpl::causal_score`，它把 `prefill_hidden` 当作"每个位置的后置 hidden"，
整块拷进 2560 宽的 staging 再投影：

- `src/models/qwen3_5/program/planning/startup.cpp:312` 用 `frame_width` 建 `prefill_hidden`，
  而 `frame_width = hc ? stream_width : hidden`（`:139`）⇒ 本工件 **10240 × 512**（探针实测
  `[frame] prefill_hidden ne=(10240,1024,1,1) mtp_width=10240 columns=51`）。
- `startup.cpp:316` 的 `score_hidden` 是 `hidden_size × kCausalScoreTile` = **2560 × 1024**。
- `program_impl.cpp:585` 的拷贝是 `cudaMemcpyAsync(dst, src, source.bytes())`：源是 10240 宽的
  切片，`source.bytes() = 10240*count*2`，既**跨列**又**越界**写进 2560 宽的缓冲。

引入点是 `a9054935`（四流 MTP 草稿头）把 `prefill_hidden` 从 `hidden` 加宽到 `frame_width`，打分侧
没有同步：对非 HC 模型 `frame == hidden`，这条路径一直是对的（35B 同一语料 9.61965 即证）；对 HC
模型从该提交起必然读错。

**量化影响（诊断用临时补丁，已手工反向替换，`git diff` 已确认干净）**：令 prefill 把 head 的 2560 宽
sample 发布进 frame 的前 2560 lane、打分侧按 10240 pitch 取列，得到 PPL **2.56558e5**——比走错缓冲
的 1.09633e5 还差。因此这条缺陷必须修，但它**不是**把模型拉回正常量级的全部原因。

**为什么本轮没修**：修复点 `src/models/qwen3_5/program/program_impl.cpp`（以及
`program/planning/startup.cpp` 若要给打分 Program 另建 2560 宽缓冲）不在本轮授权面内（授权为
`execution/**`、`qwen4_exp/**`、`artifact/**`、`runtime/**`、`ops/linear_attention|moe/**`、
`tests/**`、`docs/**`）。执行侧 `prefill_text_chunk` 只有 `prefill_hidden` 这一个出口，且它同时承担
MTP 的 10240 宽 continuation frame，无法在 execution 内两者兼得。

## 5. 已用独立参考验证为**正确**的部分

均以"从工件权重独立复算 vs 引擎导出张量"的方式对拍（脚本在 `.temp/qwen4exp/`，不入库）：

| 子系统 | 方法 | 结果 |
|---|---|---|
| embedding 行 | q8_g32 行反量化 vs 引擎 `x` | rel **1.3e-3**（BF16 舍入级） |
| hc_expand_norm | `embed*inv*(1+w)` vs 引擎 `normalized` | rel **2.8e-3** |
| **MoE 整块**（路由 + 512 专家 + shared + 系数） | 从工件的 q4/q5/q6 专家 bank 独立复算，喂引擎自己的 block input | rel **3.3e-3**（renormalized top-10）；改为不归一化则 rel **0.371** ⇒ 现行为正确，`norm_topk_prob=true` 与工件/上游一致 |
| HC down/inject/up 权重 | 与 GGUF 源逐元素相关 | corr **0.998 / 0.997 / 0.998**，rms 一致 |
| GDN z 投影、norm、conv 权重形状 | 形状/格式/绑定核对 | 与 HF `in_proj_qkvz` 拆分一致（q2048/k2048/v6144 + z6144，conv 10240） |

**仍未解释**：以引擎自己的 `normalized` 为输入复算 `hc_mix`（sigmoid 门、`/4`、s=row/4 的行序，与
`src/ops/gated_residual/mix/kernel/up_q5_mix.cuh:44,105,120` 逐项对应）得到与引擎导出值 rel **1.7**；
同一参考下 `injection = 2σ(b/4)` 也与引擎不符（引擎 b ≈ −11.6…+2.0，复算 ≈ −30…−20）。权重已对
GGUF 验证、输入已与引擎导出逐字节相同，因此这一处要么是 `hc_mix` 的操作数在调用前被换过，要么是
本复算漏了某个引擎侧约定——**未闭环，是下一步的第一嫌疑**。

## 6. MTP / ids 相关（本轮未做验收）

- 同 prompt、同 greedy，`--spec none` 的 ids 在 cache 14000 / 8192 下仍不同（既定行为，见
  `v9-moe-execution-register.md` 附）；本轮修复后 14000 档首 24 枚为
  `2495 391 24 318 17 17 79 912 18 220 38855 96898 68 220 105381 696 78 70662 8814 206128 2469 370 6122 3510`，
  解码文本仍是词形串（"Strly9 (22pays3 qi… _o-aloneista BOL emailpesifa"），即主路径尚不可用于
  MTP 判据。
- **因此 `--spec none` vs `--spec mtp -d3` 的 ids 逐位一致 + accepted 50%+ 本轮不成立**，不做虚报。

## 7. 回归

| 项 | 结果 |
|---|---|
| `ctest -R "gdn\|qwen3_5_dense\|kv_capacity\|expert_cache\|mtp_stem"` | **17/17 通过**（dense 真机用例按设计 skip，需 `NINFER_TEST_ARTIFACT`） |
| `ctest -R "softmax_attention\|bench_support"` | **2/2 通过**（bench_support 0.06 s、softmax_attention 633.16 s） |
| 35B 同语料 PPL（对照） | **9.61965**（未变；35B 配置无 `output_gate_type`，走 silu 分支，逐位不受影响） |
| 27B `qwen3_8_27b_v3` 同语料 | 加载后未跑出分数（进程在 weights ready 后退出，未定位；与本轮改动无关，本轮未改其路径） |

## 8. 未闭环

1. **PPL 仍未回到正常量级**（1.10e5，对照 35B 9.62）。已排除：PLE、QSA indexer、MoE（设备与主机
   两路、路由系数、专家反量化）、embedding、hc_expand_norm、HC 权重身份、残差爆量/NaN。
   第一嫌疑：`hc_mix`（§5 末）与 36 层 GDN 的数值（本轮只修了它的输出门激活）。
2. **causal scoring 的 4 流 frame/2560 hidden 混用**（§4）：需要改
   `program/program_impl.cpp`（+ 可能 `program/planning/startup.cpp`），本轮授权面外。
3. **`--spec none` vs `--spec mtp` 的 ids 判据**：主路径修好前不可判定。
4. 27B 同语料跑分未完成（进程在 `weights ready` 后退出）。
