# Weight conversion

NInfer's converter creates `.ninfer` artifacts from local weights and a Python recipe. A recipe
can reuse an official conversion, change selected layers or projections, combine sources, or call
your own conversion method. The artifact contains the resulting configuration, encoded weights,
logical bindings and frontend resources.

Run the commands below from the repository root. The converter needs Python 3.11+ with PyTorch;
a `.gguf` source additionally uses the reader in `tools/convert/sources/gguf.py`, which needs
NumPy only.

## Convert a GGUF source (Qwen4Exp)

When only a GGUF distribution exists, the same converter reads it directly:

```bash
python -m tools.convert \
  --model /path/to/model.gguf \
  --recipe qwen4_exp \
  --components text \
  --device cuda \
  --gguf-config /path/to/config.json \
  --name Qwen4Exp \
  --out models/qwen4_exp.ninfer
```

What the GGUF supplies, and what it does not:

| Item | Source |
|---|---|
| Weights | The GGUF tensors, dequantized to float and re-quantized by the recipe. Readable types: F32, F16, BF16, F64, I8/16/32/64, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0 and the K-quants Q2_K/Q3_K/Q4_K/Q5_K/Q6_K. Any other type is refused by name. |
| Geometry | GGUF metadata (`qwen4exp.*`: block/embedding/head counts, expert sizes, `ssm.*` for the linear-attention geometry, `hyper_connection.*`, `attention.indexer.*`, `ple.*`, `attention.compress_ratios`, `rope.freq_base`). |
| `rope_parameters`, `indexer_compress_ratio`, vision/MTP subtrees | Not in the GGUF. Pass `--gguf-config` with the checkpoint's `config.json`; the sidecar wins over the synthesized values. |
| Tokenizer / chat template | Not in the GGUF. Put `tokenizer.json`, `tokenizer_config.json` and `generation_config.json` beside the file (or use `--resource`, and the maintained template is installed automatically). `--gguf-root` selects the directory holding them. |

Scope and mapping notes:

* The text component, plus the draft head when `--mtp-gguf` names it (below). Vision ships as a
  separate mmproj GGUF; `--components` other than `text`/`mtp` is refused for a `.gguf` input.
* GGUF uses llama.cpp's `blk.N.*` names; the inverse of llama.cpp's QWEN4EXP tensor map is
  applied in `tools/convert/gguf_qwen4exp.py`. Tensors llama.cpp fuses for one matmul (expert
  gate|up, the indexer q|k projection, the linear-attention qkv|z pair, beta|alpha, the
  hyper-connection down|inject pair) are split back by row range; tensors it stores without the
  `.weight` suffix (`ssm_a`, `ssm_dt.bias`) and the 1-D shared-expert router bias are handled too.
* **Split files are one source.** Passing the first part of `name-0000N-of-0000M.gguf` opens every
  part: the metadata comes from the part that carries `general.architecture` (in practice the
  first, which may hold no tensors at all) and the tensor directories are merged, so a 100 GiB
  model reads as one file. A missing part is refused by name.
* **A sidecar `config.json` wins over the GGUF-derived fields** (`--gguf-config`). The flattened
  GGUF metadata cannot express the rope subtree, `indexer_compress_ratio`, `indexer_kv_heads`,
  `ple_embed_dim` or `split_ngram_parts` faithfully; those come from the sidecar when given, and
  otherwise are derived from tensor shapes (indexer key-projection rows, `ple_key` width) or
  refused.
* Linear attention stores V heads in llama.cpp's *tiled* order; the converter restores the HF
  *grouped* order (`tiled_to_grouped`) for every tensor that carries V heads.
* The PLE hash controls (`ple.layer_multipliers`, `ple.head_offsets`, `ple.head_vocab_sizes`) are
  GGUF metadata arrays; they are written to the artifact as exact int64 bytes.
* The PLE table arrives as the single `per_layer_token_embd` tensor (llama.cpp merged the
  shards), so `split_ngram_parts` must divide its row count; the value comes from the sidecar
  config. The `qwen4_exp` recipe stores that table as Q4: it is a lookup, the source carries it at
  about 4.5 bits already, and bf16 would make it 102 GiB for a 320M-row table.
* A GGUF has already lost the original float precision. Re-quantizing a Q4_K source stores the
  GGUF's approximation, not the checkpoint's — this is a conversion of what the file contains.

Verification: `tests/convert/test_gguf_qwen4exp.py` checks the container and every dequantizer
against gguf-py's own writer/dequantizers on random blocks, asserts the Qwen4Exp name/shape
mapping, and runs the real converter on a synthetic Qwen4Exp GGUF, comparing the artifact's PLE
controls byte for byte and a bf16 tensor word for word.

## Convert a Qwen4Exp draft head (`--mtp-gguf`)

llama.cpp writes the next-token-prediction head of a Qwen4Exp checkpoint as one extra block: the
GGUF's `block_count` counts the draft block, and `blk.<block_count-1>` holds `nextn.*` plus a copy
of `token_embd` and `output`. Such a file is **not** a text checkpoint. Passing it as `--model` is
refused by name; it is converted as the `mtp` component of the text artifact:

```bash
python -m tools.convert \
  --model /path/to/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf \
  --mtp-gguf /path/to/mtp-Qwen3.8-Flash-Next-Q4_K_M.gguf \
  --recipe qwen4_exp \
  --components text,mtp \
  --gguf-config /path/to/config.json --gguf-root /path/to/hf \
  --device cpu \
  --name Qwen3.8-Flash-Next \
  --out models/qwen3_8_flash_next.ninfer
```

Naming `--mtp-gguf` is what produces the component (it adds `mtp` to `--components` when it is not
given), and the text side is converted exactly as it would be without it. `GGUFSource.for_mtp`
presents the draft block as the HF `mtp.*` names the builder reads and derives its geometry from the
target's `text_config` (the MTP GGUF's own `block_count` is never read as a text layer count).

Mapping notes specific to the draft head:

* **`eh_proj` is split at the artifact, fused at the source.** llama.cpp stores the HF pair
  `fc_embedding`/`fc_hidden` as one `nextn.eh_proj.weight [h,2h]`; the converter splits it by
  column into `mtp/fc_embedding` (`[0,h)`) and `mtp/fc_hidden` (`[h,2h)`), which is the HF
  checkpoint's own form and the only one with a registered kernel geometry at `h = 2560` (the
  linear shape registry has `[2560,2560]` but no `[2560,5120]`). Both are stored bf16, as the
  source keeps them; the block's other projections follow the text recipe.
* **The gate is `attention/output_gate`.** Q and the output gate are llama.cpp's interleaved
  `blk.N.attn_q` parent; the builder splits them by head-interleaved rows into
  `attention/query` + `attention/output_gate`, and the loader binds the latter under the engine's
  `attention/gate` name for an indexed layer.
* **The shared-expert router is a vector** (`ffn_gate_inp_shexp`, `[h]`); it is reshaped to the
  `[1,h]` matrix `moe/shared_score` the routing Op reads.
* **`token_embd` and `output` are dropped.** The engine reuses the text component's embedding and
  LM head for the draft head (`src/models/qwen3_5/load/mtp.cpp`), so the component stores neither;
  keeping them would duplicate about 1.3 GB of identical BF16 words at this vocabulary.
* **Widths follow the target.** `hidden_norm` is `[h*hc_count]` (10240 here): it normalizes the
  target's four-stream residual along the whole width. The head's final correction
  (`nextn.hc_head_{norm,down,up}`) becomes `mtp/final_hc/*`, and `mtp/final_norm` aliases its
  norm.
* **Capability downgrade — the draft layer runs dense attention.** The source draft layer is an
  indexed (QSA) block (`compress_ratios[block] = 4`). The converter and loader carry its indexer
  matrices and the output-gate alias, but the runtime executes the draft mixer as **dense full
  attention**: QSA block selection (*MTD*) is not wired on this layer. The draft side therefore
  costs a full-history attention read at every draft step; it is exact, not an approximation of the
  sparse path, and it is the one declared behavioural difference from the source.

Two shapes of the same component exist, and the artifact states which one it is:

| Form | Target | Draft stem | Draft block | Head |
|---|---|---|---|---|
| Qwen3.5 single stream | one residual stream of width `h` | `e = RMSNorm(embed)`, `h = RMSNorm(hidden)`, `x = fc_hidden(h) + fc_embedding(e)` | ordinary block, `mtp/final_norm` is `[h]` | plain final RMSNorm |
| Qwen4Exp four streams (HC) | four residual streams of width `4h`; the draft head conditions on the **pre-final-mixer residual** | `v = offsetRMSNorm_10240(hidden)`, `h_init[s] = fc_hidden(v[s]) + fc_embedding(e)` | HC block: `hc_mix` → attention → `hc_combine_norm` → `hc_mix` → MoE → `hc_combine_norm` | `hc_head_norm` then the final HC mixer (`down`/`up`), no extra RMSNorm |

The HC recipe keeps the mixers at Q5 (`q5_g64_fp16`), because the separate-form HC mix Op admits
only Q5 row-split planes; every other draft projection follows the text rules. An unassigned draft
projection would be stored at a width the runtime refuses by name, so `official_recipes.py`
treats `mtp/` exactly like `text/` (except the two bf16 stem matrices, which stay bf16).

Running it: `--spec mtp --draft-tokens N` loads the component; without `--spec mtp` the component
is simply not materialized. `load/mtp.cpp` requires the component's `architectures` to match the
target's mathematics (`Qwen3_5MTP`, `Qwen3_5MoeMTP` or `Qwen4ExpMTP`).

Two boundaries follow from the runtime, and both are properties of the target it was built for
rather than of the component's tensors:

* **The replay fold is a registry of compile-time geometries, not a formula.** A speculative round
  commits the accepted prefix through `gdn_replay_fold`, and that kernel is instantiated once per
  `(linear layers, q/k heads, v heads, conv channels)` tuple (`FoldGeometry<...>` in
  `ops/linear_attention/gated_delta_net/recurrent.cuh`): the four values are template constants that
  the kernel folds into its strides and bounds, so they cannot be passed as run-time parameters.
  The same tuple is listed twice and the two lists must agree — `is_registered_fold_geometry` in
  `replay.cpp` is the start-up check and the dispatch in `recurrent.cu` is the launch, which throws
  `unregistered geometry` if it is reached with a tuple the check let through. Only the tuples the
  build instantiates are usable (`(48,16,48,10240)`, `(30,16,32,8192)`, `(36,8,24,5120)` today).
  A hybrid target whose linear layers do not match one of them, e.g. 36 linear layers that keep the
  16/48/10240 head geometry, is refused at start-up (`gdn_replay_fold: unsupported record
  geometry`) with `--spec mtp`, while `--spec none` is unaffected because it never builds replay
  records. Registering another shape is a kernel instantiation plus a dispatch branch, i.e. a build
  change, and the conv channels are `2·key_width + value_width`, not `2·hidden`, so the same head
  geometry at a narrower hidden width still lands on a different tuple.
* **The draft layer runs dense full attention.** The source draft block is an indexed (QSA) block
  (`compress_ratios[block] = 4`); the converter and the loader carry its indexer matrices and the
  output-gate alias, but the runtime executes the mixer as dense full attention because QSA block
  selection (*MTD*) is not wired on this layer. It is exact rather than an approximation of the
  sparse path, and it costs a full-history attention read at every draft step. This is the declared
  behavioural difference from the source.

## Upgrade an existing v2 artifact

The offline upgrade tool supports the official Qwen3.6/3.8-27B groupwise-int and NVFP4 artifacts,
and Qwen3.6-35B-A3B groupwise-int. Update your checkout to the current `master` and
[rebuild NInfer](../README.md#quick-start), then run with Python 3.11:

```bash
python3 tools/upgrade_ninfer_v2_to_v3.py \
  models/qwen3_8_27b_nvfp4.ninfer \
  models/qwen3_8_27b_nvfp4.v3.ninfer
```

The output must use a new path. After upgrading, use it directly or rename it to replace the
original file. Stored weight values and formats are preserved. The upgrade also installs the
matching template from `tools/chat_templates/`. Published SHA-256 checksums apply only to
downloaded files.

## Start with an official recipe

Source-weight conversion requires a Python 3.11 environment with PyTorch and NumPy. It uses CUDA
by default; `--device cpu` selects CPU conversion. The input paths below are placeholders for your
local checkpoint directories.

For Qwen3.6-27B floating-point source weights:

```bash
python3 -m tools.convert \
  --model /path/to/Qwen3.6-27B \
  --recipe qwen3_6_27b \
  --components text,vision,mtp \
  --resource chat_template.jinja=tools/chat_templates/qwen3_6.jinja \
  --proposal \
  --name qwen3.6-27b \
  --out models/qwen3_6_27b.ninfer
```

`--components` defaults to `text`. Include only the optional components you want to distribute.
`--proposal` adds the indexed proposal head used by speculative decoding; it uses the repository's
token ranking and defaults to 131,072 rows. The ordinary full-vocabulary output head is retained.

The built-in recipes are ordinary Python functions in
[`official_recipes.py`](../tools/convert/official_recipes.py):

| Recipe | Main representation choices | Additional source |
|---|---|---|
| `qwen3_6_27b` | Q4/Q5 projections, Q6 vocabulary weights | None |
| `qwen3_8_27b` | Q4/Q5 projections, Q8 vocabulary weights | None |
| `qwen3_6_35b_a3b` | Q4 experts, Q5/Q6 expert down, Q8 shared/projection weights | None |
| `qwen4_exp` | Qwen4Exp text: Q4 routed experts/attention V-head projections, Q5 projections, Q8 vocabulary | None |
| `qwen3_6_27b_nvfp4` | Imported NVFP4, selected BF16 projections, Q8 vocabulary weights | `quantized` |
| `qwen3_8_27b_nvfp4` | Imported NVFP4/FP8, FP8 embedding generated from BF16 | `quantized` |

These names select conversion choices. Runtime execution is selected from the architecture,
configuration and actual bindings stored in the artifact. `--name` sets the public model name;
it does not select kernels.

For a Qwen3.8-27B NVFP4/FP8 artifact with DFlash2:

```bash
python3 -m tools.convert \
  --model /path/to/Qwen3.8-27B \
  --recipe qwen3_8_27b_nvfp4 \
  --source quantized=/path/to/Qwen3.8-27B-NVFP4 \
  --source dflash2=/path/to/Qwen3.8-27B-DFlash2 \
  --components text,vision,mtp,dflash2 \
  --resource chat_template.jinja=tools/chat_templates/qwen3_8.jinja \
  --proposal \
  --name qwen3.8-27b \
  --out models/qwen3_8_27b_nvfp4.ninfer
```

MTP and Vision use the main source. DFlash and DFlash2 use the corresponding named source, supplied
as `--source dflash=PATH` or `--source dflash2=PATH`. An artifact may contain several optional
components; the Engine loads only the ones selected at startup, including at most one speculative
backend. Component availability and startup selection are independent.

## Change part of a recipe

Save the following as `my_recipe.py`:

```python
from tools.convert.official_recipes import qwen3_6_27b


def configure(model, recipe, sources):
    qwen3_6_27b(model, recipe, sources)
    recipe.assign(
        "text/layers/0/mlp/down",
        format="q6_g64_fp16",
        method="grouped_absmax",
    )
```

Run it with the same source and component selection:

```bash
python3 -m tools.convert \
  --model /path/to/Qwen3.6-27B \
  --recipe my_recipe.py \
  --components text,vision,mtp \
  --proposal \
  --out models/my_qwen.ninfer
```

The default entry function is `configure`; `--recipe my_recipe.py:customize` selects another
function. Alternatively, use an official `--recipe` and put only the changes in an `--override`
file. Overrides run after the base recipe and optional proposal-head setup.

`model.parameters` maps logical names to their shape, source and mathematical inputs. To see the
available names for your selected components, a recipe can print them:

```python
for name, parameter in model.parameters.items():
    print(name, parameter.shape, parameter.inputs)
```

`recipe.assign` accepts one name, a list of names, or shell-style patterns such as
`text/layers/*/mlp/down`. A selector that matches nothing fails. Assignments run in Python order;
later assignments replace only the explicitly supplied choices. Changing `format` does not
automatically change `method`. Use `layout="auto"` to select the registered layout for the format
when overriding an earlier explicit layout choice.

The principal choices are:

| Argument | Meaning |
|---|---|
| `format` | Persistent numeric format |
| `layout` | Physical encoding; inferred from format unless explicitly set |
| `method` | Built-in method name or Python callable |
| `source` | Logical values or encoded rows from the selected source |
| `parameters` | JSON-serializable numerical parameters passed to the method |
| `rows=(begin, end)` | Override complete leading-axis rows in a half-open range |
| `activation_policy` | Permission for activation precision at the parameter's mathematical inputs |

For example, `rows=(0, 128)` can give the first 128 rows a different format. This creates multiple
physical parts when necessary. The container can represent that result; the intended Op must also
support consuming those parts. Current native projections generally require a contiguous parent
region, so arbitrary splits of one projection are not automatically executable.

## Formats, methods and activation precision

The converter currently writes these formats:

| Format | Built-in method for floating-point input | Import of already encoded input |
|---|---|---|
| `bf16`, `fp32`, `int32` | `cast_direct` | Direct words through the source reader |
| `q4_g64_fp16`, `q5_g64_fp16`, `q6_g64_fp16`, `q8_g32_fp16` | `grouped_absmax` | Supply a custom method/source if needed |
| `fp8_e4m3fn_row_bf16` | `fp8_row_maxabs` | `import_encoded` |
| `nvfp4` | Supply a custom quantizer | `import_encoded` |

`grouped_absmax` stores one FP16 scale per group and signed integer codes. `fp8_row_maxabs` first
rounds input values to BF16, then produces E4M3FN codes and one BF16 multiplier per row.
`import_encoded` preserves compatible code and scale words, including NVFP4's matrix weight divisor.
It does not dequantize and requantize them.

The exact numeric and packing rules are in [numeric formats](maintainer/tensor-formats.md) and
[storage layouts](maintainer/storage-layouts.md). Source format names alone do not establish
compatibility: scale direction, granularity, code meaning and axis order must also match.

Activation permissions are independent of the stored weight format:

| Policy | Permitted activation paths |
|---|---|
| `A16Only` | A16 |
| `AllowA8` | A16, A8 |
| `AllowA4` | A16, A8, A4 |

They permit choices; they do not force a kernel to use the lowest precision. A fused operation that
shares one activation across several projections must respect the intersection of their
permissions. `recipe.use(parameter, input_name, ...)` can set one mathematical input independently;
the names are available in `parameter.inputs`.

An NVFP4 A4 input requires a positive finite activation divisor. `import_encoded` obtains it from
the selected source, or a recipe supplies it through
`recipe.use(..., auxiliaries={"activation_input_divisor": value})`. Shared weights retain separate
Use records; sharing weights does not share calibration implicitly.

## Fused parents and logical projections

The Qwen adapter exposes Q, K, gate and V separately, even when they came from fused source tensors.
It also supplies finite packing groups for attention, GDN, MLP and MoE. Compatible selections are
packed into a shared parent automatically by the built-in methods.

For the Dense groupwise recipe, attention Q/K form one Q4 parent and gate/V form one Q5 parent.
The native fused Op receives two weights. To use the supported single-parent FP8 form, an override
can assign all four projections together:

```python
def configure(model, recipe, sources):
    for layer, kind in enumerate(model.config["layer_types"]):
        if kind != "full_attention":
            continue
        prefix = f"text/layers/{layer}/attention/"
        recipe.assign(
            [prefix + role for role in ("query", "key", "gate", "value")],
            format="fp8_e4m3fn_row_bf16",
            layout="auto",
            method="fp8_row_maxabs",
            activation_policy="AllowA8",
        )
```

Use this file as `--override` after `qwen3_6_27b` or `qwen3_8_27b`. It leaves the other parameters
under that recipe. The adapter handles source Q/gate row order; the override works with logical
projections. Changing their representation can change numerical results and the physical kernels.

For explicit organization, `recipe.group([names...])` concatenates compatible selections in the
given order, `recipe.separate(names)` disables automatic grouping for those parameters, and
`recipe.share(parameter, target)` binds equal-shaped parameters to the same physical data.
Explicit groups must be disjoint and use unsplit selections with matching format, layout, method
and method parameters. NVFP4 parents also require a common weight divisor. Automatic grouping is
limited to built-in methods; custom methods can request explicit groups.

Grouping chooses storage. Model execution code chooses the supported fused implementation. The
loader uploads the stored representation, without repacking an inconvenient arrangement.

## Read another source

`--model` supplies the main config, default resources and the source named `base`. Add other
Safetensors sources with repeated `--source NAME=PATH`; they are opened when used. Single-file
Safetensors and indexed shards are supported. Additional tensor-only sources can omit model config;
sources carrying config are checked against the relevant model geometry.

To replace a logical parameter from another compatible checkpoint in a recipe:

```python
name = "text/layers/0/mlp/down"
recipe.assign(
    name,
    source=model.source(name, sources["alternate"]),
    format="q6_g64_fp16",
    method="grouped_absmax",
)
```

Supply `--source alternate=/path/to/alternate-checkpoint`. `model.source` applies the architecture's
source-name and axis mapping, including Q/gate extraction. The built-in compressed-tensors reader
understands the implemented per-row FP8 and NVFP4 code/scale conventions. It can expose decoded
values for another quantizer or encoded rows for exact import.

For another file format or quantization convention, provide a `LogicalSource`. Its value reader
accepts flat C-order element bounds and returns exactly that range. For example, a recipe can read
a logical matrix from a NumPy file stored beside the recipe:

```python
from pathlib import Path
import numpy as np
import torch
from tools.convert.sources.logical import LogicalSource


def configure(model, recipe, sources):
    name = "text/layers/0/mlp/down"
    path = Path(__file__).with_name("mlp-down.npy")
    data = np.load(path, mmap_mode="r")
    if tuple(data.shape) != model.parameters[name].shape or not data.flags.c_contiguous:
        raise ValueError("mlp-down.npy must have the logical shape and C-order storage")

    def read_values(begin, end):
        values = data.reshape(-1)[begin:end].astype(np.float32, copy=True)
        return torch.from_numpy(values)

    source = LogicalSource(tuple(data.shape), str(path), read_values)
    recipe.assign(name, source=source, format="q6_g64_fp16", method="grouped_absmax")
```

Use this as an override. The NumPy file must already follow the logical row/column order. For an
unfamiliar quantized source, its reader performs the corresponding decoding before returning
values. To preserve existing compatible encoded words, also provide `read_encoded` returning
`EncodedRows`, and the format's required divisor accessors. Their definitions are in
[`sources/logical.py`](../tools/convert/sources/logical.py).

## Write a conversion method

A method receives a `PrepareRequest` and returns `request.job(produce=...)`. Preparation validates
the target and determines auxiliary values. The `produce` function reads bounded source regions
and writes values or codes/scales through `TensorOutput`; the writer owns placement and file I/O.

This example adds explicit clipping before the existing grouped quantizer. It demonstrates the
method interface; the clipping threshold is a numerical choice made by the recipe author.

```python
import math
import torch
from tools.artifact.formats import QuantFormat, get_format
from tools.convert.quantization.groupwise import quantize_matrix


def clipped_grouped(request):
    if len(request.target.shape) != 2 or not isinstance(
        get_format(request.target.format), QuantFormat
    ):
        raise ValueError("clipped_grouped requires a grouped-integer matrix")
    limit = float(request.parameters["clip"])
    if not math.isfinite(limit) or limit <= 0 or request.rows_per_chunk <= 0:
        raise ValueError("clip and rows_per_chunk must be positive")
    n, k = request.target.shape

    def produce(output):
        for begin in range(0, n, request.rows_per_chunk):
            end = min(n, begin + request.rows_per_chunk)
            values = request.values(begin * k, end * k).reshape(end - begin, k)
            if not bool(torch.isfinite(values).all()):
                raise ValueError("source contains non-finite values")
            encoded = quantize_matrix(
                values.clamp(-limit, limit),
                request.target.format,
                device=request.device,
            )
            output.write_codes(begin, encoded.codes, encoded.scales)

    return request.job(produce=produce)


def configure(model, recipe, sources):
    recipe.assign(
        "text/layers/0/mlp/down",
        format="q6_g64_fp16",
        method=clipped_grouped,
        parameters={"clip": 1.0},
    )
```

Use this as an override. `request.values` traverses the prepared logical inputs in parent order,
including explicit groups. `output.write_codes` performs the registered packing and validates
codes/scales; the method should not duplicate that byte-layout logic. Direct output uses
`output.write_values`. Keep source blocks and temporary device tensors bounded to the method's
working set. `--rows-per-chunk` defaults to 512; custom methods own how they use it.

## Resources, files and inspection

Text includes `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja` and
`generation_config.json`. Vision adds its image and video processor configs. Resources come from
`--model`; `--resource ROLE=PATH` replaces a selected resource:

```text
--resource chat_template.jinja=/path/to/chat_template.jinja
```

The official conversion examples select these maintained templates:

| Model | Template | Defaults |
|---|---|---|
| Qwen3.6 Dense/MoE | [qwen3_6.jinja](../tools/chat_templates/qwen3_6.jinja) | thinking on; closed-turn reasoning omitted |
| Qwen3.8 | [qwen3_8.jinja](../tools/chat_templates/qwen3_8.jinja) | thinking on; effort `xhigh`; closed-turn reasoning retained |

Use your own Jinja file to change the artifact's default template. A startup
[`--chat-template FILE`](cli.md#text-input) overrides the stored template.
`generation_config.json` is preserved; sampling presets remain determined by the architecture
and explicit application/request settings.

The default maximum file size is 32,000,000,000 bytes, including framing. Smaller artifacts remain
one file. Larger artifacts use an entry such as `models/my_qwen.ninfer` plus
`my_qwen.ninfer.part-0001`, `my_qwen.ninfer.part-0002`, and so on in the same directory. Pass only the
entry path to NInfer and keep all its recorded parts together. `--max-file-bytes` changes the limit.

Conversion writes `models/my_qwen.ninfer.conversion.json` alongside the artifact, recording sources,
methods, formats, component configs, files and timing. Existing output files are not overwritten.
The report is useful for reproducing a recipe; the Engine reads the artifact itself.

```bash
python3 -m tools.artifact.inspect models/my_qwen.ninfer --objects --bindings
python3 -m tools.artifact.inspect models/my_qwen.ninfer --json
```

Inspection reads directory facts without running inference. Conversion rejects missing logical
coverage, invalid source geometry, unsupported encodings and invalid method output. Actual Op
support is checked by consumers during preparation, resource queries, warmup or execution. A
valid file may need additional Op support before its chosen combination can run. Exercise the
phases and optional components you intend to use through the normal [CLI](cli.md) or
[serving](serving.md) route.
