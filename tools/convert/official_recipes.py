"""Official representation recipes built from the same public conversion functions."""

from __future__ import annotations

from .methods import cast_direct, fp8_row_maxabs, grouped_absmax, import_encoded

Q4 = "q4_g64_fp16"
Q5 = "q5_g64_fp16"
Q6 = "q6_g64_fp16"
Q8 = "q8_g32_fp16"
FP8 = "fp8_e4m3fn_row_bf16"


def _assign(recipe, name, format, *, source=None):
    method = grouped_absmax if format in (Q4, Q5, Q6, Q8) else cast_direct
    recipe.assign(name, format=format, method=method, source=source)


def _optional(model, recipe):
    for name, parameter in model.parameters.items():
        if not parameter.projection:
            continue
        if name.startswith("vision/"):
            if name == "vision/patch_embedding":
                format = Q6
            elif name.startswith("vision/merger/"):
                format = Q8
            elif name.endswith(
                ("/attention/query", "/attention/key", "/attention/value", "/mlp/fc1")
            ):
                format = Q4
            else:
                format = Q5
            _assign(recipe, name, format)
        elif name.startswith(("mtp/", "dflash/", "dflash2/")):
            if name in ("mtp/fc_embedding", "mtp/fc_hidden"):
                # The draft stem's two [h,h] matrices stay bf16, as the checkpoint keeps them: the
                # linear shape registry has [2560,2560] in bf16 only, and no [2560,5120] at all.
                continue
            if name.endswith(
                (
                    "/moe/router",
                    "/moe/shared_score",
                    "/attention_conv/kernel_projection",
                    "/mlp_conv/kernel_projection",
                    "/candidate_selector/hidden_projection",
                )
            ):
                continue
            _assign(recipe, name, Q8)
    for backend in ("dflash", "dflash2"):
        if backend not in model.components:
            continue
        layers = model.components[backend]["config"]["num_hidden_layers"]
        for layer in range(layers):
            prefix = f"{backend}/layers/{layer}/attention/"
            for role in ("key", "value"):
                recipe.share(prefix + "context_" + role, prefix + role)


def _dense_groupwise(model, recipe, vocabulary):
    if "num_experts" in model.config:
        raise ValueError("this official recipe requires Qwen3.5 Dense mathematics")
    _optional(model, recipe)
    _assign(recipe, "text/token_embedding", vocabulary)
    _assign(recipe, "text/output_head", vocabulary)
    for name, parameter in model.parameters.items():
        if not name.startswith("text/layers/") or not parameter.projection:
            continue
        if name.endswith(("/gdn/a_projection", "/gdn/b_projection")):
            recipe.separate(name)
            continue
        if name.endswith(
            (
                "/attention/query",
                "/attention/key",
                "/gdn/query",
                "/gdn/key",
                "/mlp/gate",
                "/mlp/up",
            )
        ):
            format = Q4
        else:
            format = Q5
        _assign(recipe, name, format)


def qwen3_6_27b(model, recipe, sources):
    _dense_groupwise(model, recipe, Q6)


def qwen3_8_27b(model, recipe, sources):
    _dense_groupwise(model, recipe, Q8)


def qwen3_8_27b_gguf(model, recipe, sources):
    """Qwen3.8-27B converted from a GGUF: follow the source's widths inside each role's formats.

    A GGUF arrives already tuned: an imatrix pass decided per tensor how many bits it gets. What
    the artifact can store is narrower than that, because each consumer admits its own formats
    (`ops/weight_input.cpp`, `linear_swiglu`, `linear_add`, `embedding`):

    | role | admitted | consequence |
    |---|---|---|
    | attention / GDN input projections | paired Q4 + Q5, or a single Q8/FP8/BF16 parent | Q and K are Q4, the gate and V are Q5 - fixed by the paired native form |
    | MLP gate/up (`linear_swiglu`) | Q4, Q8 (at these shapes: also NVFP4/FP8) | Q4; Q8 would add about 3 GiB |
    | MLP down, attention/GDN output (`linear_add`) | Q4 only at 5120x6144, Q5/Q8 otherwise | Q5: the down matrix is 5120x17408, which the Q4 kernel does not register |
    | vocabulary embedding (gather) | Q6, Q8, BF16 | Q6 at minimum: the gather has no Q4 path |
    | output head (`linear`, and the draft's `linear_topk`) | Q4 only for the 131072-row proposal head; Q8/FP8 for the full vocabulary | Q8: `--spec dflash2` reduces the full head |

    Only the vocabulary embedding can follow the source here: every other matrix is pinned by a
    kernel whitelist. The artifact therefore lands slightly below the one it replaces instead of
    above it.
    """
    if "num_experts" in model.config:
        raise ValueError("this official recipe requires Qwen3.5 Dense mathematics")
    qwen3_8_27b(model, recipe, sources)
    base = sources["base"]
    if not hasattr(base, "code_format"):
        raise ValueError(
            "qwen3_8_27b_gguf follows the source's code widths; it needs a GGUF source"
        )
    by_format = {"q4": Q4, "q5": Q5, "q6": Q6, "q8": Q8}

    def width(name):
        source_name = model.parameters[name].source_name
        if source_name is None:
            return None
        try:
            return base.code_format(source_name)
        except ValueError:
            return None

    def snap(name, admitted):
        """Smallest admitted width at or above the source's; `None` keeps the recipe's choice."""
        source = width(name)
        if source is None:
            return None
        for candidate in admitted:
            if by_format[candidate] >= by_format[source]:
                return candidate
        return admitted[-1]

    pairs_q4 = ("/attention/query", "/attention/key", "/gdn/query", "/gdn/key")
    pairs_q5 = ("/attention/gate", "/attention/value", "/gdn/value", "/gdn/z")
    gate_up = ("/mlp/gate", "/mlp/up")
    for name, parameter in model.parameters.items():
        if not name.startswith("text/"):
            continue
        if name.endswith(pairs_q4):
            _assign(recipe, name, Q4)
        elif name.endswith(pairs_q5):
            _assign(recipe, name, Q5)
        elif name.endswith(gate_up):
            # The SwiGLU parent carries one format for both halves, so it takes the widest width
            # either half stores; Q5/Q6 are admitted by the folded gate/up kernels.
            layer = "/".join(name.split("/")[:-1])
            widths = [width(f"{layer}/gate"), width(f"{layer}/up")]
            widths = [value for value in widths if value]
            chosen = max(widths, key=by_format.get) if widths else None
            if chosen is not None:
                _assign(recipe, name, by_format[chosen])
        elif name == "text/token_embedding":
            chosen = snap(name, ("q6", "q8"))
            if chosen is not None:
                _assign(recipe, name, by_format[chosen])
        elif name == "text/output_head":
            # The draft path reduces the full head through linear_topk, which registers Q8 (and
            # FP8) for all 248320 rows; Q4 exists only for the 131072-row proposal head. Keeping
            # the head at Q8 is what keeps `--spec dflash2` loadable.
            _assign(recipe, name, Q8)


def qwen3_6_35b_a3b(model, recipe, sources):
    if "num_experts" not in model.config:
        raise ValueError("this official recipe requires Qwen3.5 MoE mathematics")
    _optional(model, recipe)
    _assign(recipe, "text/token_embedding", Q8)
    _assign(recipe, "text/output_head", Q6)
    for name, parameter in model.parameters.items():
        if not name.startswith("text/layers/") or not parameter.projection:
            continue
        if name.endswith(
            (
                "/gdn/a_projection",
                "/gdn/b_projection",
                "/moe/router",
                "/moe/shared_score",
            )
        ):
            continue
        if "/moe/experts/" in name:
            layer = int(name.split("/")[2])
            format = (
                (Q6 if layer in (34, 38, 39) else Q5) if name.endswith("/down") else Q4
            )
        else:
            format = Q8
        _assign(recipe, name, format)


def qwen3_6_27b_nvfp4(model, recipe, sources):
    if "num_experts" in model.config:
        raise ValueError("this official recipe requires Qwen3.5 Dense mathematics")
    _optional(model, recipe)
    quantized = sources["quantized"]
    _assign(recipe, "text/token_embedding", Q8)
    _assign(recipe, "text/output_head", Q8)
    for name, parameter in model.parameters.items():
        if not name.startswith("text/layers/") or not parameter.projection:
            continue
        layer = int(name.split("/")[2])
        if name.endswith(("/gdn/a_projection", "/gdn/b_projection")):
            recipe.separate(name)
            continue
        direct = (
            ("/attention/" in name and not name.endswith("/output") and layer < 24)
            or (name.endswith("/attention/output") and layer in (3, 7))
            or (name.endswith("/gdn/output") and layer == 4)
        )
        if direct:
            continue
        recipe.assign(
            name,
            format="nvfp4",
            method=import_encoded,
            source=model.source(name, quantized, "nvfp4"),
            activation_policy="AllowA4",
        )


def qwen3_8_27b_nvfp4(model, recipe, sources):
    if "num_experts" in model.config:
        raise ValueError("this official recipe requires Qwen3.5 Dense mathematics")
    _optional(model, recipe)
    quantized = sources["quantized"]
    recipe.assign("text/token_embedding", format=FP8, method=fp8_row_maxabs)
    for name, parameter in model.parameters.items():
        if not name.startswith("text/") or name == "text/token_embedding":
            continue
        source = model.source(name, quantized)
        if not parameter.projection or name.endswith(
            ("/gdn/a_projection", "/gdn/b_projection")
        ):
            recipe.assign(name, source=source)
            continue
        layer = int(name.split("/")[2]) if name.startswith("text/layers/") else -1
        format = "nvfp4" if "/mlp/" in name and layer < 56 else FP8
        recipe.assign(
            name,
            format=format,
            method=import_encoded,
            source=model.source(name, quantized, format),
            activation_policy="AllowA4" if format == "nvfp4" else "AllowA8",
        )


def qwen4_exp(model, recipe, sources):
    """Qwen4Exp text: Q4 routed experts and hash table, Q5 projections, Q8 vocabulary.

    The same shape the Qwen3.5 MoE recipe uses, applied to the architecture's own parameter
    names. Two entries are specific to this architecture: the PLE n-gram table is a lookup rather
    than a projection (bf16 would make it 102 GiB for Qwen3.8-Flash-Next, while the source file
    already carries it as IQ4_NL at about 4.5 bits, so Q4 keeps both the precision and the size),
    and the HC mixers stay at their float representation. A GGUF source arrives here after
    dequantization, so the formats below follow the artifact's needs, not the source's.
    """
    _optional(model, recipe)
    _assign(recipe, "text/token_embedding", Q8)
    _assign(recipe, "text/output_head", Q8)
    # The draft head follows the text rules verbatim. It must: the HC mix Op admits only
    # Q5_G64_FP16 row-split planes, so an unassigned `.../attention_hc/down` would be stored at a
    # width the runtime refuses by name rather than a merely larger one.
    for name, parameter in model.parameters.items():
        if not name.startswith(("text/", "mtp/")) or (
            not parameter.projection and "/ple/table" not in name
        ):
            continue
        if name in ("mtp/fc_embedding", "mtp/fc_hidden"):
            # bf16: these are the draft stem's two [h,h] projections, and bf16 is the only
            # registered [2560,2560] route (the checkpoint keeps them unquantized too).
            continue
        if name.endswith(
            ("/gdn/a_projection", "/gdn/b_projection", "/moe/router", "/moe/shared_score")
        ):
            continue
        if "/ple/table" in name:
            format = Q4
        elif "/moe/experts/" in name:
            format = Q6 if name.endswith("/down") else Q4
        elif name.endswith(
            (
                "/attention/query",
                "/attention/output_gate",
                "/attention/key",
                "/attention/index_query",
                "/attention/index_key",
                "/gdn/query",
                "/gdn/key",
                "/gdn/z",
            )
        ):
            format = Q4
        else:
            format = Q5
        _assign(recipe, name, format)


RECIPES = {
    "qwen3_6_27b": qwen3_6_27b,
    "qwen3_6_27b_nvfp4": qwen3_6_27b_nvfp4,
    "qwen3_8_27b": qwen3_8_27b,
    "qwen3_8_27b_gguf": qwen3_8_27b_gguf,
    "qwen3_8_27b_nvfp4": qwen3_8_27b_nvfp4,
    "qwen3_6_35b_a3b": qwen3_6_35b_a3b,
    "qwen4_exp": qwen4_exp,
}
