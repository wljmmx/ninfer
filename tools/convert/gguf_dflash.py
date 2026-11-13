"""llama.cpp `dflash` draft GGUF -> NInfer logical source (the DFlash/DFlash2 component).

Why this exists: a GGUF distribution of a target ships its speculative head as its own file
(`general.architecture = dflash`), while every other draft input to this converter is a
Safetensors checkpoint. This module presents that file as the duck-typed source
`qwen3_5._Builder.draft` already consumes, so the builder, the official recipes and the format
writers are unchanged and either text family (`qwen35`, `qwen4exp`) can carry the component.

Four things are specific to this source, and each is answered here rather than in the builder:

* Target block ids. llama.cpp's converter writes `dflash.target_layers` as 1-based boundaries
  (`[i + 1 for i in target_layer_ids]`, `conversion/qwen.py: DFlashModel.set_gguf_parameters`)
  because its graph captures the hidden state *after* a block. NInfer's
  `dflash_config.target_layer_ids` are zero-based block ids (`docs/maintainer/dflash.md`,
  "Target conditioning and context"), so the view subtracts one. Copying the GGUF values would
  shift every conditioning tap by one block and still pass every range check.
* Plain norms. The draft normalizes with `w * x / sqrt(mean(x^2) + eps)` - no unit offset. The
  runtime builds the draft with `build_norm(..., LLM_NORM_RMS)` and multiplies the stored vector
  directly, and llama.cpp's converter maps `model.hidden_norm`/`model.norm` without the `w + 1`
  transform it applies to the Qwen3.5 text norms. The vectors therefore pass through as stored.
* Names. HF's `model.hidden_norm` is llama.cpp's `enc.output_norm` (the norm after `fc`);
  `output_norm` is the final `model.norm`. The conv kernels, the candidate selector and the
  per-layer tensors have their own spellings, all listed in `_GLOBAL_TENSORS`/`_layer_tensors`.
* Scope. Only the draft file's own tensors are exposed, and a tensor the table does not cover is
  an error that names it: a dropped draft tensor is silent weight loss, not a smaller artifact.

RoPE is one-dimensional and full width. For an M-RoPE target llama.cpp writes degenerate sections
`[head_dim / 2, 0, 0, 0]` (`DFlashModel.set_gguf_parameters`), which the config derivation accepts
exactly; a partial-width draft rope has no field in the draft config and is refused by name.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .gguf_views import _Direct, _Info, View
from .qwen3_5 import draft_config
from .sources.gguf import QUANT_NAMES, GGUFFile, open_gguf

_ARCH = "dflash"
_BACKENDS = ("dflash", "dflash2")

# llama.cpp tensor type -> the artifact format whose code width matches it. Kept in step with the
# text GGUF sources: Q8_0 keeps its codes and fp16 scales (a repack, not a requant).
_CODE_FORMAT = {
    "Q8_0": "q8",
    "Q6_K": "q6",
    "Q5_K": "q5",
    "Q4_K": "q4",
    "IQ4_XS": "q4",
    "IQ4_NL": "q4",
    "Q3_K": "q4",
}


# ------------------------------------------------------------------------------------ metadata


def _meta(file: GGUFFile, suffix: str):
    return file.get(f"{_ARCH}.{suffix}")


def _need(file: GGUFFile, suffix: str):
    value = _meta(file, suffix)
    if value is None:
        raise ValueError(f"{file.path}: GGUF metadata is missing {_ARCH}.{suffix}")
    return value


def build_draft_config(file: GGUFFile, text_config: dict, backend: str) -> dict:
    """The HF `DFlash2DraftModel`/`DFlashDraftModel` config the shared validator reads.

    `text_config` is the target's own config: a draft is not a model in its own right, so its
    width, vocabulary and layer count are the target's, and the file is refused when it disagrees.
    """
    if backend not in _BACKENDS:
        raise ValueError(
            f"a `dflash` GGUF is a draft head; {backend!r} is not a draft component "
            f"(expected one of {list(_BACKENDS)})"
        )
    arch = str(file.get("general.architecture", ""))
    if arch and arch != _ARCH:
        raise ValueError(
            f"{file.path}: GGUF architecture is {arch!r}, not {_ARCH!r}; this reader opens draft "
            "heads only"
        )

    blocks = int(_need(file, "block_count"))
    hidden = int(_need(file, "embedding_length"))
    if hidden != int(text_config["hidden_size"]):
        raise ValueError(
            f"{file.path}: the draft is {hidden} wide but the target text config is "
            f"{text_config['hidden_size']}; the draft reads the target's hidden state"
        )
    heads = int(_need(file, "attention.head_count"))
    kv_heads = int(_need(file, "attention.head_count_kv"))
    head_dim = int(_need(file, "attention.key_length"))
    value_dim = int(file.get(f"{_ARCH}.attention.value_length", head_dim) or head_dim)
    if value_dim != head_dim:
        raise ValueError(
            f"{file.path}: draft key width {head_dim} differs from value width {value_dim}; the "
            "draft config has one head dimension"
        )
    intermediate = int(_need(file, "feed_forward_length"))
    context = int(_need(file, "context_length"))
    eps = float(_need(file, "attention.layer_norm_rms_epsilon"))

    # One-dimensional, full-head rope. llama.cpp writes the degenerate M-RoPE sections for a draft
    # whose target ropes on three axes; anything else is a partial-width rope this config cannot
    # express.
    sections = _meta(file, "rope.dimension_sections")
    if sections is not None:
        section = [int(x) for x in sections]
        while section and section[-1] == 0:
            section.pop()
        if section != [head_dim // 2]:
            raise ValueError(
                f"{file.path}: draft rope sections {list(sections)} do not describe a full-width "
                f"one-axis rope over {head_dim} dims; a partial draft rope is not representable"
            )

    causal = _meta(file, "attention.causal")
    if causal is not None and bool(causal):
        raise ValueError(
            f"{file.path}: draft declares causal attention; the masked-block draft is non-causal"
        )

    window = _meta(file, "attention.sliding_window")
    pattern = _meta(file, "attention.sliding_window_pattern")
    if pattern is not None:
        pattern = [bool(x) for x in pattern]
        if len(pattern) != blocks:
            raise ValueError(
                f"{file.path}: sliding_window_pattern has {len(pattern)} entries for {blocks} blocks"
            )
        layer_types = [
            "sliding_attention" if sliding else "full_attention" for sliding in pattern
        ]
    else:
        layer_types = ["full_attention"] * blocks

    taps = [int(index) for index in _need(file, "target_layers")]
    # The GGUF stores llama.cpp's 1-based boundaries; NInfer conditions on zero-based block ids.
    zero_based = [index - 1 for index in taps]

    mask = file.get("tokenizer.ggml.mask_token_id")
    if mask is None:
        raise ValueError(
            f"{file.path}: GGUF metadata is missing tokenizer.ggml.mask_token_id; the draft config "
            "requires the mask token the block forward fills with"
        )

    dflash_config = {
        "target_layer_ids": zero_based,
        "mask_token_id": int(mask),
    }
    conv_fields = {
        key: _meta(file, key)
        for key in ("conv_kernel_size", "conv_group_size", "selector_rank", "selector_top_k")
    }
    has_dflash2_meta = all(value is not None for value in conv_fields.values())
    if has_dflash2_meta and backend != "dflash2":
        raise ValueError(
            f"{file.path}: the file carries DFlash2 metadata (conv kernel/group and selector "
            f"rank/top-k), so it is the 'dflash2' component, not {backend!r}"
        )
    if not has_dflash2_meta and backend == "dflash2":
        missing = sorted(key for key, value in conv_fields.items() if value is None)
        raise ValueError(
            f"{file.path}: the 'dflash2' component needs dflash metadata {missing}; this file is a "
            "plain DFlash draft"
        )
    if has_dflash2_meta:
        dflash_config.update(
            {key: int(value) for key, value in conv_fields.items()}
        )

    raw = {
        "architectures": ["DFlash2DraftModel" if backend == "dflash2" else "DFlashDraftModel"],
        "model_type": "qwen3",
        "hidden_act": "silu",
        "attention_bias": False,
        "is_causal": False,
        "hidden_size": hidden,
        "vocab_size": int(text_config["vocab_size"]),
        "num_target_layers": int(text_config["num_hidden_layers"]),
        "intermediate_size": intermediate,
        "num_attention_heads": heads,
        "num_key_value_heads": kv_heads,
        "head_dim": head_dim,
        "num_hidden_layers": blocks,
        "max_position_embeddings": context,
        "rms_norm_eps": eps,
        "rope_parameters": {"rope_theta": float(_need(file, "rope.freq_base"))},
        "layer_types": layer_types,
        "dflash_config": dflash_config,
    }
    if "sliding_attention" in layer_types:
        if not window:
            raise ValueError(
                f"{file.path}: a sliding draft layer needs dflash.attention.sliding_window"
            )
        raw["sliding_window"] = int(window)
    # The shared validator owns the cross-field rules (head divisibility, tap range, mask range,
    # selector geometry), so the derived dict goes through exactly the path the 27B checkpoint
    # takes instead of a second, weaker copy of them.
    return draft_config(raw, text_config, backend)


# ------------------------------------------------------------------------------------ the source


def _global_tensors(hidden: int, taps: int) -> dict[str, tuple[str, tuple[int, ...]]]:
    return {
        "fc.weight": ("fc.weight", (hidden, taps * hidden)),
        "hidden_norm.weight": ("enc.output_norm.weight", (hidden,)),
        "norm.weight": ("output_norm.weight", (hidden,)),
    }


def _layer_tensors(config: dict, layer: int, hidden: int) -> dict[str, tuple[str, tuple[int, ...]]]:
    q = int(config["num_attention_heads"]) * int(config["head_dim"])
    k = int(config["num_key_value_heads"]) * int(config["head_dim"])
    intermediate = int(config["intermediate_size"])
    head_dim = int(config["head_dim"])
    gguf, local = f"blk.{layer}.", f"layers.{layer}."
    table = {
        local + "input_layernorm.weight": (gguf + "attn_norm.weight", (hidden,)),
        local + "post_attention_layernorm.weight": (gguf + "ffn_norm.weight", (hidden,)),
        local + "self_attn.q_proj.weight": (gguf + "attn_q.weight", (q, hidden)),
        local + "self_attn.k_proj.weight": (gguf + "attn_k.weight", (k, hidden)),
        local + "self_attn.v_proj.weight": (gguf + "attn_v.weight", (k, hidden)),
        local + "self_attn.o_proj.weight": (gguf + "attn_output.weight", (hidden, q)),
        local + "self_attn.q_norm.weight": (gguf + "attn_q_norm.weight", (head_dim,)),
        local + "self_attn.k_norm.weight": (gguf + "attn_k_norm.weight", (head_dim,)),
        local + "mlp.gate_proj.weight": (gguf + "ffn_gate.weight", (intermediate, hidden)),
        local + "mlp.up_proj.weight": (gguf + "ffn_up.weight", (intermediate, hidden)),
        local + "mlp.down_proj.weight": (gguf + "ffn_down.weight", (hidden, intermediate)),
    }
    draft = config["dflash_config"]
    if "conv_kernel_size" in draft:
        kernel = int(draft["conv_kernel_size"])
        groups = hidden // int(draft["conv_group_size"])
        for branch, stem in (("attention_conv", "attn_conv"), ("mlp_conv", "ffn_conv")):
            table[local + branch + ".base_kernel"] = (
                gguf + stem + "_base",
                (2, kernel, hidden),
            )
            table[local + branch + ".kernel_projection.weight"] = (
                gguf + stem + "_proj.weight",
                (2 * kernel * groups, hidden),
            )
    return table


def _selector_tensors(config: dict, hidden: int, vocab: int) -> dict[str, tuple[str, tuple[int, ...]]]:
    draft = config["dflash_config"]
    if "selector_rank" not in draft:
        return {}
    rank = int(draft["selector_rank"])
    return {
        "candidate_selector.hidden_projection.weight": ("selector_hidden.weight", (rank, hidden)),
        # The checkpoint stores the codebooks without a `.weight` suffix; llama.cpp appends one.
        "candidate_selector.predecessor_codebook": (
            "selector_predecessor.weight",
            (vocab, rank),
        ),
        "candidate_selector.successor_codebook": (
            "selector_successor.weight",
            (vocab, rank),
        ),
    }


def mapping(config: dict, target: dict) -> dict[str, tuple[str, tuple[int, ...]]]:
    """Logical (HF) name -> (GGUF tensor, expected shape) for the whole draft.

    Widths that the draft config itself does not carry (the hidden width, the vocabulary) come
    from the target: the draft reads the target's hidden state and embeds its vocabulary.
    """
    hidden = int(target["hidden_size"])
    vocab = int(target["vocab_size"])
    taps = len(config["dflash_config"]["target_layer_ids"])
    table = _global_tensors(hidden, taps)
    for layer in range(int(config["num_hidden_layers"])):
        table.update(_layer_tensors(config, layer, hidden))
    table.update(_selector_tensors(config, hidden, vocab))
    return table


class DFlashGGUFSource:
    """A `dflash` draft GGUF presented under the HF names `qwen3_5._Builder.draft` asks for."""

    def __init__(
        self,
        path: str | Path,
        text_config: dict,
        *,
        backend: str = "dflash2",
        root: str | Path | None = None,
    ) -> None:
        self.path = Path(path)
        self.file = open_gguf(self.path)
        self.backend = backend
        self.target = dict(text_config)
        self.config = build_draft_config(self.file, text_config, backend)
        self.root = Path(root) if root else self.path.parent
        self._views: dict[str, View] = {}
        self._formats: dict[str, str | None] = {}
        self.pairs: dict[str, str] = {}
        self._build_views()
        self.weight_map = {name: name for name in self._views}

    # -- lifecycle ------------------------------------------------------
    def close(self) -> None:
        self.file.close()

    def __enter__(self) -> "DFlashGGUFSource":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    # -- duck-typed source surface --------------------------------------
    @property
    def bytes_read(self) -> int:
        return self.file.bytes_read

    def describe(self, name: str) -> _Info:
        try:
            return self._views[name].info()
        except KeyError as error:
            raise ValueError(
                f"{self.path}: dflash GGUF has no tensor for logical {name!r}"
            ) from error

    def has(self, name: str) -> bool:
        """Duck-typing for `matrix_source`: a GGUF carries no compressed-tensors companions."""
        return name in self._views

    def read_flat(self, name: str, begin: int = 0, end: int | None = None):
        view = self._views.get(name)
        if view is None:
            raise ValueError(f"{self.path}: dflash GGUF has no tensor for logical {name!r}")
        elements = int(np.prod(view.shape))
        end = elements if end is None else end
        if not 0 <= begin <= end <= elements:
            raise ValueError(f"{name}: range [{begin},{end}) exceeds {view.shape}")
        values = view.read(begin, end)
        import torch  # the conversion stack; the container/quant layer above needs none

        if view.dtype == "I64":
            return torch.from_numpy(np.array(values, dtype=np.int64, copy=True))
        return torch.from_numpy(np.array(values, dtype=np.float32, copy=True)).to(torch.bfloat16)

    def code_format(self, name: str) -> str | None:
        if name not in self._formats:
            raise ValueError(f"{self.path}: the draft source does not carry {name!r}")
        return self._formats[name]

    # -- mapping --------------------------------------------------------
    def _add(self, name: str, gguf_name: str, shape: tuple[int, ...]) -> None:
        info = self.file.tensors.get(gguf_name)
        if info is None:
            raise ValueError(f"{self.path}: draft tensor {gguf_name!r} is missing")
        if tuple(info.shape) != tuple(shape):
            raise ValueError(
                f"{self.path}/{gguf_name}: shape {tuple(info.shape)} != the draft contract "
                f"{tuple(shape)} for {name!r}"
            )
        self._views[name] = _Direct(self.file, gguf_name)
        self._formats[name] = _CODE_FORMAT.get(QUANT_NAMES.get(info.ggml_type, ""))
        self.pairs[name] = gguf_name

    def _build_views(self) -> None:
        self.table = mapping(self.config, self.target)
        for name, (gguf_name, shape) in self.table.items():
            self._add(name, gguf_name, shape)
        self._check_coverage()

    def _check_coverage(self) -> None:
        """Every tensor of the file must be a value the draft consumes.

        A draft tensor this table does not name is a weight the artifact would silently lack: the
        run would still succeed and the draft would quietly propose from a half-initialized head,
        so the surplus is refused by name instead.
        """
        unknown = sorted(set(self.file.tensors) - set(self.pairs.values()))
        if unknown:
            raise ValueError(
                f"{self.path}: the draft GGUF carries {len(unknown)} tensor(s) this reader does "
                f"not map, starting with {unknown[:4]}; refusing rather than dropping weights"
            )


def open_companion(
    path: str | Path,
    text_config: dict,
    *,
    backend: str = "dflash2",
    root: str | Path | None = None,
) -> DFlashGGUFSource:
    """Open a `dflash` GGUF as the named draft component of `text_config`."""
    return DFlashGGUFSource(path, text_config, backend=backend, root=root)
