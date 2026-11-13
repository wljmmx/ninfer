"""Qwen3.5 dense GGUF -> NInfer logical source (text, MTP and vision components).

Why this exists: Qwen3.8-27B ships as GGUF (a `qwen35` text file that also carries the draft block,
a standalone MTP-only file, and a `clip` mmproj for the tower), while the converter's normal input
is Safetensors + config.json. This module presents those files as the duck-typed source
`tools/convert/qwen3_5.py` already consumes, so the builder, the recipes and the format writers are
unchanged.

Four things are specific to this source; each is answered here rather than in the recipe:

* Geometry. llama.cpp flattens the HF config into `qwen35.*`; the rope subtree (MRoPE sections,
  partial rotary factor) and the vision subtree are reconstructed from `rope.dimension_*`,
  `attention.*` and the mmproj's `clip.*` block rather than assumed.
* Offset norms. llama.cpp writes every `*norm.weight` as `w + 1` (except
  `linear_attn.norm.weight`), because HF's Qwen3.5 norms are unit-offset; the artifact stores the
  zero-centred `w` and the runtime adds one. Measured against the existing artifact: every such
  vector differs by exactly `-1.000`, while `ssm_norm` matches bit for bit.
* `A_log`. llama.cpp writes GDN's state decay as `ssm_a = -exp(A_log)`; the artifact keeps
  `A_log`, so the view takes `log(-x)`.
* Value-head order. llama.cpp stores the GDN value heads tiled; HF (and the runtime) keeps them
  grouped. Every V-carrying tensor (qkv's V rows, z, alpha/beta, A_log, dt_bias, the conv V
  channels and out_proj's columns) is re-ordered with one permutation. The direction is measured
  against the existing artifact, not copied: this file needs `(per_key, key_heads, head_dim)`
  transposed, the opposite of what `gguf_qwen4exp` needed for its Qwen4Exp file.

Scope: text, the mtp component from a standalone MTP GGUF, and the vision component from an
mmproj. A `.gguf` source still cannot supply DFlash weights; those travel as an artifact component
(see `tools/artifact/carry.py`).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .gguf_views import (
    View,
    _ArrayView,
    _ColPermute,
    _ConcatRows,
    _Direct,
    _FlatPermute,
    _Info,
    _RowPermute,
    _Rows,
)
from .sources.gguf import QUANT_NAMES, GGUFFile, open_gguf

_TEXT_ARCH = "qwen35"
_CLIP_ARCH = "clip"

# llama.cpp's tensor type -> the artifact format whose code width matches it. Q8_0 keeps its codes
# and fp16 scales (a repack, not a requant); Q3_K is stored one step up because the artifact has no
# 3-bit plane.
_CODE_FORMAT = {
    "Q8_0": "q8",
    "Q6_K": "q6",
    "Q5_K": "q5",
    "Q4_K": "q4",
    "IQ4_XS": "q4",
    "IQ4_NL": "q4",
    "Q3_K": "q4",
}

_HIDDEN_ACT = "silu"
_VISION_HIDDEN_ACT = "gelu_pytorch_tanh"


# ------------------------------------------------------------------------------------- views


class _Minus1(View):
    """`w - 1`: llama.cpp writes unit-offset RMSNorm weights, the artifact stores the offset."""

    def __init__(self, inner: View) -> None:
        if len(inner.shape) != 1:
            raise ValueError("offset-norm view requires a vector")
        self.inner = inner
        self.shape = inner.shape
        self.dtype = inner.dtype

    def read(self, begin: int, end: int) -> np.ndarray:
        return self.inner.read(begin, end) - 1.0


class _LogOfNegative(View):
    """`log(-ssm_a)`: llama.cpp writes GDN's decay as `-exp(A_log)`."""

    def __init__(self, inner: View) -> None:
        if len(inner.shape) != 1:
            raise ValueError("A_log view requires a vector")
        self.inner = inner
        self.shape = inner.shape
        self.dtype = inner.dtype

    def read(self, begin: int, end: int) -> np.ndarray:
        values = self.inner.read(begin, end)
        if not np.all(values < 0):
            raise ValueError(
                f"{self.inner.shape}: ssm_a is not negative; it is not -exp(A_log) in this file"
            )
        return np.log(-values)


def value_head_order(count: int, key_heads: int, per_key: int, head_dim: int) -> np.ndarray:
    """GGUF (tiled) -> HF (grouped) index map for every V-head-carrying axis.

    GGUF position = head_in_group * key_heads + key_head; HF position = key_head * per_key +
    head_in_group. Measured on the existing artifact: applying this map gives a relative error of
    0.00-0.09 against its weights, while the other direction gives 1.2-1.4 (i.e. a different
    tensor). `tiled_to_grouped` is the other direction and is what the Qwen4Exp file needed.
    """
    if key_heads * per_key * head_dim != count:
        raise ValueError(
            f"V-head order: {key_heads} x {per_key} x {head_dim} does not cover {count}"
        )
    return np.transpose(
        np.arange(count).reshape(per_key, key_heads, head_dim), (1, 0, 2)
    ).reshape(-1)


# ------------------------------------------------------------------------------------ metadata


def _meta(file: GGUFFile, suffix: str):
    return file.get(f"{_TEXT_ARCH}.{suffix}")


def _need(file: GGUFFile, suffix: str):
    value = _meta(file, suffix)
    if value is None:
        raise ValueError(
            f"{file.path}: GGUF metadata is missing {_TEXT_ARCH}.{suffix}; pass --gguf-config "
            "with the checkpoint's config.json"
        )
    return value


def build_config(file: GGUFFile, sidecar: Path | None) -> dict:
    """Reconstruct the config.json shape `qwen3_5.text_config` validates.

    A GGUF carries the hyperparameters llama.cpp writes from the HF config. What it cannot carry
    (the rope subtree, the MTP marker fields, vision) is reconstructed from the geometric
    relationships llama.cpp itself relies on, or taken from the sidecar when one is given.
    """
    arch = str(file.get("general.architecture", ""))
    if arch and arch != _TEXT_ARCH:
        raise ValueError(
            f"{file.path}: GGUF architecture is {arch!r}, not {_TEXT_ARCH!r}; this path converts "
            "Qwen3.5 dense (qwen35) only"
        )
    base: dict = {}
    if sidecar is not None:
        base = json.loads(Path(sidecar).read_text())
    config = dict(base)
    config.setdefault("architectures", ["Qwen3_5ForCausalLM"])
    config.setdefault("model_type", "qwen3_5_text")
    text = dict(base.get("text_config") or {})

    blocks = int(_need(file, "block_count"))
    nextn = int(file.get("qwen35.nextn_predict_layers", 0) or 0)
    layers = blocks - nextn
    if layers <= 0:
        raise ValueError(f"{file.path}: block_count {blocks} leaves no text layer")
    hidden = int(_need(file, "embedding_length"))
    heads = int(_need(file, "attention.head_count"))
    kv_heads = int(_need(file, "attention.head_count_kv"))
    head_dim = int(_need(file, "attention.key_length"))
    interval = int(file.get("qwen35.full_attention_interval", 0) or 0)
    if interval <= 0:
        raise ValueError(
            f"{file.path}: GGUF has no full_attention_interval; the dense/linear layer layout is "
            "required"
        )
    layer_types = [
        "full_attention" if (index + 1) % interval == 0 else "linear_attention"
        for index in range(layers)
    ]
    nk = int(_need(file, "ssm.group_count"))
    nv = int(_need(file, "ssm.time_step_rank"))
    dk = int(_need(file, "ssm.state_size"))
    inner = int(_need(file, "ssm.inner_size"))
    if inner % nv:
        raise ValueError(f"{file.path}: ssm.inner_size {inner} is not a multiple of {nv}")
    rope_width = int(_need(file, "rope.dimension_count"))
    sections = _need(file, "rope.dimension_sections")
    section = [int(x) for x in sections]
    if len(section) > 3 and section[-1] == 0:
        section = section[:3]

    derived = {
        "hidden_size": hidden,
        "num_hidden_layers": layers,
        "num_attention_heads": heads,
        "num_key_value_heads": kv_heads,
        "head_dim": head_dim,
        "layer_types": layer_types,
        "linear_num_key_heads": nk,
        "linear_key_head_dim": dk,
        "linear_num_value_heads": nv,
        "linear_value_head_dim": inner // nv,
        "linear_conv_kernel_dim": int(_need(file, "ssm.conv_kernel")),
        "intermediate_size": int(_need(file, "feed_forward_length")),
        "hidden_act": _HIDDEN_ACT,
        "attention_bias": False,
        "attn_output_gate": True,
        "mamba_ssm_dtype": "float32",
        "mtp_num_hidden_layers": 1,
        "mtp_use_dedicated_embeddings": False,
    }
    text = {**derived, **text}
    if "token_embd.weight" in file.tensors:
        rows = int(file.tensors["token_embd.weight"].shape[0])
        if text.get("vocab_size") not in (None, rows):
            raise ValueError(
                f"{file.path}: the sidecar says vocab_size={text['vocab_size']} but the embedding "
                f"has {rows} rows"
            )
        text["vocab_size"] = rows
    text.setdefault("max_position_embeddings", int(_meta(file, "context_length") or 0))
    text.setdefault(
        "rms_norm_eps", float(_meta(file, "attention.layer_norm_rms_epsilon") or 1e-6)
    )
    text.setdefault("tie_word_embeddings", False)
    if "rope_parameters" not in text:
        factor = rope_width / head_dim
        text["rope_parameters"] = {
            "rope_theta": float(_need(file, "rope.freq_base")),
            "partial_rotary_factor": factor,
            "mrope_section": section,
        }
    config["text_config"] = text
    missing = [
        key
        for key in ("hidden_size", "vocab_size", "num_hidden_layers", "max_position_embeddings")
        if not text.get(key)
    ]
    if missing:
        raise ValueError(f"{file.path}: cannot reconstruct {missing}")
    return config


def build_vision_config(file: GGUFFile, text_hidden: int) -> dict:
    """The `qwen3_5.vision_config` input, from the mmproj's `clip.*` block."""
    arch = str(file.get("general.architecture", ""))
    if arch and arch != _CLIP_ARCH:
        raise ValueError(
            f"{file.path}: GGUF architecture is {arch!r}, not {_CLIP_ARCH!r}; an mmproj is required"
        )
    depth = int(file.require("clip.vision.block_count"))
    hidden = int(file.require("clip.vision.embedding_length"))
    patches = sorted(name for name in file.tensors if name.startswith("v.patch_embd.weight"))
    if not patches:
        raise ValueError(f"{file.path}: mmproj has no v.patch_embd.weight")
    positions = file.tensors.get("v.position_embd.weight")
    if positions is None:
        raise ValueError(f"{file.path}: mmproj has no v.position_embd.weight")
    projection = int(file.require("clip.vision.projection_dim"))
    if projection != text_hidden:
        raise ValueError(
            f"{file.path}: the tower projects to {projection} but the text width is {text_hidden}"
        )
    return {
        "vision_config": {
            "in_channels": 3,
            "hidden_act": _VISION_HIDDEN_ACT,
            "deepstack_visual_indexes": [],
            "out_hidden_size": projection,
            "depth": depth,
            "hidden_size": hidden,
            "intermediate_size": int(file.require("clip.vision.feed_forward_length")),
            "num_heads": int(file.require("clip.vision.attention.head_count")),
            "patch_size": int(file.require("clip.vision.patch_size")),
            "temporal_patch_size": len(patches),
            "spatial_merge_size": int(file.require("clip.vision.spatial_merge_size")),
            "num_position_embeddings": int(positions.shape[0]),
        }
    }


# ------------------------------------------------------------------------------------ the source


class GGUFSource:
    """A `qwen35` GGUF presented as the converter's logical source."""

    def __init__(
        self,
        path: str | Path,
        *,
        config_path: str | Path | None = None,
        root: str | Path | None = None,
    ) -> None:
        self.path = Path(path)
        self.file = open_gguf(self.path)
        sidecar = Path(config_path) if config_path else None
        if sidecar is None and (self.path.parent / "config.json").is_file():
            sidecar = self.path.parent / "config.json"
        self.config = build_config(self.file, sidecar)
        self.root = Path(root) if root else (sidecar.parent if sidecar else self.path.parent)
        self._views: dict[str, View] = {}
        self._formats: dict[str, str | None] = {}
        self._build_views()
        self._alias_language_model()
        self.weight_map = {name: name for name in self._views}

    @classmethod
    def for_mtp(
        cls, path: str | Path, text_config: dict, *, root: str | Path | None = None
    ) -> "GGUFSource":
        """A standalone MTP GGUF presented as the HF `mtp.*` names the builder reads.

        llama.cpp writes the draft head as one extra block (`blk.<block_count-1>` holding
        `nextn.*`) plus copies of the token embedding and the output head. Those two copies are
        dropped: the engine reuses the text component's embedding and LM head.
        """
        source = cls.__new__(cls)
        source.path = Path(path)
        source.file = open_gguf(source.path)
        source.config = {"text_config": dict(text_config)}
        source.root = Path(root) if root else source.path.parent
        source._views = {}
        source._formats = {}
        source._build_mtp_views(int(text_config["hidden_size"]))
        source.weight_map = {name: name for name in source._views}
        return source

    # -- lifecycle ------------------------------------------------------
    def close(self) -> None:
        self.file.close()

    def __enter__(self) -> "GGUFSource":
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
            raise ValueError(f"{self.path}: GGUF has no tensor for logical {name!r}") from error

    def has(self, name: str) -> bool:
        """Duck-typing for `matrix_source`: a GGUF carries no compressed-tensors companions."""
        return name in self._views

    def read_flat(self, name: str, begin: int = 0, end: int | None = None):
        view = self._views.get(name)
        if view is None:
            raise ValueError(f"{self.path}: GGUF has no tensor for logical {name!r}")
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
        """The artifact format that matches this logical parameter's source code width.

        `None` means the source stores it at 16 bits (a norm, a bias or a bare float matrix) and
        the recipe's default applies.
        """
        if name not in self._formats:
            raise ValueError(f"{self.path}: the source does not carry {name!r}")
        return self._formats[name]

    # -- mapping --------------------------------------------------------
    def _has(self, name: str) -> bool:
        return name in self.file.tensors

    def _direct(self, name: str) -> View:
        return _Direct(self.file, name)

    def _add(self, name: str, view: View, *, gguf_name: str | None = None) -> None:
        self._views[name] = view
        source = gguf_name if gguf_name is not None else name
        info = self.file.tensors.get(source)
        self._formats[name] = (
            None if info is None else _CODE_FORMAT.get(QUANT_NAMES.get(info.ggml_type, ""))
        )

    def _norm(self, name: str, gguf_name: str) -> None:
        """A unit-offset norm: llama.cpp added one, the artifact stores the offset."""
        self._add(name, _Minus1(self._direct(gguf_name)), gguf_name=gguf_name)

    def _alias_language_model(self) -> None:
        """Expose the text tensors under both HF prefixes.

        The builder reads `model.*` for a config without a `text_config` subtree and
        `model.language_model.*` for one with it; this source carries the subtree (the MTP and
        vision components read it), so both spellings are registered and either resolution lands
        on the same view.
        """
        for name, view in list(self._views.items()):
            if not name.startswith("model.") or name.startswith("model.language_model."):
                continue
            alias = "model.language_model." + name[len("model."):]
            self._views[alias] = view
            self._formats[alias] = self._formats.get(name)

    def _build_views(self) -> None:
        cfg = self.config["text_config"]
        layers = int(cfg["num_hidden_layers"])
        hidden = int(cfg["hidden_size"])
        heads = int(cfg["num_attention_heads"])
        head_dim = int(cfg["head_dim"])
        nk = int(cfg["linear_num_key_heads"])
        nv = int(cfg["linear_num_value_heads"])
        dk = int(cfg["linear_key_head_dim"])
        dv = int(cfg["linear_value_head_dim"])
        taps = int(cfg["linear_conv_kernel_dim"])
        per_key = nv // nk
        order = value_head_order(nv * dv, nk, per_key, dv)
        flat_order = value_head_order(nv, nk, per_key, 1)

        if not self._has("token_embd.weight"):
            raise ValueError(f"{self.path}: GGUF is missing token_embd.weight")
        self._add("model.embed_tokens.weight", self._direct("token_embd.weight"),
                  gguf_name="token_embd.weight")
        if not self._has("output.weight"):
            raise ValueError(
                f"{self.path}: GGUF has no output.weight; NInfer stores the output head as its own "
                "physical parameter and a tied head is not accepted by the converter"
            )
        self._add("lm_head.weight", self._direct("output.weight"), gguf_name="output.weight")
        self._norm("model.norm.weight", "output_norm.weight")

        for layer in range(layers):
            gguf = f"blk.{layer}."
            local = f"model.layers.{layer}."
            self._norm(local + "input_layernorm.weight", gguf + "attn_norm.weight")
            self._norm(
                local + "post_attention_layernorm.weight", gguf + "post_attention_norm.weight"
            )
            if cfg["layer_types"][layer] == "full_attention":
                # Q and the output gate are one (2*Q, H) parent with head-interleaved rows, which
                # is what llama.cpp keeps under attn_q; the builder separates them by row range.
                self._add(local + "self_attn.q_proj.weight",
                          self._direct(gguf + "attn_q.weight"),
                          gguf_name=gguf + "attn_q.weight")
                for suffix, name in (
                    ("k_proj", "attn_k"),
                    ("v_proj", "attn_v"),
                    ("o_proj", "attn_output"),
                ):
                    self._add(local + "self_attn." + suffix + ".weight",
                              self._direct(gguf + name + ".weight"),
                              gguf_name=gguf + name + ".weight")
                for suffix, name in (("q_norm", "attn_q_norm"), ("k_norm", "attn_k_norm")):
                    self._norm(local + "self_attn." + suffix + ".weight", gguf + name + ".weight")
            else:
                self._gdn(gguf, local, hidden, nk, nv, dk, dv, per_key, order, flat_order, taps)
            for suffix, name in (
                ("gate_proj", "ffn_gate"),
                ("up_proj", "ffn_up"),
                ("down_proj", "ffn_down"),
            ):
                self._add(local + "mlp." + suffix + ".weight",
                          self._direct(gguf + name + ".weight"),
                          gguf_name=gguf + name + ".weight")

    def _gdn(self, gguf: str, local: str, hidden: int, nk: int, nv: int, dk: int, dv: int,
             per_key: int, order: np.ndarray, flat_order: np.ndarray, taps: int) -> None:
        q, v = nk * dk, nv * dv
        channels = 2 * q + v
        name = local + "linear_attn."
        qkv_name = gguf + "attn_qkv.weight"
        if not self._has(qkv_name):
            raise ValueError(f"{self.path}: {qkv_name} is missing")
        qkv = _Rows(self._direct(qkv_name), 0, channels, hidden)
        # q | k | v, with the V section put back in grouped order.
        qk = _Rows(qkv, 0, 2 * q, hidden)
        vpart = _RowPermute(_Rows(qkv, 2 * q, v, hidden), order)
        self._add(name + "in_proj_qkv.weight", _ConcatRows([(qk, 0, 2 * q), (vpart, 0, v)]),
                  gguf_name=qkv_name)
        if not self._has(gguf + "attn_gate.weight"):
            raise ValueError(f"{self.path}: {gguf}attn_gate.weight is missing")
        self._add(name + "in_proj_z.weight",
                  _RowPermute(self._direct(gguf + "attn_gate.weight"), order),
                  gguf_name=gguf + "attn_gate.weight")
        for suffix, tensor in (("in_proj_a", "ssm_alpha"), ("in_proj_b", "ssm_beta")):
            self._add(name + suffix + ".weight",
                      _RowPermute(self._direct(gguf + tensor + ".weight"), flat_order),
                      gguf_name=gguf + tensor + ".weight")
        # llama.cpp squeezed [C, 1, taps] to [C, taps]; a size-1 axis moves no element. The V
        # channels are tiled, and that permutation needs the whole kernel at once.
        raw = self._direct(gguf + "ssm_conv1d.weight")
        rows = raw.read(0, channels * taps).reshape(channels, taps)
        rows = np.concatenate([rows[: 2 * q], rows[2 * q :][order]])
        self._add(name + "conv1d.weight", _ArrayView(rows.reshape(channels, 1, taps)),
                  gguf_name=gguf + "ssm_conv1d.weight")
        self._add(name + "A_log",
                  _LogOfNegative(_FlatPermute(self._direct(gguf + "ssm_a"), flat_order)),
                  gguf_name=gguf + "ssm_a")
        self._add(name + "dt_bias",
                  _FlatPermute(self._direct(gguf + "ssm_dt.bias"), flat_order),
                  gguf_name=gguf + "ssm_dt.bias")
        self._add(name + "norm.weight", self._direct(gguf + "ssm_norm.weight"),
                  gguf_name=gguf + "ssm_norm.weight")
        self._add(name + "out_proj.weight",
                  _ColPermute(self._direct(gguf + "ssm_out.weight"), order),
                  gguf_name=gguf + "ssm_out.weight")

    def _build_mtp_views(self, hidden: int) -> None:
        block = None
        for name in self.file.tensors:
            if name.endswith(".nextn.eh_proj.weight"):
                block = name.rsplit(".nextn.", 1)[0]
                break
        if block is None:
            raise ValueError(f"{self.path}: GGUF carries no nextn draft block")
        gguf, local = block + ".", "mtp.layers.0."
        self._add("mtp.fc.weight", self._direct(gguf + "nextn.eh_proj.weight"),
                  gguf_name=gguf + "nextn.eh_proj.weight")
        for suffix, name in (
            ("pre_fc_norm_embedding", "nextn.enorm"),
            ("pre_fc_norm_hidden", "nextn.hnorm"),
            ("norm", "nextn.shared_head_norm"),
        ):
            self._norm("mtp." + suffix + ".weight", gguf + name + ".weight")
        self._norm(local + "input_layernorm.weight", gguf + "attn_norm.weight")
        self._norm(local + "post_attention_layernorm.weight", gguf + "post_attention_norm.weight")
        self._add(local + "self_attn.q_proj.weight", self._direct(gguf + "attn_q.weight"),
                  gguf_name=gguf + "attn_q.weight")
        for suffix, name in (
            ("k_proj", "attn_k"),
            ("v_proj", "attn_v"),
            ("o_proj", "attn_output"),
        ):
            self._add(local + "self_attn." + suffix + ".weight",
                      self._direct(gguf + name + ".weight"),
                      gguf_name=gguf + name + ".weight")
        for suffix, name in (("q_norm", "attn_q_norm"), ("k_norm", "attn_k_norm")):
            self._norm(local + "self_attn." + suffix + ".weight", gguf + name + ".weight")
        for suffix, name in (
            ("gate_proj", "ffn_gate"),
            ("up_proj", "ffn_up"),
            ("down_proj", "ffn_down"),
        ):
            self._add(local + "mlp." + suffix + ".weight",
                      self._direct(gguf + name + ".weight"),
                      gguf_name=gguf + name + ".weight")


class VisionGGUFSource:
    """An mmproj GGUF presented as the HF `model.visual.*` names the builder reads."""

    def __init__(self, path: str | Path, text_hidden: int, *, root: str | Path | None = None) -> None:
        self.path = Path(path)
        self.file = open_gguf(self.path)
        self.config = build_vision_config(self.file, text_hidden)
        self.root = Path(root) if root else self.path.parent
        self._views: dict[str, View] = {}
        self._build_views(self.config["vision_config"])
        self.weight_map = {name: name for name in self._views}

    def close(self) -> None:
        self.file.close()

    def __enter__(self) -> "VisionGGUFSource":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    @property
    def bytes_read(self) -> int:
        return self.file.bytes_read

    def describe(self, name: str) -> _Info:
        try:
            return self._views[name].info()
        except KeyError as error:
            raise ValueError(f"{self.path}: mmproj has no tensor for {name!r}") from error

    def has(self, name: str) -> bool:
        return name in self._views

    def read_flat(self, name: str, begin: int = 0, end: int | None = None):
        view = self._views.get(name)
        if view is None:
            raise ValueError(f"{self.path}: mmproj has no tensor for {name!r}")
        elements = int(np.prod(view.shape))
        end = elements if end is None else end
        if not 0 <= begin <= end <= elements:
            raise ValueError(f"{name}: range [{begin},{end}) exceeds {view.shape}")
        values = view.read(begin, end)
        import torch

        return torch.from_numpy(np.array(values, dtype=np.float32, copy=True)).to(torch.bfloat16)

    def _add(self, name: str, gguf_name: str) -> None:
        self._views[name] = _Direct(self.file, gguf_name)

    def _build_views(self, cfg: dict) -> None:
        hidden = int(cfg["hidden_size"])
        patch = int(cfg["patch_size"])
        temporal = int(cfg["temporal_patch_size"])
        patches = [f"v.patch_embd.weight{'' if i == 0 else '.' + str(i)}" for i in range(temporal)]
        for name in patches:
            if name not in self.file.tensors:
                raise ValueError(f"{self.path}: mmproj has no {name}")
        # HF keeps one conv3d kernel [h, 3, temporal, patch, patch]; llama.cpp splits it per
        # temporal frame, so the frames are stacked back along the third axis.
        frames = [self.file.read_tensor(name).reshape(hidden, 3, patch, patch) for name in patches]
        stacked = np.stack(frames, axis=2).reshape(hidden, 3, temporal, patch, patch)
        self._views["model.visual.patch_embed.proj.weight"] = _ArrayView(stacked)
        self._add("model.visual.patch_embed.proj.bias", "v.patch_embd.bias")
        self._add("model.visual.pos_embed.weight", "v.position_embd.weight")
        for layer in range(int(cfg["depth"])):
            gguf, local = f"v.blk.{layer}.", f"model.visual.blocks.{layer}."
            for hf, gg in (("norm1", "ln1"), ("norm2", "ln2")):
                for kind in ("weight", "bias"):
                    self._add(local + hf + "." + kind, gguf + gg + "." + kind)
            for kind in ("weight", "bias"):
                self._add(local + "attn.qkv." + kind, gguf + "attn_qkv." + kind)
            for kind in ("weight", "bias"):
                self._add(local + "attn.proj." + kind, gguf + "attn_out." + kind)
            for kind in ("weight", "bias"):
                self._add(local + "mlp.linear_fc1." + kind, gguf + "ffn_up." + kind)
                self._add(local + "mlp.linear_fc2." + kind, gguf + "ffn_down." + kind)
        for role, gguf in (
            ("norm", "v.post_ln"),
            ("linear_fc1", "mm.0"),
            ("linear_fc2", "mm.2"),
        ):
            for kind in ("weight", "bias"):
                self._add("model.visual.merger." + role + "." + kind, gguf + "." + kind)


def open_source(path: str | Path, *, config_path: str | Path | None = None,
                root: str | Path | None = None) -> GGUFSource:
    return GGUFSource(path, config_path=config_path, root=root)


def open_vision_source(path: str | Path, text_hidden: int, *,
                       root: str | Path | None = None) -> VisionGGUFSource:
    return VisionGGUFSource(path, text_hidden, root=root)
