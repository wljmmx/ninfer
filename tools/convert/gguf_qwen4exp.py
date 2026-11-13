"""Qwen4Exp GGUF -> NInfer logical source (text component).

Why this exists: when a Qwen4Exp distribution ships only as GGUF, the converter's normal input
(Safetensors + config.json) is unavailable. This module presents a GGUF as the same duck-typed
source the Qwen4Exp builder already consumes (`config`, `root`, `weight_map`, `describe`,
`read_flat`), so `tools/convert/qwen4_exp.py` maps and quantizes exactly as it does for a
Safetensors checkpoint.

Three things make that non-trivial, and each is answered here rather than in the recipe:

* Names. GGUF uses llama.cpp's `blk.N.*` spelling. The mapping below is the inverse of llama.cpp's
  own QWEN4EXP tensor map (conversion/qwen4exp.py + gguf-py's TensorNameMap, read out of that
  map), not a guess.
* Merged tensors. llama.cpp fuses several weights for one matmul - expert gate|up
  (`ffn_gate_up_exps`), the indexer q|k projection, the linear-attention qkv|z pair
  (`attn_qkv_gate`), beta|alpha (`ssm_beta_alpha_cat`), hyper-connection down|inject
  (`hc_*_down_inject`). The builder asks for the unfused projections, so they are split back by
  row range here.
* Head order. For linear attention llama.cpp stores V heads in *tiled* order
  (`_LinearAttentionVReorderBase`), while NInfer's kernels read the HF *grouped* order. The
  inverse permutation is applied whenever `linear_num_key_heads != linear_num_value_heads`.

Scope: text. Vision lives in a separate mmproj and MTP has its own MTP-only conversion; asking
for those components from a GGUF is refused with that reason instead of producing a silently
incomplete artifact.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .sources.gguf import open_gguf

_TEXT_ARCH = "qwen4exp"


# ------------------------------------------------------------------------------------- views
# The view toolkit is shared with the Qwen3.5 dense adapter and lives in `gguf_views`, so both
# sources read tensor ranges the same way.

from .gguf_views import (  # noqa: E402  (after the architecture constant, with the other tables)
    View,
    _ArrayView,
    _Cols,
    _ConcatRows,
    _Direct,
    _FlatPermute,
    _Reshape,
    _RowPermute,
    _Rows,
    _empty,
    tiled_to_grouped,
)

# ------------------------------------------------------------------- head-order permutation


def _ple_layers(cfg: dict) -> list[int]:
    """0-based PLE layer ids. The config carries HF's 1-based ids (qwen4_exp.text_config
    subtracts one); the mapping loop indexes layers from zero."""
    return [int(i) - 1 for i in (cfg.get("ple_layer_ids") or [])]


# ---------------------------------------------------------------------------------- the source


class GGUFSource:
    """A GGUF file presented as the converter's logical source."""

    def __init__(self, path: str | Path, *, config_path: str | Path | None = None,
                 root: str | Path | None = None) -> None:
        self.path = Path(path)
        # A split set (name-0000N-of-0000M.gguf) is opened as one source: the metadata lives in
        # the part that carries it, the tensors are spread over all parts.
        self.file = open_gguf(self.path)
        sidecar = Path(config_path) if config_path else None
        if sidecar is None and (self.path.parent / "config.json").is_file():
            sidecar = self.path.parent / "config.json"
        self.config = build_config(self.file, sidecar)
        self.root = Path(root) if root else (sidecar.parent if sidecar else self.path.parent)
        self._views: dict[str, View] = {}
        self._build_views()
        self._attach_ple_table()
        self.weight_map = {name: name for name in self._views}

    @classmethod
    def for_mtp(cls, path: str | Path, text_config: dict, *,
                root: str | Path | None = None) -> "GGUFSource":
        """A standalone MTP GGUF presented as HF `mtp.*` names.

        llama.cpp writes the next-token-prediction head as one extra block (`blk.<block_count-1>`,
        with the `nextn.*` tensors) plus a copy of the token embedding and the output head. `block_count`
        therefore counts that block, so reading the file as a text model would invent `block_count`
        text layers and find neither PLE nor GDN geometry. This view exposes only the draft head,
        under the target's own derived geometry, and deliberately omits `token_embd`/`output`: the
        engine reuses the text component's embedding and LM head, so keeping them would store ~1.3 GB
        of the same BF16 words twice.
        """
        source = cls.__new__(cls)
        source.path = Path(path)
        source.file = open_gguf(source.path)
        source.config = {"text_config": dict(text_config)}
        source.root = Path(root) if root else source.path.parent
        source._views = {}
        source._build_mtp_views()
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
        # copy=True: torch refuses a read-only buffer, and the view may hand back a slice of one.
        return torch.from_numpy(np.array(values, dtype=np.float32, copy=True)).to(torch.bfloat16)

    # -- mapping --------------------------------------------------------
    def _has(self, name: str) -> bool:
        return name in self.file.tensors

    def _direct(self, name: str) -> View:
        return _Direct(self.file, name)

    def _add(self, name: str, view: View) -> None:
        self._views[name] = view

    def _build_views(self) -> None:
        cfg = self.config["text_config"]
        layers = cfg["num_hidden_layers"]
        hidden = cfg["hidden_size"]
        experts = cfg["num_experts"]
        inter = cfg["moe_intermediate_size"]
        nk = cfg["linear_num_key_heads"]
        nv = cfg["linear_num_value_heads"]
        dk = cfg["linear_key_head_dim"]
        dv = cfg["linear_value_head_dim"]
        tiled = nk != nv

        # ---- globals
        if not self._has("token_embd.weight"):
            raise ValueError(f"{self.path}: GGUF is missing token_embd.weight")
        self._add("model.language_model.embed_tokens.weight", self._direct("token_embd.weight"))
        if not self._has("output.weight"):
            raise ValueError(
                f"{self.path}: GGUF has no output.weight; NInfer stores the output head as its "
                "own physical parameter and a tied head is not accepted by the converter"
            )
        self._add("lm_head.weight", self._direct("output.weight"))
        for gguf_name, suffix in (
            ("output_hc_norm", "hc_norm"),
            ("output_hc_down", "input_mix_weight_down"),
            ("output_hc_up", "input_mix_weight_up"),
        ):
            if self._has(gguf_name + ".weight"):
                self._add(f"model.language_model.hyper_connection_mixer.{suffix}.weight",
                          self._direct(gguf_name + ".weight"))

        layer_types = cfg["layer_types"]
        for layer in range(layers):
            prefix = f"blk.{layer}."
            target = f"model.language_model.layers.{layer}."
            linear = layer_types[layer] == "linear_attention"

            self._hc_pair(prefix, target + "attn_hyper_connection.", hidden, cfg,
                          down="hc_attn_down", up="hc_attn_up", norm="hc_attn_norm",
                          inject="hc_attn_inject", merged="hc_attn_down_inject")
            self._hc_pair(prefix, target + "mlp_hyper_connection.", hidden, cfg,
                          down="hc_ffn_down", up="hc_ffn_up", norm="hc_ffn_norm",
                          inject="hc_ffn_inject", merged="hc_ffn_down_inject")

            for gguf_name, suffix in (
                ("ffn_gate_inp", "mlp.gate"),
                ("ffn_gate_inp_shexp", "mlp.shared_expert_gate"),
                ("ffn_gate_shexp", "mlp.shared_expert.gate_proj"),
                ("ffn_up_shexp", "mlp.shared_expert.up_proj"),
                ("ffn_down_shexp", "mlp.shared_expert.down_proj"),
            ):
                if self._has(prefix + gguf_name + ".weight"):
                    view = self._direct(prefix + gguf_name + ".weight")
                    if suffix == "mlp.shared_expert_gate" and len(view.shape) == 1:
                        # llama.cpp stores this router bias as a vector; the builder wants the
                        # 1xH matrix the HF checkpoint carries.
                        view = _Reshape(view, (1, view.shape[0]))
                    self._add(target + suffix + ".weight", view)

            fused_name = prefix + "ffn_gate_up_exps.weight"
            gate_name = prefix + "ffn_gate_exps.weight"
            up_name = prefix + "ffn_up_exps.weight"
            if self._has(fused_name) or (self._has(gate_name) and self._has(up_name)):
                self._add(target + "mlp.experts.gate_up_proj",
                          _FusedExperts(
                              self._direct(fused_name) if self._has(fused_name) else None,
                              self._direct(gate_name) if self._has(gate_name) else None,
                              self._direct(up_name) if self._has(up_name) else None,
                              experts, inter, hidden,
                          ))
            down_name = prefix + "ffn_down_exps.weight"
            if self._has(down_name):
                self._add(target + "mlp.experts.down_proj", self._direct(down_name))
            if linear:
                self._gdn(prefix, target, cfg, tiled, nk, nv, dk, dv)
            else:
                self._qsa(prefix, target, cfg)

            if layer in _ple_layers(cfg):
                self._ple(prefix, target, cfg)

    def _hc_pair(self, prefix: str, target: str, hidden: int, cfg: dict, *, down: str, up: str,
                 norm: str, inject: str, merged: str) -> None:
        c = cfg["hc_count"]
        r = cfg["hc_lowrank"]
        if self._has(prefix + norm + ".weight"):
            self._add(target + "hc_norm.weight", self._direct(prefix + norm + ".weight"))
        if self._has(prefix + up + ".weight"):
            self._add(target + "input_mix_weight_up.weight",
                      self._direct(prefix + up + ".weight"))
        if self._has(prefix + merged + ".weight"):
            block = self._direct(prefix + merged + ".weight")
            self._add(target + "input_mix_weight_down.weight", _Rows(block, 0, r, c * hidden))
            self._add(target + "block_inject_weight.weight", _Rows(block, r, c, c * hidden))
        else:
            if self._has(prefix + down + ".weight"):
                self._add(target + "input_mix_weight_down.weight",
                          self._direct(prefix + down + ".weight"))
            if self._has(prefix + inject + ".weight"):
                self._add(target + "block_inject_weight.weight",
                          self._direct(prefix + inject + ".weight"))

    def _moe(self, prefix: str, target: str, cfg: dict, hidden: int) -> None:
        """The routed/shared FFN of one block, in the HF spelling `qwen4_exp.Builder.moe` reads."""
        experts = cfg["num_experts"]
        inter = cfg["moe_intermediate_size"]
        for gguf_name, suffix in (
            ("ffn_gate_inp", "mlp.gate"),
            ("ffn_gate_inp_shexp", "mlp.shared_expert_gate"),
            ("ffn_gate_shexp", "mlp.shared_expert.gate_proj"),
            ("ffn_up_shexp", "mlp.shared_expert.up_proj"),
            ("ffn_down_shexp", "mlp.shared_expert.down_proj"),
        ):
            if self._has(prefix + gguf_name + ".weight"):
                view = self._direct(prefix + gguf_name + ".weight")
                if suffix == "mlp.shared_expert_gate" and len(view.shape) == 1:
                    # llama.cpp stores this router bias as a vector; the builder wants the 1xH
                    # matrix the HF checkpoint carries.
                    view = _Reshape(view, (1, view.shape[0]))
                self._add(target + suffix + ".weight", view)
        fused_name = prefix + "ffn_gate_up_exps.weight"
        gate_name = prefix + "ffn_gate_exps.weight"
        up_name = prefix + "ffn_up_exps.weight"
        if self._has(fused_name) or (self._has(gate_name) and self._has(up_name)):
            self._add(target + "mlp.experts.gate_up_proj",
                      _FusedExperts(
                          self._direct(fused_name) if self._has(fused_name) else None,
                          self._direct(gate_name) if self._has(gate_name) else None,
                          self._direct(up_name) if self._has(up_name) else None,
                          experts, inter, hidden,
                      ))
        down_name = prefix + "ffn_down_exps.weight"
        if self._has(down_name):
            self._add(target + "mlp.experts.down_proj", self._direct(down_name))

    def _build_mtp_views(self) -> None:
        cfg = self.config["text_config"]
        hidden = int(cfg["hidden_size"])
        block = None
        for name in self.file.tensors:
            if name.endswith(".nextn.eh_proj.weight"):
                block = name.rsplit(".nextn.", 1)[0]
                break
        if block is None:
            raise ValueError(f"{self.path}: GGUF carries no nextn draft block")
        prefix = block + "."
        target = "mtp.layers.0."
        self._hc_pair(prefix, target + "attn_hyper_connection.", hidden, cfg,
                      down="hc_attn_down", up="hc_attn_up", norm="hc_attn_norm",
                      inject="hc_attn_inject", merged="hc_attn_down_inject")
        self._hc_pair(prefix, target + "mlp_hyper_connection.", hidden, cfg,
                      down="hc_ffn_down", up="hc_ffn_up", norm="hc_ffn_norm",
                      inject="hc_ffn_inject", merged="hc_ffn_down_inject")
        self._qsa(prefix, target, cfg)
        self._moe(prefix, target, cfg, hidden)
        # eh_proj is the fused [h,2h] matrix of the HF pair: llama.cpp writes one parent, the HF
        # checkpoint (and the engine's stem, whose registered kernels stop at [h,h]) keeps
        # fc_embedding over columns [0,h) and fc_hidden over [h,2h).
        for gguf_name, suffix, start in (
            ("nextn.eh_proj", "fc_embedding", 0),
            ("nextn.enorm", "pre_fc_norm_embedding", None),
            ("nextn.hnorm", "pre_fc_norm_hidden", None),
        ):
            if not self._has(prefix + gguf_name + ".weight"):
                raise ValueError(f"{self.path}: {prefix}{gguf_name}.weight is missing; the draft "
                                 "head needs its whole stem")
            view = self._direct(prefix + gguf_name + ".weight")
            if suffix == "fc_embedding":
                fused = self._direct(prefix + "nextn.eh_proj.weight")
                if len(fused.shape) != 2 or fused.shape[1] != 2 * hidden:
                    raise ValueError(f"{self.path}: {prefix}nextn.eh_proj.weight is not [h,2h]")
                self._add("mtp.fc_embedding.weight", _Cols(fused, 0, hidden, 2 * hidden))
                self._add("mtp.fc_hidden.weight", _Cols(fused, hidden, hidden, 2 * hidden))
                continue
            self._add(f"mtp.{suffix}.weight", view)
        # The head's final correction is the same norm+pair shape a block carries, without an
        # injection stream.
        for gguf_name, suffix in (
            ("nextn.hc_head_norm", "hc_norm"),
            ("nextn.hc_head_down", "input_mix_weight_down"),
            ("nextn.hc_head_up", "input_mix_weight_up"),
        ):
            if not self._has(prefix + gguf_name + ".weight"):
                raise ValueError(f"{self.path}: {prefix}{gguf_name}.weight is missing; the draft "
                                 "head needs its final hyper-connection mixer")
            self._add(f"mtp.hyper_connection_mixer.{suffix}.weight",
                      self._direct(prefix + gguf_name + ".weight"))

    def _qsa(self, prefix: str, target: str, cfg: dict) -> None:
        hidden = cfg["hidden_size"]
        iq = cfg["indexer_n_heads"] * cfg["indexer_head_dim"]
        ik = cfg["indexer_kv_heads"] * cfg["indexer_head_dim"]
        # Q and the output gate are ONE (2*Q, H) parent with head-interleaved rows, which is what
        # llama.cpp keeps under attn_q; the builder separates them by row range.
        if not self._has(prefix + "attn_q.weight"):
            raise ValueError(f"{self.path}: {prefix}attn_q.weight is missing")
        self._add(target + "self_attn.q_proj.weight", self._direct(prefix + "attn_q.weight"))
        for suffix, gguf_name in (
            ("k_proj", "attn_k"),
            ("v_proj", "attn_v"),
            ("o_proj", "attn_output"),
        ):
            if self._has(prefix + gguf_name + ".weight"):
                self._add(target + f"self_attn.{suffix}.weight",
                          self._direct(prefix + gguf_name + ".weight"))
        for suffix, gguf_name in (("q_norm", "attn_q_norm"), ("k_norm", "attn_k_norm")):
            if self._has(prefix + gguf_name + ".weight"):
                self._add(target + f"self_attn.{suffix}.weight",
                          self._direct(prefix + gguf_name + ".weight"))
        # The indexer q|k projection is one logical tensor in the HF checkpoint; llama.cpp
        # splits it into indexer.q_proj / indexer.k_proj, so it is fused back here.
        q_name = prefix + "indexer.q_proj.weight"
        k_name = prefix + "indexer.k_proj.weight"
        if self._has(q_name) and self._has(k_name):
            self._add(target + "self_attn.indexer.index_qk_proj.weight",
                      _ConcatRows([
                          (_Rows(self._direct(q_name), 0, iq, hidden), 0, iq),
                          (_Rows(self._direct(k_name), 0, ik, hidden), 0, ik),
                      ]))
        for suffix, gguf_name in (("q_layernorm", "indexer.q_norm"),
                                  ("k_layernorm", "indexer.k_norm")):
            if self._has(prefix + gguf_name + ".weight"):
                self._add(target + f"self_attn.indexer.{suffix}.weight",
                          self._direct(prefix + gguf_name + ".weight"))

    def _gdn(self, prefix: str, target: str, cfg: dict, tiled: bool, nk: int, nv: int, dk: int,
             dv: int) -> None:
        hidden = cfg["hidden_size"]
        taps = cfg["linear_conv_kernel_dim"]
        q = nk * dk
        v = nv * dv
        c = 2 * q + v
        per_key = nv // nk if nk else 1
        perm_v = tiled_to_grouped(v, nk, per_key, dv) if tiled else None
        name = target + "linear_attn."
        merged_name = prefix + "attn_qkv_gate.weight"
        qkv_name = prefix + "attn_qkv.weight"
        z_name = prefix + "attn_gate.weight"
        if self._has(merged_name):
            merged = self._direct(merged_name)
            qkv = _Rows(merged, 0, c, hidden)
            z = _Rows(merged, c, v, hidden)
            if perm_v is not None:
                z = _RowPermute(z, perm_v)
            self._add(name + "in_proj_z.weight", z)
        elif self._has(qkv_name):
            qkv = _Rows(self._direct(qkv_name), 0, c, hidden)
            if self._has(z_name):
                z = self._direct(z_name)
                if perm_v is not None:
                    z = _RowPermute(z, perm_v)
                self._add(name + "in_proj_z.weight", z)
        else:
            raise ValueError(f"{self.path}: {prefix}attn_qkv.weight is missing")
        # q | k | v, with the V section put back in grouped order.
        if perm_v is not None:
            qk = _Rows(qkv, 0, 2 * q, hidden)
            vpart = _RowPermute(_Rows(qkv, 2 * q, v, hidden), perm_v)
            qkv = _ConcatRows([(qk, 0, 2 * q), (vpart, 0, v)])
        self._add(name + "in_proj_qkv.weight", qkv)
        for suffix, gguf_name in (("in_proj_a", "ssm_alpha"), ("in_proj_b", "ssm_beta")):
            if self._has(prefix + gguf_name + ".weight"):
                weight = self._direct(prefix + gguf_name + ".weight")
                if tiled:
                    weight = _RowPermute(weight, tiled_to_grouped(nv, nk, per_key, 1))
                self._add(name + f"{suffix}.weight", weight)
        if self._has(prefix + "ssm_conv1d.weight"):
            raw = self._direct(prefix + "ssm_conv1d.weight")
            # llama.cpp squeezed [C, 1, taps] to [C, taps]; a size-1 axis moves no element, so it
            # only has to be reported again - except the V channels are tiled, and that
            # permutation needs the whole kernel at once.
            if tiled:
                rows = raw.read(0, c * taps).reshape(c, taps)
                qk_channels = 2 * q
                rows = np.concatenate([rows[:qk_channels], rows[qk_channels:][perm_v]])
                conv = _ArrayView(rows.reshape(c, 1, taps))
            else:
                conv = _Reshape(raw, (c, 1, taps))
            self._add(name + "conv1d.weight", conv)
        for suffix, gguf_name in (("A_log", "ssm_a"), ("dt_bias", "ssm_dt")):
            # llama.cpp names these without the weight suffix (ssm_a, ssm_dt.bias); the earlier
            # spelling is kept as a fallback for files produced by other converters.
            for candidate in (gguf_name, gguf_name + ".weight", gguf_name + ".bias"):
                if self._has(prefix + candidate):
                    weight = self._direct(prefix + candidate)
                    if tiled:
                        weight = _FlatPermute(weight, tiled_to_grouped(nv, nk, per_key, 1))
                    self._add(name + suffix, weight)
                    break
            else:
                raise ValueError(f"{self.path}: {prefix}{gguf_name} is missing")
        if self._has(prefix + "ssm_norm.weight"):
            self._add(name + "norm.weight", self._direct(prefix + "ssm_norm.weight"))
        if self._has(prefix + "ssm_out.weight"):
            out = self._direct(prefix + "ssm_out.weight")
            if tiled:
                out = _ColPermute(out, tiled_to_grouped(v, nk, per_key, dv))
            self._add(name + "out_proj.weight", out)

    def _ple(self, prefix: str, target: str, cfg: dict) -> None:
        hidden = cfg["hidden_size"]
        c = cfg["hc_count"]
        taps = cfg["ple_conv_kernel_size"]
        name = target + "ple."
        for suffix, gguf_name in (
            ("key_proj", "ple_key"),
            ("value_proj", "ple_value"),
            ("norm_key", "ple_norm_key"),
            ("norm_query", "ple_norm_query"),
            ("norm_conv", "ple_norm_conv"),
        ):
            if self._has(prefix + gguf_name + ".weight"):
                self._add(name + suffix + ".weight",
                          self._direct(prefix + gguf_name + ".weight"))
        if self._has(prefix + "ple_conv1d.weight"):
            self._add(name + "conv1d.weight",
                      _Reshape(self._direct(prefix + "ple_conv1d.weight"),
                               (hidden * c, 1, taps)))
        # The PLE hash controls are GGUF *metadata* arrays (u64), not tensors: llama.cpp writes
        # them with add_ple_layer_multipliers / head_offsets / head_vocab_sizes. HF keeps them as
        # int64, and u64 -> int64 is exact for every value below 2^63, which these multipliers
        # (45-bit range) and offsets are.
        controls = {
            "layer_multipliers": ("ple.layer_multipliers", cfg["ngram_size"]),
            "ngram_heads_vocab_sizes": ("ple.head_vocab_sizes",
                                        (cfg["ngram_size"] - 1) * cfg["heads_per_ngram"]),
            "ngram_heads_offsets": ("ple.head_offsets",
                                    (cfg["ngram_size"] - 1) * cfg["heads_per_ngram"]),
        }
        for role, (key, expected) in controls.items():
            values = _meta(self.file, key)
            if values is None:
                raise ValueError(
                    f"{self.path}: GGUF metadata {_TEXT_ARCH}.{key} is missing; the PLE hash "
                    "controls are required by the layer that carries PLE"
                )
            array = np.asarray(values, dtype=np.int64)
            if array.shape != (expected,):
                raise ValueError(
                    f"{self.path}: {_TEXT_ARCH}.{key} has {array.shape}, expected ({expected},)"
                )
            self._add(name + "ple_embedding." + role, _ArrayView(array, "I64"))

    def _attach_ple_table(self) -> None:
        cfg = self.config["text_config"]
        layers = _ple_layers(cfg)
        if not layers:
            return
        name = "per_layer_token_embd.weight"
        if not self._has(name):
            raise ValueError(
                f"{self.path}: the config marks PLE layer(s) {layers} but the GGUF has no "
                "per_layer_token_embd.weight"
            )
        table = self._direct(name)
        rows_total, ed = table.shape
        parts = int(cfg["split_ngram_parts"])
        if rows_total % parts != 0:
            raise ValueError(
                f"{self.path}: the PLE table has {rows_total} rows, not divisible by "
                f"split_ngram_parts={parts}; GGUF merged the original shards, so their row "
                "split cannot be recovered - set split_ngram_parts in the sidecar config"
            )
        rows = rows_total // parts
        base = f"model.language_model.layers.{int(layers[0])}.ple.ple_embedding." \
               "ngram_embedding."
        for index in range(parts):
            self._add(base + f"shard_{index}.weight", _Rows(table, index * rows, rows, ed))


# ---------------------------------------------------------------------------- config synthesis

_HF_TEXT_FIELDS = (
    "hidden_size",
    "vocab_size",
    "num_hidden_layers",
    "max_position_embeddings",
    "hc_count",
    "hc_lowrank",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "linear_num_key_heads",
    "linear_num_value_heads",
    "linear_key_head_dim",
    "linear_value_head_dim",
    "linear_conv_kernel_dim",
    "indexer_n_heads",
    "indexer_kv_heads",
    "indexer_head_dim",
    "indexer_compress_ratio",
    "indexer_budget",
    "num_experts",
    "num_experts_per_tok",
    "moe_intermediate_size",
    "shared_expert_intermediate_size",
)


def _meta(file: GGUFFile, key: str):
    return file.get(f"{_TEXT_ARCH}.{key}")


def _need(file: GGUFFile, key: str, value):
    if value is None or (isinstance(value, (int, float)) and value <= 0):
        raise ValueError(
            f"{file.path}: cannot reconstruct {_TEXT_ARCH}.{key} from the GGUF metadata; pass "
            "--gguf-config with the checkpoint's config.json"
        )
    return value


def build_config(file: GGUFFile, sidecar: Path | None) -> dict:
    """Reconstruct the config.json shape qwen4_exp.text_config validates.

    A GGUF carries the qwen4exp hyperparameters llama.cpp writes from the HF config, so the
    geometry is recoverable. What it does not carry (the rope_parameters subtree, the raw
    compression ratio, vision/mtp subtrees) comes from the optional sidecar config.json. A field
    neither source supplies is a refusal - never a default that would silently change the math.
    """
    for name in file.tensors:
        if name.endswith(".nextn.eh_proj.weight"):
            raise ValueError(
                f"{file.path}: this GGUF carries llama.cpp's next-token-prediction head "
                f"({name.rsplit('.nextn.', 1)[0]}); its block_count counts the draft block, so it "
                "is not a text checkpoint - pass it with --mtp-gguf instead"
            )
    arch = str(file.get("general.architecture", ""))
    if arch and arch != _TEXT_ARCH:
        raise ValueError(
            f"{file.path}: GGUF architecture is {arch!r}, not {_TEXT_ARCH!r}; this path converts "
            "Qwen4Exp only"
        )
    base: dict = {}
    if sidecar is not None:
        base = json.loads(Path(sidecar).read_text())
    text = dict(base.get("text_config") or {})
    config = dict(base)
    config.setdefault("architectures", ["Qwen4ExpForCausalLM"])

    layers = _need(file, "block_count", _meta(file, "block_count"))
    hidden = _need(file, "embedding_length", _meta(file, "embedding_length"))
    heads = _need(file, "attention.head_count", _meta(file, "attention.head_count"))
    head_dim = _need(file, "attention.key_length", _meta(file, "attention.key_length"))
    kv_heads = _need(file, "attention.head_count_kv", _meta(file, "attention.head_count_kv"))
    hc_count = _need(file, "hyper_connection.count", _meta(file, "hyper_connection.count"))
    hc_rank = _need(file, "hyper_connection.low_rank", _meta(file, "hyper_connection.low_rank"))
    experts = _need(file, "expert_count", _meta(file, "expert_count"))
    experts_used = _need(file, "expert_used_count", _meta(file, "expert_used_count"))
    moe_inter = _need(file, "expert_feed_forward_length",
                      _meta(file, "expert_feed_forward_length"))
    shared_inter = _need(file, "expert_shared_feed_forward_length",
                         _meta(file, "expert_shared_feed_forward_length"))
    nk = _need(file, "ssm.group_count", _meta(file, "ssm.group_count"))
    nv = _need(file, "ssm.time_step_rank", _meta(file, "ssm.time_step_rank"))
    dk = _need(file, "ssm.state_size", _meta(file, "ssm.state_size"))
    inner = _need(file, "ssm.inner_size", _meta(file, "ssm.inner_size"))
    conv_kernel = _need(file, "ssm.conv_kernel", _meta(file, "ssm.conv_kernel"))
    idx_heads = _need(file, "attention.indexer.head_count",
                      _meta(file, "attention.indexer.head_count"))
    idx_dim = _need(file, "attention.indexer.key_length",
                    _meta(file, "attention.indexer.key_length"))
    idx_topk = _need(file, "attention.indexer.top_k", _meta(file, "attention.indexer.top_k"))

    ratios = _meta(file, "attention.compress_ratios")
    if ratios is None or len(ratios) < layers:
        raise ValueError(
            f"{file.path}: GGUF has no per-layer attention.compress_ratios; the sparse/dense "
            "layer layout is required"
        )
    layer_types = ["full_attention" if int(ratios[i]) > 0 else "linear_attention"
                   for i in range(layers)]

    # Derived from the file; the sidecar (an authoritative HF config.json) wins wherever it
    # speaks, because the GGUF's flattened metadata cannot express every field faithfully.
    derived = {
        "hidden_size": int(hidden),
        "num_hidden_layers": int(layers),
        "num_attention_heads": int(heads),
        "head_dim": int(head_dim),
        "num_key_value_heads": int(kv_heads),
        "hc_count": int(hc_count),
        "hc_lowrank": int(hc_rank),
        "num_experts": int(experts),
        "num_experts_per_tok": int(experts_used),
        "moe_intermediate_size": int(moe_inter),
        "shared_expert_intermediate_size": int(shared_inter),
        "linear_num_key_heads": int(nk),
        "linear_num_value_heads": int(nv),
        "linear_key_head_dim": int(dk),
        "linear_value_head_dim": int(inner) // int(nv),
        "linear_conv_kernel_dim": int(conv_kernel),
        "indexer_n_heads": int(idx_heads),
        "indexer_head_dim": int(idx_dim),
        "indexer_budget": int(idx_topk),
        "layer_types": layer_types,
    }
    # The indexer's key projection is its own geometry (Qwen3.8-Flash-Next: one key head), not
    # the attention's `head_count_kv`; the tensor is the authority when the sidecar is silent.
    indexer_kv = None
    for name, info in file.tensors.items():
        if name.endswith(".indexer.k_proj.weight"):
            indexer_kv = int(info.shape[0]) // int(idx_dim)
            break
    derived["indexer_kv_heads"] = int(indexer_kv if indexer_kv else kv_heads)
    ple_layers = _meta(file, "ple.layers") or []
    if ple_layers:
        derived["ple_layer_ids"] = [int(i) + 1 for i in ple_layers]  # HF is 1-based
        derived["ngram_size"] = int(_meta(file, "ple.ngram_size"))
        derived["heads_per_ngram"] = int(_meta(file, "ple.heads_per_ngram"))
        derived["ple_conv_kernel_size"] = int(_meta(file, "ple.conv_kernel"))
        # ple_embed_dim is the width of the PLE *projections*; the GGUF's
        # embedding_length_per_layer_input is the width of the hash table, a different number.
        for name, info in file.tensors.items():
            if name.endswith(".ple_key.weight"):
                derived["ple_embed_dim"] = int(info.shape[1])
                break
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
    text.setdefault("rms_norm_eps",
                    float(_meta(file, "attention.layer_norm_rms_epsilon") or 1e-6))
    text.setdefault("hidden_act", "silu")
    text.setdefault("output_gate_type", "sigmoid")
    text.setdefault("tie_word_embeddings", False)
    text.setdefault("attention_bias", False)
    text.setdefault("mamba_ssm_dtype", "float32")
    text.setdefault("norm_topk_prob", True)
    if ple_layers:
        text.setdefault("split_ngram_parts", 1)
    if "eos_token_id" not in text:
        eos = _meta(file, "ple.eos_token_id")
        if eos is not None:
            text["eos_token_id"] = int(eos)
    if "rope_parameters" not in text:
        base_freq = _meta(file, "rope.freq_base")
        if base_freq is None:
            raise ValueError(
                f"{file.path}: GGUF has no rope.freq_base and the sidecar has no "
                "rope_parameters; rotary geometry is required"
            )
        sections = _meta(file, "rope.dimension_sections")
        text["rope_parameters"] = {
            "rope_type": "default",
            "rope_theta": float(base_freq),
            "mrope_section": [int(x) for x in sections] if sections is not None else [0, 0, 0],
        }
    config["text_config"] = text
    missing = [key for key in _HF_TEXT_FIELDS if not text.get(key)]
    if missing:
        raise ValueError(
            f"{file.path}: cannot reconstruct {missing}; pass --gguf-config with the "
            "checkpoint's config.json"
        )
    return config


def open_source(path: str | Path, *, config_path: str | Path | None = None,
                root: str | Path | None = None) -> GGUFSource:
    return GGUFSource(path, config_path=config_path, root=root)
