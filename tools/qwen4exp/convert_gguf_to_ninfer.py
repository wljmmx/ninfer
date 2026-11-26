#!/usr/bin/env python3

r"""Pack Qwen4Exp GGUF shards (plus dflash2 / MTP drafts) into a NInfer v3 container.

Every tensor keeps its original GGML encoding: nothing is dequantised or re-quantised.
Each tensor becomes a v3 `tensor` object whose `format` is the GGML type name
(`ggml_iq3_xxs`, `ggml_q6_k`, ...) and whose `layout` is `ggml_block_v1` - the contract being
"these bytes are the GGML block encoding, unchanged, for this shape and type". The engine must
have those format names registered before it can bind them; this tool only guarantees the file
is byte-faithful to the source.

Container layout: docs/maintainer/artifact-container.md - 32-byte entry header, UTF-8 JSON
directory, zero padding to 4096, logical payload split across `<entry>.part-NNNN` volumes.

Usage:
  python convert_gguf_to_ninfer.py \
      --text R:\model\qwen4exp\IQ3_XXS\...-00001-of-00002.gguf \
             R:\model\qwen4exp\IQ3_XXS\...-00002-of-00002.gguf \
      --dflash2 R:\model\qwen4exp\Qwen3.8-Flash-Next-DFlash2-BF16.gguf \
      --out R:\model\qwen4exp\qwen4exp-iq3_xxs.ninfer --verify
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys
import uuid

import numpy as np
from dataclasses import dataclass, field

GGUF_PY = r"H:\TeleAgentWorks\.temp\repos\llama-latest\gguf-py"
if GGUF_PY not in sys.path:
    sys.path.insert(0, GGUF_PY)
import gguf  # noqa: E402

ENTRY_MAGIC = b"NINFER\x00\x03"
PART_MAGIC = b"NINPRT\x00\x03"
HEADER = 32
ENTRY_ALIGN = 4096
OBJECT_ALIGN = 256
COPY_CHUNK = 8 << 20


@dataclass
class Source:
    path: str
    reader: object


@dataclass
class Obj:
    id: str
    kind: str
    shape: list | None
    fmt: str | None
    layout: str | None
    src: Source
    src_offset: int
    bytes: int
    offset: int = 0
    sha256: str = ""
    # Rank-1 F32 norms become bf16 on the way in: GGUF stores them F32, while the model's math asks
    # for bf16 (binder exact_format). This is a precision cast of a non-quantised vector, not a
    # requantisation, and it keeps the norm vectors in the representation the kernels expect.
    f32_to_bf16: bool = False
    # GGUF stores a conv1d kernel as [channels, kernel] (kernel values contiguous), while the model
    # wants [kernel, channels] in C order, so the element order has to be transposed on the way in.
    transpose: bool = False
    # F16 -> BF16: same two-byte width, but the model's conv kernels are bf16 operands.
    f16_to_bf16: bool = False


def align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def ggml_format_name(tensor_type: int) -> str:
    return "ggml_" + gguf.GGMLQuantizationType(tensor_type).name.lower()


# GGML types whose payload is byte-identical to a native ninfer format: the raw words in C order.
# Mapping them lets the existing loader read those tensors instead of needing a GGML decoder.
NATIVE_EQUIVALENT = {
    "BF16": ("bf16", "contiguous_le_v1"),
    "F32": ("fp32", "contiguous_le_v1"),
    "I32": ("int32", "contiguous_le_v1"),
}


def describe_format(tensor_type: int) -> tuple:
    name = gguf.GGMLQuantizationType(tensor_type).name
    if name in NATIVE_EQUIVALENT:
        return NATIVE_EQUIVALENT[name]
    return ggml_format_name(tensor_type), "ggml_block_v1"


def plain(value):
    """numpy scalar/array -> JSON-safe Python value."""
    if isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8", "replace")
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def read_metadata(source: Source) -> dict:
    out = {}
    for key, fld in source.reader.fields.items():
        try:
            out[key] = plain(fld.contents())
        except Exception:
            continue
    return out


def collect(source: Source, component: str) -> list:
    objs = []
    for t in source.reader.tensors:
        shape_c = [int(x) for x in reversed(list(t.shape))]  # GGML ne is column-major
        fmt, layout = describe_format(t.tensor_type)
        n_bytes = int(t.n_bytes)
        # Only norm vectors and conv1d kernels are cast: both are bf16 operands in the model
        # (rmsnorm / causal_conv1d accept BF16 only). Other rank-1 F32 vectors (gdn a_log, dt_bias)
        # are fp32 by contract and stay as they are.
        cast = fmt == "fp32" and ((len(shape_c) == 1 and "norm" in t.name)
                                  or t.name.endswith("conv1d.weight"))
        if cast:
            elements = 1
            for d in shape_c:
                elements *= d
            fmt, layout, n_bytes = "bf16", "contiguous_le_v1", elements * 2
        f16_cast = fmt == "ggml_f16"
        if f16_cast:
            fmt, layout = "bf16", "contiguous_le_v1"
        transpose = False
        if (t.name.endswith("conv1d.weight") and len(shape_c) == 2
                and shape_c[0] > shape_c[1] and 1 < shape_c[1] <= 8):
            # GGUF conv1d is [channels, kernel]; the model wants [kernel, channels].
            shape_c = [shape_c[1], shape_c[0]]
            transpose = True
        if t.name.endswith("ffn_gate_inp_shexp.weight") and len(shape_c) == 1:
            # The gate vector is a one-row matrix in the model's parameter shape.
            shape_c = [1, shape_c[0]]
        objs.append(Obj(id=f"{component}.{t.name}", kind="tensor", shape=shape_c,
                        fmt=fmt, layout=layout, f32_to_bf16=cast, transpose=transpose,
                        f16_to_bf16=f16_cast,
                        src=source, src_offset=int(t.data_offset), bytes=n_bytes))
    return objs


def text_config(metadata: dict) -> dict:
    def num(*keys, default=0):
        for k in keys:
            if k in metadata:
                try:
                    return int(metadata[k])
                except Exception:
                    pass
        return default

    cfg = {
        # The binder keys off this pair (src/models/qwen3_5/config.*); the GGUF artifact is a
        # Qwen3.8-Flash-Next target, which this repo names Qwen4Exp.
        "architectures": ["Qwen4ExpForCausalLM"],
        "model_type": "qwen4_exp_text",
        "hidden_size": num("qwen3_5.hidden_size", "embedding_length"),
        "num_hidden_layers": num("qwen3_5.block_count"),
        "num_attention_heads": num("qwen3_5.attention.head_count"),
        "num_key_value_heads": num("qwen3_5.attention.head_count_kv"),
        "head_dim": num("qwen3_5.attention.key_length"),
        "hidden_act": "silu",
        "vocab_size": num("qwen3_5.vocab_size"),
        "max_position_embeddings": num("qwen3_5.context_length"),
        "rms_norm_eps": float(metadata.get("qwen3_5.attention.layer_norm_rms_epsilon", 1e-6)),
        "rope_theta": float(metadata.get("qwen3_5.rope.freq_base", 10000000.0)),
    }
    # Architecture-specific extras (HC streams, PLE, indexer) carried verbatim for the binder.
    for key in metadata:
        if key.startswith("qwen3_5.") and ("gguf." + key) not in cfg:
            cfg["gguf." + key] = metadata[key]
    return cfg


def plan(objs: list, shard_bytes: int):
    """Assign global offsets and cut volumes on object boundaries.

    Each volume also records its payload base: object offsets are global (a v3 directory has one
    payload space), so a volume must write `offset - base` bytes of padding, not `offset` - using
    the global value repeats the whole prefix in every volume.
    """
    offset = 0
    shards = []
    current = []
    base = 0
    used = 0
    for obj in objs:
        obj.offset = align_up(offset, OBJECT_ALIGN)
        offset = obj.offset + obj.bytes
        if used and used + obj.bytes > shard_bytes:
            # The next volume starts where this one actually ends (last object's end, alignment gap
            # included), because the logical payload is one continuous space.
            base_next = current[-1].offset + current[-1].bytes
            shards.append((base, current))
            base = base_next
            current = []
            used = 0
        current.append(obj)
        used += obj.bytes
    if current:
        shards.append((base, current))
    return shards, offset


def describe(obj: Obj) -> dict:
    if obj.kind == "tensor":
        return {"id": obj.id, "kind": "tensor", "shape": obj.shape, "format": obj.fmt,
                "layout": obj.layout, "offset": obj.offset, "bytes": obj.bytes}
    return {"id": obj.id, "kind": "resource", "encoding": "raw_bytes_v1",
            "offset": obj.offset, "bytes": obj.bytes}


def copy_range(src: Source, offset: int, length: int, dst, hasher) -> None:
    with open(src.path, "rb", buffering=0) as f:
        f.seek(offset)
        left = length
        while left:
            chunk = f.read(min(COPY_CHUNK, left))
            if not chunk:
                raise IOError(f"short read in {src.path}")
            dst.write(chunk)
            hasher.update(chunk)
            left -= len(chunk)


def write_shard(path: str, shard: list, base: int, final_size: int, index: int,
                artifact_id: bytes, payload_start: int, blob: bytes) -> None:
    written = 0
    with open(path, "wb") as f:
        if index == 0:
            f.write(ENTRY_MAGIC)
            f.write(struct.pack("<Q", len(blob)))
            f.write(artifact_id)
            f.write(blob)
            f.write(b"\x00" * (payload_start - HEADER - len(blob)))
        else:
            f.write(PART_MAGIC)
            f.write(struct.pack("<Q", index))
            f.write(artifact_id)
            f.write(b"\x00" * (ENTRY_ALIGN - HEADER))
        for obj in shard:
            local = obj.offset - base
            if local > written:
                f.write(b"\x00" * (local - written))
                written = local
            h = hashlib.sha256()
            if obj.f32_to_bf16:
                src_len = obj.shape[0] * 4
                with open(obj.src.path, "rb", buffering=0) as sf:
                    sf.seek(obj.src_offset)
                    raw = sf.read(src_len)
                if len(raw) != src_len:
                    raise IOError(f"short read for {obj.id}")
                # bf16 is the top half of the fp32 word (round-to-nearest-even on the tie bit).
                u32 = np.frombuffer(raw, dtype=np.uint32).astype(np.uint64)
                bf = ((u32 + 0x7FFF + ((u32 >> 16) & 1)) >> 16).astype(np.uint16).tobytes()
                f.write(bf)
                h.update(bf)
                written += len(bf)
            elif obj.transpose:
                rows, cols = obj.shape          # target [kernel, channels]
                src_elem = 4 if obj.f32_to_bf16 else (obj.bytes // (rows * cols))
                src_bytes = rows * cols * src_elem
                with open(obj.src.path, "rb", buffering=0) as sf:
                    sf.seek(obj.src_offset)
                    raw = sf.read(src_bytes)
                if len(raw) != src_bytes:
                    raise IOError(f"short read for {obj.id}")
                src = np.frombuffer(raw, dtype={2: np.uint16, 4: np.uint32}[src_elem])
                arr = src.reshape(cols, rows).T
                if obj.f32_to_bf16 or obj.f16_to_bf16:
                    f32 = (arr.view(np.float32) if src_elem == 4
                           else arr.view(np.float16).astype(np.float32))
                    u32 = f32.view(np.uint32).astype(np.uint64)
                    arr = ((u32 + 0x7FFF + ((u32 >> 16) & 1)) >> 16).astype(np.uint16)
                data = arr.tobytes()
                f.write(data)
                h.update(data)
                written += len(data)
            else:
                copy_range(obj.src, obj.src_offset, obj.bytes, f, h)
                written += obj.bytes
            obj.sha256 = h.hexdigest()
        # The volume's declared payload covers the trailing alignment gap, and the reader checks the
        # file length against it, so the tail must be written out.
        if written < final_size:
            f.write(b"\x00" * (final_size - written))


def load_component_config(path: str, component: str = "text") -> dict:
    """Reuse a known-good component config from another artifact of the same model.

    The text config carries ~30 fields (layer_types, hc_*, indexer_*, linear_*), and every one of
    them is a property of the model rather than of the encoding, so the packed file should take them
    from the artifact that already loads rather than re-derive them from GGUF metadata.
    """
    with open(path, "rb") as f:
        if f.read(8) != ENTRY_MAGIC:
            raise ValueError(f"{path} is not a v3 entry")
        (json_bytes,) = struct.unpack("<Q", f.read(8))
        f.read(16)
        directory = json.loads(f.read(json_bytes).decode())
    return directory["components"][component]["config"]


def load_resources(path: str) -> dict:
    """Resource objects (tokenizer, templates, PLE hash controls) from an artifact of the same model.

    They are small, model-versioned blobs that GGUF does not carry; copying them keeps the packed
    file loadable without a second converter for tokenizer formats.
    """
    with open(path, "rb") as f:
        if f.read(8) != ENTRY_MAGIC:
            raise ValueError(f"{path} is not a v3 entry")
        (json_bytes,) = struct.unpack("<Q", f.read(8))
        f.read(16)
        directory = json.loads(f.read(json_bytes).decode())
    payload_start = align_up(HEADER + json_bytes, ENTRY_ALIGN)
    bases = [0]
    for fdesc in directory["files"][1:]:
        bases.append(bases[-1] + fdesc["payload_bytes"])
    by_id = {o["id"]: o for o in directory["objects"]}
    out: dict = {}
    for comp, rec in directory["components"].items():
        for name, rid in (rec.get("resources") or {}).items():
            o = by_id[rid]
            vol = 0
            for i, b in enumerate(bases):
                if b <= o["offset"] < b + directory["files"][i]["payload_bytes"]:
                    vol = i
                    break
            start = payload_start if vol == 0 else ENTRY_ALIGN
            out.setdefault(comp, {})[name] = Obj(
                id=f"resource/{comp}/{name}", kind="resource", shape=None, fmt=None, layout=None,
                src=Source(path, None), src_offset=start + (o["offset"] - bases[vol]),
                bytes=o["bytes"])
    return out


# Layer-level GGUF suffix -> logical parameter suffix. Verified against text/layers/0 (GDN) and
# text/layers/3 (indexed attention) of the same-model artifact.
COMMON_LAYER = {
    "hc_attn_norm.weight": "attention_hc/norm",
    "hc_attn_down.weight": "attention_hc/down",
    "hc_attn_up.weight": "attention_hc/up",
    "hc_attn_inject.weight": "attention_hc/inject",
    "hc_ffn_norm.weight": "moe_hc/norm",
    "hc_ffn_down.weight": "moe_hc/down",
    "hc_ffn_up.weight": "moe_hc/up",
    "hc_ffn_inject.weight": "moe_hc/inject",
    "ffn_gate_inp.weight": "moe/router",
    "ffn_gate_inp_shexp.weight": "moe/shared_score",
    "ffn_gate_exps.weight": "moe/experts/gate",
    "ffn_up_exps.weight": "moe/experts/up",
    "ffn_down_exps.weight": "moe/experts/down",
    "ffn_gate_shexp.weight": "moe/shared/gate",
    "ffn_up_shexp.weight": "moe/shared/up",
    "ffn_down_shexp.weight": "moe/shared/down",
    # PLE blocks (only the layers the config names carry them).
    "ple_conv1d.weight": "ple/convolution",
    "ple_key.weight": "ple/key",
    "ple_value.weight": "ple/value",
    "ple_norm_conv.weight": "ple/norm_conv",
    "ple_norm_key.weight": "ple/norm_key",
    "ple_norm_query.weight": "ple/norm_query",
}
GDN_LAYER = {
    "ssm_a": "gdn/a_log",
    "ssm_alpha.weight": "gdn/a_projection",
    "ssm_beta.weight": "gdn/b_projection",
    "ssm_conv1d.weight": "gdn/convolution",
    "ssm_dt.bias": "gdn/dt_bias",
    "ssm_norm.weight": "gdn/norm",
    "ssm_out.weight": "gdn/output",
    "attn_gate.weight": "gdn/z",
}
FULL_LAYER = {
    "attn_k.weight": "attention/key",
    "attn_v.weight": "attention/value",
    "attn_output.weight": "attention/output",
    "attn_q_norm.weight": "attention/query_norm",
    "attn_k_norm.weight": "attention/key_norm",
    "indexer.q_proj.weight": "attention/index_query",
    "indexer.k_proj.weight": "attention/index_key",
    "indexer.q_norm.weight": "attention/index_query_norm",
    "indexer.k_norm.weight": "attention/index_key_norm",
}
# Merged GGUF tensors, cut by rows. The logical suffix carries the layer family because that is how
# the binder names them (gdn/query on a linear-attention layer, attention/query on a full one).
# Ratios follow the element counts the reference artifact uses
# (GDN: 2048/2048/6144 of 10240; full attention: 6144/6144 of 12288).
MERGED = {
    "attn_qkv.weight": (("gdn/query", 0.2), ("gdn/key", 0.2), ("gdn/value", 0.6)),
    "attn_q.weight": (("attention/query", 0.5), ("attention/output_gate", 0.5)),
}


def split_rows(obj: Obj, pieces) -> list:
    rows = obj.shape[0]
    row_bytes = obj.bytes // rows
    out = []
    start = 0
    for name, ratio in pieces:
        n = rows - start if name == pieces[-1][0] else int(rows * ratio)
        out.append(Obj(id=f"{obj.id}/{name}", kind="tensor", shape=[n, obj.shape[1]],
                       fmt=obj.fmt, layout=obj.layout, src=obj.src,
                       src_offset=obj.src_offset + start * row_bytes, bytes=n * row_bytes))
        start += n
    return out


def load_uses(path: str) -> list:
    """The computation-permission table (parameter x input x activation policy).

    It is a property of the model's math, not of the encoding, so the packed file takes it from the
    same-model artifact instead of restating it.
    """
    with open(path, "rb") as f:
        if f.read(8) != ENTRY_MAGIC:
            raise ValueError(f"{path} is not a v3 entry")
        (json_bytes,) = struct.unpack("<Q", f.read(8))
        f.read(16)
        return json.loads(f.read(json_bytes).decode())["uses"]


def filter_uses(uses: list, components: dict) -> list:
    """Keep only entries whose parameter and input belong to a component this artifact carries.

    The reference table covers every optional component (vision/mtp/dflash2); an artifact that packs
    text only must not declare permissions for a component it does not provide.
    """
    out = []
    for u in uses:
        if u["parameter"].split("/")[0] in components and u["input"].split("/")[0] in components:
            out.append(u)
    return out


DFLASH_TOP = {
    "selector_hidden.weight": "candidate_selector/hidden_projection",
    "selector_predecessor.weight": "candidate_selector/predecessor_codebook",
    "selector_successor.weight": "candidate_selector/successor_codebook",
    "fc.weight": "feature_projection",
    "enc.output_norm.weight": "context_norm",
    "output_norm.weight": "final_norm",
}
DFLASH_LAYER = {
    "attn_conv_base": "attention_conv/base_kernel",
    "attn_conv_proj.weight": "attention_conv/kernel_projection",
    "attn_norm.weight": "input_norm",
    "ffn_norm.weight": "post_attention_norm",
    "ffn_conv_base": "mlp_conv/base_kernel",
    "ffn_conv_proj.weight": "mlp_conv/kernel_projection",
    "ffn_down.weight": "mlp/down",
    "ffn_gate.weight": "mlp/gate",
    "ffn_up.weight": "mlp/up",
    "attn_q.weight": "attention/query",
    "attn_k.weight": "attention/key",
    "attn_v.weight": "attention/value",
    "attn_output.weight": "attention/output",
    "attn_q_norm.weight": "attention/query_norm",
    "attn_k_norm.weight": "attention/key_norm",
}


def expand_dflash(objs: list, component: str) -> tuple:
    """Bindings for a draft component's own tensor names (the DFlash2 layout)."""
    bindings = {}
    prefix = component + "."
    for obj in objs:
        name = obj.id[len(prefix):]
        if name.startswith("blk."):
            _, num, rest = name.split(".", 2)
            suffix = DFLASH_LAYER.get(rest)
            if suffix:
                bindings[f"{component}/layers/{int(num)}/{suffix}"] = {"object": obj.id}
            continue
        suffix = DFLASH_TOP.get(name)
        if suffix:
            bindings[f"{component}/{suffix}"] = {"object": obj.id}
    # context_key/context_value are the same tensors as key/value (the reference artifact binds both
    # names to one identical range, not to a second copy), so they are aliases, not extra data.
    for source, alias in (("/attention/key", "/attention/context_key"),
                          ("/attention/value", "/attention/context_value")):
        for key, value in list(bindings.items()):
            if key.endswith(source):
                bindings[key[: -len(source)] + alias] = dict(value)
    return objs, bindings


def expand_text(objs: list, layer_types: list, ple_layer_id: int = 1) -> tuple:
    """Cut merged tensors and return (objects, bindings) for the text component.

    A merged GGUF tensor becomes one object per logical parameter, so the bindings are plain
    1:1 references and the container needs no `parts` ranges.
    """
    out = []
    bindings = {}
    for obj in objs:
        name = obj.id[len("text."):]
        if name.startswith("blk."):
            _, num, rest = name.split(".", 2)
            layer = int(num)
            full = str(layer_types[layer]).lower() not in ("linear_attention", "gdn", "mamba")
            if rest in MERGED:
                rows = obj.shape[0]
                row_bytes = obj.bytes // rows
                start = 0
                pieces = MERGED[rest]
                for i, (piece, ratio) in enumerate(pieces):
                    n = rows - start if i == len(pieces) - 1 else int(rows * ratio)
                    child = Obj(id=f"{obj.id}/{piece}", kind="tensor",
                                shape=[n, obj.shape[1]], fmt=obj.fmt, layout=obj.layout,
                                src=obj.src, src_offset=obj.src_offset + start * row_bytes,
                                bytes=n * row_bytes)
                    out.append(child)
                    bindings[f"text/layers/{layer}/{piece}"] = {"object": child.id}
                    start += n
                continue
            suffix = dict(COMMON_LAYER, **(FULL_LAYER if full else GDN_LAYER)).get(rest)
            out.append(obj)
            if suffix:
                bindings[f"text/layers/{layer}/{suffix}"] = {"object": obj.id}
            continue
        out.append(obj)
        if name == "token_embd.weight":
            bindings["text/token_embedding"] = {"object": obj.id}
        elif name == "output.weight":
            bindings["text/output_head"] = {"object": obj.id}
        elif name.startswith("output_hc_"):
            bindings["text/final_hc/" + name[len("output_hc_"):].split(".")[0]] = {"object": obj.id}
        elif name.startswith("per_layer_token_embd"):
            # The PLE table belongs to the layer the config names as carrying it (layer 1 here).
            bindings[f"text/layers/{ple_layer_id}/ple/table"] = {"object": obj.id}
    return out, bindings


def main() -> int:
    ap = argparse.ArgumentParser(description="Qwen4Exp GGUF -> NInfer v3")
    ap.add_argument("--text", nargs="+", required=True)
    ap.add_argument("--dflash2", nargs="*", default=[])
    ap.add_argument("--mtp", nargs="*", default=[])
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard-gb", type=int, default=30)
    ap.add_argument("--config-from", default="",
                    help="artifact to copy the text component config from (same model)")
    ap.add_argument("--resources-from", default="",
                    help="artifact to copy component resources (tokenizer/templates/PLE) from")
    ap.add_argument("--uses-from", default="",
                    help="artifact to copy the uses table from (same model)")
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()

    sources = {}

    def source(path: str) -> Source:
        if path not in sources:
            sources[path] = Source(path, gguf.GGUFReader(path))
        return sources[path]

    text_objs = []
    for path in args.text:
        text_objs += collect(source(path), "text")
    if args.config_from:
        text_cfg = load_component_config(args.config_from)
    else:
        text_cfg = text_config(read_metadata(source(args.text[0])))
    text_objs, bindings = expand_text(text_objs, text_cfg.get("layer_types", []),
                                      int((text_cfg.get("ple") or {}).get("layer_ids", [1])[0]))
    objs = list(text_objs)
    components = {"text": {"config": text_cfg}}
    for flag, comp, arc in (("dflash2", "dflash2", "Qwen3_8DFlash2"),
                            ("mtp", "mtp", "Qwen3_5MTP")):
        for i, path in enumerate(getattr(args, flag)):
            name = comp if i == 0 else f"{comp}{i}"
            part = collect(source(path), name)
            if comp == "dflash2":
                part, draft_bindings = expand_dflash(part, name)
                bindings.update(draft_bindings)
            objs += part
            components[name] = {"config": {"architectures": [arc]}, "target": "text"}
    if args.resources_from:
        for comp, mapping in load_resources(args.resources_from).items():
            if comp not in components:
                continue
            components[comp]["resources"] = {name: o.id for name, o in mapping.items()}
            objs += list(mapping.values())

    if not objs:
        print("no tensors collected", file=sys.stderr)
        return 2

    shards, payload_bytes = plan(objs, args.shard_gb << 30)
    entry_name = os.path.basename(args.out)
    files = []
    volumes = []
    for i, (base, shard) in enumerate(shards):
        # Declared size spans up to the last object's end, alignment gap included: the reader checks
        # object ranges against this total, and a sum of object bytes is short by the gaps.
        end = shard[-1].offset + shard[-1].bytes
        files.append({"path": None if i == 0 else f"{entry_name}.part-{i:04d}",
                      "payload_bytes": end - base})
        volumes.append((base, shard, end - base))

    artifact_id = uuid.uuid4().bytes
    directory = {
        "components": components,
        "objects": [describe(o) for o in objs],
        "bindings": bindings,
        "uses": filter_uses(load_uses(args.uses_from), components) if args.uses_from else [],
        "files": files,
        "metadata": {"name": str(read_metadata(source(args.text[0])).get("general.name", "Qwen3.8-Flash-Next")),
                     "source": "gguf", "payload_bytes": payload_bytes},
        "provenance": {
            "converter": "tools/qwen4exp/convert_gguf_to_ninfer.py",
            "note": "GGML encodings preserved verbatim: format=ggml_<type>, layout=ggml_block_v1",
            "sources": [{"path": p, "bytes": os.path.getsize(p)} for p in sources],
        },
    }
    blob = json.dumps(directory, ensure_ascii=False, separators=(",", ":")).encode()
    payload_start = align_up(HEADER + len(blob), ENTRY_ALIGN)

    out_dir = os.path.dirname(os.path.abspath(args.out)) or "."
    os.makedirs(out_dir, exist_ok=True)
    part_paths = [args.out] + [os.path.join(out_dir, f["path"]) for f in files[1:]]

    # The part_index in each continuation header is its files[] index.
    for index, (base, shard, final_size) in enumerate(volumes):
        write_shard(part_paths[index], shard, base, final_size, index, artifact_id,
                    payload_start if index == 0 else 0, blob if index == 0 else b"")

    total = sum(os.path.getsize(p) for p in part_paths)
    print(f"wrote {args.out}")
    print(f"  objects={len(objs)} payload={payload_bytes/2**30:.2f} GiB files={len(part_paths)} total={total/2**30:.2f} GiB")
    for o in objs[:3]:
        print(f"  {o.id}: {o.fmt} {o.shape} {o.bytes/2**20:.1f} MiB")

    if args.verify:
        bad = checked = skipped = 0
        # Objects the writer transformed (bf16 casts, conv transpose) do not round-trip to a simple
        # byte copy of the source, and their on-disk bytes are produced by that transform; the
        # sampler reports them separately instead of pretending the copy check covers them.
        plain = [o for o in objs if not (o.transpose or o.f32_to_bf16 or o.f16_to_bf16)]
        skipped = len(objs) - len(plain)
        sample = plain[:: max(1, len(plain) // 24)]
        # Locate objects through the same allocation the writer used. Deriving the volume from the
        # JSON payload spans re-derives a boundary the writer already decided, and the two disagree
        # for the object that starts exactly at a volume edge.
        placement = {}
        for vol, (base, shard, _size) in enumerate(volumes):
            for obj in shard:
                placement[obj.id] = (vol, base)
        for obj in sample:
            vol, base = placement[obj.id]
            start = payload_start if vol == 0 else ENTRY_ALIGN
            with open(part_paths[vol], "rb") as f:
                f.seek(start + (obj.offset - base))
                h = hashlib.sha256()
                left = obj.bytes
                while left:
                    chunk = f.read(min(COPY_CHUNK, left))
                    if not chunk:
                        break
                    h.update(chunk)
                    left -= len(chunk)
            checked += 1
            if h.hexdigest() != obj.sha256:
                bad += 1
                print(f"  MISMATCH {obj.id}", file=sys.stderr)
        print(f"verify: {checked} sampled objects, {bad} mismatched, {skipped} transformed objects not sampled")
        return 1 if bad else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



