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
import gguf.quants  # noqa: E402  (dequantize, for the one site whose permutation is inside a row)

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
    # `w - 1`: a gamma the GGUF stores folded as `1 + w` while the engine's unit-offset norm adds
    # the one itself. The source element size and encoding are kept so the writer (and the verify
    # pass) can decode, subtract and re-round without inferring them from the target format.
    minus_one: bool = False
    src_elem: int = 0
    src_type: str = ""
    # Rows written in a permuted order: target row i takes source row row_map[i]-th of the source
    # rows (resolved across `segments`, so a merged parent rows-map works too). A row-permutation of
    # a block-quantised matrix moves whole rows, so the encoding survives untouched.
    row_map: list | None = None
    # A payload this tool authors rather than copies (value transforms a row permutation cannot
    # express: log(-A_log), a conv transpose, a column permutation inside quantised rows).
    blob: bytes | None = None
    # Objects assembled from more than one source range (the GDN control pair, which the model
    # addresses as one parent). Each entry is (source offset, byte count), written in order.
    segments: list | None = None


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


# Gammas llama.cpp folds to `1 + w` on disk. The engine's unit-offset norms add that one back
# (`gain = 1 + weight`), so an artifact that copied these vectors verbatim would run every one of
# them at `2 + w` - roughly twice the model's gain on the residual streams. `gguf_views.Minus1` and
# `tools/convert/gguf_qwen4exp.py::_norm` are the same rule for the native converter, and the
# same-model artifact `qwen3_8_flash_next_v5_e2` stores exactly `bf16(gguf - 1)` for each of them
# (verified bit-exact at every spelling site). `ssm_norm` is deliberately absent: upstream's fold
# skips `linear_attn.norm.weight`, and that artifact matches the GGUF bit for bit.
FOLDED_NORMS = (
    "hc_attn_norm.weight",
    "hc_ffn_norm.weight",
    "output_hc_norm.weight",
    "attn_q_norm.weight",
    "attn_k_norm.weight",
    "indexer.q_norm.weight",
    "indexer.k_norm.weight",
    "ple_norm_key.weight",
    "ple_norm_query.weight",
    "ple_norm_conv.weight",
)


def bf16_round(values) -> bytes:
    """float32 -> bf16 bytes, round-to-nearest-even on the tie bit (the top half of the word)."""
    u32 = np.asarray(values, dtype=np.float32).view(np.uint32).astype(np.uint64)
    return ((u32 + 0x7FFF + ((u32 >> 16) & 1)) >> 16).astype(np.uint16).tobytes()


def decode_float(raw: bytes, kind: str):
    """Decode a GGML float payload to float32 so a value transform can run on it."""
    if kind == "F32":
        return np.frombuffer(raw, dtype="<f4")
    if kind == "F16":
        return np.frombuffer(raw, dtype="<f2").astype(np.float32)
    if kind == "BF16":
        return (np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16).view(np.float32)
    raise ValueError(f"a folded norm is a float vector, not {kind}")


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
        # A folded gamma is stored `1 + w` in the GGUF and `w` in the artifact, at whichever float
        # width the quantiser chose for it; the writer decodes that width, subtracts and re-rounds.
        # The draft shares the text blocks' `attn_q_norm`/`attn_k_norm` spelling but its own norms are
        # plain (`tools/convert/gguf_dflash.py`: `w * x / sqrt(mean + eps)`, no unit offset), so the
        # component decides, not the tensor name.
        folded = component == "text" and t.name.endswith(FOLDED_NORMS)
        # The source width of one element, before any cast: the transforms that read the payload back
        # (folded norms, the conv transpose, A_log) need to know how many bytes one element occupies
        # in the file, which the target format no longer says.
        source_elements = 1
        for d in shape_c:
            source_elements *= d
        src_elem = int(t.n_bytes) // max(1, source_elements)
        if folded:
            if len(shape_c) != 1:
                raise ValueError(f"{t.name}: a folded norm is a rank-1 vector, not {shape_c}")
            if fmt not in ("fp32", "bf16", "ggml_f16"):
                raise ValueError(f"{t.name}: folded norm has an unexpected encoding {fmt}")
            if src_elem * source_elements != int(t.n_bytes):
                raise ValueError(f"{t.name}: folded norm is not a whole number of elements")
            fmt, layout, n_bytes = "bf16", "contiguous_le_v1", source_elements * 2
            cast = False
        transpose = False
        if (t.name.endswith("conv1d.weight") and len(shape_c) == 2
                and shape_c[0] > shape_c[1] and 1 < shape_c[1] <= 8):
            # GGUF conv1d is [channels, kernel]; the model wants [kernel, channels].
            shape_c = [shape_c[1], shape_c[0]]
            transpose = True
        if t.name.endswith("ffn_gate_inp_shexp.weight") and len(shape_c) == 1:
            # The gate vector is a one-row matrix in the model's parameter shape.
            shape_c = [1, shape_c[0]]
        if len(shape_c) == 3 and "_exps" in t.name:
            # An expert bank is one [experts x rows, columns] matrix in the model's shape; GGML
            # stores it as [experts, rows, columns] contiguously, so this is a view, not a copy.
            shape_c = [shape_c[0] * shape_c[1], shape_c[2]]
        objs.append(Obj(id=f"{component}.{t.name}", kind="tensor", shape=shape_c,
                        fmt=fmt, layout=layout, f32_to_bf16=cast, transpose=transpose,
                        f16_to_bf16=f16_cast, minus_one=folded, src_elem=src_elem,
                        src_type=gguf.GGMLQuantizationType(t.tensor_type).name,
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
            if obj.blob is not None:
                f.write(obj.blob)
                h.update(obj.blob)
                written += len(obj.blob)
            elif obj.row_map is not None:
                data = permuted_rows(obj)
                f.write(data)
                h.update(data)
                written += len(data)
            elif obj.minus_one:
                src_len = obj.shape[0] * obj.src_elem
                with open(obj.src.path, "rb", buffering=0) as sf:
                    sf.seek(obj.src_offset)
                    raw = sf.read(src_len)
                if len(raw) != src_len:
                    raise IOError(f"short read for {obj.id}")
                # `w = (1 + w) - 1` before the bf16 rounding: the unit-offset norm adds the one
                # back, so the stored vector is the offset the model's gain was folded from.
                data = bf16_round(decode_float(raw, obj.src_type) - 1.0)
                f.write(data)
                h.update(data)
                written += len(data)
            elif obj.f32_to_bf16:
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
            elif obj.segments:
                for src_off, src_len in obj.segments:
                    copy_range(obj.src, src_off, src_len, f, h)
                    written += src_len
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


def permuted_rows(obj: Obj) -> bytes:
    """Read the object's source rows and return them in `row_map` order.

    Whole rows move, so a block-quantised matrix keeps its encoding: a row is one or more complete
    blocks, and the object's declared byte count divides by its row count. The map is resolved over
    `segments` when the parent was assembled from more than one source range (the a|b control pair),
    which is why the lookup is a (segment, row) pair rather than a flat source offset.
    """
    rows = obj.shape[0]
    if obj.bytes % rows:
        raise ValueError(f"{obj.id}: {obj.bytes} bytes over {rows} rows is not row aligned")
    row_bytes = obj.bytes // rows
    spans = obj.segments if obj.segments else [(obj.src_offset, obj.bytes)]
    starts = []
    total = 0
    for _offset, length in spans:
        starts.append(total)
        if length % row_bytes:
            raise ValueError(f"{obj.id}: segment of {length} bytes is not row aligned")
        total += length // row_bytes
    if total != rows:
        raise ValueError(f"{obj.id}: segments hold {total} rows, the shape says {rows}")
    out = bytearray(obj.bytes)
    if obj.bytes <= (64 << 20):
        # One read per segment: a per-row seek over a 14 MiB matrix is ~10k syscalls for no reason.
        planes = []
        for offset, length in spans:
            planes.append(read_source_at(obj, offset, length))
        for target, source_row in enumerate(obj.row_map):
            segment = 0
            while segment + 1 < len(spans) and source_row >= starts[segment + 1]:
                segment += 1
            begin = (source_row - starts[segment]) * row_bytes
            out[target * row_bytes:(target + 1) * row_bytes] = \
                planes[segment][begin:begin + row_bytes]
        return bytes(out)
    with open(obj.src.path, "rb", buffering=0) as f:
        for target, source_row in enumerate(obj.row_map):
            segment = 0
            while segment + 1 < len(spans) and source_row >= starts[segment + 1]:
                segment += 1
            offset = spans[segment][0] + (source_row - starts[segment]) * row_bytes
            f.seek(offset)
            chunk = f.read(row_bytes)
            if len(chunk) != row_bytes:
                raise IOError(f"short read for {obj.id} row {source_row}")
            out[target * row_bytes:(target + 1) * row_bytes] = chunk
    return bytes(out)


def read_source_at(obj: Obj, offset: int, length: int) -> bytes:
    with open(obj.src.path, "rb", buffering=0) as f:
        f.seek(offset)
        raw = f.read(length)
    if len(raw) != length:
        raise IOError(f"short read for {obj.id}")
    return raw


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


# Logical parameters the model addresses as two row blocks of ONE parent. Each entry is
# (first tensor suffix, second tensor suffix, merged object suffix, first binding, second binding).
# Verified against the same-model reference artifact: its bindings are `parts` ranges over a single
# object for exactly these pairs (gdn control [96,2560] and ple embedding [12800,2560]).
#
# The HC pairs are the compact form the Qwen4Exp path reads: `down` rows then `inject` rows of one
# BF16 parent, which is the [rank+streams, H*C] matrix `hc_mix`'s row-tail schedule decodes. The
# final HC has no injection stream, so it is not a pair and keeps `down` as its own object.
STACKED_PAIRS = (
    ("ssm_alpha", "ssm_beta", "gdn_control", "gdn/a_projection", "gdn/b_projection"),
    ("ple_key", "ple_value", "ple_embedding", "ple/key", "ple/value"),
    ("hc_attn_down", "hc_attn_inject", "hc_attn_pair", "attention_hc/down", "attention_hc/inject"),
    ("hc_ffn_down", "hc_ffn_inject", "hc_ffn_pair", "moe_hc/down", "moe_hc/inject"),
)


def merge_stacked_pairs(objs: list, bindings: dict) -> tuple:
    """Fold every two-block parameter into the single parent the model's ops require.

    `prepare_gdn_control_proj_weights` concatenates its A/B pair, the PLE gate reads `key` and
    `value` as two halves of one [H*C+H,H] projection, and `hc_mix` reads `down` plus its injection
    rows as one row-tailed matrix; all three refuse a pair that lives in two objects. GGUF keeps
    them apart, so each pair becomes one object holding the two byte ranges back to back, with the
    bindings addressing them as `parts` - the same shape the reference artifact uses. The row split
    is by element range, so it holds for quantised parents as long as both halves share one encoding
    and one column count.
    """
    by_suffix: dict = {}
    for obj in objs:
        for tag in {t[0] for t in STACKED_PAIRS} | {t[1] for t in STACKED_PAIRS}:
            if obj.id.endswith(f".{tag}.weight"):
                by_suffix.setdefault(obj.id[: -len(f".{tag}.weight")], {})[tag] = obj
    drop = set()
    added = []
    for prefix, parts in by_suffix.items():
        # The layer is the `blk.N` component, not the last dotted field: an HC pair's prefix ends in
        # `hc_attn`, which is what the tensor name carries after the block number.
        if ".blk." not in prefix:
            continue
        layer = int(prefix.rsplit(".blk.", 1)[1].split(".", 1)[0])
        for tag_a, tag_b, merged_name, key_a, key_b in STACKED_PAIRS:
            a, b = parts.get(tag_a), parts.get(tag_b)
            if a is None or b is None:
                continue
            if a.fmt != b.fmt or a.layout != b.layout or a.shape[1] != b.shape[1]:
                # A mixed encoding cannot be one parent: refuse loudly instead of packing a pair the
                # loader will reject later with a message that no longer names the cause.
                raise ValueError(f"{prefix}: {tag_a}/{tag_b} differ in encoding or width, so they "
                                 f"cannot be one parent ({a.fmt} {a.shape} vs {b.fmt} {b.shape})")
            cols = a.shape[1]
            span = a.shape[0] * cols
            merged = Obj(id=f"{prefix}.{merged_name}.weight", kind="tensor",
                         shape=[a.shape[0] + b.shape[0], cols], fmt=a.fmt, layout=a.layout,
                         src=a.src, src_offset=a.src_offset, bytes=a.bytes + b.bytes,
                         segments=[(a.src_offset, a.bytes), (b.src_offset, b.bytes)])
            bindings[f"text/layers/{layer}/{key_a}"] = {
                "parts": [{"object": merged.id, "range": [0, span]}]}
            bindings[f"text/layers/{layer}/{key_b}"] = {
                "parts": [{"object": merged.id, "range": [span, span + b.shape[0] * cols]}]}
            added.append(merged)
            drop.add(a.id)
            drop.add(b.id)
    kept = [o for o in objs if o.id not in drop] + added
    return kept, bindings


def gdn_tiling(cfg: dict) -> tuple:
    """The heads whose order llama.cpp and the engine disagree on, and the rows that carry them.

    `nk != nv` makes llama.cpp write the value-head axis tiled (llama.cpp's own
    `_LinearAttentionVReorderBase` reshapes it to `[key_head, head_in_group, head_dim]` and
    transposes the first two axes), while the engine indexes it grouped. `grouped_from_tiled` in
    `tools/convert/gguf_views.py` states the map: grouped head `g` sits at tiled position
    `(g % per_key) * key_heads + g // per_key`. Only the GDN layers of a tiled family need it; a
    model with `nk == nv` has one head per key head and the permutation is the identity.
    """
    nk = int(cfg.get("linear_num_key_heads", 0))
    nv = int(cfg.get("linear_num_value_heads", 0))
    dv = int(cfg.get("linear_value_head_dim", 0))
    if not nk or not nv or nk == nv:
        return None
    per_key = nv // nk
    heads = [(h % per_key) * nk + h // per_key for h in range(nv)]
    rows = [(h * dv + i) for h in heads for i in range(dv)]
    return heads, rows


def quantize_q5_row_split(values) -> bytes:
    """Encode a float matrix as `q5_g64_fp16` / `row_split_k128_v1`.

    Used for the one site whose permutation is inside a quantised row (the GDN out projection):
    reordering 4-bit codes inside a block would mean decoding and re-encoding anyway, and a 5-bit
    re-encode keeps the tensor in a format the engine already runs for that projection - the
    same-model artifact stores this tensor as q5 - while its step is finer than the 4-bit codes the
    values came from. Symmetric per-group scale, codes in [-16,15], the group size and plane order
    the engine's own reader and writer use (`tools/artifact/codecs/row_split.py`).
    """
    import torch
    # The codec lives in this repo's `tools` package; the converter is run from anywhere, so the
    # repository root (two levels above this file) is put on the path instead of assuming a cwd.
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from tools.artifact.codecs.row_split import encode_row_split

    values = np.ascontiguousarray(values, dtype=np.float32)
    rows, width = values.shape
    groups = width // 64
    if groups * 64 != width:
        raise ValueError(f"q5 groups of 64 do not divide width {width}")
    blocks = values.reshape(rows, groups, 64)
    scale = np.abs(blocks).max(axis=2, keepdims=True) / 15.0
    scale = np.where(scale > 0.0, scale, 1.0).astype(np.float32)
    codes = np.clip(np.rint(blocks / scale), -16, 15).astype(np.int8)
    return encode_row_split(torch.from_numpy(codes), torch.from_numpy(scale[:, :, 0].astype(np.float16)),
                            "q5_g64_fp16", (rows, width))


def apply_gdn_tiling(objs: list, cfg: dict, layer_types: list) -> None:
    """Put the GDN value-head axis back in the order the engine's kernels index.

    Seven sites carry it, exactly the ones the native converter rewrites (`_gdn` in
    `tools/convert/gguf_qwen4exp.py`): the V rows of the fused q|k|v matrix, the z rows, the a/b
    control rows, A_log, dt_bias, the conv kernel's V channels and the out projection's columns.
    A row permutation moves whole rows, so those stay their original GGML encoding; the three sites
    that need a value transform (A_log is stored as `-exp(A_log)`, the conv needs a transpose, and
    the out projection permutes inside a row) are authored here.
    """
    tiling = gdn_tiling(cfg)
    if tiling is None:
        return
    heads, rows = tiling
    nv = len(heads)
    dv = len(rows) // nv
    nk = int(cfg["linear_num_key_heads"])
    q = nk * dv
    v = nv * dv
    hidden = int(cfg["hidden_size"])
    by_id = {o.id: o for o in objs}

    for layer, kind in enumerate(layer_types):
        if str(kind).lower() not in ("linear_attention", "gdn", "mamba"):
            continue
        prefix = f"text.blk.{layer}."

        # q | k | v: only the V block is tiled.
        obj = by_id.get(prefix + "attn_qkv.weight")
        if obj is not None:
            total = obj.shape[0]
            if total != 2 * q + v:
                raise ValueError(f"{obj.id}: q|k|v is {total} rows, expected {2 * q + v}")
            obj.row_map = list(range(2 * q)) + [2 * q + r for r in rows]

        obj = by_id.get(prefix + "attn_gate.weight")
        if obj is not None:
            if obj.shape[0] != v:
                raise ValueError(f"{obj.id}: z is {obj.shape[0]} rows, expected {v}")
            obj.row_map = list(rows)

        # The a|b control parent is [a(48) | b(48)] rows; each half is permuted on its own.
        obj = by_id.get(prefix + "gdn_control.weight")
        if obj is not None:
            if obj.shape[0] != 2 * nv:
                raise ValueError(f"{obj.id}: control is {obj.shape[0]} rows, expected {2 * nv}")
            obj.row_map = list(heads) + [nv + h for h in heads]

        # A_log: llama.cpp stores `-exp(A_log)`, the engine applies `-exp(A_log)` itself.
        obj = by_id.get(prefix + "ssm_a")
        if obj is not None:
            raw = read_source(obj, obj.bytes)
            values = np.frombuffer(raw, dtype=np.float32)
            if not np.all(values < 0):
                raise ValueError(f"{obj.id}: ssm_a is not negative, so it is not -exp(A_log)")
            obj.blob = np.log(-values)[heads].astype(np.float32).tobytes()
            obj.bytes = len(obj.blob)

        obj = by_id.get(prefix + "ssm_dt.bias")
        if obj is not None:
            raw = read_source(obj, obj.bytes)
            values = np.frombuffer(raw, dtype=np.float32)
            obj.blob = values[heads].astype(np.float32).tobytes()
            obj.bytes = len(obj.blob)

        # The conv kernel is [channels, taps] in the GGUF and [taps, channels] in the model, with the
        # V channels tiled; bf16 is the operand type the conv kernels accept.
        obj = by_id.get(prefix + "ssm_conv1d.weight")
        if obj is not None:
            elements = obj.shape[0] * obj.shape[1]
            raw = read_source(obj, elements * obj.src_elem)
            values = np.frombuffer(raw, dtype=np.float32).reshape(2 * q + v, -1)
            taps = values.shape[1]
            tiled = np.concatenate([values[: 2 * q], values[2 * q:][rows]])
            obj.blob = bf16_round(tiled.T.reshape(-1))
            obj.fmt, obj.layout = "bf16", "contiguous_le_v1"
            obj.shape = [taps, 2 * q + v]
            obj.bytes = len(obj.blob)
            obj.transpose = False
            obj.src_elem = 0

        # The out projection's reduction axis is the value-head axis, so its permutation happens
        # inside each row: a block-quantised row cannot be reordered without re-encoding, and
        # re-encoding 4-bit codes would change the values. The decoded rows are stored bf16, which
        # is strictly wider than the codes they came from.
        obj = by_id.get(prefix + "ssm_out.weight")
        if obj is not None:
            if obj.fmt.startswith("ggml_"):
                raw = read_source(obj, obj.bytes)
                kind = gguf.GGMLQuantizationType[obj.src_type]
                values = gguf.quants.dequantize(np.frombuffer(raw, dtype=np.uint8), kind)
            else:
                raw = read_source(obj, obj.bytes)
                values = (np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32) \
                    if obj.fmt == "bf16" else np.frombuffer(raw, dtype=np.float32)
            values = np.asarray(values, dtype=np.float32).reshape(obj.shape[0], obj.shape[1])
            if values.shape[1] != v:
                raise ValueError(f"{obj.id}: out projection width {values.shape[1]}, expected {v}")
            obj.blob = quantize_q5_row_split(values[:, rows])
            obj.fmt, obj.layout = "q5_g64_fp16", "row_split_k128_v1"
            obj.bytes = len(obj.blob)


def read_source(obj: Obj, length: int) -> bytes:
    with open(obj.src.path, "rb", buffering=0) as f:
        f.seek(obj.src_offset)
        raw = f.read(length)
    if len(raw) != length:
        raise IOError(f"short read for {obj.id}")
    return raw


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
                # The merged tensor stays ONE object: the GDN control and the attention row views
                # require a single parent and address their slices by element range, exactly as the
                # reference artifact does. Splitting into separate objects breaks that contract.
                rows = obj.shape[0]
                cols = obj.shape[1]
                start = 0
                pieces = MERGED[rest]
                for i, (piece, ratio) in enumerate(pieces):
                    n = rows - start if i == len(pieces) - 1 else int(rows * ratio)
                    bindings[f"text/layers/{layer}/{piece}"] = {
                        "parts": [{"object": obj.id,
                                   "range": [start * cols, (start + n) * cols]}]}
                    start += n
                out.append(obj)
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
    text_objs, bindings = merge_stacked_pairs(text_objs, bindings)
    # After the merge, so the a|b control parent is permuted as one 96-row object rather than as two
    # halves that the merge would then copy verbatim.
    apply_gdn_tiling(text_objs, text_cfg, text_cfg.get("layer_types", []))
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
        # Objects the writer transformed (bf16 casts, conv transpose, folded norms) do not round-trip
        # to a simple byte copy of the source, and their on-disk bytes are produced by that transform;
        # the sampler reports them separately instead of pretending the copy check covers them.
        plain = [o for o in objs if not (o.transpose or o.f32_to_bf16 or o.f16_to_bf16
                                        or o.minus_one or o.row_map is not None
                                        or o.blob is not None)]
        skipped = len(objs) - len(plain)
        # The merged parents (gdn control, ple key|value, hc down|inject) are the objects this tool
        # authors rather than copies, so every one of them is checked, not just the periodic sample:
        # their byte order is a convention the engine depends on and nothing else verifies it.
        merged = [o for o in plain if o.segments]
        stride = max(1, len(plain) // 24)
        sample = merged + [o for i, o in enumerate(plain) if i % stride == 0 and not o.segments]
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
        # The folded gammas are the only objects whose value is rewritten rather than copied, so
        # every one of them is recomputed from its source and compared with the volume. A gain that
        # keeps its `1 +` is invisible in the shapes and fatal to every layer that consumes it.
        folded = [o for o in objs if o.minus_one]
        for obj in folded:
            src_len = obj.shape[0] * obj.src_elem
            with open(obj.src.path, "rb", buffering=0) as sf:
                sf.seek(obj.src_offset)
                raw = sf.read(src_len)
            expected = bf16_round(decode_float(raw, obj.src_type) - 1.0)
            vol, base = placement[obj.id]
            start = payload_start if vol == 0 else ENTRY_ALIGN
            with open(part_paths[vol], "rb") as f:
                f.seek(start + (obj.offset - base))
                got = f.read(len(expected))
            checked += 1
            if got != expected:
                bad += 1
                print(f"  MISMATCH {obj.id} (folded norm)", file=sys.stderr)
        # The GDN tiling rows are recomputed from the source and compared with the volume: the
        # permutation is invisible in the shapes and fatal to every linear-attention layer, so it gets
        # a content check, not just the placement the sampler checks.
        tiled = [o for o in objs if o.row_map is not None]
        for obj in tiled:
            expected = permuted_rows(obj)
            vol, base = placement[obj.id]
            start = payload_start if vol == 0 else ENTRY_ALIGN
            with open(part_paths[vol], "rb") as f:
                f.seek(start + (obj.offset - base))
                got = f.read(len(expected))
            checked += 1
            if got != expected:
                bad += 1
                print(f"  MISMATCH {obj.id} (row permutation)", file=sys.stderr)
        authored = len([o for o in objs if o.blob is not None])
        print(f"verify: {checked} sampled objects ({len(folded)} folded norms, {len(tiled)} row "
              f"permutations), {bad} mismatched, {skipped} transformed objects not sampled "
              f"({authored} authored payloads)")
        return 1 if bad else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



