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


def align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def ggml_format_name(tensor_type: int) -> str:
    return "ggml_" + gguf.GGMLQuantizationType(tensor_type).name.lower()


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
        objs.append(Obj(id=f"{component}.{t.name}", kind="tensor", shape=shape_c,
                        fmt=ggml_format_name(t.tensor_type), layout="ggml_block_v1",
                        src=source, src_offset=int(t.data_offset), bytes=int(t.n_bytes)))
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
        "architectures": ["Qwen3_5Text"],
        "model_type": "qwen3_5",
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
            shards.append((base, current))
            base += sum(o.bytes for o in current)
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


def write_shard(path: str, shard: list, base: int, index: int, artifact_id: bytes,
                payload_start: int, blob: bytes) -> None:
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
            copy_range(obj.src, obj.src_offset, obj.bytes, f, h)
            obj.sha256 = h.hexdigest()
            written += obj.bytes


def main() -> int:
    ap = argparse.ArgumentParser(description="Qwen4Exp GGUF -> NInfer v3")
    ap.add_argument("--text", nargs="+", required=True)
    ap.add_argument("--dflash2", nargs="*", default=[])
    ap.add_argument("--mtp", nargs="*", default=[])
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard-gb", type=int, default=30)
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()

    sources = {}

    def source(path: str) -> Source:
        if path not in sources:
            sources[path] = Source(path, gguf.GGUFReader(path))
        return sources[path]

    objs = []
    for path in args.text:
        objs += collect(source(path), "text")
    components = {"text": {"config": text_config(read_metadata(source(args.text[0])))}}
    for flag, comp, arc in (("dflash2", "dflash2", "Qwen3_8DFlash2"),
                            ("mtp", "mtp", "Qwen3_5MTP")):
        for i, path in enumerate(getattr(args, flag)):
            name = comp if i == 0 else f"{comp}{i}"
            objs += collect(source(path), name)
            components[name] = {"config": {"architectures": [arc]}, "target": "text"}

    if not objs:
        print("no tensors collected", file=sys.stderr)
        return 2

    shards, payload_bytes = plan(objs, args.shard_gb << 30)
    entry_name = os.path.basename(args.out)
    files = []
    for i, (_base, shard) in enumerate(shards):
        files.append({"path": None if i == 0 else f"{entry_name}.part-{i:04d}",
                      "payload_bytes": sum(o.bytes for o in shard)})

    artifact_id = uuid.uuid4().bytes
    directory = {
        "components": components,
        "objects": [describe(o) for o in objs],
        "bindings": {},
        "uses": [],
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
    for index, (base, shard) in enumerate(shards):
        write_shard(part_paths[index], shard, base, index, artifact_id,
                    payload_start if index == 0 else 0, blob if index == 0 else b"")

    total = sum(os.path.getsize(p) for p in part_paths)
    print(f"wrote {args.out}")
    print(f"  objects={len(objs)} payload={payload_bytes/2**30:.2f} GiB files={len(part_paths)} total={total/2**30:.2f} GiB")
    for o in objs[:3]:
        print(f"  {o.id}: {o.fmt} {o.shape} {o.bytes/2**20:.1f} MiB")

    if args.verify:
        bad = checked = 0
        sample = objs[:: max(1, len(objs) // 24)]
        base_of_volume = []
        running = 0
        for fdesc in files:
            base_of_volume.append(running)
            running += fdesc["payload_bytes"]
        for obj in sample:
            for vol, fdesc in enumerate(files):
                if base_of_volume[vol] <= obj.offset < base_of_volume[vol] + fdesc["payload_bytes"]:
                    start = payload_start if vol == 0 else ENTRY_ALIGN
                    with open(part_paths[vol], "rb") as f:
                        f.seek(start + (obj.offset - base_of_volume[vol]))
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
                    break
        print(f"verify: {checked} sampled objects, {bad} mismatched")
        return 1 if bad else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


