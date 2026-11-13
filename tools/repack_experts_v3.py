#!/usr/bin/env python3
"""Rewrite a v3 artifact so that every expert owns its objects.

A fused MoE artifact stores one matrix per (layer, matrix) with the experts in
consecutive row blocks, so loading one expert is six plane copies from ranges
scattered across two objects. This tool splits those objects per expert:

    weight/000003  (all 256 experts' gate|up)  ->  256 tensors of (2*ir, h)
    weight/000004  (all 256 experts' down)     ->  256 tensors of (h, ir)

Each new object keeps the artifact's own plane order (codes, then high, then
scales), which is exactly the per-expert image the loader builds, so the reader
needs to know nothing about this having happened.

Only `bindings` changes: `uses` names parameters and their inputs, never
objects. Weight bytes are copied, never decoded, so every expert stays
bit-identical and the PPL / decode-hash gates are the check.

usage: repack_experts_v3.py INPUT.ninfer OUTPUT.ninfer [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.artifact.framing import HEADER, PAYLOAD_ALIGNMENT  # noqa: E402
from tools.artifact.layouts import encoded_size, row_split_geometry  # noqa: E402
from tools.artifact.reader import Artifact  # noqa: E402
from tools.artifact.schema import (  # noqa: E402
    ResourceSpec,
    TensorSpec,
    plan_objects,
)
from tools.artifact.writer import layout_directory  # noqa: E402

EXPERT_BINDING = re.compile(r"^(?P<prefix>.*?/experts/)(?P<expert>\d+)/(?P<role>gate|up|down)$")
FUSED_EXPERT_BINDING = re.compile(r"^(?P<prefix>.*?/experts/)(?P<role>gate|up|down)$")
MAGIC = b"NINFER\x00\x03"


def read_description(path: Path) -> dict:
    with open(path, "rb") as handle:
        raw = handle.read(HEADER.size)
        if len(raw) != HEADER.size:
            raise SystemExit(f"{path}: truncated header")
        magic, length, _ = HEADER.unpack(raw)
        if magic != MAGIC:
            raise SystemExit(f"{path}: not a v3 artifact ({magic!r})")
        return json.loads(handle.read(length))


def planes_of(shape, fmt: str, layout: str):
    """(offset, bytes, row_bytes) for the populated planes of one tensor."""
    if layout != "row_split_k128_v1":
        raise SystemExit(f"unsupported layout for expert splitting: {layout}")
    geometry = row_split_geometry(fmt, shape)
    planes = [
        (geometry.base_offset, geometry.base_bytes, geometry.base_row_bytes),
        (geometry.high_offset, geometry.high_bytes, geometry.high_row_bytes),
        (geometry.scale_offset, geometry.scale_bytes, geometry.scale_row_bytes),
    ]
    return [plane for plane in planes if plane[1]]


def synthesize_expert_bindings(description: dict) -> int:
    """Fan a fused expert binding out into one binding per expert.

    The Qwen3.5 MoE adapter writes `experts/<i>/<role>`, one binding per expert; the Qwen4Exp
    adapter writes one binding per (layer, matrix) over a bank holding every expert's rows. Both
    load - the binder tries the per-expert name and falls back to the fused one - but only the
    per-expert form names the blocks this tool splits, so a fused artifact fans out first.

    The ranges are exact row blocks of the same object: every expert owns `rows/experts` rows, and
    the object's row-split layout is row-major over that block, which is what the split below
    copies plane by plane.
    """
    config = description.get("components", {}).get("text", {}).get("config", {})
    experts = int(config.get("num_experts") or 0)
    if experts <= 0:
        return 0
    bindings = description["bindings"]
    added = 0
    fanned: dict[str, tuple[list[str], str, str]] = {}
    for name, binding in list(bindings.items()):
        match = FUSED_EXPERT_BINDING.match(name)
        if match is None:
            continue
        parts = binding["parts"]
        if len(parts) != 1:
            raise SystemExit(f"{name}: fused expert weight spans several objects")
        begin, end = parts[0]["range"]
        total = end - begin
        if total % experts:
            raise SystemExit(f"{name}: {total} elements do not divide into {experts} experts")
        block = total // experts
        names: list[str] = []
        for expert in range(experts):
            per_expert = f"{match.group('prefix')}{expert}/{match.group('role')}"
            bindings[per_expert] = {
                "parts": [{"object": parts[0]["object"],
                           "range": [begin + expert * block, begin + (expert + 1) * block]}]
            }
            names.append(per_expert)
            added += 1
        fanned[name] = (names, match.group("prefix"), match.group("role"))
        # The fused object is split away below, so the fused binding that names it cannot survive:
        # every reader that resolves this matrix now goes through the per-expert names.
        del bindings[name]

    # `uses` names parameters, so every entry that named a fused expert bank has to name the experts
    # instead; the directory parser refuses a use whose parameter is not a binding. The down matrix
    # also reads its own expert's product, which the fused artifact spells once for the whole bank.
    if fanned and description.get("uses"):
        expanded = []
        for use in description["uses"]:
            entry = fanned.get(use["parameter"])
            if entry is None:
                expanded.append(use)
                continue
            names, prefix, role = entry
            for expert, name in enumerate(names):
                expanded.append({**use, "parameter": name})
        description["uses"] = expanded
    return added


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    description = read_description(args.source)
    objects = {obj["id"]: obj for obj in description["objects"]}

    if not any(EXPERT_BINDING.match(name) for name in description["bindings"]):
        fanned = synthesize_expert_bindings(description)
        if fanned:
            print(f"fused expert bindings fanned out into {fanned} per-expert bindings")

    # object -> expert -> role -> (element begin, element end, binding name)
    grouped: dict[str, dict[int, dict[str, tuple[int, int, str]]]] = {}
    for name, binding in description["bindings"].items():
        match = EXPERT_BINDING.match(name)
        if match is None:
            continue
        parts = binding["parts"]
        if len(parts) != 1:
            raise SystemExit(f"{name}: expert weight spans several objects")
        part = parts[0]
        grouped.setdefault(part["object"], {}).setdefault(int(match.group("expert")), {})[
            match.group("role")
        ] = (part["range"][0], part["range"][1], name)

    if not grouped:
        raise SystemExit("no expert bindings found; nothing to do")
    print(f"expert objects: {len(grouped)} covering {sum(len(v) for v in grouped.values())} experts")

    specs = []
    bindings = dict(description["bindings"])
    # new object id -> [(source byte offset, byte count)]
    splits: dict[str, list[tuple[int, int]]] = {}

    for object_id, experts in grouped.items():
        source = objects[object_id]
        count = len(experts)
        rows, columns = source["shape"]
        if rows % count:
            raise SystemExit(f"{object_id}: {rows} rows do not divide into {count} experts")
        per_expert_rows = rows // count
        shape = (per_expert_rows, columns)
        suffix = "gate_up" if any("down" not in roles for roles in experts.values()) else "down"

        planes = planes_of(source["shape"], source["format"], source["layout"])
        sizes = planes_of(shape, source["format"], source["layout"])
        expected = encoded_size(source["layout"], source["format"], shape)
        if sum(size[1] for size in sizes) != expected:
            raise SystemExit(f"{object_id}: plane sizes disagree with {expected}")

        element_block = per_expert_rows * columns
        for expert in range(count):
            new_id = f"weight/expert/{object_id.rsplit('/', 1)[-1]}/{expert:03d}/{suffix}"
            specs.append(TensorSpec(new_id, shape, source["format"], source["layout"]))
            splits[new_id] = [
                (source["offset"] + offset + expert * per_expert_rows * row_bytes, size)
                for (offset, _, row_bytes), (_, size, _) in zip(planes, sizes)
            ]
            for role, (begin, end, name) in experts[expert].items():
                shift = expert * element_block
                if begin < shift or end > shift + element_block:
                    raise SystemExit(f"{name}: range [{begin},{end}) leaves its expert block")
                bindings[name] = {
                    "parts": [{"object": new_id, "range": [begin - shift, end - shift]}]
                }

    kept = [
        ResourceSpec(obj["id"], obj["bytes"], obj["encoding"])
        if obj["kind"] == "resource"
        else TensorSpec(obj["id"], tuple(obj["shape"]), obj["format"], obj["layout"])
        for obj in description["objects"]
        if obj["id"] not in grouped
    ]
    planned = plan_objects(kept + specs)
    print(f"objects: {len(description['objects'])} -> {len(planned)} (split {len(specs)})")

    directory_description = {
        "components": description["components"],
        "objects": [obj.to_json() for obj in planned],
        "bindings": bindings,
        "uses": description["uses"],
        "metadata": description.get("metadata", {}),
        "provenance": description.get("provenance", {}),
    }
    payload = planned[-1].offset + planned[-1].bytes
    if args.dry_run:
        print(f"payload bytes: {description['files'][0]['payload_bytes']} -> {payload}")
        return 0
    if args.destination.exists():
        raise SystemExit(f"{args.destination}: already exists")

    directory, encoded, payload_offset = layout_directory(
        args.destination.name, directory_description, payload, max_file_bytes=1 << 62
    )
    if len(directory.files) != 1:
        raise SystemExit("this box writes a single entry; the payload must not need sharding")

    reader = Artifact.open(args.source)
    temporary = args.destination.with_name(f".{args.destination.name}.{uuid.uuid4().hex[:8]}.tmp")
    try:
        with open(temporary, "wb", buffering=0) as handle:
            handle.write(HEADER.pack(MAGIC, len(encoded), uuid.uuid4().bytes))
            handle.write(encoded)
            for obj in planned:
                handle.seek(payload_offset + obj.offset)
                if obj.id in splits:
                    for offset, size in splits[obj.id]:
                        for chunk in reader.iter_range(offset, size):
                            handle.write(chunk)
                else:
                    source = objects[obj.id]
                    if obj.bytes != source["bytes"]:
                        raise SystemExit(f"{obj.id}: size changed without a split")
                    for chunk in reader.iter_range(source["offset"], source["bytes"]):
                        handle.write(chunk)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, args.destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    finally:
        reader.close()
    print(f"wrote {args.destination} ({args.destination.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
