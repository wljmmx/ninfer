"""Carry whole components from one v3 artifact into another, byte for byte.

Why this exists: some components cannot be re-derived from the files a conversion reads. The
DFlash/DFlash2 draft heads ship as their own checkpoints, and a GGUF distribution of the target
model does not contain them. Rebuilding them would mean decoding and re-quantizing weights that
are already encoded here, which is exactly the loss this tool avoids: the carried objects are
copied as payload ranges, never interpreted.

The result is a new artifact with
  * the target's own components, objects, bindings, uses, metadata and provenance,
  * the carried components' directory records, verbatim,
  * the carried objects under fresh ids (`carried/<component>/<id>`), so nothing collides,
  * the use records that name a carried component - including the target parameters that take a
    carried activation as an input (the text output head consumes `dflash2/final_hidden`).

Usage:

    python -m tools.artifact.carry extended.ninfer \\
        --source old.ninfer --component dflash2 --out final.ninfer
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from .file_io import IO_CHUNK_BYTES
from .reader import Artifact
from .schema import ResourceSpec, TensorSpec
from .writer import DEFAULT_MAX_FILE_BYTES, ArtifactWriter


def _spec(obj) -> ResourceSpec | TensorSpec:
    if obj.kind == "resource":
        return ResourceSpec(obj.id, obj.bytes, obj.encoding)
    return TensorSpec(obj.id, tuple(obj.shape), obj.format, obj.layout)


def _named(record: dict, component: str) -> bool:
    for key in ("parameter", "input"):
        value = record.get(key)
        if isinstance(value, str) and value.startswith(component + "/"):
            return True
    return False


def carry(
    target: str | Path,
    *,
    source: str | Path,
    components: tuple[str, ...],
    output: str | Path,
    max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
) -> dict:
    target_path, source_path, output_path = Path(target), Path(source), Path(output)
    if output_path.exists():
        raise FileExistsError(f"carry output already exists: {output_path}")
    with Artifact(target_path) as new, Artifact(source_path) as old:
        records = {}
        for component in components:
            if component not in old.directory.components:
                raise ValueError(f"{source_path}: no component {component!r}")
            if component in new.directory.components:
                raise ValueError(f"{target_path}: already carries {component!r}")
            records[component] = old.directory.components[component]

        # Every object a carried binding names, renamed so the two artifacts' ids cannot collide.
        rename: dict[str, str] = {}
        bindings = dict(new.directory.bindings)
        for component in components:
            for name, entry in old.directory.bindings.items():
                if not name.startswith(component + "/"):
                    continue
                parts = entry.get("parts")
                if parts is None:
                    parts = [entry]
                fresh = []
                for part in parts:
                    original = part["object"]
                    rename.setdefault(original, f"carried/{component}/{original}")
                    fresh.append({**part, "object": rename[original]})
                bindings[name] = {"parts": fresh} if "parts" in entry else fresh[0]

        uses = list(new.directory.uses)
        for component in components:
            for record in old.directory.uses:
                if _named(record, component) and record not in uses:
                    uses.append(record)

        # The target's objects keep their ids and bytes; a carried object gets the fresh id and
        # keeps its source id only for reading.
        specs = [_spec(obj) for obj in new.directory.objects]
        sources: dict[str, tuple[Artifact, str]] = {
            obj.id: (new, obj.id) for obj in new.directory.objects
        }
        for original, fresh in rename.items():
            obj = old.by_id[original]
            if obj.kind == "resource":
                specs.append(ResourceSpec(fresh, obj.bytes, obj.encoding))
            else:
                specs.append(TensorSpec(fresh, tuple(obj.shape), obj.format, obj.layout))
            sources[fresh] = (old, original)

        components_out = dict(new.directory.components)
        components_out.update(records)
        provenance = dict(new.directory.provenance)
        provenance["carried_components"] = {
            component: {
                "from": str(source_path),
                "artifact_id": old.artifact_id.hex(),
                "objects": sum(1 for fresh in rename.values()
                               if fresh.startswith(f"carried/{component}/")),
            }
            for component in components
        }
        with ArtifactWriter(
            output_path,
            specs,
            components=components_out,
            bindings=bindings,
            uses=uses,
            metadata=new.directory.metadata,
            provenance=provenance,
            max_file_bytes=max_file_bytes,
        ) as writer:
            for object_id, (reader, original) in sources.items():
                writer.write_object(
                    object_id,
                    reader.iter_object(original, chunk_bytes=IO_CHUNK_BYTES),
                )
        return {
            "output": str(output_path),
            "components": ",".join(sorted(components_out)),
            "objects": len(writer.directory.objects),
            "payload_bytes": writer.directory.payload_bytes,
            "carried_objects": len(rename),
        }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", type=Path, help="artifact to extend")
    parser.add_argument("--source", type=Path, required=True,
                        help="artifact the components are copied from")
    parser.add_argument("--component", action="append", required=True,
                        help="component name to carry (repeatable)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-file-bytes", type=int, default=DEFAULT_MAX_FILE_BYTES)
    args = parser.parse_args(argv)
    report = carry(
        args.target,
        source=args.source,
        components=tuple(args.component),
        output=args.out,
        max_file_bytes=args.max_file_bytes,
    )
    for key, value in report.items():
        print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
