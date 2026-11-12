"""Convert explicitly selected local weights into a NInfer v3 artifact."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import importlib.util
from pathlib import Path
import sys
from collections.abc import Mapping

from .official_recipes import RECIPES
from .pipeline import convert
from .proposal import DEFAULT_RANKING, add_official_proposal
from . import qwen3_5, qwen4_exp
from .gguf_qwen4exp import GGUFSource, open_source as open_gguf_source
from .recipe import Recipe
from .sources.safetensors import SafetensorsSource


def _open_base(path, *, gguf_config=None, gguf_root=None):
    """Open the primary checkpoint. A .gguf file is presented through the GGUF source layer."""
    if Path(path).suffix.lower() == ".gguf":
        return open_gguf_source(path, config_path=gguf_config, root=gguf_root)
    return SafetensorsSource(path)


def _builder(config):
    """Pick the architecture adapter from the checkpoint's declared architecture."""
    architectures = [str(item) for item in (config.get("architectures") or [])]
    if any(item.startswith("Qwen4Exp") for item in architectures):
        return qwen4_exp.build_model
    return qwen3_5.build_model


class SourceInputs(Mapping):
    """Named optional sources are opened only when a recipe or component requests one."""

    def __init__(self, base, paths, stack):
        self._sources = {"base": base}
        self._paths = dict(paths)
        self._stack = stack

    def __getitem__(self, name):
        if name not in self._sources:
            if name not in self._paths:
                raise ValueError(
                    f"selected recipe requires source {name!r}; provide --source {name}=PATH"
                )
            self._sources[name] = self._stack.enter_context(
                SafetensorsSource(self._paths[name])
            )
        return self._sources[name]

    def __iter__(self):
        return iter(dict.fromkeys((*self._sources, *self._paths)))

    def __len__(self):
        return len(set(self._sources) | set(self._paths))

    def provenance(self):
        return {
            name: {"path": str(source.path)} for name, source in self._sources.items()
        }


def _pairs(values, label):
    result = {}
    for value in values:
        name, separator, path = value.partition("=")
        if not separator or not name or not path or name in result:
            raise ValueError(f"{label} requires unique NAME=PATH entries")
        result[name] = Path(path)
    return result


def _function(value: str):
    if value in RECIPES:
        return RECIPES[value]
    # "file.py:function" selects an entry point. Split on the last colon only when what follows
    # is an identifier: a bare Windows path ("C:\work\recipe.py") also contains a colon, and
    # splitting there would hand the loader a drive letter.
    filename, function = value, "configure"
    head, separator, tail = value.rpartition(":")
    if separator and tail.isidentifier():
        filename, function = head, tail
    path = Path(filename).resolve()
    spec = importlib.util.spec_from_file_location("ninfer_user_recipe", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load recipe file {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    result = getattr(module, function)
    if not callable(result):
        raise TypeError(f"{value}: recipe entry must be callable")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="primary checkpoint/config and default resources (.safetensors dir or .gguf)",
    )
    parser.add_argument(
        "--recipe",
        required=True,
        help="official name or Python file[:function]; defaults to configure in a file",
    )
    parser.add_argument(
        "--override",
        help="Python configuration applied after the selected recipe and proposal",
    )
    parser.add_argument(
        "--source",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="named source such as quantized, dflash or dflash2",
    )
    parser.add_argument(
        "--components",
        default="text",
        help="comma-separated text,vision,mtp,dflash,dflash2",
    )
    parser.add_argument(
        "--resource",
        action="append",
        default=[],
        metavar="ROLE=PATH",
        help="override a final frontend resource",
    )
    parser.add_argument(
        "--proposal",
        action="store_true",
        help="include the existing indexed proposal head",
    )
    parser.add_argument("--proposal-rows", type=int, default=131072)
    parser.add_argument("--ranking", type=Path, default=DEFAULT_RANKING)
    parser.add_argument("--name", help="public instance name saved in metadata")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--rows-per-chunk", type=int, default=512)
    parser.add_argument("--max-file-bytes", type=int, default=32_000_000_000)
    # GGUF input only: a GGUF carries weights and the llama.cpp hyperparameters, but not every
    # field of the HF config.json (rope_parameters' mrope layout, vision/mtp subtrees). The
    # sidecar is read when present; --gguf-root points at the directory holding the frontend
    # resources (tokenizer.json, chat_template.jinja) when they do not sit beside the GGUF.
    parser.add_argument("--gguf-config", type=Path,
                        help="config.json sidecar for a .gguf input")
    parser.add_argument("--gguf-root", type=Path,
                        help="resource directory for a .gguf input (defaults to its folder)")
    parser.add_argument("--mtp-gguf", type=Path,
                        help="standalone MTP GGUF (llama.cpp --nextn head) added as the mtp component")
    args = parser.parse_args(argv)
    components = tuple(args.components.split(","))
    if len(components) != len(set(components)):
        raise ValueError("components must not repeat")
    gguf_extra = set(components) - {"text", "mtp"} if Path(args.model).suffix.lower() == ".gguf" else set()
    if gguf_extra or (Path(args.model).suffix.lower() == ".gguf" and "mtp" in components
                      and args.mtp_gguf is None):
        raise ValueError(
            "a .gguf source converts the text component, plus mtp when --mtp-gguf names the draft "
            "head (vision ships as a separate mmproj)"
        )
    if args.mtp_gguf is not None and "mtp" not in components:
        raise ValueError("--mtp-gguf requires --components text,mtp")
    paths = _pairs(args.source, "source")
    if "base" in paths:
        raise ValueError("select the base source with --model")
    overrides = _pairs(args.resource, "resource")
    with ExitStack() as stack:
        base = stack.enter_context(
            _open_base(args.model, gguf_config=args.gguf_config, gguf_root=args.gguf_root)
        )
        sources = SourceInputs(base, paths, stack)
        companions = {
            key: sources[key] for key in ("dflash", "dflash2") if key in components
        }
        extra_builder = {}
        if args.mtp_gguf is not None:
            arch = [str(item) for item in (base.config.get("architectures") or [])]
            if not any(item.startswith("Qwen4Exp") for item in arch):
                raise ValueError("--mtp-gguf is implemented for the Qwen4Exp draft head")
            extra_builder["mtp_source"] = GGUFSource.for_mtp(
                args.mtp_gguf, base.config["text_config"],
                root=args.gguf_root or args.model.parent,
            )
            stack.callback(extra_builder["mtp_source"].close)
        model = _builder(base.config)(
            base,
            components=components,
            companions=companions,
            resource_overrides=overrides,
            **extra_builder,
        )
        recipe = Recipe(model)
        _function(args.recipe)(model, recipe, sources)
        if args.proposal:
            add_official_proposal(recipe, ranking=args.ranking, rows=args.proposal_rows)
        if args.override:
            _function(args.override)(model, recipe, sources)

        def progress(index, total, job):
            label = job.parameters[0]
            if len(job.parameters) > 1:
                label += f" (+{len(job.parameters)-1})"
            print(
                f"[{index+1}/{total}] {label}: {job.spec.format} {job.spec.shape}",
                flush=True,
            )

        provenance = {
            "converter": "ninfer-v3",
            "recipe": args.recipe,
            "sources": sources.provenance(),
        }
        if args.override:
            provenance["override"] = args.override
        if args.proposal:
            provenance["ranking"] = str(args.ranking)
        report = convert(
            model,
            recipe,
            args.out,
            name=args.name,
            provenance=provenance,
            device=args.device,
            rows_per_chunk=args.rows_per_chunk,
            max_file_bytes=args.max_file_bytes,
            progress=progress,
        )
        print(
            f"wrote {args.out}: {report['objects']} objects, {len(report['files'])} files, {report['seconds']:.1f}s",
            flush=True,
        )


if __name__ == "__main__":
    main()
