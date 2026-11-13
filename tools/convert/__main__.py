"""Convert explicitly selected local weights into a NInfer v3 artifact."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import importlib.util
from math import prod
from pathlib import Path
import sys
from collections.abc import Mapping

from .official_recipes import RECIPES
from .pipeline import convert
from .proposal import DEFAULT_RANKING, add_official_proposal
from . import gguf_dflash, gguf_qwen3_5, qwen3_5, qwen4_exp
from .gguf_qwen4exp import GGUFSource, open_source as open_gguf_source
from .recipe import Recipe
from .sources.gguf import open_gguf
from .sources.safetensors import SafetensorsSource


def _gguf_architecture(path):
    """The file's `general.architecture`, read without materializing any tensor."""
    file = open_gguf(path)
    try:
        return str(file.get("general.architecture", ""))
    finally:
        file.close()


def _open_base(path, *, gguf_config=None, gguf_root=None):
    """Open the primary checkpoint. A .gguf file is presented through its own source layer."""
    if Path(path).suffix.lower() == ".gguf":
        architecture = _gguf_architecture(path)
        if architecture == "qwen35":
            return gguf_qwen3_5.open_source(
                path, config_path=gguf_config, root=gguf_root
            )
        if architecture == "qwen4exp":
            return open_gguf_source(path, config_path=gguf_config, root=gguf_root)
        raise ValueError(
            f"{path}: GGUF architecture is {architecture!r}; this converter reads qwen4exp and "
            "qwen35 text sources (plus qwen35 mmproj vision)"
        )
    return SafetensorsSource(path)


def _builder(config):
    """Pick the architecture adapter from the checkpoint's declared architecture."""
    architectures = [str(item) for item in (config.get("architectures") or [])]
    if any(item.startswith("Qwen4Exp") for item in architectures):
        return qwen4_exp.build_model
    return qwen3_5.build_model


class SourceInputs(Mapping):
    """Named optional sources are opened only when a recipe or component requests one.

    A named source may be a `.gguf`: a draft head ships that way when only a GGUF distribution of
    the target exists, and its reader differs per architecture (`dflash` here). The dispatch is by
    extension, so `--source dflash2=X.gguf` and `--source dflash2=X.safetensors-dir` are the same
    command with different files.
    """

    def __init__(self, base, paths, stack, *, text_config, root):
        self._sources = {"base": base}
        self._paths = dict(paths)
        self._stack = stack
        self._text_config = text_config
        self._root = root

    def _open(self, name, path):
        if Path(path).suffix.lower() != ".gguf":
            return SafetensorsSource(path)
        if name not in ("dflash", "dflash2"):
            raise ValueError(
                f"--source {name}={path}: a .gguf named source is a draft head (dflash/dflash2); "
                "the text and vision GGUF files travel through --model, --mtp-gguf and --mmproj"
            )
        return gguf_dflash.open_companion(
            path, self._text_config, backend=name, root=self._root
        )

    def __getitem__(self, name):
        if name not in self._sources:
            if name not in self._paths:
                raise ValueError(
                    f"selected recipe requires source {name!r}; provide --source {name}=PATH"
                )
            self._sources[name] = self._stack.enter_context(
                self._open(name, self._paths[name])
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


def _report_plan(model, sources, components):
    """Print what a conversion would write: component geometry, counts, and source mapping.

    A 122 GiB conversion is not something to discover after the fact: this prints the component
    configs, the parameter/element count per component, and - for every GGUF draft - the tensor
    mapping that the artifact would take, then stops before touching the output path.
    """
    print(f"[dry-run] components: {', '.join(components)}")
    counts: dict[str, list[int]] = {}
    for name, parameter in model.parameters.items():
        component = name.split("/", 1)[0]
        entry = counts.setdefault(component, [0, 0])
        entry[0] += 1
        entry[1] += prod(parameter.shape)
    for component in components:
        count, elements = counts.get(component, [0, 0])
        geometry = ""
        config = model.components.get(component, {}).get("config", {})
        if component in ("dflash", "dflash2"):
            draft = config.get("dflash_config", {})
            geometry = (
                f" layers={config.get('num_hidden_layers')}"
                f" heads={config.get('num_attention_heads')}/{config.get('num_key_value_heads')}"
                f" head_dim={config.get('head_dim')}"
                f" inter={config.get('intermediate_size')}"
                f" ctx={config.get('max_position_embeddings')}"
                f" theta={config.get('rope_parameters', {}).get('rope_theta')}"
                f" taps={draft.get('target_layer_ids')}"
            )
            if "selector_rank" in draft:
                geometry += (
                    f" conv={draft.get('conv_kernel_size')}/{draft.get('conv_group_size')}"
                    f" selector={draft.get('selector_rank')}/{draft.get('selector_top_k')}"
                )
        print(
            f"[dry-run] {component}: {count} parameters, {elements:,} elements{geometry}"
        )
    for backend in ("dflash", "dflash2"):
        if backend not in components:
            continue
        source = sources[backend]
        pairs = getattr(source, "pairs", None)
        if pairs is None:
            print(
                f"[dry-run] {backend} source: {source.path} "
                f"({len(getattr(source, 'weight_map', {}))} tensors, HF names)"
            )
            continue
        print(f"[dry-run] {backend} source: {source.path} ({len(pairs)} tensors)")
        for logical, tensor in pairs.items():
            shape = tuple(source.file.tensors[tensor].shape)
            print(
                f"[dry-run]   {logical} <- {tensor} {shape} "
                f"{source.code_format(logical) or 'bf16'}"
            )
    total = sum(value[0] for value in counts.values())
    print(f"[dry-run] would write {total} parameters; nothing written")


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
    parser.add_argument("--mmproj", type=Path,
                        help="mmproj GGUF (llama.cpp vision projector) added as the vision component")
    parser.add_argument("--dflash2-gguf", type=Path,
                        help="standalone DFlash2 draft GGUF (llama.cpp `dflash` architecture) added "
                             "as the dflash2 component")
    parser.add_argument("--dflash-gguf", type=Path,
                        help="standalone DFlash draft GGUF (llama.cpp `dflash` architecture) added "
                             "as the dflash component")
    parser.add_argument("--dry-run", action="store_true",
                        help="build and report the plan (components, counts, source mapping) without "
                             "writing the artifact")
    args = parser.parse_args(argv)
    components = tuple(args.components.split(","))
    if len(components) != len(set(components)):
        raise ValueError("components must not repeat")
    if args.mtp_gguf is not None and "mtp" not in components:
        # A named draft source is the component: asking for it is enough.
        components = (*components, "mtp")
    if args.mmproj is not None and "vision" not in components:
        components = (*components, "vision")
    paths = _pairs(args.source, "source")
    if "base" in paths:
        raise ValueError("select the base source with --model")
    for backend, named in (
        ("dflash", args.dflash_gguf),
        ("dflash2", args.dflash2_gguf),
    ):
        if named is None:
            continue
        if backend in paths and paths[backend] != named:
            raise ValueError(
                f"--source {backend}={paths[backend]} and --{backend}-gguf={named} name two files"
            )
        paths[backend] = named
        if backend not in components:
            components = (*components, backend)
    is_gguf = Path(args.model).suffix.lower() == ".gguf"
    if is_gguf:
        gguf_extra = set(components) - {"text", "mtp", "vision", "dflash", "dflash2"}
        if gguf_extra:
            raise ValueError(
                f"a .gguf source converts {sorted(gguf_extra)} only when their own GGUF is named "
                "(--mtp-gguf, --mmproj, --dflash2-gguf)"
            )
        if "mtp" in components and args.mtp_gguf is None:
            raise ValueError("a .gguf source has no standalone draft head; pass --mtp-gguf")
        if "vision" in components and args.mmproj is None:
            raise ValueError("a .gguf source carries no tower; pass --mmproj")
        for backend in ("dflash", "dflash2"):
            if backend in components and backend not in paths:
                raise ValueError(
                    f"a .gguf text file carries no {backend} weights; pass their own draft GGUF "
                    f"with --{backend}-gguf PATH or --source {backend}=PATH"
                )
    elif args.mmproj is not None:
        raise ValueError("--mmproj applies to a .gguf source; a Safetensors checkpoint has the tower")
    overrides = _pairs(args.resource, "resource")
    with ExitStack() as stack:
        base = stack.enter_context(
            _open_base(args.model, gguf_config=args.gguf_config, gguf_root=args.gguf_root)
        )
        sources = SourceInputs(
            base,
            paths,
            stack,
            text_config=base.config.get("text_config") or base.config,
            root=args.gguf_root or args.model.parent,
        )
        companions = {
            key: sources[key] for key in ("dflash", "dflash2") if key in components
        }
        extra_builder = {}
        architectures = [str(item) for item in (base.config.get("architectures") or [])]
        is_qwen4exp = any(item.startswith("Qwen4Exp") for item in architectures)
        if args.mtp_gguf is not None:
            text_config = base.config["text_config"]
            source_type = GGUFSource if is_qwen4exp else gguf_qwen3_5.GGUFSource
            extra_builder["mtp_source"] = source_type.for_mtp(
                args.mtp_gguf, text_config,
                root=args.gguf_root or args.model.parent,
            )
            stack.callback(extra_builder["mtp_source"].close)
        if args.mmproj is not None:
            # An mmproj is a llama.cpp `clip` file: it describes the tower, not the text
            # architecture, so the same reader serves both families. qwen4_exp's builder already
            # accepts a `vision_source` (qwen4_exp.py:539, `tower = vision_source or base`); the
            # family check that used to reject it here was the only thing blocking the conversion.
            hidden = int(base.config["text_config"]["hidden_size"])
            extra_builder["vision_source"] = gguf_qwen3_5.open_vision_source(
                args.mmproj, hidden, root=args.gguf_root or args.model.parent
            )
            stack.callback(extra_builder["vision_source"].close)
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
        if args.dry_run:
            _report_plan(model, sources, components)
            return

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
