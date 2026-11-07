#!/usr/bin/env python3
"""Generate docs/maintainer/sm89-fallback-table.md from the bf16 shape sources.

Upstream targets sm_120a and serves the large-token tail of its bf16 linear shapes with
TMA schedules, which sm_89 lacks. Each shape guards that tail with
`#if defined(NINFER_ENABLE_TMA)` and falls back in `#else`. This tool reads the shape files
and prints the *like-for-like* Ada replacement for every TMA instance, following the
inheritance declared in bf16_schedule.cuh.

Usage:
    python tools/port/gen_bf16_fallback_table.py [--shapes DIR] > docs/maintainer/sm89-fallback-table.md

T004 additions (draft mode, used to size the upstream merge):
    python tools/port/gen_bf16_fallback_table.py --shapes <upstream>/src/ops/linear --recursive \
        --instances <upstream>/src/ops/linear/kernels/bf16/bf16_instances.cuh --draft

`--draft` prints only a banner plus one ladder per shape (no page header/coverage text), so
the output can be appended to the table as a review draft. `--recursive` walks every
`shapes/` directory under `--shapes`; `--instances` resolves the named bf16 aliases
(Bf16A16TmaR64T64K64S3 ...) that the newest shape files use instead of inline schedules.
The mapping rule itself is unchanged: only the producer swizzle changes (Tma128 -> Xor64)
and ConsumerKUnroll is dropped.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

USING = re.compile(r"using\s+(S\d+|[A-Za-z_]\w*)\s*=\s*(.*?);", re.S)
TMA = re.compile(r"Bf16A16TmaMmaSchedule\s*<([^>]*)>")
ELSE = re.compile(r"return\s+launch_bf16_(\w+)\s*<\s*(S\d+)\s*>")

# TMA selectors, in priority order: a launcher call, then the three route forms the newer
# shape files use (`routes.template tma<S5>()`, `tma.template operator()<Alias>()`, `tma<S>()`).
SELECTOR = (
    re.compile(r"\blaunch_\w*tma\w*\s*<\s*(\w+)"),
    re.compile(r"\.template\s+tma\s*<\s*(\w+)"),
    re.compile(r"\btma\.template\s+operator\s*\(\s*\)\s*<\s*(\w+)"),
    re.compile(r"\btma\s*<\s*(\w+)"),
)
CALL = re.compile(
    r"\blaunch_\w+\s*<|\.template\s+(?:operator\s*\(\s*\)|\w+)\s*<"
)
BOUND = re.compile(r"tokens\s*<=\s*(\d+)")

HEADER = """# sm_89 fallback table for the upstream bf16 shapes

Upstream targets **sm_120a (RTX 5090)**. Its bf16 linear shapes serve the large-token tail
with TMA schedules (`Bf16A16TmaMmaSchedule`), which do not exist on sm_89. Each shape wraps
that tail in `#if defined(NINFER_ENABLE_TMA)` and, in the `#else`, falls back to **this
shape's own cp.async schedule** (a coarse, conservative choice). This table records the
**like-for-like** Ada replacement for every TMA instance, so stage 2 of
`upstream_absorption_plan.md` becomes a mechanical edit plus measurement rather than a
design exercise.

## Mapping rule

The replacement follows the inheritance already declared in
`src/ops/linear/bf16/bf16_schedule.cuh` (the TMA schedule derives from the cp.async one
with identical tile geometry):

```
Bf16A16TmaMmaSchedule<R,T,K,WR,WT,S,M[,Raster[,RGR[,ConsumerKUnroll]]]>
  : Bf16A16MmaSchedule<R,T,K,WR,WT,S,M, cg, cg, PingPong, Raster, Tma128, RGR>

Ada fallback:
Bf16A16MmaSchedule<R,T,K,WR,WT,S,M, cg, cg, PingPong, Raster, Xor64, RGR>
```

Only the swizzle changes (**Tma128** producer staging -> cp.async **Xor64**); tile geometry,
stages, raster, min-blocks-per-SM and any outer wrapper (`Bf16ScheduleInstance`,
`Bf16RowTailSchedule`, `Bf16KTailSchedule`) are preserved. `ConsumerKUnroll` has no cp.async
analogue and is dropped. Each fallback satisfies the cp.async schedule's static asserts on
sm_89 (`kBlockRows % kWarpRows == 0`, `kBlockTokens % kWarpTokens == 0`, `kBlockK % 64 == 0`,
`2 <= kStages <= 8`, `kSharedBytes <= 99 KiB`).

## Status

- **Wired in and measured on sm_89 (RTX 4090).** The `#else` branches of all sixteen
  ladders below run the recorded cp.async fallback per token range. Verified by
  `ninfer_linear_bf16_a16_test` and by interleaved A/B `linear_bench --sweep 512:4096:512`
  (min, repeat 200, flush 96 MiB): every wired rung is a win or neutral except the two
  documented deviations in `n1152_k1152.cu` and `n1152_k4304.cu` (see below).
- **Measured deviations.** `n1152_k1152`: the 32x64 rung regressed +21% at (128, 256] and
  the 64x128 / 128x128 rungs wave-quantize (9 row tiles x tokens/128 = 144 blocks = 2
  unbalanced waves on 128 SMs), so that shape runs S5 to 256 and the 64x64 F7 beyond.
  `n1152_k4304`: the 64x64 / 64x128 rungs regressed +23%/+14% over (256, 1024] and the
  128x128 rung quantizes at 2048 tokens, so those spans keep S4 while (116, 256] and
  (1280, 1792] / (2048, inf] keep the wired fallbacks.
- **Stage 2 is deferred** (no target model family): every shape here is inert for the
  current baseline, qwen3.8-27b (hidden = 5120).
- Regenerate with `python tools/port/gen_bf16_fallback_table.py`. The generator reads the
  `#if NINFER_ENABLE_TMA` ladder only; the two measured deviations live in the `#else`
  branches and are not regenerated.

`token range` is the token interval that selects the instance; `inf` is the final `return`
(no upper bound). Only TMA-selected steps get a fallback row; a non-TMA step interleaved
inside the guard still advances the interval.

## Coverage

Sixteen shape files carry an inline TMA ladder (listed below). Three further files
(`n14336_k5120.cu`, `n256_k5120.cu`, `n5120_k6144.cu`) reach TMA through the pre-existing
aliases in `bf16_instances.cuh`; they predate this merge and already ship an `#else`
fallback. `n48_k2560.cu` and `n96_k2560.cu` contain no TMA ladder.

---

"""

DRAFT_HEADER = """## T004 初稿：上游新增 bf16 形状的 sm_89 fallback（待 T016 复核/实测）

> 范围：`upstream/master`（a754fcf3）中**本地树尚无同名 shape**、且不在 T001 判死组
> （`*/dgx_spark/*`，本次排除 22 个 dgx_spark bf16 梯度）里的 bf16 TMA 梯度：
> `linear/add/bf16` ×1、`linear/split/bf16/rtx5090` ×3、`linear/split/bf16` 共享 ×1，
> 共 5 个 shape / 20 行。映射规则同上，仅换生产者 swizzle（Tma128→Xor64）并丢掉
> ConsumerKUnroll；**未编译、未实测**，行内 token 区间只表达上游的分档意图。
>
> 生成命令：
> `python tools/port/gen_bf16_fallback_table.py --shapes <upstream>/src/ops/linear --recursive
> --instances <upstream>/src/ops/linear/kernels/bf16/bf16_instances.cuh --draft
> --only split/bf16/rtx5090 --only split/bf16/shapes/ --only add/bf16`
>
> 上游这批文件没有 `#if defined(NINFER_ENABLE_TMA)`（上游只按目录/目标选编译单元），搬入时
> 需先补门控，`#else` 分支按本表落 cp.async 回退，再由 T016 实测确认/修正。
>
> 新增入口 `--recursive`（遍历 shapes/）、`--instances`（解析 `Bf16A16TmaR64T64K64S3` 这类别名）、
> `--only`（路径过滤）、`--draft`（只出 banner + 梯度表），默认输出与旧版逐字一致。

"""


def ada_schedule(tma_args: str) -> str:
    a = [x.strip() for x in tma_args.split(",")]
    core = a[:7]
    raster = a[7] if len(a) > 7 else "Bf16MmaRaster::TokenFast"
    rgr = a[8] if len(a) > 8 else "1"
    return ("Bf16A16MmaSchedule<" + ", ".join(core) +
            f", Cache::cg, Cache::cg, Bf16MmaFragmentPipeline::PingPong, {raster}, "
            f"Bf16MmaSwizzle::Xor64, {rgr}>")


def flat(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def load_aliases(path: Path | None) -> dict[str, tuple[str, str]]:
    """`using Bf16A16TmaR64T64K64S3 = Bf16A16TmaMmaSchedule<...>;` -> (rhs, ada_rhs)."""
    if not path or not path.is_file():
        return {}
    text = path.read_text(encoding="utf-8", errors="replace")
    out = {}
    for m in USING.finditer(text):
        name, rhs = m.group(1), m.group(2).strip()
        t = TMA.search(rhs)
        if t:
            out[name] = (rhs, rhs.replace(t.group(0), ada_schedule(t.group(1))))
    return out


def tma_selector(line: str):
    for rx in SELECTOR:
        m = rx.search(line)
        if m:
            return m.group(1)
    return None


def analyze(path: Path, aliases: dict[str, tuple[str, str]]):
    text = path.read_text(encoding="utf-8", errors="replace")
    defs = {}
    for m in USING.finditer(text):
        sid, rhs = m.group(1), m.group(2).strip()
        t = TMA.search(rhs)
        if t:
            defs[sid] = (rhs, rhs.replace(t.group(0), ada_schedule(t.group(1))))
    defs.update({k: v for k, v in aliases.items() if k not in defs})

    g = re.search(r"#if\s+defined\(NINFER_ENABLE_TMA\)(.*?)#else(.*?)#endif", text, re.S)
    if g:
        body = g.group(1)
        # Interval starts at the last token bound before the guard.
        pre = [int(x) for x in BOUND.findall(text[:g.start()])]
        prev = pre[-1] if pre else 0
        em = ELSE.search(g.group(2))
        else_txt = f"`{em.group(2)}` (`{em.group(1)}`)" if em else "`None`"
    else:
        # Upstream has no NINFER_ENABLE_TMA gate: the ladder runs over the whole file and
        # the port has to add the gate (and an #else) when it merges the shape.
        body = text
        prev = 0
        else_txt = "`None`（上游无门控，搬入时补）"

    steps = []
    pending = None       # token bound whose call has not been seen yet
    awaiting = False
    for raw in body.splitlines():
        line = raw.strip()
        m = BOUND.search(line)
        if m:
            pending, awaiting = int(m.group(1)), True
        elif line.startswith("else") and not line.startswith("else if"):
            pending, awaiting = None, True  # final else: no upper bound
        if not CALL.search(line):
            continue
        hi = pending if awaiting else None
        sid = tma_selector(line)
        if sid and sid in defs:
            steps.append((prev, hi, sid))
            if hi is not None:
                prev = hi
        elif hi is not None:
            prev = hi  # a non-TMA rung still advances the token interval
        pending, awaiting = None, False
    return defs, steps, else_txt


def render(path: Path, defs, steps, else_txt: str, note: str | None = None,
           title: str | None = None):
    head = f"### {title or '`' + path.name + '`'}" + (f"\n\n`{note}`" if note else "") + "\n"
    out = [head]
    out.append("| token range | TMA instance (sm_120a) | Ada fallback (sm_89) |")
    out.append("|---|---|---|")
    rows = 0
    missing = []
    for lo, hi, sid in steps:
        if sid not in defs:
            missing.append(sid)
            continue
        h = str(hi) if hi is not None else "inf"
        rhs, ada_rhs = defs[sid]
        out.append(f"| ({lo}, {h}] | `{flat(rhs)}` | `{flat(ada_rhs)}` |")
        rows += 1
    if missing:
        out.append(f"\n未解析的实例（上游未在本文件/别名表内声明）：`{', '.join(sorted(set(missing)))}` —— 待移植时实测确定\n")
    out.append(f"\ncurrent `#else` fallback: {else_txt}\n")
    return "\n".join(out), rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", default=None, help="shape dir (default <repo>/src/ops/linear/bf16/shapes)")
    ap.add_argument("--out", default=None, help="write to file (UTF-8, LF) instead of stdout")
    ap.add_argument("--recursive", action="store_true", help="walk every shapes/ dir under --shapes")
    ap.add_argument("--instances", default=None,
                    help="bf16_instances.cuh to resolve named schedule aliases")
    ap.add_argument("--draft", action="store_true", help="emit only a draft banner plus ladders")
    ap.add_argument("--only", action="append", default=[],
                    help="keep only shapes whose path contains this substring (repeatable)")
    args = ap.parse_args()
    root = Path(args.shapes) if args.shapes else Path(__file__).resolve().parents[2] / "src/ops/linear/bf16/shapes"
    if not root.is_dir():
        print(f"error: {root} is not a directory", file=sys.stderr)
        return 2
    aliases = load_aliases(Path(args.instances)) if args.instances else {}

    def has_tma(p: Path) -> bool:
        text = p.read_text(encoding="utf-8", errors="replace")
        if "Bf16A16TmaMmaSchedule" in text:
            return True
        # Newer shapes name the schedule through an alias (Bf16A16TmaR64T64K64S3 ...).
        return any(a in text for a in aliases)

    pattern = "**/*.cu" if args.recursive else "*.cu"
    files = sorted(p for p in root.glob(pattern) if has_tma(p))
    if args.only:
        files = [p for p in files
                 if any(s in p.as_posix() for s in args.only)]
    out = [DRAFT_HEADER.rstrip("\n"), ""] if args.draft else [HEADER, "\n## Per-shape ladders\n"]
    total = 0
    for p in files:
        r = analyze(p, aliases)
        if not r:
            continue
        defs, steps, else_txt = r
        if not steps:
            continue
        if args.draft:
            rel = p.as_posix()
            m = re.search(r"src/ops/(.*)/shapes/.*$", rel)
            family = m.group(1) if m else "?"
            title = f"`{p.name}` — `{family}`"
            text, rows = render(p, defs, steps, else_txt, note=(m.group(0) if m else rel),
                                title=title)
        else:
            text, rows = render(p, defs, steps, else_txt)
        total += rows
        out.append(text)
    text = "\n".join(out)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8", newline="\n")
    else:
        print(text)
    if args.draft:
        print(f"\n<!-- draft rows: {total} -->", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
