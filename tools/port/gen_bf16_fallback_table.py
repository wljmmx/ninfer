#!/usr/bin/env python3
"""Generate docs/maintainer/sm89-fallback-table.md from the bf16 shape sources.

Upstream targets sm_120a and serves the large-token tail of its bf16 linear shapes with
TMA schedules, which sm_89 lacks. Each shape guards that tail with
`#if defined(NINFER_ENABLE_TMA)` and falls back in `#else`. This tool reads the shape files
and prints the *like-for-like* Ada replacement for every TMA instance, following the
inheritance declared in bf16_schedule.cuh.

Usage:
    python tools/port/gen_bf16_fallback_table.py [--shapes DIR] > docs/maintainer/sm89-fallback-table.md
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

USING = re.compile(r"using\s+(S\d+)\s*=\s*(.*?);", re.S)
TMA = re.compile(r"Bf16A16TmaMmaSchedule\s*<([^>]*)>")
STEP = re.compile(r"(?:if\s*\(\s*tokens\s*<=\s*(\d+)\s*\)\s*)?return\s+launch_bf16_(\w+)\s*<\s*(S\d+)\s*>")
ELSE = re.compile(r"return\s+launch_bf16_(\w+)\s*<\s*(S\d+)\s*>")

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

- **Recorded, not wired in.** The `#else` branches still keep their conservative reuse; the
  fallbacks below are proposed replacements, verified by static reasoning only - not yet
  compiled or benchmarked on device.
- **Stage 2 is deferred** (no target model family): every shape here is inert for the
  current baseline, qwen3.8-27b (hidden = 5120).
- Regenerate with `python tools/port/gen_bf16_fallback_table.py`.

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


def analyze(path: Path):
    text = path.read_text(encoding="utf-8", errors="replace")
    defs = {}
    for m in USING.finditer(text):
        sid, rhs = m.group(1), m.group(2).strip()
        t = TMA.search(rhs)
        if t:
            defs[sid] = (rhs, rhs.replace(t.group(0), ada_schedule(t.group(1))))
    g = re.search(r"#if\s+defined\(NINFER_ENABLE_TMA\)(.*?)#else(.*?)#endif", text, re.S)
    if not g:
        return None
    pre = [int(x) for x in re.findall(r"tokens\s*<=\s*(\d+)", text[:g.start()])]
    prev = pre[-1] if pre else 0
    steps = []
    for mm in STEP.finditer(g.group(1)):
        up = int(mm.group(1)) if mm.group(1) else None
        steps.append((prev, up, mm.group(2), mm.group(3)))
        if up is not None:
            prev = up
    em = ELSE.search(g.group(2))
    else_txt = f"`{em.group(2)}` (`{em.group(1)}`)" if em else "`None`"
    return defs, steps, else_txt


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", default=None, help="shape dir (default <repo>/src/ops/linear/bf16/shapes)")
    ap.add_argument("--out", default=None, help="write to file (UTF-8, LF) instead of stdout")
    args = ap.parse_args()
    root = Path(args.shapes) if args.shapes else Path(__file__).resolve().parents[2] / "src/ops/linear/bf16/shapes"
    if not root.is_dir():
        print(f"error: {root} is not a directory", file=sys.stderr)
        return 2
    files = sorted(p for p in root.glob("*.cu")
                   if "Bf16A16TmaMmaSchedule" in p.read_text(encoding="utf-8", errors="replace"))
    out = [HEADER, "\n## Per-shape ladders\n"]
    for p in files:
        r = analyze(p)
        if not r:
            continue
        defs, steps, else_txt = r
        out.append(f"### `{p.name}`\n")
        out.append("| token range | TMA instance (sm_120a) | Ada fallback (sm_89) |")
        out.append("|---|---|---|")
        for lo, hi, launcher, sid in steps:
            if launcher != "tma_mma":
                continue
            h = str(hi) if hi is not None else "inf"
            rhs, ada_rhs = defs[sid]
            out.append(f"| ({lo}, {h}] | `{flat(rhs)}` | `{flat(ada_rhs)}` |")
        out.append(f"\ncurrent `#else` fallback: {else_txt}\n")
    text = "\n".join(out)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8", newline="\n")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
