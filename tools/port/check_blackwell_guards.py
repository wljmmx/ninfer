#!/usr/bin/env python3
"""Pre-flight check: Blackwell-only entry points called outside their feature gate.

The upstream tree targets sm_120a only. Our port keeps the same sources but compiles the
Blackwell-only paths out on sm_89, so every call into them must sit inside the matching
feature gate. A missed gate is not caught until the compiler reaches that translation unit
minutes into a build; this script reports those call sites in a second so a merge can be
triaged before a full rebuild. The compiler stays the authority; this is a fail-fast aid.

Rule families (one gate set each):

  tma        launch_*tma* <...>          TMA tensor-core launchers (new upstream variants such
                                         as launch_bf16_a16_tma_mma / launch_q8_a16_tma_k32 are
                                         already covered by the launch_\\w*tma\\w* shape).
  tma_bulk   \\w*tma_load\\w*( / \\w*tma_store\\w*( / cp_async_bulk*( /
             cuTensorMapEncodeTiled(
                                         calls into the bulk-tensor copy helpers and the
                                         tensor-map descriptor API. The raw PTX and the helper
                                         definitions live in the *tma* / pipeline headers, which
                                         are only reachable from gated call sites, so the asm
                                         text itself is not what this rule chases.
  cluster    __cluster_dims__, cluster.sync(), cooperative_groups cluster API,
             cudaLaunchAttributeClusterDimension, cudaOccupancyMaxActiveClusters,
             cudaFuncAttributeNonPortableClusterSizeAllowed
                                         thread-block clusters (sm_90+, no sm_89 equivalent).
  tp2        moe_tp2 / tp2_fixture / test_tp2 / ops/moe/tp2
                                         the TP2 multi-rank group sealed as dead in T001: it is
                                         not merged, so any reference in src/ is a violation.
                                         Scanned in tests/ and bench/ only with
                                         --include-tests (the group's own files live there).

Deliberate scoping decisions (unchanged from the first version):

  * NVFP4 entry points are exempt from the tma rule: they are declared in both branches of
    NINFER_ENABLE_NVFP4 (the sm_89 branch defines a throwing stub), so an unguarded call
    compiles and reports at run time by design. The count of skipped sites is printed.
  * A call inside a *templated* lambda body ([&]<class S>() {...}) is not reported: such a
    body is only instantiated where the lambda is called, and the callers are already
    guarded. Plain lambdas are not exempt, so a genuine unguarded call is still caught.
  * Tokens that are portable to Ada are NOT checked even though they appear next to TMA code:
    bare mbarrier / cuda::barrier (sm_80+), cudaLaunchKernelEx without a cluster attribute
    (used upstream for cooperative split-K, see gdn control/kernels/mma_launch.cuh),
    cp.async (non-bulk). Flagging them would bury the real findings.
  * A definition is not a call (`void launch_x(...)`, `__device__ void tma_load_2d(...)`): a
    definition compiles on any arch, only a call from an ungated TU is the hazard. Comments
    and string literals are skipped too (the latter also silences the tp2 rule on the
    "TP2/ADP2" error message in src/models/*/parallel/plan.cpp).

Usage:
    python tools/port/check_blackwell_guards.py [--root SRC_DIR] [--include-tests] [--verbose]

Exit code 0 when clean, 1 when a Blackwell-only call site is outside its gate, 2 on bad root.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# --- rule table ---------------------------------------------------------------------------

TMA_GATES = ("NINFER_ENABLE_TMA", "NINFER_ENABLE_NVFP4")
CLUSTER_GATES = ("NINFER_ENABLE_TMA", "NINFER_ENABLE_CLUSTER")
TP2_GATES = ("NINFER_ENABLE_TP2",)


class Rule:
    def __init__(self, name, pattern, gates, hint, nvfp4_exempt=False):
        self.name = name
        self.pattern = re.compile(pattern)
        self.gates = gates
        self.hint = hint
        self.nvfp4_exempt = nvfp4_exempt


RULES = (
    Rule(
        "tma",
        r"\blaunch_\w*tma\w*\s*[<(]",
        TMA_GATES,
        "wrap the call in #if defined(NINFER_ENABLE_TMA) with an sm_89 fallback",
        nvfp4_exempt=True,
    ),
    Rule(
        "tma_bulk",
        r"\w*tma_load\w*\s*\(|\w*tma_store\w*\s*\(|\bcp_async_bulk\w*\s*\(|"
        r"\bcuTensorMapEncodeTiled\s*\(",
        TMA_GATES,
        "wrap the bulk-copy / tensor-map call in #if defined(NINFER_ENABLE_TMA) "
        "with an sm_89 cp.async path",
    ),
    Rule(
        "cluster",
        r"__cluster_dims__|cudaLaunchAttributeClusterDimension|"
        r"cudaFuncAttributeNonPortableClusterSizeAllowed|cudaOccupancyMaxActiveClusters|"
        r"\bcluster\.sync\s*\(|\bcluster_group\b|cooperative_groups::this_cluster|"
        r"\bthis_cluster\s*\(",
        CLUSTER_GATES,
        "clusters need sm_90+; wrap in #if defined(NINFER_ENABLE_CLUSTER) with an sm_89 path",
    ),
    Rule(
        "tp2",
        r"\bmoe_tp2\b|\btp2_fixture\b|\btest_tp2\b|\bops/moe/tp2\b|\bTP2\b|\btp2\b",
        TP2_GATES,
        "TP2 is sealed out (T001 dead list); do not reference it from src/",
    ),
)

# A call inside a templated lambda body is only instantiated at its (guarded) call sites.
TEMPLATED_LAMBDA = re.compile(r"\]\s*<[^>]*>\s*\(")
DIRECTIVE = re.compile(r"^\s*#\s*(if|ifdef|ifndef|elif|else|endif)\b(.*)$")
# A definition is not a call: `void launch_x(...)` / `__device__ void tma_load_2d(...)` compiles
# whether or not the feature is on; only a *call* from an ungated TU fails or misbehaves.
DEFINITION_PREFIX = re.compile(r"\b(?:void|auto)\s*$|__forceinline__\s*$")
COMMENT_PREFIX = ("//", "/*", "*", "*/")


def macro_mentioned(cond, macro):
    if re.search(r"!\s*defined\s*\(?\s*" + re.escape(macro), cond):
        return False
    if re.search(r"defined\s*\(?\s*" + re.escape(macro), cond):
        return True
    return cond.strip() == macro


class Gate:
    """Tracks whether the current line is inside a gate region for one rule."""

    def __init__(self, macros):
        self.macros = macros
        self.depth = None
        self.active = False

    def directive(self, kind, rest, depth):
        if kind in ("if", "ifdef", "ifndef"):
            if any(macro_mentioned(rest, g) for g in self.macros) and not self.active:
                self.active, self.depth = True, depth
        elif kind == "elif":
            if self.depth is not None:
                self.active = any(macro_mentioned(rest, g) for g in self.macros)
        elif kind == "else":
            if self.depth is not None:
                self.active = False
        elif kind == "endif":
            if self.depth is not None:
                self.active, self.depth = False, None

    def close(self, depth):
        if self.depth is not None and depth < self.depth:
            self.active, self.depth = False, None


def is_nvfp4(match_text):
    return "nvfp4" in match_text.lower()


def looks_like_definition(code, pos):
    return bool(DEFINITION_PREFIX.search(code[:pos]))


def inside_string(code, pos):
    """Odd number of unescaped double quotes before pos means the hit is in a literal."""
    return code.count('"', 0, pos) % 2 == 1


def scan(path):
    findings = []
    skipped = 0
    gates = {r.name: Gate(r.gates) for r in RULES}
    depth = 0
    defer_from = None
    in_defer = False

    for lineno, line in enumerate(
        path.read_text(encoding="utf-8", errors="replace").splitlines(), 1
    ):
        m = DIRECTIVE.match(line)
        if m:
            kind, rest = m.group(1), m.group(2)
            for gate in gates.values():
                gate.directive(kind, rest, depth)
            continue

        # Raw bulk-copy PTX and tensor-map helpers live in the *tma* / pipeline
        # implementation headers; those are only reachable from gated call sites, so the
        # rule checks calls, not the asm text, and skips definitions.
        stripped = line.lstrip()
        code = line.split("//", 1)[0] if "//" in line else line
        if stripped.startswith(COMMENT_PREFIX):
            code = ""

        # Detect the templated-lambda opener before the call check, so a body written on
        # the same line as its header ([&]<class S>() { call; }) is exempt too. A call
        # that starts before the opener, or in a plain lambda, is still reported.
        lambda_opens = TEMPLATED_LAMBDA.search(line)
        if lambda_opens:
            defer_from = depth

        for rule in RULES:
            reported = False
            for hit in rule.pattern.finditer(code):
                if reported or gates[rule.name].active or in_defer:
                    continue
                if lambda_opens and hit.start() >= lambda_opens.start():
                    continue
                if looks_like_definition(code, hit.start()):
                    continue
                if inside_string(code, hit.start()):
                    continue
                if rule.nvfp4_exempt and is_nvfp4(hit.group(0)):
                    skipped += 1
                    continue
                reported = True
                findings.append(
                    (rule, lineno, line.strip()[:110])
                )

        depth += line.count("{") - line.count("}")
        if defer_from is not None and depth > defer_from:
            in_defer = True
        if in_defer and depth <= defer_from:
            in_defer, defer_from = False, None
        for gate in gates.values():
            gate.close(depth)
    return findings, skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None, help="source root (default <repo>/src)")
    ap.add_argument("--include-tests", action="store_true",
                    help="also scan tests/ and bench/ (the TP2 group lives there)")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[2]
    roots = [Path(args.root)] if args.root else [repo / "src"]
    if args.include_tests and not args.root:
        roots += [repo / "tests", repo / "bench"]
    for root in roots:
        if not root.is_dir():
            print(f"error: {root} is not a directory", file=sys.stderr)
            return 2

    files = []
    for root in roots:
        files += [p for p in sorted(root.rglob("*"))
                  if p.suffix in {".cu", ".cuh", ".cpp", ".h"}]

    findings = []
    skipped = 0
    for p in files:
        hits, nvfp4_skipped = scan(p)
        skipped += nvfp4_skipped
        findings.extend((h[0], p, h[1], h[2]) for h in hits)
        if args.verbose and not hits:
            print(f"ok   {p.relative_to(repo)}")

    if findings:
        counts = {}
        for rule, _, _, _ in findings:
            counts[rule.name] = counts.get(rule.name, 0) + 1
        summary = ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
        print(f"FAIL: {len(findings)} ungated Blackwell-only call site(s) ({summary})\n")
        for rule, path, lineno, text in findings:
            rel = path.relative_to(repo) if path.is_relative_to(repo) else path
            print(f"  {rule.name:9s} {rel}:{lineno}: {text}")
        print()
        for rule in RULES:
            if rule.name in counts:
                print(f"  [{rule.name}] {rule.hint}")
        return 1

    extra = f", {skipped} NVFP4 site(s) exempt" if skipped else ""
    print(f"ok: {len(files)} file(s) scanned, no ungated Blackwell-only call sites{extra}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
