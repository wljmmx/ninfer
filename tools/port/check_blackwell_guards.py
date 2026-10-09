#!/usr/bin/env python3
"""Pre-flight check: TMA entry points called outside NINFER_ENABLE_TMA.

The upstream tree targets sm_120a. Some launchers (launch_bf16_tma_mma) are declared only
under #if defined(NINFER_ENABLE_TMA), so calling one from a build that lacks TMA fails to
compile ("identifier ... is undefined") minutes into the build. This script reports those
call sites in a second, so a merge can be triaged before a full rebuild. The compiler
stays the authority; this is a fail-fast aid.

Two deliberate scoping decisions:
  * NVFP4 is not checked: those entry points are declared in both branches of
    NINFER_ENABLE_NVFP4 (the sm_89 branch defines a throwing stub), so an unguarded call
    is the intended design.
  * A call inside a *templated* lambda body ([&]<class S>() {...}) is not reported: such a
    body is only instantiated where the lambda is called, and the callers are already
    guarded. Plain lambdas are not exempt, so a genuine unguarded call is still caught.

Usage:
    python tools/port/check_blackwell_guards.py [--root SRC_DIR] [--verbose]

Exit code 0 when clean, 1 when a TMA call site is outside its gate.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

CALL = re.compile(r"\blaunch_\w*tma\w*\s*<")
MACRO = "NINFER_ENABLE_TMA"
# NVFP4 code only exists on Blackwell, so its guard also gates TMA calls.
GATING_MACROS = ("NINFER_ENABLE_TMA", "NINFER_ENABLE_NVFP4")
TEMPLATED_LAMBDA = re.compile(r"\]\s*<[^>]*>\s*\(")
DIRECTIVE = re.compile(r"^\s*#\s*(if|ifdef|ifndef|elif|else|endif)\b(.*)$")


def macro_mentioned(cond, macro=MACRO):
    if re.search(r"!\s*defined\s*\(?\s*" + re.escape(macro), cond):
        return False
    if re.search(r"defined\s*\(?\s*" + re.escape(macro), cond):
        return True
    return cond.strip() == macro


def scan(path):
    findings = []
    guard_depth = None       # brace depth at which the TMA guard region opened
    depth = 0
    in_guard = False
    defer_from = None        # brace depth at which a templated lambda body opened
    in_defer = False

    for lineno, line in enumerate(
        path.read_text(encoding="utf-8", errors="replace").splitlines(), 1
    ):
        m = DIRECTIVE.match(line)
        if m:
            kind, rest = m.group(1), m.group(2)
            if kind in ("if", "ifdef", "ifndef"):
                if any(macro_mentioned(rest, g) for g in GATING_MACROS) and not in_guard:
                    in_guard, guard_depth = True, depth
            elif kind == "elif":
                if guard_depth is not None:
                    in_guard = any(macro_mentioned(rest, g) for g in GATING_MACROS)
            elif kind == "else":
                if guard_depth is not None:
                    in_guard = False
            elif kind == "endif":
                if guard_depth is not None:
                    in_guard, guard_depth = False, None
            continue

        # Detect the templated-lambda opener before the call check, so a body written
        # on the same line as its header ([&]<class S>() { call; }) is exempt too. A
        # call that starts before the opener, or in a plain lambda, is still reported.
        lambda_opens = TEMPLATED_LAMBDA.search(line)
        if lambda_opens:
            defer_from = depth

        call = CALL.search(line)
        if call and not in_guard and not in_defer:
            if not (lambda_opens and call.start() >= lambda_opens.start()):
                findings.append(
                    f"{path}:{lineno}: TMA call outside #if defined({MACRO}): {line.strip()[:100]}"
                )

        depth += line.count("{") - line.count("}")
        if defer_from is not None and depth > defer_from:
            in_defer = True
        if in_defer and depth <= defer_from:
            in_defer, defer_from = False, None
        if guard_depth is not None and depth < guard_depth:
            in_guard, guard_depth = False, None
    return findings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None, help="source root (default <repo>/src)")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    root = Path(args.root) if args.root else Path(__file__).resolve().parents[2] / "src"
    if not root.is_dir():
        print(f"error: {root} is not a directory", file=sys.stderr)
        return 2

    files = [
        p
        for p in sorted(root.rglob("*"))
        if p.suffix in {".cu", ".cuh", ".cpp", ".h"}
    ]
    findings = []
    for p in files:
        hits = scan(p)
        findings.extend(hits)
        if args.verbose and not hits:
            print(f"ok   {p}")

    if findings:
        print(f"FAIL: {len(findings)} TMA call site(s) outside #if defined({MACRO})\n")
        for f in findings:
            print("  " + f)
        print("\nWrap each site in #if defined(NINFER_ENABLE_TMA) with an sm_89 fallback.")
        return 1
    print(f"ok: {len(files)} file(s) scanned, no ungated TMA call sites")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
