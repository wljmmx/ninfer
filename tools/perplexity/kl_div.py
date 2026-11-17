#!/usr/bin/env python3
"""KL(p||q) per position segment from two ``ninfer-perplexity --token-topk`` exports.

The input is the TSV that ``ninfer-perplexity --token-topk K`` writes into its output directory
(``token_topk.tsv``):

    # index<TAB>token<TAB>logprob<TAB>log_normalizer<TAB>topk_ids<TAB>topk_logprobs
    1	1234	-0.00411	11.87	1234,77,...	-0.00411,-3.5,...

Every row carries p over a *candidate list* plus the column's full log-normalizer, which is what a
KL needs: the plain ``--token-logprobs`` export has one number per position and no support, so the
mean of its differences is a cross-entropy difference (CE_Q - CE_P), never KL(p||q).

The estimator, for one position with reference p and candidate q. Both sides are proper
distributions over the same two-part support - the candidates the two lists share, plus one bucket:

    KL = sum over i in candidates(p) n candidates(q) of  p_i * (log p_i - log q_i)
       + P_bucket * (log P_bucket - log Q_bucket)

    P_bucket = p's mass outside the shared candidates = p's unranked candidates + 1 - coverage(p)
    Q_bucket = q's mass outside the shared candidates = q's unranked candidates + 1 - coverage(q)

  * ``log p_i`` is the exported log-probability of candidate i; ``p_i = exp(log p_i)``, and the
    same relation holds for q's exported values. Both exported values are used as they are: with
    the two buckets above, each side already sums to exactly 1 over {shared candidates, bucket}.
  * The bucket is where q's residual mass is charged, and where p's is, symmetrically. Every token
    one model ranks and the other does not lands in it, so no mass is dropped and no mass is
    counted twice - the two failure modes of a per-token patching rule. It also means two identical
    runs produce a bucket of 0 on both sides and KL exactly 0.
  * Because both sides sum to 1 over {shared candidates, bucket}, the value is a KL between two
    normalized distributions and cannot be negative. If q's list covers everything and p puts mass
    outside it, the bucket's q probability is floored instead of being zero-probability (-inf).

The bucket is a coarse object: it aggregates tokens whose individual probabilities were never
exported, so the number describes the exported support plus the mass outside it, not every token.
``coverage_p``/``coverage_q`` are reported for every segment so a reader can see how much of each
distribution the comparison actually covered.

Usage:

    python tools/perplexity/kl_div.py a/token_topk.tsv b/token_topk.tsv
    python tools/perplexity/kl_div.py a/token_topk.tsv b/token_topk.tsv --json kl.json
    python tools/perplexity/kl_div.py a.tsv b.tsv --segments 512,1024,2048

The default segments end at index 1K, 2K and 4K (``1K``, ``2K``, ``4K``, ``4K+`` plus a total row).
KL is not symmetric: the first file is p (the reference), the second is q.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

TSV_COLUMNS = ("index", "token", "logprob", "log_normalizer", "topk_ids", "topk_logprobs")
DEFAULT_SEGMENTS = (1024, 2048, 4096)
# Probability floor for a token q's candidate list does not contain while q's residual mass is zero
# (q's list already holds every bit of its mass, so the token is -inf under q). Finite floor keeps
# the row readable and the count of such tokens is reported instead.
RESIDUAL_FLOOR = 1.0e-30
RESIDUAL_SUM_TOLERANCE = 1.0e-3


class KlError(Exception):
    """Input problem that stops the comparison instead of producing a number."""


@dataclass(frozen=True)
class Position:
    """One scored position: the observed token, p's candidates, and the column normalizer."""

    index: int
    token: int
    logprob: float
    log_normalizer: float
    ids: tuple[int, ...]
    logprobs: tuple[float, ...]

    @property
    def k(self) -> int:
        return len(self.ids)

    def coverage(self) -> float:
        return math.fsum(math.exp(value) for value in self.logprobs)


@dataclass(frozen=True)
class KlTerms:
    """One position's contribution and the two truncation diagnostics that qualify it."""

    kl: float
    coverage_p: float
    coverage_q: float
    # Candidates p ranks that q's list does not contain.
    tail_tokens: int
    # p's mass in the aggregated bucket: those candidates plus p's own residual 1 - coverage(p).
    tail_mass: float


def _coerce_int(value: str, path: Path, line: int, name: str) -> int:
    try:
        return int(value)
    except ValueError:
        raise KlError(f"{path}:{line}: {name} is not an integer: {value!r}") from None


def _coerce_float(value: str, path: Path, line: int, name: str) -> float:
    try:
        number = float(value)
    except ValueError:
        raise KlError(f"{path}:{line}: {name} is not a number: {value!r}") from None
    if not math.isfinite(number):
        raise KlError(f"{path}:{line}: {name} is not finite: {value!r}")
    return number


def _parse_list(value: str, path: Path, line: int, name: str) -> tuple[str, ...]:
    items = tuple(part.strip() for part in value.split(","))
    if not items or any(not item for item in items):
        raise KlError(f"{path}:{line}: {name} must be a non-empty comma-separated list")
    return items


def read_topk(path: Path) -> list[Position]:
    """Parse one top-K export, refusing anything that would make a KL ambiguous."""
    if not path.is_file():
        raise KlError(f"{path}: no such file")
    lines = path.read_text(encoding="utf-8", errors="strict").splitlines()
    columns: tuple[str, ...] | None = None
    positions: list[Position] = []
    k: int | None = None
    for offset, raw in enumerate(lines):
        line = offset + 1
        text = raw.strip()
        if not text:
            continue
        if text.startswith("#"):
            names = tuple(name.strip() for name in text.lstrip("#").split("\t") if name.strip())
            if all(name in names for name in TSV_COLUMNS):
                columns = names
            continue
        if columns is None:
            raise KlError(
                f"{path}:{line}: data before the '# index\\ttoken\\tlogprob\\tlog_normalizer\\t"
                "topk_ids\\ttopk_logprobs' header; refusing to guess the column order"
            )
        fields = [part.strip() for part in raw.split("\t")]
        if len(fields) != len(columns):
            raise KlError(
                f"{path}:{line}: expected {len(columns)} tab-separated fields, got {len(fields)}"
            )
        row = dict(zip(columns, fields))
        index = _coerce_int(row["index"], path, line, "index")
        token = _coerce_int(row["token"], path, line, "token")
        logprob = _coerce_float(row["logprob"], path, line, "logprob")
        log_normalizer = _coerce_float(row["log_normalizer"], path, line, "log_normalizer")
        ids = tuple(
            _coerce_int(item, path, line, "topk_ids")
            for item in _parse_list(row["topk_ids"], path, line, "topk_ids")
        )
        logprobs = tuple(
            _coerce_float(item, path, line, "topk_logprobs")
            for item in _parse_list(row["topk_logprobs"], path, line, "topk_logprobs")
        )
        if len(ids) != len(logprobs):
            raise KlError(
                f"{path}:{line}: {len(ids)} candidate ids but {len(logprobs)} candidate logprobs"
            )
        if len(set(ids)) != len(ids):
            raise KlError(f"{path}:{line}: the candidate list repeats a token")
        if any(later > earlier for earlier, later in zip(logprobs, logprobs[1:])):
            raise KlError(f"{path}:{line}: candidate logprobs are not in decreasing order")
        if logprobs and abs(logprobs[0] - logprob) > 1.0e-6:
            raise KlError(
                f"{path}:{line}: the top candidate's logprob {logprobs[0]!r} is not the row's "
                f"logprob {logprob!r}"
            )
        if k is None:
            k = len(ids)
        elif len(ids) != k:
            raise KlError(
                f"{path}:{line}: candidate count {len(ids)} differs from the earlier rows' {k}; "
                "the two files must be produced with the same --token-topk"
            )
        positions.append(
            Position(
                index=index,
                token=token,
                logprob=logprob,
                log_normalizer=log_normalizer,
                ids=ids,
                logprobs=logprobs,
            )
        )
    if not positions:
        raise KlError(f"{path}: contains no positions")
    return positions


def position_kl(reference: Position, candidate: Position) -> KlTerms:
    """One position's KL(p||q) over {shared candidates, one bucket per side}.

    The bucket carries each side's mass outside the shared candidates - the candidates the other
    model does not rank plus the residual the export does not cover - so both sides sum to 1 and the
    value cannot be negative, while two identical runs still give exactly 0.
    """
    coverage_p = reference.coverage()
    coverage_q = candidate.coverage()
    if coverage_p <= 0.0 or coverage_q <= 0.0:
        raise KlError(f"position {reference.index}: a candidate list carries no probability mass")
    if coverage_p > 1.0 + RESIDUAL_SUM_TOLERANCE or coverage_q > 1.0 + RESIDUAL_SUM_TOLERANCE:
        raise KlError(
            f"position {reference.index}: candidate probabilities sum above 1 "
            f"(coverage p={coverage_p:.6f}, q={coverage_q:.6f})"
        )

    candidate_logprob = dict(zip(candidate.ids, candidate.logprobs))
    reference_ids = set(reference.ids)

    kl = 0.0
    shared_p_terms: list[float] = []
    tail_tokens = 0
    for token, logprob_p in zip(reference.ids, reference.logprobs):
        other = candidate_logprob.get(token)
        if other is None:
            tail_tokens += 1
            continue
        probability_p = math.exp(logprob_p)
        shared_p_terms.append(probability_p)
        kl += probability_p * (logprob_p - other)

    # Both buckets are the complement of the shared candidates, so each side sums to exactly 1 over
    # {shared candidates, bucket}: p's unranked candidates and residual on one side, q's on the
    # other. This is where q's residual mass is charged, under p's own weight. The two complements
    # are summed the same way (fsum over the shared mass) so two identical exports give a bucket of
    # exactly 0 on both sides and a KL of exactly 0.
    shared_p = math.fsum(shared_p_terms)
    bucket_p = max(0.0, 1.0 - shared_p)
    bucket_q = max(
        0.0,
        1.0 - math.fsum(
            math.exp(logprob) for token, logprob in zip(candidate.ids, candidate.logprobs)
            if token in reference_ids
        ),
    )
    if bucket_p > 0.0:
        kl += bucket_p * (math.log(bucket_p) - math.log(max(bucket_q, RESIDUAL_FLOOR)))
    return KlTerms(
        kl=kl,
        coverage_p=coverage_p,
        coverage_q=coverage_q,
        tail_tokens=tail_tokens,
        tail_mass=bucket_p,
    )
    return KlTerms(
        kl=kl,
        coverage_p=coverage_p,
        coverage_q=coverage_q,
        tail_tokens=tail_tokens,
        tail_mass=tail_mass,
    )
    return KlTerms(
        kl=kl,
        coverage_p=coverage_p,
        coverage_q=coverage_q,
        tail_tokens=tail_tokens,
        tail_mass=tail_mass,
    )


def _segment_label(low: int, high: int | None) -> str:
    return f"{low}-{high - 1}" if high is not None else f"{low}+"


def _boundaries(segments: Sequence[int]) -> list[tuple[int, int | None]]:
    ordered = sorted(set(segments))
    if any(value <= 0 for value in ordered) or len(ordered) != len(segments):
        raise KlError("segment boundaries must be distinct positive integers")
    bounds: list[tuple[int, int | None]] = []
    low = 0
    for high in ordered:
        bounds.append((low, high))
        low = high
    bounds.append((low, None))
    return bounds


def compare(path_p: Path, path_q: Path, segments: Sequence[int]) -> dict[str, object]:
    """Align two exports by index and return the per-segment report payload."""
    positions_p = read_topk(path_p)
    positions_q = read_topk(path_q)
    if len(positions_p) != len(positions_q):
        raise KlError(
            f"position count differs: {path_p} has {len(positions_p)}, "
            f"{path_q} has {len(positions_q)}; both runs must score the same text"
        )
    for position_p, position_q in zip(positions_p, positions_q):
        if position_p.index != position_q.index:
            raise KlError(
                f"position mismatch: index {position_p.index} vs {position_q.index}"
            )
        if position_p.token != position_q.token:
            raise KlError(
                f"position {position_p.index}: token {position_p.token} vs {position_q.token}; "
                "the two runs did not score the same text"
            )

    bounds = _boundaries(segments)
    buckets: list[tuple[str, int | None, int | None, list[KlTerms]]] = [("all", None, None, [])]
    for low, high in bounds:
        buckets.append((_segment_label(low, high), low, high, []))

    rows: list[dict[str, object]] = []
    for label, low, high, members in buckets:
        for reference, candidate in zip(positions_p, positions_q):
            index = reference.index
            if low is not None and index < low:
                continue
            if high is not None and index >= high:
                continue
            members.append(position_kl(reference, candidate))
        positions = len(members)
        kl_sum = math.fsum(term.kl for term in members)
        rows.append(
            {
                "segment": label,
                "low": low,
                "high": high,
                "positions": positions,
                "kl_sum": kl_sum,
                "kl_mean": kl_sum / positions if positions else None,
                "coverage_p": (
                    math.fsum(term.coverage_p for term in members) / positions
                    if positions
                    else None
                ),
                "coverage_q": (
                    math.fsum(term.coverage_q for term in members) / positions
                    if positions
                    else None
                ),
                "tail_positions": sum(1 for term in members if term.tail_tokens),
                "tail_tokens": sum(term.tail_tokens for term in members),
                "tail_mass": math.fsum(term.tail_mass for term in members),
            }
        )
    return {
        "p": {"path": str(path_p), "candidates": positions_p[0].k},
        "q": {"path": str(path_q), "candidates": positions_q[0].k},
        "segments": rows,
        "note": (
            "KL(p||q) in nats per position over the candidates both lists share plus one bucket "
            "per side (each side's unranked mass, which is where q's residual is charged); read "
            "it with coverage_p, coverage_q and tail_mass"
        ),
    }


def _number(value: object, width: int, digits: int = 6) -> str:
    if value is None:
        return format("n/a", f">{width}")
    return format(float(value), f">{width}.{digits}f")


def _print_table(report: dict[str, object]) -> None:
    p = report["p"]
    q = report["q"]
    assert isinstance(p, dict) and isinstance(q, dict)
    print(f"p: {p['path']} ({p['candidates']} candidates)")
    print(f"q: {q['path']} ({q['candidates']} candidates)")
    print(
        f"{'segment':>10} {'positions':>10} {'KL sum':>12} {'KL/pos':>12} "
        f"{'cov p':>9} {'cov q':>9} {'tail tok':>9} {'tail mass':>10}"
    )
    rows = report["segments"]
    assert isinstance(rows, list)
    for row in rows:
        assert isinstance(row, dict)
        print(
            f"{row['segment']:>10} {row['positions']:>10} {_number(row['kl_sum'], 12)} "
            f"{_number(row['kl_mean'], 12, 9)} {_number(row['coverage_p'], 9)} "
            f"{_number(row['coverage_q'], 9)} {row['tail_tokens']:>9} "
            f"{_number(row['tail_mass'], 10, 3)}"
        )
    print(str(report["note"]))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="KL(p||q) between two ninfer-perplexity --token-topk exports.",
        epilog="See tools/perplexity/README.md for the record formats.",
    )
    parser.add_argument("p", type=Path, help="reference export (p)")
    parser.add_argument("q", type=Path, help="candidate export (q)")
    parser.add_argument(
        "--segments",
        default=",".join(str(value) for value in DEFAULT_SEGMENTS),
        metavar="N[,N...]",
        help=(
            "segment boundaries by index "
            "(default 1024,2048,4096 -> 0-1023,1024-2047,2048-4095,4096+)"
        ),
    )
    parser.add_argument(
        "--json",
        nargs="?",
        const="-",
        default=None,
        metavar="PATH",
        help="write the report as JSON to PATH ('-' or no value: stdout)",
    )
    args = parser.parse_args(argv)

    try:
        segments = [int(item) for item in args.segments.split(",") if item.strip()]
        if not segments:
            raise KlError("--segments needs at least one boundary")
        report = compare(args.p, args.q, segments)
    except (KlError, OSError) as error:
        parser.exit(1, f"kl_div: {error}\n")
        return 1

    if args.json is None:
        _print_table(report)
        return 0
    text = json.dumps(report, indent=2, allow_nan=False)
    if args.json == "-":
        print(text)
    else:
        Path(args.json).write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
