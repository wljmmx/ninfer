#!/usr/bin/env python3
"""Compare two per-token logprob records and report only the quantities they support.

Input is the record stream produced by ``ninfer-perplexity --token-logprobs`` (apps/perplexity),
which writes one row per scored token:

    # index<TAB>token<TAB>logprob
    1<TAB>1234<TAB>-2.91732822

``index`` is the token position inside its stream (plan_windows targets every token after x[0], so a
stream of N tokens yields positions 1..N-1), ``token`` is the observed corpus token id, and
``logprob`` is ``log p(x[index] | x[0..index))`` under the evaluated artifact.

That format carries the observed token only. It has no distribution, no alternative candidates and no
model argmax, so:

  * KL(P||Q) is NOT computable from the plain TSV. It needs p and q over the same support. This tool
    prints ``n/a`` there instead of guessing.
  * top-1 disagreement is NOT computable from the plain TSV, for the same reason: the recorded token
    is the corpus token, not either model's argmax.
  * The mean logprob difference IS computable and is exact for what it measures: with
    mean_delta = mean(log p(x) - log q(x)) over scored tokens, mean_delta = CE_Q - CE_P in nats per
    token. It equals KL(P||Q) only if the scored tokens were sampled from P, which a natural corpus
    does not guarantee. Read it as a cross-entropy difference, not as a KL divergence.

A JSONL record may also carry ``topk`` (``[{"token": 7, "logprob": -0.31}, ...]`` or
``[[7, -0.31], ...]``). When every paired record in a bucket has it on both sides, this tool reports
KL(P||Q) restricted to the union of the two candidate sets, with the remaining probability mass
lumped into one residual bucket (exact only if P and Q distribute their residual identically) and
top-1 disagreement over the two argmaxes. The report labels those numbers with the restriction.

Every input problem raises instead of producing a number, and names the file, line and field.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

LN2 = math.log(2.0)
TSV_COLUMNS = ("index", "token", "logprob")
RESIDUAL_SUM_TOLERANCE = 1.0e-3

LIMITATIONS = (
    "target logprob only: KL(P||Q) and top-1 disagreement need per-position candidate lists; "
    "plain --token-logprobs rows provide neither",
    "mean logprob difference is CE_Q - CE_P (nats/token), not KL(P||Q)",
    "KL from topk is restricted to the union of the candidate sets, with residual mass lumped",
    "position indexes are per stream; a multi-stream corpus pools equal positions across streams",
)


class CompareError(Exception):
    """Input problem that must stop the comparison instead of producing a number."""


@dataclass(frozen=True)
class Record:
    index: int
    token: int
    logprob: float
    topk: tuple[tuple[int, float], ...] | None
    line: int


@dataclass
class Bucket:
    label: str
    low: int | None
    high: int | None
    tokens: int = 0
    sum_logprob_p: float = 0.0
    sum_logprob_q: float = 0.0
    kl_sum: float = 0.0
    kl_tokens: int = 0
    kl_support_miss: int = 0
    top1_tokens: int = 0
    top1_diff: int = 0

    def add(self, record_p: Record, record_q: Record) -> None:
        """Accumulate one aligned position pair into this bucket."""
        self.tokens += 1
        self.sum_logprob_p += record_p.logprob
        self.sum_logprob_q += record_q.logprob
        if record_p.topk is None or record_q.topk is None:
            return
        kl, missed = _topk_kl(record_p, record_q)
        self.kl_tokens += 1
        self.kl_support_miss += int(missed)
        if not missed:
            self.kl_sum += kl
        self.top1_tokens += 1
        if record_p.topk[0][0] != record_q.topk[0][0]:
            self.top1_diff += 1


@dataclass
class Report:
    path_p: Path
    path_q: Path
    format_p: str
    format_q: str
    has_topk_p: bool
    has_topk_q: bool
    buckets: list[Bucket] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _coerce_int(value: object, path: Path, line: int, name: str) -> int:
    if isinstance(value, bool):
        raise CompareError(f"{path}:{line}: {name} is a boolean, expected an integer")
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            pass
    raise CompareError(f"{path}:{line}: {name} is not an integer: {value!r}")


def _coerce_logprob(value: object, path: Path, line: int, name: str) -> float:
    if isinstance(value, bool):
        raise CompareError(f"{path}:{line}: {name} is a boolean, expected a number")
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        raise CompareError(f"{path}:{line}: {name} is not a number: {value!r}") from None
    if not math.isfinite(number):
        raise CompareError(f"{path}:{line}: {name} is not finite: {value!r}")
    if number > 1.0e-6:
        raise CompareError(f"{path}:{line}: {name} must be <= 0 for a log probability: {number!r}")
    return number


def _parse_topk(value: object, path: Path, line: int) -> tuple[tuple[int, float], ...] | None:
    if value is None:
        return None
    if not isinstance(value, list) or not value:
        raise CompareError(f"{path}:{line}: topk must be a non-empty array")
    entries: list[tuple[int, float]] = []
    seen: set[int] = set()
    for position, item in enumerate(value):
        if isinstance(item, dict):
            raw_token = item.get("token")
            raw_logprob = item.get("logprob")
            if raw_token is None or raw_logprob is None:
                raise CompareError(f"{path}:{line}: topk[{position}] needs token and logprob")
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            raw_token, raw_logprob = item[0], item[1]
        else:
            raise CompareError(
                f"{path}:{line}: topk[{position}] must be {{token, logprob}} or [token, logprob]"
            )
        token = _coerce_int(raw_token, path, line, f"topk[{position}].token")
        logprob = _coerce_logprob(raw_logprob, path, line, f"topk[{position}].logprob")
        if token in seen:
            raise CompareError(f"{path}:{line}: topk repeats token {token}")
        if entries and logprob >= entries[-1][1]:
            raise CompareError(
                f"{path}:{line}: topk must be ordered by decreasing logprob (entry {position} is "
                f"{logprob!r} after {entries[-1][1]!r}); top-1 would be ambiguous"
            )
        seen.add(token)
        entries.append((token, logprob))
    return tuple(entries)


def _topk_kl(record_p: Record, record_q: Record) -> tuple[float, bool]:
    """Return (KL, support_missed) over the union of two candidate lists."""
    assert record_p.topk is not None and record_q.topk is not None
    support_p = {token: math.exp(logprob) for token, logprob in record_p.topk}
    support_q = {token: math.exp(logprob) for token, logprob in record_q.topk}
    p_residual = 1.0 - math.fsum(support_p.values())
    q_residual = 1.0 - math.fsum(support_q.values())
    if p_residual < -RESIDUAL_SUM_TOLERANCE or q_residual < -RESIDUAL_SUM_TOLERANCE:
        raise CompareError(
            f"lines {record_p.line}/{record_q.line}: topk probabilities sum above 1 "
            f"(1 - sum p = {p_residual:.6f}, 1 - sum q = {q_residual:.6f})"
        )
    p_residual = max(0.0, p_residual)
    q_residual = max(0.0, q_residual)

    missing = False
    total = 0.0
    for token, probability in support_p.items():
        if probability <= 0.0:
            continue
        other = support_q.get(token, 0.0)
        if other <= 0.0:
            missing = True
            continue
        total += probability * math.log(probability / other)
    if p_residual > 0.0:
        if q_residual <= 0.0:
            missing = True
        else:
            total += p_residual * math.log(p_residual / q_residual)
    return total, missing


def _read_records(path: Path) -> tuple[list[Record], bool]:
    """Return (records, jsonl) for one input file; ``jsonl`` selects the record parser."""
    if not path.is_file():
        raise CompareError(f"{path}: no such file")
    with path.open("r", encoding="utf-8", errors="strict") as handle:
        lines = handle.read().splitlines()

    first_data: int | None = None
    for position, raw in enumerate(lines):
        text = raw.strip()
        if text and not text.startswith("#"):
            first_data = position
            break
    if first_data is None:
        raise CompareError(f"{path}: contains no records")
    if lines[first_data].strip().startswith("{"):
        return _read_jsonl(lines, path, first_data)
    return _read_tsv(lines, path, first_data)


def _read_tsv(lines: Sequence[str], path: Path, first_data: int) -> tuple[list[Record], bool]:
    columns: tuple[str, ...] | None = None
    for raw in lines[:first_data]:
        text = raw.strip()
        if not text.startswith("#"):
            continue
        names = tuple(name.strip() for name in text.lstrip("#").split("\t") if name.strip())
        if all(name in names for name in TSV_COLUMNS):
            columns = names
            break
    if columns is None:
        raise CompareError(
            f"{path}: missing the '# index<TAB>token<TAB>logprob' header written by "
            "ninfer-perplexity; refusing to guess the column order"
        )

    records: list[Record] = []
    for offset in range(first_data, len(lines)):
        raw = lines[offset]
        line_no = offset + 1
        text = raw.strip()
        if not text or text.startswith("#"):
            continue
        fields = [part.strip() for part in raw.split("\t")]
        if len(fields) != len(columns):
            raise CompareError(
                f"{path}:{line_no}: expected {len(columns)} tab-separated fields, got {len(fields)}"
            )
        row = dict(zip(columns, fields))
        records.append(
            Record(
                index=_coerce_int(row["index"], path, line_no, "index"),
                token=_coerce_int(row["token"], path, line_no, "token"),
                logprob=_coerce_logprob(row["logprob"], path, line_no, "logprob"),
                topk=None,
                line=line_no,
            )
        )
    return records, False


def _read_jsonl(lines: Sequence[str], path: Path, first_data: int) -> tuple[list[Record], bool]:
    records: list[Record] = []
    for offset in range(first_data, len(lines)):
        raw = lines[offset]
        line_no = offset + 1
        text = raw.strip()
        if not text or text.startswith("#"):
            continue
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as error:
            raise CompareError(f"{path}:{line_no}: invalid JSON ({error.msg})") from None
        if not isinstance(payload, dict):
            raise CompareError(f"{path}:{line_no}: JSONL record must be an object")
        for name in ("index", "token", "logprob"):
            if name not in payload:
                raise CompareError(f"{path}:{line_no}: record lacks '{name}'")
        records.append(
            Record(
                index=_coerce_int(payload["index"], path, line_no, "index"),
                token=_coerce_int(payload["token"], path, line_no, "token"),
                logprob=_coerce_logprob(payload["logprob"], path, line_no, "logprob"),
                topk=_parse_topk(payload.get("topk"), path, line_no),
                line=line_no,
            )
        )
    return records, True


def _align(
    records_p: Sequence[Record],
    records_q: Sequence[Record],
    path_p: Path,
    path_q: Path,
    allow_token_mismatch: bool,
) -> None:
    if len(records_p) != len(records_q):
        detail = ""
        for position, (near, far) in enumerate(zip(records_p, records_q)):
            if near.index != far.index:
                detail = (
                    f"; first position mismatch at record {position + 1}: "
                    f"{near.index} ({path_p}:{near.line}) vs {far.index} ({path_q}:{far.line})"
                )
                break
        if not detail:
            extra = records_p if len(records_p) > len(records_q) else records_q
            extra_path = path_p if len(records_p) > len(records_q) else path_q
            shared = min(len(records_p), len(records_q))
            detail = (
                f"; the extra records start at position {shared + 1} "
                f"({extra_path}:{extra[shared].line})"
            )
        raise CompareError(
            f"record count differs: {path_p} has {len(records_p)}, "
            f"{path_q} has {len(records_q)}{detail}"
        )
    for position, (near, far) in enumerate(zip(records_p, records_q)):
        if near.index != far.index:
            raise CompareError(
                f"position mismatch at record {position + 1}: index {near.index} "
                f"({path_p}:{near.line}) vs {far.index} ({path_q}:{far.line})"
            )
        if not allow_token_mismatch and near.token != far.token:
            raise CompareError(
                f"token mismatch at record {position + 1} (index {near.index}): token {near.token} "
                f"({path_p}:{near.line}) vs {far.token} ({path_q}:{far.line}); the two files did not "
                "score the same corpus text"
            )


def _parse_buckets(text: str) -> tuple[int, ...]:
    boundaries: list[int] = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            value = int(item)
        except ValueError:
            raise argparse.ArgumentTypeError(f"bucket boundary is not an integer: {item}") from None
        if value <= 0:
            raise argparse.ArgumentTypeError(f"bucket boundary must be positive: {value}")
        boundaries.append(value)
    if not boundaries:
        raise argparse.ArgumentTypeError("at least one bucket boundary is required")
    if boundaries != sorted(boundaries) or len(set(boundaries)) != len(boundaries):
        raise argparse.ArgumentTypeError("bucket boundaries must be strictly increasing")
    return tuple(boundaries)


def _buckets_from(boundaries: Sequence[int]) -> list[Bucket]:
    buckets = [Bucket("all", None, None)]
    low = 0
    for high in boundaries:
        buckets.append(Bucket(f"{low}-{high - 1}", low, high))
        low = high
    buckets.append(Bucket(f"{low}+", low, None))
    return buckets


def _bucket_of(buckets: Sequence[Bucket], index: int) -> Bucket:
    for bucket in buckets[1:]:
        if bucket.low is None or index < bucket.low:
            continue
        if bucket.high is None or index < bucket.high:
            return bucket
    return buckets[-1]


def compare(
    path_p: Path,
    path_q: Path,
    boundaries: Sequence[int],
    allow_token_mismatch: bool,
) -> Report:
    records_p, jsonl_p = _read_records(path_p)
    records_q, jsonl_q = _read_records(path_q)
    _align(records_p, records_q, path_p, path_q, allow_token_mismatch)

    buckets = _buckets_from(boundaries)
    for record_p, record_q in zip(records_p, records_q):
        buckets[0].add(record_p, record_q)
        _bucket_of(buckets, record_p.index).add(record_p, record_q)

    has_topk_p = all(record.topk is not None for record in records_p)
    has_topk_q = all(record.topk is not None for record in records_q)
    report = Report(
        path_p=path_p,
        path_q=path_q,
        format_p="jsonl" if jsonl_p else "tsv",
        format_q="jsonl" if jsonl_q else "tsv",
        has_topk_p=has_topk_p,
        has_topk_q=has_topk_q,
        buckets=buckets,
    )
    if has_topk_p != has_topk_q:
        report.warnings.append(
            "only one side carries topk on every record; KL and top-1 disagreement stay n/a"
        )
    elif not has_topk_p:
        report.warnings.append(
            "target-only logprobs: KL(P||Q) and top-1 disagreement are not computable"
        )

    indexes = [record.index for record in records_p]
    if indexes != sorted(indexes):
        report.warnings.append(
            "indexes are not globally increasing (multi-stream input); bucket counts pool equal "
            "positions across streams"
        )
    return report


def _bucket_json(bucket: Bucket) -> dict[str, object]:
    mean_p = bucket.sum_logprob_p / bucket.tokens if bucket.tokens else None
    mean_q = bucket.sum_logprob_q / bucket.tokens if bucket.tokens else None
    mean_delta = (
        (bucket.sum_logprob_p - bucket.sum_logprob_q) / bucket.tokens if bucket.tokens else None
    )
    kl: float | None = None
    kl_status = "n/a: input has no per-position candidate lists"
    if bucket.kl_tokens:
        if bucket.kl_support_miss:
            kl_status = (
                f"n/a: {bucket.kl_support_miss}/{bucket.kl_tokens} positions have P mass outside "
                "Q's candidate list"
            )
        else:
            kl = bucket.kl_sum / bucket.kl_tokens
            kl_status = "topk-restricted, residual lumped"
            if bucket.kl_tokens < bucket.tokens:
                kl_status += f" ({bucket.kl_tokens}/{bucket.tokens} positions)"
    top1_rate: float | None = None
    if bucket.top1_tokens:
        top1_rate = bucket.top1_diff / bucket.top1_tokens
    return {
        "bucket": bucket.label,
        "low": bucket.low,
        "high": bucket.high,
        "tokens": bucket.tokens,
        "mean_logprob_delta": mean_delta,
        "delta_bits_per_token": None if mean_delta is None else mean_delta / LN2,
        "mean_logprob_p": mean_p,
        "mean_logprob_q": mean_q,
        "ppl_p": None if mean_p is None else math.exp(-mean_p),
        "ppl_q": None if mean_q is None else math.exp(-mean_q),
        "ppl_ratio_p_over_q": None
        if mean_p is None or mean_q is None
        else math.exp(mean_q - mean_p),
        "kl_p_q": kl,
        "kl_status": kl_status,
        "kl_tokens": bucket.kl_tokens,
        "top1_disagreement": top1_rate,
        "top1_tokens": bucket.top1_tokens,
    }


def _format_number(value: object, digits: int = 6) -> str:
    if value is None:
        return "n/a"
    number = float(value)  # type: ignore[arg-type]
    if not math.isfinite(number):
        return "inf" if number > 0 else "-inf"
    return f"{number:.{digits}f}"


def _source_label(prefix: str, path: Path, fmt: str, complete: bool, kl_positions: int) -> str:
    if complete:
        detail = "topk on every record"
    elif kl_positions:
        detail = f"partial topk ({kl_positions} positions)"
    else:
        detail = "target-only"
    return f"{prefix}: {path} ({fmt}, {detail})"


def _print_table(report: Report) -> None:
    totals = _bucket_json(report.buckets[0])
    kl_positions = report.buckets[0].kl_tokens
    print(_source_label("P", report.path_p, report.format_p, report.has_topk_p, kl_positions))
    print(_source_label("Q", report.path_q, report.format_q, report.has_topk_q, kl_positions))
    print(
        f"{'bucket':>12} {'tokens':>9} {'dlogp':>12} {'dbits':>10} {'PPL_P':>11} "
        f"{'PPL_Q':>11} {'P/Q':>9} {'KL(P||Q)':>12} {'top1diff':>9}"
    )
    for bucket in report.buckets:
        row = _bucket_json(bucket)
        print(
            f"{bucket.label:>12} {bucket.tokens:>9} "
            f"{_format_number(row['mean_logprob_delta']):>12} "
            f"{_format_number(row['delta_bits_per_token']):>10} "
            f"{_format_number(row['ppl_p'], 4):>11} {_format_number(row['ppl_q'], 4):>11} "
            f"{_format_number(row['ppl_ratio_p_over_q'], 5):>9} "
            f"{_format_number(row['kl_p_q']):>12} {_format_number(row['top1_disagreement'], 4):>9}"
        )
    print(f"KL status: {totals['kl_status']}")
    print("dlogp = mean(log p(x) - log q(x)) = CE_Q - CE_P; positive means P is less surprised.")
    print("This is not KL(P||Q) unless the scored tokens were sampled from P.")


def _write_json(report: Report, destination: str) -> None:
    payload = {
        "p": {
            "path": str(report.path_p),
            "format": report.format_p,
            "topk_on_every_record": report.has_topk_p,
        },
        "q": {
            "path": str(report.path_q),
            "format": report.format_q,
            "topk_on_every_record": report.has_topk_q,
        },
        "totals": _bucket_json(report.buckets[0]),
        "buckets": [_bucket_json(bucket) for bucket in report.buckets[1:]],
        "limitations": list(LIMITATIONS),
        "warnings": report.warnings,
    }
    text = json.dumps(payload, indent=2, allow_nan=False)
    if destination == "-":
        print(text)
        return
    Path(destination).write_text(text + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare two per-token logprob record files.",
        epilog="See tools/perplexity/README.md for the supported fields and known limits.",
    )
    parser.add_argument("p", type=Path, help="first record file (P)")
    parser.add_argument("q", type=Path, help="second record file (Q)")
    parser.add_argument(
        "--buckets",
        type=_parse_buckets,
        default=(1024, 2048, 4096),
        metavar="N[,N...]",
        help="position boundaries (default 1024,2048,4096 -> 0-1023, 1024-2047, 2048-4095, 4096+)",
    )
    parser.add_argument(
        "--json",
        nargs="?",
        const="-",
        default=None,
        metavar="PATH",
        help="write JSON to PATH instead of the table ('-' or no value: stdout)",
    )
    parser.add_argument(
        "--allow-token-mismatch",
        action="store_true",
        help="do not fail when the two files recorded different corpus tokens at one position",
    )
    args = parser.parse_args(argv)

    try:
        report = compare(args.p, args.q, args.buckets, args.allow_token_mismatch)
    except (CompareError, OSError) as error:
        parser.exit(1, f"kl_compare: {error}\n")

    for warning in report.warnings:
        print(f"note: {warning}", file=sys.stderr)
    if args.json is None:
        _print_table(report)
    else:
        _write_json(report, args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
