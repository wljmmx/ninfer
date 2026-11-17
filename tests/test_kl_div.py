"""Hand-computable checks for the KL estimator in tools/perplexity/kl_div.py.

The estimator is the only place a number is produced from two candidate lists, so it is graded on
distributions whose KL is known in closed form, including the two ways the candidate lists truncate.
"""

from __future__ import annotations

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.perplexity.kl_div import (  # noqa: E402
    KlError,
    Position,
    compare,
    position_kl,
)


def position(index: int, distribution: dict[int, float], normalizer: float = 0.0) -> Position:
    """Build a position whose candidate list is ordered by decreasing probability."""
    ordered = sorted(distribution.items(), key=lambda item: (-item[1], item[0]))
    return Position(
        index=index,
        token=ordered[0][0],
        logprob=math.log(ordered[0][1]),
        log_normalizer=normalizer,
        ids=tuple(token for token, _ in ordered),
        logprobs=tuple(math.log(probability) for _, probability in ordered),
    )


class IdenticalRunsTest(unittest.TestCase):
    def test_identical_distributions_are_exactly_zero(self) -> None:
        p = position(1, {5: 0.5, 7: 0.3, 11: 0.15, 13: 0.05})
        terms = position_kl(p, p)
        self.assertEqual(terms.kl, 0.0)
        self.assertAlmostEqual(terms.coverage_p, 1.0, places=12)
        self.assertEqual(terms.tail_tokens, 0)

    def test_identical_truncated_lists_keep_zero(self) -> None:
        # Identical runs export identical candidate lists, so nothing is charged to the tail even
        # when the list does not cover the distribution.
        distribution = {1: 0.4, 2: 0.3}
        p = position(2, distribution)
        self.assertAlmostEqual(p.coverage(), 0.7, places=12)
        self.assertEqual(position_kl(p, position(2, distribution)).kl, 0.0)


class KnownKlTest(unittest.TestCase):
    def test_uniform_support_against_peaked_reference(self) -> None:
        # KL(p||q) with p = (0.5,0.25,0.125,0.125) and q uniform over the same four tokens:
        #   0.5*ln2 + 0.25*ln1 + 2*(0.125*ln0.5) = 0.346574 - 0.173287 = 0.173287 nats.
        p = position(10, {0: 0.5, 1: 0.25, 2: 0.125, 3: 0.125})
        q = position(10, {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25})
        expected = 0.5 * math.log(2.0) + 2.0 * 0.125 * math.log(0.5)
        terms = position_kl(p, q)
        self.assertAlmostEqual(terms.kl, expected, delta=1e-6)
        self.assertAlmostEqual(terms.coverage_p, 1.0, places=12)
        self.assertAlmostEqual(terms.coverage_q, 1.0, places=12)
        self.assertEqual(terms.tail_tokens, 0)

    def test_kl_is_not_symmetric(self) -> None:
        p = position(11, {0: 0.9, 1: 0.1})
        q = position(11, {0: 0.5, 1: 0.5})
        forward = position_kl(p, q).kl
        backward = position_kl(q, p).kl
        self.assertAlmostEqual(forward, 0.9 * math.log(1.8) + 0.1 * math.log(0.2), delta=1e-9)
        self.assertGreater(backward, forward)

    def test_reference_token_missing_from_the_candidate_list_charges_its_tail(self) -> None:
        # p = (0.6 over 0, 0.3 over 2) and q's list holds (0.5 over 0, 0.3 over 1). The shared
        # candidate is token 0, so p's bucket is 1 - 0.6 = 0.4 (its unranked 0.3 plus its residual
        # 0.1) and q's bucket is 1 - 0.5 = 0.5 (its unranked 0.3 plus its residual 0.2):
        #   0.6*ln(0.6/0.5) + 0.4*ln(0.4/0.5) = 0.020135 nats.
        p = position(12, {0: 0.6, 2: 0.3})
        q = position(12, {0: 0.5, 1: 0.3})
        terms = position_kl(p, q)
        expected = 0.6 * math.log(0.6 / 0.5) + 0.4 * math.log(0.4 / 0.5)
        self.assertAlmostEqual(terms.kl, expected, delta=1e-6)
        self.assertAlmostEqual(terms.kl, 0.0201350, delta=1e-6)
        self.assertEqual(terms.tail_tokens, 1)
        self.assertAlmostEqual(terms.tail_mass, 0.4, places=12)
        self.assertAlmostEqual(terms.coverage_p, 0.9, places=12)
        self.assertAlmostEqual(terms.coverage_q, 0.8, places=12)

    def test_every_unshared_candidate_lands_in_the_same_bucket(self) -> None:
        # Two candidates absent from q's list must NOT each carry q's whole residual, because that
        # counts the same q mass twice. They join p's residual in ONE bucket per side:
        #   p = (0.4 over 0, 0.3 over 2, 0.2 over 3), q = (0.5 over 0, 0.4 over 1)
        #   bucket p = 1 - 0.4 = 0.6, bucket q = 1 - 0.5 = 0.5
        #   0.4*ln(0.4/0.5) + 0.6*ln(0.6/0.5) = 0.020135 nats.
        p = position(13, {0: 0.4, 2: 0.3, 3: 0.2})
        q = position(13, {0: 0.5, 1: 0.4})
        terms = position_kl(p, q)
        expected = 0.4 * math.log(0.4 / 0.5) + 0.6 * math.log(0.6 / 0.5)
        self.assertAlmostEqual(terms.kl, expected, delta=1e-6)
        self.assertEqual(terms.tail_tokens, 2)
        self.assertAlmostEqual(terms.tail_mass, 0.6, places=12)
        self.assertGreaterEqual(terms.kl, 0.0)

    def test_equal_coverage_cannot_produce_a_negative_kl(self) -> None:
        # p and q cover the same mass but rank different tokens. Both sides still sum to 1 over
        # {shared candidate, bucket}, so the value cannot be negative - the property the raw
        # per-token charging rule broke on real data.
        p = position(14, {0: 0.4, 2: 0.25, 3: 0.25})
        q = position(14, {0: 0.55, 1: 0.35})
        terms = position_kl(p, q)
        expected = 0.4 * math.log(0.4 / 0.55) + 0.6 * math.log(0.6 / 0.45)
        self.assertAlmostEqual(p.coverage(), q.coverage(), places=12)
        self.assertAlmostEqual(terms.kl, expected, delta=1e-6)
        self.assertGreaterEqual(terms.kl, 0.0)

    def test_different_candidate_counts_are_allowed(self) -> None:
        # Two exports may be produced with different --token-topk; only the shared candidates are
        # summed, and each side's bucket is the complement of that shared set:
        #   0.6*ln(0.6/0.5) + 0.3*ln(1) + 0.1*ln(0.1/0.2) = 0.040078 nats.
        p = position(15, {0: 0.6, 1: 0.3})
        q = position(15, {0: 0.5, 1: 0.3, 2: 0.15})
        terms = position_kl(p, q)
        expected = (
            0.6 * math.log(0.6 / 0.5)
            + 0.3 * math.log(0.3 / 0.3)
            + 0.1 * math.log(0.1 / 0.2)
        )
        self.assertAlmostEqual(terms.kl, expected, delta=1e-6)
        self.assertEqual(terms.tail_tokens, 0)

    def test_mass_outside_a_fully_covered_candidate_list_is_reported_not_dropped(self) -> None:
        # q's list holds all of its mass, so p's mass outside the shared candidate has no q
        # probability at all and the true KL is infinite. The bucket floors q there instead of
        # dropping the term, which keeps the number finite and unmistakably large, not quietly
        # small.
        p = position(16, {0: 0.5, 1: 0.3, 2: 0.2})
        q = position(16, {0: 0.6, 1: 0.4})
        terms = position_kl(p, q)
        self.assertEqual(terms.tail_tokens, 1)
        self.assertGreater(terms.kl, 10.0)


class FileTest(unittest.TestCase):
    def write(self, directory: Path, name: str, rows: list[str]) -> Path:
        path = directory / name
        path.write_text(
            "# index\ttoken\tlogprob\tlog_normalizer\ttopk_ids\ttopk_logprobs\n"
            + "".join(row + "\n" for row in rows),
            encoding="utf-8",
        )
        return path

    def test_segments_split_by_index_and_agree_with_the_core(self) -> None:
        rows_p: list[str] = []
        rows_q: list[str] = []
        for index in range(1, 6):
            rows_p.append(
                f"{index}\t0\t{math.log(0.7):.17g}\t0.0\t0,1\t"
                f"{math.log(0.7):.17g},{math.log(0.3):.17g}"
            )
            rows_q.append(
                f"{index}\t0\t{math.log(0.4):.17g}\t0.0\t0,2\t"
                f"{math.log(0.4):.17g},{math.log(0.35):.17g}"
            )
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            report = compare(
                self.write(directory, "p.tsv", rows_p),
                self.write(directory, "q.tsv", rows_q),
                [3],
            )
            json.dumps(report, allow_nan=False)
        segments = {row["segment"]: row for row in report["segments"]}
        self.assertEqual(segments["0-2"]["positions"], 2)
        self.assertEqual(segments["3+"]["positions"], 3)
        self.assertEqual(segments["all"]["positions"], 5)
        # Token 1 is absent from q's list, so p's bucket is 1 - 0.7 = 0.3 (that candidate) and q's
        # is 1 - 0.4 = 0.6 (its candidate 2 at 0.35 plus its residual 0.25).
        expected = 0.7 * math.log(0.7 / 0.4) + 0.3 * math.log(0.3 / 0.6)
        self.assertAlmostEqual(segments["all"]["kl_mean"], expected, delta=1e-6)
        self.assertEqual(segments["all"]["tail_tokens"], 5)

    def test_mismatched_candidate_counts_are_refused(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            p = self.write(directory, "p.tsv", ["1\t0\t-0.1\t0.0\t0,1\t-0.1,-2.3"])
            q = self.write(directory, "q.tsv", ["1\t0\t-0.1\t0.0\t0\t-0.1"])
            with self.assertRaises(KlError):
                compare(p, q, [1024])

    def test_order_violation_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            bad = self.write(directory, "p.tsv", ["1\t0\t-0.1\t0.0\t0,1\t-2.3,-0.1"])
            good = self.write(directory, "q.tsv", ["1\t0\t-0.1\t0.0\t0,1\t-0.1,-2.3"])
            with self.assertRaises(KlError):
                compare(bad, good, [1024])

    def test_top_candidate_must_be_the_rows_logprob(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            bad = self.write(directory, "p.tsv", ["1\t0\t-0.7\t0.0\t0,1\t-0.1,-2.3"])
            good = self.write(directory, "q.tsv", ["1\t0\t-0.1\t0.0\t0,1\t-0.1,-2.3"])
            with self.assertRaises(KlError):
                compare(bad, good, [1024])


if __name__ == "__main__":
    unittest.main()
