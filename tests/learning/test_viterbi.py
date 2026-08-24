from __future__ import annotations

import unittest

from vdt_span.types import ScoredSpan, ViterbiPath
from vdt_span.viterbi import semi_markov_viterbi


class SemiMarkovViterbiTests(unittest.TestCase):
    def test_finds_best_complete_segmentation(self) -> None:
        candidates = [
            ScoredSpan(0, 1, 1.0),
            ScoredSpan(1, 2, 1.0),
            ScoredSpan(2, 3, 1.0),
            ScoredSpan(3, 4, 1.0),
            ScoredSpan(0, 2, 3.0),
            ScoredSpan(2, 4, 3.0),
        ]
        self.assertEqual(
            semi_markov_viterbi(4, candidates),
            ViterbiPath(score=6.0, spans=((0, 2), (2, 4))),
        )

    def test_tie_prefers_fewer_spans(self) -> None:
        candidates = [
            ScoredSpan(0, 1, 2.0),
            ScoredSpan(1, 2, 2.0),
            ScoredSpan(0, 2, 4.0),
        ]
        self.assertEqual(
            semi_markov_viterbi(2, candidates).spans,
            ((0, 2),),
        )

    def test_remaining_tie_is_lexicographic(self) -> None:
        candidates = [
            ScoredSpan(0, 1, 1.0),
            ScoredSpan(1, 4, 2.0),
            ScoredSpan(0, 2, 1.0),
            ScoredSpan(2, 4, 2.0),
        ]
        self.assertEqual(
            semi_markov_viterbi(4, candidates).spans,
            ((0, 1), (1, 4)),
        )

    def test_empty_sequence_and_unreachable_sequence(self) -> None:
        self.assertEqual(semi_markov_viterbi(0, []), ViterbiPath(0.0, ()))
        with self.assertRaises(ValueError):
            semi_markov_viterbi(3, [ScoredSpan(0, 1, 1.0)])


if __name__ == "__main__":
    unittest.main()
