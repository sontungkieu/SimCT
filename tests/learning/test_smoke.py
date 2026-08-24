"""Smoke checks that must pass before and after solving the TODOs."""

from __future__ import annotations

import sys
import unittest

import vdt_span
from vdt_span.types import AlignedSpan, SpanPolicyConfig


class LearningLabSmokeTests(unittest.TestCase):
    def test_supported_python(self) -> None:
        self.assertGreaterEqual(sys.version_info, (3, 10))

    def test_all_exercises_are_exported(self) -> None:
        self.assertEqual(len(vdt_span.TODO_EXERCISES), 6)
        for name in (
            "adaptive_max_span_width",
            "align_exact_byte_boundaries",
            "coarsen_distribution",
            "continuation_logprob",
            "enumerate_span_candidates",
            "semi_markov_viterbi",
        ):
            self.assertTrue(callable(getattr(vdt_span, name)))

    def test_scaffolding_data_types(self) -> None:
        atom = AlignedSpan(0, 1, 0, 2, 0, 3)
        self.assertEqual((atom.teacher_width, atom.student_width, atom.byte_width), (1, 2, 3))
        self.assertEqual(SpanPolicyConfig().min_width, 1)


if __name__ == "__main__":
    unittest.main()
