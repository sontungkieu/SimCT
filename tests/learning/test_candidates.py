from __future__ import annotations

import unittest

from vdt_span.candidates import enumerate_span_candidates
from vdt_span.types import AlignedSpan


ATOMS = (
    AlignedSpan(0, 2, 0, 1, 0, 2),
    AlignedSpan(2, 3, 1, 3, 2, 5),
    AlignedSpan(3, 4, 3, 4, 5, 6),
)


class CandidateEnumerationTests(unittest.TestCase):
    def test_enumerates_all_feasible_consecutive_unions(self) -> None:
        candidates = enumerate_span_candidates(
            ATOMS,
            max_teacher_tokens=3,
            max_student_tokens=3,
            max_bytes=5,
        )
        self.assertEqual(
            [(c.atom_start, c.atom_end) for c in candidates],
            [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3)],
        )

    def test_candidate_ranges_are_exact_unions(self) -> None:
        candidates = enumerate_span_candidates(
            ATOMS,
            max_teacher_tokens=4,
            max_student_tokens=4,
            max_bytes=6,
        )
        whole = candidates[2]
        self.assertEqual((whole.atom_start, whole.atom_end), (0, 3))
        self.assertEqual(
            (
                whole.teacher_start,
                whole.teacher_end,
                whole.student_start,
                whole.student_end,
                whole.byte_start,
                whole.byte_end,
            ),
            (0, 4, 0, 4, 0, 6),
        )

    def test_limits_are_inclusive_and_order_is_deterministic(self) -> None:
        candidates = enumerate_span_candidates(
            ATOMS,
            max_teacher_tokens=2,
            max_student_tokens=2,
            max_bytes=3,
        )
        self.assertEqual(
            [(c.atom_start, c.atom_end) for c in candidates],
            [(0, 1), (1, 2), (2, 3)],
        )

    def test_noncontiguous_atoms_are_rejected(self) -> None:
        broken = (ATOMS[0], AlignedSpan(3, 4, 1, 2, 2, 3))
        with self.assertRaises(ValueError):
            enumerate_span_candidates(
                broken,
                max_teacher_tokens=4,
                max_student_tokens=4,
                max_bytes=8,
            )


if __name__ == "__main__":
    unittest.main()
