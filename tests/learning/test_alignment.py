from __future__ import annotations

import unittest

from vdt_span.alignment import align_exact_byte_boundaries
from vdt_span.types import AlignedSpan


class ExactByteAlignmentTests(unittest.TestCase):
    def test_shared_boundaries_form_atomic_spans(self) -> None:
        actual = align_exact_byte_boundaries(
            [b"ab", b"c", b"def"],
            [b"a", b"bc", b"de", b"f"],
        )
        expected = (
            AlignedSpan(0, 2, 0, 2, 0, 3),
            AlignedSpan(2, 3, 2, 4, 3, 6),
        )
        self.assertEqual(actual, expected)

    def test_incomplete_utf8_pieces_are_aligned_as_bytes(self) -> None:
        actual = align_exact_byte_boundaries(
            [b"\xf0\x9f", b"\x98\x80"],
            [b"\xf0", b"\x9f\x98", b"\x80"],
        )
        self.assertEqual(actual, (AlignedSpan(0, 2, 0, 3, 0, 4),))

    def test_mismatched_decoded_bytes_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            align_exact_byte_boundaries([b"same", b"?"], [b"same", b"!"])

    def test_empty_token_piece_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            align_exact_byte_boundaries([b"a", b""], [b"a"])


if __name__ == "__main__":
    unittest.main()
