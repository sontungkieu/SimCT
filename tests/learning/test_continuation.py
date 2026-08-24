from __future__ import annotations

import math
import unittest

from vdt_span.continuation import continuation_logprob


class ContinuationScoringTests(unittest.TestCase):
    def test_sums_conditional_logprobabilities(self) -> None:
        table = {
            ((7,), 8): -0.1,
            ((7, 8), 9): -0.2,
            ((7, 8, 9), 10): -0.3,
        }
        score = continuation_logprob((7,), (8, 9, 10), lambda p, t: table[(p, t)])
        self.assertAlmostEqual(score, -0.6)

    def test_prefix_grows_after_each_continuation_token(self) -> None:
        seen: list[tuple[tuple[int, ...], int]] = []

        def scorer(prefix: tuple[int, ...], token: int) -> float:
            seen.append((prefix, token))
            return -1.0

        continuation_logprob((1, 2), (3, 4), scorer)
        self.assertEqual(seen, [((1, 2), 3), ((1, 2, 3), 4)])

    def test_empty_continuation_has_logprob_zero(self) -> None:
        calls = 0

        def scorer(prefix: tuple[int, ...], token: int) -> float:
            nonlocal calls
            calls += 1
            return -1.0

        self.assertEqual(continuation_logprob((1,), (), scorer), 0.0)
        self.assertEqual(calls, 0)

    def test_invalid_logprob_is_rejected(self) -> None:
        for invalid in (0.01, math.nan, math.inf, -math.inf):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    continuation_logprob((), (1,), lambda _p, _t, x=invalid: x)


if __name__ == "__main__":
    unittest.main()
