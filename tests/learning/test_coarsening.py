from __future__ import annotations

import math
import unittest

from vdt_span.coarsening import coarsen_distribution


class MassPreservingCoarseningTests(unittest.TestCase):
    def test_collisions_are_aggregated(self) -> None:
        coarse = coarsen_distribution(
            {"a": 0.4, "b": 0.3, "c": 0.2, "tail": 0.1},
            {"a": "AB", "b": "AB", "c": "C"},
        )
        self.assertEqual(set(coarse), {"AB", "C", "__OTHER__"})
        self.assertAlmostEqual(coarse["AB"], 0.7)
        self.assertAlmostEqual(coarse["C"], 0.2)

    def test_unmapped_tail_mass_is_preserved(self) -> None:
        coarse = coarsen_distribution(
            {"known": 0.6, "tail-1": 0.25, "tail-2": 0.15},
            {"known": "KNOWN"},
            residual_bucket="TAIL",
        )
        self.assertAlmostEqual(coarse["KNOWN"], 0.6)
        self.assertAlmostEqual(coarse["TAIL"], 0.4)
        self.assertAlmostEqual(math.fsum(coarse.values()), 1.0)

    def test_no_renormalization_when_every_event_is_mapped(self) -> None:
        coarse = coarsen_distribution(
            {"x": 0.1, "y": 0.2, "z": 0.7},
            {"x": 0, "y": 0, "z": 1},
        )
        self.assertEqual(set(coarse), {0, 1})
        self.assertAlmostEqual(coarse[0], 0.3)
        self.assertAlmostEqual(coarse[1], 0.7)

    def test_invalid_distribution_is_rejected(self) -> None:
        invalid_cases = (
            {"a": -0.1, "b": 1.1},
            {"a": 0.2, "b": 0.2},
            {"a": math.nan, "b": math.nan},
        )
        for distribution in invalid_cases:
            with self.subTest(distribution=distribution):
                with self.assertRaises(ValueError):
                    coarsen_distribution(distribution, {})


if __name__ == "__main__":
    unittest.main()
