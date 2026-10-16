import math
import unittest

from h2_word_metrics import aggregate, compare_distributions, distribution, entropy, entropy_match, js, kendall_tau_b


class WordMetricsTests(unittest.TestCase):
    def test_other_validation(self):
        self.assertEqual(distribution([0.2, 0.3]), [0.2, 0.3, 0.5])
        for values in ([-1e-12, 0.1], [math.nan], [math.inf], [0.6, 0.400002], []):
            with self.assertRaises(ValueError):
                distribution(values)
        p = distribution([0.6, 0.4000005])
        self.assertAlmostEqual(sum(p), 1)
        self.assertEqual(p[-1], 0)

    def test_zero_and_self_js(self):
        p = [1, 0, 0]
        self.assertEqual(js(p, p), 0)
        self.assertAlmostEqual(js(p, [0, 1, 0]), math.log(2))
        compared = compare_distributions([0.1, 0.3], [0.1, 0.3], [0.1, 0.3])
        self.assertEqual(compared["X_raw"], 0)
        self.assertEqual(compared["X_shape"], 0)
        self.assertTrue(compared["top_cells"]["S"]["other_dominant"])

    def test_tie_and_zero_support_infeasibility(self):
        tied = entropy_match([0.5, 0.5, 0], 0.1)
        self.assertFalse(tied["feasible"])
        self.assertEqual(tied["reason"], "below_tied_maxima_entropy")
        zero_support = entropy_match([1, 0, 0], 0.5)
        self.assertFalse(zero_support["feasible"])
        self.assertEqual(zero_support["reason"], "above_positive_support_entropy")
        self.assertIsNone(compare_distributions([0.3, 0.3], [1, 0], [0.2, 0.3])["X_shape"])

    def test_entropy_match_and_endpoints(self):
        original = [0.7, 0.2, 0.1, 0]
        for target in (0.2, 0.8, 1.0):
            match = entropy_match(original, target)
            self.assertTrue(match["feasible"])
            self.assertAlmostEqual(entropy(match["distribution"]), target, places=8)
            self.assertEqual(match["distribution"][-1], 0)
        maximum = entropy_match(original, math.log(3))
        self.assertEqual(maximum["limit"], "uniform_support")
        minimum = entropy_match(original, 0)
        self.assertEqual(minimum["distribution"], [1, 0, 0, 0])

    def test_tau_b_ties(self):
        self.assertEqual(kendall_tau_b([1, 2, 2], [1, 2, 2]), 1)
        self.assertEqual(kendall_tau_b([1, 2, 2], [2, 1, 1]), -1)
        self.assertIsNone(kendall_tau_b([1, 1], [1, 2]))
        self.assertAlmostEqual(kendall_tau_b([1, 2, 2], [1, 2, 3]), 2 / math.sqrt(6))

    def test_prompt_cluster_mean(self):
        rows = [{"prompt_id": "long", "category": "a", "sampling_weight": 1, "X_raw": 1, "X_shape": None} for _ in range(100)]
        rows += [{"prompt_id": "short", "category": "b", "sampling_weight": 1, "X_raw": -1, "X_shape": -0.5}]
        result = aggregate(rows, "X_raw")
        self.assertEqual(result["mean"], 0)
        self.assertEqual(result["valid_prompt_count"], 2)
        self.assertEqual(result["categories"]["a"]["mean"], 1)
        self.assertEqual(result["missing_flags"]["X_shape"], 100)
        self.assertEqual(result, aggregate(rows, "X_raw"))
        shape = aggregate(rows, "X_shape")
        self.assertEqual(shape["mean"], -0.5)
        self.assertEqual(shape["valid_position_count"], 1)

    def test_weighted_mean_and_missing(self):
        rows = [{"prompt_id": "p", "sampling_weight": 3, "X_raw": 1},
                {"prompt_id": "p", "sampling_weight": 1, "X_raw": -1},
                {"prompt_id": "q", "sampling_weight": 1, "X_raw": None}]
        result = aggregate(rows, "X_raw", repeats=0)
        self.assertEqual(result["mean"], 0.5)
        self.assertEqual(result["valid_prompt_count"], 1)
        self.assertEqual(result["missing_position_count"], 1)
        self.assertIsNone(result["ci95"])
        self.assertIsNone(aggregate([], "X_raw")["mean"])


if __name__ == "__main__":
    unittest.main()
