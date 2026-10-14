import math
import unittest
from h2_measurements import (prefix_map, strict_pairs, byte_spans, js,
                             bootstrap_prompt_mean, next_word_event, assert_prefix_free)


class H2MeasurementsTest(unittest.TestCase):
    def test_prefix_projection_longest_duplicate_residual_and_eos(self):
        vocab = [b'a', b'ab', b'ab', None]
        mapping, size = prefix_map(vocab, [b'abc', b'ax', b'z', None], {3}, {3})
        self.assertEqual(mapping, [1, 0, 5, 4]); self.assertEqual(size, 6)
        mass = [0.] * size
        for i, p in zip(mapping, [.2, .3, .1, .4]): mass[i] += p
        self.assertAlmostEqual(sum(mass), 1)
        self.assertAlmostEqual(js(mass, mass), 0)

    def test_split_unicode_bytes_have_exact_offsets(self):
        raw = 'é'.encode(); vocab = [raw[:1], raw[1:], raw]
        student = byte_spans([0, 1], vocab, 'é'); teacher = byte_spans([2], vocab, 'é')
        self.assertEqual(strict_pairs(student, teacher), {})
        self.assertEqual(strict_pairs(teacher, teacher), {0: 0})
        with self.assertRaises(ValueError): byte_spans([0], vocab, 'é')

    def test_bootstrap_does_not_weight_long_response_as_extra_prompts(self):
        rows = [{'prompt_id': 'a', 'x': 0}] * 100 + [{'prompt_id': 'b', 'x': 1}]
        result = bootstrap_prompt_mean(rows, 'x', repeats=100)
        self.assertEqual(result['mean'], .5); self.assertEqual(result['prompts'], 2)

    def test_candidate_boundaries_and_prefix_overlap(self):
        self.assertEqual(next_word_event('  hello world'), '  hello ')
        self.assertEqual(next_word_event('123+456'), '123+')
        self.assertIsNone(next_word_event('123'))
        assert_prefix_free(['1+', '12+'])
        with self.assertRaises(ValueError): assert_prefix_free(['1', '12'])

    def test_js_zero_support_is_finite_and_bounded(self):
        self.assertAlmostEqual(js([1, 0], [0, 1]), math.log(2))
        with self.assertRaises(ValueError): js([.9], [.9])


if __name__ == '__main__': unittest.main()
