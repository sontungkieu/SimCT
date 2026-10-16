import unittest

from h2_tokenizer_audit import aggregate_audits, audit_pair


def spans(lengths, ids=None):
    result, offset = [], 0
    for index, length in enumerate(lengths):
        result.append((offset, offset + length, ids[index] if ids else index))
        offset += length
    return result


class TokenizerAuditTests(unittest.TestCase):
    def test_strict_and_id_independence(self):
        result = audit_pair(spans([1, 2], [9, 10]), spans([1, 2], [20, 21]), "abc")
        self.assertEqual(result["coverage"]["strict_student_token_fraction"], 1)
        self.assertEqual(result["counts"]["strict_pairs_same_numeric_id"], 0)
        self.assertEqual(result["boundaries"]["jaccard"], 1)
        self.assertEqual(result["student_token_trace"][1]["category"], "word_internal")
        other = audit_pair(spans([1, 2], [1, 2]), spans([3], [1]), "abc")
        self.assertEqual(other["counts"]["overlap_pairs_same_numeric_id_different_bytes"], 1)
        self.assertEqual(other["coverage"]["strict_byte_fraction"], 0)

    def test_merge_split_and_nm(self):
        merged = audit_pair(spans([1, 1, 2]), spans([2, 1, 1]), "abcd")
        self.assertEqual(merged["interval_classes"]["N:1"]["intervals"], 1)
        self.assertEqual(merged["interval_classes"]["1:N"]["intervals"], 1)
        self.assertEqual(merged["coverage"]["strict_student_token_fraction"], 0)
        self.assertEqual(merged["boundaries"]["jaccard"], 1 / 3)
        mixed = audit_pair(spans([1, 3]), spans([2, 2]), "abcd")
        self.assertEqual(mixed["interval_classes"]["N:M"]["intervals"], 1)
        self.assertEqual(mixed["student_token_trace"][1]["teacher_count"], 2)
        self.assertFalse(mixed["student_token_trace"][1]["context_aligned"])
        self.assertEqual(mixed["student_token_trace"][1]["minimal_coarse_interval"], {"start": 0, "end": 4, "class": "N:M"})

    def test_unicode_newlines_digits_latex_whitespace(self):
        text = "é\n123\\x "
        result = audit_pair(spans([1, 1, 1, 1, 2, 2, 1]), spans([2, 1, 3, 2, 1]), text)
        self.assertEqual(result["source"]["utf8_bytes"], 9)
        self.assertEqual(result["source"]["unicode_characters"], 8)
        self.assertEqual(result["source"]["non_ascii_characters"], 1)
        self.assertFalse(result["student_token_trace"][0]["utf8_end_boundary"])
        self.assertEqual(result["student_token_trace"][2]["category"], "whitespace")
        self.assertTrue(result["student_token_trace"][2]["contains_newline_bytes"])
        self.assertEqual(result["numeric_runs"][0]["character_length"], 3)
        self.assertFalse(result["numeric_runs"][0]["exact_run_segmentation_match"])
        self.assertEqual(result["denominators"]["whitespace_source_bytes"], 2)
        self.assertEqual(result["student_token_trace"][-2]["category"], "latex")

    def test_zero_boundaries_and_empty_text(self):
        one = audit_pair(spans([3]), spans([3]), "abc")
        self.assertEqual(one["boundaries"]["jaccard"], 1)
        self.assertEqual(one["counts"]["shared_interior_boundaries"], 0)
        one_sided = audit_pair(spans([3]), spans([1, 2]), "abc")
        self.assertEqual(one_sided["boundaries"]["recall_teacher_against_student"], 0)
        empty = audit_pair([], [], "")
        self.assertIsNone(empty["coverage"]["strict_byte_fraction"])
        self.assertEqual(empty["boundaries"]["jaccard"], 1)

    def test_split_unicode_whitespace_denominators(self):
        result = audit_pair(spans([1, 2, 1]), spans([3, 1]), "\u2003x")
        self.assertEqual(result["denominators"]["whitespace_source_bytes"], 3)
        self.assertEqual(result["denominators"]["nonwhitespace_student_tokens"], 1)
        self.assertEqual(result["denominators"]["nonwhitespace_teacher_tokens"], 1)
        self.assertEqual(result["excluding_whitespace"]["strict_student_token_fraction"], 1)
        self.assertEqual(result["excluding_whitespace"]["strict_teacher_token_fraction"], 1)
        self.assertEqual(result["excluding_whitespace"]["strict_byte_fraction"], 1)

    def test_validation(self):
        for malformed in ([(1, 3, 0)], [(0, 0, 0), (0, 3, 1)], [(0, 2, 0)]):
            with self.assertRaises(ValueError):
                audit_pair(malformed, spans([3]), "abc")
        with self.assertRaises(ValueError):
            audit_pair([{"start": 0, "end": 3, "token_id": 1, "special": True}], spans([3]), "abc")

    def test_micro_macro_and_quantiles(self):
        yes = audit_pair(spans([1, 1, 1, 1]), spans([1, 1, 1, 1]), "abcd")
        no = audit_pair(spans([2]), spans([1, 1]), "ab")
        rows = [{"prompt_id": "a", "rollout_id": 0, "audit": yes},
                {"prompt_id": "a", "rollout_id": 1, "audit": yes},
                {"prompt_id": "b", "rollout_id": 0, "audit": no}]
        result = aggregate_audits(rows)
        key = "strict_student_token_fraction"
        self.assertEqual(result["global_micro"]["coverage"][key], 8 / 9)
        self.assertEqual(result["global_macro_rollout_coverage"][key], 2 / 3)
        self.assertEqual(result["global_macro_prompt_coverage"][key], 0.5)
        self.assertEqual(result["rollout_strict_coverage_quantiles"][key]["p10"], 0.2)
        self.assertEqual(result["prompt_strict_coverage_quantiles"][key]["p50"], 0.5)
        with self.assertRaises(ValueError):
            aggregate_audits([rows[0], rows[0]])


if __name__ == "__main__":
    unittest.main()
