#!/usr/bin/env python3
"""Deterministic public grader for the CPU-only dynamic-span learning lab."""

from __future__ import annotations

import argparse
import io
import sys
import unittest
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@dataclass(frozen=True)
class TestSpec:
    group: str
    name: str
    points: int


RUBRIC = (
    TestSpec("alignment", "tests.learning.test_alignment.ExactByteAlignmentTests.test_shared_boundaries_form_atomic_spans", 5),
    TestSpec("alignment", "tests.learning.test_alignment.ExactByteAlignmentTests.test_incomplete_utf8_pieces_are_aligned_as_bytes", 5),
    TestSpec("alignment", "tests.learning.test_alignment.ExactByteAlignmentTests.test_mismatched_decoded_bytes_are_rejected", 4),
    TestSpec("alignment", "tests.learning.test_alignment.ExactByteAlignmentTests.test_empty_token_piece_is_rejected", 4),
    TestSpec("candidates", "tests.learning.test_candidates.CandidateEnumerationTests.test_enumerates_all_feasible_consecutive_unions", 4),
    TestSpec("candidates", "tests.learning.test_candidates.CandidateEnumerationTests.test_candidate_ranges_are_exact_unions", 4),
    TestSpec("candidates", "tests.learning.test_candidates.CandidateEnumerationTests.test_limits_are_inclusive_and_order_is_deterministic", 4),
    TestSpec("candidates", "tests.learning.test_candidates.CandidateEnumerationTests.test_noncontiguous_atoms_are_rejected", 4),
    TestSpec("viterbi", "tests.learning.test_viterbi.SemiMarkovViterbiTests.test_finds_best_complete_segmentation", 5),
    TestSpec("viterbi", "tests.learning.test_viterbi.SemiMarkovViterbiTests.test_tie_prefers_fewer_spans", 5),
    TestSpec("viterbi", "tests.learning.test_viterbi.SemiMarkovViterbiTests.test_remaining_tie_is_lexicographic", 5),
    TestSpec("viterbi", "tests.learning.test_viterbi.SemiMarkovViterbiTests.test_empty_sequence_and_unreachable_sequence", 5),
    TestSpec("continuation", "tests.learning.test_continuation.ContinuationScoringTests.test_sums_conditional_logprobabilities", 4),
    TestSpec("continuation", "tests.learning.test_continuation.ContinuationScoringTests.test_prefix_grows_after_each_continuation_token", 4),
    TestSpec("continuation", "tests.learning.test_continuation.ContinuationScoringTests.test_empty_continuation_has_logprob_zero", 3),
    TestSpec("continuation", "tests.learning.test_continuation.ContinuationScoringTests.test_invalid_logprob_is_rejected", 3),
    TestSpec("coarsening", "tests.learning.test_coarsening.MassPreservingCoarseningTests.test_collisions_are_aggregated", 4),
    TestSpec("coarsening", "tests.learning.test_coarsening.MassPreservingCoarseningTests.test_unmapped_tail_mass_is_preserved", 4),
    TestSpec("coarsening", "tests.learning.test_coarsening.MassPreservingCoarseningTests.test_no_renormalization_when_every_event_is_mapped", 4),
    TestSpec("coarsening", "tests.learning.test_coarsening.MassPreservingCoarseningTests.test_invalid_distribution_is_rejected", 4),
    TestSpec("policy", "tests.learning.test_policy.AdaptiveSpanPolicyTests.test_exact_auditable_policy_fixture", 4),
    TestSpec("policy", "tests.learning.test_policy.AdaptiveSpanPolicyTests.test_training_progress_can_expand_span_budget", 4),
    TestSpec("policy", "tests.learning.test_policy.AdaptiveSpanPolicyTests.test_context_risk_and_disagreement_shrink_budget", 4),
    TestSpec("policy", "tests.learning.test_policy.AdaptiveSpanPolicyTests.test_invalid_state_or_context_is_rejected", 4),
)

SMOKE_SUITE = "tests.learning.test_smoke"


def run_suite(name: str) -> tuple[bool, str]:
    stream = io.StringIO()
    suite = unittest.defaultTestLoader.loadTestsFromName(name)
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    return result.wasSuccessful(), stream.getvalue()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="run environment/import smoke tests only")
    parser.add_argument("--exercise", choices=sorted({spec.group for spec in RUBRIC}), help="grade one exercise group")
    parser.add_argument("--verbose", action="store_true", help="print unittest failure details")
    args = parser.parse_args()

    smoke_ok, smoke_output = run_suite(SMOKE_SUITE)
    print(f"SMOKE: {'PASS' if smoke_ok else 'FAIL'}")
    if args.verbose or not smoke_ok:
        print(smoke_output.rstrip())
    if args.smoke:
        return 0 if smoke_ok else 2
    if not smoke_ok:
        print("Environment/import smoke checks failed; exercise score was not computed.")
        return 2

    selected = [spec for spec in RUBRIC if args.exercise in (None, spec.group)]
    earned = 0
    possible = sum(spec.points for spec in selected)
    group_totals: dict[str, list[int]] = {}
    details: list[str] = []

    for spec in selected:
        passed, output = run_suite(spec.name)
        if passed:
            earned += spec.points
        scores = group_totals.setdefault(spec.group, [0, 0])
        scores[1] += spec.points
        if passed:
            scores[0] += spec.points
        short_name = spec.name.rsplit(".", 1)[-1]
        print(f"[{spec.group:12}] {short_name:62} {'PASS' if passed else 'FAIL'}  +{spec.points if passed else 0}/{spec.points}")
        if args.verbose and not passed:
            details.append(f"\n--- {spec.name} ---\n{output.rstrip()}")

    print("\nGROUPS")
    for group, (group_earned, group_possible) in group_totals.items():
        print(f"  {group:12}: {group_earned:3}/{group_possible:3}")
    print(f"TOTAL: {earned}/{possible}")

    if details:
        print("\nFAILURE DETAILS")
        print("\n".join(details))
    return 0 if earned == possible else 1


if __name__ == "__main__":
    raise SystemExit(main())
