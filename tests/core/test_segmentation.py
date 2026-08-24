import math

import pytest

from vdt_span.segmentation import (
    AdaptiveBoundaryPolicy,
    enumerate_candidate_spans,
    fixed_width_partition,
    semi_markov_log_partition,
    semi_markov_span_marginals,
    semi_markov_viterbi,
    validate_partition,
)


def test_enumerate_candidate_spans() -> None:
    assert enumerate_candidate_spans(3, 2) == ((0, 1), (0, 2), (1, 2), (1, 3), (2, 3))


def test_fixed_width_partition() -> None:
    partition = fixed_width_partition(7, 3)
    assert partition == ((0, 3), (3, 6), (6, 7))
    validate_partition(partition, 7)


def test_viterbi_recovers_best_partition() -> None:
    scores = {
        (0, 1): 0.0,
        (1, 2): 0.0,
        (2, 3): 0.0,
        (0, 2): 2.0,
        (1, 3): -1.0,
    }
    result = semi_markov_viterbi(3, 2, lambda start, end: scores[(start, end)])
    assert result == ((0, 2), (2, 3))


def test_viterbi_tie_break_is_explicit() -> None:
    zero = lambda _start, _end: 0.0
    assert semi_markov_viterbi(3, 3, zero, tie_break="finer") == ((0, 1), (1, 2), (2, 3))
    assert semi_markov_viterbi(3, 3, zero, tie_break="coarser") == ((0, 3),)


def test_log_partition_and_marginals_are_consistent() -> None:
    zero = lambda _start, _end: 0.0
    log_z = semi_markov_log_partition(3, 2, zero)
    assert math.exp(log_z) == pytest.approx(3.0)
    marginals = semi_markov_span_marginals(3, 2, zero)
    for atomic_index in range(3):
        coverage = sum(
            probability
            for (start, end), probability in marginals.items()
            if start <= atomic_index < end
        )
        assert coverage == pytest.approx(1.0)


def test_adaptive_policy_can_change_with_training_progress() -> None:
    policy = AdaptiveBoundaryPolicy(
        disagreement_weight=0.0,
        instability_weight=0.0,
        association_weight=0.0,
        progress_weight=1.0,
        threshold=0.5,
        max_span=4,
    )
    features = dict(disagreement=[0.0, 0.0], stability=[1.0, 1.0], association=[0.0, 0.0])
    assert policy.partition(**features, progress=0.0) == ((0, 3),)
    assert policy.partition(**features, progress=1.0) == ((0, 1), (1, 2), (2, 3))


def test_adaptive_policy_can_change_with_context() -> None:
    policy = AdaptiveBoundaryPolicy(
        disagreement_weight=0.0,
        instability_weight=0.0,
        association_weight=0.0,
        context_weight=1.0,
        threshold=0.5,
        max_span=4,
    )
    shared = dict(
        disagreement=[0.0, 0.0],
        stability=[1.0, 1.0],
        association=[0.0, 0.0],
    )
    assert policy.partition(**shared, context=[0.0, 0.0]) == ((0, 3),)
    assert policy.partition(**shared, context=[1.0, 0.0]) == ((0, 1), (1, 3))


def test_adaptive_policy_respects_max_span() -> None:
    policy = AdaptiveBoundaryPolicy(threshold=10.0, max_span=2)
    result = policy.partition(
        disagreement=[0.0, 0.0, 0.0, 0.0],
        stability=[1.0, 1.0, 1.0, 1.0],
        association=[1.0, 1.0, 1.0, 1.0],
    )
    assert result == ((0, 2), (2, 4), (4, 5))
