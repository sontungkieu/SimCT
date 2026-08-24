import math

import pytest

from vdt_span.scoring import (
    coarsen_distribution,
    continuation_score,
    kl_divergence,
    kl_information_gap,
    normalized_candidate_log_probs,
)


def test_continuation_score_separates_sum_from_length_normalization() -> None:
    logps = [math.log(0.5), math.log(0.25)]
    assert continuation_score(logps, length_normalized=False) == pytest.approx(math.log(0.125))
    assert continuation_score(logps) == pytest.approx(math.log(0.125) / 2)


def test_candidate_log_probs_normalize() -> None:
    log_probs = normalized_candidate_log_probs([0.0, math.log(3.0)])
    assert [math.exp(value) for value in log_probs] == pytest.approx([0.25, 0.75])


def test_mass_preserving_coarsening() -> None:
    coarse = coarsen_distribution([0.1, 0.2, 0.3, 0.4], [(0, 1), (2, 3)])
    assert coarse == pytest.approx((0.3, 0.7))
    assert sum(coarse) == pytest.approx(1.0)


def test_coarsening_cannot_increase_forward_kl() -> None:
    teacher = [0.45, 0.05, 0.10, 0.40]
    student = [0.10, 0.40, 0.20, 0.30]
    groups = [(0, 1), (2, 3)]
    fine = kl_divergence(teacher, student)
    gap = kl_information_gap(teacher, student, groups)
    coarse = kl_divergence(
        coarsen_distribution(teacher, groups),
        coarsen_distribution(student, groups),
    )
    assert gap >= 0.0
    assert fine == pytest.approx(coarse + gap)


def test_invalid_coarsening_fails_closed() -> None:
    with pytest.raises(ValueError, match="partition"):
        coarsen_distribution([0.5, 0.5], [(0,), (0,)])
