from __future__ import annotations

import pytest

from vdt_span.scoring import (
    reverse_kl_from_candidate_scores,
    reverse_kl_student_score_gradient,
)


def test_python_reverse_kl_gradient_matches_finite_differences():
    student = [0.2, -0.3, 1.1]
    teacher = [-0.4, 0.9, 0.1]
    temperature = 0.7
    analytic = reverse_kl_student_score_gradient(
        student,
        teacher,
        temperature=temperature,
    )
    epsilon = 1e-6
    finite_difference = []
    for index in range(len(student)):
        left = list(student)
        right = list(student)
        left[index] -= epsilon
        right[index] += epsilon
        finite_difference.append(
            (
                reverse_kl_from_candidate_scores(
                    right,
                    teacher,
                    temperature=temperature,
                )
                - reverse_kl_from_candidate_scores(
                    left,
                    teacher,
                    temperature=temperature,
                )
            )
            / (2.0 * epsilon)
        )
    assert analytic == pytest.approx(finite_difference, abs=1e-8)
