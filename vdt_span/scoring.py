"""Reference scoring utilities for cross-tokenizer span experiments."""

from __future__ import annotations

import math
from typing import Iterable, Sequence


def _validate_probability_vector(values: Sequence[float], name: str) -> None:
    if not values:
        raise ValueError(f"{name} must not be empty")
    if any(value < 0.0 or not math.isfinite(value) for value in values):
        raise ValueError(f"{name} must contain finite non-negative values")
    if not math.isclose(sum(values), 1.0, rel_tol=0.0, abs_tol=1e-8):
        raise ValueError(f"{name} must sum to one")


def continuation_score(log_probabilities: Sequence[float], *, length_normalized: bool = True) -> float:
    """Score one realized continuation from its causal token log-probabilities."""

    if not log_probabilities:
        raise ValueError("a continuation must contain at least one token")
    if any(not math.isfinite(value) for value in log_probabilities):
        raise ValueError("log-probabilities must be finite")
    total = float(sum(log_probabilities))
    return total / len(log_probabilities) if length_normalized else total


def normalized_candidate_log_probs(scores: Sequence[float]) -> tuple[float, ...]:
    """Apply log-softmax to finite candidate scores.

    The result is a candidate-normalized operational distribution, not a claim
    that the candidate set captures all generative probability mass.
    """

    if not scores:
        raise ValueError("scores must not be empty")
    if any(not math.isfinite(score) for score in scores):
        raise ValueError("scores must be finite")
    maximum = max(scores)
    log_z = maximum + math.log(sum(math.exp(score - maximum) for score in scores))
    return tuple(float(score - log_z) for score in scores)


def reverse_kl_from_candidate_scores(
    student_scores: Sequence[float],
    teacher_scores: Sequence[float],
    *,
    temperature: float = 1.0,
) -> float:
    """Reference ``KL(q_student || q_teacher)`` on SimCT candidates.

    ``scores`` are the paper's mean token log-probability scores before the
    finite-candidate softmax.  This is an operational candidate distribution,
    not a mass-preserving projection of either model vocabulary.
    """

    if len(student_scores) != len(teacher_scores):
        raise ValueError("student and teacher scores must have equal length")
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be positive and finite")
    student_log_probs = normalized_candidate_log_probs(
        [float(value) / temperature for value in student_scores]
    )
    teacher_log_probs = normalized_candidate_log_probs(
        [float(value) / temperature for value in teacher_scores]
    )
    return float(
        sum(
            math.exp(student_log_prob)
            * (student_log_prob - teacher_log_prob)
            for student_log_prob, teacher_log_prob in zip(
                student_log_probs,
                teacher_log_probs,
                strict=True,
            )
        )
    )


def reverse_kl_student_score_gradient(
    student_scores: Sequence[float],
    teacher_scores: Sequence[float],
    *,
    temperature: float = 1.0,
) -> tuple[float, ...]:
    """Analytic paper-reference gradient with respect to student scores."""

    loss = reverse_kl_from_candidate_scores(
        student_scores,
        teacher_scores,
        temperature=temperature,
    )
    student_log_probs = normalized_candidate_log_probs(
        [float(value) / temperature for value in student_scores]
    )
    teacher_log_probs = normalized_candidate_log_probs(
        [float(value) / temperature for value in teacher_scores]
    )
    return tuple(
        math.exp(student_log_prob)
        * (student_log_prob - teacher_log_prob - loss)
        / temperature
        for student_log_prob, teacher_log_prob in zip(
            student_log_probs,
            teacher_log_probs,
            strict=True,
        )
    )


def kl_divergence(p: Sequence[float], q: Sequence[float]) -> float:
    """Compute forward ``KL(p || q)`` for discrete probability vectors."""

    if len(p) != len(q):
        raise ValueError("p and q must have equal length")
    _validate_probability_vector(p, "p")
    _validate_probability_vector(q, "q")
    result = 0.0
    for p_value, q_value in zip(p, q):
        if p_value == 0.0:
            continue
        if q_value == 0.0:
            return math.inf
        result += p_value * math.log(p_value / q_value)
    return result


def coarsen_distribution(
    probabilities: Sequence[float],
    groups: Iterable[Sequence[int]],
) -> tuple[float, ...]:
    """Mass-preservingly aggregate atomic events into a strict partition."""

    _validate_probability_vector(probabilities, "probabilities")
    materialized = tuple(tuple(group) for group in groups)
    flattened = [index for group in materialized for index in group]
    if any(not group for group in materialized):
        raise ValueError("coarsening groups must be non-empty")
    if sorted(flattened) != list(range(len(probabilities))):
        raise ValueError("groups must partition every event index exactly once")
    return tuple(sum(probabilities[index] for index in group) for group in materialized)


def kl_information_gap(
    p: Sequence[float],
    q: Sequence[float],
    groups: Iterable[Sequence[int]],
) -> float:
    """Return fine KL minus mass-preserving coarse KL."""

    materialized = tuple(tuple(group) for group in groups)
    fine = kl_divergence(p, q)
    coarse = kl_divergence(
        coarsen_distribution(p, materialized),
        coarsen_distribution(q, materialized),
    )
    gap = fine - coarse
    if gap < -1e-10:
        raise ArithmeticError("coarse KL exceeded fine KL beyond numerical tolerance")
    return max(0.0, gap)
