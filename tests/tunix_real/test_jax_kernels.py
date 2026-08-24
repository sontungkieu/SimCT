from __future__ import annotations

import math

import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from vdt_span.scoring import (
    continuation_score,
    normalized_candidate_log_probs,
    reverse_kl_from_candidate_scores,
    reverse_kl_student_score_gradient,
)
from vdt_tunix.jax_kernels import (
    candidate_log_probs,
    paper_simct_aligned_batch_loss,
    paper_candidate_scores,
    paper_simct_reverse_kl,
    paper_simple_opd_aligned_batch_loss,
    reverse_kl_loss_and_student_score_gradient,
)


def _python_log_softmax(row):
    maximum = max(row)
    log_z = maximum + math.log(sum(math.exp(value - maximum) for value in row))
    return [value - log_z for value in row]


def test_paper_candidate_scores_match_pure_python_reference():
    logits = [
        [[2.0, 0.0, -1.0], [0.0, 2.0, -1.0]],
        [[-1.0, 1.0, 0.0], [4.0, 1.0, -2.0]],
        [[0.0, 0.5, 1.0], [0.0, 0.0, 0.0]],
    ]
    token_ids = [[0, 1], [1, 0], [2, 0]]
    mask = [[1, 1], [1, 1], [1, 0]]
    expected_scores = []
    for rows, ids, active in zip(logits, token_ids, mask, strict=True):
        selected = [
            _python_log_softmax(row)[token_id]
            for row, token_id, keep in zip(rows, ids, active, strict=True)
            if keep
        ]
        expected_scores.append(continuation_score(selected))

    observed = paper_candidate_scores(
        jnp.asarray(logits),
        jnp.asarray(token_ids),
        jnp.asarray(mask),
    )
    assert list(observed) == pytest.approx(expected_scores, abs=1e-6)
    assert list(candidate_log_probs(observed)) == pytest.approx(
        normalized_candidate_log_probs(expected_scores),
        abs=1e-6,
    )


def test_reverse_kl_loss_and_student_gradient_match_python_reference():
    student = [0.2, -0.3, 1.1]
    teacher = [-0.4, 0.9, 0.1]
    temperature = 0.7
    loss, gradient = reverse_kl_loss_and_student_score_gradient(
        jnp.asarray(student),
        jnp.asarray(teacher),
        temperature=temperature,
    )
    assert float(loss) == pytest.approx(
        reverse_kl_from_candidate_scores(
            student,
            teacher,
            temperature=temperature,
        ),
        abs=1e-6,
    )
    assert list(gradient) == pytest.approx(
        reverse_kl_student_score_gradient(
            student,
            teacher,
            temperature=temperature,
        ),
        abs=1e-6,
    )


def test_end_to_end_kernel_stops_teacher_gradient():
    student_logits = jnp.asarray(
        [
            [[2.0, 0.0, -1.0], [0.0, 2.0, -1.0]],
            [[0.0, 1.0, -1.0], [1.0, 0.0, -1.0]],
        ]
    )
    teacher_logits = jnp.asarray(
        [
            [[0.0, 2.0, -1.0], [2.0, 0.0, -1.0]],
            [[1.0, 0.0, -1.0], [0.0, 1.0, -1.0]],
        ]
    )
    token_ids = jnp.asarray([[0, 1], [1, 0]])
    mask = jnp.ones_like(token_ids)

    def loss_fn(student_values, teacher_values):
        return paper_simct_reverse_kl(
            student_values,
            token_ids,
            mask,
            teacher_values,
            token_ids,
            mask,
        )

    student_gradient, teacher_gradient = jax.grad(
        loss_fn,
        argnums=(0, 1),
    )(student_logits, teacher_logits)
    assert bool(jnp.all(jnp.isfinite(student_gradient)))
    assert float(jnp.linalg.norm(student_gradient)) > 0.0
    assert float(jnp.linalg.norm(teacher_gradient)) == pytest.approx(0.0)


def test_aligned_batch_loss_matches_two_row_python_reference():
    student_logits = jnp.asarray(
        [[
            [2.0, 0.0, -1.0, 1.0],
            [0.0, 3.0, 1.0, -1.0],
            [1.0, -1.0, 2.0, 0.0],
        ]]
    )
    teacher_logits = jnp.asarray(
        [[
            [0.0, 2.0, -1.0, 1.0, 0.5],
            [2.0, 0.0, 1.0, -1.0, 0.2],
        ]]
    )
    student_ids = jnp.asarray([[0, 1, 2]])
    teacher_ids = jnp.asarray([[1, 0]])
    bounds = jnp.asarray([[[0, 1, 0, 2], [1, 2, 2, 3]]])
    unit_mask = jnp.asarray([[1.0, 1.0]])
    span_mask = jnp.asarray([[True, False]])

    s_lp = [_python_log_softmax(row) for row in student_logits[0].tolist()]
    t_lp = [_python_log_softmax(row) for row in teacher_logits[0].tolist()]
    expected_rows = [
        reverse_kl_from_candidate_scores(
            [s_lp[0][0], s_lp[0][3], (s_lp[0][0] + s_lp[1][1]) / 2.0],
            [t_lp[0][1], t_lp[0][4], t_lp[0][1]],
        ),
        reverse_kl_from_candidate_scores(
            [s_lp[2][0], s_lp[2][3]],
            [t_lp[1][1], t_lp[1][4]],
        ),
    ]
    observed = paper_simct_aligned_batch_loss(
        student_logits,
        student_ids,
        teacher_logits,
        teacher_ids,
        jnp.asarray([0, 3]),
        jnp.asarray([1, 4]),
        bounds,
        unit_mask,
        span_mask,
    )
    assert float(observed) == pytest.approx(sum(expected_rows) / 2.0, abs=1e-6)


def test_aligned_batch_loss_stops_teacher_gradient_and_uses_padding_mask():
    student_logits = jnp.asarray(
        [[[2.0, 0.0, -1.0], [0.0, 2.0, -1.0]]]
    )
    teacher_logits = jnp.asarray(
        [[[0.0, 2.0, -1.0], [2.0, 0.0, -1.0]]]
    )
    labels = jnp.asarray([[0, 1]])
    bounds = jnp.asarray([[[0, 2, 0, 2], [0, 0, 0, 0]]])
    unit_mask = jnp.asarray([[1.0, 0.0]])
    span_mask = jnp.asarray([[True, False]])

    def loss_fn(student_values, teacher_values):
        return paper_simct_aligned_batch_loss(
            student_values,
            labels,
            teacher_values,
            labels,
            jnp.asarray([0, 1]),
            jnp.asarray([0, 1]),
            bounds,
            unit_mask,
            span_mask,
        )

    student_gradient, teacher_gradient = jax.grad(loss_fn, argnums=(0, 1))(
        student_logits, teacher_logits
    )
    assert bool(jnp.all(jnp.isfinite(student_gradient)))
    assert float(jnp.linalg.norm(student_gradient)) > 0.0
    assert float(jnp.linalg.norm(teacher_gradient)) == pytest.approx(0.0)


def test_simple_opd_uses_only_one_to_one_units_and_stops_teacher_gradient():
    student_logits = jnp.asarray(
        [[[2.0, 0.0, -1.0], [0.0, 2.0, -1.0]]]
    )
    teacher_logits = jnp.asarray(
        [[[0.0, 2.0, -1.0], [2.0, 0.0, -1.0]]]
    )
    bounds = jnp.asarray([[[0, 1, 0, 1], [1, 2, 1, 2]]])
    unit_mask = jnp.asarray([[1.0, 1.0]])
    span_mask = jnp.asarray([[False, True]])

    expected = reverse_kl_from_candidate_scores(
        [2.0, 0.0], [0.0, 2.0]
    ) / 2.0

    def loss_fn(student_values, teacher_values):
        return paper_simple_opd_aligned_batch_loss(
            student_values,
            teacher_values,
            jnp.asarray([0, 1]),
            jnp.asarray([0, 1]),
            bounds,
            unit_mask,
            span_mask,
        )

    observed = loss_fn(student_logits, teacher_logits)
    student_gradient, teacher_gradient = jax.grad(loss_fn, argnums=(0, 1))(
        student_logits, teacher_logits
    )
    assert float(observed) == pytest.approx(expected, abs=1e-6)
    assert float(jnp.linalg.norm(student_gradient)) > 0.0
    assert float(jnp.linalg.norm(teacher_gradient)) == pytest.approx(0.0)
