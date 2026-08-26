"""JAX-native paper-math SimCT scoring and reverse-KL kernels.

JAX is imported lazily so the dependency-free configuration and checkpoint
contracts remain usable in the study environment.  The teacher side is always
stop-gradient in :func:`paper_simct_reverse_kl`; only student logits can receive
training gradients.
"""

from __future__ import annotations

from typing import Any


class JaxKernelUnavailable(RuntimeError):
    """Raised when a JAX kernel is requested without a usable JAX runtime."""


def _jax_modules() -> tuple[Any, Any]:
    try:
        import jax
        import jax.numpy as jnp
    except ImportError as exc:  # pragma: no cover - exercised without JAX
        raise JaxKernelUnavailable(
            "paper SimCT kernels require JAX; no fallback kernel is used"
        ) from exc
    return jax, jnp


def _require_rank(name: str, value: Any, rank: int) -> None:
    if getattr(value, "ndim", None) != rank:
        raise ValueError(f"{name} must have rank {rank}")


def paper_candidate_scores(
    position_logits: Any,
    realized_token_ids: Any,
    token_mask: Any,
) -> Any:
    """Compute SimCT Eq. (7) mean selected-token log-probabilities.

    Args:
        position_logits: ``[candidate, token, vocabulary]`` causal logits.  A
          separate row is required for every candidate continuation prefix.
        realized_token_ids: ``[candidate, token]`` selected continuation ids.
        token_mask: ``[candidate, token]`` positive for real tokens and zero
          for padding.  Every candidate must contain at least one real token.

    Returns:
        One paper-faithful mean-log-probability score per candidate.
    """

    jax, jnp = _jax_modules()
    logits = jnp.asarray(position_logits)
    token_ids = jnp.asarray(realized_token_ids)
    mask = jnp.asarray(token_mask)
    _require_rank("position_logits", logits, 3)
    _require_rank("realized_token_ids", token_ids, 2)
    _require_rank("token_mask", mask, 2)
    if logits.shape[:2] != token_ids.shape or token_ids.shape != mask.shape:
        raise ValueError(
            "candidate logits, token ids, and token mask shapes are inconsistent"
        )
    if logits.shape[-1] < 2:
        raise ValueError("candidate vocabulary must contain at least two entries")
    if not jnp.issubdtype(token_ids.dtype, jnp.integer):
        raise ValueError("realized_token_ids must have an integer dtype")

    lengths = jnp.sum(mask > 0, axis=-1)
    # This check is concrete in eager/CPU tests.  Under jit, callers are
    # responsible for the same static contract because Python bool conversion
    # of a tracer is intentionally unsupported by JAX.
    try:
        if bool(jnp.any(lengths == 0)):
            raise ValueError("every candidate must contain at least one token")
        if bool(jnp.any(token_ids < 0)) or bool(
            jnp.any(token_ids >= logits.shape[-1])
        ):
            raise ValueError("realized token id is outside the candidate vocabulary")
    except jax.errors.TracerBoolConversionError:
        pass

    log_probs = jax.nn.log_softmax(logits.astype(jnp.float32), axis=-1)
    selected = jnp.take_along_axis(log_probs, token_ids[..., None], axis=-1)[
        ..., 0
    ]
    selected = jnp.where(mask > 0, selected, 0.0)
    return jnp.sum(selected, axis=-1) / lengths.astype(jnp.float32)


def candidate_log_probs(scores: Any, *, temperature: float = 1.0) -> Any:
    """Softmax-normalize paper scores over the finite candidate set."""

    jax, jnp = _jax_modules()
    values = jnp.asarray(scores, dtype=jnp.float32)
    _require_rank("scores", values, 1)
    if values.shape[0] < 2:
        raise ValueError("SimCT requires at least two candidates")
    if not isinstance(temperature, (int, float)) or temperature <= 0.0:
        raise ValueError("temperature must be positive")
    return jax.nn.log_softmax(values / float(temperature), axis=-1)


def reverse_kl_from_scores(
    student_scores: Any,
    teacher_scores: Any,
    *,
    temperature: float = 1.0,
) -> Any:
    """Compute candidate-normalized ``KL(q_student || q_teacher)``."""

    _, jnp = _jax_modules()
    student = jnp.asarray(student_scores, dtype=jnp.float32)
    teacher = jnp.asarray(teacher_scores, dtype=jnp.float32)
    if student.shape != teacher.shape:
        raise ValueError("student and teacher candidate scores must match")
    student_log_probs = candidate_log_probs(student, temperature=temperature)
    teacher_log_probs = candidate_log_probs(teacher, temperature=temperature)
    return jnp.sum(
        jnp.exp(student_log_probs) * (student_log_probs - teacher_log_probs)
    )


def reverse_kl_loss_and_student_score_gradient(
    student_scores: Any,
    teacher_scores: Any,
    *,
    temperature: float = 1.0,
) -> tuple[Any, Any]:
    """Return reverse KL and its exact JAX gradient over student scores."""

    jax, jnp = _jax_modules()
    student = jnp.asarray(student_scores, dtype=jnp.float32)
    teacher = jax.lax.stop_gradient(
        jnp.asarray(teacher_scores, dtype=jnp.float32)
    )

    def loss_fn(values: Any) -> Any:
        return reverse_kl_from_scores(
            values,
            teacher,
            temperature=temperature,
        )

    return jax.value_and_grad(loss_fn)(student)


def paper_simct_reverse_kl(
    student_position_logits: Any,
    student_token_ids: Any,
    student_token_mask: Any,
    teacher_position_logits: Any,
    teacher_token_ids: Any,
    teacher_token_mask: Any,
    *,
    temperature: float = 1.0,
) -> Any:
    """End-to-end paper SimCT loss with a frozen teacher computation."""

    jax, _ = _jax_modules()
    student_scores = paper_candidate_scores(
        student_position_logits,
        student_token_ids,
        student_token_mask,
    )
    teacher_scores = jax.lax.stop_gradient(
        paper_candidate_scores(
            teacher_position_logits,
            teacher_token_ids,
            teacher_token_mask,
        )
    )
    return reverse_kl_from_scores(
        student_scores,
        teacher_scores,
        temperature=temperature,
    )


def _selected_log_probabilities(position_logits: Any, token_ids: Any) -> Any:
    """Return the realized-token log-probability at every causal position."""

    jax, jnp = _jax_modules()
    logits = jnp.asarray(position_logits)
    labels = jnp.asarray(token_ids)
    _require_rank("position_logits", logits, 3)
    _require_rank("token_ids", labels, 2)
    if logits.shape[:2] != labels.shape:
        raise ValueError("position logits and token ids must share batch/length")
    log_probs = jax.nn.log_softmax(logits.astype(jnp.float32), axis=-1)
    return jnp.take_along_axis(log_probs, labels[..., None], axis=-1)[..., 0]


def paper_teacher_sufficient_statistics(
    position_logits: Any,
    token_ids: Any,
    overlap_ids: Any,
) -> tuple[Any, Any]:
    """Reduce frozen-teacher logits to the exact statistics SimCT consumes.

    The full-vocabulary normalization is preserved.  The returned tensors are
    shared-token log-probabilities ``[batch, token, overlap]`` and realized
    selected-token log-probabilities ``[batch, token]``.  Keeping this
    reduction inside the teacher forward JIT lets XLA avoid exporting a
    persistent ``batch x sequence x vocabulary`` tensor.
    """

    jax, jnp = _jax_modules()
    logits = jnp.asarray(position_logits)
    labels = jnp.asarray(token_ids)
    shared = jnp.asarray(overlap_ids)
    _require_rank("position_logits", logits, 3)
    _require_rank("token_ids", labels, 2)
    _require_rank("overlap_ids", shared, 1)
    if logits.shape[:2] != labels.shape:
        raise ValueError("position logits and token ids must share batch/length")
    if shared.shape[0] < 1:
        raise ValueError("teacher sufficient statistics require overlap tokens")
    if not jnp.issubdtype(labels.dtype, jnp.integer):
        raise ValueError("token_ids must have an integer dtype")
    if not jnp.issubdtype(shared.dtype, jnp.integer):
        raise ValueError("overlap_ids must have an integer dtype")
    try:
        invalid_labels = jnp.any(labels < 0) | jnp.any(labels >= logits.shape[-1])
        invalid_shared = jnp.any(shared < 0) | jnp.any(shared >= logits.shape[-1])
        if bool(invalid_labels) or bool(invalid_shared):
            raise ValueError("teacher token id is outside the vocabulary")
    except jax.errors.TracerBoolConversionError:
        pass

    values = logits.astype(jnp.float32)
    log_normalizer = jax.scipy.special.logsumexp(values, axis=-1)
    shared_log_probs = (
        jnp.take(values, shared, axis=-1) - log_normalizer[..., None]
    )
    selected_logits = jnp.take_along_axis(
        values, labels[..., None], axis=-1
    )[..., 0]
    selected_log_probs = selected_logits - log_normalizer
    return (
        jax.lax.stop_gradient(shared_log_probs),
        jax.lax.stop_gradient(selected_log_probs),
    )


def paper_simct_aligned_batch_loss_from_teacher_statistics(
    student_position_logits: Any,
    student_token_ids: Any,
    teacher_shared_log_probs: Any,
    teacher_selected_log_probs: Any,
    student_overlap_ids: Any,
    segment_bounds: Any,
    segment_mask: Any,
    span_mask: Any,
    *,
    temperature: float = 1.0,
    normalizer: Any | None = None,
) -> Any:
    """Paper SimCT loss using exact reduced frozen-teacher statistics."""

    jax, jnp = _jax_modules()
    student_logits = jnp.asarray(student_position_logits)
    student_labels = jnp.asarray(student_token_ids)
    teacher_shared = jax.lax.stop_gradient(
        jnp.asarray(teacher_shared_log_probs, dtype=jnp.float32)
    )
    teacher_selected = jax.lax.stop_gradient(
        jnp.asarray(teacher_selected_log_probs, dtype=jnp.float32)
    )
    student_shared = jnp.asarray(student_overlap_ids)
    bounds = jnp.asarray(segment_bounds)
    units = jnp.asarray(segment_mask, dtype=jnp.float32)
    spans = jnp.asarray(span_mask, dtype=bool)

    _require_rank("student_position_logits", student_logits, 3)
    _require_rank("student_token_ids", student_labels, 2)
    _require_rank("teacher_shared_log_probs", teacher_shared, 3)
    _require_rank("teacher_selected_log_probs", teacher_selected, 2)
    _require_rank("student_overlap_ids", student_shared, 1)
    _require_rank("segment_bounds", bounds, 3)
    _require_rank("segment_mask", units, 2)
    _require_rank("span_mask", spans, 2)
    if student_logits.shape[:2] != student_labels.shape:
        raise ValueError("student logits/labels shape mismatch")
    if teacher_shared.shape[:2] != teacher_selected.shape:
        raise ValueError("teacher sufficient-statistic shapes mismatch")
    if teacher_shared.shape[-1] != student_shared.shape[0]:
        raise ValueError("student overlap and teacher shared-score widths mismatch")
    if bounds.shape[:2] != units.shape or units.shape != spans.shape:
        raise ValueError("segment bounds and masks must share batch/unit axes")
    if bounds.shape[-1] != 4:
        raise ValueError("segment_bounds must end in four interval coordinates")
    if student_shared.shape[0] < 1:
        raise ValueError("SimCT requires at least one shared-vocabulary token")

    student_selected = _selected_log_probabilities(
        student_logits, student_labels
    )
    student_all_log_probs = jax.nn.log_softmax(
        student_logits.astype(jnp.float32), axis=-1
    )

    ts, te, ss, se = (bounds[..., index] for index in range(4))
    batch = jnp.arange(bounds.shape[0], dtype=jnp.int32)[:, None]
    student_first = student_all_log_probs[batch, ss]
    student_shared_scores = jnp.take(
        student_first, student_shared, axis=-1
    )
    teacher_shared_scores = teacher_shared[batch, ts]

    def interval_means(selected: Any, starts: Any, ends: Any) -> Any:
        prefix = jnp.pad(jnp.cumsum(selected, axis=-1), ((0, 0), (1, 0)))
        total = prefix[batch, ends] - prefix[batch, starts]
        width = jnp.maximum(ends - starts, 1)
        return total / width.astype(jnp.float32)

    student_span_scores = interval_means(student_selected, ss, se)
    teacher_span_scores = interval_means(teacher_selected, ts, te)
    inactive = jnp.asarray(-1.0e30, dtype=jnp.float32)
    student_span_scores = jnp.where(spans, student_span_scores, inactive)
    teacher_span_scores = jnp.where(spans, teacher_span_scores, inactive)
    student_scores = jnp.concatenate(
        (student_shared_scores, student_span_scores[..., None]), axis=-1
    )
    teacher_scores = jnp.concatenate(
        (teacher_shared_scores, teacher_span_scores[..., None]), axis=-1
    )

    student_log_q = jax.nn.log_softmax(
        student_scores / float(temperature), axis=-1
    )
    teacher_log_q = jax.lax.stop_gradient(
        jax.nn.log_softmax(teacher_scores / float(temperature), axis=-1)
    )
    row_kl = jnp.sum(
        jnp.exp(student_log_q) * (student_log_q - teacher_log_q), axis=-1
    )
    numerator = jnp.sum(row_kl * units)
    denominator = (
        jnp.sum(units)
        if normalizer is None
        else jnp.asarray(normalizer, dtype=jnp.float32)
    )
    return numerator / jnp.maximum(denominator, 1.0)


def paper_simct_aligned_batch_loss(
    student_position_logits: Any,
    student_token_ids: Any,
    teacher_position_logits: Any,
    teacher_token_ids: Any,
    student_overlap_ids: Any,
    teacher_overlap_ids: Any,
    segment_bounds: Any,
    segment_mask: Any,
    span_mask: Any,
    *,
    temperature: float = 1.0,
    normalizer: Any | None = None,
) -> Any:
    """Vectorized paper-math SimCT loss on precomputed aligned segments.

    ``segment_bounds`` has shape ``[batch, unit, 4]`` and stores
    ``(teacher_start, teacher_end, student_start, student_end)`` over the
    completion-token axes.  Padding units are ignored by ``segment_mask``.

    Each real row compares the shared vocabulary plus, for a tokenizer-
    mismatched unit, one realized continuation candidate.  Shared candidates
    use the original next-token log-probability at the unit prefix.  The span
    candidate uses Eq. (7), i.e. the mean autoregressive token
    log-probability.  This is the written-paper coordinate system; it never
    substitutes mean raw logits and never applies the post-paper ``G(h)``
    safeguard.
    """

    _, jnp = _jax_modules()
    student_shared = jnp.asarray(student_overlap_ids)
    teacher_shared = jnp.asarray(teacher_overlap_ids)
    if student_shared.shape != teacher_shared.shape:
        raise ValueError("student/teacher overlap id arrays must have equal length")
    teacher_shared_log_probs, teacher_selected_log_probs = (
        paper_teacher_sufficient_statistics(
            teacher_position_logits,
            teacher_token_ids,
            teacher_shared,
        )
    )
    return paper_simct_aligned_batch_loss_from_teacher_statistics(
        student_position_logits,
        student_token_ids,
        teacher_shared_log_probs,
        teacher_selected_log_probs,
        student_shared,
        segment_bounds,
        segment_mask,
        span_mask,
        temperature=temperature,
        normalizer=normalizer,
    )


def paper_simple_opd_aligned_batch_loss(
    student_position_logits: Any,
    teacher_position_logits: Any,
    student_overlap_ids: Any,
    teacher_overlap_ids: Any,
    segment_bounds: Any,
    segment_mask: Any,
    span_mask: Any,
    *,
    temperature: float = 1.0,
    normalizer: Any | None = None,
) -> Any:
    """Reverse KL on overlap vocabulary at exact one-to-one aligned units.

    This is the no-span control paired with SimCT.  A row is active only when
    the realized teacher and student tokens share both prefix and suffix byte
    boundaries.  The candidate support is the normalized overlap vocabulary;
    mismatched multi-token units receive no SimpleOPD credit.
    """

    _, jnp = _jax_modules()
    student_shared = jnp.asarray(student_overlap_ids)
    teacher_shared = jnp.asarray(teacher_overlap_ids)
    if student_shared.shape != teacher_shared.shape:
        raise ValueError("student/teacher overlap id arrays must have equal length")
    teacher_shared_scores = jnp.take(
        jnp.asarray(teacher_position_logits), teacher_shared, axis=-1
    )
    return paper_simple_opd_aligned_batch_loss_from_teacher_statistics(
        student_position_logits,
        teacher_shared_scores,
        student_shared,
        segment_bounds,
        segment_mask,
        span_mask,
        temperature=temperature,
        normalizer=normalizer,
    )


def paper_simple_opd_aligned_batch_loss_from_teacher_statistics(
    student_position_logits: Any,
    teacher_shared_scores: Any,
    student_overlap_ids: Any,
    segment_bounds: Any,
    segment_mask: Any,
    span_mask: Any,
    *,
    temperature: float = 1.0,
    normalizer: Any | None = None,
) -> Any:
    """SimpleOPD loss using overlap-only frozen-teacher scores."""

    jax, jnp = _jax_modules()
    student_logits = jnp.asarray(student_position_logits)
    teacher_scores_by_position = jax.lax.stop_gradient(
        jnp.asarray(teacher_shared_scores, dtype=jnp.float32)
    )
    student_shared = jnp.asarray(student_overlap_ids)
    bounds = jnp.asarray(segment_bounds)
    units = jnp.asarray(segment_mask, dtype=jnp.float32)
    spans = jnp.asarray(span_mask, dtype=bool)

    _require_rank("student_position_logits", student_logits, 3)
    _require_rank("teacher_shared_scores", teacher_scores_by_position, 3)
    _require_rank("student_overlap_ids", student_shared, 1)
    _require_rank("segment_bounds", bounds, 3)
    _require_rank("segment_mask", units, 2)
    _require_rank("span_mask", spans, 2)
    if bounds.shape[:2] != units.shape or units.shape != spans.shape:
        raise ValueError("segment bounds and masks must share batch/unit axes")
    if bounds.shape[-1] != 4:
        raise ValueError("segment_bounds must end in four interval coordinates")
    if teacher_scores_by_position.shape[-1] != student_shared.shape[0]:
        raise ValueError("student overlap and teacher shared-score widths mismatch")
    if student_shared.shape[0] < 2:
        raise ValueError("SimpleOPD requires at least two shared-vocabulary tokens")
    if not isinstance(temperature, (int, float)) or temperature <= 0.0:
        raise ValueError("temperature must be positive")

    ts, _, ss, _ = (bounds[..., index] for index in range(4))
    batch = jnp.arange(bounds.shape[0], dtype=jnp.int32)[:, None]
    student_scores = jnp.take(
        student_logits[batch, ss], student_shared, axis=-1
    )
    teacher_scores = teacher_scores_by_position[batch, ts]
    student_log_probs = jax.nn.log_softmax(
        student_scores.astype(jnp.float32) / float(temperature), axis=-1
    )
    teacher_log_probs = jax.lax.stop_gradient(
        jax.nn.log_softmax(
            teacher_scores.astype(jnp.float32) / float(temperature), axis=-1
        )
    )
    row_kl = jnp.sum(
        jnp.exp(student_log_probs) * (student_log_probs - teacher_log_probs),
        axis=-1,
    )
    active = units * jnp.logical_not(spans).astype(jnp.float32)
    numerator = jnp.sum(row_kl * active)
    denominator = (
        jnp.sum(units)
        if normalizer is None
        else jnp.asarray(normalizer, dtype=jnp.float32)
    )
    return numerator / jnp.maximum(denominator, 1.0)
