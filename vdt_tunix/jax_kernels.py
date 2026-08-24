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
