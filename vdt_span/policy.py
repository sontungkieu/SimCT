"""Exercise 6: a bounded context- and training-dependent span policy."""

from __future__ import annotations

from vdt_span.types import PolicyContext, SpanPolicyConfig, TrainingState


def adaptive_max_span_width(
    context: PolicyContext,
    training: TrainingState,
    config: SpanPolicyConfig = SpanPolicyConfig(),
) -> int:
    """Choose a deterministic maximum span width for this state/context.

    The public exercise uses a deliberately small, auditable policy rather
    than a learned router. See Exercise 6 in ``STUDY.md`` for its exact
    schedule, risk factors, bounds, and validation requirements.
    """

    raise NotImplementedError("TODO E6: adaptive span policy")
