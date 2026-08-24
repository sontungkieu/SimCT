"""Exercise 2: enumerate feasible unions of atomic aligned spans."""

from __future__ import annotations

from collections.abc import Sequence

from vdt_span.types import AlignedSpan, SpanCandidate


def enumerate_span_candidates(
    atoms: Sequence[AlignedSpan],
    *,
    max_teacher_tokens: int,
    max_student_tokens: int,
    max_bytes: int,
) -> tuple[SpanCandidate, ...]:
    """Enumerate every feasible non-empty consecutive union of ``atoms``.

    Results must be unique and ordered lexicographically by
    ``(atom_start, atom_end)``. See Exercise 2 in ``STUDY.md`` for the
    contiguity invariant and resource constraints.
    """

    raise NotImplementedError("TODO E2: candidate span enumeration")
