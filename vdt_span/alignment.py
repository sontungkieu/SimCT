"""Exercise 1: exact alignment on the decoded-byte boundary lattice."""

from __future__ import annotations

from collections.abc import Sequence

from vdt_span.types import AlignedSpan


def align_exact_byte_boundaries(
    teacher_tokens: Sequence[bytes],
    student_tokens: Sequence[bytes],
) -> tuple[AlignedSpan, ...]:
    """Return atomic spans induced by shared cumulative byte boundaries.

    Token pieces may contain incomplete UTF-8 byte sequences. Therefore the
    contract is defined on raw bytes, not on characters or independently
    decoded token strings. See Exercise 1 in ``STUDY.md`` for validation and
    maximality requirements.
    """

    raise NotImplementedError("TODO E1: exact byte-boundary alignment")
