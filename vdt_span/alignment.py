"""Exact byte-boundary alignment independent of model frameworks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence


class AlignmentError(ValueError):
    """Raised when two token-piece sequences cannot describe one byte string."""


@dataclass(frozen=True, slots=True)
class AlignedUnit:
    """One finest unit bounded by both tokenizers.

    Token intervals are half open. Byte offsets refer to the common decoded
    byte string and are also half open.
    """

    teacher_start: int
    teacher_end: int
    student_start: int
    student_end: int
    byte_start: int
    byte_end: int

    @property
    def teacher_width(self) -> int:
        return self.teacher_end - self.teacher_start

    @property
    def student_width(self) -> int:
        return self.student_end - self.student_start

    @property
    def byte_width(self) -> int:
        return self.byte_end - self.byte_start


def _piece_bytes(piece: str | bytes) -> bytes:
    if isinstance(piece, bytes):
        value = piece
    elif isinstance(piece, str):
        value = piece.encode("utf-8")
    else:
        raise TypeError(f"piece must be str or bytes, got {type(piece).__name__}")
    if not value:
        raise AlignmentError(
            "zero-byte pieces make token-boundary indices ambiguous; remove or "
            "represent special tokens explicitly before alignment"
        )
    return value


def _boundaries(pieces: Sequence[str | bytes]) -> tuple[list[bytes], dict[int, int]]:
    encoded = [_piece_bytes(piece) for piece in pieces]
    offset_to_token_index = {0: 0}
    offset = 0
    for index, piece in enumerate(encoded, start=1):
        offset += len(piece)
        if offset in offset_to_token_index:
            raise AlignmentError("duplicate byte boundary encountered")
        offset_to_token_index[offset] = index
    return encoded, offset_to_token_index


def _first_mismatch(left: bytes, right: bytes) -> int:
    limit = min(len(left), len(right))
    for index in range(limit):
        if left[index] != right[index]:
            return index
    return limit


def minimal_joint_segments(
    teacher_pieces: Sequence[str | bytes],
    student_pieces: Sequence[str | bytes],
) -> tuple[AlignedUnit, ...]:
    """Return the unique finest partition at common decoded-byte boundaries.

    This function solves mechanical alignment only. It does not claim that the
    returned units are semantically optimal or optimal for gradient credit.
    """

    if not teacher_pieces and not student_pieces:
        return ()
    if not teacher_pieces or not student_pieces:
        raise AlignmentError("one side is empty while the other side is not")

    teacher_bytes, teacher_boundaries = _boundaries(teacher_pieces)
    student_bytes, student_boundaries = _boundaries(student_pieces)
    teacher_text = b"".join(teacher_bytes)
    student_text = b"".join(student_bytes)
    if teacher_text != student_text:
        mismatch = _first_mismatch(teacher_text, student_text)
        raise AlignmentError(
            "decoded byte strings differ at offset "
            f"{mismatch}: teacher_len={len(teacher_text)}, "
            f"student_len={len(student_text)}"
        )

    common_offsets = sorted(set(teacher_boundaries) & set(student_boundaries))
    if common_offsets[0] != 0 or common_offsets[-1] != len(teacher_text):
        raise AlignmentError("common boundaries do not cover the decoded response")

    units: list[AlignedUnit] = []
    for byte_start, byte_end in zip(common_offsets, common_offsets[1:]):
        units.append(
            AlignedUnit(
                teacher_start=teacher_boundaries[byte_start],
                teacher_end=teacher_boundaries[byte_end],
                student_start=student_boundaries[byte_start],
                student_end=student_boundaries[byte_end],
                byte_start=byte_start,
                byte_end=byte_end,
            )
        )
    return tuple(units)


def coarsen_aligned_units(
    units: Sequence[AlignedUnit],
    partition: Iterable[tuple[int, int]],
) -> tuple[AlignedUnit, ...]:
    """Merge consecutive atomic units according to an index partition."""

    spans = tuple(partition)
    if not units:
        if spans:
            raise ValueError("cannot partition an empty aligned-unit sequence")
        return ()

    cursor = 0
    merged: list[AlignedUnit] = []
    for start, end in spans:
        if start != cursor or end <= start or end > len(units):
            raise ValueError(
                "partition must be contiguous, non-empty, and cover units exactly"
            )
        left = units[start]
        right = units[end - 1]
        merged.append(
            AlignedUnit(
                teacher_start=left.teacher_start,
                teacher_end=right.teacher_end,
                student_start=left.student_start,
                student_end=right.student_end,
                byte_start=left.byte_start,
                byte_end=right.byte_end,
            )
        )
        cursor = end
    if cursor != len(units):
        raise ValueError("partition does not cover every aligned unit")
    return tuple(merged)
