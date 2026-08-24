"""Host-side construction of SimCT's shared support and aligned-unit layout."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import dataclasses
from typing import Any

from vdt_span.alignment import AlignedUnit, minimal_joint_segments
from vdt_tunix.contracts import TokenSequence


class SupervisionError(ValueError):
    """Raised when tokenizers cannot define one deterministic SimCT support."""


@dataclasses.dataclass(frozen=True, slots=True)
class OverlapVocabulary:
    """Deterministic paired IDs for the public SimCT overlap heuristic."""

    normalized_tokens: tuple[str, ...]
    student_ids: tuple[int, ...]
    teacher_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.normalized_tokens:
            raise SupervisionError("student and teacher vocabularies do not overlap")
        if not (
            len(self.normalized_tokens)
            == len(self.student_ids)
            == len(self.teacher_ids)
        ):
            raise SupervisionError("overlap token/id arrays have inconsistent lengths")
        if len(set(self.normalized_tokens)) != len(self.normalized_tokens):
            raise SupervisionError("normalized overlap tokens must be unique")


@dataclasses.dataclass(frozen=True, slots=True)
class AlignedLayout:
    """Padded-free single-sample layout over completion-token positions."""

    units: tuple[AlignedUnit, ...]

    @property
    def bounds(self) -> tuple[tuple[int, int, int, int], ...]:
        return tuple(
            (
                unit.teacher_start,
                unit.teacher_end,
                unit.student_start,
                unit.student_end,
            )
            for unit in self.units
        )

    @property
    def span_mask(self) -> tuple[bool, ...]:
        return tuple(
            unit.teacher_width != 1 or unit.student_width != 1
            for unit in self.units
        )


def _raw_vocabulary(tokenizer: Any) -> Mapping[str, int]:
    get_vocab = getattr(tokenizer, "get_vocab", None)
    if callable(get_vocab):
        value = get_vocab()
        if isinstance(value, Mapping):
            return value
    vocab_size = getattr(tokenizer, "vocab_size", None)
    if callable(vocab_size):
        vocab_size = vocab_size()
    id_to_piece = getattr(tokenizer, "id_to_piece", None)
    if not callable(id_to_piece):
        id_to_piece = getattr(tokenizer, "IdToPiece", None)
    if isinstance(vocab_size, int) and vocab_size > 0 and callable(id_to_piece):
        return {str(id_to_piece(index)): index for index in range(vocab_size)}
    raise SupervisionError(
        "tokenizer must expose get_vocab(), or vocab_size plus id_to_piece()"
    )


def _normalize_public_token(token: str) -> str:
    # This intentionally matches the released SimCT implementation.  It is a
    # public-code compatibility heuristic, not a proof of decoded-byte identity.
    return token.replace("Ġ", "▁")


def _normalized_vocabulary(tokenizer: Any) -> dict[str, int]:
    result: dict[str, int] = {}
    collisions: set[str] = set()
    for token, token_id in _raw_vocabulary(tokenizer).items():
        if isinstance(token_id, bool) or not isinstance(token_id, int):
            raise SupervisionError("tokenizer vocabulary IDs must be integers")
        normalized = _normalize_public_token(str(token))
        previous = result.get(normalized)
        if previous is not None and previous != token_id:
            collisions.add(normalized)
            continue
        result[normalized] = int(token_id)
    # A normalization collision makes the token-to-ID map ambiguous.  Dropping
    # it is deterministic and safer than whichever insertion happened last.
    for token in collisions:
        result.pop(token, None)
    return result


def _special_id(tokenizer: Any, attribute: str, method: str) -> int | None:
    value = getattr(tokenizer, attribute, None)
    if value is None:
        candidate = getattr(tokenizer, method, None)
        value = candidate() if callable(candidate) else candidate
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return int(value)
    return None


def build_overlap_vocabulary(
    student_tokenizer: Any,
    teacher_tokenizer: Any,
) -> OverlapVocabulary:
    """Reproduce the released overlap-token mapping with deterministic order."""

    student = _normalized_vocabulary(student_tokenizer)
    teacher = _normalized_vocabulary(teacher_tokenizer)
    shared = sorted(student.keys() & teacher.keys())
    student_ids = [student[token] for token in shared]
    teacher_ids = [teacher[token] for token in shared]

    student_eos = _special_id(student_tokenizer, "eos_token_id", "eos_id")
    teacher_eos = _special_id(teacher_tokenizer, "eos_token_id", "eos_id")
    if student_eos is not None and teacher_eos is not None:
        pair = (student_eos, teacher_eos)
        if pair not in set(zip(student_ids, teacher_ids, strict=True)):
            shared.append("<paired-eos>")
            student_ids.append(student_eos)
            teacher_ids.append(teacher_eos)
    return OverlapVocabulary(
        normalized_tokens=tuple(shared),
        student_ids=tuple(student_ids),
        teacher_ids=tuple(teacher_ids),
    )


def build_aligned_layout(
    student: TokenSequence,
    teacher: TokenSequence,
) -> AlignedLayout:
    """Build the paper's finest common decoded-byte partition for one rollout."""

    if student.text != teacher.text:
        raise SupervisionError("student and teacher must tokenize identical text")
    units = minimal_joint_segments(teacher.pieces, student.pieces)
    if not units:
        raise SupervisionError("an OPD completion must contain an aligned unit")
    return AlignedLayout(units=units)


def pad_layouts(
    layouts: Sequence[AlignedLayout],
) -> tuple[list[list[list[int]]], list[list[float]], list[list[bool]]]:
    """Pad host layouts into JSON/JAX-friendly batch arrays."""

    if not layouts:
        raise SupervisionError("layout batch must not be empty")
    width = max(len(layout.units) for layout in layouts)
    bounds: list[list[list[int]]] = []
    unit_mask: list[list[float]] = []
    span_mask: list[list[bool]] = []
    for layout in layouts:
        real = [list(item) for item in layout.bounds]
        spans = list(layout.span_mask)
        padding = width - len(real)
        bounds.append(real + [[0, 0, 0, 0] for _ in range(padding)])
        unit_mask.append([1.0] * len(real) + [0.0] * padding)
        span_mask.append(spans + [False] * padding)
    return bounds, unit_mask, span_mask
