"""Shared immutable data structures for the learning exercises."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AlignedSpan:
    """One maximal interval between two consecutive shared byte boundaries."""

    teacher_start: int
    teacher_end: int
    student_start: int
    student_end: int
    byte_start: int
    byte_end: int

    def __post_init__(self) -> None:
        if not (0 <= self.teacher_start < self.teacher_end):
            raise ValueError("teacher range must be non-empty and non-negative")
        if not (0 <= self.student_start < self.student_end):
            raise ValueError("student range must be non-empty and non-negative")
        if not (0 <= self.byte_start < self.byte_end):
            raise ValueError("byte range must be non-empty and non-negative")

    @property
    def teacher_width(self) -> int:
        return self.teacher_end - self.teacher_start

    @property
    def student_width(self) -> int:
        return self.student_end - self.student_start

    @property
    def byte_width(self) -> int:
        return self.byte_end - self.byte_start


@dataclass(frozen=True, slots=True)
class SpanCandidate:
    """A candidate made by joining consecutive atomic aligned spans."""

    atom_start: int
    atom_end: int
    teacher_start: int
    teacher_end: int
    student_start: int
    student_end: int
    byte_start: int
    byte_end: int

    def __post_init__(self) -> None:
        if not (0 <= self.atom_start < self.atom_end):
            raise ValueError("atom range must be non-empty and non-negative")
        if not (0 <= self.teacher_start < self.teacher_end):
            raise ValueError("teacher range must be non-empty and non-negative")
        if not (0 <= self.student_start < self.student_end):
            raise ValueError("student range must be non-empty and non-negative")
        if not (0 <= self.byte_start < self.byte_end):
            raise ValueError("byte range must be non-empty and non-negative")

    @property
    def atom_width(self) -> int:
        return self.atom_end - self.atom_start

    @property
    def teacher_width(self) -> int:
        return self.teacher_end - self.teacher_start

    @property
    def student_width(self) -> int:
        return self.student_end - self.student_start

    @property
    def byte_width(self) -> int:
        return self.byte_end - self.byte_start


@dataclass(frozen=True, slots=True)
class ScoredSpan:
    """A semi-Markov edge over atomic positions ``[start, end)``."""

    start: int
    end: int
    score: float


@dataclass(frozen=True, slots=True)
class ViterbiPath:
    """Best complete segmentation and its additive score."""

    score: float
    spans: tuple[tuple[int, int], ...]


@dataclass(frozen=True, slots=True)
class PolicyContext:
    """Observable reliability/risk signals for the current context."""

    boundary_confidence: float
    teacher_disagreement: float
    context_risk: float


@dataclass(frozen=True, slots=True)
class TrainingState:
    """Training schedule state used by the adaptive policy."""

    step: int
    warmup_steps: int


@dataclass(frozen=True, slots=True)
class SpanPolicyConfig:
    """Hard width bounds for the adaptive policy."""

    min_width: int = 1
    max_width: int = 8
