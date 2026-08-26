"""In-memory interfaces between rollout, scoring, and SimCT consumers."""

from __future__ import annotations

import dataclasses
from typing import Any, Protocol, runtime_checkable


INTERFACE_CONTRACT_VERSION = 1


class ContractError(ValueError):
    """Raised when a backend returns a malformed contract object."""


def _piece_bytes(piece: str | bytes) -> bytes:
    if isinstance(piece, bytes):
        value = piece
    elif isinstance(piece, str):
        value = piece.encode("utf-8")
    else:
        raise ContractError("token pieces must be str or bytes")
    if not value:
        raise ContractError(
            "zero-byte token pieces are unsupported; omit terminal special tokens"
        )
    return value


@dataclasses.dataclass(frozen=True, slots=True)
class TokenSequence:
    """One tokenizer's loss-bearing tokens for exactly one decoded text.

    ``pieces`` exclude padding and terminal special tokens. Concatenating their
    UTF-8 bytes must reproduce ``text`` exactly, which gives SimCT a tokenizer-
    independent alignment coordinate without assuming one-to-one tokens.
    """

    text: str
    token_ids: tuple[int, ...]
    pieces: tuple[str | bytes, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.text, str) or not self.text:
            raise ContractError("token sequence text must be non-empty")
        if len(self.token_ids) != len(self.pieces):
            raise ContractError("token_ids and pieces must have equal length")
        if not self.token_ids:
            raise ContractError("token sequence must contain at least one token")
        if any(
            isinstance(token_id, bool) or not isinstance(token_id, int)
            for token_id in self.token_ids
        ):
            raise ContractError("token_ids must contain integers")
        if any(token_id < 0 for token_id in self.token_ids):
            raise ContractError("token_ids must be non-negative")
        decoded = b"".join(_piece_bytes(piece) for piece in self.pieces)
        if decoded != self.text.encode("utf-8"):
            raise ContractError(
                "token pieces do not reproduce text exactly in UTF-8 bytes"
            )


@dataclasses.dataclass(frozen=True, slots=True)
class PromptRecord:
    prompt_id: str
    student_prompt: str
    teacher_prompt: str

    def __post_init__(self) -> None:
        for name in ("prompt_id", "student_prompt", "teacher_prompt"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ContractError(f"{name} must be a non-empty string")


@dataclasses.dataclass(frozen=True, slots=True)
class RolloutRequest:
    run_id: str
    step: int
    prompts: tuple[PromptRecord, ...]
    samples_per_prompt: int

    def __post_init__(self) -> None:
        if not self.run_id:
            raise ContractError("run_id must be non-empty")
        if self.step < 0:
            raise ContractError("step must be non-negative")
        if not self.prompts:
            raise ContractError("rollout request must contain prompts")
        if self.samples_per_prompt < 1:
            raise ContractError("samples_per_prompt must be positive")
        prompt_ids = [prompt.prompt_id for prompt in self.prompts]
        if len(prompt_ids) != len(set(prompt_ids)):
            raise ContractError("prompt ids must be unique")


@dataclasses.dataclass(frozen=True, slots=True)
class StudentRolloutSample:
    sample_id: str
    prompt_id: str
    student_prompt_token_ids: tuple[int, ...]
    completion: TokenSequence
    rollout_log_probs: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.sample_id or not self.prompt_id:
            raise ContractError("sample_id and prompt_id must be non-empty")
        if not self.student_prompt_token_ids:
            raise ContractError("student prompt token ids must be non-empty")
        if len(self.rollout_log_probs) != len(self.completion.token_ids):
            raise ContractError(
                "rollout_log_probs must match the student completion length"
            )


@dataclasses.dataclass(frozen=True, slots=True)
class StudentRolloutBatch:
    contract_version: int
    run_id: str
    step: int
    model_id: str
    model_revision: str
    tokenizer_id: str
    tokenizer_revision: str
    samples: tuple[StudentRolloutSample, ...]

    def __post_init__(self) -> None:
        if self.contract_version != INTERFACE_CONTRACT_VERSION:
            raise ContractError("unsupported student rollout contract version")
        if self.step < 0 or not self.run_id:
            raise ContractError("invalid rollout run coordinate")
        if not self.samples:
            raise ContractError("rollout batch must contain samples")
        sample_ids = [sample.sample_id for sample in self.samples]
        if len(sample_ids) != len(set(sample_ids)):
            raise ContractError("student rollout sample ids must be unique")


@dataclasses.dataclass(frozen=True, slots=True)
class TeacherScoreRequest:
    rollouts: StudentRolloutBatch
    prompts: tuple[PromptRecord, ...]

    def __post_init__(self) -> None:
        prompt_ids = {prompt.prompt_id for prompt in self.prompts}
        if any(sample.prompt_id not in prompt_ids for sample in self.rollouts.samples):
            raise ContractError("teacher score request is missing prompt metadata")


def _payload_shape(values: Any) -> tuple[int, ...] | None:
    shape = getattr(values, "shape", None)
    if shape is not None:
        return tuple(int(size) for size in shape)
    if isinstance(values, (tuple, list)):
        if not values:
            return (0,)
        first = values[0]
        if isinstance(first, (tuple, list)):
            width = len(first)
            if any(not isinstance(row, (tuple, list)) or len(row) != width for row in values):
                raise ContractError("logit payload rows must have a fixed width")
            return (len(values), width)
    return None


@dataclasses.dataclass(frozen=True, slots=True)
class LogitsPayload:
    """Backend-owned full-vocabulary logits plus explicit shape metadata."""

    values: Any = dataclasses.field(repr=False, compare=False)
    shape: tuple[int, int]
    dtype: str

    def __post_init__(self) -> None:
        if len(self.shape) != 2 or self.shape[0] < 1 or self.shape[1] < 2:
            raise ContractError("teacher logits must have shape [tokens, vocab>=2]")
        if not self.dtype:
            raise ContractError("teacher logits dtype must be declared")
        observed = _payload_shape(self.values)
        if observed is not None and observed != self.shape:
            raise ContractError(
                f"teacher logits payload shape {observed} != declared {self.shape}"
            )


@dataclasses.dataclass(frozen=True, slots=True)
class TeacherSufficientStatisticsPayload:
    """Exact reduced teacher scores used by the paper SimCT loss."""

    shared_log_probs: Any = dataclasses.field(repr=False, compare=False)
    selected_log_probs: Any = dataclasses.field(repr=False, compare=False)
    shape: tuple[int, int]
    dtype: str

    def __post_init__(self) -> None:
        if len(self.shape) != 2 or self.shape[0] < 1 or self.shape[1] < 1:
            raise ContractError(
                "teacher shared log-probabilities must have shape "
                "[tokens, overlap>=1]"
            )
        shared_shape = _payload_shape(self.shared_log_probs)
        if shared_shape is not None and shared_shape != self.shape:
            raise ContractError(
                "teacher shared-score payload shape "
                f"{shared_shape} != declared {self.shape}"
            )
        selected_shape = _payload_shape(self.selected_log_probs)
        if selected_shape is not None and selected_shape != (self.shape[0],):
            raise ContractError(
                "teacher selected-score payload must match the token axis"
            )
        if not self.dtype:
            raise ContractError("teacher sufficient-statistic dtype must be declared")


@dataclasses.dataclass(frozen=True, slots=True)
class TeacherScoreSample:
    sample_id: str
    prompt_id: str
    teacher_prompt_token_ids: tuple[int, ...]
    completion: TokenSequence
    position_logits: LogitsPayload | None = None
    sufficient_statistics: TeacherSufficientStatisticsPayload | None = None

    def __post_init__(self) -> None:
        if not self.sample_id or not self.prompt_id:
            raise ContractError("sample_id and prompt_id must be non-empty")
        if not self.teacher_prompt_token_ids:
            raise ContractError("teacher prompt token ids must be non-empty")
        if (self.position_logits is None) == (self.sufficient_statistics is None):
            raise ContractError(
                "teacher score must provide exactly one score representation"
            )
        token_rows = (
            self.position_logits.shape[0]
            if self.position_logits is not None
            else self.sufficient_statistics.shape[0]
        )
        if token_rows != len(self.completion.token_ids):
            raise ContractError(
                "teacher score rows must match teacher completion tokens"
            )


@dataclasses.dataclass(frozen=True, slots=True)
class TeacherScoreBatch:
    contract_version: int
    run_id: str
    step: int
    model_id: str
    model_revision: str
    tokenizer_id: str
    tokenizer_revision: str
    samples: tuple[TeacherScoreSample, ...]

    def __post_init__(self) -> None:
        if self.contract_version != INTERFACE_CONTRACT_VERSION:
            raise ContractError("unsupported teacher score contract version")
        if self.step < 0 or not self.run_id:
            raise ContractError("invalid teacher score run coordinate")
        if not self.samples:
            raise ContractError("teacher score batch must contain samples")
        sample_ids = [sample.sample_id for sample in self.samples]
        if len(sample_ids) != len(set(sample_ids)):
            raise ContractError("teacher score sample ids must be unique")


@runtime_checkable
class StudentRolloutBackend(Protocol):
    backend_name: str
    real_model_integration: bool

    def rollout(self, request: RolloutRequest) -> StudentRolloutBatch:
        """Generate on-policy student completions and rollout log-probabilities."""


@runtime_checkable
class TeacherScoreBackend(Protocol):
    backend_name: str
    real_model_integration: bool

    def score(self, request: TeacherScoreRequest) -> TeacherScoreBatch:
        """Score shared completion text under the single teacher tokenizer."""


@dataclasses.dataclass(frozen=True, slots=True)
class BackendBundle:
    student: StudentRolloutBackend
    teacher: TeacherScoreBackend

    def __post_init__(self) -> None:
        if not isinstance(self.student, StudentRolloutBackend):
            raise ContractError("student backend does not implement rollout contract")
        if not isinstance(self.teacher, TeacherScoreBackend):
            raise ContractError("teacher backend does not implement score contract")
