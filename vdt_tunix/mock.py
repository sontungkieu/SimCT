"""Deterministic CPU doubles for contract tests; never scientific evidence."""

from __future__ import annotations

from vdt_tunix.config import RunConfig
from vdt_tunix.contracts import (
    INTERFACE_CONTRACT_VERSION,
    LogitsPayload,
    RolloutRequest,
    StudentRolloutBatch,
    StudentRolloutSample,
    TeacherScoreBatch,
    TeacherScoreRequest,
    TeacherScoreSample,
    TokenSequence,
)


def _ids(text: str, modulus: int) -> tuple[int, ...]:
    return tuple((sum(char.encode("utf-8")) % modulus) + 1 for char in text)


def _character_tokens(text: str) -> TokenSequence:
    pieces = tuple(text)
    return TokenSequence(text=text, token_ids=_ids(text, 251), pieces=pieces)


def _two_character_tokens(text: str) -> TokenSequence:
    pieces = tuple(text[index : index + 2] for index in range(0, len(text), 2))
    token_ids = tuple(
        (sum(piece.encode("utf-8")) % 257) + 1 for piece in pieces
    )
    return TokenSequence(text=text, token_ids=token_ids, pieces=pieces)


class MockStudentRolloutBackend:
    backend_name = "cpu-mock-student-rollout"
    real_model_integration = False

    def __init__(self, config: RunConfig):
        self._config = config

    def rollout(self, request: RolloutRequest) -> StudentRolloutBatch:
        samples: list[StudentRolloutSample] = []
        for prompt in request.prompts:
            for sample_index in range(request.samples_per_prompt):
                completion = _character_tokens(
                    f"mock:{prompt.prompt_id}:{sample_index}"
                )
                samples.append(
                    StudentRolloutSample(
                        sample_id=f"{prompt.prompt_id}/{sample_index}",
                        prompt_id=prompt.prompt_id,
                        student_prompt_token_ids=_ids(prompt.student_prompt, 251),
                        completion=completion,
                        rollout_log_probs=(-0.125,) * len(completion.token_ids),
                    )
                )
        model = self._config.student
        return StudentRolloutBatch(
            contract_version=INTERFACE_CONTRACT_VERSION,
            run_id=request.run_id,
            step=request.step,
            model_id=model.model_id,
            model_revision=model.model_revision,
            tokenizer_id=model.tokenizer_id,
            tokenizer_revision=model.tokenizer_revision,
            samples=tuple(samples),
        )


class MockTeacherScoreBackend:
    backend_name = "cpu-mock-teacher-score"
    real_model_integration = False

    def __init__(self, config: RunConfig):
        self._config = config

    def score(self, request: TeacherScoreRequest) -> TeacherScoreBatch:
        prompt_by_id = {prompt.prompt_id: prompt for prompt in request.prompts}
        scored: list[TeacherScoreSample] = []
        vocab_size = 11
        for rollout in request.rollouts.samples:
            prompt = prompt_by_id[rollout.prompt_id]
            completion = _two_character_tokens(rollout.completion.text)
            values = tuple(
                tuple(float((row + column) % vocab_size) for column in range(vocab_size))
                for row in range(len(completion.token_ids))
            )
            scored.append(
                TeacherScoreSample(
                    sample_id=rollout.sample_id,
                    prompt_id=rollout.prompt_id,
                    teacher_prompt_token_ids=_ids(prompt.teacher_prompt, 257),
                    completion=completion,
                    position_logits=LogitsPayload(
                        values=values,
                        shape=(len(values), vocab_size),
                        dtype="float32",
                    ),
                )
            )
        model = self._config.teacher
        return TeacherScoreBatch(
            contract_version=INTERFACE_CONTRACT_VERSION,
            run_id=request.rollouts.run_id,
            step=request.rollouts.step,
            model_id=model.model_id,
            model_revision=model.model_revision,
            tokenizer_id=model.tokenizer_id,
            tokenizer_revision=model.tokenizer_revision,
            samples=tuple(scored),
        )
