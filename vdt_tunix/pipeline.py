"""Contract-only orchestration for one student rollout and teacher score pass."""

from __future__ import annotations

import dataclasses
from typing import Any

from vdt_tunix.config import RunConfig
from vdt_tunix.contracts import (
    BackendBundle,
    INTERFACE_CONTRACT_VERSION,
    PromptRecord,
    RolloutRequest,
    TeacherScoreRequest,
)


class PipelineContractError(RuntimeError):
    """Raised when a backend violates configuration or identity contracts."""


@dataclasses.dataclass(frozen=True, slots=True)
class CanaryReport:
    status: str
    run_id: str
    config_sha256: str
    student_backend: str
    teacher_backend: str
    sample_count: int
    student_completion_tokens: int
    teacher_completion_tokens: int
    cross_tokenization_observed: bool
    real_model_integration: bool
    hardware: dict[str, Any] | None
    scientific_evidence: bool = False
    simct_update_executed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _model_tuple(batch: Any) -> tuple[str, str, str, str]:
    return (
        batch.model_id,
        batch.model_revision,
        batch.tokenizer_id,
        batch.tokenizer_revision,
    )


def run_contract_canary(
    config: RunConfig,
    prompts: tuple[PromptRecord, ...],
    backends: BackendBundle,
    *,
    require_real_integration: bool,
    hardware: dict[str, Any] | None = None,
) -> CanaryReport:
    """Exercise transport contracts; this deliberately performs no optimizer step."""

    if len(prompts) != config.rollout.prompt_batch_size:
        raise PipelineContractError(
            "prompt count does not match rollout.prompt_batch_size"
        )
    real = bool(
        backends.student.real_model_integration
        and backends.teacher.real_model_integration
    )
    if require_real_integration and not real:
        raise PipelineContractError(
            "the Kaggle canary requires real student and teacher integrations"
        )

    request = RolloutRequest(
        run_id=config.run_id,
        step=0,
        prompts=prompts,
        samples_per_prompt=config.rollout.samples_per_prompt,
    )
    rollouts = backends.student.rollout(request)
    if rollouts.contract_version != INTERFACE_CONTRACT_VERSION:
        raise PipelineContractError("student returned an unsupported contract")
    if (rollouts.run_id, rollouts.step) != (config.run_id, 0):
        raise PipelineContractError("student returned a different run coordinate")
    expected_student = (
        config.student.model_id,
        config.student.model_revision,
        config.student.tokenizer_id,
        config.student.tokenizer_revision,
    )
    if _model_tuple(rollouts) != expected_student:
        raise PipelineContractError("student model/tokenizer provenance mismatch")
    expected_samples = (
        config.rollout.prompt_batch_size * config.rollout.samples_per_prompt
    )
    if len(rollouts.samples) != expected_samples:
        raise PipelineContractError("student returned the wrong sample count")

    scores = backends.teacher.score(
        TeacherScoreRequest(rollouts=rollouts, prompts=prompts)
    )
    if (scores.run_id, scores.step) != (config.run_id, 0):
        raise PipelineContractError("teacher returned a different run coordinate")
    expected_teacher = (
        config.teacher.model_id,
        config.teacher.model_revision,
        config.teacher.tokenizer_id,
        config.teacher.tokenizer_revision,
    )
    if _model_tuple(scores) != expected_teacher:
        raise PipelineContractError("teacher model/tokenizer provenance mismatch")

    rollout_by_id = {sample.sample_id: sample for sample in rollouts.samples}
    score_by_id = {sample.sample_id: sample for sample in scores.samples}
    if rollout_by_id.keys() != score_by_id.keys():
        raise PipelineContractError("teacher score ids do not match rollout ids")

    cross_tokenization_observed = False
    for sample_id, rollout in rollout_by_id.items():
        score = score_by_id[sample_id]
        if rollout.prompt_id != score.prompt_id:
            raise PipelineContractError("teacher changed a sample prompt id")
        if rollout.completion.text != score.completion.text:
            raise PipelineContractError(
                "teacher must score the exact student completion text"
            )
        if (
            rollout.completion.token_ids != score.completion.token_ids
            or rollout.completion.pieces != score.completion.pieces
        ):
            cross_tokenization_observed = True
    if not cross_tokenization_observed:
        raise PipelineContractError(
            "cross-tokenizer canary observed identical completion tokenization"
        )

    return CanaryReport(
        status="contract_passed",
        run_id=config.run_id,
        config_sha256=config.digest(),
        student_backend=backends.student.backend_name,
        teacher_backend=backends.teacher.backend_name,
        sample_count=len(rollouts.samples),
        student_completion_tokens=sum(
            len(sample.completion.token_ids) for sample in rollouts.samples
        ),
        teacher_completion_tokens=sum(
            len(sample.completion.token_ids) for sample in scores.samples
        ),
        cross_tokenization_observed=True,
        real_model_integration=real,
        hardware=hardware,
    )
