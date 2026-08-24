from __future__ import annotations

import dataclasses

import pytest

from vdt_tunix.contracts import (
    BackendBundle,
    ContractError,
    PromptRecord,
    TeacherScoreBatch,
    TokenSequence,
)
from vdt_tunix.mock import MockStudentRolloutBackend, MockTeacherScoreBackend
from vdt_tunix.pipeline import PipelineContractError, run_contract_canary


def _prompt(run_config):
    return PromptRecord(
        prompt_id=run_config.canary.prompt_id,
        student_prompt=run_config.canary.student_prompt,
        teacher_prompt=run_config.canary.teacher_prompt,
    )


def test_cpu_mocks_exercise_cross_tokenizer_transport(run_config):
    bundle = BackendBundle(
        student=MockStudentRolloutBackend(run_config),
        teacher=MockTeacherScoreBackend(run_config),
    )
    report = run_contract_canary(
        run_config,
        (_prompt(run_config),),
        bundle,
        require_real_integration=False,
    )

    assert report.status == "contract_passed"
    assert report.sample_count == 2
    assert report.student_completion_tokens > report.teacher_completion_tokens
    assert report.cross_tokenization_observed is True
    assert report.real_model_integration is False
    assert report.simct_update_executed is False
    assert report.scientific_evidence is False


def test_utf8_piece_contract_fails_closed_on_changed_text():
    with pytest.raises(ContractError, match="do not reproduce"):
        TokenSequence(text="xin chào", token_ids=(1, 2), pieces=("xin ", "chau"))


def test_real_canary_refuses_mock_backends(run_config):
    bundle = BackendBundle(
        student=MockStudentRolloutBackend(run_config),
        teacher=MockTeacherScoreBackend(run_config),
    )
    with pytest.raises(PipelineContractError, match="requires real"):
        run_contract_canary(
            run_config,
            (_prompt(run_config),),
            bundle,
            require_real_integration=True,
        )


def test_pipeline_rejects_teacher_sample_identity_drift(run_config):
    class WrongTeacher(MockTeacherScoreBackend):
        def score(self, request):
            batch = super().score(request)
            first = dataclasses.replace(batch.samples[0], sample_id="wrong-id")
            return TeacherScoreBatch(
                contract_version=batch.contract_version,
                run_id=batch.run_id,
                step=batch.step,
                model_id=batch.model_id,
                model_revision=batch.model_revision,
                tokenizer_id=batch.tokenizer_id,
                tokenizer_revision=batch.tokenizer_revision,
                samples=(first, *batch.samples[1:]),
            )

    bundle = BackendBundle(
        student=MockStudentRolloutBackend(run_config),
        teacher=WrongTeacher(run_config),
    )
    with pytest.raises(PipelineContractError, match="ids do not match"):
        run_contract_canary(
            run_config,
            (_prompt(run_config),),
            bundle,
            require_real_integration=False,
        )
