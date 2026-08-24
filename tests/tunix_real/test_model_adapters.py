from __future__ import annotations

import dataclasses
from types import SimpleNamespace
from typing import ClassVar

import numpy as np
import pytest

from vdt_tunix.contracts import (
    PromptRecord,
    RolloutRequest,
    TeacherScoreRequest,
)
from vdt_tunix.model_adapters import (
    ModelAdapterError,
    ModelRuntimeDependencies,
    TokenizerByteAdapter,
)
from vdt_tunix.pipeline import run_contract_canary
from vdt_tunix.real_backend import (
    RealBackendUnavailable,
    _LoadedTunixModel,
    build_backends,
    load_native_student,
)


class PieceTokenizer:
    pad_token_id = 0
    eos_token_id = 1
    all_special_ids: ClassVar[list[int]] = [0, 1]

    def __init__(self, pieces):
        self._id_to_piece = {index + 2: piece for index, piece in enumerate(pieces)}
        self._piece_to_id = {piece: index for index, piece in self._id_to_piece.items()}

    @property
    def vocab_size(self):
        return max(self._id_to_piece) + 1

    def id_for(self, piece):
        return self._piece_to_id[piece]

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        result = []
        cursor = 0
        ordered = sorted(self._piece_to_id, key=len, reverse=True)
        while cursor < len(text):
            piece = next(
                (item for item in ordered if text.startswith(item, cursor)),
                None,
            )
            if piece is None:
                raise ValueError(f"no fake token piece at {text[cursor:]!r}")
            result.append(self._piece_to_id[piece])
            cursor += len(piece)
        return result

    def decode(self, token_ids, **kwargs):
        del kwargs
        return "".join(self._id_to_piece.get(int(value), "") for value in token_ids)


def _fake_dependencies(student_tokenizer, teacher_tokenizer):
    load_calls = []
    stop_gradient_calls = []

    def load_tokenizer(config):
        if "student" in config.tokenizer_id:
            return student_tokenizer
        return teacher_tokenizer

    def load_model(config, trainable):
        load_calls.append((config.model_id, trainable))
        tokenizer = student_tokenizer if trainable else teacher_tokenizer
        return {"trainable": trainable, "tokenizer": tokenizer}

    def forward_model(model, input_ids, segment_ids):
        ids = np.asarray(input_ids)
        segments = np.asarray(segment_ids)
        tokenizer = model["tokenizer"]
        logits = np.full(
            (*ids.shape, tokenizer.vocab_size),
            -5.0,
            dtype=np.float32,
        )
        if model["trainable"]:
            desired = [
                tokenizer.id_for("ha"),
                tokenizer.id_for("pp"),
                tokenizer.id_for("y"),
                tokenizer.eos_token_id,
            ]
            for row in range(ids.shape[0]):
                active = int(segments[row].sum())
                generated = active - 1  # the fake prompt is the single piece P:
                next_id = desired[min(generated, len(desired) - 1)]
                logits[row, active - 1, next_id] = 8.0
        else:
            # Teacher scoring needs full-vocabulary causal rows, not a sampled
            # continuation.  Non-constant rows also check payload slicing.
            for row in range(ids.shape[0]):
                for position in range(ids.shape[1]):
                    logits[row, position] += np.arange(tokenizer.vocab_size) * 0.1
        return logits

    def stop_gradient(value):
        stop_gradient_calls.append(True)
        return value

    dependencies = ModelRuntimeDependencies(
        name="cpu-fake-model-runtime",
        production=False,
        validate_model_spec=lambda config: None,
        load_tokenizer=load_tokenizer,
        load_model=load_model,
        forward_model=forward_model,
        stop_gradient=stop_gradient,
        to_host=np.asarray,
    )
    return dependencies, load_calls, stop_gradient_calls


def test_dependency_injected_adapters_exercise_real_backend_contract(real_config):
    student_tokenizer = PieceTokenizer(["P:", "ha", "pp", "y"])
    teacher_tokenizer = PieceTokenizer(["P:", "hap", "py"])
    dependencies, load_calls, stop_gradient_calls = _fake_dependencies(
        student_tokenizer,
        teacher_tokenizer,
    )
    bundle = build_backends(real_config, dependencies=dependencies)

    assert bundle.student.model_adapter.trainable is True
    assert bundle.teacher.model_adapter.trainable is False
    assert bundle.student.model_adapter.model_loaded is False
    assert bundle.teacher.model_adapter.model_loaded is False
    assert bundle.student.real_model_integration is False
    assert bundle.teacher.real_model_integration is False

    report = run_contract_canary(
        real_config,
        (
            PromptRecord(
                prompt_id=real_config.canary.prompt_id,
                student_prompt=real_config.canary.student_prompt,
                teacher_prompt=real_config.canary.teacher_prompt,
            ),
        ),
        bundle,
        require_real_integration=False,
    )

    assert report.status == "contract_passed"
    assert report.cross_tokenization_observed is True
    assert report.real_model_integration is False
    assert load_calls == [
        (real_config.student.model_id, True),
        (real_config.teacher.model_id, False),
    ]
    assert stop_gradient_calls

    prompts = (
        PromptRecord(
            prompt_id=real_config.canary.prompt_id,
            student_prompt=real_config.canary.student_prompt,
            teacher_prompt=real_config.canary.teacher_prompt,
        ),
    )
    rollouts = bundle.student.rollout(
        RolloutRequest(
            run_id=real_config.run_id,
            step=1,
            prompts=prompts,
            samples_per_prompt=real_config.rollout.samples_per_prompt,
        )
    )
    scores = bundle.teacher.score(
        TeacherScoreRequest(rollouts=rollouts, prompts=prompts)
    )
    assert rollouts.samples[0].completion.text == "happy"
    assert rollouts.samples[0].completion.pieces == (b"ha", b"pp", b"y")
    assert scores.samples[0].completion.text == "happy"
    assert scores.samples[0].completion.pieces == (b"hap", b"py")


def test_teacher_tokenization_without_prompt_boundary_fails_closed(real_config):
    tokenizer = PieceTokenizer(["P:h", "appy"])
    adapter = TokenizerByteAdapter(tokenizer, real_config.teacher)
    with pytest.raises(ModelAdapterError, match="no exact prompt/completion"):
        adapter.tokenize_continuation(prompt_text="P:", completion_text="happy")


def test_builder_rejects_missing_local_checkpoint(real_config):
    student_tokenizer = PieceTokenizer(["P:", "ha", "pp", "y"])
    teacher_tokenizer = PieceTokenizer(["P:", "hap", "py"])
    dependencies, _, _ = _fake_dependencies(student_tokenizer, teacher_tokenizer)
    missing = dataclasses.replace(
        real_config.student,
        maxtext_checkpoint_uri="/definitely/missing/vdt-student/items",
    )
    config = dataclasses.replace(real_config, student=missing)
    with pytest.raises(RealBackendUnavailable, match="unavailable"):
        build_backends(config, dependencies=dependencies)


def test_builder_rejects_unimplemented_pipeline_layout(real_config):
    student_tokenizer = PieceTokenizer(["P:", "ha", "pp", "y"])
    teacher_tokenizer = PieceTokenizer(["P:", "hap", "py"])
    dependencies, _, _ = _fake_dependencies(student_tokenizer, teacher_tokenizer)
    tpu = dataclasses.replace(
        real_config.tpu,
        tensor_parallelism=4,
        pipeline_parallelism=2,
    )
    config = dataclasses.replace(real_config, tpu=tpu)
    with pytest.raises(RealBackendUnavailable, match="PP1"):
        build_backends(config, dependencies=dependencies)


def test_student_only_loader_never_materializes_teacher(real_config):
    tokenizer = PieceTokenizer(["P:", "ha", "pp", "y"])
    validate_calls = []
    tokenizer_calls = []
    model_calls = []

    def validate_model_spec(config):
        validate_calls.append(config.model_id)

    def load_tokenizer(config):
        tokenizer_calls.append(config.model_id)
        return tokenizer

    def load_model(config, trainable):
        model_calls.append((config.model_id, trainable))
        return _LoadedTunixModel(
            model="student-model",
            mesh="student-mesh",
            forward_fn="student-forward",
            model_config=SimpleNamespace(
                num_layers=2,
                num_kv_heads=1,
                head_dim=4,
            ),
        )

    dependencies = ModelRuntimeDependencies(
        name="student-only-cpu-fake",
        production=False,
        validate_model_spec=validate_model_spec,
        load_tokenizer=load_tokenizer,
        load_model=load_model,
        forward_model=lambda *args: None,
        stop_gradient=lambda value: value,
        to_host=np.asarray,
    )
    loaded, adapter = load_native_student(real_config, dependencies=dependencies)

    assert loaded.model == "student-model"
    assert adapter.raw_tokenizer is tokenizer
    assert validate_calls == [real_config.student.model_id]
    assert tokenizer_calls == [real_config.student.model_id]
    assert model_calls == [(real_config.student.model_id, True)]
