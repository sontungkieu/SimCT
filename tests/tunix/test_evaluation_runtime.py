from __future__ import annotations

import contextlib
from types import SimpleNamespace

import pytest

from vdt_tunix.checkpoint import DataCursor
from vdt_tunix.evaluation_runtime import (
    EvaluationRuntimeError,
    NativeTunixGenerator,
    RestoredStudent,
    restore_student_for_inference,
)
from vdt_tunix.real_backend import _LoadedTunixModel
from vdt_tunix.tunix_checkpoint import ResumeState


def test_restore_student_uses_model_only_verified_coordinate(
    config_payload, tmp_path, monkeypatch
):
    from vdt_tunix import evaluation_runtime as runtime
    from vdt_tunix.config import RunConfig

    config = RunConfig.from_mapping(config_payload)
    root = tmp_path / "checkpoints"
    root.mkdir()
    state = SimpleNamespace(
        completed_steps=10,
        run_id=config.run_id,
        student_parameters=SimpleNamespace(sha256="a" * 64),
        dataset_manifest_sha256="b" * 64,
    )
    loaded = _LoadedTunixModel(
        model="model",
        mesh="mesh",
        forward_fn="forward",
        model_config=SimpleNamespace(num_layers=2, num_kv_heads=1, head_dim=4),
    )
    tokenizer = SimpleNamespace(raw_tokenizer="raw-tokenizer")
    monkeypatch.setattr(runtime, "load_latest_checkpoint", lambda *a, **k: state)
    monkeypatch.setattr(
        runtime,
        "load_native_student",
        lambda cfg: (loaded, tokenizer),
    )
    monkeypatch.setattr(
        runtime,
        "require_tpu_v5e8",
        lambda **kwargs: (object(), {"device_count": kwargs["expected_device_count"]}),
    )
    seen = {}

    class FakeController:
        def __init__(self, run_config, model, optimizer, **kwargs):
            seen["config"] = run_config
            seen["model"] = model
            seen["optimizer"] = optimizer
            seen.update(kwargs)

        def initialize_or_resume(self):
            return ResumeState(
                completed_steps=0,
                data_cursor=DataCursor(epoch=0, next_prompt_index=0),
                rng_state=(),
                initialization="warm_start",
                source_checkpoint_steps=10,
                source_checkpoint_run_id=config.run_id,
                source_student_parameters_sha256="a" * 64,
                source_dataset_manifest_sha256="b" * 64,
            )

        def close(self):
            seen["closed"] = True

    monkeypatch.setattr(runtime, "TunixCheckpointController", FakeController)
    restored = restore_student_for_inference(config, root)
    assert restored.model == "model"
    assert restored.tokenizer == "raw-tokenizer"
    assert restored.checkpoint_steps == 10
    assert restored.student_parameters_sha256 == "a" * 64
    assert seen["optimizer"] is None
    assert seen["config"].checkpoint.resume_from is None
    assert seen["config"].checkpoint.warm_start_from == str(root.resolve())
    assert seen["closed"] is True


def test_native_generator_checks_prompt_capacity_and_returns_lengths(
    monkeypatch
):

    class CacheConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class Sampler:
        def __init__(self, transformer, tokenizer, cache_config):
            self.transformer = transformer
            self.tokenizer = tokenizer
            self.cache_config = cache_config

        def tokenize(self, prompt):
            return list(range(len(prompt)))

        def __call__(self, prompts, **kwargs):
            assert kwargs["top_p"] == 0.95
            return SimpleNamespace(
                text=[f"answer-{index}" for index in range(len(prompts))],
                tokens=[list(range(index + 1)) for index in range(len(prompts))],
            )

    sampler_module = SimpleNamespace(CacheConfig=CacheConfig, Sampler=Sampler)
    generate_module = SimpleNamespace(sampler=sampler_module)
    monkeypatch.setitem(
        __import__("sys").modules,
        "tunix",
        SimpleNamespace(generate=generate_module),
    )
    monkeypatch.setitem(__import__("sys").modules, "tunix.generate.sampler", sampler_module)
    monkeypatch.setitem(
        __import__("sys").modules,
        "tunix.generate",
        generate_module,
    )
    monkeypatch.setitem(
        __import__("sys").modules,
        "jax",
        SimpleNamespace(set_mesh=lambda mesh: contextlib.nullcontext()),
    )
    restored = RestoredStudent(
        model="model",
        tokenizer="tokenizer",
        mesh="mesh",
        model_config=SimpleNamespace(num_layers=2, num_kv_heads=1, head_dim=4),
        checkpoint_steps=10,
        checkpoint_run_id="run",
        student_parameters_sha256="a" * 64,
        dataset_manifest_sha256="b" * 64,
        hardware={"device_count": 8},
    )
    generator = NativeTunixGenerator(
        restored, max_prompt_tokens=8, max_completion_tokens=16
    )
    texts, prompt_lengths, completion_lengths = generator.generate(
        ["abc", "defg"],
        max_new_tokens=8,
        temperature=0.6,
        top_p=0.95,
        seed=42,
    )
    assert texts == ("answer-0", "answer-1")
    assert prompt_lengths == (3, 4)
    assert completion_lengths == (1, 2)
    with pytest.raises(EvaluationRuntimeError, match="exceeds"):
        generator.prompt_lengths(["too-long-prompt"])
