from __future__ import annotations

import copy
from pathlib import Path

import pytest

from vdt_tunix.config import RunConfig


@pytest.fixture
def config_payload(tmp_path: Path) -> dict:
    return {
        "contract_version": 1,
        "run_id": "cpu-contract-test",
        "student": {
            "model_id": "example/student",
            "model_revision": "student-immutable-revision",
            "tokenizer_id": "example/student-tokenizer",
            "tokenizer_revision": "student-tokenizer-immutable-revision",
            "maxtext_checkpoint_uri": str(tmp_path / "student" / "items"),
        },
        "teacher": {
            "model_id": "example/teacher",
            "model_revision": "teacher-immutable-revision",
            "tokenizer_id": "example/teacher-tokenizer",
            "tokenizer_revision": "teacher-tokenizer-immutable-revision",
            "maxtext_checkpoint_uri": str(tmp_path / "teacher" / "items"),
        },
        "simct": {
            "algorithm": "simct",
            "divergence": "reverse_kl",
            "alignment_unit": "utf8_bytes",
            "virtual_support": "shared_tokens_plus_realized_spans",
            "temperature": 1.0,
            "span_gh_mask_threshold": 2.0,
        },
        "rollout": {
            "prompt_batch_size": 1,
            "samples_per_prompt": 2,
            "max_prompt_tokens": 32,
            "max_completion_tokens": 16,
            "temperature": 0.0,
            "top_p": 1.0,
        },
        "training": {
            "max_steps": 4,
            "micro_batch_size": 1,
            "gradient_accumulation_steps": 2,
            "learning_rate": 1e-5,
        },
        "tpu": {
            "accelerator_type": "v5e-8",
            "expected_device_count": 8,
            "tensor_parallelism": 8,
            "pipeline_parallelism": 1,
        },
        "checkpoint": {
            "root": str(tmp_path / "checkpoints"),
            "save_every_steps": 1,
            "resume_from": None,
        },
        "canary": {
            "prompt_id": "prompt-0",
            "student_prompt": "Compute 2 + 2.",
            "teacher_prompt": "Compute 2 + 2.",
        },
    }


@pytest.fixture
def run_config(config_payload: dict) -> RunConfig:
    return RunConfig.from_mapping(copy.deepcopy(config_payload))
