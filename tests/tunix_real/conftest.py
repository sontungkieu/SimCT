from __future__ import annotations

from pathlib import Path

import pytest

from vdt_tunix.config import RunConfig


@pytest.fixture
def real_config(tmp_path: Path) -> RunConfig:
    student_checkpoint = tmp_path / "student" / "items"
    teacher_checkpoint = tmp_path / "teacher" / "items"
    student_checkpoint.mkdir(parents=True)
    teacher_checkpoint.mkdir(parents=True)
    return RunConfig.from_mapping(
        {
            "contract_version": 1,
            "run_id": "real-adapter-cpu-contract",
            "student": {
                "model_id": "Qwen/Qwen3-0.6B",
                "model_revision": "student-immutable-revision",
                "tokenizer_id": "local/student-tokenizer",
                "tokenizer_revision": "student-tokenizer-immutable-revision",
                "maxtext_checkpoint_uri": str(student_checkpoint),
            },
            "teacher": {
                "model_id": "Qwen/Qwen2.5-0.5B",
                "model_revision": "teacher-immutable-revision",
                "tokenizer_id": "local/teacher-tokenizer",
                "tokenizer_revision": "teacher-tokenizer-immutable-revision",
                "maxtext_checkpoint_uri": str(teacher_checkpoint),
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
                "max_prompt_tokens": 8,
                "max_completion_tokens": 4,
                "temperature": 0.0,
                "top_p": 1.0,
            },
            "training": {
                "max_steps": 2,
                "micro_batch_size": 1,
                "gradient_accumulation_steps": 1,
                "learning_rate": 1e-5,
            },
            "tpu": {
                "accelerator_type": "v5e-8",
                "expected_device_count": 8,
                "tensor_parallelism": 8,
                "pipeline_parallelism": 1,
            },
            "checkpoint": {
                "root": str(tmp_path / "run-checkpoints"),
                "save_every_steps": 1,
                "resume_from": None,
            },
            "canary": {
                "prompt_id": "prompt-0",
                "student_prompt": "P:",
                "teacher_prompt": "P:",
            },
        }
    )
