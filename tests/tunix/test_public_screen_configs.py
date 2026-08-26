from __future__ import annotations

from pathlib import Path

import pytest

from vdt_tunix.config import load_config


REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    ("name", "algorithm", "learning_rate", "completion_tokens"),
    [
        ("sft", "simct", 2e-6, 1024),
        ("simple_opd", "simple_opd", 5e-7, 256),
        ("simct", "simct", 5e-7, 256),
    ],
)
def test_public_screen_configs_are_small_pinned_updates(
    name, algorithm, learning_rate, completion_tokens
):
    config = load_config(
        REPO_ROOT
        / "configs"
        / "reproduction"
        / f"qwen25_7b_to_gemma2_2b_public_{name}_screen.json"
    )
    assert config.run_id == f"vdt-public-{name}-screen"
    # ``model_id`` is the Tunix architecture key, not a Hugging Face repo ID.
    # These exact spellings route to ModelConfig.gemma2_2b_it and
    # ModelConfig.qwen2p5_7b_instruct in the pinned Tunix revision.
    assert config.student.model_id == "gemma-2-2b-it"
    assert config.teacher.model_id == "qwen2.5-7b-instruct"
    assert config.student.tokenizer_id == "google/gemma-2-2b-it"
    assert config.teacher.tokenizer_id == "Qwen/Qwen2.5-7B-Instruct"
    assert config.simct.algorithm == algorithm
    assert config.training.max_steps == 10
    assert config.training.gradient_accumulation_steps == 1
    assert config.training.micro_batch_size == 1
    assert config.training.learning_rate == learning_rate
    assert config.rollout.prompt_batch_size == 1
    assert config.rollout.max_completion_tokens == completion_tokens
    assert config.checkpoint.save_every_steps == 10
    if name == "sft":
        assert config.checkpoint.warm_start_from is None
    else:
        assert config.checkpoint.warm_start_from is not None


def test_paper_canary_uses_pinned_tunix_architecture_keys():
    config = load_config(
        REPO_ROOT
        / "configs"
        / "reproduction"
        / "qwen25_7b_to_gemma2_2b_paper_canary.json"
    )
    assert config.student.model_id == "gemma-2-2b-it"
    assert config.teacher.model_id == "qwen2.5-7b-instruct"


@pytest.mark.parametrize("algorithm", ["simple_opd", "simct"])
def test_paper4k_configs_freeze_table4_operational_contract(algorithm):
    config = load_config(
        REPO_ROOT
        / "configs"
        / "reproduction"
        / f"qwen25_7b_to_gemma2_2b_paper4k_{algorithm}.json"
    )
    assert config.simct.algorithm == algorithm
    assert config.rollout.max_sequence_tokens == 4096
    assert config.rollout.max_completion_tokens == 4096
    assert config.rollout.temperature == 0.6
    assert config.rollout.top_p == 0.95
    assert config.rollout.samples_per_prompt == 1
    assert config.training.max_steps == 314
    assert config.training.max_steps_unit == "optimizer_update"
    assert config.training.gradient_accumulation_steps == 64
    assert config.training.learning_rate == 1e-6


@pytest.mark.parametrize("algorithm", ["simple_opd", "simct"])
def test_public8k_configs_freeze_released_script_ablation(algorithm):
    config = load_config(
        REPO_ROOT
        / "configs"
        / "reproduction"
        / f"qwen25_7b_to_gemma2_2b_public8k_{algorithm}.json"
    )
    assert config.simct.algorithm == algorithm
    assert config.rollout.max_sequence_tokens == 8192
    assert config.rollout.max_completion_tokens == 4096
    assert config.rollout.temperature == 0.6
    assert config.rollout.top_p == 1.0
    assert config.training.max_steps == 157
    assert config.training.max_steps_unit == "optimizer_update"
    assert config.training.gradient_accumulation_steps == 64
    assert config.training.learning_rate == 5e-7
