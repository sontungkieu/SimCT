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
