from __future__ import annotations

import copy

from vdt_tunix.config import RunConfig
from vdt_tunix.contracts import PromptRecord, RolloutRequest
from vdt_tunix.real_backend import _rollout_seed, _stable_seed


PROMPTS = (
    PromptRecord(
        prompt_id="p0", student_prompt="student", teacher_prompt="teacher"
    ),
)


def test_explicit_training_seed_decouples_rng_from_run_id(config_payload):
    first = copy.deepcopy(config_payload)
    first["training"]["seed"] = 43
    first["run_id"] = "seeded-run-a"
    second = copy.deepcopy(first)
    second["run_id"] = "seeded-run-b"
    request_a = RolloutRequest(
        run_id="seeded-run-a", step=2, prompts=PROMPTS, samples_per_prompt=1
    )
    request_b = RolloutRequest(
        run_id="seeded-run-b", step=2, prompts=PROMPTS, samples_per_prompt=1
    )
    seed_a = _rollout_seed(RunConfig.from_mapping(first), request_a, "sample")
    seed_b = _rollout_seed(RunConfig.from_mapping(second), request_b, "sample")
    assert seed_a == seed_b


def test_legacy_training_seed_preserves_run_id_rng(config_payload):
    config = RunConfig.from_mapping(copy.deepcopy(config_payload))
    request = RolloutRequest(
        run_id=config.run_id, step=2, prompts=PROMPTS, samples_per_prompt=1
    )
    assert _rollout_seed(config, request, "sample") == _stable_seed(
        config.run_id, 2, "sample"
    )
