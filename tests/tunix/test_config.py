from __future__ import annotations

import copy
import json

import pytest

from vdt_tunix.config import ConfigError, RunConfig, load_config


def test_config_round_trip_and_digest_are_stable(config_payload, tmp_path):
    config = RunConfig.from_mapping(copy.deepcopy(config_payload))
    reordered = dict(reversed(list(copy.deepcopy(config_payload).items())))
    assert RunConfig.from_mapping(reordered).digest() == config.digest()

    path = tmp_path / "config.json"
    path.write_text(json.dumps(reordered), encoding="utf-8")
    assert load_config(path) == config
    assert len(config.digest()) == 64


def test_explicit_training_seed_changes_identity_without_drifting_legacy(config_payload):
    legacy = RunConfig.from_mapping(copy.deepcopy(config_payload))
    seeded_payload = copy.deepcopy(config_payload)
    seeded_payload["training"]["seed"] = 43
    seeded = RunConfig.from_mapping(seeded_payload)
    assert legacy.training.seed is None
    assert seeded.training.seed == 43
    assert seeded.digest() != legacy.digest()

    negative = copy.deepcopy(config_payload)
    negative["training"]["seed"] = -1
    with pytest.raises(ConfigError, match="training.seed must be non-negative"):
        RunConfig.from_mapping(negative)


def test_performance_controls_are_opt_in_and_part_of_identity(config_payload):
    legacy = RunConfig.from_mapping(copy.deepcopy(config_payload))
    payload = copy.deepcopy(config_payload)
    payload["training"].update(
        {
            "teacher_sequence_buckets": [128, 256, 512],
            "student_sequence_buckets": [128, 256, 512],
            "student_completion_buckets": [64, 128, 256],
            "alignment_unit_buckets": [32, 64, 128],
            "alignment_bucket_size": 64,
            "synchronize_phase_timings": True,
        }
    )
    configured = RunConfig.from_mapping(payload)
    assert configured.training.teacher_sequence_buckets == (128, 256, 512)
    assert configured.training.student_sequence_buckets == (128, 256, 512)
    assert configured.training.student_completion_buckets == (64, 128, 256)
    assert configured.training.alignment_unit_buckets == (32, 64, 128)
    assert configured.training.alignment_bucket_size == 64
    assert configured.training.synchronize_phase_timings
    assert configured.digest() != legacy.digest()

    invalid = copy.deepcopy(payload)
    invalid["training"]["teacher_sequence_buckets"] = [256, 128]
    with pytest.raises(ConfigError, match="strictly increasing"):
        RunConfig.from_mapping(invalid)


def test_cached_teacher_scoring_is_explicit_and_part_of_identity(config_payload):
    legacy = RunConfig.from_mapping(copy.deepcopy(config_payload))
    payload = copy.deepcopy(config_payload)
    payload["training"]["teacher_scoring_mode"] = "cached_teacher_forcing"
    configured = RunConfig.from_mapping(payload)

    assert legacy.training.teacher_scoring_mode == "dense"
    assert configured.training.teacher_scoring_mode == "cached_teacher_forcing"
    assert configured.digest() != legacy.digest()

    invalid = copy.deepcopy(payload)
    invalid["training"]["teacher_scoring_mode"] = "approximate"
    with pytest.raises(ConfigError, match="dense or cached_teacher_forcing"):
        RunConfig.from_mapping(invalid)


def test_sequence_probe_and_optimizer_update_contracts_are_explicit(config_payload):
    legacy = RunConfig.from_mapping(copy.deepcopy(config_payload))
    payload = copy.deepcopy(config_payload)
    payload["rollout"].update(
        {
            "max_sequence_tokens": 4096,
            "force_max_completion": True,
            "minimum_actual_sequence_tokens": 3968,
        }
    )
    payload["training"].update(
        {
            "max_steps_unit": "optimizer_update",
            "gradient_accumulation_steps": 64,
        }
    )
    configured = RunConfig.from_mapping(payload)
    assert configured.rollout.max_sequence_tokens == 4096
    assert configured.rollout.force_max_completion
    assert configured.rollout.minimum_actual_sequence_tokens == 3968
    assert configured.training.max_steps_unit == "optimizer_update"
    assert configured.digest() != legacy.digest()

    invalid = copy.deepcopy(payload)
    invalid["rollout"]["minimum_actual_sequence_tokens"] = 4097
    with pytest.raises(ConfigError, match="cannot exceed max_sequence_tokens"):
        RunConfig.from_mapping(invalid)

    invalid = copy.deepcopy(payload)
    invalid["training"]["max_steps_unit"] = "micro_step"
    with pytest.raises(ConfigError, match="trainer_call or optimizer_update"):
        RunConfig.from_mapping(invalid)


def test_contract_is_single_teacher_and_rejects_unknown_teachers(config_payload):
    payload = copy.deepcopy(config_payload)
    payload["teachers"] = [payload["teacher"]]
    with pytest.raises(ConfigError, match="unsupported keys.*teachers"):
        RunConfig.from_mapping(payload)


def test_contract_rejects_same_tokenizer(config_payload):
    payload = copy.deepcopy(config_payload)
    payload["teacher"]["tokenizer_id"] = payload["student"]["tokenizer_id"]
    with pytest.raises(ConfigError, match="distinct student and teacher"):
        RunConfig.from_mapping(payload)


def test_contract_rejects_mutable_revision(config_payload):
    payload = copy.deepcopy(config_payload)
    payload["teacher"]["model_revision"] = "main"
    with pytest.raises(ConfigError, match="immutable revision"):
        RunConfig.from_mapping(payload)


def test_contract_rejects_non_v5e8_layout(config_payload):
    payload = copy.deepcopy(config_payload)
    payload["tpu"]["expected_device_count"] = 4
    with pytest.raises(ConfigError, match="must be 8"):
        RunConfig.from_mapping(payload)


def test_paper_math_mode_requires_post_paper_safeguard_disabled(config_payload):
    payload = copy.deepcopy(config_payload)
    payload["simct"]["reproduction_mode"] = "paper_math"
    with pytest.raises(ConfigError, match="requires span_gh_mask_threshold=0"):
        RunConfig.from_mapping(payload)
    payload["simct"]["span_gh_mask_threshold"] = 0
    assert RunConfig.from_mapping(payload).simct.reproduction_mode == "paper_math"


def test_simple_opd_requires_overlap_only_paper_contract(config_payload):
    payload = copy.deepcopy(config_payload)
    payload["simct"].update(
        {
            "algorithm": "simple_opd",
            "virtual_support": "shared_tokens_only",
            "reproduction_mode": "paper_math",
            "span_gh_mask_threshold": 0.0,
        }
    )
    config = RunConfig.from_mapping(payload)
    assert config.simct.algorithm == "simple_opd"

    wrong_support = copy.deepcopy(payload)
    wrong_support["simct"]["virtual_support"] = (
        "shared_tokens_plus_realized_spans"
    )
    with pytest.raises(ConfigError, match="shared_tokens_only"):
        RunConfig.from_mapping(wrong_support)

    wrong_mode = copy.deepcopy(payload)
    wrong_mode["simct"]["reproduction_mode"] = "public_code_cf0f33a_default"
    wrong_mode["simct"]["span_gh_mask_threshold"] = 2.0
    with pytest.raises(ConfigError, match="simple_opd currently requires"):
        RunConfig.from_mapping(wrong_mode)


def test_native_tunix_fsdp8_layout_and_source_are_explicit(config_payload):
    payload = copy.deepcopy(config_payload)
    for role in ("student", "teacher"):
        payload[role].update(
            {
                "model_source": "huggingface",
                "model_path": f"/kaggle/input/{role}-weights",
                "tokenizer_type": "huggingface",
                "tokenizer_path": f"/kaggle/input/{role}-weights",
            }
        )
    payload["tpu"].update(
        {
            "tensor_parallelism": 1,
            "pipeline_parallelism": 1,
            "fsdp_parallelism": 8,
        }
    )
    config = RunConfig.from_mapping(payload)
    assert config.student.resolved_model_path.endswith("student-weights")
    assert config.tpu.fsdp_parallelism == 8
