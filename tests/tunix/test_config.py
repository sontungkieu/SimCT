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
