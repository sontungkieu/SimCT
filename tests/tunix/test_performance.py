from __future__ import annotations

import copy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from scripts.tpu.build_performance_canary_matrix import build_matrix
from vdt_tunix.config import RunConfig
from vdt_tunix.performance import (
    PerformanceContractError,
    jit_cache_size,
    numeric_shape_signature,
    select_length_bucket,
)


def test_length_buckets_are_fail_closed():
    assert select_length_bucket(129, (128, 256, 512)) == 256
    assert select_length_bucket(129, ()) == 129
    with pytest.raises(PerformanceContractError, match="exceeds largest bucket"):
        select_length_bucket(513, (128, 256, 512))


def test_shape_signature_is_stable_and_wandb_safe():
    first = numeric_shape_signature(batch=1, teacher=256)
    assert first == numeric_shape_signature(teacher=256, batch=1)
    assert first != numeric_shape_signature(batch=2, teacher=256)
    assert 0 <= first < 2**52


def test_jit_cache_size_is_best_effort():
    class Observable:
        def _cache_size(self):
            return 3

    assert jit_cache_size(Observable()) == 3
    assert jit_cache_size(object()) == -1


def test_performance_matrix_is_two_separate_length_ladders(config_payload):
    payload = copy.deepcopy(config_payload)
    payload["simct"].update(
        {
            "algorithm": "simple_opd",
            "virtual_support": "shared_tokens_only",
            "reproduction_mode": "paper_math",
            "span_gh_mask_threshold": 0.0,
        }
    )
    matrix = build_matrix(RunConfig.from_mapping(payload))
    assert list(matrix) == [
        "paper4k-fsdp8-b1",
        "paper4k-fsdp8-b2",
        "paper4k-fsdp8-b4",
        "paper4k-fsdp8-b8",
        "public8k-fsdp8-b1",
        "public8k-fsdp8-b2",
        "public8k-fsdp8-b4",
        "public8k-fsdp8-b8",
    ]
    for name, configured in matrix.items():
        batch = configured["rollout"]["prompt_batch_size"]
        assert configured["training"]["max_steps"] == 1
        assert configured["training"]["max_steps_unit"] == "optimizer_update"
        assert configured["training"]["gradient_accumulation_steps"] == 64 // batch
        assert configured["rollout"]["samples_per_prompt"] == 1
        assert configured["rollout"]["force_max_completion"] is True
        assert configured["tpu"] == {
            "accelerator_type": "v5e-8",
            "expected_device_count": 8,
            "tensor_parallelism": 1,
            "pipeline_parallelism": 1,
            "fsdp_parallelism": 8,
        }
        expected_length = 4096 if name.startswith("paper4k") else 8192
        assert configured["rollout"]["max_sequence_tokens"] == expected_length
        expected_completion = 3840 if name.startswith("paper4k") else 4096
        assert configured["rollout"]["max_completion_tokens"] == expected_completion
        assert (
            configured["rollout"]["max_prompt_tokens"] + expected_completion
            == expected_length
        )
        assert (
            configured["training"]["teacher_scoring_mode"]
            == "cached_teacher_forcing"
        )


def test_checked_in_performance_matrix_matches_builder():
    repo = Path(__file__).resolve().parents[2]
    baseline_payload = json.loads(
        (
            repo
            / "configs/reproduction/qwen25_7b_to_gemma2_2b_public_simple_opd_screen.json"
        ).read_text(encoding="utf-8")
    )
    expected = build_matrix(RunConfig.from_mapping(baseline_payload))
    for name, payload in expected.items():
        checked_in = json.loads(
            (repo / "configs/performance" / f"{name}.json").read_text(
                encoding="utf-8"
            )
        )
        assert checked_in == payload


def test_resource_probe_dataset_is_fixed_and_large_enough(tmp_path):
    output = tmp_path / "resource-probes"
    subprocess.run(
        [
            sys.executable,
            "scripts/tpu/build_resource_probe_dataset.py",
            "--output-dir",
            str(output),
        ],
        check=True,
    )
    for protocol in ("paper4k", "public8k"):
        manifest = json.loads(
            (output / protocol / "manifest.json").read_text(encoding="utf-8")
        )
        records = (output / protocol / manifest["records_path"]).read_text(
            encoding="utf-8"
        )
        assert manifest["record_count"] == 64
        assert len(records.splitlines()) == 64
    public_row = json.loads(
        (output / "public8k" / "records.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()[0]
    )
    assert public_row["student_prompt"].count(" probe") == 4000
