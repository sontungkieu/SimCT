from __future__ import annotations

import copy

import pytest

from vdt_tunix.checkpoint import (
    ArtifactRef,
    CheckpointError,
    CheckpointState,
    DataCursor,
    load_latest_checkpoint,
    save_checkpoint,
)
from vdt_tunix.config import RunConfig


def _state(run_config, *, completed_steps=1, suffix="a"):
    return CheckpointState.create(
        run_config,
        dataset_manifest_sha256="d" * 64,
        completed_steps=completed_steps,
        data_cursor=DataCursor(epoch=0, next_prompt_index=completed_steps * 2),
        rng_state={"student_rollout": f"rollout-{completed_steps}", "trainer": f"train-{completed_steps}"},
        student_parameters=ArtifactRef(
            uri=f"orbax://student/{completed_steps}", sha256=suffix * 64
        ),
        optimizer_state=ArtifactRef(
            uri=f"orbax://optimizer/{completed_steps}", sha256="b" * 64
        ),
    )


def test_checkpoint_round_trip_and_idempotent_save(run_config, tmp_path):
    root = tmp_path / "checkpoint"
    state = _state(run_config)
    manifest = save_checkpoint(root, state)
    assert manifest.name == "resume_manifest.json"
    assert save_checkpoint(root, state) == manifest
    assert load_latest_checkpoint(root, config=run_config) == state


def test_checkpoint_rejects_different_state_at_same_coordinate(run_config, tmp_path):
    root = tmp_path / "checkpoint"
    save_checkpoint(root, _state(run_config, suffix="a"))
    with pytest.raises(CheckpointError, match="different state"):
        save_checkpoint(root, _state(run_config, suffix="c"))


def test_checkpoint_detects_manifest_corruption(run_config, tmp_path):
    root = tmp_path / "checkpoint"
    manifest = save_checkpoint(root, _state(run_config))
    manifest.write_text(manifest.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(CheckpointError, match="SHA-256 mismatch"):
        load_latest_checkpoint(root, config=run_config)


def test_resume_rejects_config_drift(run_config, config_payload, tmp_path):
    root = tmp_path / "checkpoint"
    save_checkpoint(root, _state(run_config))
    changed = copy.deepcopy(config_payload)
    changed["training"]["learning_rate"] = 2e-5
    changed_config = RunConfig.from_mapping(changed)
    with pytest.raises(CheckpointError, match="resume identity mismatch"):
        load_latest_checkpoint(root, config=changed_config)


def test_checkpoint_identity_ignores_storage_and_resume_paths(config_payload):
    original = RunConfig.from_mapping(copy.deepcopy(config_payload))
    moved = copy.deepcopy(config_payload)
    moved["checkpoint"]["root"] = "/different/output/root"
    moved["checkpoint"]["resume_from"] = "/mounted/input/checkpoint"
    assert RunConfig.from_mapping(moved).digest() == original.digest()


def test_checkpoint_refuses_latest_rollback(run_config, tmp_path):
    root = tmp_path / "checkpoint"
    save_checkpoint(root, _state(run_config, completed_steps=2))
    with pytest.raises(CheckpointError, match="backwards"):
        save_checkpoint(root, _state(run_config, completed_steps=1))
