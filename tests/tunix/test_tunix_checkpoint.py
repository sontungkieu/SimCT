from __future__ import annotations

import copy

import pytest

from vdt_tunix.checkpoint import DataCursor
from vdt_tunix.config import RunConfig
from vdt_tunix.tunix_checkpoint import (
    TunixCheckpointController,
    TunixCheckpointError,
    directory_sha256,
)


class FakeManager:
    def __init__(self, root, store):
        self.root = root
        self.store = store
        self.closed = False

    def save(self, step, model, optimizer, *, force, custom_metadata):
        assert force is True
        step_dir = self.root / str(step)
        step_dir.mkdir(parents=True, exist_ok=True)
        (step_dir / "model.bin").write_bytes(str(model).encode("utf-8"))
        (step_dir / "optimizer.bin").write_bytes(str(optimizer).encode("utf-8"))
        self.store[(str(self.root), step)] = dict(custom_metadata)
        return True

    def maybe_restore(self, model, optimizer, *, step):
        del model, optimizer
        return step, self.store[(str(self.root), step)]

    def close(self):
        self.closed = True


def _factory(store):
    def build(root, save_every_steps):
        assert save_every_steps == 1
        return FakeManager(root, store)

    return build


def test_directory_sha256_changes_with_content(tmp_path):
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    payload = artifact / "params"
    payload.write_bytes(b"one")
    first = directory_sha256(artifact)
    payload.write_bytes(b"two")
    assert directory_sha256(artifact) != first


def test_tunix_checkpoint_save_and_restore(config_payload):
    store = {}
    config = RunConfig.from_mapping(copy.deepcopy(config_payload))
    controller = TunixCheckpointController(
        config,
        model="student-state",
        optimizer="optimizer-state",
        dataset_manifest_sha256="d" * 64,
        manager_factory=_factory(store),
    )
    cursor = DataCursor(epoch=1, next_prompt_index=3)
    state = controller.save(
        completed_steps=2,
        data_cursor=cursor,
        rng_state={"rollout": "seed-2", "trainer": "seed-3"},
    )
    controller.close()
    assert state.completed_steps == 2
    assert state.dataset_manifest_sha256 == "d" * 64

    resume_payload = copy.deepcopy(config_payload)
    resume_payload["checkpoint"]["resume_from"] = resume_payload["checkpoint"]["root"]
    resumed_config = RunConfig.from_mapping(resume_payload)
    resumed = TunixCheckpointController(
        resumed_config,
        model="new-student",
        optimizer="new-optimizer",
        dataset_manifest_sha256="d" * 64,
        manager_factory=_factory(store),
    ).restore_if_requested()
    assert resumed.completed_steps == 2
    assert resumed.data_cursor == cursor
    assert dict(resumed.rng_state) == {
        "rollout": "seed-2",
        "trainer": "seed-3",
    }


def test_tunix_checkpoint_detects_artifact_tampering(config_payload):
    store = {}
    config = RunConfig.from_mapping(copy.deepcopy(config_payload))
    controller = TunixCheckpointController(
        config,
        model="student-state",
        optimizer="optimizer-state",
        dataset_manifest_sha256="d" * 64,
        manager_factory=_factory(store),
    )
    controller.save(
        completed_steps=1,
        data_cursor=DataCursor(epoch=0, next_prompt_index=1),
        rng_state={"rollout": "seed"},
    )
    controller.close()
    root = controller.root
    (root / "1" / "model.bin").write_bytes(b"tampered")

    resume_payload = copy.deepcopy(config_payload)
    resume_payload["checkpoint"]["resume_from"] = str(root)
    resumed = TunixCheckpointController(
        RunConfig.from_mapping(resume_payload),
        model="student",
        optimizer="optimizer",
        dataset_manifest_sha256="d" * 64,
        manager_factory=_factory(store),
    )
    with pytest.raises(TunixCheckpointError, match="artifact SHA-256 mismatch"):
        resumed.restore_if_requested()


def test_tunix_checkpoint_rejects_dataset_drift(config_payload):
    store = {}
    config = RunConfig.from_mapping(copy.deepcopy(config_payload))
    controller = TunixCheckpointController(
        config,
        model="student-state",
        optimizer="optimizer-state",
        dataset_manifest_sha256="d" * 64,
        manager_factory=_factory(store),
    )
    controller.save(
        completed_steps=1,
        data_cursor=DataCursor(epoch=0, next_prompt_index=1),
        rng_state={"rollout": "seed"},
    )
    controller.close()

    resume_payload = copy.deepcopy(config_payload)
    resume_payload["checkpoint"]["resume_from"] = str(controller.root)
    resumed = TunixCheckpointController(
        RunConfig.from_mapping(resume_payload),
        model="student",
        optimizer="optimizer",
        dataset_manifest_sha256="e" * 64,
        manager_factory=_factory(store),
    )
    with pytest.raises(TunixCheckpointError, match="dataset manifest"):
        resumed.restore_if_requested()


def test_tunix_checkpoint_model_only_warm_start_allows_new_phase(
    config_payload, tmp_path
):
    store = {}
    sft_config = RunConfig.from_mapping(copy.deepcopy(config_payload))
    sft = TunixCheckpointController(
        sft_config,
        model="sft-model",
        optimizer="sft-optimizer",
        dataset_manifest_sha256="d" * 64,
        manager_factory=_factory(store),
    )
    sft.save(
        completed_steps=4,
        data_cursor=DataCursor(epoch=2, next_prompt_index=0),
        rng_state={"trainer": "sft-4"},
    )
    sft.close()

    opd_payload = copy.deepcopy(config_payload)
    opd_payload["run_id"] = "opd-phase"
    opd_payload["training"]["learning_rate"] = 2e-6
    opd_payload["checkpoint"].update(
        {
            "root": str(tmp_path / "opd-checkpoints"),
            "resume_from": None,
            "warm_start_from": str(sft.root),
        }
    )
    opd = TunixCheckpointController(
        RunConfig.from_mapping(opd_payload),
        model="opd-model",
        optimizer="opd-optimizer",
        dataset_manifest_sha256="e" * 64,
        manager_factory=_factory(store),
    )
    initialized = opd.initialize_or_resume()
    assert initialized.initialization == "warm_start"
    assert initialized.completed_steps == 0
    assert initialized.source_checkpoint_steps == 4
    assert initialized.source_dataset_manifest_sha256 == "d" * 64
