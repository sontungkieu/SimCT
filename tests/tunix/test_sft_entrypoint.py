from __future__ import annotations

import copy
import hashlib
import json
from types import SimpleNamespace

from scripts.tpu import kaggle_v5e8_sft as sft_entrypoint
from vdt_tunix.checkpoint import DataCursor
from vdt_tunix.sft_trainer import SFTUpdateMetrics
from vdt_tunix.tunix_checkpoint import ResumeState


def _sft_manifest(tmp_path):
    records = tmp_path / "sft.jsonl"
    records.write_text(
        "".join(
            json.dumps(
                {
                    "prompt_id": f"p-{index}",
                    "student_prompt": f"student {index}",
                    "teacher_prompt": f"teacher {index}",
                    "target_response": f"answer {index}",
                    "source": "fixture",
                    "source_id": str(index),
                    "source_license": "MIT",
                }
            )
            + "\n"
            for index in range(2)
        ),
        encoding="utf-8",
    )
    manifest = tmp_path / "sft_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "contract_version": 1,
                "dataset_id": "simct/sft-fixture",
                "dataset_revision": "fixture-v1",
                "split": "train",
                "records_path": records.name,
                "records_sha256": hashlib.sha256(records.read_bytes()).hexdigest(),
                "record_count": 2,
            }
        ),
        encoding="utf-8",
    )
    return manifest


def test_sft_entrypoint_trains_and_saves(tmp_path, config_payload, monkeypatch):
    payload = copy.deepcopy(config_payload)
    payload["rollout"]["samples_per_prompt"] = 1
    payload["training"].update(
        {"max_steps": 2, "gradient_accumulation_steps": 1}
    )
    config = tmp_path / "config.json"
    config.write_text(json.dumps(payload), encoding="utf-8")
    manifest = _sft_manifest(tmp_path)
    output = tmp_path / "summary.json"
    metrics = tmp_path / "metrics.jsonl"
    saves = []

    class FakeTrainer:
        def __init__(self, run_config, backends):
            del run_config, backends
            self.loaded_student = SimpleNamespace(model="model")
            self.optimizer = "optimizer"

        def step(self, rows, *, step):
            assert len(rows) == 1
            return SFTUpdateMetrics(
                loss=1.0 / (step + 1),
                gradient_norm=2.0,
                parameter_norm=3.0,
                sample_count=1,
                target_tokens=4,
            )

    class FakeController:
        def __init__(self, *args, **kwargs):
            del args, kwargs

        def restore_if_requested(self):
            return ResumeState(
                completed_steps=0,
                data_cursor=DataCursor(epoch=0, next_prompt_index=0),
                rng_state=(),
            )

        def save(self, **kwargs):
            saves.append(kwargs)

        def close(self):
            pass

    monkeypatch.setattr(sft_entrypoint, "load_real_backend_bundle", lambda cfg: object())
    monkeypatch.setattr(
        sft_entrypoint,
        "require_tpu_v5e8",
        lambda **kwargs: (object(), {"device_count": kwargs["expected_device_count"]}),
    )
    monkeypatch.setattr(sft_entrypoint, "TunixSFTTrainer", FakeTrainer)
    monkeypatch.setattr(sft_entrypoint, "TunixCheckpointController", FakeController)
    result = sft_entrypoint.main(
        [
            "--config",
            str(config),
            "--dataset-manifest",
            str(manifest),
            "--output",
            str(output),
            "--metrics",
            str(metrics),
        ]
    )
    assert result == 0
    summary = json.loads(output.read_text(encoding="utf-8"))
    assert summary["completed_steps"] == 2
    assert summary["scientific_evidence"] is False
    assert [item["completed_steps"] for item in saves] == [1, 2]
