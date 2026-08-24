from __future__ import annotations

import copy
import hashlib
import json
from types import SimpleNamespace

from scripts.tpu import kaggle_v5e8_train as train_entrypoint
from vdt_tunix.checkpoint import DataCursor
from vdt_tunix.trainer import UpdateMetrics
from vdt_tunix.tunix_checkpoint import ResumeState


def _dataset_manifest(tmp_path):
    records = tmp_path / "prompts.jsonl"
    records.write_text(
        "".join(
            json.dumps(
                {
                    "prompt_id": f"p-{index}",
                    "student_prompt": f"student {index}",
                    "teacher_prompt": f"teacher {index}",
                }
            )
            + "\n"
            for index in range(2)
        ),
        encoding="utf-8",
    )
    manifest = tmp_path / "dataset.json"
    manifest.write_text(
        json.dumps(
            {
                "contract_version": 1,
                "dataset_id": "simct/fixture",
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


def test_train_entrypoint_runs_steps_and_publishes_checkpoints(
    tmp_path, config_payload, monkeypatch
):
    payload = copy.deepcopy(config_payload)
    payload["simct"].update(
        {
            "reproduction_mode": "paper_math",
            "span_gh_mask_threshold": 0.0,
        }
    )
    payload["rollout"]["samples_per_prompt"] = 1
    payload["training"].update(
        {
            "max_steps": 2,
            "gradient_accumulation_steps": 1,
        }
    )
    config = tmp_path / "config.json"
    config.write_text(json.dumps(payload), encoding="utf-8")
    manifest = _dataset_manifest(tmp_path)
    output = tmp_path / "summary.json"
    metrics = tmp_path / "metrics.jsonl"
    saves = []

    class FakeTrainer:
        def __init__(self, run_config, backends):
            del run_config, backends
            self.loaded_student = SimpleNamespace(model="model")
            self.optimizer = "optimizer"

        def step(self, prompts, *, step):
            assert len(prompts) == 1
            return UpdateMetrics(
                loss=1.0 / (step + 1),
                gradient_norm=2.0,
                parameter_norm=3.0,
                sample_count=1,
                student_completion_tokens=2,
                teacher_completion_tokens=3,
                aligned_units=1,
                aligned_spans=1,
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

    monkeypatch.setattr(train_entrypoint, "load_real_backend_bundle", lambda cfg: object())
    monkeypatch.setattr(
        train_entrypoint,
        "require_tpu_v5e8",
        lambda **kwargs: (object(), {"device_count": kwargs["expected_device_count"]}),
    )
    monkeypatch.setattr(train_entrypoint, "PaperSimCTTrainer", FakeTrainer)
    monkeypatch.setattr(train_entrypoint, "TunixCheckpointController", FakeController)

    result = train_entrypoint.main(
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
    rows = [json.loads(line) for line in metrics.read_text(encoding="utf-8").splitlines()]
    assert [row["step"] for row in rows] == [1, 2]
    assert [item["completed_steps"] for item in saves] == [1, 2]
