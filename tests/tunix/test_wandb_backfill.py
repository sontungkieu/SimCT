from __future__ import annotations

import json
from pathlib import Path

import pytest

from vdt_tunix.wandb_backfill import (
    WandbBackfillError,
    backfill_wandb_evidence,
)


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


class _FakeRun:
    def __init__(self, *, objective: str, kwargs: dict, captures: list[dict]):
        self.objective = objective
        self.kwargs = kwargs
        self.captures = captures
        self.logged_steps = 0

    def log_metrics(self, values, *, step, namespace="train"):
        self.logged_steps += 1
        self.captures.append(
            {
                "objective": self.objective,
                "values": dict(values),
                "step": step,
                "namespace": namespace,
            }
        )

    def finish(self, *, training_status, exit_code):
        assert training_status == "complete"
        assert exit_code == 0

    def summary(self):
        return {
            "provider": "wandb",
            "requested": True,
            "fail_open": True,
            "status": "finished",
            "reason": "training_finished",
            "run_url": f"https://wandb.ai/entity/project/runs/{self.objective}",
            "logged_steps": self.logged_steps,
            "project": self.kwargs["project"],
            "run_name": self.kwargs["run_name"],
            "group": self.kwargs["group"],
            "error_type": "",
            "error": "",
            "evidence_mode": self.kwargs["evidence_mode"],
            "source_artifact_sha256": self.kwargs[
                "source_artifact_sha256"
            ],
        }


def _fixture(tmp_path: Path):
    training = tmp_path / "train_summary.json"
    metrics = tmp_path / "train_metrics.jsonl"
    generation = tmp_path / "generation_summary.json"
    scoring = tmp_path / "scoring_summary.json"
    config = "a" * 64
    dataset = "b" * 64
    params = "c" * 64
    protocol = "d" * 64
    _write_json(
        training,
        {
            "status": "complete",
            "phase": "sft_training",
            "run_id": "fixture-sft",
            "completed_steps": 2,
            "config_sha256": config,
            "dataset_manifest_sha256": dataset,
            "final_student_parameters_sha256": params,
        },
    )
    metrics.write_text(
        "\n".join(
            json.dumps(
                {
                    "run_id": "fixture-sft",
                    "step": step,
                    "loss": float(step),
                }
            )
            for step in (1, 2)
        )
        + "\n",
        encoding="utf-8",
    )
    _write_json(
        generation,
        {
            "status": "complete",
            "variant": "sft",
            "checkpoint_run_id": "fixture-sft",
            "training_config_sha256": config,
            "student_parameters_sha256": params,
            "protocol_sha256": protocol,
            "benchmarks": [
                {"benchmark": "gsm8k", "batch_count": 2},
                {"benchmark": "mbpp", "batch_count": 1},
            ],
        },
    )
    _write_json(
        scoring,
        {
            "status": "complete",
            "variant": "sft",
            "checkpoint_sha256": params,
            "protocol_sha256": protocol,
            "benchmarks": [
                {
                    "benchmark": "gsm8k",
                    "score": 0.5,
                    "correct": 1,
                    "total": 2,
                },
                {
                    "benchmark": "mbpp",
                    "score": 1.0,
                    "correct": 2,
                    "total": 2,
                },
            ],
        },
    )
    return training, metrics, generation, scoring


def test_backfill_replays_exact_counts_without_mutating_sources(tmp_path):
    paths = _fixture(tmp_path)
    before = [path.read_bytes() for path in paths]
    captures = []

    def factory(**kwargs):
        return _FakeRun(
            objective=kwargs["objective"], kwargs=kwargs, captures=captures
        )

    result = backfill_wandb_evidence(
        training_summary_path=paths[0],
        training_metrics_path=paths[1],
        generation_summary_path=paths[2],
        scoring_summary_path=paths[3],
        project="project",
        group="multiseed",
        run_factory=factory,
    )

    assert result["status"] == "passed"
    assert result["mode"] == "historical_backfill"
    assert result["scientific_evidence"] is False
    assert [path.read_bytes() for path in paths] == before
    summaries = result["summaries"]
    assert summaries["training"]["observability"]["logged_steps"] == 2
    assert summaries["generation"]["observability"]["logged_steps"] == 3
    assert summaries["scoring"]["observability"]["logged_steps"] == 2
    assert all(
        summary["observability"]["evidence_mode"] == "backfill"
        for summary in summaries.values()
    )
    assert len(captures) == 7


def test_backfill_rejects_noncontiguous_training_metrics(tmp_path):
    paths = _fixture(tmp_path)
    paths[1].write_text(
        json.dumps({"run_id": "fixture-sft", "step": 2, "loss": 1.0})
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(WandbBackfillError, match="steps drifted"):
        backfill_wandb_evidence(
            training_summary_path=paths[0],
            training_metrics_path=paths[1],
            generation_summary_path=paths[2],
            scoring_summary_path=paths[3],
            project="project",
            group="multiseed",
        )
