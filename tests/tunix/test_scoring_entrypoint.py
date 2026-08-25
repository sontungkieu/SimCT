from __future__ import annotations

import json
from pathlib import Path

from scripts.evaluation import score_generated_predictions as scoring_entrypoint


REPO_ROOT = Path(__file__).resolve().parents[2]


class _FakeWandbRun:
    def __init__(self):
        self.logs = []
        self.finished = False

    def log_metrics(self, values, *, step, namespace="train"):
        self.logs.append((dict(values), step, namespace))

    def finish(self, *, training_status, exit_code):
        assert training_status == "complete"
        assert exit_code == 0
        self.finished = True

    def summary(self):
        return {
            "provider": "wandb",
            "requested": True,
            "fail_open": True,
            "status": "finished" if self.finished else "active",
            "reason": "training_finished" if self.finished else "online_run_started",
            "run_url": "https://wandb.ai/entity/project/runs/scoring",
            "logged_steps": len(self.logs),
            "project": "project",
            "run_name": "scoring",
            "group": "multiseed",
            "error_type": "",
            "error": "",
        }


def test_scoring_entrypoint_logs_each_terminal_benchmark(tmp_path, monkeypatch):
    generation = tmp_path / "generation"
    output = tmp_path / "scoring"
    generation.mkdir()
    (generation / "generation_summary.json").write_text(
        json.dumps(
            {
                "variant": "simct",
                "checkpoint_run_id": "vdt-public-simct-seed43",
                "training_config_sha256": "a" * 64,
                "student_parameters_sha256": "b" * 64,
            }
        ),
        encoding="utf-8",
    )
    benchmarks = [
        {
            "benchmark": name,
            "score": 0.5,
            "correct": 1,
            "total": 2,
            "empty_output_count": 0,
            "extraction_failed_count": 0,
        }
        for name in ("gsm8k", "math500", "mbpp", "live-code-bench-v6")
    ]
    monkeypatch.setattr(
        scoring_entrypoint,
        "score_generation_root",
        lambda **kwargs: {
            "status": "complete",
            "phase": "paper_released_evaluator_scoring",
            "variant": "simct",
            "benchmarks": benchmarks,
        },
    )
    logger = _FakeWandbRun()
    monkeypatch.setattr(
        scoring_entrypoint, "start_wandb_run", lambda **kwargs: logger
    )
    result = scoring_entrypoint.main(
        [
            "--generation-root",
            str(generation),
            "--evaluation-root",
            str(tmp_path / "evaluation"),
            "--generation-protocol",
            str(
                REPO_ROOT
                / "configs/evaluation/simct_paper_one_seed_generation.json"
            ),
            "--evaluator-source",
            str(tmp_path / "evaluation.py"),
            "--output-dir",
            str(output),
        ]
    )
    assert result == 0
    assert logger.finished is True
    assert [item[1] for item in logger.logs] == [1, 2, 3, 4]
    assert logger.logs[0][2] == "evaluation/gsm8k"
    summary = json.loads((output / "scoring_summary.json").read_text())
    assert summary["observability"]["status"] == "finished"
    assert summary["observability"]["logged_steps"] == 4
