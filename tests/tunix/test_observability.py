from __future__ import annotations

import sys
from types import SimpleNamespace

from vdt_tunix.observability import start_wandb_run


def _start():
    return start_wandb_run(
        run_id="fixture-run",
        objective="simct",
        config_sha256="a" * 64,
        dataset_manifest_sha256="b" * 64,
        metadata={"max_steps": 2},
    )


def test_missing_or_staged_placeholder_key_disables_wandb(monkeypatch):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    assert _start().summary()["status"] == "disabled"
    monkeypatch.setenv("WANDB_API_KEY", "__KJO_SECRET_WANDB_API_KEY__")
    assert _start().summary()["status"] == "disabled"


def test_wandb_logs_numeric_metrics_and_finishes(monkeypatch):
    calls = []

    class FakeRun:
        def log(self, payload, *, step):
            calls.append(("log", payload, step))

        def finish(self, *, exit_code):
            calls.append(("finish", exit_code))

        def get_url(self):
            return "https://wandb.ai/fixture/project/runs/abc"

    fake_wandb = SimpleNamespace(
        Settings=lambda **kwargs: ("settings", kwargs),
        init=lambda **kwargs: calls.append(("init", kwargs)) or FakeRun(),
    )
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    monkeypatch.setenv("WANDB_API_KEY", "fixture-secret")
    logger = _start()
    logger.log_metrics(
        {"loss": 1.5, "count": 2, "name": "ignored", "bad": float("nan")},
        step=1,
    )
    logger.finish(training_status="complete", exit_code=0)

    assert logger.summary()["status"] == "finished"
    assert logger.summary()["logged_steps"] == 1
    assert logger.summary()["run_url"].endswith("/abc")
    init = next(item for item in calls if item[0] == "init")[1]
    assert "WANDB_API_KEY" not in init["config"]
    assert init["config"]["evidence_mode"] == "native"
    assert init["config"]["source_artifact_sha256"] == ""
    logged = next(item for item in calls if item[0] == "log")
    assert logged[1] == {
        "train/loss": 1.5,
        "train/count": 2,
        "trainer/global_step": 1,
    }


def test_backfill_metadata_is_explicit(monkeypatch):
    calls = []

    class FakeRun:
        def finish(self, *, exit_code):
            calls.append(("finish", exit_code))

        def get_url(self):
            return "https://wandb.ai/fixture/project/runs/backfill"

    monkeypatch.setitem(
        sys.modules,
        "wandb",
        SimpleNamespace(
            Settings=lambda **kwargs: kwargs,
            init=lambda **kwargs: calls.append(("init", kwargs)) or FakeRun(),
        ),
    )
    monkeypatch.setenv("WANDB_API_KEY", "fixture-secret")
    logger = start_wandb_run(
        run_id="fixture-run",
        objective="sft_training_backfill",
        config_sha256="a" * 64,
        dataset_manifest_sha256="b" * 64,
        metadata={"historical_backfill": True},
        project="project",
        run_name="backfill",
        group="multiseed",
        evidence_mode="backfill",
        source_artifact_sha256="c" * 64,
    )
    logger.finish(training_status="complete", exit_code=0)

    summary = logger.summary()
    assert summary["evidence_mode"] == "backfill"
    assert summary["source_artifact_sha256"] == "c" * 64
    init = next(item for item in calls if item[0] == "init")[1]
    assert init["config"]["evidence_mode"] == "backfill"
    assert init["config"]["source_artifact_sha256"] == "c" * 64
    assert init["tags"][0] == "historical-backfill"
    assert "tpu-v5e8" not in init["tags"]


def test_wandb_init_failure_is_fail_open_and_redacted(monkeypatch):
    redaction_marker = "fixture-secret"

    def fail(**kwargs):
        del kwargs
        raise RuntimeError(f"network rejected {redaction_marker}")

    monkeypatch.setitem(
        sys.modules,
        "wandb",
        SimpleNamespace(Settings=lambda **kwargs: kwargs, init=fail),
    )
    monkeypatch.setenv("WANDB_API_KEY", redaction_marker)
    logger = _start()
    logger.log_metrics({"loss": 1.0}, step=1)
    logger.finish(training_status="blocked", exit_code=69)

    summary = logger.summary()
    assert summary["status"] == "degraded"
    assert summary["reason"] == "init_failed"
    assert redaction_marker not in summary["error"]
    assert "<redacted>" in summary["error"]
