from __future__ import annotations

import copy

import pytest

from vdt_tunix.wandb_evidence import (
    WandbEvidenceError,
    validate_native_wandb_evidence,
)


def _observability(name: str, steps: int) -> dict:
    return {
        "provider": "wandb",
        "requested": True,
        "fail_open": True,
        "status": "finished",
        "reason": "training_finished",
        "run_url": f"https://wandb.ai/entity/project/runs/{name}",
        "logged_steps": steps,
        "project": "project",
        "run_name": name,
        "group": "multiseed",
        "error_type": "",
        "error": "",
    }


def _summaries():
    training = {"completed_steps": 10, "observability": _observability("train", 10)}
    generation = {
        "benchmarks": [{"batch_count": 2}, {"batch_count": 3}],
        "observability": _observability("generate", 5),
    }
    scoring = {
        "benchmarks": [{"benchmark": "a"}, {"benchmark": "b"}],
        "observability": _observability("score", 2),
    }
    return training, generation, scoring


def test_native_wandb_evidence_requires_all_three_finished_runs():
    training, generation, scoring = _summaries()
    report = validate_native_wandb_evidence(
        training_summary=training,
        generation_summary=generation,
        scoring_summary=scoring,
    )
    assert report["status"] == "passed"
    assert set(report["phases"]) == {"training", "generation", "scoring"}


def test_native_wandb_evidence_accepts_required_fail_closed_phases():
    training, generation, scoring = _summaries()
    for summary in (generation, scoring):
        summary["observability"]["required"] = True
        summary["observability"]["fail_open"] = False
    report = validate_native_wandb_evidence(
        training_summary=training,
        generation_summary=generation,
        scoring_summary=scoring,
    )
    assert report["status"] == "passed"


@pytest.mark.parametrize(
    ("phase", "field", "value", "match"),
    [
        ("training", "status", "degraded", "drifted"),
        ("generation", "logged_steps", 4, "must be 5"),
        ("scoring", "run_url", "", "not a W&B URL"),
        ("training", "fail_open", "yes", "must be a boolean"),
    ],
)
def test_native_wandb_evidence_fails_closed(phase, field, value, match):
    summaries = list(_summaries())
    index = {"training": 0, "generation": 1, "scoring": 2}[phase]
    summaries[index] = copy.deepcopy(summaries[index])
    summaries[index]["observability"][field] = value
    with pytest.raises(WandbEvidenceError, match=match):
        validate_native_wandb_evidence(
            training_summary=summaries[0],
            generation_summary=summaries[1],
            scoring_summary=summaries[2],
        )


def test_native_wandb_evidence_rejects_inconsistent_required_policy():
    training, generation, scoring = _summaries()
    generation["observability"]["required"] = True
    with pytest.raises(WandbEvidenceError, match="must equal not required"):
        validate_native_wandb_evidence(
            training_summary=training,
            generation_summary=generation,
            scoring_summary=scoring,
        )


def test_backfilled_wandb_evidence_requires_explicit_opt_in():
    training, generation, scoring = _summaries()
    for summary in (training, generation, scoring):
        summary["observability"]["evidence_mode"] = "backfill"
        summary["observability"]["source_artifact_sha256"] = "a" * 64
    with pytest.raises(WandbEvidenceError, match="backfilled rather than native"):
        validate_native_wandb_evidence(
            training_summary=training,
            generation_summary=generation,
            scoring_summary=scoring,
        )
    report = validate_native_wandb_evidence(
        training_summary=training,
        generation_summary=generation,
        scoring_summary=scoring,
        allow_backfill=True,
    )
    assert report["evidence_mode"] == "backfill"
