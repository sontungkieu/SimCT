"""Fail-closed validation for W&B monitoring evidence.

W&B never controls model updates, but a run can still be rejected by the
orchestrator when the requested monitoring channel did not finish cleanly.
This keeps scientific artifacts recoverable while making observability an
explicit, auditable gate.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


class WandbEvidenceError(RuntimeError):
    """Raised when a terminal summary lacks the required W&B evidence."""


def _positive_int(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise WandbEvidenceError(f"{context} must be a positive integer")
    return value


def _validate_observability(
    summary: Mapping[str, Any],
    *,
    context: str,
    expected_logged_steps: int,
    allow_backfill: bool,
) -> dict[str, Any]:
    observability = summary.get("observability")
    if not isinstance(observability, Mapping):
        raise WandbEvidenceError(f"{context}.observability must be an object")
    expected = {
        "provider": "wandb",
        "requested": True,
        "status": "finished",
    }
    drift = {
        key: (observability.get(key), value)
        for key, value in expected.items()
        if observability.get(key) != value
    }
    if drift:
        raise WandbEvidenceError(f"{context} W&B evidence drifted: {drift}")
    fail_open = observability.get("fail_open")
    if not isinstance(fail_open, bool):
        raise WandbEvidenceError(f"{context}.fail_open must be a boolean")
    required = observability.get("required")
    if required is not None:
        if not isinstance(required, bool):
            raise WandbEvidenceError(f"{context}.required must be a boolean")
        if fail_open is required:
            raise WandbEvidenceError(
                f"{context}.fail_open must equal not required"
            )
    logged_steps = _positive_int(
        observability.get("logged_steps"), f"{context}.logged_steps"
    )
    if logged_steps != expected_logged_steps:
        raise WandbEvidenceError(
            f"{context}.logged_steps must be {expected_logged_steps}, got "
            f"{logged_steps}"
        )
    run_url = observability.get("run_url")
    if not isinstance(run_url, str) or not run_url.startswith("https://wandb.ai/"):
        raise WandbEvidenceError(f"{context}.run_url is not a W&B URL")
    for field in ("project", "run_name", "group"):
        value = observability.get(field)
        if not isinstance(value, str) or not value.strip():
            raise WandbEvidenceError(f"{context}.{field} is missing")
    if observability.get("error_type") or observability.get("error"):
        raise WandbEvidenceError(f"{context} W&B summary contains an error")
    evidence_mode = observability.get("evidence_mode", "native")
    if evidence_mode not in {"native", "backfill"}:
        raise WandbEvidenceError(
            f"{context}.evidence_mode must be native or backfill"
        )
    if evidence_mode == "backfill":
        if not allow_backfill:
            raise WandbEvidenceError(
                f"{context} contains backfilled rather than native W&B evidence"
            )
        source_sha256 = observability.get("source_artifact_sha256")
        if not isinstance(source_sha256, str) or len(source_sha256) != 64:
            raise WandbEvidenceError(
                f"{context}.source_artifact_sha256 is missing"
            )
    return dict(observability)


def validate_native_wandb_evidence(
    *,
    training_summary: Mapping[str, Any],
    generation_summary: Mapping[str, Any],
    scoring_summary: Mapping[str, Any],
    allow_backfill: bool = False,
) -> dict[str, Any]:
    """Validate online training, generation, and scoring runs for one variant."""

    completed_steps = _positive_int(
        training_summary.get("completed_steps"), "training.completed_steps"
    )
    generation_benchmarks = generation_summary.get("benchmarks")
    scoring_benchmarks = scoring_summary.get("benchmarks")
    if not isinstance(generation_benchmarks, list) or not generation_benchmarks:
        raise WandbEvidenceError("generation.benchmarks must be non-empty")
    if not isinstance(scoring_benchmarks, list) or not scoring_benchmarks:
        raise WandbEvidenceError("scoring.benchmarks must be non-empty")
    generation_steps = sum(
        _positive_int(item.get("batch_count"), "generation.batch_count")
        for item in generation_benchmarks
        if isinstance(item, Mapping)
    )
    if len(generation_benchmarks) != sum(
        isinstance(item, Mapping) for item in generation_benchmarks
    ):
        raise WandbEvidenceError("generation benchmark entry must be an object")
    if len(scoring_benchmarks) != sum(
        isinstance(item, Mapping) for item in scoring_benchmarks
    ):
        raise WandbEvidenceError("scoring benchmark entry must be an object")

    phases = {
        "training": _validate_observability(
            training_summary,
            context="training",
            expected_logged_steps=completed_steps,
            allow_backfill=allow_backfill,
        ),
        "generation": _validate_observability(
            generation_summary,
            context="generation",
            expected_logged_steps=generation_steps,
            allow_backfill=allow_backfill,
        ),
        "scoring": _validate_observability(
            scoring_summary,
            context="scoring",
            expected_logged_steps=len(scoring_benchmarks),
            allow_backfill=allow_backfill,
        ),
    }
    projects = {item["project"] for item in phases.values()}
    groups = {item["group"] for item in phases.values()}
    urls = {item["run_url"] for item in phases.values()}
    if len(projects) != 1:
        raise WandbEvidenceError(f"W&B project drifted across phases: {projects}")
    if len(groups) != 1:
        raise WandbEvidenceError(f"W&B group drifted across phases: {groups}")
    if len(urls) != len(phases):
        raise WandbEvidenceError("training/generation/scoring reused a W&B run URL")
    evidence_modes = {
        item.get("evidence_mode", "native") for item in phases.values()
    }
    if len(evidence_modes) != 1:
        raise WandbEvidenceError(
            f"W&B evidence mode drifted across phases: {evidence_modes}"
        )
    return {
        "status": "passed",
        "project": next(iter(projects)),
        "group": next(iter(groups)),
        "evidence_mode": next(iter(evidence_modes)),
        "phases": phases,
    }
