"""Provenance-preserving historical W&B backfill for completed runs.

The source summaries remain immutable.  This module validates their shared
lineage, replays only metrics that are already present in audited artifacts,
and writes derived summary copies whose observability blocks point at three
distinct W&B runs.  A backfill is monitoring evidence, not evidence that the
original run was observed live.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from vdt_tunix.observability import BestEffortWandbRun, start_wandb_run
from vdt_tunix.wandb_evidence import validate_native_wandb_evidence


class WandbBackfillError(RuntimeError):
    """Raised when source evidence or the W&B replay fails closed."""


RunFactory = Callable[..., BestEffortWandbRun]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise WandbBackfillError(f"invalid JSON artifact: {path}") from exc
    if not isinstance(payload, dict):
        raise WandbBackfillError(f"JSON artifact must be an object: {path}")
    return payload


def _load_training_metrics(
    path: Path, *, completed_steps: int, run_id: str
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise WandbBackfillError(f"training metrics are missing: {path}") from exc
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise WandbBackfillError(
                f"training metrics line {line_number} is invalid JSON"
            ) from exc
        if not isinstance(row, dict):
            raise WandbBackfillError(
                f"training metrics line {line_number} must be an object"
            )
        rows.append(row)
    expected_steps = list(range(1, completed_steps + 1))
    observed_steps = [row.get("step") for row in rows]
    if observed_steps != expected_steps:
        raise WandbBackfillError(
            "training metric steps drifted: "
            f"expected {expected_steps}, got {observed_steps}"
        )
    if any(row.get("run_id") != run_id for row in rows):
        raise WandbBackfillError("training metric run_id drifted")
    for row in rows:
        for name, value in row.items():
            if isinstance(value, float) and not math.isfinite(value):
                raise WandbBackfillError(
                    f"training metric {name} at step {row['step']} is not finite"
                )
    return rows


def _require_complete(summary: Mapping[str, Any], context: str) -> None:
    if summary.get("status") != "complete":
        raise WandbBackfillError(f"{context} summary is not complete")


def _require_sha256(value: Any, context: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise WandbBackfillError(f"{context} must be a lowercase SHA-256")
    return value


def _generation_plan(
    summary: Mapping[str, Any],
) -> list[tuple[str, int]]:
    benchmarks = summary.get("benchmarks")
    if not isinstance(benchmarks, list) or not benchmarks:
        raise WandbBackfillError("generation.benchmarks must be non-empty")
    plan: list[tuple[str, int]] = []
    for benchmark in benchmarks:
        if not isinstance(benchmark, Mapping):
            raise WandbBackfillError("generation benchmark must be an object")
        name = benchmark.get("benchmark")
        batch_count = benchmark.get("batch_count")
        if not isinstance(name, str) or not name:
            raise WandbBackfillError("generation benchmark name is missing")
        if (
            isinstance(batch_count, bool)
            or not isinstance(batch_count, int)
            or batch_count < 1
        ):
            raise WandbBackfillError(
                f"generation batch_count is invalid for {name}"
            )
        plan.append((name, batch_count))
    return plan


def _scoring_plan(
    summary: Mapping[str, Any],
) -> list[tuple[str, dict[str, Any]]]:
    benchmarks = summary.get("benchmarks")
    if not isinstance(benchmarks, list) or not benchmarks:
        raise WandbBackfillError("scoring.benchmarks must be non-empty")
    plan: list[tuple[str, dict[str, Any]]] = []
    for benchmark in benchmarks:
        if not isinstance(benchmark, Mapping):
            raise WandbBackfillError("scoring benchmark must be an object")
        name = benchmark.get("benchmark")
        if not isinstance(name, str) or not name:
            raise WandbBackfillError("scoring benchmark name is missing")
        values = {
            key: benchmark[key]
            for key in (
                "score",
                "correct",
                "total",
                "empty_output_count",
                "extraction_failed_count",
                "generation_truncated_count",
            )
            if key in benchmark
        }
        score = values.get("score")
        if (
            isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(float(score))
        ):
            raise WandbBackfillError(f"scoring score is invalid for {name}")
        for field in (
            "correct",
            "total",
            "empty_output_count",
            "extraction_failed_count",
            "generation_truncated_count",
        ):
            if field not in values:
                continue
            value = values[field]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise WandbBackfillError(
                    f"scoring {field} is invalid for {name}"
                )
        plan.append((name, values))
    return plan


def _finished_observability(
    logger: BestEffortWandbRun,
    *,
    context: str,
    expected_steps: int,
) -> dict[str, Any]:
    logger.finish(training_status="complete", exit_code=0)
    summary = logger.summary()
    if summary.get("status") != "finished":
        raise WandbBackfillError(
            f"{context} W&B backfill did not finish: {summary.get('reason')}"
        )
    if summary.get("logged_steps") != expected_steps:
        raise WandbBackfillError(
            f"{context} W&B backfill logged "
            f"{summary.get('logged_steps')} of "
            f"{expected_steps} expected steps"
        )
    run_url = summary.get("run_url")
    if not isinstance(run_url, str) or not run_url.startswith("https://wandb.ai/"):
        raise WandbBackfillError(f"{context} W&B backfill has no run URL")
    return summary


def backfill_wandb_evidence(
    *,
    training_summary_path: Path,
    training_metrics_path: Path,
    generation_summary_path: Path,
    scoring_summary_path: Path,
    project: str,
    group: str,
    run_factory: RunFactory = start_wandb_run,
) -> dict[str, Any]:
    """Replay audited metrics and return derived summaries plus provenance."""

    paths = {
        "training_summary": training_summary_path,
        "training_metrics": training_metrics_path,
        "generation_summary": generation_summary_path,
        "scoring_summary": scoring_summary_path,
    }
    source_hashes = {name: _sha256(path) for name, path in paths.items()}
    training = _load_object(training_summary_path)
    generation = _load_object(generation_summary_path)
    scoring = _load_object(scoring_summary_path)
    for context, summary in (
        ("training", training),
        ("generation", generation),
        ("scoring", scoring),
    ):
        _require_complete(summary, context)

    run_id = training.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise WandbBackfillError("training.run_id is missing")
    completed_steps = training.get("completed_steps")
    if (
        isinstance(completed_steps, bool)
        or not isinstance(completed_steps, int)
        or completed_steps < 1
    ):
        raise WandbBackfillError("training.completed_steps must be positive")
    config_sha256 = _require_sha256(
        training.get("config_sha256"), "training.config_sha256"
    )
    dataset_sha256 = _require_sha256(
        training.get("dataset_manifest_sha256"),
        "training.dataset_manifest_sha256",
    )
    final_parameters_sha256 = _require_sha256(
        training.get("final_student_parameters_sha256"),
        "training.final_student_parameters_sha256",
    )
    variant = generation.get("variant")
    if variant not in {"sft", "simple_opd", "simct"}:
        raise WandbBackfillError("generation.variant is unsupported")
    lineage_drift = {
        "generation.checkpoint_run_id": (
            generation.get("checkpoint_run_id"),
            run_id,
        ),
        "generation.training_config_sha256": (
            generation.get("training_config_sha256"),
            config_sha256,
        ),
        "generation.student_parameters_sha256": (
            generation.get("student_parameters_sha256"),
            final_parameters_sha256,
        ),
        "scoring.variant": (scoring.get("variant"), variant),
        "scoring.checkpoint_sha256": (
            scoring.get("checkpoint_sha256"),
            final_parameters_sha256,
        ),
        "scoring.protocol_sha256": (
            scoring.get("protocol_sha256"),
            generation.get("protocol_sha256"),
        ),
    }
    mismatches = {
        name: values
        for name, values in lineage_drift.items()
        if values[0] != values[1]
    }
    if mismatches:
        raise WandbBackfillError(f"source lineage drifted: {mismatches}")
    protocol_sha256 = _require_sha256(
        generation.get("protocol_sha256"), "generation.protocol_sha256"
    )
    training_rows = _load_training_metrics(
        training_metrics_path,
        completed_steps=completed_steps,
        run_id=run_id,
    )
    generation_plan = _generation_plan(generation)
    scoring_plan = _scoring_plan(scoring)

    train_logger = run_factory(
        run_id=run_id,
        objective=f"{variant}_training_backfill",
        config_sha256=config_sha256,
        dataset_manifest_sha256=dataset_sha256,
        metadata={
            "variant": variant,
            "source_phase": training.get("phase", ""),
            "completed_steps": completed_steps,
            "historical_backfill": True,
        },
        project=project,
        run_name=f"{run_id}-{variant}-training-backfill",
        group=group,
        evidence_mode="backfill",
        source_artifact_sha256=source_hashes["training_summary"],
    )
    for row in training_rows:
        train_logger.log_metrics(row, step=row["step"], namespace="train")
    training_observability = _finished_observability(
        train_logger, context="training", expected_steps=completed_steps
    )

    generation_logger = run_factory(
        run_id=run_id,
        objective=f"{variant}_generation_backfill",
        config_sha256=config_sha256,
        dataset_manifest_sha256=protocol_sha256,
        metadata={
            "variant": variant,
            "protocol_sha256": protocol_sha256,
            "historical_backfill": True,
        },
        project=project,
        run_name=f"{run_id}-{variant}-generation-backfill",
        group=group,
        evidence_mode="backfill",
        source_artifact_sha256=source_hashes["generation_summary"],
    )
    generation_step = 0
    for name, batch_count in generation_plan:
        for batch_index in range(1, batch_count + 1):
            generation_step += 1
            generation_logger.log_metrics(
                {
                    "batch_index": batch_index,
                    "batch_count": batch_count,
                    "backfilled_completed_batch": 1,
                },
                step=generation_step,
                namespace=f"generation/{name}",
            )
    generation_observability = _finished_observability(
        generation_logger,
        context="generation",
        expected_steps=generation_step,
    )

    scoring_logger = run_factory(
        run_id=run_id,
        objective=f"{variant}_scoring_backfill",
        config_sha256=config_sha256,
        dataset_manifest_sha256=protocol_sha256,
        metadata={
            "variant": variant,
            "protocol_sha256": protocol_sha256,
            "historical_backfill": True,
        },
        project=project,
        run_name=f"{run_id}-{variant}-scoring-backfill",
        group=group,
        evidence_mode="backfill",
        source_artifact_sha256=source_hashes["scoring_summary"],
    )
    for step, (name, values) in enumerate(scoring_plan, start=1):
        scoring_logger.log_metrics(
            values,
            step=step,
            namespace=f"evaluation/{name}",
        )
    scoring_observability = _finished_observability(
        scoring_logger,
        context="scoring",
        expected_steps=len(scoring_plan),
    )

    derived_training = copy.deepcopy(training)
    derived_generation = copy.deepcopy(generation)
    derived_scoring = copy.deepcopy(scoring)
    derived_training["observability"] = training_observability
    derived_generation["observability"] = generation_observability
    derived_scoring["observability"] = scoring_observability
    wandb_evidence = validate_native_wandb_evidence(
        training_summary=derived_training,
        generation_summary=derived_generation,
        scoring_summary=derived_scoring,
        allow_backfill=True,
    )
    return {
        "status": "passed",
        "mode": "historical_backfill",
        "scientific_evidence": False,
        "variant": variant,
        "run_id": run_id,
        "source_paths": {name: str(path) for name, path in paths.items()},
        "source_sha256": source_hashes,
        "wandb_evidence": wandb_evidence,
        "summaries": {
            "training": derived_training,
            "generation": derived_generation,
            "scoring": derived_scoring,
        },
    }
