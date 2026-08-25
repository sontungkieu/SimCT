"""Fail-open experiment observability for Kaggle TPU training.

Weights & Biases is an operational side channel, never part of the training
objective or the scientific-evidence contract. A missing key, unavailable
package, or network failure must therefore degrade logging without changing a
model update or terminating training.
"""

from __future__ import annotations

import importlib
import json
import math
import os
from dataclasses import dataclass, field
from typing import Any, Mapping


_SECRET_PLACEHOLDER_PREFIX = "__KJO_SECRET_"


def _safe_error(exc: BaseException) -> str:
    message = str(exc)
    if os.environ.get("WANDB_API_KEY", ""):
        message = message.replace(
            os.environ["WANDB_API_KEY"], "<redacted>"
        )
    return message[-1000:]


def _numeric_metrics(values: Mapping[str, Any], namespace: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, value in values.items():
        if isinstance(value, bool):
            result[f"{namespace}/{name}"] = value
        elif isinstance(value, int):
            result[f"{namespace}/{name}"] = value
        elif isinstance(value, float) and math.isfinite(value):
            result[f"{namespace}/{name}"] = value
    return result


@dataclass
class BestEffortWandbRun:
    """Small W&B adapter whose public methods never raise."""

    requested: bool
    project: str
    run_name: str
    group: str
    status: str = "disabled"
    reason: str = "WANDB_API_KEY_not_present"
    run_url: str = ""
    logged_steps: int = 0
    error_type: str = ""
    error: str = ""
    evidence_mode: str = "native"
    source_artifact_sha256: str = ""
    _run: Any = field(default=None, repr=False)
    _logging_disabled: bool = field(default=False, repr=False)

    def log_metrics(
        self,
        values: Mapping[str, Any],
        *,
        step: int,
        namespace: str = "train",
    ) -> None:
        if self._run is None or self._logging_disabled:
            return
        payload = _numeric_metrics(values, namespace)
        payload["trainer/global_step"] = step
        try:
            self._run.log(payload, step=step)
            self.logged_steps += 1
        except Exception as exc:  # W&B must never change training control flow.
            self.status = "degraded"
            self.reason = "log_failed"
            self.error_type = type(exc).__name__
            self.error = _safe_error(exc)
            self._logging_disabled = True
            self._emit("log_failed")

    def finish(self, *, training_status: str, exit_code: int) -> None:
        if self._run is not None:
            try:
                self._run.finish(exit_code=exit_code)
                if self.status == "active":
                    self.status = "finished"
                    self.reason = "training_finished"
            except Exception as exc:  # W&B must never mask the real exit code.
                self.status = "degraded"
                self.reason = "finish_failed"
                self.error_type = type(exc).__name__
                self.error = _safe_error(exc)
        self._emit("finish", training_status=training_status, exit_code=exit_code)

    def summary(self) -> dict[str, Any]:
        return {
            "provider": "wandb",
            "requested": self.requested,
            "fail_open": True,
            "secret_name": "WANDB_API_KEY",
            "project": self.project,
            "run_name": self.run_name,
            "group": self.group,
            "status": self.status,
            "reason": self.reason,
            "run_url": self.run_url,
            "logged_steps": self.logged_steps,
            "error_type": self.error_type,
            "error": self.error,
            "evidence_mode": self.evidence_mode,
            "source_artifact_sha256": self.source_artifact_sha256,
        }

    def _emit(self, event: str, **extra: Any) -> None:
        payload = {"event": event, **self.summary(), **extra}
        print("VDT_WANDB_STATUS " + json.dumps(payload, sort_keys=True), flush=True)


def start_wandb_run(
    *,
    run_id: str,
    objective: str,
    config_sha256: str,
    dataset_manifest_sha256: str,
    metadata: Mapping[str, Any],
    project: str | None = None,
    run_name: str | None = None,
    group: str | None = None,
    evidence_mode: str = "native",
    source_artifact_sha256: str = "",
) -> BestEffortWandbRun:
    """Start one online W&B run when the staged key was injected.

    The API key is consumed only through W&B's environment-variable contract;
    it is never passed as config, logged, hashed, or returned in the summary.
    """

    key = os.environ.get("WANDB_API_KEY", "")
    requested = bool(key) and not key.startswith(_SECRET_PLACEHOLDER_PREFIX)
    project = project or os.environ.get(
        "WANDB_PROJECT", "vdt-simct-tunix-reproduction"
    )
    run_name = run_name or os.environ.get(
        "WANDB_RUN_NAME", f"{run_id}-{objective}"
    )
    group = group or os.environ.get(
        "WANDB_RUN_GROUP", "public-substitute-one-seed"
    )
    if evidence_mode not in {"native", "backfill"}:
        raise ValueError("evidence_mode must be native or backfill")
    logger = BestEffortWandbRun(
        requested=requested,
        project=project,
        run_name=run_name,
        group=group,
        evidence_mode=evidence_mode,
        source_artifact_sha256=source_artifact_sha256,
    )
    if not requested:
        logger._emit("disabled")
        return logger

    try:
        wandb = importlib.import_module("wandb")
        init_kwargs: dict[str, Any] = {
            "project": project,
            "name": run_name,
            "group": group,
            "job_type": objective,
            "mode": os.environ.get("WANDB_MODE", "online"),
            "config": {
                "run_id": run_id,
                "objective": objective,
                "config_sha256": config_sha256,
                "dataset_manifest_sha256": dataset_manifest_sha256,
                "evidence_mode": evidence_mode,
                "source_artifact_sha256": source_artifact_sha256,
                **dict(metadata),
            },
            "tags": (
                ["kaggle", "tpu-v5e8", objective, "public-substitute"]
                if evidence_mode == "native"
                else ["historical-backfill", objective, "public-substitute"]
            ),
        }
        entity = os.environ.get("WANDB_ENTITY", "")
        if entity:
            init_kwargs["entity"] = entity
        settings_type = getattr(wandb, "Settings", None)
        if settings_type is not None:
            init_kwargs["settings"] = settings_type(
                init_timeout=int(os.environ.get("WANDB_INIT_TIMEOUT", "30"))
            )
        logger._run = wandb.init(**init_kwargs)
        if logger._run is None:
            raise RuntimeError("wandb.init returned no run")
        logger.status = "active"
        logger.reason = "online_run_started"
        get_url = getattr(logger._run, "get_url", None)
        if callable(get_url):
            logger.run_url = str(get_url() or "")
        elif getattr(logger._run, "url", None):
            logger.run_url = str(logger._run.url)
        logger._emit("started")
    except Exception as exc:
        logger.status = "degraded"
        logger.reason = "init_failed"
        logger.error_type = type(exc).__name__
        logger.error = _safe_error(exc)
        logger._run = None
        logger._emit("init_failed")
    return logger
