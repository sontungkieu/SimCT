#!/usr/bin/env python3
"""Kaggle TPU v5e-8 contract canary.

This entrypoint intentionally exits non-zero before JAX hardware discovery
while the real model adapter is absent. It never substitutes the CPU mocks.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vdt_tunix.config import ConfigError, load_config
from vdt_tunix.contracts import PromptRecord
from vdt_tunix.integration import (
    RealModelIntegrationUnavailable,
    load_real_backend_bundle,
)
from vdt_tunix.pipeline import PipelineContractError, run_contract_canary
from vdt_tunix.runtime import TPUPreflightError, require_tpu_v5e8


EX_UNAVAILABLE = 69
EX_CONFIG = 78


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _finish(payload: dict[str, Any], output: Path, exit_code: int) -> int:
    _write_json(output, payload)
    stream = sys.stdout if exit_code == 0 else sys.stderr
    print(json.dumps(payload, sort_keys=True), file=stream, flush=True)
    return exit_code


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)

    try:
        config = load_config(args.config)
    except (ConfigError, OSError) as exc:
        return _finish(
            {
                "phase": "configuration",
                "status": "blocked",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "configuration_validated": False,
                "real_model_integration": False,
                "hardware_probe_attempted": False,
                "simct_update_executed": False,
                "scientific_evidence": False,
            },
            args.output,
            EX_CONFIG,
        )

    try:
        backends = load_real_backend_bundle(config)
    except RealModelIntegrationUnavailable as exc:
        return _finish(
            {
                "phase": "real_model_integration",
                "status": "blocked",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "run_id": config.run_id,
                "config_sha256": config.digest(),
                "configuration_validated": True,
                "real_model_integration": False,
                "hardware_probe_attempted": False,
                "simct_update_executed": False,
                "scientific_evidence": False,
            },
            args.output,
            EX_UNAVAILABLE,
        )

    try:
        _, hardware = require_tpu_v5e8(
            expected_device_count=config.tpu.expected_device_count
        )
        prompts = tuple(
            PromptRecord(
                prompt_id=(
                    config.canary.prompt_id
                    if config.rollout.prompt_batch_size == 1
                    else f"{config.canary.prompt_id}/{index}"
                ),
                student_prompt=config.canary.student_prompt,
                teacher_prompt=config.canary.teacher_prompt,
            )
            for index in range(config.rollout.prompt_batch_size)
        )
        report = run_contract_canary(
            config,
            prompts,
            backends,
            require_real_integration=True,
            hardware=hardware,
        )
    except (TPUPreflightError, PipelineContractError) as exc:
        return _finish(
            {
                "phase": "tpu_contract_canary",
                "status": "blocked",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "run_id": config.run_id,
                "config_sha256": config.digest(),
                "configuration_validated": True,
                "real_model_integration": True,
                "hardware_probe_attempted": True,
                "simct_update_executed": False,
                "scientific_evidence": False,
            },
            args.output,
            EX_UNAVAILABLE,
        )

    payload = report.to_dict()
    payload["phase"] = "tpu_contract_canary"
    payload["status"] = "passed"
    return _finish(payload, args.output, 0)


if __name__ == "__main__":
    raise SystemExit(main())
