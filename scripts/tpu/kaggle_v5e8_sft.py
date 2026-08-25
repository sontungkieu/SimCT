#!/usr/bin/env python3
"""Resume-safe Gemma warm-start SFT on a provenance-checked corpus."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vdt_tunix.config import ConfigError, load_config
from vdt_tunix.integration import RealModelIntegrationUnavailable, load_real_backend_bundle
from vdt_tunix.model_adapters import ModelAdapterError
from vdt_tunix.observability import start_wandb_run
from vdt_tunix.runtime import TPUPreflightError, require_tpu_v5e8
from vdt_tunix.sft_trainer import SFTTrainingError, TunixSFTTrainer
from vdt_tunix.training_data import TrainingDataError, load_sft_dataset
from vdt_tunix.tunix_checkpoint import TunixCheckpointController, TunixCheckpointError


EX_UNAVAILABLE = 69
EX_CONFIG = 78


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True) + "\n")
        stream.flush()


def _finish(path: Path, payload: dict[str, Any], exit_code: int) -> int:
    _atomic_json(path, payload)
    stream = sys.stdout if exit_code == 0 else sys.stderr
    print(json.dumps(payload, sort_keys=True), file=stream, flush=True)
    return exit_code


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--dataset-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--metrics", required=True, type=Path)
    args = parser.parse_args(argv)

    try:
        config = load_config(args.config)
        dataset = load_sft_dataset(args.dataset_manifest)
        if config.checkpoint.warm_start_from is not None:
            raise ConfigError("SFT accepts resume_from, not warm_start_from")
    except (ConfigError, TrainingDataError, OSError) as exc:
        return _finish(
            args.output,
            {
                "phase": "sft_configuration_and_data",
                "status": "blocked",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "scientific_evidence": False,
            },
            EX_CONFIG,
        )

    controller: TunixCheckpointController | None = None
    observability = start_wandb_run(
        run_id=config.run_id,
        objective="sft",
        config_sha256=config.digest(),
        dataset_manifest_sha256=dataset.manifest.digest(),
        metadata={
            "student_model_id": config.student.model_id,
            "teacher_model_id": config.teacher.model_id,
            "max_steps": config.training.max_steps,
            "learning_rate": config.training.learning_rate,
            "training_seed": config.training.seed,
        },
    )
    training_started = time.monotonic()
    try:
        backends = load_real_backend_bundle(config)
        _, hardware = require_tpu_v5e8(
            expected_device_count=config.tpu.expected_device_count
        )
        trainer = TunixSFTTrainer(config, backends)
        controller = TunixCheckpointController(
            config,
            trainer.loaded_student.model,
            trainer.optimizer,
            dataset_manifest_sha256=dataset.manifest.digest(),
        )
        resume = controller.restore_if_requested()
        remaining = config.training.max_steps - resume.completed_steps
        if remaining < 0:
            raise SFTTrainingError("resume step exceeds training.max_steps")
        cursor = resume.data_cursor
        completed = resume.completed_steps
        last_metrics: dict[str, Any] | None = None
        final_checkpoint = None
        for rows, next_cursor in dataset.batches(
            cursor=cursor,
            batch_size=config.rollout.prompt_batch_size,
            max_steps=remaining,
        ):
            step_started = time.monotonic()
            update = trainer.step(rows, step=completed)
            row = {
                "run_id": config.run_id,
                "step": completed + 1,
                "step_elapsed_s": time.monotonic() - step_started,
                "elapsed_s": time.monotonic() - training_started,
                **update.to_dict(),
            }
            if any(
                not math.isfinite(float(row[name]))
                for name in ("loss", "gradient_norm", "parameter_norm")
            ):
                raise SFTTrainingError("non-finite SFT update metric")
            _append_jsonl(args.metrics, row)
            observability.log_metrics(row, step=completed + 1)
            completed += 1
            cursor = next_cursor
            last_metrics = row
            if (
                completed % config.checkpoint.save_every_steps == 0
                or completed == config.training.max_steps
            ):
                final_checkpoint = controller.save(
                    completed_steps=completed,
                    data_cursor=cursor,
                    rng_state={"trainer_completed_steps": str(completed)},
                )
    except (
        RealModelIntegrationUnavailable,
        ModelAdapterError,
        TPUPreflightError,
        SFTTrainingError,
        TunixCheckpointError,
    ) as exc:
        observability.finish(training_status="blocked", exit_code=EX_UNAVAILABLE)
        return _finish(
            args.output,
            {
                "phase": "sft_training",
                "status": "blocked",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "run_id": config.run_id,
                "config_sha256": config.digest(),
                "dataset_manifest_sha256": dataset.manifest.digest(),
                "observability": observability.summary(),
                "scientific_evidence": False,
            },
            EX_UNAVAILABLE,
        )
    except Exception as exc:
        observability.finish(training_status="blocked", exit_code=EX_UNAVAILABLE)
        return _finish(
            args.output,
            {
                "phase": "sft_training_unexpected",
                "status": "blocked",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "run_id": config.run_id,
                "config_sha256": config.digest(),
                "dataset_manifest_sha256": dataset.manifest.digest(),
                "observability": observability.summary(),
                "scientific_evidence": False,
            },
            EX_UNAVAILABLE,
        )
    finally:
        if controller is not None:
            controller.close()

    observability.finish(training_status="complete", exit_code=0)
    return _finish(
        args.output,
        {
            "phase": "sft_training",
            "status": "complete",
            "run_id": config.run_id,
            "config_sha256": config.digest(),
            "dataset_manifest_sha256": dataset.manifest.digest(),
            "dataset_id": dataset.manifest.dataset_id,
            "dataset_revision": dataset.manifest.dataset_revision,
            "start_step": resume.completed_steps,
            "completed_steps": completed,
            "data_cursor": {
                "epoch": cursor.epoch,
                "next_prompt_index": cursor.next_prompt_index,
            },
            "last_update_metrics": last_metrics,
            "hardware": hardware,
            "checkpoint_root": config.checkpoint.root,
            "final_student_parameters_sha256": (
                None
                if final_checkpoint is None
                else final_checkpoint.student_parameters.sha256
            ),
            "observability": observability.summary(),
            "scientific_evidence": False,
            "remaining_gate": "shared downstream evaluation and OPD comparison",
        },
        0,
    )


if __name__ == "__main__":
    raise SystemExit(main())
