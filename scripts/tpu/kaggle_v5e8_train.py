#!/usr/bin/env python3
"""Resume-safe single-teacher OPD training on a Kaggle TPU v5e-8.

This entrypoint produces training evidence and durable model/optimizer state.
It does not evaluate downstream tasks and therefore never labels its output as
scientific reproduction evidence by itself.
"""

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
from vdt_tunix.training_data import TrainingDataError, load_prompt_dataset
from vdt_tunix.trainer import (
    PaperSimCTTrainer,
    PaperSimpleOPDTrainer,
    TrainingError,
)
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


def _target_trainer_calls(config: Any) -> int:
    if config.training.max_steps_unit == "optimizer_update":
        return config.training.max_steps * config.training.gradient_accumulation_steps
    return config.training.max_steps


def _aggregate_optimizer_update(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise TrainingError("optimizer update aggregation requires micro-step rows")
    result = dict(rows[-1])
    additive = {
        "step_elapsed_s",
        "rollout_s",
        "teacher_score_s",
        "teacher_tokenize_s",
        "teacher_forward_s",
        "alignment_s",
        "batch_prepare_s",
        "student_update_dispatch_s",
        "student_update_sync_s",
        "student_update_s",
        "student_fwd_bwd_s",
        "compile_s",
        "sample_count",
        "student_completion_tokens",
        "teacher_completion_tokens",
        "actual_prompt_tokens",
        "actual_completion_tokens",
        "actual_total_tokens",
        "aligned_units",
        "aligned_spans",
        "truncation_count",
    }
    maximum = {
        "maximum_prompt_tokens",
        "maximum_completion_tokens",
        "maximum_total_tokens",
        "teacher_sequence_required",
        "teacher_sequence_bucket",
        "teacher_completion_bucket",
        "alignment_bucket",
        "memory_bytes_in_use",
        "memory_peak_bytes_in_use",
        "memory_bytes_limit",
        "shape_signature_changed",
        "jit_cache_miss",
    }
    for name in additive:
        result[name] = sum(float(row.get(name, 0.0)) for row in rows)
    for name in maximum:
        result[name] = max(int(row.get(name, -1)) for row in rows)
    result["minimum_total_tokens"] = min(
        int(row["minimum_total_tokens"]) for row in rows
    )
    result["micro_steps_per_optimizer_update"] = len(rows)
    rollout_s = float(result.get("rollout_s", 0.0))
    teacher_s = float(result.get("teacher_score_s", 0.0))
    student_s = float(result.get("student_update_s", 0.0))
    result["rollout_tokens_s"] = (
        float(result["actual_completion_tokens"]) / rollout_s if rollout_s > 0 else 0.0
    )
    result["teacher_score_tokens_s"] = (
        float(result["teacher_completion_tokens"]) / teacher_s
        if teacher_s > 0
        else 0.0
    )
    result["student_update_tokens_s"] = (
        float(result["actual_total_tokens"]) / student_s if student_s > 0 else 0.0
    )
    result["student_fwd_bwd_tokens_s"] = result["student_update_tokens_s"]
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--dataset-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--metrics", required=True, type=Path)
    parser.add_argument("--profile-dir", type=Path)
    parser.add_argument("--profile-step", type=int, default=0)
    args = parser.parse_args(argv)

    try:
        config = load_config(args.config)
        dataset = load_prompt_dataset(args.dataset_manifest)
    except (ConfigError, TrainingDataError, OSError) as exc:
        return _finish(
            args.output,
            {
                "phase": "configuration_and_data",
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
        objective=config.simct.algorithm,
        config_sha256=config.digest(),
        dataset_manifest_sha256=dataset.manifest.digest(),
        metadata={
            "student_model_id": config.student.model_id,
            "teacher_model_id": config.teacher.model_id,
            "max_steps": config.training.max_steps,
            "max_steps_unit": config.training.max_steps_unit,
            "learning_rate": config.training.learning_rate,
            "training_seed": config.training.seed,
            "prompt_batch_size": config.rollout.prompt_batch_size,
            "gradient_accumulation_steps": (
                config.training.gradient_accumulation_steps
            ),
            "effective_global_batch": (
                config.rollout.prompt_batch_size
                * config.rollout.samples_per_prompt
                * config.training.gradient_accumulation_steps
            ),
            "max_sequence_tokens": config.rollout.max_sequence_tokens,
            "max_completion_tokens": config.rollout.max_completion_tokens,
            "force_max_completion": config.rollout.force_max_completion,
            "fsdp_parallelism": config.tpu.fsdp_parallelism,
            "tensor_parallelism": config.tpu.tensor_parallelism,
            "pipeline_parallelism": config.tpu.pipeline_parallelism,
        },
    )
    training_started = time.monotonic()
    try:
        observability.require_active()
        backends = load_real_backend_bundle(config)
        _, hardware = require_tpu_v5e8(
            expected_device_count=config.tpu.expected_device_count
        )
        trainer_type = (
            PaperSimCTTrainer
            if config.simct.algorithm == "simct"
            else PaperSimpleOPDTrainer
        )
        trainer = trainer_type(config, backends)
        controller = TunixCheckpointController(
            config,
            trainer.loaded_student.model,
            trainer.optimizer,
            dataset_manifest_sha256=dataset.manifest.digest(),
        )
        resume = controller.initialize_or_resume()
        target_trainer_calls = _target_trainer_calls(config)
        if args.profile_dir is not None:
            if not 1 <= args.profile_step <= target_trainer_calls:
                raise TrainingError(
                    "profile_step must identify one executed trainer call"
                )
            args.profile_dir.mkdir(parents=True, exist_ok=True)
        if resume.completed_steps > target_trainer_calls:
            raise TrainingError("resume step exceeds the trainer-call target")
        remaining = target_trainer_calls - resume.completed_steps
        cursor = resume.data_cursor
        completed = resume.completed_steps
        completed_optimizer_steps = (
            completed // config.training.gradient_accumulation_steps
        )
        last_metrics: dict[str, Any] | None = None
        final_checkpoint = None
        pending_micro_rows: list[dict[str, Any]] = []
        micro_metrics_path = args.metrics.with_name(
            f"{args.metrics.stem}.micro.jsonl"
        )

        for prompts, next_cursor in dataset.batches(
            cursor=cursor,
            batch_size=config.rollout.prompt_batch_size,
            max_steps=remaining,
        ):
            step_started = time.monotonic()
            profile_this_step = (
                args.profile_dir is not None and completed + 1 == args.profile_step
            )
            if profile_this_step:
                trainer._jax.profiler.start_trace(str(args.profile_dir))
            try:
                update = trainer.step(prompts, step=completed)
            finally:
                if profile_this_step:
                    trainer._jax.profiler.stop_trace()
            row = {
                "run_id": config.run_id,
                "objective": config.simct.algorithm,
                "micro_step": completed + 1,
                "step_elapsed_s": time.monotonic() - step_started,
                "elapsed_s": time.monotonic() - training_started,
                "prompt_batch_size": config.rollout.prompt_batch_size,
                "gradient_accumulation_steps": (
                    config.training.gradient_accumulation_steps
                ),
                "effective_global_batch": (
                    config.rollout.prompt_batch_size
                    * config.rollout.samples_per_prompt
                    * config.training.gradient_accumulation_steps
                ),
                "fsdp_parallelism": config.tpu.fsdp_parallelism,
                "tensor_parallelism": config.tpu.tensor_parallelism,
                "pipeline_parallelism": config.tpu.pipeline_parallelism,
                "tpu_device_count": config.tpu.expected_device_count,
                **update.to_dict(),
            }
            if any(
                not math.isfinite(float(row[name]))
                for name in ("loss", "gradient_norm", "parameter_norm")
            ):
                raise TrainingError("non-finite update metric")
            completed += 1
            cursor = next_cursor
            optimizer_updated = (
                completed % config.training.gradient_accumulation_steps == 0
            )
            row["optimizer_updated"] = int(optimizer_updated)
            row["optimizer_step"] = (
                completed // config.training.gradient_accumulation_steps
            )
            _append_jsonl(micro_metrics_path, row)
            pending_micro_rows.append(row)
            should_log_optimizer = (
                config.training.max_steps_unit == "optimizer_update"
                and optimizer_updated
            )
            if config.training.max_steps_unit == "trainer_call":
                should_log_optimizer = True
            if should_log_optimizer:
                optimizer_row = _aggregate_optimizer_update(pending_micro_rows)
                optimizer_row["step"] = (
                    row["optimizer_step"]
                    if config.training.max_steps_unit == "optimizer_update"
                    else completed
                )
                optimizer_row["optimizer_step"] = row["optimizer_step"]
                optimizer_row["optimizer_updated"] = int(optimizer_updated)
                _append_jsonl(args.metrics, optimizer_row)
                observability.log_metrics(optimizer_row, step=optimizer_row["step"])
                last_metrics = optimizer_row
                pending_micro_rows.clear()
            if optimizer_updated:
                completed_optimizer_steps = row["optimizer_step"]
            checkpoint_due = optimizer_updated and (
                completed_optimizer_steps % config.checkpoint.save_every_steps == 0
                or completed == target_trainer_calls
            )
            if config.training.max_steps_unit == "trainer_call":
                checkpoint_due = (
                    completed % config.checkpoint.save_every_steps == 0
                    or completed == target_trainer_calls
                )
            if checkpoint_due:
                final_checkpoint = controller.save(
                    completed_steps=completed,
                    data_cursor=cursor,
                    rng_state={
                        "rollout_next_step": str(completed),
                        "trainer_completed_steps": str(completed),
                    },
                )
    except (
        RealModelIntegrationUnavailable,
        ModelAdapterError,
        TPUPreflightError,
        TrainingError,
        TunixCheckpointError,
    ) as exc:
        observability.finish(training_status="blocked", exit_code=EX_UNAVAILABLE)
        return _finish(
            args.output,
            {
                "phase": f"{config.simct.algorithm}_training",
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
                "phase": f"{config.simct.algorithm}_training_unexpected",
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
            "phase": f"{config.simct.algorithm}_training",
            "status": "complete",
            "objective": config.simct.algorithm,
            "run_id": config.run_id,
            "config_sha256": config.digest(),
            "dataset_manifest_sha256": dataset.manifest.digest(),
            "dataset_id": dataset.manifest.dataset_id,
            "dataset_revision": dataset.manifest.dataset_revision,
            "start_step": resume.completed_steps,
            "initialization": resume.initialization,
            "source_checkpoint_steps": resume.source_checkpoint_steps,
            "source_checkpoint_run_id": resume.source_checkpoint_run_id,
            "source_student_parameters_sha256": (
                resume.source_student_parameters_sha256
            ),
            "source_dataset_manifest_sha256": (
                resume.source_dataset_manifest_sha256
            ),
            "completed_steps": completed,
            "completed_trainer_calls": completed,
            "completed_optimizer_steps": completed_optimizer_steps,
            "max_steps_unit": config.training.max_steps_unit,
            "target_optimizer_steps": (
                config.training.max_steps
                if config.training.max_steps_unit == "optimizer_update"
                else math.ceil(
                    config.training.max_steps
                    / config.training.gradient_accumulation_steps
                )
            ),
            "micro_metrics_path": str(micro_metrics_path),
            "profile_step": args.profile_step,
            "profile_dir": (
                None if args.profile_dir is None else str(args.profile_dir)
            ),
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
            "remaining_gate": "downstream evaluation under the comparison contract",
        },
        0,
    )


if __name__ == "__main__":
    raise SystemExit(main())
