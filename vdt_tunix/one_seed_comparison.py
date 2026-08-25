"""Assemble a fail-closed one-seed comparison from terminal run evidence."""

from __future__ import annotations

import dataclasses
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from vdt_tunix.config import RunConfig, load_config
from vdt_tunix.evaluation_contract import ComparisonContract, PAPER_BENCHMARKS
from vdt_tunix.generation_contract import GenerationProtocol, load_generation_protocol


class OneSeedComparisonError(RuntimeError):
    """Raised when three run bundles are not comparable."""


@dataclasses.dataclass(frozen=True, slots=True)
class VariantEvidencePaths:
    name: str
    training_config: Path
    training_summary: Path
    generation_summary: Path
    scoring_summary: Path


_VARIANTS = ("sft", "simple_opd", "simct")
_TRAIN_PHASES = {
    "sft": "sft_training",
    "simple_opd": "simple_opd_training",
    "simct": "simct_training",
}


def _load_object(path: Path, context: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise OneSeedComparisonError(f"invalid {context}: {path}") from exc
    if not isinstance(value, dict):
        raise OneSeedComparisonError(f"{context} must be a JSON object: {path}")
    return value


def _require_equal(actual: Any, expected: Any, context: str) -> None:
    if actual != expected:
        raise OneSeedComparisonError(
            f"{context} drifted: observed={actual!r}, expected={expected!r}"
        )


def _require_sha256(value: Any, context: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise OneSeedComparisonError(f"{context} must be a lowercase SHA-256")
    return value


def _require_positive_int(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise OneSeedComparisonError(f"{context} must be a positive integer")
    return value


def _benchmark_map(value: Any, context: str) -> dict[str, dict[str, Any]]:
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise OneSeedComparisonError(f"{context} must be an array of objects")
    result = {str(item.get("benchmark")): item for item in value}
    if len(result) != len(value) or set(result) != set(PAPER_BENCHMARKS):
        raise OneSeedComparisonError(
            f"{context} must contain each paper benchmark exactly once"
        )
    return result


def _prompt_preflight(
    value: Any, protocol: GenerationProtocol, context: str
) -> list[dict[str, Any]]:
    items = _benchmark_map(value, context)
    protocol_benchmarks = {item.name: item for item in protocol.benchmarks}
    for benchmark_name, item in items.items():
        expected = protocol_benchmarks[benchmark_name]
        _require_equal(
            item.get("record_count"),
            expected.record_count,
            f"{context}.{benchmark_name}.record_count",
        )
        _require_equal(
            item.get("max_prompt_tokens_contract"),
            protocol.max_prompt_tokens,
            f"{context}.{benchmark_name}.max_prompt_tokens_contract",
        )
        for key in ("minimum_prompt_tokens", "maximum_prompt_tokens"):
            _require_positive_int(item.get(key), f"{context}.{benchmark_name}.{key}")
        if item["minimum_prompt_tokens"] > item["maximum_prompt_tokens"]:
            raise OneSeedComparisonError(
                f"{context}.{benchmark_name} prompt token bounds are inverted"
            )
        if item["maximum_prompt_tokens"] > protocol.max_prompt_tokens:
            raise OneSeedComparisonError(
                f"{context}.{benchmark_name} exceeds max_prompt_tokens"
            )
    return [items[item.name] for item in protocol.benchmarks]


def _verified_benchmark_specs(
    protocol: GenerationProtocol, evaluation_root: Path
) -> list[dict[str, Any]]:
    specs = []
    for benchmark in protocol.benchmarks:
        manifest = _load_object(
            evaluation_root / benchmark.name / "manifest.json",
            f"{benchmark.name} materialization manifest",
        )
        _require_equal(manifest.get("name"), benchmark.name, f"{benchmark.name}.name")
        _require_equal(
            manifest.get("records_sha256"),
            benchmark.records_sha256,
            f"{benchmark.name}.records_sha256",
        )
        _require_equal(
            manifest.get("record_count"),
            benchmark.record_count,
            f"{benchmark.name}.record_count",
        )
        dataset_id = manifest.get("dataset_id")
        dataset_revision = manifest.get("dataset_revision")
        split = manifest.get("split")
        if not all(
            isinstance(item, str) and item.strip()
            for item in (dataset_id, dataset_revision, split)
        ):
            raise OneSeedComparisonError(
                f"{benchmark.name} dataset identity is incomplete"
            )
        specs.append(
            {
                "name": benchmark.name,
                "dataset_id": dataset_id,
                "dataset_revision": dataset_revision,
                "split": split,
                "records_sha256": benchmark.records_sha256,
                "metric": PAPER_BENCHMARKS[benchmark.name],
                "max_new_tokens": benchmark.max_new_tokens,
            }
        )
    return specs


def _validate_training(
    name: str,
    config: RunConfig,
    summary: Mapping[str, Any],
) -> dict[str, Any]:
    _require_equal(summary.get("status"), "complete", f"{name} training status")
    _require_equal(summary.get("phase"), _TRAIN_PHASES[name], f"{name} training phase")
    _require_equal(summary.get("scientific_evidence"), False, f"{name} training evidence boundary")
    _require_equal(summary.get("run_id"), config.run_id, f"{name} training run_id")
    _require_equal(summary.get("config_sha256"), config.digest(), f"{name} training config")
    steps = _require_positive_int(summary.get("completed_steps"), f"{name}.completed_steps")
    _require_equal(steps, config.training.max_steps, f"{name} optimizer steps")
    student_sha = _require_sha256(
        summary.get("final_student_parameters_sha256"),
        f"{name}.final_student_parameters_sha256",
    )
    dataset_sha = _require_sha256(
        summary.get("dataset_manifest_sha256"),
        f"{name}.dataset_manifest_sha256",
    )
    hardware = summary.get("hardware")
    if not isinstance(hardware, Mapping):
        raise OneSeedComparisonError(f"{name}.hardware must be an object")
    _require_equal(hardware.get("backend"), "tpu", f"{name} training backend")
    _require_equal(hardware.get("device_count"), 8, f"{name} training TPU count")
    _require_equal(hardware.get("v5e_kind_match"), True, f"{name} TPU kind")
    if name == "sft":
        if summary.get("source_student_parameters_sha256") is not None:
            raise OneSeedComparisonError("SFT may not declare an OPD warm start")
        source_sha = None
    else:
        _require_equal(summary.get("objective"), name, f"{name} objective")
        _require_equal(summary.get("initialization"), "warm_start", f"{name} initialization")
        source_sha = _require_sha256(
            summary.get("source_student_parameters_sha256"),
            f"{name}.source_student_parameters_sha256",
        )
    return {
        "steps": steps,
        "student_sha": student_sha,
        "dataset_sha": dataset_sha,
        "source_sha": source_sha,
    }


def _validate_generation_and_scoring(
    *,
    name: str,
    config: RunConfig,
    training: Mapping[str, Any],
    generation: Mapping[str, Any],
    scoring: Mapping[str, Any],
    protocol: GenerationProtocol,
) -> dict[str, Any]:
    protocol_sha = protocol.digest()
    generation_expected = {
        "status": "complete",
        "phase": "native_tunix_generation",
        "variant": name,
        "protocol_id": protocol.protocol_id,
        "protocol_sha256": protocol_sha,
        "training_config_sha256": config.digest(),
        "checkpoint_run_id": config.run_id,
        "checkpoint_steps": training["steps"],
        "student_parameters_sha256": training["student_sha"],
        "training_dataset_manifest_sha256": training["dataset_sha"],
        "scientific_evidence": False,
    }
    for key, expected in generation_expected.items():
        _require_equal(generation.get(key), expected, f"{name} generation {key}")
    generation_hardware = generation.get("hardware")
    if not isinstance(generation_hardware, Mapping):
        raise OneSeedComparisonError(f"{name} generation hardware must be an object")
    _require_equal(
        generation_hardware.get("backend"), "tpu", f"{name} generation backend"
    )
    _require_equal(
        generation_hardware.get("device_count"), 8, f"{name} generation TPU count"
    )
    generation_benchmarks = _benchmark_map(
        generation.get("benchmarks"), f"{name} generation benchmarks"
    )
    protocol_benchmarks = {item.name: item for item in protocol.benchmarks}
    for benchmark_name, item in generation_benchmarks.items():
        expected = protocol_benchmarks[benchmark_name]
        _require_equal(
            item.get("record_count"),
            expected.record_count,
            f"{name}.{benchmark_name} generated count",
        )
        _require_sha256(
            item.get("predictions_sha256"), f"{name}.{benchmark_name}.predictions_sha256"
        )

    scoring_expected = {
        "status": "complete",
        "phase": "paper_released_evaluator_scoring",
        "variant": name,
        "protocol_id": protocol.protocol_id,
        "protocol_sha256": protocol_sha,
        "checkpoint_sha256": training["student_sha"],
        "evaluator_revision": protocol.evaluator_revision,
        "evaluator_classification": "paper-released; not official benchmark harness",
        "evaluation_evidence": True,
        "scientific_evidence": True,
        "paper_reproduction": False,
    }
    for key, expected in scoring_expected.items():
        _require_equal(scoring.get(key), expected, f"{name} scoring {key}")
    evaluator_id = scoring.get("evaluator_id")
    evaluator_source_sha = _require_sha256(
        scoring.get("evaluator_source_sha256"), f"{name}.evaluator_source_sha256"
    )
    if not isinstance(evaluator_id, str) or not evaluator_id.strip():
        raise OneSeedComparisonError(f"{name}.evaluator_id is missing")
    environment = scoring.get("environment")
    if not isinstance(environment, Mapping):
        raise OneSeedComparisonError(f"{name}.environment must be an object")

    scoring_benchmarks = _benchmark_map(
        scoring.get("benchmarks"), f"{name} scoring benchmarks"
    )
    scores: dict[str, float] = {}
    for benchmark_name, item in scoring_benchmarks.items():
        expected = protocol_benchmarks[benchmark_name]
        _require_equal(
            item.get("metric"),
            PAPER_BENCHMARKS[benchmark_name],
            f"{name}.{benchmark_name}.metric",
        )
        total = _require_positive_int(item.get("total"), f"{name}.{benchmark_name}.total")
        _require_equal(total, expected.record_count, f"{name}.{benchmark_name} scored count")
        correct = item.get("correct")
        if isinstance(correct, bool) or not isinstance(correct, int) or not 0 <= correct <= total:
            raise OneSeedComparisonError(f"{name}.{benchmark_name}.correct is invalid")
        score = item.get("score")
        if (
            isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(float(score))
        ):
            raise OneSeedComparisonError(f"{name}.{benchmark_name}.score is invalid")
        if not math.isclose(float(score), correct / total, rel_tol=0.0, abs_tol=1e-12):
            raise OneSeedComparisonError(
                f"{name}.{benchmark_name}.score disagrees with correct/total"
            )
        _require_sha256(
            item.get("scored_predictions_sha256"),
            f"{name}.{benchmark_name}.scored_predictions_sha256",
        )
        scores[benchmark_name] = float(score)
    prompt_preflight = _prompt_preflight(
        generation.get("prompt_preflight"),
        protocol,
        f"{name} prompt preflight",
    )
    return {
        "scores": scores,
        "environment": dict(environment),
        "evaluator_id": evaluator_id,
        "evaluator_source_sha256": evaluator_source_sha,
        "prompt_preflight_sha256": _require_sha256(
            generation.get("prompt_preflight_sha256"),
            f"{name}.prompt_preflight_sha256",
        ),
        "prompt_preflight": prompt_preflight,
    }


def assemble_one_seed_comparison(
    *,
    comparison_id: str,
    generation_protocol_path: str | Path,
    evaluation_root: str | Path,
    variants: tuple[VariantEvidencePaths, ...],
) -> dict[str, Any]:
    """Validate and combine exactly SFT, SimpleOPD, and SimCT evidence."""

    by_name = {item.name: item for item in variants}
    if len(by_name) != len(variants) or set(by_name) != set(_VARIANTS):
        raise OneSeedComparisonError(
            "comparison requires exactly sft, simple_opd, and simct evidence"
        )
    protocol = load_generation_protocol(generation_protocol_path)
    benchmark_specs = _verified_benchmark_specs(protocol, Path(evaluation_root))

    validated: dict[str, dict[str, Any]] = {}
    comparison_variants = []
    for name in _VARIANTS:
        paths = by_name[name]
        config = load_config(paths.training_config)
        training_summary = _load_object(paths.training_summary, f"{name} training summary")
        generation_summary = _load_object(paths.generation_summary, f"{name} generation summary")
        scoring_summary = _load_object(paths.scoring_summary, f"{name} scoring summary")
        training = _validate_training(name, config, training_summary)
        downstream = _validate_generation_and_scoring(
            name=name,
            config=config,
            training=training,
            generation=generation_summary,
            scoring=scoring_summary,
            protocol=protocol,
        )
        validated[name] = {
            "config": config,
            "training": training,
            "downstream": downstream,
        }
        comparison_variants.append(
            {
                "name": name,
                "objective": name,
                "run_id": config.run_id,
                "completed_steps": training["steps"],
                "student_model_revision": config.student.model_revision,
                "student_tokenizer_revision": config.student.tokenizer_revision,
                "training_dataset_manifest_sha256": training["dataset_sha"],
                "student_parameters_sha256": training["student_sha"],
                "source_student_parameters_sha256": training["source_sha"],
            }
        )

    sft_sha = validated["sft"]["training"]["student_sha"]
    for name in ("simple_opd", "simct"):
        _require_equal(
            validated[name]["training"]["source_sha"],
            sft_sha,
            f"{name} exact SFT warm start",
        )
    shared_fields = (
        "environment",
        "evaluator_id",
        "evaluator_source_sha256",
        "prompt_preflight_sha256",
        "prompt_preflight",
    )
    for field in shared_fields:
        expected = validated["sft"]["downstream"][field]
        for name in ("simple_opd", "simct"):
            _require_equal(
                validated[name]["downstream"][field],
                expected,
                f"shared {field} ({name})",
            )

    contract_payload = {
        "contract_version": 1,
        "comparison_id": comparison_id,
        "protocol": {
            "paper_revision": protocol.paper_revision,
            "prompt_protocol": protocol.prompt_protocol,
            "evaluator_id": validated["sft"]["downstream"]["evaluator_id"],
            "evaluator_revision": protocol.evaluator_revision,
            "temperature": protocol.temperature,
            "top_p": protocol.top_p,
            "samples_per_instance": protocol.samples_per_instance,
            "run_seeds": [protocol.seed],
            "scope": "one_seed_screen",
            "benchmarks": benchmark_specs,
        },
        "variants": comparison_variants,
    }
    contract = ComparisonContract.from_mapping(contract_payload)
    scores = {
        name: validated[name]["downstream"]["scores"] for name in _VARIANTS
    }
    deltas = {
        name: {
            benchmark: scores[name][benchmark] - scores["sft"][benchmark]
            for benchmark in sorted(PAPER_BENCHMARKS)
        }
        for name in ("simple_opd", "simct")
    }
    return {
        "contract_version": 1,
        "status": "complete",
        "phase": "one_seed_comparison",
        "comparison_id": comparison_id,
        "comparison_contract": contract.to_dict(),
        "comparison_contract_sha256": contract.digest(),
        "protocol_id": protocol.protocol_id,
        "protocol_sha256": protocol.digest(),
        "scores": scores,
        "deltas_vs_sft": deltas,
        "shared_scoring_environment": validated["sft"]["downstream"]["environment"],
        "shared_prompt_preflight_sha256": validated["sft"]["downstream"][
            "prompt_preflight_sha256"
        ],
        "evaluation_evidence": True,
        "scientific_evidence": True,
        "paper_reproduction": False,
        "scope": "bounded public-substitute one-seed screen",
        "limitations": [
            "one seed only",
            "public substitute training corpus, not the unavailable paper 10K corpus",
            "paper-released evaluator, not official Math-Verify or LiveCodeBench harnesses",
        ],
    }


def comparison_markdown(summary: Mapping[str, Any]) -> str:
    """Render the compact score table without weakening the JSON contract."""

    scores = summary["scores"]
    deltas = summary["deltas_vs_sft"]
    lines = [
        f"# {summary['comparison_id']}",
        "",
        "| Benchmark | SFT | SimpleOPD | SimCT | SimpleOPD - SFT | SimCT - SFT |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for benchmark in sorted(PAPER_BENCHMARKS):
        lines.append(
            "| "
            + " | ".join(
                (
                    benchmark,
                    f"{scores['sft'][benchmark]:.6f}",
                    f"{scores['simple_opd'][benchmark]:.6f}",
                    f"{scores['simct'][benchmark]:.6f}",
                    f"{deltas['simple_opd'][benchmark]:+.6f}",
                    f"{deltas['simct'][benchmark]:+.6f}",
                )
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "Scientific evidence: bounded public-substitute one-seed screen.",
            "Paper reproduction: false.",
            "",
        ]
    )
    return "\n".join(lines)
