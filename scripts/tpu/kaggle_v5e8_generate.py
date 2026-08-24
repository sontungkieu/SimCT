#!/usr/bin/env python3
"""Resume-safe native Tunix generation for the four-paper benchmark screen."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vdt_tunix.config import ConfigError, load_config
from vdt_tunix.evaluation_runtime import (
    EvaluationRuntimeError,
    NativeTunixGenerator,
    restore_student_for_inference,
)
from vdt_tunix.generation_contract import (
    GenerationContractError,
    GenerationExample,
    GenerationProtocol,
    iter_generation_examples,
    load_generation_protocol,
    stable_batch_seed,
)

EX_UNAVAILABLE = 69
EX_CONFIG = 78


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _canonical_row(payload: dict[str, Any]) -> bytes:
    return (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        + "\n"
    ).encode("utf-8")


def _variant_matches(variant: str, config) -> bool:
    if variant == "sft":
        return config.run_id == "vdt-public-sft-screen"
    return config.simct.algorithm == variant


def _read_batch(
    path: Path,
    *,
    expected: tuple[GenerationExample, ...],
    protocol: GenerationProtocol,
    protocol_sha256: str,
    variant: str,
    checkpoint_sha256: str,
    batch_index: int,
) -> tuple[dict[str, Any], ...]:
    rows = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise GenerationContractError(f"invalid batch artifact: {path}") from exc
            if not isinstance(row, dict):
                raise GenerationContractError(f"batch row must be an object: {path}")
            rows.append(row)
    if len(rows) != len(expected):
        raise GenerationContractError(f"batch row count drifted: {path}")
    expected_keys = {
        "contract_version",
        "protocol_sha256",
        "variant",
        "checkpoint_sha256",
        "benchmark",
        "instance_id",
        "source_index",
        "prompt_sha256",
        "batch_index",
        "batch_position",
        "seed",
        "temperature",
        "top_p",
        "max_new_tokens",
        "prompt_tokens",
        "completion_tokens",
        "truncated",
        "completion",
    }
    benchmark = next(
        item for item in protocol.benchmarks if item.name == expected[0].benchmark
    )
    seed = stable_batch_seed(protocol, expected[0].benchmark, batch_index)
    for position, (row, example) in enumerate(zip(rows, expected, strict=True)):
        if set(row) != expected_keys:
            raise GenerationContractError(
                f"batch schema drift in {path}: "
                f"missing={sorted(expected_keys - set(row))}, "
                f"unsupported={sorted(set(row) - expected_keys)}"
            )
        static = {
            "contract_version": 1,
            "protocol_sha256": protocol_sha256,
            "variant": variant,
            "checkpoint_sha256": checkpoint_sha256,
            "benchmark": example.benchmark,
            "instance_id": example.instance_id,
            "source_index": example.source_index,
            "prompt_sha256": hashlib.sha256(
                example.prompt.encode("utf-8")
            ).hexdigest(),
            "batch_index": batch_index,
            "batch_position": position,
            "seed": seed,
            "temperature": protocol.temperature,
            "top_p": protocol.top_p,
            "max_new_tokens": benchmark.max_new_tokens,
        }
        drift = {
            key: (row.get(key), value)
            for key, value in static.items()
            if row.get(key) != value
        }
        if drift:
            raise GenerationContractError(f"batch identity drift in {path}: {drift}")
        if not isinstance(row.get("completion"), str):
            raise GenerationContractError(f"batch completion is not text: {path}")
        for key, upper in (
            ("prompt_tokens", protocol.max_prompt_tokens),
            ("completion_tokens", benchmark.max_new_tokens),
        ):
            value = row.get(key)
            if isinstance(value, bool) or not isinstance(value, int):
                raise GenerationContractError(
                    f"batch {key} is not an integer: {path}"
                )
            lower = 1 if key == "prompt_tokens" else 0
            if value < lower or value > upper:
                raise GenerationContractError(
                    f"batch {key} is outside [{lower}, {upper}]: {path}"
                )
        truncated = row.get("truncated")
        if not isinstance(truncated, bool) or truncated != (
            row["completion_tokens"] == benchmark.max_new_tokens
        ):
            raise GenerationContractError(
                f"batch truncation flag is inconsistent: {path}"
            )
    return tuple(rows)


def _write_batch(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        for row in rows:
            stream.write(_canonical_row(row))
        stream.flush()
    temporary.replace(path)


def _finalize_predictions(
    path: Path,
    batch_paths: list[Path],
) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as output:
        for batch_path in batch_paths:
            with batch_path.open("rb") as source:
                for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
                    output.write(block)
        output.flush()
    temporary.replace(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-config", required=True, type=Path)
    parser.add_argument("--generation-protocol", required=True, type=Path)
    parser.add_argument("--evaluation-root", required=True, type=Path)
    parser.add_argument("--checkpoint-root", required=True, type=Path)
    parser.add_argument("--variant", choices=("sft", "simple_opd", "simct"), required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)

    try:
        config = load_config(args.training_config)
        protocol = load_generation_protocol(args.generation_protocol)
        if not _variant_matches(args.variant, config):
            raise GenerationContractError("variant and training config disagree")
        protocol_sha256 = protocol.digest()
        examples_by_name = {
            benchmark.name: tuple(
                iter_generation_examples(protocol, args.evaluation_root, benchmark)
            )
            for benchmark in protocol.benchmarks
        }
    except (ConfigError, GenerationContractError, OSError) as exc:
        _atomic_json(
            args.output_dir / "generation_summary.json",
            {
                "status": "blocked",
                "phase": "generation_configuration_and_data",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "scientific_evidence": False,
            },
        )
        return EX_CONFIG

    try:
        restored = restore_student_for_inference(config, args.checkpoint_root)
        generator = NativeTunixGenerator(
            restored,
            max_prompt_tokens=protocol.max_prompt_tokens,
            max_completion_tokens=max(
                benchmark.max_new_tokens for benchmark in protocol.benchmarks
            ),
        )
        prompt_preflight = []
        for benchmark in protocol.benchmarks:
            examples = examples_by_name[benchmark.name]
            prompt_lengths = []
            for start in range(0, len(examples), protocol.batch_size):
                prompt_lengths.extend(
                    generator.prompt_lengths(
                        [
                            item.prompt
                            for item in examples[start : start + protocol.batch_size]
                        ]
                    )
                )
            if len(prompt_lengths) != len(examples):
                raise EvaluationRuntimeError(
                    f"prompt preflight count drifted for {benchmark.name}"
                )
            prompt_preflight.append(
                {
                    "benchmark": benchmark.name,
                    "record_count": len(prompt_lengths),
                    "minimum_prompt_tokens": min(prompt_lengths),
                    "maximum_prompt_tokens": max(prompt_lengths),
                    "max_prompt_tokens_contract": protocol.max_prompt_tokens,
                }
            )
        prompt_preflight_path = args.output_dir / "prompt_preflight.json"
        _atomic_json(
            prompt_preflight_path,
            {
                "status": "passed",
                "protocol_sha256": protocol_sha256,
                "benchmarks": prompt_preflight,
                "scientific_evidence": False,
            },
        )
        benchmark_summaries = []
        for benchmark in protocol.benchmarks:
            examples = examples_by_name[benchmark.name]
            benchmark_dir = args.output_dir / benchmark.name
            batches_dir = benchmark_dir / "batches"
            batch_paths = []
            completion_tokens_total = 0
            truncated_count = 0
            batch_count = math.ceil(len(examples) / protocol.batch_size)
            for batch_index in range(batch_count):
                start = batch_index * protocol.batch_size
                expected = examples[start : start + protocol.batch_size]
                batch_path = batches_dir / f"batch_{batch_index:06d}.jsonl"
                batch_paths.append(batch_path)
                if batch_path.is_file():
                    rows = _read_batch(
                        batch_path,
                        expected=expected,
                        protocol=protocol,
                        protocol_sha256=protocol_sha256,
                        variant=args.variant,
                        checkpoint_sha256=restored.student_parameters_sha256,
                        batch_index=batch_index,
                    )
                else:
                    padded = list(expected)
                    while len(padded) < protocol.batch_size:
                        padded.append(expected[-1])
                    batch_seed = stable_batch_seed(
                        protocol, benchmark.name, batch_index
                    )
                    texts, prompt_lengths, completion_lengths = generator.generate(
                        [item.prompt for item in padded],
                        max_new_tokens=benchmark.max_new_tokens,
                        temperature=protocol.temperature,
                        top_p=protocol.top_p,
                        seed=batch_seed,
                    )
                    new_rows = []
                    for position, example in enumerate(expected):
                        completion_tokens = completion_lengths[position]
                        new_rows.append(
                            {
                                "contract_version": 1,
                                "protocol_sha256": protocol_sha256,
                                "variant": args.variant,
                                "checkpoint_sha256": restored.student_parameters_sha256,
                                "benchmark": benchmark.name,
                                "instance_id": example.instance_id,
                                "source_index": example.source_index,
                                "prompt_sha256": hashlib.sha256(
                                    example.prompt.encode("utf-8")
                                ).hexdigest(),
                                "batch_index": batch_index,
                                "batch_position": position,
                                "seed": batch_seed,
                                "temperature": protocol.temperature,
                                "top_p": protocol.top_p,
                                "max_new_tokens": benchmark.max_new_tokens,
                                "prompt_tokens": prompt_lengths[position],
                                "completion_tokens": completion_tokens,
                                "truncated": completion_tokens
                                == benchmark.max_new_tokens,
                                "completion": texts[position],
                            }
                        )
                    _write_batch(batch_path, new_rows)
                    rows = tuple(new_rows)
                completion_tokens_total += sum(
                    int(row["completion_tokens"]) for row in rows
                )
                truncated_count += sum(bool(row["truncated"]) for row in rows)
            predictions = benchmark_dir / "predictions.jsonl"
            _finalize_predictions(predictions, batch_paths)
            benchmark_summary = {
                "benchmark": benchmark.name,
                "record_count": len(examples),
                "batch_count": batch_count,
                "completion_tokens": completion_tokens_total,
                "truncated_count": truncated_count,
                "predictions_path": str(predictions),
                "predictions_sha256": _sha256(predictions),
            }
            _atomic_json(benchmark_dir / "generation_summary.json", benchmark_summary)
            benchmark_summaries.append(benchmark_summary)
    except Exception as exc:  # noqa: BLE001 - persist a fail-closed resume summary
        _atomic_json(
            args.output_dir / "generation_summary.json",
            {
                "status": "blocked",
                "phase": "native_tunix_generation",
                "variant": args.variant,
                "protocol_sha256": protocol_sha256,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "scientific_evidence": False,
            },
        )
        return EX_UNAVAILABLE

    summary = {
        "status": "complete",
        "phase": "native_tunix_generation",
        "variant": args.variant,
        "protocol_id": protocol.protocol_id,
        "protocol_sha256": protocol_sha256,
        "training_config_sha256": config.digest(),
        "checkpoint_run_id": restored.checkpoint_run_id,
        "checkpoint_steps": restored.checkpoint_steps,
        "student_parameters_sha256": restored.student_parameters_sha256,
        "training_dataset_manifest_sha256": restored.dataset_manifest_sha256,
        "hardware": restored.hardware,
        "prompt_preflight_sha256": _sha256(prompt_preflight_path),
        "prompt_preflight": prompt_preflight,
        "benchmarks": benchmark_summaries,
        "scientific_evidence": False,
        "remaining_gate": "official benchmark scoring and comparison contract",
    }
    _atomic_json(args.output_dir / "generation_summary.json", summary)
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
