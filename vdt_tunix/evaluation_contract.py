"""Fail-closed comparison contract for SFT, SimpleOPD, and SimCT.

Training success is not evaluation evidence.  This module records the exact
checkpoint lineage and one shared downstream protocol needed before scores
from the three variants may be compared.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


COMPARISON_CONTRACT_VERSION = 1
PAPER_BENCHMARKS = {
    "gsm8k": "accuracy",
    "math500": "accuracy",
    "mbpp": "pass@1",
    "live-code-bench-v6": "pass@1",
}


class EvaluationContractError(RuntimeError):
    """Raised when comparison identity or fairness cannot be proven."""


def _strict_keys(
    value: Mapping[str, Any], *, context: str, required: set[str]
) -> None:
    missing = sorted(required - value.keys())
    extra = sorted(value.keys() - required)
    if missing or extra:
        raise EvaluationContractError(
            f"{context} key mismatch: missing={missing}, unsupported={extra}"
        )


def _string(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EvaluationContractError(f"{context} must be a non-empty string")
    return value


def _sha256(value: Any, context: str) -> str:
    result = _string(value, context)
    if len(result) != 64 or any(char not in "0123456789abcdef" for char in result):
        raise EvaluationContractError(f"{context} must be a lowercase SHA-256")
    return result


def _immutable_revision(value: Any, context: str) -> str:
    result = _string(value, context)
    if result.lower() in {"main", "master", "head", "latest"}:
        raise EvaluationContractError(f"{context} must be immutable")
    return result


@dataclasses.dataclass(frozen=True, slots=True)
class BenchmarkSpec:
    name: str
    dataset_id: str
    dataset_revision: str
    split: str
    records_sha256: str
    metric: str
    max_new_tokens: int

    def __post_init__(self) -> None:
        if self.name not in PAPER_BENCHMARKS:
            raise EvaluationContractError(f"unsupported paper benchmark: {self.name}")
        _string(self.dataset_id, f"{self.name}.dataset_id")
        _immutable_revision(
            self.dataset_revision, f"{self.name}.dataset_revision"
        )
        _string(self.split, f"{self.name}.split")
        _sha256(self.records_sha256, f"{self.name}.records_sha256")
        if self.metric != PAPER_BENCHMARKS[self.name]:
            raise EvaluationContractError(
                f"{self.name}.metric must be {PAPER_BENCHMARKS[self.name]}"
            )
        if self.max_new_tokens < 1:
            raise EvaluationContractError("max_new_tokens must be positive")

    @classmethod
    def from_mapping(cls, value: Any) -> "BenchmarkSpec":
        if not isinstance(value, Mapping):
            raise EvaluationContractError("benchmark must be an object")
        required = {
            "name",
            "dataset_id",
            "dataset_revision",
            "split",
            "records_sha256",
            "metric",
            "max_new_tokens",
        }
        _strict_keys(value, context="benchmark", required=required)
        max_tokens = value["max_new_tokens"]
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int):
            raise EvaluationContractError("max_new_tokens must be an integer")
        return cls(
            name=_string(value["name"], "benchmark.name"),
            dataset_id=_string(value["dataset_id"], "benchmark.dataset_id"),
            dataset_revision=_immutable_revision(
                value["dataset_revision"], "benchmark.dataset_revision"
            ),
            split=_string(value["split"], "benchmark.split"),
            records_sha256=_sha256(
                value["records_sha256"], "benchmark.records_sha256"
            ),
            metric=_string(value["metric"], "benchmark.metric"),
            max_new_tokens=max_tokens,
        )


@dataclasses.dataclass(frozen=True, slots=True)
class EvaluationProtocol:
    paper_revision: str
    prompt_protocol: str
    evaluator_id: str
    evaluator_revision: str
    temperature: float
    top_p: float
    samples_per_instance: int
    run_seeds: tuple[int, ...]
    scope: str
    benchmarks: tuple[BenchmarkSpec, ...]

    def __post_init__(self) -> None:
        _immutable_revision(self.paper_revision, "paper_revision")
        _string(self.prompt_protocol, "prompt_protocol")
        _string(self.evaluator_id, "evaluator_id")
        _immutable_revision(self.evaluator_revision, "evaluator_revision")
        if self.temperature != 0.6 or self.top_p != 0.95:
            raise EvaluationContractError(
                "paper comparison requires temperature=0.6 and top_p=0.95"
            )
        if self.samples_per_instance != 1:
            raise EvaluationContractError(
                "paper comparison requires one completion per instance"
            )
        if not self.run_seeds or len(set(self.run_seeds)) != len(self.run_seeds):
            raise EvaluationContractError("run_seeds must be non-empty and unique")
        if any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in self.run_seeds):
            raise EvaluationContractError("run seeds must be non-negative integers")
        expected_runs = {"one_seed_screen": 1, "paper_five_run": 5}
        if self.scope not in expected_runs:
            raise EvaluationContractError(
                "scope must be one_seed_screen or paper_five_run"
            )
        if len(self.run_seeds) != expected_runs[self.scope]:
            raise EvaluationContractError(
                f"{self.scope} requires {expected_runs[self.scope]} run seed(s)"
            )
        names = [benchmark.name for benchmark in self.benchmarks]
        if len(names) != len(set(names)):
            raise EvaluationContractError("benchmark names must be unique")
        if set(names) != set(PAPER_BENCHMARKS):
            raise EvaluationContractError(
                "comparison must include GSM8K, MATH-500, MBPP, and LCB-v6"
            )

    @classmethod
    def from_mapping(cls, value: Any) -> "EvaluationProtocol":
        if not isinstance(value, Mapping):
            raise EvaluationContractError("evaluation protocol must be an object")
        required = {
            "paper_revision",
            "prompt_protocol",
            "evaluator_id",
            "evaluator_revision",
            "temperature",
            "top_p",
            "samples_per_instance",
            "run_seeds",
            "scope",
            "benchmarks",
        }
        _strict_keys(value, context="evaluation protocol", required=required)
        temperature = value["temperature"]
        top_p = value["top_p"]
        samples = value["samples_per_instance"]
        seeds = value["run_seeds"]
        benchmarks = value["benchmarks"]
        if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
            raise EvaluationContractError("temperature must be numeric")
        if isinstance(top_p, bool) or not isinstance(top_p, (int, float)):
            raise EvaluationContractError("top_p must be numeric")
        if isinstance(samples, bool) or not isinstance(samples, int):
            raise EvaluationContractError("samples_per_instance must be an integer")
        if not isinstance(seeds, Sequence) or isinstance(seeds, (str, bytes)):
            raise EvaluationContractError("run_seeds must be an array")
        if not isinstance(benchmarks, Sequence) or isinstance(
            benchmarks, (str, bytes)
        ):
            raise EvaluationContractError("benchmarks must be an array")
        return cls(
            paper_revision=_immutable_revision(
                value["paper_revision"], "paper_revision"
            ),
            prompt_protocol=_string(value["prompt_protocol"], "prompt_protocol"),
            evaluator_id=_string(value["evaluator_id"], "evaluator_id"),
            evaluator_revision=_immutable_revision(
                value["evaluator_revision"], "evaluator_revision"
            ),
            temperature=float(temperature),
            top_p=float(top_p),
            samples_per_instance=samples,
            run_seeds=tuple(seeds),
            scope=_string(value["scope"], "scope"),
            benchmarks=tuple(BenchmarkSpec.from_mapping(item) for item in benchmarks),
        )


@dataclasses.dataclass(frozen=True, slots=True)
class VariantCheckpoint:
    name: str
    objective: str
    run_id: str
    completed_steps: int
    student_model_revision: str
    student_tokenizer_revision: str
    training_dataset_manifest_sha256: str
    student_parameters_sha256: str
    source_student_parameters_sha256: str | None

    def __post_init__(self) -> None:
        expected_objectives = {
            "sft": "sft",
            "simple_opd": "simple_opd",
            "simct": "simct",
        }
        if self.name not in expected_objectives:
            raise EvaluationContractError(f"unsupported variant: {self.name}")
        if self.objective != expected_objectives[self.name]:
            raise EvaluationContractError(
                f"variant {self.name} requires objective {expected_objectives[self.name]}"
            )
        _string(self.run_id, f"{self.name}.run_id")
        if self.completed_steps < 1:
            raise EvaluationContractError("completed_steps must be positive")
        _immutable_revision(
            self.student_model_revision, f"{self.name}.student_model_revision"
        )
        _immutable_revision(
            self.student_tokenizer_revision,
            f"{self.name}.student_tokenizer_revision",
        )
        _sha256(
            self.training_dataset_manifest_sha256,
            f"{self.name}.training_dataset_manifest_sha256",
        )
        _sha256(
            self.student_parameters_sha256,
            f"{self.name}.student_parameters_sha256",
        )
        if self.name == "sft":
            if self.source_student_parameters_sha256 is not None:
                raise EvaluationContractError("SFT may not declare an OPD warm start")
        elif self.source_student_parameters_sha256 is None:
            raise EvaluationContractError(
                f"{self.name} must identify its SFT warm-start parameter hash"
            )
        else:
            _sha256(
                self.source_student_parameters_sha256,
                f"{self.name}.source_student_parameters_sha256",
            )

    @classmethod
    def from_mapping(cls, value: Any) -> "VariantCheckpoint":
        if not isinstance(value, Mapping):
            raise EvaluationContractError("variant checkpoint must be an object")
        required = {
            "name",
            "objective",
            "run_id",
            "completed_steps",
            "student_model_revision",
            "student_tokenizer_revision",
            "training_dataset_manifest_sha256",
            "student_parameters_sha256",
            "source_student_parameters_sha256",
        }
        _strict_keys(value, context="variant checkpoint", required=required)
        completed = value["completed_steps"]
        if isinstance(completed, bool) or not isinstance(completed, int):
            raise EvaluationContractError("completed_steps must be an integer")
        source_sha = value["source_student_parameters_sha256"]
        if source_sha is not None:
            source_sha = _sha256(source_sha, "source_student_parameters_sha256")
        return cls(
            name=_string(value["name"], "variant.name"),
            objective=_string(value["objective"], "variant.objective"),
            run_id=_string(value["run_id"], "variant.run_id"),
            completed_steps=completed,
            student_model_revision=_immutable_revision(
                value["student_model_revision"], "student_model_revision"
            ),
            student_tokenizer_revision=_immutable_revision(
                value["student_tokenizer_revision"], "student_tokenizer_revision"
            ),
            training_dataset_manifest_sha256=_sha256(
                value["training_dataset_manifest_sha256"],
                "training_dataset_manifest_sha256",
            ),
            student_parameters_sha256=_sha256(
                value["student_parameters_sha256"], "student_parameters_sha256"
            ),
            source_student_parameters_sha256=source_sha,
        )


@dataclasses.dataclass(frozen=True, slots=True)
class ComparisonContract:
    contract_version: int
    comparison_id: str
    protocol: EvaluationProtocol
    variants: tuple[VariantCheckpoint, ...]

    def __post_init__(self) -> None:
        if self.contract_version != COMPARISON_CONTRACT_VERSION:
            raise EvaluationContractError("unsupported comparison contract version")
        _string(self.comparison_id, "comparison_id")
        by_name = {variant.name: variant for variant in self.variants}
        if len(by_name) != len(self.variants) or set(by_name) != {
            "sft",
            "simple_opd",
            "simct",
        }:
            raise EvaluationContractError(
                "comparison requires exactly SFT, SimpleOPD, and SimCT"
            )
        sft = by_name["sft"]
        for name in ("simple_opd", "simct"):
            variant = by_name[name]
            if variant.source_student_parameters_sha256 != sft.student_parameters_sha256:
                raise EvaluationContractError(
                    f"{name} did not start from the exact evaluated SFT parameters"
                )
            if (
                variant.student_model_revision != sft.student_model_revision
                or variant.student_tokenizer_revision
                != sft.student_tokenizer_revision
            ):
                raise EvaluationContractError(
                    f"{name} student model/tokenizer differs from SFT"
                )
        if (
            by_name["simple_opd"].training_dataset_manifest_sha256
            != by_name["simct"].training_dataset_manifest_sha256
        ):
            raise EvaluationContractError(
                "SimpleOPD and SimCT must use the same OPD prompt manifest"
            )
        if by_name["simple_opd"].completed_steps != by_name["simct"].completed_steps:
            raise EvaluationContractError(
                "SimpleOPD and SimCT must use the same number of optimizer steps"
            )

    @classmethod
    def from_mapping(cls, value: Any) -> "ComparisonContract":
        if not isinstance(value, Mapping):
            raise EvaluationContractError("comparison contract must be an object")
        required = {"contract_version", "comparison_id", "protocol", "variants"}
        _strict_keys(value, context="comparison contract", required=required)
        version = value["contract_version"]
        variants = value["variants"]
        if isinstance(version, bool) or not isinstance(version, int):
            raise EvaluationContractError("contract_version must be an integer")
        if not isinstance(variants, Sequence) or isinstance(variants, (str, bytes)):
            raise EvaluationContractError("variants must be an array")
        return cls(
            contract_version=version,
            comparison_id=_string(value["comparison_id"], "comparison_id"),
            protocol=EvaluationProtocol.from_mapping(value["protocol"]),
            variants=tuple(VariantCheckpoint.from_mapping(item) for item in variants),
        )

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def digest(self) -> str:
        encoded = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


def load_comparison_contract(path: str | Path) -> ComparisonContract:
    contract_path = Path(path)
    try:
        payload = json.loads(contract_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise EvaluationContractError("comparison contract is invalid JSON") from exc
    return ComparisonContract.from_mapping(payload)
