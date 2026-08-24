"""Pinned generation protocol for the SimCT one-seed evaluation screen."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from vdt_tunix.evaluation_data import (
    PINNED_BENCHMARKS,
    load_verified_materialization,
)

GENERATION_CONTRACT_VERSION = 1
GEMMA_MERGED_SYSTEM_TEMPLATE = "gemma2-merged-system-zero-shot-v1"

GSM8K_SYSTEM_PROMPT = (
    "Please reason step by step, and put your final answer after ####."
)
MATH500_SYSTEM_PROMPT = (
    "Please reason step by step, and put your final answer within \\boxed{}."
)
MBPP_SYSTEM_PROMPT = (
    "You are a Python programming assistant. Write a Python function to solve "
    "the given task. Your code must pass the provided test cases. "
    "Only output the Python code, without any explanation."
)
LCB_SYSTEM_PROMPT = (
    "You are a Python programming assistant. Solve the given competitive "
    "programming problem. Read from standard input and write to standard output. "
    "Only output the Python code, without any explanation."
)


class GenerationContractError(RuntimeError):
    """Raised when generation inputs or resume artifacts drift."""


def _sha256(value: str, context: str) -> str:
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise GenerationContractError(f"{context} must be a lowercase SHA-256")
    return value


def _strict_keys(
    value: Mapping[str, Any], *, context: str, required: set[str]
) -> None:
    missing = sorted(required - value.keys())
    extra = sorted(value.keys() - required)
    if missing or extra:
        raise GenerationContractError(
            f"{context} key mismatch: missing={missing}, unsupported={extra}"
        )


@dataclasses.dataclass(frozen=True, slots=True)
class GenerationBenchmark:
    name: str
    records_sha256: str
    record_count: int
    max_new_tokens: int

    def __post_init__(self) -> None:
        pinned = {source.name: source for source in PINNED_BENCHMARKS}
        if self.name not in pinned:
            raise GenerationContractError(f"unsupported benchmark: {self.name}")
        _sha256(self.records_sha256, f"{self.name}.records_sha256")
        if self.record_count != pinned[self.name].expected_count:
            raise GenerationContractError(
                f"{self.name}.record_count does not match its pinned source"
            )
        expected_tokens = 2048 if self.name == "mbpp" else 4096
        if self.max_new_tokens != expected_tokens:
            raise GenerationContractError(
                f"{self.name}.max_new_tokens must be {expected_tokens}"
            )

    @classmethod
    def from_mapping(cls, value: Any) -> GenerationBenchmark:
        if not isinstance(value, Mapping):
            raise GenerationContractError("generation benchmark must be an object")
        _strict_keys(
            value,
            context="generation benchmark",
            required={
                "name",
                "records_sha256",
                "record_count",
                "max_new_tokens",
            },
        )
        if not isinstance(value["name"], str):
            raise GenerationContractError("benchmark.name must be a string")
        if not isinstance(value["records_sha256"], str):
            raise GenerationContractError("records_sha256 must be a string")
        for key in ("record_count", "max_new_tokens"):
            if isinstance(value[key], bool) or not isinstance(value[key], int):
                raise GenerationContractError(f"benchmark.{key} must be an integer")
        return cls(
            name=value["name"],
            records_sha256=value["records_sha256"],
            record_count=value["record_count"],
            max_new_tokens=value["max_new_tokens"],
        )


@dataclasses.dataclass(frozen=True, slots=True)
class GenerationProtocol:
    contract_version: int
    protocol_id: str
    paper_revision: str
    prompt_protocol: str
    evaluator_revision: str
    seed: int
    temperature: float
    top_p: float
    samples_per_instance: int
    batch_size: int
    max_prompt_tokens: int
    benchmarks: tuple[GenerationBenchmark, ...]

    def __post_init__(self) -> None:
        if self.contract_version != GENERATION_CONTRACT_VERSION:
            raise GenerationContractError("unsupported generation contract version")
        if not all(
            isinstance(value, str) and value.strip()
            for value in (
                self.protocol_id,
                self.paper_revision,
                self.prompt_protocol,
                self.evaluator_revision,
            )
        ):
            raise GenerationContractError("generation identity strings are required")
        if self.prompt_protocol != GEMMA_MERGED_SYSTEM_TEMPLATE:
            raise GenerationContractError(
                f"prompt_protocol must be {GEMMA_MERGED_SYSTEM_TEMPLATE}"
            )
        if self.seed < 0:
            raise GenerationContractError("seed must be non-negative")
        if self.temperature != 0.6 or self.top_p != 0.95:
            raise GenerationContractError(
                "paper one-seed generation requires temperature=0.6 and top_p=0.95"
            )
        if self.samples_per_instance != 1:
            raise GenerationContractError("one completion per instance is required")
        if self.batch_size < 1 or self.max_prompt_tokens < 1:
            raise GenerationContractError("batch and prompt limits must be positive")
        names = [item.name for item in self.benchmarks]
        expected = {source.name for source in PINNED_BENCHMARKS}
        if len(names) != len(set(names)) or set(names) != expected:
            raise GenerationContractError("all four pinned benchmarks are required once")

    @classmethod
    def from_mapping(cls, value: Any) -> GenerationProtocol:
        if not isinstance(value, Mapping):
            raise GenerationContractError("generation protocol must be an object")
        required = {
            "contract_version",
            "protocol_id",
            "paper_revision",
            "prompt_protocol",
            "evaluator_revision",
            "seed",
            "temperature",
            "top_p",
            "samples_per_instance",
            "batch_size",
            "max_prompt_tokens",
            "benchmarks",
        }
        _strict_keys(value, context="generation protocol", required=required)
        for key in (
            "contract_version",
            "seed",
            "samples_per_instance",
            "batch_size",
            "max_prompt_tokens",
        ):
            if isinstance(value[key], bool) or not isinstance(value[key], int):
                raise GenerationContractError(f"{key} must be an integer")
        for key in ("temperature", "top_p"):
            if isinstance(value[key], bool) or not isinstance(value[key], (int, float)):
                raise GenerationContractError(f"{key} must be numeric")
        if not isinstance(value["benchmarks"], Sequence) or isinstance(
            value["benchmarks"], (str, bytes)
        ):
            raise GenerationContractError("benchmarks must be an array")
        return cls(
            contract_version=value["contract_version"],
            protocol_id=value["protocol_id"],
            paper_revision=value["paper_revision"],
            prompt_protocol=value["prompt_protocol"],
            evaluator_revision=value["evaluator_revision"],
            seed=value["seed"],
            temperature=float(value["temperature"]),
            top_p=float(value["top_p"]),
            samples_per_instance=value["samples_per_instance"],
            batch_size=value["batch_size"],
            max_prompt_tokens=value["max_prompt_tokens"],
            benchmarks=tuple(
                GenerationBenchmark.from_mapping(item)
                for item in value["benchmarks"]
            ),
        )

    def canonical_json(self) -> str:
        return json.dumps(
            dataclasses.asdict(self),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )

    def digest(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


@dataclasses.dataclass(frozen=True, slots=True)
class GenerationExample:
    benchmark: str
    instance_id: str
    source_index: int
    prompt: str


def load_generation_protocol(path: str | Path) -> GenerationProtocol:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GenerationContractError(f"invalid generation protocol: {path}") from exc
    return GenerationProtocol.from_mapping(payload)


def gemma_merged_system_prompt(system: str, user: str) -> str:
    if not system.strip() or not user.strip():
        raise GenerationContractError("system and user prompt text are required")
    return (
        "<start_of_turn>user\n"
        + system.strip()
        + "\n\n"
        + user.strip()
        + "<end_of_turn>\n<start_of_turn>model\n"
    )


def _record_prompt(name: str, row: Mapping[str, Any]) -> tuple[str, str]:
    if name == "gsm8k":
        return GSM8K_SYSTEM_PROMPT, str(row["question"])
    if name == "math500":
        return MATH500_SYSTEM_PROMPT, str(row["problem"])
    if name == "mbpp":
        tests = row["test_list"]
        if not isinstance(tests, list) or any(not isinstance(item, str) for item in tests):
            raise GenerationContractError("MBPP test_list must be an array of strings")
        user = (
            str(row["text"])
            + "\n\nYour code should pass these tests:\n"
            + "\n".join(tests)
        )
        return MBPP_SYSTEM_PROMPT, user
    if name == "live-code-bench-v6":
        return LCB_SYSTEM_PROMPT, str(row["question_content"])
    raise GenerationContractError(f"unsupported benchmark: {name}")


def _instance_id(name: str, row: Mapping[str, Any], index: int) -> str:
    if name == "gsm8k":
        return f"gsm8k-{index:04d}"
    if name == "math500":
        return str(row["unique_id"])
    if name == "mbpp":
        return str(row["task_id"])
    if name == "live-code-bench-v6":
        return str(row["question_id"])
    raise GenerationContractError(f"unsupported benchmark: {name}")


def generation_example_from_record(
    benchmark: str,
    row: Mapping[str, Any],
    index: int,
) -> GenerationExample:
    """Build the exact generation identity for one verified benchmark row.

    Scoring uses this public helper to recompute the instance ID and prompt
    digest without rereading a multi-gigabyte benchmark through a second
    iterator.  Generation and scoring therefore share one prompt transform.
    """

    system, user = _record_prompt(benchmark, row)
    return GenerationExample(
        benchmark=benchmark,
        instance_id=_instance_id(benchmark, row, index),
        source_index=index,
        prompt=gemma_merged_system_prompt(system, user),
    )


def iter_generation_examples(
    protocol: GenerationProtocol,
    evaluation_root: str | Path,
    benchmark: GenerationBenchmark,
) -> Iterator[GenerationExample]:
    source = next(item for item in PINNED_BENCHMARKS if item.name == benchmark.name)
    manifest = load_verified_materialization(source, evaluation_root)
    if manifest is None:
        raise GenerationContractError(
            f"benchmark materialization failed verification: {benchmark.name}"
        )
    if manifest["records_sha256"] != benchmark.records_sha256:
        raise GenerationContractError(
            f"generation contract hash drift for {benchmark.name}"
        )
    records = Path(evaluation_root) / benchmark.name / "records.jsonl"
    observed_ids: set[str] = set()
    count = 0
    with records.open("r", encoding="utf-8") as stream:
        for index, line in enumerate(stream):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise GenerationContractError(
                    f"invalid {benchmark.name} JSONL row {index}"
                ) from exc
            if not isinstance(row, Mapping):
                raise GenerationContractError(
                    f"{benchmark.name} row {index} must be an object"
                )
            example = generation_example_from_record(benchmark.name, row, index)
            instance_id = example.instance_id
            if not instance_id or instance_id in observed_ids:
                raise GenerationContractError(
                    f"duplicate/empty instance ID in {benchmark.name}: {instance_id!r}"
                )
            observed_ids.add(instance_id)
            count += 1
            yield example
    if count != benchmark.record_count:
        raise GenerationContractError(
            f"{benchmark.name} expected {benchmark.record_count} rows, got {count}"
        )


def stable_batch_seed(
    protocol: GenerationProtocol, benchmark: str, batch_index: int
) -> int:
    payload = f"{protocol.seed}\x1f{benchmark}\x1f{batch_index}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big") & 0x7FFFFFFF
