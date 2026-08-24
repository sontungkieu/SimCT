"""Score native generations with the evaluator released by the SimCT paper.

This is deliberately not described as an official benchmark evaluator.  The
paper repository's ``evaluation.py`` is the metric implementation that
produced the reported numbers, but it is not the official LiveCodeBench or
Math-Verify harness.  We verify the released file byte-for-byte, extract the
exact scoring functions from its AST, and record the runtime environment.
"""

from __future__ import annotations

import ast
import base64
import dataclasses
import functools
import hashlib
import json
import multiprocessing
import os
import pickle
import platform
import re
import zlib
from collections.abc import Callable, Mapping
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from vdt_tunix.evaluation_data import (
    PINNED_BENCHMARKS,
    load_verified_materialization,
    sha256_file,
)
from vdt_tunix.generation_contract import (
    GenerationBenchmark,
    GenerationProtocol,
    generation_example_from_record,
    stable_batch_seed,
)

PAPER_EVALUATOR_ID = "sunjie279/SimCT-/scripts/evaluation/evaluation.py"
PAPER_EVALUATOR_REVISION = "cf0f33a0e6c967d4b74ea32b2dba12be01b73b9e"
PAPER_EVALUATOR_SHA256 = (
    "21378cfb1aa1d2f3ddab684a1bcb671fd588919c76fd410fac424bd062db2839"
)
_RELEASED_FUNCTIONS = {
    "strip_thinking_content",
    "extract_number_from_answer",
    "normalize_answer_string",
    "extract_boxed_answer",
    "try_parse_number",
    "is_math_equivalent",
    "_extract_code_block",
    "_run_mbpp_tests",
    "_run_lcb_tests",
}
_PREDICTION_KEYS = {
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


class PaperReleasedScoringError(RuntimeError):
    """Raised when source fidelity, prediction identity, or scoring fails."""


@dataclasses.dataclass(frozen=True, slots=True)
class ReleasedEvaluator:
    source_path: Path
    functions: Mapping[str, Callable[..., Any]]

    def __getitem__(self, name: str) -> Callable[..., Any]:
        return self.functions[name]


@functools.lru_cache(maxsize=4)
def load_released_evaluator(source_path: str | Path) -> ReleasedEvaluator:
    """Load only exact scorer functions from the byte-pinned paper source."""

    path = Path(source_path).resolve()
    if not path.is_file():
        raise PaperReleasedScoringError(
            f"paper evaluator source is unavailable: {path}"
        )
    observed = sha256_file(path)
    if observed != PAPER_EVALUATOR_SHA256:
        raise PaperReleasedScoringError(
            "paper evaluator source hash drifted: "
            f"expected {PAPER_EVALUATOR_SHA256}, got {observed}"
        )
    try:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
    except (OSError, UnicodeDecodeError, SyntaxError) as exc:
        raise PaperReleasedScoringError(
            f"paper evaluator source cannot be parsed: {path}"
        ) from exc
    selected = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in _RELEASED_FUNCTIONS
    ]
    names = {node.name for node in selected}
    if names != _RELEASED_FUNCTIONS:
        raise PaperReleasedScoringError(
            "paper evaluator functions drifted: "
            f"missing={sorted(_RELEASED_FUNCTIONS - names)}"
        )
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__",
                names=[ast.alias(name="annotations")],
                level=0,
            ),
            *selected,
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    namespace: dict[str, Any] = {
        "__name__": "_vdt_paper_released_evaluator",
        "os": os,
        "re": re,
    }
    exec(compile(module, str(path), "exec"), namespace)  # noqa: S102
    return ReleasedEvaluator(
        source_path=path,
        functions={name: namespace[name] for name in sorted(_RELEASED_FUNCTIONS)},
    )


def released_environment() -> dict[str, Any]:
    """Record optional math-parser availability instead of assuming it."""

    sympy_version = None
    latex_parser_operational = False
    latex_parser_error = None
    try:
        import sympy
        from sympy.parsing.latex import parse_latex

        sympy_version = sympy.__version__
        parse_latex("1")
        latex_parser_operational = True
    except Exception as exc:  # noqa: BLE001 - match released fallback semantics
        latex_parser_error = f"{type(exc).__name__}: {exc}"
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "multiprocessing_start_method": multiprocessing.get_start_method(),
        "sympy": sympy_version,
        "sympy_latex_parser_operational": latex_parser_operational,
        "sympy_latex_parser_error": latex_parser_error,
    }


def require_released_fork_semantics() -> None:
    """Normalize Python 3.14+ to the Linux start method used by the release.

    The released MBPP function defines its worker locally, so it is not
    pickleable under ``spawn`` or Python 3.14's new ``forkserver`` default.
    Kaggle's release-era Linux runtime used ``fork``.  Scoring runs in a
    dedicated process, making this global normalization bounded and explicit.
    """

    if "fork" not in multiprocessing.get_all_start_methods():
        raise PaperReleasedScoringError(
            "paper-released MBPP evaluator requires Linux fork semantics"
        )
    if multiprocessing.get_start_method(allow_none=True) != "fork":
        try:
            multiprocessing.set_start_method("fork", force=True)
        except RuntimeError as exc:
            raise PaperReleasedScoringError(
                "cannot establish fork semantics before MBPP evaluation"
            ) from exc


def decode_released_lcb_tests(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Reproduce the paper evaluator's public/private LCB test decoding."""

    public = row.get("public_test_cases", row.get("test_cases", row.get("tests", [])))
    if isinstance(public, str):
        try:
            public = json.loads(public)
        except json.JSONDecodeError:
            public = []
    tests = list(public) if isinstance(public, list) else []
    private = row.get("private_test_cases", "")
    if private and isinstance(private, str):
        try:
            decoded = base64.b64decode(private)
            decompressed = zlib.decompress(decoded)
            payload = pickle.loads(decompressed)
            private_tests = json.loads(payload) if isinstance(payload, str) else payload
            if isinstance(private_tests, list):
                tests.extend(private_tests)
        except Exception:  # noqa: BLE001,S110 - exact released behavior
            # Exact behavior of the released evaluator: decoding failure leaves
            # the public tests in place and does not abort the whole benchmark.
            pass
    return tests


def _released_flags(
    evaluator: ReleasedEvaluator, completion: str
) -> tuple[str, bool, bool, bool]:
    stripped, had_thinking, think_truncated = evaluator[
        "strip_thinking_content"
    ](completion)
    return stripped, had_thinking, think_truncated, not bool(completion.strip())


def score_released_gsm8k(
    evaluator: ReleasedEvaluator, completion: str, gold: Mapping[str, Any]
) -> dict[str, Any]:
    stripped, had_thinking, think_truncated, empty = _released_flags(
        evaluator, completion
    )
    predicted = evaluator["extract_number_from_answer"](stripped)
    target = evaluator["extract_number_from_answer"](str(gold["answer"]))
    correct = (
        predicted is not None
        and target is not None
        and abs(predicted - target) < 1e-6
    )
    return {
        "correct": bool(correct),
        "predicted_answer": predicted,
        "gold_answer": target,
        "had_thinking": had_thinking,
        "think_truncated": think_truncated,
        "empty_output": empty,
        "extraction_failed": predicted is None and not empty,
    }


def score_released_math500(
    evaluator: ReleasedEvaluator, completion: str, gold: Mapping[str, Any]
) -> dict[str, Any]:
    stripped, had_thinking, think_truncated, empty = _released_flags(
        evaluator, completion
    )
    predicted = evaluator["extract_boxed_answer"](stripped)
    target = str(gold["answer"])
    return {
        "correct": bool(evaluator["is_math_equivalent"](predicted, target)),
        "predicted_answer": predicted,
        "gold_answer": target,
        "had_thinking": had_thinking,
        "think_truncated": think_truncated,
        "empty_output": empty,
        "extraction_failed": not predicted and not empty,
    }


def score_released_mbpp(
    evaluator: ReleasedEvaluator, completion: str, gold: Mapping[str, Any]
) -> dict[str, Any]:
    require_released_fork_semantics()
    stripped, had_thinking, think_truncated, empty = _released_flags(
        evaluator, completion
    )
    code = evaluator["_extract_code_block"](stripped)
    tests = gold["test_list"]
    if not isinstance(tests, list) or any(not isinstance(item, str) for item in tests):
        raise PaperReleasedScoringError("MBPP test_list is not an array of strings")
    passed = evaluator["_run_mbpp_tests"](
        code,
        tests,
        str(gold.get("test_setup_code", "")),
    )
    return {
        "correct": bool(passed),
        "had_thinking": had_thinking,
        "think_truncated": think_truncated,
        "empty_output": empty,
        "extraction_failed": not code.strip() and not empty,
    }


def _lcb_worker(
    task: tuple[str, str, list[dict[str, Any]]]
) -> tuple[int, int]:
    evaluator_path, code, tests = task
    evaluator = load_released_evaluator(evaluator_path)
    return evaluator["_run_lcb_tests"](code, tests)


def _score_lcb_batch(
    *,
    evaluator_path: Path,
    pending: list[
        tuple[dict[str, Any], dict[str, Any], str, list[dict[str, Any]]]
    ],
    pool: ProcessPoolExecutor | None,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    tasks = [
        (str(evaluator_path), code, tests)
        for _, _, code, tests in pending
    ]
    if pool is None:
        results = [_lcb_worker(task) for task in tasks]
    else:
        results = list(pool.map(_lcb_worker, tasks, chunksize=1))
    scored = []
    for (prediction, flags, _code, _tests), (passed, total) in zip(
        pending, results, strict=True
    ):
        scored.append(
            (
                prediction,
                {
                    **flags,
                    "correct": bool(total > 0 and passed == total),
                    "tests_passed": int(passed),
                    "tests_total": int(total),
                },
            )
        )
    return scored


def _canonical_json_line(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            dict(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        + "\n"
    ).encode("utf-8")


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _load_predictions(
    path: Path,
    *,
    benchmark: GenerationBenchmark,
    protocol: GenerationProtocol,
    protocol_sha256: str,
    variant: str,
    checkpoint_sha256: str,
) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for index, line in enumerate(stream):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise PaperReleasedScoringError(
                    f"invalid prediction JSON at {path}:{index + 1}"
                ) from exc
            if not isinstance(row, dict) or set(row) != _PREDICTION_KEYS:
                raise PaperReleasedScoringError(
                    f"prediction schema drift at {path}:{index + 1}"
                )
            expected = {
                "contract_version": 1,
                "protocol_sha256": protocol_sha256,
                "variant": variant,
                "checkpoint_sha256": checkpoint_sha256,
                "benchmark": benchmark.name,
                "source_index": index,
                "batch_index": index // protocol.batch_size,
                "batch_position": index % protocol.batch_size,
                "seed": stable_batch_seed(
                    protocol, benchmark.name, index // protocol.batch_size
                ),
                "temperature": protocol.temperature,
                "top_p": protocol.top_p,
                "max_new_tokens": benchmark.max_new_tokens,
            }
            drift = {
                key: (row.get(key), value)
                for key, value in expected.items()
                if row.get(key) != value
            }
            if drift:
                raise PaperReleasedScoringError(
                    f"prediction identity drift at {path}:{index + 1}: {drift}"
                )
            for key in (
                "batch_index",
                "batch_position",
                "seed",
                "prompt_tokens",
                "completion_tokens",
            ):
                if isinstance(row.get(key), bool) or not isinstance(row.get(key), int):
                    raise PaperReleasedScoringError(
                        f"prediction {key} is not an integer at {path}:{index + 1}"
                    )
            if not 1 <= row["prompt_tokens"] <= protocol.max_prompt_tokens:
                raise PaperReleasedScoringError(
                    f"prediction prompt_tokens is outside the protocol at "
                    f"{path}:{index + 1}"
                )
            if not 0 <= row["completion_tokens"] <= benchmark.max_new_tokens:
                raise PaperReleasedScoringError(
                    f"prediction completion_tokens is outside the protocol at "
                    f"{path}:{index + 1}"
                )
            if not isinstance(row.get("completion"), str) or not isinstance(
                row.get("truncated"), bool
            ):
                raise PaperReleasedScoringError(
                    f"prediction payload type drift at {path}:{index + 1}"
                )
            if row["truncated"] != (
                row["completion_tokens"] == benchmark.max_new_tokens
            ):
                raise PaperReleasedScoringError(
                    f"prediction truncation flag drift at {path}:{index + 1}"
                )
            rows.append(row)
    if len(rows) != benchmark.record_count:
        raise PaperReleasedScoringError(
            f"{benchmark.name} expected {benchmark.record_count} predictions, "
            f"got {len(rows)}"
        )
    if len({row["instance_id"] for row in rows}) != len(rows):
        raise PaperReleasedScoringError(
            f"{benchmark.name} prediction instance IDs are not unique"
        )
    return tuple(rows)


def _metric_row(
    prediction: Mapping[str, Any], score: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "contract_version": 1,
        "benchmark": prediction["benchmark"],
        "instance_id": prediction["instance_id"],
        "source_index": prediction["source_index"],
        "checkpoint_sha256": prediction["checkpoint_sha256"],
        "completion_sha256": hashlib.sha256(
            prediction["completion"].encode("utf-8")
        ).hexdigest(),
        "generation_truncated": prediction["truncated"],
        **score,
    }


def score_generation_root(
    *,
    generation_root: str | Path,
    evaluation_root: str | Path,
    protocol: GenerationProtocol,
    evaluator_source: str | Path,
    output_root: str | Path,
    workers: int,
) -> dict[str, Any]:
    """Score all four benchmark files under one verified generation root."""

    if workers < 1:
        raise PaperReleasedScoringError("workers must be positive")
    require_released_fork_semantics()
    if protocol.evaluator_revision != PAPER_EVALUATOR_REVISION:
        raise PaperReleasedScoringError(
            "generation protocol does not pin the released evaluator revision"
        )
    generation_dir = Path(generation_root).resolve()
    output_dir = Path(output_root).resolve()
    try:
        generation_summary = json.loads(
            (generation_dir / "generation_summary.json").read_text(encoding="utf-8")
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PaperReleasedScoringError("generation summary is unavailable") from exc
    protocol_sha256 = protocol.digest()
    expected_summary = {
        "status": "complete",
        "phase": "native_tunix_generation",
        "protocol_sha256": protocol_sha256,
        "scientific_evidence": False,
    }
    drift = {
        key: (generation_summary.get(key), value)
        for key, value in expected_summary.items()
        if generation_summary.get(key) != value
    }
    if drift:
        raise PaperReleasedScoringError(f"generation summary drifted: {drift}")
    prompt_preflight_path = generation_dir / "prompt_preflight.json"
    if sha256_file(prompt_preflight_path) != generation_summary.get(
        "prompt_preflight_sha256"
    ):
        raise PaperReleasedScoringError("generation prompt preflight hash drifted")
    try:
        prompt_preflight = json.loads(
            prompt_preflight_path.read_text(encoding="utf-8")
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PaperReleasedScoringError(
            "generation prompt preflight is unavailable"
        ) from exc
    preflight_expected = {
        "status": "passed",
        "protocol_sha256": protocol_sha256,
        "scientific_evidence": False,
    }
    preflight_drift = {
        key: (prompt_preflight.get(key), value)
        for key, value in preflight_expected.items()
        if prompt_preflight.get(key) != value
    }
    preflight_by_name = {
        item.get("benchmark"): item
        for item in prompt_preflight.get("benchmarks", [])
        if isinstance(item, Mapping)
    }
    if set(preflight_by_name) != {item.name for item in protocol.benchmarks}:
        preflight_drift["benchmarks"] = (
            sorted(preflight_by_name),
            sorted(item.name for item in protocol.benchmarks),
        )
    for benchmark in protocol.benchmarks:
        item = preflight_by_name.get(benchmark.name, {})
        expected_item = {
            "record_count": benchmark.record_count,
            "max_prompt_tokens_contract": protocol.max_prompt_tokens,
        }
        for key, value in expected_item.items():
            if item.get(key) != value:
                preflight_drift[f"{benchmark.name}.{key}"] = (
                    item.get(key),
                    value,
                )
        maximum = item.get("maximum_prompt_tokens")
        minimum = item.get("minimum_prompt_tokens")
        if (
            isinstance(maximum, bool)
            or not isinstance(maximum, int)
            or isinstance(minimum, bool)
            or not isinstance(minimum, int)
            or not 1 <= minimum <= maximum <= protocol.max_prompt_tokens
        ):
            preflight_drift[f"{benchmark.name}.token_range"] = (
                minimum,
                maximum,
            )
    if preflight_drift:
        raise PaperReleasedScoringError(
            f"generation prompt preflight drifted: {preflight_drift}"
        )
    variant = generation_summary.get("variant")
    checkpoint_sha256 = generation_summary.get("student_parameters_sha256")
    if variant not in {"sft", "simple_opd", "simct"}:
        raise PaperReleasedScoringError("generation variant is unsupported")
    if (
        not isinstance(checkpoint_sha256, str)
        or len(checkpoint_sha256) != 64
        or any(char not in "0123456789abcdef" for char in checkpoint_sha256)
    ):
        raise PaperReleasedScoringError("generation checkpoint hash is invalid")
    evaluator = load_released_evaluator(evaluator_source)
    source_by_name = {source.name: source for source in PINNED_BENCHMARKS}
    generation_benchmarks = {
        item.get("benchmark"): item
        for item in generation_summary.get("benchmarks", [])
        if isinstance(item, Mapping)
    }
    if set(generation_benchmarks) != {item.name for item in protocol.benchmarks}:
        raise PaperReleasedScoringError("generation benchmark summary is incomplete")

    benchmark_summaries: list[dict[str, Any]] = []
    for benchmark in protocol.benchmarks:
        prediction_path = generation_dir / benchmark.name / "predictions.jsonl"
        summary_entry = generation_benchmarks[benchmark.name]
        if sha256_file(prediction_path) != summary_entry.get("predictions_sha256"):
            raise PaperReleasedScoringError(
                f"{benchmark.name} prediction file hash drifted"
            )
        predictions = _load_predictions(
            prediction_path,
            benchmark=benchmark,
            protocol=protocol,
            protocol_sha256=protocol_sha256,
            variant=variant,
            checkpoint_sha256=checkpoint_sha256,
        )
        materialization = load_verified_materialization(
            source_by_name[benchmark.name], evaluation_root
        )
        if materialization is None or materialization.get("records_sha256") != benchmark.records_sha256:
            raise PaperReleasedScoringError(
                f"{benchmark.name} evaluation materialization drifted"
            )

        scored_path = output_dir / benchmark.name / "scored_predictions.jsonl"
        scored_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = scored_path.with_name(scored_path.name + ".tmp")
        correct = 0
        extraction_failed = 0
        empty_output = 0
        think_truncated = 0
        generation_truncated = 0
        observed = 0
        pending_lcb: list[
            tuple[dict[str, Any], dict[str, Any], str, list[dict[str, Any]]]
        ] = []
        lcb_pool = (
            ProcessPoolExecutor(max_workers=workers)
            if benchmark.name == "live-code-bench-v6" and workers > 1
            else None
        )

        def write_scored(stream, prediction, score):
            nonlocal correct, extraction_failed, empty_output
            nonlocal think_truncated, generation_truncated, observed
            row = _metric_row(prediction, score)
            stream.write(_canonical_json_line(row))
            observed += 1
            correct += int(bool(row["correct"]))
            extraction_failed += int(bool(row["extraction_failed"]))
            empty_output += int(bool(row["empty_output"]))
            think_truncated += int(bool(row["think_truncated"]))
            generation_truncated += int(bool(row["generation_truncated"]))

        records_path = Path(evaluation_root) / benchmark.name / "records.jsonl"
        try:
            with records_path.open("r", encoding="utf-8") as records, temporary.open(
                "wb"
            ) as scored_stream:
                for index, line in enumerate(records):
                    try:
                        gold = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise PaperReleasedScoringError(
                            f"invalid gold JSON for {benchmark.name} row {index}"
                        ) from exc
                    if not isinstance(gold, Mapping) or index >= len(predictions):
                        raise PaperReleasedScoringError(
                            f"gold/prediction cardinality drift for {benchmark.name}"
                        )
                    prediction = predictions[index]
                    example = generation_example_from_record(benchmark.name, gold, index)
                    identity = {
                        "instance_id": example.instance_id,
                        "source_index": example.source_index,
                        "prompt_sha256": hashlib.sha256(
                            example.prompt.encode("utf-8")
                        ).hexdigest(),
                    }
                    identity_drift = {
                        key: (prediction.get(key), value)
                        for key, value in identity.items()
                        if prediction.get(key) != value
                    }
                    if identity_drift:
                        raise PaperReleasedScoringError(
                            f"gold/prediction identity drift for {benchmark.name}: "
                            f"{identity_drift}"
                        )
                    completion = prediction["completion"]
                    if benchmark.name == "gsm8k":
                        score = score_released_gsm8k(evaluator, completion, gold)
                        write_scored(scored_stream, prediction, score)
                    elif benchmark.name == "math500":
                        score = score_released_math500(evaluator, completion, gold)
                        write_scored(scored_stream, prediction, score)
                    elif benchmark.name == "mbpp":
                        score = score_released_mbpp(evaluator, completion, gold)
                        write_scored(scored_stream, prediction, score)
                    else:
                        stripped, had_thinking, truncated, empty = _released_flags(
                            evaluator, completion
                        )
                        code = evaluator["_extract_code_block"](stripped)
                        pending_lcb.append(
                            (
                                prediction,
                                {
                                    "had_thinking": had_thinking,
                                    "think_truncated": truncated,
                                    "empty_output": empty,
                                    "extraction_failed": not code.strip() and not empty,
                                },
                                code,
                                decode_released_lcb_tests(gold),
                            )
                        )
                        if len(pending_lcb) >= workers:
                            for scored_prediction, score in _score_lcb_batch(
                                evaluator_path=evaluator.source_path,
                                pending=pending_lcb,
                                pool=lcb_pool,
                            ):
                                write_scored(
                                    scored_stream, scored_prediction, score
                                )
                            pending_lcb.clear()
                if pending_lcb:
                    for scored_prediction, score in _score_lcb_batch(
                        evaluator_path=evaluator.source_path,
                        pending=pending_lcb,
                        pool=lcb_pool,
                    ):
                        write_scored(scored_stream, scored_prediction, score)
                    pending_lcb.clear()
                scored_stream.flush()
        finally:
            if lcb_pool is not None:
                lcb_pool.shutdown(wait=True, cancel_futures=True)
        if observed != benchmark.record_count:
            temporary.unlink(missing_ok=True)
            raise PaperReleasedScoringError(
                f"{benchmark.name} scored {observed}, expected {benchmark.record_count}"
            )
        temporary.replace(scored_path)
        metric = correct / observed
        benchmark_summary = {
            "benchmark": benchmark.name,
            "metric": "accuracy" if benchmark.name in {"gsm8k", "math500"} else "pass@1",
            "score": metric,
            "correct": correct,
            "total": observed,
            "extraction_failed_count": extraction_failed,
            "empty_output_count": empty_output,
            "think_truncated_count": think_truncated,
            "generation_truncated_count": generation_truncated,
            "scored_predictions_sha256": sha256_file(scored_path),
        }
        _atomic_json(output_dir / benchmark.name / "metrics.json", benchmark_summary)
        benchmark_summaries.append(benchmark_summary)

    summary = {
        "contract_version": 1,
        "status": "complete",
        "phase": "paper_released_evaluator_scoring",
        "variant": variant,
        "protocol_id": protocol.protocol_id,
        "protocol_sha256": protocol_sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "evaluator_id": PAPER_EVALUATOR_ID,
        "evaluator_revision": PAPER_EVALUATOR_REVISION,
        "evaluator_source_sha256": PAPER_EVALUATOR_SHA256,
        "evaluator_classification": "paper-released; not official benchmark harness",
        "environment": released_environment(),
        "benchmarks": benchmark_summaries,
        "evaluation_evidence": True,
        "scientific_evidence": True,
        "paper_reproduction": False,
        "limitations": [
            "one seed only",
            "public substitute training corpus, not the paper 10K corpus",
            "released paper evaluator is not the official LiveCodeBench/Math-Verify harness",
        ],
    }
    _atomic_json(output_dir / "scoring_summary.json", summary)
    return summary
