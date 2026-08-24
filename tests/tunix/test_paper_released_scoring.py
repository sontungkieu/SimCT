from __future__ import annotations

import base64
import hashlib
import json
import pickle
import zlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from vdt_tunix.generation_contract import (
    generation_example_from_record,
    stable_batch_seed,
)
from vdt_tunix.paper_released_scoring import (
    PAPER_EVALUATOR_REVISION,
    PAPER_EVALUATOR_SHA256,
    PaperReleasedScoringError,
    decode_released_lcb_tests,
    load_released_evaluator,
    score_generation_root,
    score_released_gsm8k,
    score_released_math500,
    score_released_mbpp,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
EVALUATOR = REPO_ROOT / "scripts" / "evaluation" / "evaluation.py"


def test_released_evaluator_source_and_extractors_are_exact():
    evaluator = load_released_evaluator(EVALUATOR)
    assert PAPER_EVALUATOR_REVISION == "cf0f33a0e6c967d4b74ea32b2dba12be01b73b9e"
    assert PAPER_EVALUATOR_SHA256 == (
        "21378cfb1aa1d2f3ddab684a1bcb671fd588919c76fd410fac424bd062db2839"
    )
    assert evaluator["extract_number_from_answer"]("work #### 1,234") == 1234.0
    assert evaluator["extract_boxed_answer"](r"work \boxed{\frac{1}{2}}") == (
        r"\frac{1}{2}"
    )


def test_released_evaluator_rejects_byte_drift(tmp_path):
    changed = tmp_path / "evaluation.py"
    changed.write_bytes(EVALUATOR.read_bytes() + b"\n# drift\n")
    with pytest.raises(PaperReleasedScoringError, match="hash drifted"):
        load_released_evaluator(changed)


def test_released_math_scorers_match_paper_logic():
    evaluator = load_released_evaluator(EVALUATOR)
    gsm = score_released_gsm8k(
        evaluator,
        "reasoning\n#### 42",
        {"answer": "gold work\n#### 42"},
    )
    math = score_released_math500(
        evaluator,
        r"reasoning \boxed{\frac{1}{2}}",
        {"answer": r"\frac{1}{2}"},
    )
    assert gsm["correct"] is True
    assert gsm["extraction_failed"] is False
    assert math["correct"] is True


def test_released_mbpp_executes_base_tests_only():
    evaluator = load_released_evaluator(EVALUATOR)
    result = score_released_mbpp(
        evaluator,
        "```python\ndef add(a, b):\n    return a + b\n```",
        {
            "test_list": ["assert add(2, 3) == 5"],
            "challenge_test_list": ["assert False"],
            "test_setup_code": "",
        },
    )
    assert result["correct"] is True


def test_released_lcb_decode_and_runner_match_source_behavior():
    evaluator = load_released_evaluator(EVALUATOR)
    private = [{"input": "world\n", "output": "world\n", "testtype": "stdin"}]
    encoded = base64.b64encode(
        zlib.compress(pickle.dumps(json.dumps(private)))
    ).decode("ascii")
    row = {
        "public_test_cases": json.dumps(
            [{"input": "hello\n", "output": "hello\n", "testtype": "stdin"}]
        ),
        "private_test_cases": encoded,
    }
    tests = decode_released_lcb_tests(row)
    assert len(tests) == 2
    passed, total = evaluator["_run_lcb_tests"](
        "import sys\nprint(sys.stdin.read().strip())",
        tests,
        timeout=1.0,
    )
    assert (passed, total) == (2, 2)


def test_full_scoring_contract_checks_identity_and_scores_four_benchmarks(
    tmp_path, monkeypatch
):
    from vdt_tunix import paper_released_scoring as scoring

    protocol_sha = "9" * 64
    benchmark_rows = {
        "gsm8k": {
            "question": "What is one plus one?",
            "answer": "work\n#### 2",
        },
        "math500": {
            "problem": "Compute 1+1.",
            "solution": r"\boxed{2}",
            "answer": "2",
            "unique_id": "math-0",
            "subject": "Algebra",
            "level": 1,
        },
        "mbpp": {
            "task_id": 10,
            "text": "Add two numbers.",
            "code": "def add(a, b): return a + b",
            "test_list": ["assert add(2, 3) == 5"],
            "test_setup_code": "",
            "challenge_test_list": [],
        },
        "live-code-bench-v6": {
            "question_id": "lcb-0",
            "question_content": "Echo stdin.",
            "public_test_cases": json.dumps(
                [{"input": "ok\n", "output": "ok\n", "testtype": "stdin"}]
            ),
            "private_test_cases": "",
            "metadata": "{}",
        },
    }
    completions = {
        "gsm8k": "reasoning\n#### 2",
        "math500": r"reasoning \boxed{2}",
        "mbpp": "def add(a, b):\n    return a + b",
        "live-code-bench-v6": "import sys\nprint(sys.stdin.read().strip())",
    }
    benchmarks = tuple(
        SimpleNamespace(
            name=name,
            records_sha256=str(index + 1) * 64,
            record_count=1,
            max_new_tokens=16,
        )
        for index, name in enumerate(benchmark_rows)
    )

    class Protocol:
        protocol_id = "fixture-generation-v1"
        evaluator_revision = PAPER_EVALUATOR_REVISION
        seed = 42
        temperature = 0.6
        top_p = 0.95
        batch_size = 4
        max_prompt_tokens = 4096

        def __init__(self):
            self.benchmarks = benchmarks

        def digest(self):
            return protocol_sha

    protocol = Protocol()
    evaluation_root = tmp_path / "evaluation"
    generation_root = tmp_path / "generation"
    checkpoint_sha = "a" * 64
    generation_benchmarks = []
    for benchmark in benchmarks:
        gold = benchmark_rows[benchmark.name]
        gold_dir = evaluation_root / benchmark.name
        gold_dir.mkdir(parents=True)
        (gold_dir / "records.jsonl").write_text(
            json.dumps(gold, sort_keys=True) + "\n", encoding="utf-8"
        )
        example = generation_example_from_record(benchmark.name, gold, 0)
        prediction = {
            "contract_version": 1,
            "protocol_sha256": protocol_sha,
            "variant": "sft",
            "checkpoint_sha256": checkpoint_sha,
            "benchmark": benchmark.name,
            "instance_id": example.instance_id,
            "source_index": 0,
            "prompt_sha256": hashlib.sha256(
                example.prompt.encode("utf-8")
            ).hexdigest(),
            "batch_index": 0,
            "batch_position": 0,
            "seed": stable_batch_seed(protocol, benchmark.name, 0),
            "temperature": 0.6,
            "top_p": 0.95,
            "max_new_tokens": 16,
            "prompt_tokens": 8,
            "completion_tokens": 4,
            "truncated": False,
            "completion": completions[benchmark.name],
        }
        prediction_path = generation_root / benchmark.name / "predictions.jsonl"
        prediction_path.parent.mkdir(parents=True)
        prediction_path.write_text(
            json.dumps(prediction, sort_keys=True) + "\n", encoding="utf-8"
        )
        generation_benchmarks.append(
            {
                "benchmark": benchmark.name,
                "predictions_sha256": hashlib.sha256(
                    prediction_path.read_bytes()
                ).hexdigest(),
            }
        )
    preflight = {
        "status": "passed",
        "protocol_sha256": protocol_sha,
        "scientific_evidence": False,
        "benchmarks": [
            {
                "benchmark": benchmark.name,
                "record_count": 1,
                "minimum_prompt_tokens": 8,
                "maximum_prompt_tokens": 8,
                "max_prompt_tokens_contract": 4096,
            }
            for benchmark in benchmarks
        ],
    }
    preflight_path = generation_root / "prompt_preflight.json"
    preflight_path.write_text(
        json.dumps(preflight, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    generation_summary = {
        "status": "complete",
        "phase": "native_tunix_generation",
        "protocol_sha256": protocol_sha,
        "scientific_evidence": False,
        "variant": "sft",
        "student_parameters_sha256": checkpoint_sha,
        "prompt_preflight_sha256": hashlib.sha256(
            preflight_path.read_bytes()
        ).hexdigest(),
        "benchmarks": generation_benchmarks,
    }
    (generation_root / "generation_summary.json").write_text(
        json.dumps(generation_summary), encoding="utf-8"
    )
    monkeypatch.setattr(
        scoring,
        "load_verified_materialization",
        lambda source, root: {
            "records_sha256": next(
                item.records_sha256
                for item in benchmarks
                if item.name == source.name
            )
        },
    )
    summary = score_generation_root(
        generation_root=generation_root,
        evaluation_root=evaluation_root,
        protocol=protocol,
        evaluator_source=EVALUATOR,
        output_root=tmp_path / "scores",
        workers=1,
    )
    assert summary["status"] == "complete"
    assert summary["paper_reproduction"] is False
    assert len(summary["benchmarks"]) == 4
    assert all(item["score"] == 1.0 for item in summary["benchmarks"])
