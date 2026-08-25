from __future__ import annotations

import json
from pathlib import Path

import pytest

from vdt_tunix.config import load_config
from vdt_tunix.generation_contract import load_generation_protocol
from vdt_tunix.one_seed_comparison import (
    OneSeedComparisonError,
    VariantEvidencePaths,
    assemble_one_seed_comparison,
    comparison_markdown,
)


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "configs/evaluation/simct_paper_one_seed_generation.json"
CONFIGS = {
    "sft": ROOT
    / "configs/reproduction/qwen25_7b_to_gemma2_2b_public_sft_screen.json",
    "simple_opd": ROOT
    / "configs/reproduction/qwen25_7b_to_gemma2_2b_public_simple_opd_screen.json",
    "simct": ROOT
    / "configs/reproduction/qwen25_7b_to_gemma2_2b_public_simct_screen.json",
}
STUDENT_HASHES = {"sft": "a" * 64, "simple_opd": "b" * 64, "simct": "c" * 64}


def _write(path: Path, value) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    return path


def _fixture(tmp_path: Path):
    protocol = load_generation_protocol(PROTOCOL)
    evaluation = tmp_path / "evaluation"
    for benchmark in protocol.benchmarks:
        _write(
            evaluation / benchmark.name / "manifest.json",
            {
                "name": benchmark.name,
                "dataset_id": f"fixture/{benchmark.name}",
                "dataset_revision": f"{benchmark.name}-immutable-r1",
                "split": "test",
                "record_count": benchmark.record_count,
                "records_sha256": benchmark.records_sha256,
            },
        )

    variants = []
    payloads = {}
    prompt_preflight = [
        {
            "benchmark": item.name,
            "record_count": item.record_count,
            "minimum_prompt_tokens": 8,
            "maximum_prompt_tokens": 32,
            "max_prompt_tokens_contract": protocol.max_prompt_tokens,
        }
        for item in protocol.benchmarks
    ]
    for variant_index, name in enumerate(("sft", "simple_opd", "simct")):
        config = load_config(CONFIGS[name])
        dataset_sha = "d" * 64 if name == "sft" else "e" * 64
        training = {
            "status": "complete",
            "phase": f"{name}_training",
            "run_id": config.run_id,
            "config_sha256": config.digest(),
            "completed_steps": config.training.max_steps,
            "final_student_parameters_sha256": STUDENT_HASHES[name],
            "dataset_manifest_sha256": dataset_sha,
            "hardware": {
                "backend": "tpu",
                "device_count": 8,
                "v5e_kind_match": True,
            },
            "scientific_evidence": False,
        }
        if name != "sft":
            training.update(
                {
                    "objective": name,
                    "initialization": "warm_start",
                    "source_student_parameters_sha256": STUDENT_HASHES["sft"],
                }
            )
        generation = {
            "status": "complete",
            "phase": "native_tunix_generation",
            "variant": name,
            "protocol_id": protocol.protocol_id,
            "protocol_sha256": protocol.digest(),
            "training_config_sha256": config.digest(),
            "checkpoint_run_id": config.run_id,
            "checkpoint_steps": config.training.max_steps,
            "student_parameters_sha256": STUDENT_HASHES[name],
            "training_dataset_manifest_sha256": dataset_sha,
            "hardware": {"backend": "tpu", "device_count": 8},
            "prompt_preflight_sha256": "1" * 64,
            "prompt_preflight": prompt_preflight,
            "benchmarks": [
                {
                    "benchmark": item.name,
                    "record_count": item.record_count,
                    "predictions_sha256": f"{variant_index + 2:x}" * 64,
                }
                for item in protocol.benchmarks
            ],
            "scientific_evidence": False,
        }
        scoring_benchmarks = []
        for benchmark_index, item in enumerate(protocol.benchmarks):
            correct = 10 + variant_index + benchmark_index
            scoring_benchmarks.append(
                {
                    "benchmark": item.name,
                    "metric": "accuracy"
                    if item.name in {"gsm8k", "math500"}
                    else "pass@1",
                    "score": correct / item.record_count,
                    "correct": correct,
                    "total": item.record_count,
                    "scored_predictions_sha256": f"{variant_index + 5:x}" * 64,
                }
            )
        scoring = {
            "status": "complete",
            "phase": "paper_released_evaluator_scoring",
            "variant": name,
            "protocol_id": protocol.protocol_id,
            "protocol_sha256": protocol.digest(),
            "checkpoint_sha256": STUDENT_HASHES[name],
            "evaluator_id": "fixture/paper-evaluator.py",
            "evaluator_revision": protocol.evaluator_revision,
            "evaluator_source_sha256": "f" * 64,
            "evaluator_classification": "paper-released; not official benchmark harness",
            "environment": {
                "python": "3.12.13",
                "multiprocessing_start_method": "fork",
            },
            "benchmarks": scoring_benchmarks,
            "evaluation_evidence": True,
            "scientific_evidence": True,
            "paper_reproduction": False,
        }
        variant_root = tmp_path / name
        training_path = _write(variant_root / "training.json", training)
        generation_path = _write(variant_root / "generation.json", generation)
        scoring_path = _write(variant_root / "scoring.json", scoring)
        variants.append(
            VariantEvidencePaths(
                name=name,
                training_config=CONFIGS[name],
                training_summary=training_path,
                generation_summary=generation_path,
                scoring_summary=scoring_path,
            )
        )
        payloads[name] = {
            "training": training_path,
            "generation": generation_path,
            "scoring": scoring_path,
        }
    return evaluation, tuple(variants), payloads


def test_assembles_three_terminal_variants_and_deltas(tmp_path):
    evaluation, variants, _ = _fixture(tmp_path)
    summary = assemble_one_seed_comparison(
        comparison_id="fixture-one-seed-v1",
        generation_protocol_path=PROTOCOL,
        evaluation_root=evaluation,
        variants=variants,
    )
    assert summary["status"] == "complete"
    assert summary["scientific_evidence"] is True
    assert summary["paper_reproduction"] is False
    assert set(summary["scores"]) == {"sft", "simple_opd", "simct"}
    assert len(summary["comparison_contract_sha256"]) == 64
    assert summary["deltas_vs_sft"]["simct"]["gsm8k"] == pytest.approx(
        2 / 1319
    )
    rendered = comparison_markdown(summary)
    assert "| Benchmark | SFT | SimpleOPD | SimCT |" in rendered
    assert "Paper reproduction: false." in rendered


def test_rejects_warm_start_lineage_drift(tmp_path):
    evaluation, variants, payloads = _fixture(tmp_path)
    path = payloads["simct"]["training"]
    value = json.loads(path.read_text())
    value["source_student_parameters_sha256"] = "9" * 64
    _write(path, value)
    with pytest.raises(OneSeedComparisonError, match="exact SFT warm start"):
        assemble_one_seed_comparison(
            comparison_id="fixture-one-seed-v1",
            generation_protocol_path=PROTOCOL,
            evaluation_root=evaluation,
            variants=variants,
        )


def test_rejects_scoring_environment_drift(tmp_path):
    evaluation, variants, payloads = _fixture(tmp_path)
    path = payloads["simple_opd"]["scoring"]
    value = json.loads(path.read_text())
    value["environment"]["python"] = "3.13.0"
    _write(path, value)
    with pytest.raises(OneSeedComparisonError, match="shared environment"):
        assemble_one_seed_comparison(
            comparison_id="fixture-one-seed-v1",
            generation_protocol_path=PROTOCOL,
            evaluation_root=evaluation,
            variants=variants,
        )


def test_rejects_score_that_disagrees_with_counts(tmp_path):
    evaluation, variants, payloads = _fixture(tmp_path)
    path = payloads["sft"]["scoring"]
    value = json.loads(path.read_text())
    value["benchmarks"][0]["score"] = 0.9
    _write(path, value)
    with pytest.raises(OneSeedComparisonError, match="correct/total"):
        assemble_one_seed_comparison(
            comparison_id="fixture-one-seed-v1",
            generation_protocol_path=PROTOCOL,
            evaluation_root=evaluation,
            variants=variants,
        )
