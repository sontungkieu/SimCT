from __future__ import annotations

import copy

import pytest

from vdt_tunix.evaluation_contract import (
    ComparisonContract,
    EvaluationContractError,
)


def _payload():
    return {
        "contract_version": 1,
        "comparison_id": "qwen25-gemma2-one-seed-v1",
        "protocol": {
            "paper_revision": "arxiv-2605.07711v2",
            "prompt_protocol": "simct-paper-zero-shot-chat-v1",
            "evaluator_id": "sunjie279/SimCT-/scripts/evaluation/evaluation.py",
            "evaluator_revision": "cf0f33a0e6c967d4b74ea32b2dba12be01b73b9e",
            "temperature": 0.6,
            "top_p": 0.95,
            "samples_per_instance": 1,
            "run_seeds": [42],
            "scope": "one_seed_screen",
            "benchmarks": [
                {
                    "name": name,
                    "dataset_id": f"fixture/{name}",
                    "dataset_revision": f"{name}-immutable-v1",
                    "split": "test",
                    "records_sha256": char * 64,
                    "metric": metric,
                    "max_new_tokens": 4096,
                }
                for (name, metric), char in zip(
                    (
                        ("gsm8k", "accuracy"),
                        ("math500", "accuracy"),
                        ("mbpp", "pass@1"),
                        ("live-code-bench-v6", "pass@1"),
                    ),
                    "abcd",
                )
            ],
        },
        "variants": [
            {
                "name": "sft",
                "objective": "sft",
                "run_id": "sft-v1",
                "completed_steps": 10,
                "student_model_revision": "gemma2-kaggle-v1",
                "student_tokenizer_revision": "gemma2-kaggle-v1",
                "training_dataset_manifest_sha256": "e" * 64,
                "student_parameters_sha256": "f" * 64,
                "source_student_parameters_sha256": None,
            },
            {
                "name": "simple_opd",
                "objective": "simple_opd",
                "run_id": "simple-opd-v1",
                "completed_steps": 20,
                "student_model_revision": "gemma2-kaggle-v1",
                "student_tokenizer_revision": "gemma2-kaggle-v1",
                "training_dataset_manifest_sha256": "1" * 64,
                "student_parameters_sha256": "2" * 64,
                "source_student_parameters_sha256": "f" * 64,
            },
            {
                "name": "simct",
                "objective": "simct",
                "run_id": "simct-v1",
                "completed_steps": 20,
                "student_model_revision": "gemma2-kaggle-v1",
                "student_tokenizer_revision": "gemma2-kaggle-v1",
                "training_dataset_manifest_sha256": "1" * 64,
                "student_parameters_sha256": "3" * 64,
                "source_student_parameters_sha256": "f" * 64,
            },
        ],
    }


def test_comparison_contract_locks_shared_warm_start_and_protocol():
    contract = ComparisonContract.from_mapping(_payload())
    assert contract.protocol.run_seeds == (42,)
    assert len(contract.digest()) == 64


def test_comparison_rejects_different_sft_warm_start():
    payload = _payload()
    payload["variants"][2]["source_student_parameters_sha256"] = "9" * 64
    with pytest.raises(EvaluationContractError, match="exact evaluated SFT"):
        ComparisonContract.from_mapping(payload)


def test_comparison_rejects_different_opd_prompt_manifests():
    payload = _payload()
    payload["variants"][2]["training_dataset_manifest_sha256"] = "9" * 64
    with pytest.raises(EvaluationContractError, match="same OPD prompt"):
        ComparisonContract.from_mapping(payload)


def test_one_seed_scope_cannot_be_mislabeled_as_five_run():
    payload = copy.deepcopy(_payload())
    payload["protocol"]["scope"] = "paper_five_run"
    with pytest.raises(EvaluationContractError, match="requires 5"):
        ComparisonContract.from_mapping(payload)


def test_protocol_requires_all_four_paper_benchmarks():
    payload = _payload()
    payload["protocol"]["benchmarks"].pop()
    with pytest.raises(EvaluationContractError, match="must include"):
        ComparisonContract.from_mapping(payload)
