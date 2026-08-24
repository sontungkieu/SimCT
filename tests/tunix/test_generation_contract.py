from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from vdt_tunix.evaluation_data import PINNED_BENCHMARKS
from vdt_tunix.generation_contract import (
    GEMMA_MERGED_SYSTEM_TEMPLATE,
    GenerationBenchmark,
    GenerationContractError,
    GenerationProtocol,
    gemma_merged_system_prompt,
    generation_example_from_record,
    iter_generation_examples,
    load_generation_protocol,
    stable_batch_seed,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _protocol(records_sha256: str) -> GenerationProtocol:
    benchmarks = []
    for source in PINNED_BENCHMARKS:
        benchmarks.append(
            GenerationBenchmark(
                name=source.name,
                records_sha256=(
                    records_sha256 if source.name == "gsm8k" else "a" * 64
                ),
                record_count=source.expected_count,
                max_new_tokens=2048 if source.name == "mbpp" else 4096,
            )
        )
    return GenerationProtocol(
        contract_version=1,
        protocol_id="fixture-v1",
        paper_revision="arxiv-fixture-v1",
        prompt_protocol=GEMMA_MERGED_SYSTEM_TEMPLATE,
        evaluator_revision="evaluator-fixture-v1",
        seed=42,
        temperature=0.6,
        top_p=0.95,
        samples_per_instance=1,
        batch_size=4,
        max_prompt_tokens=4096,
        benchmarks=tuple(benchmarks),
    )


def test_checked_in_generation_protocol_pins_all_paper_records():
    protocol = load_generation_protocol(
        REPO_ROOT
        / "configs"
        / "evaluation"
        / "simct_paper_one_seed_generation.json"
    )
    assert protocol.seed == 42
    assert protocol.temperature == 0.6
    assert protocol.top_p == 0.95
    assert sum(item.record_count for item in protocol.benchmarks) == 3374
    assert len(protocol.digest()) == 64


def test_gemma_prompt_merges_system_into_one_user_turn():
    prompt = gemma_merged_system_prompt("system rule", "question")
    assert prompt == (
        "<start_of_turn>user\n"
        "system rule\n\nquestion"
        "<end_of_turn>\n<start_of_turn>model\n"
    )
    assert "<start_of_turn>system" not in prompt


def test_public_example_builder_matches_gsm8k_identity():
    example = generation_example_from_record(
        "gsm8k",
        {"question": "What is 1+1?", "answer": "#### 2"},
        7,
    )
    assert example.instance_id == "gsm8k-0007"
    assert example.source_index == 7
    assert "What is 1+1?" in example.prompt


def test_stable_batch_seed_is_coordinate_specific():
    protocol = _protocol("b" * 64)
    assert stable_batch_seed(protocol, "gsm8k", 0) == stable_batch_seed(
        protocol, "gsm8k", 0
    )
    assert stable_batch_seed(protocol, "gsm8k", 0) != stable_batch_seed(
        protocol, "gsm8k", 1
    )
    assert stable_batch_seed(protocol, "gsm8k", 0) != stable_batch_seed(
        protocol, "math500", 0
    )


def test_iter_examples_verifies_materialization_and_order(tmp_path):
    source = next(item for item in PINNED_BENCHMARKS if item.name == "gsm8k")
    root = tmp_path / "gsm8k"
    root.mkdir()
    records = root / "records.jsonl"
    with records.open("w", encoding="utf-8") as stream:
        for index in range(source.expected_count):
            stream.write(
                json.dumps(
                    {
                        "question": f"What is {index}+1?",
                        "answer": f"#### {index + 1}",
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )
    records_sha = hashlib.sha256(records.read_bytes()).hexdigest()
    manifest = {
        "contract_version": 1,
        "name": source.name,
        "dataset_id": source.dataset_id,
        "dataset_revision": source.dataset_revision,
        "config": source.config,
        "split": source.split,
        "records_path": "records.jsonl",
        "records_sha256": records_sha,
        "record_count": source.expected_count,
        "required_fields": list(source.required_fields),
        "source_files": [
            {
                "path": item.path,
                "url": item.url,
                "bytes": 1,
                "sha256": "c" * 64,
            }
            for item in source.files
        ],
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    protocol = _protocol(records_sha)
    benchmark = next(item for item in protocol.benchmarks if item.name == "gsm8k")
    examples = tuple(iter_generation_examples(protocol, tmp_path, benchmark))
    assert len(examples) == 1319
    assert examples[0].instance_id == "gsm8k-0000"
    assert examples[-1].instance_id == "gsm8k-1318"
    assert "Please reason step by step" in examples[0].prompt


def test_protocol_rejects_public_script_top_p_drift():
    protocol = _protocol("d" * 64)
    payload = json.loads(protocol.canonical_json())
    payload["top_p"] = 1.0
    with pytest.raises(GenerationContractError, match="top_p=0.95"):
        GenerationProtocol.from_mapping(payload)
