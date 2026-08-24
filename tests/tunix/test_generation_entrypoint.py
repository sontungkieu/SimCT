from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from scripts.tpu import kaggle_v5e8_generate as generation_entrypoint
from vdt_tunix.generation_contract import GenerationExample, load_generation_protocol

REPO_ROOT = Path(__file__).resolve().parents[2]


def _install_fake_runtime(monkeypatch, calls):
    protocol = load_generation_protocol(
        REPO_ROOT
        / "configs"
        / "evaluation"
        / "simct_paper_one_seed_generation.json"
    )
    examples = {
        benchmark.name: tuple(
            GenerationExample(
                benchmark=benchmark.name,
                instance_id=f"{benchmark.name}-{index}",
                source_index=index,
                prompt=f"prompt {benchmark.name} {index}",
            )
            for index in range(3)
        )
        for benchmark in protocol.benchmarks
    }

    monkeypatch.setattr(
        generation_entrypoint,
        "iter_generation_examples",
        lambda loaded, root, benchmark: iter(examples[benchmark.name]),
    )
    monkeypatch.setattr(
        generation_entrypoint,
        "restore_student_for_inference",
        lambda config, root: SimpleNamespace(
            checkpoint_run_id=config.run_id,
            checkpoint_steps=10,
            student_parameters_sha256="a" * 64,
            dataset_manifest_sha256="b" * 64,
            hardware={"device_count": 8},
        ),
    )

    class FakeGenerator:
        def __init__(self, restored, **kwargs):
            del restored, kwargs

        def prompt_lengths(self, prompts):
            return tuple(2 for _ in prompts)

        def generate(self, prompts, **kwargs):
            calls.append((tuple(prompts), kwargs))
            return (
                tuple(f"completion {index}" for index in range(len(prompts))),
                tuple(2 for _ in prompts),
                tuple(1 for _ in prompts),
            )

    monkeypatch.setattr(generation_entrypoint, "NativeTunixGenerator", FakeGenerator)


def _args(tmp_path, output):
    return [
        "--training-config",
        str(
            REPO_ROOT
            / "configs"
            / "reproduction"
            / "qwen25_7b_to_gemma2_2b_public_sft_screen.json"
        ),
        "--generation-protocol",
        str(
            REPO_ROOT
            / "configs"
            / "evaluation"
            / "simct_paper_one_seed_generation.json"
        ),
        "--evaluation-root",
        str(tmp_path / "evaluation"),
        "--checkpoint-root",
        str(tmp_path / "checkpoint"),
        "--variant",
        "sft",
        "--output-dir",
        str(output),
    ]


def test_generation_entrypoint_writes_and_resumes_atomic_batches(
    tmp_path, monkeypatch
):
    calls = []
    _install_fake_runtime(monkeypatch, calls)
    output = tmp_path / "generation"
    assert generation_entrypoint.main(_args(tmp_path, output)) == 0
    assert len(calls) == 4
    summary = json.loads(
        (output / "generation_summary.json").read_text(encoding="utf-8")
    )
    assert summary["status"] == "complete"
    assert summary["scientific_evidence"] is False
    assert len(summary["benchmarks"]) == 4
    assert all(item["record_count"] == 3 for item in summary["benchmarks"])
    assert all(
        item["maximum_prompt_tokens"] == 2
        for item in summary["prompt_preflight"]
    )
    assert len(summary["prompt_preflight_sha256"]) == 64

    calls.clear()
    assert generation_entrypoint.main(_args(tmp_path, output)) == 0
    assert calls == []


def test_generation_resume_rejects_typed_field_tampering(tmp_path, monkeypatch):
    calls = []
    _install_fake_runtime(monkeypatch, calls)
    output = tmp_path / "generation"
    assert generation_entrypoint.main(_args(tmp_path, output)) == 0
    batch = output / "gsm8k" / "batches" / "batch_000000.jsonl"
    rows = [json.loads(line) for line in batch.read_text(encoding="utf-8").splitlines()]
    rows[0]["completion_tokens"] = "1"
    batch.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )

    calls.clear()
    assert generation_entrypoint.main(_args(tmp_path, output)) == 69
    assert calls == []
    summary = json.loads(
        (output / "generation_summary.json").read_text(encoding="utf-8")
    )
    assert summary["status"] == "blocked"
    assert summary["scientific_evidence"] is False
    assert "completion_tokens is not an integer" in summary["error"]
