from __future__ import annotations

from vdt_tunix.training_data import load_prompt_dataset, load_sft_dataset
from vdt_tunix.training_data_materialize import (
    GSM8K_TRAIN_URL,
    GSM8K_TEST_URL,
    MBPP_TRAIN_URL,
    MBPP_TEST_URL,
    materialize_public_substitute,
)


def _gsm_rows():
    return [
        {"question": f"math question {index}", "answer": f"answer {index}"}
        for index in range(7473)
    ]


def _mbpp_rows():
    return [
        {
            "task_id": index,
            "text": f"write function {index}",
            "code": f"def f_{index}():\n    return {index}",
        }
        for index in range(374)
    ]


def _gsm_test_rows():
    return [
        {"question": ("math question 0" if index == 0 else f"test math {index}")}
        for index in range(1319)
    ]


def _mbpp_test_rows():
    return [
        {
            "task_id": 1000 + index,
            "text": ("write function 0" if index == 0 else f"test code {index}"),
        }
        for index in range(500)
    ]


def test_public_substitute_is_deterministic_loadable_and_disclosed(tmp_path):
    payloads = {
        GSM8K_TRAIN_URL: b"gsm-train",
        GSM8K_TEST_URL: b"gsm-test",
        MBPP_TRAIN_URL: b"mbpp-train",
        MBPP_TEST_URL: b"mbpp-test",
    }

    def decoder(content, *, source):
        return {
            b"gsm-train": _gsm_rows,
            b"gsm-test": _gsm_test_rows,
            b"mbpp-train": _mbpp_rows,
            b"mbpp-test": _mbpp_test_rows,
        }[content]()

    first = materialize_public_substitute(
        tmp_path,
        per_source=3,
        seed=42,
        fetcher=lambda url: payloads[url],
        parquet_decoder=decoder,
    )
    sft = load_sft_dataset(tmp_path / "sft" / "manifest.json")
    opd = load_prompt_dataset(tmp_path / "opd" / "manifest.json")
    first_sft_bytes = (tmp_path / "sft" / "records.jsonl").read_bytes()
    second = materialize_public_substitute(
        tmp_path,
        per_source=3,
        seed=42,
        fetcher=lambda url: payloads[url],
        parquet_decoder=decoder,
    )
    assert first == second
    assert first_sft_bytes == (tmp_path / "sft" / "records.jsonl").read_bytes()
    assert len(sft) == len(opd) == 6
    assert [row.prompt_id for row in sft] == [row.prompt_id for row in opd]
    assert {row.source_license for row in sft} == {"MIT", "CC-BY-4.0"}
    assert first["public_data_substitute"] is True
    assert first["paper_training_corpus_reproduced"] is False
    assert first["paper_selected_teacher_trajectories_reproduced"] is False
    assert all("/train" in row.source for row in sft)
    test_sources = [
        source for source in first["sources"] if source["split"] == "test"
    ]
    assert len(test_sources) == 2
    assert {source["purpose"] for source in test_sources} == {
        "decontamination_only"
    }
    assert first["selection"]["excluded_train_rows"] == {
        "gsm8k": 1,
        "mbpp": 1,
    }
    assert all("question 0" not in row.student_prompt for row in sft)
    assert all("function 0" not in row.student_prompt for row in sft)
