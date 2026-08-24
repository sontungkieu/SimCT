from __future__ import annotations

import hashlib
import json

import pytest

from vdt_tunix.checkpoint import DataCursor
from vdt_tunix.training_data import TrainingDataError, load_prompt_dataset


def _write_dataset(tmp_path, rows):
    records = tmp_path / "prompts.jsonl"
    records.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "contract_version": 1,
                "dataset_id": "simct/test-prompts",
                "dataset_revision": "sha256-fixture-v1",
                "split": "train",
                "records_path": records.name,
                "records_sha256": hashlib.sha256(records.read_bytes()).hexdigest(),
                "record_count": len(rows),
            }
        ),
        encoding="utf-8",
    )
    return manifest, records


def _rows(count=5):
    return [
        {
            "prompt_id": f"p-{index}",
            "student_prompt": f"student {index}",
            "teacher_prompt": f"teacher {index}",
        }
        for index in range(count)
    ]


def test_load_prompt_dataset_verifies_identity_and_content(tmp_path):
    manifest, _ = _write_dataset(tmp_path, _rows())
    dataset = load_prompt_dataset(manifest)
    assert len(dataset) == 5
    assert dataset[0].prompt_id == "p-0"
    assert len(dataset.manifest.digest()) == 64


def test_prompt_dataset_rejects_tampering(tmp_path):
    manifest, records = _write_dataset(tmp_path, _rows())
    records.write_text(records.read_text(encoding="utf-8") + "{}\n", encoding="utf-8")
    with pytest.raises(TrainingDataError, match="SHA-256 mismatch"):
        load_prompt_dataset(manifest)


def test_prompt_dataset_rejects_duplicate_ids(tmp_path):
    rows = _rows(2)
    rows[1]["prompt_id"] = rows[0]["prompt_id"]
    manifest, _ = _write_dataset(tmp_path, rows)
    with pytest.raises(TrainingDataError, match="duplicate prompt_id"):
        load_prompt_dataset(manifest)


def test_batches_drop_tail_and_return_exact_resume_cursor(tmp_path):
    manifest, _ = _write_dataset(tmp_path, _rows())
    dataset = load_prompt_dataset(manifest)
    batches = list(
        dataset.batches(
            cursor=DataCursor(epoch=0, next_prompt_index=0),
            batch_size=2,
            max_steps=3,
        )
    )
    assert [[record.prompt_id for record in batch] for batch, _ in batches] == [
        ["p-0", "p-1"],
        ["p-2", "p-3"],
        ["p-0", "p-1"],
    ]
    assert [cursor for _, cursor in batches] == [
        DataCursor(epoch=0, next_prompt_index=2),
        DataCursor(epoch=0, next_prompt_index=4),
        DataCursor(epoch=1, next_prompt_index=2),
    ]


def test_batches_reject_cursor_outside_dataset(tmp_path):
    manifest, _ = _write_dataset(tmp_path, _rows(2))
    dataset = load_prompt_dataset(manifest)
    with pytest.raises(TrainingDataError, match="exceeds dataset length"):
        next(
            dataset.batches(
                cursor=DataCursor(epoch=0, next_prompt_index=2),
                batch_size=1,
            )
        )
