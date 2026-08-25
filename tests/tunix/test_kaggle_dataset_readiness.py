from __future__ import annotations

import pytest

from vdt_tunix.kaggle_dataset_readiness import (
    DatasetReadinessError,
    DatasetSnapshot,
    parse_dataset_status,
    parse_files_csv,
    wait_for_dataset_ready,
)


def test_parse_files_csv_is_deterministic() -> None:
    payload = "name,size,creationDate\nb/records.jsonl,9,x\na/manifest.json,3,x\n"
    assert parse_files_csv(payload) == (("a/manifest.json", 3), ("b/records.jsonl", 9))


def test_parse_dataset_status_ignores_cli_permission_warning() -> None:
    payload = "Warning: Kaggle API key is readable by other users\nready\n"
    assert parse_dataset_status(payload) == "ready"


def test_wait_requires_two_identical_ready_snapshots() -> None:
    snapshots = iter(
        [
            DatasetSnapshot("creating", (("a", 10),)),
            DatasetSnapshot("ready", (("a", 10), ("b", 20))),
            DatasetSnapshot("ready", (("a", 10), ("b", 21))),
            DatasetSnapshot("ready", (("a", 10), ("b", 21))),
        ]
    )
    result = wait_for_dataset_ready(
        snapshot_reader=lambda: next(snapshots),
        expected_files=("a", "b"),
        min_total_bytes=30,
        stable_checks=2,
        timeout_s=10,
        interval_s=0,
        sleep=lambda _: None,
    )
    assert result.total_bytes == 31


def test_wait_fails_closed_on_missing_file(monkeypatch: pytest.MonkeyPatch) -> None:
    ticks = iter((0.0, 0.0, 2.0, 2.0))
    monkeypatch.setattr("vdt_tunix.kaggle_dataset_readiness.time.monotonic", lambda: next(ticks))
    with pytest.raises(DatasetReadinessError, match="missing_files"):
        wait_for_dataset_ready(
            snapshot_reader=lambda: DatasetSnapshot("ready", (("a", 10),)),
            expected_files=("a", "b"),
            min_total_bytes=1,
            timeout_s=1,
            interval_s=0,
            sleep=lambda _: None,
        )
