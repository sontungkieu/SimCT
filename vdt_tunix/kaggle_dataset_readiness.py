"""Fail-closed readiness checks for Kaggle dataset dependencies."""

from __future__ import annotations

import csv
import hashlib
import io
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence


AUTH_ENV_NAMES = (
    "KAGGLE_API_V1_TOKEN",
    "KAGGLE_API_TOKEN",
    "KAGGLE_USERNAME",
    "KAGGLE_KEY",
)


class DatasetReadinessError(RuntimeError):
    """Raised when a dataset never reaches a stable, complete state."""


@dataclass(frozen=True)
class DatasetSnapshot:
    status: str
    files: tuple[tuple[str, int], ...]

    @property
    def total_bytes(self) -> int:
        return sum(size for _, size in self.files)

    @property
    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        for name, size in self.files:
            digest.update(f"{name}\t{size}\n".encode())
        return digest.hexdigest()


def parse_files_csv(payload: str) -> tuple[tuple[str, int], ...]:
    rows: list[tuple[str, int]] = []
    for row in csv.DictReader(io.StringIO(payload)):
        name = (row.get("name") or "").strip()
        raw_size = (row.get("size") or "").strip()
        if not name or not raw_size:
            continue
        try:
            size = int(raw_size)
        except ValueError as exc:
            raise DatasetReadinessError(f"invalid dataset file size for {name!r}") from exc
        rows.append((name, size))
    return tuple(sorted(rows))


def parse_dataset_status(payload: str) -> str:
    """Return the canonical status while ignoring Kaggle CLI warning lines."""
    for line in reversed(payload.splitlines()):
        candidate = line.strip().lower()
        if candidate in {"ready", "creating", "error"}:
            return candidate
        if candidate.startswith("dataset status:"):
            return candidate.partition(":")[2].strip()
    raise DatasetReadinessError("Kaggle dataset status response was not recognized")


def _run_kaggle(
    kaggle_bin: str,
    config_dir: Path,
    args: Sequence[str],
    *,
    timeout_s: float,
) -> str:
    env = os.environ.copy()
    for name in AUTH_ENV_NAMES:
        env.pop(name, None)
    env["KAGGLE_CONFIG_DIR"] = str(config_dir)
    completed = subprocess.run(
        [kaggle_bin, *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
        timeout=timeout_s,
        env=env,
    )
    if completed.returncode != 0:
        tail = completed.stderr.strip()[-500:]
        raise DatasetReadinessError(
            f"Kaggle command failed ({completed.returncode}): {tail}"
        )
    return completed.stdout


def read_dataset_snapshot(
    *,
    kaggle_bin: str,
    config_dir: Path,
    dataset_source: str,
    command_timeout_s: float = 60.0,
) -> DatasetSnapshot:
    status = parse_dataset_status(
        _run_kaggle(
            kaggle_bin,
            config_dir,
            ("datasets", "status", dataset_source),
            timeout_s=command_timeout_s,
        )
    )
    files = parse_files_csv(
        _run_kaggle(
            kaggle_bin,
            config_dir,
            ("datasets", "files", dataset_source, "--csv"),
            timeout_s=command_timeout_s,
        )
    )
    return DatasetSnapshot(status=status, files=files)


def wait_for_dataset_ready(
    *,
    snapshot_reader: Callable[[], DatasetSnapshot],
    expected_files: Sequence[str],
    min_total_bytes: int,
    stable_checks: int = 2,
    timeout_s: float = 300.0,
    interval_s: float = 10.0,
    sleep: Callable[[float], None] = time.sleep,
) -> DatasetSnapshot:
    if stable_checks < 1:
        raise ValueError("stable_checks must be positive")
    deadline = time.monotonic() + timeout_s
    previous_fingerprint = ""
    consecutive = 0
    expected = set(expected_files)
    last_reason = "no snapshot"
    while True:
        snapshot = snapshot_reader()
        names = {name for name, _ in snapshot.files}
        missing = sorted(expected - names)
        if snapshot.status != "ready":
            last_reason = f"status={snapshot.status!r}"
            consecutive = 0
        elif missing:
            last_reason = f"missing_files={missing}"
            consecutive = 0
        elif snapshot.total_bytes < min_total_bytes:
            last_reason = (
                f"total_bytes={snapshot.total_bytes} < min_total_bytes={min_total_bytes}"
            )
            consecutive = 0
        else:
            if snapshot.fingerprint == previous_fingerprint:
                consecutive += 1
            else:
                previous_fingerprint = snapshot.fingerprint
                consecutive = 1
            last_reason = f"stable_checks={consecutive}/{stable_checks}"
            if consecutive >= stable_checks:
                return snapshot
        if time.monotonic() >= deadline:
            raise DatasetReadinessError(
                f"dataset did not become stably ready: {last_reason}"
            )
        sleep(interval_s)
