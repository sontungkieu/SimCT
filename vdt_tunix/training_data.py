"""Deterministic, provenance-checked prompt datasets for TPU training.

The public SimCT repository does not ship the exact 10K corpus used in the
paper.  This module therefore refuses an implicit or mutable dataset: every
training JSONL file must be named by a strict manifest and verified before a
prompt can enter an on-policy rollout.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from vdt_tunix.checkpoint import DataCursor
from vdt_tunix.contracts import PromptRecord


DATASET_CONTRACT_VERSION = 1


class TrainingDataError(RuntimeError):
    """Raised when dataset identity, content, or cursor state is unsafe."""


def _strict_keys(
    value: Mapping[str, Any], *, context: str, required: set[str]
) -> None:
    missing = sorted(required - value.keys())
    extra = sorted(value.keys() - required)
    if missing or extra:
        raise TrainingDataError(
            f"{context} key mismatch: missing={missing}, unsupported={extra}"
        )


def _nonempty_string(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TrainingDataError(f"{context} must be a non-empty string")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclasses.dataclass(frozen=True, slots=True)
class PromptDatasetManifest:
    contract_version: int
    dataset_id: str
    dataset_revision: str
    split: str
    records_path: str
    records_sha256: str
    record_count: int

    def __post_init__(self) -> None:
        if self.contract_version != DATASET_CONTRACT_VERSION:
            raise TrainingDataError("unsupported prompt dataset contract version")
        for name in ("dataset_id", "dataset_revision", "split", "records_path"):
            _nonempty_string(getattr(self, name), f"dataset.{name}")
        if self.dataset_revision.lower() in {"main", "master", "head", "latest"}:
            raise TrainingDataError("dataset_revision must be immutable")
        if len(self.records_sha256) != 64 or any(
            char not in "0123456789abcdef" for char in self.records_sha256
        ):
            raise TrainingDataError("records_sha256 must be a lowercase SHA-256")
        if self.record_count < 1:
            raise TrainingDataError("record_count must be positive")

    @classmethod
    def from_mapping(cls, value: Any) -> "PromptDatasetManifest":
        if not isinstance(value, Mapping):
            raise TrainingDataError("dataset manifest must be an object")
        required = {
            "contract_version",
            "dataset_id",
            "dataset_revision",
            "split",
            "records_path",
            "records_sha256",
            "record_count",
        }
        _strict_keys(value, context="dataset manifest", required=required)
        version = value["contract_version"]
        count = value["record_count"]
        if isinstance(version, bool) or not isinstance(version, int):
            raise TrainingDataError("contract_version must be an integer")
        if isinstance(count, bool) or not isinstance(count, int):
            raise TrainingDataError("record_count must be an integer")
        return cls(
            contract_version=version,
            dataset_id=_nonempty_string(value["dataset_id"], "dataset_id"),
            dataset_revision=_nonempty_string(
                value["dataset_revision"], "dataset_revision"
            ),
            split=_nonempty_string(value["split"], "split"),
            records_path=_nonempty_string(value["records_path"], "records_path"),
            records_sha256=_nonempty_string(
                value["records_sha256"], "records_sha256"
            ),
            record_count=count,
        )

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def digest(self) -> str:
        encoded = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


@dataclasses.dataclass(frozen=True, slots=True)
class VerifiedPromptDataset(Sequence[PromptRecord]):
    manifest: PromptDatasetManifest
    manifest_path: Path
    records_path: Path
    records: tuple[PromptRecord, ...]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int | slice) -> PromptRecord | tuple[PromptRecord, ...]:
        return self.records[index]

    def batches(
        self,
        *,
        cursor: DataCursor,
        batch_size: int,
        max_steps: int | None = None,
    ) -> Iterator[tuple[tuple[PromptRecord, ...], DataCursor]]:
        """Yield deterministic wraparound batches and the next durable cursor."""

        if batch_size < 1:
            raise TrainingDataError("batch_size must be positive")
        if max_steps is not None and max_steps < 0:
            raise TrainingDataError("max_steps must be non-negative")
        if cursor.next_prompt_index >= len(self.records):
            raise TrainingDataError(
                "data cursor next_prompt_index exceeds dataset length"
            )
        epoch = cursor.epoch
        index = cursor.next_prompt_index
        emitted = 0
        while max_steps is None or emitted < max_steps:
            remaining = len(self.records) - index
            if remaining < batch_size:
                # Dropping the incomplete tail matches the fixed-size training
                # batch contract and makes resume coordinates unambiguous.
                epoch += 1
                index = 0
            batch = self.records[index : index + batch_size]
            if len(batch) != batch_size:
                raise TrainingDataError(
                    "dataset is smaller than the requested prompt batch size"
                )
            index += batch_size
            if index == len(self.records):
                epoch += 1
                index = 0
            emitted += 1
            yield batch, DataCursor(epoch=epoch, next_prompt_index=index)


@dataclasses.dataclass(frozen=True, slots=True)
class SFTRecord:
    prompt_id: str
    student_prompt: str
    teacher_prompt: str
    target_response: str
    source: str
    source_id: str
    source_license: str


@dataclasses.dataclass(frozen=True, slots=True)
class VerifiedSFTDataset(Sequence[SFTRecord]):
    manifest: PromptDatasetManifest
    manifest_path: Path
    records_path: Path
    records: tuple[SFTRecord, ...]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int | slice) -> SFTRecord | tuple[SFTRecord, ...]:
        return self.records[index]

    def batches(
        self,
        *,
        cursor: DataCursor,
        batch_size: int,
        max_steps: int | None = None,
    ) -> Iterator[tuple[tuple[SFTRecord, ...], DataCursor]]:
        prompt_records = tuple(
            PromptRecord(
                prompt_id=row.prompt_id,
                student_prompt=row.student_prompt,
                teacher_prompt=row.teacher_prompt,
            )
            for row in self.records
        )
        proxy = VerifiedPromptDataset(
            manifest=self.manifest,
            manifest_path=self.manifest_path,
            records_path=self.records_path,
            records=prompt_records,
        )
        records_by_id = {record.prompt_id: record for record in self.records}
        for batch, next_cursor in proxy.batches(
            cursor=cursor, batch_size=batch_size, max_steps=max_steps
        ):
            yield (
                tuple(records_by_id[record.prompt_id] for record in batch),
                next_cursor,
            )


def _load_manifest_and_records_path(
    manifest_path: str | Path,
) -> tuple[Path, PromptDatasetManifest, Path]:
    path = Path(manifest_path).resolve()
    try:
        raw_manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TrainingDataError(f"invalid dataset manifest {path}: {exc}") from exc
    manifest = PromptDatasetManifest.from_mapping(raw_manifest)
    records_path = Path(manifest.records_path)
    if not records_path.is_absolute():
        records_path = path.parent / records_path
    records_path = records_path.resolve()
    if not records_path.is_file():
        raise TrainingDataError(f"dataset records do not exist: {records_path}")
    observed_sha = _sha256_file(records_path)
    if observed_sha != manifest.records_sha256:
        raise TrainingDataError(
            "dataset records SHA-256 mismatch: "
            f"declared={manifest.records_sha256} observed={observed_sha}"
        )
    return path, manifest, records_path


def load_prompt_dataset(manifest_path: str | Path) -> VerifiedPromptDataset:
    path, manifest, records_path = _load_manifest_and_records_path(manifest_path)

    records: list[PromptRecord] = []
    seen_ids: set[str] = set()
    for line_number, line in enumerate(
        records_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            raise TrainingDataError(f"blank JSONL row at line {line_number}")
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise TrainingDataError(
                f"invalid JSONL row at line {line_number}: {exc}"
            ) from exc
        if not isinstance(value, Mapping):
            raise TrainingDataError(f"row {line_number} must be an object")
        _strict_keys(
            value,
            context=f"row {line_number}",
            required={"prompt_id", "student_prompt", "teacher_prompt"},
        )
        prompt_id = _nonempty_string(value["prompt_id"], f"row {line_number}.prompt_id")
        if prompt_id in seen_ids:
            raise TrainingDataError(f"duplicate prompt_id {prompt_id!r}")
        seen_ids.add(prompt_id)
        records.append(
            PromptRecord(
                prompt_id=prompt_id,
                student_prompt=_nonempty_string(
                    value["student_prompt"], f"row {line_number}.student_prompt"
                ),
                teacher_prompt=_nonempty_string(
                    value["teacher_prompt"], f"row {line_number}.teacher_prompt"
                ),
            )
        )
    if len(records) != manifest.record_count:
        raise TrainingDataError(
            "dataset record_count mismatch: "
            f"declared={manifest.record_count} observed={len(records)}"
        )
    return VerifiedPromptDataset(
        manifest=manifest,
        manifest_path=path,
        records_path=records_path,
        records=tuple(records),
    )


def load_sft_dataset(manifest_path: str | Path) -> VerifiedSFTDataset:
    path, manifest, records_path = _load_manifest_and_records_path(manifest_path)
    required = {
        "prompt_id",
        "student_prompt",
        "teacher_prompt",
        "target_response",
        "source",
        "source_id",
        "source_license",
    }
    records: list[SFTRecord] = []
    seen_ids: set[str] = set()
    for line_number, line in enumerate(
        records_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            raise TrainingDataError(f"blank JSONL row at line {line_number}")
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise TrainingDataError(
                f"invalid JSONL row at line {line_number}: {exc}"
            ) from exc
        if not isinstance(value, Mapping):
            raise TrainingDataError(f"row {line_number} must be an object")
        _strict_keys(value, context=f"row {line_number}", required=required)
        strings = {
            name: _nonempty_string(value[name], f"row {line_number}.{name}")
            for name in required
        }
        if strings["prompt_id"] in seen_ids:
            raise TrainingDataError(
                f"duplicate prompt_id {strings['prompt_id']!r}"
            )
        seen_ids.add(strings["prompt_id"])
        records.append(SFTRecord(**strings))
    if len(records) != manifest.record_count:
        raise TrainingDataError(
            "dataset record_count mismatch: "
            f"declared={manifest.record_count} observed={len(records)}"
        )
    return VerifiedSFTDataset(
        manifest=manifest,
        manifest_path=path,
        records_path=records_path,
        records=tuple(records),
    )
