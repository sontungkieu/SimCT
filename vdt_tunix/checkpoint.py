"""Atomic metadata contract for exact, fail-closed resume decisions.

The scaffold does not serialize JAX arrays. Real Tunix/MaxText integration must
first persist student parameters and optimizer state (for example via Orbax),
then pass immutable artifact references into this manifest layer.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

from vdt_tunix.config import RunConfig


CHECKPOINT_CONTRACT_VERSION = 1
LATEST_POINTER = "latest.json"


class CheckpointError(RuntimeError):
    """Raised when checkpoint identity, integrity, or ordering is unsafe."""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_sha256(value: str, context: str) -> None:
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise CheckpointError(f"{context} must be a lowercase SHA-256 digest")


def _strict_keys(
    value: Mapping[str, Any], *, context: str, required: set[str]
) -> None:
    missing = sorted(required - value.keys())
    extra = sorted(value.keys() - required)
    if missing or extra:
        raise CheckpointError(
            f"{context} key mismatch: missing={missing}, unsupported={extra}"
        )


@dataclasses.dataclass(frozen=True, slots=True)
class ArtifactRef:
    uri: str
    sha256: str

    def __post_init__(self) -> None:
        if not self.uri:
            raise CheckpointError("artifact uri must be non-empty")
        _require_sha256(self.sha256, "artifact sha256")

    @classmethod
    def from_dict(cls, value: Any) -> ArtifactRef:
        if not isinstance(value, Mapping):
            raise CheckpointError("artifact reference must be an object")
        _strict_keys(value, context="artifact", required={"uri", "sha256"})
        if not isinstance(value["uri"], str) or not isinstance(value["sha256"], str):
            raise CheckpointError("artifact uri and sha256 must be strings")
        return cls(uri=value["uri"], sha256=value["sha256"])


@dataclasses.dataclass(frozen=True, slots=True)
class DataCursor:
    epoch: int
    next_prompt_index: int

    def __post_init__(self) -> None:
        if self.epoch < 0 or self.next_prompt_index < 0:
            raise CheckpointError("data cursor values must be non-negative")

    @classmethod
    def from_dict(cls, value: Any) -> DataCursor:
        if not isinstance(value, Mapping):
            raise CheckpointError("data_cursor must be an object")
        _strict_keys(
            value,
            context="data_cursor",
            required={"epoch", "next_prompt_index"},
        )
        epoch = value["epoch"]
        index = value["next_prompt_index"]
        if isinstance(epoch, bool) or not isinstance(epoch, int):
            raise CheckpointError("data_cursor.epoch must be an integer")
        if isinstance(index, bool) or not isinstance(index, int):
            raise CheckpointError(
                "data_cursor.next_prompt_index must be an integer"
            )
        return cls(epoch=epoch, next_prompt_index=index)


@dataclasses.dataclass(frozen=True, slots=True)
class CheckpointState:
    contract_version: int
    run_id: str
    config_sha256: str
    completed_steps: int
    student_model_revision: str
    student_tokenizer_revision: str
    teacher_model_revision: str
    teacher_tokenizer_revision: str
    data_cursor: DataCursor
    rng_state: tuple[tuple[str, str], ...]
    student_parameters: ArtifactRef
    optimizer_state: ArtifactRef

    def __post_init__(self) -> None:
        if self.contract_version != CHECKPOINT_CONTRACT_VERSION:
            raise CheckpointError("unsupported checkpoint contract version")
        if not self.run_id:
            raise CheckpointError("checkpoint run_id must be non-empty")
        _require_sha256(self.config_sha256, "config_sha256")
        if self.completed_steps < 0:
            raise CheckpointError("completed_steps must be non-negative")
        if not all(
            (
                self.student_model_revision,
                self.student_tokenizer_revision,
                self.teacher_model_revision,
                self.teacher_tokenizer_revision,
            )
        ):
            raise CheckpointError("all model/tokenizer revisions are required")
        keys = [key for key, _ in self.rng_state]
        if len(keys) != len(set(keys)) or keys != sorted(keys):
            raise CheckpointError("rng_state keys must be unique and sorted")
        if not keys or any(not key or not value for key, value in self.rng_state):
            raise CheckpointError("rng_state entries must be non-empty")

    @classmethod
    def create(
        cls,
        config: RunConfig,
        *,
        completed_steps: int,
        data_cursor: DataCursor,
        rng_state: Mapping[str, str],
        student_parameters: ArtifactRef,
        optimizer_state: ArtifactRef,
    ) -> CheckpointState:
        if any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in rng_state.items()
        ):
            raise CheckpointError("rng_state keys and values must be strings")
        state = cls(
            contract_version=CHECKPOINT_CONTRACT_VERSION,
            run_id=config.run_id,
            config_sha256=config.digest(),
            completed_steps=completed_steps,
            student_model_revision=config.student.model_revision,
            student_tokenizer_revision=config.student.tokenizer_revision,
            teacher_model_revision=config.teacher.model_revision,
            teacher_tokenizer_revision=config.teacher.tokenizer_revision,
            data_cursor=data_cursor,
            rng_state=tuple(sorted(rng_state.items())),
            student_parameters=student_parameters,
            optimizer_state=optimizer_state,
        )
        validate_resume(config, state)
        return state

    @classmethod
    def from_dict(cls, value: Any) -> CheckpointState:
        if not isinstance(value, Mapping):
            raise CheckpointError("checkpoint manifest must be an object")
        required = {
            "contract_version",
            "run_id",
            "config_sha256",
            "completed_steps",
            "student_model_revision",
            "student_tokenizer_revision",
            "teacher_model_revision",
            "teacher_tokenizer_revision",
            "data_cursor",
            "rng_state",
            "student_parameters",
            "optimizer_state",
        }
        _strict_keys(value, context="checkpoint", required=required)
        rng = value["rng_state"]
        if not isinstance(rng, Mapping):
            raise CheckpointError("rng_state must be an object")
        if any(
            not isinstance(key, str) or not isinstance(item, str)
            for key, item in rng.items()
        ):
            raise CheckpointError("rng_state keys and values must be strings")
        completed = value["completed_steps"]
        version = value["contract_version"]
        if isinstance(completed, bool) or not isinstance(completed, int):
            raise CheckpointError("completed_steps must be an integer")
        if isinstance(version, bool) or not isinstance(version, int):
            raise CheckpointError("contract_version must be an integer")
        string_fields = {
            name: value[name]
            for name in (
                "run_id",
                "config_sha256",
                "student_model_revision",
                "student_tokenizer_revision",
                "teacher_model_revision",
                "teacher_tokenizer_revision",
            )
        }
        if any(not isinstance(item, str) for item in string_fields.values()):
            raise CheckpointError("checkpoint identity fields must be strings")
        return cls(
            contract_version=version,
            run_id=string_fields["run_id"],
            config_sha256=string_fields["config_sha256"],
            completed_steps=completed,
            student_model_revision=string_fields["student_model_revision"],
            student_tokenizer_revision=string_fields["student_tokenizer_revision"],
            teacher_model_revision=string_fields["teacher_model_revision"],
            teacher_tokenizer_revision=string_fields["teacher_tokenizer_revision"],
            data_cursor=DataCursor.from_dict(value["data_cursor"]),
            rng_state=tuple(sorted(rng.items())),
            student_parameters=ArtifactRef.from_dict(value["student_parameters"]),
            optimizer_state=ArtifactRef.from_dict(value["optimizer_state"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract_version": self.contract_version,
            "run_id": self.run_id,
            "config_sha256": self.config_sha256,
            "completed_steps": self.completed_steps,
            "student_model_revision": self.student_model_revision,
            "student_tokenizer_revision": self.student_tokenizer_revision,
            "teacher_model_revision": self.teacher_model_revision,
            "teacher_tokenizer_revision": self.teacher_tokenizer_revision,
            "data_cursor": dataclasses.asdict(self.data_cursor),
            "rng_state": dict(self.rng_state),
            "student_parameters": dataclasses.asdict(self.student_parameters),
            "optimizer_state": dataclasses.asdict(self.optimizer_state),
        }


def validate_resume(config: RunConfig, state: CheckpointState) -> None:
    expected = {
        "run_id": config.run_id,
        "config_sha256": config.digest(),
        "student_model_revision": config.student.model_revision,
        "student_tokenizer_revision": config.student.tokenizer_revision,
        "teacher_model_revision": config.teacher.model_revision,
        "teacher_tokenizer_revision": config.teacher.tokenizer_revision,
    }
    mismatches = {
        name: (getattr(state, name), expected_value)
        for name, expected_value in expected.items()
        if getattr(state, name) != expected_value
    }
    if mismatches:
        raise CheckpointError(f"resume identity mismatch: {mismatches}")
    if state.completed_steps > config.training.max_steps:
        raise CheckpointError(
            "checkpoint completed_steps exceeds training.max_steps"
        )


def _canonical_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        + "\n"
    ).encode("utf-8")


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(content)
    temporary.replace(path)


def _load_pointer(root: Path) -> dict[str, Any] | None:
    pointer_path = root / LATEST_POINTER
    if not pointer_path.is_file():
        return None
    try:
        value = json.loads(pointer_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CheckpointError("latest checkpoint pointer is invalid JSON") from exc
    if not isinstance(value, Mapping):
        raise CheckpointError("latest checkpoint pointer must be an object")
    _strict_keys(
        value,
        context="latest pointer",
        required={"contract_version", "completed_steps", "manifest", "manifest_sha256"},
    )
    if any(
        isinstance(value[name], bool) or not isinstance(value[name], int)
        for name in ("contract_version", "completed_steps")
    ):
        raise CheckpointError("latest pointer version and step must be integers")
    if not isinstance(value["manifest"], str) or not isinstance(
        value["manifest_sha256"], str
    ):
        raise CheckpointError("latest pointer path and digest must be strings")
    return dict(value)


def save_checkpoint(root: str | Path, state: CheckpointState) -> Path:
    """Atomically publish one immutable manifest and then its latest pointer."""

    checkpoint_root = Path(root)
    pointer = _load_pointer(checkpoint_root)
    if pointer is not None:
        latest_steps = pointer["completed_steps"]
        if isinstance(latest_steps, bool) or not isinstance(latest_steps, int):
            raise CheckpointError("latest pointer completed_steps must be an integer")
        if state.completed_steps < latest_steps:
            raise CheckpointError("refusing to move latest checkpoint backwards")

    relative = PurePosixPath(
        f"step_{state.completed_steps:08d}", "resume_manifest.json"
    )
    manifest_path = checkpoint_root.joinpath(*relative.parts)
    content = _canonical_bytes(state.to_dict())
    if manifest_path.exists():
        if manifest_path.read_bytes() != content:
            raise CheckpointError(
                "checkpoint coordinate already exists with different state"
            )
    else:
        _atomic_write(manifest_path, content)

    pointer_content = _canonical_bytes(
        {
            "contract_version": CHECKPOINT_CONTRACT_VERSION,
            "completed_steps": state.completed_steps,
            "manifest": relative.as_posix(),
            "manifest_sha256": _sha256_bytes(content),
        }
    )
    _atomic_write(checkpoint_root / LATEST_POINTER, pointer_content)
    return manifest_path


def load_latest_checkpoint(
    root: str | Path, *, config: RunConfig
) -> CheckpointState:
    checkpoint_root = Path(root)
    pointer = _load_pointer(checkpoint_root)
    if pointer is None:
        raise CheckpointError(f"no {LATEST_POINTER} under {checkpoint_root}")
    if pointer["contract_version"] != CHECKPOINT_CONTRACT_VERSION:
        raise CheckpointError("unsupported latest pointer contract version")
    relative = PurePosixPath(pointer["manifest"])
    if relative.is_absolute() or ".." in relative.parts:
        raise CheckpointError("latest pointer manifest path is unsafe")
    manifest_path = checkpoint_root.joinpath(*relative.parts)
    try:
        manifest_path.resolve().relative_to(checkpoint_root.resolve())
    except ValueError as exc:
        raise CheckpointError("latest pointer escapes checkpoint root") from exc
    if not manifest_path.is_file():
        raise CheckpointError(f"checkpoint manifest does not exist: {manifest_path}")
    raw = manifest_path.read_bytes()
    declared_sha = pointer["manifest_sha256"]
    _require_sha256(declared_sha, "manifest_sha256")
    if _sha256_bytes(raw) != declared_sha:
        raise CheckpointError("checkpoint manifest SHA-256 mismatch")
    try:
        state = CheckpointState.from_dict(json.loads(raw.decode("utf-8")))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CheckpointError("checkpoint manifest is invalid JSON") from exc
    if state.completed_steps != pointer["completed_steps"]:
        raise CheckpointError("checkpoint pointer step does not match manifest")
    validate_resume(config, state)
    return state
