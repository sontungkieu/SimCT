"""Durable Tunix model/optimizer checkpoints plus fail-closed manifests."""

from __future__ import annotations

import dataclasses
import hashlib
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from vdt_tunix.checkpoint import (
    ArtifactRef,
    CheckpointError,
    CheckpointState,
    DataCursor,
    load_latest_checkpoint,
    save_checkpoint,
)
from vdt_tunix.config import RunConfig


class TunixCheckpointError(CheckpointError):
    """Raised when array persistence and the resume manifest disagree."""


def directory_sha256(path: str | Path) -> str:
    """Hash file names, sizes, and bytes in a checkpoint directory."""

    root = Path(path)
    if not root.is_dir():
        raise TunixCheckpointError(f"checkpoint artifact directory is missing: {root}")
    digest = hashlib.sha256()
    files = sorted(item for item in root.rglob("*") if item.is_file() or item.is_symlink())
    if not files:
        raise TunixCheckpointError(f"checkpoint artifact directory is empty: {root}")
    for item in files:
        if item.is_symlink():
            raise TunixCheckpointError(
                f"checkpoint artifact may not contain symlinks: {item}"
            )
        relative = item.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        size = item.stat().st_size
        digest.update(size.to_bytes(8, "big"))
        with item.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def _artifact_uri(step: int, item: str) -> str:
    return f"orbax://{step}#{item}"


def _metadata(
    config: RunConfig,
    *,
    dataset_manifest_sha256: str,
    completed_steps: int,
    data_cursor: DataCursor,
    rng_state: Mapping[str, str],
) -> dict[str, Any]:
    return {
        "run_id": config.run_id,
        "config_sha256": config.digest(),
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "completed_steps": completed_steps,
        "data_cursor": dataclasses.asdict(data_cursor),
        "rng_state": dict(sorted(rng_state.items())),
    }


def _default_manager(root: Path, save_every_steps: int) -> Any:
    """Construct the pinned Tunix manager lazily on the TPU runtime."""

    try:
        from orbax.checkpoint import v1 as ocp
        from tunix.sft import checkpoint_manager, checkpoint_options
    except ImportError as exc:
        raise TunixCheckpointError(
            f"Tunix/Orbax checkpoint dependencies are unavailable: {exc}"
        ) from exc
    options = checkpoint_options.create_checkpointing_options(
        save_decision_policy=(
            ocp.training.save_decision_policies.FixedIntervalPolicy(
                save_every_steps
            )
        ),
        preservation_policy=ocp.training.preservation_policies.LatestN(n=2),
        enable_async_checkpointing=False,
    )
    return checkpoint_manager.CheckpointManager(
        root_directory=str(root), options=options
    )


@dataclasses.dataclass(frozen=True, slots=True)
class ResumeState:
    completed_steps: int
    data_cursor: DataCursor
    rng_state: tuple[tuple[str, str], ...]


class TunixCheckpointController:
    """Coordinate Orbax arrays with the stricter VDT identity manifest."""

    def __init__(
        self,
        config: RunConfig,
        model: Any,
        optimizer: Any,
        *,
        dataset_manifest_sha256: str,
        manager_factory: Callable[[Path, int], Any] | None = None,
    ):
        if len(dataset_manifest_sha256) != 64 or any(
            char not in "0123456789abcdef"
            for char in dataset_manifest_sha256
        ):
            raise TunixCheckpointError(
                "dataset_manifest_sha256 must be a lowercase SHA-256"
            )
        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.dataset_manifest_sha256 = dataset_manifest_sha256
        self.root = Path(config.checkpoint.root).resolve()
        self.resume_root = Path(
            config.checkpoint.resume_from or config.checkpoint.root
        ).resolve()
        self._manager_factory = manager_factory or _default_manager
        self._save_manager: Any | None = None

    def _manager(self, root: Path) -> Any:
        return self._manager_factory(root, self.config.checkpoint.save_every_steps)

    def restore_if_requested(self) -> ResumeState:
        if self.config.checkpoint.resume_from is None:
            return ResumeState(
                completed_steps=0,
                data_cursor=DataCursor(epoch=0, next_prompt_index=0),
                rng_state=(),
            )
        state = load_latest_checkpoint(self.resume_root, config=self.config)
        if state.dataset_manifest_sha256 != self.dataset_manifest_sha256:
            raise TunixCheckpointError(
                "resume dataset manifest does not match the current dataset"
            )
        step_dir = self.resume_root / str(state.completed_steps)
        observed_sha = directory_sha256(step_dir)
        expected_uris = {
            state.student_parameters.uri: _artifact_uri(
                state.completed_steps, "model_params"
            ),
            state.optimizer_state.uri: _artifact_uri(
                state.completed_steps, "optimizer_state"
            ),
        }
        if any(actual != expected for actual, expected in expected_uris.items()):
            raise TunixCheckpointError("checkpoint artifact URI mismatch")
        expected_hashes = {
            state.student_parameters.sha256,
            state.optimizer_state.sha256,
        }
        if expected_hashes != {observed_sha}:
            raise TunixCheckpointError("checkpoint artifact SHA-256 mismatch")

        manager = self._manager(self.resume_root)
        try:
            restored_step, restored_metadata = manager.maybe_restore(
                self.model,
                self.optimizer,
                step=state.completed_steps,
            )
        finally:
            manager.close()
        if restored_step != state.completed_steps:
            raise TunixCheckpointError(
                "Tunix restored a different checkpoint coordinate"
            )
        expected_metadata = _metadata(
            self.config,
            dataset_manifest_sha256=self.dataset_manifest_sha256,
            completed_steps=state.completed_steps,
            data_cursor=state.data_cursor,
            rng_state=dict(state.rng_state),
        )
        if restored_metadata != expected_metadata:
            raise TunixCheckpointError(
                "Orbax custom metadata does not match the resume manifest"
            )
        return ResumeState(
            completed_steps=state.completed_steps,
            data_cursor=state.data_cursor,
            rng_state=state.rng_state,
        )

    def save(
        self,
        *,
        completed_steps: int,
        data_cursor: DataCursor,
        rng_state: Mapping[str, str],
    ) -> CheckpointState:
        if completed_steps < 1:
            raise TunixCheckpointError("completed_steps must be positive when saving")
        if self._save_manager is None:
            self._save_manager = self._manager(self.root)
        metadata = _metadata(
            self.config,
            dataset_manifest_sha256=self.dataset_manifest_sha256,
            completed_steps=completed_steps,
            data_cursor=data_cursor,
            rng_state=rng_state,
        )
        saved = self._save_manager.save(
            completed_steps,
            self.model,
            self.optimizer,
            force=True,
            custom_metadata=metadata,
        )
        if not saved:
            raise TunixCheckpointError("Tunix checkpoint manager did not save")
        step_dir = self.root / str(completed_steps)
        artifact_sha = directory_sha256(step_dir)
        state = CheckpointState.create(
            self.config,
            dataset_manifest_sha256=self.dataset_manifest_sha256,
            completed_steps=completed_steps,
            data_cursor=data_cursor,
            rng_state=rng_state,
            student_parameters=ArtifactRef(
                uri=_artifact_uri(completed_steps, "model_params"),
                sha256=artifact_sha,
            ),
            optimizer_state=ArtifactRef(
                uri=_artifact_uri(completed_steps, "optimizer_state"),
                sha256=artifact_sha,
            ),
        )
        save_checkpoint(self.root, state)
        return state

    def close(self) -> None:
        if self._save_manager is not None:
            self._save_manager.close()
            self._save_manager = None
