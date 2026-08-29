"""Deterministic Kaggle model-source metadata and canary notebook helpers."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Sequence


_MODEL_SOURCE = re.compile(
    r"^[a-z0-9][a-z0-9_-]*/"
    r"[A-Za-z0-9][A-Za-z0-9._-]*/"
    r"[A-Za-z0-9][A-Za-z0-9._-]*/"
    r"[A-Za-z0-9][A-Za-z0-9._-]*/"
    r"[1-9][0-9]*$"
)
class KaggleModelSourceError(ValueError):
    """Raised when a generated Kaggle package would have ambiguous inputs."""


def validate_model_source(source: str) -> str:
    """Validate Kaggle's owner/model/framework/variation/version handle."""

    if not isinstance(source, str) or not _MODEL_SOURCE.fullmatch(source):
        raise KaggleModelSourceError(
            "model source must be owner/model/framework/variation/version: "
            f"{source!r}"
        )
    return source


def model_source_mount(source: str) -> PurePosixPath:
    """Return Kaggle's owner-preserving model mount for an exact source."""

    validated = validate_model_source(source)
    return PurePosixPath("/kaggle/input/models") / validated


def resolve_model_source_mount(
    source: str,
    *,
    model_download: Callable[[str], str | Path] | None = None,
) -> Path:
    """Resolve an attached Kaggle model without assuming its mount layout.

    Kaggle's supported notebook interface is ``kagglehub.model_download``.
    The attached model is resolved from Kaggle's local cache, while the exact
    versioned handle keeps the dependency pinned.  ``model_download`` is
    injectable so the behavior can be tested without network or Kaggle state.
    """

    validated = validate_model_source(source)
    if model_download is None:
        try:
            import kagglehub
        except ImportError as exc:  # pragma: no cover - Kaggle runtime contract
            raise KaggleModelSourceError(
                "kagglehub is required to resolve attached Kaggle models"
            ) from exc
        model_download = kagglehub.model_download
    try:
        resolved = Path(model_download(validated))
    except Exception as exc:
        raise KaggleModelSourceError(
            f"failed to resolve attached Kaggle model {validated}"
        ) from exc
    if not resolved.is_dir():
        raise KaggleModelSourceError(
            f"Kaggle model resolver returned a missing directory for "
            f"{validated}: {resolved}"
        )
    return resolved


def bind_runtime_model_mount(
    config_section: dict[str, Any], mount: str | Path
) -> dict[str, str]:
    """Rewrite one model config section to a resolved runtime directory."""

    required = ("model_path", "tokenizer_path", "maxtext_checkpoint_uri")
    missing = [key for key in required if not isinstance(config_section.get(key), str)]
    if missing:
        raise KaggleModelSourceError(
            f"model config section is missing string fields: {missing}"
        )
    resolved = Path(mount)
    if not resolved.is_dir():
        raise KaggleModelSourceError(f"resolved model mount is absent: {resolved}")
    previous_model = PurePosixPath(config_section["model_path"])
    previous_tokenizer = PurePosixPath(config_section["tokenizer_path"])
    if previous_tokenizer == previous_model:
        tokenizer = resolved
    else:
        tokenizer = resolved / previous_tokenizer.name
        if not tokenizer.exists():
            raise KaggleModelSourceError(
                f"tokenizer asset {previous_tokenizer.name!r} is absent in {resolved}"
            )
    config_section["model_path"] = str(resolved)
    config_section["maxtext_checkpoint_uri"] = str(resolved)
    config_section["tokenizer_path"] = str(tokenizer)
    return {
        "model_path": str(resolved),
        "maxtext_checkpoint_uri": str(resolved),
        "tokenizer_path": str(tokenizer),
    }


def _read_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise KaggleModelSourceError(f"required JSON file is absent: {path}") from exc
    except json.JSONDecodeError as exc:
        raise KaggleModelSourceError(f"invalid JSON file: {path}") from exc
    if not isinstance(payload, dict):
        raise KaggleModelSourceError(f"JSON root must be an object: {path}")
    return payload


def _write_object(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _fingerprint(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    return {
        "path": str(path),
        "exists": True,
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def attach_model_sources(
    metadata_path: str | Path,
    stage_manifest_path: str | Path,
    sources: Sequence[str],
) -> dict[str, Any]:
    """Attach exact model versions and refresh KJO's metadata fingerprint."""

    metadata_path = Path(metadata_path).resolve()
    stage_manifest_path = Path(stage_manifest_path).resolve()
    normalized = [validate_model_source(source) for source in sources]
    if not normalized:
        raise KaggleModelSourceError("at least one model source is required")
    if len(set(normalized)) != len(normalized):
        raise KaggleModelSourceError("model sources must be unique")

    metadata = _read_object(metadata_path)
    existing = metadata.get("model_sources")
    if existing not in (None, normalized):
        raise KaggleModelSourceError(
            f"existing model_sources drifted: {existing!r} != {normalized!r}"
        )
    metadata["model_sources"] = normalized
    _write_object(metadata_path, metadata)

    manifest = _read_object(stage_manifest_path)
    fingerprints = manifest.get("fingerprints")
    if not isinstance(fingerprints, dict) or "metadata" not in fingerprints:
        raise KaggleModelSourceError(
            "stage manifest does not contain a metadata fingerprint"
        )
    manifest["model_sources"] = normalized
    fingerprints["metadata"] = _fingerprint(metadata_path)
    _write_object(stage_manifest_path, manifest)
    return {
        "ok": True,
        "metadata": str(metadata_path),
        "stage_manifest": str(stage_manifest_path),
        "model_sources": normalized,
        "metadata_fingerprint": fingerprints["metadata"],
    }


def verify_attached_model_sources(
    metadata_path: str | Path,
    stage_manifest_path: str | Path,
    sources: Sequence[str],
) -> dict[str, Any]:
    """Fail closed unless a staged package statically attaches exact models."""

    metadata_path = Path(metadata_path).resolve()
    stage_manifest_path = Path(stage_manifest_path).resolve()
    normalized = [validate_model_source(source) for source in sources]
    if not normalized:
        raise KaggleModelSourceError("at least one model source is required")
    if len(set(normalized)) != len(normalized):
        raise KaggleModelSourceError("model sources must be unique")

    metadata = _read_object(metadata_path)
    manifest = _read_object(stage_manifest_path)
    if metadata.get("model_sources") != normalized:
        raise KaggleModelSourceError(
            "model sources are not statically attached in kernel metadata: "
            f"{metadata.get('model_sources')!r} != {normalized!r}"
        )
    if manifest.get("model_sources") != normalized:
        raise KaggleModelSourceError(
            "model sources are not recorded in the stage manifest: "
            f"{manifest.get('model_sources')!r} != {normalized!r}"
        )
    fingerprints = manifest.get("fingerprints")
    recorded = fingerprints.get("metadata") if isinstance(fingerprints, dict) else None
    current = _fingerprint(metadata_path)
    if not isinstance(recorded, dict) or recorded.get("sha256") != current["sha256"]:
        raise KaggleModelSourceError(
            "stage manifest metadata fingerprint is stale after model attachment"
        )
    return {
        "ok": True,
        "verified": True,
        "metadata": str(metadata_path),
        "stage_manifest": str(stage_manifest_path),
        "model_sources": normalized,
        "metadata_fingerprint": current,
    }


def render_model_source_mount_probe_notebook(
    *, model_sources: Sequence[str]
) -> dict[str, Any]:
    """Render a CPU-safe probe for exact Kaggle model-source accessibility.

    The probe resolves attached sources through Kaggle's supported
    ``kagglehub.model_download`` interface, records only bounded path metadata,
    and deliberately does not load model tensors.  Its output is operational
    evidence, never scientific evidence.
    """

    normalized = [validate_model_source(source) for source in model_sources]
    if not normalized:
        raise KaggleModelSourceError("at least one model source is required")
    if len(set(normalized)) != len(normalized):
        raise KaggleModelSourceError("model sources must be unique")

    probe = f'''from pathlib import Path
import json
import kagglehub

MODEL_SOURCES = {normalized!r}
records = []
for source in MODEL_SOURCES:
    mount = Path(kagglehub.model_download(source))
    if not mount.is_dir():
        raise FileNotFoundError(f"model source did not resolve to a directory: {{source}} -> {{mount}}")
    top_level = sorted(path.name for path in mount.iterdir())
    records.append({{
        "source": source,
        "mount": str(mount),
        "mount_exists": True,
        "top_level_count": len(top_level),
        "top_level_head": top_level[:100],
    }})

summary = {{
    "status": "passed",
    "model_sources": records,
    "scientific_evidence": False,
}}
output = Path("/kaggle/working/model_source_mount_probe.json")
output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
print("VDT_MODEL_SOURCE_MOUNT_PROBE " + json.dumps(summary, sort_keys=True))'''

    return {
        "cells": [
            {
                "cell_type": "markdown",
                "id": "model-source-probe-intro",
                "metadata": {},
                "source": [
                    "# Kaggle model-source mount probe\n",
                    "\n",
                    "Operational access check only; no model tensors are loaded.\n",
                ],
            },
            {
                "cell_type": "code",
                "id": "model-source-probe",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [line + "\n" for line in probe.splitlines()],
            },
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def render_canary_notebook(
    *,
    config_relative_path: str,
    repo_dataset_source: str,
    student_model_source: str,
    teacher_model_source: str,
) -> dict[str, Any]:
    """Render the source notebook that KJO will instrument and stage."""

    config_relative = PurePosixPath(config_relative_path)
    if config_relative.is_absolute() or ".." in config_relative.parts:
        raise KaggleModelSourceError("config_relative_path must stay inside repo")
    repo_parts = repo_dataset_source.split("/")
    if len(repo_parts) != 2 or not all(repo_parts):
        raise KaggleModelSourceError(
            "repo_dataset_source must be owner/slug"
        )
    repo_owner, repo_slug = repo_parts
    student_source = validate_model_source(student_model_source)
    teacher_source = validate_model_source(teacher_model_source)

    copy_repo = f'''from pathlib import Path
import json
import os
import shutil

KJO_REPO_DATASET_SOURCE = {repo_dataset_source!r}
KJO_REPO_DATASET_SLUG = {repo_slug!r}
KJO_REPO_DIR_NAME = "repo"
KJO_REPO_WORKING_DIR = Path("/kaggle/working/repo")
input_root = Path(os.environ.get("KJO_KAGGLE_INPUT_ROOT", "/kaggle/input"))
legacy_root = input_root / KJO_REPO_DATASET_SLUG
direct_root = input_root / "datasets" / {repo_owner!r} / KJO_REPO_DATASET_SLUG
version_root = direct_root / "versions"
if legacy_root.is_dir():
    dataset_root = legacy_root
elif direct_root.is_dir() and not version_root.is_dir():
    dataset_root = direct_root
else:
    versions = sorted(
        (path for path in version_root.glob("*") if path.is_dir()),
        key=lambda path: int(path.name) if path.name.isdigit() else -1,
    )
    if len(versions) != 1:
        available = sorted(str(path.relative_to(input_root)) for path in input_root.rglob("*") if path.is_dir())
        raise FileNotFoundError(
            "Kaggle dataset is not mounted at a supported layout. "
            f"dataset_source={{KJO_REPO_DATASET_SOURCE}} versions={{versions}} "
            f"available_inputs={{available[:100]}}"
        )
    dataset_root = versions[0]
source_repo = dataset_root / KJO_REPO_DIR_NAME
if not source_repo.is_dir():
    source_repo = dataset_root
required_repo_paths = (
    source_repo / "pyproject.toml",
    source_repo / "environments/kaggle-tpu/pyproject.toml",
    source_repo / "environments/kaggle-tpu/provider-constraints.json",
    source_repo / "environments/kaggle-tpu/uv.lock",
    source_repo / "vdt_tunix",
    source_repo / "vdt_tunix/kaggle_uv.py",
    source_repo / {config_relative.as_posix()!r},
)
if any(not path.exists() for path in required_repo_paths):
    raise FileNotFoundError(
        f"repo payload is incomplete at {{source_repo}}; "
        f"required={{[str(path) for path in required_repo_paths]}}"
    )
if KJO_REPO_WORKING_DIR.exists():
    shutil.rmtree(KJO_REPO_WORKING_DIR)
shutil.copytree(source_repo, KJO_REPO_WORKING_DIR, symlinks=False)
summary = {{
    "dataset_source": KJO_REPO_DATASET_SOURCE,
    "dataset_root": str(dataset_root),
    "source_repo": str(source_repo),
    "working_dir": str(KJO_REPO_WORKING_DIR),
    "file_count": sum(1 for path in KJO_REPO_WORKING_DIR.rglob("*") if path.is_file()),
}}
print("KJO_REPO_DATASET_COPY_SUMMARY " + json.dumps(summary, sort_keys=True))'''

    setup = f'''from pathlib import Path
import json
import os
import platform
import subprocess
import sys

REPO = Path("/kaggle/working/repo")
CONFIG_PATH = REPO / {config_relative.as_posix()!r}
STUDENT_SOURCE = {student_source!r}
TEACHER_SOURCE = {teacher_source!r}
sys.path.insert(0, str(REPO))
from vdt_tunix.kaggle_model_sources import (
    bind_runtime_model_mount,
    resolve_model_source_mount,
)
STUDENT_MOUNT = resolve_model_source_mount(STUDENT_SOURCE)
TEACHER_MOUNT = resolve_model_source_mount(TEACHER_SOURCE)
for required in (REPO, CONFIG_PATH, STUDENT_MOUNT, TEACHER_MOUNT):
    if not required.exists():
        available = sorted(path.name for path in Path("/kaggle/input").glob("*"))
        raise FileNotFoundError(f"required input missing: {{required}}; inputs={{available}}")
config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
student_runtime = bind_runtime_model_mount(config["student"], STUDENT_MOUNT)
teacher_runtime = bind_runtime_model_mount(config["teacher"], TEACHER_MOUNT)
RUNTIME_CONFIG = Path("/kaggle/working/vdt_simct_canary/runtime_config.json")
RUNTIME_CONFIG.parent.mkdir(parents=True, exist_ok=True)
RUNTIME_CONFIG.write_text(
    json.dumps(config, indent=2, sort_keys=True) + "\\n", encoding="utf-8"
)
print("VDT_MODEL_SOURCE_PROVENANCE " + json.dumps({{
    "student": STUDENT_SOURCE,
    "student_mount": str(STUDENT_MOUNT),
    "student_runtime": student_runtime,
    "teacher": TEACHER_SOURCE,
    "teacher_mount": str(TEACHER_MOUNT),
    "teacher_runtime": teacher_runtime,
    "python": platform.python_version(),
}}, sort_keys=True))'''

    dependencies = '''from vdt_tunix.kaggle_uv import (
    bootstrap_locked_kaggle_environment,
    runtime_subprocess_environment,
)

LOCKED_ENVIRONMENT = bootstrap_locked_kaggle_environment(
    REPO,
    Path("/tmp/vdt_simct_canary_environment"),
    summary_path=Path(
        "/kaggle/working/vdt_simct_canary/locked_environment_summary.json"
    ),
)
RUNTIME_PYTHON = Path(LOCKED_ENVIRONMENT["runtime_python"])
RUNTIME_SUBPROCESS_ENV = runtime_subprocess_environment(REPO, LOCKED_ENVIRONMENT)'''

    run = '''import shutil

work = Path("/kaggle/working/vdt_simct_canary")
output = work / "canary.json"
cache = work / "base-model-cache"
command = [
    str(RUNTIME_PYTHON),
    str(REPO / "scripts/tpu/kaggle_v5e8_canary.py"),
    "--config", str(RUNTIME_CONFIG),
    "--output", str(output),
]
try:
    result = subprocess.run(
        command,
        cwd=REPO,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=RUNTIME_SUBPROCESS_ENV,
    )
    print(result.stdout, end="")
    print(result.stderr, end="", file=sys.stderr)
finally:
    if cache.is_dir():
        shutil.rmtree(cache)
if result.returncode:
    raise RuntimeError(f"VDT canary failed with exit code {result.returncode}")
payload = json.loads(output.read_text(encoding="utf-8"))
expected = {
    "status": "passed",
    "real_model_integration": True,
    "cross_tokenization_observed": True,
    "simct_update_executed": True,
    "scientific_evidence": False,
}
drift = {key: (payload.get(key), value) for key, value in expected.items() if payload.get(key) != value}
if drift:
    raise RuntimeError(f"VDT canary evidence contract mismatch: {drift}")
print("VDT_CANARY_SUMMARY " + json.dumps(payload, sort_keys=True))'''

    def code_cell(source: str) -> dict[str, Any]:
        return {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in source.splitlines()],
        }

    return {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# SimCT paper-math Tunix TPU canary\n",
                    "One real optimizer update; this is not a scientific reproduction result.\n",
                ],
            },
            code_cell(copy_repo),
            code_cell(setup),
            code_cell(dependencies),
            code_cell(run),
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def _validate_dataset_source(source: str, context: str) -> tuple[str, str]:
    parts = source.split("/")
    if len(parts) != 2 or not all(parts):
        raise KaggleModelSourceError(f"{context} must be owner/slug")
    owner, slug = parts
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", owner):
        raise KaggleModelSourceError(f"{context} owner is invalid")
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", slug):
        raise KaggleModelSourceError(f"{context} slug is invalid")
    return owner, slug


def _safe_relative_path(
    value: str,
    context: str,
    *,
    allow_root: bool = False,
) -> PurePosixPath:
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or ".." in path.parts
        or (path == PurePosixPath(".") and not allow_root)
    ):
        raise KaggleModelSourceError(f"{context} must stay inside its input root")
    return path


def render_training_notebook(
    *,
    phase: str,
    config_relative_path: str,
    repo_dataset_source: str,
    training_dataset_source: str,
    training_manifest_relative_path: str,
    student_model_source: str,
    teacher_model_source: str,
    source_run_id: str | None = None,
    expected_run_id: str | None = None,
    training_seed: int | None = None,
    wandb_group: str = "public-substitute-one-seed",
    warm_start_dataset_source: str | None = None,
    warm_start_kernel_source: str | None = None,
    warm_start_kernel_version: int | None = None,
    warm_start_relative_path: str | None = None,
    checkpoint_initialization: str = "warm_start",
    expected_resume_trainer_calls: int | None = None,
    remote_teacher_profile_dataset_source: str | None = None,
    remote_teacher_profile_relative_path: str | None = None,
    remote_teacher_tokenizer_relative_path: str | None = None,
    remote_teacher_timeout_s: float = 300.0,
    remote_teacher_max_parallel_requests: int = 4,
    remote_teacher_max_attempts: int = 3,
    remote_teacher_retry_backoff_s: float = 2.0,
    profile_step: int = 0,
) -> dict[str, Any]:
    """Render a provenance-checked SFT or OPD notebook with durable output.

    The notebook only establishes training and checkpoint evidence.  Its
    summary is required to keep ``scientific_evidence=false`` until a separate
    evaluation contract is completed.
    """

    allowed_phases = {"sft", "simple_opd", "simct"}
    if phase not in allowed_phases:
        raise KaggleModelSourceError(
            f"phase must be one of {sorted(allowed_phases)}"
        )
    default_run_id = "vdt-public-" + phase + "-screen"
    if source_run_id is None:
        source_run_id = default_run_id
    if expected_run_id is None:
        expected_run_id = source_run_id
    if not isinstance(source_run_id, str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_-]*", source_run_id
    ):
        raise KaggleModelSourceError("source_run_id must be a safe run id")
    if not isinstance(expected_run_id, str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_-]*", expected_run_id
    ):
        raise KaggleModelSourceError("expected_run_id must be a safe run id")
    if not isinstance(wandb_group, str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_.-]*", wandb_group
    ):
        raise KaggleModelSourceError("wandb_group must be a safe W&B group")
    if (
        training_seed is not None
        and (
            isinstance(training_seed, bool)
            or not isinstance(training_seed, int)
            or training_seed < 0
        )
    ):
        raise KaggleModelSourceError("training_seed must be non-negative")
    if (
        isinstance(profile_step, bool)
        or not isinstance(profile_step, int)
        or profile_step < 0
    ):
        raise KaggleModelSourceError("profile_step must be a non-negative integer")
    if phase == "sft" and profile_step:
        raise KaggleModelSourceError("profile_step is only supported for OPD canaries")
    if checkpoint_initialization not in {"warm_start", "resume"}:
        raise KaggleModelSourceError(
            "checkpoint_initialization must be warm_start or resume"
        )
    if expected_resume_trainer_calls is not None and (
        isinstance(expected_resume_trainer_calls, bool)
        or not isinstance(expected_resume_trainer_calls, int)
        or expected_resume_trainer_calls < 1
    ):
        raise KaggleModelSourceError(
            "expected_resume_trainer_calls must be a positive integer"
        )
    config_relative = _safe_relative_path(
        config_relative_path, "config_relative_path"
    )
    manifest_relative = _safe_relative_path(
        training_manifest_relative_path,
        "training_manifest_relative_path",
    )
    repo_owner, repo_slug = _validate_dataset_source(
        repo_dataset_source, "repo_dataset_source"
    )
    data_owner, data_slug = _validate_dataset_source(
        training_dataset_source, "training_dataset_source"
    )
    student_source = validate_model_source(student_model_source)
    teacher_source = validate_model_source(teacher_model_source)

    remote_values = (
        remote_teacher_profile_dataset_source,
        remote_teacher_profile_relative_path,
        remote_teacher_tokenizer_relative_path,
    )
    remote_teacher_enabled = any(value is not None for value in remote_values)
    if remote_teacher_enabled and not all(value is not None for value in remote_values):
        raise KaggleModelSourceError(
            "remote teacher dataset source, profile path, and tokenizer path "
            "must be provided together"
        )
    if remote_teacher_enabled and phase == "sft":
        raise KaggleModelSourceError("remote teacher is only supported for OPD phases")
    if (
        isinstance(remote_teacher_timeout_s, bool)
        or not isinstance(remote_teacher_timeout_s, (int, float))
        or not math.isfinite(float(remote_teacher_timeout_s))
        or remote_teacher_timeout_s <= 0
    ):
        raise KaggleModelSourceError("remote_teacher_timeout_s must be positive")
    if (
        isinstance(remote_teacher_max_parallel_requests, bool)
        or not isinstance(remote_teacher_max_parallel_requests, int)
        or remote_teacher_max_parallel_requests < 1
    ):
        raise KaggleModelSourceError(
            "remote_teacher_max_parallel_requests must be a positive integer"
        )
    if (
        isinstance(remote_teacher_max_attempts, bool)
        or not isinstance(remote_teacher_max_attempts, int)
        or remote_teacher_max_attempts < 1
    ):
        raise KaggleModelSourceError(
            "remote_teacher_max_attempts must be a positive integer"
        )
    if (
        isinstance(remote_teacher_retry_backoff_s, bool)
        or not isinstance(remote_teacher_retry_backoff_s, (int, float))
        or not math.isfinite(float(remote_teacher_retry_backoff_s))
        or remote_teacher_retry_backoff_s < 0
    ):
        raise KaggleModelSourceError(
            "remote_teacher_retry_backoff_s must be finite and non-negative"
        )
    if remote_teacher_enabled:
        remote_owner, remote_slug = _validate_dataset_source(
            str(remote_teacher_profile_dataset_source),
            "remote_teacher_profile_dataset_source",
        )
        remote_profile_relative = _safe_relative_path(
            str(remote_teacher_profile_relative_path),
            "remote_teacher_profile_relative_path",
            allow_root=True,
        )
        remote_tokenizer_relative = _safe_relative_path(
            str(remote_teacher_tokenizer_relative_path),
            "remote_teacher_tokenizer_relative_path",
        )
    else:
        remote_owner = remote_slug = ""
        remote_profile_relative = remote_tokenizer_relative = None

    if phase == "sft":
        if (
            warm_start_dataset_source is not None
            or warm_start_kernel_source is not None
            or warm_start_kernel_version is not None
            or warm_start_relative_path is not None
            or checkpoint_initialization != "warm_start"
            or expected_resume_trainer_calls is not None
        ):
            raise KaggleModelSourceError(
                "SFT may not declare a checkpoint initialization source"
            )
        warm_owner = warm_slug = ""
        warm_relative = None
    else:
        warm_source_count = sum(
            source is not None
            for source in (
                warm_start_dataset_source,
                warm_start_kernel_source,
            )
        )
        if warm_source_count != 1 or warm_start_relative_path is None:
            raise KaggleModelSourceError(
                "OPD phases require exactly one completed SFT dataset or "
                "kernel source and a relative path"
            )
        if warm_start_dataset_source is not None:
            warm_owner, warm_slug = _validate_dataset_source(
                warm_start_dataset_source, "warm_start_dataset_source"
            )
            if warm_start_kernel_version is not None:
                raise KaggleModelSourceError(
                    "dataset warm-start may not declare a kernel version"
                )
        else:
            warm_owner, warm_slug = _validate_dataset_source(
                str(warm_start_kernel_source), "warm_start_kernel_source"
            )
            if warm_start_kernel_version is None or warm_start_kernel_version < 1:
                raise KaggleModelSourceError(
                    "warm_start_kernel_version must be a positive integer"
                )
        warm_relative = _safe_relative_path(
            warm_start_relative_path, "warm_start_relative_path"
        )
        if checkpoint_initialization == "resume":
            if warm_start_kernel_source is None:
                raise KaggleModelSourceError(
                    "resume requires a versioned kernel checkpoint source"
                )
            if expected_resume_trainer_calls is None:
                raise KaggleModelSourceError(
                    "resume requires expected_resume_trainer_calls"
                )
        elif expected_resume_trainer_calls is not None:
            raise KaggleModelSourceError(
                "expected_resume_trainer_calls is only valid for resume"
            )

    copy_repo = f'''from pathlib import Path
import json
import os
import shutil

KJO_REPO_DATASET_SOURCE = {repo_dataset_source!r}
KJO_REPO_DATASET_SLUG = {repo_slug!r}
KJO_REPO_DIR_NAME = "repo"
KJO_REPO_WORKING_DIR = Path("/kaggle/working/repo")
input_root = Path(os.environ.get("KJO_KAGGLE_INPUT_ROOT", "/kaggle/input"))
legacy_root = input_root / KJO_REPO_DATASET_SLUG
direct_root = input_root / "datasets" / {repo_owner!r} / KJO_REPO_DATASET_SLUG
version_root = direct_root / "versions"
if legacy_root.is_dir():
    dataset_root = legacy_root
elif direct_root.is_dir() and not version_root.is_dir():
    dataset_root = direct_root
else:
    versions = sorted(
        (path for path in version_root.glob("*") if path.is_dir()),
        key=lambda path: int(path.name) if path.name.isdigit() else -1,
    )
    if len(versions) != 1:
        available = sorted(str(path.relative_to(input_root)) for path in input_root.rglob("*") if path.is_dir())
        raise FileNotFoundError(
            "Kaggle repo dataset is not mounted at a supported layout. "
            f"dataset_source={{KJO_REPO_DATASET_SOURCE}} versions={{versions}} "
            f"available_inputs={{available[:100]}}"
        )
    dataset_root = versions[0]
source_repo = dataset_root / KJO_REPO_DIR_NAME
if not source_repo.is_dir():
    source_repo = dataset_root
required_repo_paths = (
    source_repo / "pyproject.toml",
    source_repo / "environments/kaggle-tpu/pyproject.toml",
    source_repo / "environments/kaggle-tpu/provider-constraints.json",
    source_repo / "environments/kaggle-tpu/uv.lock",
    source_repo / "vdt_tunix",
    source_repo / "vdt_tunix/kaggle_uv.py",
    source_repo / {config_relative.as_posix()!r},
)
if any(not path.exists() for path in required_repo_paths):
    raise FileNotFoundError(
        f"repo payload is incomplete at {{source_repo}}; "
        f"required={{[str(path) for path in required_repo_paths]}}"
    )
if KJO_REPO_WORKING_DIR.exists():
    shutil.rmtree(KJO_REPO_WORKING_DIR)
shutil.copytree(source_repo, KJO_REPO_WORKING_DIR, symlinks=False)
print("KJO_REPO_DATASET_COPY_SUMMARY " + json.dumps({{
    "dataset_source": KJO_REPO_DATASET_SOURCE,
    "dataset_root": str(dataset_root),
    "source_repo": str(source_repo),
    "working_dir": str(KJO_REPO_WORKING_DIR),
    "file_count": sum(1 for path in KJO_REPO_WORKING_DIR.rglob("*") if path.is_file()),
}}, sort_keys=True))'''

    resolve_inputs = f'''from pathlib import Path, PurePosixPath
import json
import os
import shutil
import zipfile

REPO = Path("/kaggle/working/repo")
CONFIG_PATH = REPO / {config_relative.as_posix()!r}
TRAINING_DATASET_SOURCE = {training_dataset_source!r}
TRAINING_DATASET_OWNER = {data_owner!r}
TRAINING_DATASET_SLUG = {data_slug!r}
TRAINING_MANIFEST_RELATIVE = PurePosixPath({manifest_relative.as_posix()!r})
WARM_START_KERNEL_SOURCE = {warm_start_kernel_source!r}
WARM_START_DATASET_SOURCE = {warm_start_dataset_source!r}
WARM_START_OWNER = {warm_owner!r}
WARM_START_SLUG = {warm_slug!r}
WARM_START_KERNEL_VERSION = {warm_start_kernel_version!r}
WARM_START_RELATIVE = {None if warm_relative is None else warm_relative.as_posix()!r}
CHECKPOINT_INITIALIZATION = {checkpoint_initialization!r}
EXPECTED_RESUME_TRAINER_CALLS = {expected_resume_trainer_calls!r}
REMOTE_TEACHER_PROFILE_DATASET_SOURCE = {remote_teacher_profile_dataset_source!r}
REMOTE_TEACHER_PROFILE_OWNER = {remote_owner!r}
REMOTE_TEACHER_PROFILE_SLUG = {remote_slug!r}
REMOTE_TEACHER_PROFILE_RELATIVE = {None if remote_profile_relative is None else remote_profile_relative.as_posix()!r}
REMOTE_TEACHER_TOKENIZER_RELATIVE = {None if remote_tokenizer_relative is None else remote_tokenizer_relative.as_posix()!r}
RUNTIME_INPUTS = Path("/kaggle/working/vdt_runtime_inputs")

def resolve_input(source, owner, slug, *, notebook_version=None):
    root = Path(os.environ.get("KJO_KAGGLE_INPUT_ROOT", "/kaggle/input"))
    legacy = root / slug
    owner_direct = root / owner / slug
    notebook_direct = root / "notebooks" / owner / slug
    kernel_direct = root / "kernels" / owner / slug
    direct = root / "datasets" / owner / slug

    def mounted_candidate(candidate, required_version=None):
        if not candidate.is_dir():
            return None
        version_root = candidate / "versions"
        if required_version is not None:
            exact = version_root / str(required_version)
            if exact.is_dir():
                return exact
        if not version_root.is_dir():
            return candidate
        versions = sorted(
            (path for path in version_root.glob("*") if path.is_dir()),
            key=lambda path: int(path.name) if path.name.isdigit() else -1,
        )
        if required_version is None and len(versions) == 1:
            return versions[0]
        return None

    static_candidates = [
        (legacy, notebook_version),
        (owner_direct, notebook_version),
    ]
    if notebook_version is None:
        static_candidates.append((direct, None))
    else:
        static_candidates.append((notebook_direct, notebook_version))
        static_candidates.append((kernel_direct, notebook_version))
    for candidate, required_version in static_candidates:
        mounted = mounted_candidate(candidate, required_version)
        if mounted is not None:
            return mounted
    if notebook_version is not None:
        handle = f"{{source}}/versions/{{notebook_version}}"
        try:
            import kagglehub

            attached = Path(kagglehub.notebook_output_download(handle))
        except Exception as exc:
            raise FileNotFoundError(
                f"notebook output {{handle}} was neither statically mounted "
                "nor dynamically attached by kagglehub"
            ) from exc
        if not attached.is_dir():
            raise FileNotFoundError(
                f"kagglehub returned a non-directory for notebook output "
                f"{{handle}}: {{attached}}"
            )
        print("VDT_NOTEBOOK_OUTPUT_ATTACH_SUMMARY " + json.dumps({{
            "handle": handle,
            "method": "kagglehub_runtime_attach",
            "root": str(attached),
            "version": notebook_version,
        }}, sort_keys=True))
        return attached
    available = sorted(
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_dir()
    )
    raise FileNotFoundError(
        f"input {{source}} is not mounted at a supported layout; "
        f"required_notebook_version={{notebook_version}} "
        f"available_inputs={{available[:100]}}"
    )

def resolve_relative(root, relative, extract_name):
    relative = PurePosixPath(relative)
    direct = root.joinpath(*relative.parts)
    if direct.exists():
        return direct
    archive = root / f"{{relative.parts[0]}}.zip"
    if not archive.is_file():
        raise FileNotFoundError(
            f"input path {{relative}} and archive {{archive.name}} are absent in {{root}}"
        )
    destination = RUNTIME_INPUTS / extract_name
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    with zipfile.ZipFile(archive) as handle:
        destination_resolved = destination.resolve()
        for member in handle.infolist():
            target = (destination / member.filename).resolve()
            if destination_resolved not in target.parents and target != destination_resolved:
                raise RuntimeError(f"unsafe archive member: {{member.filename}}")
        handle.extractall(destination)
    candidates = (
        destination.joinpath(*relative.parts),
        destination.joinpath(*relative.parts[1:]),
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"archive {{archive}} did not contain requested path {{relative}}"
    )

training_root = resolve_input(
    TRAINING_DATASET_SOURCE, TRAINING_DATASET_OWNER, TRAINING_DATASET_SLUG
)
TRAINING_MANIFEST = resolve_relative(
    training_root, TRAINING_MANIFEST_RELATIVE, "training-data"
)
if WARM_START_KERNEL_SOURCE is None and WARM_START_DATASET_SOURCE is None:
    WARM_START_ROOT = None
else:
    warm_mount = resolve_input(
        WARM_START_KERNEL_SOURCE or WARM_START_DATASET_SOURCE,
        WARM_START_OWNER,
        WARM_START_SLUG,
        notebook_version=(
            WARM_START_KERNEL_VERSION
            if WARM_START_KERNEL_SOURCE is not None
            else None
        ),
    )
    WARM_START_ROOT = resolve_relative(
        warm_mount, PurePosixPath(WARM_START_RELATIVE), "warm-start"
    )
if REMOTE_TEACHER_PROFILE_DATASET_SOURCE is None:
    REMOTE_TEACHER_PROFILE_ROOT = None
    REMOTE_TEACHER_TOKENIZER_ROOT = None
else:
    remote_teacher_mount = resolve_input(
        REMOTE_TEACHER_PROFILE_DATASET_SOURCE,
        REMOTE_TEACHER_PROFILE_OWNER,
        REMOTE_TEACHER_PROFILE_SLUG,
    )
    REMOTE_TEACHER_PROFILE_ROOT = resolve_relative(
        remote_teacher_mount,
        PurePosixPath(REMOTE_TEACHER_PROFILE_RELATIVE),
        "remote-teacher-profile",
    )
    REMOTE_TEACHER_TOKENIZER_ROOT = resolve_relative(
        remote_teacher_mount,
        PurePosixPath(REMOTE_TEACHER_TOKENIZER_RELATIVE),
        "remote-teacher-tokenizer",
    )
    required_remote_teacher_files = (
        REMOTE_TEACHER_PROFILE_ROOT / "teacher_overlap_lm_head.manifest.json",
        REMOTE_TEACHER_PROFILE_ROOT / "teacher_ids.i32le",
        REMOTE_TEACHER_PROFILE_ROOT / "teacher_overlap_lm_head.bf16le",
        REMOTE_TEACHER_TOKENIZER_ROOT / "tokenizer.json",
        REMOTE_TEACHER_TOKENIZER_ROOT / "tokenizer_config.json",
    )
    missing_remote_teacher_files = [
        str(path) for path in required_remote_teacher_files if not path.is_file()
    ]
    if missing_remote_teacher_files:
        raise FileNotFoundError(
            "remote teacher profile payload is incomplete: "
            f"{{missing_remote_teacher_files}}"
        )
print("VDT_TRAINING_INPUT_PROVENANCE " + json.dumps({{
    "phase": {phase!r},
    "training_dataset_source": TRAINING_DATASET_SOURCE,
    "training_manifest": str(TRAINING_MANIFEST),
    "warm_start_dataset_source": WARM_START_DATASET_SOURCE,
    "warm_start_kernel_source": WARM_START_KERNEL_SOURCE,
    "warm_start_kernel_version": WARM_START_KERNEL_VERSION,
    "warm_start_root": None if WARM_START_ROOT is None else str(WARM_START_ROOT),
    "checkpoint_initialization": CHECKPOINT_INITIALIZATION,
    "expected_resume_trainer_calls": EXPECTED_RESUME_TRAINER_CALLS,
    "remote_teacher_profile_dataset_source": REMOTE_TEACHER_PROFILE_DATASET_SOURCE,
    "remote_teacher_profile_root": None if REMOTE_TEACHER_PROFILE_ROOT is None else str(REMOTE_TEACHER_PROFILE_ROOT),
    "remote_teacher_tokenizer_root": None if REMOTE_TEACHER_TOKENIZER_ROOT is None else str(REMOTE_TEACHER_TOKENIZER_ROOT),
}}, sort_keys=True))'''

    wandb_secret = '''import os

os.environ["WANDB_API_KEY"] = "__KJO_SECRET_WANDB_API_KEY__"'''

    remote_teacher_secret = None
    if remote_teacher_enabled:
        remote_teacher_secret = f'''import json
import os
from pathlib import Path

REMOTE_TEACHER_URL = "__KJO_SECRET_VDT_REMOTE_TEACHER_URL__"
REMOTE_TEACHER_TOKEN = "__KJO_SECRET_VDT_REMOTE_TEACHER_TOKEN__"
REMOTE_TEACHER_SECRET_DIR = Path("/kaggle/working/.vdt-remote-teacher")
REMOTE_TEACHER_SECRET_DIR.mkdir(mode=0o700, parents=True, exist_ok=True)
REMOTE_TEACHER_SECRET_DIR.chmod(0o700)
REMOTE_TEACHER_TOKEN_FILE = REMOTE_TEACHER_SECRET_DIR / "bearer-token"
REMOTE_TEACHER_TOKEN_FILE.write_text(REMOTE_TEACHER_TOKEN, encoding="utf-8")
REMOTE_TEACHER_TOKEN_FILE.chmod(0o600)
os.environ["VDT_REMOTE_TEACHER_URL"] = REMOTE_TEACHER_URL
os.environ["VDT_REMOTE_TEACHER_TOKEN_FILE"] = str(REMOTE_TEACHER_TOKEN_FILE)
os.environ["VDT_REMOTE_TEACHER_PROFILE_DIR"] = str(REMOTE_TEACHER_PROFILE_ROOT)
os.environ["VDT_REMOTE_TEACHER_TOKENIZER_DIR"] = str(REMOTE_TEACHER_TOKENIZER_ROOT)
os.environ["VDT_REMOTE_TEACHER_TIMEOUT_S"] = {str(float(remote_teacher_timeout_s))!r}
os.environ["VDT_REMOTE_TEACHER_MAX_PARALLEL"] = {str(remote_teacher_max_parallel_requests)!r}
os.environ["VDT_REMOTE_TEACHER_MAX_ATTEMPTS"] = {str(remote_teacher_max_attempts)!r}
os.environ["VDT_REMOTE_TEACHER_RETRY_BACKOFF_S"] = {str(float(remote_teacher_retry_backoff_s))!r}
print("VDT_REMOTE_TEACHER_CONFIG_SUMMARY " + json.dumps({{
    "enabled": True,
    "max_attempts": {remote_teacher_max_attempts!r},
    "max_parallel_requests": {remote_teacher_max_parallel_requests!r},
    "profile_dataset_source": REMOTE_TEACHER_PROFILE_DATASET_SOURCE,
    "profile_root": str(REMOTE_TEACHER_PROFILE_ROOT),
    "timeout_s": {float(remote_teacher_timeout_s)!r},
    "retry_backoff_s": {float(remote_teacher_retry_backoff_s)!r},
    "token_file_mode": oct(REMOTE_TEACHER_TOKEN_FILE.stat().st_mode & 0o777),
    "tokenizer_root": str(REMOTE_TEACHER_TOKENIZER_ROOT),
}}, sort_keys=True))
del REMOTE_TEACHER_TOKEN'''

    if remote_teacher_enabled:
        teacher_mount_setup = '''TEACHER_MOUNT = None
for required in (
    REPO,
    CONFIG_PATH,
    TRAINING_MANIFEST,
    STUDENT_MOUNT,
    REMOTE_TEACHER_PROFILE_ROOT,
    REMOTE_TEACHER_TOKENIZER_ROOT,
):
    if not required.exists():
        raise FileNotFoundError(f"required input missing: {required}")'''
        teacher_runtime_setup = '''teacher_runtime = {
    "mode": "remote_vllm_exact",
    "profile_dataset_source": REMOTE_TEACHER_PROFILE_DATASET_SOURCE,
    "profile_dir": str(REMOTE_TEACHER_PROFILE_ROOT),
    "tokenizer_dir": str(REMOTE_TEACHER_TOKENIZER_ROOT),
}'''
    else:
        teacher_mount_setup = '''TEACHER_MOUNT = resolve_model_source_mount(TEACHER_SOURCE)
for required in (REPO, CONFIG_PATH, TRAINING_MANIFEST, STUDENT_MOUNT, TEACHER_MOUNT):
    if not required.exists():
        raise FileNotFoundError(f"required input missing: {required}")'''
        teacher_runtime_setup = '''teacher_runtime = bind_runtime_model_mount(
    config["teacher"], TEACHER_MOUNT
)'''

    setup = f'''import hashlib
import json
import os
import platform
import subprocess
import sys

os.environ.setdefault("WANDB_PROJECT", "vdt-simct-tunix-reproduction")
os.environ.setdefault("WANDB_RUN_GROUP", {wandb_group!r})
os.environ.setdefault("WANDB_MODE", "online")
os.environ.setdefault("WANDB_INIT_TIMEOUT", "30")
os.environ.setdefault("VDT_REQUIRE_WANDB", "1")

STUDENT_SOURCE = {student_source!r}
TEACHER_SOURCE = {teacher_source!r}
sys.path.insert(0, str(REPO))
from vdt_tunix.kaggle_model_sources import (
    bind_runtime_model_mount,
    resolve_model_source_mount,
)
STUDENT_MOUNT = resolve_model_source_mount(STUDENT_SOURCE)
{teacher_mount_setup}
config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
source_run_id = {source_run_id!r}
if config["run_id"] != source_run_id:
    raise RuntimeError(
        f"unexpected source run_id in {{CONFIG_PATH}}: {{config['run_id']}}"
    )
config["run_id"] = {expected_run_id!r}
if {training_seed!r} is not None:
    config["training"]["seed"] = {training_seed!r}
config["checkpoint"]["root"] = str(
    Path("/kaggle/working")
    / config["run_id"].replace("-", "_")
    / "checkpoints"
)
student_runtime = bind_runtime_model_mount(config["student"], STUDENT_MOUNT)
{teacher_runtime_setup}
if config["run_id"] != {expected_run_id!r}:
    raise RuntimeError(f"runtime run_id drifted: {{config['run_id']}}")
expected_algorithm = {("simct" if phase == "sft" else phase)!r}
if config["simct"]["algorithm"] != expected_algorithm:
    raise RuntimeError("training phase and algorithm drifted")
WORK = Path(config["checkpoint"]["root"]).parent
WORK.mkdir(parents=True, exist_ok=True)
config["checkpoint"]["root"] = str(WORK / "checkpoints")
if CHECKPOINT_INITIALIZATION == "resume":
    accumulation = config["training"]["gradient_accumulation_steps"]
    if EXPECTED_RESUME_TRAINER_CALLS % accumulation:
        raise RuntimeError(
            "resume checkpoint is not on an optimizer-update boundary"
        )
    EXPECTED_START_STEP = EXPECTED_RESUME_TRAINER_CALLS // accumulation
    config["checkpoint"]["resume_from"] = str(WARM_START_ROOT)
    config["checkpoint"]["warm_start_from"] = None
else:
    EXPECTED_START_STEP = 0
    config["checkpoint"]["resume_from"] = None
    config["checkpoint"]["warm_start_from"] = (
        None if WARM_START_ROOT is None else str(WARM_START_ROOT)
    )
RUNTIME_CONFIG = WORK / "runtime_config.json"
RUNTIME_CONFIG.write_text(
    json.dumps(config, indent=2, sort_keys=True) + "\\n", encoding="utf-8"
)
print("VDT_MODEL_SOURCE_PROVENANCE " + json.dumps({{
    "student": STUDENT_SOURCE,
    "student_mount": str(STUDENT_MOUNT),
    "student_runtime": student_runtime,
    "teacher": TEACHER_SOURCE,
    "teacher_mount": str(TEACHER_MOUNT),
    "teacher_runtime": teacher_runtime,
    "python": platform.python_version(),
    "runtime_config_sha256": hashlib.sha256(RUNTIME_CONFIG.read_bytes()).hexdigest(),
}}, sort_keys=True))'''

    dependencies = f'''from vdt_tunix.kaggle_uv import (
    bootstrap_locked_kaggle_environment,
    runtime_subprocess_environment,
)

LOCKED_ENVIRONMENT = bootstrap_locked_kaggle_environment(
    REPO,
    Path("/tmp/vdt_{phase}_training_environment"),
    summary_path=WORK / "locked_environment_summary.json",
)
RUNTIME_PYTHON = Path(LOCKED_ENVIRONMENT["runtime_python"])
RUNTIME_SUBPROCESS_ENV = runtime_subprocess_environment(REPO, LOCKED_ENVIRONMENT)'''

    entrypoint = (
        "scripts/tpu/kaggle_v5e8_sft.py"
        if phase == "sft"
        else "scripts/tpu/kaggle_v5e8_train.py"
    )
    expected_phase = "sft_training" if phase == "sft" else f"{phase}_training"
    objective_contract = (
        "" if phase == "sft" else f'    "objective": {phase!r},\n'
    )
    if phase == "sft":
        initialization_contract = ""
    elif checkpoint_initialization == "warm_start":
        initialization_contract = '''if payload.get("initialization") != "warm_start":
    drift["initialization"] = (payload.get("initialization"), "warm_start")
'''
    else:
        initialization_contract = '''if payload.get("initialization") != "resume":
    drift["initialization"] = (payload.get("initialization"), "resume")
'''
    resume_contract = (
        ""
        if checkpoint_initialization != "resume"
        else '''if payload.get("source_checkpoint_steps") != EXPECTED_RESUME_TRAINER_CALLS:
    drift["source_checkpoint_steps"] = (
        payload.get("source_checkpoint_steps"), EXPECTED_RESUME_TRAINER_CALLS
    )
if payload.get("source_checkpoint_run_id") != config["run_id"]:
    drift["source_checkpoint_run_id"] = (
        payload.get("source_checkpoint_run_id"), config["run_id"]
    )
'''
    )
    start_step_contract = (
        "EXPECTED_START_STEP" if checkpoint_initialization == "resume" else "0"
    )
    start_trainer_call_contract = (
        "EXPECTED_RESUME_TRAINER_CALLS"
        if checkpoint_initialization == "resume"
        else "0"
    )

    run = f'''import hashlib
import shutil

SUMMARY = WORK / "train_summary.json"
METRICS = WORK / "train_metrics.jsonl"
ARTIFACT_MANIFEST = WORK / "artifact_manifest.json"
command = [
    str(RUNTIME_PYTHON),
    str(REPO / {entrypoint!r}),
    "--config", str(RUNTIME_CONFIG),
    "--dataset-manifest", str(TRAINING_MANIFEST),
    "--output", str(SUMMARY),
    "--metrics", str(METRICS),
]
if {profile_step!r}:
    command.extend([
        "--profile-dir", str(WORK / "jax-profile"),
        "--profile-step", str({profile_step!r}),
    ])
result = None
try:
    result = subprocess.run(
        command,
        cwd=REPO,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=RUNTIME_SUBPROCESS_ENV,
    )
    print(result.stdout, end="")
    print(result.stderr, end="", file=sys.stderr)
finally:
    cache = WORK / "base-model-cache"
    if cache.is_dir():
        shutil.rmtree(cache)
if result is None or result.returncode:
    raise RuntimeError(
        f"VDT {phase} training failed with exit code "
        f"{{None if result is None else result.returncode}}"
    )
payload = json.loads(SUMMARY.read_text(encoding="utf-8"))
expected = {{
    "phase": {expected_phase!r},
    "status": "complete",
    "run_id": config["run_id"],
    "start_step": {start_step_contract},
    "start_trainer_call": {start_trainer_call_contract},
    "completed_steps": config["training"]["max_steps"],
    "scientific_evidence": False,
{objective_contract}}}
drift = {{
    key: (payload.get(key), value)
    for key, value in expected.items()
    if payload.get(key) != value
}}
parameter_sha = payload.get("final_student_parameters_sha256")
if not isinstance(parameter_sha, str) or len(parameter_sha) != 64:
    drift["final_student_parameters_sha256"] = (parameter_sha, "sha256")
{initialization_contract}{resume_contract}if drift:
    raise RuntimeError(f"VDT training evidence contract mismatch: {{drift}}")
artifact = {{
    "contract_version": 1,
    "phase": {phase!r},
    "run_id": config["run_id"],
    "source_repo_dataset": KJO_REPO_DATASET_SOURCE,
    "training_dataset_source": TRAINING_DATASET_SOURCE,
    "warm_start_dataset_source": WARM_START_DATASET_SOURCE,
    "warm_start_kernel_source": WARM_START_KERNEL_SOURCE,
    "checkpoint_initialization": CHECKPOINT_INITIALIZATION,
    "source_checkpoint_steps": payload.get("source_checkpoint_steps"),
    "summary_sha256": hashlib.sha256(SUMMARY.read_bytes()).hexdigest(),
    "metrics_sha256": hashlib.sha256(METRICS.read_bytes()).hexdigest(),
    "final_student_parameters_sha256": parameter_sha,
    "scientific_evidence": False,
    "remaining_gate": "shared downstream one-seed evaluation contract",
}}
ARTIFACT_MANIFEST.write_text(
    json.dumps(artifact, indent=2, sort_keys=True) + "\\n", encoding="utf-8"
)
if RUNTIME_INPUTS.is_dir():
    shutil.rmtree(RUNTIME_INPUTS)
if REPO.is_dir():
    shutil.rmtree(REPO)
print("VDT_TRAINING_SUMMARY " + json.dumps(payload, sort_keys=True))
print("VDT_TRAINING_ARTIFACT " + json.dumps(artifact, sort_keys=True))'''

    def code_cell(source: str) -> dict[str, Any]:
        return {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in source.splitlines()],
        }

    cells = [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    f"# VDT Tunix {phase} public-data screen\n",
                    "Training/checkpoint evidence only; not a scientific reproduction result.\n",
                ],
            },
            code_cell(copy_repo),
            code_cell(resolve_inputs),
            code_cell(wandb_secret),
    ]
    if remote_teacher_secret is not None:
        cells.append(code_cell(remote_teacher_secret))
    cells.extend(
        [
            code_cell(setup),
            code_cell(dependencies),
            code_cell(run),
        ]
    )

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
