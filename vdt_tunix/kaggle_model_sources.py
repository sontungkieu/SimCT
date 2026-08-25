"""Deterministic Kaggle model-source metadata and canary notebook helpers."""

from __future__ import annotations

import hashlib
import json
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
TUNIX_COMMIT = "50f5752a17edec56e2aa30aabfc03859949adf6f"


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
    source_repo / "vdt_tunix",
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
import importlib.metadata
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

    dependencies = f'''requirements = REPO / "requirements-tpu.txt"
before = {{name: importlib.metadata.version(name) for name in ("jax", "jaxlib")}}
subprocess.run([
    sys.executable, "-m", "pip", "install", "--no-input", "--no-deps",
    "-r", str(requirements),
], check=True, cwd=REPO)
after = {{name: importlib.metadata.version(name) for name in ("jax", "jaxlib")}}
if before != after:
    raise RuntimeError(f"provider-managed JAX stack changed: {{before}} -> {{after}}")
direct = json.loads(
    importlib.metadata.distribution("google-tunix").read_text("direct_url.json") or "{{}}"
)
observed_tunix_commit = direct.get("vcs_info", {{}}).get("commit_id")
if observed_tunix_commit != {TUNIX_COMMIT!r}:
    raise RuntimeError(f"installed Tunix commit drifted: {{observed_tunix_commit}}")
import flax
import orbax.checkpoint
import optax
import sentencepiece
import transformers
import tunix.models.automodel
import tunix.generate.sampler
print("VDT_DEPENDENCY_PROVENANCE " + json.dumps({{
    "jax_before": before,
    "jax_after": after,
    "flax": flax.__version__,
    "huggingface_hub": importlib.metadata.version("huggingface-hub"),
    "transformers": transformers.__version__,
    "tunix_commit": observed_tunix_commit,
}}, sort_keys=True))'''

    run = '''import shutil

work = Path("/kaggle/working/vdt_simct_canary")
output = work / "canary.json"
cache = work / "base-model-cache"
command = [
    sys.executable,
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
        env={**os.environ, "PYTHONPATH": str(REPO)},
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


def _safe_relative_path(value: str, context: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or path == PurePosixPath("."):
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
    warm_start_kernel_source: str | None = None,
    warm_start_kernel_version: int | None = None,
    warm_start_relative_path: str | None = None,
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

    if phase == "sft":
        if (
            warm_start_kernel_source is not None
            or warm_start_kernel_version is not None
            or warm_start_relative_path is not None
        ):
            raise KaggleModelSourceError("SFT may not declare a warm-start source")
        warm_owner = warm_slug = ""
        warm_relative = None
    else:
        if (
            warm_start_kernel_source is None
            or warm_start_kernel_version is None
            or warm_start_relative_path is None
        ):
            raise KaggleModelSourceError(
                "OPD phases require a completed SFT kernel source, positive "
                "version, and relative path"
            )
        warm_owner, warm_slug = _validate_dataset_source(
            warm_start_kernel_source, "warm_start_kernel_source"
        )
        if warm_start_kernel_version < 1:
            raise KaggleModelSourceError(
                "warm_start_kernel_version must be a positive integer"
            )
        warm_relative = _safe_relative_path(
            warm_start_relative_path, "warm_start_relative_path"
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
    source_repo / "vdt_tunix",
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
WARM_START_OWNER = {warm_owner!r}
WARM_START_SLUG = {warm_slug!r}
WARM_START_KERNEL_VERSION = {warm_start_kernel_version!r}
WARM_START_RELATIVE = {None if warm_relative is None else warm_relative.as_posix()!r}
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
if WARM_START_KERNEL_SOURCE is None:
    WARM_START_ROOT = None
else:
    warm_mount = resolve_input(
        WARM_START_KERNEL_SOURCE,
        WARM_START_OWNER,
        WARM_START_SLUG,
        notebook_version=WARM_START_KERNEL_VERSION,
    )
    WARM_START_ROOT = resolve_relative(
        warm_mount, PurePosixPath(WARM_START_RELATIVE), "warm-start"
    )
print("VDT_TRAINING_INPUT_PROVENANCE " + json.dumps({{
    "phase": {phase!r},
    "training_dataset_source": TRAINING_DATASET_SOURCE,
    "training_manifest": str(TRAINING_MANIFEST),
    "warm_start_kernel_source": WARM_START_KERNEL_SOURCE,
    "warm_start_kernel_version": WARM_START_KERNEL_VERSION,
    "warm_start_root": None if WARM_START_ROOT is None else str(WARM_START_ROOT),
}}, sort_keys=True))'''

    setup = f'''import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys

os.environ["WANDB_API_KEY"] = "__KJO_SECRET_WANDB_API_KEY__"
os.environ.setdefault("WANDB_PROJECT", "vdt-simct-tunix-reproduction")
os.environ.setdefault("WANDB_RUN_GROUP", "public-substitute-one-seed")
os.environ.setdefault("WANDB_MODE", "online")
os.environ.setdefault("WANDB_INIT_TIMEOUT", "30")

STUDENT_SOURCE = {student_source!r}
TEACHER_SOURCE = {teacher_source!r}
sys.path.insert(0, str(REPO))
from vdt_tunix.kaggle_model_sources import (
    bind_runtime_model_mount,
    resolve_model_source_mount,
)
STUDENT_MOUNT = resolve_model_source_mount(STUDENT_SOURCE)
TEACHER_MOUNT = resolve_model_source_mount(TEACHER_SOURCE)
for required in (REPO, CONFIG_PATH, TRAINING_MANIFEST, STUDENT_MOUNT, TEACHER_MOUNT):
    if not required.exists():
        raise FileNotFoundError(f"required input missing: {{required}}")
config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
student_runtime = bind_runtime_model_mount(config["student"], STUDENT_MOUNT)
teacher_runtime = bind_runtime_model_mount(config["teacher"], TEACHER_MOUNT)
if config["run_id"] != {('vdt-public-' + phase + '-screen')!r}:
    raise RuntimeError(f"unexpected run_id in {{CONFIG_PATH}}: {{config['run_id']}}")
expected_algorithm = {("simct" if phase == "sft" else phase)!r}
if config["simct"]["algorithm"] != expected_algorithm:
    raise RuntimeError("training phase and algorithm drifted")
WORK = Path(config["checkpoint"]["root"]).parent
WORK.mkdir(parents=True, exist_ok=True)
config["checkpoint"]["root"] = str(WORK / "checkpoints")
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

    dependencies = f'''requirements = REPO / "requirements-tpu.txt"
before = {{name: importlib.metadata.version(name) for name in ("jax", "jaxlib")}}
subprocess.run([
    sys.executable, "-m", "pip", "install", "--no-input", "--no-deps",
    "-r", str(requirements),
], check=True, cwd=REPO)
after = {{name: importlib.metadata.version(name) for name in ("jax", "jaxlib")}}
if before != after:
    raise RuntimeError(f"provider-managed JAX stack changed: {{before}} -> {{after}}")
direct = json.loads(
    importlib.metadata.distribution("google-tunix").read_text("direct_url.json") or "{{}}"
)
observed_tunix_commit = direct.get("vcs_info", {{}}).get("commit_id")
if observed_tunix_commit != {TUNIX_COMMIT!r}:
    raise RuntimeError(f"installed Tunix commit drifted: {{observed_tunix_commit}}")
import flax
import orbax.checkpoint
import optax
import sentencepiece
import transformers
import tunix.models.automodel
import tunix.generate.sampler
print("VDT_DEPENDENCY_PROVENANCE " + json.dumps({{
    "jax_before": before,
    "jax_after": after,
    "flax": flax.__version__,
    "huggingface_hub": importlib.metadata.version("huggingface-hub"),
    "transformers": transformers.__version__,
    "tunix_commit": observed_tunix_commit,
}}, sort_keys=True))'''

    entrypoint = (
        "scripts/tpu/kaggle_v5e8_sft.py"
        if phase == "sft"
        else "scripts/tpu/kaggle_v5e8_train.py"
    )
    expected_phase = "sft_training" if phase == "sft" else f"{phase}_training"
    objective_contract = (
        "" if phase == "sft" else f'    "objective": {phase!r},\n'
    )
    warm_start_contract = (
        ""
        if phase == "sft"
        else '''if payload.get("initialization") != "warm_start":
    drift["initialization"] = (payload.get("initialization"), "warm_start")
'''
    )

    run = f'''import hashlib
import shutil

SUMMARY = WORK / "train_summary.json"
METRICS = WORK / "train_metrics.jsonl"
ARTIFACT_MANIFEST = WORK / "artifact_manifest.json"
command = [
    sys.executable,
    str(REPO / {entrypoint!r}),
    "--config", str(RUNTIME_CONFIG),
    "--dataset-manifest", str(TRAINING_MANIFEST),
    "--output", str(SUMMARY),
    "--metrics", str(METRICS),
]
result = None
try:
    result = subprocess.run(
        command,
        cwd=REPO,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env={{**os.environ, "PYTHONPATH": str(REPO)}},
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
    "start_step": 0,
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
{warm_start_contract}if drift:
    raise RuntimeError(f"VDT training evidence contract mismatch: {{drift}}")
artifact = {{
    "contract_version": 1,
    "phase": {phase!r},
    "run_id": config["run_id"],
    "source_repo_dataset": KJO_REPO_DATASET_SOURCE,
    "training_dataset_source": TRAINING_DATASET_SOURCE,
    "warm_start_kernel_source": WARM_START_KERNEL_SOURCE,
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

    return {
        "cells": [
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
