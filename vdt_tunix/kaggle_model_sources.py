"""Deterministic Kaggle model-source metadata and canary notebook helpers."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path, PurePosixPath
from typing import Any, Sequence


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
    """Return the legacy Kaggle mount path for a validated model source."""

    _, model, framework, variation, version = validate_model_source(source).split(
        "/"
    )
    return PurePosixPath("/kaggle/input") / model / framework / variation / version


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
    student_mount = str(model_source_mount(student_source))
    teacher_mount = str(model_source_mount(teacher_source))

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
version_root = input_root / "datasets" / {repo_owner!r} / KJO_REPO_DATASET_SLUG / "versions"
if legacy_root.is_dir():
    dataset_root = legacy_root
else:
    versions = sorted(
        (path for path in version_root.glob("*") if path.is_dir()),
        key=lambda path: int(path.name) if path.name.isdigit() else -1,
    )
    if len(versions) != 1:
        available = sorted(str(path.relative_to(input_root)) for path in input_root.rglob("*") if path.is_dir())
        raise FileNotFoundError(
            "Kaggle dataset is not mounted at either supported layout. "
            f"dataset_source={{KJO_REPO_DATASET_SOURCE}} versions={{versions}} "
            f"available_inputs={{available[:100]}}"
        )
    dataset_root = versions[0]
source_repo = dataset_root / KJO_REPO_DIR_NAME
if not source_repo.is_dir():
    raise FileNotFoundError(f"repo directory is missing from dataset: {{source_repo}}")
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
STUDENT_MOUNT = Path({student_mount!r})
TEACHER_MOUNT = Path({teacher_mount!r})
for required in (REPO, CONFIG_PATH, STUDENT_MOUNT, TEACHER_MOUNT):
    if not required.exists():
        available = sorted(path.name for path in Path("/kaggle/input").glob("*"))
        raise FileNotFoundError(f"required input missing: {{required}}; inputs={{available}}")
config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
if Path(config["student"]["model_path"]) != STUDENT_MOUNT:
    raise RuntimeError("student model mount drifted from config")
if Path(config["teacher"]["model_path"]) != TEACHER_MOUNT:
    raise RuntimeError("teacher model mount drifted from config")
print("VDT_MODEL_SOURCE_PROVENANCE " + json.dumps({{
    "student": STUDENT_SOURCE,
    "student_mount": str(STUDENT_MOUNT),
    "teacher": TEACHER_SOURCE,
    "teacher_mount": str(TEACHER_MOUNT),
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
    "--config", str(CONFIG_PATH),
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
