"""Deterministic Kaggle notebook rendering for native checkpoint generation."""

from __future__ import annotations

from typing import Any

from vdt_tunix.kaggle_model_sources import (
    TUNIX_COMMIT,
    KaggleModelSourceError,
    _safe_relative_path,
    _validate_dataset_source,
    model_source_mount,
    validate_model_source,
)


def render_generation_notebook(
    *,
    variant: str,
    training_config_relative_path: str,
    generation_protocol_relative_path: str,
    repo_dataset_source: str,
    evaluation_dataset_source: str,
    checkpoint_kernel_source: str,
    checkpoint_relative_path: str,
    student_model_source: str,
) -> dict[str, Any]:
    """Render generation-only evidence for one trained variant."""

    if variant not in {"sft", "simple_opd", "simct"}:
        raise KaggleModelSourceError("variant must be sft, simple_opd, or simct")
    config_relative = _safe_relative_path(
        training_config_relative_path, "training_config_relative_path"
    )
    protocol_relative = _safe_relative_path(
        generation_protocol_relative_path, "generation_protocol_relative_path"
    )
    checkpoint_relative = _safe_relative_path(
        checkpoint_relative_path, "checkpoint_relative_path"
    )
    repo_owner, repo_slug = _validate_dataset_source(
        repo_dataset_source, "repo_dataset_source"
    )
    eval_owner, eval_slug = _validate_dataset_source(
        evaluation_dataset_source, "evaluation_dataset_source"
    )
    checkpoint_owner, checkpoint_slug = _validate_dataset_source(
        checkpoint_kernel_source, "checkpoint_kernel_source"
    )
    student_source = validate_model_source(student_model_source)
    student_mount = str(model_source_mount(student_source))

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
        raise FileNotFoundError(
            f"repo dataset {{KJO_REPO_DATASET_SOURCE}} is not mounted: {{versions}}"
        )
    dataset_root = versions[0]
source_repo = dataset_root / KJO_REPO_DIR_NAME
if not source_repo.is_dir():
    source_repo = dataset_root
required = (
    source_repo / "pyproject.toml",
    source_repo / "vdt_tunix",
    source_repo / {config_relative.as_posix()!r},
    source_repo / {protocol_relative.as_posix()!r},
)
if any(not path.exists() for path in required):
    raise FileNotFoundError(f"repo payload is incomplete: {{required}}")
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
TRAINING_CONFIG = REPO / {config_relative.as_posix()!r}
GENERATION_PROTOCOL = REPO / {protocol_relative.as_posix()!r}
EVALUATION_DATASET_SOURCE = {evaluation_dataset_source!r}
EVALUATION_OWNER = {eval_owner!r}
EVALUATION_SLUG = {eval_slug!r}
CHECKPOINT_KERNEL_SOURCE = {checkpoint_kernel_source!r}
CHECKPOINT_OWNER = {checkpoint_owner!r}
CHECKPOINT_SLUG = {checkpoint_slug!r}
CHECKPOINT_RELATIVE = PurePosixPath({checkpoint_relative.as_posix()!r})
EVALUATION_ROOT = Path("/kaggle/working/vdt_evaluation_inputs")

def resolve_input(owner, slug, allow_versions):
    root = Path(os.environ.get("KJO_KAGGLE_INPUT_ROOT", "/kaggle/input"))
    candidates = (
        root / slug,
        root / owner / slug,
        root / "kernels" / owner / slug,
        root / "datasets" / owner / slug,
    )
    for candidate in candidates:
        version_root = candidate / "versions"
        if candidate.is_dir() and not version_root.is_dir():
            return candidate
        if allow_versions and version_root.is_dir():
            versions = sorted(
                (path for path in version_root.glob("*") if path.is_dir()),
                key=lambda path: int(path.name) if path.name.isdigit() else -1,
            )
            if len(versions) == 1:
                return versions[0]
    raise FileNotFoundError(f"input {{owner}}/{{slug}} is not mounted")

evaluation_mount = resolve_input(EVALUATION_OWNER, EVALUATION_SLUG, True)
if EVALUATION_ROOT.exists():
    shutil.rmtree(EVALUATION_ROOT)
EVALUATION_ROOT.mkdir(parents=True)
for benchmark in ("gsm8k", "math500", "mbpp", "live-code-bench-v6"):
    direct = evaluation_mount / benchmark
    target = EVALUATION_ROOT / benchmark
    if (direct / "manifest.json").is_file() and (direct / "records.jsonl").is_file():
        target.symlink_to(direct, target_is_directory=True)
        continue
    archive = evaluation_mount / f"{{benchmark}}.zip"
    if not archive.is_file():
        raise FileNotFoundError(f"benchmark input is absent: {{benchmark}}")
    temporary = EVALUATION_ROOT / f".extract-{{benchmark}}"
    temporary.mkdir()
    with zipfile.ZipFile(archive) as handle:
        resolved = temporary.resolve()
        for member in handle.infolist():
            member_path = (temporary / member.filename).resolve()
            if resolved not in member_path.parents and member_path != resolved:
                raise RuntimeError(f"unsafe benchmark archive member: {{member.filename}}")
        handle.extractall(temporary)
    nested = temporary / benchmark
    source = nested if (nested / "manifest.json").is_file() else temporary
    if not (source / "manifest.json").is_file() or not (source / "records.jsonl").is_file():
        raise FileNotFoundError(f"archive did not contain {{benchmark}} records")
    if source == temporary:
        temporary.replace(target)
    else:
        source.replace(target)
        temporary.rmdir()

checkpoint_mount = resolve_input(CHECKPOINT_OWNER, CHECKPOINT_SLUG, False)
CHECKPOINT_ROOT = checkpoint_mount.joinpath(*CHECKPOINT_RELATIVE.parts)
if not (CHECKPOINT_ROOT / "latest.json").is_file():
    raise FileNotFoundError(f"checkpoint root is incomplete: {{CHECKPOINT_ROOT}}")
print("VDT_GENERATION_INPUT_PROVENANCE " + json.dumps({{
    "variant": {variant!r},
    "evaluation_dataset_source": EVALUATION_DATASET_SOURCE,
    "evaluation_root": str(EVALUATION_ROOT),
    "checkpoint_kernel_source": CHECKPOINT_KERNEL_SOURCE,
    "checkpoint_root": str(CHECKPOINT_ROOT),
}}, sort_keys=True))'''

    setup = f'''import importlib.metadata
import json
import platform
import subprocess
import sys

STUDENT_SOURCE = {student_source!r}
STUDENT_MOUNT = Path({student_mount!r})
for required in (
    REPO, TRAINING_CONFIG, GENERATION_PROTOCOL, EVALUATION_ROOT,
    CHECKPOINT_ROOT, STUDENT_MOUNT,
):
    if not required.exists():
        raise FileNotFoundError(f"required generation input missing: {{required}}")
training_config = json.loads(TRAINING_CONFIG.read_text(encoding="utf-8"))
if Path(training_config["student"]["model_path"]) != STUDENT_MOUNT:
    raise RuntimeError("student model mount drifted from config")
print("VDT_MODEL_SOURCE_PROVENANCE " + json.dumps({{
    "student": STUDENT_SOURCE,
    "student_mount": str(STUDENT_MOUNT),
    "teacher_loaded": False,
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
import tunix.generate.sampler
print("VDT_DEPENDENCY_PROVENANCE " + json.dumps({{
    "jax_before": before,
    "jax_after": after,
    "flax": flax.__version__,
    "huggingface_hub": importlib.metadata.version("huggingface-hub"),
    "transformers": transformers.__version__,
    "tunix_commit": observed_tunix_commit,
}}, sort_keys=True))'''

    run = f'''WORK = Path("/kaggle/working/vdt_generation_{variant}")
SCORE_WORK = Path("/kaggle/working/vdt_scoring_{variant}")
SUMMARY = WORK / "generation_summary.json"
command = [
    sys.executable,
    str(REPO / "scripts/tpu/kaggle_v5e8_generate.py"),
    "--training-config", str(TRAINING_CONFIG),
    "--generation-protocol", str(GENERATION_PROTOCOL),
    "--evaluation-root", str(EVALUATION_ROOT),
    "--checkpoint-root", str(CHECKPOINT_ROOT),
    "--variant", {variant!r},
    "--output-dir", str(WORK),
]
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
if result.returncode:
    raise RuntimeError(f"native generation failed with exit code {{result.returncode}}")
payload = json.loads(SUMMARY.read_text(encoding="utf-8"))
expected = {{
    "status": "complete",
    "phase": "native_tunix_generation",
    "variant": {variant!r},
    "scientific_evidence": False,
}}
drift = {{
    key: (payload.get(key), value)
    for key, value in expected.items()
    if payload.get(key) != value
}}
if len(payload.get("benchmarks", [])) != 4:
    drift["benchmarks"] = (len(payload.get("benchmarks", [])), 4)
if drift:
    raise RuntimeError(f"generation evidence contract mismatch: {{drift}}")
score_command = [
    sys.executable,
    str(REPO / "scripts/evaluation/score_generated_predictions.py"),
    "--generation-root", str(WORK),
    "--evaluation-root", str(EVALUATION_ROOT),
    "--generation-protocol", str(GENERATION_PROTOCOL),
    "--evaluator-source", str(REPO / "scripts/evaluation/evaluation.py"),
    "--output-dir", str(SCORE_WORK),
    "--workers", str(min(16, os.cpu_count() or 1)),
]
score_result = subprocess.run(
    score_command,
    cwd=REPO,
    text=True,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    env={{**os.environ, "PYTHONPATH": str(REPO)}},
)
print(score_result.stdout, end="")
print(score_result.stderr, end="", file=sys.stderr)
if score_result.returncode:
    raise RuntimeError(
        f"paper-released scoring failed with exit code {{score_result.returncode}}"
    )
score_payload = json.loads(
    (SCORE_WORK / "scoring_summary.json").read_text(encoding="utf-8")
)
score_expected = {{
    "status": "complete",
    "phase": "paper_released_evaluator_scoring",
    "variant": {variant!r},
    "scientific_evidence": True,
    "paper_reproduction": False,
}}
score_drift = {{
    key: (score_payload.get(key), value)
    for key, value in score_expected.items()
    if score_payload.get(key) != value
}}
if len(score_payload.get("benchmarks", [])) != 4:
    score_drift["benchmarks"] = (len(score_payload.get("benchmarks", [])), 4)
if score_drift:
    raise RuntimeError(f"scoring evidence contract mismatch: {{score_drift}}")
cache = Path(training_config["checkpoint"]["root"]).parent / "base-model-cache"
if cache.is_dir():
    shutil.rmtree(cache)
if EVALUATION_ROOT.exists():
    shutil.rmtree(EVALUATION_ROOT)
if REPO.exists():
    shutil.rmtree(REPO)
print("VDT_GENERATION_SUMMARY " + json.dumps(payload, sort_keys=True))
print("VDT_SCORING_SUMMARY " + json.dumps(score_payload, sort_keys=True))'''

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
                    f"# VDT native Tunix generation: {variant}\n",
                    "Native generation plus the paper-released scorer; this is not the official benchmark harness or a paper reproduction.\n",
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
