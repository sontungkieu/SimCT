"""Local-only Kaggle TPU v5e-8 staging for the VDT contract canary."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import tempfile
import textwrap
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from vdt_tunix.config import ConfigError, RunConfig, load_config


PACKAGE_VERSION = 1
SIMCT_COMMIT = "cf0f33a0e6c967d4b74ea32b2dba12be01b73b9e"
TUNIX_COMMIT = "50f5752a17edec56e2aa30aabfc03859949adf6f"
SUBMIT_ACCELERATOR = "TpuV5E8"
ACCELERATOR_LABEL = "TPU VM v5-8"
DEFAULT_OUTPUT_ROOT = Path("/mnt/d/dev/codex/vdt-dynamic-span")
DEFAULT_KJO_CLI = Path(
    "/mnt/c/Users/Tung/.codex/skills/kaggle-job-ops/scripts/"
    "kaggle_job_ops.py"
)
_OWNER = re.compile(r"^[a-z0-9][a-z0-9_-]{1,49}$")
_SLUG = re.compile(r"^[a-z0-9][a-z0-9-]{2,49}$")
_SHA40 = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_PLACEHOLDER = re.compile(r"replace|placeholder|changeme|todo", re.I)
_REQUIRED_SOURCE_FILES = (
    "requirements-tpu.txt",
    "environments/kaggle-tpu/pyproject.toml",
    "environments/kaggle-tpu/provider-constraints.json",
    "environments/kaggle-tpu/uv.lock",
    "scripts/tpu/kaggle_v5e8_canary.py",
    "vdt_tunix/config.py",
    "vdt_tunix/kaggle_uv.py",
    "vdt_tunix/real_backend.py",
    "vdt_tunix/runtime.py",
)
_AUTH_ENV = (
    "KAGGLE_USERNAME",
    "KAGGLE_KEY",
    "KAGGLE_API_TOKEN",
    "KAGGLE_API_V1_TOKEN",
)


class KagglePackageError(ValueError):
    """A local prerequisite is absent or ambiguous."""


def _object(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise KagglePackageError(f"{context} must be a JSON object")
    return value


def _keys(value: Mapping[str, Any], context: str, required: set[str]) -> None:
    missing = sorted(required - value.keys())
    extra = sorted(value.keys() - required)
    if missing:
        raise KagglePackageError(f"{context} missing keys: {missing}")
    if extra:
        raise KagglePackageError(f"{context} unsupported keys: {extra}")


def _text(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise KagglePackageError(f"{context} must be a non-empty string")
    if _PLACEHOLDER.search(value):
        raise KagglePackageError(f"{context} still contains a placeholder")
    return value


def _path(value: Any, context: str, boundary: Path) -> Path:
    path = Path(_text(value, context)).expanduser()
    if not path.is_absolute():
        raise KagglePackageError(f"{context} must be absolute")
    try:
        path.resolve().relative_to(boundary.resolve())
    except ValueError as exc:
        raise KagglePackageError(f"{context} must be under {boundary}") from exc
    if any(
        part.lower() in {".secrets", ".env", "kaggle.json", "all-kaggle.json"}
        for part in path.parts
    ):
        raise KagglePackageError(f"{context} points at a forbidden secret path")
    return path.resolve()


def _relative(value: Any, context: str) -> Path:
    path = Path(_text(value, context))
    if path.is_absolute() or ".." in path.parts or path == Path("."):
        raise KagglePackageError(f"{context} must be a safe relative path")
    return path


def _dataset_source(value: Any, context: str, owner: str) -> str:
    source = _text(value, context)
    parts = source.split("/")
    if (
        len(parts) != 2
        or parts[0] != owner
        or not _OWNER.fullmatch(parts[0])
        or not _SLUG.fullmatch(parts[1])
    ):
        raise KagglePackageError(
            f"{context} must be an exact same-owner Kaggle owner/slug"
        )
    return source


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_dir(path: Path, context: str) -> Path:
    if not path.is_dir() or not any(path.iterdir()):
        raise KagglePackageError(f"{context} is absent or empty: {path}")
    return path


@dataclasses.dataclass(frozen=True, slots=True)
class CheckpointAsset:
    local_root: Path
    dataset_source: str
    checkpoint_subpath: Path
    manifest_subpath: Path
    manifest_sha256: str


@dataclasses.dataclass(frozen=True, slots=True)
class PackageSpec:
    run_id: str
    owner: str
    slug: str
    title: str
    config_path: Path
    config: RunConfig
    source_manifest: Path
    source_dataset: str
    source_tree_sha256: str
    requirements_sha256: str
    uv_project_sha256: str
    provider_constraints_sha256: str
    uv_lock_sha256: str
    tokenizer_root: Path
    tokenizer_dataset: str
    hf_home_subpath: Path
    student_checkpoint: CheckpointAsset
    teacher_checkpoint: CheckpointAsset

    @property
    def kernel_id(self) -> str:
        return f"{self.owner}/{self.slug}"

    @property
    def dataset_sources(self) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                (
                    self.source_dataset,
                    self.tokenizer_dataset,
                    self.student_checkpoint.dataset_source,
                    self.teacher_checkpoint.dataset_source,
                )
            )
        )


def _checkpoint(
    value: Any,
    *,
    context: str,
    owner: str,
    boundary: Path,
) -> CheckpointAsset:
    raw = _object(value, context)
    _keys(
        raw,
        context,
        {
            "local_root",
            "dataset_source",
            "checkpoint_subpath",
            "manifest_subpath",
            "manifest_sha256",
        },
    )
    root = _require_dir(
        _path(raw["local_root"], f"{context}.local_root", boundary), context
    )
    checkpoint_subpath = _relative(
        raw["checkpoint_subpath"], f"{context}.checkpoint_subpath"
    )
    _require_dir(root / checkpoint_subpath, f"{context}.checkpoint")
    manifest_subpath = _relative(
        raw["manifest_subpath"], f"{context}.manifest_subpath"
    )
    manifest = root / manifest_subpath
    if not manifest.is_file():
        raise KagglePackageError(f"{context} manifest is absent: {manifest}")
    expected = _text(raw["manifest_sha256"], f"{context}.manifest_sha256")
    if not _SHA256.fullmatch(expected) or _sha256(manifest) != expected:
        raise KagglePackageError(f"{context} manifest SHA-256 mismatch")
    return CheckpointAsset(
        local_root=root,
        dataset_source=_dataset_source(
            raw["dataset_source"], f"{context}.dataset_source", owner
        ),
        checkpoint_subpath=checkpoint_subpath,
        manifest_subpath=manifest_subpath,
        manifest_sha256=expected,
    )


def _validate_tokenizer_cache(config: RunConfig, hf_home: Path) -> None:
    for role, model in (("student", config.student), ("teacher", config.teacher)):
        if not _SHA40.fullmatch(model.model_revision):
            raise KagglePackageError(f"{role}.model_revision must be exact 40-hex")
        if not _SHA40.fullmatch(model.tokenizer_revision):
            raise KagglePackageError(f"{role}.tokenizer_revision must be exact 40-hex")
        model_dir = f"models--{model.tokenizer_id.replace('/', '--')}"
        candidates = (
            hf_home / "hub" / model_dir / "snapshots" / model.tokenizer_revision,
            hf_home / model_dir / "snapshots" / model.tokenizer_revision,
        )
        if not any(path.is_dir() and any(path.iterdir()) for path in candidates):
            raise KagglePackageError(
                f"{role} tokenizer cache is absent at revision "
                f"{model.tokenizer_revision}"
            )


def _source_snapshot(
    path: Path, owner: str
) -> tuple[str, str, str, str, str, str]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise KagglePackageError(f"required source snapshot is absent: {path}") from exc
    except json.JSONDecodeError as exc:
        raise KagglePackageError("source snapshot manifest is invalid JSON") from exc
    if payload.get("ok") is not True:
        raise KagglePackageError("source snapshot manifest is not ok")
    if payload.get("git", {}).get("commit") != SIMCT_COMMIT:
        raise KagglePackageError("source snapshot has the wrong SimCT commit")
    source = _dataset_source(
        payload.get("dataset_source"), "source.dataset_source", owner
    )
    content = payload.get("content_manifest", {})
    tree = content.get("tree_sha256")
    if not isinstance(tree, str) or not _SHA256.fullmatch(tree):
        raise KagglePackageError("source snapshot tree_sha256 is invalid")
    repo = Path(content.get("dataset_dir", "")) / "repo"
    for relative in _REQUIRED_SOURCE_FILES:
        if not (repo / relative).is_file():
            raise KagglePackageError(f"source snapshot lacks {relative}")
    requirements = repo / "requirements-tpu.txt"
    if TUNIX_COMMIT not in requirements.read_text(encoding="utf-8"):
        raise KagglePackageError("source requirements do not pin the Tunix commit")
    uv_project = repo / "environments/kaggle-tpu/pyproject.toml"
    provider_constraints = repo / "environments/kaggle-tpu/provider-constraints.json"
    uv_lock = repo / "environments/kaggle-tpu/uv.lock"
    return (
        source,
        tree,
        _sha256(requirements),
        _sha256(uv_project),
        _sha256(provider_constraints),
        _sha256(uv_lock),
    )


def load_package_spec(
    path: str | Path,
    *,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> PackageSpec:
    """Validate every local prerequisite without accessing credentials."""

    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except OSError as exc:
        raise KagglePackageError(f"package spec is absent: {path}") from exc
    except json.JSONDecodeError as exc:
        raise KagglePackageError("package spec is invalid JSON") from exc
    raw = _object(raw, "package spec")
    _keys(
        raw,
        "package spec",
        {
            "contract_version",
            "run_id",
            "provenance",
            "kaggle",
            "canary_config",
            "source_snapshot_manifest",
            "tokenizer_cache",
            "student_checkpoint",
            "teacher_checkpoint",
        },
    )
    if raw["contract_version"] != PACKAGE_VERSION:
        raise KagglePackageError(f"contract_version must be {PACKAGE_VERSION}")
    provenance = _object(raw["provenance"], "provenance")
    _keys(provenance, "provenance", {"upstream_simct_commit", "tunix_commit"})
    if provenance["upstream_simct_commit"] != SIMCT_COMMIT:
        raise KagglePackageError("wrong upstream SimCT commit")
    if provenance["tunix_commit"] != TUNIX_COMMIT:
        raise KagglePackageError("wrong Tunix commit")
    kaggle = _object(raw["kaggle"], "kaggle")
    _keys(kaggle, "kaggle", {"owner", "slug", "title"})
    owner = _text(kaggle["owner"], "kaggle.owner")
    slug = _text(kaggle["slug"], "kaggle.slug")
    if not _OWNER.fullmatch(owner):
        raise KagglePackageError("Kaggle owner is invalid")
    if not _SLUG.fullmatch(slug):
        raise KagglePackageError("Kaggle slug is invalid")
    run_id = _text(raw["run_id"], "run_id")
    if not _SLUG.fullmatch(run_id):
        raise KagglePackageError("run_id is not Kaggle-safe")
    title = _text(kaggle["title"], "kaggle.title")
    boundary = output_root.resolve()
    if not boundary.is_dir():
        raise KagglePackageError(f"output root is absent: {boundary}")
    config_path = _path(raw["canary_config"], "canary_config", boundary)
    try:
        config = load_config(config_path)
    except (ConfigError, OSError) as exc:
        raise KagglePackageError(f"invalid canary config: {exc}") from exc
    if config.run_id != run_id:
        raise KagglePackageError("canary config run_id does not match")
    student = _checkpoint(
        raw["student_checkpoint"],
        context="student_checkpoint",
        owner=owner,
        boundary=boundary,
    )
    teacher = _checkpoint(
        raw["teacher_checkpoint"],
        context="teacher_checkpoint",
        owner=owner,
        boundary=boundary,
    )
    if Path(config.student.maxtext_checkpoint_uri).resolve() != (
        student.local_root / student.checkpoint_subpath
    ).resolve():
        raise KagglePackageError("student checkpoint spec does not match canary config")
    if Path(config.teacher.maxtext_checkpoint_uri).resolve() != (
        teacher.local_root / teacher.checkpoint_subpath
    ).resolve():
        raise KagglePackageError("teacher checkpoint spec does not match canary config")
    tokenizer = _object(raw["tokenizer_cache"], "tokenizer_cache")
    _keys(
        tokenizer,
        "tokenizer_cache",
        {"local_root", "dataset_source", "hf_home_subpath"},
    )
    tokenizer_root = _require_dir(
        _path(tokenizer["local_root"], "tokenizer_cache.local_root", boundary),
        "tokenizer_cache.local_root",
    )
    hf_home_subpath = _relative(
        tokenizer["hf_home_subpath"], "tokenizer_cache.hf_home_subpath"
    )
    _validate_tokenizer_cache(
        config,
        _require_dir(tokenizer_root / hf_home_subpath, "tokenizer HF_HOME"),
    )
    source_manifest = _path(
        raw["source_snapshot_manifest"], "source_snapshot_manifest", boundary
    )
    (
        source_dataset,
        source_tree,
        requirements_sha,
        uv_project_sha,
        provider_constraints_sha,
        uv_lock_sha,
    ) = _source_snapshot(
        source_manifest, owner
    )
    return PackageSpec(
        run_id=run_id,
        owner=owner,
        slug=slug,
        title=title,
        config_path=config_path,
        config=config,
        source_manifest=source_manifest,
        source_dataset=source_dataset,
        source_tree_sha256=source_tree,
        requirements_sha256=requirements_sha,
        uv_project_sha256=uv_project_sha,
        provider_constraints_sha256=provider_constraints_sha,
        uv_lock_sha256=uv_lock_sha,
        tokenizer_root=tokenizer_root,
        tokenizer_dataset=_dataset_source(
            tokenizer["dataset_source"], "tokenizer_cache.dataset_source", owner
        ),
        hf_home_subpath=hf_home_subpath,
        student_checkpoint=student,
        teacher_checkpoint=teacher,
    )


def _runtime_checkpoint(asset: CheckpointAsset) -> dict[str, str]:
    return {
        "dataset_source": asset.dataset_source,
        "checkpoint_subpath": asset.checkpoint_subpath.as_posix(),
        "manifest_subpath": asset.manifest_subpath.as_posix(),
        "manifest_sha256": asset.manifest_sha256,
    }


def render_source_notebook(spec: PackageSpec) -> dict[str, Any]:
    """Render readable cells; KJO adds logging, repo copy, and TPU probe."""

    runtime = {
        "provenance": {
            "upstream_simct_commit": SIMCT_COMMIT,
            "tunix_commit": TUNIX_COMMIT,
            "source_tree_sha256": spec.source_tree_sha256,
            "requirements_tpu_sha256": spec.requirements_sha256,
            "uv_project_sha256": spec.uv_project_sha256,
            "provider_constraints_sha256": spec.provider_constraints_sha256,
            "uv_lock_sha256": spec.uv_lock_sha256,
            "submit_accelerator": SUBMIT_ACCELERATOR,
        },
        "tokenizer": {
            "dataset_source": spec.tokenizer_dataset,
            "hf_home_subpath": spec.hf_home_subpath.as_posix(),
        },
        "student": _runtime_checkpoint(spec.student_checkpoint),
        "teacher": _runtime_checkpoint(spec.teacher_checkpoint),
        "config": spec.config.to_dict(),
    }
    literal = repr(json.dumps(runtime, sort_keys=True))
    setup = textwrap.dedent(
        f"""
        import hashlib, json, os
        from pathlib import Path
        RUNTIME = json.loads({literal})

        def mount(source):
            owner, slug = source.split("/", 1)
            legacy = Path("/kaggle/input") / slug
            if legacy.is_dir():
                return legacy
            versions = Path("/kaggle/input/datasets") / owner / slug / "versions"
            matches = [path for path in versions.glob("*") if path.is_dir()]
            if len(matches) != 1:
                raise RuntimeError(f"expected one mount for {{source}}, found {{matches}}")
            return matches[0]

        def sha256(path):
            digest = hashlib.sha256()
            with Path(path).open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
            return digest.hexdigest()

        def checkpoint(asset):
            root = mount(asset["dataset_source"])
            manifest = root / asset["manifest_subpath"]
            target = root / asset["checkpoint_subpath"]
            if not manifest.is_file() or sha256(manifest) != asset["manifest_sha256"]:
                raise RuntimeError(f"checkpoint manifest mismatch: {{asset['dataset_source']}}")
            if not target.is_dir() or not any(target.iterdir()):
                raise RuntimeError(f"checkpoint missing: {{asset['dataset_source']}}")
            return target

        repo = Path("/kaggle/working/repo")
        requirements = repo / "requirements-tpu.txt"
        if sha256(requirements) != RUNTIME["provenance"]["requirements_tpu_sha256"]:
            raise RuntimeError("requirements-tpu.txt hash mismatch")
        uv_project = repo / "environments/kaggle-tpu/pyproject.toml"
        provider_constraints = repo / "environments/kaggle-tpu/provider-constraints.json"
        uv_lock = repo / "environments/kaggle-tpu/uv.lock"
        if sha256(uv_project) != RUNTIME["provenance"]["uv_project_sha256"]:
            raise RuntimeError("Kaggle TPU uv project hash mismatch")
        if sha256(provider_constraints) != RUNTIME["provenance"]["provider_constraints_sha256"]:
            raise RuntimeError("Kaggle TPU provider constraints hash mismatch")
        if sha256(uv_lock) != RUNTIME["provenance"]["uv_lock_sha256"]:
            raise RuntimeError("Kaggle TPU uv.lock hash mismatch")
        hf_home = mount(RUNTIME["tokenizer"]["dataset_source"]) / RUNTIME["tokenizer"]["hf_home_subpath"]
        if not hf_home.is_dir():
            raise RuntimeError("tokenizer cache is not mounted")
        os.environ.update({{"HF_HOME": str(hf_home), "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_TOKEN": ""}})
        config = RUNTIME["config"]
        config["student"]["maxtext_checkpoint_uri"] = str(checkpoint(RUNTIME["student"]))
        config["teacher"]["maxtext_checkpoint_uri"] = str(checkpoint(RUNTIME["teacher"]))
        work = Path("/kaggle/working/vdt_simct_canary")
        work.mkdir(parents=True, exist_ok=True)
        config["checkpoint"]["root"] = str(work / "checkpoints")
        config_path = work / "canary_config.json"
        config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
        print("VDT_PACKAGE_PROVENANCE " + json.dumps(RUNTIME["provenance"], sort_keys=True))
        """
    ).strip()
    dependencies = textwrap.dedent(
        """
        from vdt_tunix.kaggle_uv import (
            bootstrap_locked_kaggle_environment,
            runtime_subprocess_environment,
        )

        LOCKED_ENVIRONMENT = bootstrap_locked_kaggle_environment(
            repo,
            Path("/tmp/vdt_simct_canary_environment"),
            summary_path=work / "locked_environment_summary.json",
        )
        RUNTIME_PYTHON = Path(LOCKED_ENVIRONMENT["runtime_python"])
        RUNTIME_SUBPROCESS_ENV = runtime_subprocess_environment(
            repo, LOCKED_ENVIRONMENT
        )
        """
    ).strip()
    run = textwrap.dedent(
        """
        import json, subprocess, sys
        output = work / "canary.json"
        result = subprocess.run([str(RUNTIME_PYTHON), str(repo / "scripts/tpu/kaggle_v5e8_canary.py"), "--config", str(config_path), "--output", str(output)], cwd=repo, capture_output=True, text=True, env=RUNTIME_SUBPROCESS_ENV)
        print(result.stdout, end="")
        print(result.stderr, end="", file=sys.stderr)
        if result.returncode:
            raise RuntimeError(f"VDT canary failed with exit code {result.returncode}")
        payload = json.loads(output.read_text(encoding="utf-8"))
        expected = {"status": "passed", "real_model_integration": True, "cross_tokenization_observed": True, "scientific_evidence": False, "simct_update_executed": True}
        if any(payload.get(key) != value for key, value in expected.items()):
            raise RuntimeError("VDT canary evidence contract mismatch")
        print("VDT_CANARY_SUMMARY " + json.dumps(payload, sort_keys=True))
        """
    ).strip()

    def cell(source: str) -> dict[str, Any]:
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
                    "# VDT single-teacher cross-tokenizer TPU v5e-8 dry run\n",
                    "One optimizer-step canary; not a scientific reproduction result.\n",
                ],
            },
            cell(setup),
            cell(dependencies),
            cell(run),
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


def _run_kjo(args: Sequence[str], *, cli: Path, cwd: Path) -> None:
    if not cli.is_file():
        raise KagglePackageError(f"Kaggle Job Ops CLI is absent: {cli}")
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    for name in _AUTH_ENV:
        env.pop(name, None)
    result = subprocess.run(
        [os.sys.executable, os.fspath(cli), *args],
        cwd=cwd,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        detail = (result.stderr or result.stdout).strip()[-3000:]
        raise KagglePackageError(f"KJO {args[0]} failed: {detail}")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _submit_command(spec: PackageSpec, package_dir: Path, cli: Path) -> str:
    command = [
        "python3",
        os.fspath(cli),
        "submit-kernel",
        "--run-dir",
        os.fspath(package_dir),
        "--metadata",
        os.fspath(package_dir / "stage/kernel-metadata.json"),
        "--submitted-notebook",
        os.fspath(package_dir / "stage/submitted_notebook.ipynb"),
        "--expected-accelerator",
        "tpu",
        "--submit-accelerator",
        SUBMIT_ACCELERATOR,
        "--owner",
        spec.owner,
        "--repo-dataset-manifest",
        os.fspath(spec.source_manifest),
        "--require-repo-dataset",
        "--require-repo-dataset-push",
        "--require-repo-dataset-manifest-verifiable",
    ]
    for source in spec.dataset_sources[1:]:
        command.extend(("--required-dataset-source", source))
    command.extend(
        (
            "--require-notebook-logging-contract",
            "--require-accelerator-probe-contract",
            "--record-registry",
            "/home/tung/vdt-dynamic-span/.secrets/kaggle_notebooks.jsonl",
            "--reservation-token",
            "${RESERVATION_TOKEN:?set from reserve-owners}",
            "--secret-mode",
            "none",
            "--is-private",
            "--artifact-mode",
            "logs-only",
            "--retention-action",
            "delete-after-download",
            "--kaggle-bin",
            "/home/tung/.codex/state/kaggle-job-ops/cli/kaggle-2.2.3/bin/kaggle",
            "--project-root",
            "/home/tung/vdt-dynamic-span",
            "--run-id",
            spec.run_id,
        )
    )
    quoted = [
        item if item.startswith("${RESERVATION_TOKEN:") else shlex.quote(item)
        for item in command
    ]
    return "#!/usr/bin/env bash\nset -euo pipefail\n" + " \\\n+  ".join(quoted) + "\n"


def stage_package(
    spec_path: str | Path,
    output_dir: str | Path,
    *,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    kjo_cli: Path = DEFAULT_KJO_CLI,
    clean: bool = False,
) -> Path:
    """Materialize and validate a local package. This function cannot submit."""

    spec = load_package_spec(spec_path, output_root=output_root)
    output_dir = _path(
        os.fspath(Path(output_dir).absolute()), "output_dir", output_root
    )
    if output_dir == output_root.resolve():
        raise KagglePackageError("output_dir cannot be the task root")
    if output_dir.exists() and not clean:
        raise KagglePackageError(f"output_dir already exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent)
    )
    try:
        source = temporary / "source/vdt_simct_v5e8_source.ipynb"
        _write_json(source, render_source_notebook(spec))
        args = [
            "stage-notebook-package",
            "--source-notebook",
            os.fspath(source),
            "--run-dir",
            os.fspath(temporary),
            "--out-dir",
            os.fspath(temporary / "stage"),
            "--owner",
            spec.owner,
            "--slug",
            spec.slug,
            "--title",
            spec.title,
            "--run-id",
            spec.run_id,
            "--accelerator",
            "tpu",
            "--submit-accelerator",
            SUBMIT_ACCELERATOR,
            "--repo-dataset-manifest",
            os.fspath(spec.source_manifest),
            "--require-repo-dataset",
            "--require-repo-dataset-manifest-verifiable",
            "--inject-repo-copy-cell",
            "--instrument-logging",
            "--instrumentation-mode",
            "inline",
            "--require-notebook-logging-contract",
            "--inject-accelerator-probe",
            "--require-accelerator-probe-contract",
            "--clean",
        ]
        for dataset in spec.dataset_sources[1:]:
            args.extend(
                ("--dataset-source", dataset, "--required-dataset-source", dataset)
            )
        _run_kjo(args, cli=kjo_cli, cwd=temporary)
        metadata_path = temporary / "stage/kernel-metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        required = {
            "id": spec.kernel_id,
            "machine_shape": SUBMIT_ACCELERATOR,
            "enable_gpu": False,
            "enable_tpu": True,
            "is_private": True,
        }
        if any(metadata.get(key) != value for key, value in required.items()):
            raise KagglePackageError(
                "generated kernel metadata violates the TPU contract"
            )
        observed_sources = tuple(metadata.get("dataset_sources", ()))
        if (
            len(observed_sources) != len(spec.dataset_sources)
            or set(observed_sources) != set(spec.dataset_sources)
        ):
            raise KagglePackageError("generated dataset_sources drifted from the spec")
        manifest = {
            "contract_version": PACKAGE_VERSION,
            "ok": True,
            "kernel_id": spec.kernel_id,
            "accelerator": {
                "kaggle_label": ACCELERATOR_LABEL,
                "submit_shape": SUBMIT_ACCELERATOR,
                "expected_device_count": 8,
            },
            "provenance": {
                "upstream_simct_commit": SIMCT_COMMIT,
                "tunix_commit": TUNIX_COMMIT,
                "source_tree_sha256": spec.source_tree_sha256,
                "uv_project_sha256": spec.uv_project_sha256,
                "provider_constraints_sha256": spec.provider_constraints_sha256,
                "uv_lock_sha256": spec.uv_lock_sha256,
                "canary_config_sha256": spec.config.digest(),
                "student_checkpoint_manifest_sha256": (
                    spec.student_checkpoint.manifest_sha256
                ),
                "teacher_checkpoint_manifest_sha256": (
                    spec.teacher_checkpoint.manifest_sha256
                ),
            },
            "dataset_sources": list(spec.dataset_sources),
            "remote_submit_performed": False,
            "scientific_evidence": False,
            "simct_update_executed": True,
            "remaining_submit_gates": [
                "successful source/checkpoint/tokenizer dataset upload evidence",
                "exact-slug absence check",
                "TPU capacity reservation",
            ],
        }
        _write_json(temporary / "vdt_kaggle_package_manifest.json", manifest)
        submit = temporary / "future_submit_command.sh"
        submit.write_text(_submit_command(spec, output_dir, kjo_cli), encoding="utf-8")
        submit.chmod(0o755)
        if output_dir.exists():
            shutil.rmtree(output_dir)
        temporary.replace(output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output_dir


def safe_summary(spec: PackageSpec) -> dict[str, Any]:
    return {
        "ok": True,
        "kernel_id": spec.kernel_id,
        "accelerator": SUBMIT_ACCELERATOR,
        "accelerator_label": ACCELERATOR_LABEL,
        "expected_device_count": 8,
        "dataset_sources": list(spec.dataset_sources),
        "upstream_simct_commit": SIMCT_COMMIT,
        "tunix_commit": TUNIX_COMMIT,
        "uv_project_sha256": spec.uv_project_sha256,
        "provider_constraints_sha256": spec.provider_constraints_sha256,
        "uv_lock_sha256": spec.uv_lock_sha256,
        "remote_submit_performed": False,
    }
