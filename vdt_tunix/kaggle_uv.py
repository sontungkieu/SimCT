"""Locked userspace environment bootstrap for Kaggle TPU notebooks.

Kaggle owns the accelerator-coupled JAX, jaxlib, and libtpu installations.
The project therefore locks every other package with uv, installs that export
into a system-site-packages virtual environment, and attests that the provider
stack was inherited rather than replaced.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path
from typing import Any, Sequence


UV_VERSION = "0.10.2"
LOCK_PROJECT_RELATIVE = Path("environments/kaggle-tpu")
PROVIDER_PACKAGES = ("jax", "jaxlib", "libtpu")
TUNIX_COMMIT = "50f5752a17edec56e2aa30aabfc03859949adf6f"
_REQUIREMENT = re.compile(r"^([A-Za-z0-9_.-]+)(?:\[.*\])?(?:==|\s*@|\s*;|$)")


class KaggleUvEnvironmentError(RuntimeError):
    """Raised when the locked environment cannot preserve the TPU runtime."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _distribution_versions(names: Sequence[str]) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def load_provider_constraints(path: str | Path) -> dict[str, Any]:
    """Load the exact Kaggle-owned runtime contract and reject loose inputs."""

    path = Path(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise KaggleUvEnvironmentError(
            f"provider constraints are missing: {path}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise KaggleUvEnvironmentError("provider constraints are invalid JSON") from exc
    expected_keys = {
        "accelerator",
        "contract_version",
        "device_count",
        "jax",
        "jaxlib",
        "python",
    }
    if not isinstance(payload, dict) or set(payload) != expected_keys:
        raise KaggleUvEnvironmentError(
            "provider constraints must contain exactly " + str(sorted(expected_keys))
        )
    if payload["contract_version"] != 1:
        raise KaggleUvEnvironmentError("unsupported provider contract version")
    if payload["accelerator"] != "TpuV5E8" or payload["device_count"] != 8:
        raise KaggleUvEnvironmentError("provider contract is not TPU v5e-8")
    for name in ("python", "jax", "jaxlib"):
        value = payload[name]
        if not isinstance(value, str) or not re.fullmatch(r"\d+\.\d+\.\d+", value):
            raise KaggleUvEnvironmentError(
                f"provider constraint {name} must be an exact three-part version"
            )
    return payload


def _run(
    command: Sequence[str | os.PathLike[str]],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        [str(part) for part in command],
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode:
        combined = (result.stdout + "\n" + result.stderr).strip()
        tail = combined[-4000:]
        raise KaggleUvEnvironmentError(
            f"command failed ({result.returncode}): {command[0]}\n{tail}"
        )
    return result


def _resolve_uv() -> Path:
    expected = f"uv {UV_VERSION}"
    candidates = [
        Path(shutil.which("uv")) if shutil.which("uv") else None,
        Path(sysconfig.get_path("scripts")) / "uv",
    ]
    for candidate in candidates:
        if candidate is None or not candidate.is_file():
            continue
        result = subprocess.run(
            [str(candidate), "--version"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if result.returncode == 0 and result.stdout.strip() == expected:
            return candidate

    _run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-input",
            "--no-deps",
            f"uv=={UV_VERSION}",
        ]
    )
    for candidate in candidates:
        if candidate is None or not candidate.is_file():
            continue
        result = _run([candidate, "--version"])
        if result.stdout.strip() == expected:
            return candidate
    raise KaggleUvEnvironmentError(f"could not resolve pinned {expected}")


def validate_exported_requirements(path: str | Path) -> tuple[str, ...]:
    """Fail if the uv export would install provider-owned accelerator packages."""

    path = Path(path)
    observed: set[str] = set()
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith(("#", "--", "\\")):
            continue
        match = _REQUIREMENT.match(line)
        if match:
            observed.add(match.group(1).lower().replace("_", "-"))
    forbidden = sorted(set(PROVIDER_PACKAGES) & observed)
    if forbidden:
        raise KaggleUvEnvironmentError(
            f"uv export contains provider-owned packages: {forbidden}"
        )
    return tuple(sorted(observed))


def _attest_runtime(runtime_python: Path, repo: Path) -> dict[str, Any]:
    script = r'''
import importlib.metadata
import json
from pathlib import Path

import flax
import jax
import jaxlib
import orbax.checkpoint
import optax
import sentencepiece
import transformers
import wandb
import tunix.generate.sampler
import tunix.models.automodel

direct = json.loads(
    importlib.metadata.distribution("google-tunix").read_text("direct_url.json")
    or "{}"
)
print(json.dumps({
    "versions": {
        name: importlib.metadata.version(name)
        for name in (
            "flax", "google-tunix", "huggingface-hub", "jax", "jaxlib",
            "optax", "orbax-checkpoint", "sentencepiece", "transformers",
            "wandb",
        )
    },
    "jax_file": str(Path(jax.__file__).resolve()),
    "jaxlib_file": str(Path(jaxlib.__file__).resolve()),
    "jax_device_count": len(jax.devices()),
    "jax_platforms": sorted({device.platform for device in jax.devices()}),
    "tunix_commit": direct.get("vcs_info", {}).get("commit_id"),
}, sort_keys=True))
'''
    environment = {
        **os.environ,
        "PYTHONPATH": str(repo),
        "VIRTUAL_ENV": str(runtime_python.parent.parent),
        "PATH": str(runtime_python.parent) + os.pathsep + os.environ.get("PATH", ""),
    }
    result = _run([runtime_python, "-c", script], cwd=repo, env=environment)
    try:
        payload = json.loads(result.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as exc:
        raise KaggleUvEnvironmentError("runtime attestation was not JSON") from exc
    if payload.get("tunix_commit") != TUNIX_COMMIT:
        raise KaggleUvEnvironmentError(
            f"installed Tunix commit drifted: {payload.get('tunix_commit')}"
        )
    return payload


def bootstrap_locked_kaggle_environment(
    repo: str | Path,
    working_dir: str | Path,
) -> dict[str, Any]:
    """Create and attest the locked hybrid uv environment used by TPU jobs."""

    repo = Path(repo).resolve()
    working_dir = Path(working_dir).resolve()
    lock_project = repo / LOCK_PROJECT_RELATIVE
    lock_file = lock_project / "uv.lock"
    project_file = lock_project / "pyproject.toml"
    provider_file = lock_project / "provider-constraints.json"
    for required in (project_file, lock_file, provider_file):
        if not required.is_file():
            raise KaggleUvEnvironmentError(f"locked environment input missing: {required}")

    provider_before = _distribution_versions(PROVIDER_PACKAGES)
    provider_constraints = load_provider_constraints(provider_file)
    expected_provider = {
        "jax": provider_constraints.get("jax"),
        "jaxlib": provider_constraints.get("jaxlib"),
    }
    if platform.python_version() != provider_constraints.get("python"):
        raise KaggleUvEnvironmentError(
            "Kaggle Python drifted from the locked provider image: "
            f"{platform.python_version()} != {provider_constraints.get('python')}"
        )
    if {name: provider_before[name] for name in ("jax", "jaxlib")} != expected_provider:
        raise KaggleUvEnvironmentError(
            f"Kaggle provider JAX stack drifted: {provider_before} != "
            f"{expected_provider}"
        )
    if provider_before["jax"] is None or provider_before["jaxlib"] is None:
        raise KaggleUvEnvironmentError(
            f"Kaggle provider JAX stack is incomplete: {provider_before}"
        )

    working_dir.mkdir(parents=True, exist_ok=True)
    uv = _resolve_uv()
    exported = working_dir / "requirements-kaggle-tpu.lock"
    venv = working_dir / ".venv"
    _run([uv, "lock", "--check", "--project", lock_project], cwd=repo)
    _run(
        [
            uv,
            "export",
            "--project",
            lock_project,
            "--locked",
            "--no-header",
            "--no-emit-project",
            "--no-emit-package",
            "jax",
            "--no-emit-package",
            "jaxlib",
            "--no-emit-package",
            "libtpu",
            "--output-file",
            exported,
        ],
        cwd=repo,
    )
    packages = validate_exported_requirements(exported)
    if venv.exists():
        shutil.rmtree(venv)
    _run(
        [
            uv,
            "venv",
            "--python",
            sys.executable,
            "--system-site-packages",
            venv,
        ],
        cwd=repo,
    )
    runtime_python = venv / "bin" / "python"
    if not runtime_python.is_file():
        raise KaggleUvEnvironmentError(f"uv did not create {runtime_python}")
    _run(
        [
            uv,
            "pip",
            "install",
            "--python",
            runtime_python,
            "--no-deps",
            "-r",
            exported,
        ],
        cwd=repo,
    )
    _run([uv, "pip", "check", "--python", runtime_python], cwd=repo)

    provider_after = _distribution_versions(PROVIDER_PACKAGES)
    if provider_before != provider_after:
        raise KaggleUvEnvironmentError(
            f"provider-managed stack changed: {provider_before} -> {provider_after}"
        )
    runtime = _attest_runtime(runtime_python, repo)
    if runtime.get("jax_device_count") != provider_constraints["device_count"]:
        raise KaggleUvEnvironmentError(
            "locked runtime did not preserve the TPU device topology: "
            f"{runtime.get('jax_device_count')} != "
            f"{provider_constraints['device_count']}"
        )
    if runtime.get("jax_platforms") != ["tpu"]:
        raise KaggleUvEnvironmentError(
            f"locked runtime is not TPU-backed: {runtime.get('jax_platforms')}"
        )
    for name in ("jax", "jaxlib"):
        if runtime["versions"].get(name) != provider_before[name]:
            raise KaggleUvEnvironmentError(
                f"runtime {name} does not match provider version: "
                f"{runtime['versions'].get(name)} != {provider_before[name]}"
            )
        module_path = Path(runtime[f"{name}_file"]).resolve()
        if venv == module_path or venv in module_path.parents:
            raise KaggleUvEnvironmentError(
                f"provider-owned {name} was installed inside the uv venv: {module_path}"
            )

    summary = {
        "contract_version": 1,
        "status": "passed",
        "environment": "uv-system-site-packages",
        "uv_version": UV_VERSION,
        "lock_project": str(LOCK_PROJECT_RELATIVE),
        "lock_sha256": _sha256(lock_file),
        "project_sha256": _sha256(project_file),
        "provider_constraints_sha256": _sha256(provider_file),
        "provider_constraints": provider_constraints,
        "export_sha256": _sha256(exported),
        "provider_packages_excluded": list(PROVIDER_PACKAGES),
        "provider_before": provider_before,
        "provider_after": provider_after,
        "resolved_package_count": len(packages),
        "runtime_python": str(runtime_python),
        "runtime": runtime,
        "pip_check": "passed",
    }
    output = working_dir / "locked_environment_summary.json"
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("VDT_LOCKED_ENVIRONMENT_PROVENANCE " + json.dumps(summary, sort_keys=True))
    return summary


def runtime_subprocess_environment(
    repo: str | Path, summary: dict[str, Any]
) -> dict[str, str]:
    """Build the environment for a child launched with the locked interpreter."""

    runtime_python = Path(summary["runtime_python"])
    return {
        **os.environ,
        "PYTHONPATH": str(Path(repo).resolve()),
        "VIRTUAL_ENV": str(runtime_python.parent.parent),
        "PATH": str(runtime_python.parent) + os.pathsep + os.environ.get("PATH", ""),
    }
