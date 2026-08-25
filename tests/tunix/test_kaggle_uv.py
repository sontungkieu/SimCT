from __future__ import annotations

import json
import tomllib
from pathlib import Path

import pytest

from vdt_tunix.kaggle_uv import (
    LOCK_PROJECT_RELATIVE,
    PROVIDER_PACKAGES,
    TUNIX_COMMIT,
    UV_VERSION,
    KaggleUvEnvironmentError,
    load_provider_constraints,
    runtime_subprocess_environment,
    validate_exported_requirements,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_kaggle_tpu_uv_project_and_lock_are_pinned():
    project_root = REPO_ROOT / LOCK_PROJECT_RELATIVE
    project = tomllib.loads(
        (project_root / "pyproject.toml").read_text(encoding="utf-8")
    )
    lock = tomllib.loads((project_root / "uv.lock").read_text(encoding="utf-8"))
    provider = json.loads(
        (project_root / "provider-constraints.json").read_text(encoding="utf-8")
    )

    assert project["project"]["requires-python"] == ">=3.12,<3.13"
    assert project["tool"]["uv"]["package"] is False
    assert project["tool"]["uv"]["environments"] == ["sys_platform == 'linux'"]
    assert lock["requires-python"] == "==3.12.*"
    assert provider == {
        "accelerator": "TpuV5E8",
        "contract_version": 1,
        "device_count": 8,
        "jax": "0.10.2",
        "jaxlib": "0.10.2",
        "python": "3.12.13",
    }

    packages = {item["name"]: item for item in lock["package"]}
    assert packages["wandb"]["version"] == "0.19.11"
    assert packages["google-tunix"]["source"]["git"].endswith(TUNIX_COMMIT)
    for provider in PROVIDER_PACKAGES:
        assert provider in packages

    readable_inputs = {
        line.strip()
        for line in (REPO_ROOT / "requirements-tpu.txt")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    assert readable_inputs == set(project["project"]["dependencies"])


def test_export_validation_accepts_locked_userspace_packages(tmp_path: Path):
    exported = tmp_path / "requirements.lock"
    exported.write_text(
        "wandb==0.19.11 ; sys_platform == 'linux'\n"
        "google-tunix @ git+https://example.invalid/tunix.git@abc\n",
        encoding="utf-8",
    )

    assert validate_exported_requirements(exported) == (
        "google-tunix",
        "wandb",
    )


@pytest.mark.parametrize("provider", PROVIDER_PACKAGES)
def test_export_validation_rejects_provider_owned_packages(
    tmp_path: Path, provider: str
):
    exported = tmp_path / "requirements.lock"
    exported.write_text(f"{provider}==1.2.3\n", encoding="utf-8")

    with pytest.raises(KaggleUvEnvironmentError, match="provider-owned"):
        validate_exported_requirements(exported)


def test_runtime_subprocess_environment_selects_locked_interpreter(tmp_path: Path):
    runtime_python = tmp_path / ".venv" / "bin" / "python"
    summary = {"runtime_python": str(runtime_python)}

    environment = runtime_subprocess_environment(REPO_ROOT, summary)

    assert environment["VIRTUAL_ENV"] == str(runtime_python.parent.parent)
    assert environment["PATH"].split(":", 1)[0] == str(runtime_python.parent)
    assert environment["PYTHONPATH"] == str(REPO_ROOT)
    assert UV_VERSION == "0.10.2"


def test_provider_contract_loader_requires_exact_versions(tmp_path: Path):
    contract = tmp_path / "provider.json"
    contract.write_text(
        json.dumps(
            {
                "accelerator": "TpuV5E8",
                "contract_version": 1,
                "device_count": 8,
                "jax": "0.10",
                "jaxlib": "0.10.2",
                "python": "3.12.13",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(KaggleUvEnvironmentError, match="exact three-part"):
        load_provider_constraints(contract)


def test_provider_contract_loader_rejects_extra_keys(tmp_path: Path):
    contract = tmp_path / "provider.json"
    payload = json.loads(
        (REPO_ROOT / LOCK_PROJECT_RELATIVE / "provider-constraints.json").read_text(
            encoding="utf-8"
        )
    )
    payload["unlocked"] = True
    contract.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(KaggleUvEnvironmentError, match="exactly"):
        load_provider_constraints(contract)
