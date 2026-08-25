from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

import vdt_tunix.kaggle_package as package


REPO_ROOT = Path(__file__).resolve().parents[2]
STUDENT_REVISION = "1" * 40
TEACHER_REVISION = "2" * 40
TREE_SHA256 = "3" * 64


def _write(path: Path, content: str = "x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def package_fixture(tmp_path, config_payload):
    boundary = tmp_path / "d-drive"
    boundary.mkdir()
    run_id = "vdt-simct-v5e8-dry-run"
    config = copy.deepcopy(config_payload)
    config["run_id"] = run_id
    config["student"].update(
        {
            "model_revision": STUDENT_REVISION,
            "tokenizer_revision": STUDENT_REVISION,
        }
    )
    config["teacher"].update(
        {
            "model_revision": TEACHER_REVISION,
            "tokenizer_revision": TEACHER_REVISION,
        }
    )

    checkpoint_assets = {}
    for role in ("student", "teacher"):
        root = boundary / "datasets" / f"{role}-checkpoint"
        _write(root / "items" / "weights.index", role)
        manifest = _write(root / "checkpoint_manifest.json", f'{{"role":"{role}"}}\n')
        config[role]["maxtext_checkpoint_uri"] = str(root / "items")
        checkpoint_assets[role] = {
            "local_root": str(root),
            "dataset_source": f"testowner/vdt-{role}-checkpoint",
            "checkpoint_subpath": "items",
            "manifest_subpath": "checkpoint_manifest.json",
            "manifest_sha256": _sha256(manifest),
        }

    config_path = _write(
        boundary / "specs" / "canary_config.json",
        json.dumps(config, indent=2),
    )
    tokenizer_root = boundary / "datasets" / "tokenizers"
    hf_home = tokenizer_root / "hf-home" / "hub"
    for model, revision in (
        (config["student"], STUDENT_REVISION),
        (config["teacher"], TEACHER_REVISION),
    ):
        model_dir = f"models--{model['tokenizer_id'].replace('/', '--')}"
        _write(hf_home / model_dir / "snapshots" / revision / "tokenizer.json", "{}")

    source_root = boundary / "source" / "repo-dataset"
    repo = source_root / "dataset" / "repo"
    for relative in package._REQUIRED_SOURCE_FILES:
        content = (
            f"google-tunix @ git+https://example.invalid/tunix.git@{package.TUNIX_COMMIT}\n"
            if relative == "requirements-tpu.txt"
            else "# source fixture\n"
        )
        _write(repo / relative, content)
    source_manifest = _write(
        source_root / "repo_dataset_manifest.json",
        json.dumps(
            {
                "ok": True,
                "dataset_source": "testowner/vdt-source",
                "git": {"commit": package.SIMCT_COMMIT, "dirty": True},
                "content_manifest": {
                    "tree_sha256": TREE_SHA256,
                    "dataset_dir": str(source_root / "dataset"),
                },
            }
        ),
    )
    spec = {
        "contract_version": 1,
        "run_id": run_id,
        "provenance": {
            "upstream_simct_commit": package.SIMCT_COMMIT,
            "tunix_commit": package.TUNIX_COMMIT,
        },
        "kaggle": {
            "owner": "testowner",
            "slug": "vdt-simct-v5e8-dry-run",
            "title": "VDT SimCT v5e-8 dry run",
        },
        "canary_config": str(config_path),
        "source_snapshot_manifest": str(source_manifest),
        "tokenizer_cache": {
            "local_root": str(tokenizer_root),
            "dataset_source": "testowner/vdt-tokenizers",
            "hf_home_subpath": "hf-home",
        },
        "student_checkpoint": checkpoint_assets["student"],
        "teacher_checkpoint": checkpoint_assets["teacher"],
    }
    spec_path = _write(boundary / "specs" / "package.json", json.dumps(spec))
    return boundary, spec, spec_path


def test_valid_spec_pins_tpu_and_provenance(package_fixture):
    boundary, _, spec_path = package_fixture
    loaded = package.load_package_spec(spec_path, output_root=boundary)
    summary = package.safe_summary(loaded)

    assert summary["kernel_id"] == "testowner/vdt-simct-v5e8-dry-run"
    assert summary["accelerator"] == "TpuV5E8"
    assert summary["accelerator_label"] == "TPU VM v5-8"
    assert summary["expected_device_count"] == 8
    assert summary["upstream_simct_commit"] == package.SIMCT_COMMIT
    assert summary["tunix_commit"] == package.TUNIX_COMMIT
    assert summary["uv_project_sha256"] == loaded.uv_project_sha256
    assert summary["provider_constraints_sha256"] == (
        loaded.provider_constraints_sha256
    )
    assert summary["uv_lock_sha256"] == loaded.uv_lock_sha256
    assert summary["remote_submit_performed"] is False


def test_spec_rejects_placeholder_owner_before_assets(package_fixture):
    boundary, spec, spec_path = package_fixture
    spec["kaggle"]["owner"] = "replace-with-owner"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")

    with pytest.raises(package.KagglePackageError, match="placeholder"):
        package.load_package_spec(spec_path, output_root=boundary)


def test_spec_does_not_hard_code_transient_account_eligibility(package_fixture):
    boundary, spec, spec_path = package_fixture
    spec["kaggle"]["owner"] = "kieutung"
    spec["tokenizer_cache"]["dataset_source"] = "kieutung/vdt-tokenizers"
    for role in ("student", "teacher"):
        spec[f"{role}_checkpoint"]["dataset_source"] = (
            f"kieutung/vdt-{role}-checkpoint"
        )
    source_manifest = Path(spec["source_snapshot_manifest"])
    source = json.loads(source_manifest.read_text(encoding="utf-8"))
    source["dataset_source"] = "kieutung/vdt-source"
    source_manifest.write_text(json.dumps(source), encoding="utf-8")
    spec_path.write_text(json.dumps(spec), encoding="utf-8")

    loaded = package.load_package_spec(spec_path, output_root=boundary)

    assert loaded.owner == "kieutung"


def test_spec_rejects_checkpoint_manifest_hash_mismatch(package_fixture):
    boundary, spec, spec_path = package_fixture
    spec["student_checkpoint"]["manifest_sha256"] = "0" * 64
    spec_path.write_text(json.dumps(spec), encoding="utf-8")

    with pytest.raises(package.KagglePackageError, match="manifest SHA-256 mismatch"):
        package.load_package_spec(spec_path, output_root=boundary)


def test_spec_rejects_missing_tokenizer_revision_cache(package_fixture):
    boundary, spec, spec_path = package_fixture
    spec["tokenizer_cache"]["hf_home_subpath"] = "missing-hf-home"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")

    with pytest.raises(package.KagglePackageError, match="tokenizer HF_HOME"):
        package.load_package_spec(spec_path, output_root=boundary)


def test_spec_rejects_missing_source_snapshot(package_fixture):
    boundary, spec, spec_path = package_fixture
    spec["source_snapshot_manifest"] = str(boundary / "missing-source.json")
    spec_path.write_text(json.dumps(spec), encoding="utf-8")

    with pytest.raises(package.KagglePackageError, match="source snapshot"):
        package.load_package_spec(spec_path, output_root=boundary)


def test_spec_rejects_wrong_source_commit(package_fixture):
    boundary, spec, spec_path = package_fixture
    manifest_path = Path(spec["source_snapshot_manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["git"]["commit"] = "0" * 40
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(package.KagglePackageError, match="wrong SimCT commit"):
        package.load_package_spec(spec_path, output_root=boundary)


def test_rendered_notebook_is_bounded_and_records_evidence(package_fixture):
    boundary, _, spec_path = package_fixture
    loaded = package.load_package_spec(spec_path, output_root=boundary)
    notebook = package.render_source_notebook(loaded)
    source = "".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )

    assert len(notebook["cells"]) == 4
    assert package.SIMCT_COMMIT in source
    assert package.TUNIX_COMMIT in source
    assert loaded.uv_project_sha256 in source
    assert loaded.provider_constraints_sha256 in source
    assert loaded.uv_lock_sha256 in source
    assert "bootstrap_locked_kaggle_environment" in source
    assert "str(RUNTIME_PYTHON)" in source
    assert "env=RUNTIME_SUBPROCESS_ENV" in source
    assert "kaggle_v5e8_canary.py" in source
    assert '"scientific_evidence": False' in source
    assert '"simct_update_executed": True' in source


def test_stage_materializes_metadata_manifest_and_future_command(
    package_fixture, monkeypatch
):
    boundary, _, spec_path = package_fixture
    output = boundary / "packages" / "dry-run"

    def fake_kjo(args, *, cli, cwd):
        assert args[0] == "stage-notebook-package"
        assert "TpuV5E8" in args
        assert "--require-accelerator-probe-contract" in args
        stage = cwd / "stage"
        stage.mkdir()
        (stage / "submitted_notebook.ipynb").write_text("{}", encoding="utf-8")
        (stage / "stage_package_manifest.json").write_text("{}", encoding="utf-8")
        (stage / "kernel-metadata.json").write_text(
            json.dumps(
                {
                    "id": "testowner/vdt-simct-v5e8-dry-run",
                    "machine_shape": "TpuV5E8",
                    "enable_gpu": False,
                    "enable_tpu": True,
                    "is_private": True,
                    "dataset_sources": [
                        "testowner/vdt-source",
                        "testowner/vdt-tokenizers",
                        "testowner/vdt-student-checkpoint",
                        "testowner/vdt-teacher-checkpoint",
                    ],
                }
            ),
            encoding="utf-8",
        )

    monkeypatch.setattr(package, "_run_kjo", fake_kjo)
    result = package.stage_package(
        spec_path,
        output,
        output_root=boundary,
        kjo_cli=REPO_ROOT / "scripts" / "tpu" / "stage_kaggle_v5e8.py",
    )

    manifest = json.loads(
        (result / "vdt_kaggle_package_manifest.json").read_text(encoding="utf-8")
    )
    command = (result / "future_submit_command.sh").read_text(encoding="utf-8")
    assert manifest["remote_submit_performed"] is False
    assert manifest["scientific_evidence"] is False
    assert manifest["accelerator"]["submit_shape"] == "TpuV5E8"
    assert manifest["provenance"]["uv_project_sha256"]
    assert manifest["provenance"]["provider_constraints_sha256"]
    assert manifest["provenance"]["uv_lock_sha256"]
    assert "submit-kernel" in command
    assert "--require-repo-dataset-push" in command
    assert "--reservation-token" in command
    assert "${RESERVATION_TOKEN:" in command


def test_cli_failure_path_imports_subprocess_at_module_scope():
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "tpu" / "stage_kaggle_v5e8.py"),
            "validate",
            "--spec",
            str(
                REPO_ROOT
                / "scripts"
                / "tpu"
                / "kaggle_v5e8_dry_run.example.json"
            ),
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 78
    payload = json.loads(result.stderr)
    assert payload["status"] == "blocked"
    assert payload["remote_submit_performed"] is False
    assert payload["error_type"] == "KagglePackageError"
    assert "NameError" not in result.stderr
