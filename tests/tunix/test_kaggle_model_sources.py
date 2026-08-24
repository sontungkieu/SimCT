from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from vdt_tunix.kaggle_model_sources import (
    KaggleModelSourceError,
    attach_model_sources,
    model_source_mount,
    render_canary_notebook,
)


STUDENT = "google/gemma-2/flax/gemma2-2b-it/1"
TEACHER = "qwen-lm/qwen2.5/transformers/7b-instruct/1"
REPO_DATASET = "testowner/simct-tunix-repro-src"


def test_model_source_mount_matches_kaggle_layout():
    assert str(model_source_mount(STUDENT)) == (
        "/kaggle/input/models/google/gemma-2/flax/gemma2-2b-it/1"
    )
    assert str(model_source_mount(TEACHER)) == (
        "/kaggle/input/models/qwen-lm/qwen2.5/transformers/7b-instruct/1"
    )


def test_attach_model_sources_refreshes_stage_fingerprint(tmp_path: Path):
    metadata = tmp_path / "kernel-metadata.json"
    manifest = tmp_path / "stage_package_manifest.json"
    metadata.write_text(json.dumps({"id": "owner/slug"}), encoding="utf-8")
    manifest.write_text(
        json.dumps({"ok": True, "fingerprints": {"metadata": {}}}),
        encoding="utf-8",
    )

    report = attach_model_sources(metadata, manifest, [STUDENT, TEACHER])

    observed_metadata = json.loads(metadata.read_text(encoding="utf-8"))
    observed_manifest = json.loads(manifest.read_text(encoding="utf-8"))
    expected_sha = hashlib.sha256(metadata.read_bytes()).hexdigest()
    assert observed_metadata["model_sources"] == [STUDENT, TEACHER]
    assert observed_manifest["model_sources"] == [STUDENT, TEACHER]
    assert observed_manifest["fingerprints"]["metadata"]["sha256"] == expected_sha
    assert report["metadata_fingerprint"]["sha256"] == expected_sha


def test_attach_model_sources_rejects_existing_drift(tmp_path: Path):
    metadata = tmp_path / "kernel-metadata.json"
    manifest = tmp_path / "stage_package_manifest.json"
    metadata.write_text(
        json.dumps({"model_sources": [STUDENT]}), encoding="utf-8"
    )
    manifest.write_text(
        json.dumps({"fingerprints": {"metadata": {}}}), encoding="utf-8"
    )
    with pytest.raises(KaggleModelSourceError, match="drifted"):
        attach_model_sources(metadata, manifest, [STUDENT, TEACHER])


def test_rendered_canary_is_pinned_bounded_and_preserves_jax():
    notebook = render_canary_notebook(
        config_relative_path=(
            "configs/reproduction/qwen25_7b_to_gemma2_2b_paper_canary.json"
        ),
        repo_dataset_source=REPO_DATASET,
        student_model_source=STUDENT,
        teacher_model_source=TEACHER,
    )
    source = "".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )
    assert len(notebook["cells"]) == 5
    assert REPO_DATASET in source
    assert 'input_root / "datasets"' in source
    assert "direct_root.is_dir()" in source
    assert "KJO_REPO_DATASET_COPY_SUMMARY" in source
    assert STUDENT in source
    assert TEACHER in source
    assert "--no-deps" in source
    assert "provider-managed JAX stack changed" in source
    assert "scientific_evidence" in source
    assert "shutil.rmtree(cache)" in source


def test_rendered_repo_copy_supports_direct_owner_slug_mount(
    tmp_path: Path, monkeypatch
):
    notebook = render_canary_notebook(
        config_relative_path=(
            "configs/reproduction/qwen25_7b_to_gemma2_2b_paper_canary.json"
        ),
        repo_dataset_source=REPO_DATASET,
        student_model_source=STUDENT,
        teacher_model_source=TEACHER,
    )
    copy_source = next(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if "KJO_REPO_DATASET_SOURCE" in "".join(cell.get("source", []))
    )
    input_root = tmp_path / "input"
    mounted = input_root / "datasets" / "testowner" / "simct-tunix-repro-src"
    (mounted / "vdt_tunix").mkdir(parents=True)
    config = (
        mounted
        / "configs"
        / "reproduction"
        / "qwen25_7b_to_gemma2_2b_paper_canary.json"
    )
    config.parent.mkdir(parents=True)
    config.write_text("{}", encoding="utf-8")
    (mounted / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    (mounted / "vdt_tunix" / "marker.py").write_text("VALUE = 1\n", encoding="utf-8")
    working = tmp_path / "working" / "repo"
    copy_source = copy_source.replace(
        'Path("/kaggle/working/repo")', f"Path({str(working)!r})"
    )
    monkeypatch.setenv("KJO_KAGGLE_INPUT_ROOT", str(input_root))
    exec(compile(copy_source, "<rendered-repo-copy>", "exec"), {})
    assert (working / "vdt_tunix" / "marker.py").is_file()


def test_render_rejects_config_escape():
    with pytest.raises(KaggleModelSourceError, match="inside repo"):
        render_canary_notebook(
            config_relative_path="../secret.json",
            repo_dataset_source=REPO_DATASET,
            student_model_source=STUDENT,
            teacher_model_source=TEACHER,
        )


def test_render_rejects_malformed_repo_dataset_source():
    with pytest.raises(KaggleModelSourceError, match="owner/slug"):
        render_canary_notebook(
            config_relative_path="configs/canary.json",
            repo_dataset_source="missing-owner",
            student_model_source=STUDENT,
            teacher_model_source=TEACHER,
        )
