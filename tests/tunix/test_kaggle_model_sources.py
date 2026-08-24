from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile

import pytest

from vdt_tunix.kaggle_model_sources import (
    KaggleModelSourceError,
    attach_model_sources,
    model_source_mount,
    render_canary_notebook,
    render_training_notebook,
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
    assert 'importlib.metadata.version("huggingface-hub")' in source
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


def _render_public_training(phase: str):
    kwargs = {
        "phase": phase,
        "config_relative_path": (
            f"configs/reproduction/qwen25_7b_to_gemma2_2b_public_{phase}_screen.json"
        ),
        "repo_dataset_source": REPO_DATASET,
        "training_dataset_source": "testowner/public-substitute-v1",
        "training_manifest_relative_path": (
            "sft/manifest.json" if phase == "sft" else "opd/manifest.json"
        ),
        "student_model_source": STUDENT,
        "teacher_model_source": TEACHER,
    }
    if phase != "sft":
        kwargs.update(
            {
                "warm_start_kernel_source": "testowner/public-sft-screen-v1",
                "warm_start_relative_path": (
                    "vdt_public_sft_screen/checkpoints"
                ),
            }
        )
    return render_training_notebook(**kwargs)


@pytest.mark.parametrize("phase", ["sft", "simple_opd", "simct"])
def test_rendered_training_notebook_is_pinned_and_syntax_valid(phase):
    notebook = _render_public_training(phase)
    source = "".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )
    assert len(notebook["cells"]) == 6
    assert REPO_DATASET in source
    assert "testowner/public-substitute-v1" in source
    assert STUDENT in source
    assert TEACHER in source
    assert "--no-deps" in source
    assert 'importlib.metadata.version("huggingface-hub")' in source
    assert "provider-managed JAX stack changed" in source
    assert "scientific_evidence" in source
    assert "shared downstream one-seed evaluation contract" in source
    assert "unsafe archive member" in source
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"<cell-{index}>", "exec")
    if phase == "sft":
        assert "kaggle_v5e8_sft.py" in source
        assert "WARM_START_KERNEL_SOURCE = None" in source
    else:
        assert "kaggle_v5e8_train.py" in source
        assert "testowner/public-sft-screen-v1" in source
        assert '"initialization"' in source


def test_training_renderer_rejects_sft_warm_start():
    with pytest.raises(KaggleModelSourceError, match="may not declare"):
        render_training_notebook(
            phase="sft",
            config_relative_path="configs/sft.json",
            repo_dataset_source=REPO_DATASET,
            training_dataset_source="testowner/public-substitute-v1",
            training_manifest_relative_path="sft/manifest.json",
            student_model_source=STUDENT,
            teacher_model_source=TEACHER,
            warm_start_kernel_source="testowner/source",
            warm_start_relative_path="checkpoints",
        )


def test_training_renderer_requires_opd_warm_start():
    with pytest.raises(KaggleModelSourceError, match="require a completed SFT"):
        render_training_notebook(
            phase="simct",
            config_relative_path="configs/simct.json",
            repo_dataset_source=REPO_DATASET,
            training_dataset_source="testowner/public-substitute-v1",
            training_manifest_relative_path="opd/manifest.json",
            student_model_source=STUDENT,
            teacher_model_source=TEACHER,
        )


@pytest.mark.parametrize("phase", ["sft", "simct"])
def test_training_input_cell_resolves_zipped_dataset_and_kernel_source(
    tmp_path, monkeypatch, phase
):
    notebook = _render_public_training(phase)
    source = next(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if "TRAINING_DATASET_SOURCE" in "".join(cell.get("source", []))
    )
    input_root = tmp_path / "input"
    dataset_root = (
        input_root
        / "datasets"
        / "testowner"
        / "public-substitute-v1"
        / "versions"
        / "1"
    )
    dataset_root.mkdir(parents=True)
    archive_name = "sft.zip" if phase == "sft" else "opd.zip"
    with zipfile.ZipFile(dataset_root / archive_name, "w") as archive:
        archive.writestr("manifest.json", "{}\n")
    runtime_inputs = tmp_path / "runtime-inputs"
    source = source.replace(
        'Path("/kaggle/working/vdt_runtime_inputs")',
        f"Path({str(runtime_inputs)!r})",
    )
    if phase != "sft":
        warm_root = input_root / "public-sft-screen-v1"
        checkpoint = warm_root / "vdt_public_sft_screen" / "checkpoints"
        checkpoint.mkdir(parents=True)
        (checkpoint / "latest.json").write_text("{}\n", encoding="utf-8")
    monkeypatch.setenv("KJO_KAGGLE_INPUT_ROOT", str(input_root))
    namespace = {}
    exec(compile(source, "<rendered-training-inputs>", "exec"), namespace)
    assert namespace["TRAINING_MANIFEST"].is_file()
    if phase == "sft":
        assert namespace["WARM_START_ROOT"] is None
    else:
        assert namespace["WARM_START_ROOT"] == checkpoint
