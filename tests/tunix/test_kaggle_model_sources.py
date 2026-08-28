from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import zipfile

import pytest

from vdt_tunix.kaggle_model_sources import (
    KaggleModelSourceError,
    attach_model_sources,
    bind_runtime_model_mount,
    model_source_mount,
    resolve_model_source_mount,
    verify_attached_model_sources,
    render_canary_notebook,
    render_model_source_mount_probe_notebook,
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


def test_resolve_model_source_mount_uses_exact_versioned_handle(tmp_path: Path):
    mounted = tmp_path / "attached-model"
    mounted.mkdir()
    observed = []

    resolved = resolve_model_source_mount(
        STUDENT,
        model_download=lambda handle: observed.append(handle) or str(mounted),
    )

    assert resolved == mounted
    assert observed == [STUDENT]


def test_resolve_model_source_mount_rejects_missing_download(tmp_path: Path):
    with pytest.raises(KaggleModelSourceError, match="missing directory"):
        resolve_model_source_mount(
            STUDENT,
            model_download=lambda handle: tmp_path / "missing",
        )


def test_bind_runtime_model_mount_preserves_tokenizer_asset_semantics(
    tmp_path: Path,
):
    mount = tmp_path / "gemma"
    mount.mkdir()
    tokenizer = mount / "tokenizer.model"
    tokenizer.write_text("tokenizer", encoding="utf-8")
    section = {
        "model_path": "/old/model",
        "maxtext_checkpoint_uri": "/old/model",
        "tokenizer_path": "/old/model/tokenizer.model",
    }

    provenance = bind_runtime_model_mount(section, mount)

    assert section["model_path"] == str(mount)
    assert section["maxtext_checkpoint_uri"] == str(mount)
    assert section["tokenizer_path"] == str(tokenizer)
    assert provenance == section

    root_tokenizer_section = {
        "model_path": "/old/qwen",
        "maxtext_checkpoint_uri": "/old/qwen",
        "tokenizer_path": "/old/qwen",
    }
    bind_runtime_model_mount(root_tokenizer_section, mount)
    assert root_tokenizer_section["tokenizer_path"] == str(mount)


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


def test_verify_attached_model_sources_fails_before_noninteractive_submit(
    tmp_path: Path,
):
    metadata = tmp_path / "kernel-metadata.json"
    manifest = tmp_path / "stage_package_manifest.json"
    metadata.write_text(json.dumps({"id": "owner/slug"}), encoding="utf-8")
    manifest.write_text(
        json.dumps({"fingerprints": {"metadata": {}}}), encoding="utf-8"
    )

    with pytest.raises(
        KaggleModelSourceError, match="not statically attached"
    ):
        verify_attached_model_sources(metadata, manifest, [STUDENT])

    attach_model_sources(metadata, manifest, [STUDENT])
    report = verify_attached_model_sources(metadata, manifest, [STUDENT])
    assert report["verified"] is True
    assert report["model_sources"] == [STUDENT]


def test_rendered_model_source_mount_probe_is_bounded_and_operational():
    notebook = render_model_source_mount_probe_notebook(
        model_sources=[STUDENT, TEACHER]
    )
    source = "".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )
    assert len(notebook["cells"]) == 2
    assert [cell["id"] for cell in notebook["cells"]] == [
        "model-source-probe-intro",
        "model-source-probe",
    ]
    assert STUDENT in source
    assert TEACHER in source
    assert "kagglehub.model_download(source)" in source
    assert "top_level_head" in source
    assert '"scientific_evidence": False' in source
    assert "model tensors are loaded" in source


def test_rendered_model_source_mount_probe_rejects_duplicates():
    with pytest.raises(KaggleModelSourceError, match="unique"):
        render_model_source_mount_probe_notebook(
            model_sources=[STUDENT, STUDENT]
        )


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
    assert "resolve_model_source_mount(STUDENT_SOURCE)" in source
    assert "resolve_model_source_mount(TEACHER_SOURCE)" in source
    assert "bind_runtime_model_mount" in source
    assert '"--config", str(RUNTIME_CONFIG)' in source
    assert "bootstrap_locked_kaggle_environment" in source
    assert "runtime_subprocess_environment" in source
    assert "str(RUNTIME_PYTHON)" in source
    assert "RUNTIME_SUBPROCESS_ENV" in source
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
    locked = mounted / "environments" / "kaggle-tpu"
    locked.mkdir(parents=True)
    (locked / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    (locked / "provider-constraints.json").write_text("{}\n", encoding="utf-8")
    (locked / "uv.lock").write_text("version = 1\n", encoding="utf-8")
    (mounted / "vdt_tunix" / "kaggle_uv.py").write_text(
        "# locked environment bootstrap\n", encoding="utf-8"
    )
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
                "warm_start_kernel_version": 1,
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
    assert len(notebook["cells"]) == 7
    assert REPO_DATASET in source
    assert "testowner/public-substitute-v1" in source
    assert STUDENT in source
    assert TEACHER in source
    assert "resolve_model_source_mount(STUDENT_SOURCE)" in source
    assert "resolve_model_source_mount(TEACHER_SOURCE)" in source
    assert "bind_runtime_model_mount" in source
    assert "bootstrap_locked_kaggle_environment" in source
    assert "runtime_subprocess_environment" in source
    assert "str(RUNTIME_PYTHON)" in source
    assert "RUNTIME_SUBPROCESS_ENV" in source
    assert "scientific_evidence" in source
    assert "shared downstream one-seed evaluation contract" in source
    secret_cells = [
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if "__KJO_SECRET_WANDB_API_KEY__" in "".join(cell.get("source", []))
    ]
    assert secret_cells == [
        'import os\n\nos.environ["WANDB_API_KEY"] = '
        '"__KJO_SECRET_WANDB_API_KEY__"\n'
    ]
    assert '__KJO_SECRET_WANDB_API_KEY__' in source
    assert 'WANDB_PROJECT' in source
    assert 'WANDB_RUN_GROUP' in source
    assert '"VDT_REQUIRE_WANDB", "1"' in source
    assert "unsafe archive member" in source
    assert f"if {phase!r} != \"sft\"" not in source
    assert '"start_step": 0' in source
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"<cell-{index}>", "exec")
    if phase == "sft":
        assert "kaggle_v5e8_sft.py" in source
        assert "WARM_START_KERNEL_SOURCE = None" in source
        assert "expected_algorithm = 'simct'" in source
        assert '"phase": \'sft_training\'' in source
        assert '"objective":' not in source
        assert 'payload.get("initialization")' not in source
    else:
        assert "kaggle_v5e8_train.py" in source
        assert "testowner/public-sft-screen-v1" in source
        assert "WARM_START_KERNEL_VERSION = 1" in source
        assert "kagglehub.notebook_output_download(handle)" in source
        assert f"expected_algorithm = {phase!r}" in source
        assert f'"phase": {f"{phase}_training"!r}' in source
        assert f'"objective": {phase!r}' in source
        assert 'payload.get("initialization") != "warm_start"' in source
        assert '"initialization"' in source


def test_tpu_requirements_pin_wandb_observability_client():
    requirements = (
        Path(__file__).resolve().parents[2] / "requirements-tpu.txt"
    ).read_text(encoding="utf-8").splitlines()

    assert [line for line in requirements if line.startswith("wandb")] == [
        "wandb==0.26.1"
    ]


def test_training_renderer_materializes_explicit_seed_runtime_config():
    notebook = render_training_notebook(
        phase="simct",
        config_relative_path=(
            "configs/reproduction/qwen25_7b_to_gemma2_2b_public_simct_screen.json"
        ),
        repo_dataset_source=REPO_DATASET,
        training_dataset_source="testowner/public-substitute-v1",
        training_manifest_relative_path="opd/manifest.json",
        student_model_source=STUDENT,
        teacher_model_source=TEACHER,
        expected_run_id="vdt-public-simct-seed43",
        training_seed=43,
        wandb_group="public-substitute-multiseed",
        warm_start_kernel_source="testowner/public-sft-seed43",
        warm_start_kernel_version=1,
        warm_start_relative_path="vdt_public_sft_seed43/checkpoints",
    )
    source = "".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )
    assert 'config["run_id"] = \'vdt-public-simct-seed43\'' in source
    assert 'config["training"]["seed"] = 43' in source
    assert 'vdt_public_simct_seed43' not in source
    assert 'config["run_id"].replace("-", "_")' in source
    assert "'public-substitute-multiseed'" in source


def test_training_renderer_accepts_resource_canary_source_run_id():
    notebook = render_training_notebook(
        phase="simple_opd",
        config_relative_path="configs/performance/paper4k-fsdp8-b1.json",
        repo_dataset_source=REPO_DATASET,
        training_dataset_source="testowner/resource-probe-paper4k-v1",
        training_manifest_relative_path="manifest.json",
        student_model_source=STUDENT,
        teacher_model_source=TEACHER,
        source_run_id="vdt-resource-simple_opd-paper4k-fsdp8-b1",
        expected_run_id="vdt-resource-simple_opd-paper4k-fsdp8-b1-owner",
        warm_start_kernel_source="testowner/public-sft-seed43",
        warm_start_kernel_version=1,
        warm_start_relative_path="checkpoints",
    )
    source = "".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )
    assert "source_run_id = 'vdt-resource-simple_opd-paper4k-fsdp8-b1'" in source
    assert (
        "config[\"run_id\"] = "
        "'vdt-resource-simple_opd-paper4k-fsdp8-b1-owner'"
    ) in source


def test_training_renderer_can_enable_one_profile_step():
    notebook = render_training_notebook(
        phase="simple_opd",
        config_relative_path="configs/simple-opd.json",
        repo_dataset_source=REPO_DATASET,
        training_dataset_source="testowner/public-substitute-v1",
        training_manifest_relative_path="opd/manifest.json",
        student_model_source=STUDENT,
        teacher_model_source=TEACHER,
        warm_start_kernel_source="testowner/public-sft-seed43",
        warm_start_kernel_version=1,
        warm_start_relative_path="checkpoints",
        profile_step=2,
    )
    source = "".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )
    assert '"--profile-dir", str(WORK / "jax-profile")' in source
    assert '"--profile-step", str(2)' in source


def test_training_renderer_can_use_remote_teacher_without_teacher_model_mount():
    notebook = render_training_notebook(
        phase="simct",
        config_relative_path=(
            "configs/performance/real-public-simct-student-dynamic.json"
        ),
        repo_dataset_source=REPO_DATASET,
        training_dataset_source="testowner/public-substitute-v1",
        training_manifest_relative_path="opd/manifest.json",
        student_model_source=STUDENT,
        teacher_model_source=TEACHER,
        warm_start_kernel_source="testowner/public-sft-seed43",
        warm_start_kernel_version=1,
        warm_start_relative_path="checkpoints",
        remote_teacher_profile_dataset_source=(
            "testowner/remote-teacher-profile-v1"
        ),
        remote_teacher_profile_relative_path=".",
        remote_teacher_tokenizer_relative_path="tokenizer",
        remote_teacher_timeout_s=600.0,
        remote_teacher_max_parallel_requests=2,
    )
    source = "".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )
    assert len(notebook["cells"]) == 8
    assert "testowner/remote-teacher-profile-v1" in source
    assert "resolve_model_source_mount(STUDENT_SOURCE)" in source
    assert "resolve_model_source_mount(TEACHER_SOURCE)" not in source
    assert '"mode": "remote_vllm_exact"' in source
    assert "__KJO_SECRET_VDT_REMOTE_TEACHER_URL__" in source
    assert "__KJO_SECRET_VDT_REMOTE_TEACHER_TOKEN__" in source
    assert 'REMOTE_TEACHER_SECRET_DIR.mkdir(mode=0o700' in source
    assert 'REMOTE_TEACHER_TOKEN_FILE.chmod(0o600)' in source
    assert 'os.environ["VDT_REMOTE_TEACHER_MAX_PARALLEL"] = \'2\'' in source
    assert "required_remote_teacher_files" in source
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"<remote-cell-{index}>", "exec")


def test_training_renderer_rejects_partial_remote_teacher_contract():
    with pytest.raises(KaggleModelSourceError, match="must be provided together"):
        render_training_notebook(
            phase="simct",
            config_relative_path="configs/simple-opd.json",
            repo_dataset_source=REPO_DATASET,
            training_dataset_source="testowner/public-substitute-v1",
            training_manifest_relative_path="opd/manifest.json",
            student_model_source=STUDENT,
            teacher_model_source=TEACHER,
            warm_start_kernel_source="testowner/public-sft-seed43",
            warm_start_kernel_version=1,
            warm_start_relative_path="checkpoints",
            remote_teacher_profile_dataset_source=(
                "testowner/remote-teacher-profile-v1"
            ),
        )


def test_training_input_cell_resolves_remote_teacher_profile_dataset(
    tmp_path, monkeypatch
):
    notebook = render_training_notebook(
        phase="simct",
        config_relative_path=(
            "configs/performance/real-public-simct-student-dynamic.json"
        ),
        repo_dataset_source=REPO_DATASET,
        training_dataset_source="testowner/public-substitute-v1",
        training_manifest_relative_path="opd/manifest.json",
        student_model_source=STUDENT,
        teacher_model_source=TEACHER,
        warm_start_kernel_source="testowner/public-sft-seed43",
        warm_start_kernel_version=1,
        warm_start_relative_path="checkpoints",
        remote_teacher_profile_dataset_source=(
            "testowner/remote-teacher-profile-v1"
        ),
        remote_teacher_profile_relative_path=".",
        remote_teacher_tokenizer_relative_path="tokenizer",
    )
    source = next(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if "TRAINING_DATASET_SOURCE" in "".join(cell.get("source", []))
    )
    source = source.replace(
        'Path("/kaggle/working/vdt_runtime_inputs")',
        f"Path({str(tmp_path / 'runtime-inputs')!r})",
    )
    input_root = tmp_path / "input"
    training_root = (
        input_root
        / "datasets"
        / "testowner"
        / "public-substitute-v1"
        / "versions"
        / "1"
    )
    (training_root / "opd").mkdir(parents=True)
    (training_root / "opd" / "manifest.json").write_text(
        "{}\n", encoding="utf-8"
    )
    warm_root = (
        input_root
        / "notebooks"
        / "testowner"
        / "public-sft-seed43"
        / "versions"
        / "1"
        / "checkpoints"
    )
    warm_root.mkdir(parents=True)
    profile_root = (
        input_root
        / "datasets"
        / "testowner"
        / "remote-teacher-profile-v1"
        / "versions"
        / "1"
    )
    tokenizer_root = profile_root / "tokenizer"
    tokenizer_root.mkdir(parents=True)
    for name in (
        "teacher_overlap_lm_head.manifest.json",
        "teacher_ids.i32le",
        "teacher_overlap_lm_head.bf16le",
    ):
        (profile_root / name).write_bytes(b"fixture")
    for name in ("tokenizer.json", "tokenizer_config.json"):
        (tokenizer_root / name).write_text("{}\n", encoding="utf-8")
    monkeypatch.setenv("KJO_KAGGLE_INPUT_ROOT", str(input_root))
    namespace = {}
    exec(compile(source, "<remote-training-inputs>", "exec"), namespace)
    assert namespace["REMOTE_TEACHER_PROFILE_ROOT"] == profile_root
    assert namespace["REMOTE_TEACHER_TOKENIZER_ROOT"] == tokenizer_root


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
            warm_start_kernel_version=1,
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
    runtime_inputs = tmp_path / "runtime-inputs"
    source = source.replace(
        'Path("/kaggle/working/vdt_runtime_inputs")',
        f"Path({str(runtime_inputs)!r})",
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
    if phase != "sft":
        warm_root = (
            input_root
            / "notebooks"
            / "testowner"
            / "public-sft-screen-v1"
        )
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


def test_training_input_cell_dynamically_attaches_versioned_notebook_output(
    tmp_path, monkeypatch
):
    notebook = _render_public_training("simct")
    source = next(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if "TRAINING_DATASET_SOURCE" in "".join(cell.get("source", []))
    )
    runtime_inputs = tmp_path / "runtime-inputs"
    source = source.replace(
        'Path("/kaggle/working/vdt_runtime_inputs")',
        f"Path({str(runtime_inputs)!r})",
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
    with zipfile.ZipFile(dataset_root / "opd.zip", "w") as archive:
        archive.writestr("manifest.json", "{}\n")
    attached = tmp_path / "runtime-attached-output"
    checkpoint = attached / "vdt_public_sft_screen" / "checkpoints"
    checkpoint.mkdir(parents=True)
    (checkpoint / "latest.json").write_text("{}\n", encoding="utf-8")
    observed = []
    monkeypatch.setitem(
        sys.modules,
        "kagglehub",
        SimpleNamespace(
            notebook_output_download=lambda handle: (
                observed.append(handle) or str(attached)
            )
        ),
    )
    monkeypatch.setenv("KJO_KAGGLE_INPUT_ROOT", str(input_root))
    namespace = {}
    exec(compile(source, "<dynamic-notebook-output>", "exec"), namespace)

    assert namespace["WARM_START_ROOT"] == checkpoint
    assert observed == ["testowner/public-sft-screen-v1/versions/1"]
