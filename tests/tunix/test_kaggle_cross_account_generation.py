from __future__ import annotations

import json
from pathlib import Path

import pytest

from vdt_tunix.kaggle_cross_account_generation import (
    compose_cross_account_generation_notebook,
)
from vdt_tunix.kaggle_generation_sources import render_generation_notebook
from vdt_tunix.kaggle_model_sources import KaggleModelSourceError
from vdt_tunix.kaggle_model_sources import render_training_notebook


SOURCE_KERNEL = "sourceowner/checkpoint-v1"
RUNTIME_OWNER = "runtimeowner"
EVALUATION_SOURCE = "runtimeowner/evaluation-v1"
STUDENT = "google/gemma-2/flax/gemma2-2b-it/1"


def _render():
    return render_generation_notebook(
        variant="sft",
        training_config_relative_path=(
            "configs/reproduction/qwen25_7b_to_gemma2_2b_public_sft_screen.json"
        ),
        generation_protocol_relative_path=(
            "configs/evaluation/simct_paper_one_seed_generation.json"
        ),
        repo_dataset_source="runtimeowner/repo-v1",
        evaluation_dataset_source=EVALUATION_SOURCE,
        checkpoint_kernel_source=SOURCE_KERNEL,
        checkpoint_relative_path="vdt_public_sft_screen/checkpoints",
        student_model_source=STUDENT,
    )


def _download_source(config_dir: str, output_dir: str) -> str:
    return f'''SOURCE_OWNER = "sourceowner"
RUNTIME_OWNER = "runtimeowner"
KERNEL_ID = "{SOURCE_KERNEL}"
CONFIG_DIR = "{config_dir}"
OUTPUT_DIR = "{output_dir}"
REMOVED_AUTH_VARIABLES = ["KAGGLE_API_V1_TOKEN"]
print("KJO_CROSS_ACCOUNT_OUTPUT_SUMMARY")'''


def _compose(tmp_path: Path):
    config = tmp_path / "source-config"
    output = tmp_path / "overlay" / "kernels" / "sourceowner" / "checkpoint-v1"
    overlay = tmp_path / "overlay"
    return compose_cross_account_generation_notebook(
        base_notebook=_render(),
        cross_account_output_source=_download_source(str(config), str(output)),
        source_kernel_id=SOURCE_KERNEL,
        runtime_owner=RUNTIME_OWNER,
        evaluation_dataset_source=EVALUATION_SOURCE,
        source_config_dir=str(config),
        cross_account_output_dir=str(output),
        overlay_input_root=str(overlay),
    )


def test_composer_inserts_guarded_cells_before_input_resolution(tmp_path):
    notebook = _compose(tmp_path)
    assert len(notebook["cells"]) == 12
    sources = ["".join(cell.get("source", [])) for cell in notebook["cells"]]
    copy_index = next(i for i, source in enumerate(sources) if "KJO_REPO_DATASET_COPY_SUMMARY" in source)
    bootstrap_index = next(i for i, source in enumerate(sources) if "KJO_KAGGLE_CLI_BOOTSTRAP" in source)
    credential_secret_index = next(
        i for i, source in enumerate(sources)
        if "__KJO_SECRET_KAGGLE_SOURCE_KEY__" in source
    )
    credential_index = next(
        i for i, source in enumerate(sources)
        if "source_credential_path.write_text" in source
    )
    download_index = next(i for i, source in enumerate(sources) if "KJO_CROSS_ACCOUNT_OUTPUT_SUMMARY" in source)
    overlay_index = next(i for i, source in enumerate(sources) if "VDT_CROSS_ACCOUNT_INPUT_OVERLAY" in source)
    resolve_index = next(i for i, source in enumerate(sources) if "CHECKPOINT_KERNEL_SOURCE" in source and "EVALUATION_DATASET_SOURCE" in source)
    assert (
        copy_index
        < bootstrap_index
        < credential_secret_index
        < credential_index
        < download_index
        < overlay_index
        < resolve_index
    )
    assert "KJO_KAGGLE_CLI_VERSION = '2.2.3'" in sources[bootstrap_index]
    assert '"pip"' in sources[bootstrap_index]
    assert sources[credential_secret_index] == (
        "SOURCE_KAGGLE_KEY = '__KJO_SECRET_KAGGLE_SOURCE_KEY__'\n"
    )
    assert "__KJO_SECRET_KAGGLE_SOURCE_KEY__" not in sources[credential_index]
    assert "KAGGLE_API_V1_TOKEN" in sources[download_index]
    for index in (
        bootstrap_index,
        credential_secret_index,
        credential_index,
        download_index,
        overlay_index,
    ):
        compile(sources[index], f"<cross-account-cell-{index}>", "exec")


def test_credential_cell_fails_closed_without_injection(tmp_path):
    notebook = _compose(tmp_path)
    source = next(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if "source_credential_path.write_text" in "".join(cell.get("source", []))
    )
    with pytest.raises(RuntimeError, match="was not injected"):
        exec(  # noqa: S102
            compile(source, "<credential-cell>", "exec"),
            {"SOURCE_KAGGLE_KEY": "__KJO_SECRET_KAGGLE_SOURCE_KEY__"},
        )


def test_credential_cell_writes_only_the_explicit_source_owner(tmp_path):
    notebook = _compose(tmp_path)
    source = next(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if "source_credential_path.write_text" in "".join(cell.get("source", []))
    )
    exec(  # noqa: S102
        compile(source, "<credential-cell>", "exec"),
        {"SOURCE_KAGGLE_KEY": "fixture-key"},
    )
    payload = json.loads((tmp_path / "source-config" / "kaggle.json").read_text())
    assert payload == {"username": "sourceowner", "key": "fixture-key"}
    # NTFS-backed pytest temp roots may not preserve POSIX mode bits, so assert
    # the runtime hardening operation itself instead of the host mount result.
    assert "source_credential_path.chmod(0o600)" in source


def test_composer_rejects_drifted_kjo_download_cell(tmp_path):
    with pytest.raises(KaggleModelSourceError, match="missing literals"):
        compose_cross_account_generation_notebook(
            base_notebook=_render(),
            cross_account_output_source="print('not a KJO cell')",
            source_kernel_id=SOURCE_KERNEL,
            runtime_owner=RUNTIME_OWNER,
            evaluation_dataset_source=EVALUATION_SOURCE,
            source_config_dir=str(tmp_path / "config"),
            cross_account_output_dir=str(tmp_path / "output"),
            overlay_input_root=str(tmp_path / "overlay"),
        )


def test_composer_supports_opd_training_input_resolution(tmp_path):
    base = render_training_notebook(
        phase="simple_opd",
        config_relative_path="configs/performance/paper4k-fsdp8-b1.json",
        repo_dataset_source="runtimeowner/repo-v1",
        training_dataset_source=EVALUATION_SOURCE,
        training_manifest_relative_path="manifest.json",
        student_model_source=STUDENT,
        teacher_model_source="qwen-lm/qwen2.5/transformers/7b-instruct/1",
        source_run_id="vdt-resource-simple_opd-paper4k-fsdp8-b1",
        warm_start_kernel_source=SOURCE_KERNEL,
        warm_start_kernel_version=1,
        warm_start_relative_path="checkpoints",
    )
    output = (
        tmp_path
        / "overlay"
        / "kernels"
        / "sourceowner"
        / "checkpoint-v1"
        / "versions"
        / "1"
    )
    notebook = compose_cross_account_generation_notebook(
        base_notebook=base,
        cross_account_output_source=_download_source(
            str(tmp_path / "source-config"), str(output)
        ),
        source_kernel_id=SOURCE_KERNEL,
        runtime_owner=RUNTIME_OWNER,
        evaluation_dataset_source=EVALUATION_SOURCE,
        source_config_dir=str(tmp_path / "source-config"),
        cross_account_output_dir=str(output),
        overlay_input_root=str(tmp_path / "overlay"),
    )
    sources = ["".join(cell.get("source", [])) for cell in notebook["cells"]]
    overlay_index = next(
        i for i, source in enumerate(sources)
        if "VDT_CROSS_ACCOUNT_INPUT_OVERLAY" in source
    )
    resolve_index = next(
        i for i, source in enumerate(sources)
        if "TRAINING_DATASET_SOURCE" in source
        and "WARM_START_KERNEL_SOURCE" in source
    )
    assert overlay_index < resolve_index
