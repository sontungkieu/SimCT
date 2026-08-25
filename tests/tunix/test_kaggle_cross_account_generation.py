from __future__ import annotations

import json
from pathlib import Path

import pytest

from vdt_tunix.kaggle_cross_account_generation import (
    compose_cross_account_generation_notebook,
)
from vdt_tunix.kaggle_generation_sources import render_generation_notebook
from vdt_tunix.kaggle_model_sources import KaggleModelSourceError


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
    assert len(notebook["cells"]) == 9
    sources = ["".join(cell.get("source", [])) for cell in notebook["cells"]]
    copy_index = next(i for i, source in enumerate(sources) if "KJO_REPO_DATASET_COPY_SUMMARY" in source)
    credential_index = next(i for i, source in enumerate(sources) if "SOURCE_KAGGLE_KEY" in source)
    download_index = next(i for i, source in enumerate(sources) if "KJO_CROSS_ACCOUNT_OUTPUT_SUMMARY" in source)
    overlay_index = next(i for i, source in enumerate(sources) if "VDT_CROSS_ACCOUNT_INPUT_OVERLAY" in source)
    resolve_index = next(i for i, source in enumerate(sources) if "CHECKPOINT_KERNEL_SOURCE" in source and "EVALUATION_DATASET_SOURCE" in source)
    assert copy_index < credential_index < download_index < overlay_index < resolve_index
    assert "__KJO_SECRET_KAGGLE_SOURCE_KEY__" in sources[credential_index]
    assert "KAGGLE_API_V1_TOKEN" in sources[download_index]
    for index in (credential_index, download_index, overlay_index):
        compile(sources[index], f"<cross-account-cell-{index}>", "exec")


def test_credential_cell_fails_closed_without_injection(tmp_path):
    notebook = _compose(tmp_path)
    source = next(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if "SOURCE_KAGGLE_KEY" in "".join(cell.get("source", []))
    )
    with pytest.raises(RuntimeError, match="was not injected"):
        exec(compile(source, "<credential-cell>", "exec"), {})  # noqa: S102


def test_credential_cell_writes_only_the_explicit_source_owner(tmp_path):
    notebook = _compose(tmp_path)
    source = next(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if "SOURCE_KAGGLE_KEY" in "".join(cell.get("source", []))
    ).replace("__KJO_SECRET_KAGGLE_SOURCE_KEY__", "fixture-key")
    exec(compile(source, "<credential-cell>", "exec"), {})  # noqa: S102
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
