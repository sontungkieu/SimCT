"""Compose generation notebooks that consume a private checkpoint cross-account."""

from __future__ import annotations

import copy
import re
from pathlib import PurePosixPath
from typing import Any, Mapping

from vdt_tunix.kaggle_model_sources import (
    KaggleModelSourceError,
    _validate_dataset_source,
)


_SECRET_PLACEHOLDER_RE = re.compile(r"^__KJO_SECRET_[A-Z0-9_]+__$")
_KAGGLE_CLI_VERSION = "2.2.3"


def _code_cell(source: str, *, cell_id: str) -> dict[str, Any]:
    return {
        "cell_type": "code",
        "id": cell_id,
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source.splitlines()],
    }


def compose_cross_account_generation_notebook(
    *,
    base_notebook: Mapping[str, Any],
    cross_account_output_source: str,
    source_kernel_id: str,
    runtime_owner: str,
    evaluation_dataset_source: str,
    source_config_dir: str = "/tmp/.kaggle_source_owner",
    cross_account_output_dir: str,
    overlay_input_root: str = "/tmp/vdt_cross_account_inputs",
    source_key_placeholder: str = "__KJO_SECRET_KAGGLE_SOURCE_KEY__",
) -> dict[str, Any]:
    """Insert guarded credential, download, and input-overlay cells.

    The supplied download source must come from KJO's
    ``render-cross-account-output-cell`` command. The source owner's key remains
    a placeholder until the private staged notebook is injected immediately
    before submit.
    """

    source_owner, source_slug = _validate_dataset_source(
        source_kernel_id, "source_kernel_id"
    )
    evaluation_owner, evaluation_slug = _validate_dataset_source(
        evaluation_dataset_source, "evaluation_dataset_source"
    )
    if not runtime_owner or "/" in runtime_owner:
        raise KaggleModelSourceError("runtime_owner must be one Kaggle owner")
    if not _SECRET_PLACEHOLDER_RE.fullmatch(source_key_placeholder):
        raise KaggleModelSourceError(
            "source_key_placeholder must be a key-specific KJO secret sentinel"
        )

    config_path = PurePosixPath(source_config_dir)
    output_path = PurePosixPath(cross_account_output_dir)
    overlay_path = PurePosixPath(overlay_input_root)
    for value, name in (
        (config_path, "source_config_dir"),
        (output_path, "cross_account_output_dir"),
        (overlay_path, "overlay_input_root"),
    ):
        if not value.is_absolute() or ".." in value.parts:
            raise KaggleModelSourceError(f"{name} must be an absolute safe path")

    required_download_literals = (
        "KJO_CROSS_ACCOUNT_OUTPUT_SUMMARY",
        source_kernel_id,
        runtime_owner,
        str(config_path),
        str(output_path),
    )
    missing = [
        literal
        for literal in required_download_literals
        if literal not in cross_account_output_source
    ]
    if missing:
        raise KaggleModelSourceError(
            f"cross-account output source drifted; missing literals: {missing}"
        )

    notebook = copy.deepcopy(dict(base_notebook))
    cells = notebook.get("cells")
    if not isinstance(cells, list):
        raise KaggleModelSourceError("base_notebook.cells must be a list")
    copy_indexes = [
        index
        for index, cell in enumerate(cells)
        if "KJO_REPO_DATASET_COPY_SUMMARY" in "".join(cell.get("source", []))
    ]
    resolve_indexes = [
        index
        for index, cell in enumerate(cells)
        if "EVALUATION_DATASET_SOURCE" in "".join(cell.get("source", []))
        and "CHECKPOINT_KERNEL_SOURCE" in "".join(cell.get("source", []))
    ]
    if len(copy_indexes) != 1 or len(resolve_indexes) != 1:
        raise KaggleModelSourceError(
            "base generation notebook must contain exactly one repo-copy cell "
            "and one generation-input cell"
        )
    insert_at = resolve_indexes[0]
    if copy_indexes[0] >= insert_at:
        raise KaggleModelSourceError(
            "repo-copy cell must precede the generation-input cell"
        )

    credential_source = f'''from pathlib import Path
import json

SOURCE_KAGGLE_OWNER = {source_owner!r}
SOURCE_KAGGLE_KEY = {source_key_placeholder!r}
SOURCE_KAGGLE_CONFIG_DIR = Path({str(config_path)!r})
if not SOURCE_KAGGLE_KEY or SOURCE_KAGGLE_KEY.startswith("__KJO_SECRET_"):
    raise RuntimeError("source-owner Kaggle key was not injected")
SOURCE_KAGGLE_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
source_credential_path = SOURCE_KAGGLE_CONFIG_DIR / "kaggle.json"
source_credential_path.write_text(
    json.dumps({{"username": SOURCE_KAGGLE_OWNER, "key": SOURCE_KAGGLE_KEY}}),
    encoding="utf-8",
)
source_credential_path.chmod(0o600)'''

    cli_bootstrap_source = f'''import importlib.util
import json
import shutil
import subprocess
import sys

KJO_KAGGLE_CLI_VERSION = {_KAGGLE_CLI_VERSION!r}

def _kjo_kaggle_cli_available():
    if shutil.which("kaggle"):
        return True
    try:
        return importlib.util.find_spec("kaggle.cli") is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False

installed = False
if not _kjo_kaggle_cli_available():
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--quiet",
            "--disable-pip-version-check",
            f"kaggle=={{KJO_KAGGLE_CLI_VERSION}}",
        ],
        check=True,
    )
    installed = True
if not _kjo_kaggle_cli_available():
    raise RuntimeError("Kaggle CLI bootstrap completed without a usable CLI")
print("KJO_KAGGLE_CLI_BOOTSTRAP " + json.dumps({{
    "available": True,
    "installed": installed,
    "version": KJO_KAGGLE_CLI_VERSION,
}}, sort_keys=True))'''

    overlay_source = f'''from pathlib import Path
import json
import os

REAL_KAGGLE_INPUT_ROOT = Path("/kaggle/input")
CROSS_ACCOUNT_INPUT_ROOT = Path({str(overlay_path)!r})
CROSS_ACCOUNT_CHECKPOINT_MOUNT = Path({str(output_path)!r})
EVALUATION_OVERLAY_OWNER = {evaluation_owner!r}
EVALUATION_OVERLAY_SLUG = {evaluation_slug!r}
evaluation_candidates = (
    REAL_KAGGLE_INPUT_ROOT / EVALUATION_OVERLAY_SLUG,
    REAL_KAGGLE_INPUT_ROOT / EVALUATION_OVERLAY_OWNER / EVALUATION_OVERLAY_SLUG,
    REAL_KAGGLE_INPUT_ROOT / "datasets" / EVALUATION_OVERLAY_OWNER / EVALUATION_OVERLAY_SLUG,
)
evaluation_source = next(
    (candidate for candidate in evaluation_candidates if candidate.is_dir()),
    None,
)
if evaluation_source is None:
    raise FileNotFoundError(
        f"evaluation dataset is not mounted: {{EVALUATION_OVERLAY_OWNER}}/"
        f"{{EVALUATION_OVERLAY_SLUG}}"
    )
if not CROSS_ACCOUNT_CHECKPOINT_MOUNT.is_dir():
    raise FileNotFoundError(
        f"cross-account checkpoint output is absent: {{CROSS_ACCOUNT_CHECKPOINT_MOUNT}}"
    )
evaluation_target = (
    CROSS_ACCOUNT_INPUT_ROOT / "datasets" / EVALUATION_OVERLAY_OWNER /
    EVALUATION_OVERLAY_SLUG
)
evaluation_target.parent.mkdir(parents=True, exist_ok=True)
if evaluation_target.is_symlink() or evaluation_target.is_file():
    evaluation_target.unlink()
elif evaluation_target.exists():
    raise RuntimeError(f"refusing to replace existing directory: {{evaluation_target}}")
evaluation_target.symlink_to(evaluation_source, target_is_directory=True)
os.environ["KJO_KAGGLE_INPUT_ROOT"] = str(CROSS_ACCOUNT_INPUT_ROOT)
print("VDT_CROSS_ACCOUNT_INPUT_OVERLAY " + json.dumps({{
    "checkpoint_mount": str(CROSS_ACCOUNT_CHECKPOINT_MOUNT),
    "evaluation_dataset_source": {evaluation_dataset_source!r},
    "evaluation_source": str(evaluation_source),
    "input_root": str(CROSS_ACCOUNT_INPUT_ROOT),
    "runtime_owner": {runtime_owner!r},
    "source_kernel_id": {source_kernel_id!r},
}}, sort_keys=True))'''

    inserted = [
        _code_cell(cli_bootstrap_source, cell_id="cross-account-cli-bootstrap"),
        _code_cell(credential_source, cell_id="cross-account-credential"),
        _code_cell(
            cross_account_output_source.rstrip("\n"),
            cell_id="cross-account-checkpoint-download",
        ),
        _code_cell(overlay_source, cell_id="cross-account-input-overlay"),
    ]
    cells[insert_at:insert_at] = inserted
    return notebook
