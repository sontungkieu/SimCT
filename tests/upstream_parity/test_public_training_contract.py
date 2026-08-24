"""Static contract checks for the public SimCT launch artifacts."""

from __future__ import annotations

import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
OPD_SCRIPTS = sorted((REPO_ROOT / "scripts" / "ctopd").glob("*span*.sh"))
SFT_CONFIGS = sorted((REPO_ROOT / "scripts" / "sft").glob("*_sft_warmup_*.yaml"))

PAPER_OPD = {
    "learning_rate": "1e-6",
    "num_epochs": "2",
    "max_len": "4096",
    "top_p": "0.95",
}


def _shell_opts(path: Path) -> dict[str, str]:
    source = path.read_text(encoding="utf-8")
    return dict(re.findall(r'OPTS\+?=" --([A-Za-z0-9_]+) ([^" ]+)"', source))


def _flat_yaml(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        values[key.strip()] = value.strip()
    return values


def test_all_three_public_opd_scripts_are_in_scope() -> None:
    assert [path.name for path in OPD_SCRIPTS] == [
        "phi4_gemma2_span_mix10k_lr5e-7.sh",
        "qwen25_gemma2_span_mix10k_lr5e-7.sh",
        "qwen25_phi4_span_mix10k_lr5e-7.sh",
    ]


def test_public_opd_scripts_lock_current_non_paper_hyperparameters() -> None:
    for path in OPD_SCRIPTS:
        opts = _shell_opts(path)
        assert opts["kd_algorithm"] == "span_ctkd"
        assert opts["kd_loss_fn"] == "rkl"
        assert opts["train_batch_size"] == "64"
        assert opts["micro_train_batch_size"] == "1"
        assert opts["temperature"] == "0.6"
        assert opts["generate_max_len"] == "4096"

        assert opts["learning_rate"] == "5e-7"
        assert opts["learning_rate"] != PAPER_OPD["learning_rate"]
        assert opts["num_epochs"] == "1"
        assert opts["num_epochs"] != PAPER_OPD["num_epochs"]
        assert opts["max_len"] == "8192"
        assert opts["max_len"] != PAPER_OPD["max_len"]
        assert "top_p" not in opts


def test_omitted_top_p_and_gh_flags_activate_non_paper_defaults() -> None:
    rollout_args = (REPO_ROOT / "kdflow" / "arguments" / "rollout_args.py").read_text(
        encoding="utf-8"
    )
    distillation_args = (
        REPO_ROOT / "kdflow" / "arguments" / "distillation_args.py"
    ).read_text(encoding="utf-8")

    assert re.search(r"top_p:\s*float\s*=\s*field\(\s*default=1\.0", rollout_args)
    assert re.search(
        r"span_gh_mask_threshold:\s*float\s*=\s*field\(\s*default=2\.0",
        distillation_args,
    )
    for path in OPD_SCRIPTS:
        source = path.read_text(encoding="utf-8")
        assert "--top_p " not in source
        assert "--span_gh_mask_threshold " not in source


def test_sft_yaml_values_differ_from_paper_batch_and_length() -> None:
    assert len(SFT_CONFIGS) == 3
    for path in SFT_CONFIGS:
        values = _flat_yaml(path)
        assert values["learning_rate"] == "2e-6"
        assert values["num_train_epochs"] == "2.0"
        assert values["per_device_train_batch_size"] == "4"
        assert values["gradient_accumulation_steps"] == "4"
        assert values["cutoff_len"] == "2048"

        public_effective_batch = 8 * int(values["per_device_train_batch_size"]) * int(
            values["gradient_accumulation_steps"]
        )
        assert public_effective_batch == 128
        assert public_effective_batch != 64


def test_generation_and_builder_wrappers_resolve_to_missing_python_paths() -> None:
    script_dir = REPO_ROOT / "scripts"
    wrapper_targets = {
        "run_generate_responses_10k_qwen.sh": "generate_teacher_responses.py",
        "run_generate_responses_10k_phi4.sh": "generate_teacher_responses.py",
        "run_build_sft_10k_qwen.sh": "build_sft_warmup_dataset.py",
        "run_build_sft_10k_phi4.sh": "build_sft_warmup_dataset.py",
    }

    for wrapper_name, target_name in wrapper_targets.items():
        wrapper = REPO_ROOT / "scripts" / "sft" / wrapper_name
        source = wrapper.read_text(encoding="utf-8")
        assert 'SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"' in source
        assert f"${{SCRIPT_DIR}}/{target_name}" in source
        assert not (script_dir / target_name).exists()
        assert (script_dir / "sft" / target_name).exists()
