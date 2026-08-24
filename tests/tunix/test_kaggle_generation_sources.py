from __future__ import annotations

import zipfile

import pytest

from vdt_tunix.kaggle_generation_sources import render_generation_notebook
from vdt_tunix.kaggle_model_sources import KaggleModelSourceError

STUDENT = "google/gemma-2/flax/gemma2-2b-it/1"


def _render(variant="sft"):
    return render_generation_notebook(
        variant=variant,
        training_config_relative_path=(
            f"configs/reproduction/qwen25_7b_to_gemma2_2b_public_{variant}_screen.json"
        ),
        generation_protocol_relative_path=(
            "configs/evaluation/simct_paper_one_seed_generation.json"
        ),
        repo_dataset_source="testowner/repo-v1",
        evaluation_dataset_source="testowner/evaluation-v1",
        checkpoint_kernel_source="testowner/checkpoint-v1",
        checkpoint_relative_path="vdt_public_sft_screen/checkpoints",
        student_model_source=STUDENT,
    )


@pytest.mark.parametrize("variant", ["sft", "simple_opd", "simct"])
def test_generation_notebook_is_pinned_and_syntax_valid(variant):
    notebook = _render(variant)
    source = "".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )
    assert len(notebook["cells"]) == 6
    assert "testowner/repo-v1" in source
    assert "testowner/evaluation-v1" in source
    assert "testowner/checkpoint-v1" in source
    assert STUDENT in source
    assert "qwen-lm/qwen2.5/transformers/7b-instruct/1" not in source
    assert '"teacher_loaded": False' in source
    assert "--no-deps" in source
    assert "kaggle_v5e8_generate.py" in source
    assert "score_generated_predictions.py" in source
    assert "paper-released scorer" in source
    assert "not the official benchmark harness" in source
    assert "VDT_SCORING_SUMMARY" in source
    assert "scientific_evidence" in source
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"<generation-cell-{index}>", "exec")


def test_generation_input_cell_resolves_four_archives_and_checkpoint(
    tmp_path, monkeypatch
):
    notebook = _render()
    source = next(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if "EVALUATION_DATASET_SOURCE" in "".join(cell.get("source", []))
    )
    input_root = tmp_path / "input"
    eval_root = (
        input_root
        / "datasets"
        / "testowner"
        / "evaluation-v1"
        / "versions"
        / "1"
    )
    eval_root.mkdir(parents=True)
    for benchmark in ("gsm8k", "math500", "mbpp", "live-code-bench-v6"):
        with zipfile.ZipFile(eval_root / f"{benchmark}.zip", "w") as archive:
            archive.writestr("manifest.json", "{}\n")
            archive.writestr("records.jsonl", "{}\n")
    checkpoint = (
        input_root
        / "kernels"
        / "testowner"
        / "checkpoint-v1"
        / "vdt_public_sft_screen"
        / "checkpoints"
    )
    checkpoint.mkdir(parents=True)
    (checkpoint / "latest.json").write_text("{}\n", encoding="utf-8")
    working_eval = tmp_path / "working-eval"
    source = source.replace(
        'Path("/kaggle/working/vdt_evaluation_inputs")',
        f"Path({str(working_eval)!r})",
    )
    monkeypatch.setenv("KJO_KAGGLE_INPUT_ROOT", str(input_root))
    namespace = {}
    exec(compile(source, "<generation-inputs>", "exec"), namespace)  # noqa: S102
    assert namespace["CHECKPOINT_ROOT"] == checkpoint
    for benchmark in ("gsm8k", "math500", "mbpp", "live-code-bench-v6"):
        assert (working_eval / benchmark / "manifest.json").is_file()
        assert (working_eval / benchmark / "records.jsonl").is_file()


def test_generation_renderer_rejects_checkpoint_escape():
    with pytest.raises(KaggleModelSourceError, match="checkpoint_relative_path"):
        render_generation_notebook(
            variant="sft",
            training_config_relative_path="configs/sft.json",
            generation_protocol_relative_path="configs/eval.json",
            repo_dataset_source="testowner/repo-v1",
            evaluation_dataset_source="testowner/eval-v1",
            checkpoint_kernel_source="testowner/checkpoint-v1",
            checkpoint_relative_path="../secret",
            student_model_source=STUDENT,
        )
