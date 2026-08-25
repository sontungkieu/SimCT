#!/usr/bin/env python3
"""Render a Kaggle TPU notebook for checkpoint-native benchmark generation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vdt_tunix.kaggle_generation_sources import render_generation_notebook
from vdt_tunix.kaggle_model_sources import KaggleModelSourceError


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=("sft", "simple_opd", "simct"), required=True)
    parser.add_argument("--training-config-relative-path", required=True)
    parser.add_argument("--generation-protocol-relative-path", required=True)
    parser.add_argument("--repo-dataset-source", required=True)
    parser.add_argument("--evaluation-dataset-source", required=True)
    parser.add_argument("--checkpoint-kernel-source", required=True)
    parser.add_argument("--checkpoint-relative-path", required=True)
    parser.add_argument("--student-model-source", required=True)
    parser.add_argument("--expected-training-run-id")
    parser.add_argument("--training-seed", type=int)
    parser.add_argument(
        "--wandb-group", default="public-substitute-multiseed"
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        notebook = render_generation_notebook(
            variant=args.variant,
            training_config_relative_path=args.training_config_relative_path,
            generation_protocol_relative_path=args.generation_protocol_relative_path,
            repo_dataset_source=args.repo_dataset_source,
            evaluation_dataset_source=args.evaluation_dataset_source,
            checkpoint_kernel_source=args.checkpoint_kernel_source,
            checkpoint_relative_path=args.checkpoint_relative_path,
            student_model_source=args.student_model_source,
            expected_training_run_id=args.expected_training_run_id,
            training_seed=args.training_seed,
            wandb_group=args.wandb_group,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_name(args.output.name + ".tmp")
        temporary.write_text(
            json.dumps(notebook, indent=1, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(args.output)
    except (KaggleModelSourceError, OSError) as exc:
        print(json.dumps({"ok": False, "error": str(exc)}), file=sys.stderr)
        return 78
    print(
        json.dumps(
            {
                "ok": True,
                "variant": args.variant,
                "output": str(args.output),
                "cell_count": len(notebook["cells"]),
                "scientific_evidence": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
