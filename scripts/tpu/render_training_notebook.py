#!/usr/bin/env python3
"""Render a pinned Kaggle TPU notebook for SFT, SimpleOPD, or SimCT."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vdt_tunix.kaggle_model_sources import (
    KaggleModelSourceError,
    render_training_notebook,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("sft", "simple_opd", "simct"), required=True)
    parser.add_argument("--config-relative-path", required=True)
    parser.add_argument("--repo-dataset-source", required=True)
    parser.add_argument("--training-dataset-source", required=True)
    parser.add_argument("--training-manifest-relative-path", required=True)
    parser.add_argument("--student-model-source", required=True)
    parser.add_argument("--teacher-model-source", required=True)
    parser.add_argument("--expected-run-id")
    parser.add_argument("--training-seed", type=int)
    parser.add_argument(
        "--wandb-group", default="public-substitute-one-seed"
    )
    parser.add_argument("--warm-start-kernel-source")
    parser.add_argument("--warm-start-kernel-version", type=int)
    parser.add_argument("--warm-start-relative-path")
    parser.add_argument("--profile-step", type=int, default=0)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        notebook = render_training_notebook(
            phase=args.phase,
            config_relative_path=args.config_relative_path,
            repo_dataset_source=args.repo_dataset_source,
            training_dataset_source=args.training_dataset_source,
            training_manifest_relative_path=args.training_manifest_relative_path,
            student_model_source=args.student_model_source,
            teacher_model_source=args.teacher_model_source,
            expected_run_id=args.expected_run_id,
            training_seed=args.training_seed,
            wandb_group=args.wandb_group,
            warm_start_kernel_source=args.warm_start_kernel_source,
            warm_start_kernel_version=args.warm_start_kernel_version,
            warm_start_relative_path=args.warm_start_relative_path,
            profile_step=args.profile_step,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_name(args.output.name + ".tmp")
        temporary.write_text(
            json.dumps(notebook, indent=1, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(args.output)
    except (KaggleModelSourceError, OSError) as exc:
        print(
            json.dumps(
                {
                    "ok": False,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 78
    print(
        json.dumps(
            {
                "ok": True,
                "phase": args.phase,
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
