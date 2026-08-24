#!/usr/bin/env python3
"""Render the model-source-backed SimCT canary notebook."""

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
    render_canary_notebook,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--config-relative-path", required=True)
    parser.add_argument("--repo-dataset-source", required=True)
    parser.add_argument("--student-model-source", required=True)
    parser.add_argument("--teacher-model-source", required=True)
    args = parser.parse_args(argv)
    try:
        payload = render_canary_notebook(
            config_relative_path=args.config_relative_path,
            repo_dataset_source=args.repo_dataset_source,
            student_model_source=args.student_model_source,
            teacher_model_source=args.teacher_model_source,
        )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except (KaggleModelSourceError, OSError) as exc:
        print(json.dumps({"ok": False, "error": str(exc)}), file=sys.stderr)
        return 78
    print(json.dumps({"ok": True, "out": str(args.out)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
