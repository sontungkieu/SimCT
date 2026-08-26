#!/usr/bin/env python3
"""Attach exact Kaggle model versions to a KJO-staged notebook package."""

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
    attach_model_sources,
    verify_attached_model_sources,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--stage-manifest", required=True, type=Path)
    parser.add_argument("--model-source", action="append", required=True)
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Fail unless metadata and manifest already contain exact sources.",
    )
    args = parser.parse_args(argv)
    try:
        operation = (
            verify_attached_model_sources
            if args.verify_only
            else attach_model_sources
        )
        result = operation(args.metadata, args.stage_manifest, args.model_source)
    except (KaggleModelSourceError, OSError) as exc:
        print(json.dumps({"ok": False, "error": str(exc)}), file=sys.stderr)
        return 78
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
