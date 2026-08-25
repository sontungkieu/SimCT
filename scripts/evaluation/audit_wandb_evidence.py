#!/usr/bin/env python3
"""Audit one variant's terminal online W&B evidence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vdt_tunix.wandb_evidence import (  # noqa: E402
    WandbEvidenceError,
    validate_native_wandb_evidence,
)


def _load(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise WandbEvidenceError(f"summary must be an object: {path}")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-summary", required=True, type=Path)
    parser.add_argument("--generation-summary", required=True, type=Path)
    parser.add_argument("--scoring-summary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--allow-backfill",
        action="store_true",
        help="Accept a complete, provenance-hashed historical W&B backfill.",
    )
    args = parser.parse_args(argv)
    try:
        report = validate_native_wandb_evidence(
            training_summary=_load(args.training_summary),
            generation_summary=_load(args.generation_summary),
            scoring_summary=_load(args.scoring_summary),
            allow_backfill=args.allow_backfill,
        )
    except (OSError, json.JSONDecodeError, WandbEvidenceError) as exc:
        report = {
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        exit_code = 1
    else:
        exit_code = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(args.output)
    print(json.dumps(report, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
