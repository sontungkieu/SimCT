#!/usr/bin/env python3
"""Materialize the four SimCT paper benchmarks at immutable revisions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vdt_tunix.evaluation_data import (
    PINNED_BENCHMARKS,
    EvaluationDataError,
    materialize_all,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument(
        "--benchmark",
        action="append",
        choices=[source.name for source in PINNED_BENCHMARKS],
        help="materialize only this benchmark (repeatable; default: all)",
    )
    args = parser.parse_args(argv)
    try:
        manifests = materialize_all(args.output_root, names=args.benchmark)
    except (EvaluationDataError, OSError) as exc:
        print(
            json.dumps(
                {
                    "status": "blocked",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 69
    print(
        json.dumps(
            {
                "status": "complete",
                "output_root": str(args.output_root.resolve()),
                "benchmarks": manifests,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
