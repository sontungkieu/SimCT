#!/usr/bin/env python3
"""Materialize the bounded public GSM8K/MBPP training substitute."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vdt_tunix.training_data_materialize import (
    PublicSubstituteError,
    materialize_public_substitute,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--per-source", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    try:
        provenance = materialize_public_substitute(
            args.output_root, per_source=args.per_source, seed=args.seed
        )
    except (OSError, PublicSubstituteError) as exc:
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
    print(json.dumps({"status": "complete", **provenance}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
