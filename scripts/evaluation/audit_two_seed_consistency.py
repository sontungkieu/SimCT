#!/usr/bin/env python3
"""Apply the predeclared paired-consistency gate to two training replicates."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vdt_tunix.multiseed_consistency import (  # noqa: E402
    MultiseedConsistencyError,
    audit_two_seed_consistency,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    for seed in ("first", "second"):
        for variant in ("sft", "simple-opd", "simct"):
            parser.add_argument(
                f"--{seed}-{variant}-scoring-root", required=True, type=Path
            )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    first = {
        "sft": args.first_sft_scoring_root,
        "simple_opd": args.first_simple_opd_scoring_root,
        "simct": args.first_simct_scoring_root,
    }
    second = {
        "sft": args.second_sft_scoring_root,
        "simple_opd": args.second_simple_opd_scoring_root,
        "simct": args.second_simct_scoring_root,
    }
    try:
        report = audit_two_seed_consistency(
            first_roots=first, second_roots=second
        )
    except (OSError, MultiseedConsistencyError) as exc:
        report = {
            "contract_version": 1,
            "status": "failed",
            "allow_third_seed": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "paper_reproduction": False,
        }
        exit_code = 1
    else:
        exit_code = 0 if report["status"] == "consistent" else 2
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
