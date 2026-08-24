#!/usr/bin/env python3
"""Stage or validate the bounded VDT Kaggle TPU package locally."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vdt_tunix.kaggle_package import (
    DEFAULT_KJO_CLI,
    DEFAULT_OUTPUT_ROOT,
    KagglePackageError,
    load_package_spec,
    safe_summary,
    stage_package,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Local-only Kaggle TPU v5e-8 package staging; never submits."
    )
    parser.add_argument("--kjo-cli", type=Path, default=DEFAULT_KJO_CLI)
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate")
    validate.add_argument("--spec", required=True, type=Path)

    stage = subparsers.add_parser("stage")
    stage.add_argument("--spec", required=True, type=Path)
    stage.add_argument("--output-dir", required=True, type=Path)
    stage.add_argument("--clean", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "validate":
            result = safe_summary(
                load_package_spec(args.spec, output_root=DEFAULT_OUTPUT_ROOT)
            )
        else:
            package = stage_package(
                args.spec,
                args.output_dir,
                output_root=DEFAULT_OUTPUT_ROOT,
                kjo_cli=args.kjo_cli,
                clean=args.clean,
            )
            result = {
                "ok": True,
                "package_dir": str(package),
                "package_manifest": str(
                    package / "vdt_kaggle_package_manifest.json"
                ),
                "future_submit_command": str(package / "future_submit_command.sh"),
                "remote_submit_performed": False,
            }
    except (KagglePackageError, OSError, subprocess.SubprocessError) as exc:
        print(
            json.dumps(
                {
                    "ok": False,
                    "status": "blocked",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "remote_submit_performed": False,
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 78
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
