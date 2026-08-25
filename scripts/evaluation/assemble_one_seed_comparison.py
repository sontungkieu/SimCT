#!/usr/bin/env python3
"""Validate three terminal variants and emit one comparison report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vdt_tunix.one_seed_comparison import (
    VariantEvidencePaths,
    assemble_one_seed_comparison,
    comparison_markdown,
)


def _variant(parser: argparse.ArgumentParser, name: str) -> None:
    prefix = name.replace("_", "-")
    parser.add_argument(f"--{prefix}-training-config", required=True, type=Path)
    parser.add_argument(f"--{prefix}-training-summary", required=True, type=Path)
    parser.add_argument(f"--{prefix}-generation-summary", required=True, type=Path)
    parser.add_argument(f"--{prefix}-scoring-summary", required=True, type=Path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison-id", required=True)
    parser.add_argument("--generation-protocol", required=True, type=Path)
    parser.add_argument("--evaluation-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    for name in ("sft", "simple_opd", "simct"):
        _variant(parser, name)
    args = parser.parse_args(argv)

    variants = tuple(
        VariantEvidencePaths(
            name=name,
            training_config=getattr(args, f"{name}_training_config"),
            training_summary=getattr(args, f"{name}_training_summary"),
            generation_summary=getattr(args, f"{name}_generation_summary"),
            scoring_summary=getattr(args, f"{name}_scoring_summary"),
        )
        for name in ("sft", "simple_opd", "simct")
    )
    summary = assemble_one_seed_comparison(
        comparison_id=args.comparison_id,
        generation_protocol_path=args.generation_protocol,
        evaluation_root=args.evaluation_root,
        variants=variants,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "comparison_summary.json"
    temporary = json_path.with_name(json_path.name + ".tmp")
    temporary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    temporary.replace(json_path)
    markdown_path = args.output_dir / "comparison_summary.md"
    temporary = markdown_path.with_name(markdown_path.name + ".tmp")
    temporary.write_text(comparison_markdown(summary), encoding="utf-8")
    temporary.replace(markdown_path)
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
