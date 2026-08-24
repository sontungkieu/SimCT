#!/usr/bin/env python3
"""Score verified native Tunix generations with the paper-released evaluator."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vdt_tunix.generation_contract import (
    GenerationContractError,
    load_generation_protocol,
)
from vdt_tunix.paper_released_scoring import (
    score_generation_root,
)

EX_UNAVAILABLE = 69
EX_CONFIG = 78


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generation-root", required=True, type=Path)
    parser.add_argument("--evaluation-root", required=True, type=Path)
    parser.add_argument("--generation-protocol", required=True, type=Path)
    parser.add_argument("--evaluator-source", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(argv)
    try:
        protocol = load_generation_protocol(args.generation_protocol)
    except (GenerationContractError, OSError) as exc:
        _atomic_json(
            args.output_dir / "scoring_summary.json",
            {
                "status": "blocked",
                "phase": "scoring_configuration",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "scientific_evidence": False,
            },
        )
        return EX_CONFIG
    try:
        summary = score_generation_root(
            generation_root=args.generation_root,
            evaluation_root=args.evaluation_root,
            protocol=protocol,
            evaluator_source=args.evaluator_source,
            output_root=args.output_dir,
            workers=args.workers,
        )
    except Exception as exc:  # noqa: BLE001 - persist a fail-closed terminal summary
        _atomic_json(
            args.output_dir / "scoring_summary.json",
            {
                "status": "blocked",
                "phase": "paper_released_evaluator_scoring",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "scientific_evidence": False,
            },
        )
        return EX_UNAVAILABLE
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
