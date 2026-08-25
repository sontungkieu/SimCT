#!/usr/bin/env python3
"""Replay completed training/generation/scoring artifacts to W&B."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vdt_tunix.wandb_backfill import (  # noqa: E402
    WandbBackfillError,
    backfill_wandb_evidence,
)


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-summary", required=True, type=Path)
    parser.add_argument("--training-metrics", required=True, type=Path)
    parser.add_argument("--generation-summary", required=True, type=Path)
    parser.add_argument("--scoring-summary", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--project", default="vdt-simct-tunix-reproduction"
    )
    parser.add_argument(
        "--group", default="public-substitute-multiseed"
    )
    args = parser.parse_args(argv)
    manifest_path = args.output_dir / "backfill_manifest.json"
    try:
        result = backfill_wandb_evidence(
            training_summary_path=args.training_summary,
            training_metrics_path=args.training_metrics,
            generation_summary_path=args.generation_summary,
            scoring_summary_path=args.scoring_summary,
            project=args.project,
            group=args.group,
        )
        summaries = result.pop("summaries")
        output_paths = {
            phase: args.output_dir / f"{phase}_summary.json"
            for phase in ("training", "generation", "scoring")
        }
        for phase, path in output_paths.items():
            _atomic_json(path, summaries[phase])
        result["output_paths"] = {
            phase: str(path) for phase, path in output_paths.items()
        }
        result["output_sha256"] = {
            phase: _sha256(path) for phase, path in output_paths.items()
        }
        _atomic_json(manifest_path, result)
    except (OSError, WandbBackfillError) as exc:
        failure = {
            "status": "failed",
            "mode": "historical_backfill",
            "scientific_evidence": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        _atomic_json(manifest_path, failure)
        print(json.dumps(failure, sort_keys=True))
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
