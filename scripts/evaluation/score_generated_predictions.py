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
from vdt_tunix.observability import start_wandb_run
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
    observability = None
    try:
        generation_summary = json.loads(
            (args.generation_root / "generation_summary.json").read_text(
                encoding="utf-8"
            )
        )
        variant = generation_summary["variant"]
        checkpoint_run_id = generation_summary["checkpoint_run_id"]
        training_config_sha256 = generation_summary["training_config_sha256"]
        if not all(
            isinstance(value, str) and value
            for value in (variant, checkpoint_run_id, training_config_sha256)
        ):
            raise GenerationContractError(
                "generation summary has incomplete W&B lineage"
            )
        observability = start_wandb_run(
            run_id=checkpoint_run_id,
            objective=f"{variant}_scoring",
            config_sha256=training_config_sha256,
            dataset_manifest_sha256=protocol.digest(),
            metadata={
                "variant": variant,
                "protocol_id": protocol.protocol_id,
                "protocol_sha256": protocol.digest(),
                "student_parameters_sha256": generation_summary.get(
                    "student_parameters_sha256", ""
                ),
                "evaluation_seed": protocol.seed,
            },
        )
        observability.require_active()
        summary = score_generation_root(
            generation_root=args.generation_root,
            evaluation_root=args.evaluation_root,
            protocol=protocol,
            evaluator_source=args.evaluator_source,
            output_root=args.output_dir,
            workers=args.workers,
        )
    except Exception as exc:  # noqa: BLE001 - persist a fail-closed terminal summary
        if observability is not None:
            observability.finish(
                training_status="blocked", exit_code=EX_UNAVAILABLE
            )
        _atomic_json(
            args.output_dir / "scoring_summary.json",
            {
                "status": "blocked",
                "phase": "paper_released_evaluator_scoring",
                "error_type": type(exc).__name__,
                "error": str(exc),
                **(
                    {}
                    if observability is None
                    else {"observability": observability.summary()}
                ),
                "scientific_evidence": False,
            },
        )
        return EX_UNAVAILABLE
    for step, benchmark in enumerate(summary["benchmarks"], start=1):
        observability.log_metrics(
            {
                "score": benchmark["score"],
                "correct": benchmark["correct"],
                "total": benchmark["total"],
                "empty_output_count": benchmark.get("empty_output_count", 0),
                "extraction_failed_count": benchmark.get(
                    "extraction_failed_count", 0
                ),
            },
            step=step,
            namespace=f"evaluation/{benchmark['benchmark']}",
        )
    observability.finish(training_status="complete", exit_code=0)
    summary["observability"] = observability.summary()
    _atomic_json(args.output_dir / "scoring_summary.json", summary)
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
