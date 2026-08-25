"""Paired correctness audit for two public-substitute training replicates."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any


BENCHMARKS = ("gsm8k", "math500", "mbpp", "live-code-bench-v6")
VARIANTS = ("sft", "simple_opd", "simct")


class MultiseedConsistencyError(RuntimeError):
    """Raised when paired seed evidence is incomplete or mismatched."""


def _correctness(root: Path, benchmark: str) -> dict[str, int]:
    path = root / benchmark / "scored_predictions.jsonl"
    rows: dict[str, int] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise MultiseedConsistencyError(f"missing scored predictions: {path}") from exc
    for line_number, line in enumerate(lines, start=1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise MultiseedConsistencyError(
                f"invalid scored row {path}:{line_number}"
            ) from exc
        instance_id = row.get("instance_id")
        correct = row.get("correct")
        if not isinstance(instance_id, str) or not instance_id:
            raise MultiseedConsistencyError(
                f"missing instance_id in {path}:{line_number}"
            )
        if not isinstance(correct, bool):
            raise MultiseedConsistencyError(
                f"correct must be boolean in {path}:{line_number}"
            )
        if instance_id in rows:
            raise MultiseedConsistencyError(
                f"duplicate instance_id {instance_id!r} in {path}"
            )
        rows[instance_id] = int(correct)
    if not rows:
        raise MultiseedConsistencyError(f"no scored rows in {path}")
    return rows


def _paired_delta(first: Mapping[str, int], second: Mapping[str, int]) -> dict[str, Any]:
    if set(first) != set(second):
        raise MultiseedConsistencyError("paired instance ids drifted between seeds")
    deltas = [second[key] - first[key] for key in sorted(first)]
    count = len(deltas)
    mean = sum(deltas) / count
    if count == 1:
        paired_se = 0.0
    else:
        variance = sum((value - mean) ** 2 for value in deltas) / (count - 1)
        paired_se = math.sqrt(variance / count)
    return {
        "record_count": count,
        "first_score": sum(first.values()) / count,
        "second_score": sum(second.values()) / count,
        "score_gap": mean,
        "absolute_score_gap": abs(mean),
        "paired_standard_error": paired_se,
        "discordant_count": sum(value != 0 for value in deltas),
    }


def audit_two_seed_consistency(
    *,
    first_roots: Mapping[str, str | Path],
    second_roots: Mapping[str, str | Path],
    absolute_score_gap_floor: float = 0.05,
    paired_standard_error_multiplier: float = 3.0,
    treatment_effect_flip_floor: float = 0.02,
) -> dict[str, Any]:
    """Return a predeclared stop/go decision before a possible third seed."""

    if set(first_roots) != set(VARIANTS) or set(second_roots) != set(VARIANTS):
        raise MultiseedConsistencyError(
            f"roots must contain exactly {list(VARIANTS)}"
        )
    if not 0 <= absolute_score_gap_floor <= 1:
        raise MultiseedConsistencyError("absolute score gap floor is invalid")
    if paired_standard_error_multiplier <= 0:
        raise MultiseedConsistencyError("paired SE multiplier must be positive")
    if not 0 <= treatment_effect_flip_floor <= 1:
        raise MultiseedConsistencyError("treatment effect flip floor is invalid")

    paired: dict[str, dict[str, Any]] = {}
    correctness: dict[str, dict[str, tuple[dict[str, int], dict[str, int]]]] = {}
    triggers: list[dict[str, Any]] = []
    for variant in VARIANTS:
        paired[variant] = {}
        correctness[variant] = {}
        for benchmark in BENCHMARKS:
            first = _correctness(Path(first_roots[variant]), benchmark)
            second = _correctness(Path(second_roots[variant]), benchmark)
            correctness[variant][benchmark] = (first, second)
            result = _paired_delta(first, second)
            threshold = max(
                absolute_score_gap_floor,
                paired_standard_error_multiplier
                * result["paired_standard_error"],
            )
            result["strong_divergence_threshold"] = threshold
            result["strong_divergence"] = result["absolute_score_gap"] > threshold
            paired[variant][benchmark] = result
            if result["strong_divergence"]:
                triggers.append(
                    {
                        "type": "seed_gap",
                        "variant": variant,
                        "benchmark": benchmark,
                        "absolute_score_gap": result["absolute_score_gap"],
                        "threshold": threshold,
                    }
                )

    treatment_effects: dict[str, Any] = {}
    for benchmark in BENCHMARKS:
        simple = paired["simple_opd"][benchmark]
        simct = paired["simct"][benchmark]
        first_effect = simct["first_score"] - simple["first_score"]
        second_effect = simct["second_score"] - simple["second_score"]
        sign_flip = (
            first_effect * second_effect < 0
            and min(abs(first_effect), abs(second_effect))
            >= treatment_effect_flip_floor
        )
        treatment_effects[benchmark] = {
            "first_simct_minus_simple_opd": first_effect,
            "second_simct_minus_simple_opd": second_effect,
            "flip_floor": treatment_effect_flip_floor,
            "strong_sign_flip": sign_flip,
        }
        if sign_flip:
            triggers.append(
                {
                    "type": "treatment_effect_sign_flip",
                    "benchmark": benchmark,
                    "first_effect": first_effect,
                    "second_effect": second_effect,
                }
            )

    investigate = bool(triggers)
    return {
        "contract_version": 1,
        "status": "investigate" if investigate else "consistent",
        "allow_third_seed": not investigate,
        "paired": paired,
        "treatment_effects": treatment_effects,
        "triggers": triggers,
        "paper_reproduction": False,
    }
