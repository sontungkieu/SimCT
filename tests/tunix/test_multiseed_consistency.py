from __future__ import annotations

import json

from vdt_tunix.multiseed_consistency import (
    BENCHMARKS,
    VARIANTS,
    audit_two_seed_consistency,
)


def _write(root, benchmark, correctness):
    path = root / benchmark / "scored_predictions.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(
                {"instance_id": f"{benchmark}-{index:04d}", "correct": value}
            )
            + "\n"
            for index, value in enumerate(correctness)
        ),
        encoding="utf-8",
    )


def _roots(tmp_path, first_scores=None, second_scores=None):
    first_scores = first_scores or {}
    second_scores = second_scores or {}
    first = {}
    second = {}
    for variant in VARIANTS:
        first[variant] = tmp_path / "first" / variant
        second[variant] = tmp_path / "second" / variant
        for benchmark in BENCHMARKS:
            _write(
                first[variant],
                benchmark,
                first_scores.get((variant, benchmark), [False] * 100),
            )
            _write(
                second[variant],
                benchmark,
                second_scores.get((variant, benchmark), [False] * 100),
            )
    return first, second


def test_identical_paired_correctness_allows_third_seed(tmp_path):
    first, second = _roots(tmp_path)
    report = audit_two_seed_consistency(
        first_roots=first, second_roots=second
    )
    assert report["status"] == "consistent"
    assert report["allow_third_seed"] is True
    assert report["triggers"] == []


def test_large_paired_seed_gap_stops_before_third_seed(tmp_path):
    changed = [True] * 10 + [False] * 90
    first, second = _roots(
        tmp_path,
        second_scores={("simct", "gsm8k"): changed},
    )
    report = audit_two_seed_consistency(
        first_roots=first, second_roots=second
    )
    assert report["status"] == "investigate"
    assert report["allow_third_seed"] is False
    assert report["triggers"][0]["type"] == "seed_gap"


def test_large_treatment_effect_sign_flip_stops(tmp_path):
    half = [True] * 50 + [False] * 50
    simct_first = [True] * 55 + [False] * 45
    simct_second = [True] * 45 + [False] * 55
    first, second = _roots(
        tmp_path,
        first_scores={
            ("simple_opd", "math500"): half,
            ("simct", "math500"): simct_first,
        },
        second_scores={
            ("simple_opd", "math500"): half,
            ("simct", "math500"): simct_second,
        },
    )
    report = audit_two_seed_consistency(
        first_roots=first, second_roots=second
    )
    assert report["allow_third_seed"] is False
    assert any(
        trigger["type"] == "treatment_effect_sign_flip"
        for trigger in report["triggers"]
    )
