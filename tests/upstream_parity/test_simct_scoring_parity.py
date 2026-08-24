"""Pure-Python fixtures for paper-vs-public-code SimCT scoring."""

from __future__ import annotations

import ast
import math
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SPAN_SOURCE_PATH = REPO_ROOT / "kdflow" / "algorithms" / "span_ctkd.py"


def _logsumexp(values: list[float]) -> float:
    maximum = max(values)
    return maximum + math.log(sum(math.exp(value - maximum) for value in values))


def _softmax(scores: list[float]) -> list[float]:
    normalizer = _logsumexp(scores)
    return [math.exp(score - normalizer) for score in scores]


def _paper_span_score(logit_rows: list[list[float]], token_ids: list[int]) -> float:
    return sum(
        row[token_id] - _logsumexp(row)
        for row, token_id in zip(logit_rows, token_ids, strict=True)
    ) / len(token_ids)


def _public_span_score(logit_rows: list[list[float]], token_ids: list[int]) -> float:
    return sum(
        row[token_id]
        for row, token_id in zip(logit_rows, token_ids, strict=True)
    ) / len(token_ids)


def _paper_candidate_probs(
    logit_rows: list[list[float]], token_ids: list[int], shared_ids: list[int]
) -> list[float]:
    first_log_normalizer = _logsumexp(logit_rows[0])
    shared_scores = [
        logit_rows[0][token_id] - first_log_normalizer for token_id in shared_ids
    ]
    return _softmax(shared_scores + [_paper_span_score(logit_rows, token_ids)])


def _public_candidate_probs(
    logit_rows: list[list[float]], token_ids: list[int], shared_ids: list[int]
) -> list[float]:
    shared_scores = [logit_rows[0][token_id] for token_id in shared_ids]
    return _softmax(shared_scores + [_public_span_score(logit_rows, token_ids)])


def _teacher_z(
    logit_rows: list[list[float]],
    span_token_ids: list[int],
    shared_ids: list[int],
) -> float:
    first_probs = _softmax(logit_rows[0])
    shared_mass = sum(first_probs[token_id] for token_id in shared_ids)
    mean_log_probability = _paper_span_score(logit_rows, span_token_ids)
    return min(1.0, shared_mass + math.exp(mean_log_probability))


def _method_source(method_name: str) -> str:
    source = SPAN_SOURCE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == method_name:
            segment = ast.get_source_segment(source, node)
            assert segment is not None
            return segment
    raise AssertionError(f"method not found: {method_name}")


def test_public_source_averages_selected_raw_logits() -> None:
    source = _method_source("_build_virtual_vocab_logits")
    assert "stu_self_logits.mean()" in source
    assert "tea_self_logits.mean()" in source
    assert "torch.log_softmax" not in source


def test_paper_and_public_scores_match_when_log_normalizers_cancel() -> None:
    rows = [[2.0, 0.0, -1.0], [0.0, 2.0, -1.0]]
    token_ids = [0, 1]
    shared_ids = [1, 2]

    paper = _paper_candidate_probs(rows, token_ids, shared_ids)
    public = _public_candidate_probs(rows, token_ids, shared_ids)

    assert paper == pytest.approx(public)


def test_paper_and_public_scores_diverge_with_position_specific_normalizers() -> None:
    rows = [[2.0, 0.0, -1.0], [10.0, 10.0, 10.0]]
    token_ids = [0, 1]
    shared_ids = [1, 2]

    paper_score = _paper_span_score(rows, token_ids)
    public_score = _public_span_score(rows, token_ids)
    paper = _paper_candidate_probs(rows, token_ids, shared_ids)
    public = _public_candidate_probs(rows, token_ids, shared_ids)

    omitted_correction = sum(_logsumexp(row) for row in rows) / len(rows)
    assert public_score - paper_score == pytest.approx(omitted_correction)
    assert max(abs(left - right) for left, right in zip(paper, public, strict=True)) > 0.1


def test_gh_source_uses_log_probabilities_but_virtual_scores_do_not() -> None:
    gh_source = _method_source("_compute_segment_teacher_Z")
    virtual_source = _method_source("_build_virtual_vocab_logits")

    assert "torch.log_softmax" in gh_source
    assert "torch.exp(logps.mean())" in gh_source
    assert "torch.log_softmax" not in virtual_source


def test_gh_threshold_keeps_equality_and_masks_low_score() -> None:
    at_boundary = _teacher_z(
        logit_rows=[[0.0] * 4, [0.0] * 4],
        span_token_ids=[1, 2],
        shared_ids=[0],
    )
    low_score = _teacher_z(
        logit_rows=[[0.0] * 10, [0.0] * 10],
        span_token_ids=[1, 2],
        shared_ids=[0],
    )

    assert at_boundary == pytest.approx(0.5)
    assert at_boundary >= 1.0 / 2.0  # public code uses >=, so equality is kept
    assert low_score == pytest.approx(0.2)
    assert not low_score >= 1.0 / 2.0
