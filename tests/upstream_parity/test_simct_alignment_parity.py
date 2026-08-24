"""Execute the public alignment method without importing its PyTorch module."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


REPO_ROOT = Path(__file__).resolve().parents[2]
SPAN_SOURCE_PATH = REPO_ROOT / "kdflow" / "algorithms" / "span_ctkd.py"


class _FakeTokenizer:
    def __init__(self, pieces: list[str]) -> None:
        self._pieces = pieces
        self.eos_token = "<eos>"

    def decode(self, token_ids: list[int]) -> str:
        assert len(token_ids) == 1
        return self._pieces[token_ids[0]]


@dataclass
class _AlignmentState:
    teacher_tokenizer: _FakeTokenizer
    student_tokenizer: _FakeTokenizer
    _debug_count: int = 3


def _load_public_alignment_method() -> Callable[..., tuple]:
    """Compile only the audited method, avoiding module-level torch imports."""

    source = SPAN_SOURCE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_align_sequences_with_spans":
            isolated = ast.Module(body=[node], type_ignores=[])
            ast.fix_missing_locations(isolated)
            namespace: dict[str, object] = {}
            exec(compile(isolated, str(SPAN_SOURCE_PATH), "exec"), namespace)
            return namespace[node.name]  # type: ignore[return-value]
    raise AssertionError("public alignment method not found")


def _public_segments(teacher_pieces: list[str], student_pieces: list[str]) -> list[tuple[int, ...]]:
    method = _load_public_alignment_method()
    state = _AlignmentState(
        teacher_tokenizer=_FakeTokenizer(teacher_pieces),
        student_tokenizer=_FakeTokenizer(student_pieces),
    )
    segments, _, _ = method(
        state,
        list(range(len(teacher_pieces))),
        list(range(len(student_pieces))),
    )
    return segments


def _paper_common_boundary_segments(
    teacher_pieces: list[str], student_pieces: list[str]
) -> list[tuple[int, int, int, int]]:
    """Reference the finest joint partition via common decoded-byte boundaries."""

    teacher_text = "".join(teacher_pieces).encode("utf-8")
    student_text = "".join(student_pieces).encode("utf-8")
    if teacher_text != student_text:
        raise ValueError("decoded byte strings differ")

    def cumulative_ends(pieces: list[str]) -> list[int]:
        ends: list[int] = []
        offset = 0
        for piece in pieces:
            offset += len(piece.encode("utf-8"))
            ends.append(offset)
        return ends

    teacher_ends = cumulative_ends(teacher_pieces)
    student_ends = cumulative_ends(student_pieces)
    common_ends = sorted(set(teacher_ends) & set(student_ends))

    segments: list[tuple[int, int, int, int]] = []
    teacher_start = student_start = 0
    for boundary in common_ends:
        teacher_end = teacher_ends.index(boundary) + 1
        student_end = student_ends.index(boundary) + 1
        segments.append((teacher_start, teacher_end, student_start, student_end))
        teacher_start, student_start = teacher_end, student_end
    return segments


def test_public_scan_merges_mismatch_with_following_shared_anchor() -> None:
    teacher = ["hap", "py", "!"]
    student = ["ha", "pp", "y", "!"]

    assert _paper_common_boundary_segments(teacher, student) == [
        (0, 2, 0, 3),
        (2, 3, 3, 4),
    ]
    assert _public_segments(teacher, student) == [(0, 3, 0, 4)]


def test_public_scan_drops_unanchored_trailing_mismatch() -> None:
    teacher = ["hap", "py"]
    student = ["ha", "pp", "y"]

    assert _paper_common_boundary_segments(teacher, student) == [(0, 2, 0, 3)]
    assert _public_segments(teacher, student) == []


def test_public_scan_matches_paper_partition_for_one_to_one_tokens() -> None:
    pieces = ["I", " am", " ok"]
    expected = [(0, 1, 0, 1), (1, 2, 1, 2), (2, 3, 2, 3)]

    assert _paper_common_boundary_segments(pieces, pieces) == expected
    assert _public_segments(pieces, pieces) == expected
