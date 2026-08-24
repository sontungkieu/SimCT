from __future__ import annotations

from vdt_tunix.contracts import TokenSequence
from vdt_tunix.supervision import (
    build_aligned_layout,
    build_overlap_vocabulary,
    pad_layouts,
)


class VocabularyTokenizer:
    eos_token_id = 9

    def __init__(self, vocabulary):
        self._vocabulary = vocabulary

    def get_vocab(self):
        return dict(self._vocabulary)


def test_overlap_mapping_is_sorted_normalized_and_eos_paired():
    student = VocabularyTokenizer({"Ġhello": 3, "world": 4, "only-s": 5})
    teacher = VocabularyTokenizer({"▁hello": 7, "world": 8, "only-t": 6})
    teacher.eos_token_id = 10
    overlap = build_overlap_vocabulary(student, teacher)
    assert overlap.normalized_tokens == ("world", "▁hello", "<paired-eos>")
    assert overlap.student_ids == (4, 3, 9)
    assert overlap.teacher_ids == (8, 7, 10)


def test_aligned_layout_and_padding_use_completion_token_coordinates():
    student = TokenSequence(
        text="happy",
        token_ids=(1, 2, 3),
        pieces=(b"ha", b"pp", b"y"),
    )
    teacher = TokenSequence(
        text="happy",
        token_ids=(4, 5),
        pieces=(b"hap", b"py"),
    )
    layout = build_aligned_layout(student, teacher)
    assert layout.bounds == ((0, 2, 0, 3),)
    assert layout.span_mask == (True,)

    second = build_aligned_layout(
        TokenSequence(text="a", token_ids=(1,), pieces=(b"a",)),
        TokenSequence(text="a", token_ids=(2,), pieces=(b"a",)),
    )
    bounds, unit_mask, span_mask = pad_layouts((layout, second))
    assert bounds == [[[0, 2, 0, 3]], [[0, 1, 0, 1]]]
    assert unit_mask == [[1.0], [1.0]]
    assert span_mask == [[True], [False]]
