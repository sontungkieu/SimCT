import pytest

from vdt_span.alignment import (
    AlignmentError,
    coarsen_aligned_units,
    minimal_joint_segments,
)


def test_minimal_joint_segments_many_to_many() -> None:
    units = minimal_joint_segments(["h", "appy"], ["ha", "pp", "y"])
    assert len(units) == 1
    unit = units[0]
    assert (unit.teacher_start, unit.teacher_end) == (0, 2)
    assert (unit.student_start, unit.student_end) == (0, 3)
    assert (unit.byte_start, unit.byte_end) == (0, 5)


def test_minimal_joint_segments_find_all_common_boundaries() -> None:
    units = minimal_joint_segments(["I", " am", " ok"], ["I", " ", "am", " ok"])
    assert [
        (unit.teacher_start, unit.teacher_end, unit.student_start, unit.student_end)
        for unit in units
    ] == [
        (0, 1, 0, 1),
        (1, 2, 1, 3),
        (2, 3, 3, 4),
    ]


def test_alignment_uses_utf8_byte_offsets() -> None:
    units = minimal_joint_segments(["A", "😊", "B"], ["A😊", "B"])
    assert [(unit.byte_start, unit.byte_end) for unit in units] == [(0, 5), (5, 6)]


def test_mismatched_decoded_text_fails_closed() -> None:
    with pytest.raises(AlignmentError, match="offset 1"):
        minimal_joint_segments(["abc"], ["axc"])


def test_zero_byte_piece_fails_closed() -> None:
    with pytest.raises(AlignmentError, match="zero-byte"):
        minimal_joint_segments(["a", ""], ["a"])


def test_coarsen_aligned_units_preserves_outer_offsets() -> None:
    units = minimal_joint_segments(["I", " am", " ok"], ["I", " ", "am", " ok"])
    merged = coarsen_aligned_units(units, [(0, 2), (2, 3)])
    assert len(merged) == 2
    assert (merged[0].teacher_start, merged[0].teacher_end) == (0, 2)
    assert (merged[0].student_start, merged[0].student_end) == (0, 3)
    assert (merged[0].byte_start, merged[0].byte_end) == (0, 4)
