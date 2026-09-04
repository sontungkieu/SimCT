"""Lossless SimCT atom construction for MP-OPD.

This module deliberately does not reuse SpanCTKD's character-length alignment:
MP-OPD needs byte-identical, response-only atomic units and must fail closed
when that contract cannot be established.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Optional, Sequence


class AtomizationFailure(str, Enum):
    EMPTY_RESPONSE = "empty_response"
    EMPTY_DECODE = "empty_decode"
    REPLACEMENT_CHARACTER = "replacement_character"
    NORMALIZATION_MISMATCH = "normalization_mismatch"
    UNSUPPORTED_ADDED_TOKEN = "unsupported_added_token"
    UNALIGNED_SUFFIX = "unaligned_suffix"


@dataclass(frozen=True)
class MPAtom:
    sample_id: str
    student_start: int
    student_end: int
    teacher_start: int
    teacher_end: int
    byte_start: int
    byte_end: int
    student_token_count: int
    teacher_token_count: int
    valid: bool = True
    failure_reason: Optional[str] = None
    boundary_type: str = "one_to_one"
    lT: Optional[float] = None
    lS_old: Optional[float] = None
    b: Optional[float] = None
    w: Optional[int] = None
    r: Optional[float] = None

    def with_credit(self, *, lT: float, lS_old: float, b: float, r: float) -> "MPAtom":
        return replace(self, lT=lT, lS_old=lS_old, b=b, w=self.student_token_count, r=r)


@dataclass(frozen=True)
class AtomizationResult:
    atoms: tuple[MPAtom, ...]
    valid: bool
    failure_reason: Optional[str]
    covered_student_events: int
    covered_teacher_events: int
    masked_student_eos: int = 0
    masked_teacher_eos: int = 0


def _decode(tokenizer: Any, ids: Sequence[int]) -> str:
    try:
        return tokenizer.decode(
            list(ids), skip_special_tokens=False, clean_up_tokenization_spaces=False
        )
    except TypeError:
        return tokenizer.decode(list(ids), skip_special_tokens=False)


def _added_ids(tokenizer: Any) -> set[int]:
    getter = getattr(tokenizer, "get_added_vocab", None)
    if getter is None:
        return set()
    return {int(value) for value in getter().values()}


class SimCTAtomizer:
    """Construct minimal synchronized response segments with exact byte checks.

    EOS is masked by default. It is not decoded into response text because two
    tokenizers need not share a lossless representation of the stopping event.
    Padding must already have been removed by the caller's loss mask.
    """

    def __init__(self, student_tokenizer: Any, teacher_tokenizer: Any, eos_policy: str = "mask"):
        if eos_policy != "mask":
            raise ValueError("MP-OPD v0 supports only eos_policy='mask'")
        self.student_tokenizer = student_tokenizer
        self.teacher_tokenizer = teacher_tokenizer
        self.eos_policy = eos_policy

    @staticmethod
    def _mask_terminal_eos(ids: list[int], eos_id: Optional[int]) -> tuple[list[int], int]:
        count = 0
        if eos_id is not None:
            while ids and ids[-1] == int(eos_id):
                ids.pop()
                count += 1
        return ids, count

    def atomize(
        self,
        student_label_ids: Sequence[int],
        teacher_label_ids: Sequence[int],
        *,
        sample_id: str,
    ) -> AtomizationResult:
        stu_ids, stu_eos = self._mask_terminal_eos(
            [int(x) for x in student_label_ids],
            getattr(self.student_tokenizer, "eos_token_id", None),
        )
        tea_ids, tea_eos = self._mask_terminal_eos(
            [int(x) for x in teacher_label_ids],
            getattr(self.teacher_tokenizer, "eos_token_id", None),
        )
        if not stu_ids or not tea_ids:
            return AtomizationResult((), False, AtomizationFailure.EMPTY_RESPONSE.value, 0, 0, stu_eos, tea_eos)

        stu_added = _added_ids(self.student_tokenizer)
        tea_added = _added_ids(self.teacher_tokenizer)
        if any(x in stu_added for x in stu_ids) or any(x in tea_added for x in tea_ids):
            return AtomizationResult(
                (), False, AtomizationFailure.UNSUPPORTED_ADDED_TOKEN.value, 0, 0, stu_eos, tea_eos
            )

        stu_prefix = [_decode(self.student_tokenizer, stu_ids[:i]) for i in range(1, len(stu_ids) + 1)]
        tea_prefix = [_decode(self.teacher_tokenizer, tea_ids[:i]) for i in range(1, len(tea_ids) + 1)]
        if any("\ufffd" in value for value in (*stu_prefix, *tea_prefix)):
            return AtomizationResult(
                (), False, AtomizationFailure.REPLACEMENT_CHARACTER.value, 0, 0, stu_eos, tea_eos
            )
        if not stu_prefix[-1] or not tea_prefix[-1]:
            return AtomizationResult((), False, AtomizationFailure.EMPTY_DECODE.value, 0, 0, stu_eos, tea_eos)
        if stu_prefix[-1].encode("utf-8") != tea_prefix[-1].encode("utf-8"):
            return AtomizationResult(
                (), False, AtomizationFailure.NORMALIZATION_MISMATCH.value, 0, 0, stu_eos, tea_eos
            )

        atoms: list[MPAtom] = []
        si = ti = 0
        prev_si = prev_ti = 0
        prev_bytes = b""
        while si < len(stu_ids) and ti < len(tea_ids):
            sb = stu_prefix[si].encode("utf-8")
            tb = tea_prefix[ti].encode("utf-8")
            if sb == tb:
                if len(sb) <= len(prev_bytes):
                    return AtomizationResult(
                        (), False, AtomizationFailure.EMPTY_DECODE.value, 0, 0, stu_eos, tea_eos
                    )
                stu_count = si + 1 - prev_si
                tea_count = ti + 1 - prev_ti
                atoms.append(
                    MPAtom(
                        sample_id=str(sample_id),
                        student_start=prev_si,
                        student_end=si + 1,
                        teacher_start=prev_ti,
                        teacher_end=ti + 1,
                        byte_start=len(prev_bytes),
                        byte_end=len(sb),
                        student_token_count=stu_count,
                        teacher_token_count=tea_count,
                        boundary_type="one_to_one" if stu_count == tea_count == 1 else "multi_token",
                    )
                )
                prev_si, prev_ti, prev_bytes = si + 1, ti + 1, sb
                si += 1
                ti += 1
            elif len(sb) < len(tb) and tb.startswith(sb):
                si += 1
            elif len(tb) < len(sb) and sb.startswith(tb):
                ti += 1
            elif len(sb) <= len(tb):
                si += 1
            else:
                ti += 1

        if prev_si != len(stu_ids) or prev_ti != len(tea_ids):
            return AtomizationResult(
                (), False, AtomizationFailure.UNALIGNED_SUFFIX.value, 0, 0, stu_eos, tea_eos
            )
        return AtomizationResult(
            tuple(atoms), True, None, len(stu_ids), len(tea_ids), stu_eos, tea_eos
        )
