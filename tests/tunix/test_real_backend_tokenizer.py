from __future__ import annotations

import pytest

from vdt_tunix.real_backend import (
    RealBackendUnavailable,
    _normalize_tokenizer_padding,
)


class _SentencePieceLike:
    def pad_id(self) -> int:
        return 0

    def eos_id(self) -> int:
        return 1

    def __getattr__(self, name: str):
        if name in {"pad_token_id", "pad_token", "eos_token"}:
            raise AssertionError(f"SentencePiece must not be queried for {name}")
        raise AttributeError(name)


class _HuggingFaceLike:
    def __init__(self, *, pad_token_id=None, eos_token="</s>") -> None:
        self.pad_token_id = pad_token_id
        self.eos_token = eos_token
        self.pad_token = None


def test_sentencepiece_padding_uses_native_numeric_ids() -> None:
    tokenizer = _SentencePieceLike()

    normalized = _normalize_tokenizer_padding(tokenizer, "sentencepiece")

    assert normalized is tokenizer
    assert normalized.pad_id() == 0
    assert normalized.eos_id() == 1


def test_huggingface_padding_falls_back_to_eos_token() -> None:
    tokenizer = _HuggingFaceLike()

    normalized = _normalize_tokenizer_padding(tokenizer, "huggingface")

    assert normalized is tokenizer
    assert tokenizer.pad_token == "</s>"


def test_huggingface_padding_preserves_existing_pad_token() -> None:
    tokenizer = _HuggingFaceLike(pad_token_id=7)

    _normalize_tokenizer_padding(tokenizer, "huggingface")

    assert tokenizer.pad_token is None


def test_huggingface_padding_requires_eos_fallback() -> None:
    tokenizer = _HuggingFaceLike(eos_token=None)

    with pytest.raises(RealBackendUnavailable, match="neither a pad token"):
        _normalize_tokenizer_padding(tokenizer, "huggingface")
