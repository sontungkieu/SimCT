from __future__ import annotations

import pytest

from vdt_tunix.real_backend import (
    RealBackendUnavailable,
    _call_native_tunix_model,
    _native_model_family,
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


class _ModelCallRecorder:
    def __init__(self) -> None:
        self.args = None
        self.kwargs = None

    def __call__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        return "logits", "cache"


@pytest.mark.parametrize(
    ("model_id", "family"),
    [
        ("gemma-2-2b-it", "gemma2"),
        ("qwen2.5-7b-instruct", "qwen2p5"),
    ],
)
def test_native_model_family_is_pinned(model_id: str, family: str) -> None:
    assert _native_model_family(model_id) == family


def test_native_model_family_rejects_unimplemented_signature() -> None:
    with pytest.raises(RealBackendUnavailable, match="Gemma 2 and Qwen 2.5"):
        _native_model_family("llama-3.1-8b")


def test_gemma2_forward_uses_attention_mask_without_segment_keyword() -> None:
    model = _ModelCallRecorder()

    logits = _call_native_tunix_model(
        model,
        family="gemma2",
        input_ids="ids",
        positions="positions",
        attention_mask="causal-mask",
        segments="segments",
    )

    assert logits == "logits"
    assert model.args == ("ids", "positions", None, "causal-mask")
    assert model.kwargs == {}


def test_qwen25_forward_uses_native_segment_ids_keyword() -> None:
    model = _ModelCallRecorder()

    logits = _call_native_tunix_model(
        model,
        family="qwen2p5",
        input_ids="ids",
        positions="positions",
        attention_mask="causal-mask",
        segments="segments",
    )

    assert logits == "logits"
    assert model.args == ("ids", "positions", None, "causal-mask")
    assert model.kwargs == {"segment_ids": "segments"}


def test_native_forward_rejects_unknown_family() -> None:
    with pytest.raises(RealBackendUnavailable, match="unsupported native"):
        _call_native_tunix_model(
            _ModelCallRecorder(),
            family="unknown",
            input_ids="ids",
            positions="positions",
            attention_mask="causal-mask",
            segments="segments",
        )
