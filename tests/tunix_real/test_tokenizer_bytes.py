"""Lossless byte alignment, including token boundaries inside UTF-8 scalars."""

from __future__ import annotations

import dataclasses
import io
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from vdt_span.alignment import minimal_joint_segments
from vdt_tunix.contracts import StudentRolloutSample
from vdt_tunix.model_adapters import ModelAdapterError, TokenizerByteAdapter


def _byte_alphabet():
    visible = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    alphabet = {value: chr(value) for value in visible}
    for index, value in enumerate(value for value in range(256) if value not in alphabet):
        alphabet[value] = chr(256 + index)
    return alphabet


class ByteTokenizer:
    """ByteLevel API double with a real lossy intermediate UTF-8 decode."""

    pad_token_id = 0
    eos_token_id = 1
    all_special_ids = (0, 1)

    def __init__(self):
        self.alphabet = _byte_alphabet()
        self.backend_tokenizer = SimpleNamespace(
            decoder=SimpleNamespace(__getstate__=lambda: b'{"type":"ByteLevel"}'),
            id_to_token=lambda value: self.alphabet[value - 2],
        )

    def encode(self, text, **kwargs):
        return [value + 2 for value in text.encode("utf-8")]

    def decode(self, ids, **kwargs):
        return bytes(value - 2 for value in ids).decode("utf-8", errors="replace")


class SentencePieceTokenizer(ByteTokenizer):
    """Tunix delegates id_to_piece/is_byte to its SentencePieceProcessor."""

    def __init__(self):
        super().__init__()
        del self.backend_tokenizer

    def is_byte(self, value):
        return value >= 130

    def id_to_piece(self, value):
        return f"<0x{value - 2:02X}>" if self.is_byte(value) else chr(value - 2)

    def is_unknown(self, value):
        return False

    def is_control(self, value):
        return value in self.all_special_ids


def _adapter(real_config, kind):
    tokenizer = ByteTokenizer() if kind == "huggingface" else SentencePieceTokenizer()
    config = dataclasses.replace(real_config.student, tokenizer_type=kind)
    return TokenizerByteAdapter(tokenizer, config)


@pytest.mark.parametrize("kind", ["huggingface", "sentencepiece"])
@pytest.mark.parametrize("text", ["é", "€", "🙂", "a🙂é\n中", "\ufffd"])
def test_partial_unicode_prefixes_preserve_every_original_token(real_config, kind, text):
    adapter = _adapter(real_config, kind)
    prompt_ids = adapter.encode("P:")
    ids = adapter.encode(text)
    sequence = adapter.continuation_from_generated_ids(
        prompt_text="P:", prompt_token_ids=prompt_ids, completion_token_ids=ids
    )
    assert sequence.text == text
    assert sequence.token_ids == ids
    assert sequence.pieces == tuple(bytes([value]) for value in text.encode("utf-8"))
    # A multi-token Unicode scalar remains one aligned unit against a single
    # teacher token. No token/log-probability position may be dropped or shifted.
    units = minimal_joint_segments((text.encode("utf-8"),), sequence.pieces)
    assert len(units) == 1
    assert units[0].student_start == 0
    assert units[0].student_end == len(ids)
    log_probs = tuple(-0.1 * (index + 1) for index in range(len(ids)))
    sample = StudentRolloutSample(
        sample_id="unicode-sample", prompt_id="prompt", student_prompt_token_ids=prompt_ids,
        completion=sequence, rollout_log_probs=log_probs,
    )
    assert sample.rollout_log_probs == log_probs
    assert len(sample.rollout_log_probs) == len(sequence.pieces)


@pytest.mark.parametrize("kind", ["huggingface", "sentencepiece"])
def test_unicode_inside_prompt_does_not_force_causal_split(real_config, kind):
    adapter = _adapter(real_config, kind)
    prompt, completion = adapter.tokenize_continuation(
        prompt_text="é:", completion_text="🙂"
    )
    assert prompt == adapter.encode("é:")
    assert completion.text == "🙂"
    assert adapter.last_continuation_tokenization_mode == "joint_exact_boundary"


@pytest.mark.parametrize("kind", ["huggingface", "sentencepiece"])
@pytest.mark.parametrize("raw", [b"\xc3", b"\xff", b"\xc3A", b"\xed\xa0\x80"])
def test_invalid_or_truncated_utf8_is_not_replaced_or_dropped(real_config, kind, raw):
    adapter = _adapter(real_config, kind)
    with pytest.raises(ModelAdapterError, match="complete UTF-8"):
        adapter.continuation_from_generated_ids(
            prompt_text="P:", prompt_token_ids=adapter.encode("P:"),
            completion_token_ids=tuple(value + 2 for value in raw),
        )


@pytest.mark.parametrize("kind", ["huggingface", "sentencepiece"])
def test_reconstructed_bytes_must_match_contextual_decoder(real_config, kind):
    adapter = _adapter(real_config, kind)
    tokenizer = adapter.raw_tokenizer
    original = tokenizer.decode
    tokenizer.decode = lambda ids, **kw: original(ids).replace("P:A", "P:B")
    with pytest.raises(ModelAdapterError, match="raw token bytes.*contextual decode"):
        adapter.continuation_from_generated_ids(
            prompt_text="P:", prompt_token_ids=adapter.encode("P:"),
            completion_token_ids=tokenizer.encode("A"),
        )


def test_unrecognized_decoder_retains_fail_closed_prefix_guard(real_config):
    tokenizer = ByteTokenizer()
    del tokenizer.backend_tokenizer
    adapter = TokenizerByteAdapter(tokenizer, real_config.student)
    with pytest.raises(ModelAdapterError, match="prefix-monotone"):
        adapter.continuation_from_generated_ids(
            prompt_text="P:", prompt_token_ids=adapter.encode("P:"),
            completion_token_ids=adapter.encode("é"),
        )


def test_sentencepiece_metaspace_in_continuation_is_not_stripped(real_config):
    tokenizer = SentencePieceTokenizer()
    original_piece = tokenizer.id_to_piece
    tokenizer.id_to_piece = lambda value: "▁" if value == 34 else original_piece(value)
    adapter = TokenizerByteAdapter(
        tokenizer, dataclasses.replace(real_config.student, tokenizer_type="sentencepiece")
    )
    sequence = adapter.continuation_from_generated_ids(
        prompt_text="P:", prompt_token_ids=adapter.encode("P:"),
        completion_token_ids=adapter.encode(" é"),
    )
    assert sequence.text == " é"
    assert sequence.pieces == (b" ", b"\xc3", b"\xa9")


@pytest.mark.parametrize("kind", ["huggingface", "sentencepiece"])
def test_zero_byte_piece_is_still_rejected(real_config, kind):
    adapter = _adapter(real_config, kind)
    tokenizer = adapter.raw_tokenizer
    if kind == "huggingface":
        tokenizer.backend_tokenizer.id_to_token = lambda value: ""
    else:
        tokenizer.id_to_piece = lambda value: ""
    with pytest.raises(ModelAdapterError, match="zero-byte"):
        adapter.continuation_from_generated_ids(
            prompt_text="P:", prompt_token_ids=adapter.encode("P:"),
            completion_token_ids=(67,),
        )


def test_sentencepiece_literal_byte_spelling_is_not_interpreted_as_byte(real_config):
    tokenizer = SentencePieceTokenizer()
    tokenizer.id_to_piece = lambda value: "<0x61>"
    original = tokenizer.decode
    tokenizer.decode = lambda ids, **kw: original(ids).replace("P:A", "P:<0x61>")
    adapter = TokenizerByteAdapter(
        tokenizer, dataclasses.replace(real_config.student, tokenizer_type="sentencepiece")
    )
    sequence = adapter.continuation_from_generated_ids(
        prompt_text="P:", prompt_token_ids=adapter.encode("P:"), completion_token_ids=(67,)
    )
    assert sequence.pieces == (b"<0x61>",)


@pytest.mark.parametrize("predicate", ["is_unknown", "is_control", "is_unused"])
def test_sentencepiece_non_text_tokens_fail_closed(real_config, predicate):
    adapter = _adapter(real_config, "sentencepiece")
    setattr(adapter.raw_tokenizer, predicate, lambda value: value == 67)
    with pytest.raises(ModelAdapterError, match="non-text SentencePiece"):
        adapter.continuation_from_generated_ids(
            prompt_text="P:", prompt_token_ids=adapter.encode("P:"), completion_token_ids=(67,)
        )


class CoreTokenizerFacade:
    """HF fast-tokenizer surface around a real Rust Tokenizers backend."""

    def __init__(self, core):
        self.backend_tokenizer = core
        self.pad_token_id = core.token_to_id("<pad>")
        self.eos_token_id = core.token_to_id("<eos>")
        self.all_special_ids = (self.pad_token_id, self.eos_token_id)

    def encode(self, text, **kwargs):
        return self.backend_tokenizer.encode(text, add_special_tokens=False).ids

    def decode(self, ids, **kwargs):
        return self.backend_tokenizer.decode(ids, skip_special_tokens=False)


def test_real_bytelevel_decoder_and_added_literal_token(real_config):
    tokenizers = pytest.importorskip("tokenizers")
    vocab = {"<pad>": 0, "<eos>": 1}
    vocab.update({char: value + 2 for value, char in _byte_alphabet().items()})
    core = tokenizers.Tokenizer(tokenizers.models.BPE(vocab, []))
    core.decoder = tokenizers.decoders.ByteLevel()
    core.pre_tokenizer = tokenizers.pre_tokenizers.ByteLevel(add_prefix_space=False)
    core.add_tokens([tokenizers.AddedToken("漢Ġ", normalized=False)])
    tokenizer = CoreTokenizerFacade(core)
    adapter = TokenizerByteAdapter(tokenizer, real_config.teacher)
    for text in ("🙂é\n中", "\ufffd", "漢Ġ"):
        ids = adapter.encode(text)
        sequence = adapter.continuation_from_generated_ids(
            prompt_text="P:", prompt_token_ids=adapter.encode("P:"), completion_token_ids=ids
        )
        assert sequence.text == text
        assert sequence.token_ids == ids
        assert b"".join(sequence.pieces) == text.encode("utf-8")
    assert sequence.pieces == ("漢Ġ".encode("utf-8"),)


def test_real_sentencepiece_byte_fallback_with_tunix_style_delegation(real_config):
    spm = pytest.importorskip("sentencepiece")
    # In-memory tiny vocabulary forces non-ASCII to byte fallback. No model
    # download, disk model, training weights, JAX or TPU is involved.
    model = io.BytesIO()
    spm.SentencePieceTrainer.train(
        sentence_iterator=iter(["Prompt: hello world", "Plain ASCII text"]),
        model_writer=model, model_type="bpe", vocab_size=300,
        byte_fallback=True, hard_vocab_limit=False, pad_id=3,
        minloglevel=2, normalization_rule_name="identity",
        remove_extra_whitespaces=False,
    )
    processor = spm.SentencePieceProcessor(model_proto=model.getvalue())

    class TunixStyleWrapper:
        def __getattr__(self, name):
            return getattr(processor, name)

        def decode(self, ids):
            return processor.decode(ids)

        def encode(self, text):
            return processor.encode(text)

    adapter = TokenizerByteAdapter(
        TunixStyleWrapper(),
        dataclasses.replace(real_config.student, tokenizer_type="sentencepiece"),
    )
    prompt_ids = adapter.encode("Prompt:")
    for text in ("é", "🙂", "\ufffd", "  é\n", "hello🙂world"):
        # Generated IDs need not be a fresh canonical encode of the completion:
        # preserve the sampler's IDs rather than introducing a dummy-space token.
        ids = tuple(processor.piece_to_id(f"<0x{value:02X}>") for value in text.encode())
        assert processor.decode(list(prompt_ids + ids)) == "Prompt:" + text
        sequence = adapter.continuation_from_generated_ids(
            prompt_text="Prompt:", prompt_token_ids=prompt_ids, completion_token_ids=ids
        )
        assert sequence.text == text
        assert sequence.pieces == tuple(bytes([value]) for value in text.encode())
        assert sequence.token_ids == ids
    for text in ("hello", " hello world", "  hello\nworld", "é🙂"):
        prefix_ids, sequence = adapter.tokenize_continuation(
            prompt_text="Prompt:", completion_text=text
        )
        assert sequence.text == text
        assert processor.decode(list(prefix_ids + sequence.token_ids)) == "Prompt:" + text


def test_cached_qwen_tokenizer_offline_regression(real_config):
    path = os.environ.get("VDT_TEST_QWEN_TOKENIZER_JSON")
    if not path:
        pytest.skip("set VDT_TEST_QWEN_TOKENIZER_JSON to a local pinned tokenizer.json")
    tokenizers = pytest.importorskip("tokenizers")
    assert Path(path).is_file()
    core = tokenizers.Tokenizer.from_file(path)
    tokenizer = CoreTokenizerFacade(core)
    tokenizer.pad_token_id = core.token_to_id("<|endoftext|>")
    tokenizer.eos_token_id = core.token_to_id("<|im_end|>")
    tokenizer.all_special_ids = (tokenizer.pad_token_id, tokenizer.eos_token_id)
    adapter = TokenizerByteAdapter(tokenizer, real_config.teacher)
    texts = ("é€🙂中", "Việt Nam 🇻🇳", "🧬🫠𐍈", " a\n\t b", "\ufffd")
    for prompt in ("P:", "🙂:", "<|im_start|>assistant\n"):
        for text in texts:
            ids = adapter.encode(text)
            sequence = adapter.continuation_from_generated_ids(
                prompt_text=prompt, prompt_token_ids=adapter.encode(prompt),
                completion_token_ids=ids,
            )
            assert sequence.text == text
            assert sequence.token_ids == ids
            assert b"".join(sequence.pieces) == text.encode()
            _, retokenized = adapter.tokenize_continuation(
                prompt_text=prompt, completion_text=text
            )
            assert retokenized.text == text
    # Qwen's 256 native byte tokens let us test arbitrary *generated* IDs too,
    # including valid alternative segmentations of the same Unicode text.
    for text in ("🙂é", "🙂" * 960):  # second case: 3840 generated byte tokens
        ids = tuple(core.token_to_id(_byte_alphabet()[value]) for value in text.encode())
        assert all(value is not None for value in ids)
        sequence = adapter.continuation_from_generated_ids(
            prompt_text="P:", prompt_token_ids=adapter.encode("P:"), completion_token_ids=ids
        )
        assert sequence.token_ids == ids
        assert sequence.pieces == tuple(bytes([value]) for value in text.encode())
        assert sequence.text == text


def test_cached_qwen_transformers_fast_wrapper_offline(real_config):
    path = os.environ.get("VDT_TEST_QWEN_TOKENIZER_JSON")
    if not path:
        pytest.skip("set VDT_TEST_QWEN_TOKENIZER_JSON to a local pinned tokenizer.json")
    transformers = pytest.importorskip("transformers")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        str(Path(path).parent), local_files_only=True, trust_remote_code=False, token=False,
    )
    assert tokenizer.is_fast
    adapter = TokenizerByteAdapter(tokenizer, real_config.teacher)
    assert adapter._byte_decoder == "byte_level"
    for prompt, text in (("P:", "hello world"), ("🙂:", "🧬🫠𐍈 Việt Nam 🇻🇳")):
        prompt_ids, completion = adapter.tokenize_continuation(
            prompt_text=prompt, completion_text=text,
        )
        assert completion.text == text
        assert tokenizer.decode(list(prompt_ids + completion.token_ids)) == prompt + text
        # On already prefix-monotone text the new pieces equal the old ones.
        if text == "hello world":
            previous = prompt.encode()
            old_pieces = []
            for width in range(1, len(completion.token_ids) + 1):
                decoded = adapter._decode(prompt_ids + completion.token_ids[:width]).encode()
                old_pieces.append(decoded[len(previous):])
                previous = decoded
            assert completion.pieces == tuple(old_pieces)
