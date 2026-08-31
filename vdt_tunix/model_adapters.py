"""Dependency-injected tokenizer and causal-model adapters.

The production dependency bundle is assembled in :mod:`vdt_tunix.real_backend`.
Keeping the contracts here framework-light lets CPU fakes exercise the exact
text/token/forward boundary without pretending to load real weights.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
import dataclasses
import json
from typing import Any

from vdt_tunix.config import ModelConfig
from vdt_tunix.contracts import TokenSequence


class ModelAdapterError(RuntimeError):
    """Raised when tokenizer or model behavior cannot satisfy the contract."""


@dataclasses.dataclass(frozen=True, slots=True)
class ModelRuntimeDependencies:
    """Injectable model-runtime operations used by the real adapters.

    ``production`` is false for CPU fakes.  Only the production dependency
    loader may cause backends to declare ``real_model_integration=True``.
    """

    name: str
    production: bool
    validate_model_spec: Callable[[ModelConfig], None]
    load_tokenizer: Callable[[ModelConfig], Any]
    load_model: Callable[[ModelConfig, bool], Any]
    forward_model: Callable[[Any, Any, Any], Any]
    stop_gradient: Callable[[Any], Any]
    to_host: Callable[[Any], Any]
    forward_sufficient_statistics: Callable[
        [Any, Any, Any, Any, Any, Any], tuple[Any, Any]
    ] | None = None
    forward_cached_sufficient_statistics: Callable[
        [Any, Any, Any, Any, Any, Any], tuple[Any, Any]
    ] | None = None


def _as_token_ids(value: Any) -> tuple[int, ...]:
    if isinstance(value, dict):
        value = value.get("input_ids")
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, tuple):
        value = list(value)
    if value and isinstance(value[0], (list, tuple)):
        if len(value) != 1:
            raise ModelAdapterError("tokenizer returned an unexpected batch")
        value = value[0]
    if not isinstance(value, list) or any(
        isinstance(item, bool) or not isinstance(item, int) for item in value
    ):
        raise ModelAdapterError("tokenizer input_ids must be a list of integers")
    if any(item < 0 for item in value):
        raise ModelAdapterError("tokenizer input_ids must be non-negative")
    return tuple(value)


def _byte_level_inverse_alphabet() -> dict[str, int]:
    """Invert the GPT-2/Tokenizers ByteLevel bijection, not Unicode decoding."""

    visible = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    result = {chr(value): value for value in visible}
    for index, value in enumerate(value for value in range(256) if value not in visible):
        result[chr(256 + index)] = value
    return result


_BYTE_LEVEL_INVERSE = _byte_level_inverse_alphabet()


class TokenizerByteAdapter:
    """Expose one exact byte piece per token, even inside a UTF-8 character."""

    def __init__(self, tokenizer: Any, config: ModelConfig):
        self._tokenizer = tokenizer
        self.config = config
        self.last_continuation_tokenization_mode = "not_run"
        self.pad_token_id = self._special_id("pad_token_id", "pad_id")
        self.eos_token_id = self._special_id("eos_token_id", "eos_id")
        self.bos_token_id = self._optional_special_id("bos_token_id", "bos_id")
        special = getattr(tokenizer, "all_special_ids", ()) or ()
        self.special_token_ids = frozenset(int(value) for value in special)
        explicit_special = {self.pad_token_id, self.eos_token_id}
        if self.bos_token_id is not None:
            explicit_special.add(self.bos_token_id)
        self.special_token_ids = self.special_token_ids | explicit_special
        self._byte_decoder = self._detect_byte_decoder()

    def _detect_byte_decoder(self) -> str:
        # Tunix delegates these native SentencePiece APIs to its processor.
        if self.config.tokenizer_type == "sentencepiece" and all(
            callable(getattr(self._tokenizer, name, None))
            for name in ("id_to_piece", "is_byte")
        ):
            return "sentencepiece"
        backend = getattr(self._tokenizer, "backend_tokenizer", None)
        decoder = getattr(backend, "decoder", None)
        if decoder is not None and callable(getattr(backend, "id_to_token", None)):
            try:
                state = json.loads(decoder.__getstate__())
            except (TypeError, ValueError, AttributeError):
                state = None
            if isinstance(state, dict) and state.get("type") == "ByteLevel":
                return "byte_level"
        # Do not guess raw-byte semantics from a token's spelling. Unknown
        # decoder pipelines retain the original strict prefix-difference path.
        return "prefix_decode"

    def _alignment_error(self, message: str) -> ModelAdapterError:
        # Deliberately exclude text, token IDs, paths and operational secrets.
        return ModelAdapterError(f"{message} (byte_decoder={self._byte_decoder})")

    def _raw_token_pieces(self, token_ids: tuple[int, ...]) -> tuple[bytes, ...] | None:
        if self._byte_decoder == "prefix_decode":
            return None
        pieces = []
        for index, token_id in enumerate(token_ids):
            if self._byte_decoder == "byte_level":
                token = self._tokenizer.backend_tokenizer.id_to_token(token_id)
                if not isinstance(token, str):
                    raise self._alignment_error(
                        f"missing token spelling at token_index={index}"
                    )
                # Tokenizers' ByteLevel decoder falls back to the whole literal
                # token if any character is outside its alphabet (added tokens).
                piece = (
                    bytes(_BYTE_LEVEL_INVERSE[char] for char in token)
                    if all(char in _BYTE_LEVEL_INVERSE for char in token)
                    else token.encode("utf-8")
                )
            else:
                for name in ("is_control", "is_unknown", "is_unused"):
                    predicate = getattr(self._tokenizer, name, None)
                    if callable(predicate) and predicate(token_id):
                        raise self._alignment_error(
                            f"non-text SentencePiece token at token_index={index}"
                        )
                token = self._tokenizer.id_to_piece(token_id)
                if not isinstance(token, str):
                    raise self._alignment_error(
                        f"missing token spelling at token_index={index}"
                    )
                if self._tokenizer.is_byte(token_id):
                    if not (
                        len(token) == 6
                        and token.startswith("<0x")
                        and token.endswith(">")
                        and all(char in "0123456789abcdefABCDEF" for char in token[3:5])
                    ):
                        raise self._alignment_error(
                            f"invalid SentencePiece byte spelling at token_index={index}"
                        )
                    piece = bytes([int(token[3:5], 16)])
                else:
                    # These are continuation pieces after a nonempty prompt;
                    # SentencePiece's initial dummy-space removal does not apply.
                    piece = token.replace("▁", " ").encode("utf-8")
            if not piece:
                raise self._alignment_error(
                    f"zero-byte completion token at token_index={index}"
                )
            pieces.append(piece)
        return tuple(pieces)

    @property
    def raw_tokenizer(self) -> Any:
        return self._tokenizer

    def _special_id(self, attribute: str, method: str) -> int:
        value = getattr(self._tokenizer, attribute, None)
        if value is None:
            candidate = getattr(self._tokenizer, method, None)
            value = candidate() if callable(candidate) else candidate
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ModelAdapterError(
                f"tokenizer must expose a non-negative {attribute}"
            )
        return int(value)

    def _optional_special_id(self, attribute: str, method: str) -> int | None:
        value = getattr(self._tokenizer, attribute, None)
        if value is None:
            candidate = getattr(self._tokenizer, method, None)
            value = candidate() if callable(candidate) else candidate
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ModelAdapterError(
                f"tokenizer must expose a non-negative {attribute} or None"
            )
        return int(value)

    def with_model_prefix(self, text_token_ids: Sequence[int]) -> tuple[int, ...]:
        """Add exactly the BOS prefix used by Tunix's native sampler.

        Loss-bearing text token IDs deliberately exclude zero-byte special
        tokens so byte alignment remains exact.  Native model forwards still
        need the same BOS-conditioned state used by ``Sampler.tokenize``.
        Tunix treats BOS id zero as disabled and de-duplicates an already
        present leading BOS; mirror those semantics here.
        """

        token_ids = _as_token_ids(list(text_token_ids))
        bos_id = self.bos_token_id
        if not bos_id:
            return token_ids
        prefixed = (bos_id, *token_ids)
        while len(prefixed) > 1 and prefixed[0] == prefixed[1] == bos_id:
            prefixed = prefixed[1:]
        return prefixed

    def encode_model_prompt(self, text: str) -> tuple[int, ...]:
        """Encode text and add the native model prefix without changing text IDs."""

        return self.with_model_prefix(self.encode(text))

    def _decode(self, token_ids: Sequence[int]) -> str:
        try:
            value = self._tokenizer.decode(
                list(token_ids),
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
        except TypeError:
            value = self._tokenizer.decode(list(token_ids))
        if not isinstance(value, str):
            raise ModelAdapterError("tokenizer.decode must return str")
        return value

    def encode(self, text: str) -> tuple[int, ...]:
        if not isinstance(text, str) or not text:
            raise ModelAdapterError("tokenizer input text must be non-empty")
        try:
            value = self._tokenizer.encode(text, add_special_tokens=False)
        except TypeError:
            value = self._tokenizer.encode(text)
        token_ids = _as_token_ids(value)
        if not token_ids:
            raise ModelAdapterError("tokenizer produced no tokens")
        if self._decode(token_ids) != text:
            raise ModelAdapterError(
                "tokenizer does not losslessly reproduce the requested text"
            )
        return token_ids

    def continuation_from_generated_ids(
        self,
        *,
        prompt_text: str,
        prompt_token_ids: Sequence[int],
        completion_token_ids: Sequence[int],
    ) -> TokenSequence:
        """Decode generated ids in their prompt context into exact byte pieces."""

        prompt_ids = tuple(int(value) for value in prompt_token_ids)
        completion_ids = tuple(int(value) for value in completion_token_ids)
        if not prompt_text or not prompt_ids:
            raise ModelAdapterError("byte alignment requires a nonempty prompt")
        if not completion_ids:
            raise ModelAdapterError("generated completion contains no text tokens")
        if any(value in self.special_token_ids for value in completion_ids):
            raise ModelAdapterError(
                "completion token ids must exclude terminal/special tokens"
            )
        prompt_bytes = prompt_text.encode("utf-8")
        if self._decode(prompt_ids).encode("utf-8") != prompt_bytes:
            raise ModelAdapterError(
                "prompt ids do not reproduce the exact prompt UTF-8 bytes"
            )
        raw_pieces = self._raw_token_pieces(completion_ids)
        if raw_pieces is not None:
            pieces = raw_pieces
            completion_bytes = b"".join(pieces)
        else:
            previous = prompt_bytes
            increments = []
            for width in range(1, len(completion_ids) + 1):
                decoded = self._decode(prompt_ids + completion_ids[:width]).encode("utf-8")
                if not decoded.startswith(previous):
                    raise self._alignment_error(
                        "tokenizer decode is not prefix-monotone at a token boundary; "
                        f"completion_index={width - 1}"
                    )
                piece = decoded[len(previous) :]
                if not piece:
                    raise self._alignment_error(
                        "zero-byte completion token is unsupported by byte alignment; "
                        f"completion_index={width - 1}"
                    )
                increments.append(piece)
                previous = decoded
            pieces = tuple(increments)
            completion_bytes = b"".join(pieces)
        try:
            completion_text = completion_bytes.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise self._alignment_error(
                "completion bytes are not complete UTF-8 text; "
                f"byte_offset={exc.start}, completion_tokens={len(completion_ids)}"
            ) from None
        if self._decode(prompt_ids + completion_ids) != prompt_text + completion_text:
            raise self._alignment_error(
                "raw token bytes do not match the full contextual decode"
            )
        return TokenSequence(
            text=completion_text,
            token_ids=completion_ids,
            pieces=tuple(pieces),
        )

    def tokenize_continuation(
        self,
        *,
        prompt_text: str,
        completion_text: str,
    ) -> tuple[tuple[int, ...], TokenSequence]:
        """Retokenize exact prompt+completion and require a prompt boundary."""

        if not completion_text:
            raise ModelAdapterError("completion text must be non-empty")
        full_ids = self.encode(prompt_text + completion_text)
        prompt_bytes = prompt_text.encode("utf-8")
        boundary = None
        if self._byte_decoder == "byte_level":
            # Prefix text can temporarily contain U+FFFD inside a multibyte
            # character. Find the joint boundary in raw bytes, not that text.
            raw_pieces = self._raw_token_pieces(full_ids)
            assert raw_pieces is not None
            if b"".join(raw_pieces) != (prompt_text + completion_text).encode("utf-8"):
                raise self._alignment_error("joint raw token bytes do not reproduce text")
            offset = 0
            for width, piece in enumerate(raw_pieces, start=1):
                offset += len(piece)
                if offset == len(prompt_bytes):
                    boundary = width
                    break
                if offset > len(prompt_bytes):
                    break
        else:
            for width in range(1, len(full_ids) + 1):
                decoded = self._decode(full_ids[:width]).encode("utf-8")
                if decoded == prompt_bytes:
                    boundary = width
                    break
                # U+FFFD may represent an incomplete prefix, not a mismatch.
                # Do not let it hide an exact boundary after the next token.
                if b"\xef\xbf\xbd" not in decoded and (
                    len(decoded) > len(prompt_bytes)
                    or not prompt_bytes.startswith(decoded)
                ):
                    break
        if boundary is None or boundary == len(full_ids):
            # A byte-level BPE may merge the final prompt bytes with the first
            # completion bytes when the concatenated text is tokenized in one
            # call.  Causal generation cannot use such a cross-boundary token:
            # the prompt tokens already exist before the first completion token
            # is sampled.  Reproduce that causal state by tokenizing the prompt
            # and completion independently, then fail closed unless their
            # concatenated IDs still decode to the exact requested UTF-8 text.
            try:
                prompt_ids = self.encode(prompt_text)
                completion_ids = self.encode(completion_text)
                sequence = self.continuation_from_generated_ids(
                    prompt_text=prompt_text,
                    prompt_token_ids=prompt_ids,
                    completion_token_ids=completion_ids,
                )
            except Exception as exc:
                raise ModelAdapterError(
                    "teacher tokenization has no exact prompt/completion byte "
                    "boundary and causal split tokenization is not lossless"
                ) from exc
            if sequence.text != completion_text:
                raise ModelAdapterError(
                    "teacher causal split tokenization changed the student "
                    "completion text"
                )
            self.last_continuation_tokenization_mode = "causal_split"
            return prompt_ids, sequence
        prompt_ids = full_ids[:boundary]
        completion_ids = full_ids[boundary:]
        sequence = self.continuation_from_generated_ids(
            prompt_text=prompt_text,
            prompt_token_ids=prompt_ids,
            completion_token_ids=completion_ids,
        )
        if sequence.text != completion_text:
            raise ModelAdapterError(
                "teacher retokenization changed the student completion text"
            )
        self.last_continuation_tokenization_mode = "joint_exact_boundary"
        return prompt_ids, sequence


class CausalModelForwardAdapter:
    """Lazy causal forward adapter with explicit student/teacher gradients."""

    def __init__(
        self,
        config: ModelConfig,
        tokenizer: TokenizerByteAdapter,
        dependencies: ModelRuntimeDependencies,
        *,
        trainable: bool,
    ):
        self.config = config
        self.tokenizer = tokenizer
        self.dependencies = dependencies
        self.trainable = bool(trainable)
        self.role = "student" if self.trainable else "teacher"
        self._model: Any | None = None

    @property
    def model_loaded(self) -> bool:
        return self._model is not None

    def _require_model(self) -> Any:
        if self._model is None:
            try:
                self._model = self.dependencies.load_model(
                    self.config,
                    self.trainable,
                )
            except Exception as exc:
                raise ModelAdapterError(
                    f"failed to load {self.role} MaxText model "
                    f"{self.config.model_id}: {type(exc).__name__}: {exc}"
                ) from exc
            if self._model is None:
                raise ModelAdapterError(
                    f"{self.role} model loader returned no model"
                )
        return self._model

    def require_loaded_model(self) -> Any:
        """Materialize and return the framework-owned model wrapper."""

        return self._require_model()

    def forward(self, input_ids: Any, segment_ids: Any) -> Any:
        """Return full causal logits; teacher results are stop-gradient."""

        try:
            logits = self.dependencies.forward_model(
                self._require_model(),
                input_ids,
                segment_ids,
            )
        except ModelAdapterError:
            raise
        except Exception as exc:
            raise ModelAdapterError(
                f"{self.role} forward failed: {type(exc).__name__}: {exc}"
            ) from exc
        if getattr(logits, "ndim", None) != 3:
            raise ModelAdapterError(
                f"{self.role} logits must have [batch, sequence, vocabulary] shape"
            )
        if not self.trainable:
            logits = self.dependencies.stop_gradient(logits)
        return logits

    def forward_sufficient_statistics(
        self,
        input_ids: Any,
        segment_ids: Any,
        completion_positions: Any,
        completion_token_ids: Any,
        overlap_ids: Any,
    ) -> tuple[Any, Any]:
        """Return exact reduced frozen-teacher statistics from one fused JIT."""

        if self.trainable:
            raise ModelAdapterError(
                "sufficient-statistic forward is reserved for the frozen teacher"
            )
        forward = self.dependencies.forward_sufficient_statistics
        if forward is None:
            raise ModelAdapterError(
                "runtime does not implement fused teacher sufficient statistics"
            )
        try:
            shared, selected = forward(
                self._require_model(),
                input_ids,
                segment_ids,
                completion_positions,
                completion_token_ids,
                overlap_ids,
            )
        except ModelAdapterError:
            raise
        except Exception as exc:
            raise ModelAdapterError(
                "teacher sufficient-statistic forward failed: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        if getattr(shared, "ndim", None) != 3:
            raise ModelAdapterError(
                "teacher shared scores must have [batch, token, overlap] shape"
            )
        if getattr(selected, "ndim", None) != 2:
            raise ModelAdapterError(
                "teacher selected scores must have [batch, token] shape"
            )
        if shared.shape[:2] != selected.shape:
            raise ModelAdapterError("teacher sufficient-statistic shapes mismatch")
        return (
            self.dependencies.stop_gradient(shared),
            self.dependencies.stop_gradient(selected),
        )

    def forward_cached_sufficient_statistics(
        self,
        prompt_ids: Any,
        prompt_mask: Any,
        completion_token_ids: Any,
        completion_mask: Any,
        overlap_ids: Any,
    ) -> tuple[Any, Any]:
        """Return exact teacher statistics via prefill plus cached forcing."""

        if self.trainable:
            raise ModelAdapterError(
                "cached sufficient-statistic forward is reserved for the frozen teacher"
            )
        forward = self.dependencies.forward_cached_sufficient_statistics
        if forward is None:
            raise ModelAdapterError(
                "runtime does not implement cached teacher sufficient statistics"
            )
        try:
            shared, selected = forward(
                self._require_model(),
                prompt_ids,
                prompt_mask,
                completion_token_ids,
                completion_mask,
                overlap_ids,
            )
        except ModelAdapterError:
            raise
        except Exception as exc:
            raise ModelAdapterError(
                "cached teacher sufficient-statistic forward failed: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        if getattr(shared, "ndim", None) != 3:
            raise ModelAdapterError(
                "cached teacher shared scores must have [batch, token, overlap] shape"
            )
        if getattr(selected, "ndim", None) != 2:
            raise ModelAdapterError(
                "cached teacher selected scores must have [batch, token] shape"
            )
        if shared.shape[:2] != selected.shape:
            raise ModelAdapterError("cached teacher sufficient-statistic shapes mismatch")
        return (
            self.dependencies.stop_gradient(shared),
            self.dependencies.stop_gradient(selected),
        )

    def to_host(self, value: Any) -> Any:
        return self.dependencies.to_host(value)
