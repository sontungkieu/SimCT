"""Remote frozen-teacher backend that leaves the trainable student on TPU."""

from __future__ import annotations

import dataclasses
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from typing import Any

from vdt_tunix.config import RunConfig
from vdt_tunix.contracts import (
    INTERFACE_CONTRACT_VERSION,
    TeacherScoreBatch,
    TeacherScoreRequest,
    TeacherScoreSample,
    TeacherSufficientStatisticsPayload,
)
from vdt_tunix.model_adapters import ModelAdapterError, TokenizerByteAdapter
from vdt_tunix.performance import PerformanceContractError, select_length_bucket
from vdt_tunix.remote_teacher import (
    RemoteTeacherClient,
    RemoteTeacherError,
    RemoteTeacherProfile,
    RemoteTeacherRuntimeConfig,
    TeacherHiddenStats,
)


@dataclasses.dataclass(slots=True)
class _RemoteTeacherAdapter:
    tokenizer: TokenizerByteAdapter
    dependencies: Any = dataclasses.field(
        default_factory=lambda: SimpleNamespace(production=True)
    )


def _project_remote_statistics(
    hidden_state_bits: Any,
    overlap_head_bits: Any,
    log_normalizer: Any,
    *,
    jax_module: Any,
    jnp_module: Any,
) -> Any:
    """Reconstruct exact shared-token log probabilities on TPU.

    Both matrices carry the native Qwen BF16 bit patterns. The only FP32
    quantity supplied by the server is the full-vocabulary log-normalizer.
    """

    hidden = jax_module.lax.bitcast_convert_type(
        jnp_module.asarray(hidden_state_bits, dtype=jnp_module.uint16),
        jnp_module.bfloat16,
    )
    head = jax_module.lax.bitcast_convert_type(
        jnp_module.asarray(overlap_head_bits, dtype=jnp_module.uint16),
        jnp_module.bfloat16,
    )
    shared_logits = jnp_module.einsum("btd,od->bto", hidden, head).astype(
        jnp_module.float32
    )
    shared = shared_logits - jnp_module.asarray(
        log_normalizer, dtype=jnp_module.float32
    )[..., None]
    return jax_module.lax.stop_gradient(shared)


class RemoteVLLMTeacherScoreBackend:
    """Use one authenticated vLLM teacher for one or more TPU students."""

    backend_name = "remote-vllm-frozen-teacher"
    real_model_integration = True

    def __init__(
        self,
        config: RunConfig,
        tokenizer: TokenizerByteAdapter,
        runtime: RemoteTeacherRuntimeConfig,
        *,
        profile: RemoteTeacherProfile | None = None,
        client: RemoteTeacherClient | None = None,
        jax_module: Any | None = None,
        jnp_module: Any | None = None,
    ) -> None:
        self._config = config
        self.runtime = runtime
        self.profile = profile or RemoteTeacherProfile.load(runtime.profile_dir)
        if self.profile.model_id != config.teacher.tokenizer_id:
            raise RemoteTeacherError(
                "remote teacher model identity does not match teacher tokenizer_id"
            )
        if self.profile.model_revision != config.teacher.model_revision:
            raise RemoteTeacherError(
                "remote teacher revision does not match the scientific config"
            )
        self.client = client or RemoteTeacherClient(runtime, self.profile)
        self.client.health()
        self.model_adapter = _RemoteTeacherAdapter(tokenizer)
        self.last_phase_timings: dict[str, float | int | str] = {}
        self._overlap_token_ids: tuple[int, ...] | None = None
        self._jax = jax_module
        self._jnp = jnp_module
        self._project = None

    def configure_overlap_token_ids(self, token_ids) -> None:
        normalized = tuple(int(value) for value in token_ids)
        if not normalized:
            raise ModelAdapterError("teacher overlap token ids must not be empty")
        self.profile.validate_overlap_ids(normalized)
        self._overlap_token_ids = normalized
        if self._jax is None or self._jnp is None:
            try:
                import jax
                import jax.numpy as jnp
            except ImportError as exc:
                raise ModelAdapterError(
                    "JAX is required to reconstruct remote teacher statistics"
                ) from exc
            self._jax = jax
            self._jnp = jnp

        def project(hidden_bits, head_bits, log_normalizer):
            return _project_remote_statistics(
                hidden_bits,
                head_bits,
                log_normalizer,
                jax_module=self._jax,
                jnp_module=self._jnp,
            )

        self._project = self._jax.jit(project)

    def _tokenize(self, request: TeacherScoreRequest):
        prompt_by_id = {prompt.prompt_id: prompt for prompt in request.prompts}
        rows = []
        mode_counts = {"joint_exact_boundary": 0, "causal_split": 0}
        for rollout in request.rollouts.samples:
            prompt = prompt_by_id[rollout.prompt_id]
            prompt_ids, completion = self.model_adapter.tokenizer.tokenize_continuation(
                prompt_text=prompt.teacher_prompt,
                completion_text=rollout.completion.text,
            )
            model_prompt_ids = self.model_adapter.tokenizer.with_model_prefix(
                prompt_ids
            )
            mode = self.model_adapter.tokenizer.last_continuation_tokenization_mode
            if mode not in mode_counts:
                raise ModelAdapterError(
                    "teacher continuation tokenization mode was not recorded"
                )
            mode_counts[mode] += 1
            if len(model_prompt_ids) > self._config.rollout.max_prompt_tokens:
                raise ModelAdapterError(
                    f"teacher prompt {prompt.prompt_id!r} exceeds max_prompt_tokens"
                )
            required = len(model_prompt_ids) + len(completion.token_ids)
            try:
                sequence_bucket = select_length_bucket(
                    required,
                    self._config.training.teacher_sequence_buckets,
                )
            except PerformanceContractError as exc:
                raise ModelAdapterError(str(exc)) from exc
            rows.append(
                (
                    rollout,
                    prompt_ids,
                    model_prompt_ids,
                    completion,
                    sequence_bucket,
                )
            )
        return rows, mode_counts

    def score(self, request: TeacherScoreRequest) -> TeacherScoreBatch:
        if self._overlap_token_ids is None or self._project is None:
            raise ModelAdapterError(
                "remote teacher requires the pinned overlap profile before scoring"
            )
        try:
            import numpy as np
        except ImportError as exc:
            raise ModelAdapterError("NumPy is required for remote teacher") from exc

        tokenize_started = time.monotonic()
        tokenized, mode_counts = self._tokenize(request)
        tokenize_s = time.monotonic() - tokenize_started

        network_started = time.monotonic()
        workers = min(self.runtime.max_parallel_requests, len(tokenized))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(
                    self.client.score_tokens,
                    row[2],
                    row[3].token_ids,
                )
                for row in tokenized
            ]
            statistics = [future.result() for future in futures]
        network_s = time.monotonic() - network_started

        completion_bucket = max(
            select_length_bucket(
                len(row[3].token_ids),
                self._config.training.teacher_sequence_buckets,
            )
            for row in tokenized
        )
        hidden_bits = np.zeros(
            (len(tokenized), completion_bucket, self.profile.hidden_size),
            dtype=np.uint16,
        )
        log_normalizer = np.zeros(
            (len(tokenized), completion_bucket), dtype=np.float32
        )
        selected_log_probs = np.zeros_like(log_normalizer)
        widths = []
        request_attempts = []
        for index, stats in enumerate(statistics):
            width = int(stats.hidden_state_bits.shape[0])
            if stats.hidden_state_bits.shape[1] != self.profile.hidden_size:
                raise RemoteTeacherError("remote teacher hidden size mismatch")
            widths.append(width)
            request_attempts.append(
                int(stats.header.get("_client_request_attempts", 1))
            )
            hidden_bits[index, :width] = stats.hidden_state_bits
            log_normalizer[index, :width] = stats.log_normalizer
            selected_log_probs[index, :width] = stats.selected_log_probs

        project_started = time.monotonic()
        shared_log_probs = self._project(
            hidden_bits,
            self.profile.overlap_head_bits,
            log_normalizer,
        )
        selected_values = self._jax.lax.stop_gradient(
            self._jnp.asarray(selected_log_probs, dtype=self._jnp.float32)
        )
        if self._config.training.synchronize_phase_timings:
            self._jax.block_until_ready((shared_log_probs, selected_values))
        project_s = time.monotonic() - project_started

        samples = []
        for index, row in enumerate(tokenized):
            rollout, prompt_ids, _, completion, _ = row
            width = widths[index]
            sample_shared = shared_log_probs[index, :width, :]
            sample_selected = selected_values[index, :width]
            samples.append(
                TeacherScoreSample(
                    sample_id=rollout.sample_id,
                    prompt_id=rollout.prompt_id,
                    teacher_prompt_token_ids=prompt_ids,
                    completion=completion,
                    sufficient_statistics=TeacherSufficientStatisticsPayload(
                        shared_log_probs=sample_shared,
                        selected_log_probs=sample_selected,
                        shape=(width, len(self._overlap_token_ids)),
                        dtype="float32",
                    ),
                )
            )

        result = TeacherScoreBatch(
            contract_version=INTERFACE_CONTRACT_VERSION,
            run_id=request.rollouts.run_id,
            step=request.rollouts.step,
            model_id=self._config.teacher.model_id,
            model_revision=self._config.teacher.model_revision,
            tokenizer_id=self._config.teacher.tokenizer_id,
            tokenizer_revision=self._config.teacher.tokenizer_revision,
            samples=tuple(samples),
        )
        self.last_phase_timings = {
            "teacher_tokenize_s": tokenize_s,
            "teacher_remote_network_s": network_s,
            "teacher_remote_projection_s": project_s,
            "teacher_remote_request_attempts": sum(request_attempts),
            "teacher_remote_retry_count": sum(
                max(0, attempts - 1) for attempts in request_attempts
            ),
            "teacher_forward_s": network_s + project_s,
            "teacher_sequence_required": max(
                len(row[2]) + len(row[3].token_ids) for row in tokenized
            ),
            "teacher_sequence_bucket": max(row[4] for row in tokenized),
            "teacher_completion_bucket": completion_bucket,
            "teacher_joint_boundary_records": mode_counts[
                "joint_exact_boundary"
            ],
            "teacher_causal_split_records": mode_counts["causal_split"],
            "teacher_remote_profile_id": self.profile.profile_id,
        }
        return result
