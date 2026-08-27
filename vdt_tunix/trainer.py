"""One-teacher paper-math SimCT optimizer path for native Tunix models."""

from __future__ import annotations

import dataclasses
import math
import time
from typing import Any

from vdt_tunix.config import RunConfig
from vdt_tunix.contracts import (
    BackendBundle,
    PromptRecord,
    RolloutRequest,
    TeacherScoreRequest,
)
from vdt_tunix.jax_kernels import (
    paper_simct_aligned_batch_loss_from_teacher_statistics,
    paper_simple_opd_aligned_batch_loss_from_hidden_projection,
    paper_teacher_sufficient_statistics,
)
from vdt_tunix.performance import (
    PerformanceContractError,
    jax_memory_metrics,
    jit_cache_size,
    numeric_shape_signature,
    select_length_bucket,
)
from vdt_tunix.real_backend import _LoadedTunixModel
from vdt_tunix.supervision import (
    OverlapVocabulary,
    build_aligned_layout,
    build_overlap_vocabulary,
    pad_layouts,
)


class TrainingError(RuntimeError):
    """Raised when a batch cannot execute the declared reproduction mode."""


@dataclasses.dataclass(frozen=True, slots=True)
class PreparedSimCTBatch:
    """Static arrays consumed by one differentiable student update."""

    student_input_ids: Any
    student_segment_ids: Any
    student_completion_positions: Any
    student_completion_token_ids: Any
    teacher_shared_log_probs: Any
    teacher_selected_log_probs: Any
    segment_bounds: Any
    segment_mask: Any
    span_mask: Any
    normalizer: Any
    sample_count: int
    student_completion_tokens: int
    teacher_completion_tokens: int
    aligned_units: int
    aligned_spans: int
    rollout_s: float = 0.0
    teacher_score_s: float = 0.0
    teacher_tokenize_s: float = 0.0
    teacher_forward_s: float = 0.0
    alignment_s: float = 0.0
    batch_prepare_s: float = 0.0
    teacher_sequence_required: int = 0
    teacher_sequence_bucket: int = 0
    teacher_completion_bucket: int = 0
    teacher_joint_boundary_records: int = 0
    teacher_causal_split_records: int = 0
    alignment_bucket: int = 0
    actual_prompt_tokens: int = 0
    actual_completion_tokens: int = 0
    actual_total_tokens: int = 0
    maximum_prompt_tokens: int = 0
    maximum_completion_tokens: int = 0
    maximum_total_tokens: int = 0
    minimum_total_tokens: int = 0
    truncation_count: int = 0


@dataclasses.dataclass(frozen=True, slots=True)
class UpdateMetrics:
    loss: float
    gradient_norm: float
    parameter_norm: float
    sample_count: int
    student_completion_tokens: int
    teacher_completion_tokens: int
    aligned_units: int
    aligned_spans: int
    rollout_s: float = 0.0
    teacher_score_s: float = 0.0
    teacher_tokenize_s: float = 0.0
    teacher_forward_s: float = 0.0
    alignment_s: float = 0.0
    batch_prepare_s: float = 0.0
    student_update_dispatch_s: float = 0.0
    student_update_sync_s: float = 0.0
    student_update_s: float = 0.0
    student_fwd_bwd_s: float = 0.0
    teacher_sequence_required: int = 0
    teacher_sequence_bucket: int = 0
    teacher_completion_bucket: int = 0
    teacher_joint_boundary_records: int = 0
    teacher_causal_split_records: int = 0
    alignment_bucket: int = 0
    shape_signature: int = 0
    shape_signature_changed: int = 0
    jit_cache_size_before: int = -1
    jit_cache_size_after: int = -1
    jit_cache_miss: int = -1
    actual_prompt_tokens: int = 0
    actual_completion_tokens: int = 0
    actual_total_tokens: int = 0
    maximum_prompt_tokens: int = 0
    maximum_completion_tokens: int = 0
    maximum_total_tokens: int = 0
    minimum_total_tokens: int = 0
    requested_prompt_tokens: int = 0
    requested_completion_tokens: int = 0
    requested_sequence_tokens: int = 0
    truncation_count: int = 0
    rollout_tokens_s: float = 0.0
    teacher_score_tokens_s: float = 0.0
    student_update_tokens_s: float = 0.0
    student_fwd_bwd_tokens_s: float = 0.0
    compile_s: float = 0.0
    memory_bytes_in_use: int = -1
    memory_peak_bytes_in_use: int = -1
    memory_bytes_limit: int = -1

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _build_optimizer(config: RunConfig, model: Any, nnx: Any, optax: Any) -> Any:
    optimizer_steps = (
        config.training.max_steps
        if config.training.max_steps_unit == "optimizer_update"
        else max(
            1,
            math.ceil(
                config.training.max_steps
                / config.training.gradient_accumulation_steps
            ),
        )
    )
    warmup_steps = int(optimizer_steps * 0.05)
    if warmup_steps > 0:
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=config.training.learning_rate,
            warmup_steps=warmup_steps,
            decay_steps=optimizer_steps,
            end_value=0.0,
        )
    else:
        schedule = optax.cosine_decay_schedule(
            init_value=config.training.learning_rate,
            decay_steps=optimizer_steps,
        )
    transform = optax.adamw(
        learning_rate=schedule,
        b1=0.9,
        b2=0.98,
        weight_decay=0.0,
    )
    if config.training.gradient_accumulation_steps > 1:
        transform = optax.MultiSteps(
            transform,
            every_k_schedule=config.training.gradient_accumulation_steps,
        )
    return nnx.Optimizer(model, transform, wrt=nnx.Param)


def _pad_device_rows(rows: list[Any], width: int, jnp: Any) -> Any:
    padded = []
    for row in rows:
        value = jnp.asarray(row)
        if value.ndim == 1:
            padded.append(jnp.pad(value, ((0, width - value.shape[0]),)))
        elif value.ndim == 2:
            padded.append(jnp.pad(value, ((0, width - value.shape[0]), (0, 0))))
        else:
            raise TrainingError("only rank-1 labels and rank-2 logits can be padded")
    return jnp.stack(padded)


def prepare_simct_batch(
    config: RunConfig,
    prompts: tuple[PromptRecord, ...],
    backends: BackendBundle,
    overlap: OverlapVocabulary,
    *,
    step: int,
) -> PreparedSimCTBatch:
    """Roll out the student, score the frozen teacher, and align completions."""

    try:
        import jax.numpy as jnp
        import numpy as np
    except ImportError as exc:
        raise TrainingError("Tunix training requires JAX and NumPy") from exc
    if config.simct.reproduction_mode != "paper_math":
        raise TrainingError(
            "the Tunix optimizer currently implements paper_math only; "
            "public-code modes must remain separate"
        )
    prepare_started = time.monotonic()
    rollout_started = time.monotonic()
    rollouts = backends.student.rollout(
        RolloutRequest(
            run_id=config.run_id,
            step=step,
            prompts=prompts,
            samples_per_prompt=config.rollout.samples_per_prompt,
        )
    )
    rollout_s = time.monotonic() - rollout_started
    teacher_started = time.monotonic()
    teacher_scores = backends.teacher.score(
        TeacherScoreRequest(rollouts=rollouts, prompts=prompts)
    )
    teacher_score_s = time.monotonic() - teacher_started
    scores = {sample.sample_id: sample for sample in teacher_scores.samples}
    if set(scores) != {sample.sample_id for sample in rollouts.samples}:
        raise TrainingError("teacher scores do not match rollout sample IDs")

    batch_size = len(rollouts.samples)
    total_length = config.rollout.max_sequence_tokens or (
        config.rollout.max_prompt_tokens + config.rollout.max_completion_tokens
    )
    pad_id = backends.student.model_adapter.tokenizer.pad_token_id
    student_input_ids = np.full(
        (batch_size, total_length), pad_id, dtype=np.int32
    )
    student_segment_ids = np.zeros_like(student_input_ids)
    completion_positions = np.zeros(
        (batch_size, config.rollout.max_completion_tokens), dtype=np.int32
    )
    completion_ids = np.zeros_like(completion_positions)
    teacher_shared_scores: list[Any] = []
    teacher_selected_scores: list[Any] = []
    layouts = []
    student_token_total = 0
    teacher_token_total = 0
    prompt_token_total = 0
    total_token_counts: list[int] = []
    prompt_token_counts: list[int] = []
    completion_token_counts: list[int] = []
    truncation_count = 0

    alignment_started = time.monotonic()
    for row_index, rollout in enumerate(rollouts.samples):
        score = scores[rollout.sample_id]
        model_prompt_ids = (
            backends.student.model_adapter.tokenizer.with_model_prefix(
                rollout.student_prompt_token_ids
            )
        )
        full_ids = model_prompt_ids + rollout.completion.token_ids
        if len(full_ids) > total_length:
            raise TrainingError("student prompt plus completion exceeds static capacity")
        student_input_ids[row_index, : len(full_ids)] = full_ids
        student_segment_ids[row_index, : len(full_ids)] = 1
        start = len(model_prompt_ids) - 1
        width = len(rollout.completion.token_ids)
        completion_positions[row_index, :width] = np.arange(
            start, start + width, dtype=np.int32
        )
        completion_ids[row_index, :width] = rollout.completion.token_ids
        score_ids = jnp.asarray(score.completion.token_ids, dtype=jnp.int32)
        if score.sufficient_statistics is not None:
            shared_scores = jnp.asarray(
                score.sufficient_statistics.shared_log_probs,
                dtype=jnp.float32,
            )
            selected_scores = jnp.asarray(
                score.sufficient_statistics.selected_log_probs,
                dtype=jnp.float32,
            )
            if shared_scores.shape[-1] != len(overlap.teacher_ids):
                raise TrainingError(
                    "teacher sufficient-statistic overlap width mismatch"
                )
        else:
            if score.position_logits is None:
                raise TrainingError("teacher score contains no score representation")
            shared_batch, selected_batch = paper_teacher_sufficient_statistics(
                jnp.asarray(score.position_logits.values)[None, ...],
                score_ids[None, ...],
                jnp.asarray(overlap.teacher_ids, dtype=jnp.int32),
            )
            shared_scores = shared_batch[0]
            selected_scores = selected_batch[0]
        teacher_shared_scores.append(shared_scores)
        teacher_selected_scores.append(selected_scores)
        layouts.append(build_aligned_layout(rollout.completion, score.completion))
        student_token_total += width
        teacher_token_total += len(score.completion.token_ids)
        prompt_width = len(model_prompt_ids)
        prompt_token_total += prompt_width
        prompt_token_counts.append(prompt_width)
        completion_token_counts.append(width)
        total_token_counts.append(prompt_width + width)
        row_completion_limit = config.rollout.max_completion_tokens
        if config.rollout.max_sequence_tokens is not None:
            row_completion_limit = min(
                row_completion_limit,
                config.rollout.max_sequence_tokens - prompt_width,
            )
        truncation_count += int(width >= row_completion_limit)

    teacher_required = max(int(row.shape[0]) for row in teacher_shared_scores)
    try:
        teacher_width = select_length_bucket(
            teacher_required, config.training.teacher_sequence_buckets
        )
    except PerformanceContractError as exc:
        raise TrainingError(str(exc)) from exc
    padded_teacher_shared = _pad_device_rows(
        teacher_shared_scores, teacher_width, jnp
    )
    padded_teacher_selected = _pad_device_rows(
        teacher_selected_scores, teacher_width, jnp
    )
    bounds, unit_mask, spans = pad_layouts(
        layouts, width=config.training.alignment_bucket_size
    )
    aligned_units = sum(len(layout.units) for layout in layouts)
    aligned_spans = sum(sum(layout.span_mask) for layout in layouts)
    maximum_total_tokens = max(total_token_counts)
    minimum_total_tokens = min(total_token_counts)
    minimum_required = config.rollout.minimum_actual_sequence_tokens
    if minimum_required is not None and minimum_total_tokens < minimum_required:
        raise TrainingError(
            "resource probe did not reach its minimum actual sequence length: "
            f"{minimum_total_tokens} < {minimum_required}"
        )
    alignment_s = time.monotonic() - alignment_started
    student_timing = getattr(backends.student, "last_phase_timings", {})
    teacher_timing = getattr(backends.teacher, "last_phase_timings", {})
    return PreparedSimCTBatch(
        student_input_ids=jnp.asarray(student_input_ids),
        student_segment_ids=jnp.asarray(student_segment_ids),
        student_completion_positions=jnp.asarray(completion_positions),
        student_completion_token_ids=jnp.asarray(completion_ids),
        teacher_shared_log_probs=padded_teacher_shared,
        teacher_selected_log_probs=padded_teacher_selected,
        segment_bounds=jnp.asarray(bounds, dtype=jnp.int32),
        segment_mask=jnp.asarray(unit_mask, dtype=jnp.float32),
        span_mask=jnp.asarray(spans, dtype=bool),
        normalizer=jnp.asarray(student_token_total, dtype=jnp.float32),
        sample_count=batch_size,
        student_completion_tokens=student_token_total,
        teacher_completion_tokens=teacher_token_total,
        aligned_units=aligned_units,
        aligned_spans=aligned_spans,
        rollout_s=float(student_timing.get("student_rollout_s", rollout_s)),
        teacher_score_s=teacher_score_s,
        teacher_tokenize_s=float(teacher_timing.get("teacher_tokenize_s", 0.0)),
        teacher_forward_s=float(teacher_timing.get("teacher_forward_s", 0.0)),
        alignment_s=alignment_s,
        batch_prepare_s=time.monotonic() - prepare_started,
        teacher_sequence_required=int(
            teacher_timing.get("teacher_sequence_required", teacher_required)
        ),
        teacher_sequence_bucket=int(
            teacher_timing.get("teacher_sequence_bucket", teacher_required)
        ),
        teacher_completion_bucket=teacher_width,
        teacher_joint_boundary_records=int(
            teacher_timing.get("teacher_joint_boundary_records", 0)
        ),
        teacher_causal_split_records=int(
            teacher_timing.get("teacher_causal_split_records", 0)
        ),
        alignment_bucket=len(bounds[0]),
        actual_prompt_tokens=prompt_token_total,
        actual_completion_tokens=student_token_total,
        actual_total_tokens=prompt_token_total + student_token_total,
        maximum_prompt_tokens=max(prompt_token_counts),
        maximum_completion_tokens=max(completion_token_counts),
        maximum_total_tokens=maximum_total_tokens,
        minimum_total_tokens=minimum_total_tokens,
        truncation_count=truncation_count,
    )


class PaperSimCTTrainer:
    """Own a Tunix NNX optimizer and execute paper-faithful SimCT updates."""

    def __init__(self, config: RunConfig, backends: BackendBundle):
        if config.simct.algorithm != "simct":
            raise TrainingError("PaperSimCTTrainer requires algorithm=simct")
        if config.simct.reproduction_mode != "paper_math":
            raise TrainingError("PaperSimCTTrainer requires reproduction_mode=paper_math")
        if not (
            backends.student.real_model_integration
            and backends.teacher.real_model_integration
        ):
            raise TrainingError("optimizer updates require real model integrations")
        try:
            import jax
            import jax.numpy as jnp
            from flax import nnx
            import optax
        except ImportError as exc:
            raise TrainingError("Tunix optimizer dependencies are unavailable") from exc

        self.config = config
        self.backends = backends
        self.overlap = build_overlap_vocabulary(
            backends.student.model_adapter.tokenizer.raw_tokenizer,
            backends.teacher.model_adapter.tokenizer.raw_tokenizer,
        )
        configure_teacher = getattr(
            backends.teacher, "configure_overlap_token_ids", None
        )
        if callable(configure_teacher):
            configure_teacher(self.overlap.teacher_ids)
        loaded = backends.student.model_adapter.require_loaded_model()
        if not isinstance(loaded, _LoadedTunixModel):
            raise TrainingError("student backend did not expose a native Tunix model")
        self.loaded_student = loaded
        self.optimizer = _build_optimizer(config, loaded.model, nnx, optax)

        student_overlap = jnp.asarray(self.overlap.student_ids, dtype=jnp.int32)
        temperature = config.simct.temperature
        forward_fn = loaded.forward_fn

        @nnx.jit
        def update_fn(
            model,
            optimizer,
            input_ids,
            segment_ids,
            completion_positions,
            completion_token_ids,
            teacher_shared_log_probs,
            teacher_selected_log_probs,
            segment_bounds,
            segment_mask,
            span_mask,
            normalizer,
        ):
            positions = jnp.maximum(jnp.cumsum(segment_ids, axis=-1) - 1, 0)

            def loss_fn(candidate_model):
                full_logits = forward_fn(
                    candidate_model, input_ids, positions, segment_ids
                )
                batch = jnp.arange(input_ids.shape[0], dtype=jnp.int32)[:, None]
                student_position_logits = full_logits[batch, completion_positions]
                return paper_simct_aligned_batch_loss_from_teacher_statistics(
                    student_position_logits,
                    completion_token_ids,
                    teacher_shared_log_probs,
                    teacher_selected_log_probs,
                    student_overlap,
                    segment_bounds,
                    segment_mask,
                    span_mask,
                    temperature=temperature,
                    normalizer=normalizer,
                )

            loss, gradients = nnx.value_and_grad(loss_fn)(model)
            gradient_norm = optax.global_norm(gradients)
            parameter_norm = optax.global_norm(nnx.state(model, nnx.Param))
            optimizer.update(model, gradients)
            return loss, gradient_norm, parameter_norm

        self._update_fn = update_fn
        self._jax = jax
        self._last_shape_signature: int | None = None

    def step(
        self,
        prompts: tuple[PromptRecord, ...],
        *,
        step: int,
    ) -> UpdateMetrics:
        batch = prepare_simct_batch(
            self.config,
            prompts,
            self.backends,
            self.overlap,
            step=step,
        )
        loaded = self.loaded_student
        shape_signature = numeric_shape_signature(
            batch_size=int(batch.student_input_ids.shape[0]),
            student_sequence=int(batch.student_input_ids.shape[1]),
            student_completion=int(batch.student_completion_positions.shape[1]),
            teacher_completion=int(batch.teacher_shared_log_probs.shape[1]),
            alignment_units=int(batch.segment_bounds.shape[1]),
        )
        shape_changed = int(
            self._last_shape_signature is not None
            and self._last_shape_signature != shape_signature
        )
        compilation_candidate = self._last_shape_signature is None or bool(
            shape_changed
        )
        cache_before = jit_cache_size(self._update_fn)
        dispatch_started = time.monotonic()
        with self._jax.set_mesh(loaded.mesh):
            loss, gradient_norm, parameter_norm = self._update_fn(
                loaded.model,
                self.optimizer,
                batch.student_input_ids,
                batch.student_segment_ids,
                batch.student_completion_positions,
                batch.student_completion_token_ids,
                batch.teacher_shared_log_probs,
                batch.teacher_selected_log_probs,
                batch.segment_bounds,
                batch.segment_mask,
                batch.span_mask,
                batch.normalizer,
            )
        dispatch_s = time.monotonic() - dispatch_started
        sync_started = time.monotonic()
        loss, gradient_norm, parameter_norm = self._jax.device_get(
            (loss, gradient_norm, parameter_norm)
        )
        sync_s = time.monotonic() - sync_started
        cache_after = jit_cache_size(self._update_fn)
        memory = jax_memory_metrics(self._jax)
        self._last_shape_signature = shape_signature
        return UpdateMetrics(
            loss=float(loss),
            gradient_norm=float(gradient_norm),
            parameter_norm=float(parameter_norm),
            sample_count=batch.sample_count,
            student_completion_tokens=batch.student_completion_tokens,
            teacher_completion_tokens=batch.teacher_completion_tokens,
            aligned_units=batch.aligned_units,
            aligned_spans=batch.aligned_spans,
            rollout_s=batch.rollout_s,
            teacher_score_s=batch.teacher_score_s,
            teacher_tokenize_s=batch.teacher_tokenize_s,
            teacher_forward_s=batch.teacher_forward_s,
            alignment_s=batch.alignment_s,
            batch_prepare_s=batch.batch_prepare_s,
            student_update_dispatch_s=dispatch_s,
            student_update_sync_s=sync_s,
            student_update_s=dispatch_s + sync_s,
            student_fwd_bwd_s=dispatch_s + sync_s,
            teacher_sequence_required=batch.teacher_sequence_required,
            teacher_sequence_bucket=batch.teacher_sequence_bucket,
            teacher_completion_bucket=batch.teacher_completion_bucket,
            teacher_joint_boundary_records=batch.teacher_joint_boundary_records,
            teacher_causal_split_records=batch.teacher_causal_split_records,
            alignment_bucket=batch.alignment_bucket,
            shape_signature=shape_signature,
            shape_signature_changed=shape_changed,
            jit_cache_size_before=cache_before,
            jit_cache_size_after=cache_after,
            jit_cache_miss=(
                int(cache_after > cache_before)
                if cache_before >= 0 and cache_after >= 0
                else -1
            ),
            actual_prompt_tokens=batch.actual_prompt_tokens,
            actual_completion_tokens=batch.actual_completion_tokens,
            actual_total_tokens=batch.actual_total_tokens,
            maximum_prompt_tokens=batch.maximum_prompt_tokens,
            maximum_completion_tokens=batch.maximum_completion_tokens,
            maximum_total_tokens=batch.maximum_total_tokens,
            minimum_total_tokens=batch.minimum_total_tokens,
            requested_prompt_tokens=self.config.rollout.max_prompt_tokens,
            requested_completion_tokens=self.config.rollout.max_completion_tokens,
            requested_sequence_tokens=(
                self.config.rollout.max_sequence_tokens
                or self.config.rollout.max_prompt_tokens
                + self.config.rollout.max_completion_tokens
            ),
            truncation_count=batch.truncation_count,
            rollout_tokens_s=(
                batch.actual_completion_tokens / batch.rollout_s
                if batch.rollout_s > 0
                else 0.0
            ),
            teacher_score_tokens_s=(
                batch.teacher_completion_tokens / batch.teacher_score_s
                if batch.teacher_score_s > 0
                else 0.0
            ),
            student_update_tokens_s=(
                batch.actual_total_tokens / (dispatch_s + sync_s)
                if dispatch_s + sync_s > 0
                else 0.0
            ),
            student_fwd_bwd_tokens_s=(
                batch.actual_total_tokens / (dispatch_s + sync_s)
                if dispatch_s + sync_s > 0
                else 0.0
            ),
            compile_s=(
                dispatch_s + sync_s
                if compilation_candidate
                else 0.0
            ),
            **memory,
        )


class PaperSimpleOPDTrainer:
    """Overlap-only reverse-KL control initialized from the same SFT state."""

    def __init__(self, config: RunConfig, backends: BackendBundle):
        if config.simct.algorithm != "simple_opd":
            raise TrainingError(
                "PaperSimpleOPDTrainer requires algorithm=simple_opd"
            )
        if config.simct.reproduction_mode != "paper_math":
            raise TrainingError(
                "PaperSimpleOPDTrainer requires reproduction_mode=paper_math"
            )
        if not (
            backends.student.real_model_integration
            and backends.teacher.real_model_integration
        ):
            raise TrainingError("optimizer updates require real model integrations")
        try:
            import jax
            import jax.numpy as jnp
            from flax import nnx
            import optax
        except ImportError as exc:
            raise TrainingError(
                "Tunix optimizer dependencies are unavailable"
            ) from exc

        self.config = config
        self.backends = backends
        self.overlap = build_overlap_vocabulary(
            backends.student.model_adapter.tokenizer.raw_tokenizer,
            backends.teacher.model_adapter.tokenizer.raw_tokenizer,
        )
        configure_teacher = getattr(
            backends.teacher, "configure_overlap_token_ids", None
        )
        if callable(configure_teacher):
            configure_teacher(self.overlap.teacher_ids)
        loaded = backends.student.model_adapter.require_loaded_model()
        if not isinstance(loaded, _LoadedTunixModel):
            raise TrainingError("student backend did not expose a native Tunix model")
        self.loaded_student = loaded
        self.optimizer = _build_optimizer(config, loaded.model, nnx, optax)

        student_overlap = jnp.asarray(self.overlap.student_ids, dtype=jnp.int32)
        temperature = config.simct.temperature
        forward_hidden_fn = loaded.forward_hidden_fn
        project_shared_fn = loaded.project_shared_fn
        if not callable(forward_hidden_fn) or not callable(project_shared_fn):
            raise TrainingError(
                "SimpleOPD requires the bounded hidden-state overlap projection"
            )

        @nnx.jit
        def update_fn(
            model,
            optimizer,
            input_ids,
            segment_ids,
            completion_positions,
            teacher_shared_log_probs,
            segment_bounds,
            segment_mask,
            span_mask,
            normalizer,
        ):
            positions = jnp.maximum(jnp.cumsum(segment_ids, axis=-1) - 1, 0)

            def loss_fn(candidate_model):
                full_hidden = forward_hidden_fn(
                    candidate_model, input_ids, positions, segment_ids
                )
                batch = jnp.arange(input_ids.shape[0], dtype=jnp.int32)[:, None]
                student_position_hidden = full_hidden[
                    batch, completion_positions
                ]
                return paper_simple_opd_aligned_batch_loss_from_hidden_projection(
                    student_position_hidden,
                    teacher_shared_log_probs,
                    segment_bounds,
                    segment_mask,
                    span_mask,
                    lambda hidden: project_shared_fn(
                        candidate_model, hidden, student_overlap
                    ),
                    temperature=temperature,
                    normalizer=normalizer,
                )

            loss, gradients = nnx.value_and_grad(loss_fn)(model)
            gradient_norm = optax.global_norm(gradients)
            parameter_norm = optax.global_norm(nnx.state(model, nnx.Param))
            optimizer.update(model, gradients)
            return loss, gradient_norm, parameter_norm

        self._update_fn = update_fn
        self._jax = jax
        self._last_shape_signature: int | None = None

    def step(
        self,
        prompts: tuple[PromptRecord, ...],
        *,
        step: int,
    ) -> UpdateMetrics:
        batch = prepare_simct_batch(
            self.config,
            prompts,
            self.backends,
            self.overlap,
            step=step,
        )
        shape_signature = numeric_shape_signature(
            batch_size=int(batch.student_input_ids.shape[0]),
            student_sequence=int(batch.student_input_ids.shape[1]),
            student_completion=int(batch.student_completion_positions.shape[1]),
            teacher_completion=int(batch.teacher_shared_log_probs.shape[1]),
            alignment_units=int(batch.segment_bounds.shape[1]),
        )
        shape_changed = int(
            self._last_shape_signature is not None
            and self._last_shape_signature != shape_signature
        )
        compilation_candidate = self._last_shape_signature is None or bool(
            shape_changed
        )
        cache_before = jit_cache_size(self._update_fn)
        dispatch_started = time.monotonic()
        with self._jax.set_mesh(self.loaded_student.mesh):
            loss, gradient_norm, parameter_norm = self._update_fn(
                self.loaded_student.model,
                self.optimizer,
                batch.student_input_ids,
                batch.student_segment_ids,
                batch.student_completion_positions,
                batch.teacher_shared_log_probs,
                batch.segment_bounds,
                batch.segment_mask,
                batch.span_mask,
                batch.normalizer,
            )
        dispatch_s = time.monotonic() - dispatch_started
        sync_started = time.monotonic()
        loss, gradient_norm, parameter_norm = self._jax.device_get(
            (loss, gradient_norm, parameter_norm)
        )
        sync_s = time.monotonic() - sync_started
        cache_after = jit_cache_size(self._update_fn)
        memory = jax_memory_metrics(self._jax)
        self._last_shape_signature = shape_signature
        return UpdateMetrics(
            loss=float(loss),
            gradient_norm=float(gradient_norm),
            parameter_norm=float(parameter_norm),
            sample_count=batch.sample_count,
            student_completion_tokens=batch.student_completion_tokens,
            teacher_completion_tokens=batch.teacher_completion_tokens,
            aligned_units=batch.aligned_units - batch.aligned_spans,
            aligned_spans=0,
            rollout_s=batch.rollout_s,
            teacher_score_s=batch.teacher_score_s,
            teacher_tokenize_s=batch.teacher_tokenize_s,
            teacher_forward_s=batch.teacher_forward_s,
            alignment_s=batch.alignment_s,
            batch_prepare_s=batch.batch_prepare_s,
            student_update_dispatch_s=dispatch_s,
            student_update_sync_s=sync_s,
            student_update_s=dispatch_s + sync_s,
            student_fwd_bwd_s=dispatch_s + sync_s,
            teacher_sequence_required=batch.teacher_sequence_required,
            teacher_sequence_bucket=batch.teacher_sequence_bucket,
            teacher_completion_bucket=batch.teacher_completion_bucket,
            teacher_joint_boundary_records=batch.teacher_joint_boundary_records,
            teacher_causal_split_records=batch.teacher_causal_split_records,
            alignment_bucket=batch.alignment_bucket,
            shape_signature=shape_signature,
            shape_signature_changed=shape_changed,
            jit_cache_size_before=cache_before,
            jit_cache_size_after=cache_after,
            jit_cache_miss=(
                int(cache_after > cache_before)
                if cache_before >= 0 and cache_after >= 0
                else -1
            ),
            actual_prompt_tokens=batch.actual_prompt_tokens,
            actual_completion_tokens=batch.actual_completion_tokens,
            actual_total_tokens=batch.actual_total_tokens,
            maximum_prompt_tokens=batch.maximum_prompt_tokens,
            maximum_completion_tokens=batch.maximum_completion_tokens,
            maximum_total_tokens=batch.maximum_total_tokens,
            minimum_total_tokens=batch.minimum_total_tokens,
            requested_prompt_tokens=self.config.rollout.max_prompt_tokens,
            requested_completion_tokens=self.config.rollout.max_completion_tokens,
            requested_sequence_tokens=(
                self.config.rollout.max_sequence_tokens
                or self.config.rollout.max_prompt_tokens
                + self.config.rollout.max_completion_tokens
            ),
            truncation_count=batch.truncation_count,
            rollout_tokens_s=(
                batch.actual_completion_tokens / batch.rollout_s
                if batch.rollout_s > 0
                else 0.0
            ),
            teacher_score_tokens_s=(
                batch.teacher_completion_tokens / batch.teacher_score_s
                if batch.teacher_score_s > 0
                else 0.0
            ),
            student_update_tokens_s=(
                batch.actual_total_tokens / (dispatch_s + sync_s)
                if dispatch_s + sync_s > 0
                else 0.0
            ),
            student_fwd_bwd_tokens_s=(
                batch.actual_total_tokens / (dispatch_s + sync_s)
                if dispatch_s + sync_s > 0
                else 0.0
            ),
            compile_s=(
                dispatch_s + sync_s
                if compilation_candidate
                else 0.0
            ),
            **memory,
        )
