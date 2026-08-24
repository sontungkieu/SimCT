"""One-teacher paper-math SimCT optimizer path for native Tunix models."""

from __future__ import annotations

import dataclasses
import math
from typing import Any

from vdt_tunix.config import RunConfig
from vdt_tunix.contracts import (
    BackendBundle,
    PromptRecord,
    RolloutRequest,
    TeacherScoreRequest,
)
from vdt_tunix.jax_kernels import paper_simct_aligned_batch_loss
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
    teacher_position_logits: Any
    teacher_completion_token_ids: Any
    segment_bounds: Any
    segment_mask: Any
    span_mask: Any
    normalizer: Any
    sample_count: int
    student_completion_tokens: int
    teacher_completion_tokens: int
    aligned_units: int
    aligned_spans: int


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

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


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
    rollouts = backends.student.rollout(
        RolloutRequest(
            run_id=config.run_id,
            step=step,
            prompts=prompts,
            samples_per_prompt=config.rollout.samples_per_prompt,
        )
    )
    teacher_scores = backends.teacher.score(
        TeacherScoreRequest(rollouts=rollouts, prompts=prompts)
    )
    scores = {sample.sample_id: sample for sample in teacher_scores.samples}
    if set(scores) != {sample.sample_id for sample in rollouts.samples}:
        raise TrainingError("teacher scores do not match rollout sample IDs")

    batch_size = len(rollouts.samples)
    total_length = (
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
    teacher_logits: list[Any] = []
    teacher_ids: list[Any] = []
    layouts = []
    student_token_total = 0
    teacher_token_total = 0

    for row_index, rollout in enumerate(rollouts.samples):
        score = scores[rollout.sample_id]
        full_ids = rollout.student_prompt_token_ids + rollout.completion.token_ids
        if len(full_ids) > total_length:
            raise TrainingError("student prompt plus completion exceeds static capacity")
        student_input_ids[row_index, : len(full_ids)] = full_ids
        student_segment_ids[row_index, : len(full_ids)] = 1
        start = len(rollout.student_prompt_token_ids) - 1
        width = len(rollout.completion.token_ids)
        completion_positions[row_index, :width] = np.arange(
            start, start + width, dtype=np.int32
        )
        completion_ids[row_index, :width] = rollout.completion.token_ids
        teacher_logits.append(score.position_logits.values)
        teacher_ids.append(jnp.asarray(score.completion.token_ids, dtype=jnp.int32))
        layouts.append(build_aligned_layout(rollout.completion, score.completion))
        student_token_total += width
        teacher_token_total += len(score.completion.token_ids)

    teacher_width = max(int(row.shape[0]) for row in teacher_logits)
    padded_teacher_logits = _pad_device_rows(teacher_logits, teacher_width, jnp)
    padded_teacher_ids = _pad_device_rows(teacher_ids, teacher_width, jnp)
    bounds, unit_mask, spans = pad_layouts(layouts)
    aligned_units = sum(len(layout.units) for layout in layouts)
    aligned_spans = sum(sum(layout.span_mask) for layout in layouts)
    return PreparedSimCTBatch(
        student_input_ids=jnp.asarray(student_input_ids),
        student_segment_ids=jnp.asarray(student_segment_ids),
        student_completion_positions=jnp.asarray(completion_positions),
        student_completion_token_ids=jnp.asarray(completion_ids),
        teacher_position_logits=padded_teacher_logits,
        teacher_completion_token_ids=padded_teacher_ids,
        segment_bounds=jnp.asarray(bounds, dtype=jnp.int32),
        segment_mask=jnp.asarray(unit_mask, dtype=jnp.float32),
        span_mask=jnp.asarray(spans, dtype=bool),
        normalizer=jnp.asarray(student_token_total, dtype=jnp.float32),
        sample_count=batch_size,
        student_completion_tokens=student_token_total,
        teacher_completion_tokens=teacher_token_total,
        aligned_units=aligned_units,
        aligned_spans=aligned_spans,
    )


class PaperSimCTTrainer:
    """Own a Tunix NNX optimizer and execute paper-faithful SimCT updates."""

    def __init__(self, config: RunConfig, backends: BackendBundle):
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
        loaded = backends.student.model_adapter.require_loaded_model()
        if not isinstance(loaded, _LoadedTunixModel):
            raise TrainingError("student backend did not expose a native Tunix model")
        self.loaded_student = loaded
        optimizer_steps = max(
            1,
            math.ceil(
                config.training.max_steps
                / config.training.gradient_accumulation_steps
            ),
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
        tx = optax.adamw(
            learning_rate=schedule,
            b1=0.9,
            b2=0.98,
            weight_decay=0.0,
        )
        if config.training.gradient_accumulation_steps > 1:
            tx = optax.MultiSteps(
                tx,
                every_k_schedule=config.training.gradient_accumulation_steps,
            )
        self.optimizer = nnx.Optimizer(
            loaded.model,
            tx,
            wrt=nnx.Param,
        )

        student_overlap = jnp.asarray(self.overlap.student_ids, dtype=jnp.int32)
        teacher_overlap = jnp.asarray(self.overlap.teacher_ids, dtype=jnp.int32)
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
            teacher_position_logits,
            teacher_completion_token_ids,
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
                return paper_simct_aligned_batch_loss(
                    student_position_logits,
                    completion_token_ids,
                    teacher_position_logits,
                    teacher_completion_token_ids,
                    student_overlap,
                    teacher_overlap,
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
        with self._jax.set_mesh(loaded.mesh):
            loss, gradient_norm, parameter_norm = self._update_fn(
                loaded.model,
                self.optimizer,
                batch.student_input_ids,
                batch.student_segment_ids,
                batch.student_completion_positions,
                batch.student_completion_token_ids,
                batch.teacher_position_logits,
                batch.teacher_completion_token_ids,
                batch.segment_bounds,
                batch.segment_mask,
                batch.span_mask,
                batch.normalizer,
            )
        loss, gradient_norm, parameter_norm = self._jax.device_get(
            (loss, gradient_norm, parameter_norm)
        )
        return UpdateMetrics(
            loss=float(loss),
            gradient_norm=float(gradient_norm),
            parameter_norm=float(parameter_norm),
            sample_count=batch.sample_count,
            student_completion_tokens=batch.student_completion_tokens,
            teacher_completion_tokens=batch.teacher_completion_tokens,
            aligned_units=batch.aligned_units,
            aligned_spans=batch.aligned_spans,
        )
