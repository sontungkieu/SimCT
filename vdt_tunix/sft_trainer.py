"""Native Tunix warm-start SFT for the same student state used by SimCT."""

from __future__ import annotations

import dataclasses
import math
from typing import Any

from vdt_tunix.config import RunConfig
from vdt_tunix.contracts import BackendBundle
from vdt_tunix.real_backend import _LoadedTunixModel
from vdt_tunix.training_data import SFTRecord


class SFTTrainingError(RuntimeError):
    """Raised when a supervised batch violates the static TPU contract."""


@dataclasses.dataclass(frozen=True, slots=True)
class PreparedSFTBatch:
    input_ids: Any
    segment_ids: Any
    label_positions: Any
    label_ids: Any
    label_mask: Any
    normalizer: Any
    sample_count: int
    target_tokens: int


@dataclasses.dataclass(frozen=True, slots=True)
class SFTUpdateMetrics:
    loss: float
    gradient_norm: float
    parameter_norm: float
    sample_count: int
    target_tokens: int

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def prepare_sft_batch(
    config: RunConfig,
    rows: tuple[SFTRecord, ...],
    backends: BackendBundle,
) -> PreparedSFTBatch:
    try:
        import jax.numpy as jnp
        import numpy as np
    except ImportError as exc:
        raise SFTTrainingError("Tunix SFT requires JAX and NumPy") from exc
    if len(rows) != config.rollout.prompt_batch_size:
        raise SFTTrainingError("SFT row count must match rollout.prompt_batch_size")
    tokenizer = backends.student.model_adapter.tokenizer
    total_length = (
        config.rollout.max_prompt_tokens + config.rollout.max_completion_tokens
    )
    label_width = config.rollout.max_completion_tokens + 1
    input_ids = np.full(
        (len(rows), total_length), tokenizer.pad_token_id, dtype=np.int32
    )
    segment_ids = np.zeros_like(input_ids)
    label_positions = np.zeros((len(rows), label_width), dtype=np.int32)
    label_ids = np.zeros_like(label_positions)
    label_mask = np.zeros_like(label_positions, dtype=np.float32)
    target_total = 0

    for row_index, row in enumerate(rows):
        prompt_ids, completion = tokenizer.tokenize_continuation(
            prompt_text=row.student_prompt,
            completion_text=row.target_response,
        )
        model_prompt_ids = tokenizer.with_model_prefix(prompt_ids)
        if len(model_prompt_ids) > config.rollout.max_prompt_tokens:
            raise SFTTrainingError(
                f"SFT prompt {row.prompt_id!r} exceeds max_prompt_tokens"
            )
        if len(completion.token_ids) > config.rollout.max_completion_tokens:
            raise SFTTrainingError(
                f"SFT response {row.prompt_id!r} exceeds max_completion_tokens"
            )
        row_input = model_prompt_ids + completion.token_ids
        if len(row_input) > total_length:
            raise SFTTrainingError("SFT input exceeds static sequence capacity")
        targets = completion.token_ids + (tokenizer.eos_token_id,)
        start = len(model_prompt_ids) - 1
        positions = np.arange(start, start + len(targets), dtype=np.int32)
        input_ids[row_index, : len(row_input)] = row_input
        segment_ids[row_index, : len(row_input)] = 1
        label_positions[row_index, : len(targets)] = positions
        label_ids[row_index, : len(targets)] = targets
        label_mask[row_index, : len(targets)] = 1.0
        target_total += len(targets)
    return PreparedSFTBatch(
        input_ids=jnp.asarray(input_ids),
        segment_ids=jnp.asarray(segment_ids),
        label_positions=jnp.asarray(label_positions),
        label_ids=jnp.asarray(label_ids),
        label_mask=jnp.asarray(label_mask),
        normalizer=jnp.asarray(target_total, dtype=jnp.float32),
        sample_count=len(rows),
        target_tokens=target_total,
    )


class TunixSFTTrainer:
    """Run full-parameter warm-start updates on the native student model."""

    def __init__(self, config: RunConfig, backends: BackendBundle):
        if not backends.student.real_model_integration:
            raise SFTTrainingError("SFT updates require a real student integration")
        try:
            import jax
            import jax.numpy as jnp
            from flax import nnx
            import optax
        except ImportError as exc:
            raise SFTTrainingError("Tunix optimizer dependencies are unavailable") from exc
        self.config = config
        self.backends = backends
        loaded = backends.student.model_adapter.require_loaded_model()
        if not isinstance(loaded, _LoadedTunixModel):
            raise SFTTrainingError("student backend did not expose a native Tunix model")
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
        self.optimizer = nnx.Optimizer(loaded.model, tx, wrt=nnx.Param)
        forward_fn = loaded.forward_fn

        @nnx.jit
        def update_fn(
            model,
            optimizer,
            input_ids,
            segment_ids,
            label_positions,
            label_ids,
            label_mask,
            normalizer,
        ):
            positions = jnp.maximum(jnp.cumsum(segment_ids, axis=-1) - 1, 0)

            def loss_fn(candidate_model):
                full_logits = forward_fn(
                    candidate_model, input_ids, positions, segment_ids
                )
                batch = jnp.arange(input_ids.shape[0], dtype=jnp.int32)[:, None]
                logits = full_logits[batch, label_positions]
                log_probs = jax.nn.log_softmax(logits, axis=-1)
                selected = jnp.take_along_axis(
                    log_probs, label_ids[..., None], axis=-1
                )[..., 0]
                return -jnp.sum(selected * label_mask) / normalizer

            loss, gradients = nnx.value_and_grad(loss_fn)(model)
            gradient_norm = optax.global_norm(gradients)
            parameter_norm = optax.global_norm(nnx.state(model, nnx.Param))
            optimizer.update(model, gradients)
            return loss, gradient_norm, parameter_norm

        self._update_fn = update_fn
        self._jax = jax

    def step(self, rows: tuple[SFTRecord, ...], *, step: int) -> SFTUpdateMetrics:
        del step
        batch = prepare_sft_batch(self.config, rows, self.backends)
        with self._jax.set_mesh(self.loaded_student.mesh):
            loss, gradient_norm, parameter_norm = self._update_fn(
                self.loaded_student.model,
                self.optimizer,
                batch.input_ids,
                batch.segment_ids,
                batch.label_positions,
                batch.label_ids,
                batch.label_mask,
                batch.normalizer,
            )
        loss, gradient_norm, parameter_norm = self._jax.device_get(
            (loss, gradient_norm, parameter_norm)
        )
        return SFTUpdateMetrics(
            loss=float(loss),
            gradient_norm=float(gradient_norm),
            parameter_norm=float(parameter_norm),
            sample_count=batch.sample_count,
            target_tokens=batch.target_tokens,
        )
