"""Native Tunix inference from a verified VDT Orbax checkpoint."""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from vdt_tunix.checkpoint import load_latest_checkpoint
from vdt_tunix.config import RunConfig
from vdt_tunix.real_backend import load_native_student
from vdt_tunix.runtime import require_tpu_v5e8
from vdt_tunix.tunix_checkpoint import TunixCheckpointController


class EvaluationRuntimeError(RuntimeError):
    """Raised when a checkpoint cannot support native, pinned inference."""


@dataclasses.dataclass(frozen=True, slots=True)
class RestoredStudent:
    model: Any
    tokenizer: Any
    mesh: Any
    model_config: Any
    checkpoint_steps: int
    checkpoint_run_id: str
    student_parameters_sha256: str
    dataset_manifest_sha256: str
    hardware: dict[str, Any]


def restore_student_for_inference(
    config: RunConfig, checkpoint_root: str | Path
) -> RestoredStudent:
    """Restore model-only state while checking the full training lineage."""

    root = Path(checkpoint_root).resolve()
    state = load_latest_checkpoint(root, config=config)
    inference_checkpoint = dataclasses.replace(
        config.checkpoint,
        resume_from=None,
        warm_start_from=str(root),
    )
    inference_config = dataclasses.replace(
        config, checkpoint=inference_checkpoint
    )
    loaded, tokenizer = load_native_student(inference_config)
    _, hardware = require_tpu_v5e8(
        expected_device_count=inference_config.tpu.expected_device_count
    )
    controller = TunixCheckpointController(
        inference_config,
        loaded.model,
        None,
        dataset_manifest_sha256=state.dataset_manifest_sha256,
    )
    try:
        restored = controller.initialize_or_resume()
    finally:
        controller.close()
    if restored.source_checkpoint_steps != state.completed_steps:
        raise EvaluationRuntimeError("restored checkpoint coordinate drifted")
    if restored.source_student_parameters_sha256 != state.student_parameters.sha256:
        raise EvaluationRuntimeError("restored student parameter digest drifted")
    return RestoredStudent(
        model=loaded.model,
        tokenizer=tokenizer.raw_tokenizer,
        mesh=loaded.mesh,
        model_config=loaded.model_config,
        checkpoint_steps=state.completed_steps,
        checkpoint_run_id=state.run_id,
        student_parameters_sha256=state.student_parameters.sha256,
        dataset_manifest_sha256=state.dataset_manifest_sha256,
        hardware=hardware,
    )


class NativeTunixGenerator:
    """One reusable KV-cached sampler for a restored Gemma student."""

    def __init__(
        self,
        restored: RestoredStudent,
        *,
        max_prompt_tokens: int,
        max_completion_tokens: int,
    ):
        if max_prompt_tokens < 1 or max_completion_tokens < 1:
            raise EvaluationRuntimeError("generation token limits must be positive")
        try:
            from tunix.generate import sampler as sampler_lib
        except ImportError as exc:
            raise EvaluationRuntimeError("Tunix sampler is unavailable") from exc
        params = restored.model_config
        for name in ("num_layers", "num_kv_heads", "head_dim"):
            if not hasattr(params, name):
                raise EvaluationRuntimeError(
                    f"student model config lacks sampler field {name}"
                )
        self.restored = restored
        self.max_prompt_tokens = max_prompt_tokens
        self.max_completion_tokens = max_completion_tokens
        self.sampler = sampler_lib.Sampler(
            transformer=restored.model,
            tokenizer=restored.tokenizer,
            cache_config=sampler_lib.CacheConfig(
                cache_size=max_prompt_tokens + max_completion_tokens + 1,
                num_layers=int(params.num_layers),
                num_kv_heads=int(params.num_kv_heads),
                head_dim=int(params.head_dim),
            ),
        )

    def prompt_lengths(self, prompts: Sequence[str]) -> tuple[int, ...]:
        lengths = tuple(len(self.sampler.tokenize(prompt)) for prompt in prompts)
        if any(length > self.max_prompt_tokens for length in lengths):
            raise EvaluationRuntimeError(
                f"prompt exceeds max_prompt_tokens={self.max_prompt_tokens}: {lengths}"
            )
        return lengths

    def generate(
        self,
        prompts: Sequence[str],
        *,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        seed: int,
    ) -> tuple[tuple[str, ...], tuple[int, ...], tuple[int, ...]]:
        if not prompts:
            raise EvaluationRuntimeError("generation batch must be non-empty")
        if max_new_tokens < 1 or max_new_tokens > self.max_completion_tokens:
            raise EvaluationRuntimeError("max_new_tokens exceeds sampler capacity")
        prompt_lengths = self.prompt_lengths(prompts)
        top_p_value = None if temperature == 0.0 else top_p
        try:
            import jax
            with jax.set_mesh(self.restored.mesh):
                sampled = self.sampler(
                    list(prompts),
                    max_generation_steps=max_new_tokens,
                    max_prompt_length=self.max_prompt_tokens,
                    temperature=temperature,
                    top_p=top_p_value,
                    seed=seed,
                    return_logits=False,
                    return_logprobs=False,
                    echo=False,
                    pad_output=False,
                )
        except Exception as exc:
            raise EvaluationRuntimeError(
                f"native Tunix generation failed: {type(exc).__name__}: {exc}"
            ) from exc
        texts = tuple(sampled.text)
        completion_lengths = tuple(len(tokens) for tokens in sampled.tokens)
        if len(texts) != len(prompts) or len(completion_lengths) != len(prompts):
            raise EvaluationRuntimeError("sampler output batch size drifted")
        if any(not isinstance(text, str) for text in texts):
            raise EvaluationRuntimeError("sampler returned a non-string completion")
        return texts, prompt_lengths, completion_lengths
