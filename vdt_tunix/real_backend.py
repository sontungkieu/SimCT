"""Bounded real Tunix backend for rollout and frozen-teacher scoring.

The production path uses only locally mounted, provenance-pinned model and
tokenizer files.  Native Tunix safetensors (Qwen), native Gemma Flax/Orbax
checkpoints, and MaxText checkpoints are explicit source modes.  Model
construction is lazy.  Any unsupported or unavailable component fails closed;
there is no mock or network-download fallback.
"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib
import math
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from vdt_tunix.config import ModelConfig, RunConfig
from vdt_tunix.performance import PerformanceContractError, select_length_bucket
from vdt_tunix.contracts import (
    INTERFACE_CONTRACT_VERSION,
    BackendBundle,
    LogitsPayload,
    RolloutRequest,
    StudentRolloutBatch,
    StudentRolloutSample,
    TeacherScoreBatch,
    TeacherScoreRequest,
    TeacherScoreSample,
    TeacherSufficientStatisticsPayload,
)
from vdt_tunix.model_adapters import (
    CausalModelForwardAdapter,
    ModelAdapterError,
    ModelRuntimeDependencies,
    TokenizerByteAdapter,
)


class RealBackendUnavailable(RuntimeError):
    """Raised when the bounded production backend cannot be constructed."""


@dataclasses.dataclass(slots=True)
class _LoadedTunixModel:
    model: Any
    mesh: Any
    forward_fn: Any
    model_config: Any
    sampler: Any | None = None
    forward_statistics_fn: Any | None = None
    forward_cached_statistics_fn: Any | None = None


def _reduce_teacher_step_statistics(
    logits: Any,
    selected_logits: Any,
    overlap_ids: Any,
    *,
    jax_module: Any,
    jnp_module: Any,
    shared_logits: Any | None = None,
) -> tuple[Any, Any]:
    """Reduce one next-token distribution to exact SimCT statistics."""

    values = logits.astype(jnp_module.float32)
    log_normalizer = _blocked_teacher_logsumexp(
        values,
        jax_module=jax_module,
        jnp_module=jnp_module,
    )
    shared = (
        jnp_module.take(values, overlap_ids, axis=-1)
        if shared_logits is None
        else shared_logits.astype(jnp_module.float32)
    )
    selected = selected_logits.astype(jnp_module.float32)
    return shared - log_normalizer[..., None], selected - log_normalizer


def _blocked_teacher_logsumexp(
    values: Any,
    *,
    jax_module: Any,
    jnp_module: Any,
    block_size: int = 4096,
) -> Any:
    """Compute an exact one-step vocabulary log-normalizer in static blocks.

    TPU XLA repeatedly produced invalid post-optimization fusions when the
    exponential reduction consumed the complete Qwen ``B x V`` projection in
    the cached-teacher scan. The projection itself is a bounded one-step
    tensor, so retain it and split only the reduction into compile-time static
    vocabulary blocks. Stable ``logaddexp`` composition is mathematically the
    same log-sum-exp while preventing a single vocabulary-wide ``reduce_sum``
    fusion. Optimization barriers keep block reductions loop-local and stop
    XLA from reconstituting the failing full-width reduction.
    """

    if block_size <= 0:
        raise ValueError("teacher logsumexp block_size must be positive")
    vocabulary_size = int(values.shape[-1])
    if vocabulary_size <= 0:
        raise ValueError("teacher logits must have a non-empty vocabulary axis")
    log_normalizer = jnp_module.full(
        values.shape[:-1],
        -jnp_module.inf,
        dtype=jnp_module.float32,
    )
    for start in range(0, vocabulary_size, block_size):
        block = values[..., start : min(start + block_size, vocabulary_size)]
        block = jax_module.lax.optimization_barrier(block)
        block_log_normalizer = jax_module.scipy.special.logsumexp(
            block,
            axis=-1,
        )
        block_log_normalizer = jax_module.lax.optimization_barrier(
            block_log_normalizer
        )
        log_normalizer = jnp_module.logaddexp(
            log_normalizer,
            block_log_normalizer,
        )
        log_normalizer = jax_module.lax.optimization_barrier(log_normalizer)
    return log_normalizer


def _project_qwen_selected_logits(
    module: Any,
    hidden_state: Any,
    token_ids: Any,
    *,
    jax_module: Any,
    jnp_module: Any,
) -> Any:
    """Project realized Qwen logits without indexing a ``B x V`` tensor.

    TPU XLA independently miscompiled selected-token gather, reduce-sum,
    batch-dot, scalar-slice, and compare/select forms after each was fused with
    the full vocabulary projection inside the cached-teacher scan. Qwen's
    ordinary input embedding already performs a dynamic row lookup on the
    immutable parameter table. Reuse that conventional layout here: gather
    only the realized LM-head row, then contract it with the one-step hidden
    state. The full projection still supplies the exact log-normalizer and
    shared coordinates, but no dynamic operation consumes its ``B x V``
    result.
    """

    config = getattr(module, "config", None)
    if config is None or not hasattr(config, "use_tied_embedding"):
        raise RealBackendUnavailable(
            "cached teacher selected-logit projection requires Qwen embedding config"
        )
    try:
        if config.use_tied_embedding:
            weight = module.embedder.input_embedding.value
            compute_dtype = module.embedder.dtype
            vocabulary_rows = weight
        else:
            weight = module.lm_head.w.value
            compute_dtype = module.lm_head.dtype
            vocabulary_rows = jnp_module.swapaxes(weight, 0, 1)
    except AttributeError as exc:
        raise RealBackendUnavailable(
            "cached teacher selected-logit projection requires Qwen LM-head weights"
        ) from exc

    hidden = jnp_module.asarray(hidden_state, dtype=compute_dtype)
    hidden = jax_module.lax.optimization_barrier(hidden)
    selected_rows = jnp_module.asarray(
        vocabulary_rows[token_ids],
        dtype=compute_dtype,
    )
    selected_rows = jax_module.lax.optimization_barrier(selected_rows)
    selected = jnp_module.einsum("bd,bd->b", hidden, selected_rows)
    selected = jax_module.lax.optimization_barrier(selected)
    return selected.astype(jnp_module.float32)


def _project_qwen_shared_logits(
    module: Any,
    hidden_state: Any,
    overlap_ids: Any,
    *,
    jax_module: Any,
    jnp_module: Any,
) -> Any:
    """Project fixed shared-vocabulary coordinates from LM-head rows."""

    config = getattr(module, "config", None)
    if config is None or not hasattr(config, "use_tied_embedding"):
        raise RealBackendUnavailable(
            "cached teacher shared-logit projection requires Qwen embedding config"
        )
    try:
        if config.use_tied_embedding:
            weight = module.embedder.input_embedding.value
            compute_dtype = module.embedder.dtype
            vocabulary_rows = weight
        else:
            weight = module.lm_head.w.value
            compute_dtype = module.lm_head.dtype
            vocabulary_rows = jnp_module.swapaxes(weight, 0, 1)
    except AttributeError as exc:
        raise RealBackendUnavailable(
            "cached teacher shared-logit projection requires Qwen LM-head weights"
        ) from exc

    hidden = jnp_module.asarray(hidden_state, dtype=compute_dtype)
    hidden = jax_module.lax.optimization_barrier(hidden)
    shared_rows = jnp_module.asarray(
        vocabulary_rows[overlap_ids],
        dtype=compute_dtype,
    )
    shared_rows = jax_module.lax.optimization_barrier(shared_rows)
    shared = jnp_module.einsum("bd,od->bo", hidden, shared_rows)
    shared = jax_module.lax.optimization_barrier(shared)
    return shared.astype(jnp_module.float32)


def _configure_qwen_compute_dtype(
    model_params: Any,
    *,
    compute_dtype: Any,
) -> Any:
    """Make the Qwen activation dtype explicit before model construction."""

    if not hasattr(model_params, "dtype"):
        raise RealBackendUnavailable(
            "Qwen model config does not expose a compute dtype"
        )
    model_params.dtype = compute_dtype
    return model_params


def _cached_teacher_forcing_scan(
    initial_state: Any,
    initial_cache: Any,
    completion_token_ids: Any,
    overlap_ids: Any,
    decode_one: Any,
    project_logits: Any,
    project_selected_logits: Any,
    project_shared_logits: Any | None = None,
    *,
    jax_module: Any,
    jnp_module: Any,
) -> tuple[Any, Any]:
    """Score a fixed completion exactly without carrying vocabulary logits.

    ``initial_state`` predicts completion token zero after projection.  Each
    scan iteration projects and reduces that state before feeding the realized
    token through the KV cache.  Keeping hidden state, rather than logits, in
    the loop carry prevents XLA from materializing a completion-wide ``T x B x
    V`` vocabulary tensor.
    """

    token_steps = jnp_module.swapaxes(completion_token_ids, 0, 1)
    step_ids = jnp_module.arange(token_steps.shape[0], dtype=jnp_module.int32)

    def scan_step(carry: tuple[Any, Any], inputs: tuple[Any, Any]):
        current_state, cache = carry
        token_ids, step = inputs
        current_logits = project_logits(current_state)
        current_selected_logits = project_selected_logits(current_state, token_ids)
        current_shared_logits = (
            None
            if project_shared_logits is None
            else project_shared_logits(current_state)
        )
        shared, selected = _reduce_teacher_step_statistics(
            current_logits,
            current_selected_logits,
            overlap_ids,
            jax_module=jax_module,
            jnp_module=jnp_module,
            shared_logits=current_shared_logits,
        )
        next_state, next_cache = decode_one(cache, token_ids, step)
        return (next_state, next_cache), (shared, selected)

    (_, _), (shared_steps, selected_steps) = jax_module.lax.scan(
        scan_step,
        (initial_state, initial_cache),
        (token_steps, step_ids),
    )
    return (
        jnp_module.swapaxes(shared_steps, 0, 1),
        jnp_module.swapaxes(selected_steps, 0, 1),
    )


def _qwen_cached_teacher_statistics(
    module: Any,
    prompt_ids: Any,
    prompt_mask: Any,
    completion_token_ids: Any,
    completion_mask: Any,
    overlap_ids: Any,
    *,
    model_params: Any,
    configured_cache_size: int,
    generate_sampler: Any,
    generate_utils: Any,
    jax_module: Any,
    jnp_module: Any,
) -> tuple[Any, Any]:
    """Run exact Qwen prompt prefill plus cached teacher forcing."""

    for field in ("num_layers", "num_kv_heads", "head_dim", "dtype"):
        if not hasattr(model_params, field):
            raise RealBackendUnavailable(
                "cached teacher forcing requires model config field " f"{field}"
            )

    prompt_active = prompt_mask.astype(jnp_module.bool_)
    completion_active = completion_mask.astype(jnp_module.bool_)
    batch_size, prompt_width = prompt_ids.shape
    completion_width = completion_token_ids.shape[1]
    cache_size = configured_cache_size + 1
    if prompt_width + completion_width > configured_cache_size:
        raise RealBackendUnavailable(
            "cached teacher forcing prompt/completion shapes exceed "
            "the configured sequence budget"
        )
    cache = generate_sampler._init_cache(
        n_layers=int(model_params.num_layers),
        cache_size=cache_size,
        batch_size=batch_size,
        num_kv_heads=int(model_params.num_kv_heads),
        head_dim=int(model_params.head_dim),
        dtype=model_params.dtype,
    )
    prompt_positions = generate_utils.build_positions_from_mask(prompt_active)
    prefill_attention = generate_utils.make_causal_attn_mask(
        prompt_active,
        cache_size,
    )
    prompt_hidden, cache = module(
        prompt_ids,
        prompt_positions,
        cache,
        prefill_attention,
        skip_lm_head=True,
    )
    initial_hidden = prompt_hidden[:, -1, :]

    trailing_width = cache_size - prompt_width - completion_width
    padding_mask = jnp_module.concatenate(
        (
            ~prompt_active,
            ~completion_active,
            jnp_module.ones(
                (batch_size, trailing_width),
                dtype=jnp_module.bool_,
            ),
        ),
        axis=-1,
    )
    prompt_lengths = jnp_module.sum(
        prompt_active,
        axis=-1,
        dtype=jnp_module.int32,
    )

    def decode_one(current_cache: Any, token_ids: Any, step: Any):
        decode_position = prompt_lengths + step
        attention_mask = generate_utils.compute_attention_masks(
            prompt_width + step,
            cache_size,
            padding_mask,
        )
        hidden, updated_cache = module(
            token_ids[:, None],
            decode_position[:, None],
            current_cache,
            attention_mask,
            skip_lm_head=True,
        )
        return hidden[:, 0, :], updated_cache

    def project_logits(hidden_state: Any):
        # XLA can otherwise lift the linear vocabulary projection out of the
        # scan and batch it over the complete teacher-token axis.  That is
        # mathematically valid but recreates the infeasible T x B x V tensor
        # that cached scoring is meant to avoid.  Bracket the one-step head so
        # projection and sufficient-statistic reduction remain loop-local.
        hidden_state = jax_module.lax.optimization_barrier(hidden_state)
        logits = module.compute_final_logits(hidden_state[:, None, :])[:, 0, :]
        return jax_module.lax.optimization_barrier(logits)

    def project_selected_logits(hidden_state: Any, token_ids: Any):
        return _project_qwen_selected_logits(
            module,
            hidden_state,
            token_ids,
            jax_module=jax_module,
            jnp_module=jnp_module,
        )

    def project_shared_logits(hidden_state: Any):
        return _project_qwen_shared_logits(
            module,
            hidden_state,
            overlap_ids,
            jax_module=jax_module,
            jnp_module=jnp_module,
        )

    return _cached_teacher_forcing_scan(
        initial_hidden,
        cache,
        completion_token_ids,
        overlap_ids,
        decode_one,
        project_logits,
        project_selected_logits,
        project_shared_logits,
        jax_module=jax_module,
        jnp_module=jnp_module,
    )


def _normalize_tokenizer_padding(tokenizer: Any, tokenizer_type: str) -> Any:
    """Add an HF pad token without imposing HF attributes on SentencePiece.

    Tunix's SentencePiece adapter exposes ``pad_id()`` and ``eos_id()``; it
    intentionally does not expose Hugging Face's ``pad_token_id`` or
    ``eos_token`` attributes.  ``TokenizerByteAdapter`` validates the numeric
    SentencePiece ids directly, so only Hugging Face tokenizers need this
    normalization step.
    """

    if tokenizer_type != "huggingface":
        return tokenizer
    if getattr(tokenizer, "pad_token_id", None) is not None:
        return tokenizer
    eos_token = getattr(tokenizer, "eos_token", None)
    if eos_token is None:
        raise RealBackendUnavailable(
            "Hugging Face tokenizer has neither a pad token nor an EOS token"
        )
    tokenizer.pad_token = eos_token
    return tokenizer


def _native_model_family(model_id: str) -> str:
    """Return the exact Tunix forward-signature family used by this run."""

    if model_id.startswith("gemma-2-"):
        return "gemma2"
    if model_id.startswith("qwen2.5-"):
        return "qwen2p5"
    raise RealBackendUnavailable(
        "the bounded native forward adapter supports Gemma 2 and Qwen 2.5; "
        f"received {model_id!r}"
    )


def _call_native_tunix_model(
    module: Any,
    *,
    family: str,
    input_ids: Any,
    positions: Any,
    attention_mask: Any,
    segments: Any,
) -> Any:
    """Call the pinned Tunix model with its family-specific signature."""

    if family == "gemma2":
        logits, _ = module(
            input_ids,
            positions,
            None,
            attention_mask,
        )
        return logits
    if family == "qwen2p5":
        logits, _ = module(
            input_ids,
            positions,
            None,
            attention_mask,
            segment_ids=segments,
        )
        return logits
    raise RealBackendUnavailable(f"unsupported native Tunix family {family!r}")


def _validate_model_path(config: ModelConfig) -> Path:
    uri = config.resolved_model_path
    if "://" in uri:
        raise RealBackendUnavailable(
            "the bounded backend supports local mounted model directories "
            f"only, not URI {uri!r}"
        )
    path = Path(uri).expanduser()
    if not path.is_absolute():
        raise RealBackendUnavailable(
            f"model path must be absolute: {uri!r}"
        )
    if not path.is_dir():
        raise RealBackendUnavailable(
            f"model directory is unavailable: {path}"
        )
    return path.resolve()


def _production_dependencies(config: RunConfig) -> ModelRuntimeDependencies:
    """Import pinned runtime APIs without installing or downloading anything."""

    try:
        jax = importlib.import_module("jax")
        jnp = importlib.import_module("jax.numpy")
        nnx = importlib.import_module("flax.nnx")
        automodel = importlib.import_module("tunix.models.automodel")
        maxtext_parallelism = importlib.import_module(
            "tunix.models.maxtext_parallelism"
        )
        generate_sampler = importlib.import_module("tunix.generate.sampler")
        generate_utils = importlib.import_module("tunix.generate.utils")
        mesh_lib = importlib.import_module("tunix.utils.mesh")
        transformers = importlib.import_module("transformers")
    except ImportError as exc:
        raise RealBackendUnavailable(
            "required Tunix real-backend dependency is unavailable: "
            f"{exc.name or exc}"
        ) from exc

    mesh_cache: dict[str, Any] = {}

    def validate_model_spec(model_config: ModelConfig) -> None:
        try:
            automodel.call_model_config(model_config.model_id)
        except Exception as exc:
            raise RealBackendUnavailable(
                f"unsupported Tunix model spec {model_config.model_id!r}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

    def load_tokenizer(model_config: ModelConfig) -> Any:
        try:
            tokenizer_path = Path(model_config.resolved_tokenizer_path).expanduser()
            if not tokenizer_path.is_absolute() or not tokenizer_path.exists():
                raise RealBackendUnavailable(
                    "tokenizer_path must be a locally mounted absolute path: "
                    f"{model_config.resolved_tokenizer_path!r}"
                )
            if model_config.tokenizer_type == "huggingface":
                tokenizer = transformers.AutoTokenizer.from_pretrained(
                    str(tokenizer_path),
                    local_files_only=True,
                    trust_remote_code=False,
                    token=None,
                )
            else:
                tokenizer_lib = importlib.import_module(
                    "tunix.generate.tokenizer_adapter"
                )
                tokenizer = tokenizer_lib.Tokenizer(
                    tokenizer_type="sentencepiece",
                    tokenizer_path=str(tokenizer_path),
                    add_bos=False,
                    add_eos=False,
                    hf_access_token=None,
                )
        except Exception as exc:
            raise RealBackendUnavailable(
                "tokenizer is not available locally at the pinned revision for "
                f"{model_config.tokenizer_id!r}: {type(exc).__name__}: {exc}"
            ) from exc
        return _normalize_tokenizer_padding(
            tokenizer,
            model_config.tokenizer_type,
        )

    def require_mesh() -> Any:
        if "mesh" in mesh_cache:
            return mesh_cache["mesh"]
        devices = list(jax.devices())
        if len(devices) != config.tpu.expected_device_count:
            raise RealBackendUnavailable(
                "model loading requires the validated eight-device runtime; "
                f"observed {len(devices)} devices"
            )
        sources = {config.student.model_source, config.teacher.model_source}
        if "maxtext" in sources and len(sources) != 1:
            raise RealBackendUnavailable(
                "student and teacher cannot mix MaxText and native Tunix meshes"
            )
        if sources == {"maxtext"}:
            axis_sizes = {
                "stage": config.tpu.pipeline_parallelism,
                "tensor": config.tpu.tensor_parallelism,
            }
            axis_names = maxtext_parallelism.MAXTEXT_MESH_AXIS_NAMES
            mesh_shape = tuple(axis_sizes.get(axis, 1) for axis in axis_names)
        else:
            if config.tpu.pipeline_parallelism != 1:
                raise RealBackendUnavailable(
                    "native Tunix model sources currently require PP1"
                )
            axis_names = ("fsdp", "tp")
            mesh_shape = (
                config.tpu.fsdp_parallelism,
                config.tpu.tensor_parallelism,
            )
        mesh_cache["mesh"] = mesh_lib.create_mesh(
            mesh_shape,
            axis_names,
            devices=devices,
        )
        return mesh_cache["mesh"]

    def load_model(model_config: ModelConfig, trainable: bool) -> Any:
        model_path = _validate_model_path(model_config)
        mesh = require_mesh()
        model_family = _native_model_family(model_config.model_id)
        sample_batch = (
            config.rollout.prompt_batch_size * config.rollout.samples_per_prompt
        )
        max_target_length = (
            config.rollout.max_prompt_tokens
            + config.rollout.max_completion_tokens
        )
        try:
            with jax.set_mesh(mesh):
                if model_config.model_source == "maxtext":
                    model, model_params = automodel.AutoModel.from_pretrained(
                        model_id=model_config.model_id,
                        model_path=str(model_path),
                        mesh=mesh,
                        model_source=automodel.ModelSource.MAXTEXT,
                        per_device_batch_size=(
                            sample_batch / config.tpu.expected_device_count
                        ),
                        max_target_length=max_target_length,
                        steps=1,
                        dataset_type="synthetic",
                        enable_checkpointing=True,
                        enable_dropout=False,
                        attention="dot_product",
                        remat_policy="full",
                        scan_layers=True,
                        scan_layers_per_stage=False,
                        skip_jax_distributed_system=True,
                        hf_access_token="",
                    )
                elif model_config.model_source == "huggingface":
                    model_params = automodel.call_model_config(
                        model_config.model_id
                    )
                    if model_family == "qwen2p5":
                        _configure_qwen_compute_dtype(
                            model_params,
                            compute_dtype=jnp.bfloat16,
                        )
                    model = automodel.create_model_from_safe_tensors(
                        model_config.model_id,
                        str(model_path),
                        model_params,
                        mesh,
                        dtype=jnp.bfloat16,
                    )
                else:
                    candidates = [
                        child
                        for child in model_path.iterdir()
                        if child.is_dir()
                        and (
                            (child / "_CHECKPOINT_METADATA").exists()
                            or (child / "checkpoint").exists()
                        )
                    ]
                    if len(candidates) != 1:
                        raise RealBackendUnavailable(
                            "Gemma Kaggle mount must contain exactly one Flax "
                            f"checkpoint directory; found {candidates}"
                        )
                    cache_role = "student" if trainable else "teacher"
                    intermediate = (
                        Path(config.checkpoint.root).parent
                        / "base-model-cache"
                        / cache_role
                    )
                    model, model_params = automodel.create_gemma_model_with_nnx_conversion(
                        model_name=model_config.model_id,
                        ckpt_path=str(model_path),
                        intermediate_ckpt_dir=str(intermediate),
                        rng_seed=0,
                        mesh=mesh,
                        model_path=candidates[0].name,
                    )
        except Exception as exc:
            raise RealBackendUnavailable(
                f"local Tunix model restore failed for "
                f"{model_config.model_id!r}: {type(exc).__name__}: {exc}"
            ) from exc

        @nnx.jit
        def forward_fn(module: Any, input_ids: Any, positions: Any, segments: Any):
            input_mask = segments.astype(jnp.bool_)
            sequence_length = input_ids.shape[-1]
            causal_mask = jnp.tril(
                jnp.ones(
                    (sequence_length, sequence_length),
                    dtype=jnp.bool_,
                )
            )
            attention_mask = input_mask[:, None, :] & causal_mask[None, ...]
            return _call_native_tunix_model(
                module,
                family=model_family,
                input_ids=input_ids,
                positions=positions,
                attention_mask=attention_mask,
                segments=segments,
            )

        @nnx.jit
        def forward_statistics_fn(
            module: Any,
            input_ids: Any,
            positions: Any,
            segments: Any,
            completion_positions: Any,
            completion_token_ids: Any,
            overlap_ids: Any,
        ):
            input_mask = segments.astype(jnp.bool_)
            sequence_length = input_ids.shape[-1]
            causal_mask = jnp.tril(
                jnp.ones(
                    (sequence_length, sequence_length),
                    dtype=jnp.bool_,
                )
            )
            attention_mask = input_mask[:, None, :] & causal_mask[None, ...]
            logits = _call_native_tunix_model(
                module,
                family=model_family,
                input_ids=input_ids,
                positions=positions,
                attention_mask=attention_mask,
                segments=segments,
            )
            batch = jnp.arange(input_ids.shape[0], dtype=jnp.int32)[:, None]
            completion_logits = logits[batch, completion_positions].astype(
                jnp.float32
            )
            log_normalizer = jax.scipy.special.logsumexp(
                completion_logits, axis=-1
            )
            shared_log_probs = (
                jnp.take(completion_logits, overlap_ids, axis=-1)
                - log_normalizer[..., None]
            )
            selected_logits = jnp.take_along_axis(
                completion_logits,
                completion_token_ids[..., None],
                axis=-1,
            )[..., 0]
            return shared_log_probs, selected_logits - log_normalizer

        @nnx.jit
        def forward_cached_statistics_fn(
            module: Any,
            prompt_ids: Any,
            prompt_mask: Any,
            completion_token_ids: Any,
            completion_mask: Any,
            overlap_ids: Any,
        ):
            if model_family != "qwen2p5":
                raise RealBackendUnavailable(
                    "cached teacher forcing currently supports Qwen 2.5 only"
                )
            configured_cache_size = (
                config.rollout.max_sequence_tokens
                or config.rollout.max_prompt_tokens
                + config.rollout.max_completion_tokens
            )
            configured_cache_size = max(
                configured_cache_size,
                prompt_ids.shape[1] + completion_token_ids.shape[1],
            )
            return _qwen_cached_teacher_statistics(
                module,
                prompt_ids,
                prompt_mask,
                completion_token_ids,
                completion_mask,
                overlap_ids,
                model_params=model_params,
                configured_cache_size=configured_cache_size,
                generate_sampler=generate_sampler,
                generate_utils=generate_utils,
                jax_module=jax,
                jnp_module=jnp,
            )

        return _LoadedTunixModel(
            model=model,
            mesh=mesh,
            forward_fn=forward_fn,
            model_config=model_params,
            forward_statistics_fn=forward_statistics_fn,
            forward_cached_statistics_fn=forward_cached_statistics_fn,
        )

    def forward_model(
        loaded: _LoadedTunixModel,
        input_ids: Any,
        segment_ids: Any,
    ) -> Any:
        ids = jnp.asarray(input_ids, dtype=jnp.int32)
        segments = jnp.asarray(segment_ids, dtype=jnp.int32)
        if ids.ndim != 2 or segments.shape != ids.shape:
            raise ModelAdapterError(
                "input_ids and segment_ids must be equal-shape rank-2 arrays"
            )
        positions = jnp.maximum(jnp.cumsum(segments, axis=-1) - 1, 0)
        with jax.set_mesh(loaded.mesh):
            return loaded.forward_fn(loaded.model, ids, positions, segments)

    def forward_sufficient_statistics(
        loaded: _LoadedTunixModel,
        input_ids: Any,
        segment_ids: Any,
        completion_positions: Any,
        completion_token_ids: Any,
        overlap_ids: Any,
    ) -> tuple[Any, Any]:
        ids = jnp.asarray(input_ids, dtype=jnp.int32)
        segments = jnp.asarray(segment_ids, dtype=jnp.int32)
        selected_positions = jnp.asarray(completion_positions, dtype=jnp.int32)
        selected_ids = jnp.asarray(completion_token_ids, dtype=jnp.int32)
        shared_ids = jnp.asarray(overlap_ids, dtype=jnp.int32)
        if ids.ndim != 2 or segments.shape != ids.shape:
            raise ModelAdapterError(
                "input_ids and segment_ids must be equal-shape rank-2 arrays"
            )
        if selected_positions.ndim != 2 or selected_ids.shape != selected_positions.shape:
            raise ModelAdapterError(
                "completion positions and ids must be equal-shape rank-2 arrays"
            )
        if selected_positions.shape[0] != ids.shape[0] or shared_ids.ndim != 1:
            raise ModelAdapterError("teacher sufficient-statistic batch shape mismatch")
        if loaded.forward_statistics_fn is None:
            raise ModelAdapterError("loaded model lacks sufficient-statistic forward")
        positions = jnp.maximum(jnp.cumsum(segments, axis=-1) - 1, 0)
        with jax.set_mesh(loaded.mesh):
            return loaded.forward_statistics_fn(
                loaded.model,
                ids,
                positions,
                segments,
                selected_positions,
                selected_ids,
                shared_ids,
            )

    def forward_cached_sufficient_statistics(
        loaded: _LoadedTunixModel,
        prompt_ids: Any,
        prompt_mask: Any,
        completion_token_ids: Any,
        completion_mask: Any,
        overlap_ids: Any,
    ) -> tuple[Any, Any]:
        prompts = jnp.asarray(prompt_ids, dtype=jnp.int32)
        prompts_active = jnp.asarray(prompt_mask, dtype=jnp.bool_)
        completion_ids = jnp.asarray(completion_token_ids, dtype=jnp.int32)
        completion_active = jnp.asarray(completion_mask, dtype=jnp.bool_)
        shared_ids = jnp.asarray(overlap_ids, dtype=jnp.int32)
        if prompts.ndim != 2 or prompts_active.shape != prompts.shape:
            raise ModelAdapterError(
                "cached prompt ids and mask must be equal-shape rank-2 arrays"
            )
        if completion_ids.ndim != 2 or completion_active.shape != completion_ids.shape:
            raise ModelAdapterError(
                "cached completion ids and mask must be equal-shape rank-2 arrays"
            )
        if completion_ids.shape[0] != prompts.shape[0] or shared_ids.ndim != 1:
            raise ModelAdapterError("cached teacher statistic batch shape mismatch")
        if loaded.forward_cached_statistics_fn is None:
            raise ModelAdapterError("loaded model lacks cached statistic forward")
        with jax.set_mesh(loaded.mesh):
            return loaded.forward_cached_statistics_fn(
                loaded.model,
                prompts,
                prompts_active,
                completion_ids,
                completion_active,
                shared_ids,
            )

    return ModelRuntimeDependencies(
        name="tunix-local-mounted",
        production=True,
        validate_model_spec=validate_model_spec,
        load_tokenizer=load_tokenizer,
        load_model=load_model,
        forward_model=forward_model,
        stop_gradient=jax.lax.stop_gradient,
        to_host=jax.device_get,
        forward_sufficient_statistics=forward_sufficient_statistics,
        forward_cached_sufficient_statistics=forward_cached_sufficient_statistics,
    )


def _stable_seed(*parts: object) -> int:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _rollout_seed(config: RunConfig, request: RolloutRequest, *parts: object) -> int:
    """Return an explicit multi-seed RNG while preserving legacy parity."""

    if config.training.seed is None:
        return _stable_seed(request.run_id, request.step, *parts)
    return _stable_seed("training-seed", config.training.seed, request.step, *parts)


def _log_softmax(values: Any) -> Any:
    import numpy as np

    array = np.asarray(values, dtype=np.float64)
    maximum = float(array.max())
    return array - (maximum + math.log(float(np.exp(array - maximum).sum())))


def _sample_next_token(
    logits: Any,
    *,
    temperature: float,
    top_p: float,
    rng: Any,
) -> tuple[int, float]:
    import numpy as np

    raw = np.asarray(logits, dtype=np.float64)
    if raw.ndim != 1 or raw.shape[0] < 2 or not np.isfinite(raw).all():
        raise ModelAdapterError("next-token logits must be a finite vocabulary vector")
    scaled = raw if temperature == 0.0 else raw / temperature
    full_log_probs = _log_softmax(scaled)
    if temperature == 0.0:
        token_id = int(np.argmax(scaled))
        return token_id, float(full_log_probs[token_id])

    probabilities = np.exp(full_log_probs)
    if top_p < 1.0:
        order = np.argsort(-probabilities)
        ordered = probabilities[order]
        keep_count = int(np.searchsorted(np.cumsum(ordered), top_p, side="left")) + 1
        keep = order[:keep_count]
        truncated = probabilities[keep]
        truncated = truncated / truncated.sum()
        token_id = int(rng.choice(keep, p=truncated))
    else:
        token_id = int(rng.choice(len(probabilities), p=probabilities))
    # Match Tunix sampler semantics: return the full tempered-model log-prob,
    # even when top-p truncation selected the token.
    return token_id, float(full_log_probs[token_id])


class TunixStudentRolloutBackend:
    backend_name = "tunix-maxtext-student-forward-rollout"

    def __init__(
        self,
        config: RunConfig,
        model_adapter: CausalModelForwardAdapter,
    ):
        self._config = config
        self.model_adapter = model_adapter
        self.real_model_integration = model_adapter.dependencies.production
        self.last_phase_timings: dict[str, float | int] = {}

    def _tunix_sampler_rollout(
        self,
        request: RolloutRequest,
        rows: Sequence[
            tuple[Any, int, str, tuple[int, ...], tuple[int, ...]]
        ],
    ) -> StudentRolloutBatch:
        """Use Tunix's KV-cached sampler for the production native-model path."""

        if self._config.student.model_source == "maxtext":
            raise ModelAdapterError(
                "the KV-cached rollout path currently requires a native Tunix model"
            )
        try:
            from flax import nnx
            from tunix.generate import sampler as sampler_lib
        except ImportError as exc:
            raise ModelAdapterError(
                f"Tunix sampler dependency is unavailable: {exc}"
            ) from exc

        loaded = self.model_adapter.require_loaded_model()
        if not isinstance(loaded, _LoadedTunixModel):
            raise ModelAdapterError("production loader returned an unknown model wrapper")
        params = loaded.model_config
        for name in ("num_layers", "num_kv_heads", "head_dim"):
            if not hasattr(params, name):
                raise ModelAdapterError(
                    f"native model config does not expose sampler field {name}"
                )
        if loaded.sampler is None:
            loaded.sampler = sampler_lib.Sampler(
                transformer=loaded.model,
                tokenizer=self.model_adapter.tokenizer.raw_tokenizer,
                cache_config=sampler_lib.CacheConfig(
                    cache_size=(
                        (
                            self._config.rollout.max_sequence_tokens
                            or self._config.rollout.max_prompt_tokens
                            + self._config.rollout.max_completion_tokens
                        )
                        + 1
                    ),
                    num_layers=int(params.num_layers),
                    num_kv_heads=int(params.num_kv_heads),
                    head_dim=int(params.head_dim),
                ),
            )
        else:
            # The student changes after every optimizer step.  Synchronize the
            # sampler before generating the next on-policy batch.
            loaded.sampler.transformer_state = nnx.state(loaded.model)

        prompts = [row[0].student_prompt for row in rows]
        top_p = (
            None
            if self._config.rollout.temperature == 0.0
            else self._config.rollout.top_p
        )
        max_generation_steps = self._config.rollout.max_completion_tokens
        if self._config.rollout.max_sequence_tokens is not None:
            max_generation_steps = min(
                max_generation_steps,
                self._config.rollout.max_sequence_tokens
                - self._config.rollout.max_prompt_tokens,
            )
        if max_generation_steps < 1:
            raise ModelAdapterError(
                "no completion capacity remains under max_sequence_tokens"
            )
        forced_kwargs: dict[str, Any] = {}
        if self._config.rollout.force_max_completion:
            forced_kwargs = {
                "eos_tokens": [self.model_adapter.tokenizer.pad_token_id],
                "forbidden_tokens": sorted(
                    self.model_adapter.tokenizer.special_token_ids
                ),
            }
        with __import__("jax").set_mesh(loaded.mesh):
            sampled = loaded.sampler(
                prompts,
                max_generation_steps=max_generation_steps,
                max_prompt_length=self._config.rollout.max_prompt_tokens,
                temperature=self._config.rollout.temperature,
                top_p=top_p,
                seed=_rollout_seed(self._config, request) % (2**31 - 1),
                return_logits=False,
                return_logprobs=True,
                echo=False,
                pad_output=False,
                **forced_kwargs,
            )
        if len(sampled.tokens) != len(rows) or sampled.logprobs is None:
            raise ModelAdapterError("Tunix sampler returned an invalid sample batch")

        samples: list[StudentRolloutSample] = []
        for row, token_ids, log_probs in zip(
            rows, sampled.tokens, sampled.logprobs, strict=True
        ):
            prompt, _, sample_id, prompt_ids, _ = row
            generated = tuple(int(value) for value in token_ids.tolist())
            completion = self.model_adapter.tokenizer.continuation_from_generated_ids(
                prompt_text=prompt.student_prompt,
                prompt_token_ids=prompt_ids,
                completion_token_ids=generated,
            )
            samples.append(
                StudentRolloutSample(
                    sample_id=sample_id,
                    prompt_id=prompt.prompt_id,
                    student_prompt_token_ids=prompt_ids,
                    completion=completion,
                    rollout_log_probs=tuple(float(value) for value in log_probs),
                )
            )
        model = self._config.student
        return StudentRolloutBatch(
            contract_version=INTERFACE_CONTRACT_VERSION,
            run_id=request.run_id,
            step=request.step,
            model_id=model.model_id,
            model_revision=model.model_revision,
            tokenizer_id=model.tokenizer_id,
            tokenizer_revision=model.tokenizer_revision,
            samples=tuple(samples),
        )

    def rollout(self, request: RolloutRequest) -> StudentRolloutBatch:
        import numpy as np

        started = time.monotonic()

        rows: list[
            tuple[Any, int, str, tuple[int, ...], tuple[int, ...]]
        ] = []
        for prompt in request.prompts:
            prompt_ids = self.model_adapter.tokenizer.encode(prompt.student_prompt)
            model_prompt_ids = self.model_adapter.tokenizer.with_model_prefix(
                prompt_ids
            )
            if len(model_prompt_ids) > self._config.rollout.max_prompt_tokens:
                raise ModelAdapterError(
                    f"student prompt {prompt.prompt_id!r} exceeds max_prompt_tokens"
                )
            for sample_index in range(request.samples_per_prompt):
                sample_id = f"{prompt.prompt_id}/{sample_index}"
                rows.append(
                    (
                        prompt,
                        sample_index,
                        sample_id,
                        prompt_ids,
                        model_prompt_ids,
                    )
                )

        if self.real_model_integration:
            requested_generation_steps = self._config.rollout.max_completion_tokens
            if self._config.rollout.max_sequence_tokens is not None:
                requested_generation_steps = min(
                    requested_generation_steps,
                    self._config.rollout.max_sequence_tokens
                    - self._config.rollout.max_prompt_tokens,
                )
            result = self._tunix_sampler_rollout(request, rows)
            self.last_phase_timings = {
                "student_rollout_s": time.monotonic() - started,
                "student_rollout_records": len(result.samples),
                "student_rollout_requested_tokens": (
                    len(result.samples) * requested_generation_steps
                ),
                "student_rollout_forced_length": int(
                    self._config.rollout.force_max_completion
                ),
            }
            return result

        max_prompt = max(len(row[4]) for row in rows)
        total_length = max_prompt + self._config.rollout.max_completion_tokens
        pad_id = self.model_adapter.tokenizer.pad_token_id
        input_ids = np.full((len(rows), total_length), pad_id, dtype=np.int32)
        segment_ids = np.zeros_like(input_ids)
        lengths = np.zeros((len(rows),), dtype=np.int32)
        generated: list[list[int]] = [[] for _ in rows]
        rollout_log_probs: list[list[float]] = [[] for _ in rows]
        finished = np.zeros((len(rows),), dtype=bool)
        rngs = [
            np.random.default_rng(
                _rollout_seed(self._config, request, row[2])
            )
            for row in rows
        ]
        for row_index, (_, _, _, _, model_prompt_ids) in enumerate(rows):
            width = len(model_prompt_ids)
            input_ids[row_index, :width] = model_prompt_ids
            segment_ids[row_index, :width] = 1
            lengths[row_index] = width

        for _ in range(self._config.rollout.max_completion_tokens):
            logits = self.model_adapter.to_host(
                self.model_adapter.forward(input_ids, segment_ids)
            )
            logits = np.asarray(logits)
            for row_index in range(len(rows)):
                if finished[row_index]:
                    continue
                position = int(lengths[row_index] - 1)
                token_id, log_prob = _sample_next_token(
                    logits[row_index, position],
                    temperature=self._config.rollout.temperature,
                    top_p=self._config.rollout.top_p,
                    rng=rngs[row_index],
                )
                if token_id == self.model_adapter.tokenizer.eos_token_id:
                    finished[row_index] = True
                    continue
                if token_id in self.model_adapter.tokenizer.special_token_ids:
                    raise ModelAdapterError(
                        "student generated a non-terminal special token"
                    )
                write_position = int(lengths[row_index])
                input_ids[row_index, write_position] = token_id
                segment_ids[row_index, write_position] = 1
                lengths[row_index] += 1
                generated[row_index].append(token_id)
                rollout_log_probs[row_index].append(log_prob)
            if bool(finished.all()):
                break

        samples: list[StudentRolloutSample] = []
        for row_index, (prompt, _, sample_id, prompt_ids, _) in enumerate(rows):
            completion = self.model_adapter.tokenizer.continuation_from_generated_ids(
                prompt_text=prompt.student_prompt,
                prompt_token_ids=prompt_ids,
                completion_token_ids=generated[row_index],
            )
            samples.append(
                StudentRolloutSample(
                    sample_id=sample_id,
                    prompt_id=prompt.prompt_id,
                    student_prompt_token_ids=prompt_ids,
                    completion=completion,
                    rollout_log_probs=tuple(rollout_log_probs[row_index]),
                )
            )
        model = self._config.student
        result = StudentRolloutBatch(
            contract_version=INTERFACE_CONTRACT_VERSION,
            run_id=request.run_id,
            step=request.step,
            model_id=model.model_id,
            model_revision=model.model_revision,
            tokenizer_id=model.tokenizer_id,
            tokenizer_revision=model.tokenizer_revision,
            samples=tuple(samples),
        )
        self.last_phase_timings = {
            "student_rollout_s": time.monotonic() - started,
            "student_rollout_records": len(result.samples),
        }
        return result


class MaxTextFrozenTeacherScoreBackend:
    backend_name = "tunix-frozen-teacher-forward"

    def __init__(
        self,
        config: RunConfig,
        model_adapter: CausalModelForwardAdapter,
    ):
        self._config = config
        self.model_adapter = model_adapter
        self.real_model_integration = model_adapter.dependencies.production
        self.last_phase_timings: dict[str, float | int] = {}
        self._overlap_token_ids: tuple[int, ...] | None = None

    def configure_overlap_token_ids(self, token_ids: Sequence[int]) -> None:
        normalized = tuple(int(value) for value in token_ids)
        if not normalized:
            raise ModelAdapterError("teacher overlap token ids must not be empty")
        self._overlap_token_ids = normalized

    def score(self, request: TeacherScoreRequest) -> TeacherScoreBatch:
        import numpy as np

        tokenize_started = time.monotonic()
        prompt_by_id = {prompt.prompt_id: prompt for prompt in request.prompts}
        tokenized: list[
            tuple[Any, tuple[int, ...], tuple[int, ...], Any]
        ] = []
        tokenization_mode_counts = {
            "joint_exact_boundary": 0,
            "causal_split": 0,
        }
        for rollout in request.rollouts.samples:
            prompt = prompt_by_id[rollout.prompt_id]
            prompt_ids, completion = (
                self.model_adapter.tokenizer.tokenize_continuation(
                    prompt_text=prompt.teacher_prompt,
                    completion_text=rollout.completion.text,
                )
            )
            model_prompt_ids = self.model_adapter.tokenizer.with_model_prefix(
                prompt_ids
            )
            tokenization_mode = (
                self.model_adapter.tokenizer.last_continuation_tokenization_mode
            )
            if tokenization_mode not in tokenization_mode_counts:
                raise ModelAdapterError(
                    "teacher continuation tokenization mode was not recorded"
                )
            tokenization_mode_counts[tokenization_mode] += 1
            if len(model_prompt_ids) > self._config.rollout.max_prompt_tokens:
                raise ModelAdapterError(
                    f"teacher prompt {prompt.prompt_id!r} exceeds max_prompt_tokens"
                )
            tokenized.append(
                (rollout, prompt_ids, model_prompt_ids, completion)
            )

        widths = [
            len(model_prompt_ids) + len(completion.token_ids)
            for _, _, model_prompt_ids, completion in tokenized
        ]
        tokenize_s = time.monotonic() - tokenize_started
        required_width = max(widths)
        completion_required = max(
            len(completion.token_ids)
            for _, _, _, completion in tokenized
        )
        cached_mode = (
            self._config.training.teacher_scoring_mode
            == "cached_teacher_forcing"
        )
        bucket_required_width = required_width
        if cached_mode:
            bucket_required_width = max(
                bucket_required_width,
                self._config.rollout.max_prompt_tokens + completion_required,
            )
        try:
            max_width = select_length_bucket(
                bucket_required_width,
                self._config.training.teacher_sequence_buckets,
            )
        except PerformanceContractError as exc:
            raise ModelAdapterError(str(exc)) from exc
        pad_id = self.model_adapter.tokenizer.pad_token_id
        input_ids = np.full((len(tokenized), max_width), pad_id, dtype=np.int32)
        segment_ids = np.zeros_like(input_ids)
        if cached_mode:
            completion_width = (
                max_width - self._config.rollout.max_prompt_tokens
            )
            if completion_required > completion_width:
                raise ModelAdapterError(
                    "teacher completion exceeds the cached scoring budget"
                )
        else:
            try:
                completion_width = select_length_bucket(
                    completion_required,
                    self._config.training.teacher_sequence_buckets,
                )
            except PerformanceContractError as exc:
                raise ModelAdapterError(str(exc)) from exc
        completion_positions = np.zeros(
            (len(tokenized), completion_width), dtype=np.int32
        )
        completion_token_ids = np.full(
            (len(tokenized), completion_width), pad_id, dtype=np.int32
        )
        completion_mask = np.zeros(
            (len(tokenized), completion_width), dtype=np.bool_
        )
        cached_prompt_ids = np.full(
            (
                len(tokenized),
                self._config.rollout.max_prompt_tokens,
            ),
            pad_id,
            dtype=np.int32,
        )
        cached_prompt_mask = np.zeros_like(cached_prompt_ids, dtype=np.bool_)
        for row_index, (_, _, model_prompt_ids, completion) in enumerate(tokenized):
            row_ids = model_prompt_ids + completion.token_ids
            input_ids[row_index, : len(row_ids)] = row_ids
            segment_ids[row_index, : len(row_ids)] = 1

            start = len(model_prompt_ids) - 1
            width = len(completion.token_ids)
            completion_positions[row_index, :width] = np.arange(
                start, start + width, dtype=np.int32
            )
            completion_token_ids[row_index, :width] = completion.token_ids
            completion_mask[row_index, :width] = True
            prompt_start = (
                self._config.rollout.max_prompt_tokens - len(model_prompt_ids)
            )
            cached_prompt_ids[row_index, prompt_start:] = model_prompt_ids
            cached_prompt_mask[row_index, prompt_start:] = True

        use_cached_statistics = (
            cached_mode
            and self._overlap_token_ids is not None
            and self.model_adapter.dependencies.forward_cached_sufficient_statistics
            is not None
        )
        if cached_mode and not use_cached_statistics:
            raise ModelAdapterError(
                "cached teacher scoring requires overlap ids and a cached runtime"
            )
        use_dense_statistics = (
            not cached_mode
            and self._overlap_token_ids is not None
            and self.model_adapter.dependencies.forward_sufficient_statistics
            is not None
        )
        # The optimizer path reduces the frozen teacher inside one JIT to the
        # exact SimCT sufficient statistics.  CPU dependency fakes and the
        # framework-light interface canary retain the full-logit fallback.
        forward_started = time.monotonic()
        if use_cached_statistics:
            shared_log_probs, selected_log_probs = (
                self.model_adapter.forward_cached_sufficient_statistics(
                    cached_prompt_ids,
                    cached_prompt_mask,
                    completion_token_ids,
                    completion_mask,
                    self._overlap_token_ids,
                )
            )
            forward_values: Any = (shared_log_probs, selected_log_probs)
        elif use_dense_statistics:
            shared_log_probs, selected_log_probs = (
                self.model_adapter.forward_sufficient_statistics(
                    input_ids,
                    segment_ids,
                    completion_positions,
                    completion_token_ids,
                    self._overlap_token_ids,
                )
            )
            forward_values: Any = (shared_log_probs, selected_log_probs)
        else:
            full_logits = self.model_adapter.forward(input_ids, segment_ids)
            forward_values = full_logits
        if self._config.training.synchronize_phase_timings:
            try:
                import jax

                jax.block_until_ready(forward_values)
            except (ImportError, AttributeError):
                self.model_adapter.to_host(forward_values)
        forward_s = time.monotonic() - forward_started
        samples: list[TeacherScoreSample] = []
        for row_index, (
            rollout,
            prompt_ids,
            model_prompt_ids,
            completion,
        ) in enumerate(tokenized):
            start = len(model_prompt_ids) - 1
            width = len(completion.token_ids)
            if use_cached_statistics or use_dense_statistics:
                sample_shared = shared_log_probs[row_index, :width, :]
                sample_selected = selected_log_probs[row_index, :width]
                score_payload = {
                    "sufficient_statistics": TeacherSufficientStatisticsPayload(
                        shared_log_probs=sample_shared,
                        selected_log_probs=sample_selected,
                        shape=tuple(int(value) for value in sample_shared.shape),
                        dtype=str(sample_shared.dtype),
                    )
                }
            else:
                position_logits = full_logits[
                    row_index, start : start + width, :
                ]
                score_payload = {
                    "position_logits": LogitsPayload(
                        values=position_logits,
                        shape=tuple(int(value) for value in position_logits.shape),
                        dtype=str(position_logits.dtype),
                    )
                }
            samples.append(
                TeacherScoreSample(
                    sample_id=rollout.sample_id,
                    prompt_id=rollout.prompt_id,
                    teacher_prompt_token_ids=prompt_ids,
                    completion=completion,
                    **score_payload,
                )
            )
        model = self._config.teacher
        result = TeacherScoreBatch(
            contract_version=INTERFACE_CONTRACT_VERSION,
            run_id=request.rollouts.run_id,
            step=request.rollouts.step,
            model_id=model.model_id,
            model_revision=model.model_revision,
            tokenizer_id=model.tokenizer_id,
            tokenizer_revision=model.tokenizer_revision,
            samples=tuple(samples),
        )
        self.last_phase_timings = {
            "teacher_tokenize_s": tokenize_s,
            "teacher_forward_s": forward_s,
            "teacher_sequence_required": required_width,
            "teacher_sequence_bucket": max_width,
            "teacher_completion_bucket": completion_width,
            "teacher_joint_boundary_records": tokenization_mode_counts[
                "joint_exact_boundary"
            ],
            "teacher_causal_split_records": tokenization_mode_counts[
                "causal_split"
            ],
        }
        return result


def build_backends(
    config: RunConfig,
    *,
    dependencies: ModelRuntimeDependencies | None = None,
) -> BackendBundle:
    """Build one trainable-student and one frozen-teacher backend.

    The optional dependency argument is the CPU-fake seam used by contract
    tests.  The public one-argument call always constructs production
    dependencies and never falls back to those fakes.
    """

    if config.tpu.pipeline_parallelism != 1:
        raise RealBackendUnavailable(
            "the bounded real backend currently supports PP1 only"
        )
    _validate_model_path(config.student)
    _validate_model_path(config.teacher)
    runtime = dependencies or _production_dependencies(config)
    try:
        runtime.validate_model_spec(config.student)
        runtime.validate_model_spec(config.teacher)
        student_tokenizer = TokenizerByteAdapter(
            runtime.load_tokenizer(config.student),
            config.student,
        )
        teacher_tokenizer = TokenizerByteAdapter(
            runtime.load_tokenizer(config.teacher),
            config.teacher,
        )
    except RealBackendUnavailable:
        raise
    except Exception as exc:
        raise RealBackendUnavailable(
            f"real backend adapter initialization failed: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    student_adapter = CausalModelForwardAdapter(
        config.student,
        student_tokenizer,
        runtime,
        trainable=True,
    )
    teacher_adapter = CausalModelForwardAdapter(
        config.teacher,
        teacher_tokenizer,
        runtime,
        trainable=False,
    )
    return BackendBundle(
        student=TunixStudentRolloutBackend(config, student_adapter),
        teacher=MaxTextFrozenTeacherScoreBackend(config, teacher_adapter),
    )


def load_native_student(
    config: RunConfig,
    *,
    dependencies: ModelRuntimeDependencies | None = None,
) -> tuple[_LoadedTunixModel, TokenizerByteAdapter]:
    """Load only the student model for inference.

    Evaluation must not materialize the frozen 7B teacher merely to sample a
    2B student checkpoint.  This path shares the production dependency bundle
    and validation used by training, but intentionally constructs one adapter.
    The dependency seam is retained for CPU contract tests; the public
    one-argument call remains fail-closed and production-only.
    """

    if config.tpu.pipeline_parallelism != 1:
        raise RealBackendUnavailable(
            "the bounded real backend currently supports PP1 only"
        )
    _validate_model_path(config.student)
    runtime = dependencies or _production_dependencies(config)
    try:
        runtime.validate_model_spec(config.student)
        tokenizer = TokenizerByteAdapter(
            runtime.load_tokenizer(config.student),
            config.student,
        )
        adapter = CausalModelForwardAdapter(
            config.student,
            tokenizer,
            runtime,
            trainable=True,
        )
        loaded = adapter.require_loaded_model()
    except RealBackendUnavailable:
        raise
    except Exception as exc:
        raise RealBackendUnavailable(
            "student-only adapter initialization failed: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    if not isinstance(loaded, _LoadedTunixModel):
        raise RealBackendUnavailable(
            "student-only loader did not return a native Tunix model"
        )
    return loaded, tokenizer
