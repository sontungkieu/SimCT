"""Bounded real Tunix/MaxText backend for the contract canary.

The production path uses only local MaxText checkpoints and locally available
tokenizer files.  Model construction is lazy: ``build_backends`` validates
dependencies, model specs, tokenizers, checkpoint paths, and the supported
TP8/PP1 layout; the first rollout/score call performs the exact weight restore.
Any unsupported or unavailable component fails closed.  There is no mock
fallback and no network/model-weight download path.
"""

from __future__ import annotations

from collections.abc import Sequence
import dataclasses
import hashlib
import importlib
import math
from pathlib import Path
from typing import Any

from vdt_tunix.config import ModelConfig, RunConfig
from vdt_tunix.contracts import (
    BackendBundle,
    INTERFACE_CONTRACT_VERSION,
    LogitsPayload,
    RolloutRequest,
    StudentRolloutBatch,
    StudentRolloutSample,
    TeacherScoreBatch,
    TeacherScoreRequest,
    TeacherScoreSample,
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
class _LoadedMaxTextModel:
    model: Any
    mesh: Any
    forward_fn: Any


def _validate_checkpoint_path(config: ModelConfig) -> Path:
    uri = config.maxtext_checkpoint_uri
    if "://" in uri:
        raise RealBackendUnavailable(
            "the bounded backend supports local MaxText checkpoint directories "
            f"only, not URI {uri!r}"
        )
    path = Path(uri).expanduser()
    if not path.is_absolute():
        raise RealBackendUnavailable(
            f"MaxText checkpoint path must be absolute: {uri!r}"
        )
    if not path.is_dir():
        raise RealBackendUnavailable(
            f"MaxText checkpoint directory is unavailable: {path}"
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
        mesh_lib = importlib.import_module("tunix.utils.mesh")
        transformers = importlib.import_module("transformers")
        try:
            importlib.import_module("maxtext.configs.pyconfig")
        except ImportError:
            importlib.import_module("maxtext.src.maxtext.configs.pyconfig")
    except ImportError as exc:
        raise RealBackendUnavailable(
            "required Tunix/MaxText real-backend dependency is unavailable: "
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
            tokenizer = transformers.AutoTokenizer.from_pretrained(
                model_config.tokenizer_id,
                revision=model_config.tokenizer_revision,
                local_files_only=True,
                trust_remote_code=False,
                token=None,
            )
        except Exception as exc:
            raise RealBackendUnavailable(
                "tokenizer is not available locally at the pinned revision for "
                f"{model_config.tokenizer_id!r}: {type(exc).__name__}: {exc}"
            ) from exc
        if getattr(tokenizer, "pad_token_id", None) is None:
            eos_token = getattr(tokenizer, "eos_token", None)
            if eos_token is None:
                raise RealBackendUnavailable(
                    f"tokenizer {model_config.tokenizer_id!r} has no pad or EOS token"
                )
            tokenizer.pad_token = eos_token
        return tokenizer

    def require_mesh() -> Any:
        if "mesh" in mesh_cache:
            return mesh_cache["mesh"]
        devices = list(jax.devices())
        if len(devices) != config.tpu.expected_device_count:
            raise RealBackendUnavailable(
                "model loading requires the validated eight-device runtime; "
                f"observed {len(devices)} devices"
            )
        axis_sizes = {
            "stage": config.tpu.pipeline_parallelism,
            "tensor": config.tpu.tensor_parallelism,
        }
        axis_names = maxtext_parallelism.MAXTEXT_MESH_AXIS_NAMES
        mesh_shape = tuple(axis_sizes.get(axis, 1) for axis in axis_names)
        mesh_cache["mesh"] = mesh_lib.create_mesh(
            mesh_shape,
            axis_names,
            devices=devices,
        )
        return mesh_cache["mesh"]

    def load_model(model_config: ModelConfig, trainable: bool) -> Any:
        del trainable  # Gradient behavior is enforced by the forward adapter.
        checkpoint_path = _validate_checkpoint_path(model_config)
        mesh = require_mesh()
        sample_batch = (
            config.rollout.prompt_batch_size * config.rollout.samples_per_prompt
        )
        max_target_length = (
            config.rollout.max_prompt_tokens
            + config.rollout.max_completion_tokens
        )
        try:
            with jax.set_mesh(mesh):
                model, _ = automodel.AutoModel.from_pretrained(
                    model_id=model_config.model_id,
                    model_path=str(checkpoint_path),
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
                    # Supplying the key prevents AutoModel from consulting an
                    # ambient HF_TOKEN.  Local checkpoints need no credential.
                    hf_access_token="",
                )
        except Exception as exc:
            raise RealBackendUnavailable(
                f"local MaxText checkpoint restore failed for "
                f"{model_config.model_id!r}: {type(exc).__name__}: {exc}"
            ) from exc

        @nnx.jit
        def forward_fn(module: Any, input_ids: Any, positions: Any, segments: Any):
            logits, _ = module(
                input_ids,
                positions,
                None,
                None,
                decoder_segment_ids=segments,
            )
            return logits

        return _LoadedMaxTextModel(model=model, mesh=mesh, forward_fn=forward_fn)

    def forward_model(
        loaded: _LoadedMaxTextModel,
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

    return ModelRuntimeDependencies(
        name="tunix-maxtext-local",
        production=True,
        validate_model_spec=validate_model_spec,
        load_tokenizer=load_tokenizer,
        load_model=load_model,
        forward_model=forward_model,
        stop_gradient=jax.lax.stop_gradient,
        to_host=jax.device_get,
    )


def _stable_seed(*parts: object) -> int:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


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

    def rollout(self, request: RolloutRequest) -> StudentRolloutBatch:
        import numpy as np

        rows: list[tuple[Any, int, str, tuple[int, ...]]] = []
        for prompt in request.prompts:
            prompt_ids = self.model_adapter.tokenizer.encode(prompt.student_prompt)
            if len(prompt_ids) > self._config.rollout.max_prompt_tokens:
                raise ModelAdapterError(
                    f"student prompt {prompt.prompt_id!r} exceeds max_prompt_tokens"
                )
            for sample_index in range(request.samples_per_prompt):
                sample_id = f"{prompt.prompt_id}/{sample_index}"
                rows.append((prompt, sample_index, sample_id, prompt_ids))

        max_prompt = max(len(row[3]) for row in rows)
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
                _stable_seed(request.run_id, request.step, row[2])
            )
            for row in rows
        ]
        for row_index, (_, _, _, prompt_ids) in enumerate(rows):
            width = len(prompt_ids)
            input_ids[row_index, :width] = prompt_ids
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
        for row_index, (prompt, _, sample_id, prompt_ids) in enumerate(rows):
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


class MaxTextFrozenTeacherScoreBackend:
    backend_name = "tunix-maxtext-frozen-teacher-forward"

    def __init__(
        self,
        config: RunConfig,
        model_adapter: CausalModelForwardAdapter,
    ):
        self._config = config
        self.model_adapter = model_adapter
        self.real_model_integration = model_adapter.dependencies.production

    def score(self, request: TeacherScoreRequest) -> TeacherScoreBatch:
        import numpy as np

        prompt_by_id = {prompt.prompt_id: prompt for prompt in request.prompts}
        tokenized: list[tuple[Any, tuple[int, ...], Any]] = []
        for rollout in request.rollouts.samples:
            prompt = prompt_by_id[rollout.prompt_id]
            prompt_ids, completion = (
                self.model_adapter.tokenizer.tokenize_continuation(
                    prompt_text=prompt.teacher_prompt,
                    completion_text=rollout.completion.text,
                )
            )
            if len(prompt_ids) > self._config.rollout.max_prompt_tokens:
                raise ModelAdapterError(
                    f"teacher prompt {prompt.prompt_id!r} exceeds max_prompt_tokens"
                )
            tokenized.append((rollout, prompt_ids, completion))

        widths = [len(prompt_ids) + len(completion.token_ids) for _, prompt_ids, completion in tokenized]
        max_width = max(widths)
        pad_id = self.model_adapter.tokenizer.pad_token_id
        input_ids = np.full((len(tokenized), max_width), pad_id, dtype=np.int32)
        segment_ids = np.zeros_like(input_ids)
        for row_index, (_, prompt_ids, completion) in enumerate(tokenized):
            row_ids = prompt_ids + completion.token_ids
            input_ids[row_index, : len(row_ids)] = row_ids
            segment_ids[row_index, : len(row_ids)] = 1

        full_logits = np.asarray(
            self.model_adapter.to_host(
                self.model_adapter.forward(input_ids, segment_ids)
            )
        )
        samples: list[TeacherScoreSample] = []
        for row_index, (rollout, prompt_ids, completion) in enumerate(tokenized):
            start = len(prompt_ids) - 1
            end = start + len(completion.token_ids)
            position_logits = full_logits[row_index, start:end, :]
            samples.append(
                TeacherScoreSample(
                    sample_id=rollout.sample_id,
                    prompt_id=rollout.prompt_id,
                    teacher_prompt_token_ids=prompt_ids,
                    completion=completion,
                    position_logits=LogitsPayload(
                        values=position_logits,
                        shape=tuple(int(value) for value in position_logits.shape),
                        dtype=str(position_logits.dtype),
                    ),
                )
            )
        model = self._config.teacher
        return TeacherScoreBatch(
            contract_version=INTERFACE_CONTRACT_VERSION,
            run_id=request.rollouts.run_id,
            step=request.rollouts.step,
            model_id=model.model_id,
            model_revision=model.model_revision,
            tokenizer_id=model.tokenizer_id,
            tokenizer_revision=model.tokenizer_revision,
            samples=tuple(samples),
        )


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

    if config.tpu.tensor_parallelism != 8 or config.tpu.pipeline_parallelism != 1:
        raise RealBackendUnavailable(
            "the bounded real backend currently supports TP8/PP1 only"
        )
    _validate_checkpoint_path(config.student)
    _validate_checkpoint_path(config.teacher)
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
