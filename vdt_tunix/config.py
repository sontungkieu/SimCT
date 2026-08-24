"""Strict, dependency-free configuration contract for the TPU scaffold."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any


CONFIG_CONTRACT_VERSION = 1


class ConfigError(ValueError):
    """Raised when a configuration does not satisfy the scaffold contract."""


def _object(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{context} must be a JSON object")
    return value


def _keys(
    value: Mapping[str, Any],
    *,
    context: str,
    required: set[str],
    optional: set[str] | None = None,
) -> None:
    optional = optional or set()
    missing = sorted(required - value.keys())
    extra = sorted(value.keys() - required - optional)
    if missing:
        raise ConfigError(f"{context} is missing required keys: {missing}")
    if extra:
        raise ConfigError(f"{context} has unsupported keys: {extra}")


def _string(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{context} must be a non-empty string")
    return value


def _integer(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{context} must be an integer")
    return value


def _number(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{context} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ConfigError(f"{context} must be finite")
    return result


@dataclasses.dataclass(frozen=True, slots=True)
class ModelConfig:
    model_id: str
    model_revision: str
    tokenizer_id: str
    tokenizer_revision: str
    maxtext_checkpoint_uri: str
    model_source: str = "maxtext"
    model_path: str | None = None
    tokenizer_type: str = "huggingface"
    tokenizer_path: str | None = None

    def __post_init__(self) -> None:
        for name in (
            "model_id",
            "model_revision",
            "tokenizer_id",
            "tokenizer_revision",
            "maxtext_checkpoint_uri",
        ):
            _string(getattr(self, name), f"model.{name}")
        if self.model_source not in {"maxtext", "huggingface", "kaggle"}:
            raise ConfigError(
                "model.model_source must be 'maxtext', 'huggingface', or 'kaggle'"
            )
        if self.model_path is not None:
            _string(self.model_path, "model.model_path")
        if self.tokenizer_type not in {"huggingface", "sentencepiece"}:
            raise ConfigError(
                "model.tokenizer_type must be 'huggingface' or 'sentencepiece'"
            )
        if self.tokenizer_path is not None:
            _string(self.tokenizer_path, "model.tokenizer_path")
        mutable_names = {"main", "master", "head", "latest"}
        if self.model_revision.lower() in mutable_names:
            raise ConfigError("model_revision must identify an immutable revision")
        if self.tokenizer_revision.lower() in mutable_names:
            raise ConfigError("tokenizer_revision must identify an immutable revision")

    @classmethod
    def from_mapping(cls, value: Any, context: str) -> ModelConfig:
        raw = _object(value, context)
        names = {
            "model_id",
            "model_revision",
            "tokenizer_id",
            "tokenizer_revision",
            "maxtext_checkpoint_uri",
        }
        optional = {
            "model_source",
            "model_path",
            "tokenizer_type",
            "tokenizer_path",
        }
        _keys(raw, context=context, required=names, optional=optional)
        required_values = {
            name: _string(raw[name], f"{context}.{name}") for name in names
        }
        model_path = raw.get("model_path")
        tokenizer_path = raw.get("tokenizer_path")
        return cls(
            **required_values,
            model_source=_string(
                raw.get("model_source", "maxtext"), f"{context}.model_source"
            ),
            model_path=(
                None
                if model_path is None
                else _string(model_path, f"{context}.model_path")
            ),
            tokenizer_type=_string(
                raw.get("tokenizer_type", "huggingface"),
                f"{context}.tokenizer_type",
            ),
            tokenizer_path=(
                None
                if tokenizer_path is None
                else _string(tokenizer_path, f"{context}.tokenizer_path")
            ),
        )

    @property
    def resolved_model_path(self) -> str:
        return self.model_path or self.maxtext_checkpoint_uri

    @property
    def resolved_tokenizer_path(self) -> str:
        return self.tokenizer_path or self.tokenizer_id


@dataclasses.dataclass(frozen=True, slots=True)
class SimCTConfig:
    algorithm: str
    divergence: str
    alignment_unit: str
    virtual_support: str
    temperature: float
    span_gh_mask_threshold: float
    reproduction_mode: str = "public_code_cf0f33a_default"

    def __post_init__(self) -> None:
        if self.algorithm != "simct":
            raise ConfigError("simct.algorithm must be 'simct'")
        if self.divergence != "reverse_kl":
            raise ConfigError("simct.divergence must be 'reverse_kl'")
        if self.alignment_unit != "utf8_bytes":
            raise ConfigError("simct.alignment_unit must be 'utf8_bytes'")
        if self.virtual_support != "shared_tokens_plus_realized_spans":
            raise ConfigError(
                "simct.virtual_support must be "
                "'shared_tokens_plus_realized_spans'"
            )
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ConfigError("simct.temperature must be positive and finite")
        modes = {
            "paper_math",
            "public_code_pre_safeguard",
            "public_code_cf0f33a_default",
            "public_scripts_cf0f33a",
        }
        if self.reproduction_mode not in modes:
            raise ConfigError(
                f"simct.reproduction_mode must be one of {sorted(modes)}"
            )
        if not math.isfinite(self.span_gh_mask_threshold) or (
            self.span_gh_mask_threshold != 0.0
            and self.span_gh_mask_threshold < 1.0
        ):
            raise ConfigError(
                "simct.span_gh_mask_threshold must be 0 (disabled) or at least 1"
            )
        if self.reproduction_mode in {
            "paper_math",
            "public_code_pre_safeguard",
        } and self.span_gh_mask_threshold != 0.0:
            raise ConfigError(
                f"{self.reproduction_mode} requires span_gh_mask_threshold=0"
            )
        if self.reproduction_mode in {
            "public_code_cf0f33a_default",
            "public_scripts_cf0f33a",
        } and self.span_gh_mask_threshold < 1.0:
            raise ConfigError(
                f"{self.reproduction_mode} requires an active G(h) threshold"
            )

    @classmethod
    def from_mapping(cls, value: Any) -> SimCTConfig:
        context = "simct"
        raw = _object(value, context)
        required = {
            "algorithm",
            "divergence",
            "alignment_unit",
            "virtual_support",
            "temperature",
            "span_gh_mask_threshold",
        }
        _keys(
            raw,
            context=context,
            required=required,
            optional={"reproduction_mode"},
        )
        return cls(
            algorithm=_string(raw["algorithm"], "simct.algorithm"),
            divergence=_string(raw["divergence"], "simct.divergence"),
            alignment_unit=_string(raw["alignment_unit"], "simct.alignment_unit"),
            virtual_support=_string(raw["virtual_support"], "simct.virtual_support"),
            temperature=_number(raw["temperature"], "simct.temperature"),
            span_gh_mask_threshold=_number(
                raw["span_gh_mask_threshold"],
                "simct.span_gh_mask_threshold",
            ),
            reproduction_mode=_string(
                raw.get(
                    "reproduction_mode", "public_code_cf0f33a_default"
                ),
                "simct.reproduction_mode",
            ),
        )


@dataclasses.dataclass(frozen=True, slots=True)
class RolloutConfig:
    prompt_batch_size: int
    samples_per_prompt: int
    max_prompt_tokens: int
    max_completion_tokens: int
    temperature: float
    top_p: float

    def __post_init__(self) -> None:
        for name in (
            "prompt_batch_size",
            "samples_per_prompt",
            "max_prompt_tokens",
            "max_completion_tokens",
        ):
            if getattr(self, name) < 1:
                raise ConfigError(f"rollout.{name} must be positive")
        if not math.isfinite(self.temperature) or self.temperature < 0:
            raise ConfigError("rollout.temperature must be finite and non-negative")
        if not math.isfinite(self.top_p) or not 0 < self.top_p <= 1:
            raise ConfigError("rollout.top_p must be in (0, 1]")

    @classmethod
    def from_mapping(cls, value: Any) -> RolloutConfig:
        context = "rollout"
        raw = _object(value, context)
        required = {
            "prompt_batch_size",
            "samples_per_prompt",
            "max_prompt_tokens",
            "max_completion_tokens",
            "temperature",
            "top_p",
        }
        _keys(raw, context=context, required=required)
        return cls(
            prompt_batch_size=_integer(
                raw["prompt_batch_size"], "rollout.prompt_batch_size"
            ),
            samples_per_prompt=_integer(
                raw["samples_per_prompt"], "rollout.samples_per_prompt"
            ),
            max_prompt_tokens=_integer(
                raw["max_prompt_tokens"], "rollout.max_prompt_tokens"
            ),
            max_completion_tokens=_integer(
                raw["max_completion_tokens"], "rollout.max_completion_tokens"
            ),
            temperature=_number(raw["temperature"], "rollout.temperature"),
            top_p=_number(raw["top_p"], "rollout.top_p"),
        )


@dataclasses.dataclass(frozen=True, slots=True)
class TrainingConfig:
    max_steps: int
    micro_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float

    def __post_init__(self) -> None:
        for name in ("max_steps", "micro_batch_size", "gradient_accumulation_steps"):
            if getattr(self, name) < 1:
                raise ConfigError(f"training.{name} must be positive")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ConfigError("training.learning_rate must be positive and finite")

    @classmethod
    def from_mapping(cls, value: Any) -> TrainingConfig:
        context = "training"
        raw = _object(value, context)
        required = {
            "max_steps",
            "micro_batch_size",
            "gradient_accumulation_steps",
            "learning_rate",
        }
        _keys(raw, context=context, required=required)
        return cls(
            max_steps=_integer(raw["max_steps"], "training.max_steps"),
            micro_batch_size=_integer(
                raw["micro_batch_size"], "training.micro_batch_size"
            ),
            gradient_accumulation_steps=_integer(
                raw["gradient_accumulation_steps"],
                "training.gradient_accumulation_steps",
            ),
            learning_rate=_number(raw["learning_rate"], "training.learning_rate"),
        )


@dataclasses.dataclass(frozen=True, slots=True)
class TPUConfig:
    accelerator_type: str
    expected_device_count: int
    tensor_parallelism: int
    pipeline_parallelism: int
    fsdp_parallelism: int = 1

    def __post_init__(self) -> None:
        if self.accelerator_type != "v5e-8":
            raise ConfigError("tpu.accelerator_type must be 'v5e-8'")
        if self.expected_device_count != 8:
            raise ConfigError("tpu.expected_device_count must be 8")
        if (
            self.tensor_parallelism < 1
            or self.pipeline_parallelism < 1
            or self.fsdp_parallelism < 1
        ):
            raise ConfigError("TPU parallelism values must be positive")
        if (
            self.tensor_parallelism
            * self.pipeline_parallelism
            * self.fsdp_parallelism
            != 8
        ):
            raise ConfigError(
                "fsdp_parallelism * tensor_parallelism * "
                "pipeline_parallelism must equal 8"
            )

    @classmethod
    def from_mapping(cls, value: Any) -> TPUConfig:
        context = "tpu"
        raw = _object(value, context)
        required = {
            "accelerator_type",
            "expected_device_count",
            "tensor_parallelism",
            "pipeline_parallelism",
        }
        _keys(
            raw,
            context=context,
            required=required,
            optional={"fsdp_parallelism"},
        )
        return cls(
            accelerator_type=_string(
                raw["accelerator_type"], "tpu.accelerator_type"
            ),
            expected_device_count=_integer(
                raw["expected_device_count"], "tpu.expected_device_count"
            ),
            tensor_parallelism=_integer(
                raw["tensor_parallelism"], "tpu.tensor_parallelism"
            ),
            pipeline_parallelism=_integer(
                raw["pipeline_parallelism"], "tpu.pipeline_parallelism"
            ),
            fsdp_parallelism=_integer(
                raw.get("fsdp_parallelism", 1), "tpu.fsdp_parallelism"
            ),
        )


@dataclasses.dataclass(frozen=True, slots=True)
class CheckpointConfig:
    root: str
    save_every_steps: int
    resume_from: str | None = None

    def __post_init__(self) -> None:
        _string(self.root, "checkpoint.root")
        if self.save_every_steps < 1:
            raise ConfigError("checkpoint.save_every_steps must be positive")
        if self.resume_from is not None:
            _string(self.resume_from, "checkpoint.resume_from")

    @classmethod
    def from_mapping(cls, value: Any) -> CheckpointConfig:
        context = "checkpoint"
        raw = _object(value, context)
        _keys(
            raw,
            context=context,
            required={"root", "save_every_steps"},
            optional={"resume_from"},
        )
        resume_from = raw.get("resume_from")
        if resume_from is not None:
            resume_from = _string(resume_from, "checkpoint.resume_from")
        return cls(
            root=_string(raw["root"], "checkpoint.root"),
            save_every_steps=_integer(
                raw["save_every_steps"], "checkpoint.save_every_steps"
            ),
            resume_from=resume_from,
        )


@dataclasses.dataclass(frozen=True, slots=True)
class CanaryConfig:
    prompt_id: str
    student_prompt: str
    teacher_prompt: str

    def __post_init__(self) -> None:
        _string(self.prompt_id, "canary.prompt_id")
        _string(self.student_prompt, "canary.student_prompt")
        _string(self.teacher_prompt, "canary.teacher_prompt")

    @classmethod
    def from_mapping(cls, value: Any) -> CanaryConfig:
        context = "canary"
        raw = _object(value, context)
        required = {"prompt_id", "student_prompt", "teacher_prompt"}
        _keys(raw, context=context, required=required)
        return cls(
            prompt_id=_string(raw["prompt_id"], "canary.prompt_id"),
            student_prompt=_string(
                raw["student_prompt"], "canary.student_prompt"
            ),
            teacher_prompt=_string(
                raw["teacher_prompt"], "canary.teacher_prompt"
            ),
        )


@dataclasses.dataclass(frozen=True, slots=True)
class RunConfig:
    contract_version: int
    run_id: str
    student: ModelConfig
    teacher: ModelConfig
    simct: SimCTConfig
    rollout: RolloutConfig
    training: TrainingConfig
    tpu: TPUConfig
    checkpoint: CheckpointConfig
    canary: CanaryConfig

    def __post_init__(self) -> None:
        if self.contract_version != CONFIG_CONTRACT_VERSION:
            raise ConfigError(
                f"contract_version must be {CONFIG_CONTRACT_VERSION}"
            )
        _string(self.run_id, "run_id")
        if self.student.tokenizer_id == self.teacher.tokenizer_id:
            raise ConfigError(
                "single-teacher SimCT requires distinct student and teacher "
                "tokenizer_id values"
            )
        if self.training.micro_batch_size > (
            self.rollout.prompt_batch_size * self.rollout.samples_per_prompt
        ):
            raise ConfigError(
                "training.micro_batch_size cannot exceed the rollout sample batch"
            )

    @classmethod
    def from_mapping(cls, value: Any) -> RunConfig:
        raw = _object(value, "config")
        required = {
            "contract_version",
            "run_id",
            "student",
            "teacher",
            "simct",
            "rollout",
            "training",
            "tpu",
            "checkpoint",
            "canary",
        }
        _keys(raw, context="config", required=required)
        return cls(
            contract_version=_integer(
                raw["contract_version"], "contract_version"
            ),
            run_id=_string(raw["run_id"], "run_id"),
            student=ModelConfig.from_mapping(raw["student"], "student"),
            teacher=ModelConfig.from_mapping(raw["teacher"], "teacher"),
            simct=SimCTConfig.from_mapping(raw["simct"]),
            rollout=RolloutConfig.from_mapping(raw["rollout"]),
            training=TrainingConfig.from_mapping(raw["training"]),
            tpu=TPUConfig.from_mapping(raw["tpu"]),
            checkpoint=CheckpointConfig.from_mapping(raw["checkpoint"]),
            canary=CanaryConfig.from_mapping(raw["canary"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def identity_dict(self) -> dict[str, Any]:
        """Return training identity without environment-local storage paths."""

        payload = self.to_dict()
        payload["checkpoint"] = {
            "save_every_steps": self.checkpoint.save_every_steps,
        }
        return payload

    def canonical_json(self) -> str:
        return json.dumps(
            self.identity_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )

    def digest(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


def load_config(path: str | Path) -> RunConfig:
    config_path = Path(path)
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigError(f"invalid JSON in {config_path}: {exc}") from exc
    return RunConfig.from_mapping(payload)
