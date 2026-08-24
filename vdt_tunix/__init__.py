"""Bounded Tunix/MaxText contracts and paper-math SimCT kernels."""

from vdt_tunix.checkpoint import (
    ArtifactRef,
    CheckpointError,
    CheckpointState,
    DataCursor,
    load_latest_checkpoint,
    save_checkpoint,
    validate_resume,
)
from vdt_tunix.config import ConfigError, RunConfig, load_config
from vdt_tunix.contracts import (
    BackendBundle,
    ContractError,
    LogitsPayload,
    PromptRecord,
    StudentRolloutBackend,
    TeacherScoreBackend,
    TokenSequence,
)
from vdt_tunix.pipeline import CanaryReport, PipelineContractError, run_contract_canary
from vdt_tunix.runtime import TPUPreflightError, require_tpu_v5e8
from vdt_tunix.jax_kernels import (
    JaxKernelUnavailable,
    candidate_log_probs,
    paper_candidate_scores,
    paper_simct_reverse_kl,
    reverse_kl_from_scores,
    reverse_kl_loss_and_student_score_gradient,
)
from vdt_tunix.model_adapters import (
    CausalModelForwardAdapter,
    ModelAdapterError,
    ModelRuntimeDependencies,
    TokenizerByteAdapter,
)

__all__ = [
    "ArtifactRef",
    "BackendBundle",
    "CanaryReport",
    "CheckpointError",
    "CheckpointState",
    "ConfigError",
    "ContractError",
    "DataCursor",
    "LogitsPayload",
    "JaxKernelUnavailable",
    "candidate_log_probs",
    "CausalModelForwardAdapter",
    "ModelAdapterError",
    "ModelRuntimeDependencies",
    "PipelineContractError",
    "PromptRecord",
    "RunConfig",
    "StudentRolloutBackend",
    "TPUPreflightError",
    "TeacherScoreBackend",
    "TokenSequence",
    "TokenizerByteAdapter",
    "load_config",
    "load_latest_checkpoint",
    "require_tpu_v5e8",
    "paper_candidate_scores",
    "paper_simct_reverse_kl",
    "reverse_kl_from_scores",
    "reverse_kl_loss_and_student_score_gradient",
    "run_contract_canary",
    "save_checkpoint",
    "validate_resume",
]
