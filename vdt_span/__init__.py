"""CPU-testable primitives for dynamic cross-tokenizer span research.

This package deliberately has no Torch, JAX, tokenizer, or model dependency.
Framework adapters should convert their tensors and decoded pieces into these
small contracts, then compare their output against the reference behavior.
"""

from vdt_span.alignment import (
    AlignedUnit,
    AlignmentError,
    coarsen_aligned_units,
    minimal_joint_segments,
)
from vdt_span.scoring import (
    coarsen_distribution,
    continuation_score,
    kl_divergence,
    kl_information_gap,
    normalized_candidate_log_probs,
    reverse_kl_from_candidate_scores,
    reverse_kl_student_score_gradient,
)
from vdt_span.segmentation import (
    AdaptiveBoundaryPolicy,
    enumerate_candidate_spans,
    fixed_width_partition,
    semi_markov_log_partition,
    semi_markov_span_marginals,
    semi_markov_viterbi,
    validate_partition,
)

__all__ = [
    "AdaptiveBoundaryPolicy",
    "AlignedUnit",
    "AlignmentError",
    "coarsen_aligned_units",
    "coarsen_distribution",
    "continuation_score",
    "enumerate_candidate_spans",
    "fixed_width_partition",
    "kl_divergence",
    "kl_information_gap",
    "minimal_joint_segments",
    "normalized_candidate_log_probs",
    "reverse_kl_from_candidate_scores",
    "reverse_kl_student_score_gradient",
    "semi_markov_log_partition",
    "semi_markov_span_marginals",
    "semi_markov_viterbi",
    "validate_partition",
]
