"""CPU-only learning lab for dynamic cross-tokenizer spans.

The public API is deliberately small. Complete the TODO functions in the
submodules; the data classes in :mod:`vdt_span.types` are provided scaffolding.
"""

from vdt_span.alignment import align_exact_byte_boundaries
from vdt_span.candidates import enumerate_span_candidates
from vdt_span.coarsening import coarsen_distribution
from vdt_span.continuation import continuation_logprob
from vdt_span.policy import adaptive_max_span_width
from vdt_span.types import (
    AlignedSpan,
    PolicyContext,
    ScoredSpan,
    SpanCandidate,
    SpanPolicyConfig,
    TrainingState,
    ViterbiPath,
)
from vdt_span.viterbi import semi_markov_viterbi

TODO_EXERCISES = (
    "exact-byte-boundary-alignment",
    "candidate-span-enumeration",
    "semi-markov-viterbi",
    "continuation-scoring",
    "mass-preserving-coarsening",
    "adaptive-span-policy",
)

__all__ = [
    "AlignedSpan",
    "PolicyContext",
    "ScoredSpan",
    "SpanCandidate",
    "SpanPolicyConfig",
    "TODO_EXERCISES",
    "TrainingState",
    "ViterbiPath",
    "adaptive_max_span_width",
    "align_exact_byte_boundaries",
    "coarsen_distribution",
    "continuation_logprob",
    "enumerate_span_candidates",
    "semi_markov_viterbi",
]
