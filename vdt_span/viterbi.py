"""Exercise 3: deterministic semi-Markov Viterbi decoding."""

from __future__ import annotations

from collections.abc import Sequence

from vdt_span.types import ScoredSpan, ViterbiPath


def semi_markov_viterbi(
    num_atoms: int,
    candidates: Sequence[ScoredSpan],
) -> ViterbiPath:
    """Find the highest-scoring complete segmentation of ``num_atoms``.

    Candidate scores are additive. A valid path starts at 0, ends at
    ``num_atoms``, and contains neither gaps nor overlaps. See Exercise 3 in
    ``STUDY.md`` for validation rules and deterministic tie-breaking.
    """

    raise NotImplementedError("TODO E3: semi-Markov Viterbi")
