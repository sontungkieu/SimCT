"""Static and dynamic partitions over atomic aligned units."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Mapping, Sequence

Segment = tuple[int, int]
SegmentScore = Callable[[int, int], float]


def validate_partition(partition: Sequence[Segment], num_units: int) -> None:
    """Validate a half-open, contiguous partition of ``range(num_units)``."""

    if num_units < 0:
        raise ValueError("num_units must be non-negative")
    if num_units == 0:
        if partition:
            raise ValueError("an empty sequence must have an empty partition")
        return
    cursor = 0
    for start, end in partition:
        if start != cursor or end <= start or end > num_units:
            raise ValueError("segments must be contiguous, non-empty, and in range")
        cursor = end
    if cursor != num_units:
        raise ValueError("partition does not cover all atomic units")


def enumerate_candidate_spans(num_units: int, max_span: int) -> tuple[Segment, ...]:
    """Enumerate every consecutive candidate span up to ``max_span`` units."""

    if num_units < 0:
        raise ValueError("num_units must be non-negative")
    if max_span < 1:
        raise ValueError("max_span must be at least one")
    return tuple(
        (start, end)
        for start in range(num_units)
        for end in range(start + 1, min(num_units, start + max_span) + 1)
    )


def fixed_width_partition(num_units: int, width: int) -> tuple[Segment, ...]:
    """Partition left-to-right into fixed-width spans (last span may be short)."""

    if num_units < 0:
        raise ValueError("num_units must be non-negative")
    if width < 1:
        raise ValueError("width must be at least one")
    return tuple((start, min(num_units, start + width)) for start in range(0, num_units, width))


def _logsumexp(values: Sequence[float]) -> float:
    if not values:
        return -math.inf
    maximum = max(values)
    if maximum == -math.inf:
        return maximum
    return maximum + math.log(sum(math.exp(value - maximum) for value in values))


def semi_markov_viterbi(
    num_units: int,
    max_span: int,
    score: SegmentScore,
    *,
    tie_break: str = "finer",
) -> tuple[Segment, ...]:
    """Return the maximum-score semi-Markov partition.

    ``score(start, end)`` must be an additive segment energy. Equal-score paths
    prefer more segments when ``tie_break='finer'`` and fewer when ``'coarser'``.
    """

    if num_units < 0:
        raise ValueError("num_units must be non-negative")
    if max_span < 1:
        raise ValueError("max_span must be at least one")
    if tie_break not in {"finer", "coarser"}:
        raise ValueError("tie_break must be 'finer' or 'coarser'")
    if num_units == 0:
        return ()

    best_score = [-math.inf] * (num_units + 1)
    best_count = [0] * (num_units + 1)
    back = [-1] * (num_units + 1)
    best_score[0] = 0.0

    for end in range(1, num_units + 1):
        for start in range(max(0, end - max_span), end):
            candidate_score = best_score[start] + float(score(start, end))
            candidate_count = best_count[start] + 1
            is_better = candidate_score > best_score[end]
            if math.isclose(candidate_score, best_score[end], rel_tol=0.0, abs_tol=1e-12):
                if tie_break == "finer":
                    is_better = candidate_count > best_count[end]
                else:
                    is_better = candidate_count < best_count[end] or back[end] == -1
            if is_better:
                best_score[end] = candidate_score
                best_count[end] = candidate_count
                back[end] = start

    if back[num_units] < 0:
        raise ValueError("no valid segmentation path")
    reversed_path: list[Segment] = []
    end = num_units
    while end:
        start = back[end]
        reversed_path.append((start, end))
        end = start
    partition = tuple(reversed(reversed_path))
    validate_partition(partition, num_units)
    return partition


def semi_markov_log_partition(num_units: int, max_span: int, score: SegmentScore) -> float:
    """Compute ``log sum_partition exp(sum_segment score)``."""

    if num_units < 0:
        raise ValueError("num_units must be non-negative")
    if max_span < 1:
        raise ValueError("max_span must be at least one")
    forward = [-math.inf] * (num_units + 1)
    forward[0] = 0.0
    for end in range(1, num_units + 1):
        forward[end] = _logsumexp(
            [
                forward[start] + float(score(start, end))
                for start in range(max(0, end - max_span), end)
            ]
        )
    return forward[num_units]


def semi_markov_span_marginals(
    num_units: int,
    max_span: int,
    score: SegmentScore,
) -> Mapping[Segment, float]:
    """Return the exact marginal probability of each candidate span."""

    if num_units < 0:
        raise ValueError("num_units must be non-negative")
    if max_span < 1:
        raise ValueError("max_span must be at least one")
    if num_units == 0:
        return {}

    forward = [-math.inf] * (num_units + 1)
    forward[0] = 0.0
    for end in range(1, num_units + 1):
        forward[end] = _logsumexp(
            [
                forward[start] + float(score(start, end))
                for start in range(max(0, end - max_span), end)
            ]
        )

    backward = [-math.inf] * (num_units + 1)
    backward[num_units] = 0.0
    for start in range(num_units - 1, -1, -1):
        backward[start] = _logsumexp(
            [
                float(score(start, end)) + backward[end]
                for end in range(start + 1, min(num_units, start + max_span) + 1)
            ]
        )

    log_z = forward[num_units]
    return {
        (start, end): math.exp(
            forward[start] + float(score(start, end)) + backward[end] - log_z
        )
        for start, end in enumerate_candidate_spans(num_units, max_span)
    }


@dataclass(frozen=True, slots=True)
class AdaptiveBoundaryPolicy:
    """Transparent first baseline for context- and training-dependent spans.

    Higher cut scores create a boundary. ``progress_weight`` is intentionally
    signed: positive values make later training finer; negative values make it
    coarser. This is a hypothesis knob, not a baked-in scientific assumption.
    """

    disagreement_weight: float = 1.0
    instability_weight: float = 1.0
    association_weight: float = 1.0
    context_weight: float = 0.0
    progress_weight: float = 0.0
    threshold: float = 0.0
    max_span: int = 4

    def cut_scores(
        self,
        *,
        disagreement: Sequence[float],
        stability: Sequence[float],
        association: Sequence[float],
        context: Sequence[float] | None = None,
        progress: float = 0.0,
    ) -> tuple[float, ...]:
        if not 0.0 <= progress <= 1.0:
            raise ValueError("progress must lie in [0, 1]")
        boundary_count = len(disagreement)
        if len(stability) != boundary_count or len(association) != boundary_count:
            raise ValueError("all boundary feature sequences must have equal length")
        if context is None:
            context = (0.0,) * boundary_count
        if len(context) != boundary_count:
            raise ValueError("context must have one value per boundary")
        return tuple(
            self.disagreement_weight * float(disagreement[index])
            + self.instability_weight * (1.0 - float(stability[index]))
            - self.association_weight * float(association[index])
            + self.context_weight * float(context[index])
            + self.progress_weight * progress
            for index in range(boundary_count)
        )

    def partition_from_scores(self, cut_scores: Sequence[float]) -> tuple[Segment, ...]:
        if self.max_span < 1:
            raise ValueError("max_span must be at least one")
        num_units = len(cut_scores) + 1
        start = 0
        spans: list[Segment] = []
        for boundary_index, cut_score in enumerate(cut_scores):
            end = boundary_index + 1
            forced_by_length = end - start >= self.max_span
            if forced_by_length or float(cut_score) >= self.threshold:
                spans.append((start, end))
                start = end
        spans.append((start, num_units))
        partition = tuple(spans)
        validate_partition(partition, num_units)
        return partition

    def partition(
        self,
        *,
        disagreement: Sequence[float],
        stability: Sequence[float],
        association: Sequence[float],
        context: Sequence[float] | None = None,
        progress: float = 0.0,
    ) -> tuple[Segment, ...]:
        return self.partition_from_scores(
            self.cut_scores(
                disagreement=disagreement,
                stability=stability,
                association=association,
                context=context,
                progress=progress,
            )
        )
