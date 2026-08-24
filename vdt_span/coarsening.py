"""Exercise 5: project fine events into coarse buckets without losing mass."""

from __future__ import annotations

from collections.abc import Hashable, Mapping


def coarsen_distribution(
    fine_mass: Mapping[Hashable, float],
    bucket_of: Mapping[Hashable, Hashable],
    *,
    residual_bucket: Hashable = "__OTHER__",
    atol: float = 1e-9,
) -> dict[Hashable, float]:
    """Aggregate a normalized fine distribution into coarse event buckets.

    Unmapped fine events belong to ``residual_bucket``; they must not be
    dropped before renormalization. See Exercise 5 in ``STUDY.md`` for input
    validation and the mass-preservation invariant.
    """

    raise NotImplementedError("TODO E5: mass-preserving coarsening")
