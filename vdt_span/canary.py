"""Deterministic end-to-end canary for the reference span contracts."""

from __future__ import annotations

from typing import Any

from vdt_span.alignment import coarsen_aligned_units, minimal_joint_segments
from vdt_span.scoring import kl_information_gap
from vdt_span.segmentation import AdaptiveBoundaryPolicy, fixed_width_partition


def run_reference_canary() -> dict[str, Any]:
    """Exercise alignment, dynamic partitioning, and KL coarsening together."""

    teacher_pieces = ["I", " like", " dynamic", " spans"]
    student_pieces = ["I", " ", "like", " dynamic", " span", "s"]
    atomic_units = minimal_joint_segments(teacher_pieces, student_pieces)

    policy = AdaptiveBoundaryPolicy(
        disagreement_weight=1.0,
        instability_weight=0.5,
        association_weight=1.0,
        progress_weight=1.25,
        threshold=0.5,
        max_span=3,
    )
    boundary_count = len(atomic_units) - 1
    disagreement = [0.2, 0.8, 0.1][:boundary_count]
    stability = [0.9, 0.4, 0.95][:boundary_count]
    association = [0.7, 0.1, 0.8][:boundary_count]
    early = policy.partition(
        disagreement=disagreement,
        stability=stability,
        association=association,
        progress=0.0,
    )
    late = policy.partition(
        disagreement=disagreement,
        stability=stability,
        association=association,
        progress=1.0,
    )

    teacher_distribution = [0.45, 0.05, 0.10, 0.40]
    student_distribution = [0.10, 0.40, 0.20, 0.30]
    groups = [(0, 1), (2, 3)]
    information_gap = kl_information_gap(
        teacher_distribution,
        student_distribution,
        groups,
    )

    return {
        "schema_version": 1,
        "atomic_unit_count": len(atomic_units),
        "atomic_units": [
            {
                "teacher": [unit.teacher_start, unit.teacher_end],
                "student": [unit.student_start, unit.student_end],
                "bytes": [unit.byte_start, unit.byte_end],
            }
            for unit in atomic_units
        ],
        "fixed_width_two": [list(span) for span in fixed_width_partition(len(atomic_units), 2)],
        "dynamic_early": [list(span) for span in early],
        "dynamic_late": [list(span) for span in late],
        "early_coarsened_unit_count": len(coarsen_aligned_units(atomic_units, early)),
        "late_coarsened_unit_count": len(coarsen_aligned_units(atomic_units, late)),
        "mass_preserving_kl_gap": information_gap,
        "claim_boundary": (
            "reference primitives only; no model, tokenizer, gradient, TPU, or "
            "downstream-quality claim"
        ),
    }
