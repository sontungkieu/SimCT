"""Metrics for canonical-path word-boundary proxy masses, not text marginals.

All information metrics use natural logarithms. OTHER is an explicit shared cell.
Shape calibration raises each cell mass to beta=1/T, preserving zero support.
Uncertainty resamples prompts, with equal prompt contribution after weighted
within-prompt averaging; positions are not treated as independent observations.
"""

import math
import random
from collections import defaultdict

MASS_TOL = 1e-6
ENTROPY_TOL = 1e-9


def distribution(masses):
    """Validate candidate masses and append OTHER; tolerate roundoff in the sum."""
    values = [float(x) for x in masses]
    if not values or any(not math.isfinite(x) or x < 0 for x in values):
        raise ValueError("candidate masses must be nonempty, finite and nonnegative")
    total = math.fsum(values)
    if total > 1 + MASS_TOL:
        raise ValueError("candidate mass exceeds one")
    if total > 1:
        values = [x / total for x in values]
        total = 1.0
    return values + [1.0 - total]


def entropy(p):
    return -math.fsum(x * math.log(x) for x in p if x > 0)


def js(p, q):
    if len(p) != len(q):
        raise ValueError("distributions need the same shared cells")
    terms = []
    for a, b in zip(p, q):
        midpoint = (a + b) / 2
        if a > 0:
            terms.append(a * math.log(a / midpoint) / 2)
        if b > 0:
            terms.append(b * math.log(b / midpoint) / 2)
    return max(0.0, math.fsum(terms))


def kendall_tau_b(p, q):
    """Pairwise tau-b; None means one ranking is entirely tied."""
    if len(p) != len(q):
        raise ValueError("rankings need equal length")
    concordant = discordant = tie_p = tie_q = 0
    for i in range(len(p)):
        for j in range(i):
            dp, dq = p[i] - p[j], q[i] - q[j]
            if dp == 0 and dq == 0:
                continue
            if dp == 0:
                tie_p += 1
            elif dq == 0:
                tie_q += 1
            elif dp * dq > 0:
                concordant += 1
            else:
                discordant += 1
    pairs = concordant + discordant
    denominator = math.sqrt((pairs + tie_p) * (pairs + tie_q))
    return (concordant - discordant) / denominator if denominator else None


def _power(p, beta):
    logs = [math.log(x) if x > 0 else None for x in p]
    maximum = max(x for x in logs if x is not None)
    values = [math.exp(beta * (x - maximum)) if x is not None else 0.0 for x in logs]
    total = math.fsum(values)
    return [x / total for x in values]


def entropy_match(p, target):
    """Match target entropy within the reachable power-temperature family.

    beta=0 is the uniform positive support limit; beta=infinity is the tied
    maxima limit. Endpoint results record the limit rather than a finite T.
    """
    if not math.isfinite(target) or target < 0:
        raise ValueError("target entropy must be finite and nonnegative")
    support = [i for i, x in enumerate(p) if x > 0]
    if not support:
        raise ValueError("distribution has no positive support")
    peak = max(p)
    peaks = [i for i, x in enumerate(p) if x == peak]
    minimum, maximum = math.log(len(peaks)), math.log(len(support))
    result = {"feasible": False, "target_entropy": target,
              "reachable_min": minimum, "reachable_max": maximum,
              "support_size": len(support), "max_tie_count": len(peaks)}
    if target < minimum - ENTROPY_TOL:
        return dict(result, reason="below_tied_maxima_entropy", distribution=None)
    if target > maximum + ENTROPY_TOL:
        return dict(result, reason="above_positive_support_entropy", distribution=None)
    if abs(entropy(p) - target) <= ENTROPY_TOL:
        return dict(result, feasible=True, reason=None, distribution=list(p),
                    beta=1.0, temperature=1.0, achieved_entropy=entropy(p))
    if target <= minimum + ENTROPY_TOL:
        matched = [1 / len(peaks) if i in peaks else 0.0 for i in range(len(p))]
        return dict(result, feasible=True, reason=None, distribution=matched,
                    beta=None, temperature=0.0, limit="tied_maxima", achieved_entropy=entropy(matched))
    if target >= maximum - ENTROPY_TOL:
        matched = [1 / len(support) if i in support else 0.0 for i in range(len(p))]
        return dict(result, feasible=True, reason=None, distribution=matched,
                    beta=0.0, temperature=None, limit="uniform_support", achieved_entropy=entropy(matched))
    low, high = 0.0, 1.0
    while entropy(_power(p, high)) > target:
        high *= 2
        if high > 2 ** 60:
            return dict(result, reason="numerical_search_limit", distribution=None)
    for _ in range(100):
        beta = (low + high) / 2
        if entropy(_power(p, beta)) > target:
            low = beta
        else:
            high = beta
    beta = (low + high) / 2
    matched = _power(p, beta)
    return dict(result, feasible=True, reason=None, distribution=matched,
                beta=beta, temperature=1 / beta, achieved_entropy=entropy(matched))


def _top(p):
    peak = max(p[:-1])
    candidates = [i for i, value in enumerate(p[:-1]) if value == peak]
    return {"candidate_top1_indices": candidates,
            "candidate_top1_mass": peak,
            "other_mass": p[-1], "other_dominant": p[-1] > peak,
            "other_tied_for_top": p[-1] == peak}


def _comparison(s, teacher):
    left, right = _top(s), _top(teacher)
    return {"js": js(s, teacher), "entropy_difference_teacher_minus_student": entropy(teacher) - entropy(s),
            "candidate_top1_overlap": bool(set(left["candidate_top1_indices"]) & set(right["candidate_top1_indices"])),
            "candidate_top1_exact_tie_set": left["candidate_top1_indices"] == right["candidate_top1_indices"],
            "candidate_kendall_tau_b": kendall_tau_b(s[:-1], teacher[:-1]),
            "all_cell_kendall_tau_b": kendall_tau_b(s, teacher)}


def compare_distributions(s, g, q):
    """Accept aligned candidate vectors WITHOUT OTHER; return raw/shape metrics."""
    if not (len(s) == len(g) == len(q)):
        raise ValueError("candidate vectors must share the same cells")
    s, g, q = map(distribution, (s, g, q))
    raw_g, raw_q = _comparison(s, g), _comparison(s, q)
    target = entropy(s)
    shape_g, shape_q = entropy_match(g, target), entropy_match(q, target)
    comparisons = {"baseline": None, "cross": None}
    for name, match in (("baseline", shape_g), ("cross", shape_q)):
        if match["feasible"]:
            comparisons[name] = _comparison(s, match["distribution"])
    shape_complete = all(x is not None for x in comparisons.values())
    return {"X_raw": raw_q["js"] - raw_g["js"],
            "X_shape": comparisons["cross"]["js"] - comparisons["baseline"]["js"] if shape_complete else None,
            "raw": {"baseline": raw_g, "cross": raw_q}, "shape": comparisons,
            "shape_calibration": {"baseline": shape_g, "cross": shape_q},
            "shape_missing": not shape_complete,
            "top_cells": {"S": _top(s), "baseline": _top(g), "cross": _top(q)},
            "student_entropy": target, "metric_units": "nats",
            "probability_contract": "canonical_path_word_boundary_proxy"}


def _quantile(values, probability):
    values = sorted(values)
    offset = probability * (len(values) - 1)
    lower = int(offset)
    fraction = offset - lower
    return values[lower] * (1 - fraction) + values[min(lower + 1, len(values) - 1)] * fraction


def _summary(rows, key, repeats):
    grouped = defaultdict(list)
    missing = 0
    for row in rows:
        value = row.get(key)
        weight = float(row.get("sampling_weight", 1.0))
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError("sampling_weight must be finite and positive")
        if value is None:
            missing += 1
            continue
        value = float(value)
        if not math.isfinite(value):
            raise ValueError("metric must be finite or None")
        grouped[str(row["prompt_id"])].append((value, weight))
    prompt_values = []
    for prompt in sorted(grouped):
        entries = grouped[prompt]
        prompt_values.append(math.fsum(v * w for v, w in entries) / math.fsum(w for _, w in entries))
    count = len(prompt_values)
    mean = math.fsum(prompt_values) / count if count else None
    result = {"mean": mean, "position_count": len(rows), "valid_position_count": len(rows) - missing,
              "missing_position_count": missing, "prompt_count": len({str(r["prompt_id"]) for r in rows}),
              "valid_prompt_count": count, "bootstrap_repeats": repeats, "bootstrap_seed": 42,
              "ci95": None, "uncertainty": "prompt_cluster_bootstrap_exploratory"}
    if count and repeats:
        rng = random.Random(42)
        draws = [math.fsum(rng.choice(prompt_values) for _ in range(count)) / count for _ in range(repeats)]
        result["ci95"] = [_quantile(draws, 0.025), _quantile(draws, 0.975)]
    return result


def aggregate(rows, key, repeats=1000):
    """Weighted positions within prompt, equal prompts, deterministic bootstrap."""
    if not isinstance(repeats, int) or repeats < 0:
        raise ValueError("repeats must be a nonnegative integer")
    rows = list(rows)
    result = _summary(rows, key, repeats)
    categories = defaultdict(list)
    for row in rows:
        categories[str(row.get("category", "unknown"))].append(row)
    result["categories"] = {name: _summary(group, key, repeats) for name, group in sorted(categories.items())}
    result["metric"] = key
    result["missing_flags"] = {name: sum(row.get(name) is None for row in rows) for name in ("X_raw", "X_shape")}
    result["estimand"] = "equal_prompt_mean_of_sampling_weighted_available_positions"
    return result
