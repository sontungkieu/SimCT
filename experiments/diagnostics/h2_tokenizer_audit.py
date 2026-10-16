"""Exact source-byte segmentation audit, with explicit matching denominators.

Input spans are non-special (byte_start, byte_end, token_id) triples partitioning
the same UTF-8 source text. Numerical token IDs across vocabularies are only
incidental labels: equal IDs never establish lexical compatibility. Statistics
retain full float precision; no unverified threshold such as '90%' is inferred.
"""

import bisect
import math
import re
import unicodedata
from collections import defaultdict

CLASSES = ("1:1", "1:N", "N:1", "N:M")


def _ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def _spans(spans, source):
    result, cursor = [], 0
    for span in spans:
        if isinstance(span, dict):
            if span.get("special", False) or span.get("is_special", False):
                raise ValueError("special tokens must be excluded before byte-span auditing")
            a, b, token = span["start"], span["end"], span["token_id"]
            if "bytes" in span and span["bytes"] != source[a:b]:
                raise ValueError("token bytes disagree with source")
        else:
            a, b, token = span
        if not isinstance(a, int) or not isinstance(b, int) or a != cursor or b <= a or b > len(source):
            raise ValueError("spans must be a positive contiguous partition of the source bytes")
        if not isinstance(token, int) or token < 0:
            raise ValueError("token_id must be a nonnegative integer")
        result.append((a, b, token))
        cursor = b
    if cursor != len(source):
        raise ValueError("spans do not cover the complete source")
    return result


def _character_context(text):
    starts, characters, offset = [], [], 0
    for character in text:
        starts.append(offset)
        characters.append(character)
        offset += len(character.encode("utf-8"))
    return starts, characters, set(starts + [offset])


def _category(source, a, b, starts, characters):
    raw = source[a:b]
    try:
        piece = raw.decode("utf-8")
    except UnicodeDecodeError:
        return "other"
    if piece.isspace():
        return "whitespace"
    stripped = piece.lstrip()
    if any(c in piece for c in "\\${}"):
        return "latex"
    if stripped and all(c.isdigit() for c in stripped):
        return "digit"
    if stripped and all(unicodedata.category(c)[0] in "PS" for c in stripped):
        return "punctuation"
    if stripped and (stripped[0].isalpha() or stripped[0] == "_"):
        first = a + len(piece[:len(piece) - len(stripped)].encode("utf-8"))
        index = bisect.bisect_left(starts, first)
        previous = characters[index - 1] if index else ""
        return "word_internal" if previous and (previous.isalnum() or previous == "_") else "word_initial"
    return "other"


def _boundary_metrics(counts):
    s, t, shared = counts["student_interior_boundaries"], counts["teacher_interior_boundaries"], counts["shared_interior_boundaries"]
    # Teacher boundaries are the predicted set; student boundaries reference.
    # Both empty is exact set agreement; a one-sided empty set has zero overlap.
    both_empty = s == 0 and t == 0
    return {"precision_teacher_against_student": shared / t if t else (1.0 if both_empty else 0.0),
            "recall_teacher_against_student": shared / s if s else (1.0 if both_empty else 0.0),
            "jaccard": shared / (s + t - shared) if s + t - shared else 1.0,
            "empty_set_convention": "both_empty=1; one_sided_empty_overlap=0",
            "excluded_boundaries": ["source_start", "source_end"]}


def _coverage(counts):
    return {"strict_student_token_fraction": _ratio(counts["strict_pairs"], counts["student_tokens"]),
            "strict_teacher_token_fraction": _ratio(counts["strict_pairs"], counts["teacher_tokens"]),
            "strict_byte_fraction": _ratio(counts["strict_bytes"], counts["bytes"])}


def audit_pair(student_spans, teacher_spans, text):
    """Audit identical byte spans and coarse alignment; no tokenizer inference."""
    if not isinstance(text, str):
        raise TypeError("text must be the exact Unicode source string")
    source = text.encode("utf-8")
    student, teacher = _spans(student_spans, source), _spans(teacher_spans, source)
    length = len(source)
    s_boundaries = {0, length, *(a for a, _, _ in student)}
    t_boundaries = {0, length, *(a for a, _, _ in teacher)}
    shared = sorted(s_boundaries & t_boundaries)
    interior_s, interior_t = s_boundaries - {0, length}, t_boundaries - {0, length}
    starts, characters, character_boundaries = _character_context(text)
    intervals, class_counts = [], {name: {"intervals": 0, "student_tokens": 0, "teacher_tokens": 0, "bytes": 0} for name in CLASSES}
    si = ti = 0
    interval_of_student = {}
    for a, b in zip(shared, shared[1:]):
        previous_s, previous_t = si, ti
        while si < len(student) and student[si][1] <= b:
            si += 1
        while ti < len(teacher) and teacher[ti][1] <= b:
            ti += 1
        ns, nt = si - previous_s, ti - previous_t
        kind = "1:1" if ns == nt == 1 else ("1:N" if ns == 1 else ("N:1" if nt == 1 else "N:M"))
        row = {"start": a, "end": b, "class": kind, "student_indices": list(range(previous_s, si)), "teacher_indices": list(range(previous_t, ti)),
               "student_count": ns, "teacher_count": nt, "bytes": b - a}
        index = len(intervals)
        intervals.append(row)
        for i in range(previous_s, si):
            interval_of_student[i] = index
        counts = class_counts[kind]
        counts["intervals"] += 1
        counts["student_tokens"] += ns
        counts["teacher_tokens"] += nt
        counts["bytes"] += b - a
    trace, category_counts, strict_pairs, strict_bytes = [], {}, 0, 0
    teacher_start = 0
    incidental_equal_id_different_bytes = 0
    whitespace_mask = [c.isspace() for c in text for _ in c.encode("utf-8")]
    whitespace_bytes = sum(whitespace_mask)
    teacher_whitespace_tokens = sum(all(whitespace_mask[a:b]) for a, b, _ in teacher)
    for index, (a, b, token) in enumerate(student):
        while teacher_start < len(teacher) and teacher[teacher_start][1] <= a:
            teacher_start += 1
        end = teacher_start
        while end < len(teacher) and teacher[end][0] < b:
            end += 1
        overlaps = list(range(teacher_start, end))
        strict = len(overlaps) == 1 and teacher[teacher_start][:2] == (a, b)
        same_head = a in t_boundaries
        category = _category(source, a, b, starts, characters)
        coarse = intervals[interval_of_student[index]]
        same_ids = [j for j in overlaps if teacher[j][2] == token]
        different_bytes_ids = [j for j in same_ids if source[teacher[j][0]:teacher[j][1]] != source[a:b]]
        incidental_equal_id_different_bytes += len(different_bytes_ids)
        # Byte-fragment tokens still have exact whitespace-byte denominators.
        whitespace_here = sum(whitespace_mask[a:b])
        counts = category_counts.setdefault(category, {"student_tokens": 0, "strict_student_tokens": 0, "bytes": 0, "strict_bytes": 0})
        counts["student_tokens"] += 1
        counts["strict_student_tokens"] += int(strict)
        counts["bytes"] += b - a
        counts["strict_bytes"] += (b - a) if strict else 0
        if strict:
            strict_pairs += 1
            strict_bytes += b - a
        trace.append({"student_index": index, "student_token_id": token, "start": a, "end": b,
                      "teacher_indices": overlaps, "teacher_count": len(overlaps), "same_head": same_head,
                      "context_aligned": same_head, "same_end": b in t_boundaries, "strict": strict,
                      "category": category, "coarse_interval_index": interval_of_student[index],
                      "minimal_coarse_interval": {"start": coarse["start"], "end": coarse["end"], "class": coarse["class"]},
                      "same_numeric_id_teacher_indices": same_ids,
                      "same_numeric_id_different_bytes_teacher_indices": different_bytes_ids,
                      "same_numeric_id_on_strict_pair": bool(strict and same_ids),
                      "same_source_bytes_on_strict_pair": strict,
                      "utf8_start_boundary": a in character_boundaries, "utf8_end_boundary": b in character_boundaries,
                      "contains_newline_bytes": b"\n" in source[a:b] or b"\r" in source[a:b],
                      "contains_non_ascii_bytes": any(x >= 128 for x in source[a:b]),
                      "whitespace_bytes": whitespace_here, "nonwhitespace_bytes": b - a - whitespace_here})
    counts = {"student_tokens": len(student), "teacher_tokens": len(teacher), "bytes": length,
              "strict_pairs": strict_pairs, "strict_bytes": strict_bytes,
              "student_interior_boundaries": len(interior_s), "teacher_interior_boundaries": len(interior_t),
              "shared_interior_boundaries": len(interior_s & interior_t),
              "student_nonwhitespace_tokens": sum(r["nonwhitespace_bytes"] > 0 for r in trace),
              "teacher_nonwhitespace_tokens": len(teacher) - teacher_whitespace_tokens,
              "strict_nonwhitespace_student_tokens": sum(r["strict"] and r["nonwhitespace_bytes"] > 0 for r in trace),
              "nonwhitespace_bytes": length - whitespace_bytes,
              "strict_nonwhitespace_bytes": sum(r["nonwhitespace_bytes"] for r in trace if r["strict"]),
              "strict_pairs_same_numeric_id": sum(r["same_numeric_id_on_strict_pair"] for r in trace),
              "overlap_pairs_same_numeric_id_different_bytes": incidental_equal_id_different_bytes,
              "context_aligned_student_tokens": sum(r["context_aligned"] for r in trace)}
    for values in class_counts.values():
        values.update(student_token_fraction=_ratio(values["student_tokens"], len(student)),
                      teacher_token_fraction=_ratio(values["teacher_tokens"], len(teacher)), byte_fraction=_ratio(values["bytes"], length))
    for values in category_counts.values():
        values.update(strict_student_token_fraction=_ratio(values["strict_student_tokens"], values["student_tokens"]),
                      strict_byte_fraction=_ratio(values["strict_bytes"], values["bytes"]))
    numeric = []
    for match in re.finditer(r"\d+", text):
        a = len(text[:match.start()].encode("utf-8"))
        b = len(text[:match.end()].encode("utf-8"))
        ss = [i for i, (x, y, _) in enumerate(student) if x < b and y > a]
        tt = [i for i, (x, y, _) in enumerate(teacher) if x < b and y > a]
        numeric.append({"text": match.group(), "start": a, "end": b, "character_length": len(match.group()), "byte_length": b - a,
                        "student_indices": ss, "teacher_indices": tt,
                        "student_boundary_offsets": sorted(x - a for x in s_boundaries if a < x < b),
                        "teacher_boundary_offsets": sorted(x - a for x in t_boundaries if a < x < b),
                        "student_run_start_boundary": a in s_boundaries, "teacher_run_start_boundary": a in t_boundaries,
                        "student_run_end_boundary": b in s_boundaries, "teacher_run_end_boundary": b in t_boundaries,
                        "exact_run_segmentation_match": {x for x in s_boundaries if a < x < b} == {x for x in t_boundaries if a < x < b}})
    return {"counts": counts, "coverage": _coverage(counts), "boundaries": _boundary_metrics(counts),
            "denominators": {"all_non_special_student_tokens": len(student), "all_non_special_teacher_tokens": len(teacher),
                             "all_source_bytes": length, "nonwhitespace_student_tokens": counts["student_nonwhitespace_tokens"],
                             "nonwhitespace_teacher_tokens": counts["teacher_nonwhitespace_tokens"],
                             "nonwhitespace_source_bytes": counts["nonwhitespace_bytes"], "whitespace_source_bytes": whitespace_bytes,
                             "excluded_special_tokens": "not present in input; count unknown"},
            "excluding_whitespace": {"strict_student_token_fraction": _ratio(counts["strict_nonwhitespace_student_tokens"], counts["student_nonwhitespace_tokens"]),
                                     "strict_teacher_token_fraction": _ratio(counts["strict_nonwhitespace_student_tokens"], counts["teacher_nonwhitespace_tokens"]),
                                     "strict_byte_fraction": _ratio(counts["strict_nonwhitespace_bytes"], counts["nonwhitespace_bytes"])},
            "interval_classes": class_counts, "common_boundary_intervals": intervals, "student_token_trace": trace,
            "categories": category_counts, "numeric_runs": numeric,
            "operations": {"teacher_splits_student_intervals": class_counts["1:N"]["intervals"],
                           "teacher_merges_student_intervals": class_counts["N:1"]["intervals"],
                           "mixed_resegmentation_intervals": class_counts["N:M"]["intervals"],
                           "student_tokens_dropped_by_strict_filter": len(student) - strict_pairs,
                           "teacher_tokens_dropped_by_strict_filter": len(teacher) - strict_pairs,
                           "literal_source_bytes_dropped": 0},
            "source": {"utf8_bytes": length, "unicode_characters": len(text), "non_ascii_characters": sum(ord(c) > 127 for c in text),
                       "newline_characters": sum(c in "\r\n" for c in text)},
            "context_contract": "same exact source prefix bytes when token start is a teacher boundary; excludes template/special context",
            "token_id_contract": "equal numeric IDs across vocabularies are incidental, not vocabulary compatibility"}


def _quantiles(values):
    values = sorted(x for x in values if x is not None)
    result = {}
    for percentile in (0, 10, 25, 50, 75, 90, 100):
        if not values:
            result[f"p{percentile}"] = None
            continue
        index = percentile / 100 * (len(values) - 1)
        lower = int(index)
        fraction = index - lower
        result[f"p{percentile}"] = values[lower] * (1 - fraction) + values[min(lower + 1, len(values) - 1)] * fraction
    return result


def _mean(values):
    values = [x for x in values if x is not None]
    return math.fsum(values) / len(values) if values else None


def _micro(audits):
    counts = {key: sum(audit["counts"][key] for audit in audits) for key in audits[0]["counts"]} if audits else {}
    if not counts:
        return {"counts": {}, "coverage": {}, "boundaries": {}}
    classes = {}
    for kind in CLASSES:
        row = {key: sum(audit["interval_classes"][kind][key] for audit in audits) for key in ("intervals", "student_tokens", "teacher_tokens", "bytes")}
        row.update(student_token_fraction=_ratio(row["student_tokens"], counts["student_tokens"]),
                   teacher_token_fraction=_ratio(row["teacher_tokens"], counts["teacher_tokens"]), byte_fraction=_ratio(row["bytes"], counts["bytes"]))
        classes[kind] = row
    categories = defaultdict(lambda: {"student_tokens": 0, "strict_student_tokens": 0, "bytes": 0, "strict_bytes": 0})
    numeric_lengths = defaultdict(int)
    numeric_count = numeric_matched = 0
    for audit in audits:
        for category, values in audit["categories"].items():
            for key in ("student_tokens", "strict_student_tokens", "bytes", "strict_bytes"):
                categories[category][key] += values[key]
        for run in audit["numeric_runs"]:
            numeric_lengths[str(run["character_length"])] += 1
            numeric_count += 1
            numeric_matched += int(run["exact_run_segmentation_match"])
    for values in categories.values():
        values.update(strict_student_token_fraction=_ratio(values["strict_student_tokens"], values["student_tokens"]),
                      strict_byte_fraction=_ratio(values["strict_bytes"], values["bytes"]))
    return {"counts": counts, "coverage": _coverage(counts), "boundaries": _boundary_metrics(counts), "interval_classes": classes,
            "categories": dict(categories), "source": {key: sum(a["source"][key] for a in audits) for key in audits[0]["source"]},
            "numeric_runs": {"count": numeric_count, "character_length_histogram": dict(sorted(numeric_lengths.items(), key=lambda item: int(item[0]))),
                             "exact_run_segmentation_match_count": numeric_matched,
                             "exact_run_segmentation_match_fraction": _ratio(numeric_matched, numeric_count)},
            "excluding_whitespace": {"strict_student_token_fraction": _ratio(counts["strict_nonwhitespace_student_tokens"], counts["student_nonwhitespace_tokens"]),
                                     "strict_teacher_token_fraction": _ratio(counts["strict_nonwhitespace_student_tokens"], counts["teacher_nonwhitespace_tokens"]),
                                     "strict_byte_fraction": _ratio(counts["strict_nonwhitespace_bytes"], counts["nonwhitespace_bytes"])}}


def aggregate_audits(rows):
    """Records {prompt_id, rollout_id, audit}; use separate calls per model pair.

    Prompt macro means first average rollout fractions inside each prompt, then
    average prompts equally. Global micro sums raw numerators/denominators.
    """
    rows = list(rows)
    prompt_groups = defaultdict(list)
    seen = set()
    for row in rows:
        identity = (str(row["prompt_id"]), str(row["rollout_id"]))
        if identity in seen:
            raise ValueError("duplicate prompt/rollout identity; aggregate model pairs separately")
        seen.add(identity)
        prompt_groups[identity[0]].append(row["audit"])
    keys = ("strict_student_token_fraction", "strict_teacher_token_fraction", "strict_byte_fraction")
    boundary_keys = ("precision_teacher_against_student", "recall_teacher_against_student", "jaccard")
    per_prompt = []
    for prompt, audits in sorted(prompt_groups.items()):
        per_prompt.append({"prompt_id": prompt, "rollout_count": len(audits), "micro": _micro(audits),
                           "macro_rollout_coverage": {key: _mean([a["coverage"][key] for a in audits]) for key in keys},
                           "macro_rollout_boundaries": {key: _mean([a["boundaries"][key] for a in audits]) for key in boundary_keys},
                           "rollout_strict_coverage_quantiles": {key: _quantiles([a["coverage"][key] for a in audits]) for key in keys}})
    audits = [r["audit"] for r in rows]
    return {"rollout_count": len(rows), "prompt_count": len(prompt_groups),
            "per_rollout": [{"prompt_id": r["prompt_id"], "rollout_id": r["rollout_id"], "counts": r["audit"]["counts"],
                             "coverage": r["audit"]["coverage"], "boundaries": r["audit"]["boundaries"]} for r in rows],
            "per_prompt": per_prompt, "global_micro": _micro(audits),
            "global_macro_rollout_coverage": {key: _mean([a["coverage"][key] for a in audits]) for key in keys},
            "global_macro_prompt_coverage": {key: _mean([p["macro_rollout_coverage"][key] for p in per_prompt]) for key in keys},
            "global_macro_rollout_boundaries": {key: _mean([a["boundaries"][key] for a in audits]) for key in boundary_keys},
            "global_macro_prompt_boundaries": {key: _mean([p["macro_rollout_boundaries"][key] for p in per_prompt]) for key in boundary_keys},
            "rollout_strict_coverage_quantiles": {key: _quantiles([a["coverage"][key] for a in audits]) for key in keys},
            "prompt_strict_coverage_quantiles": {key: _quantiles([p["macro_rollout_coverage"][key] for p in per_prompt]) for key in keys},
            "aggregation_contract": "micro sums counts; rollout macro equal rollout weight; prompt macro equal prompt weight after equal rollout weight",
            "threshold_claim": "no reference source supplied; no 90 percent match claim assumed"}
