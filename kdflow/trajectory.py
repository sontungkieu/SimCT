"""Explicit token trajectories for text-only on-policy diagnostics."""

from statistics import mean, median


def trajectory_tokens(prompt_ids, response_ids, eos_id):
    """Keep sampled IDs intact; append a masked EOS sentinel only when absent.

    The sentinel preserves the legacy MP-OPD denominator. It is not evidence
    of a sampled stopping event, and the atomizer excludes it from credit.
    Masks select predictor positions, whose next token is a response label.
    """
    prompt = list(prompt_ids)
    response = list(response_ids)
    if not prompt or eos_id is None:
        raise ValueError("trajectory requires a nonempty prompt and an EOS ID")
    if any(type(x) is not int or x < 0 for x in prompt + response):
        raise ValueError("trajectory IDs must be nonnegative integers")
    synthetic_eos = not response or response[-1] != eos_id
    labels = response + ([eos_id] if synthetic_eos else [])
    ids = prompt + labels
    mask = [False] * (len(prompt) - 1) + [True] * len(labels) + [False]
    assert len(ids) == len(mask)
    return ids, mask, synthetic_eos


def collapse_observation(lengths, baseline, bad_streak):
    """Predeclared two-batch collapse gate; lengths exclude sampled EOS."""
    if not lengths:
        raise ValueError("empty rollout batch")
    avg = mean(lengths)
    med = median(lengths)
    empty = sum(x == 0 for x in lengths) / len(lengths)
    baseline = avg if baseline is None else baseline
    bad = med == 0 or empty >= 0.25 or avg < 0.1 * baseline
    streak = bad_streak + 1 if bad else 0
    return {
        "content_length_mean": avg,
        "content_length_median": med,
        "empty_response_fraction": empty,
        "content_length_baseline": baseline,
        "collapse_bad_streak": streak,
        "collapse_stop": streak >= 2,
    }
