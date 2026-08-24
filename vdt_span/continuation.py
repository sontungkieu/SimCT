"""Exercise 4: score an autoregressive continuation at evolving prefixes."""

from __future__ import annotations

from collections.abc import Callable, Sequence

NextTokenLogprob = Callable[[tuple[int, ...], int], float]


def continuation_logprob(
    prefix: Sequence[int],
    continuation: Sequence[int],
    next_token_logprob: NextTokenLogprob,
) -> float:
    """Return the joint log-probability of ``continuation`` after ``prefix``.

    The callback must be queried once per continuation token with the prefix
    extended by all earlier continuation tokens. See Exercise 4 in
    ``STUDY.md`` for the exact scoring and validation contract.
    """

    raise NotImplementedError("TODO E4: continuation scoring")
