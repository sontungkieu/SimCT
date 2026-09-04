import itertools
import math

import pytest
import torch

from kdflow.algorithms._mp_opd_oracle import enumerate_partitions
from kdflow.algorithms._mp_opd_semimarkov import semi_markov_partition


def score_partition(energies, partition):
    return sum(energies[i, j - i - 1] for i, j in partition)


@pytest.mark.parametrize("n,max_length", [(1, 4), (2, 2), (5, 3), (8, 4)])
def test_dp_matches_bruteforce_logz_marginals_and_coverage(n, max_length):
    torch.manual_seed(100 + n)
    energies = torch.randn(n, min(max_length, n), dtype=torch.float64)
    result = semi_markov_partition(energies, temperature=0.8)
    partitions = list(enumerate_partitions(n, max_length))
    scores = torch.stack([score_partition(energies, part) / 0.8 for part in partitions])
    brute_logz = torch.logsumexp(scores, 0)
    brute_mu = torch.zeros_like(result.marginals, dtype=torch.float64)
    probabilities = torch.softmax(scores, 0)
    for probability, partition in zip(probabilities, partitions):
        for start, end in partition:
            brute_mu[start, end - start - 1] += probability
    assert torch.allclose(result.log_z.double(), brute_logz, atol=2e-6)
    assert torch.allclose(result.marginals.double(), brute_mu, atol=2e-6)
    assert float(result.coverage_max_error) < 1e-6
    assert torch.allclose(probabilities.sum(), torch.tensor(1.0, dtype=probabilities.dtype))


def test_n_zero_is_neutral_and_all_invalid_fails_closed():
    empty = semi_markov_partition(torch.empty(0, 1))
    assert empty.marginals.shape == (0, 1) and float(empty.log_z) == 0.0
    with pytest.raises(ValueError, match="masked"):
        semi_markov_partition(torch.zeros(2, 2), valid_mask=torch.zeros(2, 2, dtype=torch.bool))


def test_mixed_precision_input_runs_dp_in_float32_without_nonfinite_outputs():
    energies = torch.tensor([[100.0, -100.0], [75.0, 0.0]], dtype=torch.float16)
    result = semi_markov_partition(energies)
    assert result.log_z.dtype == torch.float32
    assert all(torch.isfinite(value).all() for value in (result.log_z, result.marginals, result.entropy))
