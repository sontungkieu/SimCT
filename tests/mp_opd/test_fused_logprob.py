"""Fused selected-token log-probabilities: parity, chunk invariance, higher-order."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from kdflow.fused_logprob import (  # noqa: E402
    accumulation_dtype,
    chunked_head_selected_logprobs,
    masked_selected_logprobs,
)


def reference(logits, labels):
    """The current production shape: fp32 log_softmax, then gather one value."""
    acc = accumulation_dtype(logits.dtype)
    return torch.log_softmax(logits.to(acc), dim=-1).gather(
        1, labels.long().unsqueeze(1)).squeeze(1)


def test_matches_the_production_helper():
    from kdflow.algorithms._mp_opd_credit import realized_token_log_probs

    torch.manual_seed(0)
    logits = torch.randn(6, 97)
    labels = torch.randint(0, 97, (6,))
    got = masked_selected_logprobs(logits, labels)
    assert torch.allclose(got, reference(logits, labels), atol=1e-6)
    assert torch.allclose(got, realized_token_log_probs(logits, labels), atol=1e-6)


def test_chunk_size_does_not_change_the_values():
    torch.manual_seed(1)
    logits = torch.randn(5, 100)
    labels = torch.randint(0, 100, (5,))
    full = masked_selected_logprobs(logits, labels, vocab_chunk=4096)
    for chunk in (1, 3, 7, 32, 99):
        got = masked_selected_logprobs(logits, labels, vocab_chunk=chunk)
        assert torch.allclose(got, full, atol=1e-5), chunk


def test_softcap_is_applied_before_normalisation():
    torch.manual_seed(2)
    logits = torch.randn(4, 64) * 5
    labels = torch.randint(0, 64, (4,))
    cap = 30.0
    capped = torch.tanh(logits / cap) * cap
    assert torch.allclose(masked_selected_logprobs(logits, labels, softcap=cap),
                          reference(capped, labels), atol=1e-6)


def test_labels_landing_in_every_chunk_position():
    torch.manual_seed(3)
    logits = torch.randn(3, 12)
    labels = torch.tensor([0, 5, 11])
    assert torch.allclose(masked_selected_logprobs(logits, labels, vocab_chunk=4),
                          reference(logits, labels), atol=1e-6)


def test_chunked_head_matches_head_then_softmax():
    torch.manual_seed(4)
    hidden = torch.randn(7, 16)
    weight = torch.randn(53, 16)
    bias = torch.randn(53)
    labels = torch.randint(0, 53, (7,))
    direct = torch.nn.functional.linear(hidden, weight, bias)
    got = chunked_head_selected_logprobs(hidden, labels, weight, bias, vocab_chunk=8)
    assert torch.allclose(got, reference(direct, labels), atol=1e-5)
    noref = torch.nn.functional.linear(hidden, weight, None)
    got2 = chunked_head_selected_logprobs(hidden, labels, weight, None, vocab_chunk=53)
    assert torch.allclose(got2, reference(noref, labels), atol=1e-5)


def test_first_gradient_matches_the_reference_path():
    torch.manual_seed(5)
    logits = torch.randn(4, 40, dtype=torch.float64, requires_grad=True)
    labels = torch.randint(0, 40, (4,))
    g1 = torch.autograd.grad(
        masked_selected_logprobs(logits, labels, vocab_chunk=7).sum(), logits)[0]
    g1_ref = torch.autograd.grad(reference(logits, labels).sum(), logits)[0]
    assert torch.allclose(g1, g1_ref, atol=1e-10)


def test_double_backward_matches_the_reference_path():
    """The energy path needs grad-of-grad; a fused path that breaks it is useless."""
    torch.manual_seed(6)
    logits = torch.randn(4, 40, dtype=torch.float64, requires_grad=True)
    labels = torch.randint(0, 40, (4,))
    u = torch.randn(4, dtype=torch.float64)

    def second_grad(fn):
        out = fn().sum()
        first = torch.autograd.grad(out, logits, create_graph=True)[0]
        return torch.autograd.grad((first * u.unsqueeze(1)).sum(), logits)[0]

    got = second_grad(lambda: masked_selected_logprobs(logits, labels, vocab_chunk=7))
    ref = second_grad(lambda: reference(logits, labels))
    assert torch.allclose(got, ref, atol=1e-9)


def test_gradcheck_and_gradgradcheck_on_tiny_fixture():
    torch.manual_seed(7)
    logits = torch.randn(3, 11, dtype=torch.float64, requires_grad=True)
    labels = torch.tensor([0, 6, 10])
    fn = lambda x: masked_selected_logprobs(x, labels, vocab_chunk=4)
    assert torch.autograd.gradcheck(fn, (logits,), eps=1e-6, atol=1e-8)
    assert torch.autograd.gradgradcheck(fn, (logits,), eps=1e-6, atol=1e-6)


def test_gradgradcheck_for_the_chunked_head():
    torch.manual_seed(8)
    hidden = torch.randn(3, 5, dtype=torch.float64, requires_grad=True)
    weight = torch.randn(9, 5, dtype=torch.float64, requires_grad=True)
    labels = torch.tensor([0, 4, 8])
    fn = lambda h, w: chunked_head_selected_logprobs(h, labels, w, None, vocab_chunk=4)
    assert torch.autograd.gradgradcheck(fn, (hidden, weight), eps=1e-6, atol=1e-6)


def test_dtype_policy_keeps_fp64_and_promotes_half():
    assert accumulation_dtype(torch.float64) is torch.float64
    assert accumulation_dtype(torch.float32) is torch.float32
    assert accumulation_dtype(torch.bfloat16) is torch.float32
    assert accumulation_dtype(torch.float16) is torch.float32


def test_invalid_inputs_are_rejected():
    with pytest.raises(ValueError):
        masked_selected_logprobs(torch.randn(4), torch.tensor([0, 1, 2, 3]))
    with pytest.raises(ValueError):
        masked_selected_logprobs(torch.randn(4, 8), torch.tensor([0, 1]))
    with pytest.raises(ValueError):
        masked_selected_logprobs(torch.randn(4, 8), torch.tensor([0, 1, 2, 3]), vocab_chunk=0)
    with pytest.raises(ValueError):
        chunked_head_selected_logprobs(torch.randn(2, 4), torch.tensor([0, 1]), torch.randn(8, 5))
