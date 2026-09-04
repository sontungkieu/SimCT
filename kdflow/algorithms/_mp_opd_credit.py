"""Dimensionally consistent scalar path credit for MP-OPD."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from ._mp_opd_atoms import MPAtom


@dataclass(frozen=True)
class AtomCreditTensors:
    teacher_log_score: torch.Tensor
    student_old_log_score: torch.Tensor
    base_credit: torch.Tensor
    weight: torch.Tensor
    rate: torch.Tensor
    current_nll: torch.Tensor


def realized_token_log_probs(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    if logits.ndim != 2 or labels.ndim != 1 or logits.shape[0] != labels.numel():
        raise ValueError("logits must be [tokens,vocab] and labels [tokens]")
    return torch.log_softmax(logits.float(), dim=-1).gather(1, labels.long().unsqueeze(1)).squeeze(1)


def build_atom_credits(
    atoms: Sequence[MPAtom],
    student_logits: torch.Tensor,
    student_labels: torch.Tensor,
    teacher_logits: torch.Tensor,
    teacher_labels: torch.Tensor,
) -> AtomCreditTensors:
    if not atoms:
        raise ValueError("at least one valid atom is required")
    student_logp = realized_token_log_probs(student_logits, student_labels)
    teacher_logp = realized_token_log_probs(teacher_logits.detach(), teacher_labels).detach()
    l_t, l_s, h, weights = [], [], [], []
    for atom in atoms:
        if not atom.valid or atom.student_token_count <= 0:
            raise ValueError("invalid atom in credit computation")
        l_t.append(teacher_logp[atom.teacher_start : atom.teacher_end].sum())
        score = student_logp[atom.student_start : atom.student_end].sum()
        l_s.append(score.detach())
        h.append(-score)
        weights.append(atom.student_token_count)
    teacher_score = torch.stack(l_t).detach()
    student_old = torch.stack(l_s).detach()
    weight = torch.tensor(weights, dtype=torch.float32, device=student_logits.device)
    base = (teacher_score - student_old).detach()
    rate = (base / weight).detach()
    return AtomCreditTensors(teacher_score, student_old, base, weight, rate, torch.stack(h))


def span_tables(base: torch.Tensor, weight: torch.Tensor, max_span_length: int):
    if base.ndim != 1 or weight.shape != base.shape or (weight <= 0).any():
        raise ValueError("base/weight must be aligned vectors with positive weights")
    n = base.numel()
    length = min(max(int(max_span_length), 1), max(n, 1))
    b_span = base.new_zeros((n, length))
    w_span = weight.new_zeros((n, length))
    valid = torch.zeros((n, length), dtype=torch.bool, device=base.device)
    pb = torch.cat((base.new_zeros(1), base.cumsum(0)))
    pw = torch.cat((weight.new_zeros(1), weight.cumsum(0)))
    for start in range(n):
        for offset in range(length):
            end = start + offset + 1
            if end <= n:
                b_span[start, offset] = pb[end] - pb[start]
                w_span[start, offset] = pw[end] - pw[start]
                valid[start, offset] = True
    rate = torch.where(valid, b_span / w_span.clamp_min(1), torch.zeros_like(b_span)).detach()
    return b_span.detach(), w_span.detach(), rate, valid


def hard_partition_loss(
    current_nll: torch.Tensor,
    base: torch.Tensor,
    weight: torch.Tensor,
    partition: Sequence[tuple[int, int]],
) -> torch.Tensor:
    total = current_nll.new_zeros(())
    cursor = 0
    for start, end in partition:
        if start != cursor or not (start < end <= current_nll.numel()):
            raise ValueError("partition must cover atoms once, contiguously, in order")
        rate = base[start:end].sum() / weight[start:end].sum()
        total = total + rate.detach() * current_nll[start:end].sum()
        cursor = end
    if cursor != current_nll.numel():
        raise ValueError("partition does not cover all atoms")
    return total


def expected_atom_rates(span_marginals: torch.Tensor, span_rates: torch.Tensor) -> torch.Tensor:
    if span_marginals.shape != span_rates.shape:
        raise ValueError("marginals and rates must have the same [n,L] shape")
    n, length = span_marginals.shape
    result = span_rates.new_zeros(n)
    for start in range(n):
        for offset in range(length):
            end = start + offset + 1
            if end <= n:
                result[start:end] += span_marginals[start, offset] * span_rates[start, offset]
    return result


def soft_partition_loss(current_nll: torch.Tensor, atom_rates: torch.Tensor) -> torch.Tensor:
    if current_nll.shape != atom_rates.shape:
        raise ValueError("current_nll and atom_rates must be aligned")
    return (atom_rates.detach() * current_nll).sum()


def credit_conservation_residual(
    base: torch.Tensor, weight: torch.Tensor, atom_rates: torch.Tensor
) -> torch.Tensor:
    return (weight * atom_rates).sum() - base.sum()
