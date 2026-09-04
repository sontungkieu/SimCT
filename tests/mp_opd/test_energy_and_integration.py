from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from kdflow.algorithms._mp_opd_energy import (
    MPAtomEnergy,
    energy_surrogate_loss,
    load_energy_checkpoint,
    save_energy_checkpoint,
)
from kdflow.algorithms._mp_opd_semimarkov import semi_markov_partition
from kdflow.algorithms.mp_opd import MetaPartitionedOPD, random_partition
from kdflow.arguments.distillation_args import DistillationArguments


def test_energy_features_detached_phi_gradient_and_no_student_gradient_leak():
    energy = MPAtomEnergy(10, hidden_dim=8, layers=2)
    student_feature_source = torch.randn(5, 10, requires_grad=True)
    logits = energy(student_feature_source, 3)
    q = semi_markov_partition(logits)
    loss = q.entropy
    loss.backward()
    assert student_feature_source.grad is None
    assert any(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in energy.parameters())


def test_energy_surrogate_detaches_utility_and_steps_only_phi():
    model = MPAtomEnergy(10, hidden_dim=8, layers=1)
    features = torch.randn(4, 10, requires_grad=True)
    utilities = torch.randn(4, 3, requires_grad=True)
    valid = torch.tensor([[1, 1, 1], [1, 1, 1], [1, 1, 0], [1, 0, 0]], dtype=torch.bool)
    objective, _ = energy_surrogate_loss(
        model, features, utilities, max_span_length=3, temperature=0.7,
        virtual_learning_rate=0.05, valid_mask=valid,
    )
    objective.backward()
    assert features.grad is None and utilities.grad is None
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_energy_checkpoint_round_trip_includes_optimizer_and_config(tmp_path):
    model = MPAtomEnergy(10, hidden_dim=8, layers=1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model(torch.randn(3, 10), 2).nan_to_num().sum().backward()
    optimizer.step()
    path = tmp_path / "energy.pt"
    save_energy_checkpoint(path, model, optimizer, step=7, extra_config={"max_span_length": 2})
    restored = MPAtomEnergy(10, hidden_dim=8, layers=1)
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-3)
    assert load_energy_checkpoint(
        path, restored, restored_optimizer, expected_extra_config={"max_span_length": 2}
    ) == 7
    for left, right in zip(model.parameters(), restored.parameters()):
        assert torch.equal(left, right)
    with pytest.raises(ValueError, match="config mismatch"):
        load_energy_checkpoint(
            path, restored, restored_optimizer, expected_extra_config={"max_span_length": 3}
        )


def test_random_partition_is_seed_deterministic_and_full_cover():
    first = random_partition(19, 4, 123)
    assert first == random_partition(19, 4, 123)
    assert first != random_partition(19, 4, 124)
    assert first[0][0] == 0 and first[-1][1] == 19
    assert all(left[1] == right[0] for left, right in zip(first, first[1:]))


class FakeTokenizer:
    def __init__(self, pieces, eos):
        self.pieces = pieces
        self.eos_token_id = eos
    def decode(self, ids, **_kwargs):
        return "".join(self.pieces[int(value)] for value in ids)
    def get_added_vocab(self):
        return {}


class FakeStudent(torch.nn.Module):
    def __init__(self, logits):
        super().__init__()
        self.logits = torch.nn.Parameter(logits)
    def forward(self, *_args, **_kwargs):
        return {"logits": self.logits}


def test_atomic_student_forward_teacher_score_backward_is_finite_and_shifted():
    args = SimpleNamespace(kd=DistillationArguments(kd_algorithm="mp_opd", mp_opd_mode="atomic", kd_ratio=1.0))
    strategy = SimpleNamespace(args=args, ring_attn_group=None)
    torch.manual_seed(5)
    student = FakeStudent(torch.randn(1, 4, 16))
    teacher_head = torch.nn.Linear(3, 24, bias=False)
    teacher_head.requires_grad_(False)
    algorithm = MetaPartitionedOPD(
        strategy,
        student,
        teacher_head,
        FakeTokenizer({1: "a", 2: "b", 3: "<s>", 15: "<eos>"}, 15),
        FakeTokenizer({10: "ab", 4: "<s>", 23: "<end>"}, 23),
    )
    batch = {
        "stu_input_ids": torch.tensor([[3, 1, 2, 15]]),
        "stu_attn_mask": torch.ones(1, 4),
        "stu_loss_mask": torch.tensor([[True, True, True, False]]),
        "tea_input_ids": torch.tensor([[4, 10, 23]]),
        "tea_loss_mask": torch.tensor([[True, True, False]]),
        "teacher_hiddens": torch.randn(2, 3),
        "avg_micro_batch_token_num": torch.tensor(2.0),
    }
    metrics = algorithm.training_step(batch)
    metrics["loss"].backward()
    assert torch.isfinite(metrics["loss"])
    assert student.logits.grad is not None and torch.isfinite(student.logits.grad).all()
    assert student.logits.grad[0, 3].abs().sum() == 0  # masked/future position
    assert all(value.ndim == 0 and torch.isfinite(value) for value in metrics.values())
    assert all(isinstance(float(value.detach()), float) for value in metrics.values())


def test_invalid_single_sample_microbatch_is_zero_gradient_and_auditable():
    args = SimpleNamespace(
        kd=DistillationArguments(kd_algorithm="mp_opd", mp_opd_mode="atomic", kd_ratio=1.0)
    )
    strategy = SimpleNamespace(args=args, ring_attn_group=None)
    torch.manual_seed(7)
    student = FakeStudent(torch.randn(1, 3, 16))
    teacher_head = torch.nn.Linear(3, 24, bias=False)
    teacher_head.requires_grad_(False)
    algorithm = MetaPartitionedOPD(
        strategy,
        student,
        teacher_head,
        FakeTokenizer({1: "a", 3: "<s>", 15: "<eos>"}, 15),
        FakeTokenizer({2: "b", 4: "<s>", 23: "<end>"}, 23),
    )
    batch = {
        "stu_input_ids": torch.tensor([[3, 1, 15]]),
        "stu_attn_mask": torch.ones(1, 3),
        "stu_loss_mask": torch.tensor([[True, True, False]]),
        "tea_input_ids": torch.tensor([[4, 2, 23]]),
        "tea_loss_mask": torch.tensor([[True, True, False]]),
        "teacher_hiddens": torch.randn(2, 3),
        "avg_micro_batch_token_num": torch.tensor(2.0),
    }
    metrics = algorithm.training_step(batch)
    metrics["loss"].backward()
    assert metrics["loss"].item() == 0.0
    assert metrics["mp_opd_valid_sample_count"].item() == 0.0
    assert metrics["mp_opd_invalid_sample_count"].item() == 1.0
    assert metrics["mp_opd_invalid_reason_normalization_mismatch"].item() == 1.0
    assert student.logits.grad is not None
    assert student.logits.grad.abs().sum().item() == 0.0
    assert all(value.ndim == 0 and torch.isfinite(value) for value in metrics.values())


def test_oracle_mode_is_fail_closed_without_instrumentation_scores():
    args = SimpleNamespace(kd=DistillationArguments(kd_algorithm="mp_opd", mp_opd_mode="oracle"))
    strategy = SimpleNamespace(args=args, ring_attn_group=None)
    algorithm = MetaPartitionedOPD(
        strategy, object(), torch.nn.Linear(1, 2), FakeTokenizer({1: "x"}, None), FakeTokenizer({2: "x"}, None)
    )
    credits = SimpleNamespace(
        current_nll=torch.ones(1, requires_grad=True), base_credit=torch.ones(1), weight=torch.ones(1)
    )
    with pytest.raises(RuntimeError, match="instrumentation-only"):
        algorithm._partition_loss(credits, [SimpleNamespace()], {}, 0)


def test_soft_mode_requires_separate_energy_checkpoint():
    with pytest.raises(ValueError, match="energy_checkpoint"):
        DistillationArguments(kd_algorithm="mp_opd", mp_opd_mode="soft")
