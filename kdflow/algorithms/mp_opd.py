"""Meta-Partitioned Credit Distillation on student on-policy responses.

``mp_opd`` is a KDFlow algorithm mode, not a replacement for distributional
SpanCTKD. It reuses only SimCT's minimal synchronized boundaries and assigns
scalar realized-path credits to contiguous atom partitions.
"""

from __future__ import annotations

import hashlib
import math
import random
import time
from pathlib import Path

import torch

from kdflow.algorithms import register_algorithm
from kdflow.loss.cross_entropy import compute_cross_entropy

from ._mp_opd_atoms import SimCTAtomizer
from ._mp_opd_credit import (
    build_atom_credits,
    credit_conservation_residual,
    expected_atom_rates,
    hard_partition_loss,
    soft_partition_loss,
    span_tables,
)
from ._mp_opd_energy import MPAtomEnergy, load_energy_checkpoint
from ._mp_opd_oracle import hard_max_partition, span_utility_table
from ._mp_opd_semimarkov import semi_markov_partition


def fixed_partition(n: int, length: int) -> tuple[tuple[int, int], ...]:
    if n < 0 or length <= 0:
        raise ValueError("n must be nonnegative and length positive")
    return tuple((start, min(start + length, n)) for start in range(0, n, length))


def random_partition(n: int, max_length: int, seed: int) -> tuple[tuple[int, int], ...]:
    if n < 0 or max_length <= 0:
        raise ValueError("n must be nonnegative and max_length positive")
    generator = random.Random(int(seed))
    parts = []
    cursor = 0
    while cursor < n:
        length = generator.randint(1, min(max_length, n - cursor))
        parts.append((cursor, cursor + length))
        cursor += length
    return tuple(parts)


def atom_features(atoms, credits) -> torch.Tensor:
    rows = []
    for index, atom in enumerate(atoms):
        w = credits.weight[index]
        rows.append(
            torch.stack(
                (
                    credits.rate[index],
                    credits.base_credit[index],
                    w,
                    w.new_tensor(float(atom.teacher_token_count)),
                    w.new_tensor(float(atom.byte_end - atom.byte_start)),
                    credits.teacher_log_score[index] / float(atom.teacher_token_count),
                    credits.student_old_log_score[index] / w,
                    w.new_tensor(float(atom.boundary_type == "one_to_one")),
                    w.new_tensor(float(atom.boundary_type == "multi_token")),
                    w.new_tensor(1.0),
                )
            )
        )
    return torch.stack(rows).detach()


def _finite_stats(prefix: str, values: torch.Tensor) -> dict[str, torch.Tensor]:
    values = values.detach().float()
    return {
        f"{prefix}_mean": values.mean(),
        f"{prefix}_std": values.std(unbiased=False),
        f"{prefix}_min": values.min(),
        f"{prefix}_max": values.max(),
        f"{prefix}_positive_fraction": (values > 0).float().mean(),
        f"{prefix}_negative_fraction": (values < 0).float().mean(),
    }


def _behavior_parity_metrics(
    student_logits: torch.Tensor,
    labels: torch.Tensor,
    behavior_log_probs: torch.Tensor,
    rollout_temperature: float,
) -> dict[str, torch.Tensor]:
    """Compare SGLang decode policy logprobs with the trainer policy.

    SGLang applies sampling temperature to output-token logprobs, while its
    input-token/prefill logprobs remain raw model logprobs. The trajectory
    stores output-token logprobs, so parity must apply the rollout temperature
    to the trainer logits as well.
    """
    temperature = float(rollout_temperature)
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("MP-OPD parity requires a positive finite rollout temperature")

    if torch.isinf(behavior_log_probs).any():
        raise RuntimeError("behavior logprobs contain infinity")
    # Synthetic terminal events are represented by NaN and intentionally have
    # no behavior probability. Infinity is never a valid sentinel.
    real = ~torch.isnan(behavior_log_probs)
    if not real.any():
        return {}
    actual = (
        (student_logits.detach().float() / temperature)
        .log_softmax(dim=-1)
        .gather(-1, labels.unsqueeze(-1))
        .squeeze(-1)
    )
    delta = (actual[real] - behavior_log_probs[real]).abs()
    if not torch.isfinite(delta).all():
        raise RuntimeError("behavior/trainer logprob parity produced non-finite deltas")

    mean = delta.mean()
    maximum = delta.max()
    p99 = torch.quantile(delta, 0.99)
    # Backend/precision tails can contain isolated finite outliers. Fail on a
    # distributional mismatch while preserving the tail as diagnostics.
    if mean > 0.1 or p99 > 0.5:
        raise RuntimeError(
            "behavior/trainer logprob parity failed: "
            f"mean={mean.item():.6f}, p99={p99.item():.6f}, max={maximum.item():.6f}"
        )
    return {
        "trajectory_logprob_abs_mean": mean,
        "trajectory_logprob_abs_p99": p99,
        "trajectory_logprob_abs_max": maximum,
        "trajectory_logprob_above_0p5_fraction": (delta > 0.5).float().mean(),
    }


@register_algorithm("mp_opd")
class MetaPartitionedOPD:
    """Scalar canonical-path credit with contiguous SimCT atom partitions.

    The real student optimizer never owns the energy parameters. ``soft``
    therefore requires an explicit, separately trained energy checkpoint.
    Oracle mode requires detached per-atom directional scores in the batch and
    is intentionally diagnostic-only.
    """

    FEATURE_DIM = 10

    def __init__(
        self,
        strategy,
        student_model,
        teacher_lm_head,
        student_tokenizer,
        teacher_tokenizer,
        **kwargs,
    ):
        self.strategy = strategy
        self.args = strategy.args
        self.student = student_model
        self.teacher_lm_head = teacher_lm_head
        self.student_tokenizer = student_tokenizer
        self.teacher_tokenizer = teacher_tokenizer
        self.atomizer = SimCTAtomizer(student_tokenizer, teacher_tokenizer)
        self.mode = self.args.kd.mp_opd_mode
        self.max_span_length = int(self.args.kd.mp_opd_max_span_length)
        self.fixed_span_length = int(self.args.kd.mp_opd_fixed_span_length)
        self.temperature = float(self.args.kd.mp_opd_partition_temperature)
        self.random_seed = int(self.args.kd.mp_opd_random_seed)
        self.energy = None
        self.energy_optimizer = None
        if self.mode == "soft":
            checkpoint = self.args.kd.mp_opd_energy_checkpoint
            if not checkpoint:
                raise ValueError("mp_opd soft mode requires mp_opd_energy_checkpoint")
            self.energy = MPAtomEnergy(
                self.FEATURE_DIM,
                hidden_dim=int(self.args.kd.mp_opd_energy_hidden_dim),
                layers=int(self.args.kd.mp_opd_energy_layers),
            ).to(self.teacher_lm_head.weight.device)
            self.energy_optimizer = torch.optim.AdamW(
                self.energy.parameters(), lr=float(self.args.kd.mp_opd_energy_lr), weight_decay=0.0
            )
            load_energy_checkpoint(
                checkpoint,
                self.energy,
                self.energy_optimizer,
                expected_extra_config={"max_span_length": self.max_span_length},
            )
            self.energy.eval()

    def get_energy_params(self):
        """Explicitly separate from ``get_projector_params``/student optimizer."""
        return [] if self.energy is None else list(self.energy.parameters())

    def _partition_loss(self, credits, atoms, micro_batch, sample_index: int):
        n = len(atoms)
        metrics = {}
        if self.mode == "atomic":
            partition = fixed_partition(n, 1)
            return hard_partition_loss(
                credits.current_nll, credits.base_credit, credits.weight, partition
            ), metrics
        if self.mode == "fixed":
            partition = fixed_partition(n, self.fixed_span_length)
            return hard_partition_loss(
                credits.current_nll, credits.base_credit, credits.weight, partition
            ), metrics
        if self.mode == "random":
            material = (
                f"{self.random_seed}:{sample_index}:"
                f"{credits.weight.detach().cpu().tolist()}"
            ).encode()
            seed = int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
            partition = random_partition(n, self.max_span_length, seed)
            return hard_partition_loss(
                credits.current_nll, credits.base_credit, credits.weight, partition
            ), metrics

        _b, _w, rates, valid = span_tables(
            credits.base_credit, credits.weight, self.max_span_length
        )
        if self.mode == "oracle":
            supplied = micro_batch.get("mp_opd_atom_directional_scores")
            if supplied is None:
                raise RuntimeError(
                    "mp_opd oracle is instrumentation-only and requires detached "
                    "mp_opd_atom_directional_scores"
                )
            z = supplied[sample_index].to(credits.base_credit).detach()
            if z.numel() != n:
                raise ValueError("oracle directional-score cardinality mismatch")
            utilities, utility_valid = span_utility_table(
                credits.base_credit, credits.weight, z, self.max_span_length
            )
            oracle = hard_max_partition(utilities, utility_valid)
            metrics["mp_opd_delta_pred"] = oracle.score.detach()
            return hard_partition_loss(
                credits.current_nll, credits.base_credit, credits.weight, oracle.partition
            ), metrics

        if self.mode == "soft":
            features = atom_features(atoms, credits)
            energies = self.energy(features, self.max_span_length)
            distribution = semi_markov_partition(
                energies, temperature=self.temperature, valid_mask=valid
            )
            if not torch.isfinite(distribution.coverage_max_error) or distribution.coverage_max_error > 1e-6:
                raise RuntimeError("MP-OPD partition coverage failed after FP64 DP; inspect before retry")
            rates_per_atom = expected_atom_rates(distribution.marginals, rates)
            metrics.update(
                {
                    "mp_opd_log_z": distribution.log_z.detach(),
                    "mp_opd_partition_entropy": distribution.entropy.detach(),
                    "mp_opd_expected_span_count": distribution.expected_span_count.detach(),
                    "mp_opd_expected_span_length": distribution.expected_span_length.detach(),
                    "mp_opd_marginal_coverage_max_error": distribution.coverage_max_error.detach(),
                    "mp_opd_credit_conservation_residual": credit_conservation_residual(
                        credits.base_credit, credits.weight, rates_per_atom
                    ).detach(),
                }
            )
            # Student path treats q_phi as fixed; phi has its own optimizer.
            return soft_partition_loss(credits.current_nll, rates_per_atom.detach()), metrics
        raise AssertionError(f"unhandled MP-OPD mode {self.mode}")

    def training_step(self, micro_batch):
        started = time.perf_counter()
        student_input_ids = micro_batch["stu_input_ids"]
        student_attn_mask = micro_batch["stu_attn_mask"]
        student_loss_mask = micro_batch["stu_loss_mask"].bool()
        teacher_input_ids = micro_batch["tea_input_ids"]
        teacher_loss_mask = micro_batch["tea_loss_mask"].bool()
        teacher_hiddens = micro_batch.get("teacher_hiddens")
        avg_token_num = micro_batch["avg_micro_batch_token_num"]
        if teacher_hiddens is None:
            raise RuntimeError("micro_batch must contain teacher_hiddens for MP-OPD")

        mm_kwargs = {key[3:]: value for key, value in micro_batch.items() if key.startswith("mm_")}
        output = self.student(
            student_input_ids,
            attention_mask=student_attn_mask,
            allgather_logits=True,
            ring_attn_group=self.strategy.ring_attn_group,
            **mm_kwargs,
        )
        student_logits_flat = output["logits"][student_loss_mask]
        student_labels = student_input_ids.roll(shifts=-1, dims=1)
        teacher_labels = teacher_input_ids.roll(shifts=-1, dims=1)
        parity_metrics = {}
        behavior = micro_batch.get("stu_behavior_log_probs")
        if behavior is not None:
            selected = behavior[student_loss_mask]
            labels_for_parity = student_labels[student_loss_mask]
            parity_metrics = _behavior_parity_metrics(
                student_logits_flat,
                labels_for_parity,
                selected,
                self.args.rollout.temperature,
            )
        teacher_logits_flat = self.teacher_lm_head(
            teacher_hiddens.to(self.teacher_lm_head.weight)
        )

        # Keep the zero baseline connected to the student graph. A fail-closed
        # sample can occupy an entire microbatch (the production recipe uses
        # micro_train_batch_size=1); it must contribute zero gradient and
        # explicit invalid-sample telemetry without aborting the surrounding
        # gradient-accumulation window.
        total_loss = student_logits_flat.sum() * 0.0
        total_atoms = total_invalid = total_one = total_multi = 0
        invalid_reasons: dict[str, int] = {}
        covered_student = covered_teacher = masked_eos = 0
        candidate_spans = 0
        byte_lengths = []
        student_lengths = []
        teacher_lengths = []
        credit_values = []
        rate_values = []
        extra_sums: dict[str, torch.Tensor] = {}
        stu_offset = tea_offset = 0
        valid_samples = 0
        for batch_index in range(student_input_ids.shape[0]):
            stu_mask = student_loss_mask[batch_index]
            tea_mask = teacher_loss_mask[batch_index]
            stu_count = int(stu_mask.sum().item())
            tea_count = int(tea_mask.sum().item())
            stu_logits = student_logits_flat[stu_offset : stu_offset + stu_count]
            tea_logits = teacher_logits_flat[tea_offset : tea_offset + tea_count]
            stu_offset += stu_count
            tea_offset += tea_count
            stu_ids = student_labels[batch_index][stu_mask].detach().cpu().tolist()
            tea_ids = teacher_labels[batch_index][tea_mask].detach().cpu().tolist()
            sample_key = hashlib.sha256(
                bytes(str(stu_ids), "utf-8") + b"|" + bytes(str(tea_ids), "utf-8")
            ).hexdigest()[:16]
            atomized = self.atomizer.atomize(stu_ids, tea_ids, sample_id=sample_key)
            masked_eos += atomized.masked_student_eos + atomized.masked_teacher_eos
            if not atomized.valid:
                total_invalid += 1
                reason = atomized.failure_reason or "unknown"
                invalid_reasons[reason] = invalid_reasons.get(reason, 0) + 1
                continue
            atoms = atomized.atoms
            credits = build_atom_credits(
                atoms,
                stu_logits,
                torch.tensor(stu_ids, device=stu_logits.device),
                tea_logits,
                torch.tensor(tea_ids, device=tea_logits.device),
            )
            sample_loss, sample_metrics = self._partition_loss(
                credits, atoms, micro_batch, batch_index
            )
            total_loss = total_loss + sample_loss
            for key, value in sample_metrics.items():
                extra_sums[key] = extra_sums.get(key, value.new_zeros(())) + value
            valid_samples += 1
            total_atoms += len(atoms)
            total_one += sum(atom.boundary_type == "one_to_one" for atom in atoms)
            total_multi += sum(atom.boundary_type == "multi_token" for atom in atoms)
            covered_student += atomized.covered_student_events
            covered_teacher += atomized.covered_teacher_events
            candidate_spans += sum(min(self.max_span_length, len(atoms) - i) for i in range(len(atoms)))
            byte_lengths.extend(atom.byte_end - atom.byte_start for atom in atoms)
            student_lengths.extend(atom.student_token_count for atom in atoms)
            teacher_lengths.extend(atom.teacher_token_count for atom in atoms)
            credit_values.append(credits.base_credit)
            rate_values.append(credits.rate)

        kd_loss = total_loss / avg_token_num
        metrics = {
            "loss": kd_loss,
            "kd_loss": kd_loss,
            "mp_opd_valid_atom_count": kd_loss.new_tensor(float(total_atoms)),
            "mp_opd_invalid_sample_count": kd_loss.new_tensor(float(total_invalid)),
            "mp_opd_valid_sample_count": kd_loss.new_tensor(float(valid_samples)),
            "mp_opd_one_to_one_atom_ratio": kd_loss.new_tensor(total_one / max(total_atoms, 1)),
            "mp_opd_multi_token_atom_ratio": kd_loss.new_tensor(total_multi / max(total_atoms, 1)),
            "mp_opd_candidate_span_count": kd_loss.new_tensor(float(candidate_spans)),
            "mp_opd_masked_eos_count": kd_loss.new_tensor(float(masked_eos)),
            "valid_student_tokens": kd_loss.new_tensor(float(covered_student)),
            "valid_teacher_tokens": kd_loss.new_tensor(float(covered_teacher)),
            "mp_opd_fail_closed_sample_ratio": kd_loss.new_tensor(
                total_invalid / max(total_invalid + valid_samples, 1)
            ),
            "mp_opd_atomization_and_loss_seconds": kd_loss.new_tensor(time.perf_counter() - started),
        }
        metrics.update(parity_metrics)
        if credit_values:
            metrics.update(_finite_stats("mp_opd_b", torch.cat(credit_values)))
            metrics.update(_finite_stats("mp_opd_r", torch.cat(rate_values)))
        metrics.update(
            {
                "mp_opd_atom_byte_length_mean": kd_loss.new_tensor(sum(byte_lengths) / max(len(byte_lengths), 1)),
                "mp_opd_student_atom_token_length_mean": kd_loss.new_tensor(sum(student_lengths) / max(len(student_lengths), 1)),
                "mp_opd_teacher_atom_token_length_mean": kd_loss.new_tensor(sum(teacher_lengths) / max(len(teacher_lengths), 1)),
            }
        )
        for reason, count in sorted(invalid_reasons.items()):
            metrics[f"mp_opd_invalid_reason_{reason}"] = kd_loss.new_tensor(float(count))
        for key, value in extra_sums.items():
            metrics[key] = value / max(valid_samples, 1)
        if self.args.kd.kd_ratio < 1:
            ce_labels = student_labels[student_loss_mask]
            ce_loss = compute_cross_entropy(student_logits_flat, ce_labels, reduction="sum") / avg_token_num
            metrics["ce_loss"] = ce_loss
            metrics["loss"] = (1 - self.args.kd.kd_ratio) * ce_loss + self.args.kd.kd_ratio * kd_loss
        for key, value in metrics.items():
            if not torch.isfinite(value).all():
                raise FloatingPointError(f"non-finite MP-OPD metric: {key}")
        return metrics
