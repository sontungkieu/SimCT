from __future__ import annotations

from types import SimpleNamespace

import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
nnx = pytest.importorskip("flax.nnx")

from vdt_tunix.config import RunConfig
from vdt_tunix.contracts import BackendBundle
from vdt_tunix.model_adapters import TokenizerByteAdapter
from vdt_tunix.real_backend import _LoadedTunixModel
from vdt_tunix.sft_trainer import TunixSFTTrainer, prepare_sft_batch
from vdt_tunix.training_data import SFTRecord


class PieceTokenizer:
    pad_token_id = 0
    eos_token_id = 1
    all_special_ids = [0, 1]

    def __init__(self, pieces):
        self._pieces = dict(pieces)

    def get_vocab(self):
        return {piece: token_id for token_id, piece in self._pieces.items()}

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        result = []
        cursor = 0
        choices = sorted(self.get_vocab(), key=len, reverse=True)
        while cursor < len(text):
            piece = next(item for item in choices if text.startswith(item, cursor))
            result.append(self.get_vocab()[piece])
            cursor += len(piece)
        return result

    def decode(self, ids, **kwargs):
        del kwargs
        return "".join(self._pieces.get(int(token_id), "") for token_id in ids)


class ToyLM(nnx.Module):
    def __init__(self):
        self.table = nnx.Param(jnp.zeros((6, 6), dtype=jnp.float32))

    def __call__(self, input_ids, *args, **kwargs):
        del args, kwargs
        return self.table.get_value()[input_ids], None


def _config():
    model = {
        "model_revision": "model-v1",
        "tokenizer_revision": "tokenizer-v1",
        "maxtext_checkpoint_uri": "/tmp/unused",
    }
    return RunConfig.from_mapping(
        {
            "contract_version": 1,
            "run_id": "sft-toy",
            "student": {
                **model,
                "model_id": "toy/student",
                "tokenizer_id": "toy/student-tokenizer",
            },
            "teacher": {
                **model,
                "model_id": "toy/teacher",
                "tokenizer_id": "toy/teacher-tokenizer",
            },
            "simct": {
                "algorithm": "simct",
                "divergence": "reverse_kl",
                "alignment_unit": "utf8_bytes",
                "virtual_support": "shared_tokens_plus_realized_spans",
                "temperature": 1.0,
                "span_gh_mask_threshold": 0.0,
                "reproduction_mode": "paper_math",
            },
            "rollout": {
                "prompt_batch_size": 1,
                "samples_per_prompt": 1,
                "max_prompt_tokens": 2,
                "max_completion_tokens": 3,
                "temperature": 0.0,
                "top_p": 1.0,
            },
            "training": {
                "max_steps": 2,
                "micro_batch_size": 1,
                "gradient_accumulation_steps": 1,
                "learning_rate": 0.05,
            },
            "tpu": {
                "accelerator_type": "v5e-8",
                "expected_device_count": 8,
                "tensor_parallelism": 1,
                "pipeline_parallelism": 1,
                "fsdp_parallelism": 8,
            },
            "checkpoint": {"root": "/tmp/sft-toy", "save_every_steps": 1},
            "canary": {
                "prompt_id": "p",
                "student_prompt": "P:",
                "teacher_prompt": "P:",
            },
        }
    )


def test_sft_trainer_updates_native_student_state():
    config = _config()
    tokenizer = TokenizerByteAdapter(
        PieceTokenizer({0: "", 1: "", 2: "P:", 3: "ha", 4: "pp", 5: "y"}),
        config.student,
    )
    model = ToyLM()
    mesh = jax.make_mesh((1, 1), ("fsdp", "tp"))

    @nnx.jit
    def forward_fn(module, input_ids, positions, segments):
        del positions, segments
        logits, _ = module(input_ids)
        return logits

    loaded = _LoadedTunixModel(
        model=model,
        mesh=mesh,
        forward_fn=forward_fn,
        model_config=SimpleNamespace(num_layers=1, num_kv_heads=1, head_dim=1),
    )
    student = SimpleNamespace(
        backend_name="toy-real-student",
        real_model_integration=True,
        rollout=lambda request: None,
        model_adapter=SimpleNamespace(
            tokenizer=tokenizer,
            require_loaded_model=lambda: loaded,
        ),
    )
    teacher = SimpleNamespace(
        backend_name="unused-teacher",
        real_model_integration=False,
        score=lambda request: None,
    )
    backends = BackendBundle(student=student, teacher=teacher)
    row = SFTRecord(
        prompt_id="p",
        student_prompt="P:",
        teacher_prompt="P:",
        target_response="happy",
        source="fixture",
        source_id="p",
        source_license="MIT",
    )
    before = jnp.array(model.table.get_value())
    metrics = TunixSFTTrainer(config, backends).step((row,), step=0)
    assert metrics.loss > 0.0
    assert metrics.gradient_norm > 0.0
    assert metrics.target_tokens == 4
    assert not bool(jnp.allclose(before, model.table.get_value()))


def test_sft_batch_uses_bos_conditioning_but_keeps_text_boundary():
    config = _config()

    class BosPieceTokenizer(PieceTokenizer):
        bos_token_id = 7
        all_special_ids = [0, 1, 7]

    tokenizer = TokenizerByteAdapter(
        BosPieceTokenizer(
            {0: "", 1: "", 2: "P:", 3: "ha", 4: "pp", 5: "y"}
        ),
        config.student,
    )
    student = SimpleNamespace(
        model_adapter=SimpleNamespace(tokenizer=tokenizer),
    )
    backends = BackendBundle(
        student=student,
        teacher=SimpleNamespace(),
    )
    row = SFTRecord(
        prompt_id="p",
        student_prompt="P:",
        teacher_prompt="P:",
        target_response="happy",
        source="fixture",
        source_id="p",
        source_license="MIT",
    )

    batch = prepare_sft_batch(config, (row,), backends)

    assert batch.input_ids[0, :5].tolist() == [7, 2, 3, 4, 5]
    assert batch.label_positions[0, :4].tolist() == [1, 2, 3, 4]
    assert batch.label_ids[0, :4].tolist() == [3, 4, 5, 1]
