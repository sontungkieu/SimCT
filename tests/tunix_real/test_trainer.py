from __future__ import annotations

from types import SimpleNamespace

import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
nnx = pytest.importorskip("flax.nnx")

from vdt_tunix.config import RunConfig
from vdt_tunix.contracts import (
    BackendBundle,
    INTERFACE_CONTRACT_VERSION,
    LogitsPayload,
    PromptRecord,
    StudentRolloutBatch,
    StudentRolloutSample,
    TeacherScoreBatch,
    TeacherScoreSample,
    TokenSequence,
)
from vdt_tunix.model_adapters import TokenizerByteAdapter
from vdt_tunix.real_backend import _LoadedTunixModel
from vdt_tunix.trainer import PaperSimCTTrainer


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
        self.table = nnx.Param(
            jnp.asarray(
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, -0.5, 0.2, 1.0, -0.2, -0.4],
                    [0.0, -0.5, 0.2, -0.3, 1.0, -0.4],
                    [0.0, -0.5, 0.2, -0.3, -0.4, 1.0],
                    [0.0, 0.5, 0.2, -0.3, -0.4, -0.5],
                ],
                dtype=jnp.float32,
            )
        )

    def __call__(self, input_ids, *args, **kwargs):
        del args, kwargs
        return self.table.get_value()[input_ids], None


class FixedStudent:
    backend_name = "toy-real-student"
    real_model_integration = True

    def __init__(self, config, adapter):
        self.config = config
        self.model_adapter = adapter

    def rollout(self, request):
        sample = StudentRolloutSample(
            sample_id="p/0",
            prompt_id="p",
            student_prompt_token_ids=(2,),
            completion=TokenSequence(
                text="happy",
                token_ids=(3, 4, 5),
                pieces=(b"ha", b"pp", b"y"),
            ),
            rollout_log_probs=(-1.0, -1.0, -1.0),
        )
        model = self.config.student
        return StudentRolloutBatch(
            contract_version=INTERFACE_CONTRACT_VERSION,
            run_id=request.run_id,
            step=request.step,
            model_id=model.model_id,
            model_revision=model.model_revision,
            tokenizer_id=model.tokenizer_id,
            tokenizer_revision=model.tokenizer_revision,
            samples=(sample,),
        )


class FixedTeacher:
    backend_name = "toy-real-teacher"
    real_model_integration = True

    def __init__(self, config, adapter):
        self.config = config
        self.model_adapter = adapter

    def score(self, request):
        sample = TeacherScoreSample(
            sample_id="p/0",
            prompt_id="p",
            teacher_prompt_token_ids=(2,),
            completion=TokenSequence(
                text="happy",
                token_ids=(3, 4),
                pieces=(b"hap", b"py"),
            ),
            position_logits=LogitsPayload(
                values=jnp.asarray(
                    [[-1.0, -0.5, 0.0, 2.0, -1.0], [1.0, -0.5, 0.0, -1.0, 2.0]]
                ),
                shape=(2, 5),
                dtype="float32",
            ),
        )
        model = self.config.teacher
        return TeacherScoreBatch(
            contract_version=INTERFACE_CONTRACT_VERSION,
            run_id=request.rollouts.run_id,
            step=request.rollouts.step,
            model_id=model.model_id,
            model_revision=model.model_revision,
            tokenizer_id=model.tokenizer_id,
            tokenizer_revision=model.tokenizer_revision,
            samples=(sample,),
        )


def _config():
    model = {
        "model_revision": "immutable-model-revision",
        "tokenizer_revision": "immutable-tokenizer-revision",
        "maxtext_checkpoint_uri": "/tmp/unused",
    }
    return RunConfig.from_mapping(
        {
            "contract_version": 1,
            "run_id": "toy-paper-update",
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
            "checkpoint": {"root": "/tmp/toy", "save_every_steps": 1},
            "canary": {
                "prompt_id": "p",
                "student_prompt": "P:",
                "teacher_prompt": "P:",
            },
        }
    )


def test_paper_trainer_executes_real_gradient_update():
    config = _config()
    student_tokenizer = TokenizerByteAdapter(
        PieceTokenizer({0: "", 1: "", 2: "P:", 3: "ha", 4: "pp", 5: "y"}),
        config.student,
    )
    teacher_tokenizer = TokenizerByteAdapter(
        PieceTokenizer({0: "", 1: "", 2: "P:", 3: "hap", 4: "py"}),
        config.teacher,
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
    student_adapter = SimpleNamespace(
        tokenizer=student_tokenizer,
        require_loaded_model=lambda: loaded,
    )
    teacher_adapter = SimpleNamespace(tokenizer=teacher_tokenizer)
    backends = BackendBundle(
        student=FixedStudent(config, student_adapter),
        teacher=FixedTeacher(config, teacher_adapter),
    )
    before = jnp.array(model.table.get_value())
    trainer = PaperSimCTTrainer(config, backends)
    metrics = trainer.step(
        (PromptRecord(prompt_id="p", student_prompt="P:", teacher_prompt="P:"),),
        step=0,
    )
    after = model.table.get_value()
    assert metrics.loss > 0.0
    assert metrics.gradient_norm > 0.0
    assert metrics.aligned_spans == 1
    assert not bool(jnp.allclose(before, after))
