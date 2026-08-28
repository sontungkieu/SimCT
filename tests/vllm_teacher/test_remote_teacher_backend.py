from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from vdt_tunix.config import load_config
from vdt_tunix.contracts import (
    INTERFACE_CONTRACT_VERSION,
    PromptRecord,
    StudentRolloutBatch,
    StudentRolloutSample,
    TeacherScoreRequest,
    TokenSequence,
)
from vdt_tunix.model_adapters import TokenizerByteAdapter
from vdt_tunix.remote_teacher import RemoteTeacherProfile, TeacherHiddenStats
from vdt_tunix.remote_teacher_backend import RemoteVLLMTeacherScoreBackend


def _bfloat16_bits(values):
    array = np.asarray(values, dtype=np.float32)
    return (array.view(np.uint32) >> 16).astype(np.uint16)


class _FakeLax:
    @staticmethod
    def bitcast_convert_type(values, dtype):
        del dtype
        bits = np.asarray(values, dtype=np.uint16).astype(np.uint32)
        return (bits << 16).view(np.float32)

    @staticmethod
    def stop_gradient(values):
        return values


class _FakeJax:
    lax = _FakeLax()

    @staticmethod
    def jit(function):
        return function

    @staticmethod
    def block_until_ready(values):
        return values


class _FakeJnp:
    uint16 = np.uint16
    bfloat16 = object()
    float32 = np.float32
    asarray = staticmethod(np.asarray)
    einsum = staticmethod(np.einsum)


class _CharTokenizer:
    pad_token_id = 0
    bos_token_id = 1
    eos_token_id = 2
    all_special_ids = (0, 1, 2)

    @staticmethod
    def encode(text, add_special_tokens=False):
        del add_special_tokens
        return [ord(character) + 10 for character in text]

    @staticmethod
    def decode(token_ids, **kwargs):
        del kwargs
        return "".join(chr(value - 10) for value in token_ids if value > 2)


class _FakeClient:
    def __init__(self):
        self.calls = []

    @staticmethod
    def health():
        return {"ok": True}

    def score_tokens(self, prompt_token_ids, completion_token_ids):
        self.calls.append((tuple(prompt_token_ids), tuple(completion_token_ids)))
        rows = len(completion_token_ids)
        return TeacherHiddenStats(
            hidden_state_bits=_bfloat16_bits([[1.0, 2.0]] * rows),
            log_normalizer=np.asarray([1.0] * rows, dtype=np.float32),
            selected_log_probs=np.asarray([-0.5] * rows, dtype=np.float32),
            selected_token_ids=np.asarray(completion_token_ids, dtype=np.int32),
            header={},
        )


def test_remote_backend_reconstructs_paper_statistics_without_teacher_model():
    config = load_config(
        "configs/reproduction/qwen25_7b_to_gemma2_2b_paper4k_simct.json"
    )
    tokenizer = TokenizerByteAdapter(_CharTokenizer(), config.teacher)
    profile = RemoteTeacherProfile(
        profile_id="test-profile",
        model_id=config.teacher.tokenizer_id,
        model_revision=config.teacher.model_revision,
        teacher_ids_sha256="1" * 64,
        overlap_head_sha256="2" * 64,
        teacher_ids=np.asarray([7, 8], dtype=np.int32),
        overlap_head_bits=_bfloat16_bits([[3.0, 4.0], [5.0, 6.0]]),
        hidden_size=2,
    )
    client = _FakeClient()
    backend = RemoteVLLMTeacherScoreBackend(
        config,
        tokenizer,
        SimpleNamespace(max_parallel_requests=1),
        profile=profile,
        client=client,
        jax_module=_FakeJax(),
        jnp_module=_FakeJnp(),
    )
    backend.configure_overlap_token_ids([7, 8])
    completion = TokenSequence(text="C", token_ids=(99,), pieces=(b"C",))
    rollouts = StudentRolloutBatch(
        contract_version=INTERFACE_CONTRACT_VERSION,
        run_id="remote-teacher-test",
        step=0,
        model_id=config.student.model_id,
        model_revision=config.student.model_revision,
        tokenizer_id=config.student.tokenizer_id,
        tokenizer_revision=config.student.tokenizer_revision,
        samples=(
            StudentRolloutSample(
                sample_id="sample-0",
                prompt_id="prompt-0",
                student_prompt_token_ids=(10,),
                completion=completion,
                rollout_log_probs=(-1.0,),
            ),
        ),
    )
    prompt = PromptRecord(
        prompt_id="prompt-0",
        student_prompt="student prompt",
        teacher_prompt="P",
    )

    result = backend.score(TeacherScoreRequest(rollouts, (prompt,)))

    assert backend.model_adapter.dependencies.production is True
    assert client.calls == [((1, ord("P") + 10), (ord("C") + 10,))]
    stats = result.samples[0].sufficient_statistics
    assert stats is not None
    np.testing.assert_allclose(stats.shared_log_probs, [[10.0, 16.0]])
    np.testing.assert_allclose(stats.selected_log_probs, [-0.5])
    assert backend.last_phase_timings["teacher_remote_profile_id"] == "test-profile"
