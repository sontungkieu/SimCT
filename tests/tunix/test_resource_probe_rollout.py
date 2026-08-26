from __future__ import annotations

import contextlib
import copy
import sys
from types import ModuleType, SimpleNamespace

from vdt_tunix.config import RunConfig
from vdt_tunix.contracts import PromptRecord, RolloutRequest, TokenSequence
from vdt_tunix.real_backend import TunixStudentRolloutBackend, _LoadedTunixModel


class _Values(list):
    def tolist(self):
        return list(self)


def test_forced_resource_probe_uses_native_full_length_sampler(
    config_payload, monkeypatch
):
    payload = copy.deepcopy(config_payload)
    payload["student"].update(
        {
            "model_source": "huggingface",
            "model_path": "/tmp/student",
            "tokenizer_type": "huggingface",
            "tokenizer_path": "/tmp/student",
        }
    )
    payload["rollout"].update(
        {
            "samples_per_prompt": 1,
            "max_prompt_tokens": 7,
            "max_completion_tokens": 6,
            "max_sequence_tokens": 8,
            "force_max_completion": True,
            "temperature": 0.6,
            "top_p": 0.95,
        }
    )
    config = RunConfig.from_mapping(payload)
    observed = {}

    class CacheConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class Sampler:
        def __init__(self, **kwargs):
            observed["constructor"] = kwargs

        def __call__(self, prompts, **kwargs):
            observed["prompts"] = prompts
            observed["call"] = kwargs
            width = kwargs["max_generation_steps"]
            return SimpleNamespace(
                tokens=[_Values([3] * width)],
                logprobs=[_Values([-1.0] * width)],
            )

    nnx = ModuleType("flax.nnx")
    nnx.state = lambda model: model
    flax = ModuleType("flax")
    flax.nnx = nnx
    sampler_module = ModuleType("tunix.generate.sampler")
    sampler_module.CacheConfig = CacheConfig
    sampler_module.Sampler = Sampler
    generate = ModuleType("tunix.generate")
    generate.sampler = sampler_module
    tunix = ModuleType("tunix")
    tunix.generate = generate
    jax = ModuleType("jax")
    jax.set_mesh = lambda mesh: contextlib.nullcontext(mesh)
    for name, module in {
        "flax": flax,
        "flax.nnx": nnx,
        "tunix": tunix,
        "tunix.generate": generate,
        "tunix.generate.sampler": sampler_module,
        "jax": jax,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    class Tokenizer:
        raw_tokenizer = object()
        pad_token_id = 0
        special_token_ids = frozenset({0, 1, 2})

        @staticmethod
        def continuation_from_generated_ids(**kwargs):
            ids = tuple(kwargs["completion_token_ids"])
            return TokenSequence(
                text="x" * len(ids),
                token_ids=ids,
                pieces=tuple(b"x" for _ in ids),
            )

    loaded = _LoadedTunixModel(
        model=object(),
        mesh=object(),
        forward_fn=object(),
        model_config=SimpleNamespace(num_layers=2, num_kv_heads=2, head_dim=4),
    )
    adapter = SimpleNamespace(
        config=config.student,
        dependencies=SimpleNamespace(production=True),
        tokenizer=Tokenizer(),
        require_loaded_model=lambda: loaded,
    )
    backend = TunixStudentRolloutBackend(config, adapter)
    prompt = PromptRecord(prompt_id="p", student_prompt="p", teacher_prompt="p")
    request = RolloutRequest(
        run_id=config.run_id,
        step=0,
        prompts=(prompt,),
        samples_per_prompt=1,
    )
    result = backend._tunix_sampler_rollout(
        request,
        [(prompt, 0, "p/0", (9,), (7, 8, 9))],
    )

    assert observed["constructor"]["cache_config"].kwargs["cache_size"] == 9
    # Tunix pads every prompt to max_prompt_length before adding generation
    # steps, so the static budget is 7 prompt tokens + 1 generated token.
    assert observed["call"]["max_generation_steps"] == 1
    assert observed["call"]["eos_tokens"] == [0]
    assert observed["call"]["forbidden_tokens"] == [0, 1, 2]
    assert observed["call"]["top_p"] == 0.95
    assert len(result.samples[0].completion.token_ids) == 1
