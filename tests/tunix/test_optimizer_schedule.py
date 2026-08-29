from __future__ import annotations

import copy

from vdt_tunix.config import RunConfig
from vdt_tunix.trainer import _build_optimizer


class _FakeOptax:
    def __init__(self) -> None:
        self.warmup_kwargs = None
        self.multi_steps = None

    def warmup_cosine_decay_schedule(self, **kwargs):
        self.warmup_kwargs = kwargs
        return ("warmup_cosine", kwargs)

    def cosine_decay_schedule(self, **kwargs):
        raise AssertionError("paper-horizon canary must retain warmup")

    def adamw(self, **kwargs):
        return ("adamw", kwargs)

    def MultiSteps(self, transform, *, every_k_schedule):
        self.multi_steps = every_k_schedule
        return ("multi_steps", transform, every_k_schedule)


class _FakeNnx:
    Param = object()

    @staticmethod
    def Optimizer(model, transform, *, wrt):
        return (model, transform, wrt)


def test_short_canary_uses_full_paper_lr_horizon(config_payload):
    payload = copy.deepcopy(config_payload)
    payload["training"].update(
        {
            "max_steps": 10,
            "max_steps_unit": "optimizer_update",
            "gradient_accumulation_steps": 32,
            "learning_rate": 1e-6,
            "lr_schedule_optimizer_steps": 314,
        }
    )
    config = RunConfig.from_mapping(payload)
    optax = _FakeOptax()

    _build_optimizer(config, object(), _FakeNnx, optax)

    assert optax.warmup_kwargs == {
        "init_value": 0.0,
        "peak_value": 1e-6,
        "warmup_steps": 15,
        "decay_steps": 314,
        "end_value": 0.0,
    }
    assert optax.multi_steps == 32
