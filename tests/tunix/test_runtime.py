from __future__ import annotations

import pytest

from vdt_tunix.runtime import TPUPreflightError, require_tpu_v5e8


class FakeDevice:
    def __init__(self, index, kind="TPU v5 lite"):
        self.index = index
        self.device_kind = kind

    def __str__(self):
        return f"TPU:{self.index}"


class FakeJax:
    __version__ = "test"

    def __init__(self, *, backend="tpu", count=8, kind="TPU v5 lite"):
        self._backend = backend
        self._devices = [FakeDevice(index, kind) for index in range(count)]

    def devices(self):
        return self._devices

    def default_backend(self):
        return self._backend


def test_v5e8_preflight_accepts_exact_runtime():
    devices, evidence = require_tpu_v5e8(jax_module=FakeJax())
    assert len(devices) == 8
    assert evidence["backend"] == "tpu"
    assert evidence["v5e_kind_match"] is True


@pytest.mark.parametrize(
    "jax_module",
    [
        FakeJax(backend="cpu"),
        FakeJax(count=4),
        FakeJax(kind="TPU v4"),
    ],
)
def test_v5e8_preflight_rejects_wrong_runtime(jax_module):
    with pytest.raises(TPUPreflightError, match="expected exactly"):
        require_tpu_v5e8(jax_module=jax_module)
