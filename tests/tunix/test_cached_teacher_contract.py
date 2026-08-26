from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from vdt_tunix.real_backend import (
    RealBackendUnavailable,
    _qwen_cached_teacher_statistics,
)


class _StopAfterCacheInitialization(RuntimeError):
    pass


class _RecordingSampler:
    def __init__(self) -> None:
        self.cache_dtype = None

    def _init_cache(self, **kwargs):
        self.cache_dtype = kwargs["dtype"]
        return {}


class _StopBeforePrefill:
    @staticmethod
    def build_positions_from_mask(mask):
        del mask
        raise _StopAfterCacheInitialization


def test_qwen_cached_teacher_uses_model_compute_dtype_for_kv_cache():
    sampler = _RecordingSampler()
    model_params = SimpleNamespace(
        num_layers=2,
        num_kv_heads=1,
        head_dim=4,
        dtype=np.float32,
    )

    with pytest.raises(_StopAfterCacheInitialization):
        _qwen_cached_teacher_statistics(
            object(),
            np.asarray([[1, 2]], dtype=np.int32),
            np.asarray([[True, True]]),
            np.asarray([[3]], dtype=np.int32),
            np.asarray([[True]]),
            np.asarray([1, 3], dtype=np.int32),
            model_params=model_params,
            configured_cache_size=3,
            generate_sampler=sampler,
            generate_utils=_StopBeforePrefill,
            jax_module=object(),
            jnp_module=np,
        )

    assert sampler.cache_dtype is np.float32


def test_qwen_cached_teacher_requires_model_compute_dtype():
    model_params = SimpleNamespace(num_layers=2, num_kv_heads=1, head_dim=4)

    with pytest.raises(RealBackendUnavailable, match="config field dtype"):
        _qwen_cached_teacher_statistics(
            object(),
            np.asarray([[1]], dtype=np.int32),
            np.asarray([[True]]),
            np.asarray([[2]], dtype=np.int32),
            np.asarray([[True]]),
            np.asarray([1], dtype=np.int32),
            model_params=model_params,
            configured_cache_size=2,
            generate_sampler=object(),
            generate_utils=object(),
            jax_module=object(),
            jnp_module=np,
        )
