from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from vdt_tunix.real_backend import (
    RealBackendUnavailable,
    _configure_qwen_compute_dtype,
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


def test_qwen_compute_dtype_is_explicitly_configured_before_restore():
    model_params = SimpleNamespace(dtype=np.float32)

    configured = _configure_qwen_compute_dtype(
        model_params,
        compute_dtype=np.float16,
    )

    assert configured is model_params
    assert model_params.dtype is np.float16


def test_qwen_compute_dtype_configuration_fails_closed_without_contract():
    with pytest.raises(RealBackendUnavailable, match="compute dtype"):
        _configure_qwen_compute_dtype(
            SimpleNamespace(),
            compute_dtype=np.float16,
        )


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


def test_qwen_cached_teacher_keeps_lm_head_inside_scalar_indexed_barriered_steps():
    class NoDynamicGatherNumpy:
        def __getattr__(self, name):
            if name == "take_along_axis":
                raise AssertionError("selected-token reduction used dynamic gather")
            return getattr(np, name)

    class FakeLax:
        def __init__(self) -> None:
            self.barrier_shapes = []
            self.dynamic_index_calls = []

        def optimization_barrier(self, value):
            self.barrier_shapes.append(tuple(value.shape))
            return value

        def dynamic_index_in_dim(self, operand, index, *, axis, keepdims):
            self.dynamic_index_calls.append(
                (tuple(operand.shape), int(index), axis, keepdims)
            )
            selected = np.take(operand, int(index), axis=axis)
            if keepdims:
                selected = np.expand_dims(selected, axis=axis)
            return selected

        @staticmethod
        def scan(fn, carry, xs):
            outputs = []
            for inputs in zip(*xs, strict=True):
                carry, output = fn(carry, inputs)
                outputs.append(output)
            return carry, tuple(
                np.stack([output[index] for output in outputs], axis=0)
                for index in range(2)
            )

    class FakeSpecial:
        @staticmethod
        def logsumexp(values, axis):
            maximum = np.max(values, axis=axis, keepdims=True)
            reduced = maximum + np.log(
                np.sum(np.exp(values - maximum), axis=axis, keepdims=True)
            )
            return np.squeeze(reduced, axis=axis)

    class FakeSampler:
        @staticmethod
        def _init_cache(*, batch_size, **kwargs):
            del kwargs
            return {"state": np.zeros(batch_size, dtype=np.float32)}

    class FakeGenerateUtils:
        @staticmethod
        def build_positions_from_mask(mask):
            return np.maximum(np.cumsum(mask, axis=-1) - 1, 0)

        @staticmethod
        def make_causal_attn_mask(mask, cache_size):
            return np.ones((mask.shape[0], mask.shape[1], cache_size), dtype=bool)

        @staticmethod
        def compute_attention_masks(position, cache_size, padding_mask):
            del position
            return np.ones(
                (padding_mask.shape[0], 1, cache_size), dtype=bool
            )

    class FakeModule:
        def __init__(self) -> None:
            self.skip_lm_head_flags = []

        def __call__(
            self,
            input_ids,
            positions,
            cache,
            attention_mask,
            *,
            skip_lm_head=False,
        ):
            del positions, attention_mask
            self.skip_lm_head_flags.append(skip_lm_head)
            if input_ids.shape[1] > 1:
                state = np.sum(input_ids, axis=-1, dtype=np.float32)
                hidden = np.stack(
                    (input_ids, input_ids + 1), axis=-1
                ).astype(np.float32)
            else:
                state = cache["state"] + input_ids[:, 0]
                hidden = np.stack((state, state + 1), axis=-1)[:, None, :]
            return hidden, {"state": state}

        @staticmethod
        def compute_final_logits(hidden):
            vocabulary = np.arange(6, dtype=np.float32)
            return -np.square(hidden[..., :1] - vocabulary)

    fake_lax = FakeLax()
    fake_jax = SimpleNamespace(
        lax=fake_lax,
        scipy=SimpleNamespace(special=FakeSpecial()),
    )
    module = FakeModule()
    model_params = SimpleNamespace(
        num_layers=1,
        num_kv_heads=1,
        head_dim=2,
        dtype=np.float32,
    )

    shared, selected = _qwen_cached_teacher_statistics(
        module,
        np.asarray([[1, 2], [2, 3]], dtype=np.int32),
        np.asarray([[True, True], [True, True]]),
        np.asarray([[3, 4, 5], [1, 0, 2]], dtype=np.int32),
        np.asarray([[True, True, True], [True, True, True]]),
        np.asarray([0, 2], dtype=np.int32),
        model_params=model_params,
        configured_cache_size=5,
        generate_sampler=FakeSampler,
        generate_utils=FakeGenerateUtils,
        jax_module=fake_jax,
        jnp_module=NoDynamicGatherNumpy(),
    )

    assert shared.shape == (2, 3, 2)
    assert selected.shape == (2, 3)
    assert module.skip_lm_head_flags == [True, True, True, True]
    assert fake_lax.barrier_shapes == [(2, 2), (2, 6), (2, 6), (2,)] * 3
    assert fake_lax.dynamic_index_calls == [
        ((6,), 3, 0, False),
        ((6,), 1, 0, False),
        ((6,), 4, 0, False),
        ((6,), 0, 0, False),
        ((6,), 5, 0, False),
        ((6,), 2, 0, False),
    ]

    centers = np.asarray(
        [[2.0, 6.0, 10.0], [3.0, 6.0, 6.0]],
        dtype=np.float32,
    )[..., None]
    dense_logits = -np.square(
        centers - np.arange(6, dtype=np.float32)[None, None, :]
    )
    dense_log_probs = dense_logits - FakeSpecial.logsumexp(dense_logits, axis=-1)[
        ..., None
    ]
    np.testing.assert_allclose(shared, dense_log_probs[..., [0, 2]])
    np.testing.assert_allclose(
        selected,
        np.take_along_axis(
            dense_log_probs,
            np.asarray([[3, 4, 5], [1, 0, 2]], dtype=np.int32)[..., None],
            axis=-1,
        )[..., 0],
    )
