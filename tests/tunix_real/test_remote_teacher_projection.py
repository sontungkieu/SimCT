from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from vdt_tunix.remote_teacher_backend import _project_remote_statistics


def _bfloat16_bits(values):
    array = jnp.asarray(values, dtype=jnp.bfloat16)
    return np.asarray(
        jax.lax.bitcast_convert_type(array, jnp.uint16),
        dtype=np.uint16,
    )


def test_remote_teacher_projection_matches_native_bfloat16_reference_under_jit():
    hidden_values = np.asarray(
        [
            [[1.0, -2.0, 0.5], [0.25, 4.0, -1.5]],
            [[-0.75, 1.5, 2.0], [3.0, -0.5, 0.125]],
        ],
        dtype=np.float32,
    )
    head_values = np.asarray(
        [[2.0, 0.5, -1.0], [-0.25, 1.25, 0.75]],
        dtype=np.float32,
    )
    log_normalizer = np.asarray(
        [[3.0, 2.5], [4.0, 1.0]],
        dtype=np.float32,
    )

    project = jax.jit(
        lambda hidden, head, normalizer: _project_remote_statistics(
            hidden,
            head,
            normalizer,
            jax_module=jax,
            jnp_module=jnp,
        )
    )
    observed = project(
        _bfloat16_bits(hidden_values),
        _bfloat16_bits(head_values),
        log_normalizer,
    )
    expected = jnp.einsum(
        "btd,od->bto",
        jnp.asarray(hidden_values, dtype=jnp.bfloat16),
        jnp.asarray(head_values, dtype=jnp.bfloat16),
    ).astype(jnp.float32) - jnp.asarray(log_normalizer)[..., None]

    np.testing.assert_array_equal(np.asarray(observed), np.asarray(expected))
    assert observed.dtype == jnp.float32
