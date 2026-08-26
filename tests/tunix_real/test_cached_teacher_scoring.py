from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from vdt_tunix.real_backend import (
    _cached_teacher_forcing_scan,
    _qwen_cached_teacher_statistics,
)


def test_cached_teacher_forcing_matches_dense_causal_statistics():
    vocabulary_size = 11
    overlap_ids = jnp.asarray([0, 3, 7, 10], dtype=jnp.int32)
    prompt_state = jnp.asarray([5, 9], dtype=jnp.int32)
    completion_ids = jnp.asarray(
        [
            [2, 4, 1, 8],
            [6, 0, 3, 5],
        ],
        dtype=jnp.int32,
    )

    def logits_from_state(state):
        vocabulary = jnp.arange(vocabulary_size, dtype=jnp.float32)
        center = jnp.mod(state, vocabulary_size).astype(jnp.float32)
        return -jnp.square(vocabulary[None, :] - center[:, None])

    def decode_one(cache, token_ids, step):
        del step
        updated = cache + token_ids
        return logits_from_state(updated), updated

    shared, selected = _cached_teacher_forcing_scan(
        logits_from_state(prompt_state),
        prompt_state,
        completion_ids,
        overlap_ids,
        decode_one,
        jax_module=jax,
        jnp_module=jnp,
    )

    dense_logits = []
    state = prompt_state
    for step in range(completion_ids.shape[1]):
        dense_logits.append(logits_from_state(state))
        state = state + completion_ids[:, step]
    dense_log_probs = jax.nn.log_softmax(jnp.stack(dense_logits, axis=1))
    expected_shared = jnp.take(dense_log_probs, overlap_ids, axis=-1)
    expected_selected = jnp.take_along_axis(
        dense_log_probs,
        completion_ids[..., None],
        axis=-1,
    )[..., 0]

    np.testing.assert_allclose(shared, expected_shared, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(selected, expected_selected, rtol=1e-6, atol=1e-6)


def test_tiny_qwen_cached_teacher_forcing_matches_dense_forward():
    nnx = pytest.importorskip("flax.nnx")
    qwen2_model = pytest.importorskip("tunix.models.qwen2.model")
    sampler_lib = pytest.importorskip("tunix.generate.sampler")
    generate_utils = pytest.importorskip("tunix.generate.utils")

    config = qwen2_model.ModelConfig.qwen2p5_0p5b()
    config.num_layers = 1
    config.vocab_size = 32
    config.embed_dim = 32
    config.hidden_dim = 64
    config.num_heads = 4
    config.head_dim = 8
    config.num_kv_heads = 2
    config.use_flash_attention = False
    config.dtype = jnp.bfloat16
    model = qwen2_model.Qwen2(config, rngs=nnx.Rngs(0))

    prompt_ids = jnp.asarray(
        [
            [0, 0, 5, 7, 9],
            [0, 3, 4, 6, 8],
        ],
        dtype=jnp.int32,
    )
    prompt_mask = prompt_ids != 0
    completion_ids = jnp.asarray(
        [
            [11, 13, 15],
            [10, 12, 14],
        ],
        dtype=jnp.int32,
    )
    completion_mask = jnp.ones_like(completion_ids, dtype=jnp.bool_)
    overlap_ids = jnp.asarray([1, 5, 9, 17], dtype=jnp.int32)
    mesh = jax.make_mesh((1, 1), ("fsdp", "tp"))

    @nnx.jit
    def cached_forward(module, prompts, prompts_mask, completions, completions_mask):
        return _qwen_cached_teacher_statistics(
            module,
            prompts,
            prompts_mask,
            completions,
            completions_mask,
            overlap_ids,
            model_params=config,
            configured_cache_size=8,
            generate_sampler=sampler_lib,
            generate_utils=generate_utils,
            jax_module=jax,
            jnp_module=jnp,
        )

    with jax.set_mesh(mesh):
        cached_shared, cached_selected = cached_forward(
            model,
            prompt_ids,
            prompt_mask,
            completion_ids,
            completion_mask,
        )

        rows = []
        positions = []
        for row in range(prompt_ids.shape[0]):
            prompt = prompt_ids[row][prompt_mask[row]]
            sequence = jnp.concatenate((prompt, completion_ids[row]))
            rows.append(jnp.pad(sequence, (0, 7 - sequence.shape[0])))
            positions.append(
                jnp.arange(
                    prompt.shape[0] - 1,
                    prompt.shape[0] - 1 + completion_ids.shape[1],
                    dtype=jnp.int32,
                )
            )
        dense_ids = jnp.stack(rows)
        dense_segments = dense_ids != 0
        dense_positions = jnp.maximum(
            jnp.cumsum(dense_segments, axis=-1) - 1,
            0,
        )
        causal = jnp.tril(jnp.ones((7, 7), dtype=jnp.bool_))
        dense_attention = dense_segments[:, None, :] & causal[None, ...]
        dense_logits, _ = model(
            dense_ids,
            dense_positions,
            None,
            dense_attention,
            segment_ids=dense_segments.astype(jnp.int32),
        )
        batch = jnp.arange(dense_ids.shape[0], dtype=jnp.int32)[:, None]
        dense_completion_logits = dense_logits[batch, jnp.stack(positions)]
        dense_log_probs = jax.nn.log_softmax(
            dense_completion_logits.astype(jnp.float32),
            axis=-1,
        )

    expected_shared = jnp.take(dense_log_probs, overlap_ids, axis=-1)
    expected_selected = jnp.take_along_axis(
        dense_log_probs,
        completion_ids[..., None],
        axis=-1,
    )[..., 0]
    np.testing.assert_allclose(
        cached_shared,
        expected_shared,
        rtol=2e-2,
        atol=2e-2,
    )
    np.testing.assert_allclose(
        cached_selected,
        expected_selected,
        rtol=2e-2,
        atol=2e-2,
    )
