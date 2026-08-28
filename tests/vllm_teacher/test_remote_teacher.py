from __future__ import annotations

import hashlib
import json
import os
import struct
from pathlib import Path

import numpy as np
import pytest

from vdt_tunix.remote_teacher import (
    REMOTE_TEACHER_CONTENT_TYPE,
    RemoteTeacherError,
    RemoteTeacherProfile,
    RemoteTeacherRuntimeConfig,
    parse_hidden_stats_response,
)


MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
MODEL_REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"
PROFILE_ID = "test-profile"
IDS_SHA = "1" * 64


def _response(prompt_ids=(1, 2), completion_ids=(3, 4)) -> bytes:
    rows = len(completion_ids)
    hidden_size = 3
    hidden = np.asarray(
        [[0x3F80, 0x4000, 0x4040], [0x4080, 0x40A0, 0x40C0]],
        dtype="<u2",
    )[:rows]
    logz = np.asarray([2.5, 3.5], dtype="<f4")[:rows]
    selected = np.asarray([-0.25, -0.5], dtype="<f4")[:rows]
    selected_ids = np.asarray(completion_ids, dtype="<i4")
    parts = [hidden.tobytes(), logz.tobytes(), selected.tobytes(), selected_ids.tobytes()]
    binary = b"".join(parts)
    offsets = [0]
    for part in parts[:-1]:
        offsets.append(offsets[-1] + len(part))
    full_ids = (*prompt_ids, *completion_ids)
    header = {
        "contract_version": 1,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "profile_id": PROFILE_ID,
        "profile_teacher_ids_sha256": IDS_SHA,
        "prompt_tokens": len(prompt_ids),
        "completion_tokens": rows,
        "hidden_size": hidden_size,
        "input_ids_sha256": hashlib.sha256(
            struct.pack(f"<{len(full_ids)}i", *full_ids)
        ).hexdigest(),
        "payload_sha256": hashlib.sha256(binary).hexdigest(),
        "arrays": [
            {
                "name": "hidden_states",
                "dtype": "bfloat16_bits_le",
                "shape": [rows, hidden_size],
                "offset": offsets[0],
                "nbytes": len(parts[0]),
            },
            {
                "name": "log_normalizer",
                "dtype": "float32_le",
                "shape": [rows],
                "offset": offsets[1],
                "nbytes": len(parts[1]),
            },
            {
                "name": "selected_log_probs",
                "dtype": "float32_le",
                "shape": [rows],
                "offset": offsets[2],
                "nbytes": len(parts[2]),
            },
            {
                "name": "selected_token_ids",
                "dtype": "int32_le",
                "shape": [rows],
                "offset": offsets[3],
                "nbytes": len(parts[3]),
            },
        ],
    }
    encoded = json.dumps(header, sort_keys=True).encode("utf-8")
    return struct.pack("<I", len(encoded)) + encoded + binary


def _parse(body: bytes):
    return parse_hidden_stats_response(
        body,
        content_type=REMOTE_TEACHER_CONTENT_TYPE,
        model_id=MODEL_ID,
        model_revision=MODEL_REVISION,
        profile_id=PROFILE_ID,
        profile_teacher_ids_sha256=IDS_SHA,
        prompt_token_ids=(1, 2),
        completion_token_ids=(3, 4),
    )


def test_parse_hidden_stats_validates_exact_binary_contract():
    result = _parse(_response())

    assert result.hidden_state_bits.shape == (2, 3)
    assert result.hidden_state_bits.dtype == np.dtype("<u2")
    np.testing.assert_array_equal(result.selected_token_ids, [3, 4])
    np.testing.assert_allclose(result.log_normalizer, [2.5, 3.5])
    np.testing.assert_allclose(result.selected_log_probs, [-0.25, -0.5])


def test_parse_hidden_stats_rejects_payload_tampering():
    body = bytearray(_response())
    body[-1] ^= 1

    with pytest.raises(RemoteTeacherError, match="payload hash"):
        _parse(bytes(body))


def test_runtime_config_requires_https_or_explicit_local_canary(tmp_path, monkeypatch):
    token = tmp_path / "token"
    token.write_text("x" * 48, encoding="utf-8")
    token.chmod(0o600)
    profile = tmp_path / "profile"
    profile.mkdir()

    with pytest.raises(RemoteTeacherError, match="HTTPS"):
        RemoteTeacherRuntimeConfig("http://127.0.0.1:18000", token, profile)

    monkeypatch.setenv("VDT_REMOTE_TEACHER_ALLOW_INSECURE_LOCALHOST", "1")
    configured = RemoteTeacherRuntimeConfig(
        "http://127.0.0.1:18000/", token, profile
    )
    assert configured.url == "http://127.0.0.1:18000"


def test_profile_loads_only_when_hashes_and_shapes_match(tmp_path):
    ids = np.asarray([2, 5, 9], dtype="<i4")
    head = np.asarray(
        [[0x3F80, 0x4000], [0x4040, 0x4080], [0x40A0, 0x40C0]],
        dtype="<u2",
    )
    ids_path = tmp_path / "teacher_ids.i32le"
    head_path = tmp_path / "teacher_overlap_lm_head.bf16le"
    ids.tofile(ids_path)
    head.tofile(head_path)
    ids_sha = hashlib.sha256(ids_path.read_bytes()).hexdigest()
    head_sha = hashlib.sha256(head_path.read_bytes()).hexdigest()
    manifest = {
        "contract_version": 1,
        "profile_id": PROFILE_ID,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "teacher_ids_sha256": ids_sha,
        "shape": [3, 2],
        "dtype": "bfloat16_le",
        "sha256": head_sha,
    }
    (tmp_path / "teacher_overlap_lm_head.manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )

    profile = RemoteTeacherProfile.load(tmp_path)

    assert profile.hidden_size == 2
    profile.validate_overlap_ids([2, 5, 9])
    with pytest.raises(RemoteTeacherError, match="overlap IDs"):
        profile.validate_overlap_ids([2, 9, 5])
