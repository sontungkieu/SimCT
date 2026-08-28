"""Authenticated client and exact binary contract for a remote vLLM teacher."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import stat
import struct
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


REMOTE_TEACHER_CONTRACT_VERSION = 1
REMOTE_TEACHER_CONTENT_TYPE = (
    "application/vnd.vdt.teacher-hidden-stats-v1"
)


class RemoteTeacherError(RuntimeError):
    """Raised when the remote teacher violates identity or payload contracts."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: Any, context: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise RemoteTeacherError(f"{context} must be a lowercase SHA-256")
    return value


@dataclasses.dataclass(frozen=True, slots=True)
class RemoteTeacherRuntimeConfig:
    """Operational settings kept outside the paper/scientific config digest."""

    url: str
    token_file: Path
    profile_dir: Path
    tokenizer_dir: Path | None = None
    timeout_s: float = 300.0
    max_parallel_requests: int = 4

    def __post_init__(self) -> None:
        normalized = self.url.rstrip("/")
        object.__setattr__(self, "url", normalized)
        if not normalized.startswith("https://"):
            allow_local = os.environ.get(
                "VDT_REMOTE_TEACHER_ALLOW_INSECURE_LOCALHOST"
            ) == "1" and normalized.startswith(
                ("http://127.0.0.1", "http://localhost")
            )
            if not allow_local:
                raise RemoteTeacherError(
                    "remote teacher URL must use HTTPS; plain HTTP is allowed "
                    "only for an explicit localhost canary"
                )
        if not self.token_file.is_file():
            raise RemoteTeacherError("remote teacher token file is unavailable")
        if os.name == "posix":
            mode = stat.S_IMODE(self.token_file.stat().st_mode)
            if mode & 0o077:
                raise RemoteTeacherError(
                    "remote teacher token file must be owner-only"
                )
        if not self.profile_dir.is_dir():
            raise RemoteTeacherError("remote teacher profile directory is unavailable")
        tokenizer_dir = self.tokenizer_dir or self.profile_dir / "tokenizer"
        object.__setattr__(self, "tokenizer_dir", tokenizer_dir)
        if not math.isfinite(self.timeout_s) or self.timeout_s <= 0:
            raise RemoteTeacherError("remote teacher timeout must be positive")
        if self.max_parallel_requests < 1:
            raise RemoteTeacherError(
                "remote teacher max_parallel_requests must be positive"
            )

    @classmethod
    def from_environment(
        cls, environ: Mapping[str, str] | None = None
    ) -> RemoteTeacherRuntimeConfig | None:
        values = os.environ if environ is None else environ
        url = values.get("VDT_REMOTE_TEACHER_URL")
        if not url:
            return None
        required = {
            "VDT_REMOTE_TEACHER_TOKEN_FILE": values.get(
                "VDT_REMOTE_TEACHER_TOKEN_FILE"
            ),
            "VDT_REMOTE_TEACHER_PROFILE_DIR": values.get(
                "VDT_REMOTE_TEACHER_PROFILE_DIR"
            ),
        }
        missing = sorted(name for name, value in required.items() if not value)
        if missing:
            raise RemoteTeacherError(
                f"remote teacher environment is missing {missing}"
            )
        try:
            timeout = float(values.get("VDT_REMOTE_TEACHER_TIMEOUT_S", "300"))
            parallel = int(values.get("VDT_REMOTE_TEACHER_MAX_PARALLEL", "4"))
        except ValueError as exc:
            raise RemoteTeacherError(
                "remote teacher timeout/parallel settings are invalid"
            ) from exc
        return cls(
            url=url,
            token_file=Path(required["VDT_REMOTE_TEACHER_TOKEN_FILE"]),
            profile_dir=Path(required["VDT_REMOTE_TEACHER_PROFILE_DIR"]),
            tokenizer_dir=(
                None
                if not values.get("VDT_REMOTE_TEACHER_TOKENIZER_DIR")
                else Path(values["VDT_REMOTE_TEACHER_TOKENIZER_DIR"])
            ),
            timeout_s=timeout,
            max_parallel_requests=parallel,
        )


@dataclasses.dataclass(frozen=True, slots=True)
class RemoteTeacherProfile:
    profile_id: str
    model_id: str
    model_revision: str
    teacher_ids_sha256: str
    overlap_head_sha256: str
    teacher_ids: Any = dataclasses.field(repr=False, compare=False)
    overlap_head_bits: Any = dataclasses.field(repr=False, compare=False)
    hidden_size: int

    @classmethod
    def load(
        cls,
        directory: Path,
        *,
        verify_hashes: bool = True,
    ) -> RemoteTeacherProfile:
        try:
            import numpy as np
        except ImportError as exc:
            raise RemoteTeacherError("NumPy is required for remote teacher") from exc
        manifest_path = directory / "teacher_overlap_lm_head.manifest.json"
        ids_path = directory / "teacher_ids.i32le"
        head_path = directory / "teacher_overlap_lm_head.bf16le"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RemoteTeacherError("remote teacher profile manifest is invalid") from exc
        if manifest.get("contract_version") != REMOTE_TEACHER_CONTRACT_VERSION:
            raise RemoteTeacherError("unsupported remote teacher profile contract")
        shape = manifest.get("shape")
        if (
            not isinstance(shape, list)
            or len(shape) != 2
            or any(isinstance(value, bool) or not isinstance(value, int) for value in shape)
            or any(value < 1 for value in shape)
        ):
            raise RemoteTeacherError("remote teacher head shape is invalid")
        if manifest.get("dtype") != "bfloat16_le":
            raise RemoteTeacherError("remote teacher head must be bfloat16_le")
        ids_sha = _require_sha256(
            manifest.get("teacher_ids_sha256"), "teacher_ids_sha256"
        )
        head_sha = _require_sha256(manifest.get("sha256"), "head sha256")
        if verify_hashes:
            if _sha256(ids_path) != ids_sha:
                raise RemoteTeacherError("remote teacher ID artifact hash mismatch")
            if _sha256(head_path) != head_sha:
                raise RemoteTeacherError("remote teacher head artifact hash mismatch")
        ids = np.memmap(ids_path, dtype="<i4", mode="r", shape=(shape[0],))
        head = np.memmap(
            head_path,
            dtype="<u2",
            mode="r",
            shape=(shape[0], shape[1]),
        )
        if len(set(int(value) for value in ids)) != shape[0]:
            raise RemoteTeacherError("remote teacher overlap IDs are not unique")
        return cls(
            profile_id=str(manifest["profile_id"]),
            model_id=str(manifest["model_id"]),
            model_revision=str(manifest["model_revision"]),
            teacher_ids_sha256=ids_sha,
            overlap_head_sha256=head_sha,
            teacher_ids=ids,
            overlap_head_bits=head,
            hidden_size=shape[1],
        )

    def validate_overlap_ids(self, token_ids: Sequence[int]) -> None:
        normalized = tuple(int(value) for value in token_ids)
        if normalized != tuple(int(value) for value in self.teacher_ids):
            raise RemoteTeacherError(
                "computed teacher overlap IDs do not match the pinned profile"
            )


@dataclasses.dataclass(frozen=True, slots=True)
class TeacherHiddenStats:
    hidden_state_bits: Any = dataclasses.field(repr=False, compare=False)
    log_normalizer: Any = dataclasses.field(repr=False, compare=False)
    selected_log_probs: Any = dataclasses.field(repr=False, compare=False)
    selected_token_ids: Any = dataclasses.field(repr=False, compare=False)
    header: Mapping[str, Any] = dataclasses.field(repr=False, compare=False)


def parse_hidden_stats_response(
    body: bytes,
    *,
    content_type: str,
    model_id: str,
    model_revision: str,
    profile_id: str,
    profile_teacher_ids_sha256: str,
    prompt_token_ids: Sequence[int],
    completion_token_ids: Sequence[int],
) -> TeacherHiddenStats:
    try:
        import numpy as np
    except ImportError as exc:
        raise RemoteTeacherError("NumPy is required for remote teacher") from exc
    if not content_type.startswith(REMOTE_TEACHER_CONTENT_TYPE):
        raise RemoteTeacherError("remote teacher returned an unsupported content type")
    if len(body) < 4:
        raise RemoteTeacherError("remote teacher response is truncated")
    header_size = struct.unpack_from("<I", body, 0)[0]
    if header_size < 2 or header_size > 1024 * 1024 or 4 + header_size > len(body):
        raise RemoteTeacherError("remote teacher header size is invalid")
    try:
        header = json.loads(body[4 : 4 + header_size])
    except json.JSONDecodeError as exc:
        raise RemoteTeacherError("remote teacher header is invalid JSON") from exc
    if not isinstance(header, dict):
        raise RemoteTeacherError("remote teacher header must be an object")
    expected_identity = {
        "contract_version": REMOTE_TEACHER_CONTRACT_VERSION,
        "model_id": model_id,
        "model_revision": model_revision,
        "profile_id": profile_id,
        "profile_teacher_ids_sha256": profile_teacher_ids_sha256,
        "prompt_tokens": len(prompt_token_ids),
        "completion_tokens": len(completion_token_ids),
    }
    for key, expected in expected_identity.items():
        if header.get(key) != expected:
            raise RemoteTeacherError(f"remote teacher header {key} mismatch")
    full_ids = tuple(int(value) for value in prompt_token_ids) + tuple(
        int(value) for value in completion_token_ids
    )
    expected_input_hash = hashlib.sha256(
        struct.pack(f"<{len(full_ids)}i", *full_ids)
    ).hexdigest()
    if header.get("input_ids_sha256") != expected_input_hash:
        raise RemoteTeacherError("remote teacher input ID hash mismatch")

    binary = memoryview(body)[4 + header_size :]
    if hashlib.sha256(binary).hexdigest() != header.get("payload_sha256"):
        raise RemoteTeacherError("remote teacher payload hash mismatch")
    arrays = header.get("arrays")
    if not isinstance(arrays, list):
        raise RemoteTeacherError("remote teacher array manifest is missing")
    by_name: dict[str, Mapping[str, Any]] = {}
    for item in arrays:
        if not isinstance(item, dict) or not isinstance(item.get("name"), str):
            raise RemoteTeacherError("remote teacher array entry is invalid")
        if item["name"] in by_name:
            raise RemoteTeacherError("remote teacher array names are duplicated")
        by_name[item["name"]] = item
    expected_names = {
        "hidden_states",
        "log_normalizer",
        "selected_log_probs",
        "selected_token_ids",
    }
    if set(by_name) != expected_names:
        raise RemoteTeacherError("remote teacher arrays do not match the contract")

    rows = len(completion_token_ids)
    hidden_size = header.get("hidden_size")
    if isinstance(hidden_size, bool) or not isinstance(hidden_size, int) or hidden_size < 1:
        raise RemoteTeacherError("remote teacher hidden size is invalid")
    specifications = {
        "hidden_states": ("bfloat16_bits_le", [rows, hidden_size], rows * hidden_size * 2),
        "log_normalizer": ("float32_le", [rows], rows * 4),
        "selected_log_probs": ("float32_le", [rows], rows * 4),
        "selected_token_ids": ("int32_le", [rows], rows * 4),
    }
    offset = 0
    for name in (
        "hidden_states",
        "log_normalizer",
        "selected_log_probs",
        "selected_token_ids",
    ):
        dtype, shape, nbytes = specifications[name]
        item = by_name[name]
        if (
            item.get("dtype") != dtype
            or item.get("shape") != shape
            or item.get("offset") != offset
            or item.get("nbytes") != nbytes
        ):
            raise RemoteTeacherError(f"remote teacher array {name} is malformed")
        offset += nbytes
    if offset != len(binary):
        raise RemoteTeacherError("remote teacher payload length mismatch")

    hidden = np.frombuffer(
        binary,
        dtype="<u2",
        count=rows * hidden_size,
        offset=by_name["hidden_states"]["offset"],
    ).reshape(rows, hidden_size)
    log_normalizer = np.frombuffer(
        binary,
        dtype="<f4",
        count=rows,
        offset=by_name["log_normalizer"]["offset"],
    )
    selected_log_probs = np.frombuffer(
        binary,
        dtype="<f4",
        count=rows,
        offset=by_name["selected_log_probs"]["offset"],
    )
    selected_token_ids = np.frombuffer(
        binary,
        dtype="<i4",
        count=rows,
        offset=by_name["selected_token_ids"]["offset"],
    )
    if selected_token_ids.tolist() != [int(value) for value in completion_token_ids]:
        raise RemoteTeacherError("remote teacher causal token alignment mismatch")
    if not np.isfinite(log_normalizer).all() or not np.isfinite(
        selected_log_probs
    ).all():
        raise RemoteTeacherError("remote teacher returned non-finite statistics")
    return TeacherHiddenStats(
        hidden_state_bits=hidden,
        log_normalizer=log_normalizer,
        selected_log_probs=selected_log_probs,
        selected_token_ids=selected_token_ids,
        header=header,
    )


class RemoteTeacherClient:
    """Synchronous fail-closed HTTPS client; bearer values are never exposed."""

    def __init__(
        self,
        runtime: RemoteTeacherRuntimeConfig,
        profile: RemoteTeacherProfile,
        *,
        opener: Any | None = None,
    ) -> None:
        self.runtime = runtime
        self.profile = profile
        token = runtime.token_file.read_text(encoding="utf-8").strip()
        if len(token) < 32:
            raise RemoteTeacherError("remote teacher bearer token is too short")
        self._authorization = "Bearer " + token
        self._opener = opener or urllib.request.urlopen

    def _request(self, path: str, payload: Mapping[str, Any] | None = None) -> tuple[bytes, str]:
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        headers = {"Authorization": self._authorization}
        if data is not None:
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(
            self.runtime.url + path,
            data=data,
            headers=headers,
            method="POST" if data is not None else "GET",
        )
        try:
            with self._opener(request, timeout=self.runtime.timeout_s) as response:
                body = response.read()
                content_type = response.headers.get("Content-Type", "")
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise RemoteTeacherError(
                f"remote teacher request failed: {type(exc).__name__}"
            ) from exc
        return body, content_type

    def health(self) -> Mapping[str, Any]:
        body, _ = self._request("/v1/vdt/teacher/health")
        try:
            result = json.loads(body)
        except json.JSONDecodeError as exc:
            raise RemoteTeacherError("remote teacher health response is invalid") from exc
        expected = {
            "ok": True,
            "contract_version": REMOTE_TEACHER_CONTRACT_VERSION,
            "model_id": self.profile.model_id,
            "model_revision": self.profile.model_revision,
            "profile_id": self.profile.profile_id,
            "profile_teacher_ids_sha256": self.profile.teacher_ids_sha256,
        }
        if not isinstance(result, dict) or any(
            result.get(key) != value for key, value in expected.items()
        ):
            raise RemoteTeacherError("remote teacher health identity mismatch")
        return result

    def score_tokens(
        self,
        prompt_token_ids: Sequence[int],
        completion_token_ids: Sequence[int],
    ) -> TeacherHiddenStats:
        prompt = tuple(int(value) for value in prompt_token_ids)
        completion = tuple(int(value) for value in completion_token_ids)
        body, content_type = self._request(
            "/v1/vdt/teacher/hidden-stats",
            {
                "prompt_token_ids": prompt,
                "completion_token_ids": completion,
                "profile_id": self.profile.profile_id,
                "profile_teacher_ids_sha256": self.profile.teacher_ids_sha256,
            },
        )
        return parse_hidden_stats_response(
            body,
            content_type=content_type,
            model_id=self.profile.model_id,
            model_revision=self.profile.model_revision,
            profile_id=self.profile.profile_id,
            profile_teacher_ids_sha256=self.profile.teacher_ids_sha256,
            prompt_token_ids=prompt,
            completion_token_ids=completion,
        )
